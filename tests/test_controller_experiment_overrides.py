from collections import OrderedDict
from types import SimpleNamespace

import pytest
from parameters import ParameterSet

import mozaik.controller
from mozaik.experiments import NoStimulation


class DummyDataStore:
    """
    Minimal datastore stub used to verify that ``run_workflow`` reaches the
    save step on the MPI root rank.
    """

    def __init__(self):
        self.saved = False

    def save(self):
        self.saved = True


class DummyModel:
    """
    Minimal model stub for controller tests.

    The controller only needs to instantiate the model and pass it forward to
    experiment construction, so no additional model behavior is required here.
    """

    def __init__(self, sim, num_threads, parameters):
        self.sim = sim
        self.num_threads = num_threads
        self.parameters = parameters


def _install_common_monkeypatches(monkeypatch, experiment_overrides):
    """
    Patch common controller dependencies so tests can focus on branch logic.

    This helper bypasses the real workflow preparation path, which would
    otherwise parse CLI arguments, load parameter files, initialize MPI-related
    state, and create result directories. Instead it injects a minimal fixed
    workflow context together with the experiment overrides under test.

    Parameters
    ----------

    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to replace controller dependencies for the duration
        of the test.

    experiment_overrides : OrderedDict
        Experiment override dictionary returned by the patched
        ``prepare_workflow`` helper.
    """
    monkeypatch.delenv("MOZAIK_EXPERANTO_CHUNK_EXPORT", raising=False)
    parameters = ParameterSet({"store_stimuli": False, "null_stimulus_period": 0})
    monkeypatch.setattr(
        mozaik.controller,
        "prepare_workflow",
        lambda simulation_name, model_class: (
            object(),
            1,
            parameters,
            experiment_overrides,
        ),
    )
    monkeypatch.setattr(
        mozaik.controller.mozaik,
        "mpi_comm",
        SimpleNamespace(rank=0),
    )
    monkeypatch.setattr(mozaik.controller.mozaik, "MPI_ROOT", 0)


def test_run_workflow_list_legacy_path(monkeypatch):
    """
    Verify that the legacy list-based experiment workflow remains unchanged.

    When no experiment overrides are present and ``create_experiments`` returns
    an already-instantiated list, the controller should pass that list straight
    through to ``run_experiments`` without synthesizing explicit experiment
    metadata.
    """
    _install_common_monkeypatches(monkeypatch, OrderedDict())
    captured = {}

    def fake_run_experiments(
        model,
        experiment_list,
        parameters,
        load_from=None,
        experiment_parameter_list=None,
    ):
        captured["experiment_list"] = experiment_list
        captured["experiment_parameter_list"] = experiment_parameter_list
        return DummyDataStore()

    monkeypatch.setattr(mozaik.controller, "run_experiments", fake_run_experiments)

    def create_experiments(model):
        return [NoStimulation(model, ParameterSet({"duration": 10.0}))]

    data_store, model = mozaik.controller.run_workflow(
        "dummy",
        DummyModel,
        create_experiments,
    )

    assert isinstance(model, DummyModel)
    assert len(captured["experiment_list"]) == 1
    assert captured["experiment_list"][0].parameters.duration == 10.0
    assert captured["experiment_parameter_list"] is None
    assert data_store.saved is True


def test_run_workflow_ordereddict_applies_overrides(monkeypatch):
    """
    Verify that the OrderedDict-based workflow applies overrides before
    experiment instantiation.

    The resulting instantiated experiment should contain the overridden value,
    and the controller should store legacy experiment metadata tuples for the
    datastore.
    """
    _install_common_monkeypatches(
        monkeypatch,
        OrderedDict(
            {
                "experiments.blank.duration": 42.0,
            }
        ),
    )
    captured = {}

    def fake_run_experiments(
        model,
        experiment_list,
        parameters,
        load_from=None,
        experiment_parameter_list=None,
    ):
        captured["experiment_list"] = experiment_list
        captured["experiment_parameter_list"] = experiment_parameter_list
        return DummyDataStore()

    monkeypatch.setattr(mozaik.controller, "run_experiments", fake_run_experiments)

    def create_experiments(model):
        return OrderedDict(
            [
                ("blank", (NoStimulation, ParameterSet({"duration": 10.0}))),
            ]
        )

    data_store, model = mozaik.controller.run_workflow(
        "dummy",
        DummyModel,
        create_experiments,
    )

    assert isinstance(model, DummyModel)
    assert len(captured["experiment_list"]) == 1
    assert captured["experiment_list"][0].parameters.duration == 42.0
    assert captured["experiment_parameter_list"] == [
        (str(NoStimulation), str(captured["experiment_list"][0].parameters))
    ]
    assert data_store.saved is True


def test_run_workflow_list_rejects_experiment_overrides(monkeypatch):
    """
    Verify that new-style experiment overrides are rejected for legacy
    list-based experiment factories.
    """
    _install_common_monkeypatches(
        monkeypatch,
        OrderedDict({"experiments.blank.duration": 42.0}),
    )

    def create_experiments(model):
        return [NoStimulation(model, ParameterSet({"duration": 10.0}))]

    with pytest.raises(ValueError, match="Experiment overrides were provided"):
        mozaik.controller.run_workflow("dummy", DummyModel, create_experiments)


@pytest.mark.parametrize("configuration", ["explicit", "environment"])
def test_run_workflow_exports_after_save(monkeypatch, tmp_path, configuration):
    from mozaik.meta_workflow import experanto_export as export

    _install_common_monkeypatches(monkeypatch, OrderedDict())
    store = DummyDataStore()
    monkeypatch.setattr(
        mozaik.controller, "run_experiments", lambda *args, **kwargs: store
    )
    monkeypatch.setattr(
        mozaik.controller.mozaik,
        "mpi_comm",
        SimpleNamespace(rank=0, bcast=lambda value, root: value),
    )
    request = export.ExperantoChunkExport(0, 0, str(tmp_path / "0_0.json"))
    captured = []

    def export_chunk(data_store, received):
        assert data_store.saved
        captured.append(received)

    monkeypatch.setattr(export, "export_experanto_chunk", export_chunk)
    options = {"experanto_export": request}
    if configuration == "environment":
        monkeypatch.setenv(
            export.EXPORT_JOB_ENV, request.job_environment()[export.EXPORT_JOB_ENV]
        )
        options = {}
    mozaik.controller.run_workflow("dummy", DummyModel, lambda model: [], **options)
    assert captured == [request]


def test_run_workflow_ordereddict_rejects_unknown_target(monkeypatch):
    """
    Verify that overrides fail fast when they reference a missing named
    experiment.
    """
    _install_common_monkeypatches(
        monkeypatch,
        OrderedDict({"experiments.missing.duration": 42.0}),
    )

    def create_experiments(model):
        return OrderedDict(
            [
                ("blank", (NoStimulation, ParameterSet({"duration": 10.0}))),
            ]
        )

    with pytest.raises(ValueError, match="Unknown experiment override target"):
        mozaik.controller.run_workflow("dummy", DummyModel, create_experiments)
