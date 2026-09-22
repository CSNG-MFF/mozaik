import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from parameters import ParameterSet

import mozaik


def sample_seed_streams():
    return {
        "model_rng": mozaik.model_rng.rand(5),
        "model_pynn_rng": mozaik.model_pynn_rng.next(
            5,
            "uniform",
            {"low": 0.0, "high": 1.0},
        ),
        "simulation_rng": mozaik.simulation_rng.rand(5),
        "experiment_rng": mozaik.experiment_rng.rand(5),
    }


def test_setup_seeds_requires_all_seeds():
    with pytest.raises(TypeError):
        mozaik.setup_seeds()
    with pytest.raises(TypeError):
        mozaik.setup_seeds(model_seed=1)
    with pytest.raises(TypeError):
        mozaik.setup_seeds(model_seed=1, simulation_seed=2)


def test_model_and_simulation_seed_streams_are_independent():
    mozaik.setup_seeds(model_seed=11, simulation_seed=22, experiment_seed=33)

    expected_model = np.random.RandomState(11).randint(2**32 - 1, size=5)
    expected_simulation = np.random.RandomState(22).randint(2**32 - 1, size=5)

    np.testing.assert_array_equal(mozaik.get_model_seeds(5), expected_model)
    np.testing.assert_array_equal(mozaik.get_simulation_seeds(5), expected_simulation)


@pytest.mark.parametrize(
    "changed_seed, affected_streams",
    [
        ("model_seed", {"model_rng", "model_pynn_rng"}),
        ("simulation_seed", {"simulation_rng"}),
        ("experiment_seed", {"experiment_rng"}),
    ],
)
def test_each_seed_changes_only_its_owned_streams(changed_seed, affected_streams):
    seeds = {"model_seed": 11, "simulation_seed": 22, "experiment_seed": 33}
    mozaik.setup_seeds(**seeds)
    baseline = sample_seed_streams()

    seeds[changed_seed] += 1
    mozaik.setup_seeds(**seeds)
    changed = sample_seed_streams()

    for stream, baseline_values in baseline.items():
        if stream in affected_streams:
            assert not np.array_equal(changed[stream], baseline_values)
        else:
            np.testing.assert_array_equal(changed[stream], baseline_values)


def test_experiment_seed_controls_experiment_stream():
    values = list(range(10))
    expected = list(range(10))

    mozaik.setup_seeds(model_seed=11, simulation_seed=22, experiment_seed=33)
    mozaik.experiment_rng.shuffle(values)
    np.random.RandomState(33).shuffle(expected)

    assert values == expected


def test_visual_experiment_shuffle_uses_only_experiment_seed():
    from mozaik.experiments.vision import VisualExperiment

    class NumberedStimuliExperiment(VisualExperiment):
        def generate_stimuli(self):
            self.stimuli.extend(range(10))

    model = SimpleNamespace(
        input_space=SimpleNamespace(
            background_luminance=0.0,
            parameters=SimpleNamespace(update_interval=1.0),
        ),
        input_layer=SimpleNamespace(
            parameters=SimpleNamespace(
                receptive_field=SimpleNamespace(spatial_resolution=1.0)
            )
        ),
    )

    def stimulus_order(model_seed, simulation_seed, experiment_seed):
        mozaik.setup_seeds(model_seed, simulation_seed, experiment_seed)
        experiment = NumberedStimuliExperiment(
            model,
            ParameterSet({"shuffle_stimuli": True}),
        )
        return experiment.stimuli

    baseline = stimulus_order(11, 22, 33)
    assert stimulus_order(12, 22, 33) == baseline
    assert stimulus_order(11, 23, 33) == baseline
    assert stimulus_order(11, 22, 34) != baseline


def test_mozaik_source_seed_stream_ownership():
    """Keep each of model, simulation and experiment randomness confined to
    the code it should affect.

    The model stream belongs to model and PyNN construction, the simulation
    stream to runtime effects and analysis, and the experiment stream to
    stimulus generation and ordering. This test scans the Mozaik source tree
    and fails when code accesses a stream that does not belong in that part of
    the repository. Valid mixed-purpose files are documented as exceptions.
    """
    model_apis = {"model_rng", "model_pynn_rng", "get_model_seeds"}
    simulation_apis = {"simulation_rng", "get_simulation_seeds"}
    experiment_apis = {"experiment_rng"}
    stream_apis = model_apis | simulation_apis | experiment_apis
    allowed_by_path = {
        Path("mozaik/analysis"): simulation_apis,
        Path("mozaik/connectors"): model_apis,
        Path("mozaik/controller.py"): simulation_apis,
        Path("mozaik/experiments"): experiment_apis,
        Path("mozaik/models"): model_apis,
        Path("mozaik/sheets"): model_apis,
        Path("mozaik/stimuli"): experiment_apis,
    }
    exceptions = {
        # General-purpose tools have no seed-stream owner.
        Path("mozaik/tools"): stream_apis,
        # simulation_rng influences LGN noise currents,
        # model_rng determines construction of the retinal sheets.
        Path("mozaik/models/vision/spatiotemporalfilter.py"): (
            model_apis | simulation_apis
        ),
        # Model-seeded selectors choose targets, while simulation-seeded StGen
        # instances control stochastic spike timing during direct stimulation.
        Path("mozaik/sheets/direct_stimulator.py"): (model_apis | simulation_apis),
    }
    violations = []

    paths_to_check = allowed_by_path | {
        path: allowed for path, allowed in exceptions.items() if path.is_dir()
    }
    # Check that each file uses only the random streams assigned to its part of
    # the codebase. This catches direct ``mozaik.<stream>`` uses, but does not
    # trace where a seed stored in a local variable originally came from.
    for path, allowed in paths_to_check.items():
        source_paths = path.rglob("*.py") if path.is_dir() else [path]
        for source_path in source_paths:
            tree = ast.parse(source_path.read_text(), filename=str(source_path))
            used = {
                node.attr
                for node in ast.walk(tree)
                if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "mozaik"
                and node.attr in stream_apis
            }
            permitted = exceptions.get(source_path, allowed)
            if not used <= permitted:
                violations.append(f"{source_path}: {sorted(used - permitted)}")

    assert not violations, "Seed-stream ownership violations:\n" + "\n".join(violations)


def test_mozaik_rngs_are_not_unseeded_or_seeded_by_literals():
    """Prevent Mozaik code from creating an unseeded or fixed-seed RNG."""

    def is_literal(expression):
        variable_nodes = (ast.Name, ast.Attribute, ast.Call, ast.Subscript)
        return not any(
            isinstance(node, variable_nodes) for node in ast.walk(expression)
        )

    constructor_names = {
        "Random",
        "RandomState",
        "default_rng",
        "NumpyRNG",
        "StGen",
    }
    violations = []

    for source_path in Path("mozaik").rglob("*.py"):
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            constructor = getattr(node.func, "id", getattr(node.func, "attr", None))
            if constructor not in constructor_names:
                continue

            supplied_values = list(node.args[:1]) + [
                keyword.value
                for keyword in node.keywords
                if keyword.arg in {"seed", "rng"}
            ]
            supplied_values = [
                value
                for value in supplied_values
                if not (isinstance(value, ast.Constant) and value.value is None)
            ]
            error = (
                "has no seed or RNG"
                if not supplied_values
                else (
                    "has a literal seed"
                    if any(is_literal(value) for value in supplied_values)
                    else None
                )
            )
            if error:
                violations.append(f"{source_path}:{node.lineno}: {constructor} {error}")

    assert not violations, "Invalid Mozaik RNG construction:\n" + "\n".join(violations)


def test_setup_seeds_can_lock_against_reinitialization(monkeypatch):
    monkeypatch.setattr(mozaik, "_seed_setup_locked", False)
    mozaik.setup_seeds(
        model_seed=11,
        simulation_seed=22,
        experiment_seed=33,
        prevent_reinitialization=True,
    )

    with pytest.raises(
        RuntimeError,
        match="random-number streams are already initialized",
    ):
        mozaik.setup_seeds(
            model_seed=11,
            simulation_seed=22,
            experiment_seed=33,
        )
