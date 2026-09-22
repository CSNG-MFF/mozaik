import copy

import numpy as np
import pytest
from parameters import ParameterSet

import mozaik
from mozaik.models import Model
from mozaik.models.vision.spatiotemporalfilter import SpatioTemporalFilterRetinaLGN
from mozaik.space import VisualRegion
import mozaik.stimuli.vision.topographica_based as topo
from mozaik.tools.distribution_parametrization import load_parameters
from tests.models.vision.spatiotemporalfilter_test_support import (
    BASE_STIM_PARAMS,
    PARAMS,
)

GRATING_DURATION = 2000.0


@pytest.fixture(scope="module")
def nest_simulator():
    from pyNN import nest

    return nest


def seed_comparison_parameters(simulation_seed, model_seed=None):
    parameters = copy.deepcopy(load_parameters(PARAMS, ParameterSet({})))
    del parameters["visual_field"]
    parameters["simulation_seed"] = simulation_seed
    if model_seed is not None:
        parameters["model_seed"] = model_seed
    parameters["reset"] = False
    parameters["null_stimulus_period"] = 150.0
    retina_parameters = parameters["sheets"]["retina_lgn"]["params"]
    retina_parameters["size"] = (1.0, 1.0)
    retina_parameters["density"] = 10
    retina_parameters["noise"]["stdev"] = 3.0
    retina_parameters["mpi_reproducible_noise"] = False
    return parameters


def seed_comparison_grating(trial):
    return topo.FullfieldDriftingSinusoidalGrating(
        **{
            **BASE_STIM_PARAMS,
            "orientation": 0.0,
            "spatial_frequency": 0.5,
            "temporal_frequency": 1.0,
            "contrast": 100.0,
            "duration": GRATING_DURATION,
            "trial": trial,
        }
    )


def construct_seed_comparison_retina(
    simulator,
    simulation_seed,
    model_seed=None,
):
    parameters = seed_comparison_parameters(simulation_seed, model_seed)
    mozaik.setup_seeds(
        model_seed=parameters["model_seed"],
        simulation_seed=parameters["simulation_seed"],
        experiment_seed=parameters["experiment_seed"],
    )
    mozaik.setup_mpi()

    model = Model(simulator, 2, parameters)
    model.visual_field = VisualRegion(
        location_x=0.0,
        location_y=0.0,
        size_x=7.0,
        size_y=7.0,
    )
    retina = SpatioTemporalFilterRetinaLGN(
        model,
        parameters["sheets"]["retina_lgn"]["params"],
    )
    model.seed_comparison_projection = model.sim.Projection(
        retina.sheets["X_ON"].pop,
        retina.sheets["X_OFF"].pop,
        model.sim.FixedProbabilityConnector(
            0.25,
            rng=mozaik.model_pynn_rng,
        ),
        synapse_type=model.sim.StaticSynapse(
            weight=0.0,
            delay=model.parameters.min_delay,
        ),
        receptor_type="excitatory",
    )

    positions = {
        rf_type: np.asarray(retina.sheets[rf_type].pop.positions).copy()
        for rf_type in retina.rf_types
    }
    connections = np.asarray(
        model.seed_comparison_projection.get(
            "weight",
            format="list",
            gather=True,
        )
    )[:, :2].astype(int)
    connections = connections[np.lexsort((connections[:, 1], connections[:, 0]))]
    return model, retina, positions, connections


def run_seed_comparison_trials(simulator, simulation_seed, num_trials):
    model, retina, positions, connections = construct_seed_comparison_retina(
        simulator,
        simulation_seed,
    )
    trial_responses = []

    try:
        for sheet in retina.sheets.values():
            sheet.pop.record("v", sampling_interval=1.0)

        simulator_time = 0.0
        for trial in range(num_trials):
            if trial:
                retina.provide_null_input(
                    model.input_space,
                    duration=model.parameters.null_stimulus_period,
                    offset=simulator_time,
                )
                model.sim.run(model.parameters.null_stimulus_period)
                simulator_time += model.parameters.null_stimulus_period
                for sheet in retina.sheets.values():
                    sheet.pop.get_data("v", clear=True)

            stimulus = seed_comparison_grating(trial)
            model.input_space.clear()
            model.input_space.add_object(str(stimulus), stimulus)
            retina.process_input(
                model.input_space,
                stimulus,
                duration=GRATING_DURATION,
                offset=simulator_time,
            )
            model.sim.run(GRATING_DURATION)
            simulator_time += GRATING_DURATION

            sheet_responses = []
            for sheet in retina.sheets.values():
                segment = sheet.pop.get_data("v", clear=True).segments[-1]
                sheet_responses.append(np.asarray(segment.analogsignals[0]))
            trial_response = np.concatenate(sheet_responses, axis=1)
            expected_samples = int(GRATING_DURATION)
            trial_responses.append(trial_response[-expected_samples:])
    finally:
        model.sim.end()

    return positions, connections, np.asarray(trial_responses)


def test_same_model_seed_different_simulation_seeds_construct_same_model(
    nest_simulator,
):
    """Model-seeded cell layouts and sampled connectivity ignore simulation_seed."""
    model_a, _, positions_a, connections_a = construct_seed_comparison_retina(
        nest_simulator,
        1023,
    )
    model_a.sim.end()
    model_b, _, positions_b, connections_b = construct_seed_comparison_retina(
        nest_simulator,
        1024,
    )
    model_b.sim.end()

    assert positions_a.keys() == positions_b.keys()
    for rf_type in positions_a:
        np.testing.assert_array_equal(positions_a[rf_type], positions_b[rf_type])
    assert 0 < len(connections_a) < 100
    np.testing.assert_array_equal(connections_a, connections_b)


def test_different_model_seeds_change_model_construction(nest_simulator):
    """With simulation_seed fixed, model_seed changes layouts and connectivity."""
    model_a, _, positions_a, connections_a = construct_seed_comparison_retina(
        nest_simulator,
        1023,
        model_seed=936395,
    )
    model_a.sim.end()
    model_b, _, positions_b, connections_b = construct_seed_comparison_retina(
        nest_simulator,
        1023,
        model_seed=936396,
    )
    model_b.sim.end()

    assert any(
        not np.array_equal(positions_a[rf_type], positions_b[rf_type])
        for rf_type in positions_a
    )
    assert 0 < len(connections_a) < 100
    assert 0 < len(connections_b) < 100
    assert not np.array_equal(connections_a, connections_b)


def test_same_model_different_simulation_seeds_produce_different_results(
    nest_simulator,
):
    """Identical cell layouts must still yield different noisy voltages."""
    positions_a, connections_a, responses_a = run_seed_comparison_trials(
        nest_simulator,
        1023,
        1,
    )
    positions_b, connections_b, responses_b = run_seed_comparison_trials(
        nest_simulator,
        1024,
        1,
    )

    for rf_type in positions_a:
        np.testing.assert_array_equal(positions_a[rf_type], positions_b[rf_type])
    np.testing.assert_array_equal(connections_a, connections_b)
    assert responses_a.shape == responses_b.shape
    assert not np.array_equal(responses_a, responses_b)


def test_five_trial_grating_responses_are_close_across_simulation_seeds(
    nest_simulator,
):
    """Compare trial-averaged LGN voltages for a 2 s, 1 Hz full-field grating."""
    _, _, responses_a = run_seed_comparison_trials(nest_simulator, 1023, 5)
    _, _, responses_b = run_seed_comparison_trials(nest_simulator, 1024, 5)
    trial_average_a = responses_a.mean(axis=0)
    trial_average_b = responses_b.mean(axis=0)
    mean_absolute_error = np.mean(np.abs(trial_average_a - trial_average_b))
    response_scale = np.mean(np.abs((trial_average_a + trial_average_b) / 2.0))
    relative_mean_error = mean_absolute_error / response_scale
    assert relative_mean_error <= 0.05, (
        "Five-trial mean responses differ by " f"{relative_mean_error:.2%}"
    )
