from collections import OrderedDict

import numpy as np
import pytest
from parameters import ParameterSet
from pyNN.parameters import Sequence

import mozaik
from mozaik.experiments.dvs import (
    DVSRecordedInput,
    group_dvs_events_by_grid,
    load_dvs_events,
)
from mozaik.sheets.direct_stimulator import SpikeSourceArrayStimulator


class FakeCell:
    local = True

    def __init__(self):
        self.spike_times = None

    def set_parameters(self, **parameters):
        self.spike_times = parameters["spike_times"]


class FakePopulation:
    celltype = type("SpikeSourceArray", (), {})()

    def __init__(self, size):
        self.size = size
        self._mask_local = np.ones(size, dtype=bool)
        self.all_cells = np.arange(size)
        self.cells = [FakeCell() for _ in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.cells[index]


class FakeSheet:
    def __init__(self, name, width=2, height=2):
        self.name = name
        self.dt = 0.1
        self.parameters = ParameterSet(
            {
                "cell": {
                    "model": "SpikeSourceArray",
                    "native_nest": False,
                    "params": {"spike_times": []},
                    "receptors": {},
                    "initial_values": {},
                }
            }
        )
        self.pop = FakePopulation(width * height)
        xs, ys = np.indices((width, height), dtype=float)
        self.pop.positions = np.array((xs.flatten(), ys.flatten(), np.zeros(width * height)))


class FakeModel:
    def __init__(self):
        self.sheets = OrderedDict(
            {
                "ON": FakeSheet("ON"),
                "OFF": FakeSheet("OFF"),
            }
        )


def sequence_values(sequence):
    assert isinstance(sequence, Sequence)
    return list(np.array(sequence.value, dtype=float))


def encode_iit_yarp_dvs_address(x, y, polarity, channel=1):
    raw_polarity = 0 if polarity > 0 else 1
    return (((channel << 10 | y) << 11 | x) << 1) | raw_polarity


def write_events(tmp_path, events, name="dvs_dataset"):
    event_dir = tmp_path / name
    event_dir.mkdir()
    (event_dir / "info.log").write_text(
        "Type: Bottle;\n"
        "Stamp: rx;\n"
        "[0.000000] /vPreProcess/right:o [connected]\n"
    )
    encoded_events = []
    for time_ms, x, y, polarity in sorted(events, key=lambda event: event[0]):
        encoded_events.append(str(int(round(time_ms * 1000.0))))
        encoded_events.append(str(encode_iit_yarp_dvs_address(int(x), int(y), polarity)))
    (event_dir / "data.log").write_text(
        "1 0.000000 AE (%s)\n" % " ".join(encoded_events)
    )
    return str(event_dir)


def test_load_dvs_events_accepts_iit_yarp_export(tmp_path):
    event_file = write_events(
        tmp_path,
        [
            [0.0, 0, 1, 1],
            [2.0, 0, 1, 1],
            [1.0, 1, 0, -1],
            [3.0, 1, 0, -1],
        ],
    )

    events = load_dvs_events(event_file)

    np.testing.assert_allclose(
        events,
        np.array(
            [
                [0.0, 0, 1, 1],
                [1.0, 1, 0, -1],
                [2.0, 0, 1, 1],
                [3.0, 1, 0, -1],
            ],
            dtype=float,
        ),
    )


@pytest.mark.parametrize(
    "events, message",
    [
        ([[1.0, 0, 1]], "shape"),
        ([[1.0, 0.5, 1, 1]], "integer pixel"),
        ([[1.0, 0, 1, 0]], "polarity"),
        ([[11.0, 0, 1, 1]], "duration"),
        ([[1.0, 2, 1, 1]], "out of range"),
    ],
)
def test_group_dvs_events_rejects_invalid_events(events, message):
    with pytest.raises(ValueError, match=message):
        group_dvs_events_by_grid(np.asarray(events, dtype=float), 2, 2, 10.0)


def test_group_dvs_events_uses_exact_pixel_grid_and_splits_polarity():
    events = np.array(
        [
            [5.0, 0, 0, 1],
            [2.0, 1, 0, -1],
            [6.0, 1, 0, -1],
            [4.0, 1, 1, 1],
            [8.0, 1, 1, 1],
            [3.0, 2, 1, -1],
            [7.0, 2, 1, -1],
            [1.0, 0, 0, 1],
        ]
    )

    on_spikes, off_spikes = group_dvs_events_by_grid(events, 3, 2, 10.0)

    assert [list(s) for s in on_spikes] == [
        [1.0, 5.0],
        [],
        [],
        [4.0, 8.0],
        [],
        [],
    ]
    assert [list(s) for s in off_spikes] == [
        [],
        [],
        [2.0, 6.0],
        [],
        [],
        [3.0, 7.0],
    ]


def test_dvs_spike_source_stimulator_offsets_and_clears_spike_times():
    sheet = FakeSheet("ON", width=2, height=2)
    stimulator = SpikeSourceArrayStimulator(
        sheet,
        ParameterSet(
            {"spike_times": [[0.0, 3.0], [], [2.0, 4.0], [0.5, 4.5]]}
        ),
    )

    stimulator.prepare_stimulation(duration=5.0, offset=10.0)

    np.testing.assert_allclose(sequence_values(sheet.pop[0].spike_times), [10.2001, 13.0])
    assert sequence_values(sheet.pop[1].spike_times) == []
    assert sequence_values(sheet.pop[2].spike_times) == [12.0, 14.0]
    assert sequence_values(sheet.pop[3].spike_times) == [10.5, 14.5]

    stimulator.inactivate(offset=15.0)

    for cell in sheet.pop.cells:
        assert sequence_values(cell.spike_times) == []


def test_dvs_recorded_input_uses_named_on_off_sheets(tmp_path):
    event_file = write_events(
        tmp_path,
        [
            [0.0, 0, 0, 1],
            [2.0, 0, 0, 1],
            [1.0, 1, 1, -1],
            [3.0, 1, 1, -1],
        ],
    )

    experiment = DVSRecordedInput(
        FakeModel(),
        ParameterSet(
            {
                "event_file": event_file,
                "duration": 5.0,
                "on_sheet_name": "ON",
                "off_sheet_name": "OFF",
                "dvs_width": 2,
                "dvs_height": 2,
            }
        ),
    )

    stimulation = experiment.direct_stimulation[0]
    assert list(stimulation.keys()) == ["ON", "OFF"]
    assert [list(s) for s in stimulation["ON"][0].parameters.spike_times] == [
        [0.0, 2.0],
        [],
        [],
        [],
    ]
    assert [list(s) for s in stimulation["OFF"][0].parameters.spike_times] == [
        [],
        [],
        [],
        [1.0, 3.0],
    ]


def test_dvs_recorded_input_rejects_sheet_grid_mismatch(tmp_path):
    event_file = write_events(tmp_path, [[0.0, 0, 0, 1]])

    with pytest.raises(ValueError, match="DVS grid"):
        DVSRecordedInput(
            FakeModel(),
            ParameterSet(
                {
                    "event_file": event_file,
                    "duration": 5.0,
                    "on_sheet_name": "ON",
                    "off_sheet_name": "OFF",
                    "dvs_width": 3,
                    "dvs_height": 2,
                }
            ),
        )


@pytest.mark.model
def test_dvs_recorded_input_records_spikes_from_two_consecutive_nest_experiments(
    tmp_path,
):
    from mozaik.models import Model
    from mozaik.sheets.vision import VisualCorticalGridSheet
    from pyNN import nest

    class TinyDVSModel(Model):
        required_parameters = ParameterSet({})

        def __init__(self, sim, num_threads, parameters):
            Model.__init__(self, sim, num_threads, parameters)
            self.on_sheet = VisualCorticalGridSheet(
                self, self.spike_source_sheet_parameters("ON")
            )
            self.off_sheet = VisualCorticalGridSheet(
                self, self.spike_source_sheet_parameters("OFF")
            )
            self.cortex_sheet = VisualCorticalGridSheet(
                self, self.cortex_sheet_parameters("V1")
            )
            self.connect_input_sheets()

        def spike_source_sheet_parameters(self, name):
            return ParameterSet(
                {
                    "name": name,
                    "sx": 200.0,
                    "sy": 200.0,
                    "density": 100.0,
                    "mpi_safe": False,
                    "magnification_factor": 1000.0,
                    "cell": {
                        "model": "SpikeSourceArray",
                        "native_nest": False,
                        "params": {"spike_times": []},
                        "receptors": {},
                        "initial_values": {},
                    },
                    "artificial_stimulators": {},
                    "recording_interval": 1.0,
                    "recorders": {
                        "all": {
                            "component": "mozaik.sheets.population_selector.RCAll",
                            "variables": "spikes",
                            "params": {},
                        }
                    },
                }
            )

        def cortex_sheet_parameters(self, name):
            return ParameterSet(
                {
                    "name": name,
                    "sx": 200.0,
                    "sy": 200.0,
                    "density": 100.0,
                    "mpi_safe": False,
                    "magnification_factor": 1000.0,
                    "cell": {
                        "model": "IF_cond_exp",
                        "native_nest": False,
                        "params": {
                            "v_thresh": -64.0,
                            "v_rest": -65.0,
                            "v_reset": -65.0,
                            "tau_refrac": 1.0,
                            "tau_m": 10.0,
                            "cm": 0.05,
                            "e_rev_E": 0.0,
                            "e_rev_I": -80.0,
                            "tau_syn_E": 0.5,
                            "tau_syn_I": 5.0,
                        },
                        "receptors": {},
                        "initial_values": {"v": -65.0},
                    },
                    "artificial_stimulators": {},
                    "recording_interval": 1.0,
                    "recorders": {
                        "all": {
                            "component": "mozaik.sheets.population_selector.RCAll",
                            "variables": ("v", "spikes"),
                            "params": {},
                        }
                    },
                }
            )

        def connect_input_sheets(self):
            synapse = self.sim.StaticSynapse(
                weight=0.04, delay=self.parameters.min_delay
            )
            self.sim.Projection(
                self.on_sheet.pop,
                self.cortex_sheet.pop,
                self.sim.OneToOneConnector(),
                synapse_type=synapse,
                receptor_type="excitatory",
            )
            self.sim.Projection(
                self.off_sheet.pop,
                self.cortex_sheet.pop,
                self.sim.OneToOneConnector(),
                synapse_type=synapse,
                receptor_type="excitatory",
            )

    def model_parameters():
        return ParameterSet(
            {
                "input_space_type": "",
                "input_space": None,
                "sheets": None,
                "results_dir": "",
                "name": "TinyDVSModel",
                "reset": False,
                "null_stimulus_period": 0.0,
                "store_stimuli": False,
                "min_delay": 0.1,
                "max_delay": 5.0,
                "time_step": 0.1,
                "pynn_seed": 936395,
                "mpi_seed": 1023,
                "explosion_monitoring": None,
                "steps_get_data": 0,
            }
        )

    def dvs_experiment(model, event_file):
        return DVSRecordedInput(
            model,
            ParameterSet(
                {
                    "event_file": event_file,
                    "duration": 10.0,
                    "on_sheet_name": "ON",
                    "off_sheet_name": "OFF",
                    "dvs_width": 2,
                    "dvs_height": 2,
                }
            ),
        )

    def spikes_by_index(sheet, segment):
        ids = list(sheet.pop.all_cells.astype(int))
        spikes = {}
        for train in segment.spiketrains:
            index = ids.index(int(train.annotations["source_id"]))
            spikes[index] = [float(t) for t in train]
        return spikes

    def assert_spikes_by_index(actual, expected):
        assert actual.keys() == expected.keys()
        for index, expected_times in expected.items():
            np.testing.assert_allclose(actual[index], expected_times)

    def cortex_voltage_delta(segment):
        signal = np.asarray(segment.analogsignals[0])
        return float(signal.max() - signal.min())

    def assert_postsynaptic_spikes_after_inputs(postsynaptic_spikes, input_times):
        assert all(spike_time > min(input_times) for spike_time in postsynaptic_spikes)
        for input_time in input_times:
            matching_spikes = [
                spike_time
                for spike_time in postsynaptic_spikes
                if input_time < spike_time <= input_time + 1.0
            ]
            assert matching_spikes

    mozaik.setup_mpi(mozaik_seed=1023, pynn_seed=936395)
    model = TinyDVSModel(nest, 1, model_parameters())
    try:
        event_file_1 = write_events(
            tmp_path,
            [
                [0.0, 0, 0, 1],
                [2.0, 0, 0, 1],
                [4.0, 1, 0, -1],
                [6.0, 1, 0, -1],
                [5.0, 1, 1, 1],
                [8.0, 1, 1, 1],
            ],
            "events_1",
        )
        event_file_2 = write_events(
            tmp_path,
            [
                [0.0, 0, 1, -1],
                [3.0, 0, 1, -1],
                [4.0, 1, 1, 1],
                [7.0, 1, 1, 1],
            ],
            "events_2",
        )

        experiment_1 = dvs_experiment(model, event_file_1)
        segments_1, _, _, _, _ = model.present_stimulus_and_record(
            experiment_1.stimuli[0], experiment_1.direct_stimulation[0]
        )
        first = {segment.annotations["sheet_name"]: segment for segment in segments_1}
        first_on_spikes = spikes_by_index(model.on_sheet, first["ON"])
        first_off_spikes = spikes_by_index(model.off_sheet, first["OFF"])
        first_cortex_spikes = spikes_by_index(model.cortex_sheet, first["V1"])
        first_cortex_voltage_delta = cortex_voltage_delta(first["V1"])

        experiment_2 = dvs_experiment(model, event_file_2)
        segments_2, _, _, _, _ = model.present_stimulus_and_record(
            experiment_2.stimuli[0], experiment_2.direct_stimulation[0]
        )
        second = {segment.annotations["sheet_name"]: segment for segment in segments_2}
        second_on_spikes = spikes_by_index(model.on_sheet, second["ON"])
        second_off_spikes = spikes_by_index(model.off_sheet, second["OFF"])
        second_cortex_spikes = spikes_by_index(model.cortex_sheet, second["V1"])
        second_cortex_voltage_delta = cortex_voltage_delta(second["V1"])
    finally:
        nest.end()

    assert_spikes_by_index(
        first_on_spikes,
        {
            0: [0.2001, 2.0],
            1: [],
            2: [],
            3: [5.0, 8.0],
        },
    )
    assert_spikes_by_index(
        first_off_spikes,
        {
            0: [],
            1: [],
            2: [4.0, 6.0],
            3: [],
        },
    )
    assert_spikes_by_index(
        second_on_spikes,
        {
            0: [],
            1: [],
            2: [],
            3: [4.0, 7.0],
        },
    )
    assert_spikes_by_index(
        second_off_spikes,
        {
            0: [],
            1: [0.2001, 3.0],
            2: [],
            3: [],
        },
    )
    assert_postsynaptic_spikes_after_inputs(first_cortex_spikes[0], [0.2001, 2.0])
    assert_postsynaptic_spikes_after_inputs(first_cortex_spikes[2], [4.0, 6.0])
    assert_postsynaptic_spikes_after_inputs(first_cortex_spikes[3], [5.0, 8.0])
    assert_postsynaptic_spikes_after_inputs(second_cortex_spikes[1], [0.2001, 3.0])
    assert_postsynaptic_spikes_after_inputs(second_cortex_spikes[3], [4.0, 7.0])
    assert first_cortex_voltage_delta > 0.0
    assert second_cortex_voltage_delta > 0.0
