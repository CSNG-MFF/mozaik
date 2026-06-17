# encoding: utf-8
r"""
Experiments and helpers for presenting DVS event streams to spike-source sheets.
"""

from collections import OrderedDict

import numpy
from parameters import ParameterSet

from mozaik.experiments import Experiment
from mozaik.sheets.direct_stimulator import SpikeSourceArrayStimulator
from mozaik.stimuli import InternalStimulus
from mozaik.tools.distribution_parametrization import MozaikExtendedParameterSet


def load_dvs_events(event_file):
    r"""
    Load DVS events from an ``.npy`` file.

    The expected format is a numeric array with columns
    ``time_ms, x, y, polarity``.
    """
    events = numpy.load(event_file)
    try:
        events = numpy.asarray(events, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("DVS event file must contain a numeric array")
    _validate_event_shape(events)
    if not numpy.all(numpy.isfinite(events)):
        raise ValueError("DVS event array must contain only finite values")
    return events


def _validate_event_shape(events):
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("DVS event array must have shape (n_events, 4)")


def group_dvs_events_by_grid(events, dvs_width, dvs_height, duration):
    r"""
    Convert DVS events into per-neuron ON and OFF spike-time lists.

    The DVS pixel grid must exactly match the sheet grid. No rescaling,
    interpolation, or nearest-neighbor matching is performed here.
    """
    events = numpy.asarray(events, dtype=float)
    _validate_event_shape(events)
    _validate_dvs_events(events, dvs_width, dvs_height, duration)

    on_spikes = [[] for _ in range(dvs_width * dvs_height)]
    off_spikes = [[] for _ in range(dvs_width * dvs_height)]

    for time, x, y, polarity in events:
        # PyNN Grid2D fills y fastest for each x, so this indexing mirrors the
        # cell ordering produced by VisualCorticalGridSheet.
        index = _pixel_index(int(x), int(y), dvs_height)
        if polarity > 0:
            on_spikes[index].append(float(time))
        else:
            off_spikes[index].append(float(time))

    return _sort_spike_times(on_spikes), _sort_spike_times(off_spikes)


def _validate_dvs_events(events, dvs_width, dvs_height, duration):
    if dvs_width <= 0 or dvs_height <= 0:
        raise ValueError("DVS grid dimensions must be positive")
    if duration <= 0:
        raise ValueError("DVS experiment duration must be positive")
    if not numpy.all(numpy.isfinite(events)):
        raise ValueError("DVS event array must contain only finite values")

    times = events[:, 0]
    pixels = events[:, 1:3]
    polarities = events[:, 3]

    if not numpy.all((times >= 0.0) & (times < duration)):
        raise ValueError("DVS event times must be within the experiment duration")
    if not numpy.all(pixels == numpy.round(pixels)):
        raise ValueError("DVS events must use integer pixel coordinates")
    if not numpy.all((polarities > 0.0) | (polarities < 0.0)):
        raise ValueError("DVS event polarity must be positive or negative")

    x = pixels[:, 0]
    y = pixels[:, 1]
    if not numpy.all((x >= 0) & (x < dvs_width) & (y >= 0) & (y < dvs_height)):
        raise ValueError("DVS event pixel coordinates are out of range")


def _pixel_index(x, y, dvs_height):
    return x * dvs_height + y


def _sort_spike_times(spike_times):
    return [sorted(times) for times in spike_times]


def infer_sheet_grid_shape(sheet):
    r"""
    Infer the x/y grid dimensions from a sheet's generated positions.
    """
    positions = numpy.asarray(sheet.pop.positions)
    x_values = numpy.unique(numpy.round(positions[0], 12))
    y_values = numpy.unique(numpy.round(positions[1], 12))
    grid_shape = (len(x_values), len(y_values))
    if len(sheet.pop) != grid_shape[0] * grid_shape[1]:
        raise ValueError(
            "DVS target sheet positions do not form a complete rectangular grid"
        )
    return grid_shape


class DVSRecordedInput(Experiment):
    r"""
    Present DVS events from a NumPy file through ON and OFF spike-source sheets.
    """

    required_parameters = ParameterSet({
        "event_file": str,
        "duration": float,
        "on_sheet_name": str,
        "off_sheet_name": str,
        "dvs_width": int,
        "dvs_height": int,
    })

    def __init__(self, model, parameters):
        Experiment.__init__(self, model, parameters)
        self._validate_target_sheets()
        events = load_dvs_events(self.parameters.event_file)
        on_spikes, off_spikes = group_dvs_events_by_grid(
            events,
            self.parameters.dvs_width,
            self.parameters.dvs_height,
            self.parameters.duration,
        )

        direct_stimulation = OrderedDict()
        direct_stimulation[self.parameters.on_sheet_name] = [
            SpikeSourceArrayStimulator(
                self.model.sheets[self.parameters.on_sheet_name],
                ParameterSet({"spike_times": on_spikes}),
            )
        ]
        direct_stimulation[self.parameters.off_sheet_name] = [
            SpikeSourceArrayStimulator(
                self.model.sheets[self.parameters.off_sheet_name],
                ParameterSet({"spike_times": off_spikes}),
            )
        ]
        self.direct_stimulation = [direct_stimulation]
        self.stimuli.append(
            InternalStimulus(
                frame_duration=self.parameters.duration,
                duration=self.parameters.duration,
                trial=0,
                direct_stimulation_name="DVSRecordedInput",
                direct_stimulation_parameters=MozaikExtendedParameterSet(
                    {
                        "event_file": self.parameters.event_file,
                        "on_sheet_name": self.parameters.on_sheet_name,
                        "off_sheet_name": self.parameters.off_sheet_name,
                        "dvs_width": self.parameters.dvs_width,
                        "dvs_height": self.parameters.dvs_height,
                    }
                ),
            )
        )

    def _validate_target_sheets(self):
        for sheet_name in [self.parameters.on_sheet_name, self.parameters.off_sheet_name]:
            if sheet_name not in self.model.sheets:
                raise ValueError("DVS target sheet %s does not exist" % sheet_name)
            shape = infer_sheet_grid_shape(self.model.sheets[sheet_name])
            if shape != (self.parameters.dvs_width, self.parameters.dvs_height):
                raise ValueError(
                    "DVS grid %s does not match sheet %s grid %s"
                    % ((self.parameters.dvs_width, self.parameters.dvs_height), sheet_name, shape)
                )

    def do_analysis(self):
        pass
