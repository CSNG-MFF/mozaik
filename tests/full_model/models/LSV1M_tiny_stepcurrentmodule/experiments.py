#!/usr/local/bin/ipython -i
from mozaik.experiments import NoStimulation
from mozaik.experiments.vision import MeasureOrientationTuningFullfield
from mozaik.sheets.population_selector import RCRandomPercentage
from parameters import ParameterSet


def create_experiments(model):
    return [
        # Lets kick the network up into activation
        # Spontaneous Activity
        NoStimulation(model, ParameterSet({"duration": 105})),
        # Measure orientation tuning with full-filed sinusoidal gratins
        MeasureOrientationTuningFullfield(
            model,
            ParameterSet(
                {
                    "num_orientations": 2,
                    "spatial_frequency": 0.8,
                    "temporal_frequency": 2,
                    "grating_duration": 210,  # 15*7
                    "contrasts": [100],
                    "num_trials": 1,
                    "shuffle_stimuli": False,
                }
            ),
        ),
    ]
