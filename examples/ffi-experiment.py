#!/usr/bin/ipython
from MozaikLite.framework.experiment import MeasureOrientationTuningFullfield, MeasureSpontaneousActivity
from MozaikLite.models.model import JensModel
from MozaikLite.framework.experiment_controller import run_experiments

from pyNN import nest as sim
jens_model = JensModel(sim)

experiment_list =   [
                       MeasureOrientationTuningFullfield(jens_model,10,0.8,2,144*7)
                       #MeasureSpontaneousActivity(jens_model,143*7)
                    ]

run_experiments(jens_model,'root_directory',experiment_list)
