#!/usr/bin/ipython
from MozaikLite.framework.experiment import MeasureOrientationTuningFullfield, MeasureSpontaneousActivity
from pyNN import nest as sim
from MozaikLite.models.model import JensModel
from MozaikLite.framework.experiment_controller import run_experiments,run_analysis, setup_experiment

params = setup_experiment('FFI',sim)
jens_model = JensModel(sim,params)

experiment_list =   [
                       MeasureOrientationTuningFullfield(jens_model,num_orientations=1,spatial_frequency=0.8,temporal_frequency=2,grating_duration=7*7,num_trials=2),
                       #MeasureSpontaneousActivity(jens_model,duration=7*50)
                    ]

if True:
    run_experiments(jens_model,experiment_list)
else:
    run_analysis("FFI2011-10-26 17:18:59.677870",experiment_list)    
