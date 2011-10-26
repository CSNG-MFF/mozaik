#!/usr/bin/ipython
from MozaikLite.framework.experiment import MeasureOrientationTuningFullfield, MeasureSpontaneousActivity
from pyNN import nest as sim
from MozaikLite.models.model import JensModel
from MozaikLite.framework.experiment_controller import run_experiments,run_analysis
import datetime
import os 

root_directory = 'FFI' + str(datetime.datetime.now())
os.mkdir(root_directory)
jens_model = JensModel(sim)
experiment_list =   [
                       MeasureOrientationTuningFullfield(jens_model,num_orientations=2,spatial_frequency=0.8,temporal_frequency=2,grating_duration=147*7,num_trials=2),
                       #MeasureSpontaneousActivity(jens_model,duration=7*50)
                    ]


if False:
    run_experiments(jens_model,root_directory,experiment_list)
else:
    run_analysis("FFI2011-10-26 17:18:59.677870",experiment_list)    
