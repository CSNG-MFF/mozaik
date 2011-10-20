#!/usr/bin/ipython
from MozaikLite.framework.experiment import MeasureOrientationTuningFullfield, MeasureSpontaneousActivity
from MozaikLite.models.model import JensModel
from MozaikLite.framework.experiment_controller import run_experiments,run_analysis
import datetime
import os 
from pyNN import nest as sim


root_directory = 'FFI' + str(datetime.datetime.now())
os.mkdir(root_directory)
jens_model = JensModel(sim)
experiment_list =   [
                       MeasureOrientationTuningFullfield(jens_model,num_orientations=2,spatial_frequency=0.8,temporal_frequency=2,grating_duration=36*7,num_trials=1),
                       #MeasureSpontaneousActivity(jens_model,duration=7*50)
                    ]


if True:
    run_experiments(jens_model,root_directory,experiment_list)
else:
    run_analysis("FFI2011-10-12 18:31:51.243966",experiment_list)    
