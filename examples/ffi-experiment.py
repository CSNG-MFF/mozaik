#!/usr/bin/ipython 
import matplotlib
matplotlib.use('GTKAgg') # do this before importing pylab
from MozaikLite.framework.experiment import MeasureOrientationTuningFullfield, MeasureSpontaneousActivity, MeasureNaturalImagesWithEyeMovement
from pyNN import nest as sim
from MozaikLite.models.model import JensModel
from MozaikLite.framework.experiment_controller import run_experiments,run_analysis, setup_experiment

params = setup_experiment('FFI',sim)
jens_model = JensModel(sim,params)

experiment_list =   [
                       #MeasureOrientationTuningFullfield(jens_model,num_orientations=6,spatial_frequency=0.8,temporal_frequency=2,grating_duration=143*7,num_trials=10),
                       #MeasureOrientationTuningFullfield(jens_model,num_orientations=2,spatial_frequency=0.8,temporal_frequency=2,grating_duration=143*7,num_trials=2),
                       #MeasureOrientationTuningFullfield(jens_model,num_orientations=2,spatial_frequency=0.8,temporal_frequency=2,grating_duration=143*7,num_trials=2),
                       MeasureSpontaneousActivity(jens_model,duration=147*7),
                       MeasureOrientationTuningFullfield(jens_model,num_orientations=2,spatial_frequency=0.8,temporal_frequency=2,grating_duration=50*7,num_trials=1),
                    ]

if True:
    run_experiments(jens_model,experiment_list)
else:
    run_analysis("FFI_20120201-160419",experiment_list)    

import pylab
pylab.show()
