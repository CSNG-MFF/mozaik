#!/home/antolikjan/virt_envs/mozaik/bin/python
from MozaikLite.framework.experiment import MeasureOrientationTuningFullfield, MeasureSpontaneousActivity
from MozaikLite.models.model import JensModel
from MozaikLite.framework.experiment_controller import run_experiments
import datetime
import os 

from pyNN import nest as sim
jens_model = JensModel(sim)

root_directory = 'FFI' + str(datetime.datetime.now())
os.mkdir(root_directory)

experiment_list =   [
                       MeasureOrientationTuningFullfield(jens_model,2,0.8,2,143*7)
                    ]

run_experiments(jens_model,root_directory,experiment_list)

import pylab
pylab.show()
