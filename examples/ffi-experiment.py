#!/usr/bin/ipython 
import matplotlib
matplotlib.use('GTKAgg') # do this before importing pylab
from MozaikLite.framework.experiment import MeasureOrientationTuningFullfield, MeasureSpontaneousActivity
from pyNN import nest as sim
from MozaikLite.models.model import JensModel
from MozaikLite.framework.experiment_controller import run_experiments,run_analysis, setup_experiment

params = setup_experiment('FFI',sim)
jens_model = JensModel(sim,params)

experiment_list =   [
                       MeasureOrientationTuningFullfield(jens_model,num_orientations=2,spatial_frequency=0.8,temporal_frequency=2,grating_duration=7*7,num_trials=1),
                       #MeasureSpontaneousActivity(jens_model,duration=7*50)
                    ]

if True:
    run_experiments(jens_model,experiment_list)
else:
    run_analysis("FFI2011-10-26 17:18:59.677870",experiment_list)    

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import pylab

fig = pylab.figure(facecolor='w')
gs = gridspec.GridSpec(1, 1)
gs.update(left=0.05, right=0.95, top=0.95, bottom=0.05)
ax = pylab.subplot(gs[0,0])        
def f(x, y):
    return np.sin(x) + np.cos(y)

x = np.linspace(0, 2 * np.pi, 120)
y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

im = ax.imshow(f(x, y), cmap=plt.get_cmap('jet'))

def updatefig(*args):
    global x,y
    x += np.pi / 15.
    y += np.pi / 20.
    im.set_array(f(x,y))
    return im,

ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
pylab.show()
