# -*- coding: utf-8 -*-
"""
This is implementation of model of self-sustained activitity in balanced networks from: 
Vogels, T. P., & Abbott, L. F. (2005). 
Signal propagation and logic gating in networks of integrate-and-fire neurons. 
The Journal of neuroscience : the official journal of the Society for Neuroscience, 25(46), 10786–95. 

To run it, use:
mpirun python run.py simulator_name number_processors parameters name_of_test

For example:
mpirun python run.py nest 2 param/defaults 'test'
"""
#from mpi4py import MPI 
from pyNN import nest
import sys
import mozaik
from mozaik.controller import run_workflow, setup_logging
from experiments import create_experiments
from model import VogelsAbbott
from mozaik.storage.datastore import Hdf5DataStore,PickledDataStore
from analysis_and_visualization import perform_analysis_and_visualization
from parameters import ParameterSet

#mpi_comm = MPI.COMM_WORLD
logger = mozaik.getMozaikLogger()

if True:
    data_store,model = run_workflow('VogelsAbbott2005',VogelsAbbott,create_experiments)
else: 
    setup_logging()
    data_store = PickledDataStore(load=True,parameters=ParameterSet({'root_directory':'VogelsAbbott2005_test_____', 'store_stimuli' : False}),replace=True)
    logger.info('Loaded data store')

#if mpi_comm.rank == 0:
print "Starting visualization" 
perform_analysis_and_visualization(data_store)
data_store.save() 
