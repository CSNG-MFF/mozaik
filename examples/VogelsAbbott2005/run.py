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
import logging
import sys

from mozaik.cli import parse_workflow_args
from mozaik.controller import run_workflow, setup_logging
from mozaik.storage.datastore import Hdf5DataStore, PickledDataStore
from mozaik.tools.misc import result_directory_name
from mpi4py import MPI
from parameters import ParameterSet
from pyNN import nest
import matplotlib

from .analysis_and_visualization import perform_analysis_and_visualization
from .experiments import create_experiments
from .model import VogelsAbbott

matplotlib.use("Agg")
# mpi_comm = MPI.COMM_WORLD
logger = logging.getLogger(__name__)
simulation_name = "VogelsAbbott2005"
simulation_run_name, _, _, _, modified_parameters = parse_workflow_args()

if True:
    data_store, model = run_workflow(simulation_name, VogelsAbbott, create_experiments)
else:
    setup_logging()
    data_store = PickledDataStore(
        load=True,
        parameters=ParameterSet(
            {
                "root_directory": result_directory_name(
                    simulation_run_name, simulation_name, modified_parameters
                ),
                "store_stimuli": False,
            }
        ),
        replace=True,
    )
    logger.info("Loaded data store")

# if mpi_comm.rank == 0:
print("Starting visualization")
perform_analysis_and_visualization(data_store)
data_store.save()
