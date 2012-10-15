"""
docstring goes here

"""

from mozaik.storage.datastore import Hdf5DataStore, PickledDataStore
from NeuroTools.parameters import ParameterSet
import sys
import os
import mozaik
import time
from datetime import datetime
from NeuroTools import logging
from NeuroTools import init_logging
from NeuroTools import visual_logging

logger = mozaik.getMozaikLogger("Mozaik")

try:
    from mpi4py import MPI
except ImportError:
    MPI = None
if MPI:
    mpi_comm = MPI.COMM_WORLD


class Global:
    """global variable container"""
    root_directory = './'


def setup_logging():
    # Set-up logging
    if MPI:
        init_logging(Global.root_directory + "log", file_level=logging.DEBUG,
                     console_level=logging.DEBUG, mpi_rank=mpi_comm.rank)  # NeuroTools version
    else:
        init_logging(Global.root_directory + "log", file_level=logging.DEBUG,
                     console_level=logging.DEBUG)  # NeuroTools version
    visual_logging.basicConfig(Global.root_directory + "visual_log.zip",
                               level=logging.INFO)


def setup_experiments(simulation_name, sim):
    # Read parameters
    if len(sys.argv) > 1:
        parameters_url = sys.argv[1]
    else:
        raise ValueError("No parameter file supplied")

    parameters = ParameterSet(parameters_url)

    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    Global.root_directory = parameters.results_dir + simulation_name + '_' + \
                              timestamp + 'rank' + str(mpi_comm.rank) + '/'
    os.mkdir(Global.root_directory)
    parameters.save(Global.root_directory + "parameters", expand_urls=True)

    setup_logging()
    return parameters


def run_experiments(model,experiment_list,load_from=None):
    # first lets run all the measurements required by the experiments
    logger.info('Starting Experiemnts')
    if load_from == None:
        data_store = PickledDataStore(load=False,
                                      parameters=ParameterSet({'root_directory': Global.root_directory}))
    else:
        data_store = PickledDataStore(load=True,
                                      parameters=ParameterSet({'root_directory': load_from}))
    
    data_store.set_neuron_positions(model.neuron_positions())
    data_store.set_neuron_annotations(model.neuron_annotations())
    
    
    t0 = time.time()
    simulation_run_time=0
    for i,experiment in enumerate(experiment_list):
        logger.info('Starting experiment: ' + experiment.__class__.__name__)
        stimuli = experiment.return_stimuli()
        unpresented_stimuli = data_store.identify_unpresented_stimuli(stimuli)
        logger.info('Running model')
        simulation_run_time += experiment.run(data_store,unpresented_stimuli)
        logger.info('Experiment %d/%d finished' % (i+1,len(experiment_list)))
    
    total_run_time = time.time() - t0
    mozaik_run_time = total_run_time - simulation_run_time
    
    logger.info('Total simulation run time: %.0fs' % total_run_time)
    logger.info('Simulator run time: %.0fs (%d%%)' % (simulation_run_time, int(simulation_run_time /total_run_time * 100)))
    logger.info('Mozaik run time: %.0fs (%d%%)' % (mozaik_run_time, int(mozaik_run_time /total_run_time * 100)))
    
    return data_store
