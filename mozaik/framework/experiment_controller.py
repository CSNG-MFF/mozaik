"""
docstring goes here

"""
from mozaik.storage.datastore import Hdf5DataStore, PickledDataStore
from mozaik.tools.distribution_parametrization import MozaikExtendedParameterSet
import sys
import os
import mozaik
import time
from datetime import datetime
from NeuroTools import logging
from NeuroTools import init_logging
from NeuroTools import visual_logging

import pyNN.nest as sim

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


def run_workflow(simulation_name, model_class, create_experiments):
    # Read parameters
    #exec("import pyNN.nest as sim" )
    
    if len(sys.argv) > 2 and len(sys.argv)%2 == 1:
        simulator_name = sys.argv[1]
        parameters_url = sys.argv[2]
        modified_params = { sys.argv[i*2+3] : sys.argv[i*2+4]  for i in xrange(0,(len(sys.argv)-3)/2)}
    else:
        raise ValueError("Usage: runscript simulator_name parameter_file_path modified_parameter_path_1 modified_parameter_value_1 ... modified_parameter_path_n modified_parameter_value_n")
    
    parameters = MozaikExtendedParameterSet(parameters_url)
    parameters.replace_values(**modified_params)
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    
    
    modified_params_str = [str(k) + ":" + str(modified_parameters[k]) for k in modified_parameters.keys()].join('_')
    Global.root_directory = parameters.results_dir + simulation_name + '_' + \
                              timestamp + 'rank' + str(mpi_comm.rank) + '_' + modified_params_str + '/'
    os.mkdir(Global.root_directory)
    parameters.save(Global.root_directory + "parameters", expand_urls=True)
    
    #let's store the modified parameters
    import pickle
    f = open(Global.root_directory+"modified_parameters","w")
    pickle.dump(modified_params,f)
    f.close()
    setup_logging()
    
    model = model_class(sim,parameters)
    data_store = run_experiments(model,create_experiments(model))
    data_store.save()
    import resource
    print "Final memory usage: %iMB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024))
    return data_store

def run_experiments(model,experiment_list,load_from=None):
    # first lets run all the measurements required by the experiments
    logger.info('Starting Experiemnts')
    if load_from == None:
        data_store = PickledDataStore(load=False,
                                      parameters=MozaikExtendedParameterSet({'root_directory': Global.root_directory}))
    else:
        data_store = PickledDataStore(load=True,
                                      parameters=MozaikExtendedParameterSet({'root_directory': load_from}))
    
    data_store.set_neuron_ids(model.neuron_ids())
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
