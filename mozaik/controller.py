"""
This is the nexus of workflow execution controll of *mozaik*.
"""
from mozaik.storage.datastore import Hdf5DataStore, PickledDataStore
from mozaik.tools.distribution_parametrization import MozaikExtendedParameterSet, load_parameters
from mozaik.tools.misc import result_directory_name
import sys
import os
import mozaik
import time
from datetime import datetime
import logging
from NeuroTools import init_logging
from NeuroTools import visual_logging

logger = mozaik.getMozaikLogger()

class Global:
    """global variable container currently only containing the root_directory variable that points to the root directory of the model specification"""
    root_directory = './'

def setup_logging():
    """
    This functions sets up logging.
    """
    if mozaik.mpi_comm:
        init_logging(Global.root_directory + "log", file_level=logging.INFO,
                     console_level=logging.INFO, mpi_rank=mozaik.mpi_comm.rank)  # NeuroTools version
    else:
        init_logging(Global.root_directory + "log", file_level=logging.INFO,
	             console_level=logging.INFO)  # NeuroTools version
    if (not mozaik.mpi_comm) or mozaik.mpi_comm.rank==mozaik.MPI_ROOT:
	    visual_logging.basicConfig(Global.root_directory + "visual_log.zip",
        	                       level=logging.INFO)


def run_workflow(simulation_name, model_class, create_experiments):
    """
    This is the main function that executes a workflow. 
    
    It expects it gets the simulation, class of the model, and a function that will create_experiments.
    The create experiments function get a instance of a model as the only parameter and it is expected to return 
    a list of Experiment instances that should be executed over the model.
    
    The run workflow will automatically parse the command line to determine the simulator to be used and the path to the root parameter file. 
    It will also accept . (point) delimited path to parameteres in the configuration tree, and corresponding values. It will replace each such provided
    parameter's value with the provided one on the command line. 
    
    Parameters
    ----------
    simulation_name : str
                    The name of the simulation.
    
    model_class : class
                The class from which the model instance will be created from.
    
    create_experiments : func
                       The function that returns the list of experiments that will be executed on the model.
    
    Examples
    --------
    The intended syntax of the commandline is as follows (note that the simulation run name is the last argument):
    
    >>> python userscript simulator_name num_threads parameter_file_path modified_parameter_path_1 modified_parameter_value_1 ... modified_parameter_path_n modified_parameter_value_n simulation_run_name
    """
    mozaik.setup_mpi()
        # Read parameters
    exec "import pyNN.nest as sim" in  globals(), locals()
    
    if len(sys.argv) > 4 and len(sys.argv)%2 == 1:
        simulation_run_name = sys.argv[-1]    
        simulator_name = sys.argv[1]
        num_threads = sys.argv[2]
        parameters_url = sys.argv[3]
        modified_parameters = { sys.argv[i*2+4] : eval(sys.argv[i*2+5])  for i in xrange(0,(len(sys.argv)-5)/2)}
    else:
        raise ValueError("Usage: runscript simulator_name num_threads parameter_file_path modified_parameter_path_1 modified_parameter_value_1 ... modified_parameter_path_n modified_parameter_value_n simulation_run_name")
        p
    parameters = load_parameters(parameters_url,modified_parameters)
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    
    ddir  = result_directory_name(simulation_run_name,simulation_name,modified_parameters)
    
    if mozaik.mpi_comm and mozaik.mpi_comm.rank != 0:
        Global.root_directory = parameters.results_dir + ddir + '/' + str(mozaik.mpi_comm.rank) + '/'
        mozaik.mpi_comm.barrier()                                  
    else:
        Global.root_directory = parameters.results_dir + ddir + '/'
    
    
    os.makedirs(Global.root_directory)
    if mozaik.mpi_comm and mozaik.mpi_comm.rank == 0:
        mozaik.mpi_comm.barrier()
    
    #let's store the full and modified parameters, if we are the 0 rank process
    if mozaik.mpi_comm.rank == 0:
        parameters.save(Global.root_directory + "parameters", expand_urls=True)        
        import pickle
        f = open(Global.root_directory+"modified_parameters","w")
        pickle.dump(modified_parameters,f)
        f.close()

    setup_logging()
    
    model = model_class(sim,num_threads,parameters)
    data_store = run_experiments(model,create_experiments(model))

    if mozaik.mpi_comm.rank == 0:
	    data_store.save()

    import resource
    print "Final memory usage: %iMB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024))
    return (data_store,model)

def run_experiments(model,experiment_list,load_from=None):
    """
    This is function called by :func:.run_workflow that executes the experiments in the `experiment_list` over the model. 
    Alternatively, if load_from is specified it will load an existing simulation from the path specified in load_from.
    
    Parameters
    ----------
    
    model : Model
          The model to execute experiments on.
    
    experiment_list : list
          The list of experiments to execute.
          
    load_from : str
              If not None it will load the simulation from the specified directory.
              
    Returns
    -------
    
    data_store : DataStore
               The data store containing the recordings.
    """
    
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
