"""
This is the nexus of workflow execution controll of *mozaik*.
"""
from mozaik.cli import parse_workflow_args
from mozaik.storage.datastore import Hdf5DataStore, PickledDataStore
from mozaik.tools.distribution_parametrization import MozaikExtendedParameterSet, load_parameters
from mozaik.tools.misc import result_directory_name
from mozaik.stimuli import EndOfSimulationBlank
from collections import OrderedDict
import sys
import os
import mozaik
import time
from datetime import datetime
import logging
from mozaik.tools.json_export import save_json, get_experimental_protocols, get_recorders, get_stimuli
from parameters import ParameterSet


logger = mozaik.getMozaikLogger()

class Global:
    """global variable container currently only containing the root_directory variable that points to the root directory of the model specification"""
    root_directory = './'

class FancyFormatter(logging.Formatter):
    """
    A log formatter that colours and indents the log message depending on the level.
    """
    
    DEFAULT_INDENTS = {
        'CRITICAL': "",
        'ERROR': "",
        'WARNING': "",
        'HEADER': "",
        'INFO': "  ",
        'DEBUG': "    ",
    }
    
    def __init__(self, fmt=None, datefmt=None, mpi_rank=None):
        logging.Formatter.__init__(self, fmt, datefmt)
        self._indents = FancyFormatter.DEFAULT_INDENTS
        if mpi_rank is None:
            self.prefix = ""
        else:
            self.prefix = "%-3d" % mpi_rank
    
    def format(self, record):
        s = logging.Formatter.format(self, record)
        if record.levelname == "HEADER":
            s = "=== %s ===" % s
        return self.prefix + self._indents[record.levelname] + s


def init_logging(filename, file_level=logging.INFO, console_level=logging.WARNING, mpi_rank=None):
    if mpi_rank is None:
        mpi_fmt = ""
    else:
        mpi_fmt = "%3d " % mpi_rank
    logging.basicConfig(level=file_level,
                        format='%%(asctime)s %s%%(name)-10s %%(levelname)-6s %%(message)s [%%(pathname)s:%%(lineno)d]' % mpi_fmt,
                        filename=filename,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(FancyFormatter('%(message)s', mpi_rank=mpi_rank))
    logging.getLogger('').addHandler(console)
    return console



def setup_logging():
    """
    This functions sets up logging.
    """
    if mozaik.mpi_comm:
        init_logging(Global.root_directory + "log", file_level=logging.INFO,
                     console_level=logging.INFO, mpi_rank=mozaik.mpi_comm.rank)  
    else:
        init_logging(Global.root_directory + "log", file_level=logging.INFO,
                 console_level=logging.INFO)  


def prepare_workflow(simulation_name, model_class):
    """
    
    Executes the following preparatory steps for simulation workflow:

    - Load simulation parameters
    - Initialize random seeds
    - Create directory for results
    - Store loaded parameters
    - Setup logging
    - Store some initial info about the simulation

    Returns
    -------

    
    sim : module
        NEST module, to use for simulation

    num_threads : int
        Number of threads to use for the simulation

    parameters : dict
        Loaded parameters to initialize the simulation and model with
                 
    """
    (
        simulation_run_name,
        simulator_name,
        num_threads,
        parameters_url,
        modified_parameters,
    ) = parse_workflow_args()


    # First we load the parameters just to retrieve seeds. We will throw them away, because at this stage the PyNNDistribution values were not yet initialized correctly.
    parameters = load_parameters(parameters_url,modified_parameters)
    p=OrderedDict()
    if 'mozaik_seed' in parameters : p['mozaik_seed'] = parameters['mozaik_seed']
    if 'pynn_seed' in parameters : p['pynn_seed'] = parameters['pynn_seed']

    # Now initialize mpi with the seeds
    print("START MPI")
    mozaik.setup_mpi(**p)

    # Now really load parameters
    print("Loading parameters")
    parameters = load_parameters(parameters_url,modified_parameters)
    print("Finished loading parameters")

    import pyNN.nest as sim

    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

    ddir  = result_directory_name(simulation_run_name,simulation_name,modified_parameters)

    if mozaik.mpi_comm and mozaik.mpi_comm.rank != mozaik.MPI_ROOT:
        Global.root_directory = parameters.results_dir + ddir + '/' + str(mozaik.mpi_comm.rank) + '/'
        mozaik.mpi_comm.barrier()
    else:
        Global.root_directory = parameters.results_dir + ddir + '/'


    os.makedirs(Global.root_directory)
    if mozaik.mpi_comm and mozaik.mpi_comm.rank == mozaik.MPI_ROOT:
        mozaik.mpi_comm.barrier()


    if mozaik.mpi_comm.rank == mozaik.MPI_ROOT:
        # Store simulation run info, if we are the 0 rank process,
        # with several components to be stored/filled in later during the simulation run
        sim_info = {
            'submission_date' : None,
            'run_date': datetime.now().strftime('%d/%m/%Y-%H:%M:%S'),
            'simulation_run_name': simulation_run_name,
            'model_name': simulation_name,
            "model_description": model_class.__doc__,
            'results': {"$ref": "results.json"},
            'stimuli': {"$ref": "stimuli.json"},
            'recorders': {"$ref": "recorders.json"},
            'experimental_protocols': {"$ref": "experimental_protocols.json"},
            'parameters': {"$ref": "parameters.json"},
        }
        save_json(sim_info, Global.root_directory + 'sim_info.json')
        save_json(parameters.to_dict(), Global.root_directory + 'parameters.json')
        save_json(modified_parameters, Global.root_directory + 'modified_parameters.json')
        recorders = get_recorders(parameters.to_dict())
        save_json(recorders, Global.root_directory + 'recorders.json')

    setup_logging()

    return sim, num_threads, parameters

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

    # Prepare workflow - read parameters, setup logging, etc.
    sim, num_threads, parameters = prepare_workflow(simulation_name, model_class)
    # Prepare model to run experiments on
    model = model_class(sim,num_threads,parameters)
    # Run experiments with previously read parameters on the prepared model
    data_store = run_experiments(model,create_experiments(model),parameters)

    if mozaik.mpi_comm.rank == mozaik.MPI_ROOT:
        data_store.save()
    import resource
    print("Final memory usage: %iMB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024)))
    return (data_store, model)

def run_experiments(model,experiment_list,parameters,load_from=None):
    """
    This is function called by :func:.run_workflow that executes the experiments in the `experiment_list` over the model. 
    Alternatively, if load_from is specified it will load an existing simulation from the path specified in load_from.
    
    Parameters
    ----------
    
    model : Model
        The model to execute experiments on.
    
    experiment_list : list
        The list of experiments to execute.
    
    parameters : ParameterSet
        The parameters given to the simulation run.
          
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
                                      parameters=MozaikExtendedParameterSet({'root_directory': Global.root_directory,'store_stimuli' : parameters.store_stimuli}))
    else: 
        data_store = PickledDataStore(load=True,
                                      parameters=MozaikExtendedParameterSet({'root_directory': load_from,'store_stimuli' : parameters.store_stimuli}))
    
    data_store.set_neuron_ids(model.neuron_ids())
    data_store.set_neuron_positions(model.neuron_positions())
    data_store.set_neuron_annotations(model.neuron_annotations())
    data_store.set_model_parameters(parameters.pretty(expand_urls=True))
    data_store.set_sheet_parameters(MozaikExtendedParameterSet(model.sheet_parameters()).pretty(expand_urls=True))
    data_store.set_experiment_parametrization_list([(str(exp.__class__),str(exp.parameters)) for exp in experiment_list])
    
    t0 = time.time()
    simulation_run_time=0
    model_exploded=False
    for i,experiment in enumerate(experiment_list):
        logger.info('Starting experiment: ' + experiment.__class__.__name__)
        stimuli = experiment.return_stimuli()
        unpresented_stimuli_indexes = data_store.identify_unpresented_stimuli(stimuli)
        logger.info('Running model')
        experiment_run_time, model_exploded = experiment.run(data_store,unpresented_stimuli_indexes)
        simulation_run_time += experiment_run_time
        if model_exploded:
            logger.info('ERROR: Model exploded, stopping simulation!')
            break
        logger.info('Experiment %d/%d finished' % (i+1,len(experiment_list)))

    last_blank_run_time = 0
    # Do a reset after the last stimulus. If reset is done as blank stimulus, this makes sure we have some blank recorded also after last stimulus.
    ds =  OrderedDict()

    if parameters.null_stimulus_period != 0:
        s = EndOfSimulationBlank(trial=0,duration=parameters.null_stimulus_period,frame_duration=parameters.null_stimulus_period)
        (segments,null_segments,input_stimulus,last_blank_run_time,_) = model.present_stimulus_and_record(s,ds)
        data_store.add_recording(segments,s)
        data_store.add_stimulus(input_stimulus,s)
        data_store.add_direct_stimulation(ds,s)
        if null_segments != []:
                data_store.add_null_recording(null_segments,s) 
    else:
        last_blank_run_time = 0
        
    total_run_time = time.time() - t0
    mozaik_run_time = total_run_time - simulation_run_time - last_blank_run_time

    # Adding the state (represented by a randomly generated number) of the rng of every MPI process to the datastore
    if mozaik.mpi_comm:
        rngs_state = mozaik.mpi_comm.gather(float(mozaik.rng.rand(1)), root=0)
        log = {'rngs_state': rngs_state, 'explosion_detected': model_exploded}
    else:
        log = {'explosion_detected': model_exploded}
    data_store.set_simulation_log(log)

    if not model_exploded and mozaik.mpi_comm.rank == mozaik.MPI_ROOT:
        logger.info('Total simulation run time: %.0fs' % total_run_time)
        logger.info('Simulator run time: %.0fs (%d%%)' % (simulation_run_time, int(simulation_run_time /total_run_time * 100)))
        logger.info('Mozaik run time: %.0fs (%d%%)' % (mozaik_run_time, int(mozaik_run_time /total_run_time * 100)))
    
    experimental_protocols = get_experimental_protocols(data_store)
    stimuli = get_stimuli(data_store,parameters.store_stimuli, parameters.input_space)
    save_json(experimental_protocols, Global.root_directory + 'experimental_protocols.json')
    save_json(stimuli, Global.root_directory + 'stimuli.json')

    return data_store
