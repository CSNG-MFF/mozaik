from MozaikLite.storage.datastore import Hdf5DataStore,PickledDataStore
from NeuroTools.parameters import ParameterSet
import sys 
import os 
from datetime import datetime
from NeuroTools import logging
from NeuroTools import init_logging
from NeuroTools import visual_logging

global root_directory
root_directory = './'

def setup_experiment(simulation_name,sim):
    global root_directory
    # Read parameters
    if len(sys.argv) > 1:
        parameters_url = sys.argv[1]
    else:
        raise ValueError , "No parameter file supplied"
    
    parameters = ParameterSet(parameters_url) 
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    root_direcotory =  parameters.results_dir + simulation_name + '_' + timestamp + '/'
    os.mkdir(root_direcotory)
    parameters.save(root_direcotory + "parameters", expand_urls=True)
    
    logger = logging.getLogger("MozaikLite")
    
    # Set-up logging
    init_logging(root_direcotory + "log", file_level=logging.DEBUG, console_level=logging.DEBUG) # NeuroTools version
    visual_logging.basicConfig(root_direcotory + "visual_log.zip", level=logging.DEBUG)
    
    logger.info("Creating Model object using the %s simulator." % sim.__name__)
    return parameters
 

def run_experiments(model,experiment_list):
    # first lets run all the measurements required by the experiments
    print 'Starting Experiemnts'
    data_store = PickledDataStore(load=False,parameters=ParameterSet({'root_directory':root_directory}))
        
    for experiment in experiment_list:
        print 'Starting experiment: ', experiment.__class__.__name__
        stimuli = experiment.return_stimuli()
        unpresented_stimuli = data_store.identify_unpresented_stimuli(stimuli)
        print 'Running model'
        experiment.run(data_store,unpresented_stimuli)
    
    print 'Starting Analysis'
    # next lets perform the corresponding analysis
    for experiment in experiment_list:    
        experiment.do_analysis(data_store)

    print 'Saving Datastore'
    data_store.save()




def run_analysis(root_directory,experiment_list):    
    data_store = PickledDataStore(load=True,parameters=ParameterSet({'root_directory':root_directory}))
    
    print 'Starting Analysis'
    # next lets perform the corresponding analysis
    for experiment in experiment_list:    
        experiment.do_analysis(data_store)
