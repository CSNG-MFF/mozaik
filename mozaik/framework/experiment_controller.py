from mozaik.storage.datastore import Hdf5DataStore,PickledDataStore
from NeuroTools.parameters import ParameterSet
import sys 
import os 
from datetime import datetime
from NeuroTools import logging
from NeuroTools import init_logging
from NeuroTools import visual_logging

class Global:
    """global variable container"""
    root_directory = './'

def setup_experiments(simulation_name,sim):
    # Read parameters
    if len(sys.argv) > 1:
        parameters_url = sys.argv[1]
    else:
        raise ValueError , "No parameter file supplied"
    
    parameters = ParameterSet(parameters_url) 
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    Global.root_directory =  parameters.results_dir + simulation_name + '_' + timestamp + '/'
    os.mkdir(Global.root_directory)
    parameters.save(Global.root_directory + "parameters", expand_urls=True)
    
    logger = logging.getLogger("mozaik")
    
    # Set-up logging
    init_logging(Global.root_directory + "log", file_level=logging.DEBUG, console_level=logging.INFO) # NeuroTools version
    visual_logging.basicConfig(Global.root_directory + "visual_log.zip", level=logging.INFO)
    
    logger.info("Creating Model object using the %s simulator." % sim.__name__)
    return parameters
 

def run_experiments(model,experiment_list):
    # first lets run all the measurements required by the experiments
    print 'Starting Experiemnts'
    data_store = PickledDataStore(load=False,parameters=ParameterSet({'root_directory':Global.root_directory}))
    data_store.set_neuron_positions(model.neuron_positions())
    data_store.set_neuron_annotations(model.neuron_annotations())
    
    for experiment in experiment_list:
        print 'Starting experiment: ', experiment.__class__.__name__
        stimuli = experiment.return_stimuli()
        unpresented_stimuli = data_store.identify_unpresented_stimuli(stimuli)
        print 'Running model'
        
        experiment.run(data_store,unpresented_stimuli)
    
    print 'Saving Datastore'
    data_store.save()
    return data_store

