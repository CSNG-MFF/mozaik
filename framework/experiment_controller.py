from MozaikLite.storage.datastore import Hdf5DataStore
from NeuroTools.parameters import ParameterSet

def run_experiments(model,root_directory,experiment_list):
    # first lets run all the measurements required by the experiments
    print 'Starting Experiemnts'
    data_store = Hdf5DataStore(load=False,parameters=ParameterSet({'root_directory':root_directory}))
        
    for experiment in experiment_list:
        print 'Starting experiment: ', experiment.__class__.__name__
        stimuli = experiment.return_stimuli()
        unpresented_stimuli = data_store.identify_unpresented_stimuli(stimuli)
        print 'Running model'
        experiment.run(model,data_store,unpresented_stimuli)
    
    print 'Starting Analysis'
    # next lets perform the corresponding analysis
    for experiment in experiment_list:    
        experiment.do_analysis(data_store)

    print 'Saving Datastore'
    data_store.save()




def run_analysis(root_directory,experiment_list):    
    
    data_store = Hdf5DataStore(load=True,parameters=ParameterSet({'root_directory':root_directory}))
    
    print 'Starting Analysis'
    # next lets perform the corresponding analysis
    for experiment in experiment_list:    
        experiment.do_analysis(data_store)
