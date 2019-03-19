# -*- coding: utf-8 -*-
"""
This is implementation of model of push-pull connectvity: 
Jens Kremkow: Correlating Excitation and Inhibition in Visual Cortical Circuits: Functional Consequences and Biological Feasibility. PhD Thesis, 2009.
"""
from pyNN import nest
import sys
sys.path.insert(0,"/home/jan/cluster/mozaik/mozaik/")


if False:
    try:
        from mpi4py import MPI
    except ImportError:
        MPI = None
    if MPI:
	mpi_comm = MPI.COMM_WORLD
    MPI_ROOT = 0

    


from model import PushPullCCModel
from experiments import create_experiments


from parameters import ParameterSet
import mozaik
from mozaik.controller import run_workflow, setup_logging
from mozaik.storage.datastore import Hdf5DataStore,PickledDataStore
print mozaik.__file__
print sys.path





logger = mozaik.getMozaikLogger()

if True:
    data_store,model = run_workflow('FeedForwardInhibition',PushPullCCModel,create_experiments)
    #model.connectors['V1L4ExcL4ExcConnection'].store_connections(data_store)    
    #model.connectors['V1L4ExcL4InhConnection'].store_connections(data_store)    
    #model.connectors['V1L4InhL4ExcConnection'].store_connections(data_store)    
    #model.connectors['V1L4InhL4InhConnection'].store_connections(data_store)    
    #model.connectors['V1AffConnectionOn'].store_connections(data_store)    
    #model.connectors['V1AffConnectionOff'].store_connections(data_store)    
    #model.connectors['V1AffInhConnectionOn'].store_connections(data_store)    
    #model.connectors['V1AffInhConnectionOff'].store_connections(data_store)    
    data_store.save()
    from analysis_and_visualization import perform_analysis_and_visualization
    perform_analysis_and_visualization(data_store)
    
else: 
    setup_logging()
    data_store = PickledDataStore(load=True,parameters=ParameterSet({'root_directory':'FeedForwardInhibition_test_____', 'store_stimuli' : False}),replace=True)
    logger.info('Loaded data store')
    #data_store.save()
    from analysis_and_visualization import perform_analysis_and_visualization
    perform_analysis_and_visualization(data_store)


