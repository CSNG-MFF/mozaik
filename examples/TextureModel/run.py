# -*- coding: utf-8 -*-
"""
This is implementation of model of push-pull connectvity:
Jens Kremkow: Correlating Excitation and Inhibition in Visual Cortical Circuits: Functional Consequences and Biological Feasibility. PhD Thesis, 2009.
"""
import logging
import sys

from mozaik.controller import run_workflow, setup_logging
from mozaik.storage.datastore import Hdf5DataStore, PickledDataStore
from parameters import ParameterSet
from pyNN import nest

from analysis_and_visualization import perform_analysis_and_visualization
from experiments import create_experiments
from model import PushPullCCModel

if False:
    try:
        from mpi4py import MPI

        mpi_comm = MPI.COMM_WORLD
        MPI_ROOT = 0
    except ImportError:
        MPI = None

logger = logging.getLogger(__name__)

if True:
    data_store, model = run_workflow(
        "FeedForwardInhibition", PushPullCCModel, create_experiments
    )
    # model.connectors['V1L4ExcL4ExcConnection'].store_connections(data_store)
    # model.connectors['V1L4ExcL4InhConnection'].store_connections(data_store)
    # model.connectors['V1L4InhL4ExcConnection'].store_connections(data_store)
    # model.connectors['V1L4InhL4InhConnection'].store_connections(data_store)
    # model.connectors['V1AffConnectionOn'].store_connections(data_store)
    # model.connectors['V1AffConnectionOff'].store_connections(data_store)
    # model.connectors['V1AffInhConnectionOn'].store_connections(data_store)
    # model.connectors['V1AffInhConnectionOff'].store_connections(data_store)
    data_store.save()
    perform_analysis_and_visualization(data_store)

else:
    setup_logging()
    data_store = PickledDataStore(
        load=True,
        parameters=ParameterSet(
            {
                "root_directory": "FeedForwardInhibition_test_____",
                "store_stimuli": False,
            }
        ),
        replace=True,
    )
    logger.info("Loaded data store")
    # data_store.save()
    perform_analysis_and_visualization(data_store)
