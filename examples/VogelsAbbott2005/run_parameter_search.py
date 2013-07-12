# -*- coding: utf-8 -*-
import sys
from mozaik.meta_workflow.parameter_search import CombinationParameterSearch,SlurmSequentialBackend
import numpy
CombinationParameterSearch(SlurmSequentialBackend(num_threads=1,num_mpi=1),{'exc_layer.ExcExcConnection.weights' : numpy.linspace(0.0,0.006,20),'inh_layer.InhExcConnection.weights' : numpy.linspace(0.0,0.06,20)}).run_parameter_search()
