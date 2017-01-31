# -*- coding: utf-8 -*-
import sys
from mozaik.meta_workflow.parameter_search import CombinationParameterSearch,SlurmSequentialBackend,LocalSequentialBackend
import numpy
CombinationParameterSearch(LocalSequentialBackend(),{'sheets.exc_layer.ExcExcConnection.weights' : numpy.linspace(0.0,0.006,5),'sheets.inh_layer.InhExcConnection.weights' : numpy.linspace(0.0,0.06,5),'sheets.exc_layer.params.cell.params.tau_m' : [10,11,12],'sheets.exc_layer.params.cell.params.cm' : [0.2,0.1]}).run_parameter_search()
