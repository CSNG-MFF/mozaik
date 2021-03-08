# -*- coding: utf-8 -*-
import sys
from mozaik.meta_workflow.parameter_search import CombinationParameterSearch, SlurmSequentialBackend
import numpy
import time

slurm_options = [
        '-J VogelsAbbott',
        '--hint=nomultithread',
        ]

if True:
    CombinationParameterSearch(SlurmSequentialBackend(num_threads=1, num_mpi=2, path_to_mozaik_env= '/home/cagnol/virtenv/mozaik_mpi', slurm_options=slurm_options),{}).run_parameter_search()
