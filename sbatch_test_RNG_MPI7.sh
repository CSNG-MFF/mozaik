#!/bin/bash

#SBATCH -c 1 
#SBATCH -n 7
#SBATCH -J RNG_MPI7_test 
#SBATCH --hint=nomultithread

# Enter your virtual envirionment here
source PATH_TO_ENV

pytest -v -s -m 'mpi and not_github' tests/full_model/test_models_mpi.py
