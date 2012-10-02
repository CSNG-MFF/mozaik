"""
docstring goes here
"""

__version__ = None

# To obtain results repeatable over identical runs of mozaik
# one should use the mozaik.rng as the random noise generator passed to all pyNN
# functions that accept rng as one of their paramters, and also use it for any auxiallary code 
# that requires rng

from pyNN.random import NumpyRNG
rng = NumpyRNG(seed=1023)

try:
    from mpi4py import MPI
except ImportError:
    MPI = None
if MPI:
    mpi_comm = MPI.COMM_WORLD

def getMozaikLogger(name):
    import logging
    logger = logging.getLogger(name)
    return logger
