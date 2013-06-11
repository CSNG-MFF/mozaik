"""
The role of mozaik is to coordinate the workings of number of tools to provide a 
consistent workflow experience for the user. Consequently the root mozaik package is very light,
and majority of functionality is in the number of subpackages each addressing different parts of the workflow. 
In future the number of subpackages is likely to grow, as the number of 
areas that the mozaik workflow covers increases. It is also likely that 
in future some of the subpackages will be removed (or replaced with dedicated packages)
as the individual external tools overcome the 'coordination' issues that 
mozaik is currently trying to address.

This module exposes several parameters to the rest of mozaik:

Parameters
----------
    rng : nest.RandomState
        The global mozaik random number generator. It is crucially any mozaik code using it has to make sure that it will ensure that the 
        random number generator will be in the same state on all processes after the codes execution.
    pynn_rng : pynn.random.NumpyRNG
        The random number generator that should be passed to all pynn objects requiring rng.
    
    mpi_comm : mpi4py.Comm
             The mpi communication object, None if MPI not available.
"""

__version__ = None
import numpy.random
rng = None
pynn_rng = None
mpi_comm = None

def setup_mpi():
    """
    Tests the presence of MPI and sets up mozaik wide random number generator.
    
    Notes
    -----
    
    To obtain results repeatable over identical runs of mozaik
    one should use the mozaik.rng as the random noise generator passed to all pyNN
    functions that accept pynn_rng as one of their paramters
    
    Any other code using random numbers should instead use the mozaik.rng that hold a numpy RandomState instance.
    It is important to make sure that any piece of  code using this random generator draws from it 
    exactly the same number of numbers in each process, so that once the code is executed, the rng 
    is in exactly the same state in each mpi process!
    """
    global rng
    global pynn_rng
    global mpi_comm
    from pyNN.random import NumpyRNG
    pynn_rng = NumpyRNG(seed=1023)
    rng = numpy.random.RandomState()
    rng.seed(513)
    try:
        from mpi4py import MPI
    except ImportError:
        mpi_comm = None
    if MPI:
        mpi_comm = MPI.COMM_WORLD

def getMozaikLogger(name):
    """
    To maintain consistent logging settings around mozaik use this method to obtain the logger isntance.
    """
    import logging
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
