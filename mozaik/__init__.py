"""
The role of mozaik is to coordinate the workings of number of tools to provide a 
consistent workflow experience for the user. Consequently the root mozaik package is very light,
and majority of functionality is in the number of subpackages each addressing different parts of the workflow. 
In future the number of subpackages is likely to grow, as the number of 
areas that the mozaik workflow covers increases. It is also likely that 
in future some of the subpackages will be removed (or replaced with dedicated packages)
as the individual external tools overcome the 'coordination' issues that 
mozaik is currently trying to address.
"""

__version__ = None

rng = None

def setup_mpi():
    """
    Tests the presence of MPI and sets up mozaik wide random number generator.
    
    To obtain results repeatable over identical runs of mozaik
    one should use the mozaik.rng as the random noise generator passed to all pyNN
    functions that accept rng as one of their paramters, and also use it for any auxiallary code 
    that requires rng
    """

    global rng
    from pyNN.random import NumpyRNG
    rng = NumpyRNG(seed=1023)
    try:
        from mpi4py import MPI
    except ImportError:
        MPI = None
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
