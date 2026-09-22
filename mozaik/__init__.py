r"""
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

    model_rng : numpy.random.RandomState
        The global model-construction random number generator. Crucially, any mozaik code using it has to make sure that it will ensure that the
        random number generator will be in the same state on all processes after the code's execution.

    simulation_rng : numpy.random.RandomState
        The global simulation-time random number generator.

    experiment_rng : numpy.random.RandomState
        The global experiment-ordering random number generator.

    model_pynn_rng : pynn.random.NumpyRNG
        The model-seeded random number generator that should be passed to all pynn objects requiring rng.

    mpi_comm : mpi4py.Comm
        The mpi communication object, None if MPI not available.


"""
__version__ = "0.1.0"
import numpy.random
model_rng = None
model_pynn_rng = None
simulation_rng = None
experiment_rng = None
_seed_setup_locked = False
mpi_comm = None
MPI_ROOT = 0


def setup_seeds(
    model_seed,
    simulation_seed,
    experiment_seed,
    prevent_reinitialization=False,
):
    r"""
    Set up Mozaik's model, simulation, and experiment random-number streams.

    Parameters
    ----------
    model_seed : int
        Seed for model construction, including PyNN connection patterns,
        neuron positions, and parameter distributions.
    simulation_seed : int
        Seed for simulation-time randomness, including stochastic input.
    experiment_seed : int
        Seed for experiment-level randomness, including stimulus ordering.
    prevent_reinitialization : bool
        If True, lock seed setup after this call. Any later call to
        ``setup_seeds`` will raise ``RuntimeError``.

    Notes
    -----

    To obtain results repeatable over identical runs of mozaik
    one should use the mozaik.model_pynn_rng as the random noise generator passed to all pyNN
    functions that accept an RNG as one of their parameters

    Model-construction code using random numbers should use ``mozaik.model_rng``.
    Simulation-time code should use ``mozaik.simulation_rng``. It is important
    that code using either generator draws exactly the same number of values in
    each process, so that the stream remains synchronized across MPI processes.

    """

    global model_rng
    global simulation_rng
    global experiment_rng
    global model_pynn_rng
    global _seed_setup_locked
    from pyNN.random import NumpyRNG

    if _seed_setup_locked:
        raise RuntimeError("Mozaik random-number streams are already initialized")

    model_pynn_rng = NumpyRNG(seed=model_seed)
    model_rng = numpy.random.RandomState(model_seed)
    simulation_rng = numpy.random.RandomState(simulation_seed)
    experiment_rng = numpy.random.RandomState(experiment_seed)
    _seed_setup_locked = prevent_reinitialization


def setup_mpi():
    r"""Set up the global MPI communicator when mpi4py is available."""
    global mpi_comm

    try:
        from mpi4py import MPI
    except ImportError:
        mpi_comm = None
    else:
        mpi_comm = MPI.COMM_WORLD



def get_model_seeds(size=None):
    r"""
    This methods returns a set of inetegers that can be used as random seeds for RNGs. The main purpose
    is that these numbers are large and random, with extremely low probability that two of the same numbers
    are returned in a single simulation run.
    
    Returns
    -------

    A set of long integer as a ndarray of shape size. If size==None returns single seed. The integers have 64bit size.
    
    Notes
    -----

    We recommand users to use this method whenever seeding a new random generator. It is 
    important that the same number of seeds are requested in each MPI process to ensure 
    reproducability of simulations!

    """
    return model_rng.randint(2**32-1,size=size)


def get_simulation_seeds(size=None):
    r"""
    Return seeds derived from the simulation-time random number generator.

    Request the same number of seeds in each MPI process to preserve
    reproducibility across processes.
    """
    return simulation_rng.randint(2**32 - 1, size=size)

def getMozaikLogger():
    r"""
    To maintain consistent logging settings around mozaik use this method to obtain the logger isntance.
    """
    import logging
    logger = logging.getLogger("Mozaik")
    logger.setLevel(logging.INFO)
    return logger

def load_component(path):
    r"""
    This function loads a model component (represented by a class instance) located with the path varialble.
    
    Parameters
    ----------

    path : str
        The path to the module containing the component.   
             
    Returns
    -------

    component : object
        The instance of the component class
    
    Notes
    -----
    
    This function is primarily used to automatically load components based on configuration files during model construction.

    """
    logger = getMozaikLogger()
    path_parts = path.split('.')
    module_name = ".".join(path_parts[:-1])
    class_name = path_parts[-1]
    _module = __import__(module_name, globals(), locals(), [class_name])
    logger.info("Loaded component %s from module %s" % (class_name, module_name))
    return getattr(_module, class_name)
