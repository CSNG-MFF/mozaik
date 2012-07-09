__version__ = None
# To obtain results repeatable over identical runs of mozaik
# one should use the mozaik.rng as the random noise generator passed to all pyNN
# functions that accept rng as one of their paramters, and also use it for any auxiallary code 
# that requires rng
from pyNN.random import NumpyRNG
rng = NumpyRNG(seed=1023)
