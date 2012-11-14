"""
docstring goes here
"""
import numpy                                                                             
from numpy import pi, sqrt, exp, power

def sample_from_bin_distribution(bins, number_of_samples):
    """
    Samples from a distribution defined by a vector
    the sum in the vector doesn't have to add up to one
    it will be automatically normalized
    the returned samples correspond to the bins
    bins - the numpy array defining the bin distribution
    number_of_samples - number of samples to generate
    """
    if len(bins) == 0:
        return []

    bins = bins / numpy.sum(bins)
    
    # create the cumulative sum
    cs = numpy.cumsum(bins)
    samples = numpy.random.rand(number_of_samples)
    si = []
    for s in samples:
        si.append(numpy.nonzero(s < cs)[0][0])

    return si

                                                                                         

_normal_function_sqertofpi = sqrt(2*pi)
def normal_function(x, mean=0, sigma=1.0):
    """
    Returns the value of normal distribution N(mean,sigma) at point x
    """
    return numpy.exp(-numpy.power((x - mean)/sigma, 2)/2) / (sigma * _normal_function_sqertofpi)
