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

def find_neuron(which,positions):
    """
    Finds a neuron depending on which:
        'center' - the most central neuron in the sheet 
        'top_right' - the top_right neuron in the sheet
        'top_left' - the top_left neuron in the sheet
        'bottom_left' - the bottom_left neuron in the sheet
        'bottom_right' - the bottom_right neuron in the sheet
    """
    minx = numpy.min(positions[0,:])
    maxx = numpy.max(positions[0,:])
    miny = numpy.min(positions[1,:])
    maxy = numpy.max(positions[1,:])
    
    def closest(x,y,positions):
        return numpy.argmin(numpy.sqrt(numpy.power(positions[0,:].flatten()-x,2) + numpy.power(positions[1,:].flatten()-y,2)))
    
    if which == 'center':
       return closest(minx+(maxx-minx)/2,miny+(maxy-miny)/2,positions)
    elif which == 'top_right':
       return closest(maxx,maxy,positions)
    elif which == 'top_left':
       return closest(minx,maxy,positions)
    elif which == 'bottom_left':
       return closest(minx,miny,positions)
    elif which == 'bottom_right':
       return closest(maxx,miny,positions)
    
