"""
Various helper functions.
"""

import numpy                                                                             
import numpy.random                                                                             
from numpy import pi, sqrt, exp, power

def sample_from_bin_distribution(bins, number_of_samples):
    """
    Samples from a distribution defined by a vector the sum in. The vector doesn't have to add up to one
    it will be automatically normalized. 
    
    
    Parameters
    ----------
    
    bins : ndarray
         The returned samples correspond to the bins in `bins` - the numpy array defining the bin distribution
      
    number_of_samples : int
                      Number of samples to generate.
    """
    if len(bins) == 0:
        return []

    bins = bins / numpy.sum(bins)
    si = numpy.random.choice(range(len(bins)),size=number_of_samples,p=bins)

    return si

_normal_function_sqertofpi = sqrt(2*pi)
def normal_function(x, mean=0, sigma=1.0):
    """
    Returns the value of probability density of normal distribution N(mean,sigma) at point `x`.
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
       cl =  closest(minx+(maxx-minx)/2,miny+(maxy-miny)/2,positions)
    elif which == 'top_right':
       cl = closest(maxx,maxy,positions)
    elif which == 'top_left':
       cl = closest(minx,maxy,positions)
    elif which == 'bottom_left':
       cl = closest(minx,miny,positions)
    elif which == 'bottom_right':
       cl = closest(maxx,miny,positions)
       
    return cl

def result_directory_name(simulation_run_name,simulation_name,modified_parameters):
    modified_params_str = '_'.join([str(k) + ":" + str(modified_parameters[k]) for k in sorted(modified_parameters.keys()) if k!='results_dir'])
    if len(modified_params_str) > 100:
        modified_params_str = '_'.join([str(k).split('.')[-1] + ":" + str(modified_parameters[k]) for k in sorted(modified_parameters.keys()) if k!='results_dir'])
    
    return simulation_name + '_' + simulation_run_name + '_____' + modified_params_str

    
    
