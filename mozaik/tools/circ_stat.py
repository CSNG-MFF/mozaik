"""
This module contains several helper functions for working with periodic variables.
"""
import logging
import numpy
from numpy import pi, sin, cos
logger = logging.getLogger("mozaik")


def circular_dist(a, b, period):
    """
    Returns the distance between a and b (scalars) in a domain with `period` period.
    """
    return  numpy.minimum(numpy.abs(a - b), period - numpy.abs(a - b))


def rad_to_complex(vector):
    """
    Converts a vector/matrix of angles (0, 2*pi) to vector/matrix of complex
    numbers (that will lie on the unit circle) and correspond to the given angle.
    """
    return cos(vector) + 1j*sin(vector)


def angle_to_pi(array):
    """
    Returns angles of complex numbers in array but in (0, 2*pi) interval unlike
    numpy.angle the returns it in (-pi, pi).
    """
    return (numpy.angle(array) + 4*pi) % (pi*2)


def circ_mean(matrix, weights=None, axis=None, low=0, high=pi*2,
              normalize=False):
    """
    Circular mean of matrix. Weighted if weights are not none.
    Mean will be computed along axis axis.
    
    Parameters
    ----------
    
    matrix : ndarray
           Matrix of data for which the compute the circular mean. 
           
    weights : ndarray, optional
            If not none, matrix of the same size as matrix. It will be used as weighting for the mean.
    
    low, high : double, optional
              The min and max values that will be mapped onto the periodic interval of (0, 2pi).
              
    axis : int, optional
         Numpy axis along which to compute the circular mean. 
    
    
    normalize : bool
              If True weights will be normalized along axis. If any weights
              that are to be jointly normalized are all zero they will be
              kept zero!

    Returns
    -------
    (angle, length) : ndarray,ndarray
                    Where angle is the circular mean, and len is the length of the resulting mean vector.
    """
    idx = numpy.nonzero(weights!=0.0)[0]
    # check whether matrix and weights are ndarrays
    if isinstance(weights,numpy.ndarray):
       assert matrix.shape == weights.shape 
    
    if axis == None:
        axis == 0

    # convert the periodic matrix to corresponding complex numbers
    m = rad_to_complex((matrix - low)/(high - low) * pi*2)

    # normalize weights
    if normalize:
        row_sums = numpy.sum(numpy.abs(weights), axis=axis)
        row_sums[numpy.where(row_sums == 0)] = 1.0
        if axis == 1:
            weights = weights / row_sums[:, numpy.newaxis]
        else:
            weights = numpy.transpose(weights) / row_sums[:, numpy.newaxis]
            weights = weights.T
            
    if isinstance(weights,numpy.ndarray):
        z = numpy.multiply(m,weights)
        m = numpy.mean(z, axis=axis)
    else:
        m = numpy.mean(m, axis=axis)
    
    a,b = ((angle_to_pi(m) / (pi*2))*(high-low) + low, abs(m))
    return a,b
    
