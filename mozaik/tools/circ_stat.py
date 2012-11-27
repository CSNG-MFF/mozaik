"""
Circular statistics
"""
import logging
import numpy
from numpy import pi, sin, cos
logger = logging.getLogger("mozaik")


def circular_dist(a, b, period):
    """
    The distance between a and b (scalars) in the periodic.
    a, b have to be in (0, period)
    """
    return  numpy.minimum(numpy.abs(a - b), period - numpy.abs(a - b))


def rad_to_complex(vector):
    """
    Converts a vector/matrix of angles (0, 2*pi) to vector/matrix of complex
    numbers (that will lie on the unit circle)
    """
    return cos(vector) + 1j*sin(vector)


def angle_to_pi(array):
    """
    returns angles of complex numbers in array but in (0, 2*pi) interval unlike
    numpy.angle the returns it in (-pi, pi)
    """
    return (numpy.angle(array) + 4*pi) % (pi*2)


def circ_mean(matrix, weights=None, axis=None, low=0, high=pi*2,
              normalize=False):
    """
    Circular mean of matrix. Weighted if weights are not none.

    matrix     - matrix of data. Mean will be computed along axis axis.
    weights    - if not none, matrix of the same size as matrix
    low, high  - the min and max values that will be mapped onto the periodic
                 interval of (0, 2pi)
    axis       - axis along which to compute the circular mean. default = 0 (columns)
    normalize  - if True weights will be normalized along axis. If any weights
                 that are to be jointly normalized are all zero they will be
                 kept zero!

    return (angle, length) - where angle is the circular mean, and len is the
                           length of the resulting mean vector
    """

    # check whether matrix and weights are ndarrays
    if weights != None:
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
            
    if weights == None:
        m = numpy.mean(m, axis=axis)
    else:
        z = numpy.multiply(m,weights)
        m = numpy.mean(z, axis=axis)

    return ((angle_to_pi(m) / (pi*2))*(high-low) + low, abs(m))
