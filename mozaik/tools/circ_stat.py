"""
Circular statistics
"""
import logging
import numpy
logger = logging.getLogger("mozaik")

def circular_dist(a,b,period):
    """
    The distance between a and b (scalars) in the periodic. 
    a,b have to be in (0,period)
    """
    return  min(abs(a-b), period - abs(a-b))
    
def rad_to_complex(vector):
    """
    Converts a vector/matrix of angles (0,2*pi) to vector/matrix of complex numbers (that will lie on the unit circle)
    """
    return numpy.cos(vector)+1j*numpy.sin(vector)
    

def angle_two_pi(array):
    """
    returns angles of complex numbers in array but in (0,2*pi) interval unlike numpy.angle the returns it in (-pi,pi)
    """
    return (numpy.angle(array)+ 4*numpy.pi) % (numpy.pi*2)
    

def circ_mean(matrix,weights=None,axis=None,low=0,high=numpy.pi*2,normalize=False):
    """
    Circular mean of matrix. Weighted if weights are not none.
    
    matrix   - 2d ndarray of data. Mean will be computed for each column.
    weights  - if not none, a vector of the same length as number of matrix rows.
    low,high - the min and max values that will be mapped onto the periodic interval of (0,2pi)
    axis     - axis along which to compute the circular mean. default = 1
    normalize - if True weights will be normalized along axis. If any weights that are to be jointly normalized are all zero they will be kept zero!
    
    return (angle,length)  - where angle is the circular mean, and len is the length of the resulting mean vector
    """
    
    # check whether matrix and weights are ndarrays
    if not isinstance(matrix,numpy.ndarray):
       logger.error("circ_mean: array not type ndarray ") 
       raise TypeError("circ_mean: array not type ndarray ") 

    if weights!= None and not isinstance(weights,numpy.ndarray):
       logger.error("circ_mean: weights not type ndarray ") 
       raise TypeError("circ_mean: weights not type ndarray ") 

    if axis == None:
       axis == 1

    # convert the periodic matrix to corresponding complex numbers
    m = rad_to_complex((matrix - low)/(high-low) * numpy.pi*2)
    
    # normalize weights
    if normalize:
       row_sums = numpy.sum(numpy.abs(weights),axis=1)
       row_sums[numpy.where(row_sums == 0)] = 1.0
       weights = weights / row_sums[:, numpy.newaxis]
       
           
    if weights == None:
       m  = numpy.mean(m,axis=axis) 
    else:
       z = m*weights
       m = numpy.mean(z,axis=axis) 
        
    return ((angle_two_pi(m) / (numpy.pi*2))*(high-low) + low, abs(m))
    
