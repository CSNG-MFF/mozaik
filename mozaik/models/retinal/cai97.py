"""
docstring goes here

"""

import numpy
from numpy import exp
from mozaik.framework.decorators import timeit


def meshgrid3D(x, y, z):
    """A slimmed-down version of http://www.scipy.org/scipy/numpy/attachment/ticket/966/meshgrid.py"""
    x = numpy.asarray(x)
    y = numpy.asarray(y)
    z = numpy.asarray(z)
    mult_fact = numpy.ones((len(x), len(y), len(z)))
    nax = numpy.newaxis
    return x[:, nax, nax] * mult_fact, \
           y[nax, :, nax] * mult_fact, \
           z[nax, nax, :] * mult_fact


@timeit
def stRF_kernel_2d(duration=200.0, dt=1000.0/120.0, size=10.0,
                   scale_factor=10.0, p={}):
    """
    scale_factor = pixel/degree
    """
    degree_per_pixel = 1/float(scale_factor)
    #p = RF_parameters()
    x = numpy.arange(-size/2. + degree_per_pixel/2,
                     size/2. + degree_per_pixel/2,
                     degree_per_pixel)
    t = numpy.arange(0.0, duration, dt)
    xm, ym, tm = meshgrid3D(x, x, t)
    kernel = stRF_2d(xm, ym, tm, p)
    return kernel


@timeit
def stRF_2d(x, y, t, p):
    """
    x, y, and t should all be 3D arrays, produced by meshgrid3D.
    If we need to optimize, it would be quicker to do G() on a 1D t array and
    F_2d() on 2D x and y arrays, and then multiply them, as Jens did in his
    original implementation.
    Timing gives 0.44 s for Jens' implementation, and 2.9 s for this one.
    """

    tmc = G(t, p.K1, p.K2, p.c1, p.c2, p.t1, p.t2, p.n1, p.n2)
    tms = G(t-p.td, p.K1, p.K2, p.c1, p.c2, p.t1, p.t2, p.n1, p.n2)

    fcm = F_2d(x, y, p.Ac, p.sigma_c)
    fsm = F_2d(x, y, p.As, p.sigma_s)

    ##rf = (fcm*tmc - fsm*tms)/(fcm - fsm).max()
    rf = fcm*tmc - fsm*tms
    if p.subtract_mean:
        # lets normalize each time slace sepparately 
        #for i in xrange(0,numpy.shape(rf)[2]):
        #    rf[:,:,i] = rf[:,:,i] - rf[:,:,i].mean()
        rf = rf - rf.mean()
    return rf


def G(t, K1, K2, c1, c2, t1, t2, n1, n2):
    p1 = K1 * ((c1*(t - t1))**n1 * exp(-c1*(t - t1))) / ((n1**n1) * exp(-n1))
    p2 = K2 * ((c2*(t - t2))**n2 * exp(-c2*(t - t2))) / ((n2**n2) * exp(-n2))
    p3 = p1 - p2
    ### norm to max == 1.
    ##p3 /= p3.max()
    return p3


def F_2d(x, y, A, sigma):
    return A * exp(-(x**2 + y**2) / (2*sigma**2))
