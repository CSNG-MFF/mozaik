import numpy
import scipy.stats

def confidence_interval(data, confidence = 0.95):
    mean = numpy.mean(data)
    std = numpy.std(data,ddof=1)
    n = len(data)
    confidence_interval = scipy.stats.t.interval(confidence, n-1,mean, scale=std / numpy.sqrt(n))
    
    return mean, *confidence_interval

def explained_variance(distribution,fit):
    return 1-numpy.sum((fit-distribution)**2)/numpy.sum((distribution-numpy.mean(distribution))**2)
