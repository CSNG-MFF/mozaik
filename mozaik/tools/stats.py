import numpy
import scipy.stats

def confidence_interval(data, confidence=0.95):
    mean = numpy.mean(data)
    n = len(data)
    sem = scipy.stats.sem(data) 
    confidence_interval = scipy.stats.t.interval(confidence, n-1,mean, scale=sem)
    return mean, *confidence_interval

def prediction_interval(data,confidence=0.95):
    n = len(data)
    t_critical = scipy.stats.t.ppf((1 + confidence) / 2, df=n-1)
    return t_critical * numpy.std(data,ddof=1) * numpy.sqrt(1+1/n)

def explained_variance(distribution,fit):
    return 1-numpy.sum((fit-distribution)**2)/numpy.sum((distribution-numpy.mean(distribution))**2)
