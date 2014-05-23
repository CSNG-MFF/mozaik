"""
This file contains various operations over Neo objects. Such as sum over lists
of Neo objects etc.
"""
import quantities as qt
from neo.core.analogsignal import AnalogSignal as NeoAnalogSignal
import numpy

def neo_sum(l):
    """
    This function gets a list of Neo objects and it
    adds them up. Importantly unlike Python sum function
    it starts adding to the first element of the list no to 0.
    """
    a = l[0]
    for z in l[1:]:
        a = a + z
    return a

def neo_mean(l):
    """
    Calculates the mean over list of Neo objects.
    See neo_sum for more details.
    """
    return neo_sum(l) / len(l)

def down_sample_analog_signal_average_method(analog_signal,new_sampling_period):
    """
    It down-samples the signal such that it will bin the time axis with bin length `new_sampling_period` and make average for each bin.
    
    Parameters
    ----------
    analog_signal : AnalogSignal
             The analog signal to downsample
    
    new_sampling_period : float(ms)
                        The desired new sampling period of the signal
    """ 
    if abs(analog_signal.t_stop.rescale(qt.ms) -  new_sampling_period) > 0.000000001:
        assert (analog_signal.t_stop.rescale(qt.ms) % new_sampling_period)  < 0.000000001, "TemporalBinAverage: The analog signal length has to be divisible by bin_length"
        div = round(analog_signal.t_stop.rescale(qt.ms) / new_sampling_period)
        return NeoAnalogSignal(numpy.mean(numpy.reshape(analog_signal,(div,)),axis=1).flatten(),
                       t_start=0*qt.ms,
                       sampling_period=bin_length*qt.ms,
                       units=analog_signal.units)

def down_sample_analog_signal(analog_signal,new_sampling_period):
    """
    It down-samples the signal such that it picks each i-th element of the analog signal based on the desired new sampling period.
    
    Parameters
    ----------
    analog_signal : AnalogSignal
             The analog signal to downsample
    
    new_sampling_period : float(ms)
                        The desired new sampling period of the signal
    """ 
    if abs(analog_signal.t_stop.rescale(qt.ms) -  new_sampling_period) > 0.000000001:
       div = round(new_sampling_period / analog_signal.sampling_period.rescale(qt.ms))
       return NeoAnalogSignal(analog_signal[::div],
                       t_start=0*qt.ms,
                       sampling_period=new_sampling_period*qt.ms,
                       units=analog_signal.units)

    else:
        return analog_signal
