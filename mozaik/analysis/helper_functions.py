"""
This module contains various utility functions often useb by analysis code.
"""

import numpy
import quantities as qt
import mozaik
import mozaik.tools.units as munits
from neo import AnalogSignal

logger = mozaik.getMozaikLogger()


def psth(spike_list, bin_length,normalize=True):
    """
    The function returns the psth of the spiketrains with bin length bin_length.
    
    Parameters
    ----------
    spike_list : list(SpikeTrain )
               The list of spike trains. They are assumed to start and end at the same time.

    bin_length : float (ms) 
               Bin length.
               
    normalized : bool
               If true the psth will return the instantenous firing rate, if False it will return spike count per bin. 

    Returns
    -------
    psth : AnalogSignal
           The PSTH of the spiketrain. 
    
    Note
    ----
    The spiketrains are assumed to start and stop at the same time!
    """
    t_start = round(spike_list[0].t_start.rescale(qt.ms),5)
    t_stop = round(spike_list[0].t_stop.rescale(qt.ms),5)
    num_bins = int(round((t_stop-t_start)/bin_length))
    r = (float(t_start), float(t_stop))

    for sp in spike_list:
        assert len(numpy.histogram(sp, bins=num_bins, range=r)[0]) == num_bins
        
    normalizer = 1.0
    if normalize:
       normalizer = (bin_length/1000.0)
       
    h = [AnalogSignal(numpy.histogram(sp, bins=num_bins, range=r)[0] /normalizer ,t_start=t_start*qt.ms,sampling_period=bin_length*qt.ms,units=munits.spike_per_sec) for sp in spike_list]
    return  h


def psth_across_trials(spike_trials, bin_length):
    """
    It returns PSTH averaged across the spiketrains
    
    
    Parameters
    ----------
    spike_trials : list(list(SpikeTrain)) 
                 should contain a list of lists of neo spiketrains objects,
                 first coresponding to different trials and second to different neurons.
                 The function returns the histogram of spikes across trials with bin length
                 bin_length.
   
    Returns
    -------
    psth : AnalogSignal
         The PSTH averages over the spiketrains in spike_trials.
                 
    Note
    ----
    The spiketrains are assumed to start and stop at the same time.
    """
    return sum([psth(st, bin_length) for st in spike_trials])/len(spike_trials)
