import numpy

def time_histogram_across_trials(spike_trials,bin_length):
    """
    spike_trials - should contain a list of SpikeList objects, each coresponding to different trial
    The function returns the histogram of spikes across trials with bin length bin_length
    """
    
    t_start = spike_trials[0].t_start
    t_stop = spike_trials[0].t_stop
    num_neurons = len(spike_trials[0])
    
    num_bins = (t_stop-t_start)/bin_length
    
    st = [[] for i in xrange(0,num_neurons)]
    
    h = []
    
    for i in xrange(0,num_neurons):
        for s in spike_trials:
            st[i].extend(s[i].spike_times)
        h.append(numpy.histogram(st[i], bins=num_bins, range=(t_start,t_stop))[0])
    
    return h
