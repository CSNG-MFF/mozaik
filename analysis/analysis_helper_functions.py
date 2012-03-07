import numpy

def time_histogram_across_trials(spike_trials,bin_length):
    """
    spike_trials - should contain a list of lists of neo spiketrains objects, first coresponding to different trials
    and second to different neurons.  The function returns the histogram of spikes across trials with bin length bin_length
    """
    
    t_start = spike_trials[0][0].t_start
    t_stop = spike_trials[0][0].t_stop
    num_neurons = len(spike_trials[0])
    
    num_bins = float((t_stop-t_start)/bin_length)
    
    h = []
    u = spike_trials[0][0].units
    
    for i in xrange(0,num_neurons):
        st = []
        for s in spike_trials:
            st.extend(s[i].rescale(u).magnitude.tolist())
        
        h.append(numpy.histogram(st, bins=num_bins, range=(float(t_start),float(t_stop)))[0]*1.0)
    return h
