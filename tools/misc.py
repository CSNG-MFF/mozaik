from NeuroTools import signals
from MozaikLite.stimuli.stimulus_generator import parse_stimuls_id,load_from_string
import numpy

def get_spikes_to_dic(spikes,pop):
    #order according to ids
    dictionary = {}  
    for idd in xrange(0,len(pop)):
        dictionary[idd]=[]

    if len(spikes) == 0:
        return dictionary

    spikes = spikes[spikes[:,0].argsort(),:]
    
    idd = spikes[0,0];
    start = 0
    
    for i in xrange(0,len(spikes)):
        if spikes[i,0] != idd:
           sp = spikes[start:i,1] 
           dictionary[idd] = sp[sp.argsort()]
           idd = spikes[i,0]
           start = i;
    
    sp = spikes[start:,1] 
    dictionary[idd] = sp[sp.argsort()]
    
    return dictionary

def get_vm_to_dic(vm,pop):
    #order according to ids
    dictionary = {} 

    for idd in xrange(0,len(pop)):
        dictionary[idd]=[]

    if len(vm) == 0:
        return dictionary

    vm = vm[vm[:,0].argsort(),:]
    
    idd = vm[0,0];
    start = 0
    
    for i in xrange(0,len(vm)):
        if vm[i,0] != idd:
           z = vm[start:i,:]
           z = z[z[:,1].argsort(),:]
           dictionary[idd] = z[:,2]
           idd = vm[i,0]
           start = i;

    z = vm[start:,:]
    z = z[z[:,1].argsort(),:]
    dictionary[idd] = z[:,2]
    return dictionary

def get_gsyn_to_dicts(gsyn,pop):
    #order according to ids
    dictionary_e = {} 
    dictionary_i = {} 

    for idd in xrange(0,len(pop)):
        dictionary_e[idd]=[]
        dictionary_i[idd]=[]

    if len(gsyn) == 0:
       return (dictionary_e,dictionary_i)

    gsyn = gsyn[gsyn[:,0].argsort(),:]
    
    idd = gsyn[0,0];
    start = 0
    
    for i in xrange(0,len(gsyn)):
        if gsyn[i,0] != idd:
           z = gsyn[start:i,:]
           z = z[z[:,1].argsort(),:]
           dictionary_e[idd] = z[:,2]
           dictionary_i[idd] = z[:,3]
           idd = gsyn[i,0]
           start = i;

    z = gsyn[start:,:]
    z = z[z[:,1].argsort(),:]
    dictionary_e[idd] = z[:,2]
    dictionary_i[idd] = z[:,3]
    return (dictionary_e,dictionary_i)


def spike_dic_to_list(d):
    sp = []
    for k in d.keys():
        for z in d[k]:
            sp.append([k,z])
    if len(sp) == 0:
        return sp
    sp = numpy.array(sp)
    return sp[sp[:,1].argsort(),:]    
    
def spike_segment_to_dict(seg):
    sheets={}
    for s in seg.sheets: 
        d = {}
        for k in seg.__getattr__(s+'_spikes'):
            t = seg._spiketrains[k]
            d[t.index] = numpy.array(t)
        sheets[s] = (spike_dic_to_list(d),d.keys(),float(t.t_start),float(t.t_stop))
    return sheets
    
def segments_to_dict_of_SpikeList(segments):
    # it turns neo segment list to a dictionary of tuples
    # each key in dictionary corresponds to a sheet and contains 
    # a tuple of arrays containing the spiketrains  and corresponding stimuli
    dd = {}
    for seg in segments:
        d = spike_segment_to_dict(seg)
        for k in d.keys():
            (spikes,idds,tstart,tstop) = d[k]
            if not dd.has_key(k):
               dd[k] = ([],[])
            (sp,st) = dd[k]        
            print tstart
            print tstop
            sp.append(signals.SpikeList(spikes,idds,t_start=tstart,t_stop=tstop))
            st.append(parse_stimuls_id(seg.stimulus))
    return dd



def segments_to_dict_of_AnalogSignalList(segments):
    return (_segments_to_dict_of_AnalogSignalList(segments,'vm'),_segments_to_dict_of_AnalogSignalList(segments,'gsyn_e'),_segments_to_dict_of_AnalogSignalList(segments,'gsyn_i'))

def _segments_to_dict_of_AnalogSignalList(segments,signal_name):
    # it turns neo segment list to a dictionary of tuples
    # each key in dictionary corresponds to a sheet and contains 
    # a tuple of arrays containing the spiketrains  and corresponding stimuli
    dd = {}
    for seg in segments:
        d = analog_segment_to_dict(seg,signal_name)
        sp = seg._analogsignals[0].sampling_period
        for k in d.keys():
            (sig,idds) = d[k]
            if not dd.has_key(k):
               dd[k] = ([],[])
            (si,st) = dd[k]        
            si.append(sig)
            st.append(parse_stimuls_id(seg.stimulus))
    return dd

def analog_segment_to_dict(seg,signal_name):
    sheets={}
    for s in seg.sheets: 
        d = {}
        for k in seg.__getattr__(s+'_'+signal_name):
            t = seg._analogsignals[k]
            d[t.index] = numpy.array(t)
        sheets[s] = ([signals.AnalogSignal(d[k],dt=t.sampling_period) for k in d.keys()],d.keys())
    return sheets


def sample_from_bin_distribution(bins, number_of_samples):
    # samples from a distribution defined by a vector
    # the sum in the vector doesn't have to add up to one
    # it will be automatically normalized
    # the returned samples correspond to the bins 
    # bins - the numpy array defining the bin distribution
    # number_of_samples - number of samples to generate 
    
    bins = numpy.array(bins) / numpy.sum(bins)

    # create the cumulative sum
    cs = numpy.cumsum(bins)
    samples = numpy.random.rand(number_of_samples)
    si = []
    for s in samples:
        si.append(numpy.nonzero(s < cs)[0][0])
    
    return si
