from neo.core.segment import Segment
from NeuroTools import signals
import numpy

"""
This class extends Neo segment with several convenience functions.
It should be moved to datastore.py once the NeoNeurotoolsWrapper is 
obsolete and this file should be discarded.
"""
class MozaikSegment(Segment):
        def __init__(self, segment):
            Segment.__init__(self, name=segment.name, description=segment.description, file_origin=segment.file_origin,
                             file_datetime=segment.file_datetime, rec_datetime=segment.rec_datetime, index=segment.index)
            
            self.original_segment = segment
            
            self.epochs = segment.epochs
            self.epocharrays = segment.epocharrays
            self.events = segment.events
            self.eventarrays = segment.eventarrays
            self.analogsignals = segment.analogsignals
            self.analogsignalarrays = segment.analogsignalarrays
            self.irregularlysampledsignals = segment.irregularlysampledsignals
            self.spikes = segment.spikes
            self.spiketrains = segment.spiketrains
            self.annotations = segment.annotations
        
        def get_vm(self):
            for a in self.analogsignalarrays:
                if a.name == 'v':
                   return a
                
        def get_esyn(self):
            for a in self.analogsignalarrays:
                if a.name == 'gsyn_exc':
                   return a
                
        def get_isyn(self):
            for a in self.analogsignalarrays:
                if a.name == 'gsyn_inh':
                   return a
        
        def neuron_num(self):
            return len(self.spiketrains[0])
            
                
"""
This is a temporary wrapper that should be completely replaced by Neurotools once they have 
been converted to use Neo data structures. 

NOTE!!! Currently it is a big memory and CPU time liability!!!!

NOTE!!!! We are also transposing all the Neo analogsignalarrays !!! This should be sorted in future!
"""
class NeoNeurotoolsWrapper(MozaikSegment):

    def __init__(self,segment):
        MozaikSegment.__init__(self,segment)  
        
        # Store stuff also in Neurotools format
        t_start = self.spiketrains[0].t_start
        t_stop = self.spiketrains[0].t_stop
        
        d = {}
        for st in self.spiketrains:
            d[st.annotations["source_id"]] = numpy.array(st)
           
        self.nt_spikes = signals.SpikeList(spike_dic_to_list(d),d.keys(),float(t_start),float(t_stop))

        self.nt_gsyn_e = []
        self.nt_gsyn_i = []
        self.nt_vm = []
        
        #for ar in self.analogsignalarrays:
            #if ar.name == 'v':
                #for a in ar:
                    #self.nt_vm.append(signals.AnalogSignal(a,dt=ar.sampling_period))

            #if ar.name == 'gsyn_exc':
                #for a in ar:
                    
                    #self.nt_gsyn_e.append(signals.AnalogSignal(a,dt=ar.sampling_period))

            #if ar.name == 'gsyn_inh':
                #for a in ar:
                    #self.nt_gsyn_i.append(signals.AnalogSignal(a,dt=ar.sampling_period))
        
    def mean_rates(self):
            return self.nt_spikes.mean_rates()
    
    
        
    
def spike_dic_to_list(d):
    sp = []
    for k in d.keys():
        for z in d[k]:
            sp.append([k,z])
    if len(sp) == 0:
        return sp
    sp = numpy.array(sp)
    return sp[sp[:,1].argsort(),:]    
