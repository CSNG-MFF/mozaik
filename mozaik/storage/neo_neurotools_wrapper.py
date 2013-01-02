"""
docstring goes here
"""
from neo.core.segment import Segment
from NeuroTools import signals
import numpy
import cPickle


class MozaikSegment(Segment):
        """
        This class extends Neo segment with several convenience functions.

        The most important function is that it allows lazy loading of the data.

        It should be moved to datastore.py once the NeoNeurotoolsWrapper is
        obsolete and this file should be discarded.
        """

        def __init__(self, segment, identifier):
            """
            """
            self.init = True
            Segment.__init__(self, name=segment.name,
                             description=segment.description,
                             file_origin=segment.file_origin,
                             file_datetime=segment.file_datetime,
                             rec_datetime=segment.rec_datetime,
                             index=segment.index)

            self.annotations = segment.annotations
            self.identifier = identifier
            # indicates whether the segment has been fully loaded
            self.full = False

        def get_spiketrains(self):
            if not self.full:
                self.load_full()
            return self._spiketrains

        def set_spiketrains(self, s):
            if self.init:
                self.init = False
                return
            raise ValueError('The spiketrains property should never be directly set in MozaikSegment!!!')

        spiketrains = property(get_spiketrains, set_spiketrains)

        def get_vm(self, neuron):
            if not self.full:
                self.load_full()

            for a in self.analogsignalarrays:
                if a.name == 'v':
                    idd = self.spiketrains[neuron].annotations['source_id']
                    return a[:, numpy.where(a.annotations['source_ids'] == idd)[0]]

        def get_esyn(self, neuron):
            if not self.full:
                self.load_full()
            for a in self.analogsignalarrays:
                if a.name == 'gsyn_exc':
                    idd = self.spiketrains[neuron].annotations['source_id']
                    return a[:, numpy.where(a.annotations['source_ids'] == idd)[0]]

        def get_isyn(self, neuron):
            if not self.full:
                self.load_full()
            for a in self.analogsignalarrays:
                if a.name == 'gsyn_inh':
                    idd = self.spiketrains[neuron].annotations['source_id']
                    return a[:, numpy.where(a.annotations['source_ids'] == idd)[0]]

        def load_full(self):
            """
            Load the full version of the Segment and set self.full to True.
            """
            pass

        def neuron_num(self):
            """
            Return number of STORED neurons in the Segment.
            """
            return len(self.spiketrains[0])
        
        def get_stored_isyn_ids(self):
            for a in self.analogsignalarrays:
                if a.name == 'isyn_exc':
                   return a.annotations['source_ids']
        
        def get_stored_esyn_ids(self):
            for a in self.analogsignalarrays:
                if a.name == 'gsyn_exc':
                   return a.annotations['source_ids']


        def get_stored_v_ids(self):
            for a in self.analogsignalarrays:
                if a.name == 'v':
                   return a.annotations['source_ids']

"""
This is a temporary wrapper that should be completely replaced by Neurotools
once they have been converted to use Neo data structures.

NOTE!!! Currently it is a big memory and CPU time liability!!!!
NOTE!!!! We are also transposing all the Neo analogsignalarrays !!!
This should be sorted in future!
"""


class NeoNeurotoolsWrapper(MozaikSegment):

        def init_Neurotools(self):
            # Store stuff also in Neurotools format
            t_start = self.spiketrains[0].t_start
            t_stop = self.spiketrains[0].t_stop

            d = {}
            for st in self.spiketrains:
                d[st.annotations["source_id"]] = numpy.array(st)

            self.nt_spikes = signals.SpikeList(spike_dic_to_list(d),
                                               d.keys(),
                                               float(t_start),
                                               float(t_stop))

            #self.nt_gsyn_e = []
            #self.nt_gsyn_i = []
            #self.nt_vm = []

            #for ar in self.analogsignalarrays:
                #if ar.name == 'v':
                    #for a in ar:
                        #self.nt_vm.append(signals.AnalogSignal(a, dt=ar.sampling_period))

                #if ar.name == 'gsyn_exc':
                    #for a in ar:

                        #self.nt_gsyn_e.append(signals.AnalogSignal(a, dt=ar.sampling_period))

                #if ar.name == 'gsyn_inh':
                    #for a in ar:
                        #self.nt_gsyn_i.append(signals.AnalogSignal(a, dt=ar.sampling_period))

        def mean_rates(self):
            if not self.full:
                self.load_full()
            return self.nt_spikes.mean_rates()
    
        def cv_isi(self):
            if not self.full:
                   self.load_full()
            return self.nt_spikes.cv_isi()
    
class PickledDataStoreNeoWrapper(NeoNeurotoolsWrapper):
        def __init__(self, segment, identifier, datastore_path):
            MozaikSegment.__init__(self, segment, identifier)
            self.datastore_path = datastore_path

        def load_full(self):
            f = open(self.datastore_path + '/' + self.identifier + ".pickle", 'rb')
            s = cPickle.load(f)
            f.close()
            self._spiketrains = s.spiketrains
            self.analogsignalarrays = s.analogsignalarrays
            self.full = True
            self.init_Neurotools()

        def __getstate__(self):
            result = self.__dict__.copy()
            if self.full:
                del result['_spiketrains']
                del result['analogsignalarrays']
            return result


def spike_dic_to_list(d):
    sp = []
    for k in d.keys():
        for z in d[k]:
            sp.append([k, z])
    if len(sp) == 0:
        return sp
    sp = numpy.array(sp)
    return sp[sp[:, 1].argsort(), :]
