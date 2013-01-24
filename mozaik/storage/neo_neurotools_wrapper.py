"""
docstring goes here
"""
from neo.core.segment import Segment
import numpy
import cPickle
import quantities as qt


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

        def get_vm(self, neuron,idd=False):
            if not self.full:
                self.load_full()

            for a in self.analogsignalarrays:
                if a.name == 'v':
                    if idd == False:
                        idd = self.spiketrains[neuron].annotations['source_id']
                    else:
                        idd = neuron
                        
                    return a[:, numpy.where(a.annotations['source_ids'] == idd)[0]]

        def get_esyn(self, neuron,idd=False):
            if not self.full:
                self.load_full()
            for a in self.analogsignalarrays:
                if a.name == 'gsyn_exc':
                    if idd == False:
                        idd = self.spiketrains[neuron].annotations['source_id']
                    else:
                        idd = neuron
                    return a[:, numpy.where(a.annotations['source_ids'] == idd)[0]]

        def get_isyn(self, neuron,idd=False):
            if not self.full:
                self.load_full()
            for a in self.analogsignalarrays:
                if a.name == 'gsyn_inh':
                    if idd == False:
                        idd = self.spiketrains[neuron].annotations['source_id']
                    else:
                        idd = neuron
                        
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
            if not self.full:
                self.load_full()
            for a in self.analogsignalarrays:
                if a.name == 'gsyn_inh':
                   return a.annotations['source_ids']
        
        def get_stored_esyn_ids(self):
            if not self.full:
                self.load_full()
            for a in self.analogsignalarrays:
                if a.name == 'gsyn_exc':
                   return a.annotations['source_ids']

        def get_stored_v_ids(self):
            if not self.full:
                self.load_full()
            for a in self.analogsignalarrays:
                if a.name == 'v':
                   return a.annotations['source_ids']


        def mean_rates(self):
            """
            Returns the mean rates of the spiketrains in spikes/s
            """
            return [len(s)/(s.t_stop.rescale(qt.s).magnitude-s.t_start.rescale(qt.s).magnitude) for s in self.spiketrains]


"""
This is a Mozaik wrapper of neo segment, that enables pickling and lazy loading.
"""    

class PickledDataStoreNeoWrapper(MozaikSegment):
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

        def __getstate__(self):
            flag = self.full
            self.full = False
            result = self.__dict__.copy()
            if flag:
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
