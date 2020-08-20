"""
This module contains wrapper for the neo Segment, that add extra functionality to the class.
Within mozaik the data are stored and passed in this format.

Most of the included functionality should in future be provided directly by neo.
When this happens most of this code should become irrelevant and the rest should be
merged into the :mod:`.datastore` module.
"""
import logging

from neo.core.segment import Segment
import numpy
import quantities as qt

from ..tools.misc import load_pickle_crosscompat

logger = logging.getLogger(__name__)


class MozaikSegment(Segment):
    """
    This class extends Neo segment with several convenience functions.

    The most important function is that it allows lazy loading of the data.

    It should be moved to datastore.py once the NeoNeurotoolsWrapper is
    obsolete and this file should be discarded.
    """

    def __init__(self, segment, identifier, null=False):
        """"""
        self.init = True
        Segment.__init__(
            self,
            name=segment.name,
            description=segment.description,
            file_origin=segment.file_origin,
            file_datetime=segment.file_datetime,
            rec_datetime=segment.rec_datetime,
            index=segment.index
        )

        self.annotations = segment.annotations
        self.identifier = identifier
        self.null = null
        # indicates whether the segment has been fully loaded
        self.full = False

    def get_spiketrains(self):
        """
        Returns the list of SpikeTrain objects stored in this segment.
        """
        if not self.full:
            self.load_full()
        return self._spiketrains

    def set_spiketrains(self, s):
        if self.init:
            self.init = False
            return
        raise ValueError(
            "The spiketrains property should never be directly set in MozaikSegment!!!"
        )

    spiketrains = property(get_spiketrains, set_spiketrains)

    def get_spiketrain(self, neuron_id):
        """
        Returns a spiktrain or a list of spike train corresponding to id(s) listed in the `neuron_id` argument.

        Parameters
        ----------

        neuron_id : int or list(int)
                  An int or a list of ints containing the ids for which to return the spiketrains.

        Returns
        -------
        A SpikeTrain object if neuron_id is int, or list of SpikeTrain objects if neuron_id is list, the order corresponds to the order in neuron_id argument.
        """

        ids = [s.annotations["source_id"] for s in self.spiketrains]
        if isinstance(neuron_id, list) or isinstance(neuron_id, numpy.ndarray):
            return [self.spiketrains[ids.index(i)] for i in neuron_id]
        else:
            return self.spiketrains[ids.index(neuron_id)]

    def get_vm(self, neuron_id):
        """
        Returns the recorded membrane potential corresponding to neurons with id(s) listed in the `neuron_id` argument.

        Parameters
        ----------

        neuron_id : int or list(int)
                  An int or a list of ints containing the ids for which to return the AnalogSignal objects.

        Returns
        -------
        A AnalogSignal object if neuron_id is int, or list of AnalogSignal objects if neuron_id is list, the order corresponds to the order in neuron_id argument.
        """

        if not self.full:
            self.load_full()

        for a in self.analogsignals:
            if a.name == "v":

                return a[:, a.annotations["source_ids"].tolist().index(neuron_id)]

    def get_esyn(self, neuron_id):
        """
        Returns the recorded excitatory conductance corresponding to neurons with id(s) listed in the `neuron_id` argument.

        Parameters
        ----------

        neuron_id : int or list(int)
                  An int or a list of ints containing the ids for which to return the AnalogSignal objects.

        Returns
        -------
        A AnalogSignal object if neuron_id is int, or list of AnalogSignal objects if neuron_id is list, the order corresponds to the order in neuron_id argument.
        """
        if not self.full:
            self.load_full()
        for a in self.analogsignals:
            if a.name == "gsyn_exc":
                return a[:, a.annotations["source_ids"].tolist().index(neuron_id)]

    def get_isyn(self, neuron_id):
        """
        Returns the recorded inhibitory conductance corresponding to neurons with id(s) listed in the `neuron_id` argument.

        Parameters
        ----------

        neuron_id : int or list(int)
                  An int or a list of ints containing the ids for which to return the AnalogSignal objects.

        Returns
        -------
        A AnalogSignal object if neuron_id is int, or list of AnalogSignal objects if neuron_id is list, the order corresponds to the order in neuron_id argument.
        """

        if not self.full:
            self.load_full()
        for a in self.analogsignals:
            if a.name == "gsyn_inh":
                return a[:, a.annotations["source_ids"].tolist().index(neuron_id)]

    def load_full(self):
        pass

    def neuron_num(self):
        """
        Return number of stored neurons in this Segment.
        """
        return len(self.spiketrains)

    def get_stored_isyn_ids(self):
        """
        Returns ids of neurons for which inhibitory conductance is stored in this segment.
        """
        if not self.full:
            self.load_full()
        for a in self.analogsignals:
            if a.name == "gsyn_inh":
                return a.annotations["source_ids"]

    def get_stored_esyn_ids(self):
        """
        Returns ids of neurons for which excitatory conductance is stored in this segment.
        """
        if not self.full:
            self.load_full()
        for a in self.analogsignals:
            if a.name == "gsyn_exc":
                return a.annotations["source_ids"]

    def get_stored_vm_ids(self):
        """
        Returns ids of neurons for which membrane potential is stored in this segment.
        """
        if not self.full:
            self.load_full()
        for a in self.analogsignals:
            if a.name == "v":
                return a.annotations["source_ids"]

    def get_stored_spike_train_ids(self):
        """
        Returns ids of neurons for which spikes are stored in this segment.
        """

        if not self.full:
            self.load_full()
        return [s.annotations["source_id"] for s in self.spiketrains]

    def mean_rates(self, start=None, end=None):
        """
        Returns the mean rates of the spiketrains in spikes/s.
        """
        if start != None:
            start = start.rescale(qt.s)
            end = end.rescale(qt.s)
            return [
                len(s.time_slice(start, end)) / (end.magnitude - start.magnitude)
                for s in self.spiketrains
            ]
        else:
            return [
                len(s)
                / (s.t_stop.rescale(qt.s).magnitude - s.t_start.rescale(qt.s).magnitude)
                for s in self.spiketrains
            ]

    def isi(self):
        """
        Returns an array containing arrays (one per each neurons) with the inter-spike intervals of the SpikeTrain objects.
        """
        return [numpy.diff(s) for s in self.spiketrains]

    def cv_isi(self):
        """
        Return array with the coefficient of variation of the isis, one per each neuron.

        cv_isi is the ratio between the standard deviation and the mean of the ISI
        The irregularity of individual spike trains is measured by the squared
        coefficient of variation of the corresponding inter-spike interval (ISI)
        distribution.
        In point processes, low values reflect more regular spiking, a
        clock-like pattern yields CV2= 0. On the other hand, CV2 = 1 indicates
        Poisson-type behavior. As a measure for irregularity in the network one
        can use the average irregularity across all neurons.

        http://en.wikipedia.org/wiki/Coefficient_of_variation
        """
        isi = self.isi()
        cv_isi = []
        for _isi in isi:
            if len(_isi) > 4:
                cv_isi.append(numpy.std(_isi) / numpy.mean(_isi))
            else:
                cv_isi.append(None)
        return cv_isi


class PickledDataStoreNeoWrapper(MozaikSegment):
    """
    This is a Mozaik wrapper of neo segment, that enables pickling and lazy loading.
    """

    def __init__(self, segment, identifier, datastore_path, null=False):
        MozaikSegment.__init__(self, segment, identifier, null)
        self.datastore_path = datastore_path

    def load_full(self):
        s = load_pickle_crosscompat(self.datastore_path + "/" + self.identifier + ".pickle")
        self._spiketrains = s.spiketrains
        self.analogsignals = s.analogsignals
        self.full = True

    def __getstate__(self):
        result = self.__dict__.copy()
        if self.full:
            del result["_spiketrains"]
            del result["analogsignals"]
        return result

    def release(self):
        self.full = False
        del self._spiketrains
        del self.analogsignals
