from mozaik.storage import neo_neurotools_wrapper as neowrap
from mozaik.storage.queries import *
import cPickle

class TestMozaikSegment:
    """
    Tests for allowing lists/numpy_arrays of neurons when retrieving vm/esyn/isyn
    """
    @staticmethod
    def load_segment():
        dir = 'model_data'
        seg_file = 'Segment'
        f = open(dir + "/" + seg_file + ".pickle", 'rb')
        s = cPickle.load(f)
        f.close()
        return neowrap.PickledDataStoreNeoWrapper(s, seg_file, dir)

    def test_get_vm(self):
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
        segment=self.load_segment()
        stored_neuron_ids = segment.get_stored_vm_ids()

        # Test individual neuron
        input = stored_neuron_ids[0]
        output = segment.get_vm(input)
        assert type(input) == numpy.int64
        assert output.shape[1] == 1

        # Test list of neurons
        input = list(stored_neuron_ids)
        output = segment.get_vm(input)
        assert type(input) == list
        # asserts that asking for the recordings of the array is the same values as asking it for each individual neuron
        assert [output[i] == segment.get_vm(input[i]) for i in range(len(input))]

        # Test numpy array of neurons
        input = stored_neuron_ids
        output = segment.get_vm(input)
        assert type(input) == numpy.ndarray
        # asserts that asking for the recordings of the array is the same values as asking it for each individual neuron
        assert [output[i] == segment.get_vm(input[i]) for i in range(len(input))]

    def test_get_esyn(self):
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
        segment=self.load_segment()
        stored_neuron_ids = segment.get_stored_vm_ids()

        # Test individual neuron
        input = stored_neuron_ids[0]
        output = segment.get_esyn(input)
        assert type(input) == numpy.int64
        assert output.shape[1] == 1

        # Test list of neurons
        input = list(stored_neuron_ids)
        output = segment.get_esyn(input)
        assert type(input) == list
        # asserts that asking for the recordings of the array is the same values as asking it for each individual neuron
        assert [output[i] == segment.get_esyn(input[i]) for i in range(len(input))]

        # Test numpy array of neurons
        input = stored_neuron_ids
        output = segment.get_esyn(input)
        assert type(input) == numpy.ndarray
        # asserts that asking for the recordings of the array is the same values as asking it for each individual neuron
        assert [output[i] == segment.get_esyn(input[i]) for i in range(len(input))]

    def test_get_isyn(self):
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
        segment=self.load_segment()
        stored_neuron_ids = segment.get_stored_vm_ids()

        #Test individual neuron
        input=stored_neuron_ids[0]
        output=segment.get_isyn(input)
        assert type(input)==numpy.int64
        assert output.shape[1]==1

        #Test list of neurons
        input=list(stored_neuron_ids)
        output=segment.get_isyn(input)
        assert type(input)==list
        # asserts that asking for the recordings of the array is the same values as asking it for each individual neuron
        assert [output[i]==segment.get_isyn(input[i]) for i in range(len(input))]

        #Test numpy array of neurons
        input=stored_neuron_ids
        output=segment.get_isyn(input)
        assert type(input) == numpy.ndarray
        # asserts that asking for the recordings of the array is the same values as asking it for each individual neuron
        assert [output[i]==segment.get_isyn(input[i]) for i in range(len(input))]

class TestPickledDataStoreNeoWrapper:
    pass