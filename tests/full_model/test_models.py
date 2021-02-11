"""
This module contains tests that run mozaik models and compare their output to a
saved reference.
"""

import numpy as np
import os
from mozaik.storage.queries import *
from mozaik.storage.datastore import PickledDataStore
from parameters import ParameterSet

import pytest


class TestModel(object):
    """
    Parent class for checking outputs of models.
    Contains functions for checking if recorded membrane potentials or spike times
    are equal in two recorded datastores.
    """

    @staticmethod
    def load_datastore(base_dir):
        """
        Load PickledDataStore for reading.

        Parameters
        ----------
        base_dir : base directory where DataStore files are saved

        Returns
        -------
        PickledDataStore with the data from base_dir
        """
        return PickledDataStore(
            load=True,
            parameters=ParameterSet(
                {"root_directory": base_dir, "store_stimuli": False}
            ),
            replace=False,
        )

    def get_segments(self, data_store, sheet_name=None):
        """
        Returns a list of segments in the DataStore, ordered by their identifier.
        Optionally filters the segments for results from a specific sheet.

        Parameters
        ----------

        data_store : Datastore to retrieve segments from
        sheet_name : name of neuron sheet (layer) to retrieve the recorded segments for

        Returns
        -------
        A list of all segments in a DataStore, optionally filtered for sheet name.
        """

        if sheet_name is None:
            # If no sheet name specified, load all sheets
            return data_store.get_segments()
        else:
            return sorted(
                param_filter_query(data_store, sheet_name=sheet_name).get_segments(),
                key=lambda x: x.identifier,
            )

    def get_voltages(self, data_store, sheet_name=None, max_neurons=None):
        """
        Returns the recorded membrane potentials for neurons recorded in the DataStore,
        ordered by segment identifiers and neuron ids, flattened into a single 1D vector.
        Can be optionally filtered for neuron sheet, and maximum number of neurons.

        Parameters
        ----------

        data_store : Datastore to retrieve voltages from
        sheet_name : name of neuron sheet (layer) to retrieve the recorded voltages from
        max_neurons : maximum number of neurons to get the voltages for

        Returns
        -------
        A 1D list of membrane potential voltages recorded in neurons in the DataStore
        """

        segments = self.get_segments(data_store, sheet_name)
        return [
            v
            for segment in segments
            for neuron_id in sorted(segment.get_stored_vm_ids())[:max_neurons]
            for v in segment.get_vm(neuron_id).flatten()
        ]

    def get_spikes(self, data_store, sheet_name=None, max_neurons=None):
        """
        Returns the recorded spike times for neurons recorded in the DataStore,
        ordered by segment identifiers and neuron ids, flattened into a single 1D vector.
        Can be optionally filtered for neuron sheet, and maximum number of neurons.

        Parameters
        ----------

        data_store : Datastore to retrieve spike times from
        sheet_name : name of neuron sheet (layer) to retrieve the recorded spike times from
        max_neurons : maximum number of neurons to get spike times for

        Returns
        -------
        A 1D list of spike times recorded in neurons in the DataStore
        """
        segments = self.get_segments(data_store, sheet_name)
        return [
            v
            for segment in segments
            for neuron_id in sorted(segment.get_stored_spike_train_ids())[:max_neurons]
            for v in segment.get_spiketrain(neuron_id).flatten()
        ]

    def check_spikes(self, ds0, ds1, sheet_name=None, max_neurons=None):
        """
        Check if spike times recorded in two DataStores are equal. Spike times are merged
        into a single 1D array and compared using numpy assertions.
        Can be optionally filtered for neuron sheet, and maximum number of neurons.

        Parameters
        ----------

        ds0, ds1 : DataStores to retrieve spike times from
        sheet_name : name of neuron sheet (layer) to check spike times for
        max_neurons : maximum number of neurons to check spike times for
        """

        np.testing.assert_equal(
            self.get_spikes(ds0, sheet_name, max_neurons),
            self.get_spikes(ds1, sheet_name, max_neurons),
        )

    def check_voltages(self, ds0, ds1, sheet_name=None, max_neurons=None):
        """
        Check if membrane potential voltages recorded in two DataStores are equal. Voltages
        are merged into a single 1D array and compared using numpy assertions.
        Can be optionally filtered for neuron sheet, and maximum number of neurons.

        Parameters
        ----------

        ds0, ds1 : DataStores to retrieve spike times from
        sheet_name : name of neuron sheet (layer) to check voltages for
        max_neurons : maximum number of neurons to check voltages for
        """

        np.testing.assert_equal(
            self.get_voltages(ds0, sheet_name, max_neurons),
            self.get_voltages(ds1, sheet_name, max_neurons),
        )


class TestVogelsAbbott2005(TestModel):
    """
    Class that runs the VogelsAbbott2005 model on construction. Its testing methods
    compare the membrane potentials of a few neurons and the spike times of all neurons
    to a saved reference.
    """

    result_path = "examples/VogelsAbbott2005/VogelsAbbott2005_pytest_____"
    ref_path = "tests/full_model/reference_data/VogelsAbbott2005"

    ds = None  # Model run datastore
    ds_ref = None  # Reference datastore

    @classmethod
    def setup_class(cls):
        """
        Runs the VogelsAbbott2005 model and loads its result and a saved reference result
        """
        # Rerun test if it already ran
        if os.path.exists(cls.result_path):
            os.system("rm -r " + cls.result_path)
        os.system(
            "cd examples/VogelsAbbott2005 && python run.py nest 2 param/defaults 'pytest' && cd ../.."
        )

        # Load DataStore of recordings from the model that just ran
        cls.ds = cls.load_datastore(cls.result_path)
        # Load DataStore of reference recordings
        cls.ds_ref = cls.load_datastore(cls.ref_path)

    @pytest.mark.model
    @pytest.mark.parametrize("sheet_name", ["Exc_Layer", "Inh_Layer"])
    def test_spikes(self, sheet_name):
        self.check_spikes(self.ds, self.ds_ref, sheet_name)

    @pytest.mark.model
    @pytest.mark.parametrize("sheet_name", ["Exc_Layer", "Inh_Layer"])
    def test_voltages(self, sheet_name):
        self.check_voltages(self.ds, self.ds_ref, sheet_name, max_neurons=5)
