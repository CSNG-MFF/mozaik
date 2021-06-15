"""
This module contains tests that run mozaik tools functions
"""

import numpy as np
import os
from mozaik.storage.queries import *
from mozaik.tools.datastore_utils import *
from mozaik.storage.datastore import PickledDataStore
from mozaik.storage import queries
from parameters import ParameterSet

import pytest


class TestMergeDatastore(object):
    """
    Class that merges two reference datastores and merge them in a new datastore
    that is the saved. It then loads this datastore and tests if the segments of
    the reference datastore have been effectively copied into the merged datastore
    """

    merged_path_gratings = "tests/tools/MergedDatastoreGratings"
    merged_path_disks = "tests/tools/MergedDatastoreDisks"
    ref_path1 = "tests/tools/reference_data/ToMerge1"
    ref_path2 = "tests/tools/reference_data/ToMerge2"
    ref_path3 = "tests/tools/reference_data/ToMerge3"
    ref_path4 = "tests/tools/reference_data/ToMerge4"

    ds1 = None  # Reference datastore 1
    ds2 = None  # Reference datastore 2

    ds_merged = None  # Merged datastore

    @classmethod
    def setup_class(cls):
        """
        Load the reference datastores, merge them, save the datastores resulting from the merging, and load them again
        """
        # Rerun merge if it already ran
        if os.path.exists(cls.merged_path_gratings):
            os.system("rm -r " + cls.merged_path_gratings)
        if os.path.exists(cls.merged_path_disks):
            os.system("rm -r " + cls.merged_path_disks)

        # Load the DataStore of references recordings
        cls.ds1 = cls.load_datastore(cls.ref_path1)
        cls.ds2 = cls.load_datastore(cls.ref_path2)
        cls.ds3 = cls.load_datastore(cls.ref_path3)
        cls.ds4 = cls.load_datastore(cls.ref_path4)

        # Merge the two DataStore into one and saves it
        cls.ds_merged_gratings = merge_datastores(
            (cls.ds1, cls.ds2), cls.merged_path_gratings, True, True, True, True
        )
        cls.ds_merged_gratings.save()
        # Merge the two DataStore into one and saves it
        cls.ds_merged_disks = merge_datastores(
            (cls.ds3, cls.ds4), cls.merged_path_disks, True, True, True, True
        )
        cls.ds_merged_disks.save()

        # Load the merged DataStores
        cls.ds_merged_gratings_saved = cls.load_datastore(cls.merged_path_gratings)
        cls.ds_merged_disks_saved = cls.load_datastore(cls.merged_path_disks)

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

    def get_spikes(self, segment):
        """
        Returns the recorded spike times for neurons from a specific segment
        ordered by neuron ids, flattened into a single 1D vector.

        Parameters
        ----------

        segment : Segment to retrieve spike times from

        Returns
        -------
        A 1D list of spike times recorded in neurons in the Segment
        """
        return [
            v
            for neuron_id in sorted(segment.get_stored_spike_train_ids())
            for v in segment.get_spiketrain(neuron_id).flatten()
        ]

    def check_segments_gratings_merge(self, ref_dss, merged_ds):
        """
        Check if all the segments from the Drifting Sinusoidal Gratings experiment of the reference datastores are present in the merged datastore

        Parameters
        ----------

        ref_dss : An iterable object containing the reference datastores that have be used for the merge
        merged_ds : The DataStore object that resulted from the merge
        """

        merged_seg_count = len(merged_ds.get_segments())
        merged_null_seg_count = len(merged_ds.get_segments(null=True))
        ref_seg_count = 0
        ref_null_seg_count = 0
        for ds in ref_dss:
            ref_seg_count += len(ds.get_segments())
            ref_null_seg_count += len(ds.get_segments(null=True))
            segs = queries.param_filter_query(
                ds, st_name="DriftingSinusoidalGrating"
            ).get_segments()

            for seg in segs:
                stimulus = eval(seg.annotations["stimulus"])
                s = queries.param_filter_query(
                    merged_ds,
                    st_name="DriftingSinusoidalGrating",
                    sheet_name=seg.annotations["sheet_name"],
                    st_orientation=stimulus["orientation"],
                    st_contrast=stimulus["contrast"],
                    st_trial=stimulus["trial"],
                ).get_segments()[0]

                np.testing.assert_equal(
                    self.get_spikes(seg),
                    self.get_spikes(s),
                )
                seg.release()
                s.release()

        assert (
            merged_seg_count == ref_seg_count
        ), "The number of segments in the merged datastore must be equal to the sum of the number of segments in each reference datastores"
        assert (
            merged_null_seg_count == ref_null_seg_count
        ), "The number of null segments in the merged datastore must be equal to the sum of the number of null segments in each reference datastores"

    def check_segments_disks_merge(self, ref_dss, merged_ds):
        """
        Check if all the segments of the reference datastores are present in the merged datastore

        Parameters
        ----------

        ref_dss : An iterable object containing the reference datastores that have be used for the merge
        merged_ds : The DataStore object that resulted from the merge
        """

        merged_seg_count = len(merged_ds.get_segments())
        merged_null_seg_count = len(merged_ds.get_segments(null=True))
        ref_seg_count = 0
        ref_null_seg_count = 0
        for ds in ref_dss:
            ref_seg_count += len(ds.get_segments())
            ref_null_seg_count += len(ds.get_segments(null=True))
            segs = queries.param_filter_query(
                ds, st_name="DriftingSinusoidalGratingDisk"
            ).get_segments()

            for seg in segs:
                stimulus = eval(seg.annotations["stimulus"])
                s = queries.param_filter_query(
                    merged_ds,
                    st_name="DriftingSinusoidalGratingDisk",
                    sheet_name=seg.annotations["sheet_name"],
                    st_orientation=stimulus["orientation"],
                    st_contrast=stimulus["contrast"],
                    st_radius=stimulus["radius"],
                    st_trial=stimulus["trial"],
                ).get_segments()[0]

                np.testing.assert_equal(
                    self.get_spikes(seg),
                    self.get_spikes(s),
                )
                seg.release()
                s.release()

            segs_spont = queries.param_filter_query(
                ds, st_name="InternalStimulus"
            ).get_segments()
            for seg in segs_spont:
                stimulus = eval(seg.annotations["stimulus"])
                s = queries.param_filter_query(
                    merged_ds,
                    sheet_name=seg.annotations["sheet_name"],
                    st_name="InternalStimulus",
                ).get_segments()[0]
                np.testing.assert_equal(
                    self.get_spikes(seg),
                    self.get_spikes(s),
                )
                seg.release()
                s.release()

        assert (
            merged_seg_count == ref_seg_count
        ), "The number of segments in the merged datastore must be equal to the sum of the number of segments in each reference datastores"
        assert (
            merged_null_seg_count == ref_null_seg_count
        ), "The number of null segments in the merged datastore must be equal to the sum of the number of null segments in each reference datastores"

    @pytest.mark.merge
    def test_merge(self):
        self.check_segments_gratings_merge(
            (self.ds1, self.ds2), self.ds_merged_gratings
        )
        self.check_segments_gratings_merge(
            (self.ds1, self.ds2), self.ds_merged_gratings_saved
        )
        self.check_segments_disks_merge((self.ds3, self.ds4), self.ds_merged_disks)
        self.check_segments_disks_merge(
            (self.ds3, self.ds4), self.ds_merged_disks_saved
        )
