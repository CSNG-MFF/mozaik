import matplotlib

matplotlib.use("Agg")
from mozaik.analysis.analysis import * 
from mozaik.storage.datastore import * 
from mozaik.storage.queries import param_filter_query 
import mozaik
import logging
import os

import pytest


class TestDatastore:

    ref_path = "tests/full_model/reference_data/LSV1M_tiny"

    @classmethod
    def setup_class(cls):
        """
        Runs the model and loads its result and a saved reference result
        """
        cls.ds = cls.load_datastore(cls.ref_path)
        TrialAveragedFiringRate(param_filter_query(cls.ds,st_name='FullfieldDriftingSinusoidalGrating'), ParameterSet({})).analyse()
        cls.ads = cls.ds.get_analysis_result()

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
    def test_ADS_sorting_homogeneity(self):
        dsv_sorted = param_filter_query(self.ds,st_name='FullfieldDriftingSinusoidalGrating')
        dsv_sorted.sort_analysis_results('sheet_name')
        adss_sorted = dsv_sorted.get_analysis_result()
        count_incorrect = 0
        for ads in adss_sorted:
            if ads not in self.ads:
                count_incorrect +=1
        for ads in self.ads:
            if ads not in adss_sorted:
                count_incorrect +=1
        assert count_incorrect == 0 
    pass

    def test_ADS_sorting_st_homogeneity(self):
        dsv_sorted = param_filter_query(self.ds)
        dsv_sorted.sort_analysis_results('st_orientation')
        adss_sorted = dsv_sorted.get_analysis_result()
        count_incorrect = 0
        for ads in adss_sorted:
            if ads not in self.ads:
                count_incorrect +=1
        for ads in self.ads:
            if ads not in adss_sorted:
                count_incorrect +=1
        assert count_incorrect == 0
    pass

    def test_ADS_sorting_order(self):
        dsv_sorted = param_filter_query(self.ds)
        dsv_sorted.sort_analysis_results('sheet_name')
        adss_sorted = dsv_sorted.get_analysis_result()
        count_correct = 0
        for ads,ads_next in zip(adss_sorted[:-1],adss_sorted[1:]):
            if not hasattr(ads,'sheet_name'):
                count_correct += 1
            elif hasattr(ads_next,'sheet_name') and getattr(ads,'sheet_name') <= getattr(ads_next,'sheet_name'):
                    count_correct +=1
        assert count_correct + 1 == len(adss_sorted)
    pass


    def test_ADS_sorting_st_order(self):
        dsv_sorted = param_filter_query(self.ds)
        dsv_sorted.sort_analysis_results('st_orientation')
        adss_sorted = dsv_sorted.get_analysis_result()
        count_correct = 0
        for ads,ads_next in zip(adss_sorted[:-1],adss_sorted[1:]):
            if ads.stimulus_id == None:
                count_correct += 1
            elif ads_next.stimulus_id and not hasattr(ads.stimulus_id,'st_orientation'):
                count_correct += 1
            elif ads_next.stimulus_id and hasattr(ads_next.stimulus_id,'st_orientation'):
                if getattr(MozaikParametrized.idd(ads.stimulus_id),'st_orientation') <= getattr(MozaikParametrized.idd(ads_next),'st_orientation'):
                    count_correct +=1
        assert count_correct + 1 == len(adss_sorted)
    pass
