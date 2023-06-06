"""
This module contains tests that run mozaik models and compare their output to a
saved reference.
"""

import numpy as np
import os
from mozaik.storage.queries import *
from mozaik.storage.datastore import PickledDataStore
from parameters import ParameterSet
from test_models import TestModel

import pytest
import mozaik


class TestLSV1MTinyMPI(TestModel):
    """
    Class that runs the a tiny version of the LSV1M model on construction from the mozaik-models
    repository and runs it with MPI. Its testing methods compare the membrane potentials of a
    few neurons and the spike times of all neurons to a saved reference.
    """

    model_run_command = "cd tests/full_model/models/LSV1M_tiny && mpirun -np 2 python run.py nest 1 param/defaults 'pytest' && cd ../../../.."
    result_path = "tests/full_model/models/LSV1M_tiny/LSV1M_pytest_____"
    ref_path = "tests/full_model/reference_data/LSV1M_tiny_mpi"

    ds = None  # Model run datastore
    ds_ref = None  # Reference datastore

    @pytest.mark.model
    @pytest.mark.mpi
    @pytest.mark.parametrize(
        "sheet_name", ["V1_Exc_L4", "V1_Inh_L4", "V1_Exc_L2/3", "V1_Inh_L2/3"]
    )
    def test_spikes(self, sheet_name):
        self.check_spikes(self.ds, self.ds_ref, sheet_name)

    @pytest.mark.model
    @pytest.mark.mpi
    @pytest.mark.parametrize(
        "sheet_name", ["V1_Exc_L4", "V1_Inh_L4", "V1_Exc_L2/3", "V1_Inh_L2/3"]
    )
    def test_voltages(self, sheet_name):
        self.check_voltages(self.ds, self.ds_ref, sheet_name, max_neurons=25)

    @pytest.mark.model
    @pytest.mark.mpi
    def test_mozaik_rng_mpi2(self):
        rngs_state = self.ds.block.annotations["simulation_log"]["rngs_state"]
        print(rngs_state)
        assert len(set(rngs_state)) == 1

class TestLSV1MTinyMPI7(TestModel):
    """
    Class that runs the a tiny version of the LSV1M model on construction from the mozaik-models
    repository and runs it with MPI using 7 processes. Its testing methods compare the membrane
    potentials of a few neurons and the spike times of all neurons to a saved reference.
    """

    model_run_command = "cd tests/full_model/models/LSV1M_tiny && mpirun -np 7 python run.py nest 1 param/defaults 'pytest' && cd ../../../.."
    result_path = "tests/full_model/models/LSV1M_tiny/LSV1M_pytest_____"
    ref_path = "tests/full_model/reference_data/LSV1M_tiny_mpi"

    ds = None  # Model run datastore
    ds_ref = None  # Reference datastore

    @pytest.mark.model
    @pytest.mark.mpi
    @pytest.mark.not_github
    def test_mozaik_rng_mpi7(self):
        rngs_state = self.ds.block.annotations["simulation_log"]["rngs_state"]
        print(rngs_state)
        assert len(set(rngs_state)) == 1
