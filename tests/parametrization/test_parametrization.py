import matplotlib

matplotlib.use("Agg")
import mozaik
from mozaik.tools.distribution_parametrization import load_parameters
import os
from collections import OrderedDict
import pytest

class TestParametrization:
    """
    Tests for the TestParametrization class
    """
    def setup_class(cls):
        os.chdir("tests/parametrization/")

        p = OrderedDict()
        p["mozaik_seed"] = 1023
        p["pynn_seed"] = 936395
        mozaik.setup_mpi(**p)
        cls.param_original_PyNNDistribution = load_parameters('param_original_PyNNDistribution/defaults')

        mozaik.setup_mpi(**p)
        cls.param_parametrized_PyNNDistribution = load_parameters('param_parametrized_PyNNDistribution/defaults')

    def test_parametrization_PyNNDistribution(self):
        assert (
            self.param_original_PyNNDistribution.num_samples.next() == self.param_parametrized_PyNNDistribution.num_samples.next()
        ), "num_samples of the two parametrization generate different numbers"


