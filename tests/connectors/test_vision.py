import matplotlib

matplotlib.use("Agg")
from mozaik.connectors import vision
from mozaik.connectors.vision import V1CorrelationBasedConnectivity as V1CBC
from mozaik.tools.distribution_parametrization import (
    MozaikExtendedParameterSet,
    load_parameters,
)
import mozaik
import numpy as np
import numpy.linalg
import logging
import itertools
import os
from collections import OrderedDict

import pytest

np.random.seed(1024)  # Make random tests deterministic


class TestMapDependentModularConnectorFunction:
    pass


class TestV1PushPullArborization:
    pass


class TestGaborArborization:
    pass


class TestV1CorrelationBasedConnectivity:
    """
    Tests for the V1CorrelationBasedConnectivity class
    """

    num_tests = 100
    relative_error_tolerance = 0.02
    absolute_error_tolerance = 0.002

    def gabor_params(length=1):
        """
        Generate random parameters of a gabor function
        """
        params = []
        for i in range(0, length):
            k = 0.1 + np.random.rand() * 9.9
            a = 1 / (1 + np.random.rand() * 9)
            b = 1 / (1 + np.random.rand() * 9)
            f = 0.1 + np.random.rand() * 0.3
            x0 = -5 + np.random.rand() * 5
            y0 = -5 + np.random.rand() * 5
            omega = np.random.rand() * np.pi
            # in vision we always work with the special case where the orientation of the gaussian is the same as orientation along which the grating varies
            theta = omega
            p = np.random.rand() * np.pi * 2
            params.append([k, a, b, x0, y0, theta, f, omega, p])
        return params

    def gabor_relative_params(length=1):
        """
        Generate relative random parameters of a gabor function
        """
        params = []
        for i in range(0, length):
            x0 = -5 + np.random.rand() * 5
            y0 = -5 + np.random.rand() * 5
            F = 0.1 + np.random.rand() * 0.3
            orr = np.random.rand() * np.pi
            P = np.random.rand() * np.pi * 2
            size = np.random.rand() * 10
            aspect_ratio = 0.2 + np.random.rand() * 5
            params.append([size, x0, y0, aspect_ratio, orr, F, P])
        return params

    @staticmethod
    def gabor_connectivity_gabor(width, posx, posy, ar, orr, freq, phase):
        XX, YY = np.ogrid[-40:40:2000j, -40:40:2000j]
        return vision.gabor(XX, YY, posx, posy, orr, freq, phase, width, ar)

    @staticmethod
    def real_gabor(K, a, b, x0, y0, theta, F, omega, P):
        """
        Return samples of a gabor function with the specified parameters

        Parameters
        ----------

        a, b    - gaussian widths (1/over)
        x0, y0  - centre of gaussian
        F, P    - spatial frequency and phase of grating
        theta,omega  - orientation angles of gaussian and grating
        K       - scaler

        """

        XX, YY = np.ogrid[-40:40:2000j, -40:40:2000j]
        x_r = (XX - x0) * np.cos(theta) + (YY - y0) * np.sin(theta)
        y_r = -(XX - x0) * np.sin(theta) + (YY - y0) * np.cos(theta)
        gaussian = K * np.exp(-np.pi * (a**2 * x_r**2 + b**2 * y_r**2))
        complex_grating = np.exp(
            1j * 2 * np.pi * F * (XX * np.cos(omega) + YY * np.sin(omega)) + 1j * P
        )
        return np.real(gaussian * complex_grating)

    @pytest.mark.parametrize(
        "p1, p2",
        itertools.zip_longest(gabor_params(num_tests), gabor_params(num_tests)),
    )
    def test_integral_of_gabor_multiplication(self, p1, p2):
        """
        Compare empirical and analytical calculation of the integral of multiplication
        of two randomly generated gabor functions.

        Parameters
        ----------

        p1, p2 : list
                 Randomly generated parameters of a gabor function

        """
        gabor1, gabor2 = self.real_gabor(*p1), self.real_gabor(*p2)
        empir = np.dot(gabor1.flatten(), gabor2.flatten()) * (80.0 / 2000) ** 2
        anal = np.array(V1CBC.integral_of_gabor_multiplication(*(p1 + p2)))[0][0]
        np.testing.assert_allclose(
            empir,
            anal,
            rtol=self.relative_error_tolerance,
            atol=self.absolute_error_tolerance,
            err_msg="The integral of multiplication of two gabors with parameters %s and %s does not match. Empirical value: %g, analytical value: %g."
            % (p1, p2, empir, anal),
        )

    @pytest.mark.parametrize(
        "p1, p2",
        itertools.zip_longest(gabor_params(num_tests), gabor_params(num_tests)),
    )
    def test_gabor_correlation(self, p1, p2):
        """
        Compare empirical and analyitical calculation of the correlation
        of two randomly generated gabor functions.

        Parameters
        ----------

        p1, p2 : list
                 Randomly generated parameters of a gabor function

        """

        g1 = self.real_gabor(*p1).flatten()
        g2 = self.real_gabor(*p2).flatten()
        empir = np.dot(g1 - g1.mean(), g2 - g2.mean()) / (len(g1) * g1.std() * g2.std())
        anal = V1CBC.gabor_correlation(*(p1 + p2))
        np.testing.assert_allclose(
            empir,
            anal,
            rtol=self.relative_error_tolerance,
            atol=self.absolute_error_tolerance,
            err_msg="The correlation of two gabors with parameters %s and %s does not match. Empirical value: %g, analytical value: %g."
            % (p1, p2, empir, anal),
        )

    @pytest.mark.parametrize(
        "p1, p2",
        itertools.zip_longest(
            gabor_relative_params(num_tests), gabor_relative_params(num_tests)
        ),
    )
    def test_gabor_correlation_with_gaussian_used_for_connections(self, p1, p2):
        """
        Compare empirical and analyitical calculation of the correlation of two randomly
        generated gabor functions, where a Gaussian is used for connections.

        Parameters
        ----------

        p1, p2 : list
                 Randomly generated parameters of a gabor function

        """
        g1 = self.gabor_connectivity_gabor(*p1).flatten()
        g2 = self.gabor_connectivity_gabor(*p2).flatten()
        empir = np.dot(g1 - g1.mean(), g2 - g2.mean()) / (len(g1) * g1.std() * g2.std())
        anal = V1CBC.gabor_correlation_rescaled_parammeters(*(p1 + p2))
        np.testing.assert_allclose(
            empir,
            anal,
            rtol=self.relative_error_tolerance,
            atol=self.absolute_error_tolerance,
            err_msg="The correlation of two gabors with a gaussian used for connections with parameters %s and %s does not match. Empirical value: %g, analytical value: %g."
            % (p1, p2, empir, anal),
        )

    @pytest.mark.parametrize(
        "param1, param2", [(gabor_params(num_tests), gabor_params(num_tests))]
    )
    def test_integral_of_gabor_multiplication_vectorized(self, param1, param2):
        """
        Compare the output of the vectorized integral of gabor multiplication function
        to a vector of the integral of gabor multiplication function.

        Parameters
        ----------

        param1, param2 - list of lists
                         Randomly generated parameters of num_tests gabors
        """

        matrix = [
            np.array(V1CBC.integral_of_gabor_multiplication(*(p1 + p2)))[0][0]
            for p1, p2 in zip(param1, param2)
        ]
        param12 = tuple(np.array(param1).T) + tuple(np.array(param2).T)
        vectorized = V1CBC.integral_of_gabor_multiplication_vectorized(*param12)

        for i, (m, v) in enumerate(zip(matrix, vectorized)):
            np.testing.assert_allclose(
                m,
                v,
                rtol=self.relative_error_tolerance,
                atol=self.absolute_error_tolerance,
                err_msg="The integral of multiplication of two gabors with parameters %s and %s does not match. Matrix version: %g, vectorized version: %g."
                % (param1[i], param2[i], m, v),
            )


class TestCoCircularModularConnectorFunction:
    pass


class TestLocalModule:
    def setup_class(cls):
        os.chdir("tests/connectors/LocalModuleTest/")
        parameters = MozaikExtendedParameterSet("param/defaults")
        p = OrderedDict()
        if "mozaik_seed" in parameters:
            p["mozaik_seed"] = parameters["mozaik_seed"]
        if "pynn_seed" in parameters:
            p["pynn_seed"] = parameters["pynn_seed"]

        mozaik.setup_mpi(**p)
        parameters = MozaikExtendedParameterSet("param/defaults")
        import pyNN.nest as sim
        from tests.connectors.LocalModuleTest.model import ModelLocalModule

        model = ModelLocalModule(sim, 1, parameters)
        pos = model.sheets["sheet_lm"].pop.positions
        cls.dist = numpy.sqrt(pos[0] * pos[0] + pos[1] * pos[1])
        cls.weights = numpy.array(
            model.connectors["RecurrentConnection"].proj.get(
                "weight", format="list", gather=True
            )
        )[:, :]
        cls.weights_lm = numpy.array(
            model.connectors["RecurrentConnectionLM"].proj.get(
                "weight", format="list", gather=True
            )
        )[:, :]
        cls.in_radius = model.connectors[
            "RecurrentConnectionLM"
        ].parameters.local_module.in_radius
        cls.out_radius = model.connectors[
            "RecurrentConnectionLM"
        ].parameters.local_module.out_radius
        os.chdir("../../../")

    def test_weights_homogeneity(self):
        assert sum(self.weights[:, 2]) - sum(self.weights_lm[:, 2]) < 0.00001

    def test_separation_local_module(self):
        lm_ids = np.nonzero(self.dist < self.in_radius)[0]
        border_ids = np.nonzero(
            np.logical_and(self.dist < self.out_radius, self.dist > self.in_radius)
        )[0]
        num_pre = 0
        for idd in lm_ids:
            pre_idx = np.nonzero(self.weights_lm[:, 1] == idd)[0]
            pre_ids = self.weights_lm[pre_idx, 0]
            num_pre += numpy.intersect1d(pre_ids, border_ids).shape[0]
        assert num_pre == 0
