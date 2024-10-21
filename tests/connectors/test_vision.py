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
from scipy.stats import pearsonr
import logging
import itertools
import os
from collections import OrderedDict

import pytest

np.random.seed(1024)  # Make random tests deterministic


class TestModularConnector:
    def setup_class(cls):
        os.chdir("tests/connectors/ModularConnectorFunctionTest/")
        parameters = MozaikExtendedParameterSet("param/defaults")
        p = OrderedDict()
        if "mozaik_seed" in parameters:
            p["mozaik_seed"] = parameters["mozaik_seed"]
        if "pynn_seed" in parameters:
            p["pynn_seed"] = parameters["pynn_seed"]

        mozaik.setup_mpi(**p)
        parameters = MozaikExtendedParameterSet("param/defaults")
        parameters_selfconnections = MozaikExtendedParameterSet(
            "param_selfconnections/defaults"
        )

        import pyNN.nest as sim
        from tests.connectors.MapDependentModularConnectorFunctionTest.model import (
            ModelMapDependentModularConnectorFunction,
        )

        model = ModelMapDependentModularConnectorFunction(sim, 1, parameters)
        pos = model.sheets["sheet"].pop.positions
        model_selfconnections = ModelMapDependentModularConnectorFunction(
            sim, 1, parameters_selfconnections
        )

        cls.weights = model.connectors["RecurrentConnection"].proj.get(
            "weight", format="array", gather=True
        )
        cls.weights_selfconnections = model_selfconnections.connectors[
            "RecurrentConnection"
        ].proj.get("weight", format="array", gather=True)
        os.chdir("../../../")

    def test_no_self_connections(self):
        cnt = 0
        for i in numpy.arange(self.weights.shape[0]):
            if not numpy.isnan(self.weights[i, i]):
                cnt += 1
                break
        assert cnt == 0

    def test_self_connections(self):
        cnt = 0
        for i in numpy.arange(self.weights_selfconnections.shape[0]):
            if not numpy.isnan(self.weights_selfconnections[i, i]):
                cnt += 1
        assert cnt > 0

    pass


class TestMapDependentModularConnectorFunction:
    def setup_class(cls):
        os.chdir("tests/connectors/MapDependentModularConnectorFunctionTest/")
        parameters = MozaikExtendedParameterSet("param/defaults")
        p = OrderedDict()
        if "mozaik_seed" in parameters:
            p["mozaik_seed"] = parameters["mozaik_seed"]
        if "pynn_seed" in parameters:
            p["pynn_seed"] = parameters["pynn_seed"]

        mozaik.setup_mpi(**p)
        parameters = MozaikExtendedParameterSet("param/defaults")
        parameters_stretch = MozaikExtendedParameterSet("param_stretch/defaults")

        import pyNN.nest as sim
        from tests.connectors.MapDependentModularConnectorFunctionTest.model import (
            ModelMapDependentModularConnectorFunction,
            ModelMapDependentModularConnectorFunctionStretch,
        )

        model = ModelMapDependentModularConnectorFunction(sim, 1, parameters)
        pos = model.sheets["sheet"].pop.positions
        model_stretch = ModelMapDependentModularConnectorFunctionStretch(
            sim,
            1,
            pos
            * parameters_stretch.sheets.sheet.RecurrentConnection.weight_functions.f1.params.map_stretch,
            parameters_stretch,
        )

        cls.orr = [
            ann["LGNAfferentOrientation"]
            for ann in model.sheets["sheet"].get_neuron_annotations()
        ]
        cls.orr_stretch = [
            ann["LGNAfferentOrientation"]
            for ann in model_stretch.sheets["sheet"].get_neuron_annotations()
        ]
        os.chdir("../../../")

    def test_stretch_orientation_map(self):
        corr = pearsonr(self.orr, self.orr_stretch)
        assert corr.statistic > 0.99 and corr.pvalue < 0.001

    pass


class TestV1PushPullArborization:
    pass


class TestGaborConnector:
    def setup_class(cls):
        os.chdir("tests/connectors/GaborConnectorTest/")
        parameters = MozaikExtendedParameterSet("param/defaults")
        p = OrderedDict()
        if "mozaik_seed" in parameters:
            p["mozaik_seed"] = parameters["mozaik_seed"]
        if "pynn_seed" in parameters:
            p["pynn_seed"] = parameters["pynn_seed"]

        mozaik.setup_mpi(**p)
        parameters = MozaikExtendedParameterSet("param/defaults")
        parameters_stretch = MozaikExtendedParameterSet("param_stretch/defaults")

        import pyNN.nest as sim
        from tests.connectors.GaborConnectorTest.model import (
            ModelGaborConnector,
            ModelGaborConnectorStretch,
        )

        model = ModelGaborConnector(sim, 1, parameters)
        pos = model.sheets["sheet"].pop.positions
        model_stretch = ModelGaborConnectorStretch(
            sim,
            1,
            pos * parameters_stretch.sheets.sheet.AfferentConnection.or_map_stretch,
            parameters_stretch,
        )

        cls.orr = [
            ann["LGNAfferentOrientation"]
            for ann in model.sheets["sheet"].get_neuron_annotations()
        ]
        cls.orr_stretch = [
            ann["LGNAfferentOrientation"]
            for ann in model_stretch.sheets["sheet"].get_neuron_annotations()
        ]

        os.chdir("../../../")

    def test_stretch_orientation_map(self):
        corr = pearsonr(self.orr, self.orr_stretch)
        assert corr.statistic > 0.99 and corr.pvalue < 0.001


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
        assert abs(sum(self.weights[:, 2]) - sum(self.weights_lm[:, 2])) < 0.00001

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


class TestModularNumSamples:
    def setup_class(cls):
        os.chdir("tests/connectors/ModularNumSamplesTest/")
        parameters = MozaikExtendedParameterSet("param/defaults")
        p = OrderedDict()
        if "mozaik_seed" in parameters:
            p["mozaik_seed"] = parameters["mozaik_seed"]
        if "pynn_seed" in parameters:
            p["pynn_seed"] = parameters["pynn_seed"]

        mozaik.setup_mpi(**p)
        parameters = MozaikExtendedParameterSet("param/defaults")
        import pyNN.nest as sim
        from tests.connectors.ModularNumSamplesTest.model import ModelModularNumSamples

        model = ModelModularNumSamples(sim, 1, parameters)
        cls.pos = model.sheets["sheet"].pop.positions
        cls.weights_lin = numpy.array(
            model.connectors["LinearConnection"].proj.get(
                "weight", format="list", gather=True
            )
        )[:, :]

        cls.weights_quad = numpy.array(
            model.connectors["QuadraticConnection"].proj.get(
                "weight", format="list", gather=True
            )
        )[:, :]

        cls.weights_exp = numpy.array(
            model.connectors["ExponentialConnection"].proj.get(
                "weight", format="list", gather=True
            )
        )[:, :]

        cls.num_samples_lin = model.connectors[
            "LinearConnection"
        ].parameters.num_samples.next()
        cls.base_weight_lin = model.connectors[
            "LinearConnection"
        ].parameters.base_weight.next()
        cls.num_samples_quad = model.connectors[
            "QuadraticConnection"
        ].parameters.num_samples.next()
        cls.base_weight_quad = model.connectors[
            "QuadraticConnection"
        ].parameters.base_weight.next()
        cls.num_samples_exp = model.connectors[
            "ExponentialConnection"
        ].parameters.num_samples.next()
        cls.base_weight_exp = model.connectors[
            "ExponentialConnection"
        ].parameters.base_weight.next()

        # These two parameters have the same values for all 3 connections for simplification
        cls.threshold = model.connectors[
            "LinearConnection"
        ].parameters.num_samples_functions.n1.params.threshold
        cls.max_decrease = model.connectors[
            "LinearConnection"
        ].parameters.num_samples_functions.n1.params.max_decrease

        cls.exponent_factor = model.connectors[
            "ExponentialConnection"
        ].parameters.num_samples_functions.n1.params.exponent_factor

        cls.size_x = model.sheets["sheet"].size_x
        cls.size_y = model.sheets["sheet"].size_y

        cls.center = []
        cls.surround = []
        for idd in range(cls.pos.shape[1]):
            posx = cls.pos[0, idd] + cls.size_x / 2
            posy = cls.pos[1, idd] + cls.size_y / 2

            if (
                posx > cls.threshold
                and cls.size_x - cls.threshold > posx
                and posy > cls.threshold
                and cls.size_y - cls.threshold > posy
            ):
                cls.center.append(idd)
            else:
                cls.surround.append(idd)

        os.chdir("../../../")

    def test_center_homogeneity_linear(self):
        if len(self.center) == 0:
            pytest.skip("All neurons are close to the borders of the sheet")
        assert (
            abs(
                numpy.sum(
                    self.weights_lin[numpy.isin(self.weights_lin[:, 1], self.center), 2]
                )
                - len(self.center) * self.base_weight_lin * self.num_samples_lin
            )
            < 0.00001
        )

    def test_center_homogeneity_quadratic(self):
        if len(self.center) == 0:
            pytest.skip("All neurons are close to the borders of the sheet")
        assert (
            abs(
                numpy.sum(
                    self.weights_quad[
                        numpy.isin(self.weights_quad[:, 1], self.center), 2
                    ]
                )
                - len(self.center) * self.base_weight_quad * self.num_samples_quad
            )
            < 0.00001
        )

    def test_center_homogeneity_exponential(self):
        if len(self.center) == 0:
            pytest.skip("All neurons are close to the borders of the sheet")
        assert (
            abs(
                numpy.sum(
                    self.weights_exp[numpy.isin(self.weights_exp[:, 1], self.center), 2]
                )
                - len(self.center) * self.base_weight_exp * self.num_samples_exp
            )
            < 0.00001
        )

    def test_max_decrease_linear(self):
        if len(self.surround) == 0:
            pytest.skip("No neurons are sufficiently close to the borders of the sheet")
        assert (
            numpy.min(
                [
                    numpy.sum(self.weights_lin[self.weights_lin[:, 1] == idd, 2])
                    for idd in self.surround
                ]
            )
            >= self.base_weight_lin * self.num_samples_lin / self.max_decrease**2
        )

    def test_max_decrease_quadratic(self):
        if len(self.surround) == 0:
            pytest.skip("No neurons are sufficiently close to the borders of the sheet")
        assert (
            numpy.min(
                [
                    numpy.sum(self.weights_quad[self.weights_quad[:, 1] == idd, 2])
                    for idd in self.surround
                ]
            )
            >= self.base_weight_quad * self.num_samples_quad / self.max_decrease**2
        )

    def test_max_decrease_exponential(self):
        if len(self.surround) == 0:
            pytest.skip("No neurons are sufficiently close to the borders of the sheet")
        assert (
            numpy.min(
                [
                    numpy.sum(self.weights_exp[self.weights_exp[:, 1] == idd, 2])
                    for idd in self.surround
                ]
            )
            >= self.base_weight_exp * self.num_samples_exp / self.max_decrease**2
        )

    def test_num_samples_surround_linear(self):
        if len(self.surround) == 0:
            pytest.skip("No neurons are sufficiently close to the borders of the sheet")

        cnt = 0
        for idd in self.surround:
            coef = 1
            posx = self.pos[0, idd] + self.size_x / 2
            posy = self.pos[1, idd] + self.size_y / 2

            if posx < self.threshold:
                coef *= 1 / self.max_decrease + (1 - 1 / self.max_decrease) * (
                    posx / self.threshold
                )
            elif self.size_x - posx < self.threshold:
                coef *= 1 / self.max_decrease + (1 - 1 / self.max_decrease) * (
                    (self.size_x - posx) / self.threshold
                )

            if posy < self.threshold:
                coef *= 1 / self.max_decrease + (1 - 1 / self.max_decrease) * (
                    posy / self.threshold
                )
            elif self.size_y - posy < self.threshold:
                coef *= 1 / self.max_decrease + (1 - 1 / self.max_decrease) * (
                    (self.size_y - posy) / self.threshold
                )

            if (
                abs(
                    numpy.sum(self.weights_lin[self.weights_lin[:, 1] == idd, 2])
                    - self.base_weight_lin * round(self.num_samples_lin * coef)
                )
                < 0.00001
            ):
                cnt += 1

        assert cnt == len(self.surround)

    def test_num_samples_surround_quadratic(self):
        if len(self.surround) == 0:
            pytest.skip("No neurons are sufficiently close to the borders of the sheet")

        cnt = 0
        for idd in self.surround:
            coef = 1
            posx = self.pos[0, idd] + self.size_x / 2
            posy = self.pos[1, idd] + self.size_y / 2

            if posx < self.threshold:
                coef *= 1 / self.max_decrease + (
                    1 - 1 / self.max_decrease
                ) * numpy.sqrt(posx / self.threshold)
            elif self.size_x - posx < self.threshold:
                coef *= 1 / self.max_decrease + (
                    1 - 1 / self.max_decrease
                ) * numpy.sqrt((self.size_x - posx) / self.threshold)

            if posy < self.threshold:
                coef *= 1 / self.max_decrease + (
                    1 - 1 / self.max_decrease
                ) * numpy.sqrt(posy / self.threshold)
            elif self.size_y - posy < self.threshold:
                coef *= 1 / self.max_decrease + (
                    1 - 1 / self.max_decrease
                ) * numpy.sqrt((self.size_y - posy) / self.threshold)

            if (
                abs(
                    numpy.sum(self.weights_quad[self.weights_quad[:, 1] == idd, 2])
                    - self.base_weight_quad * round(self.num_samples_quad * coef)
                )
                < 0.00001
            ):
                cnt += 1

        assert cnt == len(self.surround)

    # We don't test that all neurons close to the border have num_samples lower than the normal value, because the rounding of the number of connections lead neurons that are only slightly below the threshold to have the same number of connections as neurons in the center
    def test_num_samples_surround_exponential(self):
        if len(self.surround) == 0:
            pytest.skip("No neurons are sufficiently close to the borders of the sheet")

        cnt = 0
        for idd in self.surround:
            coef = 1
            posx = self.pos[0, idd] + self.size_x / 2
            posy = self.pos[1, idd] + self.size_y / 2

            if posx < self.threshold:
                coef *= 1 / self.max_decrease + (1 - 1 / self.max_decrease) * (
                    -numpy.exp(-self.exponent_factor * posx / self.threshold) + 1
                ) / (-numpy.exp(-self.exponent_factor) + 1)
            elif self.size_x - posx < self.threshold:
                coef *= 1 / self.max_decrease + (1 - 1 / self.max_decrease) * (
                    -numpy.exp(
                        -self.exponent_factor * (self.size_x - posx) / self.threshold
                    )
                    + 1
                ) / (-numpy.exp(-self.exponent_factor) + 1)

            if posy < self.threshold:
                coef *= 1 / self.max_decrease + (1 - 1 / self.max_decrease) * (
                    -numpy.exp(-self.exponent_factor * posy / self.threshold) + 1
                ) / (-numpy.exp(-self.exponent_factor) + 1)
            elif self.size_y - posy < self.threshold:
                coef *= 1 / self.max_decrease + (1 - 1 / self.max_decrease) * (
                    -numpy.exp(
                        -self.exponent_factor * (self.size_y - posy) / self.threshold
                    )
                    + 1
                ) / (-numpy.exp(-self.exponent_factor) + 1)

            if (
                abs(
                    numpy.sum(self.weights_exp[self.weights_exp[:, 1] == idd, 2])
                    - self.base_weight_exp * round(self.num_samples_exp * coef)
                )
                < 0.00001
            ):
                cnt += 1

        assert cnt == len(self.surround)
