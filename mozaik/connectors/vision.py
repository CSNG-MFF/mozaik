"""
Vision specific connectors.
"""
import logging

from parameters import ParameterSet
from scipy.interpolate import NearestNDInterpolator
import matplotlib.pyplot as plt
import numpy as np

from ..controller import Global
from ..tools.circ_stat import *
from ..tools.misc import *
from .modular_connector_functions import ModularConnectorFunction

logger = logging.getLogger(__name__)


class MapDependentModularConnectorFunction(ModularConnectorFunction):
    """
    Corresponds to: distance*linear_scaler + constant_scaler
    """

    required_parameters = ParameterSet(
        {
            # It has to point to a file containing a single pickled 2d numpy array, containing values in the interval [0..1].
            "map_location": str,
            # How sharply does the wieght fall off with the increasing distance between the map values (exp(-0.5*(distance/sigma)*2)/(sigma*sqrt(2*np.pi)))
            "sigma": float,
            # if true, the values in map will be treated as periodic (and consequently the distance between two values will be computed as circular distance).
            "periodic": bool
        }
    )

    def __init__(self, source, target, parameters):
        import pickle

        ModularConnectorFunction.__init__(self, source, target, parameters)
        t_size = target.size_in_degrees()
        f = open(self.parameters.map_location, "r")
        mmap = pickle.load(f)
        coords_x = np.linspace(-t_size[0] / 2.0, t_size[0] / 2.0, np.shape(mmap)[0])
        coords_y = np.linspace(-t_size[1] / 2.0, t_size[1] / 2.0, np.shape(mmap)[1])
        X, Y = np.meshgrid(coords_x, coords_y)
        self.mmap = NearestNDInterpolator(
            list(zip(X.flatten(), Y.flatten())), mmap.flatten()
        )
        self.val_source = (
            self.mmap(
                np.transpose(
                    np.array(
                        [self.source.pop.positions[0], self.source.pop.positions[1]]
                    )
                )
            )
            * np.pi
        )

        for (index, neuron2) in enumerate(target.pop.all()):
            val_target = self.mmap(
                self.target.pop.positions[0][index], self.target.pop.positions[1][index]
            )
            self.target.add_neuron_annotation(
                index, "LGNAfferentOrientation", val_target * np.pi, protected=False
            )

    def evaluate(self, index):
        val_target = self.target.get_neuron_annotation(index, "LGNAfferentOrientation")
        if self.parameters.periodic:
            distance = circular_dist(self.val_source, val_target, np.pi)
        else:
            distance = np.abs(self.val_source - val_target)
        return np.exp(-0.5 * (distance / self.parameters.sigma) ** 2) / (
            self.parameters.sigma * np.sqrt(2 * np.pi)
        )


class V1PushPullArborization(ModularConnectorFunction):
    """
    This connector function implements the standard V1 functionally specific
    connection rule:

    Excitatory synapses are more likely on cooriented in-phase neurons
    Inhibitory synapses are more likely to cooriented anti-phase neurons


    Notes
    -----
    The `push_pull_ratio` parameter essentially adds constant to the probability distribution
    over the orientation and phase dependent connectivity probability space. The formula looks
    like this: push_pull_ratio + (1-push_pull_ratio) * push_pull_term. Where the push_pull_term
    is the term determining the pure push pull connectivity. However note that due to the Gaussian
    terms being 0 in infinity and the finite distances in the orientation and phase space, the push-pull
    connectivity effectively itself adds a constant probability to all phase and orientation combinations, and
    the value of this constant is dependent on the phase and orientation sigma parameters.
    Therefore one has to be carefull in interpreting the push_pull_ratio parameter. If the phase and/or
    orientation sigma parameters are rather small, it is however safe to interpret it as the ratio
    of connections that were drawn randomly to the number of connection drawn based on push-pull type of connectivity.

    Other parameters
    ----------------
    or_sigma : float
                  How sharply does the probability of connection fall off with orientation difference.

    phase_sigma : float
                  how sharply does the probability of connection fall off with phase difference.

    target_synapses : str
                    Either 'excitatory' or 'inhibitory': what type is the target excitatory/inhibitory

    push_pull_ratio : float
                    The ratio of push-pull connections, the rest will be random drawn randomly.
    """

    required_parameters = ParameterSet(
        {
            "or_sigma": float,  # how sharply does the probability of connection fall off with orientation difference
            "phase_sigma": float,  # how sharply does the probability of connection fall off with phase difference
            "target_synapses": str,  # what type is the target excitatory/inhibitory
            # the ratio of push-pull connections, the rest will be random drawn randomly
            "push_pull_ratio": float
        }
    )

    def __init__(self, source, target, parameters):
        ModularConnectorFunction.__init__(self, source, target, parameters)
        self.source_or = np.array(
            [
                self.source.get_neuron_annotation(i, "LGNAfferentOrientation")
                for i in range(0, self.source.pop.size)
            ]
        )
        self.source_phase = np.array(
            [
                self.source.get_neuron_annotation(i, "LGNAfferentPhase")
                for i in range(0, self.source.pop.size)
            ]
        )

    def evaluate(self, index):
        target_or = self.target.get_neuron_annotation(index, "LGNAfferentOrientation")
        target_phase = self.target.get_neuron_annotation(index, "LGNAfferentPhase")
        assert np.all(self.source_or >= 0) and np.all(self.source_or <= np.pi)
        assert np.all(target_or >= 0) and np.all(target_or <= np.pi)
        assert np.all(self.source_phase >= 0) and np.all(self.source_phase <= 2 * np.pi)
        assert np.all(target_phase >= 0) and np.all(target_phase <= 2 * np.pi)

        or_dist = circular_dist(self.source_or, target_or, np.pi)
        if self.parameters.target_synapses == "excitatory":
            phase_dist = circular_dist(self.source_phase, target_phase, 2 * np.pi)
        else:
            phase_dist = np.pi - circular_dist(
                self.source_phase, target_phase, 2 * np.pi
            )

        assert np.all(or_dist >= 0) and np.all(or_dist <= np.pi / 2)
        assert np.all(phase_dist >= 0) and np.all(phase_dist <= np.pi)

        or_gauss = normal_function(or_dist, mean=0, sigma=self.parameters.or_sigma)
        phase_gauss = normal_function(
            phase_dist, mean=0, sigma=self.parameters.phase_sigma
        )

        # normalize the product with the product of the two normal distribution at 0.
        m = np.multiply(phase_gauss, or_gauss) / (
            normal_function(np.array([0]), mean=0, sigma=self.parameters.or_sigma)[0]
            * normal_function(np.array([0]), mean=0, sigma=self.parameters.phase_sigma)[
                0
            ]
        )

        return (
            1.0 - self.parameters.push_pull_ratio
        ) + self.parameters.push_pull_ratio * m


def gabor(x1, y1, x2, y2, orientation, frequency, phase, size, aspect_ratio):
    X = (x1 - x2) * np.cos(orientation) + (y1 - y2) * np.sin(orientation)
    Y = -(x1 - x2) * np.sin(orientation) + (y1 - y2) * np.cos(orientation)
    ker = -(X * X + Y * Y * (aspect_ratio ** 2)) / (2 * (size ** 2))
    return np.exp(ker) * np.cos(2 * np.pi * X * frequency + phase)


def gauss(x1, y1, x2, y2, orientation, size, aspect_ratio):
    X = (x1 - x2) * np.cos(orientation) + (y1 - y2) * np.sin(orientation)
    Y = -(x1 - x2) * np.sin(orientation) + (y1 - y2) * np.cos(orientation)
    ker = -(X * X + Y * Y * (aspect_ratio ** 2)) / (2 * (size ** 2))
    return np.exp(ker)


class GaborArborization(ModularConnectorFunction):
    """
    This connector function implements the standard Gabor-like afferent V1 connectivity. It takes the parameters of gabors from
    the annotations that have to be before assigned to neurons.

    Other parameters
    ----------------
    ON : bool
         Whether this is gabor on ON or OFF cells.
    """

    required_parameters = ParameterSet(
        {"ON": bool}  # Whether this is gabor on ON or OFF cells.
    )

    def evaluate(self, index):
        target_or = self.target.get_neuron_annotation(index, "LGNAfferentOrientation")
        target_phase = self.target.get_neuron_annotation(index, "LGNAfferentPhase")
        target_ar = self.target.get_neuron_annotation(index, "LGNAfferentAspectRatio")
        target_freq = self.target.get_neuron_annotation(index, "LGNAfferentFrequency")
        target_size = self.target.get_neuron_annotation(index, "LGNAfferentSize")
        target_posx = self.target.get_neuron_annotation(index, "LGNAfferentX")
        target_posy = self.target.get_neuron_annotation(index, "LGNAfferentY")

        w = gabor(
            self.source.pop.positions[0],
            self.source.pop.positions[1],
            target_posx,
            target_posy,
            target_or + np.pi / 2,
            target_freq,
            target_phase,
            target_size,
            target_ar
        )

        if self.parameters.ON:
            return np.maximum(0, w)  # + 0.03 * g
        else:
            return -np.minimum(0, w)  # + 0.03 * g


class V1CorrelationBasedConnectivity(ModularConnectorFunction):
    """
    This connector function implements  a correlation based rules for neurons with
    gabor like RFs, where excitatory synapses are more-likely between neurons with correlated
    RFs while inhibitory synapses are more likely among anti-correlated synapses.

    Note that this is very similar to how push-pull connectvitity is typically defined, but
    there are important differences mainly that the phase of the cells does not play role when connecting
    orthogonally orientated neurons, and that position of the RFs are also taken into account.

    The connections are drawn from a Gaussian distributaiton centered on 1 (for excitatory synapses) or -1
    (for inhibitory synapses), where the input to the distribution is the correlation between the RFs of the
    neurons and the *sigma* parameter defines the width of the Gaussian.

    Note that this connector uses the same annotation as generated by the GaborArborization connetor function
    to determine the parameters of the neuron's afferent RF (of coruse any other connector that creates the same annotation can be utilized with this connector).

    Other parameters
    ----------------
    sigma : float
                  How sharply does the probability of connection fall off depending on the afferent RF correlation of the two neurons.

    target_synapses : str
                    Either 'excitatory' or 'inhibitory': what type is the target excitatory/inhibitory
    """

    required_parameters = ParameterSet(
        {
            # how sharply does the probability of connection fall off depending on the afferent RF correlation of the two neurons.
            "sigma": float,
            "target_synapses": str,  # what type is the target excitatory/inhibitory
        }
    )

    @staticmethod
    def u_func(F, omega):
        return np.matrix([[F * np.cos(omega), F * np.sin(omega)]]).T

    def __init__(self, source, target, parameters):
        ModularConnectorFunction.__init__(self, source, target, parameters)
        self.source_or = np.array(
            [
                self.source.get_neuron_annotation(i, "LGNAfferentOrientation")
                for i in range(0, self.source.pop.size)
            ]
        )
        self.source_phase = np.array(
            [
                self.source.get_neuron_annotation(i, "LGNAfferentPhase")
                for i in range(0, self.source.pop.size)
            ]
        )
        self.source_ar = np.array(
            [
                self.source.get_neuron_annotation(i, "LGNAfferentAspectRatio")
                for i in range(0, self.source.pop.size)
            ]
        )
        self.source_freq = np.array(
            [
                self.source.get_neuron_annotation(i, "LGNAfferentFrequency")
                for i in range(0, self.source.pop.size)
            ]
        )
        self.source_size = np.array(
            [
                self.source.get_neuron_annotation(i, "LGNAfferentSize")
                for i in range(0, self.source.pop.size)
            ]
        )
        self.source_posx = np.array(
            [
                self.source.get_neuron_annotation(i, "LGNAfferentX")
                for i in range(0, self.source.pop.size)
            ]
        )
        self.source_posy = np.array(
            [
                self.source.get_neuron_annotation(i, "LGNAfferentY")
                for i in range(0, self.source.pop.size)
            ]
        )

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.hist(self.source_size,bins=20)

        # self.flag=True

    @staticmethod
    def omega(widthx, widthy, gauss_or):
        a = widthx ** 2 * np.cos(gauss_or) ** 2 + widthy ** 2 * np.sin(gauss_or) ** 2
        b = np.cos(gauss_or) * np.sin(gauss_or) * (widthx ** 2 - widthy ** 2)
        d = widthx ** 2 * np.sin(gauss_or) ** 2 + widthy ** 2 * np.cos(gauss_or) ** 2
        return (a, b, b, d)

    @staticmethod
    def integral_of_gabor_multiplication_vectorized(
        K1,
        widthx1,
        widthy1,
        posx1,
        posy1,
        gauss_or1,
        freq1,
        sine_orientation1,
        phase1,
        K2,
        widthx2,
        widthy2,
        posx2,
        posy2,
        gauss_or2,
        freq2,
        sine_orientation2,
        phase2
    ):
        def dot2x2times2x1(a, b, c, d, x, y):
            return (
                a * x + b * y,
                c * x + d * y
            )  # calculates out the dot([[a,b],[c,d]],[[x],[y]])

        dotVTtimes2x2timeV = (
            lambda a, b, c, d, x, y: a * x * x + b * y * x + c * x * y + d * y * y
        )  # calculates out the dot([x,y],dot([[a,b],[c,d]],[[x],[y]]))

        Omega1a, Omega1b, Omega1c, Omega1d = V1CorrelationBasedConnectivity.omega(
            widthx1, widthy1, gauss_or1
        )
        Omega2a, Omega2b, Omega2c, Omega2d = V1CorrelationBasedConnectivity.omega(
            widthx2, widthy2, gauss_or2
        )

        OmegaSa = Omega1a + Omega2a
        OmegaSb = Omega1b + Omega2b
        OmegaSc = Omega1c + Omega2c
        OmegaSd = Omega1d + Omega2d

        z = 1 / (OmegaSa * OmegaSd - OmegaSb * OmegaSc)
        OmegaSInva = z * OmegaSd
        OmegaSInvb = -z * OmegaSb
        OmegaSInvc = -z * OmegaSc
        OmegaSInvd = z * OmegaSa

        _x1, _y1 = dot2x2times2x1(Omega1a, Omega1b, Omega1c, Omega1d, posx1, posy1)
        _x2, _y2 = dot2x2times2x1(Omega2a, Omega2b, Omega2c, Omega2d, posx2, posy2)
        _x = _x1 + _x2
        _y = _y1 + _y2

        xs1, xs2 = dot2x2times2x1(
            OmegaSInva, OmegaSInvb, OmegaSInvc, OmegaSInvd, _x, _y
        )

        ux1 = freq1 * np.cos(sine_orientation1)
        uy1 = freq1 * np.sin(sine_orientation1)
        ux2 = freq2 * np.cos(sine_orientation2)
        uy2 = freq2 * np.sin(sine_orientation2)

        K_s = (
            K1
            * K2
            * np.exp(
                -np.pi
                * (
                    dotVTtimes2x2timeV(Omega1a, Omega1b, Omega1c, Omega1d, posx1, posy1)
                    + dotVTtimes2x2timeV(
                        Omega2a, Omega2b, Omega2c, Omega2d, posx2, posy2
                    )
                    - dotVTtimes2x2timeV(OmegaSa, OmegaSb, OmegaSc, OmegaSd, xs1, xs2)
                )
            )
        )

        def integral_complex_gabors(ux, uy, P_s):
            return (
                K_s
                / np.sqrt(OmegaSa * OmegaSd - OmegaSb * OmegaSc)
                * np.exp(
                    -np.pi
                    * (
                        dotVTtimes2x2timeV(
                            OmegaSInva, OmegaSInvb, OmegaSInvc, OmegaSInvd, ux, uy
                        )
                    )
                )
                * np.exp(1j * 2 * np.pi * (ux * xs1 + uy * xs2) + 1j * P_s)
            )

        return (
            1.0
            / 2
            * (
                np.real(integral_complex_gabors(ux1 - ux2, uy1 - uy2, phase1 - phase2))
                + np.real(
                    integral_complex_gabors(ux1 + ux2, uy1 + uy2, phase1 + phase2)
                )
            )
        )

    @staticmethod
    def integral_of_gabor_multiplication(
        K1,
        widthx1,
        widthy1,
        posx1,
        posy1,
        gauss_or1,
        freq1,
        sine_orientation1,
        phase1,
        K2,
        widthx2,
        widthy2,
        posx2,
        posy2,
        gauss_or2,
        freq2,
        sine_orientation2,
        phase2
    ):
        def R_theta_func(theta):
            return np.matrix(
                [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
            )

        def gabor_matrices(a, b, xc, yc, theta, F, omega):
            V = np.matrix([[a, 0], [0, b]])
            x = np.matrix([[xc, yc]]).T
            R_theta = R_theta_func(theta)
            A = np.dot(V, R_theta)
            u = V1CorrelationBasedConnectivity.u_func(F, omega)
            Omega = np.dot(A.T, A)
            return Omega, x, u

        Omega1, x1, u1 = gabor_matrices(
            widthx1, widthy1, posx1, posy1, gauss_or1, freq1, sine_orientation1
        )
        Omega2, x2, u2 = gabor_matrices(
            widthx2, widthy2, posx2, posy2, gauss_or2, freq2, sine_orientation2
        )

        Omega_s = Omega1 + Omega2
        x_s = np.dot(np.linalg.inv(Omega_s), np.dot(Omega1, x1) + np.dot(Omega2, x2))
        P_s = phase1 + phase1
        K_s = (
            K1
            * K2
            * np.exp(
                -np.pi
                * (x1.T * Omega1 * x1 + x2.T * Omega2 * x2 - x_s.T * Omega_s * x_s)
            )
        )

        def integral_complex_gabors(u_s, P_s):
            return (
                K_s
                / np.sqrt(np.linalg.det(Omega_s))
                * np.exp(-np.pi * (u_s.T * np.linalg.inv(Omega_s) * u_s))
                * np.exp(1j * 2 * np.pi * u_s.T * x_s + 1j * P_s)
            )

        return (
            1.0
            / 2
            * (
                np.real(integral_complex_gabors(u1 - u2, phase1 - phase2))
                + np.real(integral_complex_gabors(u1 + u2, phase1 + phase2))
            )
        )

    @staticmethod
    def gabor_correlation(
        K1,
        widthx1,
        widthy1,
        posx1,
        posy1,
        gauss_or1,
        freq1,
        sine_orientation1,
        phase1,
        K2,
        widthx2,
        widthy2,
        posx2,
        posy2,
        gauss_or2,
        freq2,
        sine_orientation2,
        phase2
    ):

        cov = V1CorrelationBasedConnectivity.integral_of_gabor_multiplication_vectorized(
            K1,
            widthx1,
            widthy1,
            posx1,
            posy1,
            gauss_or1,
            freq1,
            sine_orientation1,
            phase1,
            K2,
            widthx2,
            widthy2,
            posx2,
            posy2,
            gauss_or2,
            freq2,
            sine_orientation2,
            phase2
        )

        var1 = V1CorrelationBasedConnectivity.integral_of_gabor_multiplication_vectorized(
            K1,
            widthx1,
            widthy1,
            posx1,
            posy1,
            gauss_or1,
            freq1,
            sine_orientation1,
            phase1,
            K1,
            widthx1,
            widthy1,
            posx1,
            posy1,
            gauss_or1,
            freq1,
            sine_orientation1,
            phase1
        )

        var2 = V1CorrelationBasedConnectivity.integral_of_gabor_multiplication_vectorized(
            K2,
            widthx2,
            widthy2,
            posx2,
            posy2,
            gauss_or2,
            freq2,
            sine_orientation2,
            phase2,
            K2,
            widthx2,
            widthy2,
            posx2,
            posy2,
            gauss_or2,
            freq2,
            sine_orientation2,
            phase2
        )
        return np.array(cov / (np.sqrt(var1) * np.sqrt(var2)))  # [0][0]

    @staticmethod
    def gabor_correlation_rescaled_parammeters(
        width1,
        posx1,
        posy1,
        ar1,
        or1,
        freq1,
        phase1,
        width2,
        posx2,
        posy2,
        ar2,
        or2,
        freq2,
        phase2
    ):

        return V1CorrelationBasedConnectivity.gabor_correlation(
            1.0,
            1 / (np.sqrt(2 * np.pi) * width1),
            ar1 / (np.sqrt(2 * np.pi) * width1),
            posx1,
            posy1,
            or1,
            freq1,
            or1,
            phase1 - np.pi * 2 * freq1 * (posx1 * np.cos(or1) + posy1 * np.sin(or1)),
            1.0,
            1 / (np.sqrt(2 * np.pi) * width2),
            ar2 / (np.sqrt(2 * np.pi) * width2),
            posx2,
            posy2,
            or2,
            freq2,
            or2,
            phase2 - np.pi * 2 * freq2 * (posx2 * np.cos(or2) + posy2 * np.sin(or2))
        )

    def evaluate(self, index):

        target_or = self.target.get_neuron_annotation(index, "LGNAfferentOrientation")
        target_phase = self.target.get_neuron_annotation(index, "LGNAfferentPhase")
        target_ar = self.target.get_neuron_annotation(index, "LGNAfferentAspectRatio")
        target_freq = self.target.get_neuron_annotation(index, "LGNAfferentFrequency")
        target_size = self.target.get_neuron_annotation(index, "LGNAfferentSize")
        target_posx = self.target.get_neuron_annotation(index, "LGNAfferentX")
        target_posy = self.target.get_neuron_annotation(index, "LGNAfferentY")

        assert np.all(self.source_or >= 0) and np.all(self.source_or <= np.pi)
        assert np.all(target_or >= 0) and np.all(target_or <= np.pi)
        assert np.all(self.source_phase >= 0) and np.all(self.source_phase <= 2 * np.pi)
        assert np.all(target_phase >= 0) and np.all(target_phase <= 2 * np.pi)

        corr = V1CorrelationBasedConnectivity.gabor_correlation_rescaled_parammeters(
            self.source_size,
            self.source_posx,
            self.source_posy,
            self.source_ar,
            self.source_or,
            self.source_freq,
            self.source_phase,
            target_size,
            target_posx,
            target_posy,
            target_ar,
            target_or,
            target_freq,
            target_phase
        )

        assert np.all(corr >= -1.0) and np.all(
            corr <= 1.000001
        ), "The correlation is %f %f" % (np.min(corr), np.max(corr))
        # import matplotlib.pyplot as plt
        # plt.figure()
        # if self.flag:
        #    print(corr)
        #    plt.hist(corr,bins=30)
        # self.flag = False
        # plt.show()

        if self.parameters.target_synapses == "excitatory":
            corr_gauss = normal_function(corr, mean=1, sigma=self.parameters.sigma)
        else:
            corr_gauss = normal_function(corr, mean=-1, sigma=self.parameters.sigma)

        return corr_gauss


class CoCircularModularConnectorFunction(ModularConnectorFunction):
    """
    This connector function implements the situation where the probability of connection between pre-synaptic neuron preN and post synaptic neuron postN
    depends on their orientation and distance in following the co-circularity rule. The law was adopted from:
    Hunt, J. J., Bosking, W. H., & Goodhill, G. J. (2011). Statistical structure of lateral connections in the primary visual cortex. Neural Systems & Circuits, 1(1), 3. https://doi.org/10.1186/2042-1001-1-3
    """

    required_parameters = ParameterSet(
        {
            # It has to point to a file containing a single pickled 2d numpy array, containing values in the interval [0..1].
            "or_map_location": str,
            "sigma": float,  # How sharply does the probability of connection fall off with the distance from idealized co-circular connectivity
        }
    )

    def __init__(self, source, target, parameters):
        import pickle

        ModularConnectorFunction.__init__(self, source, target, parameters)
        t_size = target.size_in_degrees()
        f = open(self.parameters.or_map_location, "r")
        mmap = pickle.load(f)
        coords_x = np.linspace(-t_size[0] / 2.0, t_size[0] / 2.0, np.shape(mmap)[0])
        coords_y = np.linspace(-t_size[1] / 2.0, t_size[1] / 2.0, np.shape(mmap)[1])
        X, Y = np.meshgrid(coords_x, coords_y)
        self.mmap = NearestNDInterpolator(
            list(zip(X.flatten(), Y.flatten())), mmap.flatten()
        )
        self.or_source = (
            self.mmap(
                np.transpose(
                    np.array(
                        [self.source.pop.positions[0], self.source.pop.positions[1]]
                    )
                )
            )
            * np.pi
        )

        for (index, neuron2) in enumerate(target.pop.all()):
            val_target = self.mmap(
                self.target.pop.positions[0][index], self.target.pop.positions[1][index]
            )
            self.target.add_neuron_annotation(
                index, "ORMapOrientation", val_target * np.pi, protected=False
            )

    def evaluate(self, index):
        logger.error("EVALUATE *********************************************")
        or_target = self.target.get_neuron_annotation(index, "ORMapOrientation")
        x_target = self.target.pop.positions[0][index]
        y_target = self.target.pop.positions[1][index]

        phi = np.arctan2(
            self.source.pop.positions[1] - y_target,
            self.source.pop.positions[0] - x_target
        )
        distance = circular_dist(self.or_source, 2 * phi - or_target, np.pi)
        prob = np.exp(-0.5 * (distance / self.parameters.sigma) ** 2) / (
            self.parameters.sigma * np.sqrt(2 * np.pi)
        )

        if index in [1000]:
            plt.figure()
            plt.subplot(131)
            plt.gca().set_aspect("equal", "box")
            s_size = self.source.size_in_degrees()
            plt.xlim(-s_size[0] / 2.0 * 1.5, s_size[0] / 2.0 * 1.5)
            plt.ylim(-s_size[0] / 2.0 * 1.5, s_size[0] / 2.0 * 1.5)

            def plot_or(x, y, phi, color):
                d = 0.06
                dx = np.cos(phi) * d
                dy = np.sin(phi) * d
                plt.plot([x + dx, x - dx], [y + dy, y - dy], c=color)

            idx = np.random.choice(np.arange(len(phi)), size=500)
            for i in idx:
                plot_or(
                    self.source.pop.positions[0][i],
                    self.source.pop.positions[1][i],
                    2 * phi[i] - or_target,
                    "k"
                )

            plot_or(x_target, y_target, or_target, "r")

            plt.subplot(132)
            plt.gca().set_aspect("equal", "box")
            logger.error(str(np.shape(self.source.pop.positions[1][idx])))
            logger.error(str(np.max(distance)))
            logger.error(str(np.max(prob)))
            plt.scatter(
                self.source.pop.positions[0],
                self.source.pop.positions[1],
                c=self.or_source,
                alpha=0.5,
                edgecolors="none",
                s=np.array(prob / np.max(prob)) * 20,
                cmap="hsv"
            )
            plt.colorbar()

            plt.subplot(133)
            plt.gca().set_aspect("equal", "box")
            plt.scatter(
                self.source.pop.positions[0],
                self.source.pop.positions[1],
                c=self.or_source,
                alpha=0.5,
                edgecolors="none",
                s=(1 - np.array(distance / np.max(distance))) * 20,
                cmap="hsv"
            )

            # plt.subplot(412)
            # plt.scatter(self.source.pop.positions[0],self.source.pop.positions[1],c=prob)

            plt.savefig(Global.root_directory + "aaa" + str(index) + ".png")

        return prob
