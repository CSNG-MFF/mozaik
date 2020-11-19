# encoding: utf-8
"""
This file contains connectors that were written for speed - as a general rule
they tend to use the more native pyNN or even backend specific pyNN methods.

To obtain speed they generally sacrifice ease customization.
"""
import logging

from parameters import ParameterSet
from pyNN import space
import numpy

import mozaik
from . import Connector, SheetWithMagnificationFactor

logger = logging.getLogger(__name__)


class DistanceDependentProbabilisticArborization(Connector):
    """
    A abstract connector that implements distance dependent connection.
    Each implementation just needs to implement the arborization_function and delay function.
    The distance input is in the 'native' metric of the sheets, i.e. degrees of visual field
    in RetinalSheet or micrometers in CorticalSheet.
    """

    required_parameters = ParameterSet(
        {
            "weights": float,  # nA, the synapse strength
            # location of the map. It has to be a file containing a single pickled 2d numpy array with values between 0 and 1.0.
            "map_location": str
        }
    )

    def arborization_function(self, distance):
        raise NotImplementedError
        pass

    def delay_function(self, distance):
        raise NotImplementedError
        pass

    def _connect(self):
        # JAHACK, 0.1 as minimal delay should be replaced with the simulations time_step
        if isinstance(self.target, SheetWithMagnificationFactor):
            self.arborization_expression = lambda d: self.arborization_function(
                self.target.dvf_2_dcs(d)
            )
            self.delay_expression = lambda d: self.delay_function(
                self.target.dvf_2_dcs(d)
            )
        else:
            self.arborization_expression = lambda d: self.arborization_function(d)
            self.delay_expression = lambda d: self.delay_function(d)

        method = self.sim.DistanceDependentProbabilityConnector(
            self.arborization_expression,
            allow_self_connections=False,
            weights=self.parameters.weights * self.weight_scaler,
            delays=self.delay_expression,
            space=space.Space(axes="xy"),
            safe=True,
            verbose=False,
            n_connections=None,
            rng=mozaik.pynn_rng
        )

        self.proj = self.sim.Projection(
            self.source.pop,
            self.target.pop,
            method,
            synapse_type=self.init_synaptic_mechanisms(),
            label=self.name,
            receptor_type=self.parameters.target_synapses
        )


class ExponentialProbabilisticArborization(DistanceDependentProbabilisticArborization):
    """
    Distance dependent arborization with exponential fall-off of the probability, and linear spike propagation.
    """

    required_parameters = ParameterSet(
        {
            # ms/μm the constant that will determinine the distance dependent delays on the connections
            "propagation_constant": float,
            # μm distance constant of the exponential decay of the probability of the connection with respect (in cortical distance)
            "arborization_constant": float,
            # to the distance from the innervation point.
            "arborization_scaler": float,  # the scaler of the exponential decay
        }
    )

    def arborization_function(self, distance):
        return (
            self.parameters.arborization_scaler
            * numpy.exp(-0.5 * (distance / self.parameters.arborization_constant) ** 2)
            / (self.parameters.arborization_constant * numpy.sqrt(2 * numpy.pi))
        )

    def delay_function(self, distance):
        return distance * self.parameters.propagation_constant


class UniformProbabilisticArborization(Connector):
    """
    Connects source with target with equal probability between any two neurons.
    """

    required_parameters = ParameterSet(
        {
            # probability of connection between two neurons from the two populations
            "connection_probability": float,
            "weights": float,  # nA, the synapse strength
            "delay": float,  # ms delay of the connections
        }
    )

    def _connect(self):
        method = self.sim.FixedProbabilityConnector(
            self.parameters.connection_probability,
            allow_self_connections=False,
            safe=True,
            rng=mozaik.pynn_rng
        )

        self.proj = self.sim.Projection(
            self.source.pop,
            self.target.pop,
            method,
            synapse_type=self.init_synaptic_mechanisms(
                weight=self.parameters.weights * self.weight_scaler,
                delay=self.parameters.delay
            ),
            label=self.name,
            space=space.Space(axes="xy"),
            receptor_type=self.parameters.target_synapses
        )


class FixedKConnector(Connector):
    """
    Connects source with target such that each neuron will have the same number of presynaptic neurons chosen randomly.
    """

    required_parameters = ParameterSet(
        {
            "k": int,  # probability of connection between two neurons from the two populations
            "weights": float,  # nA, the synapse strength
            "delay": float,  # ms delay of the connections
        }
    )

    def _connect(self):
        method = self.sim.FixedNumberPreConnector(
            self.parameters.k,
            allow_self_connections=False,
            safe=True,
            rng=mozaik.pynn_rng
        )

        self.proj = self.sim.Projection(
            self.source.pop,
            self.target.pop,
            method,
            synapse_type=self.init_synaptic_mechanisms(
                weight=self.parameters.weights * self.weight_scaler,
                delay=self.parameters.delay
            ),
            label=self.name,
            space=space.Space(axes="xy"),
            receptor_type=self.parameters.target_synapses
        )
