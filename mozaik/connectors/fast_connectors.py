# encoding: utf-8
import mozaik
from mozaik.connectors import MozaikConnector
from parameters import ParameterSet, ParameterDist
from pyNN import space
import numpy

logger = mozaik.getMozaikLogger()

"""
This file contains connectors that were written for speed - as a general rule 
they tend to use the more native pyNN or even backend specific pyNN methods.

To obtain speed they generally sacrifice ease customization.
"""

class DistanceDependentProbabilisticArborization(MozaikConnector):
    """
    A abstract connector that implements distance dependent connection.
    Each implementation just needs to implement the arborization_function and delay function.
    The distance input is in the 'native' metric of the sheets, i.e. degrees of visual field 
    in RetinalSheet or micrometers in CorticalSheet.
    """
    required_parameters = ParameterSet({
        'weights': float,   # nA, the synapse strength 
        'map_location' : str, # location of the map. It has to be a file containing a single pickled 2d numpy array with values between 0 and 1.0. 
    })
    
    def arborization_function(distance):
        raise NotImplementedError
        pass
    
    def delay_function(distance):
        raise NotImplementedError
        pass
        
    def _connect(self):
        # JAHACK, 0.1 as minimal delay should be replaced with the simulations time_step        
        if isinstance(self.target, SheetWithMagnificationFactor):
            self.arborization_expression = lambda d: self.arborization_function(self.target.dvf_2_dcs(d))
            self.delay_expression = lambda d: self.delay_function(self.target.dvf_2_dcs(d)) 
        else:
            self.arborization_expression = lambda d: self.arborization_function(d)
            self.delay_expression = lambda d: self.delay_function(d)
        
        method = self.sim.DistanceDependentProbabilityConnector(self.arborization_expression,
                                                                allow_self_connections=False, 
                                                                weights=self.parameters.weights, 
                                                                delays=self.delay_expression, 
                                                                space=space.Space(axes='xy'), 
                                                                safe=True, 
                                                                verbose=False, 
                                                                n_connections=None,rng=mozaik.pynn_rng)
                                                                
        self.proj = self.sim.Projection(self.source.pop, 
                                        self.target.pop, 
                                        method, 
                                        synapse_type=self.init_synaptic_mechanisms(), 
                                        label=self.name, 
                                        receptor_type=self.parameters.target_synapses)
    


class ExponentialProbabilisticArborization(DistanceDependentProbabilisticArborization):
    """
    Distance dependent arborization with exponential fall-off of the probability, and linear spike propagation.
    """
    required_parameters = ParameterSet({
        'propagation_constant': float,   # ms/μm the constant that will determinine the distance dependent delays on the connections
        'arborization_constant': float,  # μm distance constant of the exponential decay of the probability of the connection with respect (in cortical distance)
                                         # to the distance from the innervation point.
        'arborization_scaler': float,    # the scaler of the exponential decay
    })

    def arborization_function(distance):
        return self.parameters.arborization_scaler*numpy.exp(-0.5*(distance/self.parameters.arborization_constant)**2)/(self.parameters.arborization_constant*numpy.sqrt(2*numpy.pi))
    
    def delay_function(distance):
        # JAHACK, 0.1 as minimal delay should be replaced with the simulations time_step        
        return numpy.maximum(distance * self.parameters.propagation_constant,0.1)
        
        
class UniformProbabilisticArborization(MozaikConnector):
    """
    Connects source with target with equal probability between any two neurons.
    """
    
    required_parameters = ParameterSet({
        'connection_probability': float,  # probability of connection between two neurons from the two populations
        'weights': float,  # nA, the synapse strength
        'delay': float,    # ms delay of the connections
    })

    def _connect(self):
        method = self.sim.FixedProbabilityConnector(
                                    self.parameters.connection_probability,
                                    allow_self_connections=False,
                                    safe=True,rng=mozaik.pynn_rng)

                                    
                                    
        self.proj = self.sim.Projection(
                                    self.source.pop,
                                    self.target.pop,
                                    method,
                                    synapse_type=self.init_synaptic_mechanisms(weights=self.parameters.weights,delays=self.parameters.delay),
                                    label=self.name,
                                    space=space.Space(axes='xy'),
                                    receptor_type=self.parameters.target_synapses)
