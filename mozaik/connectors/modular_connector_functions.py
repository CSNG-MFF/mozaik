# encoding: utf-8
import numpy
from mozaik.framework.interfaces import MozaikParametrizeObject
from NeuroTools.parameters import ParameterSet, ParameterDist
from mozaik.tools.circ_stat import *
from mozaik.tools.misc import *

class ModularConnectorFunction(MozaikParametrizeObject):
    def __init__(self, source,target, parameters):
        MozaikParametrizeObject.__init__(self, parameters)
        self.source = source
        self.target = target
        
    def evaluate(self):
        raise NotImplemented

class DistanceDependentModularConnectorFunction(MozaikParametrizeObject):
    """
    Helper abstract class to ease the definitions of purely distance dependent connector functions.
    
    The distance is defined as the *horizontal* distance between the retinotopical positions of the neurons (one in source and one in destination sheet). 
    The distane is translated into the native coordinates of the target sheet (e.g. micrometers for CorticlaSheet)!
    
    For the special case where source = target, this coresponds to the intuitive lateral distance of the neurons.
    """
    def distance_dependent_function(distance):
        """
        The is the function, dependent only on distance that each DistanceDependentModularConnectorFunction has to implement.
        """
        raise NotImplemented
    
    def evaluate(self):
        weights = numpy.zeros((self.source.pop.size,self.target.pop.size))    
        
        for i in xrange(0,self.target.pop.size):
            for j in xrange(0,self.source.pop.size):
                weights[j,i] = self.distance_dependent_function(self.target.dvf_2_dcs(numpy.linalg.norm(self.target.pop.positions[:,i]-self.source.pop.positions[:,j])))
                
        return weights

class GaussianDecayModularConnectorFunction(MozaikParametrizeObject):
    """
    Distance dependent arborization with gaussian fall-off of the connections.
    """
    required_parameters = ParameterSet({
        'arborization_constant': float,  # μm distance constant of the gaussian decay of the connections with respect (in cortical distance)
                                         # to the distance from the innervation point.
        'arborization_scaler': float,    # the scaler of the gaussian decay
    })
    
    def distance_dependent_function(d):
        return self.parameters.arborization_scaler*numpy.exp(-0.5*(distance/self.parameters.arborization_constant)**2)/(self.parameters.arborization_constant*numpy.sqrt(2*numpy.pi))
        

class ExponentialDecayModularConnectorFunction(MozaikParametrizeObject):
    """
    Distance dependent arborization with exponential fall-off of the connections.
    """
    required_parameters = ParameterSet({
        'arborization_constant': float,  # μm distance constant of the exponential decay of the connections with respect (in cortical distance)
                                         # to the distance from the innervation point.
        'arborization_scaler': float,    # the scaler of the exponential decay
    })
    
    def distance_dependent_function(d):
        return self.parameters.arborization_scaler*numpy.exp(-distance*self.parameters.arborization_constant)

class LinearModularConnectorFunction(MozaikParametrizeObject):
    """
    Corresponds to: distance*linear_scaler + constant_scaler
    """
    required_parameters = ParameterSet({
        'constant_scaler': float,    # the scaler of the exponential decay
        'linear_scaler': float,    # the scaler of the exponential decay
    })
    
    def distance_dependent_function(d):
        return self.parameters.linear_scaler*distance + self.parameters.constant_scaler

class V1PushPullArborization(ModularConnectorFunction):
    """
    This connector function implements the standard V1 functionally specific
    connection rule:

    Excitatory synapses are more likely on cooriented in-phase neurons
    Inhibitory synapses are more likely to cooriented anti-phase neurons
    """

    required_parameters = ParameterSet({
        'or_sigma': float,  # how sharply does the probability of connection fall off with orientation difference
        'phase_sigma': float,  # how sharply does the probability of connection fall off with phase difference
        'target_synapses' : str, # what type is the target excitatory/inhibitory
    })

    def evaluate(self):
        weights = numpy.zeros((self.source.pop.size,self.target.pop.size))    

        for i in xrange(0,self.target.pop.size):
            for j in xrange(0,self.source.pop.size):
                or_dist = circular_dist(self.target.get_neuron_annotation(i, 'LGNAfferentOrientation'),
                                        self.source.get_neuron_annotation(j, 'LGNAfferentOrientation'),
                                        pi) / (pi/2)

                if self.parameters.target_synapses == 'excitatory':
                        phase_dist = circular_dist(self.target.get_neuron_annotation(i, 'LGNAfferentPhase'),
                                                   self.source.get_neuron_annotation(j, 'LGNAfferentPhase'),
                                                   2*pi) / pi
                elif self.parameters.target_synapses == 'inhibitory':
                        phase_dist = (pi - circular_dist(self.target.get_neuron_annotation(i, 'LGNAfferentPhase'),
                                                         self.source.get_neuron_annotation(j, 'LGNAfferentPhase'),
                                                         2*pi)) / pi
                else:
                    logger.error('Unknown type of synapse!')
                    return

                or_gauss = normal_function(or_dist, mean=0, sigma=self.parameters.or_sigma)
                phase_gauss = normal_function(phase_dist, mean=0, sigma=self.parameters.phase_sigma)
                w = phase_gauss * or_gauss
                weights[j,i]=w

        return weights
