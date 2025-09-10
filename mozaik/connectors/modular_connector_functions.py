# encoding: utf-8
import numpy
from mozaik.core import ParametrizedObject
from parameters import ParameterSet
from mozaik.tools.distribution_parametrization import PyNNDistribution
from mozaik.tools.misc import *
from parameters import ParameterSet
import mozaik

logger = mozaik.getMozaikLogger()


class ModularConnectorFunction(ParametrizedObject):
    r"""
    Abstract class defining the interface of modular connector functions.
    
    Each instance has to implement the evaluate(u) function that returns the pre-synaptic weights
    of neuron i.
    """
    
    def __init__(self, source,target, parameters):
        ParametrizedObject.__init__(self, parameters)
        self.source = source
        self.target = target
        
    def evaluate(self,index,**params):
        raise NotImplemented

class ConstantModularConnectorFunction(ModularConnectorFunction):
      r"""
      Triavial modular connection function assigning each connections the same weight
      """
      def evaluate(self,index,**params):
          return numpy.zeros(len(self.source.pop)) + 1

class PyNNDistributionConnectorFunction(ModularConnectorFunction):
      r"""
      ConnectorFunction which draws the values from the PyNNDistribution
      
      """
      required_parameters = ParameterSet({
        'pynn_distribution': PyNNDistribution,  # The distribution
      })

      def evaluate(self,index,seed=None):
          if seed:
              return self.parameters.pynn_distribution.copy(seed).next(len(self.source.pop))
          else:
              return self.parameters.pynn_distribution.next(len(self.source.pop))

          
class DistanceDependentModularConnectorFunction(ModularConnectorFunction):
    r"""
    Helper abstract class to ease the definitions of purely distance dependent connector functions.
    
    The distance is defined as the *horizontal* distance between the retinotopical positions of the neurons (one in source and one in destination sheet). 
    The distane is translated into the native coordinates of the target sheet (e.g. micrometers for CorticlaSheet)!
    
    For the special case where source = target, this coresponds to the intuitive lateral distance of the neurons.
    """
    def distance_dependent_function(self,distance):
        r"""
        This is the function, dependent only on distance that each DistanceDependentModularConnectorFunction has to implement.
        The distance can be matrix.
        """
        raise NotImplemented
    
    def evaluate(self,index,**params):
        return self.distance_dependent_function(self.source.dvf_2_dcs(numpy.sqrt(
                                numpy.power(self.source.pop.positions[0,:]-self.target.pop.positions[0,index],2) + numpy.power(self.source.pop.positions[1,:]-self.target.pop.positions[1,index],2)
                    )))
        
        

class GaussianDecayModularConnectorFunction(DistanceDependentModularConnectorFunction):
    r"""
    Distance dependent arborization with gaussian fall-off of the connections: k * exp(-0.5*(distance/a)*2) / (a*sqrt(2*pi))
    where a = arborization_constant, k = arborization_scaler
    """
    required_parameters = ParameterSet({
        'arborization_constant': float,  # μm distance constant of the gaussian decay of the connections with respect (in cortical distance)
                                         # to the distance from the innervation point.
        'arborization_scaler': float,    # the scaler of the gaussian decay
    })
    

    def distance_dependent_function(self,distance):
        return self.parameters.arborization_scaler*numpy.exp(-0.5*(distance/self.parameters.arborization_constant)**2)/(self.parameters.arborization_constant*numpy.sqrt(2*numpy.pi))


class ExponentialDecayModularConnectorFunction(DistanceDependentModularConnectorFunction):
    r"""
    Distance dependent arborization with exponential fall-off of the connections: k * exp(-distance/a)
    """
    required_parameters = ParameterSet({
        'arborization_constant': float,  # μm distance constant of the exponential decay of the connections with respect (in cortical distance)
                                         # to the distance from the innervation point.
        'arborization_scaler': float,    # the scaler of the exponential decay
    })
    

    def distance_dependent_function(self,distance):
        return self.parameters.arborization_scaler*numpy.exp(-distance/self.parameters.arborization_constant)


class LinearModularConnectorFunction(DistanceDependentModularConnectorFunction):
    r"""
    Corresponds to: distance*linear_scaler + constant_scaler, where distance is in micrometers
    """
    required_parameters = ParameterSet({
        'constant_scaler': float,    # the aditive constant of the decay
        'linear_scaler': float,    # the scaler of the linear decay
    })
    
    def distance_dependent_function(self,distance):
        return self.parameters.linear_scaler*distance + self.parameters.constant_scaler


class LinearModularConnectorFunction1(DistanceDependentModularConnectorFunction):
    r"""
    Corresponds to: distance*linear_scaler + constant_scaler, where distance is in micrometers
    """
    required_parameters = ParameterSet({
        'constant_scaler': PyNNDistribution,    # the aditive constant of the decay
        'linear_scaler': PyNNDistribution,    # the scaler of the linear decay
    })
    
    def distance_dependent_function(self,distance):
        return self.parameters.linear_scaler.next()*distance + self.parameters.constant_scaler.next()


class HyperbolicModularConnectorFunction(DistanceDependentModularConnectorFunction):
    r"""
    Corresponds to: exp(-alpha*sqrt(\theta^2 + distance^2)) , where distance is in micrometers
    And is the best fit I could so far find to the data from: 
    Stepanyants, A., Hirsch, J. a, Martinez, L. M., Kisvárday, Z. F., Ferecskó, A. S., & Chklovskii, D. B. (2008). 
    Local potential connectivity in cat primary visual cortex. Cerebral cortex, 18(1), 13–28. doi:10.1093/cercor/bhm027
    """
    required_parameters = ParameterSet({
        'alpha' : float, # see description
        'theta': float,  # see description
    })
    
    def distance_dependent_function(self,distance):
        return numpy.exp(-numpy.multiply(self.parameters.alpha,numpy.sqrt(numpy.power(self.parameters.theta,2) + numpy.power(distance,2))))

class ModularNumSamplesConnectorFunction(ParametrizedObject):
    r"""
    Abstract class defining the interface of modular connector functions for the
    number incoming connections.
    
    Each instance has to implement the evaluate(u) function that returns the number of 
    incoming connections of neuron i.
    """

    def __init__(self, target, parameters):
        ParametrizedObject.__init__(self, parameters)
        self.target = target

    def evaluate(self,index,**params):
        raise NotImplemented

class ThresholdLinearModularNumSamplesConnectorFunction(ModularNumSamplesConnectorFunction):
    r"""
    Number of incoming connection decreases quadratically when neurons are
    sufficiently close to the border (distance less than threshold).
    """
    required_parameters = ParameterSet({
        'threshold' : float, # see description
        'max_decrease': float,  # For each spatial dimension, the number
                                # of incoming connection cannot decrease
                                # more than by that factor
    })

    def evaluate(self,index):
        posx = self.target.pop.positions[0,index] + self.target.size_x/2
        posy = self.target.pop.positions[1,index] + self.target.size_y/2
        coef = 1
        if posx < self.parameters.threshold:
            coef *= 1/self.parameters.max_decrease + (1 - 1/self.parameters.max_decrease) * (posx/self.parameters.threshold)
        elif self.target.size_x - posx < self.parameters.threshold:
            coef *=  1/self.parameters.max_decrease + (1 - 1/self.parameters.max_decrease) * ((self.target.size_x - posx)/self.parameters.threshold)
        

        if posy < self.parameters.threshold:
            coef *= 1/self.parameters.max_decrease + (1 - 1/self.parameters.max_decrease) * (posy/self.parameters.threshold)
        elif self.target.size_y - posy < self.parameters.threshold:
            coef *=  1/self.parameters.max_decrease + (1 - 1/self.parameters.max_decrease) * ((self.target.size_y - posy)/self.parameters.threshold)

        return coef


class ThresholdQuadraticModularNumSamplesConnectorFunction(ModularNumSamplesConnectorFunction):
    r"""
    Number of incoming connection decreases quadratically when neurons are
    sufficiently close to the border (distance less than threshold).
    """
    required_parameters = ParameterSet({
        'threshold' : float, # see description
        'max_decrease': float,  # For each spatial dimension, the number
                                # of incoming connection cannot decrease
                                # more than by that factor
    })

    def evaluate(self,index):
        posx = self.target.pop.positions[0,index] + self.target.size_x/2
        posy = self.target.pop.positions[1,index] + self.target.size_y/2
        coef = 1
        if posx < self.parameters.threshold:
            coef *= 1/self.parameters.max_decrease + (1 - 1/self.parameters.max_decrease) * numpy.sqrt(posx/self.parameters.threshold)
        elif self.target.size_x - posx < self.parameters.threshold:
            coef *=  1/self.parameters.max_decrease + (1 - 1/self.parameters.max_decrease) * numpy.sqrt((self.target.size_x - posx)/self.parameters.threshold)


        if posy < self.parameters.threshold:
            coef *= 1/self.parameters.max_decrease + (1 - 1/self.parameters.max_decrease) * numpy.sqrt(posy/self.parameters.threshold)
        elif self.target.size_y - posy < self.parameters.threshold:
            coef *=  1/self.parameters.max_decrease + (1 - 1/self.parameters.max_decrease) * numpy.sqrt((self.target.size_y - posy)/self.parameters.threshold)

        return coef

class ThresholdExponentialModularNumSamplesConnectorFunction(ModularNumSamplesConnectorFunction):
    r"""
    Number of incoming connection decreases quadratically when neurons are
    sufficiently close to the border (distance less than threshold).
    """
    required_parameters = ParameterSet({
        'threshold' : float, # see description
        'max_decrease': float,  # For each spatial dimension, the number
                                # of incoming connection cannot decrease
                                # more than by that factor
        'exponent_factor': float, #the factor of the exponential
    })

    def evaluate(self,index):
        posx = self.target.pop.positions[0,index] + self.target.size_x/2
        posy = self.target.pop.positions[1,index] + self.target.size_y/2
        coef = 1
        if posx < self.parameters.threshold:
            coef *= 1/self.parameters.max_decrease + (1 - 1/self.parameters.max_decrease) * (-numpy.exp(-self.parameters.exponent_factor * posx/self.parameters.threshold)+1)/(-numpy.exp(-self.parameters.exponent_factor)+1)
        elif self.target.size_x - posx < self.parameters.threshold:
            coef *=  1/self.parameters.max_decrease + (1 - 1/self.parameters.max_decrease) *  (-numpy.exp(-self.parameters.exponent_factor * (self.target.size_x - posx)/self.parameters.threshold)+1)/(-numpy.exp(-self.parameters.exponent_factor)+1)


        if posy < self.parameters.threshold:
            coef *= 1/self.parameters.max_decrease + (1 - 1/self.parameters.max_decrease) * (-numpy.exp(-self.parameters.exponent_factor * posy/self.parameters.threshold)+1)/(-numpy.exp(-self.parameters.exponent_factor)+1)
        elif self.target.size_y - posy < self.parameters.threshold:
            coef *=  1/self.parameters.max_decrease + (1 - 1/self.parameters.max_decrease) *  (-numpy.exp(-self.parameters.exponent_factor * (self.target.size_y - posy)/self.parameters.threshold)+1)/(-numpy.exp(-self.parameters.exponent_factor)+1)

        return coef

