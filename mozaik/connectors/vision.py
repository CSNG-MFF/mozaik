"""
Vision specific connectors.
"""
import numpy
from modular_connector_functions import ModularConnectorFunction
from mozaik.tools.circ_stat import *
from mozaik.tools.misc import *
from parameters import ParameterSet
from scipy.interpolate import NearestNDInterpolator

class MapDependentModularConnectorFunction(ModularConnectorFunction):
    """
    Corresponds to: distance*linear_scaler + constant_scaler
    """
    required_parameters = ParameterSet({
        'map_location': str,  # It has to point to a file containing a single pickled 2d numpy array, containing values in the interval [0..1].
        'sigma': float,  # How sharply does the wieght fall off with the increasing distance between the map values (exp(-0.5*(distance/sigma)*2)/(sigma*sqrt(2*pi)))
        'periodic' : bool, # if true, the values in or_map will be treated as periodic (and consequently the distance between two values will be computed as circular distance).
    })
    
    def __init__(self, source,target, parameters):
        import pickle
        ModularConnectorFunction.__init__(self, source,target, parameters)
        t_size = target.size_in_degrees()
        f = open(self.parameters.map_location, 'r')
        mmap = pickle.load(f)
        coords_x = numpy.linspace(-t_size[0]/2.0,
                                  t_size[0]/2.0,
                                  numpy.shape(mmap)[0])
        coords_y = numpy.linspace(-t_size[1]/2.0,
                                  t_size[1]/2.0,
                                  numpy.shape(mmap)[1])
        X, Y = numpy.meshgrid(coords_x, coords_y)
        self.mmap = NearestNDInterpolator(zip(X.flatten(), Y.flatten()),
                                       mmap.flatten())    
        self.val_source=self.mmap(numpy.transpose(numpy.array([self.source.pop.positions[0],self.source.pop.positions[1]])))
        
    def evaluate(self,index):
            val_target=self.mmap(self.target.pop.positions[0][index],self.target.pop.positions[1][index])
            self.target.add_neuron_annotation(index, 'LGNAfferentOrientation', val_target*numpy.pi, protected=False) 
            if self.parameters.periodic:
                distance = circular_dist(self.val_source,val_target,1.0)
            else:
                distance = numpy.abs(self.val_source-val_target)
            return numpy.exp(-0.5*(distance/self.parameters.sigma)**2)/(self.parameters.sigma*numpy.sqrt(2*numpy.pi))
    

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
    """

    required_parameters = ParameterSet({
        'or_sigma': float,  # how sharply does the probability of connection fall off with orientation difference
        'phase_sigma': float,  # how sharply does the probability of connection fall off with phase difference
        'target_synapses' : str, # what type is the target excitatory/inhibitory
        'push_pull_ratio' : float, # the ratio of push-pull connections, the rest will be random drawn randomly
    })

    def __init__(self, source,target, parameters):
        ModularConnectorFunction.__init__(self, source,target,  parameters)
        self.source_or = numpy.array([self.source.get_neuron_annotation(i, 'LGNAfferentOrientation') for i in xrange(0,self.source.pop.size)])
        self.source_phase = numpy.array([self.source.get_neuron_annotation(i, 'LGNAfferentPhase') for i in xrange(0,self.source.pop.size)])

    def evaluate(self,index):
        target_or = self.target.get_neuron_annotation(index, 'LGNAfferentOrientation')
        target_phase = self.target.get_neuron_annotation(index, 'LGNAfferentPhase')
        assert numpy.all(self.source_or >= 0) and numpy.all(self.source_or <= pi)
        assert numpy.all(target_or >= 0) and numpy.all(target_or <= pi)
        assert numpy.all(self.source_phase >= 0) and numpy.all(self.source_phase <= 2*pi)
        assert numpy.all(target_phase >= 0) and numpy.all(target_phase <= 2*pi)
        
        or_dist = circular_dist(self.source_or,target_or,pi) 
        if self.parameters.target_synapses == 'excitatory':
            phase_dist = circular_dist(self.source_phase,target_phase,2*pi) 
        else:
            phase_dist = (pi - circular_dist(self.source_phase,target_phase,2*pi)) 
            
        assert numpy.all(or_dist >= 0) and numpy.all(or_dist <= pi/2)
        assert numpy.all(phase_dist >= 0) and numpy.all(phase_dist <= pi)
        
        or_gauss = normal_function(or_dist, mean=0, sigma=self.parameters.or_sigma)
        phase_gauss = normal_function(phase_dist, mean=0, sigma=self.parameters.phase_sigma)
        
        return (1.0-self.parameters.push_pull_ratio) +  self.parameters.push_pull_ratio*numpy.multiply(phase_gauss, or_gauss)




