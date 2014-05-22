# encoding: utf-8
import pickle
import numpy
from scipy.interpolate import NearestNDInterpolator
from mozaik.core import BaseComponent
from mozaik import load_component
from parameters import ParameterSet, ParameterDist
from mozaik.connectors.modular import ModularSamplingProbabilisticConnector

"""
This file contains meta-connectors. These are classes that represent some higher-level 
algorithms for connecting neurons in mozaik.

A typical use case is when several projections have to be somehow synchronized and therefore
the code generating the connections cannot be completely broken up into individual MozaikConnector
classes. Use of meta-connectors is discoureged unless there is no way to implement the given 
connectivity with MozaikConnector class (or some of it derivatives).

An example of usage is the case of Gabor connectivity from LGN to V1. Here one needs to make sure that a given V1 neuron
follows the pattern of connectivity with respect to a given gabor pattern both in the LGNOn and LGNOff population - therefore
information has to be shared between these two projection during the creation process (e.g. the same orientation of the gabor for 
a given neuron has to be used when calclulating both the ON and OFF connections to a given neuron)
"""


class GaborConnector(BaseComponent):
    """
    Connector that creates Gabor projections.

    The individual Gabor parameters are drawn from distributions specified in
    the parameter set:

    `aspect_ratio`  -  aspect ratio of the gabor
    `size`          -  the size of the gabor  RFs in degrees of visual field
    `orientation`   -  the orientation of the gabor RFs
    `phase`         -  the phase of the gabor RFs
    `frequency`     -  the frequency of the gabor in degrees of visual field

    Other parameters:
    
    `topological`          -  should the receptive field centers vary with the
                              position of the given neurons in the target sheet
                              (note positions of neurons are always stored in
                              visual field coordinates)
    
    `delay`                -  (ms/μm) the delay on the projections

    `short_term_plasticity` - short term plasticity configuration (see basic connector)
    
    `base_weight`          - The weight of the synapses
    `num_samples`           - The number of synapses per cortical neuron from each of the ON and OFF LGN populations (so effectively there will be 2 * num_samples LGN synapses)

    `or_map`             - is an orientation map supplied?
    `or_map_location`    - if or_map is True where can one find the map. It
                           has to be a file containing a single pickled 2d
                           numpy array
    `phase_map`          - is a phase map supplied?
    `phase_map_location` - if phase_map is True where can one find the map.
                           It has to be a file containing a single pickled 2d
                           numpy array
    """

    required_parameters = ParameterSet({
        'aspect_ratio': ParameterDist,  # aspect ratio of the gabor
        'size':         ParameterDist,  # the size of the gabor  RFs in degrees of visual field
        'orientation_preference':  ParameterDist,  # the orientation preference of the gabor RFs (note this is the orientation of the gabor + pi/2)
        'phase':        ParameterDist,  # the phase of the gabor RFs
        'frequency':    ParameterDist,  # the frequency of the gabor in degrees of visual field

        'topological': bool,  # should the receptive field centers vary with the position of the given neurons
                              # (note positions of neurons are always stored in visual field coordinates)
        'delay': float,         # ms/μm the delay on the projections

        'short_term_plasticity': ParameterSet,
        'base_weight' : float, # the weights of synapses
        'num_samples' : int , # number of synapses per cortical neuron from each of the ON and OFF LGN populations (so effectively there will be 2 * num_samples LGN synapses)

        'or_map': bool,  # is a orientation map supplied?
        'or_map_location': str,  # if or_map is True where can one find the map. It has to be a file containing a single pickled 2d numpy array

        'phase_map': bool,  # is a phase map supplied?
        'phase_map_location': str,  # if phase_map is True where can one find the map. It has to be a file containing a single pickled 2d numpy array
    })

    def __init__(self, network, lgn_on, lgn_off, target, parameters, name):
        from numpy import random
        random.seed(1023)
        BaseComponent.__init__(self, network, parameters)
        self.name = name

        t_size = target.size_in_degrees()
        or_map = None
        if self.parameters.or_map:

            f = open(self.parameters.or_map_location, 'r')
            or_map = pickle.load(f)*numpy.pi
            coords_x = numpy.linspace(-t_size[0]/2.0,
                                      t_size[0]/2.0,
                                      numpy.shape(or_map)[0])
            coords_y = numpy.linspace(-t_size[1]/2.0,
                                      t_size[1]/2.0,
                                      numpy.shape(or_map)[1])
            print min(coords_x), max(coords_x)
            print min(coords_y), max(coords_y)
                                      
            X, Y = numpy.meshgrid(coords_x, coords_y)
            
            or_map = NearestNDInterpolator(zip(X.flatten(), Y.flatten()),
                                           or_map.flatten())

        phase_map = None
        if self.parameters.phase_map:
            f = open(self.parameters.phase_map_location, 'r')
            phase_map = pickle.load(f)
            coords_x = numpy.linspace(-t_size[0]/2.0,
                                      t_size[0]/2.0,
                                      numpy.shape(phase_map)[0])
            coords_y = numpy.linspace(-t_size[1]/2.0,
                                      t_size[1]/2.0,
                                      numpy.shape(phase_map)[1])
            X, Y = numpy.meshgrid(coords_x, coords_y)
            phase_map = NearestNDInterpolator(zip(X.flatten(), Y.flatten()),
                                              phase_map.flatten())
        
        print min(target.pop.positions[0]), max(target.pop.positions[0])
        print min(target.pop.positions[1]), max(target.pop.positions[1])
        for (j, neuron2) in enumerate(target.pop.all()):
            if or_map:
                orientation = or_map(target.pop.positions[0][j],
                                     target.pop.positions[1][j])
            else:
                orientation = parameters.orientation_preference.next()[0]

            if phase_map:
                phase = phase_map(target.pop.positions[0][j],
                                  target.pop.positions[1][j])
            else:
                phase = parameters.phase.next()[0]

            aspect_ratio = parameters.aspect_ratio.next()[0]
            frequency = parameters.frequency.next()[0]
            size = parameters.size.next()[0]

            assert orientation < numpy.pi

            target.add_neuron_annotation(j, 'LGNAfferentOrientation', orientation, protected=True)
            target.add_neuron_annotation(j, 'LGNAfferentAspectRatio', aspect_ratio, protected=True)
            target.add_neuron_annotation(j, 'LGNAfferentFrequency', frequency, protected=True)
            target.add_neuron_annotation(j, 'LGNAfferentSize', size, protected=True)
            target.add_neuron_annotation(j, 'LGNAfferentPhase', phase, protected=True)
            
            if self.parameters.topological:
                target.add_neuron_annotation(j, 'LGNAfferentX', target.pop.positions[0][j], protected=True)
                target.add_neuron_annotation(j, 'LGNAfferentY', target.pop.positions[1][j], protected=True)
            else:
                target.add_neuron_annotation(j, 'LGNAfferentX', 0, protected=True)
                target.add_neuron_annotation(j, 'LGNAfferentY', 0, protected=True)
                

        ps = ParameterSet({   'target_synapses' : 'excitatory',               
                              'weight_functions' : {  'f1' : {
                                                                 'component' : 'mozaik.connectors.vision.GaborArborization',
                                                                 'params' : {
                                                                                'ON' : True,
                                                                            }
                                                             }                                                                              
                                                   },
                             'delay_functions' : {},
                             'weight_expression' : 'f1', # a python expression that can use variables f1..fn where n is the number of functions in weight_functions, and fi corresponds to the name given to a ModularConnectorFunction in weight_function ParameterSet. It determines how are the weight functions combined to obtain the weights
                             'delay_expression' : str(self.parameters.delay),
                             'short_term_plasticity' : self.parameters.short_term_plasticity,
                             'base_weight' : self.parameters.base_weight,
                             'num_samples' : self.parameters.num_samples,
                          })
        ModularSamplingProbabilisticConnector(network,name+'On',lgn_on,target,ps).connect()
        ps['weight_functions.f1.params.ON']=False
        ModularSamplingProbabilisticConnector(network,name+'Off',lgn_off,target,ps).connect()
           
           
           
