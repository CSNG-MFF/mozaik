# encoding: utf-8
import pickle
import numpy
from numpy import sin, cos, pi, exp
from scipy.interpolate import NearestNDInterpolator
from mozaik.framework.interfaces import MozaikComponent
from mozaik.framework import load_component
from NeuroTools.parameters import ParameterSet, ParameterDist
from mozaik.connectors import SpecificProbabilisticArborization, SpecificArborization

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



def gabor(x1, y1, x2, y2, orientation, frequency, phase, size, aspect_ratio):
    X = (x1 - x2) * numpy.cos(orientation) + (y1 - y2) * numpy.sin(orientation)
    Y = -(x1 - x2) * numpy.sin(orientation) + (y1 - y2) * numpy.cos(orientation)
    ker = - (X*X + Y*Y*(aspect_ratio**2)) / (2*(size**2))
    return numpy.exp(ker)*numpy.cos(2*numpy.pi*X*frequency + phase)


class GaborConnector(MozaikComponent):
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
    `probabilistic`        -  should the weights be probabilistic or directly
                              proportianal to the gabor profile
    `delay`                -  (ms/μm) the delay on the projections

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
        'probabilistic': bool,  # should the weights be probabilistic or directly proportianal to the gabor profile
        'delay': float,         # ms/μm the delay on the projections

        'specific_arborization': ParameterSet,

        'or_map': bool,  # is a orientation map supplied?
        'or_map_location': str,  # if or_map is True where can one find the map. It has to be a file containing a single pickled 2d numpy array

        'phase_map': bool,  # is a phase map supplied?
        'phase_map_location': str,  # if phase_map is True where can one find the map. It has to be a file containing a single pickled 2d numpy array
    })

    def __init__(self, network, lgn_on, lgn_off, target, parameters, name):
        MozaikComponent.__init__(self, network, parameters)
        import pickle
        self.name = name
        on = lgn_on.pop
        off = lgn_off.pop

        on_weights = []
        off_weights = []

        t_size = target.size_in_degrees()
        or_map = None
        if self.parameters.or_map:
            f = open(self.parameters.or_map_location, 'r')
            or_map = pickle.load(f)*pi
            coords_x = numpy.linspace(-t_size[0]/2.0,
                                      t_size[0]/2.0,
                                      numpy.shape(or_map)[0])
            coords_y = numpy.linspace(-t_size[1]/2.0,
                                      t_size[1]/2.0,
                                      numpy.shape(or_map)[1])
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
        
        on_weights = numpy.zeros((lgn_on.pop.size,target.pop.size))    
        off_weights = numpy.zeros((lgn_on.pop.size,target.pop.size))    
        on_delays = numpy.zeros((lgn_on.pop.size,target.pop.size))   + self.parameters.delay
        off_delays = numpy.zeros((lgn_on.pop.size,target.pop.size))    + self.parameters.delay
        
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

            assert orientation < pi

            target.add_neuron_annotation(j, 'LGNAfferentOrientation', orientation, protected=True)
            target.add_neuron_annotation(j, 'LGNAfferentAspectRatio', aspect_ratio, protected=True)
            target.add_neuron_annotation(j, 'LGNAfferentFrequency', frequency, protected=True)
            target.add_neuron_annotation(j, 'LGNAfferentSize', size, protected=True)
            target.add_neuron_annotation(j, 'LGNAfferentPhase', phase, protected=True)



            if parameters.topological:
                on_weights[:,j] =  numpy.maximum(0, gabor(on.positions[0],
                                                       on.positions[1],
                                                       target.pop.positions[0][j],
                                                       target.pop.positions[1][j],
                                                       orientation + pi/2,
                                                       frequency,
                                                       phase,
                                                       size,
                                                       aspect_ratio))
                off_weights[:,j] = -numpy.minimum(0, gabor(off.positions[0],
                                                         off.positions[1],
                                                         target.pop.positions[0][j],
                                                         target.pop.positions[1][j],
                                                         orientation + pi/2,
                                                         frequency,
                                                         phase,
                                                         size,
                                                         aspect_ratio))
            else:
                on_weights[:,j] = numpy.maximum(0, gabor(on.positions[0],
                                                       on.positions[1],
                                                       0,
                                                       0,
                                                       orientation + pi/2,
                                                       frequency,
                                                       phase,
                                                       size,
                                                       aspect_ratio))
                                                       
                off_weights[:,j] = -numpy.minimum(0, gabor(off.positions[0],
                                                         off.positions[1],
                                                         0,
                                                         0,
                                                         orientation + pi/2,
                                                         frequency,
                                                         phase,
                                                         size,
                                                         aspect_ratio))
                    
        if parameters.probabilistic:
            on_proj = SpecificProbabilisticArborization(
                            network, lgn_on, target, on_weights,on_delays,
                            parameters.specific_arborization,
                            'ON_to_[' + target.name + ']')
            off_proj = SpecificProbabilisticArborization(
                            network, lgn_off, target, off_weights,off_delays,
                            parameters.specific_arborization,
                            'OFF_to_[' + target.name + ']')
        else:
            on_proj = SpecificArborization(
                            network, lgn_on, target, on_weights,on_delays,
                            parameters.specific_arborization,
                            'ON_to_[' + target.name + ']')
            off_proj = SpecificArborization(
                            network, lgn_off, target, off_weights,off_delays,
                            parameters.specific_arborization,
                            'OFF_to_[' + target.name + ']')

        on_proj.connect()
        off_proj.connect()
        #on_proj.connection_field_plot_continuous(0)
        #off_proj.connection_field_plot_continuous(0)
        
