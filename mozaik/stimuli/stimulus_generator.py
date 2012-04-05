"""
The stimulus taxaonomy

will get more sofisticated but for now only a list of unique stimulus identifiers with the number of their parameters 
For simplicity - for now - a stimulus is uniquely identified by a string which contains its identified followed 
by space sepparated number of (floating) numbers that correspond to its parameters
eg:   'SinusoidalGrating 0.0 1.0 3.14 4.3 3'
each stimulus should have at least 5 parameters corresponding to those all mozaik VisualStimulus object require:
frame_duration
size_in_degrees x// note this is a bounding box of the stimulus which should correspond to the size of the visual field rather than the 'geometrical size'
size_in_degrees y// note this is a bounding box of the stimulus which should correspond to the size of the visual field rather than the 'geometrical size'
location_x
location_y
max_luminance 
stimulus_duration
density
trial
which have to be the first 7 parameters in this order

JACOMMENT: for now we support only square stimuli - (i.e. the size is the same in both axis)

StimulusTaxonomy contains the list of known stimuli with the number of their free parameters
"""

from mozaik.framework.interfaces import VisualStimulus
from NeuroTools.parameters import ParameterSet, ParameterDist
import quantities as qt
import numpy
import sys


base_stimulus_parameters = [('frame duration',qt.ms),('size_in_degrees_x',qt.degrees),('size_in_degrees_y',qt.degrees),('x center coor',qt.degrees),('y center coor',qt.degrees),('maximum luminance',qt.dimensionless),('stimulus duration',qt.ms),('density',qt.dimensionless),('trial',qt.dimensionless)]

StimulusTaxonomy = {
                        'FullfieldDriftingSinusoidalGrating' : base_stimulus_parameters + [('orientation',qt.rad),('spatial_frequency',1/qt.degree),('temporal_frequency',qt.Hz)],
                        'NaturalImageWithEyeMovement' : base_stimulus_parameters + [('size',qt.degrees),('eye_movement_period',qt.ms),('idd',qt.dimensionless)],
                        'Null' : base_stimulus_parameters,
                   }

def load_from_string(string):
    return parse_stimuls_id(string).load_stimulus()

def parse_stimuls_id(string):
    words = string.rsplit();
    return StimulusID(words[0],words[1:])

def fromat_stimulus_id(stimulus_id):
    string = ''
    for p in stimulus_id.parameters:
        if p != '*' and p != 'x':
            string = string + ' ' + str(p)
    return string


class StimulusID(object):
      
      def __init__(self, name, parameters):
          assert StimulusTaxonomy.has_key(name), 'No stimulus with name <%s>  in Stimulus Taxonomy' % name
          assert len(StimulusTaxonomy[name]) == len(parameters), 'Stimulus <%s>  requires %d parameters, %d given' % name
          self.parameters = parameters  
          self.name = name
          self.num_parameters = len(parameters)
    
      def __str__(self):
            string = self.name
            for p in self.parameters:
                string = string + ' ' + str(p)
            return string
            
      def __eq__(self, other):
            if other.__class__ == self.__class__:
               if self.parameters == other.parameters:
                   return True
            return False
      
      def get_paramter_name(self,paramter_index):      
          """
          Returns the name of the prameter_index-th parameter of stimulus stimulus_name.
          """
          return StimulusTaxonomy[self.name][paramter_index][0]

    
      def get_parameter_units(self,parameter_index):
          """
          Returns the units of the prameter_index-th parameter of stimulus stimulus_name.
          """
          return StimulusTaxonomy[self.name][parameter_index][1]

           
      def load_stimulus(self):
          for p in self.parameters:
              assert (not (isinstance(p,int) or isinstance(p,float))) , 'The parameters are not in correct format, perhaps parameter collapsing was performed' 
          cls = globals()[self.name]
          return cls([float(a) for a in self.parameters])  
          
          


def _colapse(dd,axis):
    d = {}
    for s in dd:
        s1 = parse_stimuls_id(s)
        s1.parameters[axis]='*'
        s1 = str(s1)
        if d.has_key(s1):
           d[s1].extend(dd[s])
        else:
           d[s1] = dd[s]
    return d
    
def colapse(value_list,stimuli_list,parameter_indexes=[]):
    ## it colapses the value_list acording to stimuli with the same value 
    ## of parameters whose indexes are listed in the <parameter_indexes> and 
    ## replaces the collapsed parameters in the 
    ## stimuli_list with *
    d = {}
    for v,s in zip(value_list,stimuli_list):
        d[str(s)]=[v]

    for ax in parameter_indexes:
        d = _colapse(d,ax)
    
    return ([d[k] for k in d.keys()] ,d.keys())

      
class Stimulus(VisualStimulus):
        def __str__(self):
            string = self.__class__.__name__
            for p in self.vparams:
                string = string + ' ' + str(p)
            return string
            
        def __eq__(self, other):
            if other.__class__ == self.__class__:
               if self.vparams == other.vparams:
                   return True
            return False

        def __init__(self, parameters):
            self.vparams = parameters
            self.params = parameters[9:] # store the rest of the params in self.params - easy way to access for the derived classes
            self.duration = parameters[6]
            self.density = parameters[7]
            self.trial = parameters[8]
            VisualStimulus.__init__(self,parameters[0],(parameters[1],parameters[2]), (parameters[3],parameters[4]), parameters[5]) 
            self.n_frames = numpy.inf # possibly very dangerous. Don't do 'for i in range(stim.n_frames)'!



