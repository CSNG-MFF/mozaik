# the stimulus taxaonomy

# will get more sofisticated but for now only a list of unique stimulus identifiers with the number of their parameters 
# For simplicity - for now - a stimulus is uniquely identified by a string which contains its identified followed 
# by space sepparated number of (floating) numbers that correspond to its parameters
# eg:   'SinusoidalGrating 0.0 1.0 3.14 4.3 3'
# each stimulus should have at least 5 parameters corresponding to those all mozaik VisualStimulus object require:
# frame_duration
# size_in_degrees // note this is a bounding box of the stimulus which should correspond to the size of the visual field rather than the 'geometrical size'
# location_x
# location_y
# max_luminance 
# stimulus_duration
# density
# trial
# which have to be the first 7 parameters in this order
#
# JACOMMENT: for now we support only square stimuli - (i.e. the size is the same in both axis)
#


#StimulusTaxonomy contains the list of known stimuli with the number of their free parameters
base_stimulus_parameters = ['frame duration','size_in_degrees','x center coor', 'y center coor','maximum luminance','stimulus duration','density','trial']

StimulusTaxonomy = {
                        'FullfieldDriftingSinusoidalGrating' : base_stimulus_parameters + ['orientation','spatial_frequency','temporal_frequency'],
                        'Null' : base_stimulus_parameters,
                   }

from MozaikLite.framework.interfaces import VisualStimulus
from MozaikLite.framework.interfaces import MozaikLiteParametrizeObject
from NeuroTools.parameters import ParameterSet, ParameterDist
import numpy
import sys
sys.path.append('/home/jan/topographica/')



def load_from_string(string):
    return parse_stimuls_id(string).load_stimulus()

def parse_stimuls_id(string):
    words = string.rsplit();
    return StimulusID(words[0],words[1:])


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
          return StimulusTaxonomy[self.name][paramter_index]
           
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

      
class Stimulus(VisualStimulus,MozaikLiteParametrizeObject):
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

        def __init__(self, parameters, external_parameters):
            self.vparams = parameters
            self.params = parameters[8:] # store the rest of the params in self.params - easy way to access for the derived classes
            self.duration = parameters[5]
            self.density = parameters[6]
            self.trial = parameters[7]
            VisualStimulus.__init__(self,parameters[0],(parameters[1],parameters[1]), (parameters[2],parameters[3]), parameters[4]) 
            MozaikLiteParametrizeObject.__init__(self,external_parameters)
            self.n_frames = numpy.inf # possibly very dangerous. Don't do 'for i in range(stim.n_frames)'!

# The following stimuli are currently a hack that uses the topographica to generate the stimulus
import  topo.pattern.basic
from topo.base.boundingregion import BoundingBox

class FullfieldDriftingSinusoidalGrating(Stimulus):
    def frames(self):
            """
            max_luminance is interpreted as scale
            and size_in_degrees as the bounding box size
            parameters are in this order (after the 7 default ones)
            orientation
            spatial_frequency
            temporal_frequency (Hz)
            """
            
            self.current_phase=0
            while True:
                
                yield (topo.pattern.basic.SineGrating(orientation=self.params[0],frequency=self.params[1],phase=self.current_phase,size=self.size_in_degrees[0],bounds=BoundingBox(radius=self.size_in_degrees[0]/2),scale=self.max_luminance,xdensity=self.density,ydensity=self.density)(),[self.current_phase])
                self.current_phase+= 2*numpy.pi*(self.frame_duration/1000.0)*self.params[2]


class Null(Stimulus):
    def frames(self):
            """
            empty stimulus
            """
            while True:
                yield topo.pattern.basic.Null(scale=0,size=self.size_in_degrees[0])(), []
                




class NaturalImageWithEyeMovement(Stimulus):
    """
    A visual stimulus that simulates an eye movement over a static image
    
    Parameter order:
    'eye_movement_period' -  # (ms) the time between two consequitve eye movements recorded in the eye_path file
    `idd`                 -  JAHACK: this is probably just a hack for now how to 
                             make two stimuli with different external(hidden) parameters  
                             to have unique parameter combinations
    """    
    
    required_parameters = ParameterSet({
          'image_location' : str, # path to the static image 
          'eye_path_location' : str,  # path to a file containing the eye_path as a pickled list of tuples containing the coordinates
    })
    
    def __init__(self, parameters, external_parameters):
            Stimulus.__init__(self,parameters, external_parameters) 
            f = open(self.parameters.eye_path_location,'r')
            self.eye_path = pickle.load(f)
            import topo.pattern.image 

    def frames(self):
            """
            
            """
            self.time=0
            while True:
                location = self.eye_path[numpy.floor(self.duration*self.time/self.eye_movement_period)]
                image = topo.pattern.image.Image(filename=self.parameters.eye_path_location,
                                             x=location[0],
                                             y=location[1],
                                             orientation=0,
                                             xdensity=self.density,
                                             ydensity=self.density,
                                             size=max(self.params[0],self.params[1]),
                                             bounds=BoundingBox(radius=self.size_in_degrees[0]/2),
                                             scale=self.parameters.max_luminance)()
            
                
                yield (image,[self.time])
                self.time = time + 1
