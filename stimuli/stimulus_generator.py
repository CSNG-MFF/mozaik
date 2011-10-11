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
# which should be the first 7 parameters in this order
#
# JACOMMENT: for now we support only square stimuli - (i.e. the size is the same in both axis)
#

#StimulusTaxonomy contains the list of known stimuli with the number of their free parameters
StimulusTaxonomy = {
                        'FullfieldDriftingSinusoidalGrating' : 10,
                        'Null' : 7,
                   }


from MozaikLite.framework.interfaces import VisualStimulus
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
          assert StimulusTaxonomy[name] == len(parameters), 'Stimulus <%s>  requires %d parameters, %d given' % name
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
      
      def load_stimulus(self):
          for p in self.parameters:
              assert (not (isinstance(p,int) or isinstance(p,float))) , 'The parameters are not in correct format, perhaps parameter collapsing was performed' 
          cls = globals()[self.name]
          return cls([float(a) for a in self.parameters])  
      
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
            self.params = parameters[7:]
            self.duration = parameters[5]
            self.density = parameters[6]
            VisualStimulus.__init__(self,parameters[0],(parameters[1],parameters[1]), (parameters[2],parameters[3]), parameters[4]) 
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
                import pylab
                #if self.current_phase==0:
                #   pylab.figure()
                #   pylab.imshow(topo.pattern.basic.SineGrating(orientation=self.params[0],frequency=self.params[1],phase=self.current_phase,size=self.parameters.size_in_degrees[0],bounds=BoundingBox(radius=self.parameters.size_in_degrees[0]/2),scale=self.parameters.max_luminance,xdensity=self.density,ydensity=self.density)())
                #   pylab.title('image')
                yield (topo.pattern.basic.SineGrating(orientation=self.params[0],frequency=self.params[1],phase=self.current_phase,size=self.parameters.size_in_degrees[0],bounds=BoundingBox(radius=self.parameters.size_in_degrees[0]/2),scale=self.parameters.max_luminance,xdensity=self.density,ydensity=self.density)(),[self.current_phase])
                self.current_phase+= 2*numpy.pi*(self.parameters.frame_duration/1000.0)*self.params[2]

    def describe(self):
        """
        Returns a string containing a description of the stimulus.
        """
        s = self.__doc__ + VisualRegion.describe(self)
        return s


class Null(Stimulus):

    def frames(self):
            """
            empty stimulus
            """
            while True:
                yield topo.pattern.basic.Null(scale=0,size=self.parameters.size_in_degrees[0])(), []
                


    def describe(self):
        """
        Returns a string containing a description of the stimulus.
        """
        s = self.__doc__ + VisualRegion.describe(self)
        return s




