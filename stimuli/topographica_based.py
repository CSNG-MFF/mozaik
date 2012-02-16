# The file contains stimuli that use topographica to generate the stimulus
from stimulus_generator import Stimulus
import  topo.pattern.basic
from topo.base.boundingregion import BoundingBox
import pickle
import numpy

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
    `size`              -    the length of the longer axis of the image in visual degrees
    `eye_movement_period` -  # (ms) the time between two consequitve eye movements recorded in the eye_path file
    `idd`                 -  JAHACK: this is probably just a hack for now how to 
                             make two stimuli with different external(hidden) parameters  
                             to have unique parameter combinations
    """    
    def __init__(self, parameters,eye_path_location,image_location):
            f = open(eye_path_location,'r')
            self.eye_path = pickle.load(f)
            self.image_location = image_location

            Stimulus.__init__(self,parameters) 

    def frames(self):
            self.time=0
            from topo.transferfn.basic import DivisiveNormalizeLinf
            import topo.pattern.image 
            
            
            pattern_sampler = topo.pattern.image.PatternSampler(size_normalization='fit_longest',whole_pattern_output_fns=[DivisiveNormalizeLinf()])
            
            while True:
                location = self.eye_path[int(numpy.floor(self.frame_duration*self.time/self.params[1]))]
                image = topo.pattern.image.FileImage(filename=self.image_location,
                                             x=location[0],
                                             y=location[1],
                                             orientation=0,
                                             xdensity=self.density,
                                             ydensity=self.density,
                                             size=self.params[0],
                                             bounds=BoundingBox(points=((-self.size_in_degrees[0]/2,-self.size_in_degrees[1]/2),(self.size_in_degrees[0]/2,self.size_in_degrees[1]/2))),
                                             scale=self.max_luminance,
                                             pattern_sampler = pattern_sampler
                                             )()
                yield (image,[self.time])
                self.time = self.time + 1

