# The file contains stimuli that use topographica to generate the stimulus
from visual_stimuli import VisualStimulus
import topo.pattern
from topo.base.boundingregion import BoundingBox
import pickle
import numpy
from mozaik.tools.mozaik_parametrized import *
from mozaik.tools.units import cpd
import quantities as qt


class FullfieldDriftingSinusoidalGrating(VisualStimulus):

    """
    max_luminance is interpreted as scale
    and size_in_degrees as the bounding box size
    """
    
    orientation = SNumber(qt.rad,doc="""Grating orientation""")
    spatial_frequency = SNumber(cpd,doc="""Spatial frequency of grating""")
    temporal_frequency = SNumber(qt.Hz,doc="""Temporal frequency of grating""")

    def frames(self):
        self.current_phase=0
        while True:
                yield (topo.pattern.SineGrating(orientation=self.orientation,frequency=self.spatial_frequency,phase=self.current_phase,size=self.size_x,bounds=BoundingBox(radius=self.size_x/2),scale=self.max_luminance,xdensity=self.density,ydensity=self.density)(),[self.current_phase])
                self.current_phase+= 2*numpy.pi*(self.frame_duration/1000.0)*self.temporal_frequency
                


class Null(VisualStimulus):
    def frames(self):
            """
            empty stimulus
            """
            while True:
                yield topo.pattern.Null(scale=0,size=self.size_in_degrees[0])(), []
                

class NaturalImageWithEyeMovement(VisualStimulus):
    """
    A visual stimulus that simulates an eye movement over a static image
    """    
    size = SNumber(qt.degrees,doc="""The length of the longer axis of the image in visual degrees""")
    eye_movement_period = SNumber(qt.ms,doc="""The time between two consequitve eye movements recorded in the eye_path file""")
    image_location = SString(doc="""Location of the image""")
    eye_path_location = SString(doc="""Location of file containing the eye path (two columns of numbers)""")
    
    def __init__(self, **params):
            VisualStimulus.__init__(self,params) 
            f = open(self.eye_path_location,'r')
            self.eye_path = pickle.load(f)
            self.pattern_sampler = topo.pattern.image.PatternSampler(size_normalization='fit_longest',whole_pattern_output_fns=[DivisiveNormalizeLinf()])

    def frames(self):
            self.time=0
            from topo.transferfn.basic import DivisiveNormalizeLinf
            import topo.pattern.image 
            
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
                                             pattern_sampler = self.pattern_sampler
                                             )()
                yield (image,[self.time])
                self.time = self.time + 1


class DriftingGratingWithEyeMovement(VisualStimulus):
    """
    A visual stimulus that simulates an eye movement over a drifting  gratings
    """
    
    orientation = SNumber(qt.rad,doc="""Grating orientation""")
    spatial_frequency = SNumber(cpd,doc="""Spatial frequency of grating""")
    temporal_frequency = SNumber(qt.Hz,doc="""Temporal frequency of grating""")
    eye_movement_period = SNumber(qt.ms,doc="""The time between two consequitve eye movements recorded in the eye_path file""")
    eye_path_location = SString(doc="""Location of file containing the eye path (two columns of numbers)""")

    def __init__(self, **params):
            VisualStimulus.__init__(self,params) 
            f = open(self.eye_path_location,'r')
            self.eye_path = pickle.load(f)
            self.pattern_sampler = topo.pattern.image.PatternSampler(size_normalization='fit_longest',whole_pattern_output_fns=[DivisiveNormalizeLinf()])

    def frames(self):
            self.time=0
            self.current_phase=0
            from topo.transferfn.basic import DivisiveNormalizeLinf
            import topo.pattern.image 
            
            while True:
                location = self.eye_path[int(numpy.floor(self.frame_duration*self.time/self.eye_movement_period))]
                
                image = topo.pattern.SineGrating(orientation=self.orientation,
                                                 x=location[0],
                                                 y=location[1],
                                                 frequency=self.spatial_frequency,
                                                 phase=self.current_phase,
                                                 size=self.size_x,
                                                 bounds=BoundingBox(points=((-self.size_x/2,-self.size_x/2),(-self.size_y/2,-self.size_y/2))),
                                                 scale=self.max_luminance,
                                                 xdensity=self.density,
                                                 ydensity=self.density)()
                self.time = self.time + 1
                self.current_phase+= 2*numpy.pi*(self.frame_duration/1000.0)*self.temporal_frequency
                yield (image,[self.time])
  
  
