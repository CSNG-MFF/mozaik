"""
The file contains stimuli that use topographica to generate the stimulus

"""

from visual_stimulus import VisualStimulus
import imagen
from imagen.image import BoundingBox
import pickle
import numpy
from mozaik.tools.mozaik_parametrized import SNumber, SString
from mozaik.tools.units import cpd
from numpy import pi
from quantities import Hz, rad, degrees, ms, dimensionless

class TopographicaBasedVisualStimulus(VisualStimulus):
    """
    As we do not handle transparency in the Topographica stimuli (i.e. all pixels of all stimuli difned here will have 0% transparancy)
    in this abstract class we disable the transparent flag defined by the :class:`mozaik.stimuli.visual_stimulus.VisualStimulus`, to improve efficiency.
    """
    def __init__(self,**params):
        VisualStimulus.__init__(self,**params)
        self.transparent = False # We will not handle transparency anywhere here for now so let's make it fast


class SparseNoise(TopographicaBasedVisualStimulus):
    """
    Sparse noise 
    """
    
    def frames(self):
        while True:
            yield (imagen.random.Sparse(scale=self.background_luminance,
                                        bounds=BoundingBox(radius=self.size_x/2),
                                        xdensity=self.density,
                                        ydensity=self.density)(),
                   [self.frame_duration])

    

class FullfieldDriftingSinusoidalGrating(TopographicaBasedVisualStimulus):
    """
    A full field sinusoidal grating stimulus. 
    
    Notes
    -----
    `max_luminance` is interpreted as scale and `size_x/2` as the bounding box radius.
    """

    orientation = SNumber(rad, period=pi, bounds=[0,pi], doc="Grating orientation")
    spatial_frequency = SNumber(cpd, doc="Spatial frequency of grating")
    temporal_frequency = SNumber(Hz, doc="Temporal frequency of grating")
    contrast = SNumber(dimensionless,bounds=[0,100.0],doc="Contrast of the stimulus")

    def frames(self):
        self.current_phase=0
        i = 0
        while True:
            i += 1
            yield (imagen.SineGrating(orientation=self.orientation,
                                      frequency=self.spatial_frequency,
                                      phase=self.current_phase,
                                      bounds=BoundingBox(radius=self.size_x/2),
                                      offset = self.background_luminance*(100.0 - self.contrast)/100.0,
                                      scale=2*self.background_luminance*self.contrast/100.0,
                                      xdensity=self.density,
                                      ydensity=self.density)(),
                   [self.current_phase])
            self.current_phase += 2*pi * (self.frame_duration/1000.0) * self.temporal_frequency
            
                            
class Null(TopographicaBasedVisualStimulus):
    """
    Blank stimulus.
    """
    def frames(self):
        while True:
            yield (imagen.Null(scale=self.background_luminance,
                              bounds=BoundingBox(radius=self.size_x/2),
                              xdensity=self.density,
                              ydensity=self.density)(),
                   [self.frame_duration])


class NaturalImageWithEyeMovement(TopographicaBasedVisualStimulus):
    """
    A visual stimulus that simulates an eye movement over a static image.
    """
    size = SNumber(degrees, doc="The length of the longer axis of the image in visual degrees")
    eye_movement_period = SNumber(ms, doc="The time between two consequitve eye movements recorded in the eye_path file")
    image_location = SString(doc="Location of the image")
    eye_path_location = SString(doc="Location of file containing the eye path (two columns of numbers)")

    def frames(self):
        self.time = 0
        f = open(self.eye_path_location, 'r')
        self.eye_path = pickle.load(f)
        self.pattern_sampler = imagen.image.PatternSampler(
                                    size_normalization='fit_longest',
                                    whole_pattern_output_fns=[imagen.image.DivisiveNormalizeLinf()])

        while True:
            location = self.eye_path[int(numpy.floor(self.frame_duration * self.time / self.eye_movement_period))]
            image = imagen.image.FileImage(
                                    filename=self.image_location,
                                    x=location[0],
                                    y=location[1],
                                    orientation=0,
                                    xdensity=self.density,
                                    ydensity=self.density,
                                    size=self.size,
                                    bounds=BoundingBox(points=((-self.size_x/2, -self.size_y/2),
                                                               (self.size_x/2, self.size_y/2))),
                                    scale=2*self.background_luminance,
                                    pattern_sampler=self.pattern_sampler
                                    )()
            yield (image, [self.time])
            self.time += 1


class DriftingGratingWithEyeMovement(TopographicaBasedVisualStimulus):
    """
    A visual stimulus that simulates an eye movement over a drifting  gratings.
    """

    orientation = SNumber(rad, period=pi, bounds=[0,pi], doc="Grating orientation")
    spatial_frequency = SNumber(cpd, doc="Spatial frequency of grating")
    temporal_frequency = SNumber(Hz, doc="Temporal frequency of grating")
    contrast = SNumber(dimensionless,bounds=[0,100.0],doc="Contrast of the stimulus")
    eye_movement_period = SNumber(ms, doc="The time between two consequitve eye movements recorded in the eye_path file")
    eye_path_location = SString(doc="Location of file containing the eye path (two columns of numbers)")

    def frames(self):
        
        f = open(self.eye_path_location, 'r')
        self.eye_path = pickle.load(f)
        self.time = 0
        self.current_phase = 0
        while True:
            location = self.eye_path[int(numpy.floor(self.frame_duration * self.time / self.eye_movement_period))]

            image = imagen.SineGrating(orientation=self.orientation,
                                       x=location[0],
                                       y=location[1],
                                       frequency=self.spatial_frequency,
                                       phase=self.current_phase,
                                       bounds=BoundingBox(points=((-self.size_x/2, -self.size_y/2),
                                                                  (self.size_x/2, self.size_y/2))),
                                       offset = self.background_luminance*(100.0 - self.contrast)/100.0,
                                       scale=2*self.background_luminance*self.contrast/100.0,
                                       xdensity=self.density,
                                       ydensity=self.density)()
            self.current_phase += 2*pi * (self.frame_duration/1000.0) * self.temporal_frequency
            yield (image, [self.time])
            self.time = self.time + 1


class DriftingSinusoidalGratingDisk(TopographicaBasedVisualStimulus):
    """
    A drifting sinusoidal grating confined to a apareture of specified radius.
    
    Notes
    -----
    size_x/2 is interpreted as the bounding box radius.
    """
    contrast = SNumber(dimensionless,bounds=[0,100.0],doc="Contrast of the stimulus")
    orientation = SNumber(rad, period=pi, bounds=[0,pi], doc="Grating orientation")
    spatial_frequency = SNumber(cpd, doc="Spatial frequency of the grating")
    temporal_frequency = SNumber(Hz, doc="Temporal frequency of the grating")
    radius = SNumber(degrees, doc="The radius of the grating disk - in degrees of visual field")

    def frames(self):
        self.current_phase=0
        while True:
            a = imagen.SineGrating(orientation=self.orientation,
                                   frequency=self.spatial_frequency,
                                   phase=self.current_phase,
                                   bounds=BoundingBox(radius=self.size_x/2),
                                   offset = self.background_luminance*(100.0 - self.contrast)/100.0,
                                   scale=2*self.background_luminance*self.contrast/100.0,
                                   xdensity=self.density,
                                   ydensity=self.density)()
            
            b = imagen.Null(scale=self.background_luminance,
                            bounds=BoundingBox(radius=self.size_x/2),
                            xdensity=self.density,
                            ydensity=self.density)()
            c = imagen.Disk(smoothing=0.0,
                            size=self.radius*2,
                            scale=1.0,
                            bounds=BoundingBox(radius=self.size_x/2),
                            xdensity=self.density,
                            ydensity=self.density)()    
            d1 = numpy.multiply(a,c)
            d2 = numpy.multiply(b,-(c-1.0))
            d =  numpy.add.reduce([d1,d2])
            yield (d,[self.current_phase])
            self.current_phase += 2*pi * (self.frame_duration/1000.0) * self.temporal_frequency


class DriftingSinusoidalGratingCenterSurroundStimulus(TopographicaBasedVisualStimulus):
    """
    A standard stimulus to probe orientation specific surround modulation:
    A drifting grating in center surrounded by a drifting grating in the surround.
    Orientations of both center and surround gratings can be varied independently.

    Notes
    -----
    max_luminance is interpreted as scale and size_x/2 as the bounding box radius.
    """
    
    center_orientation = SNumber(rad, period=pi, bounds=[0,pi], doc="Center grating orientation")
    surround_orientation = SNumber(rad, period=pi, bounds=[0,pi], doc="Surround grating orientation")
    spatial_frequency = SNumber(cpd, doc="Spatial frequency of the grating (same for center and surround)")
    temporal_frequency = SNumber(Hz, doc="Temporal frequency of the grating (same for center and surround)")
    gap = SNumber(degrees, doc="The gap between center and surround grating - in degrees of visual field")
    center_radius = SNumber(degrees, doc="The (outside) radius of the center grating disk - in degrees of visual field")
    surround_radius = SNumber(degrees, doc="The (outside) radius of the surround grating disk - in degrees of visual field")
    contrast = SNumber(dimensionless,bounds=[0,100.0],doc="Contrast of the stimulus")

    def frames(self):
        self.current_phase = 0
        while True:
            center = imagen.SineGrating(mask_shape=imagen.pattern.Disk(smoothing=0.0, size=self.center_radius*2),
                                        orientation=self.center_orientation,
                                        frequency=self.spatial_frequency,
                                        phase=self.current_phase,
                                        bounds=BoundingBox(radius=self.size_x/2),
                                        offset = self.background_luminance*(100.0 - self.contrast)/100.0,
                                        scale=2*self.background_luminance*self.contrast/100.0,  
                                        xdensity=self.density,
                                        ydensity=self.density)()
            r = (self.center_radius + self.surround_radius + self.gap)/2
            t = (self.surround_radius - self.surround_radius - self.gap)/2
            surround = imagen.SineGrating(mask_shape=imagen.pattern.Ring(thickness=t, smoothing=0, size=r*2),
                                          orientation=self.surround_orientation,
                                          frequency=self.spatial_frequency,
                                          phase=self.current_phase,
                                          bounds=BoundingBox(radius=self.size_x/2),
                                          offset = self.background_luminance*(100.0 - self.contrast)/100.0,
                                          scale=2*self.background_luminance*self.contrast/100.0,   
                                          xdensity=self.density,
                                          ydensity=self.density)()
            yield (numpy.add.reduce([center, surround]), [self.current_phase])
            self.current_phase += 2*pi * (self.frame_duration/1000.0) * self.temporal_frequency
