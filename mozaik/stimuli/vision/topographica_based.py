# -*- coding: utf-8 -*-
# TODO: Remove this once we switch to Python 3
"""
The file contains stimuli that use topographica to generate the stimulus

"""

from visual_stimulus import VisualStimulus
import imagen
import imagen.random
from imagen.transferfn import TransferFn
import param
from imagen.image import BoundingBox
import pickle
import numpy
import numpy as np
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
    Sparse noise stimulus.
    
    Produces a matrix filled with 0.5 values and one random entry with 0 or 1.
    The output is then transformed with the following rule:
    output = output * scale  + offset.
    """
    
    experiment_seed = SNumber(dimensionless, doc="The seed of a given experiment")
    duration = SNumber(ms, doc="Total duration of the frames")
    time_per_image = SNumber(ms, doc ="Duration of one image")
    grid_size = SNumber(dimensionless, doc = "Grid Size ")
    grid = SNumber(dimensionless, doc = "Boolean string to decide whether there is grid or not")

    def __init__(self,**params):
        TopographicaBasedVisualStimulus.__init__(self, **params)
        assert (self.time_per_image/self.frame_duration) % 1.0 == 0.0, "The duration of image presentation should be multiple of frame duration."
                
    def frames(self):
  
        aux = imagen.random.SparseNoise(
                                      grid_density = self.grid_size * 1.0 / self.size_x,
                                      grid = self.grid,
                                      offset= 0,
                                      scale= 2 * self.background_luminance,
                                      bounds=BoundingBox(radius=self.size_x/2),
                                      xdensity=self.density,
                                      ydensity=self.density,
                                      random_generator=numpy.random.RandomState(seed=self.experiment_seed))
        while True:
            aux2 = aux()
            for i in range(int(self.time_per_image/self.frame_duration)):
                yield (aux2,[0])
            

class DenseNoise(TopographicaBasedVisualStimulus):
    """
    Dense Noise 


    Produces a matrix with the values 0, 0.5 and 1 allocated at random
    and then scaled and translated by scale and offset with the next
    transformation rule:  result*scale + offset
    """
    
    experiment_seed = SNumber(dimensionless, doc="The seed of a given experiment") 
    duration = SNumber(ms, doc='Total duration of the frames')
    time_per_image = SNumber(ms, doc ='Duration of one image')
    grid_size = SNumber(dimensionless, doc = "Grid Size ")
       
    def __init__(self,**params):
        TopographicaBasedVisualStimulus.__init__(self, **params)
        assert (self.time_per_image/self.frame_duration) % 1.0 == 0.0
  
    def frames(self):
        aux = imagen.random.DenseNoise(
                                       grid_density = self.grid_size * 1.0 / self.size_x,
                                       offset = 0,
                                       scale = 2 * self.background_luminance, 
                                       bounds = BoundingBox(radius=self.size_x/2),
                                       xdensity = self.density,
                                       ydensity = self.density,
                                       random_generator=numpy.random.RandomState(seed=self.experiment_seed))
        
        while True:
            aux2 = aux()
            for i in range(self.time_per_image/self.frame_duration):
                yield (aux2,[0])


                    
class FullfieldDriftingSinusoidalGrating(TopographicaBasedVisualStimulus):
    """
    A full field sinusoidal grating stimulus. 
     
    A movies in which luminance is modulated as a sinusoid along one 
    axis and is constant in the perpendicular axis. The phase of 
    the sinusoid is increasing with time leading to a drifting pattern. 

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


class FullfieldDriftingSquareGrating(TopographicaBasedVisualStimulus):
    """
    A full field square grating stimulus.

    A movies composed of interlaced dark and bright bars spanning the width  
    the visual space. The bars are moving a direction perpendicular to their
    long axis. The speed is dictated by the *temporal_freuquency* parameter
    the width of the bars by *spatial_frequency* parameter.
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
            yield (imagen.SquareGrating(
                    orientation = self.orientation,
                    frequency = self.spatial_frequency,
                    phase = self.current_phase,
                    bounds = BoundingBox( radius=self.size_x/2 ),
                    offset = self.background_luminance*(100.0 - self.contrast)/100.0,
                    scale = 2*self.background_luminance*self.contrast/100.0,
                    xdensity = self.density,
                    ydensity = self.density)(),
                [self.current_phase])
            self.current_phase += 2*pi * (self.frame_duration/1000.0) * self.temporal_frequency

class FullfieldDriftingSinusoidalGratingA(TopographicaBasedVisualStimulus):
    """
    A full field square grating stimulus.

    A movies composed of interlaced dark and bright bars spanning the width  
    the visual space. The bars are moving a direction perpendicular to their
    long axis. The speed is dictated by the *temporal_freuquency* parameter
    the width of the bars by *spatial_frequency* parameter.
    """

    orientation = SNumber(rad, period=pi, bounds=[0,pi], doc="Grating orientation")
    spatial_frequency = SNumber(cpd, doc="Spatial frequency of grating")
    temporal_frequency = SNumber(Hz, doc="Temporal frequency of grating")
    contrast = SNumber(dimensionless,bounds=[0,100.0],doc="Contrast of the stimulus")
    offset_time = SNumber(dimensionless,bounds=[0,None],doc="")
    onset_time = SNumber(dimensionless,bounds=[0,None],doc="")

    def frames(self):
        self.current_phase=0
        i = 0
        t = 0
        while True:
            i += 1
            st = imagen.SineGrating(
                    orientation = self.orientation,
                    frequency = self.spatial_frequency,
                    phase = self.current_phase,
                    bounds = BoundingBox( radius=self.size_x/2 ),
                    offset = self.background_luminance*(100.0 - self.contrast)/100.0,
                    scale = 2*self.background_luminance*self.contrast/100.0,
                    xdensity = self.density,
                    ydensity = self.density)()
            if t > self.offset_time:
                st = st * 0 + self.background_luminance
            if t < self.onset_time:
                st = st * 0 + self.background_luminance
            
            yield (st,[self.current_phase])
            self.current_phase += 2*pi * (self.frame_duration/1000.0) * self.temporal_frequency
            t=t+self.frame_duration

 

class FlashingSquares(TopographicaBasedVisualStimulus):
    """
    A pair of displaced flashing squares. 

    A pair of squares separated by a constant distance of dimensions dictated by provided *spatial_frequency* parameter
    and flashing at frequency provided by the *temporal_frequency* parameter. 
    """
    orientation = SNumber(rad, period=pi, bounds=[0,pi], doc="Orientation of the square axis")
    spatial_frequency = SNumber(cpd, doc="Spatial frequency created by the squares and the gap between them")
    separation = SNumber(degrees, doc="The separation between the two squares")
    temporal_frequency = SNumber(Hz, doc="Temporal frequency of the flashing")
    contrast = SNumber(dimensionless,bounds=[0,100.0],doc="Contrast of the stimulus")
    separated = SNumber(dimensionless, doc = "Boolean string to decide whether the separation is specified or not")

    def frames(self):
        # the size length of square edge is given by half the period
        size = 1./(2*self.spatial_frequency)
        # if a separation size is provided we use it, otherwise we use the same as size
        if self.separated:
            halfseparation = self.separation/2.
        else:
            halfseparation = size
        # if the separation is less than the size of a square, the two squares will overlap and the luminance will be too much
        if halfseparation < size/2.:
            halfseparation = size/2.
        # flashing squares with a temporal frequency of 6Hz are happening every 1000/6=167ms
        time = self.duration/self.frame_duration
        stim_period = time/self.temporal_frequency
        t = 0
        t0 = 0
        # total time of the stimulus
        while t <= time:
            # frequency tick
            if (t-t0) >= stim_period:
                t0 = t
            # Squares presence on screen is half of the period.
            # Since the two patterns will be added together, 
            # the offset level is half it should be, to sum into the required level, 
            # and the scale level is twice as much, in order to overcome the presence of the other pattern
            if t <= t0+(stim_period/2):
                a = imagen.RawRectangle(
                        x = -halfseparation, 
                        y = 0,
                        orientation = self.orientation,
                        bounds = BoundingBox( radius=self.size_x/2 ),
                        offset = 0.5*self.background_luminance*(100.0 - self.contrast)/100.0, 
                        scale = 2*self.background_luminance*self.contrast/100.0,
                        xdensity = self.density,
                        ydensity = self.density,
                        size = size)()
                b = imagen.RawRectangle(
                        x = halfseparation, 
                        y = 0,
                        orientation = self.orientation,
                        bounds = BoundingBox( radius=self.size_x/2 ),
                        offset = 0.5*self.background_luminance*(100.0 - self.contrast)/100.0,
                        scale = 2*self.background_luminance*self.contrast/100.0,
                        xdensity = self.density,
                        ydensity = self.density,
                        size = size)()
                yield (numpy.add(a,b),[t])
            else:
                yield (imagen.Constant(
                        scale=self.background_luminance*(100.0 - self.contrast)/100.0,
                        bounds=BoundingBox(radius=self.size_x/2),
                        xdensity=self.density,
                        ydensity=self.density)(),
                    [t])
            # time
            t += 1


class Null(TopographicaBasedVisualStimulus):
    """
    Blank stimulus.

    All pixels of the visual field are set to background luminance.
    """
    def frames(self):
        while True:
            yield (imagen.Constant(scale=self.background_luminance,
                              bounds=BoundingBox(radius=self.size_x/2),
                              xdensity=self.density,
                              ydensity=self.density)(),
                   [self.frame_duration])


class MaximumDynamicRange(TransferFn):
    """
    It linearly maps 0 to the minimum of the image and 1.0 to the maximum in the image.
    """
    norm_value = param.Number(default=1.0)
    
    def __call__(self,x):
        mi = numpy.min(x)
        ma = numpy.max(x)

        if ma-mi != 0:
                x -= mi
                x *= 1/(ma-mi)

class NaturalImageWithEyeMovement(TopographicaBasedVisualStimulus):
    """
    A visual stimulus that simulates an eye movement over a static image.

    This is a movie that is generated by translating a 
    static image along a pre-specified path (presumably containing path
    that corresponds to eye-movements).
    
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
                                    whole_pattern_output_fns=[MaximumDynamicRange()])

        image = imagen.image.FileImage(         
                                    filename=self.image_location,
                                    x=0,
                                    y=0,
                                    orientation=0,
                                    xdensity=self.density,
                                    ydensity=self.density,
                                    size=self.size,
                                    bounds=BoundingBox(points=((-self.size_x/2, -self.size_y/2),
                                                               (self.size_x/2, self.size_y/2))),
                                    scale=2*self.background_luminance,
                                    pattern_sampler=self.pattern_sampler)

        while True:
            location = self.eye_path[int(numpy.floor(self.frame_duration * self.time / self.eye_movement_period))]
            image.x = location[0]
            image.y = location[1]
            yield (image(), [self.time])
            self.time += 1


class DriftingGratingWithEyeMovement(TopographicaBasedVisualStimulus):
    """
    A visual stimulus that simulates an eye movement over a drifting  gratings.

    This is a movie that is generated by translating a 
    full-field drifting sinusoidal gratings along a pre-specified path 
    (presumably containing path that corresponds to eye-movements).
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
    A drifting sinusoidal grating confined to a aperture of specified radius.

    A movies in which luminance is modulated as a sinusoid along one 
    axis and is constant in the perpendicular axis. The phase of 
    the sinusoid is advancing with time leading to a drifting pattern.
    The whole stimulus is confined to an aperture of  

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
            
            b = imagen.Constant(scale=self.background_luminance,
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


class FlatDisk(TopographicaBasedVisualStimulus):
    """
    A flat luminance aperture of specified radius.

    This stimulus corresponds to a disk of constant luminance of 
    pre-specified *radius* flashed for the *duration* of milliseconds
    on a constant background of *background_luminance* luminance.
    The luminance of the disk is specified by the *contrast* parameter,
    and is thus *background_luminance* + *background_luminance* \* (*self.contrast*/100.0).

    Notes
    -----
    size_x/2 is interpreted as the bounding box.
    """
    contrast = SNumber(dimensionless,bounds=[0,100.0],doc="Contrast of the stimulus")
    radius = SNumber(degrees, doc="The radius of the disk - in degrees of visual field")

    def frames(self):
        self.current_phase=0
        while True:  
            d = imagen.Disk(smoothing=0.0,
                            size=self.radius*2,
                            offset = self.background_luminance,
                            scale = self.background_luminance*(self.contrast/100.0),
                            bounds=BoundingBox(radius=self.size_x/2),
                            xdensity=self.density,
                            ydensity=self.density)()  
         
            yield (d,[self.current_phase])

class FlashedBar(TopographicaBasedVisualStimulus):
    """
    A flashed bar.

    This stimulus corresponds to flashing a bar of specific *orientation*,
    *width* and *length* at pre-specified position for *flash_duration* of milliseconds. 
    For the remaining time, until the *duration* of the stimulus, constant *background_luminance* 
    is displayed.
    """
    relative_luminance = SNumber(dimensionless,bounds=[0,1.0],doc="The scale of the stimulus. 0 is dark, 1.0 is double the background luminance")
    orientation = SNumber(rad, period=pi, bounds=[0,pi], doc="Grating orientation")
    width = SNumber(cpd, doc="Spatial frequency of the grating")
    length = SNumber(Hz, doc="Temporal frequency of the grating")
    flash_duration = SNumber(ms, doc="The duration of the bar presentation.")
    x = SNumber(degrees, doc="The x location of the center of the bar.")
    y = SNumber(degrees, doc="The y location of the center of the bar.")
    
    def frames(self):
        num_frames = 0
        while True:
    
            d = imagen.RawRectangle(offset = self.background_luminance,
                                    scale = self.background_luminance*(self.relative_luminance-0.5),
                                    bounds=BoundingBox(radius=self.size_x/2),
                                    xdensity=self.density,
                                    ydensity=self.density,
                                    x = self.x,
                                    y = self.y,
                                    orientation=self.orientation,
                                    size = self.width,
                                    aspect_ratio = self.length/ self.width)()  

                                    
            b = imagen.Constant(scale=self.background_luminance,
                    bounds=BoundingBox(radius=self.size_x/2),
                    xdensity=self.density,
                    ydensity=self.density)()
                    
            num_frames += 1;
            if (num_frames-1) * self.frame_duration < self.flash_duration: 
                yield (d,[1])
            else:
                yield (b,[0])
            
            
class DriftingSinusoidalGratingCenterSurroundStimulus(TopographicaBasedVisualStimulus):
    """
    Orientation-contrast surround stimulus.

    This is a standard stimulus to probe orientation specific surround modulation:
    a drifting sinusoidal grating in the center surrounded by a drifting grating 
    in the surround. Orientations of both center (*center_orientation* parameter) 
    and surround gratings (*surround_orientation* parameter) can be varied independently, 
    but they have common *spatial_frequency* and *temporal_frequency*. Gap of *gap* degrees
    of visual field can be placed between the center and surround stimulus.


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
            center = imagen.SineGrating(mask_shape=imagen.Disk(smoothing=0.0, size=self.center_radius*2.0),
                                        orientation=self.center_orientation,
                                        frequency=self.spatial_frequency,
                                        phase=self.current_phase,
                                        bounds=BoundingBox(radius=self.size_x/2.0),
                                        offset = 0,
                                        scale=2*self.background_luminance*self.contrast/100.0,  
                                        xdensity=self.density,
                                        ydensity=self.density)()
            r = (self.center_radius + self.surround_radius + self.gap)/2.0
            t = (self.surround_radius - self.center_radius - self.gap)/2.0
            surround = imagen.SineGrating(mask_shape=imagen.Ring(thickness=t*2.0, smoothing=0.0, size=r*2.0),
                                          orientation=self.surround_orientation,
                                          frequency=self.spatial_frequency,
                                          phase=self.current_phase,
                                          bounds=BoundingBox(radius=self.size_x/2.0),
                                          offset = 0,
                                          scale=2*self.background_luminance*self.contrast/100.0,   
                                          xdensity=self.density,
                                          ydensity=self.density)()
            
            offset = imagen.Constant(mask_shape=imagen.Disk(smoothing=0.0, size=self.surround_radius*2.0),
                                 bounds=BoundingBox(radius=self.size_x/2.0),
                                 scale=self.background_luminance*(100.0 - self.contrast)/100.0,
                                 xdensity=self.density,
                                 ydensity=self.density)()

            background = (imagen.Disk(smoothing=0.0,
                                     size=self.surround_radius*2.0, 
                                     bounds=BoundingBox(radius=self.size_x/2.0),
                                     xdensity=self.density,
                                     ydensity=self.density)()-1)*-self.background_luminance
            
            yield (numpy.add.reduce([numpy.maximum(center, surround),offset,background]), [self.current_phase])
            self.current_phase += 2*pi * (self.frame_duration/1000.0) * self.temporal_frequency

class DriftingSinusoidalGratingRing(TopographicaBasedVisualStimulus):
    """
    A standard stimulus to probe orientation specific surround modulation:
    A drifting grating in center surrounded by a drifting grating in the surround.
    Orientations of both center and surround gratings can be varied independently.

    Notes
    -----
    max_luminance is interpreted as scale and size_x/2 as the bounding box radius.
    """
    
    orientation = SNumber(rad, period=pi, bounds=[0,pi], doc="Center grating orientation")
    spatial_frequency = SNumber(cpd, doc="Spatial frequency of the grating ")
    temporal_frequency = SNumber(Hz, doc="Temporal frequency of the grating ")
    outer_appareture_radius = SNumber(degrees, doc="The outside radius of the grating ring - in degrees of visual field")
    inner_appareture_radius = SNumber(degrees, doc="The inside radius of the  grating ring - in degrees of visual field")
    contrast = SNumber(dimensionless,bounds=[0,100.0],doc="Contrast of the stimulus")

    def frames(self):
        self.current_phase = 0
        while True:
            r = (self.inner_appareture_radius + self.outer_appareture_radius)/2.0
            t = (self.outer_appareture_radius - self.inner_appareture_radius)/2.0
            ring = imagen.SineGrating(mask_shape=imagen.Ring(thickness=t*2.0, smoothing=0.0, size=r*2.0),
                                          orientation=self.orientation,
                                          frequency=self.spatial_frequency,
                                          phase=self.current_phase,
                                          bounds=BoundingBox(radius=self.size_x/2.0),
                                          offset = 0,
                                          scale=2*self.background_luminance*self.contrast/100.0,   
                                          xdensity=self.density,
                                          ydensity=self.density)()
            
            bg = imagen.Constant(bounds=BoundingBox(radius=self.size_x/2.0),
                                 scale=self.background_luminance,
                                 xdensity=self.density,
                                 ydensity=self.density)()

            correction = imagen.Ring(smoothing=0.0,
                                     thickness=t*2.0, 
                                     size=r*2.0,
                                     scale=-self.background_luminance,
                                     bounds=BoundingBox(radius=self.size_x/2.0),
                                     xdensity=self.density,
                                     ydensity=self.density)()
            
            yield (numpy.add.reduce([ring,bg,correction]), [self.current_phase])
            self.current_phase += 2*pi * (self.frame_duration/1000.0) * self.temporal_frequency


class FlashedInterruptedBar(TopographicaBasedVisualStimulus):
    """
    A flashed bar.

    This stimulus corresponds to flashing a bar of specific *orientation*,
    *width* and *length* at pre-specified position for *flash_duration* of milliseconds. 
    For the remaining time, until the *duration* of the stimulus, constant *background_luminance* 
    is displayed.
    """
    relative_luminance = SNumber(dimensionless,bounds=[0,1.0],doc="The scale of the stimulus. 0 is dark, 1.0 is double the background luminance.")
    orientation = SNumber(rad, period=pi, bounds=[0,pi], doc="Grating orientation.")
    disalignment = SNumber(rad, period=pi, bounds=[-pi/2,pi/2], doc="The orientation by which the flanking bars are rotated away from the principal orientation axis.")
    width = SNumber(cpd, doc="Width of the bar")
    length = SNumber(Hz, doc="Length of the bar`")
    flash_duration = SNumber(ms, doc="The duration of the bar presentation.")
    x = SNumber(degrees, doc="The x location of the center of the bar (where the gap will appear).")
    y = SNumber(degrees, doc="The y location of the center of the bar (where the gap will appear).")
    gap_length = SNumber(Hz, doc="Length of the gap in the center of the bar")
    
    def frames(self):
        num_frames = 0
        while True:
            
            z = self.gap_length/4.0 + self.length/4.0 

            d1 = imagen.RawRectangle(offset = self.background_luminance,
                                    scale = 2*self.background_luminance*(self.relative_luminance-0.5),
                                    bounds=BoundingBox(radius=self.size_x/2),
                                    xdensity=self.density,
                                    ydensity=self.density,
                                    x = self.x + numpy.cos(self.orientation) * (z),
                                    y = self.y + numpy.sin(self.orientation) * (z),
                                    orientation=self.orientation+self.disalignment,
                                    size = self.width,
                                    aspect_ratio = (self.length-self.gap_length)/2/self.width)()  

            d2 = imagen.RawRectangle(offset = self.background_luminance,
                                    scale = 2*self.background_luminance*(self.relative_luminance-0.5),
                                    bounds=BoundingBox(radius=self.size_x/2),
                                    xdensity=self.density,
                                    ydensity=self.density,
                                    x = self.x + numpy.cos(self.orientation) * (-z),
                                    y = self.y + numpy.sin(self.orientation) * (-z),
                                    orientation=self.orientation+self.disalignment,
                                    size = self.width,
                                    aspect_ratio = (self.length-self.gap_length)/2/self.width)()  

                                    
            b = imagen.Constant(scale=self.background_luminance,
                    bounds=BoundingBox(radius=self.size_x/2),
                    xdensity=self.density,
                    ydensity=self.density)()
                    
            num_frames += 1;
            if (num_frames-1) * self.frame_duration < self.flash_duration: 
                if self.relative_luminance > 0.5: 
                   yield (numpy.maximum(d1,d2),[1])
                else:
                   yield (numpy.minimum(d1,d2),[1]) 
            else:
                yield (b,[0])



class FlashedInterruptedCorner(TopographicaBasedVisualStimulus):
    """
    A flashed bar.

    This stimulus corresponds to flashing a bar of specific *orientation*,
    *width* and *length* at pre-specified position for *flash_duration* of milliseconds. 
    For the remaining time, until the *duration* of the stimulus, constant *background_luminance* 
    is displayed.
    """
    relative_luminance = SNumber(dimensionless,bounds=[0,1.0],doc="The scale of the stimulus. 0 is dark, 1.0 is double the background luminance")
    orientation = SNumber(rad, period=pi, bounds=[0,pi], doc="Orientation of the corner")
    width = SNumber(degrees, doc="width of lines forming the corner")
    length = SNumber(Hz, doc="length of the corner if it were colinear")
    flash_duration = SNumber(ms, doc="The duration of the corner presentation.")
    x = SNumber(degrees, doc="The x location of the center of the bar (where the gap will appear).")
    y = SNumber(degrees, doc="The y location of the center of the bar (where the gap will appear).")
    left_angle = SNumber(rad, period=pi, bounds=[0,pi], doc="orientation of the left arm")
    right_angle = SNumber(rad, period=pi, bounds=[0,pi], doc="orientation of the right arm")
    gap_length = SNumber(Hz, doc="Length of the gap in the center of the bar")

    def frames(self):
        num_frames = 0
    
        while True:            
            length = self.length/2-self.gap_length/2.0
            shift = length/2.0+self.gap_length/2.0

            r1=imagen.Rectangle(x= shift*numpy.cos(self.right_angle),
            		            y= shift*numpy.sin(right_angle),
            		            offset = self.background_luminance,
                                scale = 2*self.background_luminance*(self.relative_luminance-0.5),
                			    orientation=numpy.pi/2+self.right_angle,
                			    smoothing=0,
                			    aspect_ratio=self.width/length,
                			    size=length,
                			    bounds=BoundingBox(radius=self.size_x/2),
                			    )


            r2=imagen.Rectangle(x=shift*numpy.cos(self.left_angle),
                			    y=shift*numpy.sin(left_angle),
                			    offset = self.background_luminance,
                                scale = 2*self.background_luminance*(self.relative_luminance-0.5),
                			    orientation=numpy.pi/2+self.left_angle,
                			    smoothing=0,
                			    aspect_ratio=self.width/length,
                			    size=length,
                			    bounds=BoundingBox(radius=self.size_x/2),
                 			    )
            	
            r=imagen.Composite(generators=[r1,r2],x=self.x,y=self.y,bounds=BoundingBox(radius=self.size_x/2),orientation=self.orientation,xdensity=self.density,ydensity=self.density)

            b = imagen.Constant(scale=self.background_luminance,
                    bounds=BoundingBox(radius=self.size_x/2),
                    xdensity=self.density,
                    ydensity=self.density)()

            num_frames += 1;
            if (num_frames-1) * self.frame_duration < self.flash_duration: 
                    yield (r(),[1])
            else:
                    yield (b,[0])


class VonDerHeydtIllusoryBar(TopographicaBasedVisualStimulus):
    """
    An illusory bar from Von Der Heydt et al. 1989.

    Von Der Heydt, R., & Peterhans, E. (1989). Mechanisms of contour perception in monkey visual cortex. I. Lines of pattern discontinuity. Journal of Neuroscience, 9(5), 1731–1748. Retrieved from https://www.jneurosci.org/content/jneuro/9/5/1731.full.pdf
    """
    orientation = SNumber(rad, period=pi, bounds=[0,pi], doc="Grating orientation")
    background_bar_width = SNumber(degrees, doc="Width of the background bar")
    occlusion_bar_width = SNumber(degrees, doc="Width of the occlusion bar")
    bar_width = SNumber(degrees, doc="Width of the bar")
    length = SNumber(Hz, doc="Length of the background bar")
    flash_duration = SNumber(ms, doc="The duration of the bar presentation.")
    x = SNumber(degrees, doc="The x location of the center of the bar (where the gap will appear).")
    y = SNumber(degrees, doc="The y location of the center of the bar (where the gap will appear).")
    
    def frames(self):
        num_frames = 0
        while True:
            

            d1 = imagen.RawRectangle(offset = 0,
                                    scale = 2*self.background_luminance,
                                    bounds=BoundingBox(radius=self.size_x/2),
                                    xdensity=self.density,
                                    ydensity=self.density,
                                    x = self.x-(self.occlusion_bar_width/2)-(self.length-self.occlusion_bar_width)/4,
                                    y = self.y,
                                    orientation=self.orientation,
                                    size = self.background_bar_width,
                                    aspect_ratio = (self.length-self.occlusion_bar_width)/2/self.background_bar_width)()  

            d2 = imagen.RawRectangle(offset = 0,
                                    scale = 2*self.background_luminance,
                                    bounds=BoundingBox(radius=self.size_x/2),
                                    xdensity=self.density,
                                    ydensity=self.density,
                                    x = self.x+(self.occlusion_bar_width/2)+(self.length-self.occlusion_bar_width)/4,
                                    y = self.y,
                                    orientation=self.orientation,
                                    size = self.background_bar_width,
                                    aspect_ratio = (self.length-self.occlusion_bar_width)/2/self.background_bar_width)()  

                                    
            b = imagen.Constant(scale=0,
                    bounds=BoundingBox(radius=self.size_x/2),
                    xdensity=self.density,
                    ydensity=self.density)()
                    
            num_frames += 1;
            if (num_frames-1) * self.frame_duration < self.flash_duration: 
                yield (numpy.add(d1,d2),[1])
            else:
                yield (b,[0])


class SimpleGaborPatch(TopographicaBasedVisualStimulus):
    """A flash of a Gabor patch

    This stimulus corresponds to flashing a Gabor patch of a specific
    *orientation*, *size*, *phase*, *spatial_frequency* at a defined position
    *x* and *y* for *flash_duration* milliseconds. For the remaining time, 
    until the *duration* of the stimulus, constant *background_luminance* 
    is displayed.
    """

    orientation = SNumber(rad, period=pi, bounds=[0,pi], doc="Gabor patch orientation")
    phase = SNumber(rad, period=2*pi, bounds=[0,2*pi], doc="Gabor patch phase")
    spatial_frequency = SNumber(cpd, doc="Spatial frequency of the grating")
    size = SNumber(degrees, doc="Size of the Gabor patch")
    flash_duration = SNumber(ms, doc="The duration of the bar presentation.")
    relative_luminance = SNumber(dimensionless,bounds=[0,1.0],doc="The scale of the stimulus. 0 is dark, 1.0 is double the background luminance")
    x = SNumber(degrees, doc="The x location of the center of the Gabor patch.")
    y = SNumber(degrees, doc="The y location of the center of the Gabor patch.")
    grid = SNumber(dimensionless, doc = "Boolean string to decide whether there is grid or not")


    def frames(self):
        num_frames = 0
        if self.grid:
            grid_pattern = hex_grid()
        while True:
            gabor = imagen.Gabor(
                        aspect_ratio = 1, # Ratio of pattern width to height.
                                          # Set since the patch has to be round
                        mask_shape=imagen.Disk(smoothing=0, size=3*self.size),
                            # Gabor patch should fit inside tide/circle
                            # the size is rescalled according to the size
                            # of Gabor patch
                        frequency = self.spatial_frequency,
                        phase = self.phase, # Initial phase of the sinusoid
                        bounds = BoundingBox(radius=self.size_x/2.), 
                            # BoundingBox of the
                            # area in which the pattern is generated,
                            # radius=1: box with side length 2!
                        size = self.size/3, # size = 2*standard_deviation
                            # => on the radius r=size, the intensity is ~0.14
                        orientation=self.orientation, # In radians
                        x = self.x,  # x-coordinate of Gabor patch center
                        y = self.y,  # y-coordinate of Gabor patch center
                        xdensity=self.density, # Number of points in one unit of length in x direction
                        ydensity=self.density, # Number of points in one unit of length in y direction
                        scale=2*self.background_luminance*self.relative_luminance, 
                                # Difference between maximal and minimal value 
                                # => min value = -scale/2, max value = scale/2
                                )()
            gabor = gabor+self.background_luminance # rescalling
            blank = imagen.Constant(scale=self.background_luminance,
                                    bounds=BoundingBox(radius=self.size_x/2),
                                    xdensity=self.density,
                                    ydensity=self.density)()
            if self.grid:
                gabor = gabor*grid_pattern
                blank = blank*grid_pattern
            num_frames += 1
            if (num_frames-1) * self.frame_duration < self.flash_duration: 
                yield (gabor, [1])
            else:
                yield (blank, [0])

    def hex_grid(self):
        """Creates an 2D array representing hexagonal grid based on given parameters

        Algorithm:
            First, it creates a tide containing two lines looking like: ___/
                the height of the tide is size/2
                the width of the tide is sqrt(3)/2*size
                NOTE: one of the parameters is not an integer -> therefore rounding
                    is present and for big grids it can be off by several pixels
                    the oddness can be derived from the ratio of width and height
                    the more close to sqrt(3) the better

            Second, it replicates the the tide to create hexagonal tiding of the 
            following form              ___ 
                                    ___/   \
                                       \___/
            Third, it replicates the hexagonal tide and rotates it

            Fourth, it computes shifts based on parameters size, size_x, size_y
                    and cuts out the relevant part of the array

        Returns:
            array with values 1 or 0, 0 representing the hexagonal grid
        """
        # imagen is used to create the slant line /
        ln = imagen.Line(bounds = BoundingBox(radius=self.size/4.), 
                    orientation = pi/3, smoothing=0,
                    xdensity = self.density, ydensity = self.density,  
                    thickness=1./self.density)()
        # cutting the relevant part of the created line
        idx = ln.argmax(axis=0).argmax()
        line = ln[:,idx:-idx]
        # Creating the horizontal line _
        hline = numpy.zeros((line.shape[0], line.shape[1]*2))
        hline[-1,:] = 1
        # Creating hexagonal tide
        tide = numpy.hstack((hline,line))  # joins horizontal line and slant line
        tide = numpy.hstack((tide, tide[::-1,:]))  # mirrors the tide in horizontal direction
        tide = numpy.vstack((tide, tide[::-1,:]))  # mirrors the tide in vertical direction
        d, k = tide.shape
        k = k/3
        # Creating hex
        x_reps = int(self.size_x/self.size) + 2
        y_reps = int(self.size_y/self.size) + 2
        # pixel sizes
        x_size = int(self.size_x*self.density)
        y_size = int(self.size_y*self.density)
        grid = numpy.hstack((numpy.vstack((tide,)*y_reps),)*x_reps)
        # starting indices in each dimension
        i = int((0.5+int(self.size_y/self.size)*1.5)*k)-int(self.density*self.size_y/2)
        j = d - int(self.size_x%self.size*self.density/2.)
        grid = grid[j:j+x_size,i:i+y_size]
        center = grid.shape[0]/2
        return 1-grid.T


class TwoStrokeGaborPatch(TopographicaBasedVisualStimulus):
    """A flash of two consecutive Gabor patches next to each other

    This stimulus corresponds to flashing a Gabor patch of a specific
    *orientation*, *size*, *phase*, *spatial_frequency* starting at a defined
    position *x* and *y* for *stroke_time* milliseconds. After that time
    Gabor patch is moved in the *x_direction* and *y_direction* to new place,
    where is presented until the *flash_duration* milliseconds from start of
    the experiment passes. For the remaining time,  until the *duration* of the
    stimulus, constant *background_luminance* is displayed.
    """

    orientation = SNumber(rad, period=pi, bounds=[0,pi], doc="Gabor patch orientation")
    phase = SNumber(rad, period=2*pi, bounds=[0,2*pi], doc="Gabor patch phase")
    spatial_frequency = SNumber(cpd, doc="Spatial frequency of the grating")
    size = SNumber(degrees, doc="Size of the Gabor patch")
    flash_duration = SNumber(ms, doc="The duration of the bar presentation.")
    first_relative_luminance = SNumber(dimensionless,bounds=[0,1.0], doc="The scale of the stimulus. 0 is dark, 1.0 is double the background luminance")
    second_relative_luminance = SNumber(dimensionless,bounds=[0,1.0],doc="The scale of the stimulus. 0 is dark, 1.0 is double the background luminance")
    x = SNumber(degrees, doc="The x location of the center of the Gabor patch.")
    y = SNumber(degrees, doc="The y location of the center of the Gabor patch.")
    stroke_time = SNumber(ms, doc="Duration of the first stroke.")
    x_direction = SNumber(degrees, doc="The x direction for the second stroke.")
    y_direction = SNumber(degrees, doc="The y direction for the second stroke.")
    grid = SNumber(dimensionless, doc = "Boolean string to decide whether there is grid or not")


    def frames(self):
        num_frames = 0
        # relative luminance of a current stroke
        current_luminance = self.first_relative_luminance
        if self.grid:
            grid_pattern = self.hex_grid()
        while True:
            gabor = imagen.Gabor(
                        aspect_ratio = 1, # Ratio of pattern width to height.
                                          # Set since the patch has to be round
                        mask_shape=imagen.Disk(smoothing=0, size=3*self.size),
                            # Gabor patch should fit inside tide/circle
                            # the size is rescalled according to the size
                            # of Gabor patch
                        frequency = self.spatial_frequency,
                        phase = self.phase, # Initial phase of the sinusoid
                        bounds = BoundingBox(radius=self.size_x/2), 
                            # BoundingBox of the area in which the pattern is
                            # generated, radius=1: box with side length 2!
                        size = self.size/3, # size = 2*standard_deviation
                            # => on the radius r=size, the intensity is ~0.14
                        orientation=self.orientation, # In radians
                        x = self.x,  # x-coordinate of Gabor patch center
                        y = self.y,  # y-coordinate of Gabor patch center
                        xdensity=self.density, # Number of points in one unit 
                                               # of length in x direction
                        ydensity=self.density, # Number of points in one unit 
                                               # of length in y direction
                        scale=2*self.background_luminance*current_luminance,
                                # Difference between maximal and minimal value 
                                # => min value = -scale/2, max value = scale/2
                                )()                
            gabor = gabor+self.background_luminance # rescalling
            blank = imagen.Constant(scale=self.background_luminance,
                                    bounds=BoundingBox(radius=self.size_x/2),
                                    xdensity=self.density,
                                    ydensity=self.density)()
            if self.grid:
                gabor = gabor*grid_pattern
                blank = blank*grid_pattern
            num_frames += 1
            if (num_frames-1) * self.frame_duration < self.stroke_time: 
                # First stroke
                yield (gabor, [1])
                if num_frames * self.frame_duration >= self.stroke_time:
                    # If next move is the second stroke -> change position 
                    # and luminance
                    self.x = self.x + self.x_direction 
                    self.y = self.y + self.y_direction 
                    current_luminance = self.second_relative_luminance
            elif (num_frames-1) * self.frame_duration < self.flash_duration: 
                # Second stroke
                yield (gabor, [1])
            else:
                yield (blank, [0])
    
    def hex_grid(self):
        """Creates an 2D array representing hexagonal grid based on the parameters

        Algorithm:
            First, it creates a tide containing two lines looking like: ___/
                the height of the tide is size/2
                the width of the tide is sqrt(3)/2*size
                NOTE: one of the parameters is not an integer -> therefore rounding
                    is present and for big grids it can be off by several pixels
                    the oddness can be derived from the ratio of width and height
                    the more close to sqrt(3) the better

            Second, it replicates the the tide to create hexagonal tiding of the 
            following form              ___ 
                                    ___/   \
                                       \___/
            Third, it replicates the hexagonal tide and rotates it

            Fourth, it computes shifts based on parameters size, size_x, size_y
                    and cuts out the relevant part of the array

        Returns:
            array with values 1 or 0, 0 representing the hexagonal grid
        """
        # imagen is used to create the slant line /
        ln = imagen.Line(bounds = BoundingBox(radius=self.size/4.), 
                    orientation = pi/3, smoothing=0,
                    xdensity = self.density, ydensity = self.density,  
                    thickness=1./self.density)()
        # cutting the relevant part of the created line
        idx = ln.argmax(axis=0).argmax()
        line = ln[:,idx:-idx]
        # Creating the horizontal line _
        hline = numpy.zeros((line.shape[0], line.shape[1]*2))
        hline[-1,:] = 1
        # Creating hexagonal tide
        tide = numpy.hstack((hline,line))  # joins horizontal line and slant line
        tide = numpy.hstack((tide, tide[::-1,:]))  # mirrors the tide in horizontal direction
        tide = numpy.vstack((tide, tide[::-1,:]))  # mirrors the tide in vertical direction
        d, k = tide.shape
        k = k/3
        # Creating hex
        x_reps = int(self.size_x/self.size) + 2
        y_reps = int(self.size_y/self.size) + 2
        # pixel sizes
        x_size = int(self.size_x*self.density)
        y_size = int(self.size_y*self.density)
        grid = numpy.hstack((numpy.vstack((tide,)*y_reps),)*x_reps)
        # starting indices in each dimension
        i = int((0.5+int(self.size_y/self.size)*1.5)*k)-int(self.density*self.size_y/2)
        j = d - int(self.size_x%self.size*self.density/2.)
        grid = grid[j:j+x_size,i:i+y_size]
        center = grid.shape[0]/2
        return 1-grid.T


class ContinuousGaborMovementAndJump(TopographicaBasedVisualStimulus):
    """
    Continuously move a Gabor patch towards a specified center position, then jump into
    the center position once the moving gabor would overlap with the Gabor patch in the
    center position. The size*, *phase*, *spatial_frequency* of the moving and center
    patches are the same, their relative luminances can be specified separately. The
    orientation of the center patch is set by *orientation*; the orientation of the
    moving Gabor can be radial or tangential (set by *moving_gabor_orientation_radial*)

    The speed of movement is simply given by *movement_length* / *movement_duration*.
    The continuous movement is made up of *movement_duration* / *frame_duration* Gabor
    positions, evenly spaced along the movement line.

    After the movement and flash are finished, until the *duration* of the
    stimulus, constant *background_luminance* is displayed.
    """

    orientation = SNumber(rad, period=pi, bounds=[0,pi], doc="Gabor patch orientation")
    phase = SNumber(rad, period=2*pi, bounds=[0,2*pi], doc="Gabor patch phase")
    spatial_frequency = SNumber(cpd, doc="Spatial frequency of the grating")
    size = SNumber(degrees, doc="Size of the Gabor patch")
    center_relative_luminance = SNumber(dimensionless,bounds=[0,1.0], doc="The scale of the center stimulus. 0 is dark, 1.0 is double the background luminance")
    moving_relative_luminance = SNumber(dimensionless,bounds=[0,1.0], doc="The scale of the moving stimulus. 0 is dark, 1.0 is double the background luminance")
    x = SNumber(degrees, doc="x coordinate of center patch")
    y = SNumber(degrees, doc="y coordinate of center patch")
    movement_duration = SNumber(ms, doc="Duration of the Gabor patch movement.")
    movement_length = SNumber(degrees, bounds=[0,np.inf], doc="Length of the Gabor patch movement")
    movement_angle = SNumber(rad, period=2*pi, bounds=[0,2*pi], doc="Incidence angle of the moving patch to the center patch.")
    moving_gabor_orientation_radial = SNumber(dimensionless, doc = "Boolean string, radial or cross patch")
    center_flash_duration = SNumber(ms, doc="Duration of flashing the Gabor patch in the center.")

    def getGabor(self,x,y,orientation,phase,spatial_frequency,size,background_luminance, relative_luminance):
        disk_width_sd = 2.5 # how many standard deviations(of Gabor Gaussian)
                            # wide should the disk (given by "size") be
        gabor = imagen.Gabor(
                    aspect_ratio = 1, # Ratio of pattern width to height.
                                      # Set since the patch has to be round
                    mask_shape=imagen.Disk(smoothing=0, size=disk_width_sd),
                        # Gabor patch should fit inside tide/circle
                        # the size is rescalled according to the size
                        # of Gabor patch
                    frequency = spatial_frequency,
                    phase = phase, # Initial phase of the sinusoid
                    bounds = BoundingBox(radius=self.size_x/2.0),
                        # BoundingBox of the area in which the pattern is
                        # generated, radius=1: box with side length 2!
                    size = size / disk_width_sd, # size = 2*standard_deviation
                        # => on the radius r=size, the intensity is ~0.14
                    orientation=orientation, # In radians
                    x = x,  # x-coordinate of Gabor patch center
                    y = y,  # y-coordinate of Gabor patch center
                    xdensity=self.density, # Number of points in one unit
                                           # of length in x direction
                    ydensity=self.density, # Number of points in one unit
                                           # of length in y direction
                    scale=2.0*background_luminance*relative_luminance
                            # Difference between maximal and minimal value
                            # => min value = -scale/2, max value = scale/2
                            )()

        gabor = gabor+self.background_luminance
        return gabor
 
    def frames(self):
        assert self.movement_duration >= 2*self.frame_duration, "Movement must be at least 2 frames long"
        assert self.center_flash_duration >= self.frame_duration, "Flash in center must be at least 1 frame long"

        x_start = self.x + (self.size + self.movement_length) * np.cos(self.movement_angle)
        y_start = self.y + (self.size + self.movement_length) * np.sin(self.movement_angle)
        x_end = self.x + self.size * np.cos(self.movement_angle)
        y_end = self.y + self.size * np.sin(self.movement_angle)

        n_pos = int(self.movement_duration / self.frame_duration)
        x_pos = np.linspace(x_start,x_end,n_pos)
        y_pos = np.linspace(y_start,y_end,n_pos)

        blank = imagen.Constant(scale=self.background_luminance,
                                bounds=BoundingBox(radius=self.size_x/2),
                                xdensity=self.density,
                                ydensity=self.density)()

        for x,y in zip(x_pos,y_pos):
            angle = self.movement_angle if self.moving_gabor_orientation_radial else self.movement_angle + np.pi/2
            yield (self.getGabor(x,y,angle,self.phase,self.spatial_frequency,self.size,self.background_luminance, self.moving_relative_luminance),[1])

        for i in xrange(int(self.center_flash_duration / self.frame_duration)):
            yield (self.getGabor(self.x,self.y,self.orientation,self.phase,self.spatial_frequency,self.size,self.background_luminance, self.center_relative_luminance),[1])

        while True:
            yield (blank, [0])
