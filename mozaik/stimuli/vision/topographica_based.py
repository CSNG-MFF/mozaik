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
        assert (self.time_per_image/self.frame_duration) % 1.0 == 0.0
                
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
            for i in range(self.time_per_image/self.frame_duration):
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
