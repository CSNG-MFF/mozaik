"""
The file contains stimuli that use topographica to generate the stimulus

"""

from visual_stimulus import VisualStimulus
import topo.pattern
from topo.base.boundingregion import BoundingBox
import pickle
import numpy
from mozaik.tools.mozaik_parametrized import *
from mozaik.tools.units import cpd
from numpy import pi
from quantities import Hz, rad, degrees, ms


class FullfieldDriftingSinusoidalGrating(VisualStimulus):
    """
    max_luminance is interpreted as scale
    and size_x/2 as the bounding box radius
    """

    orientation = SNumber(rad, period=pi, doc="Grating orientation")
    spatial_frequency = SNumber(cpd, doc="Spatial frequency of grating")
    temporal_frequency = SNumber(Hz, doc="Temporal frequency of grating")

    def frames(self):
        self.current_phase=0
        while True:
            yield (topo.pattern.SineGrating(orientation=self.orientation,
                                            frequency=self.spatial_frequency,
                                            phase=self.current_phase,
                                            bounds=BoundingBox(radius=self.size_x/2),
                                            scale=self.max_luminance,
                                            xdensity=self.density,
                                            ydensity=self.density)(),
                   [self.current_phase])
            self.current_phase += 2*pi * (self.frame_duration/1000.0) * self.temporal_frequency


class Null(VisualStimulus):
    def frames(self):
        """
        Empty stimulus
        """
        while True:
            yield topo.pattern.Null(scale=0, bounds=BoundingBox(radius=self.size_x/2))(), []


class NaturalImageWithEyeMovement(VisualStimulus):
    """
    A visual stimulus that simulates an eye movement over a static image
    """
    size = SNumber(degrees, doc="The length of the longer axis of the image in visual degrees")
    eye_movement_period = SNumber(ms, doc="The time between two consequitve eye movements recorded in the eye_path file")
    image_location = SString(doc="Location of the image")
    eye_path_location = SString(doc="Location of file containing the eye path (two columns of numbers)")

    def frames(self):
        self.time = 0
        from topo.transferfn import DivisiveNormalizeLinf
        import topo.pattern.image

        f = open(self.eye_path_location, 'r')
        self.eye_path = pickle.load(f)
        self.pattern_sampler = topo.pattern.image.PatternSampler(
                                           size_normalization='fit_longest',
                                           whole_pattern_output_fns=[DivisiveNormalizeLinf()])

        while True:
            location = self.eye_path[int(numpy.floor(self.frame_duration * self.time / self.eye_movement_period))]
            image = topo.pattern.image.FileImage(
                                         filename=self.image_location,
                                         x=location[0],
                                         y=location[1],
                                         orientation=0,
                                         xdensity=self.density,
                                         ydensity=self.density,
                                         size=self.size,
                                         bounds=BoundingBox(points=((-self.size_x/2, -self.size_y/2),
                                                                    (self.size_x/2, self.size_y/2))),
                                         scale=self.max_luminance,
                                         pattern_sampler=self.pattern_sampler
                                         )()
            yield (image, [self.time])
            self.time += 1


class DriftingGratingWithEyeMovement(VisualStimulus):
    """
    A visual stimulus that simulates an eye movement over a drifting  gratings
    """

    orientation = SNumber(rad, period=pi, doc="Grating orientation")
    spatial_frequency = SNumber(cpd, doc="Spatial frequency of grating")
    temporal_frequency = SNumber(Hz, doc="Temporal frequency of grating")
    eye_movement_period = SNumber(ms, doc="The time between two consequitve eye movements recorded in the eye_path file")
    eye_path_location = SString(doc="Location of file containing the eye path (two columns of numbers)")

    def frames(self):
        f = open(self.eye_path_location, 'r')
        self.eye_path = pickle.load(f)
        self.pattern_sampler = topo.pattern.image.PatternSampler(
                                        size_normalization='fit_longest',
                                        whole_pattern_output_fns=[DivisiveNormalizeLinf()])
        self.time = 0
        self.current_phase = 0
        from topo.transferfn.basic import DivisiveNormalizeLinf
        import topo.pattern.image

        while True:
            location = self.eye_path[int(numpy.floor(self.frame_duration * self.time / self.eye_movement_period))]

            image = topo.pattern.SineGrating(orientation=self.orientation,
                                             x=location[0],
                                             y=location[1],
                                             frequency=self.spatial_frequency,
                                             phase=self.current_phase,
                                             bounds=BoundingBox(points=((-self.size_x/2, -self.size_x/2),
                                                                        (-self.size_y/2, -self.size_y/2))),
                                             scale=self.max_luminance,
                                             xdensity=self.density,
                                             ydensity=self.density)()
            self.time = self.time + 1
            self.current_phase += 2*pi * (self.frame_duration/1000.0) * self.temporal_frequency
            yield (image, [self.time])


class DriftingSinusoidalGratingDisk(VisualStimulus):
    """
    max_luminance is interpreted as scale
    and size_x/2 as the bounding box radius
    """

    orientation = SNumber(rad, period=pi, doc="Grating orientation")
    spatial_frequency = SNumber(cpd, doc="Spatial frequency of the grating")
    temporal_frequency = SNumber(Hz, doc="Temporal frequency of the grating")
    radius = SNumber(degrees, doc="The radius of the grating disk - in degrees of visual field")

    def frames(self):
        self.current_phase=0
        while True:
            yield (topo.pattern.SineGrating(mask_shape=topo.pattern.Disk(smoothing=0.0, size=self.radius*2),
                                            orientation=self.orientation,
                                            frequency=self.spatial_frequency,
                                            phase=self.current_phase,
                                            bounds=BoundingBox(radius=self.size_x/2),
                                            scale=self.max_luminance,
                                            xdensity=self.density,
                                            ydensity=self.density)(),
                   [self.current_phase])
            self.current_phase += 2*pi * (self.frame_duration/1000.0) * self.temporal_frequency


class DriftingSinusoidalGratingCenterSurroundStimulus(VisualStimulus):
    """
    A standard stimulus to probe orientation specific surround modulation:
    A drifting grating in center surrounded by a drifting grating in the surround.
    Orientations of both center and surround gratings can be varied independently.

    max_luminance is interpreted as scale and size_x/2 as the bounding box radius.
    """

    center_orientation = SNumber(rad, period=pi, doc="Center grating orientation")
    surr_orientation = SNumber(rad, period=pi, doc="Surround grating orientation")
    spatial_frequency = SNumber(cpd, doc="Spatial frequency of the grating (same for center and surround)")
    temporal_frequency = SNumber(Hz, doc="Temporal frequency of the grating (same for center and surround)")
    gap = SNumber(degrees, doc="The gap between center and surround grating - in degrees of visual field")
    center_radius = SNumber(degrees, doc="The (outside) radius of the center grating disk - in degrees of visual field")
    surround_radius = SNumber(degrees, doc="The (outside) radius of the surround grating disk - in degrees of visual field")

    def frames(self):
        self.current_phase = 0
        while True:
            center = topo.pattern.SineGrating(mask_shape=topo.pattern.Disk(smoothing=0.0, size=self.center_radius*2),
                                              orientation=self.center_orientation,
                                              frequency=self.spatial_frequency,
                                              phase=self.current_phase,
                                              bounds=BoundingBox(radius=self.size_x/2),
                                              scale=self.max_luminance,
                                              xdensity=self.density,
                                              ydensity=self.density)()
            r = (self.center_radius + self.surround_radius + self.gap)/2
            t = (self.surround_radius - self.surround_radius - self.gap)/2
            surround = topo.pattern.SineGrating(mask_shape=topo.pattern.Ring(thickness=t, smoothing=0, size=r*2),
                                                orientation=self.surround_orientation,
                                                frequency=self.spatial_frequency,
                                                phase=self.current_phase,
                                                bounds=BoundingBox(radius=self.size_x/2),
                                                scale=self.max_luminance,
                                                xdensity=self.density,
                                                ydensity=self.density)()
            yield (numpy.add.reduce([center, surround]), [self.current_phase])
            self.current_phase += 2*pi * (self.frame_duration/1000.0) * self.temporal_frequency
