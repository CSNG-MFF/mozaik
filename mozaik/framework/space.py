"""
Module for dealing with visual space.

For now, we deal only with two-dimensions, i.e. everything projected onto a
plane. We ignore distortions in going from a flat plane to the curved retina.

Could consider using matplotlib.transforms for some of this.

"""

import os.path
import numpy
import logging
from scipy.ndimage import interpolation
from PIL import Image
from NeuroTools.parameters import ParameterSet
from mozaik.framework.interfaces import MozaikParametrizeObject

from mozaik import __version__

TRANSPARENT = -1
logger = logging.getLogger("mozaik")


def xy2ij(coordinates):
    assert len(coordinates) == 2
    return numpy.array([coordinates[1], coordinates[0]], float)



class InputSpace(MozaikParametrizeObject):
    """
    A class to structure and simplify operations taking place in the respective sensory
    space, such as stimulus presentation.
    """
    required_parameters = ParameterSet({
        'update_interval' : float # [ms] how fast the input is changed
        })

    def __init__(self, params):
        MozaikParametrizeObject.__init__(self, params)
        self.content = {}
        self.input = None

    def add_object(self, name, input_object): # <-- previously: add_object
        """Add a inputObject to the input scene."""
        logger.info("Adding %s with name '%s' to the input scene." % (input_object, name))
        self.content[name] = input_object
        self.input = input_object # really self.input should be a list, and we should append to it, but at the moment NeuroTools.datastore can't handle multiple inputs to a component

    def reset(self):
        """
        Reset each VisualObject in the scene to its initial state.
        """
        self.frame_number = 0
        for obj in self.content.values():
            obj.reset()

    def clear(self):
        """
        Reset the visual space and clear stimuli in it 
        """
        self.content = {}
        self.input = None
        self.reset()

    def update(self):
        """
        Tell each VisualObject in the scene to update itself.
        Returns the current time within the scene.
        """
        for obj in self.content.values():
            obj.update()
        self.frame_number += 1

        return self.frame_number * self.parameters.update_interval

    def get_maximum_duration(self):
        duration = 0
        for obj in self.content.values():
            duration = max(duration, self.update_interval*obj.n_frames)
        return duration

    def get_duration(self):
        return self.parameters['duration']

    def set_duration(self, duration):
        assert duration <= self.get_maximum_duration()
        self.parameters['duration'] = duration

    def time_points(self, duration=None):
        duration = duration or self.get_maximum_duration()
        return numpy.arange(0, duration, self.update_interval)


class VisualSpace(InputSpace):
    """
    A class to structure and simplify operations taking place in visual
    space, such as stimulus presentation.
    """

    version = __version__

    required_parameters = ParameterSet({
        'background_luminance': float,
        })
    def __init__(self, params):
        InputSpace.__init__(self, params) # InputSpace requires only the update interval
        self.background_luminance = self.parameters.background_luminance
        self.update_interval = self.parameters.update_interval
        self.size_in_degrees = 0.1
        self.size_in_pixels = 1 # remove this? the viewing component should choose the pixel density it wants, and we use interpolation
        self.content = {}
        self.frame_number = 0
        self.input = None
#        return self.frame_number * self.update_interval 

    def view(self, region, pixel_size):
        """
        Show the scene within a specific region.
        
        `region` should be a VisualRegion object.
        `pixel_size` should be in degrees.
        
        Returns a numpy array containing luminance values.
        """
        size_in_pixels = numpy.ceil(xy2ij((region.size_x,region.size_y))/float(pixel_size)).astype(int)
        scene = TRANSPARENT*numpy.ones(size_in_pixels)

        for obj in self.content.values():
            if obj.is_visible:
                if region.overlaps(obj):
                    obj_view = obj.display(region, pixel_size)
                    try:
                        scene = numpy.where(obj_view > scene, obj_view, scene) # later objects overlay earlier ones with no transparency
                    except ValueError:
                        logger.error("Array dimensions mismatch. obj_view.shape=%s, scene.shape=%s" % (obj_view.shape, scene.shape))
                        logger.error("  region: %s" % region.describe())
                        logger.error("  visual object: %s" % obj.describe())
                        raise
                else:
                    #logger.debug("Warning: region %s does not overlap this object (%s)." % (region.describe(), obj.describe()))
                    pass
        return numpy.where(scene != TRANSPARENT, scene, self.background_luminance)

    def get_max_luminance(self):
        return max(obj.max_luminance for obj in self.content.values())

    def export(self, region, pixel_size, output_dir="."):
        """
        Export a sequence of views of the visual space as image files.
        Return a list of file paths.
        """
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self.reset()
        assert self.get_duration()%self.update_interval == 0
        num_frames = int(self.get_duration()/self.update_interval)
        filename_fmt = os.path.join(output_dir, "frame%%0%dd.png" % len(str(num_frames)))
        filenames = []
        for i in range(num_frames):
            scene = self.view(region, pixel_size)*255/self.get_max_luminance()
            scene = scene.astype('uint8')
            rgb_scene = numpy.array((scene, scene, scene)).transpose(1,2,0)
            img = Image.fromarray(rgb_scene, 'RGB')
            img.save(filename_fmt % i)
            filenames.append(filename_fmt % i)
            self.update()
        return filenames

    def describe(self):
        return "visual space with background luminance %g cd/m2, updating every %g ms, containing %d objects" % \
            (self.background_luminance, self.update_interval, len(self.content))

the_final_frontier = True   # <--- What's beyond? 
