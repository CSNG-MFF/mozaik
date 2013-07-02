"""
This modules implements the API for input space.
"""

import os.path
import numpy
import mozaik
from PIL import Image
from parameters import ParameterSet
from mozaik.core import ParametrizedObject

from mozaik import __version__

TRANSPARENT = -1
logger = mozaik.getMozaikLogger()


def xy2ij(coordinates):
    assert len(coordinates) == 2
    return numpy.array([coordinates[1], coordinates[0]], float)


class InputSpace(ParametrizedObject):
    """
    A class to structure and unify operations taking place in the respective sensory space, such as stimulus presentation.
    
    The basic idea of the InputSpace API is following:
    
    The InputSpace assumes there is a scene, and a set of stimuli in this scene (visual objects, sounds, smells etc.).
    These objects can be removed or added to the scene. After each interval lasting  `update_interval` miliseconds all the stimuli in the scene
    are sequentially updated (following some rules of overlapping, this is left for the specific implementation of the input space).
    After that update the scene is ready to be pased to the associated input sheet of the given model, which will use it to 
    generate the responses of neurons in the input sheet.
    
    Other parameters
    ----------------
    
    update_interval : float (ms)
                    How often does the input space update.
    """
    
    required_parameters = ParameterSet({
        'update_interval': float  # [ms] how fast the input is changed
        })

    def __init__(self, params):
        ParametrizedObject.__init__(self, params)
        self.content = {}
        self.input = None

    def add_object(self, name, input_object):  
        """Add an inputObject to the input scene."""
        logger.debug("Adding %s with name '%s' to the input scene." % (input_object, name))
        self.content[name] = input_object
        self.input = input_object  # really self.input should be a list, and we should append to it, but at the moment NeuroTools.datastore can't handle multiple inputs to a component

    def reset(self):
        """
        Reset each Object in the scene to its initial state.
        """
        self.frame_number = 0
        for obj in self.content.values():
            obj.reset()

    def clear(self):
        """
        Reset the input space and clear stimuli in it.
        """
        self.content = {}
        self.input = None
        self.reset()

    def update(self):
        """
        Tell each Object in the scene to update itself.
        Returns the current time within the scene.
        """
        for obj in self.content.values():
            obj.update()
        self.frame_number += 1

        return self.frame_number * self.parameters.update_interval

    def get_maximum_duration(self):
        """
        The maximum duration of any of the stimuli in the inpust space.
        """
        duration = 0
        for obj in self.content.values():
            duration = max(duration, self.update_interval*obj.n_frames)
        return duration

    def get_duration(self):
        """
        Get the duration of the stimulation in the input space.
        """
        return self.parameters['duration']

    def set_duration(self, duration):
        """
        Set the duration of the stimulation in the input space.
        """
        assert duration <= self.get_maximum_duration()
        self.parameters['duration'] = duration

    def time_points(self, duration=None):
        """
        Returns the time points of updates in the period 0,duration.
        """
        duration = duration or self.get_maximum_duration()
        return numpy.arange(0, duration, self.update_interval)


class VisualSpace(InputSpace):
    """
    A class to structure and simplify operations taking place in visual
    space, such as stimulus presentation.
    
    In VisualSpace the stimuli are sequentially drawn into the scene in the order in which
    thay have been added to the scene (thus later stimuli will draw over earlier added ones).
    Stimuli can specify areas in which they are transparent by using the `TRANSPARENT` value.
    
    The key operation of VisualSpace is the :func:`.view` function that recieves a region and resolution 
    and returns rendered scene as a 2D array, within that region and down-sampled to the specified 
    resolution. This allows for implementation of a visual system that changes the viewed area of the scene.
    The :func:`.view` function itself makes sequential calls to the :func:`mozaik.visual_stimulus.VisualStimulus.display` 
    function passing the region and pixel size, which have to render themselves within the region and 
    return 2D array of luminance values. :func:`.view` then assambles the final scene rendering from this data.
    
    Notes
    -----
    
    For now, we deal only with two-dimensions, i.e. everything projected onto a
    plane. We ignore distortions in going from a flat plane to the curved retina.
    Could consider using matplotlib.transforms for some of this.
    """

    version = __version__

    required_parameters = ParameterSet({
        'background_luminance': float,
        })

    def __init__(self, params):
        InputSpace.__init__(self, params)  # InputSpace requires only the update interval
        self.background_luminance = self.parameters.background_luminance
        self.update_interval = self.parameters.update_interval
        self.content = {}
        self.frame_number = 0
        self.input = None

    def view(self, region, pixel_size):
        """
        Show the scene within a specific region.
        
        Parameters
        ----------
        region : VisualRegion
               Should be a VisualRegion object.
        pixel_size : float (degrees)
                   The size of a single pixel in degrees of visual field.

        Returns
        -------
                array : nd_array 
                       A numpy 2D array containing luminance values, corresponding to 
                       to the visual scene in the visual region specified in `region` 
                       downsample such that one pixel has `pixel_size` degree.
        """
        # Let's make it more efficient if there is only one object in the scene that is not transparrent (which is often the case):
        o = self.content.values()[0]
        if len(self.content.values()) == 1 and not o.transparent and o.is_visible:
           return o.display(region, pixel_size)

        size_in_pixels = numpy.ceil(xy2ij((region.size_x, region.size_y)) / float(pixel_size)).astype(int)
        scene = TRANSPARENT*numpy.ones(size_in_pixels)
        for obj in self.content.values():
            if obj.is_visible:
                if region.overlaps(obj.region):
                    obj_view = obj.display(region, pixel_size)
                    try:
                        scene = numpy.where(obj_view > scene, obj_view, scene)  # later objects overlay earlier ones with no transparency
                    except ValueError:
                        logger.error("Array dimensions mismatch. obj_view.shape=%s, scene.shape=%s" % (obj_view.shape, scene.shape))
                        logger.error("  region: %s" % region.describe())
                        logger.error("  visual object: %s" % obj.describe())
                        raise
                else:
                    #logger.debug("Warning: region %s does not overlap this object (%s)." % (region.describe(), obj.describe()))
                    pass
        return numpy.where(scene > TRANSPARENT, scene, self.background_luminance)

    def get_max_luminance(self):
        """
        Returns the maximum luminance in the scene.
        """
        return max(obj.max_luminance for obj in self.content.values())

    def export(self, region, pixel_size, output_dir="."):
        """
        Export a sequence of views of the visual space as image files.
        Return a list of file paths.
        """
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self.reset()
        assert self.get_duration() % self.update_interval == 0
        num_frames = int(self.get_duration()/self.update_interval)
        filename_fmt = os.path.join(output_dir, "frame%%0%dd.png" % len(str(num_frames)))
        filenames = []
        for i in range(num_frames):
            scene = self.view(region, pixel_size) * 255 / self.get_max_luminance()
            scene = scene.astype('uint8')
            rgb_scene = numpy.array((scene, scene, scene)).transpose(1, 2, 0)
            img = Image.fromarray(rgb_scene, 'RGB')
            img.save(filename_fmt % i)
            filenames.append(filename_fmt % i)
            self.update()
        return filenames

    def describe(self):
        return "visual space with background luminance %g cd/m2, updating every %g ms, containing %d objects" % \
            (self.background_luminance, self.update_interval, len(self.content))


class VisualRegion(object):
    """
    A rectangular region of visual space.
    
    Parameters
    ----------
    location_x : float (degrees)
               The x coordinate of the center of the region in the visual space. 
            
    location_y : float (degrees)
               The y coordinate of the center of the region in the visual space. 

    size_x : float (degrees)
               The x size of the region in the visual space. 

    size_y : float (degrees)
               The y size of the region in the visual space. 
    """

    def __init__(self, location_x, location_y, size_x, size_y):

        self.location_x = location_x
        self.location_y = location_y
        self.size_x = size_x
        self.size_y = size_y

        assert self.size_x > 0 and self.size_y > 0

        half_width = self.size_x/2.0
        half_height = self.size_y/2.0
        self.left = self.location_x - half_width
        self.right = self.location_x + half_width
        self.top = self.location_y + half_height
        self.bottom = self.location_y - half_height
        self.width = self.right - self.left
        self.height = self.top - self.bottom

    def __eq__(self, other):
        return (self.location_x == other.location_x
                  and self.location_y == other.location_y
                  and self.size_x == other.size_x
                  and self.size_y == other.size_y)

    def __ne__(self, other):
        return not self.__eq__(other)

    def overlaps(self, another_region):
        """
        Returns whether this region overlaps with the one in the `another_region` argument.
        """

        lr = self.right <= another_region.left or self.left >= another_region.right
        tb = self.top <= another_region.bottom or self.bottom >= another_region.top
        return not(lr or tb)

    def intersection(self, another_region):
        """
        Returns VisualRegion corresponding to the intersection of this VisualRegion and the one in the `another_region` argument.
        """
        if not self.overlaps(another_region):
            raise Exception("Regions do not overlap.")
        left = max(self.left, another_region.left)
        right = min(self.right, another_region.right)
        assert left <= right, "self: %s\nanother_region: %s" % (self.describe(), another_region.describe())
        bottom = max(self.bottom, another_region.bottom)
        top = min(self.top, another_region.top)
        assert bottom <= top
        return VisualRegion(location_x=(left + right)/2.0,
                            location_y=(top + bottom)/2.0,
                            size_x=right - left,
                            size_y=top - bottom)

    def describe(self):
        s = """Region of visual space centred at (%(location_x),%(location_y)) s of size (%(size_x),%(size_y))s.
               Edges: left=%(left)g, right=%(right)g, top=%(top)g, bottom=%(bottom)g""" % self.__dict__
        return s


the_final_frontier = True
