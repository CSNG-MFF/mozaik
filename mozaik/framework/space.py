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
from mozaik.tools.mozaik_parametrized import SNumber, MozaikParametrized
import quantities as qt
from NeuroTools.parameters import ParameterSet

import quantities as qt
from mozaik import __version__

TRANSPARENT = -1
logger = logging.getLogger("mozaik")


def xy2ij(coordinates):
    assert len(coordinates) == 2
    return numpy.array([coordinates[1], coordinates[0]], float)

class VisualRegion(MozaikParametrized):
    """A rectangular region of visual space."""
    location_x = SNumber(qt.degrees,doc="""x location of the center of  visual region.""")
    location_y = SNumber(qt.degrees,doc="""y location of the center of  visual region.""")
    size_x = SNumber(qt.degrees,doc="""The size of the region in degrees (asimuth).""")
    size_y = SNumber(qt.degrees,doc="""The size of the region in degrees (elevation).""")
    
    def __init__(self,**params):
        MozaikParametrized.__init__(self,**params)
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
        return self.get_param_values() == other.get_param_values()
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def overlaps(self, another_region):                    
        lr = self.right <= another_region.left or self.left >= another_region.right
        tb = self.top <= another_region.bottom or self.bottom >= another_region.top
        return not(lr or tb)
    
    def intersection(self, another_region):
        if not self.overlaps(another_region):
            raise Exception("Regions do not overlap.")
        left = max(self.left, another_region.left)
        right = min(self.right, another_region.right)
        assert left <= right, "self: %s\nanother_region: %s" % (self.describe(), another_region.describe())
        bottom = max(self.bottom, another_region.bottom)
        top = min(self.top, another_region.top)
        assert bottom <= top
        centre = ((left + right)/2.0, (top + bottom)/2.0)
        size = (right-left, top-bottom)
        return VisualRegion(centre, size)
    
    def describe(self):
        s = """Region of visual space centred at (%(location_x),%(location_y)) s of size (%(size_x),%(size_y))s.
               Edges: left=%(left)g, right=%(right)g, top=%(top)g, bottom=%(bottom)g""" % self.__dict__
        return s

class VisualObject(VisualRegion):
    """Abstract base class."""
    
    def __init__(self,**params):
        VisualRegion.__init__(self,**params)
        self._zoom_cache = {}
        self.is_visible = True
        
    def update(self):
        raise NotImplementedException("Must be implemented by child class")

    def _calculate_zoom(self, actual_pixel_size, desired_pixel_size):
        """
        Sometimes the interpolation procedure returns a new array that is too
        small due to rounding error. This is a crude attempt to work around that.
        """
        
        zoom = actual_pixel_size/desired_pixel_size
        for i in self.img.shape:
            if int(zoom*i) != round(zoom*i):
                zoom *= (1 + 1e-15)
        return zoom

    def display(self, region, pixel_size):
        assert isinstance(region, VisualRegion), "region must be a VisualRegion-descended object. Actually a %s" % type(region)
        size_in_pixels = numpy.ceil(xy2ij(region.size)/float(pixel_size)).astype(int)
        view = TRANSPARENT*numpy.ones(size_in_pixels)
        if region.overlaps(self):
            intersection = region.intersection(self)
            assert intersection == self.intersection(region) # just a consistency check. Could be removed if necessary for performance.
            #logger.debug("this object: %s" % self.describe())
            #logger.debug("region: %s" % region.describe())
            #logger.debug("intersection: %s" % intersection.describe())
            # convert coordinates in degrees to array index values
            img_relative_left = (intersection.left - self.left)/self.width
            ##img_relative_right = (intersection.right - self.left)/self.width
            img_relative_width = intersection.width/self.width
            img_relative_top = (intersection.top - self.bottom)/self.height ##
            img_relative_bottom = (intersection.bottom - self.bottom)/self.height
            img_relative_height = intersection.height/self.height 
            view_relative_left = (intersection.left - region.left)/region.width
            ##view_relative_right = (intersection.right - region.left)/region.width
            view_relative_width = intersection.width/region.width
            view_relative_top = (intersection.top - region.bottom)/region.height ##
            view_relative_bottom = (intersection.bottom - region.bottom)/region.height
            view_relative_height = intersection.height/region.height
            
            img_pixel_size = xy2ij(self.size)/self.img.shape # is self.size a tuple or an array?
            #logger.debug("img_pixel_size = %s" % str(img_pixel_size))
            #logger.debug("img relative_left, _right, _top, _bottom = %g,%g,%g,%g" % (img_relative_left, img_relative_right, img_relative_top, img_relative_bottom))
            #logger.debug("view relative_left, _right, _top, _bottom = %g,%g,%g,%g" % (view_relative_left, view_relative_right, view_relative_top, view_relative_bottom))
            #logger.debug("self.img.shape = %s" % str(self.img.shape))
            assert img_pixel_size[0] == img_pixel_size[1]
            
            if pixel_size == img_pixel_size[0]:
                #logger.debug("Image pixel size matches desired size (%g degrees)" % pixel_size)
                img = self.img
            else:
                # note that if the image is much larger than the view region, we might save some
                # time by not rescaling the whole image, only the part within the view region.
                zoom = self._calculate_zoom(img_pixel_size[0], pixel_size) #img_pixel_size[0]/pixel_size
                #logger.debug("Image pixel size (%g deg) does not match desired size (%g deg). Zooming image by a factor %g" % (img_pixel_size[0], pixel_size, zoom))
                if zoom in self._zoom_cache:
                    img = self._zoom_cache[zoom]
                else:                  
                    img = interpolation.zoom(self.img, zoom)
                    self._zoom_cache[zoom] = img
                
            j_start = numpy.round(img_relative_left * img.shape[1]).astype(int)
            delta_j = numpy.round(img_relative_width * img.shape[1]).astype(int)
            i_start = img.shape[0] - numpy.round(img_relative_top * img.shape[0]).astype(int)
            delta_i = numpy.round(img_relative_height * img.shape[0]).astype(int)
            
            l_start = numpy.round(view_relative_left * size_in_pixels[1]).astype(int) 
            delta_l = numpy.round(view_relative_width * size_in_pixels[1]).astype(int)
            k_start = size_in_pixels[0] - numpy.round(view_relative_top * size_in_pixels[0]).astype(int) 
            delta_k = numpy.round(view_relative_height * size_in_pixels[0]).astype(int) 
            
            assert delta_j == delta_l, "delta_j = %g, delta_l = %g" % (delta_j, delta_l)
            assert delta_i == delta_k
            
            i_stop = i_start + delta_i
            j_stop = j_start + delta_j
            k_stop = k_start + delta_k
            l_stop = l_start + delta_l
            ##logger.debug("i_start = %d, i_stop = %d, j_start = %d, j_stop = %d" % (i_start, i_stop, j_start, j_stop))
            ##logger.debug("k_start = %d, k_stop = %d, l_start = %d, l_stop = %d" % (k_start, k_stop, l_start, l_stop))

            
            try:
                view[k_start:k_start+delta_k, l_start:l_start+delta_l] = img[i_start:i_start+delta_i, j_start:j_start+delta_j]
            except ValueError, errmsg:
                logger.error("i_start = %d, i_stop = %d, j_start = %d, j_stop = %d" % (i_start, i_stop, j_start, j_stop))
                logger.error("k_start = %d, k_stop = %d, l_start = %d, l_stop = %d" % (k_start, k_stop, l_start, l_stop))
                logger.error("img.shape = %s, view.shape = %s" % (img.shape, view.shape))
                logger.error("img[i_start:i_stop, j_start:j_stop].shape = %s" % str(img[i_start:i_stop, j_start:j_stop].shape))
                logger.error("view[k_start:k_stop, l_start:l_stop].shape = %s" % str(view[k_start:k_stop, l_start:l_stop].shape))
                raise
        assert view.shape == tuple(size_in_pixels), "view.shape = %s, should be %s" % (view.shape, tuple(size_in_pixels))
        return view


class VisualSpace(object):
    """
    A class to structure and simplify operations taking place in visual
    space, such as stimulus presentation.
    """
    
    version = __version__
    
    def __init__(self, update_interval, background_luminance=0.0):
        self.update_interval = update_interval
        self.background_luminance = background_luminance
        self.parameters = ParameterSet({
                            'update_interval': update_interval,
                            'background_luminance': background_luminance
                          })
        self.size_in_degrees = 0.1
        self.size_in_pixels = 1 # remove this? the viewing component should choose the pixel density it wants, and we use interpolation
        self.content = {}
        self.frame_number = 0
        self.input = None
    
    def add_object(self, name, visual_object):
        """Add a VisualObject to the visual scene."""
        logger.info("Adding %s with name '%s' to the visual scene." % (visual_object, name))
        self.content[name] = visual_object
        self.input = visual_object # really self.input should be a list, and we should append to it, but at the moment NeuroTools.datastore can't handle multiple inputs to a component
        # update size_in_X, or make them properties that are recalculated each time
        
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
        
        return self.frame_number * self.update_interval
    
    def view(self, region, pixel_size):
        """
        Show the scene within a specific region.
        
        `region` should be a VisualRegion object.
        `pixel_size` should be in degrees.
        
        Returns a numpy array containing luminance values.
        """
        size_in_pixels = numpy.ceil(xy2ij(region.size)/float(pixel_size)).astype(int)
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
        
the_final_frontier = True
