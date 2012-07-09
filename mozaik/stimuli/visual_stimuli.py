import numpy
from operator import *
import quantities as qt
from mozaik.stimuli.stimulus import Stimulus
from mozaik.framework.space import TRANSPARENT, xy2ij
from mozaik.tools.mozaik_parametrized import SNumber, SInteger, SString
from mozaik import __version__
from mozaik.tools.units import lux
from mozaik.framework.interfaces import MozaikParametrizeObject
from scipy.ndimage import interpolation

class VisualRegion(Stimulus):
    """A rectangular region of visual space."""
    location_x = SNumber(qt.degrees,doc="""x location of the center of  visual region.""")
    location_y = SNumber(qt.degrees,doc="""y location of the center of  visual region.""")
    size_x = SNumber(qt.degrees,doc="""The size of the region in degrees (asimuth).""")
    size_y = SNumber(qt.degrees,doc="""The size of the region in degrees (elevation).""")
    
    def __init__(self, **params):
        Stimulus.__init__(self, **params)
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
        return (self.location_x == other.location_x) and (self.location_y == other.location_y) and (self.size_x == other.size_x) and (self.size_y == other.size_y)
    
    def __ne__(self, other):
        return not self.__eq__(other)

    def frames(self):
        return None

    def update(self):
        return None
    
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
        return VisualRegion(location_x=(left + right)/2.0,location_y=(top + bottom)/2.0,size_x=right-left,size_y=top-bottom, \
                duration=-1., frame_duration=1., trial=0) # to match the Stimulus requirements
    
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
        size_in_pixels = numpy.ceil(xy2ij((region.size_x,region.size_y))/float(pixel_size)).astype(int)
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
            
            img_pixel_size = xy2ij((self.size_x,self.size_y))/self.img.shape # is self.size a tuple or an array?
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

class VisualStimulus(VisualObject):
    """Abstract base class for visual stimuli."""
    # merged with previously the stimulus_generator.Stimulus object

    version = __version__ # for simplicity, we take the global version, but it
                          # would be more efficient to take the revision for the
                          # last time this particular file was changed.

    max_luminance = SNumber(lux,doc="""Maximum luminance""")
    density = SNumber(1/(qt.degree),doc="""The density of stimulus - units per degree""")

    def __init__(self, **params):
        VisualObject.__init__(self, **params) # for now, we always put the stimulus in the centre of the visual field

    def update(self):
        """
        Sets the current frame to the next frame in the sequence.
        """
        try:
            self.img, self.variables = self._frames.next()

        except StopIteration:
            self.visible = False
        else:
            assert self.img.min() >= 0 or self.img.min() == TRANSPARENT, "frame minimum is less than zero: %g" % self.img.min()
            assert self.img.max() <= self.max_luminance, "frame maximum (%g) is greater than the maximum luminance (%g)" % (self.img.max(), self.max_luminance)
        self._zoom_cache = {}
    
    def reset(self):
        """
        Reset to the first frame in the sequence.
        """
        self.visible = True
        self._frames = self.frames()
        self.update()

    def export(self, path=None):
        """
        Save the frames to disk. Returns a list of paths to the individual
        frames.
        
        path - the directory in which the individual frames will be saved. If
               path is None, then a temporary directory is created.
        """
        raise NotImplementedError
    
    def next_frame(self):
        """For creating movies with NeuroTools.visualization."""
        self.update()
        return [self.img]
