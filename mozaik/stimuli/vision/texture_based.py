"""
The file contains stimuli generated based on a texture image

"""

from visual_stimulus import VisualStimulus
import visual_stimulus
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
from oct2py import octave #octave interface
import scipy.misc #for testing


class TextureBasedVisualStimulus(VisualStimulus):
    """
    As we do not handle transparency in texture stimuli (i.e. all pixels of all stimuli difned here will have 0% transparancy)
    in this abstract class we disable the transparent flag defined by the :class:`mozaik.stimuli.visual_stimulus.VisualStimulus`, to improve efficiency.

    We require a path to the image from which we generate stimuli
    """
    texture_path = SString(doc="path to the image from which we generate stimuli")

    def __init__(self,**params):
        VisualStimulus.__init__(self,**params)
        #self.transparent = False # We will not handle transparency anywhere here for now so let's make it fast
    
            


                    
class PSTextureStimulus(TextureBasedVisualStimulus):
    """
    A stimulus generated using the Portilla-Simoncelli algorithm (see textureLib/textureBasedStimulus.m)
    with statistics matched to the original image according to the Type. 
     
    Types:
        0 - original image
        1 - naturalistic texture image (matched higher order statistics)
        2 - spectrally matched noise (matched marginal statistics only). 

    Notes
    -----
    frames_number - the number of frames for which each image is presented
    ALERT!!!

    Because of optimization issues, the stimulus is not re-generated on every trial.
    We should add new 'empty' paramter to replace trial to force recalculaton of the stimuls.
    """

    stats_type = SNumber(dimensionless,bounds=[0,3],doc="Type of statistial matching of the stimulus")
    seed = int

    def frames(self):
        print("TRIAL NUMBER " + str(self.trial))
        fieldsize_x = self.size_x * self.density
        fieldsize_y = self.size_y * self.density
        libpath = visual_stimulus.__file__.replace("/visual_stimulus.pyc", "") + "/textureLib" #path to the image processing library
        octave.addpath(libpath)     
        im = octave.textureBasedStimulus(self.texture_path,
                                         self.stats_type,
                                         self.seed, 
                                         fieldsize_x, 
                                         fieldsize_y,
                                         libpath)

        im = im / 255* 2*self.background_luminance
        #scipy.misc.toimage(im, cmin=0.0, cmax=2*self.background_luminance).save('/home/kaktus/Documents/mozaik/examples/img' + str(self.trial) + '.jpg')
        scipy.misc.toimage(im, cmin=0.0, cmax=2*self.background_luminance).save('/home/kaktus/Documents/mozaik/examples/img' + str(len(self.texture_path)) + "type" + str(self.stats_type) + '.jpg')
        
        assert (im.shape == (fieldsize_x, fieldsize_y)), "Image dimensions do not correspond to visual field size"
        while True:
            yield (im, [0])



