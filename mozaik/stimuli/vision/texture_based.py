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
from mozaik.controller import Global
from numpy import pi
from quantities import Hz, rad, degrees, ms, dimensionless
from oct2py import octave #octave interface
import scipy.misc #for testing
from PIL import Image #for testing
import os

class TextureBasedVisualStimulus(VisualStimulus):
    """
    As we do not handle transparency in texture stimuli (i.e. all pixels of all stimuli difned here will have 0% transparancy)
    in this abstract class we disable the transparent flag defined by the :class:`mozaik.stimuli.visual_stimulus.VisualStimulus`, to improve efficiency.

    We require a path to the image from which we generate stimuli
    """
    texture_path = SString(doc="path to the image from which we generate stimuli")
    texture = SString(doc="name of the original texture from which we generate stimuli")


    def __init__(self,**params):
        VisualStimulus.__init__(self,**params)
        #self.transparent = False # We will not handle transparency anywhere here for now so let's make it fast
    
            


                    
class PSTextureStimulus(TextureBasedVisualStimulus):
    """
    A stimulus generated using the Portilla-Simoncelli algorithm (see textureLib/textureBasedStimulus.m)
    with statistics matched to the original image according to the Type.
    It is presented for *stimulus_duration* milliseconds. For the remaining time, 
    until the *duration* of the stimulus, constant *background_luminance* 
    is displayed.. 
     
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
    sample = SNumber(dimensionless,doc="Index of the stimulus in its texture family")
    stimulus_duration = SNumber(ms, doc="The duration of the stumulus presentation.")

    seed = SNumber(dimensionless, doc="The seed used for this stimulus")

    def frames(self):
        fieldsize_x = self.size_x * self.density
        fieldsize_y = self.size_y * self.density
        folder_name = Global.root_directory + "/TextureImagesStimuli"
        libpath = visual_stimulus.__file__.replace("/visual_stimulus.pyc", "") + "/textureLib" #path to the image processing library
        matlabPyrToolspath = os.path.join(libpath,"textureSynth","matlabPyrTools")
        if not os.path.isdir(matlabPyrToolspath):
            raise IOError("matlabPyrTools should be downloaded from https://github.com/LabForComputationalVision/matlabPyrTools and its content should be put in the directory "+matlabPyrToolspath)

        octave.addpath(libpath)     
        im = octave.textureBasedStimulus(self.texture_path,
                                         self.stats_type,
                                         self.seed, 
                                         fieldsize_x, 
                                         fieldsize_y,
                                         libpath)
        scale = 2. * self.background_luminance/ (numpy.max(im) - numpy.min(im))
        im = (im - numpy.min(im)) * scale
        im = im.astype(numpy.uint8)

        if not os.path.exists(folder_name): 
            os.mkdir(folder_name)

        IM = Image.fromarray(im)
        IM.save(folder_name + "/" + self.texture + "sample" + str(self.sample) + "type" + str(self.stats_type) + '.jpg')
        #scipy.misc.toimage(im, cmin=0.0, cmax=2*self.background_luminance).save(folder_name + "/" + self.texture + "sample" + str(self.sample) + "type" + str(self.stats_type) + '.jpg')
        assert (im.shape == (fieldsize_x, fieldsize_y)), "Image dimensions do not correspond to visual field size"

        blank = imagen.Constant(scale=self.background_luminance,
                                bounds=BoundingBox(radius=self.size_x/2),
                                xdensity=self.density,
                                ydensity=self.density)()
        num_frames = 0
        while True:
            num_frames += 1
            if (num_frames-1) * self.frame_duration < self.stimulus_duration: 
                yield (im, [0])
            else:
                yield (blank, [0])
