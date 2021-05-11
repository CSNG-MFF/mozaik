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
        assert (im.shape == (fieldsize_x, fieldsize_y)), "Image dimensions do not correspond to visual field size"

        while True:
            yield (im, [0])

class PSTextureStimulusDisk(TextureBasedVisualStimulus):
    """
    A stimulus generated using the Portilla-Simoncelli algorithm (see textureLib/textureBasedStimulus.m)
    with statistics matched to the original image according to the Type.
    It is confined to an aperture of specific radius
    """

    radius = SNumber(degrees, doc="The radius of the disk - in degrees of visual field")
    stats_type = SNumber(dimensionless,bounds=[0,3],doc="Type of statistial matching of the stimulus")
    sample = SNumber(dimensionless,doc="Index of the stimulus in its texture family")
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

        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        assert (im.shape == (fieldsize_x, fieldsize_y)), "Image dimensions do not correspond to visual field size"

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

        d1 = numpy.multiply(im,c)
        d2 = numpy.multiply(b,-(c-1.0))
        d =  numpy.add.reduce([d1,d2])

        d = d.astype(numpy.uint8)
        IM = Image.fromarray(d)
        IM.save(folder_name + "/" + self.texture + "sample" + str(self.sample) + "type" + str(self.stats_type) + 'radius' + str(self.radius) + '.jpg')

        while True:
            yield (d, [0])
            

class VictorUninformativeSyntheticStimulus(VisualStimulus):
    """
    A stimulus generated using Jonathan Victor's maximum entropy algorithm: Victor, J. D., & Conte, M. M. (2012). Local image statistics: maximum-entropy constructions and perceptual salience. JOSA A, 29(7), 1313-1345.
    Constrain some uniformative 4 pixels correlation statistics as indicated in the pixel statistics list, and generate the rest of the statistics by maximizing the entropy of the image distribution
    """
    pixel_statistics = SList(dimensionless, bounds = (2,2), doc ='The values of pixel correlations statistics. The statistics in the tuple are correspond to the 4 point correlations wye and foot of in Yu, Y., Schmid, A. M., & Victor, J. D. (2015). Visual processing of informative multipoint correlations arises primarily in V2. Elife, 4, e06604.')
    seed = SNumber(dimensionless, doc="The seed used for this stimulus")
    spatial_frequency = SNumber(dimensionless, doc="The spatial frequency of the stimulus")

    def frames(self):
        assert len(numpy.nonzero(self.pixel_statistics)) == 1, 'One and only one value in the tuple pixel_statistics must be non zero'
        for i, stat in enumerate(self.pixel_statistics):
            assert isinstance(stat, float) or isinstance(stat, int), 'Value at index %d is not a number' % i
            assert stat <= 1 and stat >= -1, 'Value at index %d is not within bounds [-1,1]' % i
        fieldsize_x = self.size_x * self.density
        fieldsize_y = self.size_y * self.density
        pixel_spatial_frequency = int(self.spatial_frequency * self.density)
        numpy.random.seed(self.seed)
        im = numpy.zeros((int(fieldsize_x), int(fieldsize_y)))

        for y in range(0,int(fieldsize_y),pixel_spatial_frequency):
            if y+pixel_spatial_frequency < im.shape[1]:
                ylim = y+pixel_spatial_frequency
            else:
                ylim = im.shape[1]

            for x in range(0,int(fieldsize_x),pixel_spatial_frequency):
                if x+pixel_spatial_frequency < im.shape[0]:
                    xlim = x+pixel_spatial_frequency
                else:
                    xlim = im.shape[0]

                if self.pixel_statistics[0] != 0 and x > pixel_spatial_frequency*2 and y > pixel_spatial_frequency*2:
                    if numpy.random.rand() < (1 + self.pixel_statistics[0])/2:
                        im[y:ylim,x:xlim] = (im[y-pixel_spatial_frequency, x-pixel_spatial_frequency] + im[y-pixel_spatial_frequency, x-2*pixel_spatial_frequency] + im[y-2*pixel_spatial_frequency, x-pixel_spatial_frequency]) % 510
                    else:
                        im[y:ylim,x:xlim] = -1 * ((im[y-pixel_spatial_frequency, x-pixel_spatial_frequency] + im[y-pixel_spatial_frequency, x-2*pixel_spatial_frequency] + im[y-2*pixel_spatial_frequency, x-pixel_spatial_frequency]) % 510 - 255)

                elif self.pixel_statistics[1] != 0 and y > pixel_spatial_frequency and x > 2*pixel_spatial_frequency:
                    if numpy.random.rand() < (1 + self.pixel_statistics[1])/2:
                        im[y:ylim,x:xlim] = (im[y-pixel_spatial_frequency, x-pixel_spatial_frequency] + im[y, x-2*pixel_spatial_frequency] + im[y-pixel_spatial_frequency, x]) % 510
                    else:
                        im[y:ylim,x:xlim] = -1 * ((im[y-pixel_spatial_frequency, x-pixel_spatial_frequency] + im[y, x-2*pixel_spatial_frequency] + im[y-pixel_spatial_frequency, x]) % 510 - 255)

                else:
                    if numpy.random.rand() < 0.5:
                        im[y:ylim,x:xlim] = 255

        while True:
            yield (im, [0])


class VictorInformativeSyntheticStimulus(VisualStimulus):
    """
    A stimulus generated using Jonathan Victor's maximum entropy algorithm
    Some statistics of each 2*2 regions of the image are constrained as indicated in the pixel statistics list, and the rest of the statistics are computed in order to maximize the entropy of the image distribution
    """
    pixel_statistics = SList(dimensionless, bounds = (10,10), doc ='The values of pixel correlations statistics. The statistics in the tuple are ranked in the same as in Fig2 in Hermundstad, A. M., Briguglio, J. J., Conte, M. M., Victor, J. D., Balasubramanian, V., & Tkacik, G. (2014). Variance predicts salience in central sensory processing. Elife, 3, e03722.')
    seed = SNumber(dimensionless, doc="The seed used for this stimulus")
    spatial_frequency = SNumber(dimensionless, doc="The spatial frequency of the stimulus")


    def frames(self):
        assert len(numpy.nonzero(self.pixel_statistics)) == 1, 'One and only one value in the tuple pixel_statistics must be non zero'
        for i, stat in enumerate(self.pixel_statistics):
            assert isinstance(stat, float) or isinstance(stat, int), 'Value at index %d is not a number' % i
            assert stat <= 1 and stat >= -1, 'Value at index %d is not within bounds [-1,1]' % i
        fieldsize_x = self.size_x * self.density
        fieldsize_y = self.size_y * self.density
        pixel_spatial_frequency = int(self.spatial_frequency * self.density)
        numpy.random.seed(self.seed)
        im = numpy.zeros((int(fieldsize_x), int(fieldsize_y)))

        for y in range(0,int(fieldsize_y),pixel_spatial_frequency):
            if y+pixel_spatial_frequency < im.shape[1]:
                ylim = y+pixel_spatial_frequency
            else:
                ylim = im.shape[1]

            for x in range(0,int(fieldsize_x),pixel_spatial_frequency):
                if x+pixel_spatial_frequency < im.shape[0]:
                    xlim = x+pixel_spatial_frequency
                else:
                    xlim = im.shape[0]

                if self.pixel_statistics[0] != 0:
                    if numpy.random.rand() < (1 + self.pixel_statistics[0])/2:
                        im[y:ylim,x:xlim] = 255

                elif self.pixel_statistics[1] != 0 and y > 0:
                    if numpy.random.rand() < (1 + self.pixel_statistics[1])/2:
                        im[y:ylim,x:xlim] = im[y-1, x]
                    else:
                        im[y:ylim,x:xlim] = -1 * (im[y-1, x] - 255)

                elif self.pixel_statistics[2] != 0 and x > 0:
                    if numpy.random.rand() < (1 + self.pixel_statistics[2])/2:
                        im[y:ylim,x:xlim] = im[y, x-1]
                    else:
                        im[y:ylim,x:xlim] = -1 * (im[y, x-1] - 255)

                elif self.pixel_statistics[3] != 0 and y > 0 and x < fieldsize_x - pixel_spatial_frequency:
                    if numpy.random.rand() < (1 + self.pixel_statistics[3])/2:
                        im[y:ylim,x:xlim] = im[y-1, x+pixel_spatial_frequency]
                    else:
                        im[y:ylim,x:xlim] = -1 * (im[y-1, x+pixel_spatial_frequency] - 255)


                elif self.pixel_statistics[4] != 0 and y > 0 and x > 0:
                    if numpy.random.rand() < (1 + self.pixel_statistics[4])/2:
                        im[y:ylim,x:xlim] = im[y-1, x-1]
                    else:
                        im[y:ylim,x:xlim] = -1 * (im[y-1, x-1] - 255)

                elif self.pixel_statistics[5] != 0 and y > 0 and x > 0:
                    if numpy.random.rand() < (1 + self.pixel_statistics[5])/2:
                        im[y:ylim,x:xlim] = -1 * ((im[y-1, x-1] + im[y, x-1]) % 510 - 255)
                    else:
                        im[y:ylim,x:xlim] = (im[y-1, x-1] + im[y, x-1]) % 510

                elif self.pixel_statistics[6] != 0 and y > 0 and x < fieldsize_x - pixel_spatial_frequency:
                    if numpy.random.rand() < (1 + self.pixel_statistics[6])/2:
                        im[y:ylim,x:xlim] = -1 * ((im[y-1, x+pixel_spatial_frequency] + im[y-1, x]) % 510 - 255)
                    else:
                        im[y:ylim,x:xlim] = (im[y-1, x+pixel_spatial_frequency] + im[y-1, x]) % 510

                elif self.pixel_statistics[7] != 0 and y > 0 and x > 0:
                    if numpy.random.rand() < (1 + self.pixel_statistics[7])/2:
                        im[y:ylim,x:xlim] = -1 * ((im[y-1, x] + im[y-1, x-1])% 510 - 255)
                    else:
                        im[y:ylim,x:xlim] = (im[y-1, x] + im[y-1, x-1]) % 510

                elif self.pixel_statistics[8] != 0 and y > 0 and x > 0:
                    if numpy.random.rand() < (1 + self.pixel_statistics[8])/2:
                        im[y:ylim,x:xlim] = -1 * ((im[y-1, x] + im[y, x-1])% 510 - 255)
                    else:
                        im[y:ylim,x:xlim] = (im[y-1, x] + im[y, x-1]) % 510

                elif self.pixel_statistics[9] != 0 and y > 0 and x > 0:
                    if numpy.random.rand() < (1 + self.pixel_statistics[9])/2:
                        im[y:ylim,x:xlim] = (im[y-1, x-1] + im[y, x-1] + im[y-1, x]) % 510
                    else:
                        im[y:ylim,x:xlim] = -1 * ((im[y-1, x-1] + im[y, x-1] + im[y-1, x]) % 510 - 255)

                else:
                    if numpy.random.rand() < 0.5:
                        im[y:ylim,x:xlim] = 255

        while True:
            yield (im, [0])

