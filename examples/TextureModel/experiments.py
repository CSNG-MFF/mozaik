#!/usr/local/bin/ipython -i 
from mozaik.experiments import *
from mozaik.experiments.vision import *
from mozaik.sheets.population_selector import RCRandomPercentage
from parameters import ParameterSet
import mozaik
    
def create_experiments(model):
    return              [
                           #Spontaneous Activity 
                           #NoStimulation(model,ParameterSet({'duration' : 147*7})),
                           
                           MeasureTextureSensitivityFullfield(model,ParameterSet({
                                                                                 'num_images' : 2,
                                                                                 'image_path' : "/home/kaktus/Documents/mozaik/mozaik/stimuli/vision/textureLib/textureSynth/metal.pgm",
                                                                                 'image_duration' : 147*7,
                                                                                 'types' : [0, 1, 2],
                                                                                 'num_trials' : 3})),
                           MeasureTextureSensitivityFullfield(model,ParameterSet({
                                                                                 'num_images' : 2,
                                                                                 'image_path' : "/home/kaktus/Documents/mozaik/mozaik/stimuli/vision/textureLib/textureSynth/reptil_skin.pgm",
                                                                                 'image_duration' : 147*7,
                                                                                 'types' : [0, 1, 2],
                                                                                 'num_trials' : 3})),
                       

                           #GRATINGS
                           #MeasureOrientationTuningFullfield(model,ParameterSet({
                        #                                                         'num_orientations' : 2,
                        #                                                         'spatial_frequency' : 0.8,
                        #                                                         'temporal_frequency'  : 2,
                        #                                                         'grating_duration' : 2*147*7,
                        #                                                         'contrasts' : [100],
                        #                                                         'num_trials' : 3})),
                       
                           
                           #MeasureFrequencySensitivity(model,ParameterSet({
                           #                                                 'orientation' : 0,
                           #                                                 'spatial_frequencies' : [0.01,0.8,1.5],
                           #                                                 'temporal_frequencies' : [2.0],
                           #                                                 'grating_duration'  : 147*7*2,
                           #                                                 'contrasts' : [100],
                           #                                                 'num_trials' : 1})),                           
                           
                           #IMAGES WITH EYEMOVEMENT
                           #MeasureNaturalImagesWithEyeMovement(model,ParameterSet({
                           #                                                            'stimulus_duration' : 2*147*7,
                           #                                                            'num_trials' : 10})),

                           #GRATINGS WITH EYEMOVEMENT
                           #MeasureDriftingSineGratingWithEyeMovement(model,ParameterSet({
                           #                                                                 'spatial_frequency' : 0.8,
                           #                                                                 'temporal_frequency' :2,
                           #                                                                 'stimulus_duration' : 147*7,
                           #                                                                 'num_trials' : 10,
                           #                                                                 'contrast' : 100),
                        ]

