from MozaikLite.stimuli.topographica_based import FullfieldDriftingSinusoidalGrating, Null, NaturalImageWithEyeMovement
from MozaikLite.visualization.plotting import GSynPlot,RasterPlot,VmPlot,CyclicTuningCurvePlot,OverviewPlot, ConductanceSignalListPlot, RetinalInputMovie
from NeuroTools.parameters import ParameterSet, ParameterDist

import numpy

class Experiment(object):
    """
    The experiment defines the list of stimuli that it needs to present to the brain.
    These stimuli presentations have to be independent - e.g. should not temporarily 
    depend on others. It should also specify the analysis of the recorded results 
    that it performs. This can be left empty if analysis will be done later.
    """
    
    stimuli = []
    
    def return_stimuli(self):
        return self.stimuli
        
    def run(self,data_store,stimuli):
        for s in stimuli:
            print 'Presenting stimulus: ',str(s) , '\n'
            (segments,retinal_input) = self.model.present_stimulus_and_record(s)
            data_store.add_recording(segments,s)
            data_store.add_retinal_stimulus(retinal_input,s)
    
    def do_analysis(self):
        raise NotImplementedError
        pass

class MeasureOrientationTuningFullfield(Experiment):
    
    def __init__(self,model,num_orientations,spatial_frequency,temporal_frequency,grating_duration,num_trials):
        self.model = model
        for j in [0.3]:
            for i in xrange(0,num_orientations):
                for k in xrange(0,num_trials):
                    self.stimuli.append(FullfieldDriftingSinusoidalGrating([
                                    7, # frame duration (roughly like a movie) - is this fast enough?
                                    model.visual_field.size[0],
                                    model.visual_field.size[0],
                                    0.0,
                                    0.0,
                                    j*90.0, #max_luminance
                                    grating_duration, # stimulus duration
                                    40, #density
                                    k, # trial number
                                    numpy.pi/num_orientations*i, #orientation
                                    spatial_frequency,
                                    temporal_frequency, #stimulus duration - we want to get one full sweep of phases
                                ]))    

    def do_analysis(self,data_store):
        pass
        
class MeasureNaturalImagesWithEyeMovement(Experiment):
    
    def __init__(self,model,stimulus_duration,num_trials):
        self.model = model
        for k in xrange(0,num_trials):
            self.stimuli.append(NaturalImageWithEyeMovement([   
                            7, # frame duration (roughly like a movie) - is this fast enough?
                            model.visual_field.size[0], 
                            model.visual_field.size[0], 
                            0.0,
                            0.0,
                            90.0, #max_luminance 
                            stimulus_duration, # stimulus duration
                            40, #density
                            k, # trial number
                            40, # x size of image
                            6.66, # eye movement period
                            1 # idd
                        ],'eye_path.pickle','image_naturelle_HIGH.bmp'))    


    def do_analysis(self,data_store):
        pass
        
class MeasureSpontaneousActivity(Experiment):
    
    def __init__(self,model,duration):
            self.model = model
            self.stimuli.append(Null([   
                            7, # frame duration (roughly like a movie) - is this fast enough?
                            model.visual_field.size[0], 
                            model.visual_field.size[0], 
                            0.0,
                            0.0,
                            90.0, #max_luminance 
                            duration, # stimulus duration
                            40, #density
                            0 # trial number
                        ]))    

    def do_analysis(self,data_store):
        pass
