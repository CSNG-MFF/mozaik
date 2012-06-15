from mozaik.stimuli.topographica_based import FullfieldDriftingSinusoidalGrating, Null, NaturalImageWithEyeMovement
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
    
    def __init__(self,model,num_orientations,spatial_frequency,temporal_frequency,grating_duration,contrasts,num_trials):
        self.model = model
        for j in contrasts:
            for i in xrange(0,num_orientations):
                for k in xrange(0,num_trials):
                    self.stimuli.append(FullfieldDriftingSinusoidalGrating(
                                    frame_duration=7, 
                                    size_x=model.visual_field.size_x,
                                    size_y=model.visual_field.size_y,
                                    location_x=0.0,
                                    location_y=0.0,
                                    max_luminance=j*90.0,
                                    duration=grating_duration,
                                    density=40,
                                    trial=k,
                                    orientation=numpy.pi/num_orientations*i, 
                                    spatial_frequency=spatial_frequency,
                                    temporal_frequency=temporal_frequency 
                                ))    

    def do_analysis(self,data_store):
        pass
        
class MeasureNaturalImagesWithEyeMovement(Experiment):
    
    def __init__(self,model,stimulus_duration,num_trials):
        self.model = model
        for k in xrange(0,num_trials):
            self.stimuli.append(NaturalImageWithEyeMovement(   
                            frame_duration=7, 
                            size_x=model.visual_field.size_x,
                            size_y=model.visual_field.size_y,
                            location_x=0.0,
                            location_y=0.0,
                            max_luminance=90.0,
                            duration=stimulus_duration,
                            density=40,
                            trial=k,
                            size=40, # x size of image
                            eye_movement_period=6.66, # eye movement period
                            eye_path_location='eye_path.pickle',
                            image_location='image_naturelle_HIGH.bmp'
                            ))    


    def do_analysis(self,data_store):
        pass
        
class MeasureSpontaneousActivity(Experiment):
    
    def __init__(self,model,duration):
            self.model = model
            self.stimuli.append(Null(   
                            frame_duration=7, 
                            size_x=model.visual_field.size_x,
                            size_y=model.visual_field.size_y,
                            location_x=0.0,
                            location_y=0.0,
                            max_luminance=90.0,
                            duration=duration,
                            density=40,
                            trial=k,
            ))    

    def do_analysis(self,data_store):
        pass
