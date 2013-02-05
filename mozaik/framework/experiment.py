"""
docstring goes here
"""
import mozaik
import mozaik.stimuli.topographica_based as topo
import numpy
import resource

logger = mozaik.getMozaikLogger("Mozaik")


class Experiment(object):
    """
    The abastract class for an experiment. The experiment defines the list of 
    stimuli that it needs to present to the brain.

    These stimulus presentations have to be independent - e.g. should not
    temporarily depend on others. It should also specify the analysis of the
    recorded results that it performs. This can be left empty if analysis will
    be done later.
    """
    def __init__(self, model):
        self.stimuli = []
    
    def return_stimuli(self):
        return self.stimuli
        
    def run(self,data_store,stimuli):
        raise NotImplementedError
        pass
        
    def do_analysis(self):
        raise NotImplementedError
        pass


class VisualExperiment(Experiment):
    """
    Visual experiment. On top of Experiment class it 
    """
    
    def __init__(self, model):
        Experiment.__init__(self, model)
        self.background_luminance = model.input_space.background_luminance 
        self.model = model

    def run(self,data_store,stimuli):
        srtsum = 0
        for i,s in enumerate(stimuli):
            logger.debug('Presenting stimulus: ' + str(s) + '\n')
            (segments,input_stimulus,simulator_run_time) = self.model.present_stimulus_and_record(s)
            srtsum += simulator_run_time
            data_store.add_recording(segments,s)
            data_store.add_stimulus(input_stimulus,s)
            logger.info('Stimulus %d/%d finished. Memory usage: %iMB' % (i+1,len(stimuli),resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
        return srtsum
    

class MeasureOrientationTuningFullfield(VisualExperiment):

    def __init__(self, model, num_orientations, spatial_frequency,
                 temporal_frequency, grating_duration, contrasts, num_trials):
        VisualExperiment.__init__(self, model)
        for c in contrasts:
            for i in xrange(0, num_orientations):
                for k in xrange(0, num_trials):
                    self.stimuli.append(topo.FullfieldDriftingSinusoidalGrating(
                                    frame_duration=7,
                                    size_x=model.visual_field.size_x,
                                    size_y=model.visual_field.size_y,
                                    location_x=0.0,
                                    location_y=0.0,
                                    background_luminance=self.background_luminance,
                                    contrast = c,
                                    duration=grating_duration,
                                    density=40,
                                    trial=k,
                                    orientation=numpy.pi/num_orientations*i,
                                    spatial_frequency=spatial_frequency,
                                    temporal_frequency=temporal_frequency
                                ))

    def do_analysis(self, data_store):
        pass


class MeasureSizeTuning(VisualExperiment):

    def __init__(self, model, num_sizes, max_size, orientation,
                 spatial_frequency, temporal_frequency, grating_duration,
                 contrasts, num_trials):
        VisualExperiment.__init__(self, model)                                          
        for c in contrasts:
            for i in xrange(0, num_sizes):
                for k in xrange(0, num_trials):
                    self.stimuli.append(topo.DriftingSinusoidalGratingDisk(
                                    frame_duration=7,
                                    size_x=model.visual_field.size_x,
                                    size_y=model.visual_field.size_y,
                                    location_x=0.0,
                                    location_y=0.0,
                                    background_luminance=self.background_luminance,
                                    contrast = c,
                                    duration=grating_duration,
                                    density=40,
                                    trial=k,
                                    orientation=orientation,
                                    radius=max_size/num_sizes*i,
                                    spatial_frequency=spatial_frequency,
                                    temporal_frequency=temporal_frequency
                                ))

    def do_analysis(self, data_store):
        pass


class MeasureOrientationContrastTuning(VisualExperiment):

    def __init__(self, model, num_orientations, orientation, center_radius,
                 surround_radius, spatial_frequency, temporal_frequency,
                 grating_duration, contrasts, num_trials):
        VisualExperiment.__init__(self, model)
        for c in contrasts:
            for i in xrange(0, num_sizes):
                for k in xrange(0, num_trials):
                    self.stimuli.append(
                        topo.DriftingSinusoidalGratingCenterSurroundStimulus(
                                    frame_duration=7,
                                    size_x=model.visual_field.size_x,
                                    size_y=model.visual_field.size_y,
                                    location_x=0.0,
                                    location_y=0.0,
                                    background_luminance=self.background_luminance,
                                    contrast = c,
                                    duration=grating_duration,
                                    density=40,
                                    trial=k,
                                    center_orientation=orientation,
                                    surround_orientation=numpy.pi/num_orientations*i,
                                    gap=0,
                                    center_radius=center_radius,
                                    surround_radius=surround_radius,
                                    spatial_frequency=spatial_frequency,
                                    temporal_frequency=temporal_frequency
                                ))

    def do_analysis(self, data_store):
        pass


class MeasureNaturalImagesWithEyeMovement(VisualExperiment):

    def __init__(self, model, stimulus_duration, num_trials):
        VisualExperiment.__init__(self, model)
        for k in xrange(0, num_trials):
            self.stimuli.append(
                topo.NaturalImageWithEyeMovement(
                            frame_duration=7,
                            size_x=model.visual_field.size_x,
                            size_y=model.visual_field.size_y,
                            location_x=0.0,
                            location_y=0.0,
                            background_luminance=self.background_luminance,
                            duration=stimulus_duration,
                            density=40,
                            trial=k,
                            size=40,  # x size of image
                            eye_movement_period=6.66,  # eye movement period
                            eye_path_location='./eye_path.pickle',
                            image_location='./image_naturelle_HIGH.bmp'
                            ))

    def do_analysis(self, data_store):
        pass


class MeasureDriftingSineGratingWithEyeMovement(VisualExperiment):

    def __init__(self, model, stimulus_duration, num_trials,spatial_frequency,temporal_frequency,contrast):
        VisualExperiment.__init__(self, model)
        for k in xrange(0, num_trials):
            self.stimuli.append(
                topo.DriftingGratingWithEyeMovement(
                            frame_duration=7,
                            size_x=model.visual_field.size_x,
                            size_y=model.visual_field.size_y,
                            location_x=0.0,
                            location_y=0.0,
                            background_luminance=self.background_luminance,
                            duration=stimulus_duration,
                            density=40,
                            trial=k,
                            contrast = contrast,
                            eye_movement_period=6.66,  # eye movement period
                            eye_path_location='./eye_path.pickle',
                            orientation=0,
                            spatial_frequency=spatial_frequency,
                            temporal_frequency=temporal_frequency
                            ))

    def do_analysis(self, data_store):
        pass



class MeasureSpontaneousActivity(VisualExperiment):
    def __init__(self,model,duration,num_trials):
            VisualExperiment.__init__(self, model)
            for k in xrange(0,num_trials):
                self.stimuli.append(
                            topo.Null(   
                                frame_duration=7, 
                                size_x=model.visual_field.size_x,
                                size_y=model.visual_field.size_y,
                                location_x=0.0,
                                location_y=0.0,
                                background_luminance=self.background_luminance,
                                duration=duration,
                                density=40,
                                trial=k
                ))    
    def do_analysis(self, data_store):
        pass
