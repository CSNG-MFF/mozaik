"""
docstring goes here
"""
import mozaik
import mozaik.stimuli.topographica_based as topo
from mozaik.stimuli.stimulus import InternalStimulus
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
    
    Also each Experiment can define the spike_stimulator and current_stimulator variables that 
    allow the experiment to stimulate specific neurons in the model with either spike trains and conductances.
    
    The (exc/inh)_spike_stimulator is a dictionary where keys are the name of the sheets, and the values are tuples (neuron_list,spike_generator)
    where neuron_list is a list of neuron_ids that should be stimulated, this varialbe can also be a string 'all', in which case all neurons 
    in the given sheet will be stimulated. The spike_generator should be function that receives a single input - duration - that returns a 
    spike train (list of spike times) of lasting the duration miliseconds. This function will be called for each stimulus presented during this experiment,
    with the duration of the stimulus as the duration parameter. 
    
    NOTE!: The spikes from (exc/inh)_spike_stimulator will be weighted by the connection with weights defined by the sheet's background_noise.(exc/inh)_weight parameter!!!
    """
    def __init__(self, model):
        self.model = model
        self.stimuli = []
        self.exc_spike_stimulators = {}
        self.inh_spike_stimulators = {}
        self.current_stimulators = {}
    
    def return_stimuli(self):
        return self.stimuli
        
    def run(self,data_store,stimuli):
        srtsum = 0
        for i,s in enumerate(stimuli):
            logger.debug('Presenting stimulus: ' + str(s) + '\n')
            for sheet_name in self.exc_spike_stimulators.keys():                
                print len(self.exc_spike_stimulators[sheet_name][1](1000))
            
            (segments,input_stimulus,simulator_run_time) = self.model.present_stimulus_and_record(s,self.exc_spike_stimulators,self.inh_spike_stimulators)
            srtsum += simulator_run_time
            data_store.add_recording(segments,s)
            data_store.add_stimulus(input_stimulus,s)
            logger.info('Stimulus %d/%d finished. Memory usage: %iMB' % (i+1,len(stimuli),resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
        return srtsum
        
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
        #JAHACK: This is kind of a hack now. There needs to be generally defined interface of what is the resolution of a visual input layer
        # possibly in the future we could force the visual_space to have resolution, perhaps something like native_resolution parameter!?
        self.density  = 1/self.model.input_layer.parameters.receptive_field.spatial_resolution # in pixels per degree of visual space 

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
                                    density=self.density,
                                    trial=k,
                                    orientation=numpy.pi/num_orientations*i,
                                    spatial_frequency=spatial_frequency,
                                    temporal_frequency=temporal_frequency))

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
                                    density=self.density,
                                    trial=k,
                                    orientation=orientation,
                                    radius=max_size/num_sizes*(i+1),
                                    spatial_frequency=spatial_frequency,
                                    temporal_frequency=temporal_frequency))

    def do_analysis(self, data_store):
        pass


class MeasureOrientationContrastTuning(VisualExperiment):

    def __init__(self, model, num_orientations, orientation, center_radius,
                 surround_radius, spatial_frequency, temporal_frequency,
                 grating_duration, contrasts, num_trials):
        VisualExperiment.__init__(self, model)
        for c in contrasts:
            for i in xrange(0, num_orientations):
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
                                    density=self.density,
                                    trial=k,
                                    center_orientation=orientation,
                                    surround_orientation=numpy.pi/num_orientations*i,
                                    gap=0,
                                    center_radius=center_radius,
                                    surround_radius=surround_radius,
                                    spatial_frequency=spatial_frequency,
                                    temporal_frequency=temporal_frequency))

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
                            density=self.density,
                            trial=k,
                            size=40,  # x size of image
                            eye_movement_period=6.66,  # eye movement period
                            eye_path_location='./eye_path.pickle',
                            image_location='./image_naturelle_HIGH.bmp'))

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
                            density=self.density,
                            trial=k,
                            contrast = contrast,
                            eye_movement_period=6.66,  # eye movement period
                            eye_path_location='./eye_path.pickle',
                            orientation=0,
                            spatial_frequency=spatial_frequency,
                            temporal_frequency=temporal_frequency))

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
                                density=self.density,
                                trial=k))    
    def do_analysis(self, data_store):
        pass

class PoissonNetworkKick(Experiment):
    """
    This experiment does not show any stimulus.
    Importantly for the duration of the experiment it will stimulate neurons 
    definded by the recording configurations in recording_configuration_list
    in the sheets specified in the sheet_list with Poisson spike train of mean 
    frequency determined by the corresponding values in lambda_list.
    """
    def __init__(self,model,duration,sheet_list,recording_configuration_list,lambda_list):
            Experiment.__init__(self, model)
            from NeuroTools import stgen
            for sheet_name,lamb,rc in zip(sheet_list,lambda_list,recording_configuration_list):
                self.exc_spike_stimulators[sheet_name] = (rc.generate_idd_list_of_neurons(),(lambda duration,lamb=lamb: stgen.StGen().poisson_generator(rate=lamb,t_start=0,t_stop=duration).spike_times))

            self.stimuli.append(
                        InternalStimulus(   
                                            frame_duration=duration, 
                                            duration=duration,
                                            trial=0,
                                            direct_stimulation_name='Kick'
                                         )
                                )
        
class NoStimulation(Experiment):
    """
    This experiment does not show any stimulus for the duration of the experiment.
    """
    def __init__(self,model,duration):
        Experiment.__init__(self, model)
        self.stimuli.append(
                        InternalStimulus(   
                                            frame_duration=duration, 
                                            duration=duration,
                                            trial=0,
                                            direct_stimulation_name='None'
                                         )
                                )
