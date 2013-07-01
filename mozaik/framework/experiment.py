"""
Module containing the experiment definition API.
"""

import mozaik
import mozaik.stimuli.topographica_based as topo
from mozaik.stimuli.stimulus import InternalStimulus
import numpy
import resource

logger = mozaik.getMozaikLogger()


class Experiment(object):
    """
    The abastract class for an experiment. The experiment defines the list of 
    stimuli that it needs to present to the brain.These stimuli presentations have to be independent - e.g. should not
    temporarily depend on each other. Experiment should also specify the analysis of the
    recorded results that it performs. This can be left empty if analysis will
    be done later.
    
    Also each Experiment can define the spike_stimulator and current_stimulator variables that 
    allow the experiment to stimulate specific neurons in the model with either spike trains and conductances.
    
    The (exc/inh)_spike_stimulator is a dictionary where keys are the name of the sheets, and the values are tuples (neuron_list,spike_generator)
    where neuron_list is a list of neuron_ids that should be stimulated, this varialbe can also be a string 'all', in which case all neurons 
    in the given sheet will be stimulated. The spike_generator should be function that receives a single input - duration - that returns a 
    spike train (list of spike times) of lasting the duration miliseconds. This function will be called for each stimulus presented during this experiment,
    with the duration of the stimulus as the duration parameter. 
    
    
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    NOTE
    ----
    When creating a new Expriment, user inherits from the Experiment class, and in the constructor fills up the `self.stimuli` array with the list of stimuli
    that the experiment presents to the model. One can also implement the do_analysis method, which should perform the analysis that the experiments requires
    at the end. Finaly the `self.exc_spike_stimulators` and `self.inh_spike_stimulators` dictionaries can also be specified in the constructor.
    
    The spikes from (exc/inh)_spike_stimulator will be weighted by the connection with weights defined by the sheet's background_noise.(exc/inh)_weight parameter!
    """
    def __init__(self, model):
        self.model = model
        self.stimuli = []
        self.exc_spike_stimulators = {}
        self.inh_spike_stimulators = {}
        self.current_stimulators = {}
    
    def return_stimuli(self):
        """
        This function is called by mozaik to retrieve the list of stimuli the experiment requires to be presented to the model.
        """
        return self.stimuli
        
    def run(self,data_store,stimuli):
        """
        This function is called to execute the experiment.
        
        Parameters
        ----------
        
        data_store : DataStore
                   The data store into which to store the recorded data.
                   
        stimuli : list(Stimulus)
                The list of stimuli to present to the model.
        
        Returns
        -------
        strsum : int (s)
               The overal simulation time it took to execute the experiment.
                
        Notes
        -----
        The reason why this function gets a list of stimuli as input is that even though the experiment itself defines the list of stimuli
        to present to the model, some of these might have already been presented. The module `mozaik.framework.experimental_controller` filters
        the list of stimuli which to present to prevent repetitions, and lets this function know via the stimuli argument which stimuli to actually present.
        """
        srtsum = 0
        for i,s in enumerate(stimuli):
            logger.debug('Presenting stimulus: ' + str(s) + '\n')
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
    Visual experiment. On top of Experiment class it defines a new variable background_luminance, 
    that it sets to be the background luminance of the model's input space, and new variable density
    which is set to over the spatial_resolution of the input layer's receptive field spatial resolution.
    All experiments in the visual sensory domain should be derived from this class.
    
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.
    """
    
    def __init__(self, model):
        Experiment.__init__(self, model)
        self.background_luminance = model.input_space.background_luminance 
        #JAHACK: This is kind of a hack now. There needs to be generally defined interface of what is the resolution of a visual input layer
        # possibly in the future we could force the visual_space to have resolution, perhaps something like native_resolution parameter!?
        self.density  = 1/self.model.input_layer.parameters.receptive_field.spatial_resolution # in pixels per degree of visual space 

class MeasureOrientationTuningFullfield(VisualExperiment):
    """
    Measure orientation tuning using a fullfiled sinusoidal grating.
    
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    num_orientations : int
          Number of orientation to present.
    
    spatial_frequency : float
                      Spatial frequency of the grating.
                      
    temporal_frequency : float
                      Temporal frequency of the grating.

    grating_duration : float
                      The duration of single presentation of a grating.
    
    contrasts : list(float) 
              List of contrasts (expressed as % : 0-100%) at which to measure the orientation tuning.
    
    num_trials : int
               Number of trials each each stimulus is shown.
    """
    
    
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
    """
    Measure size tuning using expanding sinusoidal grating disk.
    
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    num_sizes : int
              Number of sizes to present.
    
    max_size : float (degrees of visual field)
             Maximum size to present.
    
    orientation : float
                The orientation (in radians) at which to measure the size tuning. (in future this will become automated)
                
    spatial_frequency : float
                      Spatial frequency of the grating.
                      
    temporal_frequency : float
                      Temporal frequency of the grating.

    grating_duration : float
                      The duration of single presentation of a grating.
    
    contrasts : list(float) 
              List of contrasts (expressed as % : 0-100%) at which to measure the orientation tuning.
    
    num_trials : int
               Number of trials each each stimulus is shown.
    """

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
    """
    Measure orientation contrast tuning using. This measures the orientation dependence of the surround of 
    a visual neuron. This is done by stimulating the center of the RF with optimal (spatial,temporal frequency and orientation) 
    sine grating, surrounded by another sinusoidal grating ring whose orientation is varied.
    
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    num_orientations : int
          Number of orientation of the surround to present.
    
    orientation : float 
                The orientation (in radians) at which to present the center stimulus. (in future this will become automated)
    
    center_radius : float 
                  The radius of the center grating disk.
    
    surround_radius : float 
                  The outer radius of the surround grating ring.
                  
        
    spatial_frequency : float
                      Spatial frequency of the center and surround grating.
                      
    temporal_frequency : float
                      Temporal frequency of the center and surround the grating.

    grating_duration : float
                      The duration of single presentation of the stimulus.
    
    contrasts : list(float) 
              List of contrasts (expressed as % : 0-100%) of the center and surround grating.
    
    num_trials : int
               Number of trials each each stimulus is shown.
    """

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
    """
    Stimulate the model with a natural image with simulated eye movement.
        
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.


    stimulus_duration : str
                      The duration of single presentation of the stimulus.
    
    num_trials : int
               Number of trials each each stimulus is shown.
               
    Notes
    -----
    Currently this implementation bound to have the image and the eye path saved in in files *./image_naturelle_HIGH.bmp* and *./eye_path.pickle*.
    In future we need to make this more general.
    """
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
    """
    Stimulate the model with a drifting sine grating with simulated eye movement.
    
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    spatial_frequency : float
                      Spatial frequency of the center and surround grating.
                      
    temporal_frequency : float
                      Temporal frequency of the center and surround the grating.

    duration : float
             The duration of single presentation of the stimulus.
    
    contrast : float 
              Contrast (expressed as % : 0-100%) of the grating.
    
    num_trials : int
               Number of trials each each stimulus is shown.
    """
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
    """
    Measure spontaneous activity while presenting blank stimulus (all pixels set to background luminance).
        
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.


    duration : str
             The duration of single presentation of the stimulus.
    
    num_trials : int
               Number of trials each each stimulus is shown.
    """

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
    
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.


    duration : str
             The duration of single presentation of the stimulus.
    
    sheet_list : int
               The list of sheets in which to do stimulation
               
    recording_configuration_list : list
                                 The list of recording configurations (one per each sheet).
                                 
    lambda_list : list
                List of the means of the Poisson spike train to be injected into the neurons specified in recording_configuration_list (one per each sheet).
    """
    def __init__(self,model,duration,sheet_list,recording_configuration_list,lambda_list):
            Experiment.__init__(self, model)
            from NeuroTools import stgen
            for sheet_name,lamb,rc in zip(sheet_list,lambda_list,recording_configuration_list):
                idlist = rc.generate_idd_list_of_neurons()
                seeds=mozaik.get_seeds((len(idlist),))
                stgens = [stgen.StGen(seed=seeds[i]) for i in xrange(0,len(idlist))]
                generator_functions = [(lambda duration,lamb=lamb,stgen=stgens[i]: stgen.poisson_generator(rate=lamb,t_start=0,t_stop=duration).spike_times) for i in xrange(0,len(idlist))]
                self.exc_spike_stimulators[sheet_name] = (list(idlist),generator_functions)

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
    
    Notes
    -----
    Unlike :class:`.MeasureSpontaneousActivity` this can be used in model with no sensory input sheet.
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
