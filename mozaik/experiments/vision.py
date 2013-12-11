import mozaik
from mozaik.experiments import Experiment
from parameters import ParameterSet
import mozaik.stimuli.vision.topographica_based as topo
import numpy


logger = mozaik.getMozaikLogger()


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


class MeasureLuminanceSensitivity(VisualExperiment):
    """
    Measure luminance sensitivity using flat luminance screen.
    
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.
        
    luminances : list(float) 
              List of luminance (expressed as cd/m^2) at which to measure the response.
    
    step_duration : float
                      The duration of single presentation of a luminance step.
    
    num_trials : int
               Number of trials each each stimulus is shown.
    """

    def __init__(self, model, luminances, step_duration, num_trials):
        VisualExperiment.__init__(self, model)    
        # stimuli creation        
        for l in luminances:
            for k in xrange(0, num_trials):
                self.stimuli.append( topo.Null(
                    frame_duration=7,
                    size_x=model.visual_field.size_x,
                    size_y=model.visual_field.size_y,
                    location_x=0.0,
                    location_y=0.0,
                    density=self.density,
                    background_luminance=l,
                    duration=step_duration,
                    trial=k))

    def do_analysis(self, data_store):
        pass


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
    
    log_spacing : bool
               Whether use logarithmic spaced sizes. By default False, meaning linear spacing 
    """

    def __init__(self, model, num_sizes, max_size, orientation,
                 spatial_frequency, temporal_frequency, grating_duration,
                 contrasts, num_trials, log_spacing=False):
        VisualExperiment.__init__(self, model)    
        # linear or logarithmic spaced sizes
        sizes = xrange(0, num_sizes)                     
        if log_spacing:
            # base2 log of max_size
            base2max = numpy.sqrt(max_size)
            sizes = numpy.logspace(0, base2max, num=num_sizes, base=2.0)  
        # stimuli creation        
        for c in contrasts:
            for s in sizes:
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
                                    radius=s,
                                    spatial_frequency=spatial_frequency,
                                    temporal_frequency=temporal_frequency))

    def do_analysis(self, data_store):
        pass


class MeasureContrastSensitivity(VisualExperiment):
    """
    Measure contrast sensitivity using sinusoidal grating disk.
    
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.
        
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

    def __init__(self, model, size, orientation,
                 spatial_frequency, temporal_frequency, grating_duration,
                 contrasts, num_trials):
        VisualExperiment.__init__(self, model)    
        size = 20.0 #DG: very large!
        # stimuli creation        
        for c in contrasts:
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
                    radius=size,
                    spatial_frequency=spatial_frequency,
                    temporal_frequency=temporal_frequency))

    def do_analysis(self, data_store):
        pass


class MeasureFrequencySensitivity(VisualExperiment):
    """
    Measure frequency sensitivity using sinusoidal grating disk.
    
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.
        
    orientation : float
                The orientation (in radians) at which to measure the size tuning. (in future this will become automated)
                
    spatial_frequency : float
                      Spatial frequency of the grating.
                      
    contrast : float
                      Temporal frequency of the grating.

    grating_duration : float
                      The duration of single presentation of a grating.
    
    spatial_frequencies : list(float) 
              List of contrasts (expressed as % : 0-100%) at which to measure the orientation tuning.
    
    num_trials : int
               Number of trials each each stimulus is shown.
    """

    def __init__(self, model, orientation,
                 spatial_frequencies, temporal_frequency, grating_duration,
                 contrast, num_trials):
        VisualExperiment.__init__(self, model)    
        size = 20.0 #DG: very large!
        # stimuli creation        
        for sf in spatial_frequencies:
            for k in xrange(0, num_trials):
                self.stimuli.append(topo.DriftingSinusoidalGratingDisk(
                    frame_duration=7,
                    size_x=model.visual_field.size_x,
                    size_y=model.visual_field.size_y,
                    location_x=0.0,
                    location_y=0.0,
                    background_luminance=self.background_luminance,
                    contrast = contrast,
                    duration=grating_duration,
                    density=self.density,
                    trial=k,
                    orientation=orientation,
                    radius=size,
                    spatial_frequency=sf,
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


class MeasureSpontaneousActivityWithPoissonStimulation(VisualExperiment):
    """
    Measure spontaneous activity while presenting blank stimulus (all pixels set to background luminance).
    Importantly for the duration of the experiment it will stimulate neurons 
    definded by the recording configurations in recording_configuration
    in the sheets specified in the sheet_list with Poisson spike train of mean 
    frequency determined by the corresponding values in lambda_list via synpases of size weight_list.
    
        
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.


    duration : str
             The duration of single presentation of the stimulus.
    
    sheet_list : int
               The list of sheets in which to do stimulation
    
    drive_period : float (ms)
                 The length of the constant drive, after which it will be linearly taken down to 0 at the end of the stimulation.   
               
    recording_configuration : list
                                 The list of recording configurations (one per each sheet).
                                 
    lambda_list : list
                List of the means of the Poisson spike train to be injected into the neurons specified in recording_configuration_list (one per each sheet).
    
    weight_list : list
                List of spike sizes of the Poisson spike train to be injected into the neurons specified in recording_configuration_list (one per each sheet).                
    """

    def __init__(self,model,duration,sheet_list,drive_period,recording_configuration,lambda_list,weight_list):
            VisualExperiment.__init__(self, model)
            from mozaik.sheets.direct_stimulator import Kick
            
            d  = {}
            for i,sheet in enumerate(sheet_list):
                d[sheet] = [Kick(model.sheets[sheet],ParameterSet({'exc_firing_rate' : lambda_list[i],
                                                      'exc_weight' : weight_list[i],
                                                      'drive_period' : drive_period,
                                                      'population_selector' : recording_configuration})
                                )]
            
            self.direct_stimulation = [d]

            self.stimuli.append(
                        topo.Null(   
                            frame_duration=7, 
                            size_x=model.visual_field.size_x,
                            size_y=model.visual_field.size_y,
                            location_x=0.0,
                            location_y=0.0,
                            background_luminance=self.background_luminance,
                            duration=duration,
                            direct_stimulation_name='Kick',
                            density=self.density,
                            trial=0))    
    def do_analysis(self, data_store):
        pass

    
