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


class MeasureFlatLuminanceSensitivity(VisualExperiment):
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
               Number of trials each stimulus is shown.
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
    
class MeasureSparse(VisualExperiment):
    """
    Basic Sparse Stimulation Experiment
    
    Parameters
    ----------
    model : Model
        The model on which to execute the experiment.
          
    time_per_image : float
        The time it takes for the experiment to change single images 
        Every time_per_image a new instance of sparse noise will be 
        presented

    total_number_images : int
        The total number of images that will be presented
    
    num_trials : int
           Number of trials each each stimulus is shown.
           
    grid_size: dimensionless
           the grid will have grid_size x grid_size spots
           
    experiment_seed :  
     sets a particular seed at the beginning of each experiment
     
     grid: 
     If true makes the patterns stick to a grid, otherwise the 
     center of the pattern is distribuited randomly
    """
    
    
    def __init__(self, model,time_per_image, total_number_of_images, num_trials, seed, grid_size, grid = True):
        VisualExperiment.__init__(self, model)
        for k in xrange(0, num_trials):
           
            self.stimuli.append(topo.SparseNoise(
                            frame_duration=7,
                            time_per_image = time_per_image,
                            duration = total_number_of_images * time_per_image,  
                            size_x=model.visual_field.size_x,
                            size_y=model.visual_field.size_y,
                            location_x=0.0,
                            location_y=0.0, 
                            background_luminance=self.background_luminance,
                            density=self.density,
                            trial = k,
                            experiment_seed = seed,
                            grid_size = grid_size,
                            grid = grid
                          ))
   
    def do_analysis(self, data_store):
        pass

class MeasureDense(VisualExperiment):
    """
    Basic Sparse Stimulation Experiment
    
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    duration: How long will the experiment take
    
    num_trials : int
               Number of trials each each stimulus is shown.
               
     grid_size: dimensionless
        the grid will have grid_size x grid_size spots     
            
    experiment_seed :  
     sets a particular seed at the begining of each experiment
    """
    
    
    def __init__(self, model, time_per_image, total_number_of_images, num_trials, seed, grid_size):
        VisualExperiment.__init__(self, model)
        for k in xrange(0, num_trials):
            self.stimuli.append(topo.DenseNoise(
                            frame_duration=7,
                            time_per_image = time_per_image,
                            duration = total_number_of_images * time_per_image, 
                            size_x=model.visual_field.size_x,
                            size_y=model.visual_field.size_y,
                            location_x=0.0,
                            location_y=0.0, 
                            background_luminance=self.background_luminance,
                            density=self.density,
                            trial = k,
                            experiment_seed = seed,
                            grid_size = grid_size
                          ))
         
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
    Measure size tuning using expanding flat luminance disks and sinusoidal grating disks.
    
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
    
    with_flat : bool
               Whether use also flat luminance disks as stimuli. By default False 
    """

    def __init__(self, model, num_sizes, max_size, orientation,
                 spatial_frequency, temporal_frequency, grating_duration,
                 contrasts, num_trials, log_spacing=False, with_flat=False):
        VisualExperiment.__init__(self, model)    
        # linear or logarithmic spaced sizes
        sizes = xrange(0, num_sizes)                     
        if log_spacing:
            # base2 log of max_size
            base2max = numpy.sqrt(max_size)
            sizes = numpy.logspace(start=-3.0, stop=base2max, num=num_sizes, base=2.0)  
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
                    if with_flat:
                        self.stimuli.append(topo.FlatDisk(
                                    frame_duration=7,
                                    size_x=model.visual_field.size_x,
                                    size_y=model.visual_field.size_y,
                                    location_x=0.0,
                                    location_y=0.0,
                                    background_luminance=self.background_luminance,
                                    disk_luminance = c/self.background_luminance,
                                    duration=grating_duration,
                                    density=self.density,
                                    trial=k,
                                    radius=s))

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
                The orientation (in radians) at which to measure the contrast. (in future this will become automated)
                
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

    def __init__(self, 
                 model, 
                 size,
                 orientation,
                 spatial_frequency, 
                 temporal_frequency, 
                 grating_duration,
                 contrasts, 
                 num_trials):
        VisualExperiment.__init__(self, model)    
        # stimuli creation        
        for c in contrasts:
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
                    orientation=orientation,
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
                
    temporal_frequencies : list(float)
                      Temporal frequency of the gratings.
                      
    contrasts : list(float)
            List of contrasts (expressed as % : 0-100%) at which measure the tuning.

    grating_duration : float
                      The duration of single presentation of a grating.
    
    spatial_frequencies : list(float) 
              List of spatial frequencies of the gratings.
    
    num_trials : int
               Number of trials each each stimulus is shown.

    square : bool
                Whether the stimulus shoul be sinusoidal or square grating
    """

    def __init__(self, model, orientation,
                 spatial_frequencies, temporal_frequencies, contrasts, 
                 grating_duration, num_trials, frame_duration=7, square=False):
        VisualExperiment.__init__(self, model)    
        # stimuli creation        
        for tf in temporal_frequencies:
            for sf in spatial_frequencies:
                for c in contrasts:
                    for k in xrange(0, num_trials):
                        if square:
                            self.stimuli.append(topo.FullfieldDriftingSquareGrating(
                                frame_duration=frame_duration,
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
                                spatial_frequency=sf,
                                temporal_frequency=tf))
                        else:
                            self.stimuli.append(topo.FullfieldDriftingSinusoidalGrating(
                                frame_duration=frame_duration,
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
                                spatial_frequency=sf,
                                temporal_frequency=tf))

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


class MeasureFeatureInducedCorrelation(VisualExperiment):
    """
    Measure feature-induced correlation between a couple of neurons (separated by some degrees in visual space) using square grating disk and flashing squares.
    
    Parameters
    ----------
    model : Model
            The model on which to execute the experiment.
                
    temporal_frequencies : list(float)
            Temporal frequency of the gratings.
                      
    contrasts : list(float)
            List of contrasts (expressed as % : 0-100%) at which measure the tuning.

    grating_duration : float
            The duration of single presentation of a grating.
    
    spatial_frequencies : list(float) 
            List of spatial frequencies of the gratings.

    separation : float
            The separation between the two neurons in degrees of visual space.
    
    num_trials : int
            Number of trials each each stimulus is shown.
    """

    def __init__(self, model, spatial_frequencies, temporal_frequency, separation, contrast, 
                 exp_duration, num_trials, frame_duration=7):
        VisualExperiment.__init__(self, model)
         # the orientation is fixed to horizontal
        orientation = 0 #numpy.pi/2
        # SQUARED GRATINGS       
        for sf in spatial_frequencies:
            for k in xrange(0, num_trials):
                self.stimuli.append(
                    topo.FullfieldDriftingSquareGrating(
                        frame_duration = frame_duration,
                        size_x=model.visual_field.size_x,
                        size_y=model.visual_field.size_y,
                        location_x=0.0,
                        location_y=0.0,
                        background_luminance=self.background_luminance,
                        contrast = contrast,
                        duration=exp_duration,
                        density=self.density,
                        trial=k,
                        orientation=orientation,
                        spatial_frequency=sf,
                        temporal_frequency=temporal_frequency
                    )
                )
        # FLASHING SQUARES
        # the spatial_frequencies matters because squares sizes is established using the spatial frequency as for the drifting grating
        for sf in spatial_frequencies:
            for k in xrange(0, num_trials):
                self.stimuli.append(
                    topo.FlashingSquares(
                        frame_duration = frame_duration,
                        size_x=model.visual_field.size_x,
                        size_y=model.visual_field.size_y,
                        location_x=0.0,
                        location_y=0.0,
                        background_luminance=self.background_luminance,
                        contrast = contrast,
                        separation = separation,
                        separated = True,
                        density = self.density,
                        trial = k,
                        duration=exp_duration,
                        orientation = orientation*2,
                        spatial_frequency = sf,
                        temporal_frequency = temporal_frequency
                    )
                )

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
