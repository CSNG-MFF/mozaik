import mozaik
from mozaik.experiments import Experiment
from parameters import ParameterSet
import mozaik.stimuli.vision.topographica_based as topo

# import mozaik.stimuli.vision.texture_based as textu #vf
import numpy
from mozaik.stimuli import InternalStimulus
from mozaik.tools.distribution_parametrization import (
    ParameterWithUnitsAndPeriod,
    MozaikExtendedParameterSet,
)

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

    def __init__(self, model, parameters):
        Experiment.__init__(self, model, parameters)
        self.background_luminance = model.input_space.background_luminance

        # JAHACK: This is kind of a hack now. There needs to be generally defined interface of what is the spatial and temporal resolution of a visual input layer
        # possibly in the future we could force the visual_space to have resolution, perhaps something like native_resolution parameter!?
        self.density = (
            1 / self.model.input_layer.parameters.receptive_field.spatial_resolution
        )  # in pixels per degree of visual space
        self.frame_duration = (
            self.model.input_space.parameters.update_interval
        )  # in pixels per degree of visual space


class MeasureFlatLuminanceSensitivity(VisualExperiment):
    """
    Measure luminance sensitivity using flat luminance screen.

    This experiment will measure luminance sensitivity by presenting a series of full-field 
    constant stimulations (i.e. all pixels of the virtual visual space will be set to a 
    constant value) of different magnitudes. The user can specify the luminance levels that
    should be presented (see the *luminances*) parameter, the length  of presentation of 
    individual steps (*step_duration* parameter), and number of trials (*num_trials* parameter).
    
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    Other parameters
    ----------------
        
    luminances : list(float) 
              List of luminance (expressed as cd/m^2) at which to measure the response.
    
    step_duration : float
                      The duration in miliseconds of single presentation of a luminance step.
    
    num_trials : int
               Number of trials each stimulus is shown.
    """

    required_parameters = ParameterSet(
        {"luminances": list, "step_duration": float, "num_trials": int,}
    )

    def __init__(self, model, parameters):
        VisualExperiment.__init__(self, model, parameters)

        # stimuli creation
        for l in self.parameters.luminances:
            for k in range(0, self.parameters.num_trials):
                self.stimuli.append(
                    topo.Null(
                        frame_duration=self.frame_duration,
                        size_x=model.visual_field.size_x,
                        size_y=model.visual_field.size_y,
                        location_x=0.0,
                        location_y=0.0,
                        density=self.density,
                        background_luminance=l,
                        duration=self.parameters.step_duration,
                        trial=k,
                    )
                )

    def do_analysis(self, data_store):
        pass


class MeasureSparse(VisualExperiment):
    """
    Sparse noise stimulation experiments.

    This experiment will show a series of images formed by a single 
    circle (dot) which will be presented in a random position in each trial.
    
    Parameter
    ----------
    model : Model
        The model on which to execute the experiment.

    Other parameters
    ----------------

    time_per_image : float
        The time it takes for the experiment to change single images 
        Every time_per_image a new instance of sparse noise will be 
        presented

    total_number_images : int
        The total number of images that will be presented
    
    num_trials : int
           Number of trials each each stimulus is shown.
           
    grid_size: int
           the grid will have grid_size x grid_size spots
           
    experiment_seed : int
     sets a particular seed at the beginning of each experiment
     
    grid: bool
     If true makes the patterns stick to a grid, otherwise the 
     center of the pattern is distribuited randomly
    """

    required_parameters = ParameterSet(
        {
            "time_per_image": float,
            "total_number_of_images": int,
            "num_trials": int,
            "experiment_seed": int,
            "grid_size": int,
            "grid": bool,
        }
    )

    def __init__(self, model, parameters):
        VisualExperiment.__init__(self, model, parameters)

        for k in range(0, self.parameters.num_trials):

            self.stimuli.append(
                topo.SparseNoise(
                    frame_duration=self.frame_duration,
                    time_per_image=self.parameters.time_per_image,
                    duration=self.parameters.total_number_of_images
                    * self.parameters.time_per_image,
                    size_x=model.visual_field.size_x,
                    size_y=model.visual_field.size_y,
                    location_x=0.0,
                    location_y=0.0,
                    background_luminance=self.background_luminance,
                    density=self.density,
                    trial=k,
                    experiment_seed=self.parameters.experiment_seed,
                    grid_size=self.parameters.grid_size,
                    grid=self.parameters.grid,
                )
            )

    def do_analysis(self, data_store):
        pass


class MeasureDense(VisualExperiment):
    """
    Dense noise stimulation experiments.
    
    This experiment will show a series of images formed by a grid
    of 'pixels', in each trial randomly set to 0 or maximum luminance.


    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    Other parameters
    ----------------

    time_per_image : float
        The time it takes for the experiment to change single images 
        Every time_per_image a new instance of sparse noise will be 
        presented

    total_number_images : int
        The total number of images that will be presented
    
    num_trials : int
           Number of trials each each stimulus is shown.
           
    grid_size: int
           the grid will have grid_size x grid_size spots
           
    experiment_seed : int
     sets a particular seed at the beginning of each experiment
    """

    required_parameters = ParameterSet(
        {
            "time_per_image": float,
            "total_number_of_images": int,
            "num_trials": int,
            "experiment_seed": int,
            "grid_size": int,
        }
    )

    def __init__(self, model, parameters):
        VisualExperiment.__init__(self, model, parameters)

        for k in range(0, self.parameters.num_trials):
            self.stimuli.append(
                topo.DenseNoise(
                    frame_duration=self.frame_duration,
                    time_per_image=self.parameters.time_per_image,
                    duration=self.parameters.total_number_of_images
                    * self.parameters.time_per_image,
                    size_x=model.visual_field.size_x,
                    size_y=model.visual_field.size_y,
                    location_x=0.0,
                    location_y=0.0,
                    background_luminance=self.background_luminance,
                    density=self.density,
                    trial=k,
                    experiment_seed=self.parameters.experiment_seed,
                    grid_size=self.parameters.grid_size,
                )
            )

    def do_analysis(self, data_store):
        pass


class MeasureOrientationTuningFullfield(VisualExperiment):
    """
    Measure orientation tuning using a fullfiled sinusoidal grating.

    This experiment will show a series of full-field sinusoidal gratings 
    that vary in orientation, while the other parameters remain constant.
    
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    Other parameters
    ----------------

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

    required_parameters = ParameterSet(
        {
            "num_orientations": int,
            "spatial_frequency": float,
            "temporal_frequency": float,
            "grating_duration": float,
            "contrasts": list,
            "num_trials": int,
        }
    )

    def __init__(self, model, parameters):
        VisualExperiment.__init__(self, model, parameters)
        for c in self.parameters.contrasts:
            for i in range(0, self.parameters.num_orientations):
                for k in range(0, self.parameters.num_trials):
                    self.stimuli.append(
                        topo.FullfieldDriftingSinusoidalGrating(
                            frame_duration=self.frame_duration,
                            size_x=model.visual_field.size_x,
                            size_y=model.visual_field.size_y,
                            location_x=0.0,
                            location_y=0.0,
                            background_luminance=self.background_luminance,
                            contrast=c,
                            duration=self.parameters.grating_duration,
                            density=self.density,
                            trial=k,
                            orientation=numpy.pi / self.parameters.num_orientations * i,
                            spatial_frequency=self.parameters.spatial_frequency,
                            temporal_frequency=self.parameters.temporal_frequency,
                        )
                    )

    def do_analysis(self, data_store):
        pass


class MeasureOrientationTuningFullfieldA(VisualExperiment):
    """
    Measure orientation tuning using a fullfiled sinusoidal grating.

    This experiment will show a series of full-field sinusoidal gratings
    that vary in orientation, while the other parameters remain constant.
    
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    Other parameters
    ----------------

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

    required_parameters = ParameterSet(
        {
            "num_orientations": int,
            "spatial_frequency": float,
            "temporal_frequency": float,
            "grating_duration": float,
            "contrasts": list,
            "num_trials": int,
            "offset_time": float,
            "onset_time": float,
        }
    )

    def __init__(self, model, parameters):
        VisualExperiment.__init__(self, model, parameters)
        for c in self.parameters.contrasts:
            for i in range(0, self.parameters.num_orientations):
                for k in range(0, self.parameters.num_trials):
                    self.stimuli.append(
                        topo.FullfieldDriftingSinusoidalGratingA(
                            frame_duration=self.frame_duration,
                            size_x=model.visual_field.size_x,
                            size_y=model.visual_field.size_y,
                            offset_time=self.parameters.offset_time,
                            onset_time=self.parameters.onset_time,
                            location_x=0.0,
                            location_y=0.0,
                            background_luminance=self.background_luminance,
                            contrast=c,
                            duration=self.parameters.grating_duration,
                            density=self.density,
                            trial=k,
                            orientation=numpy.pi / self.parameters.num_orientations * i,
                            spatial_frequency=self.parameters.spatial_frequency,
                            temporal_frequency=self.parameters.temporal_frequency,
                        )
                    )

    def do_analysis(self, data_store):
        pass


class MeasureSizeTuning(VisualExperiment):
    """
    Size tuning experiment.

    This experiment will show a series of sinusoidal gratings or constant flat stimuli 
    (see *with_flat* parameter) confined to an apparature whose radius will vary.

    
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    Other parameters
    ----------------

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
               Whether use flat luminance disks as stimuli. If not it is the standard grating stimulus.
    """

    required_parameters = ParameterSet(
        {
            "num_sizes": int,
            "max_size": float,
            "orientation": float,
            "spatial_frequency": float,
            "temporal_frequency": float,
            "grating_duration": float,
            "contrasts": list,
            "num_trials": int,
            "log_spacing": bool,
        }
    )

    def __init__(self, model, parameters):
        VisualExperiment.__init__(self, model, parameters)

        # linear or logarithmic spaced sizes
        if self.parameters.log_spacing:
            base2max = numpy.log2(self.parameters.max_size)
            sizes = numpy.logspace(
                start=-3.0, stop=base2max, num=self.parameters.num_sizes, base=2.0
            )
        else:
            sizes = numpy.linspace(
                0, self.parameters.max_size, self.parameters.num_sizes
            )

        # stimuli creation
        for c in self.parameters.contrasts:
            for s in sizes:
                for k in range(0, self.parameters.num_trials):
                    self.stimuli.append(
                        topo.DriftingSinusoidalGratingDisk(
                            frame_duration=self.frame_duration,
                            size_x=model.visual_field.size_x,
                            size_y=model.visual_field.size_y,
                            location_x=0.0,
                            location_y=0.0,
                            background_luminance=self.background_luminance,
                            contrast=c,
                            duration=self.parameters.grating_duration,
                            density=self.density,
                            trial=k,
                            orientation=self.parameters.orientation,
                            radius=s,
                            spatial_frequency=self.parameters.spatial_frequency,
                            temporal_frequency=self.parameters.temporal_frequency,
                        )
                    )

    def do_analysis(self, data_store):
        pass


class MeasureContrastSensitivity(VisualExperiment):
    """
    Measure contrast sensitivity using sinusoidal gratings.

    This experiment shows a series of full-field sinusoidal gratings of varying 
    contrast. Using the responses to these stimuli one can construct the contrast
    sensitivity tuning curve for the measured neurons.
    
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    Other parameters
    ----------------
        
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

    required_parameters = ParameterSet(
        {
            "orientation": float,
            "spatial_frequency": float,
            "temporal_frequency": float,
            "grating_duration": float,
            "contrasts": list,
            "num_trials": int,
        }
    )

    def __init__(self, model, parameters):
        VisualExperiment.__init__(self, model, parameters)

        # stimuli creation
        for c in self.parameters.contrasts:
            for k in range(0, self.parameters.num_trials):
                self.stimuli.append(
                    topo.FullfieldDriftingSinusoidalGrating(
                        frame_duration=self.frame_duration,
                        size_x=model.visual_field.size_x,
                        size_y=model.visual_field.size_y,
                        location_x=0.0,
                        location_y=0.0,
                        background_luminance=self.background_luminance,
                        contrast=c,
                        duration=self.parameters.grating_duration,
                        density=self.density,
                        trial=k,
                        orientation=self.parameters.orientation,
                        spatial_frequency=self.parameters.spatial_frequency,
                        temporal_frequency=self.parameters.temporal_frequency,
                    )
                )

    def do_analysis(self, data_store):
        pass


class MeasureContrastSensitivityA(VisualExperiment):
    """
    Measure contrast sensitivity using sinusoidal gratings.

    This experiment shows a series of full-field sinusoidal gratings of varying 
    contrast. Using the responses to these stimuli one can construct the contrast
    sensitivity tuning curve for the measured neurons.
    
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    Other parameters
    ----------------
        
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

    required_parameters = ParameterSet(
        {
            "orientation": float,
            "spatial_frequency": float,
            "temporal_frequency": float,
            "grating_duration": float,
            "contrasts": list,
            "num_trials": int,
            "offset_time": float,
            "onset_time": float,
        }
    )

    def __init__(self, model, parameters):
        VisualExperiment.__init__(self, model, parameters)

        # stimuli creation
        for c in self.parameters.contrasts:
            for k in range(0, self.parameters.num_trials):
                self.stimuli.append(
                    topo.FullfieldDriftingSinusoidalGratingA(
                        frame_duration=self.frame_duration,
                        size_x=model.visual_field.size_x,
                        size_y=model.visual_field.size_y,
                        location_x=0.0,
                        location_y=0.0,
                        background_luminance=self.background_luminance,
                        contrast=c,
                        duration=self.parameters.grating_duration,
                        density=self.density,
                        trial=k,
                        offset_time=self.parameters.offset_time,
                        onset_time=self.parameters.onset_time,
                        orientation=self.parameters.orientation,
                        spatial_frequency=self.parameters.spatial_frequency,
                        temporal_frequency=self.parameters.temporal_frequency,
                    )
                )

    def do_analysis(self, data_store):
        pass


class MeasureFrequencySensitivity(VisualExperiment):
    """
    Measure frequency sensitivity using sinusoidal grating disk.
    
    This experiment shows a series of full-field drifting sinusoidal gratings 
    of varying spatial and temporal frequencies. Using the responses to these 
    stimuli one can construct the spatial and/or temporal frequency tuning 
    curve for the measured neurons.

    

    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    Other parameters
    ----------------
        
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

    required_parameters = ParameterSet(
        {
            "orientation": float,
            "spatial_frequency": list,
            "temporal_frequency": list,
            "grating_duration": float,
            "contrasts": list,
            "num_trials": int,
            "square": bool,
        }
    )

    def __init__(self, model, parameters):
        VisualExperiment.__init__(self, model, parameters)

        # stimuli creation
        for tf in self.parameters.temporal_frequencies:
            for sf in self.parameters.spatial_frequencies:
                for c in self.parameters.contrasts:
                    for k in range(0, self.parameters.num_trials):
                        if self.parameters.square:
                            self.stimuli.append(
                                topo.FullfieldDriftingSquareGrating(
                                    frame_duration=self.frame_duration,
                                    size_x=model.visual_field.size_x,
                                    size_y=model.visual_field.size_y,
                                    location_x=0.0,
                                    location_y=0.0,
                                    background_luminance=self.background_luminance,
                                    contrast=c,
                                    duration=self.parameters.grating_duration,
                                    density=self.density,
                                    trial=k,
                                    orientation=self.parameters.orientation,
                                    spatial_frequency=sf,
                                    temporal_frequency=tf,
                                )
                            )
                        else:
                            self.stimuli.append(
                                topo.FullfieldDriftingSinusoidalGrating(
                                    frame_duration=self.frame_duration,
                                    size_x=model.visual_field.size_x,
                                    size_y=model.visual_field.size_y,
                                    location_x=0.0,
                                    location_y=0.0,
                                    background_luminance=self.background_luminance,
                                    contrast=c,
                                    duration=self.parameters.grating_duration,
                                    density=self.density,
                                    trial=k,
                                    orientation=self.parameters.orientation,
                                    spatial_frequency=sf,
                                    temporal_frequency=tf,
                                )
                            )

    def do_analysis(self, data_store):
        pass


class MeasureOrientationContrastTuning(VisualExperiment):
    """
    Measure orientation contrast tuning using. 

    This measures the orientation dependence of the RF surround 
    of a neuron. This is done by stimulating the center of the RF 
    with optimal (spatial,temporal frequency and orientation) 
    sine grating, surrounded by another sinusoidal grating 
    ring whose orientation is varied.
    
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    Other parameters
    ----------------

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

    required_parameters = ParameterSet(
        {
            "num_orientations": int,
            "orientation": float,
            "center_radius": float,
            "surround_radius": float,
            "spatial_frequency": float,
            "temporal_frequency": float,
            "grating_duration": float,
            "contrasts": list,
            "num_trials": int,
        }
    )

    def __init__(self, model, parameters):
        VisualExperiment.__init__(self, model, parameters)

        for c in self.parameters.contrasts:
            for i in range(0, self.parameters.num_orientations):
                for k in range(0, self.parameters.num_trials):
                    self.stimuli.append(
                        topo.DriftingSinusoidalGratingCenterSurroundStimulus(
                            frame_duration=self.frame_duration,
                            size_x=model.visual_field.size_x,
                            size_y=model.visual_field.size_y,
                            location_x=0.0,
                            location_y=0.0,
                            background_luminance=self.background_luminance,
                            contrast=c,
                            duration=self.parameters.grating_duration,
                            density=self.density,
                            trial=k,
                            center_orientation=self.parameters.orientation,
                            surround_orientation=numpy.pi
                            / self.parameters.num_orientations
                            * i,
                            gap=0,
                            center_radius=self.parameters.center_radius,
                            surround_radius=self.parameters.surround_radius,
                            spatial_frequency=self.parameters.spatial_frequency,
                            temporal_frequency=self.parameters.temporal_frequency,
                        )
                    )

    def do_analysis(self, data_store):
        pass


class MeasureFeatureInducedCorrelation(VisualExperiment):
    """
    Feature-induced inter-neurons correlations.

    This experiment shows a sequence of two square grating disks followed by 
    a sequence of flashing squares (see parameter **) that are separated in 
    visual space by a constant distance. The spatial and temporal frequency 
    will be varied.

    
    Parameters
    ----------
    model : Model
            The model on which to execute the experiment.

    Other parameters
    ----------------
                
    temporal_frequencies : list(float)
            Temporal frequency of the gratings.
                      
    contrast : float
            Contrast (expressed as % : 0-100%) at which to performe measurument.

    grating_duration : float
            The duration of single presentation of a grating.
    
    spatial_frequencies : list(float) 
            List of spatial frequencies of the gratings.

    separation : float
            The separation between the two neurons in degrees of visual space.
    
    num_trials : int
            Number of trials each each stimulus is shown.
    """

    required_parameters = ParameterSet(
        {
            "spatial_frequencies": list,
            "temporal_frequencies": list,
            "grating_duration": float,
            "contrasts": list,
            "separation": float,
            "num_trials": int,
        }
    )

    def __init__(self, model, parameters):
        VisualExperiment.__init__(self, model, parameters)

        # the orientation is fixed to horizontal
        orientation = 0  # numpy.pi/2
        # SQUARED GRATINGS
        for sf in self.parameters.spatial_frequencies:
            for k in range(0, self.parameters.num_trials):
                self.stimuli.append(
                    topo.FullfieldDriftingSquareGrating(
                        frame_duration=self.frame_duration,
                        size_x=model.visual_field.size_x,
                        size_y=model.visual_field.size_y,
                        location_x=0.0,
                        location_y=0.0,
                        background_luminance=self.background_luminance,
                        contrast=self.parameters.contrast,
                        duration=self.parameters.grating_duration,
                        density=self.density,
                        trial=k,
                        orientation=orientation,
                        spatial_frequency=self.parameters.sf,
                        temporal_frequency=self.parameters.temporal_frequency,
                    )
                )
        # FLASHING SQUARES
        # the spatial_frequencies matters because squares sizes is established using the spatial frequency as for the drifting grating
        for sf in self.parameters.spatial_frequencies:
            for k in range(0, self.parameters.num_trials):
                self.stimuli.append(
                    topo.FlashingSquares(
                        frame_duration=self.frame_duration,
                        size_x=model.visual_field.size_x,
                        size_y=model.visual_field.size_y,
                        location_x=0.0,
                        location_y=0.0,
                        background_luminance=self.background_luminance,
                        contrast=self.parameters.contrast,
                        separation=self.parameters.separation,
                        separated=True,
                        density=self.density,
                        trial=k,
                        duration=self.parameters.grating_duration,
                        orientation=orientation * 2,
                        spatial_frequency=sf,
                        temporal_frequency=self.parameters.temporal_frequency,
                    )
                )

    def do_analysis(self, data_store):
        pass


class MeasureNaturalImagesWithEyeMovement(VisualExperiment):
    """
    Stimulate the model with a natural image with simulated eye movement.

    This experiment presents a movie that is generated by translating a 
    static image along a pre-specified path (presumably containing path
    that corresponds to eye-movements).
        
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.


    Other parameters
    ----------------
    
    stimulus_duration : str
                      The duration of single presentation of the stimulus.
    
    num_trials : int
               Number of trials each each stimulus is shown.
               
    Notes
    -----
    Currently this implementation bound to have the image and the eye path saved in in files *./image_naturelle_HIGH.bmp* and *./eye_path.pickle*.
    In future we need to make this more general.
    """

    required_parameters = ParameterSet({"stimulus_duration": float, "num_trials": int,})

    def __init__(self, model, parameters):
        VisualExperiment.__init__(self, model, parameters)

        for k in range(0, self.parameters.num_trials):
            self.stimuli.append(
                topo.NaturalImageWithEyeMovement(
                    frame_duration=self.frame_duration,
                    size_x=model.visual_field.size_x,
                    size_y=model.visual_field.size_y,
                    location_x=0.0,
                    location_y=0.0,
                    background_luminance=self.background_luminance,
                    duration=self.parameters.stimulus_duration,
                    density=self.density,
                    trial=k,
                    size=60,  # x size of image
                    eye_movement_period=6.66,  # eye movement period
                    eye_path_location="./eye_path.pickle",
                    image_location="./image_naturelle_HIGHC.bmp",
                )
            )

    def do_analysis(self, data_store):
        pass


class MeasureDriftingSineGratingWithEyeMovement(VisualExperiment):
    """
    Present drifting sine grating with simulated eye movement.

    This experiment presents a movie that is generated by translating a 
    full-field drifting sinusoidal grating movie along a pre-specified path 
    (presumably containing path that corresponds to eye-movements).
    
    
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    Other parameters
    ----------------

    spatial_frequency : float
                      Spatial frequency of the center and surround grating.
                      
    temporal_frequency : float
                      Temporal frequency of the center and surround the grating.

    grating_duration : float
             The duration of single presentation of the stimulus.
    
    contrast : float 
              Contrast (expressed as % : 0-100%) of the grating.
    
    num_trials : int
               Number of trials each each stimulus is shown.
    """

    required_parameters = ParameterSet(
        {
            "spatial_frequency": float,
            "temporal_frequency": float,
            "grating_duration": float,
            "contrast": float,
            "num_trials": int,
        }
    )

    def __init__(self, model, parameters):
        VisualExperiment.__init__(self, model, parameters)

        for k in range(0, self.parameters.num_trials):
            self.stimuli.append(
                topo.DriftingGratingWithEyeMovement(
                    frame_duration=self.frame_duration,
                    size_x=model.visual_field.size_x,
                    size_y=model.visual_field.size_y,
                    location_x=0.0,
                    location_y=0.0,
                    background_luminance=self.background_luminance,
                    duration=self.parameters.grating_duration,
                    density=self.density,
                    trial=k,
                    contrast=self.parameters.contrast,
                    eye_movement_period=6.66,  # eye movement period
                    eye_path_location="./eye_path.pickle",
                    orientation=0,
                    spatial_frequency=self.parameters.spatial_frequency,
                    temporal_frequency=self.parameters.temporal_frequency,
                )
            )

    def do_analysis(self, data_store):
        pass


class MeasureSpontaneousActivity(VisualExperiment):
    """
    Measure spontaneous activity.

    This experiment presents a blank stimulus (all pixels set to background luminance).
        
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    Other parameters
    ----------------

    duration : str
             The duration of single presentation of the stimulus.
    
    num_trials : int
               Number of trials each each stimulus is shown.
    """

    required_parameters = ParameterSet({"duration": float, "num_trials": int,})

    def __init__(self, model, parameters):
        VisualExperiment.__init__(self, model, parameters)

        for k in range(0, self.parameters.num_trials):
            self.stimuli.append(
                topo.Null(
                    frame_duration=self.frame_duration,
                    size_x=model.visual_field.size_x,
                    size_y=model.visual_field.size_y,
                    location_x=0.0,
                    location_y=0.0,
                    background_luminance=self.background_luminance,
                    duration=self.parameters.duration,
                    density=self.density,
                    trial=k,
                )
            )

    def do_analysis(self, data_store):
        pass


class MapPhaseResponseWithBarStimulus(VisualExperiment):
    """
    Map RF with a bar stimuli.

    This experiment presents a series of flashed bars at pre-specified range of 
    displacements from the center along the line that is  perpendicularly to 
    the elongated axis of the bars. This is an experiment commonly used to obtain
    1D approximation of the 2D receptive field of orientation selective cortical
    cells.
    
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    Other parameters
    ----------------
    model : Model
          The model on which to execute the experiment.
    
    x : float
      The x corrdinates (of center) of the area in which the mapping will be done.

    y : float
      The y corrdinates (of center) of the area in which the mapping will be done.
        
    length : float
          The length of the bar.
    
    width : float
          The width of the bar.
             
    orientation : float
                The orientation of the bar.

    max_offset : float
               The maximum offset from the central position (defined by x and y) prependicular to the length of the bar at which the bars will be flashed.

    steps : int
         The number of steps in which the bars will be flashed between the two extreme positions defined by the max_offset parameter, along the axis prependicular to the length of the bar.
    
    duration : float
             The duration of single presentation of the stimulus.
    
    flash_duration : float
             The duration of the presence of the bar.
    
    relative_luminance : float 
              Luminance of the bar relative to background luminance. 0 is dark, 1.0 is double the background luminance.
    
    num_trials : int
               Number of trials each each stimulus is shown.
    """

    required_parameters = ParameterSet(
        {
            "x": float,
            "y": float,
            "length": float,
            "width": float,
            "orientation": float,
            "max_offset": float,
            "steps": int,
            "duration": float,
            "flash_duration": float,
            "relative_luminance": float,
            "num_trials": int,
        }
    )

    def __init__(self, model, parameters):
        VisualExperiment.__init__(self, model, parameters)
        for k in range(0, self.parameters.num_trials):
            for s in range(0, self.parameters.steps):
                self.stimuli.append(
                    topo.FlashedBar(
                        frame_duration=self.frame_duration,
                        size_x=model.visual_field.size_x,
                        size_y=model.visual_field.size_y,
                        location_x=0.0,
                        location_y=0.0,
                        background_luminance=self.background_luminance,
                        duration=self.parameters.duration,
                        density=self.density,
                        relative_luminance=self.parameters.relative_luminance,
                        orientation=self.parameters.orientation,
                        width=self.parameters.width,
                        length=self.parameters.length,
                        flash_duration=self.parameters.flash_duration,
                        x=self.parameters.x
                        + numpy.cos(self.parameters.orientation + numpy.pi / 2)
                        * (
                            -self.parameters.max_offset
                            + (2 * self.parameters.max_offset)
                            / (self.parameters.steps - 1)
                            * s
                        ),
                        y=self.parameters.y
                        + numpy.sin(self.parameters.orientation + numpy.pi / 2)
                        * (
                            -self.parameters.max_offset
                            + (2 * self.parameters.max_offset)
                            / (self.parameters.steps - 1)
                            * s
                        ),
                        trial=k,
                    )
                )

    def do_analysis(self, data_store):
        pass


class CorticalStimulationWithStimulatorArrayAndHomogeneousOrientedStimulus(Experiment):
    """
    Stimulation with artificial stimulator array simulating homogeneously
    oriented visual stimulus.  

    This experiment creates a array of artificial stimulators covering an area of 
    cortex, and than stimulates the array based on the orientation preference of 
    neurons around the given stimulator, such that the stimulation resambles 
    presentation uniformly oriented stimulus, e.g. sinusoidal grating.
    
    This experiment does not show any actual visual stimulus.
    
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    Other parameters
    ----------------
  
    sheet_list : int
               The list of sheets in which to do stimulation.
    """

    required_parameters = ParameterSet(
        {
            "sheet_list": list,
            "num_trials": int,
            "localstimulationarray_parameters": ParameterSet,
        }
    )

    def __init__(self, model, parameters):
        Experiment.__init__(self, model, parameters)
        from mozaik.sheets.direct_stimulator import LocalStimulatorArrayChR

        d = {}
        for sheet in self.parameters.sheet_list:
            d[sheet] = [
                LocalStimulatorArrayChR(
                    model.sheets[sheet],
                    self.parameters.localstimulationarray_parameters,
                )
            ]

        self.direct_stimulation = []

        for i in range(0, self.parameters.num_trials):
            self.direct_stimulation.append(d)
            self.stimuli.append(
                InternalStimulus(
                    frame_duration=self.parameters.localstimulationarray_parameters.stimulating_signal_parameters.duration,
                    duration=self.parameters.localstimulationarray_parameters.stimulating_signal_parameters.duration,
                    trial=i,
                    direct_stimulation_name="LocalStimulatorArray",
                    direct_stimulation_parameters=self.parameters.localstimulationarray_parameters,
                )
            )


class CorticalStimulationWithStimulatorArrayAndOrientationTuningProtocol(Experiment):
    """
    Stimulation with artificial stimulator array simulating homogeneously
    oriented visual stimulus.  

    This experiment creates a array of artificial stimulators covering an area of 
    cortex, and than stimulates the array based on the orientation preference of 
    neurons around the given stimulator, such that the stimulation resambles 
    presentation uniformly oriented stimulus, e.g. sinusoidal grating.
    
    This experiment does not show any actual visual stimulus.
    
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    Other parameters
    ----------------
  
    sheet_list : int
               The list of sheets in which to do stimulation.
    """

    required_parameters = ParameterSet(
        {
            "sheet_list": list,
            "num_trials": int,
            "num_orientations": int,
            "intensities": list,
            "localstimulationarray_parameters": ParameterSet,
        }
    )

    def __init__(self, model, parameters):
        Experiment.__init__(self, model, parameters)
        from mozaik.sheets.direct_stimulator import LocalStimulatorArrayChR

        self.direct_stimulation = []
        first = True

        for s in self.parameters.intensities:
            for i in range(self.parameters.num_orientations):
                p = MozaikExtendedParameterSet(
                    self.parameters.localstimulationarray_parameters.tree_copy().as_dict()
                )
                p.stimulating_signal_parameters.orientation = ParameterWithUnitsAndPeriod(
                    numpy.pi / self.parameters.num_orientations * i, period=numpy.pi
                )
                p.stimulating_signal_parameters.scale = ParameterWithUnitsAndPeriod(
                    float(s), period=None
                )
                d = {}
                if first:
                    for sheet in self.parameters.sheet_list:
                        d[sheet] = [LocalStimulatorArrayChR(model.sheets[sheet], p)]
                    first = False
                else:
                    for sheet in self.parameters.sheet_list:
                        d[sheet] = [
                            LocalStimulatorArrayChR(
                                model.sheets[sheet],
                                p,
                                shared_scs=self.direct_stimulation[0][sheet][0].scs,
                            )
                        ]

                for i in range(0, self.parameters.num_trials):
                    self.direct_stimulation.append(d)
                    self.stimuli.append(
                        InternalStimulus(
                            frame_duration=self.parameters.localstimulationarray_parameters.stimulating_signal_parameters.duration,
                            duration=self.parameters.localstimulationarray_parameters.stimulating_signal_parameters.duration,
                            trial=i,
                            direct_stimulation_name="LocalStimulatorArray",
                            direct_stimulation_parameters=p,
                        )
                    )


class CorticalStimulationWithStimulatorArrayAndOrientationTuningProtocol_ContrastBased(
    Experiment
):
    """
    Stimulation with artificial stimulator array simulating homogeneously
    oriented visual stimulus.  

    This experiment creates a array of artificial stimulators covering an area of 
    cortex, and than stimulates the array based on the orientation preference of 
    neurons around the given stimulator, such that the stimulation resambles 
    presentation uniformly oriented stimulus, e.g. sinusoidal grating.
    
    This experiment does not show any actual visual stimulus.
    
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    Other parameters
    ----------------
  
    sheet_list : int
               The list of sheets in which to do stimulation.
    """

    required_parameters = ParameterSet(
        {
            "sheet_list": list,
            "num_trials": int,
            "num_orientations": int,
            "contrasts": list,
            "localstimulationarray_parameters": ParameterSet,
        }
    )

    def __init__(self, model, parameters):
        Experiment.__init__(self, model, parameters)
        from mozaik.sheets.direct_stimulator import LocalStimulatorArrayChR

        self.direct_stimulation = []
        first = True

        for c in self.parameters.contrasts:
            for i in range(self.parameters.num_orientations):
                p = MozaikExtendedParameterSet(
                    self.parameters.localstimulationarray_parameters.tree_copy().as_dict()
                )
                p.stimulating_signal_parameters.orientation = ParameterWithUnitsAndPeriod(
                    numpy.pi / self.parameters.num_orientations * i, period=numpy.pi
                )
                p.stimulating_signal_parameters.contrast = ParameterWithUnitsAndPeriod(
                    float(c), period=None
                )
                d = {}
                if first:
                    for sheet in self.parameters.sheet_list:
                        d[sheet] = [LocalStimulatorArrayChR(model.sheets[sheet], p)]
                    first = False
                else:
                    for sheet in self.parameters.sheet_list:
                        d[sheet] = [
                            LocalStimulatorArrayChR(
                                model.sheets[sheet],
                                p,
                                shared_scs=self.direct_stimulation[0][sheet][0].scs,
                            )
                        ]

                for i in range(0, self.parameters.num_trials):
                    self.direct_stimulation.append(d)
                    self.stimuli.append(
                        InternalStimulus(
                            frame_duration=self.parameters.localstimulationarray_parameters.stimulating_signal_parameters.duration,
                            duration=self.parameters.localstimulationarray_parameters.stimulating_signal_parameters.duration,
                            trial=i,
                            direct_stimulation_name="LocalStimulatorArray",
                            direct_stimulation_parameters=p,
                        )
                    )


class VonDerHeydtIllusoryBarProtocol(VisualExperiment):
    """
    An illusory bar from Von Der Heydt et al. 1989.

    Von Der Heydt, R., & Peterhans, E. (1989). Mechanisms of contour perception in monkey visual cortex. I. Lines of pattern discontinuity. Journal of Neuroscience, 9(5), 1731–1748. Retrieved from https://www.jneurosci.org/content/jneuro/9/5/1731.full.pdf
    
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    Other parameters
    ----------------
    model : Model
          The model on which to execute the experiment.
    
    x : float
      The x corrdinates (of center) of the area in which the mapping will be done.

    y : float
      The y corrdinates (of center) of the area in which the mapping will be done.
        
    length : float
          The length of the bar.
    
    background_bar_width : float
                         Width of the background bar

    occlusion_bar_width : float
                         Width of the occlusion bar
    bar_width : float
              Width of the bar
             
    orientation : float
                The orientation of the bar.

    duration : float
             The duration of single presentation of the stimulus.
    
    flash_duration : float
             The duration of the presence of the bar.
     
    num_trials : int
               Number of trials each each stimulus is shown.
    """

    required_parameters = ParameterSet(
        {
            "x": float,
            "y": float,
            "length": float,
            "bar_width": float,
            "orientation": float,
            "background_bar_width": float,
            "occlusion_bar_width": list,
            "duration": float,
            "flash_duration": float,
            "num_trials": int,
        }
    )

    def __init__(self, model, parameters):
        VisualExperiment.__init__(self, model, parameters)
        for k in range(0, self.parameters.num_trials):
            for obw in self.parameters.occlusion_bar_width:
                self.stimuli.append(
                    topo.FlashedInterruptedBar(
                        frame_duration=7,
                        size_x=model.visual_field.size_x,
                        size_y=model.visual_field.size_y,
                        location_x=0.0,
                        location_y=0.0,
                        background_luminance=self.background_luminance,
                        duration=self.parameters.duration,
                        bar_width=self.parameters.bar_width,
                        background_bar_width=self.parameters.background_bar_width,
                        occlusion_bar_width=self.parameters.occlusion_bar_width,
                        density=self.density,
                        orientation=self.parameters.orientation,
                        flash_duration=self.parameters.flash_duration,
                        x=self.parameters.x,
                        y=self.parameters.y,
                        trial=k,
                    )
                )

    def do_analysis(self, data_store):
        pass


class MeasureTextureSensitivityFullfield(VisualExperiment):
    """
    Measure sensitivity to second order image correlations using stimuli
    based on a texture image.
    This experiment will show a series of texture based images
    that vary in matched statistics.
    
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.
    Other parameters
    ----------------
    num_images : int
          Number of images of each type to present.
    
    image_path : string
                      Path to initial image.
    image_duration : float
                      The duration of single presentation of an image.
    
    types : list(int) 
              List of types indicating which statistics to match:
                0 - original image
                1 - naturalistic texture image (matched higher order statistics)
                2 - spectrally matched noise (matched marginal statistics only).
    
    num_trials : int
               Number of trials each each stimulus is shown.
    """

    required_parameters = ParameterSet(
        {
            "num_images": int,  # n. of images of each type, different synthesized instances
            "image_path": str,
            "image_duration": float,
            "types": list,
            "num_trials": int,  # n. of same instance
            #'offset_time' : float,
            #'onset_time' : float,
        }
    )

    def __init__(self, model, parameters):
        # we place this import here to avoid the need for octave dependency unless this experiment is actually used.
        import mozaik.stimuli.vision.texture_based as textu  # vf

        VisualExperiment.__init__(self, model, parameters)
        for ty, t in enumerate(self.parameters.types):
            for i in range(0, self.parameters.num_images):
                for k in range(0, self.parameters.num_trials):
                    print("TRIAL vision NUMBER " + str(k))
                    im = textu.PSTextureStimulus(
                        frame_duration=self.frame_duration,
                        duration=self.parameters.image_duration,
                        trial=k,
                        background_luminance=self.background_luminance,
                        density=self.density,
                        location_x=0.0,
                        location_y=0.0,
                        size_x=model.visual_field.size_x,
                        size_y=model.visual_field.size_y,
                        texture_path=self.parameters.image_path,
                        stats_type=t,
                        seed=523 * (i + 1) + 5113 * (ty + 1),
                    )
                    self.stimuli.append(im)

    def do_analysis(self, data_store):
        pass


class MapResponseToInterruptedBarStimulus(VisualExperiment):
    """
    Map response to interrupted bar stimuli. 

    The experiments is intended as the simplest test for line-completion.
    This experiment presents a series of flashed interrupted  bars at 
    pre-specified range of displacements from the center along the line 
    that is  perpendicularly to the elongated axis of the bars, and at 
    range of different gaps in the middle of the bar.
    
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    Other parameters
    ----------------
    model : Model
          The model on which to execute the experiment.
    
    x : float
      The x corrdinates (of center) of the area in which the mapping will be done.

    y : float
      The y corrdinates (of center) of the area in which the mapping will be done.
        
    length : float
          The length of the bar.
    
    width : float
          The width of the bar.
             
    orientation : float
                The orientation of the bar.

    max_offset : float
               The maximum offset from the central position (defined by x and y) prependicular to the length of the bar at which the bars will be flashed.

    steps : int
         The number of steps in which the bars will be flashed between the two extreme positions defined by the max_offset parameter, along the axis prependicular to the length of the bar.
    
    duration : float
             The duration of single presentation of the stimulus.
    
    flash_duration : float
             The duration of the presence of the bar.
    
    relative_luminances : list(float) 
              List of luminance of the bar relative to background luminance at which the bar's will be presented. 0 is dark, 1.0 is double the background luminance.

    gap_lengths : list(float)
                List of length of the gap that the bar will have in the middle.

    
    num_trials : int
               Number of trials each each stimulus is shown.
    """

    required_parameters = ParameterSet(
        {
            "x": float,
            "y": float,
            "length": float,
            "width": float,
            "orientation": float,
            "max_offset": float,
            "steps": int,
            "duration": float,
            "flash_duration": float,
            "relative_luminances": list,
            "gap_lengths": list,
            "num_trials": int,
        }
    )

    def __init__(self, model, parameters):
        VisualExperiment.__init__(self, model, parameters)
        for k in range(0, self.parameters.num_trials):
            for s in range(0, self.parameters.steps):
                for r in self.parameters.relative_luminances:
                    for l in self.parameters.gap_lengths:
                        if self.parameters.steps > 1:
                            z = (
                                s
                                * (2 * self.parameters.max_offset)
                                / (self.parameters.steps - 1)
                            )
                        else:
                            z = self.parameters.max_offset
                        self.stimuli.append(
                            topo.FlashedInterruptedBar(
                                frame_duration=self.frame_duration,
                                size_x=model.visual_field.size_x,
                                size_y=model.visual_field.size_y,
                                location_x=0.0,
                                location_y=0.0,
                                background_luminance=self.background_luminance,
                                duration=self.parameters.duration,
                                density=self.density,
                                relative_luminance=r,
                                orientation=self.parameters.orientation,
                                width=self.parameters.width,
                                length=self.parameters.length,
                                flash_duration=self.parameters.flash_duration,
                                gap_length=l,
                                x=self.parameters.x
                                + numpy.cos(self.parameters.orientation + numpy.pi / 2)
                                * (-self.parameters.max_offset + z),
                                y=self.parameters.y
                                + numpy.sin(self.parameters.orientation + numpy.pi / 2)
                                * (-self.parameters.max_offset + z),
                                trial=k,
                            )
                        )

    def do_analysis(self, data_store):
        pass


class MapResponseToInterruptedCornerStimulus(VisualExperiment):
    """
    Map response with interrupted corner stimuli. 

    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    Other parameters
    ----------------
    model : Model
          The model on which to execute the experiment.
    
    x : float
      The x corrdinates (of center) of the area in which the mapping will be done.

    y : float
      The y corrdinates (of center) of the area in which the mapping will be done.
        
    length : float
          The length of the corner stimulus if unfolded ().
    
    width : float
          The width of the bar.
             
    orientation : float
                The orientation of the bar.

    max_offset : float
               The maximum offset from the central position (defined by x and y) prependicular to the length of the bar at which the bars will be flashed.

    steps : int
         The number of steps in which the bars will be flashed between the two extreme positions defined by the max_offset parameter, along the axis prependicular to the length of the bar.
    
    duration : float
             The duration of single presentation of the stimulus.
    
    flash_duration : float
             The duration of the presence of the bar.
    
    relative_luminances : list(float) 
              List of luminance of the bar relative to background luminance at which the bar's will be presented. 0 is dark, 1.0 is double the background luminance.

    gap_length : float
                List of length of the gap that the bar will have in the middle.
    
    angels    : list(float)
                List of angles (rad) in which both left and right (first left then right) will be angled at.
    
    num_trials : int
               Number of trials each each stimulus is shown.
    """

    required_parameters = ParameterSet(
        {
            "x": float,
            "y": float,
            "length": float,
            "width": float,
            "orientation": float,
            "max_offset": float,
            "steps": int,
            "duration": float,
            "flash_duration": float,
            "relative_luminances": list,
            "gap_length": float,
            "num_trials": int,
            "angles": list,
        }
    )

    def __init__(self, model, parameters):
        VisualExperiment.__init__(self, model, parameters)
        for k in range(0, self.parameters.num_trials):
            for s in range(0, self.parameters.steps):
                for r in self.parameters.relative_luminances:
                    for a in self.parameters.angles:
                        if self.parameters.steps > 1:
                            z = (
                                s
                                * (2 * self.parameters.max_offset)
                                / (self.parameters.steps - 1)
                            )
                        else:
                            z = self.parameters.max_offset
                        self.stimuli.append(
                            topo.FlashedInterruptedBar(
                                frame_duration=self.frame_duration,
                                size_x=model.visual_field.size_x,
                                size_y=model.visual_field.size_y,
                                location_x=0.0,
                                location_y=0.0,
                                background_luminance=self.background_luminance,
                                duration=self.parameters.duration,
                                density=self.density,
                                relative_luminance=r,
                                orientation=self.parameters.orientation,
                                width=self.parameters.width,
                                length=self.parameters.length,
                                flash_duration=self.parameters.flash_duration,
                                gap_length=self.parameters.gap_length,
                                left_angle=a,
                                right_angle=0,
                                x=self.parameters.x
                                + numpy.cos(self.parameters.orientation + numpy.pi / 2)
                                * (-self.parameters.max_offset + z),
                                y=self.parameters.y
                                + numpy.sin(self.parameters.orientation + numpy.pi / 2)
                                * (-self.parameters.max_offset + z),
                                trial=k,
                            )
                        )

                        self.stimuli.append(
                            topo.FlashedInterruptedBar(
                                frame_duration=self.frame_duration,
                                size_x=model.visual_field.size_x,
                                size_y=model.visual_field.size_y,
                                location_x=0.0,
                                location_y=0.0,
                                background_luminance=self.background_luminance,
                                duration=self.parameters.duration,
                                density=self.density,
                                relative_luminance=r,
                                orientation=self.parameters.orientation,
                                width=self.parameters.width,
                                length=self.parameters.length,
                                flash_duration=self.parameters.flash_duration,
                                gap_length=self.parameters.gap_length,
                                left_angle=a,
                                right_angle=a,
                                x=self.parameters.x
                                + numpy.cos(self.parameters.orientation + numpy.pi / 2)
                                * (-self.parameters.max_offset + z),
                                y=self.parameters.y
                                + numpy.sin(self.parameters.orientation + numpy.pi / 2)
                                * (-self.parameters.max_offset + z),
                                trial=k,
                            )
                        )

    def do_analysis(self, data_store):
        pass
