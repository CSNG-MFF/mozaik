import mozaik
from mozaik.controller import Global
from mozaik.experiments import Experiment
from parameters import ParameterSet
import mozaik.stimuli.vision.topographica_based as topo
import numpy
from mozaik.stimuli import InternalStimulus
from mozaik.tools.distribution_parametrization import ParameterWithUnitsAndPeriod, MozaikExtendedParameterSet
from mozaik.sheets.direct_stimulator import Depolarization
from collections import OrderedDict



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
    
    def __init__(self,model,parameters):
        Experiment.__init__(self, model,parameters)
        self.background_luminance = model.input_space.background_luminance
      
        #JAHACK: This is kind of a hack now. There needs to be generally defined interface of what is the spatial and temporal resolution of a visual input layer
        # possibly in the future we could force the visual_space to have resolution, perhaps something like native_resolution parameter!?
        self.density  = 1/self.model.input_layer.parameters.receptive_field.spatial_resolution # in pixels per degree of visual space 
        self.frame_duration = self.model.input_space.parameters.update_interval # in pixels per degree of visual space 

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
    
    required_parameters = ParameterSet({
            'luminances': list, 
            'step_duration' : float, 
            'num_trials' : int,
    })
    
    def __init__(self,model,parameters):
        VisualExperiment.__init__(self, model,parameters)
        
        # stimuli creation        
        for l in self.parameters.luminances:
            for k in range(0, self.parameters.num_trials):
                self.stimuli.append( topo.Null(
                    frame_duration = self.frame_duration,
                    size_x=model.visual_field.size_x,
                    size_y=model.visual_field.size_y,
                    location_x=0.0,
                    location_y=0.0,
                    density=self.density,
                    background_luminance=l,
                    duration=self.parameters.step_duration,
                    trial=k))

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

    blank_time : float
        The duration of the blank stimulus between image presentations

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
    
    required_parameters = ParameterSet({
            'time_per_image': float, 
            'blank_time' : float,
            'total_number_of_images' : int, 
            'num_trials' : int,
            'experiment_seed' : int,
            'stim_size' : float,
            'grid_size' : int,
            'grid' : bool
    })
    
    def __init__(self,model,parameters):
        VisualExperiment.__init__(self, model,parameters)
    
        for k in range(0, self.parameters.num_trials):
           
            self.stimuli.append(topo.SparseNoise(
                            frame_duration = self.frame_duration,
                            time_per_image = self.parameters.time_per_image,
                            blank_time = self.parameters.blank_time,
                            duration = self.parameters.total_number_of_images * (self.parameters.time_per_image + self.parameters.blank_time),  
                            size_x=self.parameters.stim_size,
                            size_y=self.parameters.stim_size,
                            location_x=0.0,
                            location_y=0.0, 
                            background_luminance=self.background_luminance,
                            density=self.density,
                            trial = k,
                            experiment_seed = self.parameters.experiment_seed,
                            grid_size = self.parameters.grid_size,
                            grid = self.parameters.grid
                          ))
   
    def do_analysis(self, data_store):
        pass

class MeasureSparseWithCurrentInjection(VisualExperiment):
    """
    Sparse noise stimulation experiments (identical to MeasureSparse) with concomitant current injection.

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

    blank_time : float
        The duration of the blank stimulus between image presentations

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

    stimulation_configuration : ParameterSet
                              The parameter set for direct stimulation specifying neurons to which the kick will be administered.

    stimulation_sheet : sheet
               The sheet in which to do stimulation

    stimulation_current : float (mA)
                     The current to inject into selected neurons.

    """
    
    required_parameters = ParameterSet({
            'time_per_image': float, 
            'blank_time' : float,
            'total_number_of_images' : int, 
            'num_trials' : int,
            'experiment_seed' : int,
            'stim_size' : float,
            'grid_size' : int,
            'grid' : bool,
            'stimulation_configuration' : ParameterSet,
            'stimulation_sheet' : str,
            'stimulation_current' : float,
      
    })
    
    def __init__(self,model,parameters):
        VisualExperiment.__init__(self, model,parameters)

        self.direct_stimulation = []

        for k in xrange(0, self.parameters.num_trials):
            d  = OrderedDict()
            p = MozaikExtendedParameterSet({
                                'population_selector' : self.parameters.stimulation_configuration,
                                'current' : self.parameters.stimulation_current
                               })

            d[self.parameters.stimulation_sheet] = [Depolarization(model.sheets[self.parameters.stimulation_sheet],p)]
            
            self.direct_stimulation.append(d)     

        p['sheet'] = self.parameters.stimulation_sheet

    
        for k in xrange(0, self.parameters.num_trials):
            self.stimuli.append(topo.SparseNoise(
                frame_duration = self.frame_duration,
                            time_per_image = self.parameters.time_per_image,
                            blank_time = self.parameters.blank_time,
                            duration = self.parameters.total_number_of_images * (self.parameters.time_per_image + self.parameters.blank_time),  
                            size_x=self.parameters.stim_size,
                            size_y=self.parameters.stim_size,
                            location_x=0.0,
                            location_y=0.0, 
                            background_luminance=self.background_luminance,
                            density=self.density,
                            trial = k,
                            experiment_seed = self.parameters.experiment_seed,
                            grid_size = self.parameters.grid_size,
                            grid = self.parameters.grid,
                            direct_stimulation_name='Injection',
                            direct_stimulation_parameters = p
                          ))
            
             

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

    required_parameters = ParameterSet({
            'time_per_image': float, 
            'total_number_of_images' : int, 
            'num_trials' : int,
            'experiment_seed' : int,
            'grid_size' : int,
    })    
    
    def __init__(self,model,parameters):
        VisualExperiment.__init__(self, model,parameters)

        for k in range(0, self.parameters.num_trials):
            self.stimuli.append(topo.DenseNoise(
                            frame_duration = self.frame_duration,
                            time_per_image = self.parameters.time_per_image,
                            duration = self.parameters.total_number_of_images * self.parameters.time_per_image, 
                            size_x=model.visual_field.size_x,
                            size_y=model.visual_field.size_y,
                            location_x=0.0,
                            location_y=0.0, 
                            background_luminance=self.background_luminance,
                            density=self.density,
                            trial = k,
                            experiment_seed = self.parameters.experiment_seed,
                            grid_size = self.parameters.grid_size
                          ))
         
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
    
    required_parameters = ParameterSet({
            'num_orientations': int, 
            'spatial_frequency' : float, 
            'temporal_frequency' : float,
            'grating_duration' : float,
            'contrasts' : list,
            'num_trials' : int,
    })  
    
    def __init__(self,model,parameters):
        VisualExperiment.__init__(self, model,parameters)
        for c in self.parameters.contrasts:
            for i in range(0, self.parameters.num_orientations):
                for k in range(0, self.parameters.num_trials):
                    self.stimuli.append(topo.FullfieldDriftingSinusoidalGrating(
                                    frame_duration = self.frame_duration,
                                    size_x=model.visual_field.size_x,
                                    size_y=model.visual_field.size_y,
                                    location_x=0.0,
                                    location_y=0.0,
                                    background_luminance=self.background_luminance,
                                    contrast = c,
                                    duration=self.parameters.grating_duration,
                                    density=self.density,
                                    trial=k,
                                    orientation=numpy.pi/self.parameters.num_orientations*i,
                                    spatial_frequency=self.parameters.spatial_frequency,
                                    temporal_frequency=self.parameters.temporal_frequency))

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
    
    required_parameters = ParameterSet({
            'num_orientations': int, 
            'spatial_frequency' : float, 
            'temporal_frequency' : float,
            'grating_duration' : float,
            'contrasts' : list,
            'num_trials' : int,
            'offset_time' : float,
            'onset_time' : float,
    })  
    
    def __init__(self,model,parameters):
        VisualExperiment.__init__(self, model,parameters)
        for c in self.parameters.contrasts:
            for i in range(0, self.parameters.num_orientations):
                for k in range(0, self.parameters.num_trials):
                    self.stimuli.append(topo.FullfieldDriftingSinusoidalGratingA(
                    frame_duration = self.frame_duration,
                                    size_x=model.visual_field.size_x,
                                    size_y=model.visual_field.size_y,
                                    offset_time=self.parameters.offset_time,
                                    onset_time=self.parameters.onset_time,
                                    location_x=0.0,
                                    location_y=0.0,
                                    background_luminance=self.background_luminance,
                                    contrast = c,
                                    duration=self.parameters.grating_duration,
                                    density=self.density,
                                    trial=k,
                                    orientation=numpy.pi/self.parameters.num_orientations*i,
                                    spatial_frequency=self.parameters.spatial_frequency,
                                    temporal_frequency=self.parameters.temporal_frequency))

    def do_analysis(self, data_store):
        pass

class MeasureSizeTuning(VisualExperiment):
    """
    Size tuning experiment.

    This experiment will show a series of sinusoidal gratings or constant flat stimuli 
    (see *with_flat* parameter) confined to an aperture whose radius will vary.

    
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

    required_parameters = ParameterSet({
            'num_sizes' : int,
            'max_size' : float,
            'orientation' : float,
            'spatial_frequency' : float, 
            'temporal_frequency' : float,
            'grating_duration' : float,
            'contrasts' : list,
            'num_trials' : int,
            'log_spacing' : bool,
    })  

    def __init__(self,model,parameters):
        VisualExperiment.__init__(self, model,parameters)
            
        # linear or logarithmic spaced sizes
        if self.parameters.log_spacing:
            base2max = numpy.log2(self.parameters.max_size)
            sizes = numpy.logspace(start=-3.0, stop=base2max, num=self.parameters.num_sizes, base=2.0) 
        else:
            sizes = numpy.linspace(0, self.parameters.max_size,self.parameters.num_sizes)                     
            
        # stimuli creation        
        for c in self.parameters.contrasts:
            for s in sizes:
                for k in range(0, self.parameters.num_trials):
                    self.stimuli.append(topo.DriftingSinusoidalGratingDisk(
                                    frame_duration = self.frame_duration,
                                    size_x=model.visual_field.size_x,
                                    size_y=model.visual_field.size_y,
                                    location_x=0.0,
                                    location_y=0.0,
                                    background_luminance=self.background_luminance,
                                    contrast = c,
                                    duration=self.parameters.grating_duration,
                                    density=self.density,
                                    trial=k,
                                    orientation=self.parameters.orientation,
                                    radius=s,
                                    spatial_frequency=self.parameters.spatial_frequency,
                                    temporal_frequency=self.parameters.temporal_frequency))

    def do_analysis(self, data_store):
        pass


class MeasureSizeTuningRing(VisualExperiment):
    """
    Size tuning experiment.

    This experiment will show a series of sinusoidal gratings confined to a ring which outer
    radius stays constant and which inner radius will varry.

    
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    Other parameters
    ----------------

    num_inner_radius : int
              Number of different inner radius to present.
    
    outer_radius : float (degrees of visual field)
             The outside radius of the grating ring - in degrees of visual field.
    
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

    required_parameters = ParameterSet({
            'num_inner_radius' : int,
            'outer_radius' : float,
            'orientation' : float,
            'spatial_frequency' : float,
            'temporal_frequency' : float,
            'grating_duration' : float,
            'contrasts' : list,
            'num_trials' : int,
            'log_spacing' : bool,
    })


    def __init__(self,model,parameters):
        VisualExperiment.__init__(self, model,parameters)

        # linear or logarithmic spaced sizes
        if self.parameters.log_spacing:
            base2max = numpy.log2(self.parameters.outer_radius)
            inner_radius = numpy.logspace(start=-3.0, stop=base2max, num=self.parameters.num_inner_radius, base=2.0)
            inner_radius[-1] = self.parameters.outer_radius
        else:
            inner_radius = numpy.linspace(0, self.parameters.outer_radius,self.parameters.num_inner_radius)

        # stimuli creation        
        for c in self.parameters.contrasts:
            for r in inner_radius:
                for k in range(0, self.parameters.num_trials):
                    self.stimuli.append(topo.DriftingSinusoidalGratingRing(
                                    frame_duration = self.frame_duration,
                                    size_x=model.visual_field.size_x,
                                    size_y=model.visual_field.size_y,
                                    location_x=0.0,
                                    location_y=0.0,
                                    background_luminance=self.background_luminance,
                                    contrast = c,
                                    duration=self.parameters.grating_duration,
                                    density=self.density,
                                    trial=k,
                                    orientation=self.parameters.orientation,
                                    inner_aperture_radius=r,
                                    outer_aperture_radius=self.parameters.outer_radius,
                                    spatial_frequency=self.parameters.spatial_frequency,
                                    temporal_frequency=self.parameters.temporal_frequency))

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

    required_parameters = ParameterSet({
            'orientation': float, 
            'spatial_frequency' : float, 
            'temporal_frequency' : float,
            'grating_duration' : float,
            'contrasts' : list,
            'num_trials' : int,
    })  
    
    def __init__(self,model,parameters):
        VisualExperiment.__init__(self, model,parameters)
            
        # stimuli creation        
        for c in self.parameters.contrasts:
            for k in range(0, self.parameters.num_trials):
                self.stimuli.append(topo.FullfieldDriftingSinusoidalGrating(
                    frame_duration = self.frame_duration,
                    size_x=model.visual_field.size_x,
                    size_y=model.visual_field.size_y,
                    location_x=0.0,
                    location_y=0.0,
                    background_luminance=self.background_luminance,
                    contrast = c,
                    duration=self.parameters.grating_duration,
                    density=self.density,
                    trial=k,
                    orientation=self.parameters.orientation,
                    spatial_frequency=self.parameters.spatial_frequency,
                    temporal_frequency=self.parameters.temporal_frequency))

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

    required_parameters = ParameterSet({
            'orientation': float, 
            'spatial_frequency' : float, 
            'temporal_frequency' : float,
            'grating_duration' : float,
            'contrasts' : list,
            'num_trials' : int,
            'offset_time' : float,
            'onset_time' : float,

    })  
    
    def __init__(self,model,parameters):
        VisualExperiment.__init__(self, model,parameters)
            
        # stimuli creation        
        for c in self.parameters.contrasts:
            for k in range(0, self.parameters.num_trials):
                self.stimuli.append(topo.FullfieldDriftingSinusoidalGratingA(
                    frame_duration = self.frame_duration,
                    size_x=model.visual_field.size_x,
                    size_y=model.visual_field.size_y,
                    location_x=0.0,
                    location_y=0.0,
                    background_luminance=self.background_luminance,
                    contrast = c,
                    duration=self.parameters.grating_duration,
                    density=self.density,
                    trial=k,
                    offset_time=self.parameters.offset_time,
                    onset_time=self.parameters.onset_time,
                    orientation=self.parameters.orientation,
                    spatial_frequency=self.parameters.spatial_frequency,
                    temporal_frequency=self.parameters.temporal_frequency))

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

    required_parameters = ParameterSet({
            'orientation': float, 
            'spatial_frequencies' : list, 
            'temporal_frequencies' : list,
            'grating_duration' : float,
            'contrasts' : list,
            'num_trials' : int,
            'square' : bool,
    })  
    

    def __init__(self,model,parameters):
        VisualExperiment.__init__(self, model,parameters)
            
        # stimuli creation        
        for tf in self.parameters.temporal_frequencies:
            for sf in self.parameters.spatial_frequencies:
                for c in self.parameters.contrasts:
                    for k in range(0, self.parameters.num_trials):
                        if self.parameters.square:
                            self.stimuli.append(topo.FullfieldDriftingSquareGrating(
                                frame_duration = self.frame_duration,
                                size_x=model.visual_field.size_x,
                                size_y=model.visual_field.size_y,
                                location_x=0.0,
                                location_y=0.0,
                                background_luminance=self.background_luminance,
                                contrast = c,
                                duration=self.parameters.grating_duration,
                                density=self.density,
                                trial=k,
                                orientation=self.parameters.orientation,
                                spatial_frequency=sf,
                                temporal_frequency=tf))
                        else:
                            self.stimuli.append(topo.FullfieldDriftingSinusoidalGrating(
                                frame_duration = self.frame_duration,
                                size_x=model.visual_field.size_x,
                                size_y=model.visual_field.size_y,
                                location_x=0.0,
                                location_y=0.0,
                                background_luminance=self.background_luminance,
                                contrast = c,
                                duration=self.parameters.grating_duration,
                                density=self.density,
                                trial=k,
                                orientation=self.parameters.orientation,
                                spatial_frequency=sf,
                                temporal_frequency=tf))

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

    required_parameters = ParameterSet({
            'num_orientations': int, 
            'orientation' : float,
            'center_radius' : float,
            'surround_radius' : float,
            'spatial_frequency' : float, 
            'temporal_frequency' : float,
            'grating_duration' : float,
            'contrasts' : list,
            'num_trials' : int,
    })  

    def __init__(self,model,parameters):
        VisualExperiment.__init__(self, model,parameters)
        
        for c in self.parameters.contrasts:
            for i in range(0, self.parameters.num_orientations):
                for k in range(0, self.parameters.num_trials):
                    self.stimuli.append(
                        topo.DriftingSinusoidalGratingCenterSurroundStimulus(
                                    frame_duration = self.frame_duration,
                                    size_x=model.visual_field.size_x,
                                    size_y=model.visual_field.size_y,
                                    location_x=0.0,
                                    location_y=0.0,
                                    background_luminance=self.background_luminance,
                                    contrast = c,
                                    duration=self.parameters.grating_duration,
                                    density=self.density,
                                    trial=k,
                                    center_orientation=self.parameters.orientation,
                                    surround_orientation=numpy.pi/self.parameters.num_orientations*i,
                                    gap=0,
                                    center_radius=self.parameters.center_radius,
                                    surround_radius=self.parameters.surround_radius,
                                    spatial_frequency=self.parameters.spatial_frequency,
                                    temporal_frequency=self.parameters.temporal_frequency))

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

    required_parameters = ParameterSet({
            'spatial_frequencies' : list, 
            'temporal_frequencies' : list,
            'grating_duration' : float,
            'contrasts' : list,
            'separation' : float,
            'num_trials' : int,
    })  

    def __init__(self,model,parameters):
        VisualExperiment.__init__(self, model,parameters)
        
         # the orientation is fixed to horizontal
        orientation = 0 #numpy.pi/2
        # SQUARED GRATINGS       
        for sf in self.parameters.spatial_frequencies:
            for k in range(0, self.parameters.num_trials):
                self.stimuli.append(
                    topo.FullfieldDriftingSquareGrating(
                        frame_duration = self.frame_duration,
                        size_x=model.visual_field.size_x,
                        size_y=model.visual_field.size_y,
                        location_x=0.0,
                        location_y=0.0,
                        background_luminance=self.background_luminance,
                        contrast = self.parameters.contrast,
                        duration=self.parameters.grating_duration,
                        density=self.density,
                        trial=k,
                        orientation=orientation,
                        spatial_frequency=self.parameters.sf,
                        temporal_frequency=self.parameters.temporal_frequency
                    )
                )
        # FLASHING SQUARES
        # the spatial_frequencies matters because squares sizes is established using the spatial frequency as for the drifting grating
        for sf in self.parameters.spatial_frequencies:
            for k in range(0, self.parameters.num_trials):
                self.stimuli.append(
                    topo.FlashingSquares(
                        frame_duration = self.frame_duration,
                        size_x=model.visual_field.size_x,
                        size_y=model.visual_field.size_y,
                        location_x=0.0,
                        location_y=0.0,
                        background_luminance=self.background_luminance,
                        contrast = self.parameters.contrast,
                        separation = self.parameters.separation,
                        separated = True,
                        density = self.density,
                        trial = k,
                        duration=self.parameters.grating_duration,
                        orientation = orientation*2,
                        spatial_frequency = sf,
                        temporal_frequency = self.parameters.temporal_frequency
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
               
    size : float
               The size of the image in degrees of visual field

    Notes
    -----
    Currently this implementation bound to have the image and the eye path saved in in files *./image_naturelle_HIGH.bmp* and *./eye_path.pickle*.
    In future we need to make this more general.
    """
    
    required_parameters = ParameterSet({
            'stimulus_duration' : float,
            'num_trials' : int,
            'size' : float,
    })  

    
    def __init__(self,model,parameters):
        VisualExperiment.__init__(self, model,parameters)
        
        for k in range(0, self.parameters.num_trials):
            self.stimuli.append(
                topo.NaturalImageWithEyeMovement(
                            frame_duration = self.frame_duration,
                            size_x=model.visual_field.size_x,
                            size_y=model.visual_field.size_y,
                            location_x=0.0,
                            location_y=0.0,
                            background_luminance=self.background_luminance,
                            duration=self.parameters.stimulus_duration,
                            density=self.density,
                            trial=k,
                            size=self.parameters.size,  # x size of image
                            eye_movement_period=6.66,  # eye movement period
                            eye_path_location='./eye_path.pickle',
                            image_location='./image_naturelle_HIGHC.bmp'))

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
    
    required_parameters = ParameterSet({
           
            'spatial_frequency' : float, 
            'temporal_frequency' : float,
            'grating_duration' : float,
            'contrast' : float,
            'num_trials' : int,
    })  
    
    def __init__(self,model,parameters):
        VisualExperiment.__init__(self, model,parameters)
        
        for k in range(0, self.parameters.num_trials):
            self.stimuli.append(
                topo.DriftingGratingWithEyeMovement(
                            frame_duration = self.frame_duration,
                            size_x=model.visual_field.size_x,
                            size_y=model.visual_field.size_y,
                            location_x=0.0,
                            location_y=0.0,
                            background_luminance=self.background_luminance,
                            duration=self.parameters.grating_duration,
                            density=self.density,
                            trial=k,
                            contrast = self.parameters.contrast,
                            eye_movement_period=6.66,  # eye movement period
                            eye_path_location='./eye_path.pickle',
                            orientation=0,
                            spatial_frequency=self.parameters.spatial_frequency,
                            temporal_frequency=self.parameters.temporal_frequency))

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

    required_parameters = ParameterSet({
            'duration' : float,
            'num_trials' : int,
    })  
    
    def __init__(self,model,parameters):
            VisualExperiment.__init__(self, model,parameters)
            
            for k in range(0,self.parameters.num_trials):
                self.stimuli.append(
                            topo.Null(   
                                frame_duration = self.frame_duration,
                                size_x=model.visual_field.size_x,
                                size_y=model.visual_field.size_y,
                                location_x=0.0,
                                location_y=0.0,
                                background_luminance=self.background_luminance,
                                duration=self.parameters.duration,
                                density=self.density,
                                trial=k))    
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
    
    required_parameters = ParameterSet({
            'x' : float,
            'y' : float,
            'length' : float,
            'width' : float,
            'orientation' : float,
            'max_offset' : float,
            'steps' : int,
            'duration' : float,
            'flash_duration' : float, 
            'relative_luminance' : float,
            'num_trials' : int,
    })  
    
    def __init__(self, model,parameters):
        VisualExperiment.__init__(self, model,parameters)
        for k in range(0, self.parameters.num_trials):
            for s in range(0, self.parameters.steps):
                self.stimuli.append(
                    topo.FlashedBar(
                                frame_duration = self.frame_duration,
                                size_x=model.visual_field.size_x,
                                size_y=model.visual_field.size_y,
                                location_x=0.0,
                                location_y=0.0,
                                background_luminance=self.background_luminance,
                                duration=self.parameters.duration,
                                density=self.density,
                                relative_luminance = self.parameters.relative_luminance,
                                orientation = self.parameters.orientation,
                                width = self.parameters.width,
                                length = self.parameters.length,
                                flash_duration = self.parameters.flash_duration,
                                x = self.parameters.x + numpy.cos(self.parameters.orientation+numpy.pi/2) * (-self.parameters.max_offset + (2*self.parameters.max_offset)/ (self.parameters.steps-1) * s),
                                y = self.parameters.y + numpy.sin(self.parameters.orientation+numpy.pi/2) * (-self.parameters.max_offset + (2*self.parameters.max_offset)/ (self.parameters.steps-1) * s),
                                trial=k))

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
    
    required_parameters = ParameterSet({
            'sheet_list' : list,
            'num_trials' : int,
            'localstimulationarray_parameters' : ParameterSet,
    })

    
    def __init__(self,model,parameters):
            Experiment.__init__(self, model,parameters)
            from mozaik.sheets.direct_stimulator import LocalStimulatorArrayChR
            
            d  = OrderedDict()
            for sheet in self.parameters.sheet_list:
                d[sheet] = [LocalStimulatorArrayChR(model.sheets[sheet],self.parameters.localstimulationarray_parameters)]
            
            self.direct_stimulation = []

            for i in range(0,self.parameters.num_trials):
                self.direct_stimulation.append(d)
                self.stimuli.append(
                            InternalStimulus(   
                                                frame_duration=self.parameters.localstimulationarray_parameters.stimulating_signal_parameters.duration, 
                                                duration=self.parameters.localstimulationarray_parameters.stimulating_signal_parameters.duration,
                                                trial=i,
                                                direct_stimulation_name='LocalStimulatorArray',
                                                direct_stimulation_parameters=self.parameters.localstimulationarray_parameters
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
    
    required_parameters = ParameterSet({
            'sheet_list' : list,
            'num_trials' : int,
            'num_orientations' : int,
            'intensities' : list,
            'localstimulationarray_parameters' : ParameterSet,
    })

    
    def __init__(self,model,parameters):
            Experiment.__init__(self, model,parameters)
            from mozaik.sheets.direct_stimulator import LocalStimulatorArrayChR
            
            self.direct_stimulation = []
            first = True

            for s in self.parameters.intensities:
                for i in range(self.parameters.num_orientations):
                    p = MozaikExtendedParameterSet(self.parameters.localstimulationarray_parameters.tree_copy().as_dict())
                    p.stimulating_signal_parameters.orientation = ParameterWithUnitsAndPeriod(numpy.pi/self.parameters.num_orientations * i,period=numpy.pi)
                    p.stimulating_signal_parameters.scale =       ParameterWithUnitsAndPeriod(float(s),period=None)
                    d  = OrderedDict()
                    if first:
                        for sheet in self.parameters.sheet_list:
                            d[sheet] = [LocalStimulatorArrayChR(model.sheets[sheet],p)]
                        first = False
                    else:
                        for sheet in self.parameters.sheet_list:
                            if self.sheet.parameters.cell.model[-3:] == '_sc': 
                                d[sheet] = [LocalStimulatorArrayChR(model.sheets[sheet],p)]
                            else:
                                d[sheet] = [LocalStimulatorArrayChR(model.sheets[sheet],p,shared_scs=self.direct_stimulation[0][sheet][0].scs)]

                    for i in range(0,self.parameters.num_trials):
                        self.direct_stimulation.append(d)
                        self.stimuli.append(
                                    InternalStimulus(   
                                                        frame_duration=self.parameters.localstimulationarray_parameters.stimulating_signal_parameters.duration, 
                                                        duration=self.parameters.localstimulationarray_parameters.stimulating_signal_parameters.duration,
                                                        trial=i,
                                                        direct_stimulation_name='LocalStimulatorArray',
                                                        direct_stimulation_parameters=p
                                                     )
                                            )                



class CorticalStimulationWithStimulatorArrayAndOrientationTuningProtocol_ContrastBased(Experiment):
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
    
    required_parameters = ParameterSet({
            'sheet_list' : list,
            'num_trials' : int,
            'num_orientations' : int,
            'contrasts' : list,
            'localstimulationarray_parameters' : ParameterSet,
    })

    
    def __init__(self,model,parameters):
            Experiment.__init__(self, model,parameters)
            from mozaik.sheets.direct_stimulator import LocalStimulatorArrayChR
            
            self.direct_stimulation = []
            first = True

            for c in self.parameters.contrasts:
                for i in range(self.parameters.num_orientations):
                    p = MozaikExtendedParameterSet(self.parameters.localstimulationarray_parameters.tree_copy().as_dict())
                    p.stimulating_signal_parameters.orientation = ParameterWithUnitsAndPeriod(numpy.pi/self.parameters.num_orientations * i,period=numpy.pi)
                    p.stimulating_signal_parameters.contrast =       ParameterWithUnitsAndPeriod(float(c),period=None)
                    d  = OrderedDict()
                    if first:
                        for sheet in self.parameters.sheet_list:
                            d[sheet] = [LocalStimulatorArrayChR(model.sheets[sheet],p)]
                        first = False
                    else:
                        for sheet in self.parameters.sheet_list:
                            d[sheet] = [LocalStimulatorArrayChR(model.sheets[sheet],p,shared_scs=self.direct_stimulation[0][sheet][0].scs)]

                    for i in range(0,self.parameters.num_trials):
                        self.direct_stimulation.append(d)
                        self.stimuli.append(
                                    InternalStimulus(   
                                                        frame_duration=self.parameters.localstimulationarray_parameters.stimulating_signal_parameters.duration, 
                                                        duration=self.parameters.localstimulationarray_parameters.stimulating_signal_parameters.duration,
                                                        trial=i,
                                                        direct_stimulation_name='LocalStimulatorArray',
                                                        direct_stimulation_parameters=p
                                                     )
                                            )                


class VonDerHeydtIllusoryBarProtocol(VisualExperiment):
    """
    An illusory bar from Von Der Heydt et al. 1989.

    Von Der Heydt, R., & Peterhans, E. (1989). Mechanisms of contour perception in monkey visual cortex. I. Lines of pattern discontinuity. Journal of Neuroscience, 9(5), 1731-1748. Retrieved from https://www.jneurosci.org/content/jneuro/9/5/1731.full.pdf
    
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
    
    required_parameters = ParameterSet({
            'x' : float,
            'y' : float,
            'length' : float,
            'bar_width' : float,
            'orientation' : float,
            'background_bar_width' : float,
            'occlusion_bar_width' : list,
            'duration' : float,
            'flash_duration' : float, 
            'num_trials' : int,
    })  
    
    def __init__(self, model,parameters):
        VisualExperiment.__init__(self, model,parameters)
        for k in range(0, self.parameters.num_trials):
              for obw in self.parameters.occlusion_bar_width:  
                            self.stimuli.append(
                                topo.FlashedInterruptedBar(
                                            frame_duration = 7,
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
                                            orientation = self.parameters.orientation,
                                            flash_duration = self.parameters.flash_duration,
                                            x = self.parameters.x,
                                            y = self.parameters.y,
                                            trial=k))

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
    num_images: int
               The number of samples generated for each texture family

    folder_path: str
               The path of of the folder containing the original naturalistic images

    images: list
               The names of the pgm files containing the original naturalistic images

    duration : float
               The duration of the presentation of a single image
    
    types : list(int) 
              List of types indicating which statistics to match:
                0 - original image
                1 - naturalistic texture image (matched higher order statistics)
                2 - spectrally matched noise (matched marginal statistics only).
    
    num_trials : int
               Number of trials each each stimulus is shown.

    random : bool
               Whether to present stimuli in a random order or not

    size_x : float
              The size of the stimulus on the x-axis

    size_y : float
              The size of the stimulus on the y-axis
    """

    required_parameters = ParameterSet({
            'num_images': int, #n. of images of each type, different synthesized instances
            'folder_path' : str,
            'images': list,
            'duration' : float,
            'types' : list,
            'num_trials' : int, #n. of same instance
            'random': bool, # Whether to present stimuli in a random order or not
            'size_x': float, # The size of the stimulus on the x-axis
            'size_y': float, # The size of the stimulus on the y-axis
    })  

    def __init__(self,model,parameters):
	# we place this import here to avoid the need for octave dependency unless this experiment is actually used.
        import mozaik.stimuli.vision.texture_based as textu #vf
        VisualExperiment.__init__(self, model,parameters)

        if self.parameters.random:
            images = self.parameters.images * len(self.parameters.types) * self.parameters.num_images * self.parameters.num_trials
            types = [t for t in self.parameters.types for _ in self.parameters.images] * self.parameters.num_images * self.parameters.num_trials
            samples = [i for i in range(self.parameters.num_images) for _ in self.parameters.images * len(self.parameters.types)] * self.parameters.num_trials
            trials = [i for i in range(self.parameters.num_trials) for _ in self.parameters.images * len(self.parameters.types) * self.parameters.num_images]
            stimuli_list = list(zip(images, types, samples, trials)) 
            mozaik.rng.shuffle(stimuli_list)
            randomized_images, randomized_types, randomized_samples, randomized_trials = zip(*stimuli_list)
        
            f = open(Global.root_directory +'/stimuli_order','w')
            f.write(str(stimuli_list))
            f.close()

            for image, stats_type, sample, trial in stimuli_list:
                im = textu.PSTextureStimulus(
                        frame_duration = self.frame_duration,
                        duration=self.parameters.duration,
                        trial=trial,
                        background_luminance=self.background_luminance,
                        density=self.density,
                        location_x=0.0,
                        location_y=0.0,
                        sample=sample,
                        size_x=self.parameters.size_x,
                        size_y=self.parameters.size_y,
                        texture_path = self.parameters.folder_path+image,
                        texture = image.replace(".pgm",""),
                        stats_type = stats_type,
                        seed = 523*(sample+1)+5113*(stats_type+1))
                self.stimuli.append(im)

        else:
            for image in self.parameters.images:
                for ty, t in enumerate(self.parameters.types):
                 for i in range(0, self.parameters.num_images):                
                     for k in range(0, self.parameters.num_trials):
                        im = textu.PSTextureStimulus(
                                frame_duration = self.frame_duration,
                                duration=self.parameters.duration,
                                trial=k,
                                background_luminance=self.background_luminance,
                                density=self.density,
                                location_x=0.0,
                                location_y=0.0,
                                sample=i,
                                size_x=self.parameters.size_x,
                                size_y=self.parameters.size_y,
                                texture_path = self.parameters.folder_path+image,
                                texture = image.replace(".pgm",""),
                                stats_type = t,
                                seed = 523*(i+1)+5113*(ty+1))
                        self.stimuli.append(im)

    def do_analysis(self, data_store):
        pass         



class MeasureTextureSizeTuning(VisualExperiment):
    """
    Size tuning experiment.

    This experiment will show a series of synthetic texture spectrally-matched noise stimuli
    confined to an apparature whose radius will vary.

    
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
    
    duration : float
                      The duration of single presentation of a grating.
    
    num_trials : int
               Number of trials each each stimulus is shown.
    
    log_spacing : bool
               Whether use logarithmic spaced sizes. By default False, meaning linear spacing 

    num_images: int
               The number of samples generated for each texture family

    folder_path: str
               The path of of the folder containing the original naturalistic images
    
    images: list
               The names of the pgm files containing the original naturalistic images
               
    types : list(int)
          List of types indicating which statistics to match:
            0 - original image
            1 - naturalistic texture image (matched higher order statistics)
            2 - spectrally matched noise (matched marginal statistics only).

    """

    required_parameters = ParameterSet({
            'num_sizes' : int,
            'max_size' : float,
            'duration' : float,
            'num_trials' : int,
            'log_spacing' : bool,
            'num_images': int, 
            'folder_path' : str,
            'images': list,
            'types': list,

    })  

    def __init__(self,model,parameters):
        # we place this import here to avoid the need for octave dependency unless this experiment is actually used.
        import mozaik.stimuli.vision.texture_based as textu #vf

        VisualExperiment.__init__(self, model,parameters)
            
        # linear or logarithmic spaced sizes
        if self.parameters.log_spacing:
            base2max = numpy.log2(self.parameters.max_size)
            sizes = numpy.logspace(start=-3.0, stop=base2max, num=self.parameters.num_sizes, base=2.0) 
        else:
            sizes = numpy.linspace(0, self.parameters.max_size,self.parameters.num_sizes)                     
            
        # stimuli creation        
        for image in self.parameters.images:
            for ty, t in enumerate(self.parameters.types):
             for i in range(0, self.parameters.num_images):
                 for s in sizes:
                     for k in range(0, self.parameters.num_trials):
                         im = textu.PSTextureStimulusDisk(
                                frame_duration = self.frame_duration,
                                duration=self.parameters.duration,
                                trial=k,
                                background_luminance=self.background_luminance,
                                density=self.density,
                                location_x=0.0,
                                location_y=0.0,
                                sample=i,
                                size_x=model.visual_field.size_x,
                                size_y=model.visual_field.size_y,
                                texture_path = self.parameters.folder_path+image,
                                texture = image.replace(".pgm",""),
                                stats_type = t,
                                radius=s,
                                seed = 523*(i+1)+5113*(ty+1))

                         self.stimuli.append(im)

    def do_analysis(self, data_store):
        pass

class MeasureInformativePixelCorrelationStatisticsResponse(VisualExperiment):
    """
    Measure sensitivity to informative pixel correlations base on synthetic stimuli 
    This experiment will show a series of synthetic stimuli  
    that vary in pixel correlations statistics.
    
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.
    Other parameters
    ----------------

    duration : float
               The duration of the presentation of a single image
    
    correlation_values: list(float) 
               List of values of the pixel correlation statistics
               that will be used to generate the stimuli

    num_trials : int
               Number of trials each each stimulus is shown.
    """

    required_parameters = ParameterSet({
            'duration' : float,
            'correlation_values': list,
            'num_trials' : int, #n. of same instance
            'spatial_frequency' : float,

    })

    def __init__(self,model,parameters):
        # we place this import here to avoid the need for octave dependency unless this experiment is actually used.
        import mozaik.stimuli.vision.texture_based as textu #vf
        VisualExperiment.__init__(self, model,parameters)

        for i in range(10):
            for value in self.parameters.correlation_values:
                for j in range(self.parameters.num_trials):
                    im = textu.VictorInformativeSyntheticStimulus(
                            frame_duration = self.frame_duration,
                            duration=self.parameters.duration,
                            trial=j,
                            background_luminance=self.background_luminance,
                            density=self.density,
                            location_x=0.0,
                            location_y=0.0,
                            size_x=model.visual_field.size_x,
                            size_y=model.visual_field.size_y,
                            spatial_frequency=self.parameters.spatial_frequency/2,
                            pixel_statistics = value,
                            correlation_type = i,
                            seed = 523+5113*(i+1))
                    self.stimuli.append(im)

    def do_analysis(self, data_store):
        pass

class MeasureUninformativePixelCorrelationStatisticsResponse(VisualExperiment):
    """
    Measure sensitivity to uninformative pixel correlations base on synthetic stimuli 
    This experiment will show a series of synthetic stimuli  
    that vary in pixel correlations statistics.
    
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.
    Other parameters
    ----------------

    duration : float
               The duration of the presentation of a single image
    
    correlation_values: list(float) 
               List of values of the pixel correlation statistics
               that will be used to generate the stimuli

    num_trials : int
               Number of trials each each stimulus is shown.
    """

    required_parameters = ParameterSet({
            'duration' : float,
            'correlation_values': list,
            'num_trials' : int, #n. of same instance
            'spatial_frequency' : float,
    })

    def __init__(self,model,parameters):
        # we place this import here to avoid the need for octave dependency unless this experiment is actually used.
        import mozaik.stimuli.vision.texture_based as textu #vf
        VisualExperiment.__init__(self, model,parameters)

        for i in range(2):
            for value in self.parameters.correlation_values:
                for j in range(self.parameters.num_trials):
                    im = textu.VictorUninformativeSyntheticStimulus(
                            frame_duration = self.frame_duration,
                            duration=self.parameters.duration,
                            trial=j,
                            background_luminance=self.background_luminance,
                            density=self.density,
                            location_x=0.0,
                            location_y=0.0,
                            size_x=model.visual_field.size_x,
                            size_y=model.visual_field.size_y,
                            spatial_frequency=self.parameters.spatial_frequency/2,
                            pixel_statistics = value,
                            correlation_type = i,
                            seed = 523+5113*(i+1))
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
    
    required_parameters = ParameterSet({
            'x' : float,
            'y' : float,
            'length' : float,
            'width' : float,
            'orientation' : float,
            'max_offset' : float,
            'steps' : int,
            'duration' : float,
            'flash_duration' : float, 
            'relative_luminances' : list,
            'gap_lengths' : list,
            'num_trials' : int,
    })  
    
    def __init__(self, model,parameters):
        VisualExperiment.__init__(self, model,parameters)
        for k in range(0, self.parameters.num_trials):
            for s in range(0, self.parameters.steps):
                for r in self.parameters.relative_luminances:
                    for l in self.parameters.gap_lengths:  
                            if self.parameters.steps>1:
                                z = s*(2*self.parameters.max_offset)/ (self.parameters.steps-1)
                            else:
                                z = self.parameters.max_offset   
                            self.stimuli.append(
                                topo.FlashedInterruptedBar(
                                            frame_duration = self.frame_duration,
                                            size_x=model.visual_field.size_x,
                                            size_y=model.visual_field.size_y,
                                            location_x=0.0,
                                            location_y=0.0,
                                            background_luminance=self.background_luminance,
                                            duration=self.parameters.duration,
                                            density=self.density,
                                            relative_luminance = r,
                                            orientation = self.parameters.orientation,
                                            width = self.parameters.width,
                                            length = self.parameters.length,
                                            flash_duration = self.parameters.flash_duration,
                                            gap_length = l,
                                            x = self.parameters.x + numpy.cos(self.parameters.orientation+numpy.pi/2) * (-self.parameters.max_offset + z),
                                            y = self.parameters.y + numpy.sin(self.parameters.orientation+numpy.pi/2) * (-self.parameters.max_offset + z),
                                            trial=k))

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
    
    required_parameters = ParameterSet({
            'x' : float,
            'y' : float,
            'length' : float,
            'width' : float,
            'orientation' : float,
            'max_offset' : float,
            'steps' : int,
            'duration' : float,
            'flash_duration' : float, 
            'relative_luminances' : list,
            'gap_length' : float,
            'num_trials' : int,
            'angles' : list
    })  
    
    def __init__(self, model,parameters):
        VisualExperiment.__init__(self, model,parameters)
        for k in range(0, self.parameters.num_trials):
            for s in range(0, self.parameters.steps):
                for r in self.parameters.relative_luminances:
                    for a in self.parameters.angles:
                            if self.parameters.steps>1:
                                z = s*(2*self.parameters.max_offset)/ (self.parameters.steps-1)
                            else:
                                z = self.parameters.max_offset   
                            self.stimuli.append(
                                topo.FlashedInterruptedBar(
                                            frame_duration = self.frame_duration,
                                            size_x=model.visual_field.size_x,
                                            size_y=model.visual_field.size_y,
                                            location_x=0.0,
                                            location_y=0.0,
                                            background_luminance=self.background_luminance,
                                            duration=self.parameters.duration,
                                            density=self.density,
                                            relative_luminance = r,
                                            orientation = self.parameters.orientation,
                                            width = self.parameters.width,
                                            length = self.parameters.length,
                                            flash_duration = self.parameters.flash_duration,
                                            gap_length = self.parameters.gap_length,
                                            left_angle = a,
                                            right_angle=0,
                                            x = self.parameters.x + numpy.cos(self.parameters.orientation+numpy.pi/2) * (-self.parameters.max_offset + z),
                                            y = self.parameters.y + numpy.sin(self.parameters.orientation+numpy.pi/2) * (-self.parameters.max_offset + z),
                                            trial=k))

                            self.stimuli.append(
                                topo.FlashedInterruptedBar(
                                            frame_duration = self.frame_duration,
                                            size_x=model.visual_field.size_x,
                                            size_y=model.visual_field.size_y,
                                            location_x=0.0,
                                            location_y=0.0,
                                            background_luminance=self.background_luminance,
                                            duration=self.parameters.duration,
                                            density=self.density,
                                            relative_luminance = r,
                                            orientation = self.parameters.orientation,
                                            width = self.parameters.width,
                                            length = self.parameters.length,
                                            flash_duration = self.parameters.flash_duration,
                                            gap_length = self.parameters.gap_length,
                                            left_angle = a,
                                            right_angle = a,
                                            x = self.parameters.x + numpy.cos(self.parameters.orientation+numpy.pi/2) * (-self.parameters.max_offset + z),
                                            y = self.parameters.y + numpy.sin(self.parameters.orientation+numpy.pi/2) * (-self.parameters.max_offset + z),
                                            trial=k))

    def do_analysis(self, data_store):
        pass


class MapSimpleGabor(VisualExperiment):
    """
    Map RF with a Gabor patch stimuli.

    This experiment presents a series of flashed Gabor patches at the centers
    of regular hexagonal tides with given range of orientations.

    
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    Other parameters
    ----------------
    relative_luminance : float
        Luminance of the Gabor patch relative to background luminance.
        0. is dark, 1.0 is double the background luminance.

    central_rel_lum : float
        Luminance of the Gabor patch at the center of the RF relative to
        background luminance.
        0. is dark, 1.0 is double the background luminance.

    orientation : float
        The initial orientation of the Gabor patch.

    phase : float
        The phase of the Gabor patch.

    spatial_frequency : float
        The spatial freqency of the Gabor patch.

    rotations : int
        Number of different orientations at each given place.
        1 only one Gabor patch with initial orientation will be presented at
        given place, N>1 N different orientations will be presented, 
        orientations are uniformly distributed between [0, 2*pi) + orientation.

    size : float
        Size of the tides. From this value the size of Gabor patch is derived 
        so that it fits into a circle with diameter equal to this size.

        Gabor patch size is set so that sigma of Gaussian envelope is size/3

    x : float
        The x corrdinates of the central tide.

    y : float
        The y corrdinates of the central tide.

    flash_duration : float
        The duration of the presentation of a single Gabor patch. 

    duration : float
        The duration of single presentation of the stimulus.

    num_trials : int
        Number of trials each each stimulus is shown.

    circles : int
        Number of "circles" where the Gabor patch is presented.
        1: only at the central point the Gabor patch is presented, 
        2: stimuli are presented at the central hexagonal tide and 6 hexes 
        forming a "circle" around the central

    grid : bool
        If True hexagonal tiding with relative luminance 0 is drawn over the 
        stimmuli.
        Mostly for testing purposes to check the stimuli are generated 
        correctly.

    Note on hexagonal tiding:
    -------------------------
        Generating coordinates of centers of regular (!) hexagonal tidings.
        It is done this way, because the centers of tides are not on circles (!)
        First it generates integer indexed centers like this:
              . . .                (-2,2) (0, 2) (2,2)
             . . . .           (-3,1) (-1,1) (1,1) (3,1)
            . . . . .   ==> (-4,0) (-2,0) (0,0) (2,0) (4,0)     (circles=3)
             . . . .           (-3,-1)(-1,-1)(1,-1)(3,-1)
              . . .                (-2,-2)(0,-2)(2,-2)

        coordinates then multiplied by non-integer factor to get the right position
            x coordinate multiplied by factor 1/2*size
            y coordinate multiplied by factor sqrt(3)/2*size

    Note on central relative luminance:
    -----------------------------------
        In the experiment they had lower luminance for Gabor patches presented
        at the central tide
    """

    required_parameters = ParameterSet({
            'relative_luminance' : float,
            'central_rel_lum' : float,
            'orientation' : float,
            'phase' : float,
            'spatial_frequency' : float,
            'size' : float,
            'flash_duration' : float, 
            'x' : float,
            'y' : float,
            'rotations' : int,
            'duration' : float,
            'num_trials' : int,
            'circles' : int,
            'grid' : bool,
    })


    def __init__(self, model, parameters):
        VisualExperiment.__init__(self, model, parameters)
        if self.parameters.grid:
            # Grid is currently working only for special cases
            # Check if it is working
            assert self.parameters.x == 0, "X shift not yet implemented"
            assert self.parameters.y == 0, "Y shift not yet implemented"
            assert model.visual_field.size_x == model.visual_field.size_y, "Different sizes not yet implemented"
        for trial in xrange(0, self.parameters.num_trials):
            for rot in xrange(0, self.parameters.rotations):
                for row in xrange(self.parameters.circles-1, -self.parameters.circles,-1):
                    colmax =  2*self.parameters.circles-2 - abs(row)
                    for column in xrange(-colmax, colmax + 1, 2):
                        # central coordinates of presented Gabor patch
                        # relative to the central tide
                        x = column*0.5*self.parameters.size
                        y = row*0.5*self.parameters.size  
                        # different luminance for central tide
                        if column == 0 and row == 0:
                            rel_lum = self.parameters.central_rel_lum
                        else:
                            rel_lum = self.parameters.relative_luminance
                        self.stimuli.append(
                            topo.SimpleGaborPatch(
                                frame_duration = self.frame_duration,
                                duration=self.parameters.duration,
                                flash_duration = self.parameters.flash_duration,
                                size_x=model.visual_field.size_x,
                                size_y=model.visual_field.size_y,
                                background_luminance=self.background_luminance,
                                relative_luminance = rel_lum,
                                orientation = (self.parameters.orientation 
                                            + numpy.pi*rot/self.parameters.rotations),
                                density=self.density,
                                phase = self.parameters.phase,
                                spatial_frequency = self.parameters.spatial_frequency,
                                size = self.parameters.size,
                                x = self.parameters.x + x,
                                y = self.parameters.y + y,
                                location_x=0.0,
                                location_y=0.0,
                                trial=trial))

    def do_analysis(self, data_store):
        pass


class MapTwoStrokeGabor(VisualExperiment):
    """
    Map RF with a two stroke Gabor patch stimuli to study response on apparent
    movement. First a Gabor patch is presented for specified time after that
    another Gabor patch is presented at neighbohring tide with same orientation
    and other properties.

    There are two configuration for the movement:
        ISO i.e. Gabor patch moves parallel to its orientation
        CROSS i.e. Gabor patch moves perpendicular to its orientation
        
        In any case it has to move into another tide, therefore orientation 
        determines the configuration

  
    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    Other parameters
    ----------------
    relative_luminance : float
        Luminance of the Gabor patch relative to background luminance.
        0. is dark, 1.0 is double the background luminance.

    central_rel_lum : float
        Luminance of the Gabor patch at the center of the RF relative to
        background luminance.
        0. is dark, 1.0 is double the background luminance.

    orientation : float
        The initial orientation of the Gabor patch.
        This changes orientation of the whole experiment, i.e. it also rotates 
        the grid (because of the iso and cross configurations of movements).

    phase : float
        The phase of the Gabor patch.

    spatial_frequency : float
        The spatial freqency of the Gabor patch.

    rotations : int
        Number of different orientations at each given place.
        1 only one Gabor patch with initial orientation will be presented at
        given place, N>1 N different orientations will be presented, 
        orientations are uniformly distributed between [0, 2*pi) + orientation.

    size : float
        Size of the tides. From this value the size of Gabor patch is derived 
        so that it fits into a circle with diameter equal to this size.

        Gabor patch size is set so that sigma of Gaussian envelope is size/3

    x : float
        The x corrdinates of the central tide.

    y : float
        The y corrdinates of the central tide.

    stroke_time : float
        The duration of the first stroke of Gabor patch

    flash_duration : float
        The total duration of the presentation of Gabor patches. Therefore,
        the second stroke is presented for time equal: 
            flash_duration - stroke_tim 

    duration : float
        The duration of single presentation of the stimulus.

    num_trials : int
        Number of trials each each stimulus is shown.

    circles : int
        Number of "circles" where the Gabor patch is presented.
        1: only at the central point the Gabor patch is presented, 
        2: stimuli are presented at the central hexagonal tide and 6 hexes 
        forming a "circle" around the central
        Trajectories starting or ending in the given number of circles are
        used, i.e. First Gabor patch can be out of the circles and vice versa.

    grid : bool
        If True hexagonal tiding with relative luminance 0 is drawn over the 
        stimmuli.
        Mostly for testing purposes to check the stimuli are generated 
        correctly.

    Note on hexagonal tiding:
    -------------------------
        Generating coordinates of centers of regular (!) hexagonal tidings.
        It is done this way, because the centers of tides are not on circles (!)
        First it generates integer indexed centers like this:
              . . .                (-2,2) (0, 2) (2,2)
             . . . .           (-3,1) (-1,1) (1,1) (3,1)
            . . . . .   ==> (-4,0) (-2,0) (0,0) (2,0) (4,0)     (circles=3)
             . . . .           (-3,-1)(-1,-1)(1,-1)(3,-1)
              . . .                (-2,-2)(0,-2)(2,-2)

        coordinates then multiplied by non-integer factor to get the right position
            x coordinate multiplied by factor 1/2*size
            y coordinate multiplied by factor sqrt(3)/2*size

    Note on central relative luminance:
    -----------------------------------
        In the experiment they had lower luminance for Gabor patches presented
        at the central tide


    Note on number of circles:
    --------------------------
        For 2 stroke the experiment includes also the trajectories that
        start inside the defined number of circles but get out as well as 
        trajectories starting in the outside layer of tides comming inside.

        For example if we have number of circles = 2 -> that means we have 
        central tide and the first circle of tides around, but for two stroke
        it is possible we start with Gabor patch at the distance 2 tides away
        from the central tide (i.e. tides that are in circles = 3) if we move 
        inside and vice versa.

        This is solved by checking the distance of the final position of the 
        Gabor patch, if the distance is bigger than a radius of a circle
        then opposite direction is taken into account.
        
        Since we have hexagonal tides this check is valid only for 
        n <= 2/(2-sqrt(3)) ~ 7.5 
        which is for given purposes satisfied, but should be mentioned.

    Note on rotations:
    ------------------
        This number is taken as a free parameter, but to replicate hexagonal
        tiding this number has to be 6 or 1 or 2. The code exploits symmetry and
        properties of the hexagonal tiding rather a lot!
        The ISO/CROSS configuration is determined from this number, so any other
        number generates moving paterns but in directions not matching hexes.

    """

    required_parameters = ParameterSet({
            'relative_luminance' : float,
            'central_rel_lum' : float,
            'orientation' : float,
            'phase' : float,
            'spatial_frequency' : float,
            'size' : float,
            'flash_duration' : float, 
            'x' : float,
            'y' : float,
            'rotations' : int,
            'duration' : float,
            'num_trials' : int,
            'circles' : int,
            'stroke_time' : float,
            'grid' : bool,
            })  


    def __init__(self, model, parameters):
        VisualExperiment.__init__(self, model, parameters)
        # Assert explained in docstring
        assert self.parameters.circles < 7, "Too many circles, this won't work"
        if self.parameters.grid:
            # Grid is currently working only for special cases
            # Check if it is working
            assert self.parameters.orientation == 0., "Rotated grid is not implemented"
            assert self.parameters.x == 0, "X shift not yet implemented"
            assert self.parameters.y == 0, "Y shift not yet implemented"
            assert model.visual_field.size_x == model.visual_field.size_y, "Different sizes not yet implemented"


        for trial in xrange(0, self.parameters.num_trials):
            for rot in xrange(0, self.parameters.rotations):
                for row in xrange(self.parameters.circles-1, -self.parameters.circles,-1):
                    colmax =  2*self.parameters.circles-2 - abs(row)
                    for column in xrange(-colmax, colmax + 1, 2):
                        for direction in (-1,1):
                            # central coordinates of presented Gabor patch
                            # relative to the central tide
                            x = column*0.5*self.parameters.size
                            y = row*0.5*numpy.sqrt(3)*self.parameters.size  
                            # rotation of the Gabor
                            angle = (self.parameters.orientation 
                                    + numpy.pi*rot/self.parameters.rotations)
                            if rot%2 == 0: # even rotations -> iso config
                                # Gabor orientation 0 -> horizontal
                                x_dir = numpy.cos(angle)*self.parameters.size
                                y_dir = numpy.sin(angle)*self.parameters.size
                            else:  # odd rotations -> cross config
                                # cross config means moving into perpendicular
                                # direction (aka + pi/2)
                                x_dir = -numpy.sin(angle)*self.parameters.size
                                y_dir = numpy.cos(angle)*self.parameters.size

                            # starting in the central tide
                            if x == 0 and y == 0:
                                first_rel_lum = self.parameters.central_rel_lum
                                second_rel_lum = self.parameters.relative_luminance
                            # ending in the central tide
                            elif ((abs(x + x_dir*direction) < self.parameters.size/2.) and
                                  (abs(y + y_dir*direction) < self.parameters.size/2.)):
                                first_rel_lum = self.parameters.relative_luminance
                                second_rel_lum = self.parameters.central_rel_lum
                            # far from the central tide
                            else:
                                first_rel_lum = self.parameters.relative_luminance
                                second_rel_lum = self.parameters.relative_luminance


                            # If the Gabor patch ends in outer circle
                            # we want also Gabor moving from outer circle to 
                            # inner circles 
                            # This condition is approximated by concentric 
                            # circles more in docstring
                            outer_circle = numpy.sqrt((x+x_dir*direction)**2 
                                        + (y+y_dir*direction)**2) > (
                                                (self.parameters.circles-1)
                                                *self.parameters.size)

                            # range here is 1 or 2
                            # In case of outer_circle == True generates two
                            # experiments, from and into the outer circle
                            # In case of outer_circle == False generates only
                            # one experiment
                            for inverse in xrange(1+outer_circle):
                                self.stimuli.append(
                                    topo.TwoStrokeGaborPatch(
                                        frame_duration = self.frame_duration,
                                        duration=self.parameters.duration,
                                        flash_duration = self.parameters.flash_duration,
                                        size_x=model.visual_field.size_x,
                                        size_y=model.visual_field.size_y,
                                        background_luminance=self.background_luminance,
                                        first_relative_luminance = first_rel_lum,
                                        second_relative_luminance = second_rel_lum,
                                        orientation = angle,
                                        density=self.density,
                                        phase = self.parameters.phase,
                                        spatial_frequency = self.parameters.spatial_frequency,
                                        size = self.parameters.size,
                                        x = self.parameters.x + x + inverse*x_dir*direction,
                                            # inverse == 0 -> original start
                                            # inverse == 1 -> start from end
                                        y = self.parameters.y + y + inverse*y_dir*direction,
                                        location_x=0.0,
                                        location_y=0.0,
                                        trial=trial,
                                        stroke_time=self.parameters.stroke_time,
                                        x_direction=x_dir*direction*((-1)**inverse),
                                            # (-1)**inverse = 1 for original one
                                            # == -1 for the inverse movement
                                        y_direction=y_dir*direction*((-1)**inverse),
                                        grid=self.parameters.grid,
                                        ))
                                # For the inverse movement we have to 
                                # switch the luminances
                                first_rel_lum, second_rel_lum = second_rel_lum, first_rel_lum

    def do_analysis(self, data_store):
        pass
