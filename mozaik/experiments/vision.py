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

    Other parameters
    ----------------

    shuffle_stimuli: bool
                     If True, stumuli are shuffled randomly


    """

    required_parameters = ParameterSet({
                                        'shuffle_stimuli': bool,
                                       })

    def generate_stimuli(self):
        """
        Experiments should implement this method and build the `self.stimuli` list there
        """
        raise NotImplementedError()

    def __init__(self,model,parameters):
        Experiment.__init__(self, model,parameters)
        self.background_luminance = model.input_space.background_luminance
      
        #JAHACK: This is kind of a hack now. There needs to be generally defined interface of what is the spatial and temporal resolution of a visual input layer
        # possibly in the future we could force the visual_space to have resolution, perhaps something like native_resolution parameter!?
        self.density  = 1/model.input_layer.parameters.receptive_field.spatial_resolution # in pixels per degree of visual space 
        self.frame_duration = model.input_space.parameters.update_interval # in pixels per degree of visual space 

        if self.parameters.shuffle_stimuli:
            mozaik.rng.shuffle(self.stimuli)

class MeasureFlatLuminanceSensitivity(VisualExperiment):
    """
    Measure luminance sensitivity using flat luminance screen.

    This experiment will measure luminance sensitivity by presenting a series of full-field 
    constant stimulations (i.e. all pixels of the virtual visual space will be set to a 
    constant value) of different magnitudes. The user can specify the luminance levels that
    hould be presented (see the *luminances*) parameter, the length  of presentation of 
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
    
    def generate_stimuli(self):
        
        # stimuli creation        
        for l in self.parameters.luminances:
            for k in range(0, self.parameters.num_trials):
                self.stimuli.append( topo.Null(
                    frame_duration = self.frame_duration,
                    size_x=self.model.visual_field.size_x,
                    size_y=self.model.visual_field.size_y,
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
    
    def generate_stimuli(self):
    
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
    
    def generate_stimuli(self):

        self.direct_stimulation = []

        for k in range(0, self.parameters.num_trials):
            d  = OrderedDict()
            p = MozaikExtendedParameterSet({
                                'population_selector' : self.parameters.stimulation_configuration,
                                'current' : self.parameters.stimulation_current
                               })

            d[self.parameters.stimulation_sheet] = [Depolarization(self.model.sheets[self.parameters.stimulation_sheet],p)]
            
            self.direct_stimulation.append(d)     

        p['sheet'] = self.parameters.stimulation_sheet

    
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
                            grid = self.parameters.grid,
                            direct_stimulation_name='Injection',
                            direct_stimulation_parameters = p
                          ))
            
             

    def do_analysis(self, data_store):
        pass


class MeasureSparseBar(VisualExperiment):
    """
    Sparse noise stimulation experiments with bars instead of pixels.

    This experiment will show a series of images formed by a single bar
    which will be presented in a random position along the axis perpendicular
    to the specified orientation in each trial.

    If possible, given the total number of images, the number of bar presentations
    at each position for white and black colors will be equal. If the total number
    of images is not divisible by 2*number_of_positions, the presentations may have
    a slight left and black bias.

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
        The total number of images that will be presented. For mapping
        a receptive field using black and white bars, it should be
        2 * n_positions * n_presentations_per_position.

    num_trials : int
        Number of trials each each stimulus is shown.

    orientation : float
        The orientation of the bars, in radians.

    bar_length : float
        The length of the presented bars.

    bar_width : float
        The width of the presented bars.

    x : float
        x coordinate of the center of the stimulus area

    y : float
        y coordinate of the center of the stimulus area

    n_positions : int
        Number of positions to present the bars at. The positions
        are spread symmetrically around the center.

    experiment_seed : int
        Random seed for the bar positions.
    """

    required_parameters = ParameterSet({
            'time_per_image': float,
            'blank_time' : float,
            'total_number_of_images' : int,
            'num_trials' : int,
            'orientation' : float,
            'bar_length' : float,
            'bar_width' : float,
            'x' : float,
            'y' : float,
            'n_positions' : int,
            'experiment_seed' : int,
    })

    def generate_stimuli(self):
        common_params = {
            "frame_duration" : self.frame_duration,
            "size_x" : self.model.visual_field.size_x,
            "size_y" : self.model.visual_field.size_y,
            "location_x" : 0.0,
            "location_y" : 0.0,
            "background_luminance" : self.background_luminance,
            "duration" : self.parameters.time_per_image + self.parameters.blank_time,
            "density" : self.density,
            "orientation" : self.parameters.orientation,
            "width" : self.parameters.bar_width,
            "length" : self.parameters.bar_length,
            "flash_duration" : self.parameters.time_per_image
        }
        rng = numpy.random.default_rng(self.parameters.experiment_seed)
        for trials in range(0, self.parameters.num_trials):

            stims = []
            radius = (self.parameters.n_positions / 2 - 0.5) * self.parameters.bar_width
            x_pos = self.parameters.x + numpy.linspace(-radius,radius,self.parameters.n_positions) * numpy.sin(self.parameters.orientation)
            y_pos = self.parameters.y - numpy.linspace(-radius,radius,self.parameters.n_positions) * numpy.cos(self.parameters.orientation)
            l = len(x_pos)
            for i in range(self.parameters.total_number_of_images):
                stim = topo.FlashedBar(
                    relative_luminance = i % (2 * l) < l,
                    x = x_pos[i % l],
                    y = y_pos[i % l],
                    **common_params
                    )
                stims.append(stim)
            rng.shuffle(stims)
            self.stimuli.extend(stims)

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
    
    def generate_stimuli(self):

        for k in range(0, self.parameters.num_trials):
            self.stimuli.append(topo.DenseNoise(
                            frame_duration = self.frame_duration,
                            time_per_image = self.parameters.time_per_image,
                            duration = self.parameters.total_number_of_images * self.parameters.time_per_image, 
                            size_x=self.model.visual_field.size_x,
                            size_y=self.model.visual_field.size_y,
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

    def generate_stimuli(self):

        stimulus_parameters = []
        # stimuli creation        
        for c in self.parameters.contrasts:
            for i in range(0, self.parameters.num_orientations):
                for k in range(0, self.parameters.num_trials):
                    self.stimuli.append(topo.FullfieldDriftingSinusoidalGrating(
                                    frame_duration = self.frame_duration,
                                    size_x=self.model.visual_field.size_x,
                                    size_y=self.model.visual_field.size_y,
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
    
    def generate_stimuli(self):
        for c in self.parameters.contrasts:
            for i in range(0, self.parameters.num_orientations):
                for k in range(0, self.parameters.num_trials):
                    self.stimuli.append(topo.FullfieldDriftingSinusoidalGratingA(
                    frame_duration = self.frame_duration,
                                    size_x=self.model.visual_field.size_x,
                                    size_y=self.model.visual_field.size_y,
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
    
    orientations : list(float)
                The orientations (in radians) at which to measure the size tuning. (in future this will become automated)

    positions : list(tuple(float,float)) 
              List of coordinates of each of the positions where the stimulus should be shown

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
            'orientations' : list,
            'positions' : list,
            'spatial_frequency' : float,
            'temporal_frequency' : float,
            'grating_duration' : float,
            'contrasts' : list,
            'num_trials' : int,
            'log_spacing' : bool,
    })

    def generate_stimuli(self):

        # linear or logarithmic spaced sizes
        if self.parameters.log_spacing:
            base2max = numpy.log2(self.parameters.max_size)
            sizes = numpy.logspace(start=-3.0, stop=base2max, num=self.parameters.num_sizes, base=2.0)
        else:
            sizes = numpy.linspace(0, self.parameters.max_size,self.parameters.num_sizes)

        stimulus_parameters = []
        # stimuli creation        
        for c in self.parameters.contrasts:
            for o in self.parameters.orientations:
                for x, y in self.parameters.positions:
                    for s in sizes:
                        for k in range(0, self.parameters.num_trials):
                            self.stimuli.append(topo.DriftingSinusoidalGratingDisk(
                                    frame_duration = self.frame_duration,
                                    size_x=self.model.visual_field.size_x,
                                    size_y=self.model.visual_field.size_y,
                                    location_x=x,
                                    location_y=y,
                                    background_luminance=self.background_luminance,
                                    contrast = c,
                                    duration=self.parameters.grating_duration,
                                    density=self.density,
                                    trial=k,
                                    orientation=o,
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


    def generate_stimuli(self):

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
                                    size_x=self.model.visual_field.size_x,
                                    size_y=self.model.visual_field.size_y,
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
    
    def generate_stimuli(self):
            
        # stimuli creation        
        for c in self.parameters.contrasts:
            for k in range(0, self.parameters.num_trials):
                self.stimuli.append(topo.FullfieldDriftingSinusoidalGrating(
                    frame_duration = self.frame_duration,
                    size_x=self.model.visual_field.size_x,
                    size_y=self.model.visual_field.size_y,
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
    
    def generate_stimuli(self):
            
        # stimuli creation        
        for c in self.parameters.contrasts:
            for k in range(0, self.parameters.num_trials):
                self.stimuli.append(topo.FullfieldDriftingSinusoidalGratingA(
                    frame_duration = self.frame_duration,
                    size_x=self.model.visual_field.size_x,
                    size_y=self.model.visual_field.size_y,
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
    

    def generate_stimuli(self):
            
        # stimuli creation        
        for tf in self.parameters.temporal_frequencies:
            for sf in self.parameters.spatial_frequencies:
                for c in self.parameters.contrasts:
                    for k in range(0, self.parameters.num_trials):
                        if self.parameters.square:
                            self.stimuli.append(topo.FullfieldDriftingSquareGrating(
                                frame_duration = self.frame_duration,
                                size_x=self.model.visual_field.size_x,
                                size_y=self.model.visual_field.size_y,
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
                                size_x=self.model.visual_field.size_x,
                                size_y=self.model.visual_field.size_y,
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

    def generate_stimuli(self):
        
        for c in self.parameters.contrasts:
            for i in range(0, self.parameters.num_orientations):
                for k in range(0, self.parameters.num_trials):
                    self.stimuli.append(
                        topo.DriftingSinusoidalGratingCenterSurroundStimulus(
                                    frame_duration = self.frame_duration,
                                    size_x=self.model.visual_field.size_x,
                                    size_y=self.model.visual_field.size_y,
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

    def generate_stimuli(self):
        
         # the orientation is fixed to horizontal
        orientation = 0 #numpy.pi/2
        # SQUARED GRATINGS       
        for sf in self.parameters.spatial_frequencies:
            for k in range(0, self.parameters.num_trials):
                self.stimuli.append(
                    topo.FullfieldDriftingSquareGrating(
                        frame_duration = self.frame_duration,
                        size_x=self.model.visual_field.size_x,
                        size_y=self.model.visual_field.size_y,
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
                        size_x=self.model.visual_field.size_x,
                        size_y=self.model.visual_field.size_y,
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
    
    stimulus_duration : float
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

    
    def generate_stimuli(self):
        
        for k in range(0, self.parameters.num_trials):
            self.stimuli.append(
                topo.NaturalImageWithEyeMovement(
                            frame_duration = self.frame_duration,
                            size_x=self.model.visual_field.size_x,
                            size_y=self.model.visual_field.size_y,
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
    
    def generate_stimuli(self):
        
        for k in range(0, self.parameters.num_trials):
            self.stimuli.append(
                topo.DriftingGratingWithEyeMovement(
                            frame_duration = self.frame_duration,
                            size_x=self.model.visual_field.size_x,
                            size_y=self.model.visual_field.size_y,
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
    
    def generate_stimuli(self):
            
            for k in range(0,self.parameters.num_trials):
                self.stimuli.append(
                            topo.Null(   
                                frame_duration = self.frame_duration,
                                size_x=self.model.visual_field.size_x,
                                size_y=self.model.visual_field.size_y,
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
    
    def generate_stimuli(self):
        for k in range(0, self.parameters.num_trials):
            for s in range(0, self.parameters.steps):
                self.stimuli.append(
                    topo.FlashedBar(
                                frame_duration = self.frame_duration,
                                size_x=self.model.visual_field.size_x,
                                size_y=self.model.visual_field.size_y,
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
    
    def generate_stimuli(self):
        for k in range(0, self.parameters.num_trials):
              for obw in self.parameters.occlusion_bar_width:  
                            self.stimuli.append(
                                topo.FlashedInterruptedBar(
                                            frame_duration = 7,
                                            size_x=self.model.visual_field.size_x,
                                            size_y=self.model.visual_field.size_y,
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
            'size_x': float, # The size of the stimulus on the x-axis
            'size_y': float, # The size of the stimulus on the y-axis
    })  

    def generate_stimuli(self):
	# we place this import here to avoid the need for octave dependency unless this experiment is actually used.
        import mozaik.stimuli.vision.texture_based as textu #vf

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

    def generate_stimuli(self):
        # we place this import here to avoid the need for octave dependency unless this experiment is actually used.
        import mozaik.stimuli.vision.texture_based as textu #vf
            
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
                                size_x=self.model.visual_field.size_x,
                                size_y=self.model.visual_field.size_y,
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

    def generate_stimuli(self):
        # we place this import here to avoid the need for octave dependency unless this experiment is actually used.
        import mozaik.stimuli.vision.texture_based as textu #vf

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
                            size_x=self.model.visual_field.size_x,
                            size_y=self.model.visual_field.size_y,
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

    def generate_stimuli(self):
        # we place this import here to avoid the need for octave dependency unless this experiment is actually used.
        import mozaik.stimuli.vision.texture_based as textu #vf

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
                            size_x=self.model.visual_field.size_x,
                            size_y=self.model.visual_field.size_y,
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
    
    def generate_stimuli(self):
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
                                            size_x=self.model.visual_field.size_x,
                                            size_y=self.model.visual_field.size_y,
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
    
    def generate_stimuli(self):
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
                                            size_x=self.model.visual_field.size_x,
                                            size_y=self.model.visual_field.size_y,
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
                                            size_x=self.model.visual_field.size_x,
                                            size_y=self.model.visual_field.size_y,
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
