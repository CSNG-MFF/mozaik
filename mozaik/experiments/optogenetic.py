import os
import numpy as np
from mozaik.experiments import Experiment
from parameters import ParameterSet
from mozaik.stimuli import InternalStimulus
from mozaik.tools.distribution_parametrization import MozaikExtendedParameterSet
from collections import OrderedDict
from mozaik.sheets.direct_stimulator import OpticalStimulatorArrayChR
import matplotlib
from copy import deepcopy
import random


class CorticalStimulationWithOptogeneticArray(Experiment):
    """
    Parent class for optogenetic stimulation of cortical sheets with an array
    of light sources.

    Creates a array of optical stimulators covering an area of cortex, and then
    stimulates the array based on the stimulating_signal function in the
    localstimulationarray_parameters.

    Does not show any actual visual stimulus.

    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

    Other parameters
    ----------------

    sheet_list : list(str)
                The list of sheets in which to do stimulation.

    sheet_intensity_scaler : list(float)
                Scale the stimulation intensity of each sheet in sheet_list by the
                constants in this list. Must have equal length to sheet_list.
                The constants must be in the range (0,infinity)

    sheet_transfection_proportion : list(float)
                Set the proportion of transfected cells in each sheet in sheet_list.
                Must have equal length to sheet_list. The constants must be in the
                range (0,1) - 0 means no cells, 1 means all cells.

    num_trials : int
                Number of trials each stimulus is shown.

    stimulator_array_parameters : ParameterSet
                Parameters for the optical stimulator array:
                    size : float (μm)
                    spacing : float (μm)
                    update_interval : float (ms)
                    depth_sampling_step : float (μm)
                    light_source_light_propagation_data : str
                These parameters are the same as the parameters of
                mozaik.sheets.direct_stimulator.OpticalStimulatorArrayChR class,
                except that it must not contain the parameters
                *stimulating_signal* and *stimulating_signal_parameters* - those
                must be set by the specific experiments.
    """

    required_parameters = ParameterSet(
        {
            "sheet_list": list,
            "sheet_intensity_scaler": list,
            "sheet_transfection_proportion": list,
            "num_trials": int,
            "stimulator_array_parameters": ParameterSet,
        }
    )

    def __init__(self, model, parameters):
        Experiment.__init__(self, model, parameters)
        stimulator_array_keys = {
            "size",
            "spacing",
            "update_interval",
            "depth_sampling_step",
            "light_source_light_propagation_data",
        }
        p = self.parameters
        assert len(p.sheet_list) == len(p.sheet_intensity_scaler), (
            "sheet_list and sheet_intensity_scaler must have equal lengths, not %d and %d"
            % (
                len(p.sheet_list),
                len(p.sheet_intensity_scaler),
            )
        )
        assert len(p.sheet_list) == len(p.sheet_transfection_proportion), (
            "sheet_list and sheet_transfection_proportion must have equal lengths, not %d and %d!"
            % (
                len(p.sheet_list),
                len(p.sheet_transfection_proportion),
            )
        )
        assert all(
            [s >= 0 for s in p.sheet_intensity_scaler]
        ), "Sheet intensity scalers must be larger than 0!"
        assert all(
            [s >= 0 and s <= 1 for s in p.sheet_transfection_proportion]
        ), "Sheet transfection proportions must be in the range of (0,1)!"
        assert (
            p.stimulator_array_parameters.keys() == stimulator_array_keys
        ), "Stimulator array keys must be: %s. Supplied: %s. Difference: %s" % (
            stimulator_array_keys,
            p.stimulator_array_parameters.keys(),
            set(stimulator_array_keys) ^ set(p.stimulator_array_parameters.keys()),
        )

    def append_direct_stim(self, model, stimulator_array_parameters):
        p = self.parameters
        if self.direct_stimulation == None:
            self.direct_stimulation = []
            self.shared_scs = OrderedDict((sheet, {}) for sheet in p.sheet_list)

        d = OrderedDict()
        for k in range(len(p.sheet_list)):
            sheet = p.sheet_list[k]
            sap = MozaikExtendedParameterSet(deepcopy(stimulator_array_parameters))
            sap.stimulating_signal_parameters.intensity *= p.sheet_intensity_scaler[k]
            sap.transfection_proportion = p.sheet_transfection_proportion[k]
            d[sheet] = [
                OpticalStimulatorArrayChR(
                    model.sheets[sheet], sap, self.shared_scs[sheet]
                )
            ]
            self.shared_scs[sheet].update(
                {
                    d[sheet][0].stimulated_cells[i]: d[sheet][0].scs[i]
                    for i in range(len(d[sheet][0].scs))
                }
            )

        sap = MozaikExtendedParameterSet(deepcopy(stimulator_array_parameters))
        sap["sheet_list"] = p.sheet_list
        sap["sheet_intensity_scaler"] = p.sheet_intensity_scaler
        sap["sheet_transfection_proportion"] = p.sheet_transfection_proportion
        for trial in range(p.num_trials):
            self.direct_stimulation.append(d)
            self.stimuli.append(
                InternalStimulus(
                    frame_duration=sap.stimulating_signal_parameters.duration,
                    duration=sap.stimulating_signal_parameters.duration,
                    trial=trial,
                    direct_stimulation_name="OpticalStimulatorArrayChR",
                    direct_stimulation_parameters=sap,
                )
            )


class SingleOptogeneticArrayStimulus(CorticalStimulationWithOptogeneticArray):
    """
    Optogenetic stimulation of cortical sheets with an array of light sources.

    Creates a array of optical stimulators covering an area of cortex, and then
    stimulates the array based on the stimulating_signal function in the
    stimulator_array_parameters, *num_trials* times.

    Does not show any actual visual stimulus.

    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

          
    Other parameters
    ----------------

    sheet_list : int
                The list of sheets in which to do stimulation.

                
    sheet_intensity_scaler : list(float)
                Scale the stimulation intensity of each sheet in sheet_list by the
                constants in this list. Must have equal length to sheet_list.
                The constants must be in the range (0,infinity)

                
    sheet_transfection_proportion : list(float)
                Set the proportion of transfected cells in each sheet in sheet_list.
                Must have equal length to sheet_list. The constants must be in the
                range (0,1) - 0 means no cells, 1 means all cells.

                
    num_trials : int
                Number of trials each stimulus is shown.

                
    stimulator_array_parameters : ParameterSet
                Parameters for the optical stimulator array:
                    size : float (μm)
                    spacing : float (μm)
                    update_interval : float (ms)
                    depth_sampling_step : float (μm)
                    light_source_light_propagation_data : str
                These parameters are the same as the parameters of
                mozaik.sheets.direct_stimulator.OpticalStimulatorArrayChR class,
                except that it must not contain the parameters
                *stimulating_signal* and *stimulating_signal_parameters* - those
                are to be entered separately in this experiment.

                
    stimulating_signal : str
                Described in more detail in
                mozaik.sheets.direct_stimulator.OpticalStimulatorArrayChR.
                Path to a function that specifying the timecourse of the optogenetic
                array stimulation, should have the following form:

                def stimulating_signal_function(sheet, coor_x, coor_y, update_interval, parameters)

                sheet - Mozaik Sheet to be stimulated
                coor_x - x coordinates of the stimulators
                coor_y - y coordinates of the stimulators
                update_interval - timestep in which the stimulator updates
                parameters - any extra user parameters for the function

                It should return a 3D numpy array of size:
                coor_x.shape[0] x coor_x.shape[1] x (stimulation_duration/update_interval)

                
    stimulating_signal_parameters : ParameterSet
                Extra user parameters for the `stimulating_signal` function,
                described above.

    """

    required_parameters = ParameterSet(
        {
            "sheet_list": list,
            "sheet_intensity_scaler": list,
            "sheet_transfection_proportion": list,
            "num_trials": int,
            "stimulator_array_parameters": ParameterSet,
            "stimulating_signal": str,
            "stimulating_signal_parameters": ParameterSet,
        }
    )

    def __init__(self, model, parameters):
        CorticalStimulationWithOptogeneticArray.__init__(self, model, parameters)
        self.parameters.stimulator_array_parameters["stimulating_signal"] = (
            self.parameters.stimulating_signal
        )
        self.parameters.stimulator_array_parameters["stimulating_signal_parameters"] = (
            self.parameters.stimulating_signal_parameters
        )

        self.append_direct_stim(model, self.parameters.stimulator_array_parameters)


class OptogeneticArrayStimulusCircles(CorticalStimulationWithOptogeneticArray):
    """
    Optogenetic stimulation of cortical sheets with an array of light sources,
    in the pattern of filled circles.

    Does not show any actual visual stimulus.

    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

          
    Other parameters
    ----------------

    sheet_list : int
                The list of sheets in which to do stimulation.

                
    sheet_intensity_scaler : list(float)
                Scale the stimulation intensity of each sheet in sheet_list by the
                constants in this list. Must have equal length to sheet_list.
                The constants must be in the range (0,infinity)

                
    sheet_transfection_proportion : list(float)
                Set the proportion of transfected cells in each sheet in sheet_list.
                Must have equal length to sheet_list. The constants must be in the
                range (0,1) - 0 means no cells, 1 means all cells.

                
    num_trials : int
                Number of trials each stimulus is shown.

                
    stimulator_array_parameters : ParameterSet
                Parameters for the optical stimulator array:
                    size : float (μm)
                    spacing : float (μm)
                    update_interval : float (ms)
                    depth_sampling_step : float (μm)
                    light_source_light_propagation_data : str
                These parameters are the same as the parameters of
                mozaik.sheets.direct_stimulator.OpticalStimulatorArrayChR class,
                except that it must not contain the parameters
                *stimulating_signal* and *stimulating_signal_parameters* - those
                are set by this experiment.

                
    intensities : list(float)
                Intensities of the stimulation. Uniform across the circle.

                
    radii : list(float (μm))
                List of circle radii (μm) to present

                
    x_center : float (μm)
                Circle center x coordinate.

                
    y_center : float (μm)
                Circle center y coordinate.

                
    inverted: bool
                If true, everything in the circle has value 0, everything outside has
                the value *intensity*

                
    duration : float (ms)
            Overall stimulus duration

            
    onset_time : float (ms)
            Time point when the stimulation turns on

            
    offset_time : float(ms)
            Time point when the stimulation turns off


    """

    required_parameters = ParameterSet(
        {
            "sheet_list": list,
            "sheet_intensity_scaler": list,
            "sheet_transfection_proportion": list,
            "num_trials": int,
            "stimulator_array_parameters": ParameterSet,
            "intensities": list,
            "radii": list,
            "x_center": float,
            "y_center": float,
            "inverted": bool,
            "duration": int,
            "onset_time": int,
            "offset_time": int,
        }
    )

    def __init__(self, model, parameters):
        CorticalStimulationWithOptogeneticArray.__init__(self, model, parameters)
        sap = self.parameters.stimulator_array_parameters
        sap["stimulating_signal"] = (
            "mozaik.sheets.direct_stimulator.stimulating_pattern_flash"
        )
        sap["stimulating_signal_parameters"] = ParameterSet(
            {
                "shape": "circle",
                "intensity": 0,
                "coords": (self.parameters.x_center, self.parameters.y_center),
                "radius": 0,
                "inverted": self.parameters.inverted,
                "duration": self.parameters.duration,
                "onset_time": self.parameters.onset_time,
                "offset_time": self.parameters.offset_time,
            }
        )

        for intensity in self.parameters.intensities:
            for r in self.parameters.radii:
                sap.stimulating_signal_parameters.intensity = intensity
                sap.stimulating_signal_parameters.radius = r
                self.append_direct_stim(model, sap)


class OptogeneticArrayStimulusHexagonalTiling(CorticalStimulationWithOptogeneticArray):
    """
    Optogenetic stimulation of cortical sheets with an array of light sources,
    in the pattern of filled hexagons. These hexagons tile the entire span of
    the stimulation array, such that no hexagon has any parts outside of the
    array.

    Does not show any actual visual stimulus.

    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

          
    Other parameters
    ----------------

    sheet_list : int
                The list of sheets in which to do stimulation.

                
    sheet_intensity_scaler : list(float)
                Scale the stimulation intensity of each sheet in sheet_list by the
                constants in this list. Must have equal length to sheet_list.
                The constants must be in the range (0,infinity)

                
    sheet_transfection_proportion : list(float)
                Set the proportion of transfected cells in each sheet in sheet_list.
                Must have equal length to sheet_list. The constants must be in the
                range (0,1) - 0 means no cells, 1 means all cells.

                
    num_trials : int
                Number of trials each stimulus is shown.

                
    stimulator_array_parameters : ParameterSet
                Parameters for the optical stimulator array:
                    size : float (μm)
                    spacing : float (μm)
                    update_interval : float (ms)
                    depth_sampling_step : float (μm)
                    light_source_light_propagation_data : str
                These parameters are the same as the parameters of
                mozaik.sheets.direct_stimulator.OpticalStimulatorArrayChR class,
                except that it must not contain the parameters
                *stimulating_signal* and *stimulating_signal_parameters* - those
                are set by this experiment.

                
    intensities : list(float)
                Intensities of the stimulation. Uniform across the circle.

                
    radius : float (μm)
                Radius (or edge length) of the hexagon to present

                
    x_center : float (μm)
                Circle center x coordinate.

    y_center : float (μm)
                Circle center y coordinate.

                
    angle : float (rad)
                Clockwise rotation angle of the hexagons. At angle 0, the hexagons
                are oriented such that two sides are parallel to the horizontal axis:
                 __
                /  \
                \__/

    shuffle: bool
                If true, shuffle the order of hexagon flashes in a single trial.
                The order does not change across trials.

                
    duration : float (ms)
            Overall stimulus duration

            
    onset_time : float (ms)
            Time point when the stimulation turns on

            
    offset_time : float(ms)
            Time point when the stimulation turns off

    """

    required_parameters = ParameterSet(
        {
            "sheet_list": list,
            "sheet_intensity_scaler": list,
            "sheet_transfection_proportion": list,
            "num_trials": int,
            "stimulator_array_parameters": ParameterSet,
            "intensities": list,
            "radius": float,
            "x_center": float,
            "y_center": float,
            "angle": float,
            "shuffle": bool,
            "duration": int,
            "onset_time": int,
            "offset_time": int,
        }
    )

    def __init__(self, model, parameters):
        CorticalStimulationWithOptogeneticArray.__init__(self, model, parameters)
        sap = self.parameters.stimulator_array_parameters
        sap["stimulating_signal"] = (
            "mozaik.sheets.direct_stimulator.stimulating_pattern_flash"
        )
        sap["stimulating_signal_parameters"] = ParameterSet(
            {
                "shape": "hexagon",
                "intensity": 0,
                "angle": self.parameters.angle,
                "coords": (0, 0),
                "radius": self.parameters.radius,
                "duration": self.parameters.duration,
                "onset_time": self.parameters.onset_time,
                "offset_time": self.parameters.offset_time,
            }
        )

        hc = self.hexagonal_tiling_centers(
            self.parameters.x_center,
            self.parameters.y_center,
            self.parameters.radius,
            self.parameters.angle,
            self.parameters.stimulator_array_parameters.size,
            self.parameters.stimulator_array_parameters.size,
            self.parameters.shuffle,
        )
        for intensity in self.parameters.intensities:
            for h in hc:
                sap.stimulating_signal_parameters.intensity = intensity
                sap.stimulating_signal_parameters.coords = h
                self.append_direct_stim(model, sap)

    def hexagonal_tiling_centers(
        self, x_c, y_c, radius, angle, xlen, ylen, shuffle=False
    ):
        xmax, ymax, xmin, ymin = xlen / 2, ylen / 2, -xlen / 2, -ylen / 2
        w = radius * np.sqrt(3) / 2
        r = radius

        hex_coords = []
        for y in np.arange(0, ymax, 3 * r):
            for x in np.arange(0, xmax, 2 * w):
                hex_coords.append([x, y])
                hex_coords.append([-x, y])
                hex_coords.append([x, -y])
                hex_coords.append([-x, -y])

        for y in np.arange(3 / 2 * r, ymax, 3 * r):
            for x in np.arange(w, xmax, 2 * w):
                hex_coords.append([x, y])
                hex_coords.append([-x, y])
                hex_coords.append([x, -y])
                hex_coords.append([-x, -y])

        transform = (
            matplotlib.transforms.Affine2D()
            .translate(x_c, y_c)
            .rotate_around(x_c, y_c, angle)
        )
        hex_coords = transform.transform_affine(np.array(hex_coords))

        hc = []
        for coord in hex_coords:
            x, y = coord
            if x + r > xmax or x - r < xmin or y + r > ymax or y - r < ymin:
                continue
            hc.append(coord)

        # Remove duplicate coordinates
        hc = sorted(list(set(map(tuple, hc))))

        if shuffle == True:
            random.shuffle(hc)
        return hc


class OptogeneticArrayImageStimulus(CorticalStimulationWithOptogeneticArray):
    """
    Optogenetic stimulation of cortical sheets with an array of light sources,
    in the pattern of a grayscale image, stored as a .npy file containing a 2D
    numpy array, with values between 0 (black) and 1 (white). If the image has a
    different aspect ratio or number of pixels as the stimulation array, it will
    be stretched to fit the array.

    The experiment accepts either the path to a single image, or a directory
    containing images. All .npy files in such a supplied directory have to
    be 2D and have values in the range of (0,1).

    The mapping between the axes of the numpy array and cortical space
    is 0->X, 1->Y.

    Does not show any actual visual stimulus.

    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

          
    Other parameters
    ----------------

    sheet_list : int
                The list of sheets in which to do stimulation.

                
    sheet_intensity_scaler : list(float)
                Scale the stimulation intensity of each sheet in sheet_list by the
                constants in this list. Must have equal length to sheet_list.
                The constants must be in the range (0,infinity)

                
    sheet_transfection_proportion : list(float)
                Set the proportion of transfected cells in each sheet in sheet_list.
                Must have equal length to sheet_list. The constants must be in the
                range (0,1) - 0 means no cells, 1 means all cells.

                
    num_trials : int
                Number of trials each stimulus is shown.

                
    stimulator_array_parameters : ParameterSet
                Parameters for the optical stimulator array:
                    size : float (μm)
                    spacing : float (μm)
                    update_interval : float (ms)
                    depth_sampling_step : float (μm)
                    light_source_light_propagation_data : str
                These parameters are the same as the parameters of
                mozaik.sheets.direct_stimulator.OpticalStimulatorArrayChR class,
                except that it must not contain the parameters
                *stimulating_signal* and *stimulating_signal_parameters* - those
                are set by this experiment.

                
    intensities : list(float)
                Intensities of the stimulation. Uniform across the circle.

                
    images_path : str
                Path to either the .npy image array to read for stimulation, or the
                directory containing the .npy image arrays to read for stimulation.

                
    duration : float (ms)
            Overall stimulus duration

            
    onset_time : float (ms)
            Time point when the stimulation turns on

            
    offset_time : float(ms)
            Time point when the stimulation turns off

    """

    required_parameters = ParameterSet(
        {
            "sheet_list": list,
            "sheet_intensity_scaler": list,
            "sheet_transfection_proportion": list,
            "num_trials": int,
            "stimulator_array_parameters": ParameterSet,
            "images_path": str,
            "intensities": list,
            "duration": int,
            "onset_time": int,
            "offset_time": int,
        }
    )

    def __init__(self, model, parameters):
        CorticalStimulationWithOptogeneticArray.__init__(self, model, parameters)
        sap = self.parameters.stimulator_array_parameters
        sap["stimulating_signal"] = (
            "mozaik.sheets.direct_stimulator.stimulating_pattern_flash"
        )
        sap["stimulating_signal_parameters"] = ParameterSet(
            {
                "shape": "image",
                "intensity": 0,
                "duration": self.parameters.duration,
                "onset_time": self.parameters.onset_time,
                "offset_time": self.parameters.offset_time,
            }
        )

        if os.path.isfile(self.parameters.images_path):
            image_paths = [self.parameters.images_path]
        elif os.path.isdir(self.parameters.images_path):
            image_paths = []
            root, files = [(r, f) for r, d, f in os.walk(self.parameters.images_path)][
                0
            ]
            image_paths = sorted(
                [os.path.join(root, f) for f in files if f[-4:] == ".npy"]
            )
        else:
            raise ValueError(
                "images_path %s is not a file or directory!"
                % self.parameters.images_path
            )
        for intensity in self.parameters.intensities:
            for image_path in image_paths:
                sap.stimulating_signal_parameters.intensity = intensity
                sap.stimulating_signal_parameters.image_path = image_path
                self.append_direct_stim(model, sap)


class OptogeneticArrayStimulusOrientationTuningProtocol(
    CorticalStimulationWithOptogeneticArray
):
    """
    Optogenetic stimulation of cortical sheets with an array of light sources, with a
    pattern based on the cortical orientation map, simulating homogeneously oriented
    visual stimuli.

    At each iteration of the orientation tuning protocol, an orientation
    is selected as the primary orientation to maximally stimulate (with *intensity*
    intensity), and the stimulation intensity for the other orientations in the cortical
    orientation map falls off as a Gaussian with the circular distance from the selected
    orientation:

    Stimulation intensity = intensity * e^(-0.5*d^2/sharpness)
    d = circular_dist(selected_orientation-or_map_orientation)

    Does not show any actual visual stimulus.

    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

          
    Other parameters
    ----------------

    sheet_list : int
               The list of sheets in which to do stimulation.

               
    num_trials : int
                Number of trials each stimulus is shown.

                
    stimulator_array_parameters : ParameterSet
                Parameters for the optical stimulator array:
                    size : float (μm)
                    spacing : float (μm)
                    update_interval : float (ms)
                    depth_sampling_step : float (μm)
                    light_source_light_propagation_data : str
                These parameters are the same as the parameters of
                mozaik.sheets.direct_stimulator.OpticalStimulatorArrayChR class,
                except that it must not contain the parameters
                *stimulating_signal* and *stimulating_signal_parameters* - those
                are set by this experiment.

                
    num_orientations : int
                Number of orientations to present

                
    intensities : list(float)
                Intensities at which to present the stimulation

                
    sharpness : float
            Variance of the Gaussian falloff

            
    duration : float (ms)
            Overall stimulus duration

            
    onset_time : float (ms)
            Time point when the stimulation turns on

            
    offset_time : float(ms)
            Time point when the stimulation turns off

    """

    required_parameters = ParameterSet(
        {
            "sheet_list": list,
            "num_trials": int,
            "stimulator_array_parameters": ParameterSet,
            "num_orientations": int,
            "intensities": list,
            "sharpness": float,
            "duration": int,
            "onset_time": int,
            "offset_time": int,
        }
    )

    def __init__(self, model, parameters):
        CorticalStimulationWithOptogeneticArray.__init__(self, model, parameters)
        sl = self.parameters.sheet_list
        self.parameters["sheet_intensity_scaler"] = [1 for s in sl]
        self.parameters["sheet_transfection_proportion"] = [1 for s in sl]
        sap = self.parameters.stimulator_array_parameters
        sap["stimulating_signal"] = (
            "mozaik.sheets.direct_stimulator.stimulating_pattern_flash"
        )
        sap["stimulating_signal_parameters"] = ParameterSet(
            {
                "shape": "or_map",
                "intensity": 0,
                "orientation": 0,
                "sharpness": self.parameters.sharpness,
                "duration": self.parameters.duration,
                "onset_time": self.parameters.onset_time,
                "offset_time": self.parameters.offset_time,
            }
        )

        orientations = np.linspace(
            0, np.pi, self.parameters.num_orientations, endpoint=False
        )
        for intensity in self.parameters.intensities:
            for orientation in orientations:
                sap.stimulating_signal_parameters.intensity = intensity
                sap.stimulating_signal_parameters.orientation = orientation
                self.append_direct_stim(model, sap)


class OptogeneticArrayStimulusContrastBasedOrientationTuningProtocol(
    CorticalStimulationWithOptogeneticArray
):
    """
    Optogenetic stimulation of cortical sheets with an array of light sources, with a
    pattern based on the cortical orientation map, simulating homogeneously oriented
    visual stimuli of some specific contrast.

    Based on the Antolik et al.:
    Antolik, Jan & Sabatier, Quentin & Galle, Charlie & Frégnac, Yves & Benosman, Ryad. (2019). Cortical visual prosthesis: a detailed large-scale simulation study. 10.1101/610378.

    Both contrast-response curves for visual stimulation and intensity-response curves
    for optogenetic stimulation can be well fitted with Naka-Rushton functions.
    Thus, to simulate the effect of some specific contrast, we need to first retrieve
    the firing rate that would result in visual stimulation with that contrast,
    and then retrieve the intensity of optogenetic stimulation that would result in
    that firing rate.

    intensity = IR^-1(CR(contrast))

    Here, IR^-1 is the inverse optogenetic intensity-response function, and
    CR is the visual contrast-response function. The parameters of these Naka-Rushton
    functions should be fit separately and provided as a parameter to this experiment.

    After the stimulation intensity has been has been calculated for a specific
    contrast, the stimulation protocol proceeds in the same way as in
    OptogeneticArrayStimulusOrientationTuningProtocol:

    At each iteration of the orientation tuning protocol, an orientation
    is selected as the primary orientation to maximally stimulate (with *intensity*
    intensity), and the stimulation intensity for the other orientations in the cortical
    orientation map falls off as a Gaussian with the circular distance from the selected
    orientation:

    Stimulation intensity = intensity * e^(-0.5*d^2/sharpness)
    d = circular_dist(selected_orientation-or_map_orientation)

    For this we use 2 Naka-Rushton functions: one for the forward and one for the inverse transformation

    Parameters
    ----------
    model : Model
          The model on which to execute the experiment.

          
    Other parameters
    ----------------

    sheet_list : int
               The list of sheets in which to do stimulation.

               
    num_trials : int
                Number of trials each stimulus is shown.

                
    stimulator_array_parameters : ParameterSet
                Parameters for the optical stimulator array:
                    size : float (μm)
                    spacing : float (μm)
                    update_interval : float (ms)
                    depth_sampling_step : float (μm)
                    light_source_light_propagation_data : str
                These parameters are the same as the parameters of
                mozaik.sheets.direct_stimulator.OpticalStimulatorArrayChR class,
                except that it must not contain the parameters
                *stimulating_signal* and *stimulating_signal_parameters* - those
                are set by this experiment.

                
    num_orientations : int
                Number of orientations to present

                
    contrasts : list(float)
                List of contrasts, for which we simulate the visual activity by
                optogenetic stimulation.

                
    contrast_response_params: ParameterSet
                Parameters of the Naka-Rushton function of the contrast-response
                curve for visual stimulus: CR(c) = r_max * c^n / (c^n + c_50)

                Fitting these parameters is not part of this experiment.

                
    intensity_response_params: ParameterSet
                Parameters of the Naka-Rushton function of the intensity-response
                curve for optogenetic stimulation: IR(i) = r_max * i^n / (i^n + c_50)

                Fitting these parameters is not part of this experiment.

                
    sharpness : float
            Variance of the Gaussian falloff

            
    duration : float (ms)
            Overall stimulus duration

            
    onset_time : float (ms)
            Time point when the stimulation turns on

            
    offset_time : float(ms)
            Time point when the stimulation turns off
            
    """

    required_parameters = ParameterSet(
        {
            "sheet_list": list,
            "num_trials": int,
            "num_orientations": int,
            "contrasts": list,
            "contrast_response_params": ParameterSet,
            "intensity_response_params": ParameterSet,
            "sharpness": float,
            "duration": int,
            "onset_time": int,
            "offset_time": int,
        }
    )

    def __init__(self, model, parameters):
        CorticalStimulationWithOptogeneticArray.__init__(self, model, parameters)
        sl = self.parameters.sheet_list
        self.parameters["sheet_intensity_scaler"] = [1 for s in sl]
        self.parameters["sheet_transfection_proportion"] = [1 for s in sl]
        sap = self.parameters.stimulator_array_parameters
        sap["stimulating_signal"] = (
            "mozaik.sheets.direct_stimulator.stimulating_pattern_flash"
        )
        sap["stimulating_signal_parameters"] = ParameterSet(
            {
                "shape": "or_map",
                "orientation": 0,
                "sharpness": self.parameters.sharpness,
                "duration": self.parameters.duration,
                "onset_time": self.parameters.onset_time,
                "offset_time": self.parameters.offset_time,
            }
        )
        orientations = np.linspace(
            0, np.pi, self.parameters.num_orientations, endpoint=False
        )
        for contrast in self.parameters.contrasts:
            for orientation in orientations:
                sap.stimulating_signal_parameters.orientation = orientation
                sap.stimulating_signal_parameters.intensity = (
                    self.calculate_optogenetic_stim_scale(
                        contrast,
                        self.parameters.contrast_response_params,
                        self.parameters.intensity_response_params,
                    )
                )
                self.append_direct_stim(model, sap)

    def calculate_optogenetic_stim_scale(self, contrast, cr, ir):
        # Forward transformation - visual stimulus to firing rate
        rate = (
            cr.r_max * np.power(contrast, cr.n) / (np.power(contrast, cr.n) + cr.c_50)
        )
        # Inverse transformation - firing rate to intensity of optogenetic stimulation
        intensity = np.power(rate * ir.c_50 / (ir.r_max - rate), 1 / ir.n)
        return intensity
