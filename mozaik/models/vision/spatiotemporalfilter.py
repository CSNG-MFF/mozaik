# encoding: utf-8
r"""
Retina/LGN model based on that developed by Jens Kremkow (CNRS-INCM/ALUF)
"""

import pylab
import numpy
import os.path
import pickle
import mozaik
from pyNN import space
from mozaik.models.vision import cai97
from mozaik.space import VisualSpace, VisualRegion
from mozaik.core import SensoryInputComponent
from mozaik.sheets.vision import RetinalUniformSheet
from mozaik.sheets.vision import VisualCorticalUniformSheet
from mozaik.tools.mozaik_parametrized import MozaikParametrized
from mozaik.tools.pyNN import *
from parameters import ParameterSet
from builtins import zip
from collections import OrderedDict
from dataclasses import dataclass
import copy

logger = mozaik.getMozaikLogger()


def meshgrid3D(x, y, z):
    r"""A slimmed-down version of http://www.scipy.org/scipy/numpy/attachment/ticket/966/meshgrid.py"""
    x = numpy.asarray(x)
    y = numpy.asarray(y)
    z = numpy.asarray(z)
    mult_fact = numpy.ones((len(x), len(y), len(z)))
    nax = numpy.newaxis
    return (
        x[:, nax, nax] * mult_fact,
        y[nax, :, nax] * mult_fact,
        z[nax, nax, :] * mult_fact,
    )


class SpatioTemporalReceptiveField(object):
    r"""
    Implements spatio-temporal receptive field.

    Parameters
    ----------

    func : function
        should be a function of x, y, t, and a ParameterSet object

    func_params : ParameterSet
        ParameterSet that is passed to `func`.

    width : float (degrees)
        x-dimension size

    height : float (degrees)
        y-dimension size

    duration : float (ms)
        length of the temporal axis of the RF

    Notes
    -----

    Coordinates x = 0 and y = 0 are at the centre of the spatial kernel.


    """

    def __init__(self, func, func_params, width, height, duration):
        self.func = func
        self.func_params = func_params
        self.width = float(width)
        self.height = float(height)
        self.duration = float(duration)
        self.kernel = None
        self.spatial_resolution = numpy.inf
        self.temporal_resolution = numpy.inf

    def quantize(self, dx, dy, dt):
        r"""
        Quantizes the the receptive field.

        Parameters
        ----------

        dx : float
            Difference between pixel positions along the x axis.

        dy : float
            Difference between pixel positions along the y axis.

        dy : float
            Difference between timesteps.

        Notes
        -----

        If `dx` does not
        divide exactly into the width, then the actual width will be slightly
        larger than the nominal width. `dx` and `dy` should be in degrees and `dt` in ms.


        """
        assert dx == dy  # For now, at least
        nx = numpy.ceil(self.width / dx)
        ny = numpy.ceil(self.height / dy)
        nt = numpy.ceil(self.duration / dt)
        width = nx * dx
        height = ny * dy
        duration = nt * dt
        # x and y are the coordinates of the centre of each pixel
        # I use linspace instead of arange as arange sometimes can return inconsistent number of elements
        # (must have something to do with rounding errors)
        # x = numpy.arange(0.0, width, dx)  + dx/2.0 - width/2.0
        # y = numpy.arange(0.0, height, dy) + dy/2.0 - height/2.0

        x = numpy.linspace(0.0, width - dx, int(width / dx)) + dx / 2.0 - width / 2.0
        y = numpy.linspace(0.0, height - dy, int(height / dy)) + dx / 2.0 - height / 2.0

        # t is the time at the beginning of each timestep
        t = numpy.arange(0.0, duration, dt)
        X, Y, T = meshgrid3D(y, x, t)  # x,y are reversed because (x,y) <--> (j,i)
        kernel = self.func(X, Y, T, self.func_params)
        # logger.debug("Created receptive field kernel: width=%gº, height=%gº, duration=%g ms, shape=%s" %
        #                 (width, height, duration, kernel.shape))
        # logger.debug("before normalization: min=%g, max=%g" %
        #                 (kernel.min(), kernel.max()))
        kernel = kernel / (
            nx * ny * nt
        )  # normalize to make the kernel sum quasi-independent of the quantization
        # logger.debug("  after normalization: min=%g, max=%g, sum=%g" %
        #                 (kernel.min(), kernel.max(), kernel.sum()))
        self.kernel = kernel
        self.spatial_resolution = dx
        self.temporal_resolution = dt

    @property
    def kernel_duration(self):
        r"""
        Returns the temporal duration of the quantized kernel.

        Notes
        -----

        This relies on the kernel having been quantized. If not, accessing this will raise an error.
        """
        return self.kernel.shape[2]

    def __str__(self):
        s = "Receptive field: width=%gº, height=%gº, duration=%g ms" % (
            self.width,
            self.height,
            self.duration,
        )
        if self.kernel is not None:
            k = self.kernel
            h, w = k.shape[0:2]
            s += (
                ", quantization=%s, actual width=%gº, actual_height=%gº, min=%g, max=%g."
                % (
                    k.shape,
                    w * self.spatial_resolution,
                    h * self.spatial_resolution,
                    k.min(),
                    k.max(),
                )
            )
        else:
            s += ". Not quantized."
        return s


@dataclass
class KernelResponse:
    r"""
    Unscaled receptive-field convolution output split into two components.

    Parameters
    ----------
    contrast : ndarray
        Response of the zero-mean spatial component of the receptive-field
        kernel. This component is normalized by background luminance when
        frames are sampled.

    luminance : ndarray
        Response of the spatially uniform component of the receptive-field
        kernel. This component tracks absolute image luminance and is used to
        model luminance-dependent gain separately from contrast-dependent gain.
    """

    contrast: numpy.ndarray
    luminance: numpy.ndarray


class CellWithReceptiveField(object):
    r"""
    A model of the input current generated by retinothalamic preprocessing for
    a simulated thalamic relay neuron. It multiplies, in space and time,
    the luminance values impinging on its receptive field by a spatiotemporal
    kernel. Spatial summation over the result of this multiplication at each
    time point gives a current in nA, that may then be injected into a
    simulated thalamic relay neuron.

    initialize() should be called once, before stimulus presentation
    view() should be called in a loop, once for each stimulus frame
    response_current() should be called at the end of stimulus presentation

    Parameters
    ----------

    x , y : float
        x and y coordinates of the center of the RF in visual space.

    receptive_field : SpatioTemporalReceptiveField
        The receptive field object containing the RFs data.

    gain_control : ParameterSet
        The calculated input current values will be multiplied by the gain
        parameter (gain_control.gain), if gain_control.non_linear_gain is None.

        Otherwise, the input current values can be further scaled by a nonlinear
        gain function according to luminance and contrast (parameters set by
        gain_control.non_linear_gain). This nonlinear function mimics the luminance
        and contrast saturation effects of retina interneurons.

    visual_space : VisualSpace
        The object representing the visual space.

    original_2024_lgn_mode : bool
        If True, reproduce the LGN input-current behavior used by the 2024
        LSV1M model version: no filter state is retained between consecutive
        stimuli and the pre-stimulus luminance state is initialized as if the
        filter had previously seen zero luminance. If False, the cell keeps the
        trailing kernel state between presentations and initializes from the
        visual-space background luminance.

    """

    def __init__(
        self,
        x,
        y,
        receptive_field,
        gain_control,
        visual_space,
        original_2024_lgn_mode=False,
    ):
        self.x = x  # position in space
        self.y = y  #
        self.visual_space = visual_space
        assert isinstance(receptive_field, SpatioTemporalReceptiveField)
        self.receptive_field = receptive_field
        self.gain_control = (
            gain_control  # (nA.m²/cd) could imagine making this a function of
        )
        # the background luminance
        self.i = 0
        self.visual_region = VisualRegion(
            location_x=self.x,
            location_y=self.y,
            size_x=self.receptive_field.width,
            size_y=self.receptive_field.height,
        )
        # logger.debug("view_array.shape = %s" % str(view_array.shape))
        # logger.debug("receptive_field.kernel.shape = %s" % str(self.receptive_field.kernel.shape))
        # logger.debug("response.shape = %s" % str(self.response.shape))
        if visual_space.update_interval % self.receptive_field.temporal_resolution != 0:
            errmsg = (
                "The receptive field temporal resolution (%g ms) must be an integer multiple of the visual space update interval (%g ms)"
                % (
                    self.receptive_field.temporal_resolution,
                    visual_space.update_interval,
                )
            )
            raise Exception(errmsg)
        self.update_factor = int(
            visual_space.update_interval / self.receptive_field.temporal_resolution
        )
        self.original_2024_lgn_mode = original_2024_lgn_mode
        self.filter_state = None

        self.background_luminance = visual_space.background_luminance
        # The kernel is separated into a luminance and contrast component,
        # which are then scaled separately by the non-linear gain.
        # This separability is based on doi:10.1038/nn1556, although they
        # do the separation there in a different way - multiplicatively, where
        # the partial kernels themselves change with luminance and contrast
        rf = self.receptive_field
        assert (
            rf.kernel.shape[0] == rf.kernel.shape[1]
        ), "With the current implementation, receptive fields must be symmetric!"
        # Luminance component is the spatial mean of the kernel
        rf.kernel_luminance_component = rf.kernel.mean(axis=(0, 1))
        # Contrast component is the remaining part of the kernel
        rf.kernel_contrast_component = rf.kernel - rf.kernel_luminance_component
        # Reshape from space x space x time to space x time
        rf.kernel_contrast_component = rf.kernel_contrast_component.reshape(
            -1, numpy.shape(rf.kernel_contrast_component)[2]
        ).T
        self.null_response = KernelResponse(
            contrast=0,
            luminance=self.receptive_field.kernel_luminance_component.sum()
            * self.background_luminance,
        )

        # Unscaled cumulative sum of the luminance kernel (length = kdur)
        kernel_cumsum = numpy.cumsum(rf.kernel_luminance_component)
        # Padded, scaled version for background-luminance step response
        self.kernel_luminance_cumsum_padded = numpy.concatenate(
            [[0], self.background_luminance * kernel_cumsum]
        )
        self.luminance_step_response = self.kernel_luminance_cumsum_padded
        self.luminance_steady_state = self.luminance_step_response[-1]

        # Set starting luminance (default = background luminance)
        self.starting_luminance = (
            0 if self.original_2024_lgn_mode else self.background_luminance
        )

        # Tail from infinite past of constant starting_luminance
        self.starting_luminance_kernel_state = self.starting_luminance * (
            kernel_cumsum[-1] - kernel_cumsum[:-1]
        )

    def initialize(self, stimulus_duration):
        r"""
        Create the array that will contain the current response, and set the
        initial values on the assumption that the system was looking at a blank
        screen of constant luminance prior to stimulus onset.

        Parameters
        ----------

        stimulus_duration : float (ms)
            The duration  of the visual stimulus.

        Notes
        -----

        The response buffer includes extra samples equal to the temporal kernel
        duration. These samples hold the filter tail and are removed when
        converting the kernel response into an injected current.
        """
        rf = self.receptive_field
        self.response_length = int(
            numpy.ceil(stimulus_duration / rf.temporal_resolution) + rf.kernel_duration
        )

        # We assume the model has been looking at starting luminance prior to
        # stimulus onset; thus the initial response is the accumulation of luminance
        # kernel convolution at starting luminance until the present time point
        if self.filter_state is None:
            self.kernel_response = KernelResponse(
                contrast=numpy.zeros(self.response_length),
                luminance=numpy.pad(
                    self.starting_luminance_kernel_state,
                    (
                        0,
                        self.response_length
                        - len(self.starting_luminance_kernel_state),
                    ),
                ),
            )
        else:
            self.kernel_response = KernelResponse(
                contrast=numpy.pad(
                    self.filter_state.contrast,
                    (0, self.response_length - len(self.filter_state.contrast)),
                ),
                luminance=numpy.pad(
                    self.filter_state.luminance,
                    (0, self.response_length - len(self.filter_state.luminance)),
                ),
            )
        assert rf.kernel_duration <= self.response_length
        self.i = 0

    def view(self):
        r"""
        Look at the visual space and update t
        Where the kernel temporal resolution is the same as the frame duration
        (visual space update interval):
        R_i = Sum[j=0,L-1] K_j.I_i-j
        where L is the kernel length/duration
        Where the kernel temporal resolution = (frame duration)/α (α an integer)
        R_k = Sum[j=0,L-1] K_j.I_i'
        where i' = (k-j)//α  (// indicates integer division, discarding the
        remainder)
        To avoid loading the entire image sequence into memory, we build up the response array one frame at a time.
        """
        view_array = self.visual_space.view(
            self.visual_region, pixel_size=self.receptive_field.spatial_resolution
        )
        # We divide the input by background luminance, so that the kernel contrast
        # response is agnostic to the overall luminance level
        contrast_time_course = numpy.dot(
            self.receptive_field.kernel_contrast_component,
            view_array.reshape(-1)[: numpy.newaxis] / self.background_luminance,
        )
        # The luminance response kernel is equal at all spatial positions, so
        # we don't calculate it for each position, rather multiply the 1D version
        # of it by the mean image luminance at each time point.
        # That is equivalent to a 3D luminance kernel which is convolved and with the
        # image and then summed
        luminance_time_course = (
            self.receptive_field.kernel_luminance_component * numpy.mean(view_array)
        )
        self.va = view_array

        d = self.receptive_field.kernel_duration
        if self.update_factor != 1.0:
            for j in range(self.i, self.i + self.update_factor):
                self.kernel_response.contrast[j : j + d] += contrast_time_course[:d]
                self.kernel_response.luminance[j : j + d] += luminance_time_course[:d]
        else:
            self.kernel_response.contrast[self.i : self.i + d] += contrast_time_course[
                :d
            ]
            self.kernel_response.luminance[
                self.i : self.i + d
            ] += luminance_time_course[:d]

        self.i += (
            self.update_factor
        )  # we assume there is only ever 1 visual space used between initializations

    def gain_function(self, response, gain, scaler):
        r"""
        Scale the response by a symmetric Naka-Rushton function to
        achieve the variable luminance/contrast gain observed in the retina.
        """
        return gain * response / (numpy.abs(response) + scaler)

    def response_current(self, kernel_response):
        r"""
        Multiply the response (units of luminance (cd/m²) if we assume the
        kernel values are dimensionless) by the 'gain', to produce a current in
        nA. Returns a dictionary containing 'times' and 'amplitudes'.

        Parameters
        ----------
        kernel_response : KernelResponse
            Contrast and luminance components of the convolution response for
            one stimulus presentation.

        Returns
        -------
        dict
            Dictionary with ``times`` in ms and ``amplitudes`` in nA. The final
            kernel-duration tail is omitted from the returned arrays. In normal
            mode, that omitted tail is retained internally and used as the
            initial filter state for the next presentation.
        """
        kdur = self.receptive_field.kernel_duration
        # We scale the luminance and contrast components separately,
        # and then add them
        nlg = self.gain_control.non_linear_gain
        contrast_response = self.gain_function(
            kernel_response.contrast, nlg.contrast_gain, nlg.contrast_scaler
        )
        luminance_response = self.gain_function(
            kernel_response.luminance, nlg.luminance_gain, nlg.luminance_scaler
        )
        response = contrast_response + luminance_response

        # Remove extra padding at the end
        response = response[:-kdur]
        time_points = self.receptive_field.temporal_resolution * numpy.arange(
            0, len(response)
        )

        if not self.original_2024_lgn_mode:
            self.filter_state = KernelResponse(
                contrast=kernel_response.contrast[-kdur:].copy(),
                luminance=kernel_response.luminance[-kdur:].copy(),
            )
        return {"times": time_points, "amplitudes": response}


class SpatioTemporalFilterRetinaLGN(SensoryInputComponent):
    r"""
    Retinothalamic preprocessing with spatiotemporal receptive fields feeding
    simulated thalamic relay neurons.

    Parameters
    ----------

    density : int (1/degree^2)
        Number of neurons to simulate per square degree of visual space.

    size : tuple (degree,degree)
        The x and y size of the visual field.

    linear_scaler : float
        The linear scaler that the RF output is multiplied with.

    mpi_reproducible_noise : bool
        If true the background noise is generated in such a way that is
        reproducible across runs using different numbers of MPI processes.
        Significant slowdown if True.

    recorders : ParameterSet
        Recorder configuration passed to the simulated thalamic relay sheets.

    recording_interval : float
        Sampling interval for recorded variables on the simulated thalamic
        relay sheets.

    receptive_field : ParameterSet
        Parameters describing the receptive-field function, its spatial and
        temporal extent, and the quantization resolution.

    original_2024_lgn_mode : bool
        If True, use the legacy LGN behavior needed to reproduce the 2024 LSV1M
        reference data. This disables filter-state carry-over, starts filters
        from zero luminance, and uses the historical optimized null-stimulus
        current injection path. If False, filter state is retained across
        consecutive presentations and null input is computed through the same
        kernel-response machinery as explicit stimuli.

    cell : ParameterSet
        PyNN cell configuration for the simulated thalamic relay cell
        populations.

    gain_control : ParameterSet
        Gain and nonlinear gain parameters used to transform receptive-field
        responses into injected currents.

    noise : ParameterSet
        Mean and standard deviation of the additive current noise.

    Notes
    -----

    Stimulus responses are cached only in memory for the lifetime of the
    component. The cache is keyed by the stimulus identity with ``trial`` ignored,
    so repeated trials can reuse the same receptive-field convolution while
    still receiving independently generated noise.

    """

    required_parameters = ParameterSet(
        {
            "density": int,  # neurons per degree squared
            "size": tuple,  # degrees of visual field
            "linear_scaler": float,  # linear scaler that the RF output is multiplied with
            "mpi_reproducible_noise": bool,  # if True, noise is precomputed and StepCurrentSource is used which makes it slower
            "recorders": ParameterSet,
            "recording_interval": float,
            "receptive_field": ParameterSet(
                {
                    "func": str,
                    "func_params": ParameterSet,
                    "width": float,
                    "height": float,
                    "spatial_resolution": float,
                    "temporal_resolution": float,
                    "duration": float,
                }
            ),
            "original_2024_lgn_mode": bool,
            "cell": ParameterSet(
                {
                    "model": str,
                    "params": ParameterSet,
                    "receptors": ParameterSet,
                    "native_nest": bool,
                    "initial_values": ParameterSet,
                }
            ),
            "gain_control": {
                "gain": float,
                "non_linear_gain": ParameterSet(
                    {
                        "luminance_gain": float,
                        "luminance_scaler": float,
                        "contrast_gain": float,
                        "contrast_scaler": float,
                    }
                ),
            },
            "noise": ParameterSet(
                {
                    "mean": float,
                    "stdev": float,  # nA
                }
            ),
        }
    )

    def __init__(self, model, parameters):
        SensoryInputComponent.__init__(self, model, parameters)
        self.shape = (self.parameters.density, self.parameters.density)
        self.sheets = OrderedDict()
        self.rf_types = ("X_ON", "X_OFF")
        sim = self.model.sim
        self.pops = OrderedDict()

        if self.parameters.cell.model[-6:] == "_sc_nc":
            self.integrated_cs = True
            import copy

            cell = copy.deepcopy(self.parameters.cell)
            cell.params.update(
                [
                    ("mean", self.parameters.noise.mean * 1000),
                    ("std", self.parameters.noise.stdev * 1000),
                    ("dt", self.model.sim.get_time_step()),
                ]
            )
        else:
            self.integrated_cs = False
            self.scs = OrderedDict()
            self.ncs = OrderedDict()
            cell = self.parameters.cell

        self.ncs_rng = OrderedDict()
        self.internal_stimulus_cache = OrderedDict()
        for rf_type in self.rf_types:

            p = RetinalUniformSheet(
                model,
                ParameterSet(
                    {
                        "sx": self.parameters.size[0],
                        "sy": self.parameters.size[1],
                        "density": self.parameters.density,
                        "cell": cell,
                        "name": rf_type,
                        "artificial_stimulators": OrderedDict(),
                        "recorders": self.parameters.recorders,
                        "recording_interval": self.parameters.recording_interval,
                        "mpi_safe": False,
                    }
                ),
            )
            self.sheets[rf_type] = p

        for rf_type in self.rf_types:
            self.ncs_rng[rf_type] = []
            seeds = mozaik.get_simulation_seeds((self.sheets[rf_type].pop.size,))

            if self.integrated_cs:
                for i, lgn_cell in enumerate(self.sheets[rf_type].pop.all_cells):
                    if self.sheets[rf_type].pop._mask_local[i]:
                        self.ncs_rng[rf_type].append(
                            numpy.random.RandomState(seed=seeds[i])
                        )
            else:
                self.scs[rf_type] = []
                self.ncs[rf_type] = []
                for i, lgn_cell in enumerate(self.sheets[rf_type].pop.all_cells):
                    scs = sim.StepCurrentSource(times=[0.0], amplitudes=[0.0])

                    if not self.parameters.mpi_reproducible_noise:
                        ncs = sim.NoisyCurrentSource(**self.parameters.noise)
                    else:
                        ncs = sim.StepCurrentSource(times=[0.0], amplitudes=[0.0])

                    if self.sheets[rf_type].pop._mask_local[i]:
                        self.ncs_rng[rf_type].append(
                            numpy.random.RandomState(seed=seeds[i])
                        )
                        self.scs[rf_type].append(scs)
                        self.ncs[rf_type].append(ncs)
                    lgn_cell.inject(scs)
                    lgn_cell.inject(ncs)

        P_rf = self.parameters.receptive_field
        rf_function = eval(P_rf.func)

        rf_ON = SpatioTemporalReceptiveField(
            rf_function, P_rf.func_params, P_rf.width, P_rf.height, P_rf.duration
        )
        rf_OFF = SpatioTemporalReceptiveField(
            lambda x, y, t, p: -1.0 * rf_function(x, y, t, p),
            P_rf.func_params,
            P_rf.width,
            P_rf.height,
            P_rf.duration,
        )

        dx = dy = P_rf.spatial_resolution
        dt = P_rf.temporal_resolution
        for rf in rf_ON, rf_OFF:
            rf.quantize(dx, dy, dt)

        self.rf = {"X_ON": rf_ON, "X_OFF": rf_OFF}

        # create population of CellWithReceptiveFields, setting the receptive
        # field centres based on the size/location of self
        logger.debug("Creating population of `CellWithReceptiveField`s")
        self.input_cells = OrderedDict()
        for rf_type in self.rf_types:
            self.input_cells[rf_type] = []
            for i in numpy.nonzero(self.sheets[rf_type].pop._mask_local)[0]:
                cell = CellWithReceptiveField(
                    self.sheets[rf_type].pop.positions[0][i],
                    self.sheets[rf_type].pop.positions[1][i],
                    self.rf[rf_type],
                    self.parameters.gain_control,
                    model.input_space,
                    self.parameters.original_2024_lgn_mode,
                )
                self.input_cells[rf_type].append(cell)

    def process_input(self, visual_space, stimulus, duration=None, offset=0):
        r"""
        Present a visual stimulus to the model and create currents for the
        simulated thalamic relay neurons.

        Parameters
        ----------

        visual_space : VisualSpace
            The visual space to which the stimuli are presented.

        stimulus : VisualStimulus
            The visual stimulus to be shown.

        duration : int (ms)
            The time for which we will simulate the stimulus

        offset : int(ms)
            The time (in absolute time of the whole simulation) at which the stimulus starts.


        Returns
        -------

        retinal_input : list(ndarray)
            List of 2D arrays containing the frames of luminances that were presented to the retina.


        """
        logger.debug("Presenting visual stimulus from visual space %s" % visual_space)
        visual_space.set_duration(duration)
        self.input = visual_space
        st = MozaikParametrized.idd(stimulus)
        st.trial = None  # to avoid recalculating RFs response to multiple trials of the same stimulus

        # We check if we haven't already presented it during this simulation run.
        # This is mainly to avoid regenerating stimuli for multiple trials.
        if str(st) in self.internal_stimulus_cache:
            kernel_responses, retinal_input = self.internal_stimulus_cache[str(st)]
        else:
            kernel_responses, retinal_input = self.calculate_kernel_responses(
                visual_space, duration
            )
        input_currents = self._calculate_input_currents(kernel_responses)
        self.inject_currents(input_currents, duration, offset)

        # also save into internal cache
        self.internal_stimulus_cache[str(st)] = (
            copy.deepcopy(kernel_responses),
            retinal_input,
        )

        return retinal_input

    def inject_currents(self, input_currents, duration=None, offset=0):
        r"""
        Inject precomputed current traces into the simulated thalamic relay
        neurons.

        Parameters
        ----------
        input_currents : OrderedDict
            Mapping from RF type (``X_ON`` or ``X_OFF``) to a list of current
            dictionaries, one per local simulated thalamic relay neuron. Each
            dictionary contains ``times`` in ms and ``amplitudes`` in nA.

        duration : float (ms)
            Duration of the current injection. Used to generate reproducible
            noise traces when ``mpi_reproducible_noise`` is enabled.

        offset : float (ms)
            Absolute simulator time at which the stimulus starts.

        Notes
        -----
        The integrated current-source cell model expects amplitudes in pA, so
        amplitudes are scaled by 1000 on that path. In legacy 2024 mode, the
        historical PyNN/NEST offset correction is preserved only for the
        integrated current-source path; ordinary StepCurrentSource injection
        keeps the old uncorrected timing.
        """
        ts = self.model.sim.get_time_step()
        if self.parameters.original_2024_lgn_mode and self.integrated_cs:
            offset = convert_time_pyNN_to_nest(self.model.sim, offset) + ts

        for rf_type in self.rf_types:
            assert isinstance(input_currents[rf_type], list)
            if self.integrated_cs:
                for i, (lgn_cell, input_current) in enumerate(
                    zip(self.sheets[rf_type].pop, input_currents[rf_type])
                ):
                    assert isinstance(input_current, dict)
                    t = input_current["times"] + offset
                    a = self.parameters.linear_scaler * input_current["amplitudes"]
                    lgn_cell.set_parameters(
                        amplitude_times=t[1:], amplitude_values=a[1:] * 1000
                    )

            else:
                for i, (lgn_cell, input_current, scs, ncs) in enumerate(
                    zip(
                        self.sheets[rf_type].pop,
                        input_currents[rf_type],
                        self.scs[rf_type],
                        self.ncs[rf_type],
                    )
                ):
                    assert isinstance(input_current, dict)
                    t = input_current["times"] + offset
                    a = self.parameters.linear_scaler * input_current["amplitudes"]
                    scs.set_parameters(times=t, amplitudes=a, copy=False)
                    if self.parameters.mpi_reproducible_noise:
                        t = numpy.arange(0, duration, ts) + offset
                        amplitudes = (
                            self.parameters.noise.mean
                            + self.parameters.noise.stdev
                            * self.ncs_rng[rf_type][i].randn(len(t))
                        )
                        ncs.set_parameters(times=t, amplitudes=amplitudes, copy=False)

    def _provide_legacy_null_input(self, visual_space, duration=None, offset=0):
        r"""
        Inject the optimized null-stimulus current used by legacy 2024 runs.

        This is intentionally separate from ``calculate_null_input`` and
        ``inject_currents`` because the old implementation did not construct a
        full time series. It injected one constant two-point current step per RF
        type and used a historical StepCurrentSource timing workaround
        (``offset + 3 * timestep``) on the non-integrated current-source path.
        """
        ts = self.model.sim.get_time_step()
        if self.integrated_cs:
            new_offset = convert_time_pyNN_to_nest(self.model.sim, offset) + ts
            times = numpy.array(
                [new_offset, duration - visual_space.update_interval + new_offset]
            )
        else:
            times = numpy.array(
                [offset + 3 * ts, duration - visual_space.update_interval + offset]
            )

        for rf_type in self.rf_types:
            cell = self.input_cells[rf_type][0]
            nlg = cell.gain_control.non_linear_gain
            amplitude = cell.null_response.luminance
            amplitude = self.parameters.linear_scaler * cell.gain_function(
                amplitude, nlg.luminance_gain, nlg.luminance_scaler
            )

            if self.integrated_cs:
                for lgn_cell in self.sheets[rf_type].pop:
                    lgn_cell.set_parameters(
                        amplitude_times=times,
                        amplitude_values=numpy.zeros_like(times) + amplitude * 1000,
                    )
            else:
                for i, (scs, ncs) in enumerate(
                    zip(self.scs[rf_type], self.ncs[rf_type])
                ):
                    scs.set_parameters(
                        times=times,
                        amplitudes=numpy.zeros_like(times) + amplitude,
                        copy=False,
                    )
                    if self.parameters.mpi_reproducible_noise:
                        t = numpy.arange(0, duration, ts) + offset
                        amplitudes = (
                            self.parameters.noise.mean
                            + self.parameters.noise.stdev
                            * self.ncs_rng[rf_type][i].randn(len(t))
                        )
                        ncs.set_parameters(times=t, amplitudes=amplitudes, copy=False)

    def calculate_null_input(self, duration=None):
        r"""
        Calculate current traces for a blank screen, retaining the convolution
        tail.

        Parameters
        ----------
        duration : float (ms)
            Duration of the blank presentation.

        Returns
        -------
        OrderedDict
            Mapping from RF type to a list of current dictionaries, one per
            local simulated thalamic relay neuron. Each dictionary contains
            ``times`` in ms and ``amplitudes`` in nA.

        Notes
        -----
        This method is used only when ``original_2024_lgn_mode`` is False.
        Legacy-mode null input is handled by ``_provide_legacy_null_input`` so
        that the old shortcut remains isolated and easy to remove.
        """
        input_currents = OrderedDict()
        for rf_type in self.rf_types:
            input_currents[rf_type] = []
            for cell in self.input_cells[rf_type]:
                num_frames = (
                    int(numpy.ceil(duration / cell.visual_space.update_interval))
                    * cell.update_factor
                )
                cell.initialize(duration)

                kdur = len(cell.receptive_field.kernel_luminance_component)
                # Build background luminance step response extended to the
                # needed length (num_frames + kdur)
                step_on = numpy.concatenate(
                    [
                        cell.luminance_step_response[1:],  # t = 1..kdur
                        numpy.full(
                            num_frames, cell.luminance_steady_state
                        ),  # t = kdur+1 .. kdur+num_frames
                    ]
                )
                # Delayed step (subtract after num_frames)
                step_off = numpy.concatenate(
                    [
                        numpy.zeros(num_frames),  # delay
                        cell.luminance_step_response[
                            1:kdur
                        ],  # same shape as step_on's tail
                    ]
                )
                added_luminance = (
                    step_on[: num_frames + kdur - 1] - step_off[: num_frames + kdur - 1]
                )

                # Pad to full kernel_response length (last sample stays zero)
                if len(added_luminance) < len(cell.kernel_response.luminance):
                    added_luminance = numpy.pad(
                        added_luminance,
                        (0, len(cell.kernel_response.luminance) - len(added_luminance)),
                    )
                cell.kernel_response.luminance += added_luminance
                input_currents[rf_type].append(
                    cell.response_current(cell.kernel_response)
                )

        return input_currents

    def provide_null_input(self, visual_space, duration=None, offset=0):
        r"""
        This function exists for optimization purposes. It is the analog to
        :func:`process_input` for the special case when a blank stimulus is
        shown.

        Parameters
        ----------

        visual_space : VisualSpace
            The visual space to which the blank stimulus are presented.

        duration : int (ms)
            The time for which we will simulate the blank stimulus

        offset : int(ms)
            The time (in absolute time of the whole simulation) at which the stimulus starts.

        Returns
        -------

        None
            The generated currents are injected into the simulated thalamic
            relay neurons directly.

        Notes
        -----
        In normal mode, blank input is represented as a finite background
        luminance step and goes through the same current-injection machinery as
        explicit stimuli. In ``original_2024_lgn_mode``, this dispatches to the
        historical optimized null path to reproduce the 2024 reference data.
        Once ``original_2024_lgn_mode`` is no longer needed, it can be merged
        with ``calculate_null_input``.
        """
        if self.parameters.original_2024_lgn_mode:
            self._provide_legacy_null_input(visual_space, duration, offset)
        else:
            self.inject_currents(self.calculate_null_input(duration), duration, offset)

    def calculate_kernel_responses(self, visual_space, duration):
        r"""
        Convolve all local receptive-field cells with the current visual input.

        Parameters
        ----------
        visual_space : VisualSpace
            Visual space containing the stimulus objects to sample.

        duration : float (ms)
            Duration over which frames should be sampled. If None, the maximum
            duration of objects in ``visual_space`` is used.

        Returns
        -------
        tuple
            ``(kernel_responses, retinal_input)`` where ``kernel_responses`` is
            an ``OrderedDict`` mapping RF type to a list of ``KernelResponse``
            objects, one per local simulated thalamic relay neuron.
            ``retinal_input`` contains stored stimulus frames when
            ``store_stimuli`` is enabled; otherwise it contains ``None``
            entries matching the frame updates.
        """
        assert isinstance(visual_space, VisualSpace)
        if duration is None:
            duration = visual_space.get_maximum_duration()

        logger.debug("Processing frames")

        t = 0
        retinal_input = []

        for rf_type in self.rf_types:
            for cell in self.input_cells[rf_type]:
                cell.initialize(duration)

        while t < duration:
            t = visual_space.update()
            for rf_type in self.rf_types:
                for cell in self.input_cells[rf_type]:
                    cell.view()

            if self.model.parameters.store_stimuli == True:
                visual_region = VisualRegion(
                    location_x=0,
                    location_y=0,
                    size_x=self.model.visual_field.size_x,
                    size_y=self.model.visual_field.size_y,
                )
                im = visual_space.view(
                    visual_region, pixel_size=self.rf["X_ON"].spatial_resolution
                )
            else:
                im = None
            retinal_input.append(im)

        kernel_responses = OrderedDict()
        for rf_type in self.rf_types:
            kernel_responses[rf_type] = [
                cell.kernel_response for cell in self.input_cells[rf_type]
            ]
        return (kernel_responses, retinal_input)

    def _calculate_input_currents(self, kernel_responses=None):
        r"""
        Calculate the input currents for all cells.

        Parameters
        ----------
        kernel_responses : OrderedDict
            Mapping from RF type to the ``KernelResponse`` objects produced by
            ``calculate_kernel_responses``.

        Returns
        -------
        OrderedDict
            Mapping from RF type to current dictionaries suitable for
            ``inject_currents``.
        """
        input_currents = OrderedDict()
        for rf_type in kernel_responses.keys():
            input_currents[rf_type] = [
                cell.response_current(kernel_response)
                for cell, kernel_response in zip(
                    self.input_cells[rf_type], kernel_responses[rf_type]
                )
            ]
        return input_currents
