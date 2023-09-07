import sys
import os
import imagen
import pytest
import pylab
import numpy as np
from pyNN import nest

import mozaik
from mozaik.models import Model
from mozaik.space import VisualRegion
import mozaik.stimuli.vision.topographica_based as topo
from mozaik.tools.distribution_parametrization import load_parameters
from mozaik.space import VisualSpace
from mozaik.models.vision.spatiotemporalfilter import (
    CellWithReceptiveField,
    SpatioTemporalReceptiveField,
    SpatioTemporalFilterRetinaLGN,
)
from mozaik.models.vision import cai97
from mozaik.stimuli.vision.topographica_based import PixelImpulse
from mozaik.tools.mozaik_parametrized import SNumber
from quantities import dimensionless
from parameters import ParameterSet


params = {
    "input_space_type": "mozaik.space.VisualSpace",
    "input_space": {"update_interval": 7.0, "background_luminance": 45.0},
    "visual_field": {
        "centre": (0.0, 0.0),
        "size": (7.0, 7.0),
    },
    "sheets": {
        "retina_lgn": {
            "component": "mozaik.models.vision.spatiotemporalfilter.SpatioTemporalFilterRetinaLGN",
            "params": {
                "density": 10,
                "size": (0.5, 0.5),
                "linear_scaler": 6.0,
                "mpi_reproducible_noise": False,
                "cached": False,
                "cache_path": "",
                "recorders": {},
                "recording_interval": 1.0,
                "receptive_field": {
                    "func": "cai97.stRF_2d",
                    "func_params": {
                        "Ac": 1.0,
                        "As": 0.3,
                        "K1": 1.05,
                        "K2": 0.7,
                        "c1": 0.14,
                        "c2": 0.12,
                        "n1": 7.0,
                        "n2": 8.0,
                        "t1": -6.0,
                        "t2": -6.0,
                        "td": 6.0,
                        "sigma_c": 0.4,
                        "sigma_s": 1.0,
                        "subtract_mean": False,
                    },
                    "width": 6.0,
                    "height": 6.0,
                    "spatial_resolution": 0.1,
                    "temporal_resolution": 7.0,
                    "duration": 200.0,
                },
                "gain_control": {
                    "gain": 1,
                    "non_linear_gain": {
                        "contrast_gain": 0.11,
                        "contrast_scaler": 0.00013,
                        "luminance_gain": 0.009,
                        "luminance_scaler": 0.4,
                    },
                },
                "cell": {
                    "model": "IF_cond_exp",
                    "native_nest": False,
                    "params": {
                        "v_thresh": -57.0,
                        "v_rest": -70.0,
                        "v_reset": -70.0,
                        "tau_refrac": 2.0,
                        "tau_m": 10.0,
                        "cm": 0.29,
                        "e_rev_E": 0.0,
                        "e_rev_I": -75.0,
                        "tau_syn_E": 1.5,
                        "tau_syn_I": 10.0,
                    },
                    "initial_values": {"v": -70.0},
                },
                "noise": {"mean": 0.0, "stdev": 0.0},
            },
        }
    },
    "results_dir": "",
    "name": "SelfSustainedPushPullV1",
    "reset": False,
    "null_stimulus_period": 150.0,
    "store_stimuli": False,
    "min_delay": 0.1,
    "max_delay": 100,
    "time_step": 0.1,
    "pynn_seed": 936395,
    "mpi_seed": 1023,
    "explosion_monitoring": None,
}

base_stim_params = {
    "frame_duration": params["input_space"]["update_interval"],
    "duration": 1,
    "trial": 1,
    "background_luminance": params["input_space"]["background_luminance"],
    "density": 1
    / params["sheets"]["retina_lgn"]["params"]["receptive_field"]["spatial_resolution"],
    "location_x": 0.0,
    "location_y": 0.0,
    "size_x": params["visual_field"]["size"][0],
    "size_y": params["visual_field"]["size"][1],
}


class TestCellWithReceptiveField:
    receptive_field_on = None
    receptive_field_off = None
    cell_on = None
    cell_off = None
    visual_space = None

    # Visual stimulus parameters
    vs_params = None

    # Receptive field parameters
    rf_params = params["sheets"]["retina_lgn"]["params"]["receptive_field"][
        "func_params"
    ]

    @classmethod
    def setup_class(cls):
        size = 3.0
        cls.vs_params = base_stim_params.copy()
        cls.vs_params.update({"size_x": size, "size_y": size})
        cls.visual_space = VisualSpace(ParameterSet(params["input_space"]))
        cls.receptive_field_on = SpatioTemporalReceptiveField(
            cai97.stRF_2d, ParameterSet(cls.rf_params), size, size, 200
        )
        cls.receptive_field_on.quantize(0.1, 0.1, cls.visual_space.update_interval)
        cls.receptive_field_off = SpatioTemporalReceptiveField(
            lambda x, y, t, p: -1.0 * cai97.stRF_2d(x, y, t, p),
            ParameterSet(cls.rf_params),
            size,
            size,
            200,
        )
        cls.receptive_field_off.quantize(0.1, 0.1, cls.visual_space.update_interval)
        gain_params = ParameterSet({"gain": 1.0, "non_linear_gain": None})
        cls.cell_on = CellWithReceptiveField(
            0, 0, cls.receptive_field_on, gain_params, cls.visual_space
        )
        cls.cell_off = CellWithReceptiveField(
            0, 0, cls.receptive_field_off, gain_params, cls.visual_space
        )

    @pytest.mark.parametrize("x", np.random.randint(0, 30, size=5))
    @pytest.mark.parametrize("y", np.random.randint(0, 30, size=5))
    @pytest.mark.parametrize("on", [True, False])
    def test_impulse_response(self, x, y, on):
        """
        Check that the impulse response of the receptive field is equal to the receptive
        field kernel at the impulse position. This is only the case when no non-linear
        gain is applied to the response.
        """
        stimulus = PixelImpulse(relative_luminance=2.0, x=x, y=y, **self.vs_params)
        self.visual_space.clear()
        self.visual_space.add_object(str(stimulus), stimulus)
        self.visual_space.update()
        # Impulse is 1st frame, 4 frames null stimulus as sanity check
        if on:
            cell = self.cell_on
            rf = self.receptive_field_on
        else:
            cell = self.cell_off
            rf = self.receptive_field_off
        cell.initialize(self.vs_params["background_luminance"], 5)
        cell.view()

        # Separately test contrast and luminance response
        # Before applying nonlinear gain, they act as linear filters
        r = cell.contrast_response[: rf.kernel.shape[2]]
        pos = y + x * rf.kernel.shape[0]
        np.testing.assert_allclose(r, rf.kernel_contrast_component[:, pos])

        # The luminance response kernel is equal at all spatial positions, so
        # we don't calculate it for each position, rather multiply the 1D version
        # of it by the mean image luminance.
        # That is equivalent to a 3D luminance kernel which is convolved and with the
        # image and then summed at each time point.
        r = cell.luminance_response[: rf.kernel.shape[2]]
        np.testing.assert_allclose(r, rf.kernel_luminance_component * cell.va.mean())


class TestSpatioTemporalFilterRetinaLGN:
    @pytest.mark.parametrize("background_luminance", [10, 20, 40, 80])
    @pytest.mark.parametrize("rf_duration", [50, 100, 200])
    def test_blank_stimulus(self, background_luminance, rf_duration):
        """
        Test that the response of the neuron to a Null stimulus
        is equal to the response of presenting a homogeneous
        stimulus of background luminance.

        This is to test that the outputs of functions provide_null_input
        and process_input are equal in this scenario.
        """
        parameters = load_parameters(params, ParameterSet({}))
        mozaik.setup_mpi(parameters["mpi_seed"], parameters["pynn_seed"])
        parameters["input_space"]["background_luminance"] = background_luminance
        parameters["sheets"]["retina_lgn"]["params"]["receptive_field"][
            "duration"
        ] = rf_duration

        vf = VisualRegion(
            location_x=parameters["visual_field"]["centre"][0],
            location_y=parameters["visual_field"]["centre"][1],
            size_x=parameters["visual_field"]["size"][0],
            size_y=parameters["visual_field"]["size"][1],
        )
        del parameters["visual_field"]
        m = Model(nest, 2, parameters)

        m.visual_field = vf
        sh = SpatioTemporalFilterRetinaLGN(
            m, parameters["sheets"]["retina_lgn"]["params"]
        )
        dur = sh.rf["X_ON"].duration + parameters["input_space"]["update_interval"]

        stim_params = base_stim_params.copy()
        stim_params["background_luminance"] = background_luminance
        stim_params["duration"] = dur
        stim = topo.Null(**stim_params)
        m.input_space.add_object("blank_stimulus", stim)

        sh.process_input(m.input_space, stim, duration=dur)
        pi_ampl = [n.amplitudes[-1] for sheet in sh.scs for n in sh.scs[sheet]]

        sh.provide_null_input(m.input_space, duration=dur)
        ni_ampl = [n.amplitudes[-1] for sheet in sh.scs for n in sh.scs[sheet]]
        np.testing.assert_allclose(pi_ampl, ni_ampl)
