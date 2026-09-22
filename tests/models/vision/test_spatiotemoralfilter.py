import pytest
import numpy as np

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
from parameters import ParameterSet
import copy
from mozaik.models.vision.spatiotemporalfilter import KernelResponse
import itertools
from tests.models.vision.spatiotemporalfilter_test_support import (
    BASE_STIM_PARAMS as base_stim_params,
    PARAMS as params,
)

PARAM_RNG = np.random.RandomState(1729)


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
        from pyNN import nest

        global nest
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

    def make_cell(
        self,
        x,
        y,
        on,
        original_2024_lgn_mode,
        visual_space=None,
    ):
        gain_params = ParameterSet(
            {
                "gain": 1.0,
                "non_linear_gain": None,
            }
        )

        rf = self.receptive_field_on if on else self.receptive_field_off

        return CellWithReceptiveField(
            x,
            y,
            rf,
            gain_params,
            visual_space or self.visual_space,
            original_2024_lgn_mode,
        )

    @pytest.mark.parametrize("x", PARAM_RNG.randint(0, 30, size=5))
    @pytest.mark.parametrize("y", PARAM_RNG.randint(0, 30, size=5))
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
        rf = self.receptive_field_on if on else self.receptive_field_off
        cell = self.make_cell(0, 0, on, original_2024_lgn_mode=True)
        cell.initialize(5)
        cell.view()

        # Separately test contrast and luminance response
        # Before applying nonlinear gain, they act as linear filters
        r = cell.kernel_response.contrast[: rf.kernel.shape[2]]
        pos = y + x * rf.kernel.shape[0]
        np.testing.assert_allclose(r, rf.kernel_contrast_component[:, pos])

        # The luminance response kernel is equal at all spatial positions, so
        # we don't calculate it for each position, rather multiply the 1D version
        # of it by the mean image luminance.
        # That is equivalent to a 3D luminance kernel which is convolved and with the
        # image and then summed at each time point.
        r = cell.kernel_response.luminance[: rf.kernel.shape[2]]
        np.testing.assert_allclose(r, rf.kernel_luminance_component * cell.va.mean())

    @pytest.mark.parametrize("dt_factor", [1, 2, 5, 10])
    def test_subframe_update(self, dt_factor):
        """
        Check that sub-frame updates produce the same response as explicit fine-grained
        updates. When the visual space update interval is a multiple of the kernel
        temporal resolution, view() internally reuses the same frame across several
        kernel time steps. This should give the same result as presenting those time
        steps individually.
        """
        rf = self.receptive_field_on
        dt = rf.temporal_resolution
        bg = self.visual_space.background_luminance

        # Create visual spaces
        vs_fine = VisualSpace(
            ParameterSet({"update_interval": dt, "background_luminance": bg})
        )
        vs_coarse = VisualSpace(
            ParameterSet(
                {"update_interval": dt_factor * dt, "background_luminance": bg}
            )
        )

        # Create cells (no state retention, zero starting luminance)
        cell_fine = self.make_cell(
            0, 0, True, original_2024_lgn_mode=True, visual_space=vs_fine
        )
        cell_coarse = self.make_cell(
            0, 0, True, original_2024_lgn_mode=True, visual_space=vs_coarse
        )

        stim_duration = dt_factor * dt  # one coarse frame

        # Helper to create FlashedBar with common parameters
        def make_flash(frame_duration, trial):
            return topo.FlashedBar(
                relative_luminance=1.0,
                orientation=0.0,
                width=self.vs_params["size_x"],
                length=self.vs_params["size_y"],
                flash_duration=stim_duration,
                x=0.0,
                y=0.0,
                location_x=0,
                location_y=0,
                frame_duration=frame_duration,
                background_luminance=bg,
                density=self.vs_params["density"],
                size_x=self.vs_params["size_x"],
                size_y=self.vs_params["size_y"],
                trial=trial,
            )

        vs_fine.add_object("flash", make_flash(dt, 0))
        vs_coarse.add_object("flash", make_flash(dt_factor * dt, 1))

        # Simulation length: enough to cover stimulus + full kernel decay, aligned to coarse step
        total_sim = int(np.ceil((stim_duration + rf.duration) / (dt_factor * dt))) * (
            dt_factor * dt
        )

        cell_fine.initialize(total_sim)
        cell_coarse.initialize(total_sim)

        # Run simulations
        def run(cell, vs):
            t = 0
            while t < total_sim:
                t = vs.update()
                cell.view()

        run(cell_fine, vs_fine)
        run(cell_coarse, vs_coarse)

        np.testing.assert_allclose(
            cell_coarse.kernel_response.contrast,
            cell_fine.kernel_response.contrast,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            cell_coarse.kernel_response.luminance,
            cell_fine.kernel_response.luminance,
            atol=1e-12,
        )

    gain_params_combinations = [
        (c_gains, c_scalers, l_gains, l_scalers)
        for c_gains, c_scalers in itertools.product(
            [0.11, 0.5, 1.0, 2.0], [0.00013, 0.01, 0.1, 1.0]
        )
        for l_gains, l_scalers in itertools.product(
            [0.009, 0.05, 0.5, 1.0], [0.4, 1.0, 5.0, 10.0]
        )
    ]

    @pytest.mark.parametrize(
        "c_gain,c_scaler,l_gain,l_scaler", gain_params_combinations
    )
    def test_nonlinear_gain(self, c_gain, c_scaler, l_gain, l_scaler):
        """
        Check that contrast and luminance responses are scaled independently by the
        non-linear gain function and then summed to produce the final current. The
        result should match a direct evaluation of the gain function applied to the
        same inputs.
        """
        rf = self.receptive_field_on
        vs = self.visual_space
        kdur = rf.kernel_duration
        total_len = kdur + 3

        gain_params = ParameterSet(
            {
                "gain": 1.0,
                "non_linear_gain": {
                    "contrast_gain": c_gain,
                    "contrast_scaler": c_scaler,
                    "luminance_gain": l_gain,
                    "luminance_scaler": l_scaler,
                },
            }
        )
        cell = CellWithReceptiveField(0, 0, rf, gain_params, vs)

        # Deterministic, random-but‑fixed inputs
        rng = np.random.RandomState(42)
        contrast = rng.uniform(-0.2, 0.2, total_len)
        luminance = (rng.rand(total_len) + 1) * vs.background_luminance

        # Expected current via Naka‑Rushton
        def nr(x, gain, scaler):
            return gain * x / (np.abs(x) + scaler)

        exp_contrast = nr(contrast, c_gain, c_scaler)
        exp_luminance = nr(luminance, l_gain, l_scaler)
        expected = (exp_contrast + exp_luminance)[:-kdur]

        kernel_response = KernelResponse(contrast=contrast, luminance=luminance)
        result = cell.response_current(kernel_response)

        np.testing.assert_allclose(result["amplitudes"], expected, rtol=1e-12)


class TestSpatioTemporalFilterRetinaLGN:
    @classmethod
    def setup_class(cls):
        from pyNN import nest

        global nest

    @staticmethod
    def make_retina(
        parameters,
        visual_field_size=(7.0, 7.0),
        visual_field_center=(0.0, 0.0),
    ):
        mozaik.setup_seeds(
            model_seed=parameters["model_seed"],
            simulation_seed=parameters["simulation_seed"],
            experiment_seed=parameters["experiment_seed"],
        )
        mozaik.setup_mpi()

        model = Model(nest, 2, parameters)

        model.visual_field = VisualRegion(
            location_x=visual_field_center[0],
            location_y=visual_field_center[1],
            size_x=visual_field_size[0],
            size_y=visual_field_size[1],
        )

        retina = SpatioTemporalFilterRetinaLGN(
            model,
            parameters["sheets"]["retina_lgn"]["params"],
        )

        return model, retina

    @staticmethod
    def current_traces(retina):
        return [
            (
                np.asarray(n.times.evaluate()),
                np.asarray(n.amplitudes.evaluate()),
            )
            for sheet in retina.scs
            for n in retina.scs[sheet]
        ]

    def test_process_input_internal_stimulus_cache(self, monkeypatch):
        """
        Check that process_input() reuses cached kernel responses across trials
        of the same stimulus, but recomputes responses for non-identical
        stimulus parametrizations.

        The test perturbs retained filter state before the repeated-trial
        presentation, so a cache miss would be visible as both an extra
        kernel-response calculation and a different injected current trace.
        """
        parameters = copy.deepcopy(load_parameters(params, ParameterSet({})))
        del parameters["visual_field"]

        model, retina = self.make_retina(parameters)
        stimulus_duration = 28.0

        grating_parameters = {
            **base_stim_params,
            "orientation": 0.0,
            "spatial_frequency": 0.5,
            "temporal_frequency": 1000.0 / stimulus_duration,
            "contrast": 50.0,
            "duration": stimulus_duration,
        }

        def make_grating(trial, contrast=50.0):
            return topo.FullfieldDriftingSinusoidalGrating(
                **{
                    **grating_parameters,
                    "contrast": contrast,
                    "trial": trial,
                }
            )

        kernel_response_calculations = {"count": 0}
        original_calculate_kernel_responses = retina.calculate_kernel_responses

        def spy_calculate_kernel_responses(visual_space, duration):
            kernel_response_calculations["count"] += 1
            return original_calculate_kernel_responses(visual_space, duration)

        monkeypatch.setattr(
            retina,
            "calculate_kernel_responses",
            spy_calculate_kernel_responses,
        )

        def present_grating(stimulus, offset):
            model.input_space.clear()
            model.input_space.add_object(str(stimulus), stimulus)
            retina.process_input(
                model.input_space,
                stimulus,
                duration=stimulus_duration,
                offset=offset,
            )
            return self.current_traces(retina)

        def perturb_filter_state():
            for rf_type in retina.rf_types:
                for cell in retina.input_cells[rf_type]:
                    cell.filter_state.contrast.fill(123.0)
                    cell.filter_state.luminance.fill(456.0)

        def assert_same_current_amplitudes(first_traces, second_traces):
            for (times_first, amplitudes_first), (
                times_second,
                amplitudes_second,
            ) in zip(
                first_traces,
                second_traces,
            ):
                assert len(times_second) == len(times_first)
                np.testing.assert_allclose(amplitudes_second, amplitudes_first)

        initial_trial_traces = present_grating(make_grating(trial=1), offset=0.0)
        assert kernel_response_calculations["count"] == 1
        assert len(retina.internal_stimulus_cache) == 1

        perturb_filter_state()

        repeated_trial_traces = present_grating(
            make_grating(trial=2),
            offset=stimulus_duration,
        )
        assert kernel_response_calculations["count"] == 1
        assert len(retina.internal_stimulus_cache) == 1
        assert_same_current_amplitudes(initial_trial_traces, repeated_trial_traces)

        changed_contrast_traces = present_grating(
            make_grating(trial=3, contrast=60.0),
            offset=2 * stimulus_duration,
        )
        assert kernel_response_calculations["count"] == 2
        assert len(retina.internal_stimulus_cache) == 2

        assert any(
            not np.allclose(amplitudes_changed, amplitudes_first)
            for (_, amplitudes_first), (_, amplitudes_changed) in zip(
                initial_trial_traces,
                changed_contrast_traces,
            )
        )

    @pytest.mark.parametrize("input_space_update_interval", [7, 14])
    @pytest.mark.parametrize("background_luminance", [10, 20, 40, 80])
    @pytest.mark.parametrize("rf_duration", [50, 100, 200])
    @pytest.mark.parametrize(
        "presentation_pair",
        [
            ("null", "null"),  # NN
            ("null", "explicit"),  # Nn
            ("explicit", "null"),  # nN
            ("explicit", "explicit"),  # nn
        ],
    )
    def test_blank_stimulus(
        self,
        background_luminance,
        rf_duration,
        presentation_pair,
        input_space_update_interval,
    ):
        """
        Test that null stimuli resulting from provide_null_input
        and explicit Null stimuli produce equivalent responses
        under all pairwise presentation combinations:
            NN, Nn, nN, nn
        """

        parameters = copy.deepcopy(load_parameters(params, ParameterSet({})))

        parameters["input_space"]["background_luminance"] = background_luminance

        parameters["input_space"]["update_interval"] = input_space_update_interval

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

        model, retina = self.make_retina(
            parameters,
            visual_field_size=(vf.size_x, vf.size_y),
            visual_field_center=(vf.location_x, vf.location_y),
        )

        duration = (retina.rf["X_ON"].kernel_duration + 100) * parameters[
            "input_space"
        ]["update_interval"]

        stim = topo.Null(
            **{
                **base_stim_params,
                "background_luminance": background_luminance,
                "duration": duration,
            }
        )

        model.input_space.add_object(
            "blank_stimulus",
            stim,
        )

        def run_presentation(kind):
            if kind == "null":
                retina.provide_null_input(
                    model.input_space,
                    duration=duration,
                )
            elif kind == "explicit":
                retina.process_input(
                    model.input_space,
                    stim,
                    duration=duration,
                )
            else:
                raise ValueError(kind)

            return self.current_traces(retina)

        traces_a = run_presentation(presentation_pair[0])
        traces_b = run_presentation(presentation_pair[1])

        for (t_a, a_a), (t_b, a_b) in zip(
            traces_a,
            traces_b,
        ):
            np.testing.assert_allclose(
                a_a,
                np.interp(t_a, t_b, a_b),
                rtol=1e-5,
                atol=1e-7,
            )

    # Test short and long three-stimulus splits. The 63 ms third segment makes
    # 70 + 70 + 63 ms equal the full RF kernel length in time bins.
    kernel_duration_split_grating_durations = (70, 70, 63)
    split_grating_durations = [
        (first_duration, second_duration, 63)
        for first_duration, second_duration in itertools.product(
            [210, 70, 14],
            [490, 280, 70, 14],
        )
    ]
    split_grating_duration_cases = [
        pytest.param(
            durations,
            id=(
                "kernel-duration"
                if durations == (70, 70, 63)
                else "%d-%d-%d" % durations
            ),
        )
        for durations in split_grating_durations
    ]

    @pytest.mark.parametrize("input_space_update_interval", [7, 14])
    @pytest.mark.parametrize("durations", split_grating_duration_cases)
    def test_continuous_vs_split_grating(
        self,
        durations,
        input_space_update_interval,
    ):
        """
        Verify that splitting a drifting grating into consecutive
        stimuli produces the same response as presenting the same
        frame sequence continuously.

        The parametrization includes a case matching the full RF kernel length
        in time bins.
        """

        parameters = copy.deepcopy(load_parameters(params, ParameterSet({})))
        del parameters["visual_field"]

        parameters["input_space"]["update_interval"] = input_space_update_interval

        m_cont, sh_cont = self.make_retina(parameters)
        m_split, sh_split = self.make_retina(parameters)

        assert sum(self.kernel_duration_split_grating_durations) == (
            sh_cont.rf["X_ON"].kernel_duration * sh_cont.rf["X_ON"].temporal_resolution
        )

        total_duration = sum(durations)

        stim_params = {
            **base_stim_params,
            "orientation": 0.0,
            "spatial_frequency": 0.5,
            # Use the largest period dividing all segment durations, so each
            # restarted split stimulus begins at phase 0
            "temporal_frequency": 1000.0 / np.gcd.reduce(durations),
            "contrast": 100.0,
        }
        stim_params["frame_duration"] = input_space_update_interval

        def make_stim(duration):
            return topo.FullfieldDriftingSinusoidalGrating(
                **{
                    **stim_params,
                    "duration": duration,
                }
            )

        # Continuous presentation

        m_cont.input_space.clear()
        m_cont.input_space.add_object(
            "continuous",
            make_stim(total_duration),
        )

        currents_cont = sh_cont._calculate_input_currents(
            sh_cont.calculate_kernel_responses(
                m_cont.input_space,
                total_duration,
            )[0]
        )

        amp_cont = currents_cont["X_ON"][0]["amplitudes"]

        # Split presentation

        split_currents = []

        for index, duration in enumerate(durations, start=1):
            m_split.input_space.clear()
            m_split.input_space.add_object(
                "part%d" % index,
                make_stim(duration),
            )

            currents = sh_split._calculate_input_currents(
                sh_split.calculate_kernel_responses(
                    m_split.input_space,
                    duration,
                )[0]
            )

            split_currents.append(currents["X_ON"][0]["amplitudes"])

        amp_split = np.concatenate(split_currents)

        np.testing.assert_allclose(
            amp_split,
            amp_cont,
            rtol=1e-5,
            atol=1e-7,
            err_msg=(
                "Split drifting grating response does not match "
                "continuous presentation."
            ),
        )

    @pytest.mark.parametrize("input_space_update_interval", [7, 14])
    @pytest.mark.parametrize("stimulus_duration", [280, 210, 75, 25])
    @pytest.mark.parametrize("null_duration", [240, 203, 150, 28, 13])
    def test_stimulus_null_equivalence(
        self, stimulus_duration, null_duration, input_space_update_interval
    ):
        """
        Verify that replacing explicit Null stimuli with provide_null_input()
        produces identical responses when stimuli are presented consecutively.

        Compare all four combinations:

            S -> N -> S -> N
            S -> N -> S -> n
            S -> n -> S -> N
            S -> n -> S -> n

        where:
            S = drifting grating
            N = explicit Null stimulus
            n = provide_null_input()
        """

        parameters = copy.deepcopy(load_parameters(params, ParameterSet({})))
        del parameters["visual_field"]

        parameters["input_space"]["update_interval"] = input_space_update_interval

        base_stimulus_params = copy.deepcopy(
            load_parameters(base_stim_params, ParameterSet({}))
        )
        base_stimulus_params["frame_duration"] = input_space_update_interval
        stim_params = {
            **base_stimulus_params,
            "orientation": 0.0,
            "spatial_frequency": 0.5,
            "temporal_frequency": 1000.0 / stimulus_duration,
            "contrast": 100.0,
            "duration": stimulus_duration,
        }

        background_luminance = parameters["input_space"]["background_luminance"]

        def make_stim():
            return topo.FullfieldDriftingSinusoidalGrating(**stim_params)

        def make_null(base_stimulus_params):
            return topo.Null(
                **{
                    **base_stimulus_params,
                    "background_luminance": background_luminance,
                    "duration": null_duration,
                }
            )

        def append_currents(currents_list, retina, visual_space, duration):
            """Append raw currents from an explicit stimulus (via process_input)."""
            currents = retina._calculate_input_currents(
                retina.calculate_kernel_responses(visual_space, duration)[0]
            )
            currents_list.extend(
                retina.parameters.linear_scaler * currents[sheet][i]["amplitudes"]
                for sheet in retina.rf_types
                for i in range(len(currents[sheet]))
            )

        def run_sequence(
            first_null_explicit, second_null_explicit, base_stimulus_params
        ):
            model, retina = self.make_retina(parameters)
            responses = []

            # Stimulus 1
            model.input_space.clear()
            model.input_space.add_object("stim1", make_stim())
            append_currents(responses, retina, model.input_space, stimulus_duration)

            # Null 1
            if first_null_explicit:
                model.input_space.clear()
                model.input_space.add_object("null1", make_null(base_stimulus_params))
                append_currents(responses, retina, model.input_space, null_duration)
            else:
                cell_resp = retina.calculate_null_input(null_duration)
                responses.extend(
                    [
                        retina.parameters.linear_scaler * resp["amplitudes"]
                        for rf_type in cell_resp.keys()
                        for resp in cell_resp[rf_type]
                    ]
                )

            # Stimulus 2
            model.input_space.clear()
            model.input_space.add_object("stim2", make_stim())
            append_currents(responses, retina, model.input_space, stimulus_duration)

            # Null 2
            if second_null_explicit:
                model.input_space.clear()
                model.input_space.add_object("null2", make_null(base_stimulus_params))
                append_currents(responses, retina, model.input_space, null_duration)
            else:
                cell_resp = retina.calculate_null_input(null_duration)
                responses.extend(
                    [
                        retina.parameters.linear_scaler * resp["amplitudes"]
                        for rf_type in cell_resp.keys()
                        for resp in cell_resp[rf_type]
                    ]
                )

            return responses

        r_NN = run_sequence(True, True, base_stimulus_params)
        r_Nn = run_sequence(True, False, base_stimulus_params)
        r_nN = run_sequence(False, True, base_stimulus_params)
        r_nn = run_sequence(False, False, base_stimulus_params)

        for i, result in enumerate([r_Nn, r_nN, r_nn]):
            for ref, res in zip(r_NN, result):
                np.testing.assert_allclose(
                    ref,
                    res,
                    rtol=1e-5,
                    atol=1e-7,
                    err_msg=f"Sequence index {i} mismatch.",
                )
