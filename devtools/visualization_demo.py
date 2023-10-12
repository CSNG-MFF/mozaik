import matplotlib

matplotlib.use("TkAgg")
import mozaik.stimuli.vision.topographica_based as topo
import mozaik.experiments.vision as exp
import mozaik.experiments.apparent_motion as am
import stimulus_visualization as viz
import dummy_model as dm
from parameters import ParameterSet
import numpy as np


def experiment_default_parameters():
    return {
        "relative_luminance": 1,
        "central_rel_lum": 1,
        "orientation": 0,
        "phase": 0,
        "spatial_frequency": 1,
        "size": 2,
        "flash_duration": 10,
        "x": 0,
        "y": 0,
        "rotations": 2,
        "duration": 1000,
        "num_trials": 1,
        "circles": 2,
        "grid": True,
    }


def base_stim_default_parameters():
    return {
        "frame_duration": 1,
        "duration": 200,
        "trial": 1,
        "direct_stimulation_name": "simulation_name",
        "direct_stimulation_parameters": None,
    }


def visual_stim_default_parameters():
    d = base_stim_default_parameters()
    d.update(
        {
            "background_luminance": 50.0,
            "density": 10.0,
            "location_x": 0.0,
            "location_y": 0.0,
            "size_x": 11.0,
            "size_y": 11.0,
        }
    )
    return d


def ContinuousGaborMovementAndJump_default_parameters():
    d = visual_stim_default_parameters()
    d.update(
        {
            "orientation": 0,
            "phase": 0,
            "spatial_frequency": 1,
            "sigma": 0.5,
            "center_relative_luminance": 1,
            "moving_relative_luminance": 1,
            "x": 0,
            "y": 0,
            "movement_duration": 100,
            "movement_length": 3,
            "movement_angle": 0,
            "moving_gabor_orientation_radial": False,
            "center_flash_duration": 20,
        }
    )
    return d


def SparseNoise_default_parameters():
    d = visual_stim_default_parameters()
    d.update(
        {
            "blank_time": 0,
            "experiment_seed": 0,
            "time_per_image": 20,
            "grid_size": 11,
            "grid": True,
        }
    )
    return d


def exp_default_parameters():
    return {"duration": 1000}


def MapSimpleGabor_default_parameters():
    d = exp_default_parameters()
    d.update(
        {
            "relative_luminance": 1.0,
            "central_rel_lum": 0.5,
            "orientation": 0,
            "phase": 0,
            "spatial_frequency": 1,
            "size": 2.0,
            "flash_duration": 20,
            "x": 0,
            "y": 0,
            "rotations": 2,
            "duration": 93,
            "num_trials": 1,
            "circles": 2,
            "grid": False,
        }
    )
    return d


def MapTwoStrokeGabor_default_parameters():
    d = exp_default_parameters()
    d.update(
        {
            "relative_luminance": 1.0,
            "central_rel_lum": 0.5,
            "orientation": 0,
            "phase": 0,
            "spatial_frequency": 1,
            "size": 2.0,
            "flash_duration": 20,
            "stroke_time": 10,
            "x": 0,
            "y": 0,
            "rotations": 6,
            "duration": 50,
            "num_trials": 1,
            "circles": 2,
            "grid": False,
        }
    )
    return d


def CompareSlowVersusFastGaborMotion_default_parameters():
    d = {
        "num_trials": 2,
        "x": 0,
        "y": 0,
        "orientation": 0,
        "phase": 1,
        "spatial_frequency": 2,
        "sigma": 0.5,
        "n_sigmas": 3.0,
        "center_relative_luminance": 0.5,
        "surround_relative_luminance": 0.7,
        "movement_speeds": [5.0, 180.0],
        "angles": list(np.linspace(0, 2 * np.pi, 12, endpoint=False)),
        "moving_gabor_orientation_radial": True,
        "n_circles": 3,
        "neuron_id": 0,
        "blank_duration": 200,
    }
    return d


def MeasureGaborFlashDuration_default_parameters():
    d = {
        "num_trials": 10,
        "x": 1,
        "y": 1,
        "orientation": 0,
        "phase": 1,
        "spatial_frequency": 0.8,
        "sigma": 3.0 / 6.0,
        "n_sigmas": 3.0,
        "relative_luminance": 1.0,
        "min_duration": 14,
        "max_duration": 42,
        "step": 7,
        "blank_duration": 100,
        "neuron_id": 0,
    }
    return d


def RunApparentMotionConfigurations_default_parameters():
    d = {
        "num_trials": 1,
        "x": 0,
        "y": 0,
        "orientation": 0,
        "phase": 0,
        "spatial_frequency": 0.8,
        "sigma": 0.5,
        "n_sigmas": 3.0,
        "center_relative_luminance": 0.5,
        "surround_relative_luminance": 1.0,
        "configurations": [
            "SECTOR_ISO",
            "SECTOR_CROSS",
            "SECTOR_CF",
            "SECTOR_RND",
            "FULL_ISO",
            "FULL_CROSS",
            "FULL_RND",
            "CENTER_ONLY",
        ],
        "random_order": True,
        "n_circles": 3,
        "flash_center": True,
        "flash_duration": 28,
        "blank_duration": 100,
        "neuron_id": 0,
    }
    return d


def MeasureSparseBar_default_parameters():
    d = {
        "time_per_image": 21,
        "blank_time": 140,
        "total_number_of_images": 20,
        "num_trials": 1,
        "orientation": 0,
        "bar_length": 10,
        "bar_width": 1,
        "x": 0,
        "y": 0,
        "n_positions": 10,
        "experiment_seed": 17,
    }
    return d


def demo_show_frame():
    params = ContinuousGaborMovementAndJump_default_parameters()
    stim = topo.ContinuousGaborMovementAndJump(**params)
    frames = viz.pop_frames(stim, 100)
    viz.show_frame(frames[0], params=params, grid=True)
    viz.show_frames(frames)


def demo_stimulus_0():
    params = ContinuousGaborMovementAndJump_default_parameters()
    stim = topo.ContinuousGaborMovementAndJump(**params)
    viz.show_stimulus(stim)


def demo_stimulus_1():
    params = SparseNoise_default_parameters()
    stim = topo.SparseNoise(**params)
    viz.show_stimulus(stim)


def demo_stimulus_0_duration():
    params = ContinuousGaborMovementAndJump_default_parameters()
    stim = topo.ContinuousGaborMovementAndJump(**params)
    viz.show_stimulus(stim, duration=130)


def demo_stimulus_0_grid():
    params = ContinuousGaborMovementAndJump_default_parameters()
    stim = topo.ContinuousGaborMovementAndJump(**params)
    viz.show_stimulus(stim, grid=2)


def demo_stimulus_0_frame_delay():
    params = ContinuousGaborMovementAndJump_default_parameters()
    stim = topo.ContinuousGaborMovementAndJump(**params)
    viz.show_stimulus(stim, frame_delay=100)


def demo_stimulus_0_animate():
    params = ContinuousGaborMovementAndJump_default_parameters()
    stim = topo.ContinuousGaborMovementAndJump(**params)
    viz.show_stimulus(stim, grid=True, animate=False)


def demo_experiment_0():
    model = dm.DummyModel(**visual_stim_default_parameters())
    params = MapTwoStrokeGabor_default_parameters()
    parameters = ParameterSet(params)
    experiment = am.MapTwoStrokeGabor(model=model, parameters=parameters)
    viz.show_experiment(experiment, merge_stimuli=True)


def demo_experiment_1():
    model = dm.DummyModel(**visual_stim_default_parameters())
    params = MapSimpleGabor_default_parameters()
    parameters = ParameterSet(params)
    experiment = am.MapSimpleGabor(model=model, parameters=parameters)
    viz.show_experiment(experiment, merge_stimuli=False, grid=True)


def demo_experiment_2():
    model_params = visual_stim_default_parameters()
    model_params["frame_duration"] = 7
    model = dm.DummyModel(**model_params)
    params = CompareSlowVersusFastGaborMotion_default_parameters()
    parameters = ParameterSet(params)
    experiment = am.CompareSlowVersusFastGaborMotion(model=model, parameters=parameters)
    viz.show_experiment(experiment, merge_stimuli=False)


def demo_experiment_3():
    model_params = visual_stim_default_parameters()
    model_params["frame_duration"] = 7
    model = dm.DummyModel(**model_params)
    params = MeasureGaborFlashDuration_default_parameters()
    params["orientation"] = np.pi / 4
    params["x"] = 1
    params["y"] = 1
    parameters = ParameterSet(params)
    experiment = am.MeasureGaborFlashDuration(model=model, parameters=parameters)
    viz.show_experiment(experiment, merge_stimuli=False)


def demo_experiment_4():
    model_params = visual_stim_default_parameters()
    model_params["frame_duration"] = 7
    model = dm.DummyModel(**model_params)
    params = RunApparentMotionConfigurations_default_parameters()
    params["orientation"] = np.pi / 4
    params["x"] = 0
    params["y"] = 0
    parameters = ParameterSet(params)
    experiment = am.RunApparentMotionConfigurations(model=model, parameters=parameters)
    viz.show_experiment(experiment, merge_stimuli=False, frame_delay=100)


def demo_experiment_5():
    model_params = visual_stim_default_parameters()
    model_params["frame_duration"] = 7
    model = dm.DummyModel(**model_params)
    params = MeasureSparseBar_default_parameters()
    parameters = ParameterSet(params)
    experiment = exp.MeasureSparseBar(model=model, parameters=parameters)
    viz.show_experiment(experiment, merge_stimuli=True, frame_delay=50)


def main():
    demo_experiment_5()

    if False:
        # Try out show_stimulus arguments
        demo_stimulus_0()
        demo_stimulus_1()
        demo_stimulus_0_duration()
        demo_stimulus_0_grid()
        demo_stimulus_0_frame_delay()
        # demo_stimulus_0_animate() # Commented out as you'd have to click through all frames

    if False:
        # Try out visualizing experiments
        demo_experiment_0()
        demo_experiment_1()

    if False:
        # Try out the underlying show_frame function
        demo_show_frame()


main()
