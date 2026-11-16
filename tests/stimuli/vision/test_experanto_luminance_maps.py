"""Check the registered luminance map through experiment construction and replay."""

import ast
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
import yaml

from parameters import ParameterSet

from mozaik.experiments.vision import RandomizedExperanto
from mozaik.stimuli.vision.luminance_maps import (
    CenteredSymmetricContrast,
    LinearLuminanceWindow,
    resolve_luminance_map,
)
from mozaik.stimuli.vision.topographica_based import NaturalImage
from mozaik.tools.experanto_export import MozaikScreenExporter
from mozaik.tools.mozaik_parametrized import MozaikParametrized


@pytest.mark.parametrize(
    "transfer, window, width",
    [
        ("CenteredSymmetricContrast", None, 11.0),
        ("CenteredSymmetricContrast", 1.0, 8.0),
        ("LinearLuminanceWindow", 10.0, 8.0),
    ],
)
def test_luminance_map_replays_and_exports_its_settings(
    tmp_path, transfer, window, width
):
    effective_window = 0.2 if window is None else window
    data = tmp_path / "screen/data"
    meta = tmp_path / "screen/meta"
    data.mkdir(parents=True)
    meta.mkdir()
    rng = np.random.default_rng(1023)
    pixels = rng.integers(0, 256, (144, 144), dtype=np.uint8)
    # The sampler transforms in place, including when source files are float32.
    np.save(data / "image.npy", pixels.astype(np.float32))
    np.save(data / "video.npy", pixels[None].astype(np.float32))
    Image.fromarray(pixels).save(tmp_path / "reference.png")
    (meta / "image.yml").write_text(
        yaml.safe_dump(
            dict(
                modality="image",
                num_frames=1,
                condition_hash="image",
                pre_blank_period=0.0,
                presentation_time=0.28,
                post_blank_period=0.28,
            )
        )
    )
    (meta / "video.yml").write_text(
        yaml.safe_dump(
            dict(
                modality="video",
                num_frames=1,
                condition_hash="video",
            )
        )
    )
    (tmp_path / "chunk.json").write_text(
        json.dumps(
            [
                {"file": "image.yml", "trial": 0},
                {"file": "video.yml", "trial": 0},
            ]
        )
    )
    model = SimpleNamespace(
        parameters=SimpleNamespace(null_stimulus_period=0),
        input_space=SimpleNamespace(
            background_luminance=40.1,
            parameters=SimpleNamespace(update_interval=7.0),
        ),
        input_layer=SimpleNamespace(
            parameters=SimpleNamespace(
                receptive_field=SimpleNamespace(spatial_resolution=0.0763889),
            )
        ),
        visual_field=SimpleNamespace(size_x=11.0, size_y=11.0),
    )
    parameters = ParameterSet(
        dict(
            base_path=str(tmp_path),
            chunk_dict_path="chunk.json",
            width=width,
            movie_frame_duration=35,
            global_frame_offset=0,
            images_per_trial=1,
            video_max_value=255.0,
            shuffle_stimuli=False,
        )
    )
    experiment = RandomizedExperanto(
        model,
        parameters,
        luminance_map=transfer,
        luminance_map_window=window,
    )
    image, blank, video = experiment.stimuli
    record = ast.literal_eval(str(image))
    assert record["luminance_map"] == transfer
    assert record["luminance_map_window"] == effective_window
    assert video.luminance_map == "MaximumDynamicRange"
    assert video.luminance_map_window is None

    reference = NaturalImage(
        image_path=str(tmp_path / "reference.png"),
        image_duration=280.0,
        blank_duration=0.0,
        duration=280.0,
        frame_duration=7.0,
        size=width,
        density=1 / 0.0763889,
        background_luminance=40.1,
        trial=0,
        size_x=11.0,
        size_y=11.0,
        location_x=0.0,
        location_y=0.0,
    )
    baseline = next(reference.frames())[0]
    if transfer == "LinearLuminanceWindow":
        expected = 40.1 - effective_window + baseline * (effective_window / 40.1)
        np.testing.assert_allclose(expected[0, 0], 30.1)
    else:
        deviations = baseline - baseline.mean()
        expected = 40.1 * (
            1 + effective_window * (deviations / np.max(np.abs(deviations)))
        )
    replay = MozaikParametrized.idd_to_instance(str(image))
    for stimulus in (image, replay):
        stimulus.reset()
        if transfer == "LinearLuminanceWindow":
            np.testing.assert_allclose(stimulus.img, expected, rtol=1e-15)
        else:
            np.testing.assert_array_equal(stimulus.img, expected)
        rendered = stimulus.img.copy()
        stimulus.update()
        np.testing.assert_array_equal(stimulus.img, rendered)

    dsv = SimpleNamespace(
        get_segments=lambda: [
            SimpleNamespace(annotations=dict(stimulus=str(stimulus), sheet_name="X_ON"))
            for stimulus in (replay, blank, video)
        ],
        get_model_parameters=lambda: {"null_stimulus_period": 0.0},
    )
    exporter = MozaikScreenExporter(str(tmp_path / "export"), trial_id=0)
    exporter.process(dsv)
    exporter.finalize()
    screen = tmp_path / "export/screen"
    entries = json.loads((screen / "combined_meta.json").read_text())
    assert entries["00000"]["mozaik_luminance_map"] == {
        "name": transfer,
        "window": effective_window,
    }
    np.testing.assert_array_equal(np.load(screen / "data/00000.npy"), pixels)


def test_centered_transfer_maps_constant_image_to_background():
    baseline = np.full((3, 5), 12.5)
    result = CenteredSymmetricContrast()(baseline, 50.0)
    np.testing.assert_array_equal(result, 50.0)


def test_linear_window_uses_fixed_endpoints():
    transfer = LinearLuminanceWindow(window=10.0)
    np.testing.assert_array_equal(
        transfer(np.array([[0.0, 25.0, 50.0, 75.0, 100.0]]), 50.0),
        [[40.0, 45.0, 50.0, 55.0, 60.0]],
    )
    np.testing.assert_array_equal(
        transfer(np.array([[25.0, 25.0]]), 50.0), [[45.0, 45.0]]
    )
    with pytest.raises(ValueError, match="cannot exceed"):
        transfer(np.array([[5.0]]), 5.0)


@pytest.mark.parametrize("window", [None, -1, float("nan"), float("inf"), True])
def test_linear_window_requires_explicit_finite_nonnegative_value(window):
    with pytest.raises(ValueError, match="finite nonnegative"):
        resolve_luminance_map("LinearLuminanceWindow", window)
