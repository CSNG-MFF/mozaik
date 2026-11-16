"""Check Experanto contrast through experiment construction and replay."""

import ast
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
import yaml

from parameters import ParameterSet

from mozaik.experiments.vision import RandomizedExperanto
from mozaik.stimuli.vision.topographica_based import NaturalImage
from mozaik.tools.json_export import get_experimental_protocols
from mozaik.tools.mozaik_parametrized import MozaikParametrized


@pytest.mark.parametrize("scaler", [None, 0.0, 0.2])
def test_experanto_contrast_is_applied_and_recorded(tmp_path, scaler):
    width = 8.0
    effective_scaler = 1.0 if scaler is None else scaler
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
    if scaler is not None:
        parameters["contrast_scaler"] = scaler
    experiment = RandomizedExperanto(model, parameters)
    image, _, video = experiment.stimuli
    assert ast.literal_eval(str(image))["contrast_scaler"] == effective_scaler
    assert video.contrast_scaler == effective_scaler

    store = SimpleNamespace(
        get_experiment_parametrization_list=lambda: [
            (str(experiment.__class__), str(experiment.parameters))
        ]
    )
    protocol = get_experimental_protocols(store)[0]
    assert (
        json.loads(json.dumps(protocol))["parameters"]["contrast_scaler"][0]
        == effective_scaler
    )

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
    expected = 40.1 + effective_scaler * (baseline - 40.1)
    replay = MozaikParametrized.idd_to_instance(str(image))
    for stimulus in (image, video, replay):
        stimulus.reset()
        np.testing.assert_allclose(stimulus.img, expected, rtol=1e-15, atol=2e-14)
        rendered = stimulus.img.copy()
        stimulus.update()
        np.testing.assert_array_equal(stimulus.img, rendered)
