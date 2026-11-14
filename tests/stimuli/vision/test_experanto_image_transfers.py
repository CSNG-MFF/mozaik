"""Check the registered image transfer through experiment construction and replay."""

import ast
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
import yaml

from parameters import ParameterSet

from mozaik.experiments.vision import RandomizedExperanto
from mozaik.stimuli.vision.image_transfers import CenteredSymmetricContrast
from mozaik.stimuli.vision.topographica_based import NaturalImage
from mozaik.tools.experanto_export import MozaikScreenExporter
from mozaik.tools.mozaik_parametrized import MozaikParametrized


@pytest.mark.parametrize("window, width", [(None, 11.0), (1.0, 8.0)])
def test_centered_image_transfer_replays_and_exports_its_settings(tmp_path, window, width):
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
    (meta / "image.yml").write_text(yaml.safe_dump(dict(
        modality="image", num_frames=1, condition_hash="image",
        pre_blank_period=0.0, presentation_time=.28, post_blank_period=.28,
    )))
    (meta / "video.yml").write_text(yaml.safe_dump(dict(
        modality="video", num_frames=1, condition_hash="video",
    )))
    (tmp_path / "chunk.json").write_text(json.dumps([
        {"file": "image.yml", "trial": 0},
        {"file": "video.yml", "trial": 0},
    ]))
    model = SimpleNamespace(
        parameters=SimpleNamespace(null_stimulus_period=0),
        input_space=SimpleNamespace(
            background_luminance=40.1,
            parameters=SimpleNamespace(update_interval=7.0),
        ),
        input_layer=SimpleNamespace(parameters=SimpleNamespace(
            receptive_field=SimpleNamespace(spatial_resolution=.0763889),
        )),
        visual_field=SimpleNamespace(size_x=11.0, size_y=11.0),
    )
    parameters = ParameterSet(dict(
        base_path=str(tmp_path), chunk_dict_path="chunk.json", width=width,
        movie_frame_duration=35, global_frame_offset=0, images_per_trial=1,
        video_max_value=255.0, shuffle_stimuli=False,
    ))
    experiment = RandomizedExperanto(
        model, parameters, image_transfer="CenteredSymmetricContrast",
        image_transfer_window=window,
    )
    image, blank, video = experiment.stimuli
    record = ast.literal_eval(str(image))
    assert record["image_transfer"] == "CenteredSymmetricContrast"
    assert record["image_transfer_window"] == effective_window
    assert video.image_transfer == "MaximumDynamicRange"
    assert video.image_transfer_window is None

    reference = NaturalImage(
        image_path=str(tmp_path / "reference.png"), image_duration=280.0,
        blank_duration=0.0, duration=280.0, frame_duration=7.0, size=width,
        density=1 / .0763889, background_luminance=40.1, trial=0,
        size_x=11.0, size_y=11.0, location_x=0.0, location_y=0.0,
    )
    baseline = next(reference.frames())[0]
    deviations = baseline - baseline.mean()
    expected = 40.1 * (1 + effective_window * (deviations / np.max(np.abs(deviations))))
    replay = MozaikParametrized.idd_to_instance(str(image))
    for stimulus in (image, replay):
        stimulus.reset()
        np.testing.assert_array_equal(stimulus.img, expected)
        stimulus.update()
        np.testing.assert_array_equal(stimulus.img, expected)

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
    assert entries["00000"]["mozaik_image_transfer"] == {
        "name": "CenteredSymmetricContrast", "window": effective_window,
    }
    np.testing.assert_array_equal(np.load(screen / "data/00000.npy"), pixels)


def test_centered_transfer_maps_constant_image_to_background():
    baseline = np.full((3, 5), 12.5)
    result = CenteredSymmetricContrast()(baseline, 50.0)
    np.testing.assert_array_equal(result, 50.0)
