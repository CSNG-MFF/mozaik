"""Screen frames follow recorded presentations, with one synthetic endpoint."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from mozaik.tools.experanto_export import MozaikScreenExporter

IMAGE_META = dict(
    condition_hash="image",
    image_size=[144, 144],
    modality="image",
    num_frames=1,
    pre_blank_period=0.3812,
    presentation_time=0.5,
    tier="train",
)
VIDEO_META = dict(
    condition_hash="video",
    image_size=[36, 64],
    modality="video",
    num_frames=3,
    tier="test",
)
IMAGE_ITEM = dict(modality="image", file="00001.yml", trial=0)
VIDEO_ITEM = dict(modality="video", file="00002.yml", trial=0)


class _Seg:
    def __init__(self, params, sheet="X_ON"):
        self.annotations = {"stimulus": params, "sheet_name": sheet}


class _DSV:
    def __init__(self, segments):
        self._segments = segments

    def get_segments(self):
        return self._segments

    def get_model_parameters(self):
        return {"null_stimulus_period": 0.0}


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source" / "screen"
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir()
    for name, meta in (("00001", IMAGE_META), ("00002", VIDEO_META)):
        (root / "meta" / f"{name}.yml").write_text(yaml.safe_dump(meta))
        shape = (meta["num_frames"], *meta["image_size"])
        np.save(root / "data" / f"{name}.npy", np.full(shape, 7.0))
    return root


def _segments(source, items, frame_duration=7.0, video_frame_duration=35.0):
    segments = []
    for item in items:
        image = item["modality"] == "image"
        pre = frame_duration * (381.2 // frame_duration) if image else 0
        post = frame_duration * (49.0 // frame_duration) if image else 0
        period = (
            frame_duration * (500.0 // frame_duration)
            if image
            else video_frame_duration
        )
        if pre:
            segments.append(
                _Seg(dict(name="InternalStimulus", trial=item["trial"], duration=pre))
            )
        segments.append(
            _Seg(
                dict(
                    name="PixelMovieExperanto",
                    trial=item["trial"],
                    movie_path=str(source / "data"),
                    movie_name=item["file"].replace(".yml", ".npy"),
                    frame_duration=frame_duration,
                    movie_frame_duration=period,
                    duration=period if image else 3 * period,
                )
            )
        )
        if post:
            segments.append(
                _Seg(dict(name="InternalStimulus", trial=item["trial"], duration=post))
            )
    return segments


def _export(tmp_path, segments):
    exporter = MozaikScreenExporter(str(tmp_path / "shard"), trial_id=0)
    exporter.process(_DSV(segments))
    exporter.finalize()
    screen = Path(exporter.screen_dir)
    return json.loads((screen / "combined_meta.json").read_text()), np.load(
        screen / "timestamps.npy"
    )


def test_timeline_matches_recorded_presentations(tmp_path, source):
    combined, timestamps = _export(
        tmp_path, _segments(source, [IMAGE_ITEM, VIDEO_ITEM])
    )
    assert [entry["modality"] for entry in combined.values()] == [
        "blank",
        "image",
        "blank",
        "video",
        "blank",
    ]
    assert [entry["num_frames"] for entry in combined.values()] == [1, 1, 1, 3, 1]
    assert [entry["first_frame_idx"] for entry in combined.values()] == [0, 1, 2, 3, 6]
    np.testing.assert_allclose(
        timestamps, [0, 0.378, 0.875, 0.924, 0.959, 0.994, 1.029]
    )
    assert np.all(np.diff(timestamps) > 0)
    for key, entry in combined.items():
        assert (
            yaml.safe_load((tmp_path / "shard/screen/meta" / f"{key}.yml").read_text())
            == entry
        )
    # Blanks adjacent to images use the image dimensions; the endpoint follows the video.
    assert [entry["image_size"] for entry in combined.values()] == [[144, 144]] * 3 + [
        [36, 64]
    ] * 2


def test_warmup_and_other_trials_are_excluded(tmp_path, source):
    segments = [_Seg(dict(name="InternalStimulus", trial=None, duration=20020))]
    segments += _segments(source, [dict(VIDEO_ITEM, trial=1), VIDEO_ITEM])
    combined, timestamps = _export(tmp_path, segments)
    assert [entry["modality"] for entry in combined.values()] == ["video", "blank"]
    np.testing.assert_allclose(timestamps, [0, 0.035, 0.070, 0.105])


@pytest.mark.parametrize("process", [False, True])
def test_nothing_to_export_writes_no_shard(tmp_path, process):
    exporter = MozaikScreenExporter(str(tmp_path / "shard"), trial_id=0)
    if process:
        exporter.process(_DSV([]))
    exporter.finalize()
    assert not (Path(exporter.screen_dir) / "combined_meta.json").exists()
    assert not (Path(exporter.screen_dir) / "timestamps.npy").exists()


@pytest.mark.parametrize(
    "items, expected_ms",
    [
        ([IMAGE_ITEM], [0, 380, 880, 925]),
        ([VIDEO_ITEM], [0, 40, 80, 120]),
        ([IMAGE_ITEM, VIDEO_ITEM], [0, 380, 880, 925, 965, 1005, 1045]),
        ([VIDEO_ITEM, IMAGE_ITEM], [0, 40, 80, 120, 500, 1000, 1045]),
    ],
    ids=["image-only", "video-only", "image-video", "video-image"],
)
def test_screen_uses_recorded_stimulus_timing(tmp_path, source, items, expected_ms):
    segments = _segments(source, items, frame_duration=5.0, video_frame_duration=40.0)
    duplicates = [
        _Seg(repr(segment.annotations["stimulus"]), sheet="V1_Exc_L4")
        for segment in segments
    ]
    combined, timestamps = _export(tmp_path, segments + duplicates)
    assert all(
        entry["image_size"] == [144, 144]
        for entry in list(combined.values())[:-1]
        if entry["modality"] == "blank"
    )
    np.testing.assert_allclose(
        timestamps, np.array(expected_ms) / 1000, rtol=0, atol=1e-15
    )


def test_image_and_blanks_use_recorded_durations(tmp_path, source):
    segments = _segments(source, [IMAGE_ITEM])
    for segment, duration in zip(segments, [350, 450, 70]):
        segment.annotations["stimulus"]["duration"] = duration
    _, timestamps = _export(tmp_path, segments)
    np.testing.assert_allclose(timestamps, [0, 0.350, 0.800, 0.870], rtol=0, atol=1e-15)


@pytest.mark.parametrize("problem", ["missing", "inconsistent"])
def test_invalid_recorded_timing_is_rejected(tmp_path, source, problem):
    segments = _segments(source, [VIDEO_ITEM])
    if problem == "missing":
        del segments[0].annotations["stimulus"]["movie_frame_duration"]
        message = "Missing or invalid recorded video timing"
    else:
        duplicate = _segments(source, [VIDEO_ITEM], video_frame_duration=70.0)[0]
        duplicate.annotations["sheet_name"] = "V1_Exc_L4"
        segments.append(duplicate)
        message = "Inconsistent recorded presentations"
    with pytest.raises(ValueError, match=message):
        _export(tmp_path, segments)


def test_single_blank_between_different_image_sizes(tmp_path, source):
    other = dict(IMAGE_META, image_size=[2, 3])
    (source / "meta/00003.yml").write_text(yaml.safe_dump(other))
    np.save(source / "data/00003.npy", np.zeros((1, 2, 3)))
    first = dict(IMAGE_META, post_blank_period=0)
    (source / "meta/00001.yml").write_text(yaml.safe_dump(first))
    segments = _segments(source, [IMAGE_ITEM])[:-1]
    segments += _segments(source, [dict(IMAGE_ITEM, file="00003.yml")])
    combined, _ = _export(tmp_path, segments)
    assert [entry["image_size"] for entry in combined.values()] == [[144, 144]] * 2 + [
        [2, 3]
    ] * 4
