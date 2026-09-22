"""
Tests for ``MozaikScreenExporter`` -- the screen half of the Experanto export.

The screen timeline and the spike timeline are written by two different exporters but describe
one experiment, and nothing downstream re-derives either: Experanto reads the timestamps as
given. So what these tests pin is the timing arithmetic: an image expands into the same
pre-blank / image / post-blank sequence the simulation presents, a video gets one timestamp per
frame, and the resulting screen clock ends exactly where the spike clock does. A drift here
desynchronises spikes from stimuli silently.
"""

import json
import os

import numpy as np
import pytest
import yaml

from mozaik.tools.experanto_export import POST_BLANK_MS, MozaikScreenExporter

FRAME_DURATION_MS = 7.0
MOVIE_FRAME_DURATION_MS = 35.0

IMAGE_META = {
    "condition_hash": "IMG0000000000000001",
    "image_size": [144, 144],
    "modality": "image",
    "num_frames": 1,
    "pre_blank_period": 0.3812,
    "presentation_time": 0.5,
    "tier": "train",
}
VIDEO_META = {
    "condition_hash": "VID0000000000000001",
    "image_size": [36, 64],
    "modality": "video",
    "num_frames": 3,
    "tier": "test",
}

# Both durations are quantised down to whole input frames (7 ms), which is why they are not
# simply 381.2 and 500: 7*(381.2//7) = 378 and 7*(500//7) = 497.
PRE_BLANK_MS = 378.0
PRESENTATION_MS = 497.0
IMAGE_TOTAL_MS = PRE_BLANK_MS + PRESENTATION_MS + POST_BLANK_MS  # 924.0
VIDEO_TOTAL_MS = VIDEO_META["num_frames"] * MOVIE_FRAME_DURATION_MS  # 105.0


class _Seg:
    """Minimal stand-in for a segment; the screen exporter reads only the stimulus annotation."""

    def __init__(self, movie_path):
        self.annotations = {"stimulus": {"movie_path": movie_path}}


class _DSV:
    def __init__(self, segments):
        self._segments = segments

    def get_segments(self):
        return self._segments


@pytest.fixture
def source(tmp_path):
    """A source screen dataset: one image, one video, plus their npy files."""
    root = tmp_path / "source"
    meta_dir = root / "screen" / "meta"
    data_dir = root / "screen" / "data"
    meta_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    for name, meta in (("00001", IMAGE_META), ("00002", VIDEO_META)):
        with open(meta_dir / (name + ".yml"), "w") as f:
            yaml.safe_dump(meta, f)
        shape = (
            tuple(meta["image_size"])
            if meta["modality"] == "image"
            else (meta["num_frames"],) + tuple(meta["image_size"])
        )
        np.save(str(data_dir / (name + ".npy")), np.full(shape, 7.0, dtype=np.float32))

    return {"root": str(root), "data_dir": str(data_dir)}


IMAGE_ITEM = {"modality": "image", "file": "00001.yml", "trial": 0}
VIDEO_ITEM = {"modality": "video", "file": "00002.yml", "trial": 0}


def _write_chunk(path, items):
    with open(path, "w") as f:
        json.dump(items, f)
    return str(path)


def _exporter(tmp_path, chunk_paths):
    return MozaikScreenExporter(
        output_dir=str(tmp_path / "shard"),
        chunk_paths=chunk_paths,
        frame_duration_ms=FRAME_DURATION_MS,
        movie_frame_duration_ms=MOVIE_FRAME_DURATION_MS,
    )


def _export(tmp_path, source, chunk_paths):
    exporter = _exporter(tmp_path, chunk_paths)
    exporter.process_batch([_DSV([_Seg(source["data_dir"])])])
    exporter.finalize()
    return os.path.join(str(tmp_path / "shard"), "screen")


def _read(screen_dir):
    with open(os.path.join(screen_dir, "combined_meta.json")) as f:
        combined = json.load(f)
    timestamps = np.load(os.path.join(screen_dir, "timestamps.npy"))
    return combined, timestamps


def test_timeline_matches_the_sequence_the_simulation_presents(tmp_path, source):
    """
    The whole timing contract in one place: an image becomes pre-blank / image / post-blank
    with durations quantised down to whole input frames, a video becomes one timestamp per
    frame, every shard ends with a trailing blank -- and the last timestamp lands exactly on
    the total presented duration, which is what the spike exporter writes as ``end_time``.
    That last equality is the one thing tying the two halves of a shard to the same clock.
    """
    chunk = _write_chunk(
        tmp_path / "0_0.json",
        [
            IMAGE_ITEM,
            VIDEO_ITEM,
        ],
    )
    combined, timestamps = _read(_export(tmp_path, source, [chunk]))
    keys = sorted(combined)

    assert [combined[k]["modality"] for k in keys] == [
        "blank",
        "image",
        "blank",
        "video",
        "blank",
    ]
    assert [combined[k]["num_frames"] for k in keys] == [1, 1, 1, 3, 1]

    # the running clock, entry by entry: 378 + 497 + 49 ms for the image, then 3 x 35 ms
    expected_ms = [
        0.0,
        PRE_BLANK_MS,
        PRE_BLANK_MS + PRESENTATION_MS,
        IMAGE_TOTAL_MS,
        IMAGE_TOTAL_MS + 35.0,
        IMAGE_TOTAL_MS + 70.0,
        IMAGE_TOTAL_MS + VIDEO_TOTAL_MS,
    ]
    np.testing.assert_allclose(timestamps, np.array(expected_ms) / 1000.0)

    assert timestamps[-1] == pytest.approx((IMAGE_TOTAL_MS + VIDEO_TOTAL_MS) / 1000.0)
    assert np.all(np.diff(timestamps) > 0)
    # first_frame_idx must index the timestamp array contiguously, entry by entry
    assert [combined[k]["first_frame_idx"] for k in keys] == [0, 1, 2, 3, 6]
    assert sum(combined[k]["num_frames"] for k in keys) == len(timestamps)


@pytest.mark.parametrize(
    "chunks",
    [
        # a blank entry inside a chunk: skipped, as the simulation skips it (no npy)
        [[{"modality": "blank", "file": "00003.yml", "trial": 0}, VIDEO_ITEM]],
        # a wholly empty chunk alongside a full one: chunks are concatenated before the
        # timeline is built, so it contributes nothing and is otherwise invisible
        [[], [VIDEO_ITEM]],
    ],
    ids=["blank entry", "empty chunk"],
)
def test_entries_with_nothing_to_show_contribute_nothing(tmp_path, source, chunks):
    paths = [
        _write_chunk(tmp_path / ("0_%d.json" % i), items)
        for i, items in enumerate(chunks)
    ]
    combined, timestamps = _read(_export(tmp_path, source, paths))

    assert [combined[k]["modality"] for k in sorted(combined)] == ["video", "blank"]
    assert timestamps[0] == 0.0


@pytest.mark.parametrize(
    "items, give_dsv",
    [([], True), ([IMAGE_ITEM], False)],
    ids=["nothing presented", "no dsv to resolve the source from"],
)
def test_nothing_to_export_writes_no_shard(tmp_path, source, items, give_dsv):
    """
    Two ways to have nothing to describe: a trial whose chunks present nothing, and no DSV to
    read the source directory from. Either way no shard is written -- emitting just the
    trailing blank would claim a one-frame experiment that never ran.
    """
    chunk = _write_chunk(tmp_path / "0_0.json", items)

    out = str(tmp_path / "shard")
    exporter = _exporter(tmp_path, [chunk])
    if give_dsv:
        exporter.process_batch([_DSV([_Seg(source["data_dir"])])])
    exporter.finalize()

    screen_dir = os.path.join(out, "screen")
    assert not os.path.exists(os.path.join(screen_dir, "combined_meta.json"))
    assert not os.path.exists(os.path.join(screen_dir, "timestamps.npy"))
