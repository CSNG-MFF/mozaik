"""
Tests for :mod:`mozaik.tools.experanto_chunks`.

The chunk JSON is a contract between two other modules -- ``RandomizedExperanto`` builds the
stimulus sequence from it, ``MozaikScreenExporter`` rebuilds the screen timeline from it -- and
which stimuli land in which chunk decides what each datastore ends up holding. So what is
pinned here is the emitted record shape, that the split is a true partition, and that a seed
determines it reproducibly.
"""

import json
import os

import pytest
import yaml

from mozaik.tools.experanto_chunks import (
    CHUNK_RECORD_FIELDS,
    generate_chunks,
    scan_screen_metadata,
)


def _write_dataset(root, specs):
    """Write a minimal Experanto screen dataset; specs are (name, modality, num_frames)."""
    meta_dir = os.path.join(str(root), "screen", "meta")
    os.makedirs(meta_dir)
    for name, modality, num_frames in specs:
        meta = {"modality": modality, "num_frames": num_frames, "condition_hash": name}
        if modality == "image":
            meta.update({"presentation_time": 0.5, "pre_blank_period": 0.5})
        with open(os.path.join(meta_dir, name + ".yml"), "w") as f:
            yaml.safe_dump(meta, f)
    return str(root)


@pytest.fixture
def dataset(tmp_path):
    return _write_dataset(
        tmp_path / "ds",
        [
            ("00000", "blank", 1),
            ("00001", "image", 1),
            ("00002", "image", 1),
            ("00003", "image", 1),
            ("00004", "video", 300),
            ("00005", "video", 900),
        ],
    )


def _load(path):
    with open(path, "r") as f:
        return json.load(f)


def test_chunks_carry_the_contract_fields_and_partition_the_stimuli(dataset, tmp_path):
    """
    Each record is exactly ``(modality, file, trial)`` -- what the experiment and the exporter
    read -- and a trial's chunks together present every stimulus exactly once. Blanks are not
    among them: they carry no npy and the simulation skips them.
    """
    out = str(tmp_path / "chunks")
    n_trials, n_chunks = 3, 2
    generate_chunks(dataset, out, n_trials=n_trials, n_chunks=n_chunks)

    expected = sorted(e["file"] for e in scan_screen_metadata(dataset))
    assert "00000.yml" not in expected  # the blank

    for trial in range(n_trials):
        records = []
        for chunk_idx in range(n_chunks):
            records += _load(os.path.join(out, "%d_%d.json" % (trial, chunk_idx)))

        for record in records:
            assert tuple(record.keys()) == CHUNK_RECORD_FIELDS
            assert record["trial"] == trial
            assert record["modality"] in ("image", "video")
            assert record["file"].endswith(".yml")

        assert sorted(r["file"] for r in records) == expected


def test_the_seed_reproducibly_determines_which_chunk_a_stimulus_lands_in(
    dataset, tmp_path
):
    """
    Chunk membership decides which datastore holds which stimulus, so a rerun at the same seed
    has to reproduce it exactly -- and a different seed has to actually change it, or the seed
    would not be doing anything.
    """
    first = str(tmp_path / "a")
    again = str(tmp_path / "b")
    other = str(tmp_path / "c")
    generate_chunks(dataset, first, n_trials=2, n_chunks=2, seed=7)
    generate_chunks(dataset, again, n_trials=2, n_chunks=2, seed=7)
    generate_chunks(dataset, other, n_trials=2, n_chunks=2, seed=8)

    for name in ("0_0.json", "0_1.json", "1_0.json", "1_1.json"):
        assert _load(os.path.join(first, name)) == _load(os.path.join(again, name))

    assert _load(os.path.join(first, "0_0.json")) != _load(
        os.path.join(other, "0_0.json")
    )
