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
    CHUNK_SETTINGS_FILE,
    chunk_list_names,
    generate_chunks,
    read_chunk_settings,
    scan_screen_metadata,
    verify_chunk_lists,
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


def test_generating_records_the_settings_beside_the_lists(
    dataset, tmp_path, monkeypatch
):
    """
    The settings a chunk set was generated from cannot be read back out of the lists, so they
    are recorded beside them -- with the dataset as an absolute path, so that the record still
    names it when read from another directory.
    """
    monkeypatch.chdir(os.path.dirname(dataset))
    out = str(tmp_path / "chunks")
    generate_chunks(os.path.basename(dataset), out, n_trials=2, n_chunks=3, seed=5)

    assert read_chunk_settings(out) == {
        "data_root": dataset,
        "n_trials": 2,
        "n_chunks": 3,
        "chunk_seed": 5,
    }
    assert read_chunk_settings(str(tmp_path)) is None


def test_the_settings_record_is_not_taken_for_a_chunk_list(dataset, tmp_path):
    out = str(tmp_path / "chunks")
    generate_chunks(dataset, out, n_trials=1, n_chunks=2)

    assert CHUNK_SETTINGS_FILE in os.listdir(out)
    assert chunk_list_names(out) == ["0_0.json", "0_1.json"]
    assert chunk_list_names(str(tmp_path / "absent")) == []


def test_verifying_accepts_the_lists_of_the_same_settings(dataset, tmp_path):
    out = str(tmp_path / "chunks")
    generate_chunks(dataset, out, n_trials=2, n_chunks=2, seed=7)
    os.remove(os.path.join(out, CHUNK_SETTINGS_FILE))

    verify_chunk_lists(out, dataset, n_trials=2, n_chunks=2, seed=7)


def test_verifying_rejects_lists_of_another_seed(dataset, tmp_path):
    """
    The point of verifying is that a wrong seed can never end up recorded, so lists generated
    at a different seed have to fail, naming the list that gives it away.
    """
    out = str(tmp_path / "chunks")
    generate_chunks(dataset, out, n_trials=2, n_chunks=2, seed=7)

    with pytest.raises(ValueError, match="0_0.json differs"):
        verify_chunk_lists(out, dataset, n_trials=2, n_chunks=2, seed=8)


@pytest.mark.parametrize(
    "n_trials, n_chunks, problem",
    [(3, 2, "missing 2_0.json, 2_1.json"), (1, 2, "unexpected 1_0.json, 1_1.json")],
)
def test_verifying_rejects_a_different_set_of_lists(
    dataset, tmp_path, n_trials, n_chunks, problem
):
    out = str(tmp_path / "chunks")
    generate_chunks(dataset, out, n_trials=2, n_chunks=2, seed=7)

    with pytest.raises(ValueError, match=problem):
        verify_chunk_lists(out, dataset, n_trials=n_trials, n_chunks=n_chunks, seed=7)
