"""Essential checks for spike placement, screen boundaries and run metadata."""

import json

import numpy as np
import pytest
import yaml

from mozaik.tools.experanto_merge import merge_experanto_shards


def _yaml(path, value):
    path.write_text(yaml.safe_dump(value))


def _read_yaml(path):
    return yaml.load(
        path.read_text(), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    )


@pytest.fixture
def shards(tmp_path):
    # An image with actual pre/post blanks, followed by a three-frame video.
    # Three units in two sheets: the middle unit is silent in the first chunk,
    # and the last unit is silent in the second chunk.
    definitions = [
        (
            0.4,
            ["blank", "image.npy", "blank"],
            [[0.0, 0.05, 0.1, 0.299, 0.35], [], [0.2, 0.399]],
            [0.0, 0.1, 0.3, 0.4],
            [("blank", 0, 1), ("image", 1, 1), ("blank", 2, 1), ("blank", 3, 1)],
        ),
        (
            0.105,
            ["video.npy"],
            [[0.0, 0.104], [0.03, 0.09], []],
            [0.0, 0.035, 0.07, 0.105],
            [("video", 0, 3), ("blank", 3, 1)],
        ),
    ]
    paths = []
    for chunk, (duration, labels, units, times, screen) in enumerate(definitions):
        path = tmp_path / f"trial0_chunk{chunk}"
        (path / "responses").mkdir(parents=True)
        (path / "screen/meta").mkdir(parents=True)
        (path / "screen/data").mkdir()
        indices = [0] + np.cumsum([len(unit) for unit in units]).tolist()
        np.save(path / "responses/spikes.npy", np.concatenate(units))
        _yaml(
            path / "responses/meta.yml",
            dict(
                modality="spikes",
                n_signals=3,
                n_signals_layerwise=[2, 1],
                sheets=["X_ON", "V1_Exc_L4"],
                sheet_unit_indices=[0, 2, 3],
                start_time=0.0,
                end_time=duration,
                trial_id=0,
                sampling_rate=1000.0,
                smoothing=None,
                spike_indices=indices,
                stimuli_order=labels,
            ),
        )
        _yaml(path / "screen/meta.yml", {"modality": "screen"})
        np.save(path / "screen/timestamps.npy", np.array(times))
        combined = {}
        for index, (modality, frame, frames) in enumerate(screen):
            key = f"{index:05d}"
            entry = dict(
                modality=modality,
                first_frame_idx=frame,
                num_frames=frames,
                image_size=[2, 2],
            )
            if modality == "blank":
                entry["interleave_value"] = 128.0
            else:
                entry.update(condition_hash=f"condition-{chunk}", tier="train")
                pixels = np.full((frames, 2, 2), chunk + 1, dtype=np.uint8)
                np.save(path / "screen/data" / f"{key}.npy", pixels)
            _yaml(path / "screen/meta" / f"{key}.yml", entry)
            combined[key] = entry
        (path / "screen/combined_meta.json").write_text(json.dumps(combined))
        paths.append(path)
    return paths


def test_unit_major_merge_and_screen_boundaries(shards, tmp_path):
    output = merge_experanto_shards(shards, tmp_path / "merged")
    spikes = np.load(output / "responses/spikes.npy")
    # Expected values specified independently, including spikes during real blanks.
    np.testing.assert_allclose(
        spikes,
        [0, 0.05, 0.1, 0.299, 0.35, 0.4, 0.504, 0.43, 0.49, 0.2, 0.399],
        rtol=0,
        atol=1e-15,
    )
    meta = _read_yaml(output / "responses/meta.yml")
    assert meta["spike_indices"] == [0, 7, 9, 11]
    assert meta["sheet_unit_indices"] == [0, 2, 3]
    assert meta["stimuli_order"] == ["blank", "image.npy", "blank", "video.npy"]
    assert meta["end_time"] == pytest.approx(0.505)
    np.testing.assert_allclose(
        np.load(output / "screen/timestamps.npy"),
        [0, 0.1, 0.3, 0.4, 0.435, 0.47, 0.505],
        rtol=0,
        atol=1e-15,
    )
    combined = json.loads((output / "screen/combined_meta.json").read_text())
    assert [entry["modality"] for entry in combined.values()] == [
        "blank",
        "image",
        "blank",
        "video",
        "blank",
    ]
    assert [entry["first_frame_idx"] for entry in combined.values()] == [0, 1, 2, 3, 6]
    for key, entry in combined.items():
        assert _read_yaml(output / "screen/meta" / f"{key}.yml") == entry
    assert (output / "screen/data/00001.npy").read_bytes() == (
        shards[0] / "screen/data/00001.npy"
    ).read_bytes()
    assert (output / "screen/data/00003.npy").read_bytes() == (
        shards[1] / "screen/data/00000.npy"
    ).read_bytes()
    assert not (output / "meta.json").exists()


def test_single_chunk_is_unchanged(shards, tmp_path):
    output = merge_experanto_shards(shards[:1], tmp_path / "merged")
    assert _read_yaml(output / "responses/meta.yml") == _read_yaml(
        shards[0] / "responses/meta.yml"
    )
    np.testing.assert_array_equal(
        np.load(output / "responses/spikes.npy"),
        np.load(shards[0] / "responses/spikes.npy"),
    )
    np.testing.assert_array_equal(
        np.load(output / "screen/timestamps.npy"),
        np.load(shards[0] / "screen/timestamps.npy"),
    )
    assert json.loads((output / "screen/combined_meta.json").read_text()) == json.loads(
        (shards[0] / "screen/combined_meta.json").read_text()
    )


def test_all_units_silent(shards, tmp_path):
    for shard in shards:
        meta = _read_yaml(shard / "responses/meta.yml")
        meta["spike_indices"] = [0, 0, 0, 0]
        _yaml(shard / "responses/meta.yml", meta)
        np.save(shard / "responses/spikes.npy", np.array([], dtype=np.float64))
    output = merge_experanto_shards(shards, tmp_path / "merged")
    assert np.load(output / "responses/spikes.npy").size == 0
    assert _read_yaml(output / "responses/meta.yml")["n_signals"] == 3


@pytest.mark.parametrize(
    "key,value",
    [
        ("trial_id", 1),
        ("sheets", ["V1_Exc_L4", "X_ON"]),
    ],
)
def test_incompatible_trial_or_unit_layout_is_rejected(shards, tmp_path, key, value):
    meta = _read_yaml(shards[1] / "responses/meta.yml")
    meta[key] = value
    _yaml(shards[1] / "responses/meta.yml", meta)
    with pytest.raises(ValueError, match="incompatible"):
        merge_experanto_shards(shards, tmp_path / "merged")


def test_incorrect_unit_boundaries_are_rejected(shards, tmp_path):
    meta = _read_yaml(shards[0] / "responses/meta.yml")
    meta["spike_indices"][-1] += 1
    _yaml(shards[0] / "responses/meta.yml", meta)
    with pytest.raises(ValueError, match="unit boundaries"):
        merge_experanto_shards(shards, tmp_path / "merged")


def test_provenance_and_nonzero_start(shards, tmp_path):
    provenance = dict(
        settings=dict(n_chunks=10, n_trials=1),
        environment_variables={"OMP_NUM_THREADS": 1},
        simulation_seeds={"0": list(range(1, 11))},
    )
    output = merge_experanto_shards(
        shards, tmp_path / "trial0", chunk_start=3, provenance=provenance
    )
    meta = json.loads((output / "meta.json").read_text())
    assert meta["data_key"] == "trial0"
    assert meta["mozaik_run"]["settings"]["chunks"] == "3-4"
    assert meta["mozaik_run"]["simulation_seeds"] == [4, 5]
    assert provenance["settings"] == dict(n_chunks=10, n_trials=1)
