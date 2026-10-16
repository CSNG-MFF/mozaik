"""Direct chunk export and completed-shard merge integration."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from mozaik.meta_workflow import experanto_export as workflow
from mozaik.tools.experanto_merge import merge_experanto_shards


def _read_yaml(path):
    return yaml.load(
        path.read_text(), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    )


class Segment:
    def __init__(self, stimulus, sheet, times):
        self.annotations = {"stimulus": stimulus, "sheet_name": sheet}
        self.times = [np.asarray(times)]

    def get_spiketrains(self):
        return self.times

    def release(self):
        pass


class Store:
    def __init__(self, path, segments):
        self.parameters = SimpleNamespace(root_directory=str(path))
        self.block = SimpleNamespace(
            annotations={"simulation_log": {"explosion_detected": False}}
        )
        self.segments = segments
        self.saved = False

    def save(self):
        self.saved = True

    def get_segments(self):
        assert self.saved  # Export must run after save.
        return self.segments

    def get_model_parameters(self):
        return {"null_stimulus_period": 0}


@pytest.mark.parametrize("blank_spikes", [True, False])
def test_chunk_exports_and_merge_match_single_chunk(
    tmp_path, monkeypatch, blank_spikes
):
    from mozaik.storage import queries

    monkeypatch.setattr(queries, "param_filter_query", lambda store: store)
    monkeypatch.setattr(
        workflow, "open_datastore_dsv", lambda *args: pytest.fail("Datastore reload")
    )
    source = tmp_path / "screen_source"
    (source / "screen/meta").mkdir(parents=True)
    (source / "screen/data").mkdir()
    image = dict(
        modality="image",
        num_frames=1,
        image_size=[2, 2],
        pre_blank_period=0.014,
        presentation_time=0.028,
        post_blank_period=0.007,
        condition_hash="image",
        tier="train",
    )
    video = dict(
        modality="video",
        num_frames=3,
        image_size=[2, 2],
        condition_hash="video",
        tier="test",
    )
    for name, meta in [("image", image), ("video", video)]:
        (source / "screen/meta" / f"{name}.yml").write_text(yaml.safe_dump(meta))
        np.save(
            source / "screen/data" / f"{name}.npy", np.zeros((meta["num_frames"], 2, 2))
        )
    tier_reference = tmp_path / "tiers.json"
    tier_reference.write_text(
        json.dumps({"0": {"condition_hash": "image", "tier": "validation"}})
    )
    settings = workflow.ExperantoExportSettings(
        sheet_names=["V1_Exc_L4", "X_ON"],
        sampling_rate=2000.0,
        export_blank_spikes=blank_spikes,
        tier_reference_path=str(tier_reference),
    )
    stimuli = [
        dict(name="InternalStimulus", trial=0, duration=14),
        dict(
            name="PixelMovieExperanto",
            trial=0,
            duration=28,
            movie_name="image.npy",
            movie_path=str(source / "screen/data"),
        ),
        dict(name="InternalStimulus", trial=0, duration=7),
        dict(
            name="PixelMovieExperanto",
            trial=0,
            duration=105,
            movie_name="video.npy",
            movie_path=str(source / "screen/data"),
        ),
    ]
    segments = [
        [
            Segment(stim, sheet, [0, stim["duration"] - 1])
            for sheet in ["X_ON", "V1_Exc_L4"]
        ]
        for stim in stimuli
    ]
    records = [
        dict(modality="image", file="image.yml", trial=0),
        dict(modality="video", file="video.yml", trial=0),
    ]
    provenance = dict(
        settings=dict(n_trials=1, n_chunks=2),
        environment_variables={},
        simulation_seeds={"0": [1, 2]},
    )
    paths = []
    datastore_root = tmp_path / "results[run]"
    for chunk, presentations in enumerate([segments[:3], segments[3:]]):
        chunk_path = tmp_path / f"0_{chunk}.json"
        chunk_path.write_text(json.dumps([records[chunk]]))
        datastore = (
            datastore_root
            / f"SelfSustainedPushPull_trial0_chunk{chunk}_____seed:{chunk + 1}"
        )
        store = Store(datastore, sum(presentations, []))
        store.save()
        paths.append(
            workflow.export_experanto_chunk(
                store,
                trial_id=0,
                chunk_id=chunk,
                chunk_path=chunk_path,
                settings=settings,
                provenance=provenance,
            )
        )
        record = json.loads((paths[-1] / workflow.CHUNK_EXPORT_META).read_text())
        assert record["export_settings"]["export_blank_spikes"] is blank_spikes
        assert record["export_settings"]["tier_reference_sha256"]
        assert (paths[-1] / "meta.json").is_file()
    assert not list(datastore_root.rglob(".experanto.export-*"))

    merge_output = workflow.merge_experanto_trial(
        trial_id=0,
        datastore_prefix=datastore_root,
        chunk_end=2,
        output_dir=tmp_path / "merged",
    )

    # Export the same presentations as one chunk and pass it through the merger.
    all_path = tmp_path / "all.json"
    all_path.write_text(json.dumps(records))
    store = Store(tmp_path / "single_datastore", sum(segments, []))
    store.save()
    single_export = workflow.export_experanto_chunk(
        store,
        trial_id=0,
        chunk_id=0,
        chunk_path=all_path,
        settings=settings,
        output_dir=tmp_path / "single_export",
    )
    single_output = merge_experanto_shards([single_export], tmp_path / "single_merged")
    expected, actual = _read_yaml(single_output / "responses/meta.yml"), _read_yaml(
        merge_output / "responses/meta.yml"
    )
    assert actual == expected
    for name in ["responses/spikes.npy", "screen/timestamps.npy"]:
        np.testing.assert_allclose(
            np.load(merge_output / name),
            np.load(single_output / name),
            rtol=0,
            atol=1e-12,
        )
    assert json.loads(
        (merge_output / "screen/combined_meta.json").read_text()
    ) == json.loads((single_output / "screen/combined_meta.json").read_text())
    for file in (single_output / "screen/data").glob("*.npy"):
        assert (
            file.read_bytes() == (merge_output / "screen/data" / file.name).read_bytes()
        )
    merged_meta = json.loads((merge_output / "meta.json").read_text())["mozaik_run"]
    assert merged_meta["settings"]["chunks"] == "0-1" and merged_meta[
        "simulation_seeds"
    ] == [1, 2]
    assert _read_yaml(paths[0] / "screen/meta/00001.yml")["tier"] == "validation"

    # Settings and order must remain verifiable even when spike metadata agrees.
    completion_path = paths[1] / workflow.CHUNK_EXPORT_META
    completion = json.loads(completion_path.read_text())
    completion["export_settings"]["export_blank_spikes"] = not blank_spikes
    completion_path.write_text(json.dumps(completion))
    with pytest.raises(ValueError, match="settings differ"):
        workflow.merge_experanto_trial(
            trial_id=0, shard_paths=paths, output_dir=tmp_path / "bad"
        )


def test_failed_export_is_not_published(tmp_path, monkeypatch):
    from mozaik.storage import queries

    monkeypatch.setattr(queries, "param_filter_query", lambda store: store)
    chunk_path = tmp_path / "chunk.json"
    chunk_path.write_text(
        json.dumps([dict(modality="image", file="image.yml", trial=0)])
    )
    output = tmp_path / "export"

    def fail(*args, **kwargs):
        raise RuntimeError("export failed")

    monkeypatch.setattr(workflow, "export_dsvs_to_experanto", fail)
    with pytest.raises(RuntimeError, match="export failed"):
        workflow.export_experanto_chunk(
            Store(tmp_path, []),
            trial_id=0,
            chunk_id=0,
            chunk_path=chunk_path,
            output_dir=output,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".export.export-*"))


def test_chunk_export_requires_single_chunk(tmp_path):
    with pytest.raises(ValueError, match="one chunk"):
        workflow.export_dsvs_to_experanto(
            [object(), object()], str(tmp_path), trial_id=0, chunk_paths=[]
        )


def test_trial_merge_requires_completed_exports(tmp_path):
    with pytest.raises(ValueError, match="no completed"):
        workflow.merge_experanto_trial(
            trial_id=0,
            shard_paths=[tmp_path / "unfinished"],
            output_dir=tmp_path / "merged",
        )
