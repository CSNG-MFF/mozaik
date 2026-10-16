"""Merge per-chunk Experanto exports without loading Mozaik datastores."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np
import yaml


CHUNK_EXPORT_META = "chunk_export.json"


def _read_yaml(path):
    with path.open() as stream:
        return yaml.load(stream, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))


def _write_yaml(path, metadata):
    with path.open("w") as stream:
        yaml.safe_dump(metadata, stream)


@dataclass
class _Shard:
    """Metadata and read-only array views belonging to one chunk."""

    path: Path
    response_meta: dict
    spikes: np.ndarray
    spike_indices: np.ndarray
    screen_meta: dict
    screen_entries: list
    timestamps: np.ndarray
    export_meta: dict = None


def _load_shard(path):
    response = _read_yaml(path / "responses/meta.yml")
    spikes = np.load(path / "responses/spikes.npy", mmap_mode="r")
    indices = np.asarray(response["spike_indices"])
    if (
        spikes.ndim != 1
        or spikes.dtype != np.float64
        or indices.ndim != 1
        or indices.dtype.kind not in "iu"
        or len(indices) != response["n_signals"] + 1
        or indices[0] != 0
        or indices[-1] != len(spikes)
        or np.any(indices[1:] < indices[:-1])
    ):
        raise ValueError(f"{path}: invalid spike array or unit boundaries")

    screen_meta = _read_yaml(path / "screen/meta.yml")
    with (path / "screen/combined_meta.json").open() as stream:
        combined = json.load(stream)
    entries = sorted(combined.items(), key=lambda item: int(item[0]))
    timestamps = np.load(path / "screen/timestamps.npy", mmap_mode="r")
    if timestamps.ndim != 1 or timestamps.dtype != np.float64 or not len(timestamps):
        raise ValueError(f"{path}: invalid screen timestamps")

    # Entry frame ranges must cover the timestamp array without gaps or overlap.
    frame = 0
    for _, entry in entries:
        if entry["first_frame_idx"] != frame or entry["num_frames"] < 1:
            raise ValueError(f"{path}: invalid screen frame range")
        frame += entry["num_frames"]
    if not entries or frame != len(timestamps):
        raise ValueError(f"{path}: screen entries do not cover the timestamps")

    # The last blank is a synthetic endpoint, distinct from recorded stimulus blanks.
    terminal = entries[-1][1]
    duration = response["end_time"]
    if (
        response["start_time"] != 0
        or not math.isfinite(duration)
        or duration < 0
        or timestamps[0] != 0
        or terminal["modality"] != "blank"
        or terminal["num_frames"] != 1
        or not math.isclose(timestamps[-1], duration, rel_tol=1e-12, abs_tol=1e-9)
        or len(response["stimuli_order"]) != len(entries) - 1
    ):
        raise ValueError(f"{path}: screen and response timelines do not agree")
    export_meta = None
    if (path / CHUNK_EXPORT_META).is_file():
        with (path / CHUNK_EXPORT_META).open() as stream:
            export_meta = json.load(stream)
    return _Shard(
        path, response, spikes, indices, screen_meta, entries, timestamps, export_meta
    )


def _merge_responses(shards, offsets, duration, output):
    # Sum each unit's spike counts, then calculate its slice in the final array.
    unit_counts = np.zeros(shards[0].response_meta["n_signals"], dtype=np.int64)
    for shard in shards:
        unit_counts += np.diff(shard.spike_indices)
    unit_indices = np.concatenate(([0], np.cumsum(unit_counts)))
    spikes = np.lib.format.open_memmap(
        output / "spikes.npy",
        mode="w+",
        dtype=np.float64,
        shape=(int(unit_indices[-1]),),
    )

    # Copy all chunks for one unit before moving to the next, writing sequentially.
    # out= avoids allocating an intermediate array for the shifted spike times.
    for unit in range(len(unit_counts)):
        cursor = unit_indices[unit]
        for shard, offset in zip(shards, offsets):
            start, stop = shard.spike_indices[unit : unit + 2]
            count = stop - start
            np.add(
                shard.spikes[start:stop], offset, out=spikes[cursor : cursor + count]
            )
            cursor += count
    spikes.flush()

    metadata = dict(shards[0].response_meta)
    metadata["spike_indices"] = unit_indices.tolist()
    metadata["end_time"] = duration
    metadata["stimuli_order"] = [
        stimulus
        for shard in shards
        for stimulus in shard.response_meta["stimuli_order"]
    ]
    _write_yaml(output / "meta.yml", metadata)


def _merge_screen(shards, offsets, output):
    (output / "meta").mkdir()
    (output / "data").mkdir()
    _write_yaml(output / "meta.yml", shards[0].screen_meta)
    # Drop one terminal timestamp per chunk, retaining the final chunk's endpoint.
    frame_count = sum(len(shard.timestamps) - 1 for shard in shards) + 1
    timestamps = np.lib.format.open_memmap(
        output / "timestamps.npy", mode="w+", dtype=np.float64, shape=(frame_count,)
    )
    combined = {}
    frame_offset = 0
    for index, (shard, offset) in enumerate(zip(shards, offsets)):
        is_last = index == len(shards) - 1
        chunk_times = shard.timestamps if is_last else shard.timestamps[:-1]
        entries = shard.screen_entries if is_last else shard.screen_entries[:-1]
        np.add(
            chunk_times,
            offset,
            out=timestamps[frame_offset : frame_offset + len(chunk_times)],
        )
        for source_key, entry in entries:
            output_key = f"{len(combined):05d}"
            metadata = dict(entry)
            metadata["first_frame_idx"] += frame_offset
            combined[output_key] = metadata
            _write_yaml(output / "meta" / f"{output_key}.yml", metadata)
            if entry["modality"] != "blank":
                shutil.copy2(
                    shard.path / "screen/data" / f"{source_key}.npy",
                    output / "data" / f"{output_key}.npy",
                )
        frame_offset += len(chunk_times)
    timestamps.flush()
    with (output / "combined_meta.json").open("w") as stream:
        json.dump(combined, stream)


def _merged_provenance(provenance, trial_id, chunk_start, chunk_count, output_name):
    chunk_end = chunk_start + chunk_count
    seeds = provenance["simulation_seeds"][str(trial_id)]
    if len(seeds) < chunk_end or provenance["settings"]["n_chunks"] < chunk_end:
        raise ValueError("Provenance does not cover the merged chunks")
    return {
        "data_key": output_name,
        "mozaik_run": {
            "trial": trial_id,
            "settings": dict(
                provenance["settings"], chunks=f"{chunk_start}-{chunk_end - 1}"
            ),
            "environment_variables": provenance["environment_variables"],
            "simulation_seeds": list(seeds[chunk_start:chunk_end]),
        },
    }


def _chunk_export_provenance(shards, trial_id, chunk_start, provenance):
    """Check simulation-side completion records and recover their common run record."""
    records = [shard.export_meta for shard in shards]
    if all(record is None for record in records):
        return provenance  # Shards without completion records use caller-supplied provenance.
    if any(record is None for record in records):
        raise ValueError(
            "Every input must have a chunk completion record when any does"
        )
    reference = records[0]
    for chunk, record in enumerate(records, chunk_start):
        if record["trial_id"] != trial_id or record["chunk_id"] != chunk:
            raise ValueError(
                f"Expected completed export for trial {trial_id}, chunk {chunk}"
            )
        if record["export_settings"] != reference["export_settings"]:
            raise ValueError("Chunk export settings differ")
        if record["provenance"] != reference["provenance"]:
            raise ValueError("Chunks were exported with different run provenance")
    recorded = reference["provenance"]
    if provenance is not None and recorded is not None and provenance != recorded:
        raise ValueError("Requested provenance differs from the chunk exports")
    return recorded if provenance is None else provenance


def merge_experanto_shards(
    shard_paths, output_dir, *, trial_id=None, chunk_start=0, provenance=None
):
    """Merge one trial's chunks, supplied in numerical presentation order.

    Inputs must be valid exports with the same unit order and export settings.
    Recorded metadata and array boundaries are checked. Simulation-side completion
    records additionally verify chunk order, export settings and run provenance.
    For shards without completion records, policy and order are caller-supplied.
    Unit identities are not stored in the format and cannot be verified.
    Times are shifted by the cumulative duration of preceding chunks. The screen
    covers only the supplied chunks, keeping one synthetic terminal blank.

    The output directory must be new. Files are built in a temporary sibling and
    published together. Spike arrays are memory mapped and copied by unit, so no
    full in-memory spike array is allocated.

    ``trial_id`` optionally checks the trial inferred from response metadata.
    ``provenance`` optionally supplies the launcher's run_config.json as a dict;
    ``chunk_start`` identifies the first supplied chunk for that run record.
    Ordering is the caller's responsibility and does not depend on path names.
    Returns the output Path.
    """
    paths = [Path(path).resolve() for path in shard_paths]
    output = Path(output_dir).resolve()
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Supply at least one shard, without duplicate paths")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if any(path in output.parents or output in path.parents for path in paths):
        raise ValueError("Output must be separate from input shards")
    if chunk_start < 0:
        raise ValueError("chunk_start must be nonnegative")

    shards = [_load_shard(path) for path in paths]
    # These three fields vary with chunk duration and contents; everything else
    # (including trial, sheet layout and sampling rate) must agree across chunks.
    chunk_fields = {"end_time", "spike_indices", "stimuli_order"}
    reference = {
        k: v for k, v in shards[0].response_meta.items() if k not in chunk_fields
    }
    for shard in shards[1:]:
        metadata = {
            k: v for k, v in shard.response_meta.items() if k not in chunk_fields
        }
        if metadata != reference or shard.screen_meta != shards[0].screen_meta:
            raise ValueError(
                f"{shard.path}: incompatible trial, sheet layout or export metadata"
            )
    if trial_id is not None and reference["trial_id"] != trial_id:
        raise ValueError("Requested trial_id differs from shard metadata")
    trial_id = reference["trial_id"]
    provenance = _chunk_export_provenance(shards, trial_id, chunk_start, provenance)

    # Use the same offsets for spikes and screen frames, including blank durations.
    offsets = []
    duration = 0.0
    for shard in shards:
        offsets.append(duration)
        duration += shard.response_meta["end_time"]
    root_meta = None
    if provenance is not None:
        root_meta = _merged_provenance(
            provenance, trial_id, chunk_start, len(shards), output.name
        )
    if shards[0].export_meta is not None:
        root_meta = root_meta or {"data_key": output.name}
        root_meta["mozaik_export"] = shards[0].export_meta["export_settings"]

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.merge-", dir=output.parent
    ) as temporary:
        staging = Path(temporary)
        (staging / "responses").mkdir()
        (staging / "screen").mkdir()
        _merge_responses(shards, offsets, duration, staging / "responses")
        _merge_screen(shards, offsets, staging / "screen")
        if root_meta is not None:
            with (staging / "meta.json").open("w") as stream:
                json.dump(root_meta, stream, indent=4)
        if output.exists():
            raise FileExistsError(f"Output already exists: {output}")
        staging.rename(output)
    return output
