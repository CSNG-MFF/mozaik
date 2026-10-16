"""Export available simulation chunks and merge completed Experanto shards."""

from dataclasses import asdict, dataclass
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from mozaik.tools.experanto_merge import CHUNK_EXPORT_META

DEFAULT_MODEL_NAME = "SelfSustainedPushPull"


@dataclass(frozen=True)
class ExperantoExportSettings:
    """Shared settings for every chunk of a run; tier mappings travel as a file path."""

    sheet_names: tuple = None
    sampling_rate: float = 1000.0
    export_blank_spikes: bool = True
    frame_duration_ms: float = 7.0
    movie_frame_duration_ms: float = 35.0
    tier_reference_path: str = None

    def __post_init__(self):
        for name in ("sampling_rate", "frame_duration_ms", "movie_frame_duration_ms"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.sheet_names is not None:
            object.__setattr__(
                self, "sheet_names", tuple(sorted(set(self.sheet_names)))
            )
        if self.tier_reference_path is not None:
            object.__setattr__(
                self,
                "tier_reference_path",
                str(Path(self.tier_reference_path).resolve()),
            )

    def to_dict(self):
        settings = asdict(self)
        if self.sheet_names is not None:
            settings["sheet_names"] = list(self.sheet_names)
        return settings

    def export_kwargs(self):
        settings = self.to_dict()
        tier_path = settings.pop("tier_reference_path")
        settings["tier_reference"] = None
        if tier_path:
            from mozaik.tools.experanto_export import load_tier_reference

            settings["tier_reference"] = load_tier_reference(tier_path)
        return settings


def resolve_datastore(datastore_prefix, trial, chunk, model_name=DEFAULT_MODEL_NAME):
    """Resolve a chunk directory by its stable prefix, without opening its datastore."""
    prefix = f"{model_name}_trial{trial}_chunk{chunk}_____"
    matches = sorted(
        glob.glob(
            os.path.join(glob.escape(str(datastore_prefix)), glob.escape(prefix) + "*")
        )
    )
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise RuntimeError(
            f"Ambiguous datastore for trial{trial}_chunk{chunk}: {matches}"
        )
    raise FileNotFoundError(f"No datastore match for trial{trial}_chunk{chunk}")


def open_datastore_dsv(path):
    """Open one saved chunk for individual export; never used by merge jobs."""
    from mozaik.storage.datastore import PickledDataStore
    from mozaik.storage.queries import param_filter_query
    from parameters import ParameterSet

    data_store = PickledDataStore(
        load=True,
        parameters=ParameterSet({"root_directory": str(path), "store_stimuli": False}),
        replace=False,
    )
    return data_store, param_filter_query(data_store)


def export_dsvs_to_experanto(
    dsv_list,
    output_dir,
    *,
    trial_id,
    chunk_paths,
    sheet_names=None,
    sampling_rate=1000.0,
    export_spikes=True,
    export_screen=True,
    tier_reference=None,
    frame_duration_ms=7.0,
    movie_frame_duration_ms=35.0,
    export_blank_spikes=True,
):
    """Export one chunk's already available DSV. Combine chunks through the merger."""
    dsvs = dsv_list if isinstance(dsv_list, (list, tuple)) else [dsv_list]
    if len(dsvs) != 1 or (export_screen and len(chunk_paths) != 1):
        raise ValueError("Export one chunk at a time, then merge its Experanto shards")
    from mozaik.tools.experanto_export import MozaikScreenExporter, MozaikTrialExporter

    if export_spikes:
        spikes = MozaikTrialExporter(
            os.path.join(output_dir, "responses"),
            trial_id=trial_id,
            sampling_rate=sampling_rate,
            sheet_names=sheet_names,
            export_blank_spikes=export_blank_spikes,
        )
        spikes.process_batch(dsvs)
        spikes.finalize()
    if export_screen:
        screen = MozaikScreenExporter(
            output_dir=output_dir,
            chunk_paths=chunk_paths,
            frame_duration_ms=frame_duration_ms,
            movie_frame_duration_ms=movie_frame_duration_ms,
            tier_reference=tier_reference,
        )
        screen.process_batch(dsvs)
        screen.finalize()


def export_experanto_chunk(
    data_store,
    *,
    trial_id,
    chunk_id,
    chunk_path,
    settings=None,
    output_dir=None,
    provenance=None,
):
    """Export one finished chunk from its available datastore; call on MPI root.

    The caller saves the datastore before exporting. Output defaults to
    <datastore>/experanto. Files and a completion record are published together
    after validation; existing output directories are refused.
    """
    from mozaik.storage.queries import param_filter_query
    from mozaik.tools.experanto_merge import _load_shard, _merged_provenance

    if trial_id < 0 or chunk_id < 0:
        raise ValueError("Trial and chunk indices must be nonnegative")
    settings = settings if settings is not None else ExperantoExportSettings()
    if data_store.block.annotations.get("simulation_log", {}).get("explosion_detected"):
        raise ValueError("Cannot export a simulation that stopped after an explosion")
    chunk_bytes = Path(chunk_path).read_bytes()
    items = json.loads(chunk_bytes)
    if not items or any(item["trial"] != trial_id for item in items):
        raise ValueError("Chunk list must be nonempty and match the requested trial")
    output = Path(
        output_dir or Path(data_store.parameters.root_directory) / "experanto"
    )
    if output.exists():
        raise FileExistsError(f"Chunk export already exists: {output}")
    recorded_settings = settings.to_dict()
    tier_path = settings.tier_reference_path
    recorded_settings["tier_reference_sha256"] = (
        hashlib.sha256(Path(tier_path).read_bytes()).hexdigest() if tier_path else None
    )
    completion = dict(
        trial_id=trial_id,
        chunk_id=chunk_id,
        chunk_sha256=hashlib.sha256(chunk_bytes).hexdigest(),
        export_settings=recorded_settings,
        provenance=provenance,
    )
    root_meta = {"data_key": f"trial{trial_id}_chunk{chunk_id}"}
    if provenance is not None:
        root_meta = _merged_provenance(
            provenance,
            trial_id,
            chunk_id,
            1,
            root_meta["data_key"],
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.export-", dir=output.parent
    ) as temporary:
        staging = Path(temporary)
        export_dsvs_to_experanto(
            [param_filter_query(data_store)],
            str(staging),
            trial_id=trial_id,
            chunk_paths=[chunk_path],
            **settings.export_kwargs(),
        )
        shard = _load_shard(staging)
        expected = [
            item["file"].replace(".yml", ".npy")
            for item in items
            if item["modality"] != "blank"
        ]
        recorded = [
            label for label in shard.response_meta["stimuli_order"] if label != "blank"
        ]
        if recorded != expected:
            raise ValueError("Recorded presentations do not match the chunk list")
        for key, entry in shard.screen_entries:
            if (
                entry["modality"] != "blank"
                and not (staging / "screen/data" / f"{key}.npy").is_file()
            ):
                raise ValueError(f"Missing screen data for entry {key}")
        (staging / "meta.json").write_text(json.dumps(root_meta, indent=4))
        (staging / CHUNK_EXPORT_META).write_text(json.dumps(completion, indent=4))
        if output.exists():
            raise FileExistsError(f"Chunk export already exists: {output}")
        staging.rename(output)
    return output


def merge_experanto_trial(
    *,
    trial_id,
    output_dir,
    shard_paths=None,
    datastore_prefix=None,
    model_name=DEFAULT_MODEL_NAME,
    chunk_start=0,
    chunk_end=None,
    provenance=None,
):
    """Merge completed chunk exports for one trial; no datastore loading occurs."""
    from mozaik.tools.experanto_merge import merge_experanto_shards

    if (shard_paths is None) == (datastore_prefix is None):
        raise ValueError("Provide shard_paths or datastore_prefix, exclusively")
    if shard_paths is None:
        if chunk_end is None or chunk_end <= chunk_start:
            raise ValueError("Provide a nonempty chunk range")
        shard_paths = [
            Path(resolve_datastore(datastore_prefix, trial_id, chunk, model_name))
            / "experanto"
            for chunk in range(chunk_start, chunk_end)
        ]
    else:
        shard_paths = list(shard_paths)
        if chunk_end is not None and chunk_end - chunk_start != len(shard_paths):
            raise ValueError("Chunk range does not match the supplied shard paths")
    for path in shard_paths:
        if not (Path(path) / CHUNK_EXPORT_META).is_file():
            raise ValueError(f"{path}: no completed simulation-side chunk export")
    return merge_experanto_shards(
        shard_paths,
        output_dir,
        trial_id=trial_id,
        chunk_start=chunk_start,
        provenance=provenance,
    )
