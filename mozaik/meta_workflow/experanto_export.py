"""Simulation-side chunk export and dependent Experanto merge jobs.

Model entry points pass an ExperantoChunkExport to run_workflow, or launchers
include its job_environment() in the simulation job's environment. Merge jobs
read completed Experanto exports only.
"""

from dataclasses import asdict, dataclass, field
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from mozaik.tools.experanto_merge import CHUNK_EXPORT_META

EXPORT_JOB_ENV = "MOZAIK_EXPERANTO_CHUNK_EXPORT"
MERGE_JOB_ENV = "MOZAIK_EXPERANTO_MERGE"
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


@dataclass(frozen=True)
class ExperantoChunkExport:
    """One simulation job's export request, consumed by run_workflow on every rank."""

    trial_id: int
    chunk_id: int
    chunk_path: str
    output_dir: str = None
    settings: ExperantoExportSettings = field(default_factory=ExperantoExportSettings)
    provenance: dict = None

    def __post_init__(self):
        if self.trial_id < 0 or self.chunk_id < 0:
            raise ValueError("Trial and chunk indices must be nonnegative")
        object.__setattr__(self, "chunk_path", str(Path(self.chunk_path).resolve()))
        if self.output_dir is not None:
            object.__setattr__(self, "output_dir", str(Path(self.output_dir).resolve()))

    def job_environment(self):
        request = asdict(self)
        request["settings"] = self.settings.to_dict()
        return {EXPORT_JOB_ENV: json.dumps(request)}

    @classmethod
    def from_environment(cls):
        value = os.environ.get(EXPORT_JOB_ENV)
        if not value:
            return None
        request = json.loads(value)
        request["settings"] = ExperantoExportSettings(**request["settings"])
        return cls(**request)


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


def export_experanto_chunk(data_store, request):
    """Export the finished datastore on MPI root, publishing a completion record last.

    The default output is <datastore>/experanto; request.output_dir can select a
    separate writable directory. This reuses the datastore and never reopens it.
    """
    from mozaik.storage.queries import param_filter_query
    from mozaik.tools.experanto_merge import _load_shard, _merged_provenance

    if data_store.block.annotations.get("simulation_log", {}).get("explosion_detected"):
        raise ValueError("Cannot export a simulation that stopped after an explosion")
    chunk_bytes = Path(request.chunk_path).read_bytes()
    items = json.loads(chunk_bytes)
    if not items or any(item["trial"] != request.trial_id for item in items):
        raise ValueError("Chunk list must be nonempty and match the requested trial")
    output = Path(
        request.output_dir or Path(data_store.parameters.root_directory) / "experanto"
    )
    if output.exists():
        raise FileExistsError(f"Chunk export already exists: {output}")
    settings = request.settings.to_dict()
    tier_path = request.settings.tier_reference_path
    settings["tier_reference_sha256"] = (
        hashlib.sha256(Path(tier_path).read_bytes()).hexdigest() if tier_path else None
    )
    completion = dict(
        trial_id=request.trial_id,
        chunk_id=request.chunk_id,
        chunk_sha256=hashlib.sha256(chunk_bytes).hexdigest(),
        export_settings=settings,
        provenance=request.provenance,
    )
    root_meta = {"data_key": f"trial{request.trial_id}_chunk{request.chunk_id}"}
    if request.provenance is not None:
        root_meta = _merged_provenance(
            request.provenance,
            request.trial_id,
            request.chunk_id,
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
            trial_id=request.trial_id,
            chunk_paths=[request.chunk_path],
            **request.settings.export_kwargs(),
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


def finish_experanto_workflow(data_store, request):
    """Save and export on root; propagate an export failure to every MPI rank."""
    import mozaik

    comm = mozaik.mpi_comm
    is_root = comm is None or comm.rank == mozaik.MPI_ROOT
    failure = None
    exception = None
    if is_root:
        try:
            data_store.save()
            export_experanto_chunk(data_store, request)
        except Exception as error:
            exception = error
            failure = f"{type(error).__name__}: {error}"
    if comm is not None:
        failure = comm.bcast(failure, root=mozaik.MPI_ROOT)
    if failure:
        raise RuntimeError(f"Experanto chunk export failed: {failure}") from exception


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


def merge_experanto_from_environment():
    """Entry called by a submitted merge job, without a CLI in the Mozaik module."""
    return merge_experanto_trial(**json.loads(os.environ[MERGE_JOB_ENV]))


def submit_experanto_merge_job(
    backend,
    *,
    simulation_job_ids,
    trial_id,
    output_dir,
    shard_paths=None,
    datastore_prefix=None,
    model_name=DEFAULT_MODEL_NAME,
    chunk_start=0,
    chunk_end=None,
    provenance=None,
    env=None,
    output=None,
    python_executable="python",
):
    """Submit a merge behind all required simulation/export jobs for one trial.

    Use the existing backend.execute_script interface. Missing job IDs are rejected
    before submission, so Slurm cannot accidentally queue a dependency-free merge.
    Paths are resolved when constructing the request; datastores, if needed for
    locating default chunk exports, are resolved by directory name at job execution.
    """
    ids = list(simulation_job_ids)
    if not ids or any(not job_id for job_id in ids):
        raise ValueError("Every simulation/export job must have a dependency ID")
    if (shard_paths is None) == (datastore_prefix is None):
        raise ValueError("Provide shard_paths or datastore_prefix, exclusively")
    if shard_paths is not None:
        shard_paths = [str(Path(path).resolve()) for path in shard_paths]
        if not shard_paths:
            raise ValueError("Provide at least one chunk export")
        if chunk_end is not None and chunk_end - chunk_start != len(shard_paths):
            raise ValueError("Chunk range does not match the supplied shard paths")
    elif chunk_end is None or chunk_end <= chunk_start:
        raise ValueError("Provide a nonempty chunk range")
    request = dict(
        trial_id=trial_id,
        output_dir=str(Path(output_dir).resolve()),
        shard_paths=shard_paths,
        datastore_prefix=str(Path(datastore_prefix).resolve())
        if datastore_prefix is not None
        else None,
        model_name=model_name,
        chunk_start=chunk_start,
        chunk_end=chunk_end,
        provenance=provenance,
    )
    job_env = dict(env or {})
    job_env[MERGE_JOB_ENV] = json.dumps(request)
    command = [
        python_executable,
        "-c",
        "from mozaik.meta_workflow.experanto_export import merge_experanto_from_environment; merge_experanto_from_environment()",
    ]
    job_id = backend.execute_script(command, env=job_env, depends_on=ids, output=output)
    if not job_id:
        raise RuntimeError("The backend did not report a merge-job ID")
    return job_id
