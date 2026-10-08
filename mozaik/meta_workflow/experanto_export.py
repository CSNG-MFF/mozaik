"""Orchestration for exporting Mozaik DataStores to Experanto format.

This is the generic *driver* layer that sits above the exporter *classes* in
``mozaik.tools.experanto_export`` (``MozaikTrialExporter`` / ``MozaikScreenExporter``): the
trial/chunk loop, datastore resolution, batching, and resume/append. It is Experanto-format-aware
but **model-agnostic** — model name, sheet selection, paths, and chunk layout are all parameters
(mirroring ``controller.run_workflow``'s ``model_class``). Project-specific concerns (env var names,
default directories, the argparse CLI) stay in the thin entry points
(``mozaik-models/experanto/export.py`` and ``run.py``).

Two levels:

* :func:`export_dsvs_to_experanto` — core: drive the exporter classes over already-built DSV(s) for
  one trial in a single shot. Used directly by the in-memory inline export (``run.py --export``).
* :func:`run_experanto_export` — high-level: the multi-chunk trial loop from ``export.py`` (resolve
  per-chunk datastores, batch through the stateful exporters, resume/append).
"""

import gc
import glob
import json
import os

from mozaik.storage.datastore import PickledDataStore
from mozaik.storage.queries import param_filter_query
from mozaik.tools.experanto_export import MozaikScreenExporter, MozaikTrialExporter
from parameters import ParameterSet

try:  # tqdm is only present in the export container; degrade gracefully elsewhere.
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    def tqdm(iterable, *args, **kwargs):
        return iterable


DEFAULT_MODEL_NAME = "SelfSustainedPushPull"

# Experiment-level metadata at the root of a shard. Experanto reads its "data_key" to name the
# session (falling back to the shard's directory name without it); the "mozaik_run" record
# beside it is ignored by Experanto, and holds the settings the shard was simulated with, so
# that they outlive the datastores.
SHARD_META_FILE = "meta.json"


def resolve_datastore(datastore_prefix, trial, chunk, model_name=DEFAULT_MODEL_NAME):
    """Resolve the datastore directory for one ``(trial, chunk)``.

    Globs the stable prefix ``<model_name>_trial{t}_chunk{c}_____*``.
    Exactly one match returns that directory; multiple matches raise
    ``RuntimeError``; no matches raise ``FileNotFoundError``.

    ``datastore_prefix`` is a literal directory name, so it is escaped before globbing:
    any ``*``, ``?`` or ``[...]`` in it would otherwise be read as pattern syntax.
    """
    run_prefix = f"{model_name}_trial{trial}_chunk{chunk}_____"
    matches = sorted(
        glob.glob(os.path.join(glob.escape(datastore_prefix), run_prefix + "*"))
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous datastore for trial{trial}_chunk{chunk}: {len(matches)} dirs match "
            f"'{run_prefix}*' under {datastore_prefix!r}: {[os.path.basename(m) for m in matches]}"
        )
    raise FileNotFoundError(f"No datastore match for trial{trial}_chunk{chunk}")


def open_datastore_dsv(path):
    """Load a ``PickledDataStore`` and return ``(data_store, dsv)``.

    The DSV has **no** stimulus-name filter (explicit Experanto ``InternalStimulus`` blanks are
    retained) and **no** sheet filter. Mozaik's automatic null-stimulus recordings are excluded by
    ``get_segments()`` and unsupported by the exporter; Experanto experiments require
    ``null_stimulus_period == 0``.
    """
    data_store = PickledDataStore(
        load=True,
        parameters=ParameterSet({"root_directory": path, "store_stimuli": False}),
        replace=False,
    )
    return data_store, param_filter_query(data_store)


def _read_shard_meta(experiment_dir):
    path = os.path.join(experiment_dir, SHARD_META_FILE)
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _shard_meta(experiment_dir, trial, provenance, chunk_start, chunk_end):
    """
    Return the ``meta.json`` a shard should have once chunks ``[chunk_start, chunk_end)`` of
    *trial* have been exported into it, or None if it should be left alone.

    Called before any spike is exported, so that a failed check leaves the shard as it was.

    *provenance* is the run record the launcher wrote (``settings``, ``environment_variables``
    and every trial's per-chunk ``simulation_seeds``). The shard's record holds the same
    settings with ``chunks`` set to the chunks the shard actually contains, and the seeds of
    just those chunks. A run's settings cannot change between its rounds, so a resumed export
    only extends ``chunks`` and the seeds -- after checking that it continues exactly where the
    shard ends, which also catches a round exported out of order or twice.

    An export without *provenance* (one run by hand) leaves ``meta.json`` alone, unless the
    shard has a record: the record would then misstate what the shard holds, so that raises.
    """
    existing = _read_shard_meta(experiment_dir) or {}
    recorded = existing.get("mozaik_run")
    if provenance is None:
        if recorded is not None:
            raise ValueError(
                "%s records the run it was exported from, so exporting into it without that "
                "run's provenance would leave the record wrong; pass the provenance"
                % experiment_dir
            )
        return None

    all_seeds = provenance["simulation_seeds"].get(str(trial))
    if all_seeds is None or len(all_seeds) < chunk_end:
        raise ValueError(
            "the provenance has no simulation seeds for trial %d chunks %d-%d"
            % (trial, chunk_start, chunk_end - 1)
        )

    if chunk_start == 0:
        first, seeds = 0, []
    else:
        if recorded is None:
            raise ValueError(
                "%s has no run record to extend: chunks %d-%d can only be appended to a "
                "shard whose earlier chunks were exported with their provenance"
                % (experiment_dir, chunk_start, chunk_end - 1)
            )
        first, last = (int(c) for c in recorded["settings"]["chunks"].split("-"))
        if last != chunk_start - 1:
            raise ValueError(
                "%s holds chunks %s, so chunks %d-%d do not continue it"
                % (
                    experiment_dir,
                    recorded["settings"]["chunks"],
                    chunk_start,
                    chunk_end - 1,
                )
            )
        recorded_settings = dict(recorded["settings"])
        del recorded_settings["chunks"]
        if (
            recorded["trial"] != trial
            or recorded_settings != provenance["settings"]
            or recorded["environment_variables"] != provenance["environment_variables"]
        ):
            raise ValueError(
                "%s was exported from a run with different settings than the provenance "
                "given now" % experiment_dir
            )
        seeds = recorded["simulation_seeds"]

    meta = dict(existing)
    meta.setdefault("data_key", os.path.basename(os.path.normpath(experiment_dir)))
    meta["mozaik_run"] = {
        "trial": trial,
        "settings": dict(
            provenance["settings"], chunks="%d-%d" % (first, chunk_end - 1)
        ),
        "environment_variables": provenance["environment_variables"],
        "simulation_seeds": seeds + list(all_seeds[chunk_start:chunk_end]),
    }
    return meta


def _write_shard_meta(experiment_dir, meta):
    with open(os.path.join(experiment_dir, SHARD_META_FILE), "w") as f:
        json.dump(meta, f, indent=4)


def _make_spike_exporter(
    output_dir,
    trial_id,
    sampling_rate,
    append_mode,
    sheet_names,
    export_blank_spikes,
):
    return MozaikTrialExporter(
        os.path.join(output_dir, "responses") + "/",
        trial_id=trial_id,
        sampling_rate=sampling_rate,
        append_mode=append_mode,
        sheet_names=sheet_names,
        export_blank_spikes=export_blank_spikes,
    )


def _make_screen_exporter(
    output_dir,
    chunk_paths,
    frame_duration_ms,
    movie_frame_duration_ms,
    tier_reference,
):
    return MozaikScreenExporter(
        output_dir=output_dir,
        chunk_paths=chunk_paths,
        frame_duration_ms=frame_duration_ms,
        movie_frame_duration_ms=movie_frame_duration_ms,
        tier_reference=tier_reference,
    )


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
    append_mode=False,
    tier_reference=None,
    frame_duration_ms=7.0,
    movie_frame_duration_ms=35.0,
    export_blank_spikes=True,
):
    """Drive the exporter classes over already-built DSV(s) for a single trial, in one shot.

    Writes ``<output_dir>/responses/`` (spikes, folding every requested sheet) and
    ``<output_dir>/screen/``. This is the shared core used by the inline single-chunk path
    (``run.py --export``, one in-memory DSV); the multi-chunk driver builds the stateful exporters
    itself so it can stream many batches before finalizing. Set ``export_blank_spikes=False`` to
    retain explicit blank timing while omitting spikes recorded during those blanks.
    """
    dsvs = dsv_list if isinstance(dsv_list, (list, tuple)) else [dsv_list]
    if export_spikes:
        spikes = _make_spike_exporter(
            output_dir,
            trial_id,
            sampling_rate,
            append_mode,
            sheet_names,
            export_blank_spikes,
        )
        spikes.process_batch(dsvs)
        spikes.finalize()
    if export_screen:
        screen = _make_screen_exporter(
            output_dir,
            chunk_paths,
            frame_duration_ms,
            movie_frame_duration_ms,
            tier_reference,
        )
        screen.process_batch(dsvs)
        screen.finalize()


def run_experanto_export(
    *,
    trials,
    n_chunks,
    output_dir_for_trial,
    chunk_paths_for_trial,
    datastore_prefix,
    model_name=DEFAULT_MODEL_NAME,
    sheet_names=None,
    chunk_start=0,
    chunk_end=None,
    batch_size=4,
    sampling_rate=1000.0,
    export_spikes=True,
    export_screen=True,
    tier_reference=None,
    frame_duration_ms=7.0,
    movie_frame_duration_ms=35.0,
    export_blank_spikes=True,
    provenance=None,
):
    """Multi-chunk trial-loop driver (the loop lifted from ``export.py``).

    For each trial: build the (stateful) spike + screen exporters, resolve each chunk's datastore via
    :func:`resolve_datastore`, open a blank-retaining all-sheet DSV, batch through the exporters
    (``batch_size`` chunks in memory before flushing), then finalize once. ``chunk_start > 0`` resumes
    an existing export via the spike exporter's append mode.

    ``output_dir_for_trial(trial)`` -> the trial's experiment dir; ``chunk_paths_for_trial(trial)`` ->
    the ordered list of **all** chunk JSON paths (screen timestamps need every chunk even when only a
    subset is processed for spikes). These closures keep project path patterns out of the package.
    ``export_blank_spikes`` is forwarded to each trial's spike exporter.

    ``provenance`` is the run record written by the launcher that simulated the chunks; with
    spikes exported, each shard's ``meta.json`` records it (see :func:`_shard_meta`). Every
    trial's shard is checked against it before any trial is exported, so a mismatch exports
    nothing. Without it, ``meta.json`` is not written, as before.
    """
    chunk_end = n_chunks if chunk_end is None else chunk_end
    is_resume = chunk_start > 0
    # Screen-only: only one chunk needs loading (just to resolve the source movie_path).
    screen_only = not export_spikes

    # The spikes are what a shard's run record describes, so a screen-only export leaves it be.
    shard_metas = {
        trial: (
            _shard_meta(
                output_dir_for_trial(trial), trial, provenance, chunk_start, chunk_end
            )
            if export_spikes
            else None
        )
        for trial in trials
    }

    for trial in tqdm(trials, disable=None):
        experiment_dir = output_dir_for_trial(trial)

        spike_exporter = (
            _make_spike_exporter(
                experiment_dir,
                trial,
                sampling_rate,
                is_resume,
                sheet_names,
                export_blank_spikes,
            )
            if export_spikes
            else None
        )
        screen_exporter = (
            _make_screen_exporter(
                experiment_dir,
                chunk_paths_for_trial(trial),
                frame_duration_ms,
                movie_frame_duration_ms,
                tier_reference,
            )
            if export_screen
            else None
        )

        if screen_only:
            chunks_to_load = range(chunk_start, min(chunk_start + 1, chunk_end))
        else:
            chunks_to_load = range(chunk_start, chunk_end)

        dsv_list = []
        for i, chunk in enumerate(tqdm(chunks_to_load, disable=None)):
            path = resolve_datastore(datastore_prefix, trial, chunk, model_name)
            data_store, dsv = open_datastore_dsv(path)
            dsv_list.append(dsv)

            # Process in batches to manage memory.
            if (i + 1) % batch_size == 0:
                if spike_exporter is not None:
                    spike_exporter.process_batch(dsv_list)
                if screen_exporter is not None:
                    screen_exporter.process_batch(dsv_list)
                dsv_list = []
                del data_store
                gc.collect()

        # Process any remaining items in the list.
        if dsv_list:
            if spike_exporter is not None:
                spike_exporter.process_batch(dsv_list)
            if screen_exporter is not None:
                screen_exporter.process_batch(dsv_list)

        if spike_exporter is not None:
            spike_exporter.finalize()
            if shard_metas[trial] is not None:
                _write_shard_meta(experiment_dir, shard_metas[trial])
        if screen_exporter is not None:
            screen_exporter.finalize()
