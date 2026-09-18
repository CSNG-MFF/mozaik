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

    The DSV has **no** stimulus-name filter (blanks/``InternalStimulus`` retained so the spike
    timeline stays aligned with the screen timeline) and **no** sheet filter (every recorded sheet is
    available; the spike exporter selects/folds sheets per ``sheet_names``).
    """
    data_store = PickledDataStore(
        load=True,
        parameters=ParameterSet({"root_directory": path, "store_stimuli": False}),
        replace=False,
    )
    return data_store, param_filter_query(data_store)


def _make_spike_exporter(output_dir, trial_id, sampling_rate, append_mode, sheet_names):
    return MozaikTrialExporter(
        os.path.join(output_dir, "responses") + "/",
        trial_id=trial_id,
        sampling_rate=sampling_rate,
        append_mode=append_mode,
        sheet_names=sheet_names,
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
):
    """Drive the exporter classes over already-built DSV(s) for a single trial, in one shot.

    Writes ``<output_dir>/responses/`` (spikes, folding every requested sheet) and
    ``<output_dir>/screen/``. This is the shared core used by the inline single-chunk path
    (``run.py --export``, one in-memory DSV); the multi-chunk driver builds the stateful exporters
    itself so it can stream many batches before finalizing.
    """
    dsvs = dsv_list if isinstance(dsv_list, (list, tuple)) else [dsv_list]
    if export_spikes:
        spikes = _make_spike_exporter(
            output_dir, trial_id, sampling_rate, append_mode, sheet_names
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
):
    """Multi-chunk trial-loop driver (the loop lifted from ``export.py``).

    For each trial: build the (stateful) spike + screen exporters, resolve each chunk's datastore via
    :func:`resolve_datastore`, open a blank-retaining all-sheet DSV, batch through the exporters
    (``batch_size`` chunks in memory before flushing), then finalize once. ``chunk_start > 0`` resumes
    an existing export via the spike exporter's append mode.

    ``output_dir_for_trial(trial)`` -> the trial's experiment dir; ``chunk_paths_for_trial(trial)`` ->
    the ordered list of **all** chunk JSON paths (screen timestamps need every chunk even when only a
    subset is processed for spikes). These closures keep project path patterns out of the package.
    """
    chunk_end = n_chunks if chunk_end is None else chunk_end
    is_resume = chunk_start > 0
    # Screen-only: only one chunk needs loading (just to resolve the source movie_path).
    screen_only = not export_spikes

    for trial in tqdm(trials):
        experiment_dir = output_dir_for_trial(trial)

        spike_exporter = (
            _make_spike_exporter(
                experiment_dir, trial, sampling_rate, is_resume, sheet_names
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
        for i, chunk in enumerate(tqdm(chunks_to_load)):
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
        if screen_exporter is not None:
            screen_exporter.finalize()
