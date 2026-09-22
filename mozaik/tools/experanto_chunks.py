"""
Build the chunk lists that drive the Experanto ``RandomizedExperanto`` experiment.

A *chunk list* is a JSON file named ``{trial}_{chunk}.json`` holding, in presentation order,
the stimuli that one simulation job presents::

    [{"modality": "image", "file": "00042.yml", "trial": 0}, ...]

Those three fields are the whole contract, and both of its consumers live in this package:
:class:`mozaik.experiments.vision.RandomizedExperanto` reads ``file`` and ``trial`` to build
the stimulus sequence, and
:class:`mozaik.tools.experanto_export.MozaikScreenExporter` reads ``file`` and ``modality``
to rebuild the screen timeline at export time.

Chunks exist because one trial of a large stimulus set does not fit in a single job's wall
time. The stimuli of a trial are shuffled and then split into ``n_chunks`` groups of roughly
equal estimated duration, so that all the jobs of a trial finish at about the same time.

Reproducibility
---------------
The shuffle (``random.Random(seed + trial)``) and the greedy assignment order in
:func:`split_time_balanced` together determine *which stimuli end up in which chunk*, and
therefore which stimuli end up in which datastore. Changing either silently invalidates
comparisons against chunk sets generated earlier, so they must be kept as they are unless a
regeneration of existing datasets is intended.
"""

import heapq
import json
import os
import random

import yaml

# Time cost estimates in seconds, from empirical measurements on one cluster; they are only
# used to balance chunks against each other, so their absolute scale does not matter, only
# their ratios. "first_*" is the first presentation of a unique stimulus (cold cache), "*" a
# repeat presentation (warm cache). Video cost is PER-FRAME rather than flat: a flat cost
# badly under-weights long (900-frame) videos in the greedy balancer. Pass a ``time_costs``
# dict of the same shape to recalibrate for a different machine.
DEFAULT_TIME_COSTS = {
    "first_image": 80,
    "image": 70,
    "blank": 75,
    "video_base": 48,
    "video_per_frame": 0.55,
}

# The fields written to a chunk record, in this order. ``num_frames`` is carried internally
# for cost estimation but deliberately not written: it is not part of the contract above.
CHUNK_RECORD_FIELDS = ("modality", "file", "trial")


def scan_screen_metadata(data_root):
    """
    Read the stimulus metadata of an Experanto screen dataset.

    Parameters
    ----------
    data_root : str
        Root of the dataset; its metadata is read from ``data_root/screen/meta/*.yml``.

    Returns
    -------
    list of dict
        One ``{"file", "modality", "num_frames"}`` entry per ``image`` or ``video`` stimulus,
        ordered by file name. ``blank`` stimuli are skipped: they carry no npy file and the
        simulation skips them.

    Raises
    ------
    FileNotFoundError
        If the metadata directory does not exist.
    ValueError
        If it contains no image or video stimulus.
    """
    meta_dir = os.path.join(data_root, "screen", "meta")
    if not os.path.isdir(meta_dir):
        raise FileNotFoundError("metadata directory not found: %s" % meta_dir)

    entries = []
    for fname in sorted(os.listdir(meta_dir)):
        if not fname.endswith(".yml"):
            continue
        with open(os.path.join(meta_dir, fname), "r") as f:
            meta = yaml.safe_load(f)
        modality = meta.get("modality")
        if modality in ("image", "video"):
            entries.append(
                {
                    "file": fname,
                    "modality": modality,
                    "num_frames": int(meta.get("num_frames", 1)),
                }
            )

    if not entries:
        raise ValueError("no image or video stimulus metadata found in %s" % meta_dir)
    return entries


def estimate_cost(item, seen_files, time_costs=None):
    """
    Estimate the wall time one stimulus presentation costs, in seconds.

    Parameters
    ----------
    item : dict
        A stimulus entry as returned by :func:`scan_screen_metadata`.
    seen_files : set
        File names already presented in the chunk this stimulus is being costed for; used to
        tell a cold first presentation from a warm repeat. **Mutated**: ``item``'s file name
        is added to it.
    time_costs : dict, optional
        Cost table; defaults to :data:`DEFAULT_TIME_COSTS`.
    """
    time_costs = DEFAULT_TIME_COSTS if time_costs is None else time_costs
    modality = item["modality"]
    filename = item["file"]
    cost = 0.0

    # Images are presented wrapped in a pre-blank and a post-blank (see
    # PixelMovieExperantoBase._append_meta_stimulus); videos are presented bare.
    if modality == "image":
        cost += time_costs["blank"] * 2

    is_first = filename not in seen_files
    if modality == "image":
        cost += time_costs["first_image"] if is_first else time_costs["image"]
    elif modality == "video":
        cost += (
            time_costs["video_base"]
            + item["num_frames"] * time_costs["video_per_frame"]
        )

    seen_files.add(filename)
    return cost


def estimate_chunk_time(chunk, time_costs=None):
    """
    Estimate the total wall time of one chunk, in seconds.

    Parameters
    ----------
    chunk : list of dict
        The stimuli of the chunk, in presentation order.
    time_costs : dict, optional
        Cost table; defaults to :data:`DEFAULT_TIME_COSTS`.
    """
    seen = set()
    total = 0.0
    for item in chunk:
        total += estimate_cost(item, seen, time_costs)
    return total


def split_time_balanced(stimuli, n_chunks, time_costs=None):
    """
    Split stimuli into ``n_chunks`` groups of roughly equal estimated wall time.

    Each stimulus in turn is assigned to the chunk with the lowest accumulated time so far
    (greedy min-heap). The stimuli are walked in the order given, so the caller's ordering is
    part of the result -- see the reproducibility note in the module docstring.

    Parameters
    ----------
    stimuli : list of dict
        Stimulus entries, in the order they should be considered.
    n_chunks : int
        Number of chunks to produce.
    time_costs : dict, optional
        Cost table; defaults to :data:`DEFAULT_TIME_COSTS`.

    Returns
    -------
    list of list of dict
        ``n_chunks`` lists, each holding the stimuli of one chunk in presentation order.
    """
    if n_chunks == 1:
        return [list(stimuli)]

    # heap entries: (accumulated time, chunk index)
    heap = [(0.0, i) for i in range(n_chunks)]
    chunks = [[] for _ in range(n_chunks)]
    # Cold-vs-warm is tracked per chunk: each chunk is presented by its own job.
    seen_per_chunk = [set() for _ in range(n_chunks)]

    for item in stimuli:
        total_time, idx = heapq.heappop(heap)
        chunks[idx].append(item)
        cost = estimate_cost(item, seen_per_chunk[idx], time_costs)
        heapq.heappush(heap, (total_time + cost, idx))

    return chunks


def _chunk_record(stimulus):
    """Reduce an internal stimulus entry to the fields written to the chunk JSON."""
    return {field: stimulus[field] for field in CHUNK_RECORD_FIELDS}


def generate_chunks(
    data_root, output_dir, n_trials, n_chunks, seed=42, time_costs=None
):
    """
    Write the chunk lists for a whole experiment.

    For each trial the full stimulus set is shuffled with ``random.Random(seed + trial)`` --
    every trial sees the same stimuli in a different but reproducible order -- and split into
    ``n_chunks`` time-balanced chunks, written as ``output_dir/{trial}_{chunk}.json``.

    Parameters
    ----------
    data_root : str
        Root of the Experanto screen dataset (see :func:`scan_screen_metadata`).
    output_dir : str
        Directory the chunk JSONs are written to; created if it does not exist.
    n_trials : int
        Number of trials.
    n_chunks : int
        Number of chunks per trial.
    seed : int, optional
        Base seed of the per-trial shuffle.
    time_costs : dict, optional
        Cost table; defaults to :data:`DEFAULT_TIME_COSTS`.

    Returns
    -------
    list of dict
        One entry per written chunk, with keys ``trial``, ``chunk``, ``path``, ``n_stimuli``,
        ``n_images``, ``n_videos`` and ``est_seconds``. Size ``n_chunks`` so that the largest
        ``est_seconds`` fits the wall time of one job.
    """
    entries = scan_screen_metadata(data_root)
    os.makedirs(output_dir, exist_ok=True)

    summary = []
    for trial in range(n_trials):
        stimuli = [dict(entry, trial=trial) for entry in entries]
        random.Random(seed + trial).shuffle(stimuli)
        chunks = split_time_balanced(stimuli, n_chunks, time_costs)

        for chunk_idx, chunk in enumerate(chunks):
            path = os.path.join(output_dir, "%d_%d.json" % (trial, chunk_idx))
            with open(path, "w") as f:
                json.dump([_chunk_record(s) for s in chunk], f)

            summary.append(
                {
                    "trial": trial,
                    "chunk": chunk_idx,
                    "path": path,
                    "n_stimuli": len(chunk),
                    "n_images": sum(1 for s in chunk if s["modality"] == "image"),
                    "n_videos": sum(1 for s in chunk if s["modality"] == "video"),
                    "est_seconds": estimate_chunk_time(chunk, time_costs),
                }
            )

    return summary
