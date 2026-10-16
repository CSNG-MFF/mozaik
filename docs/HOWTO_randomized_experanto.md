# HOWTO — run the **Randomized Experanto** experiment: simulation → Experanto export

This document is for a **Mozaik user** who wants to run a visual experiment and export the result in
**Experanto** format. It covers **what you need to prepare and why**, then **how to run it**. For a tiny,
copy-paste smoke run, see `HOWTO_test3_sim_then_export.md`.

`RandomizedExperanto` is a Mozaik visual-experiment protocol that drives a model over an explicit,
pre-computed list of images and videos — a **chunk** — so a large stimulus set can be split into
walltime-balanced pieces, simulated independently, and then exported. The protocol, chunk-list builder, exporter and merger live in Mozaik. Model construction,
experiment selection and launcher argument parsing belong to the companion model project.

**Terms used here.** A **chunk** is one JSON list of stimuli (a slice of the full set) that a single run
simulates. A **shard** is one exported Experanto directory for a trial: `responses/` (the spike responses)
plus `screen/` (the stimulus frames and their metadata) — the on-disk unit the Experanto library reads. A
**trial** is a repeat of the whole stimulus set (same network, different background noise).

## What you need, and why

- **A working Mozaik installation** — the model runs on PyNN/NEST exactly as described in the Mozaik
  `README.rst`: Mozaik + the PyNNStepCurrentModule build of PyNN + NEST + the `stepcurrentmodule` NEST
  module, in a virtualenv. Nothing here changes that install. *(This stack can also run inside a Docker or
  an Apptainer image — see "Running on a cluster with Apptainer"; that image is environment-specific,
  not part of Mozaik.)*
- **NumPy and PyYAML for merging** — merging reads Experanto arrays and metadata directly, without
  loading Mozaik datastores or importing the simulator.
- **An input stimulus dataset in Experanto *screen* form** — a directory with `screen/meta/*.yml` and
  `screen/data/*.npy` (one metadata file + one pixel array per image/video). This is *what the model sees*;
  you point `BASE_PATH` at it.
- **Chunk lists** — generated once from that dataset (Step 0). *Why:* to split a large stimulus set into
  independently-simulable, walltime-balanced pieces you can run in parallel.

### The Experanto experiment family

`RandomizedExperanto` is one of three subclasses of **`PixelMovieExperantoBase`** (`vision.py`), which owns
the shared stimulus construction, timing resolver, and default timing (pre-blank → image → post-blank; videos bare). Blank periods
come from image metadata, with a default 49 ms post-blank when `post_blank_period` is absent. The three
differ only in **how they enumerate the stimuli** to present:

| Class | Enumerates stimuli by | Status |
|---|---|---|
| **`RandomizedExperanto`** | an explicit, pre-computed **chunk JSON** (`chunk_dict_path`) — you control the grouping into trials/chunks | this HOWTO |
| `MeasurePixelMovieExperanto` | scanning a whole Experanto **screen directory** — the directory contents/order define the grouping | same output |
| `SingleMoviePixelMovieExperanto` | a single **movie file** presented as frame chunks | single-movie protocol |

All three call the **same** per-stimulus construction (`_append_meta_stimulus`) and record the **same** data
with the **same** metadata — so `MeasurePixelMovieExperanto` is **not** more limited than `RandomizedExperanto`.
The only difference is how the stimuli are enumerated and grouped: `RandomizedExperanto` takes an explicit,
pre-balanced chunk list (you decide the trial/chunk grouping, and can split it across parallel jobs), whereas
`MeasurePixelMovieExperanto` presents everything in a screen directory (listing order, optional window). This
HOWTO uses `RandomizedExperanto` because that chunk list is what the parallel-friendly export consumes. Other
`vision.py` protocols (`MeasureNaturalImages`, `MeasurePixelMovieFromFile`, …) are unrelated to the Experanto
export and out of scope.

The workflow has two phases: each simulation job exports its finished chunk, then
one dependent job per trial merges the Experanto shards. Single-chunk trials use
the same path.
Model launchers use the library hooks below; no model-specific launcher is bundled here.

## TL;DR

```bash
# Run from the mozaik-models LSV1M directory. The launcher builds the chunk lists.
python run_parameter_search_experanto.py run_experanto.py nest param_experanto/defaults \
    --data-root dataset --n-trials 20 --n-chunks 12 \
    --mozaik-env "$VIRTUAL_ENV/bin/activate"
# Add --skip-merge to export chunks without scheduling a final merge.
```

---

## Running on a cluster with Apptainer

The commands below run inside an **Apptainer container**. This is **not** required by Mozaik: 
a standard `README.rst` install runs the exact same
`python run_experanto.py …` command directly and use the library hooks below for exports. If you have that install, ignore the
`apptainer exec …` / `--bind` wrapper and run the `python -u …` line on its own.

In a container the paths map as follows (all bind mounts; `$PWD` is the **Mozaik repo root**):

| Host | In container | What it is |
|---|---|---|
| `$PWD` (Mozaik repo root) | `/mozaik` | Mozaik itself — the experiment protocol + exporter library. |
| `mozaik-models/LSV1M/` | `/project` | The companion project: `run_experanto.py`, `run_parameter_search_experanto.py`, `param_experanto/`. |
| a clone of the **experanto** library ([repo](https://github.com/goirik-chakrabarty/experanto)) | `/experanto` | The Experanto data-format package the export writes through. |
| your **working directory** | `/data` | Holds the input dataset, the chunk lists, **and the output shards**. |

- **The image (`.sif`)** is an Apptainer/Singularity image bundling Mozaik + NEST. It is **specific to a
  site and is not part of the Mozaik repo** — build or obtain one whose PyNN is the PyNNStepCurrentModule
  build (0.13.0, which provides `freeze_time`). Select the image for your site.

Regardless of environment you still need:

- **An input dataset in Experanto *screen* form** (`screen/meta/*.yml` + `screen/data/*.npy`), pointed to by
  `BASE_PATH`.
- **Chunk lists** `{CHUNK_DIR}/{trial}_{chunk}.json` — built in Step 0.

---

## Step 0 — generate the chunk lists

The launcher generates chunk lists automatically. To prepare them separately,
call the library in an environment with Mozaik and PyYAML:

```python
from mozaik.tools.experanto_chunks import generate_chunks

generate_chunks(
    data_root="dataset", output_dir="chunks",
    n_trials=20, n_chunks=12, seed=42,
)
```

- Scans `screen/meta/*.yml`, keeping `image` and `video` stimuli.
- **Per trial**, shuffles the full stimulus list with `Random(seed + trial)` — every trial sees the same
  stimuli in a different, reproducible order.
- **Balances** into `n_chunks` by greedy min-heap (each stimulus → the currently-cheapest chunk). Cost is a
  walltime estimate: images carry two blanks; **video cost is per-frame**
  (`video_base + num_frames × video_per_frame`), so one long video does not swamp a chunk.
- Emits `{trial}_{chunk}.json`, each record exactly `{modality, file, trial}` — the only fields
  `RandomizedExperanto.generate_stimuli` reads. It prints each chunk's stimulus count and **estimated
  walltime**; size `n_chunks` so a chunk fits the job time limit.
- When generated through `mozaik.tools.experanto_chunks.generate_chunks`, the lists are accompanied by
  `chunk_settings.json` — the dataset (absolute path), trial count, chunk count and seed they were
  generated from. The launcher checks reused lists against it.

The chunk carries **no seed** — per-trial noise is set at launch (see "Seeds").

---

## Simulate each chunk

### Step 1 — Simulation (one job per `(trial, chunk)`)

Each job simulates one chunk. Select it with `TRIAL` / `CHUNK` / `CHUNK_DIR`; vary `simulation_seed` per
trial for independent noise. Direct in-container invocation (portable; adapt the outer loop / array to your
scheduler):

```bash
cd "$MOZAIK_ROOT"                                 # your Mozaik repo root ($PWD, below, must be this dir)
module load apptainer                             # cluster-specific; skip on a standard README.rst install
SIF=$PWD/../mozaik-sif/<mozaik-nest-image>.sif    # the site Apptainer image (see "Running on a cluster …")
EXPERANTO=/path/to/experanto                      # your clone of the experanto library
WORKDIR=/path/to/workdir                          # holds the input dataset, chunk lists, and output shards

TRIAL=0 CHUNK=0                                   # ← the chunk to simulate
NRANKS=12                                         # MPI ranks — set to the physical cores you allocated (see note)
apptainer exec --cleanenv \
  --env PYTHONPATH=/mozaik --env OMPI_MCA_orte_tmpdir_base=/tmp \
  --env TRIAL=$TRIAL --env CHUNK=$CHUNK --env CHUNK_DIR=/data/mozaik_chunk \
  --env BASE_PATH=/data/<input-screen-dataset> --env NRANKS=$NRANKS \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
  --bind "$PWD:/mozaik" --bind "$PWD/../mozaik-models/experanto:/project" \
  --bind "$EXPERANTO:/experanto" \                 # your experanto library clone
  --bind "$WORKDIR:/data" \                        # your working dir: dataset + chunk lists + output shards
  "$SIF" bash -lc '
    unset HTTP_PROXY HTTPS_PROXY FTP_PROXY http_proxy https_proxy ftp_proxy
    cd /project
    mpirun -n "$NRANKS" \
      -x OMP_NUM_THREADS -x MKL_NUM_THREADS -x OPENBLAS_NUM_THREADS -x PYTHONPATH \
      -x TRIAL -x CHUNK -x CHUNK_DIR -x BASE_PATH -x NRANKS \
      python -u run_experanto.py nest "$NRANKS" param_experanto/defaults \
        results_dir "'"'"'/data/<fresh-out-dir>/'"'"'" \
        simulation_seed '"$(( TRIAL*1000 + CHUNK + 1 ))"' \
        trial'"$TRIAL"'_chunk'"$CHUNK"'
  '
```

- **`--bind "$PWD:/mozaik"` binds the repo ROOT** (`$PWD`), not `$PWD/mozaik`. With `PYTHONPATH=/mozaik`,
  `import mozaik` resolves to `/mozaik/mozaik` (the package). Binding the package dir `$PWD/mozaik` as `/mozaik`
  makes `import mozaik` fail with `ModuleNotFoundError`. This matches the compose scripts' "mozaik root → /mozaik".
- **Ranks:** `NRANKS` is used for **both** `mpirun -n` and `nest N` — they must match. Set it to the number of
  **physical cores** you allocated; one rank per physical core (hyperthreads don't speed up NEST). On a proper
  SLURM task allocation `mpirun -n "$NRANKS"` just works (this is what the cluster runner does). On a bare
  interactive node where mpirun sees one slot, add `--oversubscribe` and pin with `taskset -c <your CPU list>`.
- **Output:** one datastore per chunk,
  `SelfSustainedPushPull_trial{T}_chunk{C}_____simulation_seed:{n}/`, under `results_dir`.
- **`results_dir` must be a quoted Python-string literal** — mozaik `eval()`s override values, so a bare
  `/data/...` fails (hence the `"'"'"'…'"'"'"` quoting through the nested shells); keep that wrapper exactly.
- **`simulation_seed` must be nonzero** (NEST rejects `rng_seed=0`); vary it per trial for per-trial noise.
- **Harmless noise:** the container prints `CMake … CMAKE_CXX_COMPILER not set` (it tries to build a NEST
  extension) and creates a `_build/` dir — ignore both. First-run wall time is dominated by retina/LGN
  filtering, which is **independent of network size** — a large-stimulus (long-video) chunk can take far
  longer than the NEST simulation itself.
- To run **all** `(trial, chunk)` pairs, loop `idx = 0 .. n_trials*n_chunks-1` with
  `TRIAL = idx / N_CHUNKS`, `CHUNK = idx % N_CHUNKS` (a SLURM array is the natural fit).

## Export after simulation

In the model entry point, call the exporter on MPI root after `run_workflow`
returns its saved datastore. Use the same export settings for every chunk:

```python
import mozaik
from mozaik.meta_workflow.experanto_export import (
    ExperantoExportSettings, export_experanto_chunk,
)

settings = ExperantoExportSettings(export_blank_spikes=True)
data_store, model = run_workflow(model_name, Model, create_experiments)
if mozaik.mpi_comm is None or mozaik.mpi_comm.rank == mozaik.MPI_ROOT:
    export_experanto_chunk(
        data_store, trial_id=trial, chunk_id=chunk,
        chunk_path=f"chunks/{trial}_{chunk}.json",
        settings=settings, provenance=run_record,
    )
```

The model entry point supplies the model, experiment factory and selected
trial/chunk. Export settings include sheet selection, blank spikes, sampling
rate, frame timings and an optional `tier_reference_path`. Frame timings must
match the experiment's timings.

Output defaults to `<datastore>/experanto`; `output_dir` selects a separate
writable directory. Each completed export contains `responses/`, `screen/`,
`meta.json` and `chunk_export.json`. Failed exports never publish the final
output directory. Existing outputs are refused. The completion record preserves
the trial, chunk, input-list digest, export settings and optional run provenance.
`provenance` is the full `run_config.json` dictionary, shared by all chunks.

## Backend integration

The mozaik-models LSV1M launcher submits one simulation/export job per chunk and
one dependent merge per trial. It passes `TRIAL`, `CHUNK`, `CHUNK_DIR`,
`BASE_PATH` and `RUN_CONFIG` through the job environment. `run_experanto.py`
reads settings and provenance from that run record and calls the exporter
directly. Allocate simulation memory and wall time for the export phase.

The launcher submits its `experanto_merge.py` through the backend's generic
`execute_script` interface, with Slurm `afterok` dependencies on the trial's
simulation jobs. It requires a complete set of job IDs. The merge runs in one
process and reads only completed chunk exports.

Use `--skip-merge` (or `skip_merge: true` in the launcher config) to leave the
individual exports for merging later. This option does not change simulation
or export settings. For runs submitted in rounds, finish intermediate rounds
with merging skipped before submitting the final round with merging enabled.
The final merge requires all chunks of the trial, and leaves their exports intact.

To merge completed exports manually, run from the mozaik-models LSV1M directory:

```bash
python experanto_merge.py 0 --results-dir runs/experiment --n-chunks 20
```

The output defaults to `runs/experiment/shards/trial0`. Use `--output-dir`
for a different fresh output directory. A subset can be selected with
`--chunk-start` (inclusive) and `--chunk-end` (exclusive).
The default lookup uses `SelfSustainedPushPull_trialT_chunkC_____*`.
For other models or layouts, call `merge_experanto_trial` with `model_name`
or ordered `shard_paths`.

## Merging individually exported shards

Manual exports can be merged directly, with or without completion records,
provided they use the same unit order and settings:

```python
from mozaik.tools.experanto_merge import merge_experanto_shards

shards = [f"chunk_exports/trial0_chunk{chunk}" for chunk in range(20)]
merge_experanto_shards(shards, "merged/trial0")
```

Inputs are in numerical chunk order; output must be new. The merger shifts spikes
and timestamps, removes intermediate synthetic terminal blanks and preserves
actual stimulus blanks and the final endpoint. Its screen covers only supplied
chunks. Completion records, when present, check order and settings and supply
provenance automatically. For shards without completion records, pass `provenance=run_record` explicitly;
`chunk_start` identifies the first supplied chunk (default 0).

## Environment knobs (the experiment driver)

`create_randomized_experanto(model)` (`mozaik-models/LSV1M/experiments.py`) reads:

| Env var | Default | Role |
|---|---|---|
| `TRIAL` | `0` | Trial index; selects the chunk JSON and (at launch) the noise seed. |
| `CHUNK` | `0` | Chunk index within the trial. Reads `{CHUNK_DIR}/{TRIAL}_{CHUNK}.json`. |
| `CHUNK_DIR` | `/data/mozaik_chunk` | Directory of chunk JSONs. |
| `BASE_PATH` | required by `run_experanto.py` | Input Experanto **screen** dataset the stimuli are read from (e.g. a P4 subset from `materialize_subset_screen.py`). |
| `STIM_WIDTH` | `11` | Presented image's longest-axis extent in **degrees** (not the visual field). E2 Cadena runs use `6.7`. |

---

## Execution chain

```text
simulation: run_workflow → save datastore → return → export_experanto_chunk (root only)
merge job:  experanto_merge.py → merge_experanto_trial → merge_experanto_shards
```

**Timing / sync invariant:** each image is `pre-blank → image (~497 ms) → post-blank`; videos are
`num_frames × 35 ms`, bare. `pre_blank_period` and `post_blank_period` are metadata values in seconds and
are rounded down to complete input frames. Missing `post_blank_period` warns and defaults to 49 ms; effective
zero-duration blanks are omitted. The simulation and screen exporter use the same resolver in
`mozaik/experiments/vision.py`, keeping their clocks equal
(`responses/meta.yml:end_time == screen/timestamps.npy[-1]`).

Spikes recorded during those explicit pre- and post-blanks are included in `responses/spikes.npy`.
Set `ExperantoExportSettings(export_blank_spikes=False)` to omit those spikes
while retaining blank timing and metadata.
Mozaik's separate automatic null-stimulus periods are not exported, so Experanto experiments and
exports require `model.parameters.null_stimulus_period == 0`.

---

## Seeds (three-stream)

`param_experanto/defaults` uses three seeds, each driving its own RNG (`controller.py` → `mozaik.setup_mozaik_seeds`):

| Seed | Drives (RNG) | Governs | Varies |
|---|---|---|---|
| `model_seed=1023` | `model_rng` / `model_pynn_rng` | **network identity** — connectivity, neuron positions, weights, connector sampling | fixed across trials |
| `simulation_seed` | `simulation_rng` | **NEST kernel noise** (per-trial background) — must be **nonzero** | **override per trial** (nonzero) for independent noise, same network |
| `experiment_seed=0` | `experiment_rng` | **experiment-level RNG** — stimulus shuffling / random draws, in experiments that shuffle at runtime | fixed |

> **Stimulus order for this pipeline is NOT set by a runtime seed.** `RandomizedExperanto` presents the
> chunk **in the order written in the JSON**, with no runtime shuffle — so `experiment_seed` has no effect on
> the order here (it matters only for shuffling experiments like the tuning protocols). The order is fixed
> **offline** by `generate_chunks(..., seed=...)` (Step 0), which shuffles per trial with `Random(seed + trial)`.
> Treat that as the pipeline's fourth, build-time seed.

Per-trial noise is set on the `run_experanto.py` CLI (`simulation_seed <n>`), **not** in the chunk JSON. The sim is
bit-reproducible under fixed seeds (same three seeds → identical spikes).

---

## Verify the export

- **Sanity checks on the shard** (load the files with numpy + PyYAML — e.g. inside the container, since the
  bare host python may lack them):
  - **timelines agree:** `responses/meta.yml`'s `end_time` equals the last value of `screen/timestamps.npy`;
  - **spike index is well-formed:** `responses/meta.yml`'s `spike_indices` (the CSR offsets into
    `responses/spikes.npy`) has length `n_signals + 1`;
  - **counts are as expected:** the number of signals (neurons) and its per-sheet breakdown (`sheets`,
    `n_signals_layerwise`), and the number of `screen/data/*.npy` files match the stimuli you presented.
