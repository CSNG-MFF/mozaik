# HOWTO — run the **Randomized Experanto** experiment: simulation → Experanto export

This document is for a **Mozaik user** who wants to run a visual experiment and export the result in
**Experanto** format. It covers **what you need to prepare and why**, then **how to run it**. For a tiny,
copy-paste smoke run, see `HOWTO_test3_sim_then_export.md`.

`RandomizedExperanto` is a Mozaik visual-experiment protocol that drives a model over an explicit,
pre-computed list of images and videos — a **chunk** — so a large stimulus set can be split into
walltime-balanced pieces, simulated independently, and then exported. The protocol is part of Mozaik
(`mozaik/experiments/vision.py`); the chunk-list builder and the exporter currently live in the companion
**`mozaik-models/experanto/`** project (`generate_chunks.py`, `run.py`, `export.py`) and may move into
Mozaik after merge.

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
- **The `experanto` library importable** — the export writes through the Experanto data-format package
  ([experanto](https://github.com/goirik-chakrabarty/experanto)).
- **An input stimulus dataset in Experanto *screen* form** — a directory with `screen/meta/*.yml` and
  `screen/data/*.npy` (one metadata file + one pixel array per image/video). This is *what the model sees*;
  you point `BASE_PATH` at it.
- **Chunk lists** — generated once from that dataset (Step 0). *Why:* to split a large stimulus set into
  independently-simulable, walltime-balanced pieces you can run in parallel.

### The Experanto experiment family

`RandomizedExperanto` is one of three subclasses of **`PixelMovieExperantoBase`** (`vision.py`), which owns
the shared stimulus construction and timing (pre-blank → image → 49 ms post-blank; videos bare). The three
differ only in **how they enumerate the stimuli** to present:

| Class | Enumerates stimuli by | Status |
|---|---|---|
| **`RandomizedExperanto`** | an explicit, pre-computed **chunk JSON** (`chunk_dict_path`) — you control the grouping into trials/chunks | this HOWTO |
| `MeasurePixelMovieExperanto` | scanning a whole Experanto **screen directory** — the directory contents/order define the grouping | same output |
| `SingleMoviePixelMovieExperanto` | a single **movie file** presented as frame chunks | legacy, no live caller |

All three call the **same** per-stimulus construction (`_append_meta_stimulus`) and record the **same** data
with the **same** metadata — so `MeasurePixelMovieExperanto` is **not** more limited than `RandomizedExperanto`.
The only difference is how the stimuli are enumerated and grouped: `RandomizedExperanto` takes an explicit,
pre-balanced chunk list (you decide the trial/chunk grouping, and can split it across parallel jobs), whereas
`MeasurePixelMovieExperanto` presents everything in a screen directory (listing order, optional window). This
HOWTO uses `RandomizedExperanto` because that chunk list is what the parallel-friendly export consumes. Other
`vision.py` protocols (`MeasureNaturalImages`, `MeasurePixelMovieFromFile`, …) are unrelated to the Experanto
export and out of scope.

There are **two** ways to produce shards:

| | **Workflow 1 — canonical (multi-chunk)** | **Workflow 2 — inline (single-chunk)** |
|---|---|---|
| Shape | sim one chunk per job, **then** a separate export per trial | sim **and** export one chunk in one process |
| Use when | any real dataset (chunks `0..N-1` concatenated per trial) | a single-chunk run you want exported immediately |
| Entry | `export.py` driver (walks chunks) | `run.py --export` (exports the one chunk it simulated) |
| Output | shards under `OUTPUT_PREFIX{trial}/` | shard **next to the datastore** (`<datastore>/experanto/`) |

> Launch is a plain `run.py` / `export.py` invocation (below). Wrapping it in a scheduler (a SLURM array,
> etc.) is up to you and your site; any site-specific launcher is out of scope for this manual.

---

## TL;DR

```bash
# 0) Build the chunk lists once (offline, outside the container is fine — pure Python + pyyaml)
python mozaik-models/experanto/generate_chunks.py \
    --data-root <dataset>/screen/... --output-dir /data/mozaik_chunk \
    --n-trials 20 --n-chunks 12

# 1) SIMULATE each (trial, chunk): one run.py per chunk, TRIAL/CHUNK/CHUNK_DIR select the chunk JSON
#    (loop the array over TRIAL*N_CHUNKS + CHUNK; see "Workflow 1 - Step 1")

# 2) EXPORT each trial: concatenate its chunks into one Experanto shard
python -u export.py <trial> --n-chunks 12          # CHUNK_DIR / OUTPUT_PREFIX / DATASTORE_PREFIX via env
```

---

## Running on a cluster with Apptainer

The commands below run inside an **Apptainer container**. This is **not** required by Mozaik: 
a standard `README.rst` install runs the exact same
`python run.py …` / `python export.py …` command directly. If you have that install, ignore the
`apptainer exec …` / `--bind` wrapper and run the `python -u …` line on its own.

In a container the paths map as follows (all bind mounts; `$PWD` is the **Mozaik repo root**):

| Host | In container | What it is |
|---|---|---|
| `$PWD` (Mozaik repo root) | `/mozaik` | Mozaik itself — the experiment protocol + exporter library. |
| `mozaik-models/experanto/` | `/project` | The companion project: `run.py`, `export.py`, `generate_chunks.py`, `param/`. |
| a clone of the **experanto** library ([repo](https://github.com/goirik-chakrabarty/experanto)) | `/experanto` | The Experanto data-format package the export writes through. |
| your **working directory** | `/data` | Holds the input dataset, the chunk lists, **and the output shards**. |

- **The image (`.sif`)** is an Apptainer/Singularity image bundling Mozaik + NEST. It is **specific to a
  site and is not part of the Mozaik repo** — build or obtain one whose PyNN is the PyNNStepCurrentModule
  build (0.13.0, which provides `freeze_time`). We record the current image in
  `mozaik-models/experanto/experiments/LOG.md`.

Regardless of environment you still need:

- **An input dataset in Experanto *screen* form** (`screen/meta/*.yml` + `screen/data/*.npy`), pointed to by
  `BASE_PATH`.
- **Chunk lists** `{CHUNK_DIR}/{trial}_{chunk}.json` — built in Step 0.

---

## Step 0 — generate the chunk lists (`generate_chunks.py`)

Run once to turn a stimulus dataset into per-`(trial, chunk)` JSON lists:

```bash
python mozaik-models/experanto/generate_chunks.py \
    --data-root /data/<dataset>/screen/... \   # dir containing screen/meta/*.yml
    --output-dir /data/mozaik_chunk \
    --n-trials 20 --n-chunks 12 \
    --seed 42                                   # base seed (default 42)
```

- Scans `screen/meta/*.yml`, keeping `image` and `video` stimuli.
- **Per trial**, shuffles the full stimulus list with `Random(seed + trial)` — every trial sees the same
  stimuli in a different, reproducible order.
- **Balances** into `n_chunks` by greedy min-heap (each stimulus → the currently-cheapest chunk). Cost is a
  walltime estimate: images carry two blanks; **video cost is per-frame**
  (`video_base + num_frames × video_per_frame`), so one long video does not swamp a chunk.
- Emits `{trial}_{chunk}.json`, each record exactly `{modality, file, trial}` — the only fields
  `RandomizedExperanto.generate_stimuli` reads. It prints each chunk's stimulus count and **estimated
  walltime**; size `--n-chunks` so a chunk fits the job time limit.

The chunk carries **no seed** — per-trial noise is set at launch (see "Seeds").

---

## Workflow 1 — canonical (sim per chunk → export per trial)

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
      python -u run.py nest "$NRANKS" param/defaults \
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

### Step 2 — Export (one shard per trial, after that trial's chunks COMPLETE)

`export.py <trial…> --n-chunks N` concatenates chunks `0..N-1` for each trial into one shard. Env selects
the paths:

```bash
# inside the same container (cd /project), or via apptainer exec as above:
CHUNK_DIR=/data/mozaik_chunk \
OUTPUT_PREFIX=/data/mozaik_data/trial \
DATASTORE_PREFIX=/data/<the results_dir the sim wrote to> \
  python -u export.py 0 --n-chunks 12                 # export trial 0
#   split a long export across jobs:  --chunk-start 0 --chunk-end 6   then   --chunk-start 6 --chunk-end 12
#   screen only:  --screen-only        spikes only:  --spikes-only     images only:  --modality-filter image
```

| Env / flag | Role |
|---|---|
| `CHUNK_DIR` | Directory of chunk JSONs (screen timeline needs **all** `N` chunks even for a spike subset). |
| `OUTPUT_PREFIX` | Shard dir per trial = `{OUTPUT_PREFIX}{trial}` → `responses/` + `screen/`. |
| `DATASTORE_PREFIX` | Dir to resolve datastores in; each is found by glob `SelfSustainedPushPull_trial{T}_chunk{C}_____*`. |
| `SHEET_NAMES` | Comma-separated sheet subset; unset = **all recorded sheets** (multi-sheet default). |
| `--n-chunks` / `--chunk-start` / `--chunk-end` | Which chunks to fold in (resume/append across jobs). |
| `--batch-size` | Chunks held in memory before flushing. |

> **Datastore resolution:** the export globs `SelfSustainedPushPull_trial{T}_chunk{C}_____*` under
> `DATASTORE_PREFIX`. If that dir has **two** matches for one `(trial, chunk)` it raises `Ambiguous
> datastore` — point `DATASTORE_PREFIX` at a dir with exactly one datastore per `(trial, chunk)`.

**Output:** `{OUTPUT_PREFIX}{trial}/responses/{spikes.npy, meta.yml}` (flat float64 spike seconds + N+1 CSR
`spike_indices`) and `{OUTPUT_PREFIX}{trial}/screen/{combined_meta.json, timestamps.npy, meta/*, data/*}`.

---

## Workflow 2 — inline (one job: simulate + export a single chunk)

`run.py … <run_name> --export` runs the sim, then (rank 0 only, after `data_store.save()`) exports the
just-simulated chunk to a **full shard next to the datastore**, reusing the same exporter library. No
separate export job; single chunk only (multi-chunk datasets → Workflow 1).

Use the Step-1 invocation above (same binds, `NRANKS`, and `mpirun`) and append `--export` to the `run.py`
line:

```bash
      python -u run.py nest "$NRANKS" param/defaults \
        results_dir "'"'"'/data/<fresh-out-dir>/'"'"'" \
        simulation_seed 1000 \
        trial0_chunk0_inline --export
```

- `--export` is stripped from `argv` before the mozaik CLI parses it, so it can sit at the end.
- The chunk it exports is the one `TRIAL`/`CHUNK`/`CHUNK_DIR` selected — same env as the sim.
- **Output:** `<datastore>/experanto/{responses,screen}` — identical shard format to Workflow 1.

---

## Environment knobs (the experiment driver)

`create_randomized_experanto(model)` (`mozaik-models/experanto/experiments.py`) reads:

| Env var | Default | Role |
|---|---|---|
| `TRIAL` | `0` | Trial index; selects the chunk JSON and (at launch) the noise seed. |
| `CHUNK` | `0` | Chunk index within the trial. Reads `{CHUNK_DIR}/{TRIAL}_{CHUNK}.json`. |
| `CHUNK_DIR` | `/data/mozaik_chunk` | Directory of chunk JSONs. |
| `BASE_PATH` | historical single-session dataset | Input Experanto **screen** dataset the stimuli are read from (e.g. a P4 subset from `materialize_subset_screen.py`). |
| `STIM_WIDTH` | `11` | Presented image's longest-axis extent in **degrees** (not the visual field). E2 Cadena runs use `6.7`. |

---

## The script chain

```
run.py                                   (mozaik-models/experanto — entry point)
  run_workflow("SelfSustainedPushPull", SelfSustainedPushPull, create_randomized_experanto)
      → build model
      → create_randomized_experanto: reads TRIAL/CHUNK/CHUNK_DIR/BASE_PATH/STIM_WIDTH
            → RandomizedExperanto(chunk_dict_path = {CHUNK_DIR}/{TRIAL}_{CHUNK}.json)   ← mozaik/experiments/vision.py
                  → generate_stimuli(): for each {file, trial} in the chunk,
                        _append_meta_stimulus(): image → pre-blank + image + 49 ms post-blank;  video → bare
      → present stimuli, record spikes → data_store.save()  (rank 0)
      → (if --export and rank 0) export_datastore_inline(...)                            ← Workflow 2 hook

export.py <trials> --n-chunks N          (mozaik-models/experanto — Workflow 1 driver)
      → run_experanto_export(...)         ← mozaik/mozaik/meta_workflow/experanto_export.py (generic driver)
            resolve datastore per (trial, chunk) by glob → load → exporter library
                  → mozaik/mozaik/tools/experanto_export.py  (MozaikTrialExporter + MozaikScreenExporter)
                        → Experanto shards under {OUTPUT_PREFIX}{trial}/
```

Both entry points call the **same** exporter library in the `mozaik` package — one export code path.

**Timing / sync invariant:** each image is `pre-blank → image (~497 ms) → 49 ms post-blank`; videos are
`num_frames × 35 ms`, bare. `POST_BLANK_MS = 49` is defined in `PixelMovieExperantoBase`
(`mozaik/experiments/vision.py`) and **mirrored** in the exporter — the spike and screen timelines share one
clock, so keep the two equal (`responses/meta.yml:end_time == screen/timestamps.npy[-1]`).

---

## Seeds (three-stream)

`param/defaults` uses three seeds, each driving its own RNG (`controller.py` → `mozaik.setup_mozaik_seeds`):

| Seed | Drives (RNG) | Governs | Varies |
|---|---|---|---|
| `model_seed=1023` | `model_rng` / `model_pynn_rng` | **network identity** — connectivity, neuron positions, weights, connector sampling | fixed across trials |
| `simulation_seed` | `simulation_rng` | **NEST kernel noise** (per-trial background) — must be **nonzero** | **override per trial** (nonzero) for independent noise, same network |
| `experiment_seed=0` | `experiment_rng` | **experiment-level RNG** — stimulus shuffling / random draws, in experiments that shuffle at runtime | fixed |

> **Stimulus order for this pipeline is NOT set by a runtime seed.** `RandomizedExperanto` presents the
> chunk **in the order written in the JSON**, with no runtime shuffle — so `experiment_seed` has no effect on
> the order here (it matters only for shuffling experiments like the tuning protocols). The order is fixed
> **offline** by `generate_chunks.py --seed` (Step 0), which shuffles per trial with `Random(seed + trial)`.
> Treat that as the pipeline's fourth, build-time seed.

Per-trial noise is set on the `run.py` CLI (`simulation_seed <n>`), **not** in the chunk JSON. The sim is
bit-reproducible under fixed seeds (same three seeds → identical spikes).

---

## Verify the export

- **PSTH notebook:** `mozaik-models/experanto/notebooks/verify_psth_export.ipynb` — point its config cell at
  a shard dir and Run All: stimulus-locking (§1–6) and export-vs-datastore PSTH parity (§7, build
  `datastore_psths.npz` first via `analysis/compute_psth_datastore.py`).
- **Sanity checks on the shard** (load the files with numpy + PyYAML — e.g. inside the container, since the
  bare host python may lack them):
  - **timelines agree:** `responses/meta.yml`'s `end_time` equals the last value of `screen/timestamps.npy`;
  - **spike index is well-formed:** `responses/meta.yml`'s `spike_indices` (the CSR offsets into
    `responses/spikes.npy`) has length `n_signals + 1`;
  - **counts are as expected:** the number of signals (neurons) and its per-sheet breakdown (`sheets`,
    `n_signals_layerwise`), and the number of `screen/data/*.npy` files match the stimuli you presented.
