# HOWTO — run the **test3** pipeline: simulation → Experanto export

> **This is the small, concrete worked example** for the general
> [`HOWTO_randomized_experanto.md`](HOWTO_randomized_experanto.md) — the same Randomized Experanto
> experiment, fixed at **3 trials × 1 chunk**. Read the general HOWTO for the full picture; use this to get
> a first end-to-end run working. The bundled fixture that makes it turnkey is in
> [`test3_fixture/`](test3_fixture/) (see "Get the data" below).
>
> **`cluster/` is local-only.** The `cluster/submit.sh` flow shown below is a convenience wrapper that is
> **specific to this cluster and not tracked/pushed**. The portable, in-tree path is the direct `run.py` /
> `export.py` invocation (Workflow 2, and the general HOWTO).

test3 is the small end-to-end case: **3 trials × 1 chunk** = 2 images + 1 video per trial.

There are **two** ways to produce Experanto shards from it:

| | **Workflow 1 — canonical (multi-chunk)** | **Workflow 2 — inline (single-chunk)** |
|---|---|---|
| Shape | sim job(s) **then** a separate export job | sim **and** export in one process |
| Use when | any real dataset (≥1 chunk/trial concatenated per trial) | a single-chunk run you want exported immediately |
| Entry | `export.py` driver (walks chunks `0..N-1`) | `run.py --export` (exports the one chunk it simulated) |
| Output | shards under `OUTPUT_PREFIX` | shard **next to the datastore** (`<datastore>/experanto/`) |

Commands run from the **mozaik repo root**: `MOZAIK-new/mozaik/`.

> **Branch:** the live `mozaik/` checkout is **`csng-mozaik-update`** (seed_refactor + #5 + #6). The
> `param/defaults` seed keys are on the three-seed schema (see "Seed scheme" below).

---

## Get the data (turnkey fixture)

test3 only touches **3 stimuli** (the same set every trial): image `09943`, image `11353`, video `01911`.
The **raw stimuli** — 3 meta YMLs + 3 `.npy` files, **2.7 MB total** — are bundled at
[`docs/test3_fixture/`](test3_fixture/). The chunk lists are **not** bundled: they are a *derived* artifact,
generated on first run from these stimuli (step below).

```
docs/test3_fixture/
  screen/meta.yml                       # dataset-level meta
  screen/meta/{09943,11353,01911}.yml   # stimulus metadata
  screen/data/{09943,11353,01911}.npy   # stimulus pixels
  MANIFEST.txt                          # source path + sha256 of every file
```

**Step 0 — generate the chunk lists once** (pure Python; run on the host, no container needed):

```bash
python ../mozaik-models/experanto/generate_chunks.py \
    --data-root docs/test3_fixture \
    --output-dir docs/test3_fixture/mozaik_chunk_test3 \
    --n-trials 3 --n-chunks 1
#   -> docs/test3_fixture/mozaik_chunk_test3/{0,1,2}_0.json   (gitignored — regenerable)
```

Then point the run at the fixture (no need for the full 6.8 GB dataset):

```bash
FIX=$PWD/docs/test3_fixture            # -> bind into the container as /fixture
#   BASE_PATH = /fixture               (the experiment reads /fixture/screen/{meta,data})
#   CHUNK_DIR = /fixture/mozaik_chunk_test3
```

> The freshly generated chunk **order** differs from the historical test3 (a different shuffle) — fine for a
> smoke run, but the P1 golden export is byte-reproduced only from its own committed launchers, not this
> fixture.

The full production dataset lives at
`/data/test_upsampling_without_hamming_30.0Hz/dynamic26872-17-20-Video-…_30hz`; the historical chunk JSONs
are at `/data/MOZAIK/mozaik_chunk_test3/`. The fixture is the 3-stimulus slice of that, so a fresh checkout
runs without hunting for paths.

---

## TL;DR

```bash
cd "$MOZAIK_ROOT"          # your Mozaik repo root

# ── Step 0: generate the chunk lists from the bundled fixture (once) ──
python ../mozaik-models/experanto/generate_chunks.py \
    --data-root docs/test3_fixture --output-dir docs/test3_fixture/mozaik_chunk_test3 \
    --n-trials 3 --n-chunks 1

# ── Workflow 2 (recommended first run): inline, one job simulates + exports a single chunk ──
#   uses the bundled fixture; no cluster/ tooling required (see "Workflow 2")

# ── Workflow 1: canonical, two submissions (local cluster/ wrapper) ──
./cluster/submit.sh cluster/experiments/sim-test3.conf       # 1) SIMULATION (array 0-2 = trial 0,1,2)
./cluster/submit.sh cluster/experiments/export-test3.conf    # 2) EXPORT (after the sim array COMPLETES)
#   add --dry-run to print the sbatch command without submitting; logs land in slurm_test3/.
```

> ⚠️ For Workflow 1, **read "Gotcha: datastore location" before the export** — the default sim and
> export confs point at *different* datastore directories.

> 💡 Just want to prove the pipeline works **fast**? See **"Fast smoke run (reduced scale)"** below —
> cut the cortical `density` ~10× and use an images-only chunk to skip the slow video/LGN filtering.

---

## Prerequisites (both workflows)

- **Repos / checkouts** (bound into the container by the compose scripts, `$PWD`-relative):
  - `mozaik/` (this repo, on `csng-mozaik-update`, → `/mozaik`)
  - `mozaik-models/experanto/` (→ `/project`; holds `run.py`, `export.py`)
  - `experanto/` sibling at `../../experanto` (→ `/experanto`)
  - `/mnt/vast-react/projects/neural_foundation_model` (→ `/data`), or the bundled `docs/test3_fixture/`.
- **SIF:** a `freeze_time`-capable image (pyNN 0.13.0) — confirm the blessed image in
  `mozaik-models/experanto/experiments/LOG.md` (e.g. `mozaik-sif/mozaik-opt-qpatch_2026-08-20.sif`).
- **Chunk files:** `{0,1,2}_0.json` — per-trial stimulus lists, **generated** from the fixture in Step 0
  (or the historical set at `/data/MOZAIK/mozaik_chunk_test3/`), referenced by `CHUNK_DIR`.

---

## Workflow 1 — canonical (sim job → export job)

*Uses the local-only `cluster/` wrapper; see the closing note. For a tooling-free run use Workflow 2.*

### Step 1 — Simulation
```bash
./cluster/submit.sh cluster/experiments/sim-test3.conf
```
`sim-test3.conf`: `WORKLOAD=sim`, `ARRAY=0-2`, `NTASKS=12`, `N_CHUNKS=1`,
`CHUNK_DIR=/data/MOZAIK/mozaik_chunk_test3`, `medium96s`/`sapphirerapids`, `TIME=02:00:00`. ≈12 min/task.

**Array index → work:** `TRIAL = idx / N_CHUNKS`, `CHUNK = idx % N_CHUNKS` → idx 0/1/2 = trial 0/1/2, chunk 0.
**Output:** one datastore per task, `SelfSustainedPushPull_trial{T}_chunk0_____<seed-key>:<n>`, in the
experanto root (`/project`) unless `RESULTS_DIR`/`WORKSPACE` redirect it.

> ⚠️ The sim runner still passes the old seed key — see "Seed scheme" for the one-line runner fix.

### Step 2 — Export (after the sim array is COMPLETED)
```bash
SIF_IMAGE=$PWD/../mozaik-sif/mozaik-opt-qpatch_2026-08-20.sif \
  ./cluster/submit.sh cluster/experiments/export-test3.conf
```
`export-test3.conf`: `WORKLOAD=export`, `ARRAY=0-2`, `NTASKS=1`, `N_CHUNKS=1`, `CHUNK_START=0`,
`CHUNK_END=1`, `BATCH_SIZE=1`, `CHUNK_DIR=/data/MOZAIK/mozaik_chunk_test3`,
`OUTPUT_PREFIX=/data/mozaik_data_test3/trial`, `DATASTORE_PREFIX=1_TEST3_EXPT`.

**Output:** Experanto shards at `/data/mozaik_data_test3/trial{0,1,2}/{responses,screen}` —
`responses/{spikes.npy, meta.yml}` + `screen/{combined_meta.json, timestamps.npy, meta/*, data/*}`.

*Validated 2026-08-04 on csng: byte-identical to the reference export (see `experiments/LOG.md`).*

---

## Workflow 2 — inline (one job: simulate + export a single chunk)

`run.py --export` runs the sim, then (on **rank 0 only**, after `data_store.save()`) exports the just-
simulated chunk to a **full Experanto shard next to the datastore**, reusing the same exporter library. No
separate export job, no `cluster/` tooling — this is the portable path.

Invocation (e.g. on a compute node), using the bundled fixture:

```bash
cd "$MOZAIK_ROOT"   # your Mozaik repo root ($PWD, below, must be this dir)
module load apptainer
SIF=$PWD/../mozaik-sif/mozaik-opt-qpatch_2026-08-20.sif   # freeze_time (pyNN 0.13.0) image; confirm in the LOG
FIX=$PWD/docs/test3_fixture
OUT=$HOME/scratch/test3_out; mkdir -p "$OUT"   # any writable dir — bound as /data for the shard output
NRANKS=2                                        # = physical cores you have; one MPI rank per physical core

apptainer exec --cleanenv \
  --env PYTHONPATH=/mozaik --env OMPI_MCA_orte_tmpdir_base=/tmp \
  --env TRIAL=0 --env CHUNK=0 --env CHUNK_DIR=/fixture/mozaik_chunk_test3 \
  --env BASE_PATH=/fixture --env NRANKS=$NRANKS \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
  --bind "$PWD:/mozaik" --bind "$PWD/../mozaik-models/experanto:/project" \
  --bind "$PWD/../../experanto:/experanto" \
  --bind "$FIX:/fixture" \
  --bind "$OUT:/data" \
  "$SIF" bash -lc '
    unset HTTP_PROXY HTTPS_PROXY FTP_PROXY http_proxy https_proxy ftp_proxy
    cd /project
    mpirun -n "$NRANKS" --oversubscribe \
      -x OMP_NUM_THREADS -x MKL_NUM_THREADS -x OPENBLAS_NUM_THREADS -x PYTHONPATH \
      -x TRIAL -x CHUNK -x CHUNK_DIR -x BASE_PATH -x NRANKS \
      python -u run.py nest "$NRANKS" param/defaults \
        results_dir "'"'"'/data/t3out/'"'"'" \
        simulation_seed 1000 \
        trial0_chunk0_inline --export
  '
```

Notes:
- **`--bind "$PWD:/mozaik"` binds the repo ROOT** (not `$PWD/mozaik`). With `PYTHONPATH=/mozaik`, `import mozaik`
  resolves to `/mozaik/mozaik`; binding the package dir instead makes `import mozaik` fail. This is the single
  most common way to get the run wrong.
- **`/data` is your output** here — the stimuli come from `/fixture`, so bind any writable dir as `/data`
  (`--bind "$OUT:/data"`) and the shard lands under it (`results_dir '/data/t3out/'`).
- **Ranks:** `NRANKS` drives both `mpirun -n` and `nest N` (they must match); set it to your **physical** core
  count, one rank per core (hyperthreads don't help NEST). `--oversubscribe` is here because a bare interactive
  node reports one slot; on a proper SLURM task allocation you can drop it.
- **`results_dir` must be a quoted Python-string literal** — mozaik `eval()`s override values, so a bare
  `/data/...` fails. Keep the `"'"'"'…'"'"'"` wrapper exactly as shown.
- `simulation_seed` **must be nonzero** (NEST rejects `rng_seed=0`); it seeds the per-trial noise.
- **Harmless:** you will see `CMake … CMAKE_CXX_COMPILER not set` and a new `_build/` dir — ignore them. Most
  of the wall time is retina/LGN filtering (independent of network size), so be patient on long stimuli.
- **Output:** `/data/t3out/<datastore>/experanto/{responses,screen}` — same shard format as Workflow 1.

*Validated 2026-08-04 on csng (sim 583 s, 12 ranks): shard structure + timeline invariant match the
reference; spikes differ (new `simulation_seed`), as designed. See `experiments/LOG.md`.*

---

## Fast smoke run (reduced scale) — optional

To prove the pipeline **mechanics** quickly (network → sim → export → a structurally valid, timeline-aligned
shard) without waiting on a full-scale run, shrink two things.

> ⚠️ **Smoke-test only.** The spikes and counts from a reduced run are **not scientifically meaningful** and
> must **not** be used for the P1 golden, reproduction, or any result. Both edits below are local — revert
> them before any real run.

**1. Shrink the network (faster build + NEST sim).** Every cortical sheet size derives from a single knob:
`density` in `mozaik-models/experanto/param/l4_cortex_exc` (the inhibitory and L2/3 sheets reference it via
`ref()`). Cut it ~10×:

```
# mozaik-models/experanto/param/l4_cortex_exc
'density': 150.0,   # was 1500.0  ->  exc sheets ~3,750 neurons (was ~37,500), inh ~937
```

Keep the reduction moderate (≈10×) so the populations stay large enough for the connectors to sample. The
retina/LGN sheets are grid-based and unaffected (still ~7,200 each).

**2. Drop the video (removes the real bottleneck).** Most of the wall time is retina/LGN filtering, which is
**independent of network size** and is dominated by long videos (~0.55 s/frame — the fixture's 300-frame clip
alone is minutes). For a quick run, use an images-only chunk instead of the generated one:

```bash
mkdir -p docs/test3_fixture/chunk_img_only
printf '%s\n' '[{"modality":"image","file":"09943.yml","trial":0},{"modality":"image","file":"11353.yml","trial":0}]' \
  > docs/test3_fixture/chunk_img_only/0_0.json
# then in the Workflow 2 invocation, point CHUNK_DIR at it:  --env CHUNK_DIR=/fixture/chunk_img_only
```

Run **Workflow 2** as above with those two changes (and `NRANKS` = your physical cores). What to expect: on a
small 2-core interactive alloc this completed in **~13–21 min** and produced a valid multi-sheet shard; on a
full CPU-partition allocation it is a few minutes. The full-scale run (full `density` + the video) is much
longer — network build and video filtering dominate.

**A successful smoke run (this exact fixture, `density=150`, images-only, `simulation_seed 1000`) yields:**
6 sheets `[X_ON, X_OFF, V1_Exc_L4, V1_Inh_L4, V1_Exc_L2/3, V1_Inh_L2/3]` with `n_signals_layerwise =
[7200, 7200, 3750, 937, 3750, 937]` (23,774 signals), **7 screen entries**, `end_time ≈ 1.946 s`, and the
**timeline invariant holds** (`responses/meta.yml:end_time == screen/timestamps.npy[-1]`). Use these as a
concrete target (the run is deterministic under fixed seeds, so the counts reproduce exactly).

---

## The complete script chain (Workflow 1)

```
./cluster/submit.sh cluster/experiments/<sim|export>-test3.conf     ← local-only wrapper (not pushed)
    reads the conf → sbatch CLI flags; exports MOZAIK_CONF=<abs conf path>
        ▼
sbatch --array=… --export=ALL,MOZAIK_CONF=… cluster/run_job.sh     (bodiless: directives arrived as flags)
        ▼
cluster/run_job.sh
    source $MOZAIK_CONF · module load apptainer · cd mozaik root · set OMP/MKL threads ·
    TRIAL = idx / N_CHUNKS, CHUNK = idx % N_CHUNKS (export: TRIAL = SLURM_ARRAY_TASK_ID) · dispatch on $WORKLOAD
```

### `WORKLOAD=sim`
```
cluster/apptainer-compose-array.sh   (binds mozaik→/mozaik, mozaik-models/experanto→/project, ../../experanto→/experanto, data→/data)
    ▼  apptainer exec <SIF>
cluster/runners/mozaik-simulation-array.sh          (cd /project)
    RUN_NAME = trial{T}_chunk{C}
    mpirun -n $NTASKS  python -u run.py  nest $NTASKS  param/defaults  simulation_seed <n>  "$RUN_NAME"
    ▼
mozaik-models/experanto/run.py
    run_workflow("SelfSustainedPushPull", …, create_randomized_experanto)
      → build model → present CHUNK_DIR/{trial}_{chunk}.json stimuli → save MOZAIK datastore
      → (if --export and rank 0) export_datastore_inline(...)          ← Workflow 2 hook
```

### `WORKLOAD=export`
```
cluster/apptainer-compose-export.sh                 (same binds; apptainer exec <SIF>)
    ▼
cluster/runners/mozaik-export-array.sh              (cd /project)
    python -u export.py $TRIAL --n-chunks $N_CHUNKS --batch-size $BATCH_SIZE [--chunk-start/-end] […]
    ▼
mozaik-models/experanto/export.py                   ← project DRIVER (argparse, trial/chunk loop)
    resolve datastore under DATASTORE_PREFIX (glob, any seed scheme); load each; call the exporters
    ▼
mozaik/mozaik/meta_workflow/experanto_export.py  →  mozaik/mozaik/tools/experanto_export.py   ← export LIBRARY (#6)
    MozaikTrialExporter + MozaikScreenExporter  →  Experanto shards under OUTPUT_PREFIX
```

Workflow 2's `run.py` imports the **same library** directly — one export code path, two entry points.

**Runtime env knobs** (set in the conf; both compose scripts pass them through as `--env`):
`SIF_IMAGE`, `PARAM_FILE` (default `param/defaults`), `CHUNK_DIR`, `BASE_PATH` (sim input screen dataset),
`RESULTS_DIR`/`WORKSPACE` (sim: redirect datastore to a bound `/ws`), `N_CHUNKS`, `CHUNK_START`/`CHUNK_END`,
`BATCH_SIZE`, `EXPORT_MODE` (`--screen-only`/`--spikes-only`), `MODALITY_FILTER`, `OUTPUT_PREFIX`,
`DATASTORE_PREFIX`.

---

## ⚠️ Gotcha: datastore location (Workflow 1: sim output vs. export input)

The default confs point at **different** datastore directories:

- **Sim** writes to the experanto root `/project` (no `RESULTS_DIR` in `sim-test3.conf`).
- **Export** reads `DATASTORE_PREFIX=1_TEST3_EXPT`, i.e. `/project/1_TEST3_EXPT/` — where an older,
  committed set of test3 datastores lives.

So a fresh sim's datastores land where the export won't look. Reconcile before exporting: set the export's
`DATASTORE_PREFIX` to wherever the sim actually wrote (or `.`/empty for the experanto root). The export
resolves each datastore by globbing `SelfSustainedPushPull_trial{T}_chunk{C}_____*`; if that dir has **two**
matches for the same trial/chunk it raises `Ambiguous datastore` — point `DATASTORE_PREFIX` at a dir with
exactly one datastore per trial/chunk. (Workflow 2 sidesteps this entirely — it exports the datastore it
just wrote.)

---

## Seed scheme (three-stream)

`param/defaults` uses three seeds (`controller.py` requires all three):
- `model_seed=1023` — **network identity** (connectivity, positions, weights, sampling); fixed across trials.
- `simulation_seed=1` — **per-trial noise** (NEST kernel); **must be nonzero** (NEST rejects `rng_seed=0`);
  override it per trial for independent background noise on the same network.
- `experiment_seed=0` — experiment-level RNG (stimulus shuffling in shuffling experiments). `RandomizedExperanto`
  presents the chunk in fixed order, so it has **no effect** on order here — the order is set offline by
  `generate_chunks.py --seed`.

Set the per-trial noise directly on the `run.py` CLI (`simulation_seed <n>`, e.g. `$(( TRIAL*1000 + CHUNK + 1 ))`).
The export resolves datastores by glob and is seed-scheme agnostic. Runs are bit-reproducible under fixed seeds.

---

## Verify the export

> ⚠️ The exported `.npy`/`.yml`/`.json` are plain files, but loading them needs **numpy + PyYAML**, which the
> cluster login/host python usually does **not** have (`ModuleNotFoundError: numpy`). Run the checks **inside
> the SIF** (`apptainer exec "$SIF" python - <<'PY' … PY`) or the project conda env — not the bare host python.

- **PSTH notebook:** `mozaik-models/experanto/notebooks/verify_psth_export.ipynb` — point its config cell at
  the shard dir (`N_TRIALS=3`) and Run All: §1–6 check stimulus-locking; §7 compares export PSTHs vs
  datastore PSTHs (build `datastore_psths.npz` first via `analysis/compute_psth_datastore.py`).
- **Structural / equivalence checks:** compare `responses/spikes.npy` (flat float64 s),
  `responses/meta.yml` CSR `spike_indices` (N+1), `screen/timestamps.npy`, `screen/combined_meta.json`, and
  `screen/data/*.npy` against a reference; assert the invariant `meta.yml:end_time == timestamps.npy[-1]`.
  Exact spike equality for the exporter is also covered by `mozaik tests/tools/test_experanto_export.py`.
