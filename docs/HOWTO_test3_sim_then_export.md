# HOWTO — test3: simulation → chunk export → merge

This small worked example uses **3 trials × 1 chunk**, with two images and one
video per trial. Each simulation saves its datastore and exports its chunk;
a merge then produces the final shard for that trial. See the
[Randomized Experanto HOWTO](HOWTO_randomized_experanto.md) for installation,
container setup and Slurm integration.

## Prepare the fixture

Run from the Mozaik repository root in an environment with Mozaik and PyYAML.
The input stimuli are bundled in [test3_fixture/](test3_fixture/):

```text
docs/test3_fixture/
  screen/meta.yml
  screen/meta/{09943,11353,01911}.yml
  screen/data/{09943,11353,01911}.npy
  MANIFEST.txt
```

Generate the three chunk lists:

```python
from mozaik.tools.experanto_chunks import generate_chunks

generate_chunks(
    data_root="docs/test3_fixture",
    output_dir="docs/test3_fixture/mozaik_chunk_test3",
    n_trials=3,
    n_chunks=1,
    seed=42,
)
```

This writes `0_0.json`, `1_0.json` and `2_0.json`, plus
`chunk_settings.json`. Use paths visible to the simulation process when
configuring the model's experiment driver:

| Setting | Value |
|---|---|
| `BASE_PATH` | Path to `docs/test3_fixture` |
| `CHUNK_DIR` | Path to `docs/test3_fixture/mozaik_chunk_test3` |
| `TRIAL` | `0`, `1` or `2`, one simulation job each |
| `CHUNK` | `0` |

## Simulate and export

The companion model entry point supplies the model and experiment factory.
Pass a chunk export request to `run_workflow` on every MPI rank:

```python
import os

from mozaik.controller import run_workflow
from mozaik.meta_workflow.experanto_export import (
    ExperantoChunkExport,
    ExperantoExportSettings,
)

trial = int(os.environ["TRIAL"])
request = ExperantoChunkExport(
    trial_id=trial,
    chunk_id=0,
    chunk_path=os.path.join(os.environ["CHUNK_DIR"], f"{trial}_0.json"),
    settings=ExperantoExportSettings(export_blank_spikes=True),
)
run_workflow(
    "SelfSustainedPushPull", Model, create_randomized_experanto,
    experanto_export=request,
)
```

Here `Model` and `create_randomized_experanto` come from the companion model
project. A launcher can also supply `request.job_environment()` to each job;
`run_workflow` consumes that request automatically.

Set a writable `results_dir` and distinct run names such as `trial0_chunk0`.
Use the same `model_seed` across trials and a nonzero `simulation_seed` per
trial. `experiment_seed` does not change the order in the chunk JSON.
Set `model.parameters.null_stimulus_period == 0`. For MPI runs, match the
`mpirun -n` count to the `nest N` argument.

After saving the datastore, MPI root exports into `<datastore>/experanto/`.
Each completed chunk export contains `responses/`, `screen/`, `meta.json`
and `chunk_export.json`. Use the request's `output_dir` to select a separate
writable directory.

## Merge each trial

Once the simulations have completed, merge each single-chunk trial:

```python
from mozaik.meta_workflow.experanto_export import merge_experanto_trial

for trial in range(3):
    merge_experanto_trial(
        trial_id=trial,
        datastore_prefix="test3/results",
        chunk_end=1,
        output_dir=f"test3/shards/trial{trial}",
    )
```

Set `datastore_prefix` to the simulation's `results_dir`; it must contain one
matching datastore per trial/chunk. Each output directory must be new.
For Slurm, use `submit_experanto_merge_job` with the trial's simulation job
IDs to schedule the merge with `afterok` dependencies, as shown in the
[backend integration example](HOWTO_randomized_experanto.md#backend-integration).

## Verify the result

Check each `test3/shards/trial{trial}/` using NumPy, PyYAML and JSON:

- `responses/meta.yml:end_time` equals the final `screen/timestamps.npy` value.
- `spike_indices` has `n_signals + 1` entries, starts at zero and ends at
  the length of `responses/spikes.npy`.
- Sheet counts agree with the configured model, and screen entries cover the
  two images, video and explicit blanks.

For stimulus-locked response checks, point the companion project's
`notebooks/verify_psth_export.ipynb` at the final shards.
