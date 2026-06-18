# DVS Input for LSV1M

## Milestones

1. Create new visual cortical sheet with regular grid of neurons.
   - Status: Complete
2. Write new experiments that inject DVS events from a file into such a sheet,
   respecting DVS pixel visual-field positions.
   - Status: Complete
3. Replace the current visual input model in LSV1M with the new DVS input sheet.
   - Status: Implemented, awaiting approval

## Milestone 1 Plan: Regular Grid Visual Cortical Sheet

Status: Complete.

### Goal

Add a new visual cortical sheet class that places neurons on a regular 2D grid
instead of using the random uniform positions used by
`VisualCorticalUniformSheet`. This sheet will later support DVS pixel-to-neuron
mapping.

### Implementation

- Add `VisualCorticalGridSheet` in `mozaik/sheets/vision.py`.
- Reuse the existing `VisualCorticalUniformSheet` implementation as much as
  possible, changing only the structure and neuron count calculation required
  for regular-grid placement.
- Keep a single `density` parameter. The grid will use the same linear density
  in x and y, so separate x/y count parameters are not needed.
- Compute the regular grid from the sheet size in cortical space and
  `density`, then create a PyNN `space.Grid2D` structure.
- Keep positions in Mozaik's existing visual-field coordinate convention.
- Preserve the existing `SheetWithMagnificationFactor` conversion API:
  `size_in_degrees()`, `vf_2_cs()`, and `cs_2_vf()`.
- Do not change the behavior of existing random visual cortical sheets.

### Validation Conditions

- A grid sheet constructs successfully with the same cell parameter shape used
  by existing visual cortical sheets.
- The number of neurons is exactly the computed x count times the computed y
  count.
- The generated positions have a regular lattice: constant spacing along x and
  constant spacing along y.
- The generated positions are centered around `(0, 0)` and remain inside the
  visual-field extent.
- The x and y grid densities are the same for square and non-square sheets.
- Existing `VisualCorticalUniformSheet` behavior remains unchanged.

### Test Plan

- Add focused pytest coverage under `tests/sheets/`.
- Test a square grid sheet to verify neuron count, unique x/y coordinates,
  spacing, centering, and bounds.
- Test a rectangular grid sheet to verify one density parameter produces equal
  x/y spacing while allowing different x/y counts.
- Test that existing `VisualCorticalUniformSheet` can still be constructed and
  retains non-grid/random placement semantics.
- Run the focused new tests and relevant existing sheet tests.

## Milestone 2 Plan: DVS Event Experiments

Status: Complete.

### Goal

Add an experiment path that reads DVS events from a `.npy` file and drives ON
and OFF regular-grid input sheets so the sheet neurons themselves output the
corresponding spikes.

### Event File Format

The event file is a numeric NumPy array with shape `(n_events, 4)`.
Columns are:

1. `time_ms`
2. `x`
3. `y`
4. `polarity`

Positive polarity events are routed to the configured ON sheet. Negative
polarity events are routed to the configured OFF sheet.

### Implementation

- Add helpers to load and validate `.npy` DVS event files.
- Add exact pixel-to-grid mapping. The DVS pixel grid must match the neuron
  grid; no scaling or interpolation is performed.
- Add a direct stimulator that sets `spike_times` on the target sheet
  population itself. Existing stimulators such as `Kick` and
  `BackgroundActivityBombardment` create separate spike-source populations and
  project into target cells, which does not satisfy this milestone.
- Require ON/OFF sheets to be `SpikeSourceArray`-compatible populations.
- Add an experiment class that receives parameters identifying:
  - `event_file`
  - `duration`
  - `on_sheet_name`
  - `off_sheet_name`
  - DVS/grid dimensions
- Use simulator-time offsetting when the direct stimulator prepares each
  presentation, so consecutive experiments schedule absolute spike times
  correctly.

### Validation Conditions

- `.npy` parsing preserves event times, pixel coordinates, and polarity.
- Invalid files fail clearly: wrong shape, unsupported polarity, out-of-range
  pixels, non-integer pixels, events outside duration, or mismatched grid size.
- Pixel-to-neuron mapping is exact and deterministic.
- Positive events are assigned only to the ON sheet and negative events only to
  the OFF sheet.
- Per-neuron spike times are sorted, offset by simulator time when prepared,
  and limited to the experiment duration.
- A small PyNN/NEST simulation with consecutive DVS experiments records the
  same ON/OFF spike times that were scheduled, proving offsetting is correct.

### Test Plan

- Unit tests for `.npy` event parsing and validation.
- Unit tests for exact pixel-to-grid index mapping on small rectangular grids.
- Unit tests for polarity split and per-neuron spike-time grouping.
- Unit tests for simulator-time offsetting on a fake spike-source sheet.
- A focused NEST-backed pytest that:
  - builds tiny ON and OFF regular-grid sheets using `SpikeSourceArray`,
  - records spikes from both sheets,
  - prepares and runs two consecutive DVS event stimulations with `reset=False`,
  - asserts recorded ON/OFF spike times match the input events with the correct
    offset for the second presentation.

## Milestone 3 Plan: CNMSS DVS Model Run

Status: Implemented, awaiting approval.

### Goal

Replace the old Retina/LGN visual input in the trimmed CNMSS model with direct
DVS-driven ON and OFF grid sheets, run a short DVS simulation, and generate a
minimal first-pass analysis/visualization.

### Implementation

- Remove the `retina_lgn` input-layer dependency from `mozaik-models/CNMSS`.
- Add two normal model sheets named `X_ON` and `X_OFF` directly in
  `SelfSustainedPushPull`.
- Use `VisualCorticalGridSheet` with `SpikeSourceArray` cells for both sheets.
- Keep the existing `GaborConnector` afferent path by passing the direct
  `X_ON` and `X_OFF` sheet objects to it.
- Add CNMSS parameter files for the ON and OFF DVS sheets.
- Add top-level DVS experiment parameters:
  - source dataset directory,
  - short dataset directory,
  - event duration,
  - DVS width and height.
- Replace the old visual experiments in CNMSS with one `DVSRecordedInput`
  experiment targeting `X_ON` and `X_OFF`.
- Create one short IIT/YARP-format DVS dataset in the CNMSS directory for the
  initial run. This is a one-off dataset artifact; no reusable shortening
  script will be added.
- Add a minimal CNMSS `analysis_and_visualization.py` that produces basic
  recorded-spike summaries and simple plots for the DVS run.

### Validation Conditions

- `SelfSustainedPushPull` builds direct sheets named `X_ON` and `X_OFF`.
- The two input sheets use `SpikeSourceArray` cells.
- The generated input grids match the DVS pixel grid exactly.
- The existing afferent `GaborConnector` path still connects `X_ON`/`X_OFF` to
  L4 excitatory and inhibitory sheets.
- `create_experiments()` returns a `DVSRecordedInput` experiment with the
  configured DVS dimensions and sheet names.
- The one-off short dataset loads through the same `bimvee`/DVS loader path as
  the full dataset.
- A first short CNMSS simulation completes, writes a datastore, and produces
  the minimal analysis/visualization outputs.

### Validation Approach

- Use temporary internal checks while developing the model package; do not add
  repo pytests for `mozaik-models/CNMSS`.
- Run focused import/build checks for the CNMSS parameters and model.
- Run one short DVS simulation with the short dataset.
- Inspect the datastore/plots enough to confirm input and V1 activity were
  recorded.
