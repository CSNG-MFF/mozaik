"""
Tests for the Experanto spike export (mozaik.tools.experanto_export.MozaikTrialExporter).

Uses synthetic, duck-typed DataStoreView / segment objects with KNOWN spike trains, so the
export can be driven and checked end-to-end without running a model. Verifies:

  a) the output format contract (meta.yml keys, CSR N+1 spike_indices, timeline end_time), and
  b) that the exported spikes are IDENTICAL to those "stored in the datastore" (the synthetic
     spike trains), with the documented per-segment offsetting and < duration windowing.

Also checks that the new stim_name_key / group_by_key parameters let the user select and name
stimuli by parameters other than movie_name / trial, with the defaults reproducing the old
behaviour.
"""

import os

import numpy as np
import pytest
import yaml

from mozaik.tools.experanto_export import MozaikTrialExporter


class _Seg:
    """Minimal stand-in for a Mozaik/neo segment as the exporter consumes it."""

    def __init__(self, stim, trains=None, sheet_name=None):
        self.annotations = {"stimulus": stim, "sheet_name": sheet_name}
        self._trains = trains or []

    def get_spiketrains(self):
        return self._trains

    def release(self):
        pass


class _DSV:
    def __init__(self, segments):
        self._segments = segments

    def get_segments(self):
        return self._segments


def _reconstruct(spikes, spike_indices):
    """Split the flat spikes array back into per-unit arrays via the CSR indices."""
    return [
        spikes[spike_indices[i] : spike_indices[i + 1]]
        for i in range(len(spike_indices) - 1)
    ]


def _load(output_dir):
    spikes = np.load(os.path.join(output_dir, "spikes.npy"))
    with open(os.path.join(output_dir, "meta.yml")) as f:
        meta = yaml.safe_load(f)
    return spikes, meta


def test_trial_export_format_and_spikes_identical(tmp_path):
    # trial 0: image A (dur 100), blank (dur 50), image C (dur 80); trial 1: image D (must be excluded)
    segs = [
        _Seg(
            {"trial": 0, "movie_name": "imgA", "duration": 100},
            [np.array([10.0, 50.0, 150.0]), np.array([20.0])],  # 150 >= dur -> dropped
        ),
        _Seg({"trial": 0, "duration": 50}),  # blank (no movie_name)
        _Seg(
            {"trial": 0, "movie_name": "imgC", "duration": 80},
            [np.array([5.0, 79.0]), np.array([])],
        ),
        _Seg(
            {"trial": 1, "movie_name": "imgD", "duration": 100},
            [np.array([1.0, 2.0, 3.0]), np.array([4.0])],  # different trial -> excluded
        ),
    ]

    out = str(tmp_path / "responses")
    exp = MozaikTrialExporter(out, trial_id=0, sampling_rate=1000.0)
    exp.process_batch([_DSV(segs)])
    exp.finalize()
    spikes, meta = _load(out)

    # --- format contract ---
    assert meta["modality"] == "spikes"
    assert meta["n_signals"] == 2
    assert meta["start_time"] == 0.0
    assert meta["sampling_rate"] == 1000.0
    assert meta["trial_id"] == 0
    assert meta["stimuli_order"] == ["imgA", "blank", "imgC"]
    idx = meta["spike_indices"]
    assert len(idx) == meta["n_signals"] + 1  # CSR N+1
    assert idx[0] == 0 and idx[-1] == len(spikes)
    assert all(idx[i] <= idx[i + 1] for i in range(len(idx) - 1))  # monotonic
    # end_time == cumulative duration of ALL trial-0 segments (incl. blank): 100+50+80 = 230 ms
    assert meta["end_time"] == 0.230

    # --- spikes identical to the datastore trains (offset per segment, windowed to < duration) ---
    # unit0: imgA [10,50] @off 0 ; imgC [5,79] @off 150  -> [10,50,155,229] ms
    # unit1: imgA [20] @off 0                             -> [20] ms
    u0, u1 = _reconstruct(spikes, idx)
    np.testing.assert_allclose(u0, np.array([10.0, 50.0, 155.0, 229.0]) / 1000.0)
    np.testing.assert_allclose(u1, np.array([20.0]) / 1000.0)


def test_custom_group_and_name_keys(tmp_path):
    # Group by a string "phase" (exercises the non-int equality path) and name by "label".
    segs = [
        _Seg(
            {"phase": "A", "label": "s1", "duration": 100},
            [np.array([10.0]), np.array([])],
        ),
        _Seg(
            {"phase": "B", "label": "s2", "duration": 100},
            [np.array([11.0]), np.array([])],
        ),
        _Seg(
            {"phase": "A", "label": "s3", "duration": 100},
            [np.array([12.0]), np.array([])],
        ),
    ]
    out = str(tmp_path / "responses")
    exp = MozaikTrialExporter(
        out,
        trial_id=0,
        group_by_key="phase",
        group_value="A",
        stim_name_key="label",
    )
    exp.process_batch([_DSV(segs)])
    exp.finalize()
    spikes, meta = _load(out)

    # only phase-A stimuli selected, named by "label"
    assert meta["stimuli_order"] == ["s1", "s3"]
    u0, _ = _reconstruct(spikes, meta["spike_indices"])
    # s1 [10] @off 0 ; s3 [12] @off 100 -> [10, 112] ms
    np.testing.assert_allclose(u0, np.array([10.0, 112.0]) / 1000.0)


def _sheet_segs():
    """Two sheets, 3 presentations each (imgA dur100, blank dur50, imgC dur80), block order
    presentation-major / sheet-minor — as a real Mozaik datastore stores per-(presentation, sheet).
    """
    L4, L23 = (
        "V1_Exc_L4",
        "V1_Exc_L2/3",
    )  # canonical order puts L4 (prio 2) before L23 (prio 4)
    return [
        # presentation 0: imgA (dur 100)
        _Seg(
            {"trial": 0, "movie_name": "imgA", "duration": 100},
            [np.array([10.0, 50.0, 150.0]), np.array([20.0])],
            sheet_name=L4,
        ),  # 150 >= dur dropped
        _Seg(
            {"trial": 0, "movie_name": "imgA", "duration": 100},
            [np.array([5.0]), np.array([]), np.array([7.0])],
            sheet_name=L23,
        ),
        # presentation 1: blank (dur 50) — per-sheet, no movie_name, no trains
        _Seg({"trial": 0, "duration": 50}, sheet_name=L4),
        _Seg({"trial": 0, "duration": 50}, sheet_name=L23),
        # presentation 2: imgC (dur 80)
        _Seg(
            {"trial": 0, "movie_name": "imgC", "duration": 80},
            [np.array([5.0, 79.0]), np.array([])],
            sheet_name=L4,
        ),
        _Seg(
            {"trial": 0, "movie_name": "imgC", "duration": 80},
            [np.array([1.0]), np.array([2.0]), np.array([80.0])],
            sheet_name=L23,
        ),  # 80 >= dur dropped
    ]


def test_multi_sheet_stacks_units_and_records_boundaries(tmp_path):
    out = str(tmp_path / "responses")
    exp = MozaikTrialExporter(
        out, trial_id=0, sampling_rate=1000.0
    )  # sheet_names=None -> all
    exp.process_batch([_DSV(_sheet_segs())])
    exp.finalize()
    spikes, meta = _load(out)

    # --- sheet layout metadata ---
    assert meta["sheets"] == ["V1_Exc_L4", "V1_Exc_L2/3"]  # canonical order
    assert meta["n_signals_layerwise"] == [2, 3]
    assert meta["sheet_unit_indices"] == [0, 2, 5]  # CSR boundaries
    assert meta["n_signals"] == 5  # grand total
    assert len(meta["spike_indices"]) == 6  # N+1
    # timeline advances ONCE per presentation (not per sheet): 100+50+80 = 230 ms
    assert meta["end_time"] == 0.230
    assert meta["stimuli_order"] == [
        "imgA",
        "blank",
        "imgC",
    ]  # one entry per presentation

    # --- each sheet's slice matches its own trains, offset per presentation ---
    units = _reconstruct(spikes, meta["spike_indices"])
    # V1_Exc_L4 unit0: imgA[10,50]@0 + imgC[5,79]@150 -> [10,50,155,229]
    np.testing.assert_allclose(units[0], np.array([10.0, 50.0, 155.0, 229.0]) / 1000.0)
    np.testing.assert_allclose(units[1], np.array([20.0]) / 1000.0)  # L4 unit1
    # V1_Exc_L2/3 (global units 2..4)
    np.testing.assert_allclose(
        units[2], np.array([5.0, 151.0]) / 1000.0
    )  # imgA[5]@0 + imgC[1]@150
    np.testing.assert_allclose(units[3], np.array([152.0]) / 1000.0)  # imgC[2]@150
    np.testing.assert_allclose(
        units[4], np.array([7.0]) / 1000.0
    )  # imgA[7]@0 (imgC[80] dropped)


def test_multi_sheet_subset_selection(tmp_path):
    out = str(tmp_path / "responses")
    exp = MozaikTrialExporter(
        out, trial_id=0, sampling_rate=1000.0, sheet_names=["V1_Exc_L2/3"]
    )
    exp.process_batch([_DSV(_sheet_segs())])
    exp.finalize()
    spikes, meta = _load(out)
    assert meta["sheets"] == ["V1_Exc_L2/3"]
    assert meta["n_signals"] == 3
    assert meta["n_signals_layerwise"] == [3]
    # subset still advances the timeline over all presentations
    assert meta["end_time"] == 0.230


def test_multi_sheet_unequal_counts_raise(tmp_path):
    # V1_Exc_L4 has 2 presentations, V1_Exc_L2/3 only 1 -> misalignment must be caught
    segs = [
        _Seg(
            {"trial": 0, "movie_name": "imgA", "duration": 100},
            [np.array([1.0])],
            sheet_name="V1_Exc_L4",
        ),
        _Seg(
            {"trial": 0, "movie_name": "imgC", "duration": 80},
            [np.array([2.0])],
            sheet_name="V1_Exc_L4",
        ),
        _Seg(
            {"trial": 0, "movie_name": "imgA", "duration": 100},
            [np.array([3.0])],
            sheet_name="V1_Exc_L2/3",
        ),
    ]
    exp = MozaikTrialExporter(str(tmp_path / "r"), trial_id=0, sampling_rate=1000.0)
    with pytest.raises(ValueError, match="Unequal segment counts"):
        exp.process_batch([_DSV(segs)])


def test_multi_sheet_missing_requested_sheet_raises(tmp_path):
    exp = MozaikTrialExporter(
        str(tmp_path / "r"),
        trial_id=0,
        sampling_rate=1000.0,
        sheet_names=["V1_Exc_L4", "NOT_A_SHEET"],
    )
    with pytest.raises(ValueError, match="not present"):
        exp.process_batch([_DSV(_sheet_segs())])
