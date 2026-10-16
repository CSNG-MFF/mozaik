import ast
import json
import math
import os
import shutil

import numpy as np
import psutil
import yaml

# Canonical export order for sheets when multiple are exported into one spikes.npy. Sheets not listed
# here are appended in sorted-name order, so unit indexing stays deterministic across trials/runs while
# still handling models that add new sheets. See docs/plan/updating-mozaik/2026-08-13_multi-sheet-export.md.
_SHEET_PRIORITY = [
    "X_ON",
    "X_OFF",
    "V1_Exc_L4",
    "V1_Inh_L4",
    "V1_Exc_L2/3",
    "V1_Inh_L2/3",
]


def canonical_sheet_order(found):
    """Order the discovered sheet names deterministically: known sheets by ``_SHEET_PRIORITY``, then
    any unknown sheet names in sorted order."""
    found = set(found)
    known = [s for s in _SHEET_PRIORITY if s in found]
    rest = sorted(s for s in found if s not in _SHEET_PRIORITY)
    return known + rest


def load_tier_reference(combined_meta_path):
    """Build a condition_hash → tier mapping from an existing combined_meta.json.

    When multiple entries share the same condition_hash, the first non-blank
    tier encountered wins. This enables passing a combined_meta.json to preserve
    train/validation/test splits from a source dataset in a Mozaik re-export.

    Parameters
    ----------
    combined_meta_path : str
        Path to a ``screen/combined_meta.json`` file.

    Returns
    -------
    dict
        Mapping of ``{condition_hash: tier}``.
    """
    with open(combined_meta_path, "r") as f:
        meta = json.load(f)
    ref = {}
    for entry in meta.values():
        ch = entry.get("condition_hash")
        tier = entry.get("tier")
        if ch and tier and ch not in ref:
            ref[ch] = tier
    return ref


def get_process_memory():
    """Returns current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def _validate_experanto_model_timing(dsv):
    """Reject Mozaik's unsupported automatic null-stimulus periods."""
    parameters = dsv.get_model_parameters()
    if isinstance(parameters, str):
        parameters = ast.parse(parameters, mode="eval").body
        null_stimulus_period = next(
            ast.literal_eval(value)
            for key, value in zip(parameters.keys, parameters.values)
            if key.value == "null_stimulus_period"
        )
    else:
        null_stimulus_period = parameters["null_stimulus_period"]

    if null_stimulus_period:
        raise ValueError(
            "Experanto export requires model.parameters.null_stimulus_period == 0 "
            "because Mozaik null-stimulus periods are not exported"
        )


class MozaikTrialExporter:
    """
    Export one chunk's presentations into a flat spike array, grouped by unit.

    Parameters
    ----------
    stim_name_key : str
        Stimulus parameter that names each non-blank stimulus in ``stimuli_order``. Explicit
        Experanto blanks are identified by the ``InternalStimulus`` class name. Default
        ``"movie_name"`` names ``PixelMovieExperanto`` stimuli.
    group_by_key : str
        Stimulus parameter used to select which segments are exported together. Default
        ``"trial"`` groups segments by trial; e.g. a long movie presented in chunks could
        instead be grouped by its per-chunk start time.
    group_value : optional
        Value of ``group_by_key`` selecting this export; defaults to ``trial_id``.
    export_blank_spikes : bool
        Include spikes recorded during explicit Experanto ``InternalStimulus`` blanks. Disabling
        this leaves blank timing and metadata intact while omitting their spikes. Default ``True``.
    """

    def __init__(
        self,
        output_dir,
        trial_id,
        sampling_rate=1000.0,
        smooth_param=None,
        stim_name_key="movie_name",
        group_by_key="trial",
        group_value=None,
        sheet_names=None,
        export_blank_spikes=True,
    ):
        self.output_dir = output_dir
        self.trial_id = trial_id
        self.sampling_rate = sampling_rate
        self.smooth_param = smooth_param
        # Which stimulus parameter names each stimulus, and which parameter (and value) selects
        # the segments to export together. Defaults select by trial and name by movie_name.
        self.stim_name_key = stim_name_key
        self.group_by_key = group_by_key
        self.group_value = trial_id if group_value is None else group_value
        self.export_blank_spikes = export_blank_spikes

        # Multi-sheet export: which sheets to fold into the single spikes.npy. None = every sheet with
        # recorded spiketrains in the DSV (discovered from the chunk). A list restricts to a subset.
        # Sheets are laid out contiguously in canonical_sheet_order(); the per-sheet unit boundaries are
        # recorded in meta.yml (`sheets`/`sheet_unit_indices`/`n_signals_layerwise`).
        self.sheet_names_requested = (
            list(sheet_names) if sheet_names is not None else None
        )
        self.sheet_names = None  # resolved (ordered) sheet list, set on first data
        self.sheet_unit_counts = None  # units per sheet, aligned with self.sheet_names
        self.sheet_unit_indices = None  # CSR boundaries, len == len(sheet_names)+1

        self.meta_segments = []
        self.all_unit_spike_lists = None
        self.num_units = 0
        self.current_time_offset = 0.0

        os.makedirs(self.output_dir, exist_ok=True)

        print(f"Exporter initialized. Target: {self.output_dir} (New Export)")

    def _in_group(self, raw):
        """Whether a segment whose ``group_by_key`` value is ``raw`` belongs to this export.

        Compares values as integers when possible, falling back to direct equality
        for values that cannot be converted to integers.
        A segment missing the grouping parameter (``raw is None``) is not selected.
        """
        if raw is None:
            return False
        try:
            return int(raw) == int(self.group_value)
        except (TypeError, ValueError):
            return raw == self.group_value

    def _parse_stimulus(self, seg):
        """Return the stimulus-parameter dict for a segment, or None on a parse error."""
        try:
            raw = seg.annotations["stimulus"]
            return ast.literal_eval(raw) if isinstance(raw, str) else raw
        except (ValueError, SyntaxError):
            print(f"Warning: Parse error for segment {seg}. Skipping.", flush=True)
            return None

    def _presentation_label(self, stim_params):
        """Return ``(label, is_blank)`` for an Experanto presentation."""
        if stim_params.get("name") == "InternalStimulus":
            return "blank", True

        label = stim_params.get(self.stim_name_key)
        if label is None:
            raise ValueError(
                "Non-blank stimulus %r has no %r parameter"
                % (stim_params.get("name"), self.stim_name_key)
            )
        return label, False

    def _bucket_by_sheet(self, dsv):
        """Group this chunk's (trial-matched) segments by sheet, preserving block/temporal order.

        Segments in a Mozaik datastore are stored one per (presentation, sheet); within a sheet the
        block order is the temporal presentation order. Returns an ``OrderedDict`` mapping
        ``sheet_name -> [(segment, stim_params), ...]`` in encounter order.
        """
        from collections import OrderedDict

        segs_by_sheet = OrderedDict()
        _validate_experanto_model_timing(dsv)
        # get_segments() intentionally uses null=False: Mozaik's automatic null-stimulus
        # periods (Model.parameters.null_stimulus_period) are not supported by this exporter.
        for seg in dsv.get_segments():
            stim_params = self._parse_stimulus(seg)
            if stim_params is None:
                continue
            if not self._in_group(stim_params.get(self.group_by_key)):
                continue
            sn = seg.annotations.get("sheet_name")
            segs_by_sheet.setdefault(sn, []).append((seg, stim_params))
        return segs_by_sheet

    def _resolve_sheets(self, present):
        """Resolve the ordered list of sheets to export."""
        if self.sheet_names_requested is not None:
            missing = [s for s in self.sheet_names_requested if s not in present]
            if missing:
                raise ValueError(
                    f"Requested sheet(s) {missing} not present in datastore (have: {sorted(present)})"
                )
            requested = set(self.sheet_names_requested)
            selected = [s for s in canonical_sheet_order(present) if s in requested]
        else:
            selected = canonical_sheet_order(present)

        self.sheet_names = selected
        return self.sheet_names

    def _init_layout(self, segs_by_sheet, i):
        """Allocate the global per-unit spike lists using each sheet's unit count at presentation ``i``.

        Every explicitly presented stimulus, including an Experanto ``InternalStimulus`` blank,
        has one spike train per recorded unit and can therefore establish the layout.
        """
        counts = [
            len(segs_by_sheet[sn][i][0].get_spiketrains()) for sn in self.sheet_names
        ]
        indices = [0]
        for c in counts:
            indices.append(indices[-1] + c)
        self.sheet_unit_counts = counts
        self.sheet_unit_indices = indices
        self.num_units = indices[-1]
        self.all_unit_spike_lists = [[] for _ in range(self.num_units)]
        print(
            f"Initialized for {self.num_units} units across {len(self.sheet_names)} sheets: "
            + ", ".join(f"{s}={c}" for s, c in zip(self.sheet_names, counts)),
            flush=True,
        )

    def process(self, dsv):
        """
        Process one chunk's DSV. Folds every requested sheet into one flat per-unit spike list,
        aligning presentations by index across sheets (segments are per-(presentation, sheet)) and
        advancing the timeline once per presentation so it stays aligned with the screen timeline.
        """
        import time

        process_t0 = time.time()
        print(
            f"Processing chunk... (Mem: {get_process_memory():.2f} MB)",
            flush=True,
        )

        # 1. Bucket segments by sheet (trial-filtered), preserving temporal order within each sheet.
        scan_t0 = time.time()
        segs_by_sheet = self._bucket_by_sheet(dsv)
        if not segs_by_sheet:
            print("No matching segments in this chunk.", flush=True)
            return

        sheet_names = self._resolve_sheets(list(segs_by_sheet.keys()))

        # 2. Every sheet must contribute the same number of presentations, so the index-zip aligns.
        lengths = {sn: len(segs_by_sheet[sn]) for sn in sheet_names}
        n_pres = lengths[sheet_names[0]]
        if any(v != n_pres for v in lengths.values()):
            raise ValueError(f"Unequal segment counts across sheets: {lengths}")

        ref = sheet_names[0]
        n_blanks = sum(
            1
            for j in range(n_pres)
            if self._presentation_label(segs_by_sheet[ref][j][1])[1]
        )
        n_stim = n_pres - n_blanks
        print(
            f"  Scan total: {time.time() - scan_t0:.1f}s — {n_pres} presentations × "
            f"{len(sheet_names)} sheets ({n_stim} stimulus, {n_blanks} blank)",
            flush=True,
        )

        t_load = 0.0  # time in get_spiketrains()
        t_loop = 0.0  # time in per-unit spike extraction
        n_processed = 0
        seg_t0 = time.time()

        # 3. Process presentation by presentation, stacking sheets on the unit axis.
        for i in range(n_pres):
            _, ref_stim = segs_by_sheet[ref][i]
            stim_label, is_blank = self._presentation_label(ref_stim)
            seg_duration = ref_stim["duration"]

            self.meta_segments.append(stim_label)

            if is_blank and not self.export_blank_spikes:
                self.current_time_offset += seg_duration
                n_processed += 1
                continue

            # Allocate the global layout from the first presentation, which may be a pre-blank.
            if self.all_unit_spike_lists is None:
                self._init_layout(segs_by_sheet, i)

            offset = self.current_time_offset
            _lists = self.all_unit_spike_lists
            for k, sn in enumerate(sheet_names):
                seg_k, stim_k = segs_by_sheet[sn][i]
                if stim_k["duration"] != seg_duration:
                    raise ValueError(
                        f"Presentation {i}: duration mismatch across sheets "
                        f"({ref}={seg_duration} vs {sn}={stim_k['duration']}) — misaligned segments"
                    )
                t0 = time.time()
                trains = seg_k.get_spiketrains()
                t_load += time.time() - t0

                base = self.sheet_unit_indices[k]
                limit_units = min(self.sheet_unit_counts[k], len(trains))
                t0 = time.time()
                for unit_idx in range(limit_units):
                    spikes = np.asarray(trains[unit_idx])
                    if len(spikes) == 0:
                        continue
                    valid = spikes[spikes < seg_duration]
                    if len(valid):
                        valid = valid + offset
                        _lists[base + unit_idx].append(valid)
                t_loop += time.time() - t0

                if hasattr(seg_k, "release"):
                    seg_k.release()

            self.current_time_offset += seg_duration
            n_processed += 1

            if n_processed % 50 == 0:
                elapsed = time.time() - seg_t0
                eta = (n_pres - n_processed) * (elapsed / n_processed)
                print(
                    f"  [{n_processed}/{n_pres}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s — "
                    f"load: {t_load:.1f}s, loop: {t_loop:.1f}s, "
                    f"offset: {self.current_time_offset/1000:.1f}s, "
                    f"mem: {get_process_memory():.0f} MB",
                    flush=True,
                )

        total_t = time.time() - process_t0
        print(
            f"Chunk complete in {total_t:.1f}s — "
            f"load: {t_load:.1f}s ({t_load/total_t*100:.0f}%), "
            f"loop: {t_loop:.1f}s ({t_loop/total_t*100:.0f}%), "
            f"other: {total_t - t_load - t_loop:.1f}s. "
            f"Offset: {self.current_time_offset:.0f} ms. "
            f"(Mem: {get_process_memory():.2f} MB)",
            flush=True,
        )

    def finalize(self):
        """
        Write the chunk's spikes.npy and meta.yml.
        """
        print("Finalizing export...")
        print(f"Memory before concat: {get_process_memory():.2f} MB")

        # Pre-compute total spike count so we can allocate once instead of
        # building an intermediate list and concatenating a second time.
        total_spikes = sum(
            sum(len(a) for a in chunks) if chunks else 0
            for chunks in self.all_unit_spike_lists
        )
        spikes_1d = np.empty(total_spikes, dtype=np.float64)

        unit_indices = [0]
        pos = 0

        for unit_idx, unit_chunks in enumerate(self.all_unit_spike_lists):
            if unit_chunks:
                unit_arr = np.concatenate(unit_chunks)
                n = len(unit_arr)
                spikes_1d[pos : pos + n] = unit_arr
                pos += n
            unit_indices.append(pos)
            self.all_unit_spike_lists[unit_idx] = None  # free as we go

        # Convert spike times ms -> seconds before saving
        spikes_1d /= 1000.0

        # Save Main Output
        np.save(os.path.join(self.output_dir, "spikes.npy"), spikes_1d)

        # Save Metadata
        meta_data = {
            "modality": "spikes",
            "n_signals": self.num_units,
            "n_signals_layerwise": self.sheet_unit_counts,
            "sheets": self.sheet_names,
            "sheet_unit_indices": self.sheet_unit_indices,
            "start_time": 0.0,
            "end_time": self.current_time_offset / 1000.0,
            "trial_id": self.trial_id,
            "sampling_rate": self.sampling_rate,
            "spike_indices": unit_indices,
            "stimuli_order": self.meta_segments,
            "smoothing": self.smooth_param,
        }

        with open(os.path.join(self.output_dir, "meta.yml"), "w") as f:
            yaml.dump(meta_data, f)

        print(
            f"Export Complete. Total Units: {self.num_units}, Total Time: {self.current_time_offset}ms"
        )
        print(f"Final Memory: {get_process_memory():.2f} MB")


class MozaikScreenExporter:
    """Export one trial's recorded presentation sequence, including explicit blanks.

    Stimulus annotations supply order and timing. Source metadata supplies pixel
    attributes and dataset splits. Each sheet must record the same sequence.
    """

    def __init__(self, output_dir, trial_id, tier_reference=None):
        self.screen_dir = os.path.join(output_dir, "screen")
        self.trial_id = trial_id
        self._presentations = []
        self._tier_reference = tier_reference
        os.makedirs(os.path.join(self.screen_dir, "meta"), exist_ok=True)
        os.makedirs(os.path.join(self.screen_dir, "data"), exist_ok=True)
        with open(os.path.join(self.screen_dir, "meta.yml"), "w") as stream:
            yaml.safe_dump({"modality": "screen"}, stream)

    def process(self, dsv):
        """Read ordered annotations without loading spike trains."""
        _validate_experanto_model_timing(dsv)
        by_sheet = {}
        for segment in dsv.get_segments():
            raw = segment.annotations["stimulus"]
            params = ast.literal_eval(raw) if isinstance(raw, str) else raw
            if (
                params.get("trial") is not None
                and int(params["trial"]) == self.trial_id
            ):
                by_sheet.setdefault(segment.annotations.get("sheet_name"), []).append(
                    params
                )
        if by_sheet:
            self._presentations = next(iter(by_sheet.values()))
            if any(sequence != self._presentations for sequence in by_sheet.values()):
                raise ValueError("Inconsistent recorded presentations across sheets")

    def finalize(self):
        """Write recorded presentations and one synthetic terminal blank."""
        if not self._presentations:
            return
        presentations = self._presentations + [
            {"name": "InternalStimulus", "duration": 0}
        ]
        # Adjacent pixel stimuli supply image sizes for recorded pre/post blanks.
        metadata = (
            [None]
            + [
                (
                    None
                    if stimulus.get("name") == "InternalStimulus"
                    else self._load_source_meta(stimulus)
                )
                for stimulus in presentations
            ]
            + [None]
        )
        timestamps_ms = []
        combined = {}
        image_size = [144, 256]
        elapsed = 0.0
        for index, stimulus in enumerate(presentations, 1):
            duration = float(stimulus["duration"])
            if not math.isfinite(duration) or duration < 0:
                raise ValueError("Invalid recorded stimulus duration")
            is_blank = stimulus.get("name") == "InternalStimulus"
            source = metadata[index]
            if is_blank:
                previous, following = metadata[index - 1], metadata[index + 1]
                source = previous or following
                # A pre-blank after a video belongs to the following image. For
                # adjacent images, source metadata identifies which one has a blank;
                # its actual duration still comes exclusively from the recording.
                if (
                    following
                    and following["modality"] == "image"
                    and (
                        not previous
                        or previous["modality"] != "image"
                        or previous.get("post_blank_period", 0.049) * 1000
                        < presentations[index - 2]["frame_duration"]
                    )
                ):
                    source = following
            if source:
                image_size = source.get("image_size", [144, 256])
            entry = (
                dict(modality="blank", image_size=image_size, interleave_value=128.0)
                if is_blank
                else dict(source)
            )
            entry["first_frame_idx"] = len(timestamps_ms)
            entry["num_frames"] = (
                source["num_frames"] if entry["modality"] == "video" else 1
            )
            if entry["modality"] == "video":
                try:
                    frame_time = float(stimulus["movie_frame_duration"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        "Missing or invalid recorded video timing"
                    ) from error
                if (
                    not math.isfinite(frame_time)
                    or frame_time <= 0
                    or not math.isclose(entry["num_frames"] * frame_time, duration)
                ):
                    raise ValueError("Recorded video timing disagrees with frame count")
                timestamps_ms.extend(
                    elapsed + frame * frame_time for frame in range(entry["num_frames"])
                )
            else:
                timestamps_ms.append(elapsed)
            elapsed += duration
            key = f"{len(combined):05d}"
            combined[key] = entry
            if not is_blank:
                shutil.copy2(
                    os.path.join(stimulus["movie_path"], stimulus["movie_name"]),
                    os.path.join(self.screen_dir, "data", f"{key}.npy"),
                )
            with open(
                os.path.join(self.screen_dir, "meta", f"{key}.yml"), "w"
            ) as stream:
                yaml.safe_dump(entry, stream)
        np.save(
            os.path.join(self.screen_dir, "timestamps.npy"),
            np.array(timestamps_ms) / 1000.0,
        )
        with open(os.path.join(self.screen_dir, "combined_meta.json"), "w") as stream:
            json.dump(combined, stream)

    def _load_source_meta(self, stimulus):
        path = os.path.join(
            os.path.dirname(stimulus["movie_path"].rstrip("/")),
            "meta",
            stimulus["movie_name"].replace(".npy", ".yml"),
        )
        with open(path) as stream:
            meta = yaml.safe_load(stream)
        if self._tier_reference is not None:
            condition = meta.get("condition_hash")
            if condition in self._tier_reference:
                meta["tier"] = self._tier_reference[condition]
        return meta
