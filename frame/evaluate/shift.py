"""Spatial-shift sensitivity (requirements 142, section 27; Phase 5).

Cross-sensor HR references are not perfectly registered to the Sentinel-2 grid, and a pixel-wise metric punishes a correctly placed detail that is a pixel or two
off just as much as an invented one. This module MEASURES how much: the SR of each system is displaced against the reference by known whole HR pixels and the same
metrics are recomputed on the overlap. It also scores an ``aligned_to_bicubic`` condition, where the displacement removed is the one phase correlation finds between the
bicubic baseline and the reference (a system-neutral estimate: no system aligns itself, and the correction is identical for all systems on a sample).

Why whole pixels: 0, 0.25, 0.5, 1 and 2 LR pixels (the requirements' list) are 0, 1, 2, 4 and 8 HR pixels at x4, so nothing is resampled and no interpolation
smoothing enters the comparison. Displacement is along the column axis. Only spatial and spectral reference metrics are computed here; classification accuracy and area error
(also named in section 27) need a downstream task, which is not part of Phase 5, and were NOT computed.

This is an interpretation aid, not a ranking: it says how large a number a systematic misregistration alone can produce, so that a difference between systems can be read
against it.
"""

from __future__ import annotations

import datetime
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from frame.evaluate import metrics as M
from frame.evaluate.aggregate import Entry, aggregate_dataset
from frame.evaluate.config import EvalConfig, resolve_path
from frame.evaluate.datasets import build_dataset
from frame.evaluate.errors import EvaluationError, SystemUnavailableError
from frame.evaluate.systems import BicubicSystem, System, build_system, check_no_training_overlap
from frame.evaluate.runner import REPO_ROOT, _clean, _resolve_device, flatten_row

#: 0, 0.25, 0.5, 1 and 2 low-resolution pixels (requirements 142, section 27)
SWEEP_LR_PIXELS: Tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0)
ALIGNED = "aligned_to_bicubic"
AXIS = "x (columns)"

#: what the sweep tables show (all are computed for every condition)
SHIFT_HEADLINE = ("reference_accuracy.psnr_db", "reference_accuracy.ssim", "reference_accuracy.sam_degrees", "reference_accuracy.ergas", "indices.NDVI.mae",
                  "spatial_detail.hf_relative_error", "spatial_detail.hf_correlation")


# ---------------------------------------------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------------------------------------------

def displace_pair(sr: np.ndarray, hr: np.ndarray, mask: np.ndarray, dy: int, dx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Displace the SR content by ``(dy, dx)`` whole HR pixels (down/right for positive values) relative to the reference and crop both to their overlap.

    In the returned frame position ``(y, x)`` holds ``sr[y - dy, x - dx]`` (original coordinates) next to ``hr[y, x]``; the mask is the reference's and is cropped
    with the reference. Nothing is resampled, wrapped or padded, and the inputs are not modified.
    """
    height, width = hr.shape[-2:]
    if abs(dy) >= height or abs(dx) >= width:
        raise ValueError(f"displacement ({dy}, {dx}) is not smaller than the image ({height}x{width}); nothing would be left to compare")
    sr_rows, hr_rows = slice(max(-dy, 0), height - max(dy, 0)), slice(max(dy, 0), height + min(dy, 0))
    sr_cols, hr_cols = slice(max(-dx, 0), width - max(dx, 0)), slice(max(dx, 0), width + min(dx, 0))
    return sr[..., sr_rows, sr_cols].copy(), hr[..., hr_rows, hr_cols].copy(), mask[hr_rows, hr_cols].copy()


def hr_shifts(lr_shifts: Sequence[float], scale: int) -> Tuple[int, ...]:
    """LR-pixel shifts as whole HR pixels; a shift that is not a whole number of HR pixels is refused (it would need resampling)."""
    out = []
    for s in lr_shifts:
        value = float(s) * scale
        if abs(value - round(value)) > 1e-9:
            raise ValueError(f"a shift of {s} LR pixels is {value} HR pixels at x{scale}, not a whole HR pixel; this experiment never resamples")
        out.append(int(round(value)))
    return tuple(out)


def condition_name(lr_shift: float) -> str:
    return f"shift_{float(lr_shift):g}_lr_px"


def estimate_alignment(baseline: np.ndarray, hr: np.ndarray, mask: np.ndarray, cfg: M.MetricConfig) -> Dict[str, Any]:
    """The whole-pixel displacement of ``baseline`` relative to ``hr`` by masked phase correlation, and the correction that removes it.

    ``dy, dx`` are the baseline's displacement (positive = down/right), ``correction = (-dy, -dx)`` is what :func:`displace_pair` needs to undo it. A degenerate image gives
    ``status = not_computable`` and a zero correction (nothing is guessed).
    """
    a = np.asarray(baseline, dtype=np.float64)
    b = np.asarray(hr, dtype=np.float64)
    raw = M._phase_shift(a.mean(axis=0), b.mean(axis=0), np.asarray(mask, dtype=bool), cfg.phase_upsample)
    if raw.get("status") != "ok" or raw["dy"] is None or not (math.isfinite(raw["dy"]) and math.isfinite(raw["dx"])):
        return {"status": "not_computable", "reason": raw.get("reason", "non-finite shift"), "dy": 0, "dx": 0, "raw_dy": None, "raw_dx": None, "correction": (0, 0)}
    dy, dx = int(round(raw["dy"])), int(round(raw["dx"]))
    return {"status": "ok", "dy": dy, "dx": dx, "raw_dy": raw["dy"], "raw_dx": raw["dx"], "correction": (-dy, -dx)}


# ---------------------------------------------------------------------------------------------------------------
# metrics under a displacement
# ---------------------------------------------------------------------------------------------------------------

def shift_metrics(sr: np.ndarray, hr: np.ndarray, mask: np.ndarray, bands: Sequence[str], scale: int, cfg: M.MetricConfig, *, dy: int, dx: int) -> Dict[str, Optional[float]]:
    """The accuracy, NDVI and detail metrics of ``sr`` displaced by ``(dy, dx)`` against ``hr`` over the overlap (flat dotted names, as in the main evaluation)."""
    sr_c, hr_c, mask_c = displace_pair(sr, hr, mask, dy, dx)
    row = {
        "reference_accuracy": M.reference_accuracy(sr_c, hr_c, mask_c, bands, scale, cfg),
        "indices": M.index_metrics(sr_c, hr_c, mask_c, bands, cfg),
        "spatial_detail": M.spatial_detail_metrics(sr_c, hr_c, mask_c, bands, cfg),
    }
    flat = flatten_row(_clean(row))
    flat["quality.compared_fraction"] = float(hr_c.shape[-2] * hr_c.shape[-1]) / float(hr.shape[-2] * hr.shape[-1])
    return flat


# ---------------------------------------------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------------------------------------------

def run_shift_sensitivity(
    config: EvalConfig,
    *,
    dataset: str,
    system_names: Optional[Sequence[str]] = None,
    lr_shifts: Sequence[float] = SWEEP_LR_PIXELS,
    output_dir: Optional[str] = None,
    base_dir: Optional[Path] = None,
    systems: Optional[Sequence[System]] = None,
    datasets: Optional[Sequence[Any]] = None,
    mamba_client_factory: Optional[Any] = None,
    progress: Optional[Any] = None,
) -> Dict[str, Any]:
    """Sweep the reference displacement for the configured systems on ONE configured dataset and write ``config.json, rows.jsonl, aggregates.json, summary.json, README.md``.

    The same safety checks as the main evaluation apply (dataset role, geometry, train/eval overlap), and nothing is dropped silently: skipped, invalid and unreadable
    samples and inference failures are listed. Results are never overwritten.
    """
    from frame.evaluate.shift_report import render_shift_readme
    from frame.train.record import environment_info, git_info

    say = progress or (lambda message: None)
    out_dir = resolve_path(output_dir or config.output_dir, base_dir)
    existing = [n for n in ("rows.jsonl", "summary.json", "aggregates.json", "README.md") if (out_dir / n).exists()]
    if existing:
        raise EvaluationError(f"{out_dir} already contains results ({', '.join(existing)}); choose a new output directory. Nothing was overwritten.")
    if dataset not in {s.name for s in config.datasets}:
        raise EvaluationError(f"dataset {dataset!r} is not in the config (datasets: {[s.name for s in config.datasets]})")
    device = _resolve_device(config.device)
    tiling, metric_cfg = config.tiling_config(), config.metric_config()

    specs = [s for s in config.datasets if s.name == dataset]
    built = [d for d in datasets if d.name == dataset] if datasets is not None else [build_dataset(spec, base_dir=base_dir) for spec in specs]
    if not built:
        raise EvaluationError(f"dataset {dataset!r} was not provided")
    ds = built[0]
    validation = ds.validate(check_files=True)

    unavailable: List[Dict[str, Any]] = []
    live: List[System] = []
    if systems is not None:
        live = [s for s in systems if system_names is None or s.name in system_names]
    else:
        for spec in config.systems:
            if system_names is not None and spec.name not in system_names:
                continue
            try:
                live.append(build_system(spec, tiling, device=device, base_dir=base_dir, mamba_client_factory=mamba_client_factory))
            except SystemUnavailableError as exc:
                unavailable.append({"name": spec.name, "kind": spec.kind, "reason": str(exc)})
    try:
        for system in live:
            check_no_training_overlap(system, dataset_kind=ds.kind, scene_ids=[r.scene_id for r in ds.records])

        valid_ids = {r.sample_id for r in validation.valid_records}
        ordered = sorted((r for r in ds.records if r.sample_id in valid_ids), key=lambda r: r.sample_id)
        # the scale factor decides whether a shift is a whole number of HR pixels, so it is read from the first loadable sample BEFORE anything is written
        preloaded: Dict[str, Any] = {}
        scale, hr_list = None, ()
        for record in ordered:
            try:
                first = ds.load(record)
            except Exception:
                continue                                                          # reported as unreadable in the loop below
            preloaded[record.sample_id], scale = first, first.scale
            break
        if scale is not None:
            try:
                hr_list = hr_shifts(lr_shifts, scale)
            except ValueError as exc:
                raise EvaluationError(str(exc)) from exc
        conditions = [condition_name(s) for s in lr_shifts] + [ALIGNED]
        entries: Dict[str, Dict[str, List[Entry]]] = {c: {s.name: [] for s in live} for c in conditions}

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(config.to_json(), encoding="utf-8")
        reference = BicubicSystem("bicubic")
        skipped, unreadable, failures, alignments, evaluated_ids = [], [], {s.name: 0 for s in live}, [], []
        with open(out_dir / "rows.jsonl", "w", encoding="utf-8") as sink:
            def write(row: Dict[str, Any]) -> None:
                sink.write(json.dumps(_clean(row), sort_keys=True, allow_nan=False) + "\n")

            for bad in validation.invalid:
                write({"type": "sample_invalid", "dataset": ds.name, **bad})
            for record in ordered:
                try:
                    sample = preloaded.pop(record.sample_id, None) or ds.load(record)
                except Exception as exc:
                    info = {"sample_id": record.sample_id, "error": f"{type(exc).__name__}: {exc}"[:300]}
                    unreadable.append(info)
                    write({"type": "sample_unreadable", "dataset": ds.name, **info})
                    continue
                if sample.quality["hr_valid_fraction"] < config.min_valid_fraction:
                    info = {"sample_id": sample.sample_id, "reason": f"insufficient_valid_reference: strict valid fraction {sample.quality['hr_valid_fraction']:.4f} < min_valid_fraction {config.min_valid_fraction}",
                            "hr_valid_fraction": sample.quality["hr_valid_fraction"]}
                    skipped.append(info)
                    write({"type": "sample_skipped", "dataset": ds.name, **info})
                    continue
                evaluated_ids.append(sample.sample_id)
                say(f"{ds.name} {sample.sample_id}")
                hr_np, mask = sample.hr.numpy().astype(np.float64), sample.hr_mask.numpy()
                baseline = reference.infer(sample.lr).sr.numpy().astype(np.float64)
                alignment = estimate_alignment(baseline, hr_np, mask, metric_cfg)
                alignments.append({"sample_id": sample.sample_id, **{k: alignment[k] for k in ("status", "dy", "dx", "raw_dy", "raw_dx")}})
                plan = [(condition_name(s), 0, dx, None) for s, dx in zip(lr_shifts, hr_list)] + [(ALIGNED, alignment["correction"][0], alignment["correction"][1], alignment)]
                for system in live:
                    try:
                        sr_np = system.infer(sample.lr).sr.numpy().astype(np.float64)
                        if sr_np.shape != hr_np.shape:
                            raise EvaluationError(f"SR shape {sr_np.shape} does not equal the HR reference shape {hr_np.shape}")
                        rows = []
                        for name, dy, dx, extra in plan:
                            flat = shift_metrics(sr_np, hr_np, mask, sample.bands, sample.scale, metric_cfg, dy=dy, dx=dx)
                            rows.append((name, dy, dx, extra, flat))
                    except Exception as exc:
                        failures[system.name] += 1
                        write({"type": "sample_result", "status": "failed", "dataset": ds.name, "sample_id": sample.sample_id, "scene_group": sample.scene_group, "system": system.name,
                               "error": f"{type(exc).__name__}: {exc}"[:400]})
                        continue
                    for name, dy, dx, extra, flat in rows:
                        row = {"type": "sample_result", "status": "ok", "dataset": ds.name, "sample_id": sample.sample_id, "scene_group": sample.scene_group, "system": system.name,
                               "condition": name, "dy_hr_px": dy, "dx_hr_px": dx, "metrics": flat}
                        if extra is not None:
                            row["alignment"] = {k: extra[k] for k in ("status", "dy", "dx", "raw_dy", "raw_dx")}
                        write(row)
                        entries[name][system.name].append((sample.sample_id, sample.scene_group, sample.category, flat))
                del sample, baseline, hr_np

        st = config.statistics
        aggregates: Dict[str, Any] = {"unit_of_analysis": "scene unit; tiles of one unit are averaged first", "statistics": {"n_boot": st.n_boot, "alpha": st.alpha, "seed": st.seed},
                                      "conditions": {}}
        for name in conditions:
            live_entries = {s: e for s, e in entries[name].items() if e}
            aggregates["conditions"][name] = aggregate_dataset(live_entries, reference_system=st.reference_system, pairs=(), headline=SHIFT_HEADLINE, n_boot=st.n_boot, alpha=st.alpha, seed=st.seed)

        magnitudes = [math.hypot(a["raw_dy"], a["raw_dx"]) for a in alignments if a["status"] == "ok" and a["raw_dy"] is not None]
        total = len(ds.records)
        summary: Dict[str, Any] = {
            "name": f"{config.name}:shift_sensitivity:{ds.name}", "status": "completed", "dataset": ds.name, "evidence_class": ds.evidence_class, "role": ds.role, "split": ds.spec.split,
            "manifest_digest": ds.manifest_digest, "config_digest": config.digest(), "finished_utc": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat(),
            "code": {"git": git_info(REPO_ROOT)}, "environment": _clean(environment_info(device)), "metric_configuration": config.metrics, "tiling": tiling.describe(),
            "axis": AXIS, "lr_shifts": [float(s) for s in lr_shifts], "hr_shifts": list(hr_list or ()), "scale": scale, "conditions": conditions,
            "alignment_reference": "bicubic baseline (system-neutral); whole-pixel phase-correlation estimate, identical for every system on a sample",
            "alignment_estimates": alignments, "alignment_magnitude_hr_px": ({"n": len(magnitudes), "median": float(np.median(magnitudes)), "mean": float(np.mean(magnitudes)),
                                                                             "max": float(np.max(magnitudes))} if magnitudes else None),
            "counts": {"total": total, "invalid": len(validation.invalid), "valid": total - len(validation.invalid), "unreadable": len(unreadable), "skipped": len(skipped),
                       "evaluated": len(evaluated_ids)},
            "invalid": validation.invalid, "skipped": skipped, "unreadable": unreadable, "failures": {k: n for k, n in failures.items() if n},
            "sample_ids": sorted(evaluated_ids), "systems": {s.name: _clean(s.provenance()) for s in live}, "systems_unavailable": unavailable,
            "headline_metrics": list(SHIFT_HEADLINE),
            "not_computed": ["classification accuracy", "area error"],
            "notes": ["An interpretation aid, not a ranking: it shows how large a difference a systematic misregistration alone produces.",
                      "Displacement is in whole HR pixels along the column axis; nothing is resampled. Statistics are descriptive unless the scene-unit count allows an interval."],
        }
        (out_dir / "aggregates.json").write_text(json.dumps(_clean(aggregates), indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        (out_dir / "summary.json").write_text(json.dumps(_clean(summary), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        (out_dir / "README.md").write_text(render_shift_readme(_clean(summary), _clean(aggregates)), encoding="utf-8")
        return _clean(summary)
    finally:
        if systems is None:
            for s in live:
                s.close()
