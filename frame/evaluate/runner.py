"""The evaluation runner (Phase 5).

    config -> datasets (role safety, reference + geometry validation) -> frozen systems (availability, train/eval overlap) -> per sample: load, strict mask,
    inference, mask-aware metrics -> aggregation over scene units -> statistics -> provenance -> experiments/evaluation/<name>/

Nothing is silently dropped. Every record of every dataset ends in exactly one of: evaluated, skipped (with the rule), invalid (with the codes),
unreadable (with the error) -- and every (sample, system) inference failure is a row with its error. An unavailable system is listed with its reason and the rest
of the run continues. Safety refusals (a benchmark record labelled train, a model that saw the evaluated scenes, a model trained on the benchmark) abort BEFORE
any output is written. Results of different evidence classes (real cross-sensor vs synthetic) are kept in separate sections and never combined.
"""

from __future__ import annotations

import datetime
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from frame.evaluate import metrics as M
from frame.evaluate.aggregate import Entry, aggregate_dataset, headline_metrics, seed_group_summary
from frame.evaluate.config import EvalConfig, resolve_path
from frame.evaluate.datasets import EvalDataset, EvalSample, build_dataset
from frame.evaluate.errors import EvaluationError, ReferenceMismatchError, SystemUnavailableError
from frame.evaluate.report import render_readme
from frame.evaluate.systems import BicubicSystem, InferenceResult, System, build_system, check_no_training_overlap, seam_lines
from frame.tiling import TilingConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
_SKIP_LEAF = ("tau", "hf_sigma_px", "half_width_px", "registration_error", "status", "reason", "definition", "opensr_test_version", "downsample_method", "rule")


# ---------------------------------------------------------------------------------------------------------------
# one sample
# ---------------------------------------------------------------------------------------------------------------

def _clean(obj: Any) -> Any:
    """JSON-safe: numpy scalars -> python, NaN/Inf -> None, tuples -> lists."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def score_sample(sample: EvalSample, system: System, result: InferenceResult, baseline_sr: torch.Tensor, cfg: M.MetricConfig, *, tiling: Optional[TilingConfig] = None) -> Dict[str, Any]:
    """All metric groups for one system on one sample (see frame.evaluate.metrics; groups are kept separate)."""
    sr = result.sr.float()
    if tuple(sr.shape) != tuple(sample.hr.shape):
        raise ReferenceMismatchError(f"{sample.sample_id}: SR shape {tuple(sr.shape)} does not equal the HR reference shape {tuple(sample.hr.shape)}.")
    sr_np, hr_np, mask = sr.numpy().astype(np.float64), sample.hr.numpy().astype(np.float64), sample.hr_mask.numpy()
    maps = M.ssim_maps(sr_np, hr_np, cfg) if mask.any() else None
    is_baseline = system.kind == "bicubic"
    seam = None
    if result.plan is not None and tiling is not None:
        rows, cols = seam_lines(result.plan, tiling)
        seam = M.seam_error(sr_np, hr_np, mask, seam_rows=rows, seam_cols=cols, half_width=8)
    row: Dict[str, Any] = {
        "type": "sample_result", "status": "ok", "dataset_kind": sample.dataset, "evidence_class": sample.evidence_class, "sample_id": sample.sample_id, "scene_group": sample.scene_group,
        "category": sample.category, "system": system.name, "seconds": round(result.seconds, 4), "tile_count": result.tile_count, "quality": dict(sample.quality),
        "reference_accuracy": M.reference_accuracy(sr_np, hr_np, mask, sample.bands, sample.scale, cfg, ssim_map=maps),
        "per_band": M.per_band_metrics(sr_np, hr_np, mask, sample.bands, cfg, ssim_map=maps),
        "indices": M.index_metrics(sr_np, hr_np, mask, sample.bands, cfg),
        "band_ratios": M.band_ratio_metrics(sr_np, hr_np, mask, sample.bands, cfg),
        "spatial_detail": M.spatial_detail_metrics(sr_np, hr_np, mask, sample.bands, cfg),
        "seam": seam,
        "self_consistency": M.self_consistency(sample.lr, sr, sample.lr_mask, sample.bands, sample.scale),
        "opensr_native": M.opensr_native(sample.lr, sr, sample.hr, hr_mask=sample.hr_mask),
    }
    if is_baseline:
        row["detail_analysis"] = None
        row["detail_analysis_note"] = "not defined for the baseline itself: the analysis measures what a system ADDED relative to the bicubic baseline"
    else:
        row["detail_analysis"] = M.hallucination_analysis(baseline_sr.numpy(), sr_np, hr_np, mask, cfg)
    return _clean(row)


def flatten_row(row: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """One number per dotted metric name (counts and labels excluded), for aggregation."""
    out: Dict[str, Optional[float]] = {}

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                if k in _SKIP_LEAF or k.startswith("n_") or k.endswith("_pixels"):
                    continue
                walk(f"{prefix}.{k}" if prefix else str(k), v)
        elif isinstance(value, bool):
            return
        elif isinstance(value, (int, float)) or value is None:
            out[prefix] = None if value is None else float(value)

    for group in ("reference_accuracy", "per_band", "indices", "band_ratios", "spatial_detail", "seam", "detail_analysis", "self_consistency", "opensr_native", "quality"):
        if group == "opensr_native":
            native = row.get(group) or {}
            walk("opensr_native", native.get("values") or {})
            continue
        if group == "spatial_detail":
            sd = {k: v for k, v in (row.get(group) or {}).items() if k != "per_band"}
            walk(group, sd)
            continue
        if row.get(group):
            walk(group, row[group])
    out = {k: v for k, v in out.items() if k not in ("quality.rule",)}
    return out


# ---------------------------------------------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------------------------------------------

def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def run_evaluation(
    config: EvalConfig,
    *,
    base_dir: Optional[Path] = None,
    systems: Optional[Sequence[System]] = None,
    datasets: Optional[Sequence[Any]] = None,
    system_factory: Optional[Callable[..., System]] = None,
    mamba_client_factory: Optional[Callable[..., Any]] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    from frame.train.record import environment_info, git_info

    say = progress or (lambda message: None)
    out_dir = resolve_path(config.output_dir, base_dir)
    existing = [n for n in ("metrics.jsonl", "summary.json", "aggregates.json", "README.md") if (out_dir / n).exists()]
    if existing:
        raise EvaluationError(f"{out_dir} already contains results ({', '.join(existing)}); choose a new output directory. Nothing was overwritten.")
    device = _resolve_device(config.device)
    tiling = config.tiling_config()
    metric_cfg = config.metric_config()

    # 1. datasets: role safety and reference/geometry validation (raises before anything is written)
    built = list(datasets) if datasets is not None else [build_dataset(spec, base_dir=base_dir) for spec in config.datasets]
    validations = {ds.name: ds.validate(check_files=True) for ds in built}

    # 2. systems: availability, then the train/eval overlap guard
    unavailable: List[Dict[str, Any]] = []
    live: List[System] = []
    if systems is not None:
        live = list(systems)
    else:
        factory = system_factory or build_system
        for spec in config.systems:
            try:
                live.append(factory(spec, tiling, device=device, base_dir=base_dir, mamba_client_factory=mamba_client_factory) if system_factory is None
                            else factory(spec, tiling, device=device, base_dir=base_dir))
            except SystemUnavailableError as exc:
                unavailable.append({"name": spec.name, "kind": spec.kind, "reason": str(exc)})
    try:
        for system in live:
            for ds in built:
                check_no_training_overlap(system, dataset_kind=ds.kind, scene_ids=[r.scene_id for r in ds.records])

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(config.to_json(), encoding="utf-8")
        group_of = {s.name: s.group for s in config.systems}
        reference = BicubicSystem("bicubic")
        primary_tau = metric_cfg.hallucination_taus[0]

        entries: Dict[str, Dict[str, List[Entry]]] = {ds.name: {s.name: [] for s in live} for ds in built}
        dataset_summary: Dict[str, Any] = {}
        first_geometry: Dict[str, Dict[str, Any]] = {}
        with open(out_dir / "metrics.jsonl", "w", encoding="utf-8") as sink:
            def write(row: Dict[str, Any]) -> None:
                sink.write(json.dumps(_clean(row), sort_keys=True, allow_nan=False) + "\n")

            for ds in built:
                v = validations[ds.name]
                valid_ids = {r.sample_id for r in v.valid_records}
                skipped, unreadable, failures, examples, categories = [], [], {s.name: 0 for s in live}, [], {}
                evaluated_ids: List[str] = []
                for bad in v.invalid:
                    write({"type": "sample_invalid", "dataset": ds.name, **bad})
                for record in sorted(ds.records, key=lambda r: r.sample_id):
                    if record.sample_id not in valid_ids:
                        continue
                    try:
                        sample = ds.load(record)
                    except Exception as exc:
                        info = {"sample_id": record.sample_id, "error": f"{type(exc).__name__}: {exc}"[:300]}
                        unreadable.append(info)
                        write({"type": "sample_unreadable", "dataset": ds.name, **info})
                        continue
                    first_geometry.setdefault(ds.name, {"lr_resolution_m": sample.lr_pixel_m, "hr_resolution_m": sample.hr_pixel_m, "bands": list(sample.bands)})
                    if sample.quality["hr_valid_fraction"] < config.min_valid_fraction:
                        info = {"sample_id": sample.sample_id, "reason": f"insufficient_valid_reference: strict valid fraction {sample.quality['hr_valid_fraction']:.4f} < min_valid_fraction {config.min_valid_fraction}",
                                "hr_valid_fraction": sample.quality["hr_valid_fraction"]}
                        skipped.append(info)
                        write({"type": "sample_skipped", "dataset": ds.name, **info})
                        continue
                    evaluated_ids.append(sample.sample_id)
                    if sample.category:
                        categories[sample.category] = categories.get(sample.category, 0) + 1
                    say(f"{ds.name} {sample.sample_id}")
                    baseline_sr = reference.infer(sample.lr).sr
                    for system in live:
                        try:
                            result = system.infer(sample.lr)
                            row = score_sample(sample, system, result, baseline_sr, metric_cfg, tiling=tiling)
                            row["dataset"] = ds.name                     # the CONFIGURED dataset (score_sample only knows the kind): two datasets of one kind must stay distinguishable
                        except Exception as exc:
                            failures[system.name] += 1
                            row = {"type": "sample_result", "status": "failed", "dataset": ds.name, "dataset_kind": sample.dataset, "evidence_class": sample.evidence_class, "sample_id": sample.sample_id,
                                   "scene_group": sample.scene_group, "system": system.name, "error": f"{type(exc).__name__}: {exc}"[:400]}
                            if len(examples) < 5:
                                examples.append({"system": system.name, "sample_id": sample.sample_id, "error": row["error"]})
                        write(row)
                        if row["status"] == "ok":
                            entries[ds.name][system.name].append((sample.sample_id, sample.scene_group, sample.category, flatten_row(row)))
                    del sample, baseline_sr
                total = len(ds.records)
                dataset_summary[ds.name] = {
                    "kind": ds.kind, "evidence_class": ds.evidence_class, "role": ds.role, "split": ds.spec.split, "manifest_digest": ds.manifest_digest, "info": ds.info,
                    "ignored_records": ds.ignored,
                    "counts": {"total": total, "invalid": len(v.invalid), "valid": total - len(v.invalid), "unreadable": len(unreadable), "skipped": len(skipped), "evaluated": len(evaluated_ids)},
                    "invalid": v.invalid, "skipped": skipped, "unreadable": unreadable, "failures": {k: n for k, n in failures.items() if n}, "failure_examples": examples,
                    "categories": dict(sorted(categories.items())), "sample_ids": sorted(evaluated_ids), "scene_units": sorted({ds.scene_group_of(r) for r in ds.records if r.sample_id in set(evaluated_ids)}),
                    "min_valid_fraction": config.min_valid_fraction,
                }

        # 3. aggregation and statistics
        headline = headline_metrics(primary_tau)
        st = config.statistics
        aggregates: Dict[str, Any] = {"metric_configuration": config.metrics, "statistics": {"n_boot": st.n_boot, "alpha": st.alpha, "seed": st.seed, "reference_system": st.reference_system,
                                                                                               "unit_of_analysis": "scene unit (NEON acquisition / source orthophoto / region); tiles of one unit are averaged first",
                                                                                               "multiple_comparisons": "no correction applied; a table of many comparisons is exploratory"},
                                      "datasets": {}}
        for ds in built:
            live_entries = {name: e for name, e in entries[ds.name].items() if e}
            aggregates["datasets"][ds.name] = aggregate_dataset(live_entries, reference_system=st.reference_system, pairs=st.pairs, headline=headline, n_boot=st.n_boot, alpha=st.alpha, seed=st.seed)
        seed_groups: Dict[str, Any] = {}
        for group in sorted({g for g in group_of.values() if g}):
            members = [s.name for s in live if group_of.get(s.name) == group]
            if members:
                seed_groups[group] = seed_group_summary(members, aggregates["datasets"], headline)

        # 4. provenance
        system_info = {s.name: {"available": True, "provenance": _clean(s.provenance())} for s in live}
        for u in unavailable:
            system_info[u["name"]] = {"available": False, "kind": u["kind"], "reason": u["reason"]}
        matrix = []
        for s in live:
            prov = system_info[s.name]["provenance"]
            for ds in built:
                d = dataset_summary[ds.name]
                geom = first_geometry.get(ds.name, {})
                n_ok = len(entries[ds.name][s.name])
                matrix.append({
                    "system": s.name, "model": prov.get("model_name") or prov.get("kind"), "dataset": ds.name, "evidence_class": ds.evidence_class, "scenes": len(d["scene_units"]),
                    "split": d["split"], "bands": geom.get("bands"), "lr_resolution_m": geom.get("lr_resolution_m"), "hr_resolution_m": geom.get("hr_resolution_m"),
                    "hard_constraint": prov.get("hard_constraint"), "tiling": prov.get("tiling"), "weights": prov.get("weights"),
                    "metric_configuration": config.metrics["version"], "n_samples": d["counts"]["total"], "n_evaluated": n_ok,
                })
        sections: Dict[str, Any] = {}
        for ds in built:
            a = aggregates["datasets"][ds.name]
            table = {sys: {m: {"mean": v["unit"]["mean"], "std": v["unit"]["std"], "n_units": v["unit"]["n"], "n_samples": v["sample"]["n"]} for m, v in info["metrics"].items() if m in headline}
                     for sys, info in a["systems"].items()}
            paired = {pair: {m: {k: t[k] for k in ("n", "mean_difference", "median_difference", "label")} | {"ci": [t["ci"]["ci_low"], t["ci"]["ci_high"]], "p": t["wilcoxon"]["p_value"]}
                             for m, t in tab.items() if m in headline} for pair, tab in a["paired"].items()}
            sections.setdefault(ds.evidence_class, {})[ds.name] = {"headline": table, "paired_headline": paired}

        summary: Dict[str, Any] = {
            "name": config.name, "status": "completed", "config_digest": config.digest(), "finished_utc": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat(),
            "code": {"git": git_info(REPO_ROOT)}, "environment": _clean(environment_info(device)), "metric_configuration": config.metrics, "tiling": tiling.describe(),
            "min_valid_fraction": config.min_valid_fraction, "systems": system_info, "systems_unavailable": unavailable, "datasets": dataset_summary, "evaluation_matrix": matrix,
            "sections": sections, "seed_groups": seed_groups, "headline_metrics": headline,
            "notes": ["Results are kept per dataset and per evidence class; nothing is pooled across datasets or across synthetic and real data. No ranking is made.",
                      "Self-consistency (SR reduced to the LR grid vs the LR) is reported separately and is not accuracy against an HR reference."],
        }
        (out_dir / "aggregates.json").write_text(json.dumps(_clean(aggregates), indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        (out_dir / "summary.json").write_text(json.dumps(_clean(summary), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        (out_dir / "README.md").write_text(render_readme(_clean(summary), _clean(aggregates), config), encoding="utf-8")
        return _clean(summary)
    finally:
        if systems is None:
            for s in live:
                s.close()
