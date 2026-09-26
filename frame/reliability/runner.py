"""The reliability runner (Phase 6).

    config -> datasets (role safety, reference + geometry validation) -> frozen models (train/eval overlap guard)
           -> per tile: load, bicubic baseline, REFERENCE ELIGIBILITY GATE (valid pixels, finiteness, registration)
           -> eligible tiles only: TTA ensemble through the tile engine -> aligned error targets and within-tile associations (frame.reliability.evidence)
           -> per (dataset, system): correlation, tile/scene reliability, risk-coverage, high-error detection, calibration (frame.reliability.analysis)
           -> experiments/uncertainty/runs/<name>/

Nothing is silently dropped and nothing is invented. Every sample ends as exactly one of: eligible (evidence), excluded from the uncertainty-error analysis with a machine-readable reason
(excluded_from_uncertainty_error_analysis; the reference gate, or a model / TTA-member failure, or a non-finite prediction), invalid (dataset validation) or unreadable. The gate is decided
from the reference and the BICUBIC baseline before any model runs, so no model's error can influence which tiles are analysed and every model sees identical evidence. Datasets are kept
apart: no correlation, interval or curve pools two datasets.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from frame.evaluate.config import resolve_path
from frame.evaluate.datasets import build_dataset
from frame.evaluate.errors import EvaluationError, SystemUnavailableError
from frame.evaluate.runner import REPO_ROOT, _clean, _resolve_device
from frame.evaluate.systems import BicubicSystem, System, build_system, check_no_training_overlap
from frame.reliability import analysis as AN
from frame.reliability.config import ReliabilityConfig
from frame.reliability.eligibility import evaluate_reference, exclusion_row
from frame.reliability.evidence import RELIABILITY_VERSION, EnsembleArrays, TileInputs, compute_tile_evidence, load_evidence, save_evidence
from frame.tiling import TiledModel
from frame.uncertainty.ensemble import run_tta_ensemble
from frame.uncertainty.transforms import DEFAULT_TRANSFORMS

OUTPUT_FILES = ("config.json", "metrics.jsonl", "summary.json", "README.md", "correlations.json", "risk_coverage.json", "detection.json", "calibration.json")
TRANSFORMS = {t.name: t for t in DEFAULT_TRANSFORMS}


def _safe(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", sample_id)


def evidence_digest(config: ReliabilityConfig, system: System, dataset: Any) -> str:
    """Identity of everything that determines a tile's evidence: the settings that shape it, the dataset revision, and the model as its provenance describes it (weights hashes included).
    Analysis-only settings (coverage grid, bootstrap size, dev/test split, thresholds) are NOT part of it: changing them re-analyses cached evidence without repeating the ensemble."""
    a = config.analysis
    payload = {"version": RELIABILITY_VERSION, "tiling": config.tiling_config().describe(), "tta": config.to_dict()["tta"], "alignment": config.to_dict()["alignment"],
               "eligibility": config.to_dict()["eligibility"], "metrics": config.metrics, "bootstrap_seed": config.bootstrap.seed,
               "analysis": {"cell_sizes_hr_px": list(a.cell_sizes_hr_px), "min_cell_valid_fraction": a.min_cell_valid_fraction, "pooled_cells_per_tile": a.pooled_cells_per_tile,
                            "pooled_pixels_per_tile": a.pooled_pixels_per_tile, "displacement_sweep_hr_px": list(a.displacement_sweep_hr_px)},
               "manifest_digest": getattr(dataset, "manifest_digest", None), "system": _clean(system.provenance())}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


class _CountingModel:
    """Counts the ensemble members the model was asked for, so a failure can be attributed to a member (the first member failing is a model failure)."""

    def __init__(self, model: Callable[[torch.Tensor], torch.Tensor]):
        self._model = model
        self.calls = 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self._model(x)


def run_tta_for_tile(system: System, sample: Any, transforms: Sequence[str], *, seed: int) -> Any:
    """The deployed ensemble (frame.uncertainty.run_tta_ensemble over the tile engine) for one tile; returns the EnsembleRunResult, or ``(reason, detail)`` on failure.

    Public because later analyses (frame.downstream) run exactly the same ensemble: the failure taxonomy (``model_failure`` for the first view, ``tta_member_failure`` naming the view,
    ``prediction_not_finite`` when the tile engine refuses a non-finite output) is shared.
    """
    tile_model = getattr(system, "tile_model", None)
    if tile_model is None:
        return ("model_failure", f"system {system.name!r} exposes no tile model")
    counting = _CountingModel(TiledModel(tile_model, system.tiling, collect_seam_diagnostic=False))
    try:
        x = sample.lr.float().to(system.device)
        return run_tta_ensemble(counting, x, [TRANSFORMS[n] for n in transforms], seed=seed, keep_per_member_predictions=True)
    except Exception as exc:                                     # noqa: BLE001 -- any failure of a model / worker is a recorded exclusion, never a crash and never a zero
        member = counting.calls - 1
        # the tile engine refuses a non-finite model output (frame.tiling): that is a non-finite prediction, not a crashed model
        reason = "prediction_not_finite" if "returned NaN or Inf" in str(exc) else ("model_failure" if member <= 0 else "tta_member_failure")
        return (reason, f"ensemble member {max(member, 0)} ({transforms[min(max(member, 0), len(transforms) - 1)]}) failed: {type(exc).__name__}: {exc}"[:400])


def _run_tta(system: System, sample: Any, config: ReliabilityConfig) -> Any:
    return run_tta_for_tile(system, sample, config.tta.transforms, seed=config.tta.seed)


def run_reliability(
    config: ReliabilityConfig,
    *,
    base_dir: Optional[Path] = None,
    systems: Optional[Sequence[System]] = None,
    datasets: Optional[Sequence[Any]] = None,
    mamba_client_factory: Optional[Callable[..., Any]] = None,
    progress: Optional[Callable[[str], None]] = None,
    reuse_evidence: bool = False,
) -> Dict[str, Any]:
    from frame.train.record import environment_info, git_info

    say = progress or (lambda message: None)
    out_dir = resolve_path(config.output_dir, base_dir)
    existing = [n for n in OUTPUT_FILES if (out_dir / n).exists()]
    if existing:
        raise EvaluationError(f"{out_dir} already contains results ({', '.join(existing)}); choose a new output directory. Nothing was overwritten.")
    device = _resolve_device(config.device)
    tiling, metric_cfg = config.tiling_config(), config.metric_config()
    cache_root = resolve_path(config.cache_dir, base_dir) / config.name if config.cache_dir else None

    built = list(datasets) if datasets is not None else [build_dataset(spec, base_dir=base_dir) for spec in config.datasets]
    validations = {ds.name: ds.validate(check_files=True) for ds in built}

    unavailable: List[Dict[str, Any]] = []
    live: List[System] = []
    if systems is not None:
        live = list(systems)
    else:
        for spec in config.systems:
            try:
                live.append(build_system(spec, tiling, device=device, base_dir=base_dir, mamba_client_factory=mamba_client_factory))
            except SystemUnavailableError as exc:
                unavailable.append({"name": spec.name, "kind": spec.kind, "reason": str(exc)})
    try:
        for system in live:
            for ds in built:
                check_no_training_overlap(system, dataset_kind=ds.kind, scene_ids=[r.scene_id for r in ds.records])
        if not live:
            raise EvaluationError(f"no system is available: {unavailable}")

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(config.to_json(), encoding="utf-8")
        reference = BicubicSystem("bicubic")
        rows: Dict[str, Dict[str, List[Dict[str, Any]]]] = {ds.name: {s.name: [] for s in live} for ds in built}
        arrays: Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]] = {ds.name: {s.name: {} for s in live} for ds in built}
        dataset_summary: Dict[str, Any] = {}

        with open(out_dir / "metrics.jsonl", "w", encoding="utf-8") as sink:
            def write(row: Dict[str, Any]) -> None:
                sink.write(json.dumps(_clean(row), sort_keys=True, allow_nan=False) + "\n")

            for ds in built:
                valid_ids = {r.sample_id for r in validations[ds.name].valid_records}
                tiles: List[Dict[str, Any]] = []
                unreadable: List[Dict[str, Any]] = []
                by_level: Dict[str, int] = {}
                by_reason: Dict[str, int] = {}
                per_system = {s.name: {"eligible": 0, "excluded_after_gate": {}, "reused_evidence": 0} for s in live}
                digests = {s.name: evidence_digest(config, s, ds) for s in live}
                for bad in validations[ds.name].invalid:
                    write({"type": "sample_invalid", "dataset": ds.name, **bad})
                for record in sorted(ds.records, key=lambda r: r.sample_id):
                    if record.sample_id not in valid_ids:
                        continue
                    try:
                        sample = ds.load(record)
                    except Exception as exc:                     # noqa: BLE001
                        info = {"sample_id": record.sample_id, "error": f"{type(exc).__name__}: {exc}"[:300]}
                        unreadable.append(info)
                        write({"type": "sample_unreadable", "dataset": ds.name, "reason": "sample_unreadable", **info})
                        continue
                    say(f"{ds.name} {sample.sample_id}")
                    baseline = reference.infer(sample.lr).sr.numpy().astype(np.float64)
                    gate = evaluate_reference(sample, baseline, config.eligibility, config.alignment, metric_cfg)
                    al = gate["alignment"] or {}
                    tiles.append({"sample_id": sample.sample_id, "scene_unit": sample.scene_group, "category": sample.category, "evidence_level": gate["evidence_level"], "reason": gate["reason"],
                                  "detail": gate["detail"], "valid_fraction": gate["valid_fraction"], "alignment_status": al.get("status"), "raw_displacement_hr_px": al.get("raw_magnitude"),
                                  "correction_hr_px": al.get("correction"), "residual_hr_px": al.get("residual_magnitude"), "quadrant_spread_hr_px": al.get("quadrant_spread")})
                    by_level[gate["evidence_level"]] = by_level.get(gate["evidence_level"], 0) + 1
                    if gate["status"] != "eligible":
                        by_reason[gate["reason"]] = by_reason.get(gate["reason"], 0) + 1
                        write(exclusion_row(dataset=ds.name, system=None, sample_id=sample.sample_id, scene_unit=sample.scene_group, category=sample.category, gate=gate))
                        continue
                    hr_np, mask_np = sample.hr.numpy().astype(np.float64), sample.hr_mask.numpy()
                    for system in live:
                        cached = None
                        stem = cache_root / ds.name / system.name / _safe(sample.sample_id) if cache_root is not None else None
                        if reuse_evidence and stem is not None:
                            try:
                                row_c, arr_c = load_evidence(stem)
                                if row_c.get("evidence_digest") == digests[system.name] and row_c.get("status") == "eligible":
                                    cached = (row_c, arr_c)
                            except (FileNotFoundError, ValueError, KeyError):
                                cached = None
                        if cached is not None:
                            row, arr = cached
                            write(row)
                            rows[ds.name][system.name].append(row)
                            arrays[ds.name][system.name][sample.sample_id] = arr
                            per_system[system.name]["eligible"] += 1
                            per_system[system.name]["reused_evidence"] += 1
                            continue
                        result = _run_tta(system, sample, config)
                        outcome: Any
                        if isinstance(result, tuple):
                            outcome = exclusion_row(dataset=ds.name, system=system.name, sample_id=sample.sample_id, scene_unit=sample.scene_group, category=sample.category, reason=result[0], detail=result[1])
                        else:
                            ens = EnsembleArrays(mean=result.mean_prediction.numpy().astype(np.float64), std=result.std_prediction.numpy().astype(np.float64),
                                                 member0=result.per_member_predictions[0].numpy().astype(np.float64), n_members=result.n, transform_names=tuple(result.transform_names),
                                                 seed=result.seed, seconds_total=float(result.total_seconds), seconds_per_member=tuple(float(s) for s in result.per_member_inference_seconds))
                            if ens.mean.shape != hr_np.shape:
                                outcome = exclusion_row(dataset=ds.name, system=system.name, sample_id=sample.sample_id, scene_unit=sample.scene_group, category=sample.category,
                                                        reason="reference_geometry_invalid", detail=f"prediction {ens.mean.shape} does not match the reference {hr_np.shape}")
                            else:
                                outcome = compute_tile_evidence(TileInputs(dataset=ds.name, system=system.name, sample_id=sample.sample_id, scene_unit=sample.scene_group, category=sample.category,
                                                                           bands=tuple(sample.bands), scale=int(sample.scale), bicubic=baseline, hr=hr_np, mask=mask_np, gate=gate, ensemble=ens), config)
                            del result, ens
                        if isinstance(outcome, dict):
                            ex = per_system[system.name]["excluded_after_gate"]
                            ex[outcome["reason"]] = ex.get(outcome["reason"], 0) + 1
                            write(outcome)
                            continue
                        row, arr = outcome
                        row["evidence_digest"] = digests[system.name]
                        write(row)
                        rows[ds.name][system.name].append(row)
                        arrays[ds.name][system.name][sample.sample_id] = arr
                        per_system[system.name]["eligible"] += 1
                        if stem is not None:
                            save_evidence(stem, row, arr)
                    del sample, baseline, hr_np
                total = len(ds.records)
                dataset_summary[ds.name] = {
                    "kind": ds.kind, "evidence_class": ds.evidence_class, "role": ds.role, "split": ds.spec.split, "manifest_digest": ds.manifest_digest, "info": ds.info,
                    "counts": {"total": total, "invalid": len(validations[ds.name].invalid), "unreadable": len(unreadable), "reference_gate_evaluated": len(tiles)},
                    "gate": {"by_evidence_level": dict(sorted(by_level.items())), "excluded_by_reason": dict(sorted(by_reason.items()))},
                    "tiles": tiles, "per_system": per_system, "invalid": validations[ds.name].invalid, "unreadable": unreadable,
                    "scene_units_eligible": sorted({t["scene_unit"] for t in tiles if t["evidence_level"] == "pixel_level_eligible"}),
                    "scene_units_all": sorted({t["scene_unit"] for t in tiles}),
                }

        analyses: Dict[str, Dict[str, Any]] = {}
        comparisons: Dict[str, Any] = {}
        for ds in built:
            analyses[ds.name] = {}
            for system in live:
                say(f"analysing {ds.name} / {system.name}")
                analyses[ds.name][system.name] = AN.analyse_system(rows[ds.name][system.name], arrays[ds.name][system.name], config)
            if len(live) > 1 and all(rows[ds.name][s.name] for s in live):
                comparisons[ds.name] = AN.compare_systems({s.name: rows[ds.name][s.name] for s in live}, config)

        system_info = {s.name: {"available": True, "provenance": _clean(s.provenance())} for s in live}
        for u in unavailable:
            system_info[u["name"]] = {"available": False, "kind": u["kind"], "reason": u["reason"]}
        summary: Dict[str, Any] = {
            "name": config.name, "status": "completed", "config_digest": config.digest(), "finished_utc": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat(),
            "code": {"git": git_info(REPO_ROOT)}, "environment": _clean(environment_info(device)), "metric_configuration": config.metrics, "tiling": tiling.describe(),
            "tta": {"transforms": list(config.tta.transforms), "n_members": len(config.tta.transforms), "seed": config.tta.seed,
                    "definition": "frame.uncertainty.run_tta_ensemble over frame.tiling.TiledModel: the model is run on each geometric view of the whole scene and every prediction is de-transformed; "
                                  "stability = per-pixel population standard deviation over the members, averaged over the four bands; the product evaluated is the ensemble mean"},
            "alignment": _clean(config.to_dict()["alignment"]), "eligibility": config.to_dict()["eligibility"], "analysis_settings": _clean(config.to_dict()["analysis"]),
            "bootstrap": config.to_dict()["bootstrap"], "systems": system_info, "systems_unavailable": unavailable, "datasets": dataset_summary,
            "overview": {d: {s: {"status": a["status"], "n_tiles": a["n_tiles"], "n_units": a["n_units"], "descriptive_only": a.get("descriptive_only")} for s, a in per.items()} for d, per in analyses.items()},
            "comparison_available_for": sorted(comparisons),
            "notes": ["Stability is a relative model-stability proxy, not a calibrated uncertainty; see calibration.json for what was and was not shown.",
                      "Only tiles whose reference passed the registration gate are analysed; excluded tiles are listed with machine-readable reasons in metrics.jsonl and here.",
                      "Datasets are analysed separately and never pooled."],
        }
        from frame.reliability.report import render_readme

        analyses_c, comparisons_c, summary_c = _clean(analyses), _clean(comparisons), _clean(summary)
        dump = lambda name, obj: (out_dir / name).write_text(json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")   # noqa: E731
        dump("correlations.json", {"definition": "association between stability and error; see summary.json for the evidence and README.md for the reading",
                                   "datasets": {d: {s: {k: a.get(k) for k in ("status", "n_tiles", "n_units", "descriptive_only", "pixel_level", "cell_level", "tile_level", "scene_level", "detail_relationship", "interpretation")}
                                                    for s, a in per.items()} for d, per in analyses_c.items()}, "comparison": comparisons_c})
        dump("risk_coverage.json", {d: {s: a.get("risk_coverage") for s, a in per.items()} for d, per in analyses_c.items()})
        dump("detection.json", {d: {s: a.get("high_error_detection") for s, a in per.items()} for d, per in analyses_c.items()})
        dump("calibration.json", {d: {s: a.get("calibration") for s, a in per.items()} for d, per in analyses_c.items()})
        dump("summary.json", {**summary_c, "cost": {d: {s: a.get("tta_cost") for s, a in per.items()} for d, per in analyses_c.items()}})
        (out_dir / "README.md").write_text(render_readme(summary_c, analyses_c, comparisons_c, config), encoding="utf-8")
        return {**summary_c, "analyses": analyses_c, "comparisons": comparisons_c}
    finally:
        if systems is None:
            for s in live:
                s.close()
