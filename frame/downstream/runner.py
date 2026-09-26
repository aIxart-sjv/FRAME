"""The downstream runner (Phase 7).

    config -> datasets (role safety, reference + geometry validation) -> frozen models (train/eval overlap guard)
           -> per tile: load, bicubic baseline, the PHASE 6 REFERENCE ELIGIBILITY GATE (valid pixels, finiteness, registration)
           -> eligible tiles only: TTA ensemble of every model (the product = ensemble mean, stability = its spread) -> the recorded alignment crop applied to every grid
              -> NDVI of the reference, of the LR (native pixel values), of bicubic and of every model on identical pixels -> fixed regions at every declared scale -> region tables (cached outside git)
           -> per dataset and scale: scene-unit metrics, paired differences against the baselines, stability / texture association, risk-coverage
           -> experiments/downstream/runs/<name>/

Only tiles whose reference passed the gate get ANY quantity derived from that reference: an excluded tile is listed with its machine-readable reason and its candidate regions are counted from the
grid geometry alone. If one model fails on a tile the tile is excluded for every system (so the comparisons of a dataset always share the same evidence). Datasets are analysed separately.
Nothing is ranked, no stability value is called calibrated, and no thresholded NDVI proxy is called land cover.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from frame.downstream import association as AS
from frame.downstream import metrics as MT
from frame.downstream.config import DownstreamConfig
from frame.downstream.errors import DownstreamError
from frame.downstream.ndvi import common_valid_mask, ndvi_map, reference_valid_mask
from frame.downstream.regions import EXCLUSION_INSUFFICIENT, candidate_region_count, extract_regions, make_grid, stack_tables
from frame.evaluate.config import resolve_path
from frame.evaluate.datasets import MASK_RULE, build_dataset
from frame.evaluate.errors import EvaluationError, SystemUnavailableError
from frame.evaluate.runner import REPO_ROOT, _clean, _resolve_device
from frame.evaluate.stats import MIN_UNITS_CI
from frame.evaluate.systems import PREPROCESSING, BicubicSystem, System, build_system, check_no_training_overlap
from frame.reliability.alignment import apply_correction
from frame.reliability.eligibility import GATE_VERSION, evaluate_reference
from frame.reliability.runner import run_tta_for_tile
from frame.reliability.targets import stability_map, texture_baseline

OUTPUT_FILES = ("config.json", "tiles.jsonl", "summary.json", "downstream_metrics.json", "association.json", "risk_coverage.json", "README.md")
LR, BICUBIC, REFERENCE = "lr_native", "bicubic", MT.REFERENCE
EXCLUDED = "excluded_from_downstream_primary_analysis"
FORMULA = "NDVI = (NIR - Red) / (NIR + Red) = (B08 - B04) / (B08 + B04)"
REGION_DEFINITION = ("Fixed square cells on the aligned HR grid (the cells frame.reliability uses): 4 x 4 HR px = 10 m, the footprint of one Sentinel-2 pixel (the requirements' '10 m pixel = 16 cells at 2.5 m'), and "
                     "16 x 16 HR px = 40 m; every region starts on a multiple of its size in prediction coordinates; a region needs at least the declared share of valid pixels")
NOTES = ("Stability is a relative model-stability proxy, not a calibrated uncertainty or confidence; nothing here changes that.",
         "A thresholded NDVI proxy is not land cover: no land-cover labels are used or implied.",
         "Regions of one scene are not independent observations: every inference is over scene units.",
         "Only tiles whose reference passed the registration gate are analysed; excluded tiles are listed with machine-readable reasons.",
         "No ranking of systems is made anywhere: values and paired differences are reported with their uncertainty when the unit count allows it.")


def _safe(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", sample_id)


def _th(value: float) -> str:
    return f"{value:g}"


def regions_digest(config: DownstreamConfig, models: Sequence[System], dataset: Any) -> str:
    """Identity of everything that determines a tile's region tables (settings, dataset revision, models by their provenance); analysis-only settings are not part of it."""
    d = config.to_dict()
    payload = {"gate": GATE_VERSION, "tiling": config.tiling_config().describe(), "tta": d["tta"], "alignment": d["alignment"], "eligibility": d["eligibility"], "ndvi": d["ndvi"],
               "thresholds": [_th(t) for t in config.thresholds()], "regions": d["regions"], "metrics": config.metrics, "manifest_digest": getattr(dataset, "manifest_digest", None),
               "models": {m.name: _clean(m.provenance()) for m in models}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tile_tables(sample: Any, gate: Dict[str, Any], baseline: np.ndarray, ensembles: Dict[str, Tuple[np.ndarray, np.ndarray]], config: DownstreamConfig) -> Dict[int, Dict[str, Any]]:
    """Region tables (one per declared scale) of one ELIGIBLE tile, on the aligned overlap."""
    bands = tuple(sample.bands)
    scale = int(sample.scale)
    lr = np.asarray(sample.lr.numpy(), dtype=np.float32)
    lr_native = np.repeat(np.repeat(lr, scale, axis=1), scale, axis=2)
    base32 = np.asarray(baseline, dtype=np.float32)
    channels: List[np.ndarray] = [lr_native, base32]
    layout: List[Tuple[str, slice]] = []
    for name, (mean, std) in ensembles.items():
        added = np.abs(np.asarray(mean, dtype=np.float64) - baseline).mean(axis=0)
        start = sum(c.shape[0] for c in channels)
        channels += [np.asarray(mean, dtype=np.float32), stability_map(std)[None].astype(np.float32), added[None].astype(np.float32)]
        layout.append((name, slice(start, start + len(bands))))
    stack = np.concatenate(channels, axis=0)
    hr = np.asarray(sample.hr.numpy(), dtype=np.float64)
    corr = gate["alignment"]["correction"]
    stack_c, hr_c, mask_c = apply_correction(stack, hr, np.asarray(sample.hr_mask.numpy(), dtype=bool), corr)
    c = len(bands)
    lr_c, bic_c = stack_c[:c], stack_c[c:2 * c]

    spec = config.ndvi
    ref_valid = reference_valid_mask(hr_c, bands, mask_c, spec)
    ndvi: Dict[str, np.ndarray] = {REFERENCE: ndvi_map(hr_c, bands, spec, valid=ref_valid)[0], LR: None, BICUBIC: None}
    computable: List[np.ndarray] = []
    for name, src in ((LR, lr_c), (BICUBIC, bic_c)):
        ndvi[name], ok = ndvi_map(src, bands, spec, valid=ref_valid)
        computable.append(ok)
    maps: Dict[str, np.ndarray] = {"texture": texture_baseline(bic_c)}
    for name, sl in layout:
        ndvi[name], ok = ndvi_map(stack_c[sl], bands, spec, valid=ref_valid)
        computable.append(ok)
        maps[f"stab:{name}"] = stack_c[sl.stop]
        maps[f"added:{name}"] = stack_c[sl.stop + 1]
    common = common_valid_mask(ref_valid, computable)
    tables: Dict[int, Dict[str, Any]] = {}
    for size in config.regions.scales_hr_px:
        grid = make_grid(common.shape, size, corr)
        tables[size] = extract_regions(maps, common, grid, min_valid_fraction=config.regions.min_valid_fraction, thresholds=config.thresholds(), systems=[LR, BICUBIC, *[n for n, _ in layout]],
                                       reference=REFERENCE, ndvi=ndvi)
    return tables


def _save_tables(stem: Path, tables: Dict[int, Dict[str, Any]]) -> Dict[int, str]:
    shas: Dict[int, str] = {}
    stem.parent.mkdir(parents=True, exist_ok=True)
    for size, t in tables.items():
        path = Path(f"{stem}__s{size}.npz")
        np.savez_compressed(path, **{k: v for k, v in t.items() if isinstance(v, np.ndarray)})
        shas[size] = _sha256_file(path)
    return shas


def _load_tables(stem: Path, meta: Dict[str, Any], scales: Sequence[int]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for size in scales:
        with np.load(Path(f"{stem}__s{size}.npz")) as data:
            t: Dict[str, Any] = {k: data[k] for k in data.files}
        info = meta["regions"][str(size)]
        t.update(size=size, n_candidate=info["n_candidate"], excluded=dict(info["excluded"]), grid=info["grid"])
        out[size] = t
    return out


def run_downstream(
    config: DownstreamConfig,
    *,
    base_dir: Optional[Path] = None,
    systems: Optional[Sequence[System]] = None,
    datasets: Optional[Sequence[Any]] = None,
    mamba_client_factory: Optional[Callable[..., Any]] = None,
    progress: Optional[Callable[[str], None]] = None,
    reuse_regions: bool = False,
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
        started = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
        reference_system = BicubicSystem(BICUBIC)
        scales = list(config.regions.scales_hr_px)
        model_names = [s.name for s in live]
        all_systems = [LR, BICUBIC, *model_names]
        thresholds = [_th(t) for t in config.thresholds()]

        dataset_summary: Dict[str, Any] = {}
        per_dataset_tables: Dict[str, Dict[int, List[Tuple[Dict[str, Any], Dict[str, Any]]]]] = {}
        with open(out_dir / "tiles.jsonl", "w", encoding="utf-8") as sink:
            def write(row: Dict[str, Any]) -> None:
                sink.write(json.dumps(_clean(row), sort_keys=True, allow_nan=False) + "\n")

            for ds in built:
                valid_ids = {r.sample_id for r in validations[ds.name].valid_records}
                digest = regions_digest(config, live, ds)
                per_dataset_tables[ds.name] = {s: [] for s in scales}
                by_level: Dict[str, int] = {}
                gate_reasons: Dict[str, int] = {}
                counts = {s: {"candidate": 0, "candidate_in_eligible_tiles": 0, "analysed": 0, "excluded_tile_level": {"regions": 0, "by_reason": {}}, "excluded_region_level": {EXCLUSION_INSUFFICIENT: 0,
                                                                                                                                                                                "alignment_crop_edge_cells": 0}} for s in scales}
                unreadable: List[Dict[str, Any]] = []
                units_all: set = set()
                units_eligible: set = set()
                reused = 0
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
                    units_all.add(sample.scene_group)
                    hr_shape = tuple(sample.hr.shape[-2:])
                    candidate = {s: candidate_region_count(hr_shape, s) for s in scales}
                    for s in scales:
                        counts[s]["candidate"] += candidate[s]
                    baseline = reference_system.infer(sample.lr).sr.numpy().astype(np.float64)
                    gate = evaluate_reference(sample, baseline, config.eligibility, config.alignment, metric_cfg)
                    by_level[gate["evidence_level"]] = by_level.get(gate["evidence_level"], 0) + 1
                    row: Dict[str, Any] = {"type": "tile", "dataset": ds.name, "sample_id": sample.sample_id, "scene_unit": sample.scene_group, "category": sample.category,
                                           "gate_version": GATE_VERSION, "evidence_level": gate["evidence_level"], "alignment": gate["alignment"], "valid_fraction": gate["valid_fraction"],
                                           "valid_fraction_after_alignment": gate["valid_fraction_after_alignment"], "candidate_regions": {str(s): candidate[s] for s in scales}}

                    def exclude(reason: str, detail: str, level_row: Dict[str, Any] = row) -> None:
                        level_row.update(status=EXCLUDED, reason=reason, detail=detail)
                        gate_reasons[reason] = gate_reasons.get(reason, 0) + 1
                        for s in scales:
                            counts[s]["excluded_tile_level"]["regions"] += candidate[s]
                            b = counts[s]["excluded_tile_level"]["by_reason"]
                            b[reason] = b.get(reason, 0) + candidate[s]
                        write(level_row)

                    if gate["status"] != "eligible":
                        exclude(gate["reason"], gate["detail"])
                        continue
                    stem = cache_root / ds.name / _safe(sample.sample_id) if cache_root is not None else None
                    tables: Optional[Dict[int, Dict[str, Any]]] = None
                    if reuse_regions and stem is not None and Path(f"{stem}.json").is_file():
                        try:
                            meta = json.loads(Path(f"{stem}.json").read_text(encoding="utf-8"))
                            if meta.get("regions_digest") == digest and meta.get("status") == "analysed":
                                tables = _load_tables(stem, meta, scales)
                                row.update({k: meta[k] for k in ("tta", "regions", "regions_sha256")})
                                reused += 1
                        except (OSError, ValueError, KeyError):
                            tables = None
                    if tables is None:
                        ensembles: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
                        timing: Dict[str, Any] = {}
                        failure: Optional[Tuple[str, str]] = None
                        for system in live:
                            result = run_tta_for_tile(system, sample, config.tta.transforms, seed=config.tta.seed)
                            if isinstance(result, tuple):
                                failure = (result[0], f"{system.name}: {result[1]}")
                                break
                            mean, std = result.mean_prediction.numpy().astype(np.float64), result.std_prediction.numpy().astype(np.float64)
                            if mean.shape != tuple(sample.hr.shape):
                                failure = ("reference_geometry_invalid", f"{system.name}: prediction {mean.shape} does not match the reference {tuple(sample.hr.shape)}")
                                break
                            ensembles[system.name] = (mean, std)
                            secs = [float(x) for x in result.per_member_inference_seconds]
                            timing[system.name] = {"n_members": int(result.n), "tta_seconds": float(result.total_seconds), "single_pass_seconds": float(np.median(secs)) if secs else None}
                            del result
                        if failure is not None:
                            exclude(*failure)
                            continue
                        tables = _tile_tables(sample, gate, baseline, ensembles, config)
                        del ensembles
                        row["tta"] = timing
                        row["regions"] = {str(s): {"analysed": int(len(t["row"])), "n_candidate": int(t["n_candidate"]), "excluded": {k: int(v) for k, v in t["excluded"].items()}, "grid": t["grid"]}
                                          for s, t in tables.items()}
                        row["regions_sha256"] = _save_tables(stem, tables) if stem is not None else {}
                        row["regions_sha256"] = {str(k): v for k, v in row["regions_sha256"].items()} if row["regions_sha256"] else {str(s): None for s in scales}
                        if stem is not None:
                            Path(f"{stem}.json").write_text(json.dumps(_clean({**row, "status": "analysed", "regions_digest": digest}), sort_keys=True, allow_nan=False), encoding="utf-8")
                    row["status"] = "analysed"
                    units_eligible.add(sample.scene_group)
                    for s, t in tables.items():
                        counts[s]["analysed"] += int(len(t["row"]))
                        counts[s]["excluded_region_level"][EXCLUSION_INSUFFICIENT] += int(t["excluded"].get(EXCLUSION_INSUFFICIENT, 0))
                        counts[s]["candidate_in_eligible_tiles"] += candidate[s]
                        counts[s]["excluded_region_level"]["alignment_crop_edge_cells"] += candidate[s] - int(t["n_candidate"])
                        per_dataset_tables[ds.name][s].append(({"dataset": ds.name, "scene_unit": sample.scene_group, "tile_id": sample.sample_id}, t))
                    write(row)
                    del sample, baseline
                dataset_summary[ds.name] = {
                    "kind": ds.kind, "evidence_class": ds.evidence_class, "role": ds.role, "split": ds.spec.split, "manifest_digest": ds.manifest_digest, "info": ds.info,
                    "counts": {"total": len(ds.records), "invalid": len(validations[ds.name].invalid), "unreadable": len(unreadable), "gate_evaluated": sum(by_level.values())},
                    "gate": {"by_evidence_level": dict(sorted(by_level.items())), "excluded_by_reason": dict(sorted(gate_reasons.items()))}, "regions": {str(s): counts[s] for s in scales},
                    "scene_units_all": sorted(units_all), "units_eligible": sorted(units_eligible), "invalid": validations[ds.name].invalid, "unreadable": unreadable, "reused_tiles": reused,
                }

        metrics_out: Dict[str, Any] = {}
        assoc_out: Dict[str, Any] = {}
        rc_out: Dict[str, Any] = {}
        overview: Dict[str, Any] = {}
        pairs = [(model_names[0], model_names[1])] if len(model_names) >= 2 else []
        for ds in built:
            m_block: Dict[str, Any] = {"scales": {}, "unit_metrics": {}}
            a_block: Dict[str, Any] = {"scales": {}}
            r_block: Dict[str, Any] = {"scales": {}}
            n_tiles = len(per_dataset_tables[ds.name][scales[0]])
            n_units = len(dataset_summary[ds.name]["units_eligible"])
            overview[ds.name] = {"status": "ok" if n_tiles else "no_eligible_evidence", "n_tiles": n_tiles, "n_units": n_units, "descriptive_only": n_units < MIN_UNITS_CI}
            if n_tiles:
                for s in scales:
                    say(f"analysing {ds.name} scale {s}")
                    table = stack_tables(per_dataset_tables[ds.name][s])
                    um = MT.unit_metrics(table, all_systems, thresholds, majority_fraction=config.decision.region_majority_fraction)
                    m_block["unit_metrics"][str(s)] = um
                    m_block["scales"][str(s)] = MT.aggregate_units(um, all_systems, thresholds, config, baselines=(BICUBIC, LR), pairs=pairs)
                    a_block["scales"][str(s)] = {m: AS.association(table, m, thresholds[0], config) for m in model_names}
                    r_block["scales"][str(s)] = {m: AS.risk_coverage(table, m, thresholds[0], config) for m in model_names}
            metrics_out[ds.name], assoc_out[ds.name], rc_out[ds.name] = m_block, a_block, r_block

        system_info = {s.name: {"available": True, "provenance": _clean(s.provenance())} for s in live}
        for u in unavailable:
            system_info[u["name"]] = {"available": False, "kind": u["kind"], "reason": u["reason"]}
        d = config.to_dict()
        summary: Dict[str, Any] = {
            "name": config.name, "run_id": f"{config.name}-{started.strftime('%Y%m%dT%H%M%SZ')}", "status": "completed", "config_digest": config.digest(),
            "finished_utc": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat(), "code": {"git": git_info(REPO_ROOT)}, "environment": _clean(environment_info(device)),
            "metric_configuration": config.metrics, "tiling": tiling.describe(),
            "ndvi": {"formula": FORMULA, "bands": {"red": config.ndvi.red_band, "nir": config.ndvi.nir_band, "band_order_of_inputs": list(PREPROCESSING["input_bands"])},
                     "input_scaling": PREPROCESSING["value_convention"], "settings": d["ndvi"],
                     "valid_pixel_rule": (f"{MASK_RULE}; then restricted to the aligned overlap and to pixels where the reference's Red + NIR reflectance exceeds {config.ndvi.min_reflectance_sum} "
                                          f"and NDVI is computable (|NIR + Red| > {config.ndvi.denominator_epsilon}) for the reference, the LR, bicubic and every model (identical pixels for every system)")},
            "decision": {**d["decision"], "threshold_policy_note": "0.3 is a conventional value; the requirements name no NDVI threshold. It was declared before any result and is neither selected nor tuned on results; the sensitivity thresholds are reported beside it."},
            "regions": {**d["regions"], "definition": REGION_DEFINITION}, "analysis": d["analysis"], "bootstrap": d["bootstrap"],
            "reference_gate": {"version": GATE_VERSION, "alignment": d["alignment"], "eligibility": d["eligibility"], "note": "the Phase 6 gate, unchanged"},
            "tta": {"transforms": list(config.tta.transforms), "seed": config.tta.seed, "n_members": len(config.tta.transforms)},
            "systems": system_info, "systems_unavailable": unavailable, "systems_compared": all_systems, "datasets": dataset_summary, "overview": overview,
            "secondary_tasks": {"landcover": "secondary_landcover_task_deferred_no_supported_reference",
                                "reason": "no dataset used here carries region-level land-cover labels or a defensible reference segmentation; the only label (NEON land-cover superclass) describes a whole tile"},
            "indian_data": {"status": "india_downstream_validation_unavailable", "reason": "no Indian HR reference and no Indian downstream labels exist in this repository; nothing Indian was evaluated or claimed"},
            "notes": list(NOTES),
        }
        from frame.downstream.report import render_readme

        summary_c, metrics_c, assoc_c, rc_c = _clean(summary), _clean(metrics_out), _clean(assoc_out), _clean(rc_out)
        dump = lambda name, obj: (out_dir / name).write_text(json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")       # noqa: E731
        dump("downstream_metrics.json", {"definition": "NDVI fidelity and vegetation-decision metrics of the LR (native), bicubic and each model against the HR reference, per scene unit and across scene units, "
                                                        "with paired differences against the baselines; see summary.json for the evidence", "datasets": metrics_c})
        dump("association.json", {"datasets": assoc_c})
        dump("risk_coverage.json", {"datasets": rc_c})
        dump("summary.json", summary_c)
        (out_dir / "README.md").write_text(render_readme(summary_c, metrics_c, assoc_c, rc_c, config), encoding="utf-8")
        return {**summary_c, "metrics": metrics_c, "association": assoc_c, "risk_coverage": rc_c}
    finally:
        if systems is None:
            for s in live:
                s.close()
