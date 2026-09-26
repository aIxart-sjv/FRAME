"""README rendering for an evaluation run (Phase 5). Reports measured numbers side by side and never ranks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from frame.evaluate.aggregate import headline_metrics

_LABELS = {
    "reference_accuracy.psnr_db": "PSNR (dB) ↑", "reference_accuracy.ssim": "SSIM ↑", "reference_accuracy.rmse": "RMSE ↓", "reference_accuracy.mae": "MAE ↓",
    "reference_accuracy.sam_degrees": "SAM (°) ↓", "reference_accuracy.ergas": "ERGAS ↓",
    "indices.NDVI.mae": "NDVI MAE ↓", "indices.NDVI.bias": "NDVI bias", "indices.NDVI.pearson_r": "NDVI r ↑", "indices.NDWI.mae": "NDWI MAE ↓",
    "band_ratios.B08/B04.bias_log_ratio": "log(B08/B04) bias",
    "spatial_detail.hf_relative_error": "detail rel. error (1 = no detail added)", "spatial_detail.hf_correlation": "detail correlation ↑", "spatial_detail.hf_energy_ratio": "detail energy ratio (1 = same amount)",
    "spatial_detail.gradient_correlation": "edge correlation ↑", "spatial_detail.phase_shift_px.magnitude": "shift (HR px)", "seam.ratio": "seam / interior error",
    "self_consistency.overall.mae": "MAE vs LR", "self_consistency.overall.rmse": "RMSE vs LR", "self_consistency.ndvi.mae": "NDVI MAE vs LR",
}
_GROUPS = [
    ("Reference accuracy (SR vs the independent HR reference)", ["reference_accuracy.psnr_db", "reference_accuracy.ssim", "reference_accuracy.rmse", "reference_accuracy.mae",
                                                                "reference_accuracy.sam_degrees", "reference_accuracy.ergas"]),
    ("Derived indices and band ratios (vs the HR reference)", ["indices.NDVI.mae", "indices.NDVI.bias", "indices.NDVI.pearson_r", "indices.NDWI.mae", "band_ratios.B08/B04.bias_log_ratio"]),
    ("Spatial / detail correctness (vs the HR reference)", ["spatial_detail.hf_relative_error", "spatial_detail.hf_correlation", "spatial_detail.hf_energy_ratio",
                                                            "spatial_detail.gradient_correlation", "spatial_detail.phase_shift_px.magnitude", "seam.ratio"]),
]


def _f(v: Optional[float], digits: int = 4) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}" if abs(v) < 100 else f"{v:.2f}"


def _cell(entry: Optional[Dict[str, Any]]) -> str:
    if not entry or entry.get("mean") is None:
        return "n/a"
    std = entry.get("std")
    return f"{_f(entry['mean'])} ± {_f(std)}" if std is not None else _f(entry["mean"])


def _table(headline: Dict[str, Any], metrics: Sequence[str]) -> str:
    systems = list(headline)
    lines = ["| system | n units | " + " | ".join(_LABELS.get(m, m) for m in metrics) + " |", "|---|---|" + "---|" * len(metrics)]
    for s in systems:
        n = max((headline[s][m]["n_units"] for m in metrics if m in headline[s]), default=0)
        lines.append(f"| {s} | {n} | " + " | ".join(_cell(headline[s].get(m)) for m in metrics) + " |")
    return "\n".join(lines)


def render_readme(summary: Dict[str, Any], aggregates: Dict[str, Any], config: Any) -> str:
    tau = summary["metric_configuration"]["hallucination_taus"][0]
    out: List[str] = [
        f"# Evaluation `{summary['name']}`",
        "",
        f"Status: **{summary['status']}**. Measured by `python -m frame.evaluate run`; every number below is a mean ± standard deviation over **scene units** (tiles of one scene averaged first), "
        "with `n units` beside it. **There is no ranking here and none should be read into it**: systems are listed in configuration order, differences are paired and labelled, and results of "
        "different datasets are never pooled (no combined score). Where the number of units is too small for an interval or a test, results are **descriptive only**.",
        "",
        f"Code: git `{(summary['code']['git'] or {}).get('revision')}` (working tree dirty: {(summary['code']['git'] or {}).get('dirty')}); metrics `{summary['metric_configuration']['version']}`; "
        f"tiling {summary['tiling']['tile_size']} px, overlap {summary['tiling']['overlap']}, {summary['tiling']['padding_mode']} padding, {summary['tiling']['blend_mode']} blend.",
        "",
        f"Preprocessing (all systems): {summary['systems'][next(iter(summary['systems']))].get('provenance', {}).get('value_convention', 'reflectance fraction = DN / reflectance_scale; no BOA offset')}; "
        "band order B04, B03, B02, B08 (FRAME's RGBN order, selected by name from the dataset files, not the files' own order).",
        "",
        "## Systems evaluated",
        "| system | model | hard constraint | weights / checkpoint |", "|---|---|---|---|",
    ]
    for name, info in summary["systems"].items():
        if not info["available"]:
            out.append(f"| {name} | — | — | **unavailable**: {info['reason']} |")
            continue
        p = info["provenance"]
        w = p.get("weights")
        weights = "none" if not w else ", ".join(f"{k}:{str(v)[:12]}" for k, v in w.items() if k.endswith("sha256") or "sha256" in k or k in ("model.safetensor", "hard_constraint.safetensor"))
        out.append(f"| {name} | {p.get('model_name') or p.get('kind')} | {p.get('hard_constraint')} | {weights or '—'} |")
    matrix = summary.get("evaluation_matrix", [])
    if matrix:
        datasets = sorted({m["dataset"] for m in matrix}, key=lambda d: [m["dataset"] for m in matrix].index(d))
        systems_in_order = list(dict.fromkeys(m["system"] for m in matrix))
        first = {m["dataset"]: m for m in matrix}
        out += ["", "## Evaluation matrix", "",
                "Each cell is samples evaluated / records in the dataset. Common to every cell of a row: model and weights as in the table above; hard constraint and tiling as stated there.", "",
                "| system | " + " | ".join(datasets) + " |", "|---|" + "---|" * len(datasets)]
        for name in systems_in_order:
            cells = []
            for d in datasets:
                m = next((x for x in matrix if x["system"] == name and x["dataset"] == d), None)
                cells.append(f"{m['n_evaluated']}/{m['n_samples']}" if m else "—")
            out.append(f"| {name} | " + " | ".join(cells) + " |")
        out += ["", "| dataset | evidence class | split | scenes | LR | HR | bands |", "|---|---|---|---|---|---|---|"]
        for d in datasets:
            m = first[d]
            lr = f"{m['lr_resolution_m']:g} m" if m["lr_resolution_m"] else "n/a"
            hr = f"{m['hr_resolution_m']:g} m" if m["hr_resolution_m"] else "n/a"
            out.append(f"| {d} | {m['evidence_class']} | {m['split']} | {m['scenes']} | {lr} | {hr} | {', '.join(m['bands'] or [])} |")
    out += ["", "Hard constraint: `True` = the intended system (low-frequency Fourier constraint inside each 512×512 tile output); `False` = a system without it (the ablation of Lite, or a trained checkpoint); `None` = not applicable (bicubic).", ""]

    for evidence, datasets in summary["sections"].items():
        out += [f"## Evidence class: `{evidence}`", ""]
        for name, block in datasets.items():
            d = summary["datasets"][name]
            c = d["counts"]
            out += [f"### Dataset `{name}` ({d['kind']}, role {d['role']}, split {d['split']})", "",
                    f"Records {c['total']}: **evaluated {c['evaluated']}**, skipped {c['skipped']}, invalid {c['invalid']}, unreadable {c['unreadable']}; {len(d['scene_units'])} scene units; manifest digest `{d['manifest_digest'][:16]}…`."]
            if d["skipped"]:
                out.append("Skipped (listed, not scored): " + "; ".join(f"{s['sample_id']} ({s['reason'].split(':')[0]})" for s in d["skipped"][:10]))
            if d["invalid"]:
                out.append("Invalid (listed, not scored): " + "; ".join(f"{i['sample_id']} {i['codes']}" for i in d["invalid"][:10]))
            if d["unreadable"]:
                out.append("Unreadable: " + "; ".join(f"{u['sample_id']} ({u['error'][:60]})" for u in d["unreadable"][:10]))
            if d["failures"]:
                out.append("Inference failures per system: " + str(d["failures"]))
            if d["categories"]:
                out.append(f"Dataset-defined categories (descriptive only): {d['categories']}")
            out.append("")
            headline = block["headline"]
            for title, metrics in _GROUPS:
                out += [f"**{title}**", "", _table(headline, metrics), ""]
            band_cols = [f"per_band.{b}.{m}" for b in ("B02", "B03", "B04", "B08") for m in ("rmse", "bias")]
            out += ["**Per-band reflectance error** (RMSE ↓ / bias = SR − reference; bands are the RGBN bands every system produces)", "",
                    "| system | " + " | ".join(f"{b} RMSE" for b in ("B02", "B03", "B04", "B08")) + " | " + " | ".join(f"{b} bias" for b in ("B02", "B03", "B04", "B08")) + " |",
                    "|---|" + "---|" * 8]
            order = [f"per_band.{b}.rmse" for b in ("B02", "B03", "B04", "B08")] + [f"per_band.{b}.bias" for b in ("B02", "B03", "B04", "B08")]
            for s, row in headline.items():
                out.append(f"| {s} | " + " | ".join(_cell(row.get(m)) for m in order) + " |")
            out.append("")
            det = [f"detail_analysis.{tau}.supported_synthesis", f"detail_analysis.{tau}.unsupported_detail", f"detail_analysis.{tau}.omission", f"detail_analysis.{tau}.mse_skill_vs_baseline"]
            out += [f"**What was added relative to bicubic, judged against the reference** (element-wise; τ = {tau} reflectance; fractions of valid elements; the reference is a different sensor, so "
                    "\"unsupported\" means \"not confirmed by this reference\", not \"false\")", "",
                    "| system | n units | supported synthesis | unsupported detail | omission | MSE skill vs bicubic |", "|---|---|---|---|---|---|"]
            for s, row in headline.items():
                if det[0] not in row:
                    continue
                out.append(f"| {s} | {row[det[0]]['n_units']} | " + " | ".join(_cell(row.get(m)) for m in det) + " |")
            out += ["", "**Self-consistency (SR → area-average → compare with the LR)**. This is NOT accuracy against an HR reference: a system can average back to the LR perfectly and still be far from the reference.", "",
                    _table(headline, ["self_consistency.overall.mae", "self_consistency.overall.rmse", "self_consistency.ndvi.mae"]), ""]
            by_cat = aggregates["datasets"].get(name, {}).get("by_category")
            if by_cat:
                out += ["**By dataset-defined category** (the dataset's own labels, never inferred; descriptive only: few samples per category, no interval or test)", "",
                        "| category | n samples | n units | system | PSNR (dB) | SAM (°) | detail rel. error |", "|---|---|---|---|---|---|---|"]
                for cat, blk in by_cat.items():
                    for sys_name, mets in blk["systems"].items():
                        cells = [_cell(mets.get(m)) for m in ("reference_accuracy.psnr_db", "reference_accuracy.sam_degrees", "spatial_detail.hf_relative_error")]
                        out.append(f"| {cat} | {blk['n_samples']} | {blk['n_units']} | {sys_name} | " + " | ".join(cells) + " |")
                out.append("")
            paired = block["paired_headline"]
            if paired:
                out += ["**Paired differences A − B over scene units** (sign convention: A minus B; for error metrics lower is smaller error; a difference is not a ranking)", "",
                        "| A − B | metric | n units | mean difference | median | 95% CI | Wilcoxon p | label |", "|---|---|---|---|---|---|---|---|"]
                for pair, table in paired.items():
                    for m in ("reference_accuracy.psnr_db", "reference_accuracy.ssim", "reference_accuracy.rmse", "reference_accuracy.sam_degrees", "reference_accuracy.ergas",
                              "indices.NDVI.mae", "spatial_detail.hf_relative_error", "self_consistency.overall.mae"):
                        t = table.get(m)
                        if not t:
                            continue
                        ci = "n/a" if t["ci"][0] is None else f"[{_f(t['ci'][0])}, {_f(t['ci'][1])}]"
                        out.append(f"| {pair} | {_LABELS.get(m, m)} | {t['n']} | {_f(t['mean_difference'])} | {_f(t['median_difference'])} | {ci} | {_f(t['p'])} | {t['label'].replace('_', ' ')} |")
                out.append("")
    if summary["seed_groups"]:
        out += ["## Multi-seed groups", "", "Systems sharing a group are the same model trained with different seeds. The tables give the mean and standard deviation ACROSS seeds of each seed's unit-level mean.", ""]
        for group, g in summary["seed_groups"].items():
            out.append(f"`{group}`: {g['n_systems']} systems ({', '.join(g['systems'])})")
            for ds, table in g["datasets"].items():
                cells = [f"{_LABELS.get(m, m)} {_f(v['mean'])} ± {_f(v['std'])}" for m, v in table.items() if m in ("reference_accuracy.psnr_db", "reference_accuracy.rmse", "reference_accuracy.sam_degrees", "spatial_detail.hf_relative_error")]
                out.append(f"- {ds}: " + "; ".join(cells))
            out.append("")
    out += [
        "## How to read this",
        "- **Valid-pixel rule**: an HR pixel is scored only if it is not all-band nodata, no evaluated band equals the HR nodata value, and all values are finite; the fractions are in `metrics.jsonl` (`quality`). "
        "Windowed metrics (SSIM, detail, gradients) additionally exclude a border of their own window width so no window touches nodata.",
        "- **Reference accuracy** compares SR with an independent HR reference. For real cross-sensor pairs (SEN2NEON, OpenSR-Test) the reference is a different sensor with its own radiometry and "
        "registration error, so no number here is an absolute error against truth; for synthetic pairs the reference is the ground truth the LR was made from.",
        "- **Detail metrics**: relative error 1.0 is exactly the score of adding no detail; below 1 the added detail matches the reference, above 1 it is worse than none. The energy ratio says how much detail "
        "was added, not whether it is right.",
        "- **Registration**: the `shift` column is the displacement of the SR relative to the reference (cross-correlation, HR pixels). A non-zero value means the reference is not registered to "
        "the Sentinel-2 grid (close to the same for every system, since all start from the same LR grid), and every pixel-wise and detail score of that scene is then pessimistic for a correctly placed detail. "
        "`python -m frame.evaluate shift CONFIG --dataset NAME` measures how much a known displacement changes the scores.",
        "- **Statistics**: units are scenes/acquisitions, not pixels; intervals are percentile bootstrap over units; the paired test is Wilcoxon signed-rank; below 5 units no interval and below 6 pairs "
        "no test is reported (descriptive only). No multiple-comparison correction.",
        "- ↑ / ↓ mark which direction is smaller error or closer agreement, not a ranking of systems.",
        "",
        "Files: `config.json` (as run), `metrics.jsonl` (one row per sample × system, plus skipped/invalid/unreadable rows), `aggregates.json` (all metrics, sample- and unit-level, paired comparisons), `summary.json` (provenance, counts, evaluation matrix).",
        "",
    ]
    return "\n".join(out)
