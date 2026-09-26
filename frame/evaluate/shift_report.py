"""README for a spatial-shift sensitivity run (see frame.evaluate.shift)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

LABELS = {
    "reference_accuracy.psnr_db": "PSNR (dB) ↑",
    "reference_accuracy.ssim": "SSIM ↑",
    "reference_accuracy.sam_degrees": "SAM (°) ↓",
    "reference_accuracy.ergas": "ERGAS ↓",
    "indices.NDVI.mae": "NDVI MAE ↓",
    "spatial_detail.hf_relative_error": "detail rel. error (1 = no detail added) ↓",
    "spatial_detail.hf_correlation": "detail correlation ↑",
}
#: metrics that also get a paired-difference table against the bicubic baseline
PAIRED = ("reference_accuracy.psnr_db", "reference_accuracy.sam_degrees", "spatial_detail.hf_relative_error")


def _f(x: Optional[float], digits: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def _cell(stat: Optional[Dict[str, Any]]) -> str:
    if not stat or stat.get("mean") is None:
        return "n/a"
    std = stat.get("std")
    return f"{stat['mean']:.3f}" + (f" ± {std:.3f}" if std is not None else "")


def _condition_label(name: str) -> str:
    if name.startswith("shift_"):
        value = name[len("shift_"):-len("_lr_px")]
        return f"`{name}` ({value} LR px)"
    return f"`{name}`"


def render_shift_readme(summary: Dict[str, Any], aggregates: Dict[str, Any]) -> str:
    c = summary["counts"]
    lines: List[str] = [
        f"# Spatial-shift sensitivity: `{summary['dataset']}`",
        "",
        f"Status: **{summary['status']}**. Requirements 142 §27: the SR of each system is displaced against the reference by known amounts and the same metrics are recomputed on the overlap. "
        "**This is an interpretation aid, not a ranking**: it shows how large a difference a systematic misregistration alone produces, so that differences between systems on this dataset can be read against it. "
        "Every number is a mean ± standard deviation over scene units and is **descriptive** unless the paired table carries an interval.",
        "",
        f"Dataset `{summary['dataset']}` ({summary['evidence_class']}, role {summary['role']}, split {summary['split']}): {c['total']} records, **{c['evaluated']} evaluated**, "
        f"{c['skipped']} skipped, {c['invalid']} invalid, {c['unreadable']} unreadable; manifest digest `{summary['manifest_digest'][:16]}…`. "
        f"Code git `{summary['code']['git'].get('revision')}` (working tree dirty: {summary['code']['git'].get('dirty')}); metrics `{summary['metric_configuration']['version']}`.",
        "",
        "**Method.** Shifts are whole HR pixels (no resampling): "
        + ", ".join(f"{lr:g} LR px = {hr} HR px" for lr, hr in zip(summary["lr_shifts"], summary["hr_shifts"]))
        + f" (scale x{summary['scale']}), along the {summary['axis']}. The SR content is displaced against the reference and both are cropped to the overlap. "
        "`aligned_to_bicubic` removes the displacement that phase correlation finds between the **bicubic baseline** and the reference, rounded to a whole HR pixel: a system-neutral estimate, "
        "identical for every system on a sample, so no system aligns itself. "
        "Not computed: classification accuracy and area error (they need a downstream task, which is outside Phase 5).",
        "",
    ]
    am = summary.get("alignment_magnitude_hr_px")
    if am:
        lines += [f"Estimated displacement of the bicubic baseline against the reference (the reference's own misregistration, HR px): median {_f(am['median'], 2)}, mean {_f(am['mean'], 2)}, "
                  f"max {_f(am['max'], 2)} over {am['n']} tiles.", ""]
    lines += ["## Systems", "| system | model | hard constraint |", "|---|---|---|"]
    for name, prov in summary["systems"].items():
        lines.append(f"| {name} | {prov.get('model_name') or prov.get('kind')} | {prov.get('hard_constraint')} |")
    lines.append("")
    for u in summary.get("systems_unavailable", []):
        lines.append(f"System `{u['name']}` UNAVAILABLE: {u['reason']}")
    if summary["failures"]:
        lines += [f"Inference failures (rows with their error in rows.jsonl): {summary['failures']}", ""]
    if summary["skipped"] or summary["invalid"] or summary["unreadable"]:
        lines += [f"Skipped {[s['sample_id'] for s in summary['skipped']]}; invalid {[i['sample_id'] for i in summary['invalid']]}; unreadable {[u['sample_id'] for u in summary['unreadable']]} (reasons in summary.json).", ""]

    conditions = summary["conditions"]
    systems = list(summary["systems"])
    for metric in summary["headline_metrics"]:
        lines += [f"## {LABELS.get(metric, metric)}", "", "| condition | " + " | ".join(systems) + " | n units |", "|---|" + "---|" * (len(systems) + 1)]
        for cond in conditions:
            block = aggregates["conditions"].get(cond, {})
            cells, n_units = [], ""
            for s in systems:
                info = block.get("systems", {}).get(s)
                stat = info["metrics"].get(metric, {}).get("unit") if info else None
                cells.append(_cell(stat))
                if stat and stat.get("n") is not None:
                    n_units = str(stat["n"])
            lines.append(f"| {_condition_label(cond)} | " + " | ".join(cells) + f" | {n_units} |")
        lines.append("")
        if metric in PAIRED:
            others = [s for s in systems if s != "bicubic"]
            if others:
                lines += [f"Paired difference (system − bicubic) over the same scene units, mean and 95% interval where the unit count allows one:", "",
                          "| condition | " + " | ".join(f"{s} − bicubic" for s in others) + " |", "|---|" + "---|" * len(others)]
                for cond in conditions:
                    paired = aggregates["conditions"].get(cond, {}).get("paired", {})
                    cells = []
                    for s in others:
                        t = paired.get(f"{s} - bicubic", {}).get(metric)
                        if not t or t.get("mean_difference") is None:
                            cells.append("n/a")
                            continue
                        ci = t.get("ci") or {}
                        interval = f" [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]" if ci.get("status") == "ok" and ci.get("ci_low") is not None else " (descriptive only)"
                        cells.append(f"{t['mean_difference']:+.3f}{interval}")
                    lines.append(f"| {_condition_label(cond)} | " + " | ".join(cells) + " |")
                lines.append("")
    lines += ["## How to read this", "",
              "* The zero row is the ordinary evaluation of this dataset (restricted to the metrics shown). Each following row scores exactly the same SR against a reference that is further displaced.",
              "* If a displacement of a fraction of an LR pixel changes a metric by more than the differences between systems, then that difference cannot be attributed to the systems on this dataset alone.",
              "* `aligned_to_bicubic` removes an estimated whole-pixel misregistration; a residual sub-pixel misregistration and any radiometric difference between sensors remain.",
              "* Nothing here is combined across datasets and nothing is a ranking.", ""]
    return "\n".join(lines)
