"""README of a downstream run (Phase 7): what was measured, what was found, what was not shown, the limitations. Tables only of measured values; no ranking; unsupported claims stated as not shown."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

MAIN_METRICS = (("ndvi_mae", "NDVI MAE"), ("ndvi_rmse", "NDVI RMSE"), ("ndvi_bias", "NDVI bias"), ("ndvi_median_abs_error", "median |NDVI err|"), ("veg_fraction_mae", "vegetation-fraction MAE"),
                ("region_decision_error_rate", "region decision error"), ("disagreement_rate", "pixel decision disagreement"), ("accuracy", "accuracy"), ("balanced_accuracy", "balanced accuracy"),
                ("false_positive_rate", "false-positive rate"), ("false_negative_rate", "false-negative rate"), ("valid_coverage", "valid coverage"))
PAIRED_METRICS = (("ndvi_mae", "NDVI MAE"), ("veg_fraction_mae", "vegetation-fraction MAE"), ("region_decision_error_rate", "region decision error"), ("disagreement_rate", "pixel decision disagreement"),
                  ("balanced_accuracy", "balanced accuracy"))
TARGET_LABEL = {"ndvi_abs_error": "|NDVI error|", "disagreement": "decision disagreement"}


def _n(x: Optional[float], d: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{d}f}"


def _ci(ci: Optional[Dict[str, Any]], d: int = 3) -> str:
    if not ci or ci.get("estimate") is None:
        return "n/a"
    if ci.get("status") == "ok" and ci.get("ci_low") is not None:
        return f"{ci['estimate']:.{d}f} [{ci['ci_low']:.{d}f}, {ci['ci_high']:.{d}f}]"
    return f"{ci['estimate']:.{d}f} (descriptive only)"


def _mean_ci(entry: Optional[Dict[str, Any]], d: int = 4) -> str:
    if not entry or entry["unit"]["mean"] is None:
        return "n/a"
    return _ci(entry["ci"], d) if entry["ci"].get("estimate") is not None else _n(entry["unit"]["mean"], d)


def _agg(block: Optional[Dict[str, Any]], d: int = 3) -> str:
    if not block or not block.get("n_tiles"):
        return "n/a"
    return _ci(block["ci"], d)


def _table(header: List[str], rows: List[List[str]]) -> List[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows] + [""]


def _paired(entry: Optional[Dict[str, Any]], d: int = 4) -> str:
    if not entry or entry.get("mean_difference") is None:
        return "n/a"
    ci = entry.get("ci") or {}
    tail = f" [{ci['ci_low']:.{d}f}, {ci['ci_high']:.{d}f}]" if ci.get("status") == "ok" and ci.get("ci_low") is not None else " (descriptive only)"
    return f"{entry['mean_difference']:+.{d}f}{tail}"


def render_readme(summary: Dict[str, Any], metrics: Dict[str, Any], association: Dict[str, Any], risk_coverage: Dict[str, Any], config: Any) -> str:
    g = summary["code"]["git"]
    primary = str(config.regions.primary_scale_hr_px)
    th = f"{config.decision.ndvi_threshold:g}"
    systems = summary["systems_compared"]
    models = [s for s in systems if s not in ("lr_native", "bicubic")]
    L: List[str] = [
        f"# Downstream analytical utility `{summary['name']}`",
        "",
        f"Status: **{summary['status']}** (run `{summary['run_id']}`). Question: does super-resolution preserve or improve a downstream NDVI-derived analytical quantity and vegetation decision relative to the LR / bicubic baseline, "
        "and is the model's instability (the TTA spread) associated with downstream mistakes once image texture is controlled for? A null or mixed answer is a result. "
        "**No ranking of systems is made**; every difference is a paired value with its uncertainty where the number of scene units allows one.",
        "",
        f"Code: git `{g.get('revision')}` (working tree dirty: {g.get('dirty')}); config digest `{summary['config_digest'][:16]}…`; metrics `{summary['metric_configuration']['version']}`; reference gate `{summary['reference_gate']['version']}`.",
        "",
        "## What was measured",
        "",
        f"* **Task**: NDVI-derived analysis, {summary['ndvi']['formula']}, reflectance fractions (`{summary['ndvi']['input_scaling']}`), bands selected by name from the inputs in order {summary['ndvi']['bands']['band_order_of_inputs']}.",
        f"* **Regions**: {summary['regions']['definition']}. Scales: {summary['regions']['scales_hr_px']} HR px (headline: {primary} HR px = 10 m). A region needs at least {summary['regions']['min_valid_fraction']:.0%} valid pixels.",
        f"* **Decision rule** (fixed): {summary['decision']['rule']}; threshold **{th}** ({summary['decision']['selection_policy'].replace('_', ' ')}: {summary['decision']['threshold_policy_note']}). "
        f"Sensitivity thresholds {summary['decision']['sensitivity_thresholds']} are reported beside it. A region is called vegetated when at least {summary['decision']['region_majority_fraction']:.0%} of its valid pixels are.",
        f"* **Valid pixels**: {summary['ndvi']['valid_pixel_rule']}.",
        f"* **Systems** (all on the same pixels): `lr_native` (the observed 10 m NDVI replicated to its cells), `bicubic`, {', '.join(f'`{m}`' for m in models)} (the ensemble mean of the deployed six-view TTA); reference = HR.",
        f"* **Reference gate** ({summary['reference_gate']['version']}, the Phase 6 gate, unchanged): bicubic-baseline registration within {summary['reference_gate']['alignment']['tolerance_hr_px']} HR px after at most one recorded whole-pixel translation "
        f"(<= {summary['reference_gate']['alignment']['max_correction_hr_px']} px), quadrant spread <= {summary['reference_gate']['alignment']['quadrant_tolerance_hr_px']} px, >= {summary['reference_gate']['eligibility']['min_valid_fraction']:.0%} valid pixels. "
        "Ineligible tiles get no NDVI, no region and no error.",
        f"* **Inference**: over scene units (NEON acquisition / source orthophoto), never over pooled regions; an interval needs >= 5 units, a correlation across units >= 10; fewer is descriptive only. Bootstrap seed {summary['bootstrap']['seed']}, {summary['bootstrap']['n_boot']} resamples.",
        "",
        "### Evidence",
        "",
    ]
    rows = []
    for name, d in summary["datasets"].items():
        r = d["regions"][primary]
        lv = d["gate"]["by_evidence_level"]
        tl = r["excluded_tile_level"]
        rows.append([f"`{name}`", str(d["counts"]["total"]), str(lv.get("pixel_level_eligible", 0)), str(lv.get("uncertain", 0)), str(lv.get("not_eligible", 0)), f"{r['candidate']:,}", f"{r['candidate_in_eligible_tiles']:,}",
                     f"{r['analysed']:,}", f"{tl['regions']:,}", f"{sum(r['excluded_region_level'].values()):,}", f"{len(d['units_eligible'])} / {len(d['scene_units_all'])}"])
    L += ["Regions at the headline scale (10 m):", ""]
    L += _table(["dataset", "records", "eligible tiles", "uncertain", "not eligible", "candidate regions", "in eligible tiles", "analysed", "excluded with their tile", "excluded in eligible tiles", "eligible scene units / all"], rows)
    for name, d in summary["datasets"].items():
        tl = d["regions"][primary]["excluded_tile_level"]["by_reason"]
        rl = d["regions"][primary]["excluded_region_level"]
        L.append(f"`{name}`: tiles excluded (regions counted from the grid geometry alone) by reason {tl or 'none'}; regions excluded inside eligible tiles {rl}; unreadable {d['counts']['unreadable']}, invalid {d['counts']['invalid']}.")
    L += ["", "Each tile's registration (raw displacement, applied translation, residual) is in `tiles.jsonl`; region tables are cached outside the repository (`cache_dir`) with their SHA-256 recorded per tile.", "", "## What was found", ""]

    for dname, m in metrics.items():
        ov = summary["overview"][dname]
        L += [f"### Dataset `{dname}`", ""]
        if ov["status"] != "ok":
            L += ["No eligible evidence: nothing is reported for this dataset.", ""]
            continue
        L += [f"{ov['n_tiles']} eligible tiles from {ov['n_units']} scene units." + (" **Descriptive only (fewer than 5 scene units).**" if ov["descriptive_only"] else ""), ""]
        block = m["scales"][primary][th]
        L += [f"**NDVI fidelity and decision utility at 10 m, threshold {th}** (mean over scene units [95% interval over units]; regions of a unit average first; decision metrics from each unit's pooled pixels):", ""]
        L += _table(["system"] + [lab for _, lab in MAIN_METRICS], [[f"`{s}`"] + [_mean_ci(block["systems"][s][k], 3) for k, _ in MAIN_METRICS] for s in systems])
        L += ["**Change relative to bicubic** (paired over the same scene units; a negative difference in an error is a smaller error; not a ranking):", ""]
        pairs = [p for p in block["paired"] if p.endswith("- bicubic")]
        L += _table(["system − bicubic", "units"] + [lab for _, lab in PAIRED_METRICS], [[f"`{p}`", str(block["paired"][p]["ndvi_mae"]["n"])] + [_paired(block["paired"][p][k], 4) for k, _ in PAIRED_METRICS] for p in pairs])
        extra = [p for p in block["paired"] if not p.endswith("- bicubic")]
        if extra:
            L += ["Against the native LR result, and between models (descriptive):", ""]
            L += _table(["A − B", "units"] + [lab for _, lab in PAIRED_METRICS], [[f"`{p}`", str(block["paired"][p]["ndvi_mae"]["n"])] + [_paired(block["paired"][p][k], 4) for k, _ in PAIRED_METRICS] for p in extra])
        sens = [t for t in m["scales"][primary] if t != th]
        L += ["**Threshold sensitivity** (pixel decision disagreement with the reference, mean over units; the primary threshold was fixed in advance and is not chosen among these):", ""]
        L += _table(["system", f"{th} (primary)"] + [str(t) for t in sens], [[f"`{s}`", _mean_ci(m["scales"][primary][th]["systems"][s]["disagreement_rate"], 3)] + [_mean_ci(m["scales"][primary][t]["systems"][s]["disagreement_rate"], 3) for t in sens]
                                                                          for s in systems])
        other = [s for s in m["scales"] if s != primary]
        for sc in other:
            b2 = m["scales"][sc][th]
            L += [f"**At the {int(sc) * 2.5:g} m scale ({sc} HR px)** (same metrics, threshold {th}):", ""]
            L += _table(["system", "NDVI MAE", "vegetation-fraction MAE", "region decision error", "pixel decision disagreement"], [[f"`{s}`"] + [_mean_ci(b2["systems"][s][k], 3) for k in ("ndvi_mae", "veg_fraction_mae", "region_decision_error_rate", "disagreement_rate")] for s in systems])
        L += ["**Is instability associated with downstream error? (within tiles, then over scene units)**", "",
              "Spearman correlation of a region's stability with its downstream error, mean over units [interval over units]; `texture` and `added detail` are trivial predictors correlated with the same error; "
              "`partial` = the stability controlling for texture (and, in the last column, for texture and added detail):", ""]
        rows = []
        for sc in m["scales"]:
            for model in models:
                a = association[dname]["scales"][sc][model]
                for tgt, lab in TARGET_LABEL.items():
                    e = a["targets"][tgt]
                    rows.append([f"{int(sc) * 2.5:g} m", f"`{model}`", lab, _agg(e["stability"]), _agg(e["texture"]), _agg(e["added_detail"]), _agg(e["partial_given_texture"]), _agg(e["partial_given_texture_and_added_detail"]),
                                 str(e["stability"]["n_units"])])
        L += _table(["region scale", "model", "downstream error", "stability", "texture", "added detail", "partial (given texture)", "partial (given texture + added detail)", "units"], rows)
        L += ["Pooled regions (a seeded subsample of each tile; the interval resamples whole scene units) and across tiles (tile mean stability against tile mean error; an interval needs >= 10 units):", ""]
        rows = []
        for model in models:
            a = association[dname]["scales"][primary][model]
            for tgt, lab in TARGET_LABEL.items():
                e = a["targets"][tgt]
                t = e["across_tiles"]["tile"]
                sp = t["spearman"]
                across = "n/a" if sp.get("value") is None else (f"{sp['value']:.3f} [{t['spearman_ci']['ci_low']:.3f}, {t['spearman_ci']['ci_high']:.3f}]" if t["spearman_ci"].get("status") == "ok" else f"{sp['value']:.3f} (descriptive only)")
                rows.append([f"`{model}`", lab, _ci(e["pooled_stability"]), _ci(e["pooled_texture"]), _ci(e["pooled_partial_given_texture"]), across])
        L += _table(["model (10 m)", "downstream error", "pooled: stability", "pooled: texture", "pooled: partial", "across tiles: stability"], rows)
        L += ["**Risk-coverage on the downstream error** (within each scene unit, drop the most unstable fraction of regions, report the relative reduction of the remaining error; texture ranked the same way; the oracle is ranked by the true error; "
              "**not calibrated selective prediction**), mean over units [interval]:", ""]
        rows = []
        for model in models:
            for tgt, lab in TARGET_LABEL.items():
                for entry in risk_coverage[dname]["scales"][primary][model]["targets"][tgt]["by_retention"]:
                    if entry["retention"] == 1.0:
                        continue
                    rows.append([f"`{model}`", lab, f"{entry['retention']:.0%} kept", _mean_ci(entry["stability"]["reduction"], 3), _mean_ci(entry["texture"]["reduction"], 3), _paired(entry.get("stability_minus_texture"), 3), _mean_ci(entry["oracle"]["reduction"], 3)])
        L += _table(["model (10 m)", "downstream error", "retained", "reduction, stability order", "reduction, texture order", "stability − texture (paired over scene units)", "reduction, oracle"], rows)

    L += ["## What was not shown", "",
          "* **Calibrated uncertainty.** Phase 6 found the TTA spread uncalibrated (16-32x smaller than the error); nothing here changes that. Any association above is an ordering, not a probability, confidence or interval.",
          "* **Causal relationships.** Texture, brightness and the amount of detail added are confounded with both instability and error; a partial correlation reduces that, it does not remove it.",
          f"* **Indian generalisation.** `{summary['indian_data']['status']}`: {summary['indian_data']['reason']}.",
          f"* **Land-cover performance.** `{summary['secondary_tasks']['landcover']}`: {summary['secondary_tasks']['reason']}. The thresholded NDVI decision is a vegetation proxy, not land cover, and its agreement is not a classification accuracy against labels.",
          "* **A downstream advantage of super-resolution, or of any system over another.** Values and paired differences are reported with their uncertainty; where an interval includes zero, or there are too few scene units for one, no advantage is claimed. No ranking of systems is made.",
          "* **Behaviour on misregistered evidence.** Tiles that failed the reference gate were excluded and are not used for any conclusion.", "",
          "## Limitations", "",
          "* **Few eligible scene units.** The registration gate leaves few units per dataset (see the evidence table); several tables are descriptive only, and dev/test protocols were not run.",
          "* **Registration residual.** Eligible tiles carry a residual misregistration of up to the gate tolerance, and OpenSR-Test tiles needed a recorded whole-pixel translation that assumes one global displacement; the reference is a different sensor with its own radiometry.",
          "* **Geographic scope.** North America (SEN2NEON) and Spain (OpenSR-Test); no Indian reference exists here.",
          "* **Texture, brightness and detail confounding.** The trivial baselines and the partial correlations are controls, not a proof of independence.",
          "* **A threshold is a choice.** 0.3 was fixed in advance from convention; the sensitivity thresholds show how much the decision depends on it. Decision metrics inherit the reference's own classification noise (it is not ground truth).",
          "* **One index, one decision rule, one perturbation set, two models.** No other index, task or ensemble was tested.", "",
          "Files: `config.json`, `tiles.jsonl` (one row per tile: registration, region counts, exclusion reason), `summary.json` (provenance, evidence, cost), `downstream_metrics.json` (per unit and across units, paired differences), `association.json`, `risk_coverage.json`.", ""]
    return "\n".join(L)
