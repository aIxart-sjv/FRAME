"""README of a reliability run (Phase 6): tables only of what was measured, with the evidence that was excluded shown beside the evidence that was used."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _n(x: Optional[float], d: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{d}f}"


def _ci(ci: Optional[Dict[str, Any]], d: int = 3) -> str:
    """estimate and interval, or the estimate marked descriptive-only, or n/a."""
    if not ci or ci.get("estimate") is None:
        return "n/a"
    if ci.get("status") == "ok" and ci.get("ci_low") is not None:
        return f"{ci['estimate']:.{d}f} [{ci['ci_low']:.{d}f}, {ci['ci_high']:.{d}f}]"
    return f"{ci['estimate']:.{d}f} (descriptive only)"


def _agg(block: Optional[Dict[str, Any]], d: int = 3) -> str:
    if not block:
        return "n/a"
    ci = dict(block["ci"]) if block.get("ci") else None
    text = _ci(ci, d)
    return text if block.get("n_tiles") else "n/a"


def _corr(block: Optional[Dict[str, Any]], d: int = 3) -> str:
    if not block:
        return "n/a"
    v = block.get("value")
    return "n/a" if v is None else f"{v:.{d}f}"


def _assoc(block: Optional[Dict[str, Any]], key: str = "spearman", ci_key: str = "spearman_ci", d: int = 3) -> str:
    if not block:
        return "n/a"
    v = (block.get(key) or {}).get("value")
    if v is None:
        return "n/a"
    ci = block.get(ci_key) or {}
    if ci.get("status") == "ok" and ci.get("ci_low") is not None:
        return f"{v:.{d}f} [{ci['ci_low']:.{d}f}, {ci['ci_high']:.{d}f}]"
    return f"{v:.{d}f} (descriptive only)"


def _table(header: List[str], rows: List[List[str]]) -> List[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows] + [""]


def render_readme(summary: Dict[str, Any], analyses: Dict[str, Any], comparisons: Dict[str, Any], config: Any) -> str:
    g = summary["code"]["git"]
    al, el = summary["alignment"], summary["eligibility"]
    L: List[str] = [
        f"# Reliability validation `{summary['name']}`",
        "",
        f"Status: **{summary['status']}**. Question: is the TTA model-stability signal informative about reconstruction error, judged **only on evidence whose reference is registered to the prediction grid**? "
        "The stability is a **relative model-stability proxy, not a calibrated uncertainty, a confidence or a probability of error**: whether it is informative is what is tested, and this README "
        "reports it faithfully whatever the answer. Every association is descriptive and non-causal; nothing is pooled across datasets; nothing is ranked.",
        "",
        f"Code: git `{g.get('revision')}` (working tree dirty: {g.get('dirty')}); metrics `{summary['metric_configuration']['version']}`; tiling {summary['tiling']}.",
        "",
        "## What is measured",
        "",
        f"* **Stability** (the deployed quantity): {summary['tta']['definition']}. Members: {', '.join(summary['tta']['transforms'])} ({summary['tta']['n_members']}); seed {summary['tta']['seed']}.",
        "* **Error targets** (against the independent HR reference, strict valid mask only): per-pixel band-mean absolute reflectance error (E1); per-pixel spectral angle (E2); at cell level "
        "(10 m = 4 HR px, 40 m = 16 HR px) their means; per tile RMSE, MAE, SAM, ERGAS; per scene unit their means. The evaluated product is the TTA ensemble mean; the single pass (identity view) is "
        "recorded separately in `metrics.jsonl`.",
        "* **Trivial predictors of error, reported next to the stability**: image texture (Sobel gradient of the bicubic input) and the amount the model moved away from bicubic. A stability that only "
        "re-encodes them adds nothing, so a partial correlation controlling for both is reported.",
        "* **Units of analysis**: within-tile correlations are summarised over scene units (NEON acquisition / source orthophoto); intervals resample whole units; fewer than 5 units is descriptive only.",
        "",
        "## The reference eligibility gate",
        "",
        f"The displacement of the bicubic baseline against the reference is estimated (`{al['method']}`, system-neutral, before any model runs). A tile is pixel-level **eligible** only if, after at most one "
        f"whole-pixel translation of {al['max_correction_hr_px']} HR px (an integer crop of both grids, no resampling, recorded per tile), the residual is within **{al['tolerance_hr_px']} HR px** and the four quadrants agree "
        f"within {al['quadrant_tolerance_hr_px']} HR px, with at least {el['min_valid_fraction']:.0%} strict-valid pixels. Between one and {al['uncertain_factor']:g} tolerances the evidence is `uncertain`; beyond it `not_eligible`. "
        "Excluded tiles receive **no** error number and are listed below with their reason (`excluded_from_uncertainty_error_analysis`).",
        "",
        "## Evidence",
        "",
    ]
    rows = []
    for name, d in summary["datasets"].items():
        gate = d["gate"]
        lv = gate["by_evidence_level"]
        rows.append([f"`{name}`", d["evidence_class"], str(d["counts"]["total"]), str(d["counts"]["invalid"]), str(d["counts"]["unreadable"]), str(lv.get("pixel_level_eligible", 0)),
                     str(lv.get("uncertain", 0)), str(lv.get("not_eligible", 0)), ", ".join(f"{k}: {v}" for k, v in gate["excluded_by_reason"].items()) or "none",
                     str(len(d["scene_units_eligible"])), str(len(d["scene_units_all"]))])
    L += _table(["dataset", "class", "records", "invalid", "unreadable", "eligible tiles", "uncertain", "not eligible", "exclusion reasons", "eligible scene units", "all scene units"], rows)
    for name, d in summary["datasets"].items():
        for s, info in d["per_system"].items():
            if info["excluded_after_gate"]:
                L.append(f"`{name}` / `{s}`: tiles that passed the gate but were excluded after inference: {info['excluded_after_gate']}")
    L.append("")
    L += ["Registration of every tile (raw displacement of the bicubic baseline, the correction applied, the residual) is in `summary.json` (`datasets.<name>.tiles`) and in each row of `metrics.jsonl`.", ""]

    for dname, per in analyses.items():
        L += [f"## Dataset `{dname}`", ""]
        for sname, a in per.items():
            L += [f"### `{sname}`", ""]
            if a["status"] != "ok":
                L += [f"No eligible evidence: **{a['status']}**. Nothing is reported for this system on this dataset.", ""]
                continue
            L += [f"{a['n_tiles']} eligible tiles from {a['n_units']} scene units." + (" **Descriptive only (fewer than 5 scene units).**" if a["descriptive_only"] else ""), ""]
            L += _pixel_cell_tables(a)
            L += _tile_tables(a)
            L += _risk_tables(a)
            L += _detection_table(a)
            L += _calibration_block(a)
            L += _detail_block(a)
            L += _cost_block(a)
        cmp = comparisons.get(dname)
        if cmp:
            L += _comparison_block(cmp)
    L += ["## How to read this", "",
          "* Correlations are between the **spread of the TTA ensemble** and **reconstruction error against the reference**. A positive value means more unstable places tend to have larger error; it does not mean instability causes error.",
          "* Compare every stability figure with the **texture** and **added-detail** baselines: they need no ensemble. The partial correlation is what remains after controlling for both.",
          "* `descriptive only` means too few scene units for an interval (or no held-out units for a threshold or a calibration). It is never turned into a claim.",
          "* Risk-coverage describes an ordering. It is **not** calibrated selective prediction; the oracle and random removal bracket it.",
          "* The raw spread is orders of magnitude smaller than the error (ensemble members share the model's bias): it is reported as **uncalibrated stability evidence**. A recalibration is judged only on held-out units.",
          "* Registration limits everything: tiles whose reference is not registered were excluded, and the residual misregistration of the accepted tiles (<= the tolerance) still adds noise.",
          "", "Files: `config.json`, `metrics.jsonl` (one row per tile and system; exclusions carry a reason and no numbers), `summary.json` (provenance, evidence, cost), `correlations.json`, `risk_coverage.json`, `detection.json`, `calibration.json`.", ""]
    return "\n".join(L)


def _pixel_cell_tables(a: Dict[str, Any]) -> List[str]:
    L = ["**Within-tile Spearman correlation between stability and error**, mean over scene units [95% interval over units]; the baselines are correlated with the same error; `partial` controls "
         "the stability for both baselines; `share > 0` is the share of tiles with a positive correlation.", ""]
    rows = []
    levels = [("pixel (2.5 m)", a["pixel_level"])] + [(f"cell {int(s) * 2.5:g} m", c) for s, c in a["cell_level"].items()]
    for label, blk in levels:
        p = blk["spearman_abs_error"]
        rows.append([label, _agg(p), _agg(blk["spearman_sam"]), _agg(blk["spearman_texture_abs_error"]), _agg(blk["spearman_added_detail_abs_error"]),
                     _agg(blk["partial_spearman_abs_error_given_baselines"]), _n(p.get("share_positive"), 2), str(p["n_tiles"]), str(p["n_units"])])
    L += _table(["scale", "stability vs abs error", "stability vs SAM", "texture vs abs error", "added detail vs abs error", "partial (stability given both)", "share > 0", "tiles", "units"], rows)
    pooled = []
    for s, c in a["cell_level"].items():
        if "pooled_spearman_abs_error" in c:
            b = c["pooled_baselines"]
            pooled.append([f"cell {int(s) * 2.5:g} m", _ci(c["pooled_spearman_abs_error"]), _ci(c["pooled_spearman_sam"]), _ci(b["texture_vs_abs_error"]), _ci(b["added_detail_vs_abs_error"]),
                           _ci(b["partial_stability_vs_abs_error_given_baselines"])])
    if pooled:
        L += ["**Pooled cell correlation** (a seeded subsample of each tile's cells; the interval resamples whole scene units):", ""]
        L += _table(["scale", "stability vs abs error", "stability vs SAM", "texture vs abs error", "added detail vs abs error", "partial"], pooled)
    sweep = a["pixel_level"].get("by_reference_displacement", {})
    if len(sweep) > 1:
        L += ["**Sensitivity to registration** (why misregistered evidence is excluded): the pixel-level correlation of the same eligible tiles with the reference displaced by known whole pixels:", ""]
        L += _table(["reference displaced by (HR px)"] + list(sweep), [["mean over units"] + [_agg(v) for v in sweep.values()]])
    return L


def _tile_tables(a: Dict[str, Any]) -> List[str]:
    t, s = a["tile_level"], a["scene_level"]
    L = ["**Tile and scene reliability**: Spearman correlation across tiles (interval resamples scene units) and across scene units (tiles of a unit averaged first).", ""]
    rows = []
    for tgt in ("rmse", "mae", "sam_degrees", "ergas"):
        st, sc = t["stability_vs"][tgt], s["stability_vs"][tgt]
        rows.append([tgt, _assoc(st), str(st["n_tiles"]), _assoc(sc), str(sc["n_units"]), _assoc(t["texture_vs"][tgt]), _assoc(t["added_detail_vs"][tgt]),
                     _assoc(t["partial_stability_given_baselines"][tgt], "partial_spearman", "partial_ci")])
    L += _table(["error target", "tile: stability", "tiles", "scene: stability", "units", "tile: texture", "tile: added detail", "tile: partial (stability given both)"], rows)
    return L


def _risk_tables(a: Dict[str, Any]) -> List[str]:
    L = ["**Risk-coverage** (most unstable removed first; risk = error of what remains; *not* calibrated selective prediction):", ""]
    for key, rc in a["risk_coverage"].items():
        if rc.get("status") != "ok":
            L.append(f"`{key}`: {rc.get('status')}")
            continue
        rows = []
        for i, c in enumerate(rc["curve"]):
            rows.append([f"{c['coverage']:.0%}", _n(c["risk"], 5), _n(rc["oracle_curve"][i]["risk"], 5), _n(rc["texture_baseline"]["curve"][i]["risk"], 5), _n(rc["added_detail_baseline"]["curve"][i]["risk"], 5)])
        L += [f"`{key}` ({rc['n']} items, {rc['n_units']} scene units). Selective efficiency (1 = oracle, 0 = random, < 0 worse than random): stability {_ci(rc['ci']['selective_efficiency'])}, "
              f"texture {_n(rc['texture_baseline']['selective_efficiency'])}, added detail {_n(rc['added_detail_baseline']['selective_efficiency'])}.", ""]
        L += _table(["coverage", "risk (stability order)", "oracle", "texture order", "added-detail order"], rows)
        red = rc["ci"]["risk_reduction_at"]
        L += ["Relative risk reduction vs no removal: " + "; ".join(f"at {float(c):.0%} coverage {_ci(v)}" for c, v in red.items()), ""]
    return L


def _detection_table(a: Dict[str, Any]) -> List[str]:
    d = a["high_error_detection"]
    L = ["**High-error detection** (10 m cells):", ""]
    if d.get("status") not in ("ok", "descriptive_only"):
        return L + [f"not computed: {d.get('status')} ({d.get('reason')})", ""]
    L += [f"Threshold {_n(d['threshold'], 5)} = the {d['quantile']:.0%} quantile of error over **{d['threshold_source']}**; prevalence of high error in the evaluated cells {_n(d['prevalence_test'], 3)} "
          f"({d['n_test_cells']} cells; test units: {len(d['test_units'])}). " + (f"**Descriptive only**: {d['reason']}." if d["status"] != "ok" else ""), ""]
    rows = []
    for tag, label in (("stability", "stability (ensemble spread)"), ("texture_baseline", "texture baseline"), ("added_detail_baseline", "added-detail baseline")):
        b = d[tag]
        rows.append([label, _ci(b["auroc_ci"]) if b["auroc_ci"].get("estimate") is not None else _n(b["auroc"]["value"]), _ci(b["auprc_ci"]) if b["auprc_ci"].get("estimate") is not None else _n(b["auprc"]["value"]),
                     _n(b["flag"]["0.1"]["precision"]), _n(b["flag"]["0.1"]["lift"], 2), _n(b["flag"]["0.2"]["precision"]), _n(b["flag"]["0.2"]["lift"], 2)])
    L += _table(["score", "AUROC", "AUPRC (chance = prevalence)", "precision, top 10% flagged", "lift", "precision, top 20%", "lift"], rows)
    return L


def _calibration_block(a: Dict[str, Any]) -> List[str]:
    c = a["calibration"]
    raw, rec = c["raw"], c["recalibration"]
    L = [f"**Calibration**: verdict `{c['verdict']}`. {c['statement']}", ""]
    cov = raw["coverage"]
    L += [f"* Scale: the median error is {_n(raw['scale']['median_error_over_median_spread'], 1)} times the median spread (median over tiles).",
          "* Coverage of `prediction +/- k * spread` against a Gaussian's nominal level: " + "; ".join(f"k = {k[1:]}: {_n(v['coverage'], 3)} (nominal {_n(v['nominal'], 3)})" for k, v in cov.items()) + "."]
    if rec["status"] == "ok":
        line = rec["line"]
        ci = rec["line_ci"]
        L += [f"* Recalibration (isotonic map stability -> expected 10 m error, fitted on {len(rec['dev_units'])} development units, judged on {len(rec['test_units'])} test units): calibration line "
              f"observed = intercept + slope x predicted: slope {_ci(ci['slope'])}, intercept {_ci(ci['intercept'], 5)} (ideal 1 and 0); expected calibration error {_n(rec['reliability']['ece'], 5)}; "
              f"skill over the development-mean constant {_n(rec['skill_vs_constant'], 3)}; verdict **{rec['verdict']}**."]
    else:
        L += [f"* Recalibration: {rec['status']} ({rec['reason']})."]
    return L + [""]


def _detail_block(a: Dict[str, Any]) -> List[str]:
    d = a["detail_relationship"]
    L = ["**Relationship with the Phase 5 detail categories** (Spearman of stability with each category; not semantic labels; instability and added detail both grow with texture): cell-level, mean over units [interval]:", ""]
    rows = []
    for s, blk in d["cell_level"].items():
        rows.append([f"cell {int(s) * 2.5:g} m", _agg(blk["unsupported"]), _agg(blk["omission"]), _agg(blk["supported"])])
    L += _table(["scale", "unsupported detail", "omission", "supported synthesis"], rows)
    rows = [[t, _assoc(v)] for t, v in d["tile_level"].items()]
    L += ["Tile-level (tile mean stability vs tile share of each category):", ""] + _table(["category", "Spearman across tiles"], rows)
    return L


def _cost_block(a: Dict[str, Any]) -> List[str]:
    c = a["tta_cost"]
    return [f"**TTA cost**: {c['n_members']} members; single pass {_n(c['single_pass_seconds_median'], 3)} s, TTA {_n(c['tta_seconds_median'], 3)} s "
            f"(x{_n(c['tta_over_single_pass'], 2)}) per scene, median over {c['n_tiles_timed']} tiles (the first tile, {_n(c['first_tile_tta_seconds'], 2)} s, carries warm-up and is excluded).", ""]


def _comparison_block(cmp: Dict[str, Any]) -> List[str]:
    L = ["### Comparison of the systems on the same eligible tiles", "", cmp["note"], "", f"{cmp['n_shared_tiles']} shared tiles, {cmp['n_shared_units']} scene units; tiles eligible for one system only: {cmp['only_in']}.", ""]
    for p in cmp["pairs"]:
        rows = []
        for path, m in p["metrics"].items():
            ci = m["ci"]
            interval = f"[{ci['ci_low']:.4g}, {ci['ci_high']:.4g}]" if ci.get("status") == "ok" and ci.get("ci_low") is not None else "(descriptive only)"
            rows.append([f"`{path}`", str(m["n"]), _n(m["mean_difference"], 4), interval])
        L += [f"**{p['a']} - {p['b']}** (paired over scene units):", ""] + _table(["metric", "units", "mean difference", "95% interval"], rows)
    return L
