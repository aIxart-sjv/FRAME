"""A small, reproducible statistical layer (Phase 5). numpy + scipy only.

Requirements grounding (docs/Requirements 142.txt Requirement 9, sections 28-32): images and tiles are spatially correlated, so the unit of
analysis is the scene/tile, never the pixel; report mean +- standard deviation; for comparisons report a paired bootstrap confidence interval of the
DIFFERENCE, a Wilcoxon signed-rank test on paired scene-level results, and an effect size, not only a p-value.

Honesty rules built in (tests enforce them):

* Below ``MIN_UNITS_CI`` (5) units no interval is computed; below ``MIN_UNITS_TEST`` (6) paired units, and whenever fewer than that many differences are
  non-zero, no test is run: with n pairs the smallest attainable exact two-sided Wilcoxon p is 2 / 2^n, which is 0.0625 for n = 5, so a test below 6 pairs
  cannot ever reach 0.05. Such results are labelled ``descriptive_only`` and carry the reason.
* Missing values are dropped AND counted (``n_missing`` / ``n_dropped``), never silently.
* No multiple-comparison correction is applied and none is implied: a table of many comparisons is exploratory.
* Every random draw uses an explicit seed, so a rerun gives identical intervals.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as scipy_stats

MIN_UNITS_CI = 5
MIN_UNITS_TEST = 6
DEFAULT_N_BOOT = 10_000
DEFAULT_ALPHA = 0.05


def _finite(values: Sequence[Optional[float]]) -> Tuple[np.ndarray, int]:
    kept = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return np.asarray(kept, dtype=np.float64), len(values) - len(kept)


def describe(values: Sequence[Optional[float]]) -> Dict[str, Any]:
    """n, mean, median, sample standard deviation (ddof = 1), min, max, and how many values were missing."""
    x, missing = _finite(values)
    if x.size == 0:
        return {"n": 0, "n_missing": missing, "mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "n": int(x.size), "n_missing": missing, "mean": float(x.mean()), "median": float(np.median(x)),
        "std": float(x.std(ddof=1)) if x.size > 1 else None, "min": float(x.min()), "max": float(x.max()),
    }


def group_means(values: Sequence[Optional[float]], units: Sequence[str]) -> Tuple[List[str], List[float], List[int]]:
    """Average ``values`` within each unit (scene/acquisition): correlated tiles of one unit count once. Returns (unit ids, unit means, valid values per unit)."""
    if len(values) != len(units):
        raise ValueError("values and units must have the same length.")
    buckets: Dict[str, List[float]] = {}
    for value, unit in zip(values, units):
        if value is None or not math.isfinite(float(value)):
            continue
        buckets.setdefault(str(unit), []).append(float(value))
    ids = sorted(buckets)
    return ids, [float(np.mean(buckets[u])) for u in ids], [len(buckets[u]) for u in ids]


def _percentile_interval(samples: np.ndarray, alpha: float) -> Tuple[float, float]:
    return float(np.percentile(samples, 100 * alpha / 2)), float(np.percentile(samples, 100 * (1 - alpha / 2)))


def bootstrap_ci(
    values: Sequence[Optional[float]],
    *,
    statistic: str = "mean",
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
    min_units: int = MIN_UNITS_CI,
) -> Dict[str, Any]:
    """Percentile bootstrap confidence interval over the units in ``values`` (resampling units with replacement)."""
    if statistic not in ("mean", "median"):
        raise ValueError(f"statistic must be 'mean' or 'median', got {statistic!r}.")
    x, missing = _finite(values)
    reduce = np.mean if statistic == "mean" else np.median
    out: Dict[str, Any] = {"n": int(x.size), "n_missing": missing, "statistic": statistic, "level": 1 - alpha, "method": "percentile bootstrap over units",
                           "n_boot": n_boot, "seed": seed, "estimate": float(reduce(x)) if x.size else None, "ci_low": None, "ci_high": None}
    if x.size < min_units:
        out.update(status="descriptive_only", reason=f"n = {x.size} < {min_units} units: too few for an interval; reported as descriptive only.")
        return out
    rng = np.random.default_rng(seed)
    draws = reduce(x[rng.integers(0, x.size, size=(n_boot, x.size))], axis=1)
    out["ci_low"], out["ci_high"] = _percentile_interval(draws, alpha)
    out.update(status="ok", reason=None)
    return out


def paired_comparison(
    a: Sequence[Optional[float]],
    b: Sequence[Optional[float]],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
    min_units_ci: int = MIN_UNITS_CI,
    min_units_test: int = MIN_UNITS_TEST,
) -> Dict[str, Any]:
    """Paired comparison of ``a`` against ``b`` over the same units: difference (a - b), its bootstrap CI, a Wilcoxon signed-rank test, an effect size."""
    if len(a) != len(b):
        raise ValueError("a and b must have the same length (one value per unit, in the same order).")
    pairs = [(float(x), float(y)) for x, y in zip(a, b) if x is not None and y is not None and math.isfinite(float(x)) and math.isfinite(float(y))]
    dropped = len(a) - len(pairs)
    xa = np.asarray([p[0] for p in pairs]); xb = np.asarray([p[1] for p in pairs])
    diff = xa - xb
    n = int(diff.size)
    result: Dict[str, Any] = {
        "n": n, "n_dropped": dropped, "mean_difference": float(diff.mean()) if n else None, "median_difference": float(np.median(diff)) if n else None,
        "std_difference": float(diff.std(ddof=1)) if n > 1 else None,
        "effect_size_dz": None, "fraction_a_greater": float(np.mean(diff > 0)) if n else None,
        "ci": bootstrap_ci(list(diff), n_boot=n_boot, alpha=alpha, seed=seed, min_units=min_units_ci),
    }
    if n > 1 and result["std_difference"] and result["std_difference"] > 0:
        result["effect_size_dz"] = float(diff.mean() / result["std_difference"])

    nonzero = int(np.count_nonzero(diff))
    wilcoxon: Dict[str, Any] = {"method": "wilcoxon signed-rank, two-sided", "statistic": None, "p_value": None, "n_nonzero": nonzero}
    if n and nonzero == 0:
        wilcoxon.update(status="no_nonzero_differences", reason="every paired difference is exactly zero; there is nothing to test.")
    elif n < min_units_test or nonzero < min_units_test:
        wilcoxon.update(status="descriptive_only", reason=(
            f"{n} pairs ({nonzero} non-zero) < {min_units_test}: with fewer than {min_units_test} pairs the smallest attainable exact two-sided p "
            "exceeds 0.05, so no test is run; the difference is descriptive only."))
    else:
        test = scipy_stats.wilcoxon(xa, xb, alternative="two-sided")
        wilcoxon.update(status="tested", reason=None, statistic=float(test.statistic), p_value=float(test.pvalue))
    result["wilcoxon"] = wilcoxon
    result["label"] = "inferential" if wilcoxon["status"] == "tested" else "descriptive_only"
    return result
