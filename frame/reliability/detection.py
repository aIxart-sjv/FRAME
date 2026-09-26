"""High-error detection metrics (Phase 6): does a higher instability flag the cases whose error is high? numpy + scipy only.

``score`` is the instability (higher = less stable) and ``label`` is 1 for a HIGH-ERROR case, whose definition (a threshold chosen on DEVELOPMENT evidence only) is the
caller's responsibility. Ties are handled exactly (AUROC gives tied pairs half credit; AUPRC treats a tied group as one threshold; flagging breaks ties with a seeded random
permutation, never by input order). Non-finite scores are dropped and counted. A single class has no AUROC / AUPRC: that is reported, not turned into 0.5.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np
from scipy import stats as scipy_stats


def _prepare(score: Sequence[float], label: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, int]:
    s, y = np.asarray(score, dtype=np.float64).ravel(), np.asarray(label).ravel()
    if s.shape != y.shape:
        raise ValueError(f"score and label must have the same length, got {s.size} and {y.size}")
    if y.dtype != bool and not np.isin(y, (0, 1)).all():
        raise ValueError("label must be binary (0/1 or bool)")
    keep = np.isfinite(s)
    return s[keep], y[keep].astype(bool), int(s.size - keep.sum())


def _undefined(n: int, dropped: int, n_pos: int) -> Dict[str, Any]:
    return {"value": None, "n": n, "n_dropped": dropped, "n_pos": n_pos, "n_neg": n - n_pos, "status": "not_computable", "reason": "a single class: AUROC/AUPRC need both high-error and other cases"}


def auroc(score: Sequence[float], label: Sequence[int]) -> Dict[str, Any]:
    """Area under the ROC curve = P(score of a random high-error case > score of a random other case), ties counting one half."""
    s, y, dropped = _prepare(score, label)
    n, n_pos = int(s.size), int(y.sum())
    if n_pos == 0 or n_pos == n:
        return _undefined(n, dropped, n_pos)
    ranks = scipy_stats.rankdata(s)
    value = (float(ranks[y].sum()) - n_pos * (n_pos + 1) / 2.0) / (n_pos * (n - n_pos))
    return {"value": float(value), "n": n, "n_dropped": dropped, "n_pos": n_pos, "n_neg": n - n_pos, "status": "ok", "reason": None}


def auprc(score: Sequence[float], label: Sequence[int]) -> Dict[str, Any]:
    """Average precision (sum over distinct thresholds of the recall step times the precision there, no interpolation). Its chance level is the prevalence."""
    s, y, dropped = _prepare(score, label)
    n, n_pos = int(s.size), int(y.sum())
    if n_pos == 0 or n_pos == n:
        return {**_undefined(n, dropped, n_pos), "prevalence": (n_pos / n) if n else None}
    order = np.argsort(-s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    tp, fp = np.cumsum(y_sorted), np.cumsum(~y_sorted)
    ends = np.r_[np.flatnonzero(np.diff(s_sorted) != 0), n - 1]                # last index of every tied group: one threshold each
    precision, recall = tp[ends] / (tp[ends] + fp[ends]), tp[ends] / n_pos
    value = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    return {"value": value, "n": n, "n_dropped": dropped, "n_pos": n_pos, "n_neg": n - n_pos, "prevalence": n_pos / n, "status": "ok", "reason": None}


def flag_metrics(score: Sequence[float], label: Sequence[int], *, fraction: float, seed: int) -> Dict[str, Any]:
    """Flag the ``fraction`` of items with the highest instability and report precision, recall and lift (precision / prevalence) of that flag."""
    s, y, dropped = _prepare(score, label)
    n, n_pos = int(s.size), int(y.sum())
    if n == 0 or n_pos == 0:
        return {"fraction": float(fraction), "n": n, "n_dropped": dropped, "status": "not_computable", "reason": "no high-error cases", "precision": None, "recall": None, "lift": None}
    k = max(1, min(n, int(round(fraction * n))))
    perm = np.random.default_rng(seed).permutation(n)
    flagged = np.lexsort((perm, -s))[:k]
    tp = int(y[flagged].sum())
    precision, prevalence = tp / k, n_pos / n
    return {"fraction": float(fraction), "n": n, "n_dropped": dropped, "n_flagged": k, "n_pos": n_pos, "true_positives": tp, "precision": precision, "recall": tp / n_pos,
            "prevalence": prevalence, "lift": precision / prevalence, "status": "ok", "reason": None}
