"""Reflectance normalization for the FRAME preprocessing layer.

Reproduces exactly the scaling used by the upstream sen2sr README examples
and by ``experiments/baseline/run_baseline.py``:

    X = (digital_number / 10_000).astype("float32")
    X = nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

Whether an input still needs the ``/10000`` scaling is never guessed from
the data itself (e.g. via a magnitude heuristic) -- the caller must state
it explicitly via ``input_scale``, so a caller can never silently apply the
scaling twice (or not at all) by accident.
"""

from __future__ import annotations

import numpy as np

from frame.preprocessing.errors import InvalidInputScaleError

RAW_DIGITAL_NUMBER = "raw_digital_number"
REFLECTANCE = "reflectance"
_VALID_INPUT_SCALES = (RAW_DIGITAL_NUMBER, REFLECTANCE)

SENTINEL2_L2A_REFLECTANCE_SCALE = 10_000.0

#: Bounds of reflectance as a fraction, from the Sentinel-2 L2A encoding (not tuned): the largest a uint16
#: DN can give is 65535 / 10000, the smallest a BOA_ADD_OFFSET-corrected DN can give is (0 - 1000) / 10000.
#: Identical to `frame.models.config` (a test compares them; this package does not import that one).
MIN_REFLECTANCE = -0.1
MAX_REFLECTANCE = 65535.0 / 10000.0

#: A scene declared as raw digital numbers whose largest valid value is at or below this is reflectance in disguise:
#: L2A DNs are integers in the hundreds to thousands, and even a black scene stays far above 1.5.
IMPLAUSIBLE_RAW_DN_MAX = 1.5


def check_scale_consistency(array: np.ndarray, valid: np.ndarray, input_scale: str) -> None:
    """Refuse a declaration the data contradicts. It never converts anything and never guesses the scale:
    it only rejects the two impossible combinations, which would otherwise pass silently
    (reflectance divided by 10000 is a black image; DNs read as reflectance are 10000 times too bright).

    Only valid, finite values are looked at (``valid`` is the (H, W) validity mask, ``array`` the (bands, H, W) stack).
    """
    if input_scale not in _VALID_INPUT_SCALES:
        raise InvalidInputScaleError(f"Unknown input_scale {input_scale!r}; expected one of {_VALID_INPUT_SCALES}.")
    values = np.asarray(array)[:, np.asarray(valid, dtype=bool)]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    lo, hi = float(values.min()), float(values.max())
    if input_scale == RAW_DIGITAL_NUMBER and hi <= IMPLAUSIBLE_RAW_DN_MAX:
        raise InvalidInputScaleError(
            f"input_scale is 'raw_digital_number' but the largest valid value is {hi:.4g}: Sentinel-2 L2A digital numbers are integers "
            "in the hundreds to thousands, so these values look like reflectance fractions. Dividing them by 10000 would produce a black image; "
            "use input_scale='reflectance'."
        )
    if input_scale == REFLECTANCE and (hi > MAX_REFLECTANCE * (1 + 1e-6) or lo < MIN_REFLECTANCE):
        raise InvalidInputScaleError(
            f"input_scale is 'reflectance' but the valid values span [{lo:.4g}, {hi:.4g}], outside the L2A reflectance range "
            f"[{MIN_REFLECTANCE}, {MAX_REFLECTANCE:.4f}]: these look like raw digital numbers; use input_scale='raw_digital_number'."
        )


def to_reflectance(array: np.ndarray, input_scale: str) -> np.ndarray:
    """Convert ``array`` to float32 surface-reflectance-like values in [0, 1].

    Args:
        array: Input band stack, any numeric dtype.
        input_scale: Either ``"raw_digital_number"`` (Sentinel-2 L2A integer
            digital numbers, scaled by 10,000 as documented by the upstream
            package) or ``"reflectance"`` (already float reflectance in
            [0, 1] -- left numerically unscaled). No other value is
            accepted.

    Returns:
        A new float32 array, same shape as ``array``, with NaN/+Inf/-Inf
        replaced by 0.0.
    """
    if input_scale not in _VALID_INPUT_SCALES:
        raise InvalidInputScaleError(
            f"Unknown input_scale {input_scale!r}; expected one of {_VALID_INPUT_SCALES}."
        )

    array = np.asarray(array)

    if input_scale == RAW_DIGITAL_NUMBER:
        # Match experiments/baseline/run_baseline.py exactly: divide in the
        # input's own precision (numpy promotes to float64), THEN cast to
        # float32 -- not the other way around, to avoid introducing any
        # avoidable discrepancy against the proven Baseline 0 numerics.
        result = (array / SENTINEL2_L2A_REFLECTANCE_SCALE).astype("float32")
    else:
        result = array.astype("float32")

    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
