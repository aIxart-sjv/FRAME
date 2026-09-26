"""Spectral band naming and explicit selection (Phase 3).

Datasets name Sentinel-2 bands differently (SEN2NEON says ``B1``/``B9``, the
OpenSR-Test/FRAME convention is ``B01``/``B09``, SEN2VENµS file names say
``b2b3b4b8``). This module gives every band ONE canonical name and makes band
selection explicit: channels are never reordered implicitly, and asking for a
band a sample does not have is an error.

FRAME's RGBN order for the Lite/Mamba models is B04, B03, B02, B08 (see
`frame.models.config.RGBN_BAND_ORDER`) -- NOT the ascending file order of
several datasets. `select_bands` is how a dataset's native order becomes it.
"""

from __future__ import annotations

import re
from typing import Sequence, Tuple, TypeVar

from frame.data.errors import ContractError
from frame.preprocessing import RGBN_BANDS

#: All twelve Sentinel-2 L2A bands in ascending order (B10 is not in L2A).
#: Identical to frame.validation.L2A_BAND_ORDER (the OpenSR-Test published
#: table); frame/tests/test_data_bands.py asserts the two never drift apart.
L2A_BANDS: Tuple[str, ...] = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")

__all__ = ["L2A_BANDS", "RGBN_BANDS", "canonical_band", "canonical_bands", "select_bands"]

_BAND = re.compile(r"[Bb]0*(\d{1,2})([Aa]?)")

ArrayLike = TypeVar("ArrayLike")


def canonical_band(name: str) -> str:
    """``"B1"`` / ``"b01"`` -> ``"B01"``; ``"B8A"`` -> ``"B8A"``. Raises `ContractError` otherwise."""
    match = _BAND.fullmatch(str(name).strip())
    if match is None:
        raise ContractError(f"Not a Sentinel-2 band name: {name!r}.")
    number, suffix = int(match.group(1)), match.group(2).upper()
    if suffix:
        if number != 8:
            raise ContractError(f"Only B8A has an 'A' suffix, got {name!r}.")
        return "B8A"
    if not 1 <= number <= 12:
        raise ContractError(f"Sentinel-2 band number out of range in {name!r}.")
    return f"B{number:02d}"


def canonical_bands(names: Sequence[str]) -> Tuple[str, ...]:
    """Canonicalise every name; duplicates are an error."""
    out = tuple(canonical_band(n) for n in names)
    if len(set(out)) != len(out):
        raise ContractError(f"Duplicate band(s) in {list(names)}.")
    return out


def select_bands(array: ArrayLike, available: Sequence[str], requested: Sequence[str]) -> ArrayLike:
    """Return ``array`` (channels first) reduced/reordered to ``requested``, by name.

    Both name lists are canonicalised first. A requested band that is not
    available raises `ContractError`; nothing is guessed or filled in.
    """
    have = canonical_bands(available)
    want = canonical_bands(requested)
    if array.shape[0] != len(have):  # type: ignore[attr-defined]
        raise ContractError(f"Array has {array.shape[0]} channel(s) but {len(have)} band name(s) were given.")  # type: ignore[attr-defined]
    missing = [b for b in want if b not in have]
    if missing:
        raise ContractError(f"Band(s) {missing} are not available; available: {list(have)}.")
    return array[[have.index(b) for b in want]]  # type: ignore[index]
