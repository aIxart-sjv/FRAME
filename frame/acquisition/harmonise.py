"""Explicit, non-learned Sentinel-2 L2A multispectral harmonisation (Requirements 142,
Requirement 2: Multispectral Preprocessing & Harmonisation).

This module does not retrieve pixels (`frame.acquisition.retrieve` already does the real,
provider-backed retrieval and the actual per-group resampling onto one grid) and it does not
run a model. It takes an already-retrieved `RetrievedScene` and makes three things explicit,
testable, and reusable that were previously only implicit in that retrieval:

    1. Full 12-band L2A coverage as a named, validated contract (`validate_full_l2a_bands`),
       distinct from "whatever subset of bands happened to be requested".
    2. A validity mask built from the real SCL layer (`frame.preprocessing.masks.ValidityMask`),
       attached to the multispectral stack rather than used to delete pixels from it.
    3. Per-band radiometric-conversion provenance (which real provider served a band, and what
       conversion that implies), so "12-band harmonisation" never silently pretends every band
       went through the same radiometric path.

Requirements 142 Section 2's own framing (§25) governs the wording used everywhere here and by
every caller of this module: after harmonisation, every channel shares one grid's tensor
dimensions, but NOT the same original information resolution -- this is a "12-band Sentinel-2
L2A stack resampled/aligned to a common computational grid", never "12 native N m bands".

Never claims a genuine 12-band learned 2.5 m SR output: `select_rgbn_adapter` below is the one,
explicit boundary between this 12-band representation and the existing, unmodified, RGBN-only
SR pipeline (frame/api/services/pipeline.py) -- see docs/CLAIMS.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from frame.acquisition.contract import RetrievalProvenance, RetrievedScene
from frame.acquisition.errors import IncompleteL2ABandsError
from frame.acquisition.providers import cdse as cdse_provider
from frame.acquisition.providers import planetary_computer
from frame.data.bands import L2A_BANDS, RGBN_BANDS, canonical_bands, select_bands
from frame.preprocessing.masks import ValidityMask

#: Human-readable, honest description of what radiometric conversion a band actually went
#: through, keyed by the real provider name that served it (frame.acquisition.providers).
#: CDSE's own Process API applies the correct per-processing-baseline offset server-side
#: (units="REFLECTANCE" -- see frame.acquisition.providers.cdse); Planetary Computer/cubo
#: returns raw digital numbers that this project converts with one fixed, documented formula
#: (frame.preprocessing.reflectance.to_reflectance: a flat DN/10000, no baseline offset). These
#: are genuinely different real conversions -- never described as if they were the same one.
RADIOMETRIC_METHOD_BY_PROVIDER: Dict[str, str] = {
    cdse_provider.PROVIDER_NAME: (
        "cdse-process-api-reflectance-units (server-side, per-processing-baseline offset applied "
        "by Sentinel Hub, not by this project)"
    ),
    planetary_computer.PROVIDER_NAME: (
        "flat-dn-div-10000 (frame.preprocessing.reflectance.to_reflectance; no processing-baseline "
        "offset correction -- a documented, real discrepancy versus the CDSE path for scenes "
        "processed under baseline >= 04.00)"
    ),
}


def validate_full_l2a_bands(available_bands) -> None:
    """Raise `IncompleteL2ABandsError` unless ``available_bands`` covers every one of the 12
    required L2A bands (B01-B09, B8A, B11, B12; B10 excluded by definition, never checked for).

    This is the "12-band product" contract Requirements 142 Requirement 2 names explicitly --
    kept separate from `frame.acquisition.retrieve`, which never requires a caller to request
    all 12 (a caller genuinely may want fewer). Band names are canonicalised first
    (frame.data.bands.canonical_band), so an unrecognised name fails with that function's own
    clear error rather than being silently ignored here.
    """
    available = canonical_bands(available_bands)
    missing = [b for b in L2A_BANDS if b not in available]
    if missing:
        raise IncompleteL2ABandsError(
            f"Not a complete 12-band L2A set: missing {missing}. Real bands present: {list(available)}."
        )


@dataclass(frozen=True)
class HarmonisedProduct:
    """A real, multi-band Sentinel-2 L2A stack on one common computational grid, with the
    per-band native-resolution and radiometric-conversion provenance that grid necessarily
    erases from the raw arrays themselves, plus a real validity mask built from the scene's own
    SCL layer (never used to delete pixels -- see `frame.preprocessing.masks.ValidityMask`).

    Wording (Requirements 142 §25): this is a "12-band Sentinel-2 L2A stack resampled/aligned to
    a common computational grid" -- never described as "N native M m bands". A band whose
    ``band_native_resolution_m`` differs from ``resolution_m`` was genuinely resampled (bilinear
    for reflectance, nearest-neighbour for SCL -- see `frame.acquisition.retrieve`); resampling
    does not create information the source band did not have.
    """

    bands: Dict[str, np.ndarray]
    band_native_resolution_m: Dict[str, float]
    band_radiometric_method: Dict[str, str]
    validity_mask: Optional[ValidityMask]
    scl: Optional[np.ndarray]
    crs: str
    transform: Tuple[float, float, float, float, float, float]
    resolution_m: float
    width: int
    height: int
    provenance: RetrievalProvenance

    @property
    def band_names(self) -> Tuple[str, ...]:
        return tuple(canonical_bands(self.bands.keys()))

    def is_complete_l2a(self) -> bool:
        """Whether this product genuinely covers all 12 required L2A bands (informational --
        does not raise; use `validate_full_l2a_bands(self.band_names)` where completeness must
        be enforced, e.g. before a strict 12-band-consuming caller runs)."""
        try:
            validate_full_l2a_bands(self.band_names)
        except IncompleteL2ABandsError:
            return False
        return True


def harmonise(retrieved: RetrievedScene) -> HarmonisedProduct:
    """Wrap an already-retrieved scene into an explicit `HarmonisedProduct`: attaches a real SCL
    validity mask (if SCL was retrieved) and per-band radiometric-method provenance. Performs no
    network access and no further resampling -- `frame.acquisition.retrieve.retrieve_scene`
    already did the one real resampling pass (bilinear for reflectance bands, nearest-neighbour
    for SCL) when it built ``retrieved``.
    """
    validity_mask = ValidityMask.from_scl(retrieved.scl) if retrieved.scl is not None else None

    band_radiometric_method = {
        band: RADIOMETRIC_METHOD_BY_PROVIDER.get(source, f"unknown provider {source!r}")
        for band, source in retrieved.provenance.band_sources.items()
        if band in retrieved.bands  # exclude SCL's own entry, which is not a radiometric band
    }

    return HarmonisedProduct(
        bands=dict(retrieved.bands),
        band_native_resolution_m=dict(retrieved.band_native_resolution_m),
        band_radiometric_method=band_radiometric_method,
        validity_mask=validity_mask,
        scl=retrieved.scl,
        crs=retrieved.crs,
        transform=retrieved.transform,
        resolution_m=retrieved.resolution_m,
        width=retrieved.width,
        height=retrieved.height,
        provenance=retrieved.provenance,
    )


def select_rgbn_adapter(product: HarmonisedProduct) -> Tuple[np.ndarray, Tuple[str, ...], List[str]]:
    """The one explicit boundary between a (potentially 12-band) `HarmonisedProduct` and the
    existing, unmodified, RGBN-only SR pipeline (frame/api/services/pipeline.py).

    Returns ``(array, RGBN_BANDS, excluded_bands)``: ``array`` is shaped
    ``(4, height, width)`` in canonical `frame.data.bands.RGBN_BANDS` order (B04, B03, B02,
    B08), built with the same `frame.data.bands.select_bands` primitive
    `frame/api/services/acquisition.py::prepare_sr_input` already uses for its own file-backed
    version of this same adapter -- there is one reusable primitive behind both, not two
    competing implementations. ``excluded_bands`` names every other real band this product has
    (e.g. the SWIR/red-edge/60 m bands of a full 12-band product); they are never silently
    dropped by this function -- naming them is the whole point of returning the list -- but this
    function does not persist them anywhere itself.

    Raises `ContractError` (frame.data.bands) if the product does not even have the 4 RGBN
    bands -- the current SR models cannot run without them, and this is never worked around by
    synthesising a missing band.
    """
    available = product.band_names
    array = np.stack([product.bands[name] for name in available], axis=0)
    rgbn_array = select_bands(array, available, RGBN_BANDS)  # raises ContractError if incomplete
    excluded_bands = [b for b in available if b not in RGBN_BANDS]
    return rgbn_array, RGBN_BANDS, excluded_bands
