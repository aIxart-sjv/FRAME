"""Step F: frame.acquisition.harmonise -- the explicit 12-band L2A contract, the real
SCL-derived validity mask, per-band radiometric-method provenance, and the one explicit RGBN
adapter boundary into the existing SR pipeline.

Uses real synthetic arrays (no network) for most tests, plus one live test against the real
Copernicus API (skipped without credentials, matching test_acquisition_live.py's convention).
"""

from __future__ import annotations

import numpy as np
import pytest

from frame.acquisition.contract import RetrievalProvenance, RetrievedScene, SceneCandidate
from frame.acquisition.errors import IncompleteL2ABandsError
from frame.acquisition.harmonise import (
    RADIOMETRIC_METHOD_BY_PROVIDER,
    harmonise,
    select_rgbn_adapter,
    validate_full_l2a_bands,
)
from frame.acquisition.providers import cdse as cdse_provider
from frame.acquisition.providers import planetary_computer
from frame.data.bands import L2A_BANDS, RGBN_BANDS
from frame.data.errors import ContractError
from frame.preprocessing.masks import DEFAULT_SCL_INVALID_CLASSES

H = 16
CRS = "EPSG:32644"
TRANSFORM = (10.0, 0.0, 500000.0, 0.0, -10.0, 1400000.0)


def _candidate() -> SceneCandidate:
    return SceneCandidate(
        id="S2B_MSIL2A_20240330T045659_N0510_R119_T44PMV_20240330T072508",
        provider="copernicus-cdse",
        stac_endpoint="https://stac.dataspace.copernicus.eu/v1/search",
        collection="sentinel-2-l2a",
        datetime="2024-03-30T04:56:59.024000Z",
        cloud_cover=2.43,
        bbox=(80.075, 12.574, 81.090, 13.568),
        geometry={"type": "Polygon", "coordinates": []},
        mgrs_tile="MGRS-44PMV",
        platform="sentinel-2b",
        processing_baseline="05.10",
        asset_hrefs={},
    )


def _retrieved_scene(
    *,
    bands=L2A_BANDS,
    band_sources=None,
    scl: "np.ndarray | None" = None,
    warnings=(),
) -> RetrievedScene:
    rng = np.random.default_rng(0)
    band_arrays = {b: (rng.random((H, H)).astype("float32") * 0.4 + 0.1) for b in bands}
    band_native_resolution_m = {}
    for b in bands:
        if b in ("B02", "B03", "B04", "B08"):
            band_native_resolution_m[b] = 10.0
        elif b in ("B05", "B06", "B07", "B8A", "B11", "B12"):
            band_native_resolution_m[b] = 20.0
        else:  # B01, B09
            band_native_resolution_m[b] = 60.0

    sources = band_sources or {b: cdse_provider.PROVIDER_NAME for b in bands}
    provenance = RetrievalProvenance(
        query=None,
        candidate_id=_candidate().id,
        search_provider="copernicus-cdse",
        search_stac_endpoint="https://stac.dataspace.copernicus.eu/v1/search",
        retrieval_provider="+".join(sorted(set(sources.values()))),
        retrieval_endpoint="https://sh.dataspace.copernicus.eu/process/v1",
        matched_acquisition_datetime=_candidate().datetime,
        retrieved_at_utc="2024-03-30T05:00:00Z",
        band_sources=sources,
        warnings=warnings,
    )
    return RetrievedScene(
        candidate=_candidate(),
        bands=band_arrays,
        band_native_resolution_m=band_native_resolution_m,
        scl=scl,
        crs=CRS,
        transform=TRANSFORM,
        resolution_m=10.0,
        width=H,
        height=H,
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Band contract
# ---------------------------------------------------------------------------

def test_validate_full_l2a_bands_accepts_all_twelve():
    validate_full_l2a_bands(L2A_BANDS)  # no raise


def test_validate_full_l2a_bands_rejects_a_missing_band():
    with pytest.raises(IncompleteL2ABandsError) as excinfo:
        validate_full_l2a_bands([b for b in L2A_BANDS if b != "B05"])
    assert "B05" in str(excinfo.value)


def test_validate_full_l2a_bands_rejects_b10_being_absent_is_correct_not_a_bug():
    # L2A_BANDS never includes B10 -- passing exactly L2A_BANDS (no B10) must pass, proving
    # completeness is judged against the real 12-band definition, not a naive "all 13" count.
    assert len(L2A_BANDS) == 12
    assert "B10" not in L2A_BANDS
    validate_full_l2a_bands(L2A_BANDS)


def test_validate_full_l2a_bands_rejects_an_unrecognised_band_name():
    with pytest.raises(ContractError):
        validate_full_l2a_bands(list(L2A_BANDS) + ["NotABand"])


def test_harmonised_product_band_names_are_canonical_regardless_of_input_order():
    scene = _retrieved_scene(bands=("b8a", "B04", "b2"))
    product = harmonise(scene)
    assert set(product.band_names) == {"B8A", "B04", "B02"}


def test_is_complete_l2a_reports_honestly():
    full = harmonise(_retrieved_scene(bands=L2A_BANDS))
    partial = harmonise(_retrieved_scene(bands=RGBN_BANDS))
    assert full.is_complete_l2a() is True
    assert partial.is_complete_l2a() is False


# ---------------------------------------------------------------------------
# Native resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("band,expected_m", [
    ("B02", 10.0), ("B03", 10.0), ("B04", 10.0), ("B08", 10.0),
    ("B05", 20.0), ("B06", 20.0), ("B07", 20.0), ("B8A", 20.0), ("B11", 20.0), ("B12", 20.0),
    ("B01", 60.0), ("B09", 60.0),
])
def test_native_resolution_is_preserved_per_band_after_harmonisation(band, expected_m):
    product = harmonise(_retrieved_scene(bands=L2A_BANDS))
    assert product.band_native_resolution_m[band] == expected_m
    # the grid itself is the target resolution, not the native one -- these differ for
    # every band except the 10 m group, and that difference must never be erased
    assert product.resolution_m == 10.0


def test_the_10m_20m_60m_groups_are_all_distinct_after_harmonisation():
    product = harmonise(_retrieved_scene(bands=L2A_BANDS))
    resolutions = set(product.band_native_resolution_m.values())
    assert resolutions == {10.0, 20.0, 60.0}


# ---------------------------------------------------------------------------
# Resampling discipline (categorical vs continuous) -- exercised at the
# frame.acquisition.retrieve level (the one place real resampling happens);
# these tests confirm harmonise() does not perform a SECOND, uncontrolled
# resampling pass and that a real SCL array's class codes survive untouched.
# ---------------------------------------------------------------------------

def test_harmonise_does_not_resample_scl_a_second_time():
    scl = np.array([[3, 4, 8], [0, 4, 11], [4, 4, 4]], dtype="float32")
    product = harmonise(_retrieved_scene(scl=scl))
    np.testing.assert_array_equal(product.scl, scl)
    # every value is still one of the real, discrete SCL class codes -- never a blended average
    assert set(np.unique(product.scl)) <= {float(v) for v in range(12)}


def test_harmonise_never_claims_a_resampling_method_of_its_own():
    """harmonise() must not silently apply bilinear/bicubic to anything -- resampling policy
    lives entirely in frame.acquisition.retrieve, which is bilinear for reflectance and
    nearest-neighbour for SCL. This is a structural check: harmonise()'s source has no call to
    any interpolation/resize function at all."""
    import inspect

    from frame.acquisition import harmonise as harmonise_module

    source = inspect.getsource(harmonise_module)
    for forbidden in ("scipy.ndimage", "cv2.resize", "PIL", "zoom(", "resize("):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# Radiometry
# ---------------------------------------------------------------------------

def test_radiometric_method_is_tagged_per_band_from_real_provenance():
    scene = _retrieved_scene(bands=RGBN_BANDS, band_sources={b: cdse_provider.PROVIDER_NAME for b in RGBN_BANDS})
    product = harmonise(scene)
    for band in RGBN_BANDS:
        assert product.band_radiometric_method[band] == RADIOMETRIC_METHOD_BY_PROVIDER[cdse_provider.PROVIDER_NAME]
        assert "baseline" in product.band_radiometric_method[band]


def test_mixed_provider_bands_get_different_honest_radiometric_methods():
    """No silent single global scaling assumption: a product with some bands from CDSE and
    others from the Planetary Computer fallback must say so per band, not present one method
    for the whole stack."""
    sources = {"B04": cdse_provider.PROVIDER_NAME, "B03": cdse_provider.PROVIDER_NAME,
               "B02": planetary_computer.PROVIDER_NAME, "B08": planetary_computer.PROVIDER_NAME}
    product = harmonise(_retrieved_scene(bands=RGBN_BANDS, band_sources=sources))

    assert product.band_radiometric_method["B04"] != product.band_radiometric_method["B02"]
    assert "flat-dn-div-10000" in product.band_radiometric_method["B02"]
    assert "cdse-process-api" in product.band_radiometric_method["B04"]


def test_an_unknown_provider_is_reported_not_hidden():
    product = harmonise(_retrieved_scene(bands=("B04",), band_sources={"B04": "some-future-provider"}))
    assert "unknown provider" in product.band_radiometric_method["B04"]


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------

def test_validity_mask_is_built_from_real_scl_using_the_existing_default_classes():
    scl = np.full((H, H), 4, dtype="float32")  # 4 = vegetation, always valid
    for column, invalid_class in enumerate(DEFAULT_SCL_INVALID_CLASSES):
        scl[0, column] = invalid_class  # 0=NoData, 1=saturated, 3=shadow, 8/9=cloud, 10=cirrus
    product = harmonise(_retrieved_scene(scl=scl))

    assert product.validity_mask is not None
    assert product.validity_mask.array.shape == (H, H)
    expected_coverage = 1.0 - (len(DEFAULT_SCL_INVALID_CLASSES) / (H * H))
    assert product.validity_mask.coverage() == pytest.approx(expected_coverage)


def test_validity_mask_is_none_when_scl_was_not_retrieved():
    product = harmonise(_retrieved_scene(scl=None))
    assert product.validity_mask is None


def test_masking_never_changes_the_multispectral_arrays_shape_or_values():
    """Masking must never delete pixels -- the bands stay full-resolution/full-extent; only a
    separate boolean mask is produced."""
    scl = np.zeros((H, H), dtype="float32")  # every pixel "NoData" (class 0) -- fully invalid
    scene = _retrieved_scene(bands=("B04",), scl=scl)
    original_b04 = scene.bands["B04"].copy()
    product = harmonise(scene)

    np.testing.assert_array_equal(product.bands["B04"], original_b04)
    assert product.validity_mask.coverage() == 0.0
    assert product.bands["B04"].shape == (H, H)


# ---------------------------------------------------------------------------
# Geospatial pass-through
# ---------------------------------------------------------------------------

def test_geospatial_metadata_passes_through_unchanged():
    scene = _retrieved_scene()
    product = harmonise(scene)
    assert product.crs == scene.crs == CRS
    assert product.transform == scene.transform == TRANSFORM
    assert product.resolution_m == scene.resolution_m == 10.0
    assert (product.width, product.height) == (scene.width, scene.height) == (H, H)


def test_provenance_object_is_preserved_not_rebuilt():
    scene = _retrieved_scene()
    product = harmonise(scene)
    assert product.provenance is scene.provenance
    assert product.provenance.candidate_id == _candidate().id


# ---------------------------------------------------------------------------
# The explicit RGBN adapter
# ---------------------------------------------------------------------------

def test_rgbn_adapter_on_a_full_12band_product_returns_exactly_rgbn_in_canonical_order():
    product = harmonise(_retrieved_scene(bands=L2A_BANDS))
    array, band_names, excluded = select_rgbn_adapter(product)

    assert array.shape == (4, H, H)
    assert band_names == RGBN_BANDS
    assert set(excluded) == set(L2A_BANDS) - set(RGBN_BANDS)
    assert len(excluded) == 8  # 12 - 4


def test_rgbn_adapter_on_an_rgbn_only_product_excludes_nothing():
    product = harmonise(_retrieved_scene(bands=RGBN_BANDS))
    array, band_names, excluded = select_rgbn_adapter(product)
    assert excluded == []
    assert array.shape == (4, H, H)


def test_rgbn_adapter_refuses_a_product_missing_a_required_band():
    product = harmonise(_retrieved_scene(bands=("B04", "B08")))
    with pytest.raises(ContractError):
        select_rgbn_adapter(product)


def test_rgbn_adapter_never_touches_the_native_resolution_record():
    """Selecting the RGBN subset must not relabel B01/B09/etc as if they were never seen --
    those stay in band_native_resolution_m even though select_rgbn_adapter doesn't return them."""
    product = harmonise(_retrieved_scene(bands=L2A_BANDS))
    select_rgbn_adapter(product)  # exercised for its side-effect-free-ness
    assert product.band_native_resolution_m["B01"] == 60.0
    assert product.band_native_resolution_m["B11"] == 20.0


# ---------------------------------------------------------------------------
# Integration: 12-band harmonisation -> RGBN adapter -> the existing, real,
# unmodified SR pipeline (mirrors test_api_acquisition_sr_handoff.py's own
# real end-to-end proof, starting one level earlier: from a full 12-band
# HarmonisedProduct rather than an already-4-band retrieval).
# ---------------------------------------------------------------------------

def test_full_12band_harmonisation_feeds_the_real_unmodified_sr_pipeline(tmp_path):
    import torch
    import torch.nn.functional as F

    from frame.geospatial import RGBN_SCALE_FACTOR, write_geotiff
    from frame.preprocessing.metadata import RasterMetadata

    def equivariant_model(x_batched: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x_batched, scale_factor=RGBN_SCALE_FACTOR, mode="bicubic", antialias=True)

    scene = _retrieved_scene(bands=L2A_BANDS)
    product = harmonise(scene)
    assert product.is_complete_l2a()

    rgbn_array, band_names, excluded = select_rgbn_adapter(product)
    assert len(excluded) == 8

    metadata = RasterMetadata(
        crs=product.crs, transform=product.transform, bounds=None, resolution_m=product.resolution_m,
        width=product.width, height=product.height, band_names=band_names,
        acquisition_timestamp=product.provenance.matched_acquisition_datetime,
        nodata_value=None, cloud_mask_coverage=None,
        sr_variant="frame.acquisition harmonisation -- RGBN adapter output",
    )
    sr_input_path = tmp_path / "sr_input.tif"
    write_geotiff(sr_input_path, rgbn_array, metadata)

    from frame.api.services.pipeline import process_upload, run_sr_job
    from frame.api.services.storage import JobStore

    store = JobStore()
    upload = process_upload(store, sr_input_path, filename="from_harmonised.tif", input_scale="reflectance")
    job = run_sr_job(store, upload.upload_id, model=equivariant_model, device="cpu", seed=42, workspace_dir=tmp_path / "jobs")

    assert job.result["status"] == "completed"
    assert job.result["bands"] == list(RGBN_BANDS)
    assert job.result["resolution"]["scale_factor"] == RGBN_SCALE_FACTOR


# ---------------------------------------------------------------------------
# Live (real network, real Copernicus API) -- skipped without credentials,
# same convention as frame/tests/test_acquisition_live.py.
# ---------------------------------------------------------------------------

import os  # noqa: E402

_HAS_LIVE_CDSE_CREDENTIALS = bool(os.environ.get("FRAME_CDSE_CLIENT_ID")) and bool(os.environ.get("FRAME_CDSE_CLIENT_SECRET"))


@pytest.mark.live_cdse  # exempts this test from conftest.py's usual CDSE-credential stripping
@pytest.mark.skipif(not _HAS_LIVE_CDSE_CREDENTIALS, reason="FRAME_CDSE_CLIENT_ID/SECRET not set -- live harmonisation test skipped.")
def test_live_full_12band_retrieval_harmonises_correctly():
    from frame.acquisition.contract import AcquisitionQuery
    from frame.acquisition.providers.cdse import CDSEAuth
    from frame.acquisition.retrieve import retrieve_scene
    from frame.acquisition.search import search_scenes

    query = AcquisitionQuery(
        bbox=(80.204, 13.006, 80.207, 13.009),  # the same tiny, already-verified AOI as test_acquisition_live.py
        start_date="2024-01-01", end_date="2024-04-01", max_cloud_cover=20.0,
    )
    candidates = search_scenes(query)
    candidate = candidates[0]

    retrieved = retrieve_scene(candidate, query, bands=L2A_BANDS, resolution_m=10.0, include_scl=True, cdse_auth=CDSEAuth())
    product = harmonise(retrieved)

    assert product.is_complete_l2a()
    assert product.validity_mask is not None
    assert 0.0 <= product.validity_mask.coverage() <= 1.0
    assert product.band_native_resolution_m["B01"] == 60.0
    assert product.band_native_resolution_m["B02"] == 10.0
    for method in product.band_radiometric_method.values():
        assert "unknown provider" not in method
    # this test's whole point is to exercise the real CDSE path (primary provider), not a
    # silent fallback to Planetary Computer -- assert that plainly rather than just hoping.
    assert all(source == cdse_provider.PROVIDER_NAME for source in retrieved.provenance.band_sources.values())

    rgbn_array, band_names, excluded = select_rgbn_adapter(product)
    assert rgbn_array.shape[0] == 4
    assert len(excluded) == 8
