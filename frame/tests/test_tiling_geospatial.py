"""Geospatial correctness of tiled output (Phase 2).

Runs the REAL pipeline (`frame.api.services.pipeline`: preprocessing ->
tile engine + uncertainty ensemble -> `derive_output_metadata` -> `write_geotiff`)
with a position-consistent fake model, then reads the written GeoTIFFs back and
checks them against the input's own georeferencing -- for a square scene, a
rectangular one and ones whose sides are not multiples of 128.

Disk hygiene: each pipeline run writes the SR raster, the uncertainty raster and
two full-size tensors (~16x the input pixels each), so every distinct scene is run
ONCE (module-scoped cache shared by all the checks) and the outputs are deleted when
the module finishes. An earlier version of this file re-ran a 511 x 777 scene for
every check and exhausted the per-user /tmp quota; keep the scenes here modest.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from rasterio.transform import Affine

from frame.api.services import pipeline
from frame.api.services.storage import JobStore
from frame.geospatial import read_geotiff, write_geotiff
from frame.preprocessing import RGBN_BANDS
from frame.preprocessing.metadata import RasterMetadata
from frame.tiling import TilingConfig

# square, rectangular, and non-multiples of 128 in both axes
SCENES = [(128, 128), (128, 256), (300, 500), (261, 389)]
DEFAULT_CRS = "EPSG:32630"
DEFAULT_ORIGIN = (720285.0, 4375125.0)


def perfect(x: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, scale_factor=4, mode="nearest")


def _make_scene(directory, h, w, *, crs, origin):
    array = (np.random.default_rng(3).random((4, h, w)).astype("float32") * 0.4) + 0.1
    x0, y0 = origin
    metadata = RasterMetadata(
        crs=crs,
        transform=(10.0, 0.0, x0, 0.0, -10.0, y0),
        bounds=(x0, y0 - h * 10.0, x0 + w * 10.0, y0),
        resolution_m=10.0,
        width=w,
        height=h,
        band_names=tuple(RGBN_BANDS),
        acquisition_timestamp="2023-01-15T10:54:11.024000",
        nodata_value=0.0,
        cloud_mask_coverage=1.0,
        sr_variant=None,
    )
    path = directory / f"scene_{h}x{w}.tif"
    write_geotiff(path, array, metadata)
    return path, array, metadata


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """``runs(h, w, overlap=32, crs=..., origin=...)`` -> (input array, input metadata, SR array,
    SR metadata, job). Each distinct request runs the pipeline once; everything is deleted at the end."""
    base = tmp_path_factory.mktemp("tiling_geospatial")
    cache = {}

    def get(h, w, *, overlap=32, crs=DEFAULT_CRS, origin=DEFAULT_ORIGIN):
        key = (h, w, overlap, crs, origin)
        if key not in cache:
            directory = base / f"run{len(cache)}"
            directory.mkdir()
            path, array, in_meta = _make_scene(directory, h, w, crs=crs, origin=origin)
            store = JobStore()
            upload = pipeline.process_upload(store, path, filename=path.name, input_scale="reflectance")
            job = pipeline.run_sr_job(
                store, upload.upload_id, model=perfect, device="cpu", seed=42, workspace_dir=directory / "jobs",
                tiling=TilingConfig(overlap=overlap),
            )
            sr_array, sr_meta = read_geotiff(job.sr_mean_path, require_crs=True)
            cache[key] = (array, in_meta, sr_array, sr_meta, job)
        return cache[key]

    yield get
    shutil.rmtree(base, ignore_errors=True)


@pytest.mark.parametrize("h,w", SCENES)
def test_output_dimensions_are_exactly_four_times_the_input(runs, h, w):
    _, in_meta, sr_array, sr_meta, job = runs(h, w)
    assert sr_array.shape == (4, 4 * h, 4 * w)
    assert (sr_meta.height, sr_meta.width) == (4 * h, 4 * w)
    assert job.result["input_shape"] == [4, h, w] and job.result["output_shape"] == [4, 4 * h, 4 * w]


@pytest.mark.parametrize("h,w", SCENES)
def test_crs_footprint_and_origin_are_unchanged(runs, h, w):
    _, in_meta, _, sr_meta, _ = runs(h, w)
    assert sr_meta.crs == in_meta.crs == DEFAULT_CRS
    assert sr_meta.bounds == pytest.approx(in_meta.bounds, abs=1e-9)  # same footprint, no drift
    assert (sr_meta.transform[2], sr_meta.transform[5]) == (in_meta.transform[2], in_meta.transform[5])  # same upper-left origin


@pytest.mark.parametrize("h,w", SCENES)
def test_pixel_size_is_divided_by_four_with_no_transform_drift(runs, h, w):
    _, in_meta, _, sr_meta, _ = runs(h, w)
    a, b, c, d, e, f = sr_meta.transform
    assert (a, b, c, d, e, f) == pytest.approx((2.5, 0.0, 720285.0, 0.0, -2.5, 4375125.0), abs=1e-12)
    assert sr_meta.resolution_m == pytest.approx(2.5)
    assert (in_meta.transform[0] / a, in_meta.transform[4] / e) == pytest.approx((4.0, 4.0))


@pytest.mark.parametrize("h,w", SCENES)
def test_the_output_grid_edges_land_exactly_on_the_input_footprint(runs, h, w):
    """Pixel alignment: the SR grid's far corner, computed from its own transform and
    size, is the input's far corner; and each input pixel is exactly a 4 x 4 block of SR pixels."""
    _, in_meta, _, sr_meta, _ = runs(h, w)
    t_in, t_sr = Affine(*in_meta.transform), Affine(*sr_meta.transform)
    assert t_sr @ (sr_meta.width, sr_meta.height) == pytest.approx(t_in @ (in_meta.width, in_meta.height), abs=1e-9)
    for row, col in [(0, 0), (h // 2, w // 2), (h - 1, w - 1)]:
        ux, uy = t_in @ (col, row)               # upper-left corner of an input pixel
        sx, sy = t_sr @ (col * 4, row * 4)       # upper-left corner of its 4 x 4 SR block
        assert (sx, sy) == pytest.approx((ux, uy), abs=1e-9)


@pytest.mark.parametrize("h,w", SCENES)
def test_content_is_at_the_right_place_no_shift_from_tiling(runs, h, w):
    """With a position-consistent model, the SR value at any world coordinate must equal
    the input value at that coordinate -- proving tiles landed where they belong, with no
    offset, mirroring or seam artefact at tile borders or padded edges."""
    array, in_meta, sr_array, sr_meta, _ = runs(h, w)
    inv_in, inv_sr = ~Affine(*in_meta.transform), ~Affine(*sr_meta.transform)
    rng = np.random.default_rng(11)
    xs = rng.uniform(in_meta.bounds[0] + 0.01, in_meta.bounds[2] - 0.01, 300)
    ys = rng.uniform(in_meta.bounds[1] + 0.01, in_meta.bounds[3] - 0.01, 300)
    # include points hugging every scene edge and the neighbourhood of the first tile seam
    xs = np.concatenate([xs, [in_meta.bounds[0] + 0.01, in_meta.bounds[2] - 0.01, in_meta.bounds[0] + 100 * 10 + 5]])
    ys = np.concatenate([ys, [in_meta.bounds[3] - 0.01, in_meta.bounds[1] + 0.01, in_meta.bounds[3] - 100 * 10 - 5]])
    inside = (xs < in_meta.bounds[2]) & (ys > in_meta.bounds[1])
    for x, y in zip(xs[inside], ys[inside]):
        c_in, r_in = inv_in @ (x, y)
        c_sr, r_sr = inv_sr @ (x, y)
        v_in = array[:, int(r_in), int(c_in)]
        v_sr = sr_array[:, int(r_sr), int(c_sr)]
        assert np.allclose(v_in, v_sr, atol=1e-5), (x, y)


def test_a_different_crs_and_origin_are_preserved_verbatim(runs):
    _, in_meta, _, sr_meta, _ = runs(261, 389, crs="EPSG:32633", origin=(500123.5, 5601987.25))
    assert sr_meta.crs == "EPSG:32633"
    assert (sr_meta.transform[2], sr_meta.transform[5]) == (500123.5, 5601987.25)
    assert sr_meta.bounds == pytest.approx(in_meta.bounds, abs=1e-9)


@pytest.mark.parametrize("overlap", [0, 16, 64])
def test_geometry_does_not_depend_on_the_overlap(runs, overlap):
    _, in_meta, sr_array, sr_meta, _ = runs(261, 389, overlap=overlap)
    assert sr_array.shape == (4, 261 * 4, 389 * 4) and sr_meta.bounds == pytest.approx(in_meta.bounds, abs=1e-9)


def test_the_uncertainty_raster_shares_the_sr_raster_geometry(runs):
    _, _, _, sr_meta, job = runs(300, 500)
    u_array, u_meta = read_geotiff(job.sr_std_path, require_crs=True)
    assert u_array.shape[1:] == (1200, 2000) and u_meta.transform == pytest.approx(sr_meta.transform) and u_meta.crs == sr_meta.crs


def test_the_pipeline_records_the_reconstruction_configuration(runs):
    *_, job = runs(261, 389, overlap=16)
    tiling = job.result["metadata"]["tiling"]
    assert tiling["tile_size"] == 128 and tiling["overlap"] == 16 and tiling["scale"] == 4
    assert tiling["padding_mode"] == "reflect" and tiling["blend_mode"] == "linear"
    assert tiling["scene_shape"] == [261, 389] and tiling["sr_shape"] == [1044, 1556]
