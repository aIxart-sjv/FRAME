"""frame.data.contract -- the paired-sample contract: valid pairs, malformed pairs, metadata."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from frame.data.contract import (
    DegradationRecord,
    HRStatus,
    PairedSample,
    PairRecord,
    PairType,
    PatchCoords,
    RasterSpec,
    Split,
)
from frame.data.errors import ContractError
from frame.preprocessing import RGBN_BANDS

LR_T = (10.0, 0.0, 500000.0, 0.0, -10.0, 4000000.0)
HR_T = (2.5, 0.0, 500000.0, 0.0, -2.5, 4000000.0)


def lr_spec(**kw):
    base = dict(band_names=RGBN_BANDS, width=32, height=32, pixel_size_m=10.0, path="lr/a.tif", dtype="uint16",
                reflectance_scale=10000.0, crs="EPSG:32630", transform=LR_T)
    base.update(kw)
    return RasterSpec(**base)


def hr_spec(**kw):
    base = dict(band_names=RGBN_BANDS, width=128, height=128, pixel_size_m=2.5, path="hr/a.tif", dtype="uint16",
                reflectance_scale=10000.0, crs="EPSG:32630", transform=HR_T)
    base.update(kw)
    return RasterSpec(**base)


def record(**kw):
    base = dict(sample_id="d:a", dataset="tinyset", scene_id="scene1", region_id="regionA", split="train",
                pair_type="real_cross_sensor", hr_status="available", scale_factor=4, lr=lr_spec(), hr=hr_spec())
    base.update(kw)
    return PairRecord(**base)


DEGRADATION = DegradationRecord("frame-sen2naipv2-style", "frame-degradation/1", {"scale": 4}, 7, "none", False)


# ------------------------------------------------------------------------------ valid pair


def test_a_valid_record_round_trips_through_json_exactly():
    r = record(lon=-3.5, lat=40.25, license="CC0-1.0", provenance={"date": "2023-01-15", "n": 3})
    assert PairRecord.from_dict(r.to_dict()) == r


def test_enums_accept_their_string_values_and_serialise_as_strings():
    r = record()
    assert r.split is Split.TRAIN and r.pair_type is PairType.REAL_CROSS_SENSOR and r.hr_status is HRStatus.AVAILABLE
    d = r.to_dict()
    assert d["split"] == "train" and d["pair_type"] == "real_cross_sensor" and d["hr_status"] == "available"


def test_band_names_are_canonicalised_on_the_raster_spec():
    assert lr_spec(band_names=("B4", "B3", "B2", "B8")).band_names == RGBN_BANDS


def test_scale_factor_is_per_record_not_fixed_at_four():
    r = record(scale_factor=2, lr=lr_spec(width=32, height=32), hr=hr_spec(width=64, height=64, pixel_size_m=5.0))
    assert r.scale_factor == 2


def test_a_synthetic_record_carries_its_degradation_through_a_round_trip():
    r = record(pair_type="synthetic", degradation=DEGRADATION)
    again = PairRecord.from_dict(r.to_dict())
    assert again.degradation == DEGRADATION and again.pair_type is PairType.SYNTHETIC


def test_an_independent_reference_may_be_lr_only_with_the_uncertainty_stated():
    r = record(pair_type="independent_hr_reference", hr=None, hr_status="unknown")
    assert r.hr is None and r.hr_status is HRStatus.UNKNOWN
    assert PairRecord.from_dict(r.to_dict()) == r


# ------------------------------------------------------------------------------ malformed records


@pytest.mark.parametrize(
    "overrides,code",
    [
        (dict(hr=None), "missing_hr"),                                             # available but no HR raster
        (dict(hr=None, hr_status="unknown"), "missing_hr"),                        # a real pair may not lack its HR
        (dict(hr_status="unknown"), "invalid_hr_status"),                          # HR described but status says unknown
        (dict(pair_type="synthetic"), "missing_degradation"),                      # synthetic without its recipe
        (dict(degradation=DEGRADATION), "unexpected_degradation"),                 # real pair pretending to be degraded
        (dict(split="training"), "invalid_split"),
        (dict(pair_type="fake"), "invalid_pair_type"),
        (dict(hr_status="maybe"), "invalid_hr_status"),
        (dict(scale_factor=0), "invalid_scale"),
        (dict(scale_factor=2.5), "invalid_scale"),
        (dict(scale_factor=True), "invalid_scale"),
        (dict(sample_id="has space"), "invalid_sample_id"),
        (dict(sample_id=""), "invalid_sample_id"),
        (dict(scene_id="../x"), "invalid_scene_id"),
        (dict(region_id="bad region"), "invalid_region_id"),
        (dict(lat=91.0), "invalid_geography"),
        (dict(lon=-181.0), "invalid_geography"),
    ],
)
def test_malformed_records_are_rejected_with_a_reason_code(overrides, code):
    with pytest.raises(ContractError) as excinfo:
        record(**overrides)
    assert excinfo.value.code == code


@pytest.mark.parametrize("path", ["/abs/lr.tif", "~/lr.tif", "C:\\data\\lr.tif", "../up/lr.tif", "a/../../b.tif"])
def test_absolute_or_escaping_paths_are_rejected(path):
    with pytest.raises(ContractError) as excinfo:
        lr_spec(path=path)
    assert excinfo.value.code == "absolute_path"


@pytest.mark.parametrize("kw", [dict(width=0), dict(height=-1), dict(width=1.5), dict(pixel_size_m=0), dict(pixel_size_m=-2.5),
                                dict(reflectance_scale=0), dict(transform=(1, 2, 3)), dict(band_names=())])
def test_malformed_raster_specs_are_rejected(kw):
    with pytest.raises(ContractError):
        lr_spec(**kw)


def test_unknown_band_names_are_rejected():
    with pytest.raises(ContractError):
        lr_spec(band_names=("B04", "red"))


def test_unknown_fields_and_schema_versions_are_rejected():
    d = record().to_dict()
    with pytest.raises(ContractError) as excinfo:
        PairRecord.from_dict({**d, "surprise": 1})
    assert excinfo.value.code == "unknown_field"
    with pytest.raises(ContractError) as excinfo:
        PairRecord.from_dict({**d, "schema_version": 99})
    assert excinfo.value.code == "unsupported_schema"
    missing = {k: v for k, v in d.items() if k != "scene_id"}
    with pytest.raises(ContractError) as excinfo:
        PairRecord.from_dict(missing)
    assert excinfo.value.code == "missing_field"


def test_records_are_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        record().split = Split.TEST  # type: ignore[misc]


# ------------------------------------------------------------------------------ PairedSample


def sample(**kw):
    rec = kw.pop("rec", record())
    base = dict(record=rec, lr=torch.rand(4, 32, 32), hr=torch.rand(4, 128, 128), lr_bands=RGBN_BANDS, hr_bands=RGBN_BANDS)
    base.update(kw)
    return PairedSample(**base)


def test_a_valid_sample_exposes_ids_and_scale():
    s = sample()
    assert s.sample_id == "d:a" and s.patch_id == "d:a" and s.scale_factor == 4


@pytest.mark.parametrize(
    "kw,code",
    [
        (dict(lr=torch.rand(3, 32, 32)), "band_mismatch"),                        # wrong channel count vs band names
        (dict(hr=torch.rand(4, 128, 100)), "shape_mismatch"),                     # HR is not LR x scale
        (dict(hr=torch.rand(4, 100, 100)), "shape_mismatch"),
        (dict(lr=torch.rand(4, 32, 32).double()), "invalid_sample"),              # dtype
        (dict(lr=torch.rand(32, 32)), "invalid_sample"),                          # rank
        (dict(hr=None, hr_bands=None), "missing_hr"),                             # the record promises an HR
        (dict(hr_bands=None), "invalid_sample"),                                  # hr without hr_bands
        (dict(lr_mask=torch.ones(32, 32)), "invalid_sample"),                     # mask must be bool
        (dict(hr_mask=torch.ones(64, 64, dtype=torch.bool)), "invalid_sample"),   # mask shape
    ],
)
def test_malformed_samples_are_rejected(kw, code):
    with pytest.raises(ContractError) as excinfo:
        sample(**kw)
    assert excinfo.value.code == code


def test_a_sample_reports_the_scale_relation_it_enforces():
    s = sample(rec=record(scale_factor=2, lr=lr_spec(), hr=hr_spec(width=64, height=64)), hr=torch.rand(4, 64, 64))
    assert s.hr.shape[-1] == s.lr.shape[-1] * 2


def test_an_lr_only_sample_is_valid_only_for_a_record_without_an_hr():
    rec = record(pair_type="independent_hr_reference", hr=None, hr_status="unknown")
    s = PairedSample(record=rec, lr=torch.rand(4, 32, 32), hr=None, lr_bands=RGBN_BANDS, hr_bands=None)
    assert s.hr is None
    assert s.metadata_dict()["hr_bands"] is None and s.metadata_dict()["hr_status"] == "unknown"


def test_metadata_dict_is_json_safe_and_carries_provenance():
    import json

    rec = record(pair_type="synthetic", degradation=DEGRADATION, dataset_revision="abc123", license="CC0-1.0")
    coords = PatchCoords(lr_row=4, lr_col=8, lr_height=16, lr_width=16, scale=4)
    s = dataclasses.replace(sample(rec=rec), patch=coords)
    meta = s.metadata_dict()
    json.dumps(meta)  # must not raise
    assert meta["sample_id"] == "d:a" and meta["patch_id"] == "d:a@r4c8"
    assert meta["dataset"] == "tinyset" and meta["scene_id"] == "scene1" and meta["region_id"] == "regionA"
    assert meta["split"] == "train" and meta["pair_type"] == "synthetic" and meta["scale_factor"] == 4
    assert meta["lr_bands"] == list(RGBN_BANDS) and meta["dataset_revision"] == "abc123" and meta["license"] == "CC0-1.0"
    assert meta["degradation"]["seed"] == 7 and meta["degradation"]["parameters_verified"] is False
    assert meta["patch"] == {"lr_row": 4, "lr_col": 8, "lr_height": 16, "lr_width": 16, "hr_row": 16, "hr_col": 32,
                             "hr_height": 64, "hr_width": 64, "scale": 4}


def test_patch_coords_define_the_hr_window_from_the_lr_window_and_scale():
    c = PatchCoords(lr_row=3, lr_col=5, lr_height=7, lr_width=9, scale=4)
    assert (c.hr_row, c.hr_col, c.hr_height, c.hr_width) == (12, 20, 28, 36)
    with pytest.raises(ContractError):
        PatchCoords(lr_row=-1, lr_col=0, lr_height=4, lr_width=4, scale=4)
    with pytest.raises(ContractError):
        PatchCoords(lr_row=0, lr_col=0, lr_height=0, lr_width=4, scale=4)
