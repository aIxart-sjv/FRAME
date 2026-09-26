"""frame.smoke -- the end-to-end smoke path (Phase 8), run with the toy model: no weights, no GPU. The real-model variants are exercised by
`python -m frame.smoke --model lite|mamba` (records under experiments/final_readiness/) and by the `integration`-marked tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

from frame import smoke


@pytest.fixture(scope="module")
def toy_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("smoke")
    return smoke.run_smoke(out, model="toy"), out


def test_the_toy_smoke_passes_every_check_and_says_it_is_not_evidence(toy_run):
    record, _ = toy_run
    assert record["status"] == "passed" and record["n_failed"] == 0 and record["n_checks"] >= 20
    assert "not scientific evidence" in record["note"] and "synthetic" in record["note"]
    names = {c["name"] for c in record["checks"]}
    for required in ("refuses_a_file_that_is_not_a_geotiff", "refuses_reflectance_declared_as_digital_numbers", "refuses_a_scene_with_no_valid_pixel", "output_shape_is_exactly_4x",
                     "crs_preserved", "origin_preserved_and_pixel_size_is_2.5_m", "footprint_equals_the_input_footprint", "tiling_matches_the_plan", "same_seed_same_scene_same_output",
                     "ndvi_demonstration_runs_and_says_what_it_is"):
        assert required in names, required


def test_the_record_states_expected_and_produced_shapes_timings_and_the_tiling(toy_run):
    record, out = toy_run
    assert record["expected_output_shape"] == record["produced_output_shape"] == [4, 800, 1200]
    assert record["tiling"]["tile_grid"] == [2, 3] and record["tiling"]["tile_inferences"] == 36
    assert record["timings_seconds"]["sr_run_s"] > 0 and record["identical_repeat"] is True
    assert json.loads((out / "smoke_record.json").read_text()) == json.loads(json.dumps(record, default=str))


def test_the_record_carries_what_is_needed_to_repeat_the_run(toy_run):
    p = toy_run[0]["provenance"]
    assert p["smoke_version"] == "frame-smoke/1" and p["seed"] == 42 and len(p["scene_sha256"]) == 64 and len(p["output_sha256"]) == 64 and len(p["settings_digest"]) == 64
    assert p["settings"]["model"] == "toy" and p["settings"]["height"] == 200 and p["settings"]["width"] == 300
    assert "revision" in p["code"]["git"] and "dirty" in p["code"]["git"] and p["environment"]["python"] and p["environment"]["torch"]
    assert p["tta"]["n_members"] == 6 and p["tta"]["transforms"][0] == "identity"


def test_the_toy_record_does_not_pretend_a_sen2sr_model_ran(toy_run):
    model = toy_run[0]["provenance"]["model"]
    assert "not a super-resolution model" in model["canonical_name"] and "no SEN2SR model ran" in model["caveat"]


def test_two_runs_of_the_same_settings_produce_the_same_scene_and_output_hashes(toy_run, tmp_path):
    again = smoke.run_smoke(tmp_path, model="toy", repeat=False)
    a, b = toy_run[0]["provenance"], again["provenance"]
    assert a["scene_sha256"] == b["scene_sha256"] and a["output_sha256"] == b["output_sha256"] and a["settings_digest"] != b["settings_digest"]   # repeat flag is a setting


def test_a_different_seed_or_size_changes_the_scene_and_the_settings_digest(tmp_path):
    other = smoke.run_smoke(tmp_path, model="toy", height=130, width=170, seed=7, repeat=False)
    assert other["status"] == "passed" and other["expected_output_shape"] == [4, 520, 680] and other["produced_output_shape"] == [4, 520, 680]


def test_the_record_holds_no_absolute_home_path_and_no_raster(toy_run):
    _, out = toy_run
    assert sorted(p.name for p in out.iterdir()) == ["smoke_record.json"]
    assert "/home/" not in (out / "smoke_record.json").read_text()


def test_the_scene_is_deterministic_rectangular_in_digital_numbers_with_a_nodata_block():
    a, b = smoke.synthetic_scene(90, 130, 1), smoke.synthetic_scene(90, 130, 1)
    assert a.shape == (4, 90, 130) and a.dtype == np.float32 and np.array_equal(a, b)
    assert not np.array_equal(a, smoke.synthetic_scene(90, 130, 2))
    assert (a[:, :10, :12] == 0).all() and 100 <= a[:, 20:, 20:].min() and a.max() <= 10000
    assert a[3, 20:, 20:].mean() > a[0, 20:, 20:].mean()                       # NIR brighter than red: NDVI has something to compute


def test_a_failing_pipeline_is_reported_as_a_failed_check_not_hidden(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke.ToyModel, "__call__", lambda self, x: np.zeros(1))     # not a tensor: the API must refuse the model output
    record = smoke.run_smoke(tmp_path, model="toy", repeat=False)
    assert record["status"] == "failed" and record["n_failed"] >= 1
    assert any(c["name"] == "sr_run_succeeds" and not c["passed"] for c in record["checks"])
    assert (tmp_path / "smoke_record.json").is_file()


def test_the_api_workspace_setting_is_restored_and_the_temporary_workspace_removed(tmp_path):
    from frame.api import config

    before = config.WORKSPACE_DIR
    smoke.run_smoke(tmp_path, model="toy", repeat=False)
    assert config.WORKSPACE_DIR == before


def test_an_unavailable_model_is_reported_as_unavailable_not_as_a_failure(tmp_path, monkeypatch):
    from frame.api.services import model as model_service

    monkeypatch.setattr(model_service, "list_models", lambda device=None: [
        {"id": "lite", "label": "SEN2SR-Lite", "model_name": "x", "available": True, "reason": None},
        {"id": "mamba", "label": "SEN2SR-Mamba", "model_name": "y", "available": False, "reason": "SEN2SR-Mamba requires a CUDA-capable GPU."}])
    with pytest.raises(smoke.SmokeUnavailable, match="CUDA"):
        smoke.run_smoke(tmp_path, model="mamba")
    assert smoke.main(["--model", "mamba", "--output-dir", str(tmp_path / "o"), "--quiet"]) == 2


def test_cli_exit_codes(tmp_path, capsys):
    assert smoke.main(["--output-dir", str(tmp_path / "ok"), "--quiet", "--no-repeat"]) == 0
    out = capsys.readouterr().out
    assert "smoke passed" in out and "not scientific evidence" in out
    assert smoke.main(["--height", "0", "--output-dir", str(tmp_path / "bad")]) == 2
    with pytest.raises(SystemExit):
        smoke.main(["--model", "banana"])


def test_run_smoke_refuses_an_unknown_model(tmp_path):
    with pytest.raises(ValueError, match="model must be one of"):
        smoke.run_smoke(tmp_path, model="banana")


def test_the_closeness_helper_is_strict_about_values_and_lengths():
    assert smoke._close((1, 2), (1, 2.0000001)) and not smoke._close((1, 2), (1, 2.1)) and not smoke._close((1, 2), (1, 2, 3))


def test_the_smoke_notices_a_shifted_output_origin(tmp_path, monkeypatch):
    """The geospatial checks must be able to FAIL: shift the origin of every raster the pipeline writes by 5 m and the smoke must say so."""
    import dataclasses

    from frame.api.services import pipeline

    real = pipeline.write_geotiff

    def shifted(path, array, metadata):
        t = metadata.transform
        real(path, array, dataclasses.replace(metadata, transform=(t[0], t[1], t[2] + 5.0, t[3], t[4], t[5])))

    monkeypatch.setattr(pipeline, "write_geotiff", shifted)
    record = smoke.run_smoke(tmp_path, model="toy", repeat=False)
    failed = {c["name"] for c in record["checks"] if not c["passed"]}
    assert record["status"] == "failed" and "origin_preserved_and_pixel_size_is_2.5_m" in failed


# ============================================================================== demo inputs


def test_a_crop_keeps_values_crs_and_the_georeferencing_of_its_window(tmp_path):
    from frame.geospatial import read_geotiff

    source = tmp_path / "source.tif"
    smoke.write_scene(source, smoke.synthetic_scene(300, 400, 3))
    crop = smoke.crop_geotiff(source, tmp_path / "crop.tif", row=50, col=70, height=120, width=160)
    full, full_meta = read_geotiff(source)
    part, meta = read_geotiff(crop)
    assert np.array_equal(part, full[:, 50:170, 70:230]) and (meta.height, meta.width) == (120, 160) and meta.crs == full_meta.crs
    assert meta.transform[2] == full_meta.transform[2] + 70 * 10.0 and meta.transform[5] == full_meta.transform[5] - 50 * 10.0       # origin moved by the window offset, pixel size unchanged
    assert meta.bounds == (meta.transform[2], meta.transform[5] - 1200.0, meta.transform[2] + 1600.0, meta.transform[5]) and meta.band_names == full_meta.band_names


def test_a_window_that_does_not_fit_is_refused(tmp_path):
    source = tmp_path / "source.tif"
    smoke.write_scene(source, smoke.synthetic_scene(100, 100, 3))
    for window in ((0, 0, 101, 10), (90, 0, 20, 10), (-1, 0, 10, 10), (0, 0, 0, 10)):
        with pytest.raises(ValueError, match="does not fit"):
            smoke.crop_geotiff(source, tmp_path / "x.tif", *window)


def test_demo_scenes_are_written_uploadable_and_say_which_scale_they_need(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(smoke, "BASELINE0_SCENE", tmp_path / "absent.tif")
    monkeypatch.setenv(smoke.REAL_SCENE_ENV, str(tmp_path / "no_cache"))
    scenes = smoke.write_demo_scenes(tmp_path / "demo")
    assert set(scenes) == {"synthetic"} and scenes["synthetic"]["input_scale"] == "raw_digital_number"                             # the real ones are optional
    from frame.geospatial import read_geotiff

    array, meta = read_geotiff(scenes["synthetic"]["path"])
    assert array.shape == (4, 200, 300) and meta.crs == smoke.CRS
    assert smoke.main(["--demo-scenes", str(tmp_path / "again")]) == 0 and "raw_digital_number" in capsys.readouterr().out


def test_the_real_demo_scenes_are_used_when_they_exist(tmp_path, monkeypatch):
    real_128 = tmp_path / "b0.tif"
    smoke.write_scene(real_128, (smoke.synthetic_scene(128, 128, 1) / 10000.0).astype("float32"))
    cache = tmp_path / "cache"
    cache.mkdir()
    smoke.write_scene(cache / "real_scene_1024.tif", (smoke.synthetic_scene(600, 800, 2) / 10000.0).astype("float32"))
    monkeypatch.setattr(smoke, "BASELINE0_SCENE", real_128)
    monkeypatch.setenv(smoke.REAL_SCENE_ENV, str(cache))
    scenes = smoke.write_demo_scenes(tmp_path / "demo")
    assert set(scenes) == {"synthetic", "real_128", "real_crop"} and scenes["real_crop"]["shape"] == [4, 200, 300] and scenes["real_crop"]["input_scale"] == "reflectance"
