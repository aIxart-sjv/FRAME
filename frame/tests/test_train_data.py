"""frame.train.data -- the manifest -> train/val datasets wiring and its leakage gate."""

from __future__ import annotations

import dataclasses
import json

import pytest
import torch

from frame.data.adapters import ADAPTERS, sen2neon
from frame.data.contract import Split
from frame.data.manifest import manifest_digest, read_manifest, write_manifest
from frame.data.roles import DATASET_PROFILES
from frame.tests.data_fixtures import (  # noqa: F401
    TINY,
    TinyAdapter,
    TinyEnv,
    build_tiny_records,
    tiny_env,
    tiny_profile,
    tiny_profile_installed,
)
from frame.tests.data_real_rows import REAL_ROWS
from frame.train.config import TrainConfig
from frame.train.data import prepare_training_data
from frame.train.errors import LeakageGuardError, TrainDataError

PATCH = 16


@pytest.fixture(autouse=True)
def tiny_adapter_registered(monkeypatch):
    monkeypatch.setitem(ADAPTERS, TINY, TinyAdapter)


def config_for(tmp_path, env=None, manifest="m.jsonl", **data):
    d = {
        "name": "unit", "model": {"name": "tiny_cnn", "params": {"width": 8, "depth": 2}}, "steps": 2, "output_dir": str(tmp_path / "run"),
        "data": {"manifest": str(tmp_path / manifest), "data_root": str(env.root) if env else None, "lr_patch": PATCH, "val_lr_patch": 32,
                 "patches_per_pair": 3, **data},
    }
    return TrainConfig.from_dict(d)


def write(tmp_path, records, name="m.jsonl"):
    return write_manifest(tmp_path / name, records)


# ============================================================================== the happy path


def test_train_and_validation_datasets_come_from_the_manifests_own_splits(tiny_env: TinyEnv, tmp_path):
    write(tmp_path, tiny_env.records)
    data = prepare_training_data(config_for(tmp_path, tiny_env))
    assert {r.split for r in data.train_records} == {Split.TRAIN} and {r.region_id for r in data.train_records} == {"A"}
    assert {r.split for r in data.val_records} == {Split.VAL} and {r.region_id for r in data.val_records} == {"B"}
    assert len(data.train) == 4 * 3 and len(data.val) == 4                # 4 train pairs x 3 patches; 4 val pairs x 1 full patch


def test_items_have_the_shapes_and_masks_the_trainer_expects(tiny_env: TinyEnv, tmp_path):
    write(tmp_path, tiny_env.records)
    data = prepare_training_data(config_for(tmp_path, tiny_env))
    t, v = data.train[0], data.val[0]
    assert t["lr"].shape == (4, PATCH, PATCH) and t["hr"].shape == (4, 4 * PATCH, 4 * PATCH) and t["hr_mask"].dtype == torch.bool
    assert v["lr"].shape == (4, 32, 32) and v["hr"].shape == (4, 128, 128) and v["metadata"]["augmentation"] == "identity"
    assert data.scale == 4 and data.band_names == ("B04", "B03", "B02", "B08")


def test_the_manifest_digest_is_recorded_exactly(tiny_env: TinyEnv, tmp_path):
    digest = write(tmp_path, tiny_env.records)
    data = prepare_training_data(config_for(tmp_path, tiny_env))
    assert data.manifest_digest == digest == manifest_digest(read_manifest(tmp_path / "m.jsonl").records)


def test_validation_is_deterministic_and_never_augmented(tiny_env: TinyEnv, tmp_path):
    write(tmp_path, tiny_env.records)
    a, b = (prepare_training_data(config_for(tmp_path, tiny_env)) for _ in range(2))
    assert all(torch.equal(a.val[i]["hr"], b.val[i]["hr"]) for i in range(len(a.val)))
    assert {a.val[i]["metadata"]["augmentation"] for i in range(len(a.val))} == {"identity"}


def test_training_patches_are_reproducible_from_the_seed_and_epoch(tiny_env: TinyEnv, tmp_path):
    write(tmp_path, tiny_env.records)
    a, b = (prepare_training_data(config_for(tmp_path, tiny_env)) for _ in range(2))
    assert all(torch.equal(a.train[i]["lr"], b.train[i]["lr"]) for i in range(len(a.train)))
    a.train.set_epoch(1)
    assert any(not torch.equal(a.train[i]["lr"], b.train[i]["lr"]) for i in range(len(a.train)))


def test_the_info_block_records_what_data_was_used(tiny_env: TinyEnv, tmp_path):
    write(tmp_path, tiny_env.records)
    info = prepare_training_data(config_for(tmp_path, tiny_env)).info
    assert info["n_train_pairs"] == 4 and info["n_val_pairs"] == 4 and info["datasets"] == [TINY] and info["ignored_records"] == {}
    assert info["train_scenes"] == ["A_s1", "A_s2"] and info["val_scenes"] == ["B_s1", "B_s2"] and info["train_regions"] == ["A"]
    assert info["qc"]["ok"] is True and info["manifest_digest"] == data_digest(tmp_path)


def data_digest(tmp_path):
    return json.loads((tmp_path / "m.jsonl").read_text().splitlines()[0])["_header"]["digest"]


def test_a_relative_manifest_path_is_resolved_against_the_config_directory(tiny_env: TinyEnv, tmp_path):
    write(tmp_path, tiny_env.records)
    cfg = config_for(tmp_path, tiny_env, manifest="m.jsonl")
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, manifest="m.jsonl"))
    assert prepare_training_data(cfg, base_dir=tmp_path).info["n_train_pairs"] == 4


# ============================================================================== test-only data never enters training


def with_test_split(tmp_path, monkeypatch):
    monkeypatch.setitem(DATASET_PROFILES, TINY, tiny_profile(splits=frozenset({Split.TRAIN, Split.VAL, Split.TEST})))
    records = build_tiny_records(tmp_path, {"A": {"s1": 2}, "B": {"s1": 2}, "C": {"s1": 2}}, splits={"A": Split.TRAIN, "B": Split.VAL, "C": Split.TEST})
    return records


def test_test_split_records_are_ignored_and_their_files_are_never_opened(tmp_path, monkeypatch):
    records = with_test_split(tmp_path, monkeypatch)
    for r in records:
        if r.split is Split.TEST:
            (tmp_path / TINY / r.lr.path).unlink()
            (tmp_path / TINY / r.hr.path).unlink()                           # if anything tried to read these, it would fail
    write(tmp_path, records)
    data = prepare_training_data(config_for(tmp_path, None, data_root=str(tmp_path)))
    assert data.info["ignored_records"] == {"test": 2} and {r.region_id for r in data.train_records + data.val_records} == {"A", "B"}
    for i in range(len(data.train)):
        data.train[i]
    for i in range(len(data.val)):
        data.val[i]


def test_a_benchmark_record_labelled_train_is_refused_before_any_pixel_is_read(tiny_env: TinyEnv, tmp_path):
    neon = [dataclasses.replace(sen2neon.record_from_row(row), split=Split.TRAIN) for row in REAL_ROWS.values()]
    write(tmp_path, tiny_env.records + neon)
    with pytest.raises(LeakageGuardError) as excinfo:
        prepare_training_data(config_for(tmp_path, tiny_env))
    assert "role_violation" in excinfo.value.codes and "sen2neon" in str(excinfo.value)


def test_benchmark_records_in_the_test_split_may_share_a_manifest_but_never_train(tiny_env: TinyEnv, tmp_path):
    neon = [sen2neon.record_from_row(row) for row in REAL_ROWS.values()]                  # split = test, files absent
    write(tmp_path, tiny_env.records + neon)
    data = prepare_training_data(config_for(tmp_path, tiny_env))
    assert all(r.dataset == TINY for r in data.train_records + data.val_records) and data.info["ignored_records"] == {"test": 3}


def test_a_benchmark_relabelled_as_validation_is_also_refused(tiny_env: TinyEnv, tmp_path):
    neon = [dataclasses.replace(sen2neon.record_from_row(row), split=Split.VAL) for row in REAL_ROWS.values()]
    write(tmp_path, tiny_env.records + neon)
    with pytest.raises(LeakageGuardError) as excinfo:
        prepare_training_data(config_for(tmp_path, tiny_env))
    assert "role_violation" in excinfo.value.codes


# ============================================================================== geographic leakage


def test_a_scene_split_across_train_and_validation_is_refused(tiny_env: TinyEnv, tmp_path):
    records = list(tiny_env.records)
    records[1] = dataclasses.replace(records[1], split=Split.VAL)                          # tile 1 of scene A_s1 -> val, tile 0 stays in train
    write(tmp_path, records)
    with pytest.raises(LeakageGuardError) as excinfo:
        prepare_training_data(config_for(tmp_path, tiny_env))
    assert "scene_leakage" in excinfo.value.codes and "A_s1" in str(excinfo.value)


def test_a_region_split_across_train_and_validation_is_refused_by_default(tiny_env: TinyEnv, tmp_path):
    records = list(tiny_env.records)
    for i, r in enumerate(records):                                                        # scene A_s2 (both tiles) -> val: scenes stay whole, region A does not
        if r.scene_id == "A_s2":
            records[i] = dataclasses.replace(r, split=Split.VAL)
    write(tmp_path, records)
    with pytest.raises(LeakageGuardError) as excinfo:
        prepare_training_data(config_for(tmp_path, tiny_env))
    assert "region_leakage" in excinfo.value.codes and "scene_leakage" not in excinfo.value.codes
    data = prepare_training_data(config_for(tmp_path, tiny_env, require_region_disjoint=False))      # a scene-level split is allowed explicitly
    assert data.info["n_val_pairs"] == 6


def test_the_guard_cannot_be_bypassed_by_the_config_beyond_the_documented_region_switch(tiny_env: TinyEnv, tmp_path):
    records = list(tiny_env.records)
    records[1] = dataclasses.replace(records[1], split=Split.VAL)
    write(tmp_path, records)
    with pytest.raises(LeakageGuardError):
        prepare_training_data(config_for(tmp_path, tiny_env, require_region_disjoint=False))          # scene leakage is never optional


# ============================================================================== unusable manifests


def test_an_edited_manifest_is_detected_by_its_digest(tiny_env: TinyEnv, tmp_path):
    write(tmp_path, tiny_env.records)
    path = tmp_path / "m.jsonl"
    lines = path.read_text().splitlines()
    lines[2] = lines[2].replace('"split":"train"', '"split":"val"') if '"split":"train"' in lines[2] else lines[2].replace('"split":"val"', '"split":"train"')
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(TrainDataError, match="digest"):
        prepare_training_data(config_for(tmp_path, tiny_env))


def test_a_missing_or_foreign_manifest_is_a_clear_error(tiny_env: TinyEnv, tmp_path):
    with pytest.raises(TrainDataError, match="not found"):
        prepare_training_data(config_for(tmp_path, tiny_env, manifest="absent.jsonl"))
    (tmp_path / "bad.jsonl").write_text('{"hello": 1}\n')
    with pytest.raises(TrainDataError, match="header"):
        prepare_training_data(config_for(tmp_path, tiny_env, manifest="bad.jsonl"))


@pytest.mark.parametrize("keep", [Split.TRAIN, Split.VAL])
def test_both_splits_must_be_present(tiny_env: TinyEnv, tmp_path, keep):
    write(tmp_path, [r for r in tiny_env.records if r.split is keep])
    with pytest.raises(TrainDataError, match="no records"):
        prepare_training_data(config_for(tmp_path, tiny_env))


def test_a_missing_data_file_for_a_used_record_is_reported(tiny_env: TinyEnv, tmp_path):
    (tiny_env.root / TINY / tiny_env.records[0].hr.path).unlink()
    write(tmp_path, tiny_env.records)
    with pytest.raises(TrainDataError, match="missing_file"):
        prepare_training_data(config_for(tmp_path, tiny_env))


def test_requested_bands_must_exist_in_every_record(tiny_env: TinyEnv, tmp_path):
    write(tmp_path, tiny_env.records)
    with pytest.raises(TrainDataError, match="B11"):
        prepare_training_data(config_for(tmp_path, tiny_env, lr_bands=["B04", "B11"], hr_bands=["B04", "B11"]))


def test_bands_are_selected_by_name_and_reach_the_batches(tiny_env: TinyEnv, tmp_path):
    write(tmp_path, tiny_env.records)
    data = prepare_training_data(config_for(tmp_path, tiny_env, lr_bands=["B08", "B04"], hr_bands=["B08", "B04"]))
    assert data.train[0]["lr"].shape[0] == 2 and data.band_names == ("B08", "B04")


def test_an_unknown_dataset_is_refused(tiny_env: TinyEnv, tmp_path):
    odd = [dataclasses.replace(r, dataset="mystery", sample_id=r.sample_id.replace(TINY, "mystery")) for r in tiny_env.records]
    write(tmp_path, odd)
    with pytest.raises(TrainDataError, match="unknown_dataset"):
        prepare_training_data(config_for(tmp_path, tiny_env))


def test_the_dataset_filter_restricts_which_datasets_are_used(tiny_env: TinyEnv, tmp_path):
    write(tmp_path, tiny_env.records)
    with pytest.raises(TrainDataError, match="no records"):
        prepare_training_data(config_for(tmp_path, tiny_env, datasets=["sen2neon"]))
    assert prepare_training_data(config_for(tmp_path, tiny_env, datasets=[TINY])).info["n_train_pairs"] == 4


def test_a_patch_larger_than_the_rasters_is_a_data_error(tiny_env: TinyEnv, tmp_path):
    write(tmp_path, tiny_env.records)
    with pytest.raises(TrainDataError, match="smaller"):
        prepare_training_data(config_for(tmp_path, tiny_env, lr_patch=64))


def test_the_role_guard_holds_even_if_a_profile_is_misconfigured_to_allow_train(tiny_env: TinyEnv, tmp_path, monkeypatch):
    """Second layer: QC trusts the profile's allowed splits. If a profile ever wrongly allowed 'train' for a benchmark role,
    training must still refuse -- the decision does not rest on one table."""
    from frame.data.contract import DatasetRole
    from frame.data.roles import VariantSpec

    broken = tiny_profile(splits=frozenset({Split.TRAIN, Split.VAL}), role=DatasetRole.INDEPENDENT_BENCHMARK)
    monkeypatch.setitem(DATASET_PROFILES, TINY, broken)
    write(tmp_path, tiny_env.records)
    with pytest.raises(LeakageGuardError, match="benchmark/holdout") as excinfo:
        prepare_training_data(config_for(tmp_path, tiny_env))
    assert excinfo.value.codes == ("role_violation",)
