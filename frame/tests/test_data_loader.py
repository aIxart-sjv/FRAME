"""frame.data.loader -- PairedPatchDataset and collate_pairs."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from frame.data.adapters.geotiff import read_geotiff_pair
from frame.data.contract import HRStatus, PairType
from frame.data.errors import ContractError, PatchError
from frame.data.loader import PairedPatchDataset, collate_pairs, dispatching_loader
from frame.preprocessing import RGBN_BANDS
from frame.tests.data_fixtures import (  # noqa: F401
    TINY,
    TinyAdapter,
    TinyEnv,
    block_mean,
    make_record,
    tiny_env,
    tiny_profile_installed,
    write_pair_files,
)

PATCH = 16  # LR px; the fixture rasters are 32x32 LR / 128x128 HR


def dataset(env: TinyEnv, **kw) -> PairedPatchDataset:
    kw.setdefault("lr_patch", PATCH)
    return PairedPatchDataset(env.records, env.adapter().load_pair, **kw)


def aligned(item, scale=4) -> bool:
    """The fixture's LR is the exact block mean of its HR, so an aligned patch averages back onto its LR."""
    return bool(np.allclose(block_mean(item["hr"].numpy(), scale), item["lr"].numpy(), atol=2e-4))


def test_grid_mode_enumerates_a_dense_deterministic_index(tiny_env: TinyEnv):
    ds = dataset(tiny_env, mode="grid")
    assert len(ds) == 8 * 4                                       # 32 = 2 x 16 -> a 2x2 grid per pair
    a, b = ds[5], ds[5]
    assert torch.equal(a["lr"], b["lr"]) and a["metadata"] == b["metadata"]
    coords = {(ds[i]["metadata"]["sample_id"], ds[i]["metadata"]["patch"]["lr_row"], ds[i]["metadata"]["patch"]["lr_col"]) for i in range(len(ds))}
    assert len(coords) == len(ds)                                 # every patch distinct


def test_items_have_the_documented_shapes_and_metadata(tiny_env: TinyEnv):
    item = dataset(tiny_env, mode="grid")[0]
    assert item["lr"].shape == (4, PATCH, PATCH) and item["hr"].shape == (4, 4 * PATCH, 4 * PATCH)
    assert item["lr"].dtype == item["hr"].dtype == torch.float32
    m = item["metadata"]
    assert m["dataset"] == TINY and m["scale_factor"] == 4 and m["lr_bands"] == list(RGBN_BANDS) and m["split"] in ("train", "val")
    assert m["patch"]["hr_row"] == 4 * m["patch"]["lr_row"] and m["patch"]["hr_height"] == 4 * m["patch"]["lr_height"]
    assert m["patch"]["hr_col"] == 4 * m["patch"]["lr_col"] and m["patch"]["hr_width"] == 4 * m["patch"]["lr_width"] and m["patch"]["scale"] == 4
    assert m["valid_fraction"] == 1.0 and m["augmentation"] == "identity" and m["epoch"] == 0 and m["sample_id"] in m["patch_id"]


def test_every_patch_the_loader_yields_is_spatially_aligned(tiny_env: TinyEnv):
    """The pair's core invariant survives the whole loader path, in both modes."""
    for ds in (dataset(tiny_env, mode="grid"), dataset(tiny_env, mode="random", patches_per_pair=3, seed=4)):
        assert all(aligned(ds[i]) for i in range(len(ds)))


def test_random_mode_is_reproducible_by_seed_and_index(tiny_env: TinyEnv):
    a, b = dataset(tiny_env, mode="random", seed=7, patches_per_pair=2), dataset(tiny_env, mode="random", seed=7, patches_per_pair=2)
    assert len(a) == 16
    for i in range(len(a)):
        assert torch.equal(a[i]["lr"], b[i]["lr"]) and a[i]["metadata"]["patch"] == b[i]["metadata"]["patch"]
    other = dataset(tiny_env, mode="random", seed=8, patches_per_pair=2)
    assert any(a[i]["metadata"]["patch"] != other[i]["metadata"]["patch"] for i in range(len(a)))


def test_the_epoch_changes_the_draws_and_is_recorded(tiny_env: TinyEnv):
    ds = dataset(tiny_env, mode="random", seed=1, patches_per_pair=4)
    first = [ds[i]["metadata"]["patch"] for i in range(len(ds))]
    ds.set_epoch(1)
    second = [ds[i]["metadata"] for i in range(len(ds))]
    assert first != [m["patch"] for m in second] and {m["epoch"] for m in second} == {1}
    ds.set_epoch(0)
    assert [ds[i]["metadata"]["patch"] for i in range(len(ds))] == first     # going back gives the same draws


def test_records_are_ordered_by_sample_id_not_by_input_order(tiny_env: TinyEnv):
    forward = PairedPatchDataset(tiny_env.records, tiny_env.adapter().load_pair, lr_patch=PATCH, mode="grid")
    backward = PairedPatchDataset(list(reversed(tiny_env.records)), tiny_env.adapter().load_pair, lr_patch=PATCH, mode="grid")
    assert [forward[i]["metadata"]["patch_id"] for i in range(len(forward))] == [backward[i]["metadata"]["patch_id"] for i in range(len(backward))]


def test_a_torch_dataloader_batches_pairs_with_their_metadata(tiny_env: TinyEnv):
    ds = dataset(tiny_env, mode="random", seed=0, patches_per_pair=2)
    batch = next(iter(DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_pairs)))
    assert batch["lr"].shape == (4, 4, PATCH, PATCH) and batch["hr"].shape == (4, 4, 4 * PATCH, 4 * PATCH)
    assert len(batch["metadata"]) == 4 and batch["metadata"][0]["dataset"] == TINY


def test_a_dataloader_with_workers_gives_the_same_data_as_without(tiny_env: TinyEnv):
    ds = dataset(tiny_env, mode="grid")
    single = [b["lr"] for b in DataLoader(ds, batch_size=8, collate_fn=collate_pairs, num_workers=0)]
    multi = [b["lr"] for b in DataLoader(ds, batch_size=8, collate_fn=collate_pairs, num_workers=2)]
    assert len(single) == len(multi) and all(torch.equal(x, y) for x, y in zip(single, multi))


def test_augmentation_keeps_lr_and_hr_aligned_and_says_what_it_did(tiny_env: TinyEnv):
    ds = dataset(tiny_env, mode="random", seed=3, patches_per_pair=6, augment=True)
    names = set()
    for i in range(len(ds)):
        item = ds[i]
        assert aligned(item), item["metadata"]["augmentation"]
        names.add(item["metadata"]["augmentation"])
    assert len(names) > 1 and names <= {"identity", "hflip", "vflip", "rot90", "rot180", "rot270"}


def test_augmentation_is_reproducible(tiny_env: TinyEnv):
    a, b = (dataset(tiny_env, mode="random", seed=3, augment=True) for _ in range(2))
    assert all(torch.equal(a[i]["hr"], b[i]["hr"]) and a[i]["metadata"]["augmentation"] == b[i]["metadata"]["augmentation"] for i in range(len(a)))


def test_records_with_different_scales_cannot_be_batched_together(tiny_env: TinyEnv, tmp_path):
    lr, hr = write_pair_files(tmp_path, TINY, "lr/x2.tif", "hr/x2.tif", scale=2, seed=9)
    odd = make_record("tinyset:x2", scene="x2", region="Z", lr=lr, hr=hr, scale=2)
    with pytest.raises(ContractError) as excinfo:
        PairedPatchDataset(tiny_env.records + [odd], tiny_env.adapter().load_pair, lr_patch=PATCH)
    assert excinfo.value.code == "inconsistent_scale"


def test_a_record_without_hr_is_refused(tiny_env: TinyEnv):
    lr_only = dataclasses.replace(tiny_env.records[0], hr=None, hr_status=HRStatus.UNKNOWN, pair_type=PairType.INDEPENDENT_HR_REFERENCE,
                                  sample_id="tinyset:lr_only")
    with pytest.raises(ContractError) as excinfo:
        PairedPatchDataset([lr_only], tiny_env.adapter().load_pair, lr_patch=PATCH)
    assert excinfo.value.code == "missing_hr"


def test_random_mode_rejects_rasters_smaller_than_the_patch(tiny_env: TinyEnv):
    with pytest.raises(PatchError, match="smaller"):
        PairedPatchDataset(tiny_env.records, tiny_env.adapter().load_pair, lr_patch=64, mode="random")


def test_bad_arguments_are_clear_errors(tiny_env: TinyEnv):
    with pytest.raises(ContractError) as excinfo:
        dataset(tiny_env, mode="bogus")
    assert excinfo.value.code == "invalid_mode"
    with pytest.raises(ContractError) as excinfo:
        PairedPatchDataset([], tiny_env.adapter().load_pair)
    assert excinfo.value.code == "empty_dataset"
    with pytest.raises(IndexError):
        dataset(tiny_env, mode="grid")[10_000]


def test_channels_are_selected_by_name(tiny_env: TinyEnv):
    adapter = tiny_env.adapter()
    load = dispatching_loader({TINY: adapter}, lr_bands=["B08", "B04"], hr_bands=["B08", "B04"])
    ds = PairedPatchDataset(tiny_env.records, load, lr_patch=PATCH, mode="grid")
    item = ds[0]
    assert item["lr"].shape[0] == 2 and item["hr"].shape[0] == 2 and item["metadata"]["lr_bands"] == ["B08", "B04"]
    full = PairedPatchDataset(tiny_env.records, adapter.load_pair, lr_patch=PATCH, mode="grid")[0]
    assert torch.equal(item["lr"][0], full["lr"][3]) and torch.equal(item["lr"][1], full["lr"][0])


def test_a_missing_adapter_is_a_clear_error(tiny_env: TinyEnv):
    load = dispatching_loader({})
    with pytest.raises(ContractError) as excinfo:
        load(tiny_env.records[0])
    assert excinfo.value.code == "unknown_dataset"


def test_mostly_nodata_random_patches_are_redrawn_and_grid_patches_report_their_valid_fraction(tmp_path, tiny_profile_installed):
    # left half of the LR (and the matching HR area) is nodata
    lr, hr = write_pair_files(tmp_path, TINY, "lr/n.tif", "hr/n.tif", lr_nodata=65535, hr_nodata=0, nodata_box=(0, 32, 0, 16), seed=2)
    record = make_record("tinyset:n", scene="n", region="N", lr=lr, hr=hr)
    adapter = TinyAdapter([record], data_root=tmp_path)
    grid = PairedPatchDataset([record], adapter.load_pair, lr_patch=PATCH, mode="grid")
    fractions = sorted(grid[i]["metadata"]["valid_fraction"] for i in range(len(grid)))
    assert fractions == [0.0, 0.0, 1.0, 1.0]                      # nodata is reported, never hidden or repaired
    rand = PairedPatchDataset([record], adapter.load_pair, lr_patch=PATCH, mode="random", patches_per_pair=20, seed=1, min_valid_fraction=0.9, max_attempts=32)
    assert all(rand[i]["metadata"]["valid_fraction"] >= 0.9 for i in range(len(rand)))


def test_a_pair_with_no_valid_pixels_at_all_is_an_error_not_silent_zeros(tmp_path, tiny_profile_installed):
    lr, hr = write_pair_files(tmp_path, TINY, "lr/z.tif", "hr/z.tif", lr_nodata=65535, hr_nodata=0, nodata_box=(0, 32, 0, 32), seed=2)
    record = make_record("tinyset:z", scene="z", region="Z", lr=lr, hr=hr)
    ds = PairedPatchDataset([record], TinyAdapter([record], data_root=tmp_path).load_pair, lr_patch=PATCH, mode="random", max_attempts=4)
    with pytest.raises(PatchError, match="no patch with any valid pixels"):
        ds[0]


def test_the_pair_cache_is_bounded_and_reads_each_pair_once_per_visit(tiny_env: TinyEnv):
    calls = []

    def counting(record):
        calls.append(record.sample_id)
        return read_geotiff_pair(record, tiny_env.root / TINY)

    ds = PairedPatchDataset(tiny_env.records, counting, lr_patch=PATCH, mode="grid", cache_size=1)
    for i in range(4):
        ds[i]                       # the four patches of the first pair
    assert len(calls) == 1
    ds[4]
    assert len(calls) == 2 and len(ds._cache) == 1


# ============================================================================== validity masks (Phase 4 needs them for the loss)


def test_masks_are_opt_in_so_existing_callers_see_the_same_items(tiny_env: TinyEnv):
    item = dataset(tiny_env, mode="grid")[0]
    assert set(item) == {"lr", "hr", "metadata"}


def test_masks_come_with_the_patch_and_have_the_patch_shapes(tiny_env: TinyEnv):
    item = dataset(tiny_env, mode="grid", return_masks=True)[0]
    assert item["lr_mask"].shape == (PATCH, PATCH) and item["hr_mask"].shape == (4 * PATCH, 4 * PATCH)
    assert item["lr_mask"].dtype == item["hr_mask"].dtype == torch.bool and bool(item["lr_mask"].all()) and bool(item["hr_mask"].all())


def nodata_record(tmp_path):
    lr, hr = write_pair_files(tmp_path, TINY, "lr/m.tif", "hr/m.tif", lr_nodata=65535, hr_nodata=0, nodata_box=(0, 32, 0, 12), seed=3)
    record = make_record("tinyset:m", scene="m", region="M", lr=lr, hr=hr)
    return record, TinyAdapter([record], data_root=tmp_path)


def test_masks_mark_the_nodata_pixels_that_the_tensors_zero(tmp_path, tiny_profile_installed):
    record, adapter = nodata_record(tmp_path)
    ds = PairedPatchDataset([record], adapter.load_pair, lr_patch=PATCH, mode="grid", return_masks=True)
    for i in range(len(ds)):
        item = ds[i]
        assert float(item["hr"][:, ~item["hr_mask"]].abs().sum()) == 0.0 and float(item["lr"][:, ~item["lr_mask"]].abs().sum()) == 0.0
    assert any(not bool(ds[i]["hr_mask"].all()) for i in range(len(ds)))          # the fixture really has masked pixels


def test_augmentation_moves_the_masks_exactly_like_the_pixels(tmp_path, tiny_profile_installed):
    """If a mask were not flipped/rotated with its image, zeroed nodata pixels would sit under mask == True."""
    record, adapter = nodata_record(tmp_path)
    ds = PairedPatchDataset([record], adapter.load_pair, lr_patch=PATCH, mode="random", patches_per_pair=24, seed=2, augment=True,
                            min_valid_fraction=0.01, return_masks=True)
    names = set()
    for i in range(len(ds)):
        item = ds[i]
        names.add(item["metadata"]["augmentation"])
        assert float(item["hr"][:, ~item["hr_mask"]].abs().sum()) == 0.0 and float(item["lr"][:, ~item["lr_mask"]].abs().sum()) == 0.0
        assert bool((item["hr"].abs().sum(0) > 0)[item["hr_mask"]].all())          # every valid pixel is a real (non-zero) observation
    assert len(names) > 2


def test_collate_stacks_masks_when_present_and_ignores_them_otherwise(tiny_env: TinyEnv):
    with_masks = collate_pairs([dataset(tiny_env, mode="grid", return_masks=True)[i] for i in range(3)])
    assert with_masks["lr_mask"].shape == (3, PATCH, PATCH) and with_masks["hr_mask"].shape == (3, 4 * PATCH, 4 * PATCH)
    assert "lr_mask" not in collate_pairs([dataset(tiny_env, mode="grid")[i] for i in range(3)])
