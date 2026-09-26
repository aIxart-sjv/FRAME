"""frame.data.splits -- geographic (scene/region) splitting and leakage checks."""

from __future__ import annotations

import dataclasses
import random
from collections import Counter, defaultdict

import pytest

from frame.data.adapters.sen2neon import record_from_row
from frame.data.contract import PairRecord, RasterSpec, Split
from frame.data.errors import RoleViolationError, SplitError, SplitLeakageError
from frame.data.splits import PROXIMITY_CELL_DEG, apply_splits, assert_no_leakage, assign_splits, check_split_integrity, unit_key
from frame.preprocessing import RGBN_BANDS
from frame.tests.data_real_rows import REAL_ROWS


def rec(sample, scene, region, split="train", dataset="tinyset", lon=None, lat=None) -> PairRecord:
    return PairRecord(sample_id=sample, dataset=dataset, scene_id=scene, region_id=region, split=split, pair_type="real_cross_sensor",
                      hr_status="available", scale_factor=4, lon=lon, lat=lat,
                      lr=RasterSpec(band_names=RGBN_BANDS, width=32, height=32), hr=RasterSpec(band_names=RGBN_BANDS, width=128, height=128))


def world(n_regions=12, scenes_per_region=3, tiles_per_scene=4, dataset="tinyset"):
    out = []
    for r in range(n_regions):
        for s in range(scenes_per_region):
            for t in range(tiles_per_scene):
                out.append(rec(f"{dataset}:r{r}s{s}t{t}", f"r{r}_s{s}", f"r{r}", dataset=dataset))
    return out


FRACTIONS = {Split.TRAIN: 0.7, Split.VAL: 0.15, Split.TEST: 0.15}


def regions_by_split(records):
    found = defaultdict(set)
    for r in records:
        found[r.split].add(r.region_id)
    return found


# ------------------------------------------------------------------------------ assignment


def test_assignment_is_deterministic_and_independent_of_input_order():
    records = world()
    a = assign_splits(records, FRACTIONS, seed=7)
    shuffled = list(records)
    random.Random(3).shuffle(shuffled)
    assert a == assign_splits(shuffled, FRACTIONS, seed=7)
    assert a == assign_splits(records, FRACTIONS, seed=7)


def test_a_different_seed_gives_a_different_assignment():
    records = world()
    assert assign_splits(records, FRACTIONS, seed=1) != assign_splits(records, FRACTIONS, seed=2)


def test_whole_regions_move_together_never_patches():
    records = apply_splits(world(), assign_splits(world(), FRACTIONS, seed=0))
    by_region = defaultdict(set)
    for r in records:
        by_region[r.region_id].add(r.split)
    assert all(len(s) == 1 for s in by_region.values())
    sets = regions_by_split(records)
    assert not (sets[Split.TRAIN] & sets[Split.VAL]) and not (sets[Split.TRAIN] & sets[Split.TEST]) and not (sets[Split.VAL] & sets[Split.TEST])


def test_no_scene_is_ever_split_across_two_splits_by_construction():
    for level in ("region", "scene"):
        for seed in range(6):
            recs = apply_splits(world(), assign_splits(world(), FRACTIONS, seed=seed, level=level), level=level)
            # splitting per SCENE is allowed to spread a region over several splits (that is what region-level splitting
            # prevents), so the region rule is only enforced for level="region"; the scene rule holds for both
            errors = [i for i in check_split_integrity(recs, check_spatial_proximity=False, require_region_disjoint=(level == "region"))
                      if i.severity == "error"]
            assert errors == [], (level, seed)
            assert not any(i.code == "scene_leakage" for i in check_split_integrity(recs, check_spatial_proximity=False))


def test_every_requested_split_is_populated_and_sizes_follow_the_fractions():
    records = apply_splits(world(n_regions=20), assign_splits(world(n_regions=20), FRACTIONS, seed=5))
    counts = Counter(r.split for r in records)
    assert all(counts[s] > 0 for s in Split)
    total = sum(counts.values())
    assert counts[Split.TRAIN] / total == pytest.approx(0.7, abs=0.11)


def test_a_small_number_of_units_still_fills_every_split():
    records = world(n_regions=3)
    recs = apply_splits(records, assign_splits(records, FRACTIONS, seed=0))
    assert {r.split for r in recs} == set(Split)


def test_units_are_weighted_by_their_sample_counts():
    big = [rec(f"tinyset:big{i}", "big_s", "big") for i in range(50)]
    small = [rec(f"tinyset:sm{r}_{i}", f"sm{r}_s", f"sm{r}") for r in range(10) for i in range(2)]
    assignment = assign_splits(big + small, {Split.TRAIN: 0.5, Split.VAL: 0.5}, seed=0)
    per_split = Counter()
    for r in big + small:
        per_split[assignment[unit_key(r, "region")]] += 1
    assert abs(per_split[Split.TRAIN] - per_split[Split.VAL]) <= 50  # the 50-sample region cannot be split, but the rest balances it


def test_explicit_assignments_override_the_hash():
    records = world()
    assignment = assign_splits(records, FRACTIONS, seed=0, explicit={"tinyset/r3": Split.TEST, "tinyset/r4": Split.TEST})
    assert assignment["tinyset/r3"] is Split.TEST and assignment["tinyset/r4"] is Split.TEST


def test_an_explicit_assignment_to_an_unknown_unit_is_an_error():
    with pytest.raises(SplitError, match="unknown unit"):
        assign_splits(world(), FRACTIONS, seed=0, explicit={"tinyset/nowhere": Split.TEST})


def test_records_without_a_region_cannot_be_split_by_region():
    records = [rec("tinyset:a", "s", None)]
    with pytest.raises(SplitError, match="no region_id"):
        assign_splits(records, FRACTIONS, seed=0, level="region")
    assert assign_splits(records, FRACTIONS, seed=0, level="scene")  # scene-level still works


def test_invalid_fractions_and_levels_are_rejected():
    with pytest.raises(SplitError):
        assign_splits(world(), {Split.TRAIN: 0.0, Split.VAL: 0.0}, seed=0)
    with pytest.raises(SplitError):
        assign_splits(world(), {Split.TRAIN: -1.0, Split.VAL: 2.0}, seed=0)
    with pytest.raises(SplitError, match="level"):
        assign_splits(world(), FRACTIONS, seed=0, level="patch")


def test_units_of_different_datasets_never_collide():
    a, b = rec("tinyset:1", "s", "SAME", dataset="tinyset"), rec("other:1", "s", "SAME", dataset="other")
    assert unit_key(a, "region") == "tinyset/SAME" and unit_key(b, "region") == "other/SAME"
    assert unit_key(a, "scene") != unit_key(b, "scene")


# ------------------------------------------------------------------------------ the requested leakage test


def test_two_patches_of_the_same_scene_in_different_splits_are_rejected():
    """The deliberate-leakage test: patches p1 and p2 come from the SAME scene but sit in train and test."""
    p1 = rec("tinyset:scene7_patch_a", "scene7", "regionX", split="train")
    p2 = rec("tinyset:scene7_patch_b", "scene7", "regionX", split="test")
    issues = check_split_integrity([p1, p2])
    scene = [i for i in issues if i.code == "scene_leakage"]
    assert len(scene) == 1 and scene[0].severity == "error"
    assert set(scene[0].sample_ids) == {"tinyset:scene7_patch_a", "tinyset:scene7_patch_b"}
    assert "scene7" in scene[0].message and "exactly one split" in scene[0].message
    with pytest.raises(SplitLeakageError, match="scene7"):
        assert_no_leakage([p1, p2])


def test_the_same_scene_in_three_splits_is_one_finding_naming_all_splits():
    recs = [rec("tinyset:a", "s", "r", "train"), rec("tinyset:b", "s", "r", "val"), rec("tinyset:c", "s", "r", "test")]
    (issue,) = [i for i in check_split_integrity(recs) if i.code == "scene_leakage"]
    assert "'test', 'train', 'val'" in issue.message.replace("[", "'").replace("]", "'") or all(s in issue.message for s in ("train", "val", "test"))


def test_a_scene_kept_whole_is_clean():
    recs = [rec("tinyset:a", "s1", "r1", "train"), rec("tinyset:b", "s1", "r1", "train"), rec("tinyset:c", "s2", "r2", "val")]
    assert check_split_integrity(recs) == []
    assert_no_leakage(recs)  # does not raise


def test_region_leakage_is_an_error_by_default_and_a_warning_when_relaxed():
    recs = [rec("tinyset:a", "s1", "r1", "train"), rec("tinyset:b", "s2", "r1", "val")]   # different scenes, SAME region
    (strict,) = [i for i in check_split_integrity(recs) if i.code == "region_leakage"]
    assert strict.severity == "error"
    (relaxed,) = [i for i in check_split_integrity(recs, require_region_disjoint=False) if i.code == "region_leakage"]
    assert relaxed.severity == "warning"
    with pytest.raises(SplitLeakageError):
        assert_no_leakage(recs)
    assert_no_leakage(recs, require_region_disjoint=False)


def test_records_without_a_region_are_only_held_to_the_scene_rule():
    recs = [rec("tinyset:a", "s1", None, "train"), rec("tinyset:b", "s2", None, "val")]
    assert check_split_integrity(recs) == []


def test_the_leakage_check_works_on_any_records_including_patch_index_entries():
    """Patch-level index entries carry the same (dataset, scene, region, split) fields, so they are checked identically."""
    patches = [dataclasses.replace(rec(f"tinyset:s:p{i}", "sceneQ", "regQ", "train" if i < 3 else "val")) for i in range(6)]
    assert any(i.code == "scene_leakage" for i in check_split_integrity(patches))


def test_fuzz_random_assignments_from_the_splitter_are_always_clean_but_random_patch_splits_never_are():
    rng = random.Random(0)
    for seed in range(15):
        records = world(n_regions=rng.randint(3, 10), scenes_per_region=rng.randint(1, 4), tiles_per_scene=rng.randint(1, 5))
        clean = apply_splits(records, assign_splits(records, FRACTIONS, seed=seed))
        assert not [i for i in check_split_integrity(clean, check_spatial_proximity=False) if i.severity == "error"]
    records = world()
    naive = [dataclasses.replace(r, split=rng.choice(list(Split))) for r in records]  # what NOT to do: random patches
    assert any(i.code == "scene_leakage" for i in check_split_integrity(naive))


# ------------------------------------------------------------------------------ dataset roles


def neon(tile="2018_MLBS_3__0_2", split=Split.TEST):
    return dataclasses.replace(record_from_row(REAL_ROWS[tile]), split=split)


def test_a_benchmark_dataset_in_the_training_split_is_a_role_violation():
    r = neon(split=Split.TRAIN)
    (issue,) = [i for i in check_split_integrity([r]) if i.code == "role_violation"]
    assert "independent_benchmark" in issue.message and "'test'" in issue.message
    with pytest.raises(RoleViolationError):
        assert_no_leakage([r])


def test_sen2neon_is_forced_into_test_by_the_splitter_whatever_the_fractions():
    records = [record_from_row(row) for row in REAL_ROWS.values()]
    assignment = assign_splits(records, {Split.TRAIN: 1.0}, seed=0)
    assert set(assignment.values()) == {Split.TEST}
    assert all(r.split is Split.TEST for r in apply_splits(records, assignment))


def test_explicit_assignment_of_a_benchmark_to_train_is_refused():
    records = [record_from_row(REAL_ROWS["2018_MLBS_3__0_2"])]
    with pytest.raises(RoleViolationError):
        assign_splits(records, FRACTIONS, seed=0, explicit={"sen2neon/MLBS": Split.TRAIN})


def test_datasets_with_different_roles_are_split_under_their_own_rules_in_one_call():
    training = world(n_regions=6, dataset="tinyset")  # unknown to the profile table -> unrestricted
    benchmark = [record_from_row(row) for row in REAL_ROWS.values()]
    assignment = assign_splits(training + benchmark, FRACTIONS, seed=1)
    assert {assignment[unit_key(r, "region")] for r in benchmark} == {Split.TEST}
    assert len({assignment[unit_key(r, "region")] for r in training}) == 3


# ------------------------------------------------------------------------------ real-data facts


def test_two_real_tiles_of_one_neon_acquisition_share_a_scene_and_a_region_so_they_cannot_be_split():
    a, b, other = (record_from_row(REAL_ROWS[k]) for k in ("2018_MLBS_3__0_2", "2018_MLBS_3__1_1", "2022_KONZ_7__5_3"))
    assert a.scene_id == b.scene_id == "2018_MLBS_3" and a.region_id == b.region_id == "MLBS"
    assert other.scene_id == "2022_KONZ_7" and other.region_id == "KONZ"
    torn = [a, dataclasses.replace(b, split=Split.VAL)]
    assert any(i.code == "scene_leakage" for i in check_split_integrity(torn, check_roles=False))


# ------------------------------------------------------------------------------ spatial proximity


def test_samples_of_different_splits_on_the_same_ground_are_flagged_as_a_warning():
    """Catches leakage BETWEEN datasets that cover the same place, which scene/region ids cannot see."""
    a = rec("tinyset:a", "s1", "r1", "train", lon=-96.500, lat=39.100)
    b = rec("otherset:b", "sX", "rX", "test", dataset="otherset", lon=-96.510, lat=39.105)
    issues = [i for i in check_split_integrity([a, b], check_roles=False) if i.code == "spatial_proximity_across_splits"]
    assert len(issues) == 1 and issues[0].severity == "warning"
    assert_no_leakage([a, b], check_roles=False)  # a warning does not raise


def test_samples_a_kilometre_apart_across_a_cell_boundary_are_still_flagged():
    """-96.500 / -96.510 fall in different 0.1 degree cells; neighbour cells are compared too."""
    a = rec("tinyset:a", "s1", "r1", "train", lon=-96.4999, lat=39.1001)
    b = rec("otherset:b", "sX", "rX", "test", dataset="otherset", lon=-96.5001, lat=39.0999)
    issues = [i for i in check_split_integrity([a, b], check_roles=False) if i.code == "spatial_proximity_across_splits"]
    assert len(issues) >= 1 and set(issues[0].sample_ids) == {"tinyset:a", "otherset:b"}


def test_distant_samples_and_single_split_cells_are_not_flagged():
    far = [rec("tinyset:a", "s1", "r1", "train", lon=-96.5, lat=39.1), rec("tinyset:b", "s2", "r2", "test", lon=10.0, lat=50.0)]
    assert check_split_integrity(far) == []
    same_split = [rec("tinyset:a", "s1", "r1", "train", lon=-96.5, lat=39.1), rec("tinyset:b", "s2", "r1", "train", lon=-96.5, lat=39.1)]
    assert check_split_integrity(same_split) == []
    assert PROXIMITY_CELL_DEG == 0.1
