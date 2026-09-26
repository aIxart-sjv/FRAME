"""frame.data.roles -- the dataset roles are the ones the requirements assign, and the layer stays isolated."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from frame.data.contract import DatasetRole, PairType, Split
from frame.data.errors import ContractError
from frame.data.roles import DATASET_PROFILES, STATUS_FIXTURE, STATUS_PLANNED, STATUS_REAL, STATUS_SYNTHETIC, get_profile

REPO = Path(__file__).resolve().parents[2]
REQUIREMENTS = REPO / "docs" / "Requirements 142.txt"


def all_variants():
    return [(p.dataset, v) for p in DATASET_PROFILES.values() for v in p.variants.values()]


def test_the_five_datasets_of_the_requirements_are_profiled():
    assert sorted(set(DATASET_PROFILES) - {"synthetic_smoke"}) == ["india_holdout", "opensr_test", "sen2naipv2", "sen2neon", "sen2venus"]
    assert sorted(DATASET_PROFILES) == ["india_holdout", "opensr_test", "sen2naipv2", "sen2neon", "sen2venus", "synthetic_smoke"]   # + FRAME's own smoke set (Phase 4)
    with pytest.raises(ContractError, match="Unknown dataset"):
        get_profile("nope")


def test_the_roles_and_permitted_splits_follow_the_requirements_tiers():
    """Train: SEN2NAIPv2 synthetic (primary) + SEN2VENuS (supplementary); validate: cross-sensor; final test: SEN2NEON, OpenSR-Test, India."""
    naip, venus = get_profile("sen2naipv2"), get_profile("sen2venus")
    for name in ("unet", "histmatch"):
        v = naip.variant(name)
        assert v.role is DatasetRole.TRAINING_PRIMARY and v.pair_type is PairType.SYNTHETIC and Split.TRAIN in v.allowed_splits and Split.TEST not in v.allowed_splits
    cross = naip.variant("crosssensor")
    assert cross.role is DatasetRole.VALIDATION and cross.pair_type is PairType.REAL_CROSS_SENSOR and cross.allowed_splits == {Split.VAL}
    for v in venus.variants.values():
        assert v.role is DatasetRole.TRAINING_SUPPLEMENTARY and Split.TRAIN in v.allowed_splits and Split.TEST not in v.allowed_splits
    for dataset, role in (("sen2neon", DatasetRole.INDEPENDENT_BENCHMARK), ("opensr_test", DatasetRole.INDEPENDENT_BENCHMARK), ("india_holdout", DatasetRole.DOMAIN_HOLDOUT)):
        for v in get_profile(dataset).variants.values():
            assert v.role is role and v.allowed_splits == {Split.TEST}, (dataset, v.name)


def test_no_benchmark_or_holdout_can_ever_be_training_or_validation_data():
    for dataset, v in all_variants():
        if v.role in (DatasetRole.INDEPENDENT_BENCHMARK, DatasetRole.DOMAIN_HOLDOUT):
            assert not ({Split.TRAIN, Split.VAL} & set(v.allowed_splits)), (dataset, v.name)


def test_the_indian_holdout_is_the_only_profile_that_may_lack_an_hr_reference():
    for dataset, v in all_variants():
        assert (v.pair_type is PairType.INDEPENDENT_HR_REFERENCE) == (dataset == "india_holdout")


def test_scale_sizes_and_pixel_sizes_are_mutually_consistent():
    for dataset, v in all_variants():
        if v.lr_size and v.hr_size:
            assert (v.lr_size[0] * v.scale_factor, v.lr_size[1] * v.scale_factor) == tuple(v.hr_size), (dataset, v.name)
        if v.lr_pixel_size_m and v.hr_pixel_size_m:
            assert v.lr_pixel_size_m / v.hr_pixel_size_m == pytest.approx(v.scale_factor), (dataset, v.name)


def test_the_documented_facts_of_each_dataset():
    naip = get_profile("sen2naipv2")
    assert all((v.lr_size, v.hr_size, v.scale_factor, len(v.lr_bands)) == ((130, 130), (520, 520), 4, 4) for v in naip.variants.values())
    assert naip.license == "CC0-1.0"
    venus = get_profile("sen2venus")
    assert (venus.variant("rgbn_10m").scale_factor, venus.variant("rededge_20m").scale_factor) == (2, 4)
    assert venus.variant("rgbn_10m").lr_bands == ("B02", "B03", "B04", "B08") and "NON-COMMERCIAL" in venus.license
    neon = get_profile("sen2neon").variant("2.5m")
    assert (len(neon.lr_bands), len(neon.hr_bands), neon.lr_size, neon.hr_size, neon.hr_pixel_size_m) == (12, 12, (256, 256), (1024, 1024), 2.5)
    opensr = get_profile("opensr_test").variant("spot")
    assert (len(opensr.lr_bands), len(opensr.hr_bands), opensr.lr_size, opensr.hr_size) == (12, 4, (128, 128), (512, 512))


def test_statuses_are_honest_about_what_was_actually_implemented():
    """Only datasets whose real data was read are 'real'; format-only adapters and the planned holdout must say so."""
    status = {d: p.status for d, p in DATASET_PROFILES.items()}
    assert status == {"sen2neon": STATUS_REAL, "opensr_test": STATUS_REAL, "sen2naipv2": STATUS_FIXTURE, "sen2venus": STATUS_FIXTURE, "india_holdout": STATUS_PLANNED,
                      "synthetic_smoke": STATUS_SYNTHETIC}
    assert "synthetic fixtures only" in STATUS_FIXTURE and "no data" in STATUS_PLANNED


def test_every_profile_states_its_licence_source_geography_units_and_caveats():
    for name, p in DATASET_PROFILES.items():
        assert p.license and p.source and p.scene_unit and p.requirements_ref and p.caveats, name
    assert get_profile("sen2neon").region_unit and "site" in get_profile("sen2neon").region_unit
    assert any("never used for training" in c for c in get_profile("sen2neon").caveats)
    assert any("no HR label is invented" in c for c in get_profile("india_holdout").caveats)


def test_synthetic_variants_are_exactly_the_ones_that_need_a_degradation_record():
    assert {(d, v.name) for d, v in all_variants() if v.pair_type is PairType.SYNTHETIC} == {("sen2naipv2", "unet"), ("sen2naipv2", "histmatch"), ("synthetic_smoke", "default")}


@pytest.mark.skipif(not REQUIREMENTS.is_file(), reason="docs/Requirements 142.txt not present")
def test_the_cited_requirement_line_ranges_actually_discuss_the_dataset():
    """`requirements_ref` points at line ranges of the primary requirements document; the named dataset must appear in them."""
    lines = REQUIREMENTS.read_text(encoding="utf-8", errors="replace").splitlines()
    keywords = {"sen2naipv2": ("sen2naip",), "sen2venus": ("ven",), "sen2neon": ("sen2neon",), "opensr_test": ("opensr",), "india_holdout": ("india",), "synthetic_smoke": ("degrad",)}
    for dataset, profile in DATASET_PROFILES.items():
        ranges = [(int(a), int(b)) for a, b in re.findall(r"(\d+)-(\d+)\)?", profile.requirements_ref) if int(b) > int(a) and int(b) <= len(lines)]
        assert ranges, dataset
        text = "\n".join("\n".join(lines[a - 1 : b]) for a, b in ranges).lower()
        assert any(k in text for k in keywords[dataset]), (dataset, ranges)


# ============================================================================== isolation


def test_importing_the_data_layer_pulls_in_no_heavy_or_unrelated_modules():
    code = (
        "import sys, frame.data, frame.data.adapters;"
        "banned = [m for m in ('opensr_test', 'skimage', 'fastapi', 'sen2sr', 'mamba_ssm', 'tacoreader', 'tacotoolbox', 'huggingface_hub') if m in sys.modules];"
        "print(banned); sys.exit(1 if banned else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr


def test_no_machine_specific_path_or_absolute_data_location_is_committed_in_the_data_layer():
    offenders = []
    for path in sorted((REPO / "frame" / "data").rglob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if re.search(r"/home/\w+|/Users/\w+|C:\\\\Users", line):
                offenders.append(f"{path.relative_to(REPO)}:{number}")
    assert offenders == []


def test_the_data_layer_does_not_touch_the_vendored_upstream_package():
    for path in sorted((REPO / "frame" / "data").rglob("*.py")):
        assert not re.search(r"^\s*(from|import)\s+sen2sr\b", path.read_text(encoding="utf-8"), re.M), path
