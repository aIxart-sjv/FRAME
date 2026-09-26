"""experiments/final_readiness/build_status.py -- the machine-readable readiness statement (Phase 8): what is read (not typed), and what may never be said in it."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "experiments" / "final_readiness" / "build_status.py"
COMMITTED = ROOT / "experiments" / "final_readiness" / "status.json"


@pytest.fixture(scope="module")
def bs():
    spec = importlib.util.spec_from_file_location("frame_build_status", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pytest_and_vitest_summaries_are_parsed(bs):
    assert bs.parse_pytest_summary("....\n2037 passed, 33 deselected, 31 warnings in 420.95s (0:07:00)\n") == {"passed": 2037, "failed": 0, "skipped": 0, "deselected": 33, "errors": 0}
    assert bs.parse_pytest_summary("1 failed, 32 passed, 2033 deselected, 3 warnings in 211.70s")["failed"] == 1
    assert bs.parse_pytest_summary(" Test Files  13 passed (13)\n      Tests  70 passed (70)\n")["passed"] == 70
    assert bs.parse_pytest_summary("nothing here") == {"passed": None, "failed": None, "skipped": None, "deselected": None, "errors": None}


def test_the_status_reads_the_code_and_the_repository_instead_of_restating_them(bs, tmp_path):
    status = bs.build(None, None, None)
    assert status["schema"] == "frame-final-readiness/1"
    assert [m["id"] for m in status["capabilities"]["models"]] == ["lite", "mamba"] and status["capabilities"]["default_model"] == "lite"
    assert status["capabilities"]["scale_factor"] == 4 and status["capabilities"]["output_pixel_grid_m"] == 2.5 and status["capabilities"]["supported_bands"] == ["B04", "B03", "B02", "B08"]
    assert "none" in status["capabilities"]["model_fallback"]
    assert isinstance(status["code"]["dirty"], bool) and len(status["code"]["git_revision"]) == 40 and status["code"]["sen2sr_unmodified"] is True
    assert status["tests"] == {"backend_unit": None, "backend_integration": None, "frontend": None}                    # no log, no invented number


def test_test_counts_come_from_the_logs_given(bs, tmp_path):
    unit, integ, front = tmp_path / "u.log", tmp_path / "i.log", tmp_path / "f.log"
    unit.write_text("2 passed, 1 deselected in 1s\n")
    integ.write_text("1 passed, 5 deselected in 1s\n")
    front.write_text("      Tests  7 passed (7)\n")
    tests = bs.build(unit, integ, front)["tests"]
    assert tests["backend_unit"]["passed"] == 2 and tests["backend_integration"]["deselected"] == 5 and tests["frontend"]["passed"] == 7


def test_smoke_records_are_summarised_with_their_hashes(bs):
    smoke = bs.build(None, None, None)["smoke"]
    assert set(smoke) == {"toy", "lite", "mamba"}
    for name, entry in smoke.items():
        assert entry["status"] == "passed" and entry["failed"] == 0 and entry["expected_output_shape"] == entry["produced_output_shape"] == [4, 800, 1200], name
        assert len(entry["scene_sha256"]) == 64 and len(entry["output_sha256"]) == 64
    assert len({e["scene_sha256"] for e in smoke.values()}) == 1                                                        # the same scene for all three


def test_the_caveats_and_non_claims_carry_the_phase_5_to_7_results_unchanged(bs):
    text = " ".join(bs.CAVEATS).lower()
    for phrase in ("uncalibrated", "weakly associated", "mixed-to-null", "no consistent advantage over bicubic", "no indian reference", "no real-data training", "synchronous", "dirty=true"):
        assert phrase in text, phrase
    assert {"calibrated uncertainty", "a consistent downstream advantage over bicubic", "region-level land-cover classification", "indian downstream validity"} <= {x.lower() for x in bs.NOT_DEMONSTRATED}


def test_no_ranking_or_marketing_language_in_the_readiness_statement(bs):
    """The positive statements (caveats, deferred work, datasets) never use it. The non-claims list may NAME a claim in order to deny it ("superior high-error detection")."""
    blob = json.dumps({"c": bs.CAVEATS, "d": bs.DEFERRED, "ds": bs.DATASETS}).lower()
    for word in ("winner", "superior", "state-of-the-art", "state of the art", "best model", "outperform", "production-ready", "real-time"):
        assert word not in blob, word


def test_the_committed_status_is_well_formed_and_records_no_failing_test():
    if not COMMITTED.is_file():
        pytest.skip("status.json has not been generated yet")
    status = json.loads(COMMITTED.read_text(encoding="utf-8"))
    assert status["schema"] == "frame-final-readiness/1" and status["code"]["sen2sr_unmodified"] is True and status["documents"]
    for key, entry in status["tests"].items():
        assert entry is not None and entry["failed"] == 0 and (entry["errors"] or 0) == 0 and entry["passed"] > 0, key
    assert status["caveats"] and status["not_demonstrated"] and status["deferred"]
    assert "/home/" not in COMMITTED.read_text(encoding="utf-8")
