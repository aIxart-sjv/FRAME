"""Build experiments/final_readiness/status.json: a small machine-readable statement of what FRAME is and is not, as of the moment it is run.

Everything that can be measured is read, not typed: the git revision and dirty state, the model list and the supported scale and bands (from the code), the smoke records, and the test counts
(parsed from the pytest / vitest output files given on the command line). The caveats and non-claims are the fixed list of docs/CLAIMS.md and are checked by a test.

    sen2sr_venv/bin/python experiments/final_readiness/build_status.py --unit-log U.log --integration-log I.log --frontend-log F.log [--output status.json]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

SCHEMA = "frame-final-readiness/1"

CAVEATS = [
    "The output is an SR-derived product on a 2.5 m pixel grid, not native 2.5 m Sentinel-2 imagery; no 2.5 m ground truth exists for Sentinel-2.",
    "The TTA stability is an uncalibrated reconstruction-variation diagnostic, only weakly associated with error in the tested evidence and not shown to beat image texture.",
    "Super-resolution changed a downstream NDVI decision only slightly, with a dataset-dependent sign: a mixed-to-null result and no consistent advantage over bicubic.",
    "Real-data evidence is North America (SEN2NEON, 30 of 2,269 tiles) and Spain (OpenSR-Test); after the registration gate 11 / 6 / 5 / 4 scene units remain per dataset.",
    "No Indian reference or Indian downstream label exists here; no synthetic Indian truth was made.",
    "No real-data training was run; the training pipeline was exercised on synthetic data only.",
    "The API is a synchronous, in-memory, single-process prototype; no production throughput is claimed.",
    "The working tree is not committed: every run record names its parent revision with dirty=true.",
]
NOT_DEMONSTRATED = [
    "calibrated uncertainty",
    "superior high-error detection from the TTA stability",
    "a consistent downstream advantage over bicubic",
    "universal geographic generalisation",
    "Indian downstream validity",
    "region-level land-cover classification",
    "production-scale throughput",
    "a universally optimal super-resolution model",
]
DEFERRED = [
    "real-data training and fine-tuning (SEN2NAIPv2 part decision recorded, not executed)",
    "Indian reference data and any domain-shift experiment",
    "land-cover and other downstream tasks needing region-level labels",
    "12-band / SWIR products, LDSR-S2, other architectures",
    "STAC acquisition inside the product, asynchronous jobs, authentication, deployment",
]
DATASETS = {
    "SEN2NEON": {"role": "independent benchmark, test only", "real_data_read": "30 of 2,269 tiles (seed 0)", "eligible_after_gate": "12 tiles / 11 scene units", "trained_on": False},
    "OpenSR-Test": {"role": "independent benchmark, test only", "real_data_read": "spot 9, spain_crops 28, spain_urban 20", "eligible_after_gate": "6 / 21 / 13 tiles; 6 / 5 / 4 scene units", "trained_on": False},
    "SEN2NAIPv2": {"role": "primary training (not used by FRAME)", "real_data_read": "130 pairs (format verification and a split check only)", "eligible_after_gate": None, "trained_on": False},
    "SEN2VENuS": {"role": "supplementary training", "real_data_read": "none (format adapter on synthetic files)", "eligible_after_gate": None, "trained_on": False},
    "India holdout": {"role": "domain holdout, test only", "real_data_read": "none (profile only)", "eligible_after_gate": None, "trained_on": False},
}
DOCUMENTS = ["docs/README.md", "docs/RELEASE_READINESS.md", "docs/CLAIMS.md", "docs/REGISTRY.md", "docs/TRACEABILITY.md", "docs/REPRODUCIBILITY.md", "docs/DEMO_RUNBOOK.md"]


def parse_pytest_summary(text: str) -> Dict[str, Optional[int]]:
    """The counts of pytest's last summary line ("2037 passed, 33 deselected, 31 warnings in 420s") or vitest's ("Tests  70 passed (70)")."""
    out: Dict[str, Optional[int]] = {"passed": None, "failed": None, "skipped": None, "deselected": None, "errors": None}
    lines = [line for line in text.splitlines() if re.search(r"\d+ (passed|failed)", line)]
    if not lines:
        return out
    line = lines[-1]
    for key, pattern in (("passed", r"(\d+) passed"), ("failed", r"(\d+) failed"), ("skipped", r"(\d+) skipped"), ("deselected", r"(\d+) deselected"), ("errors", r"(\d+) errors?")):
        m = re.search(pattern, line)
        out[key] = int(m.group(1)) if m else 0
    return out


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _read_log(path: Optional[Path]) -> Optional[str]:
    return Path(path).read_text(encoding="utf-8", errors="replace") if path and Path(path).is_file() else None


def build(unit_log: Optional[Path], integration_log: Optional[Path], frontend_log: Optional[Path]) -> Dict[str, Any]:
    from frame.geospatial import RGBN_SCALE_FACTOR
    from frame.models import config as models_cfg
    from frame.models.selection import MODEL_SPECS
    from frame.api import config as api_config

    status = _git("status", "--porcelain")
    sen2sr_diff = _git("diff", "--", "sen2sr")
    smoke: Dict[str, Any] = {}
    for name in ("toy", "lite", "mamba"):
        path = HERE / f"smoke_{name}" / "smoke_record.json"
        if path.is_file():
            record = json.loads(path.read_text(encoding="utf-8"))
            smoke[name] = {"status": record["status"], "checks": record["n_checks"], "failed": record["n_failed"], "expected_output_shape": record["expected_output_shape"],
                           "produced_output_shape": record["produced_output_shape"], "sr_run_seconds": record["timings_seconds"].get("sr_run_s"),
                           "inference_seconds_reported_by_api": record["timings_seconds"].get("inference_reported_by_api"), "identical_repeat": record["identical_repeat"],
                           "scene_sha256": record["provenance"]["scene_sha256"], "output_sha256": record["provenance"]["output_sha256"], "settings_digest": record["provenance"]["settings_digest"],
                           "record": f"experiments/final_readiness/smoke_{name}/smoke_record.json"}
    tests = {}
    for key, log in (("backend_unit", unit_log), ("backend_integration", integration_log), ("frontend", frontend_log)):
        text = _read_log(log)
        tests[key] = parse_pytest_summary(text) if text is not None else None
    return {
        "schema": SCHEMA,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "project": {"name": "FRAME", "problem_statement": "SIH 2026 PS 26142 (NTRO)", "scope": "orchestration, validation and demonstration layer around the unmodified SEN2SR models"},
        "code": {"git_revision": _git("rev-parse", "HEAD"), "dirty": None if status is None else bool(status), "uncommitted_paths": None if status is None else len(status.splitlines()),
                 "sen2sr_unmodified": sen2sr_diff == "" if sen2sr_diff is not None else None},
        "capabilities": {
            "models": [{"id": spec.id, "label": spec.label, "canonical_name": spec.model_name} for spec in MODEL_SPECS.values()],
            "default_model": models_cfg.DEFAULT_MODEL,
            "scale_factor": RGBN_SCALE_FACTOR,
            "input_resolution_m": 10.0,
            "output_pixel_grid_m": 10.0 / RGBN_SCALE_FACTOR,
            "supported_bands": list(models_cfg.RGBN_BAND_ORDER),
            "model_tile_size": models_cfg.INPUT_SIZE,
            "scene_size": f"any height and width up to {api_config.MAX_INPUT_PIXELS:,} input pixels (tiled)",
            "model_fallback": "none (a model that cannot run is a 503; nothing is substituted)",
            "execution": "synchronous, single process, in-memory job registry",
        },
        "datasets": DATASETS,
        "tests": tests,
        "smoke": smoke,
        "caveats": CAVEATS,
        "not_demonstrated": NOT_DEMONSTRATED,
        "deferred": DEFERRED,
        "documents": DOCUMENTS,
    }


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--unit-log", type=Path)
    parser.add_argument("--integration-log", type=Path)
    parser.add_argument("--frontend-log", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "status.json")
    args = parser.parse_args(argv)
    status = build(args.unit_log, args.integration_log, args.frontend_log)
    args.output.write_text(json.dumps(status, indent=2, sort_keys=False, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
