"""Environment isolation: the main FRAME process must never import the Mamba runtime.

Run in a fresh interpreter so imports made by other tests cannot mask a
regression. This is the property that lets the main environment (torch 2.14)
and the Mamba environment (torch 2.6.0+cu118) stay independent.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = ("mamba_ssm", "causal_conv1d", "selective_scan_cuda", "sen2sr")

MODULES_THE_MAIN_PROCESS_LOADS = (
    "frame.models",
    "frame.models.config",
    "frame.models.contract",
    "frame.models.selection",
    "frame.models.protocol",
    "frame.models.mamba_client",
    "frame.api.services.model",
    "frame.api.routes",
    "frame.api.app",
)


def _forbidden_modules_after_importing(modules) -> list:
    code = (
        "import sys\n"
        + "".join(f"import {m}\n" for m in modules)
        + f"bad = sorted(m for m in sys.modules if m.split('.')[0] in {FORBIDDEN!r})\n"
        + "print(','.join(bad))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return [m for m in out.stdout.strip().split(",") if m]


def test_main_process_imports_do_not_pull_in_the_mamba_runtime():
    assert _forbidden_modules_after_importing(MODULES_THE_MAIN_PROCESS_LOADS) == []


def test_selecting_the_mamba_model_id_does_not_import_the_runtime_either():
    """Selection, availability checks and error mapping are all cheap and runtime-free."""
    code = (
        "import sys\n"
        "from frame.models.selection import normalize_model_name\n"
        "from frame.api.services import model as m\n"
        "normalize_model_name('mamba'); m.list_models('cpu')\n"
        f"print(','.join(sorted(x for x in sys.modules if x.split('.')[0] in {FORBIDDEN!r})))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""


def test_worker_only_needs_light_frame_modules():
    """The worker runs in an environment without rasterio/fastapi/pydantic/opensr_test; it
    must not import frame modules that need them."""
    code = (
        "import sys\n"
        "import frame.models.mamba_worker, frame.models.mamba_adapter\n"
        "heavy = ('rasterio','fastapi','pydantic','opensr_test','skimage','scipy','mlstac')\n"
        "print(','.join(sorted(x for x in sys.modules if x.split('.')[0] in heavy)))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""
