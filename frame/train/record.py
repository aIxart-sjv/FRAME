"""What an experiment records about itself (Phase 4): environment, git state, summary.json and README.md.

Requirements grounding (docs/Requirements 142.txt section 47): every experiment records dataset version, scene IDs, model
checkpoint, code commit, random seed, input bands, hardware and software versions. This module supplies the environment and
code-state parts and renders a human-readable README from the machine-readable summary. Stdlib + torch + numpy only.

The README is deliberately modest: it reports measured numbers side by side, says what the data was, and never declares a
winner or presents a synthetic-data result as evidence about real imagery.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from frame.train.config import TrainConfig

_TRACKED_PACKAGES = ("rasterio", "scikit-image", "opensr-test", "safetensors", "mlstac", "mamba-ssm", "causal-conv1d", "triton")
_METRIC_COLUMNS = (("loss.total", "loss"), ("psnr_db", "PSNR (dB)"), ("ssim", "SSIM"), ("rmse", "RMSE"), ("sam_degrees", "SAM (deg)"), ("ergas", "ERGAS"))
SYNTHETIC_DATASETS = ("synthetic_smoke",)


def environment_info(device: str = "auto") -> Dict[str, Any]:
    cuda = torch.cuda.is_available()
    gpu: Optional[Dict[str, Any]] = None
    if cuda:
        props = torch.cuda.get_device_properties(0)
        gpu = {"name": props.name, "total_memory_mib": round(props.total_memory / 2**20), "compute_capability": f"{props.major}.{props.minor}",
               "driver_cuda": torch.version.cuda}
    try:
        ram_gib = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        ram_gib = None
    packages = {}
    for name in _TRACKED_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {
        "python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
        "ram_gib": ram_gib, "torch": str(torch.__version__), "numpy": str(np.__version__), "cuda_available": cuda,
        "cudnn": str(torch.backends.cudnn.version()) if cuda else None, "gpu": gpu, "packages": packages, "requested_device": device,
    }


def git_info(repo: Path) -> Dict[str, Optional[Any]]:
    """The HEAD revision and whether the working tree has uncommitted or untracked changes; None/None outside a repository."""
    def run(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    revision = run("rev-parse", "HEAD")
    if revision is None:
        return {"revision": None, "dirty": None}
    status = run("status", "--porcelain")
    return {"revision": revision, "dirty": None if status is None else bool(status)}


def write_summary(output_dir: Path, summary: Dict[str, Any]) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.json"
    temporary = path.with_name("summary.json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


# ---------------------------------------------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------------------------------------------

def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}" if abs(value) < 100 else f"{value:.2f}"
    return str(value)


def _pick(block: Optional[Dict[str, Any]], dotted: str) -> Any:
    if not block:
        return None
    where, _, key = dotted.rpartition(".")
    source = block.get(where) if where else block.get("metrics")
    return None if source is None else source.get(key)


def _table(rows: Dict[str, Optional[Dict[str, Any]]]) -> str:
    header = "| model | patches | " + " | ".join(label for _, label in _METRIC_COLUMNS) + " |"
    lines = [header, "|" + "---|" * (len(_METRIC_COLUMNS) + 2)]
    for name, block in rows.items():
        cells = [_fmt(_pick(block, key), 3 if key.endswith("db") else 4) for key, _ in _METRIC_COLUMNS]
        lines.append(f"| {name} | {block['n_patches'] if block else 'n/a'} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_readme(summary: Dict[str, Any], config: TrainConfig) -> str:
    data = summary.get("dataset", {})
    training = summary.get("training", {})
    env = summary.get("environment", {})
    gpu = env.get("gpu")
    synthetic = bool(data.get("datasets")) and set(data["datasets"]) <= set(SYNTHETIC_DATASETS)
    model = summary.get("model", {})
    params = model.get("parameters", {})
    git = summary.get("git", {})

    rows: Dict[str, Optional[Dict[str, Any]]] = {}
    if summary.get("initial_validation"):
        rows[f"{config.model.name} - step 0 (before training)"] = summary["initial_validation"]
    if summary.get("validation"):
        rows[f"{config.model.name} - trained, step {training.get('steps', '?')}"] = summary["validation"]
    for name, block in (summary.get("baselines") or {}).items():
        rows[f"baseline: {name}"] = block

    out = [
        f"# Experiment `{summary.get('name', config.name)}`",
        "",
        f"Status: **{summary.get('status', 'unknown')}**. A training smoke experiment from `python -m frame.train`; "
        "numbers are means over the validation patches listed below, measured by this run, and are not benchmark results. No winner is declared.",
        "",
    ]
    if synthetic:
        out += ["> The data are FRAME's synthetic smoke scenes: these results are **not evidence about real** Sentinel-2 imagery or about any model's real-world quality.", ""]
    out += [
        "## Data",
        f"- datasets: {', '.join(data.get('datasets', [])) or 'n/a'}; manifest digest `{summary.get('manifest_digest')}`",
        f"- train: {data.get('n_train_pairs')} pairs from scenes {data.get('train_scenes')}",
        f"- validation: {data.get('n_val_pairs')} pairs from scenes {data.get('val_scenes')}",
        f"- records in other splits (never read): {data.get('ignored_records') or 'none'}",
        f"- bands (by name): {config.data.lr_bands} -> {config.data.hr_bands}; LR train patch {config.data.lr_patch}, validation patch {config.data.val_lr_patch}",
        "",
        "## Model and training",
        f"- model: `{config.model.name}` {config.model.params or ''}; parameters {params.get('total')} ({params.get('trainable')} trainable, {params.get('megabytes')} MiB)",
        f"- optimizer {config.optim.name} lr {config.optim.lr}, scheduler {config.scheduler.name}, loss {config.loss.reconstruction}"
        f" (+ spectral {config.loss.spectral_weight}, consistency {config.loss.consistency_weight})",
        f"- steps {config.steps}, batch {config.batch_size} x accumulation {config.grad_accum}, precision {training.get('precision')}, seed {config.seed}",
        f"- training time {training.get('train_seconds')} s on {training.get('device')}; peak VRAM {training.get('peak_vram_mib')} MiB",
        "",
        "## Validation (deterministic, same patches for every row)",
        _table(rows),
        "",
        "Rows are not comparable in kind: baselines may carry pretrained weights or a hard constraint the trained model does not.",
        "",
    ]
    train_eval = summary.get("train_evaluation")
    if train_eval:
        train_rows: Dict[str, Optional[Dict[str, Any]]] = {f"{config.model.name} - trained, step {training.get('steps', '?')}": train_eval["model"]}
        for name, block in (train_eval.get("baselines") or {}).items():
            train_rows[f"baseline: {name}"] = block
        out += [
            "## Training scenes (over-fit check)",
            "The same evaluation on the scenes the model TRAINED on (no augmentation). A gap to the validation table shows how much of the fit "
            "is memorisation; this is a sanity check that the loop can fit data, not a quality result.",
            _table(train_rows),
            "",
        ]
    out += [
        "## Environment",
        f"- python {env.get('python')}, torch {env.get('torch')}, numpy {env.get('numpy')}, {env.get('platform')}",
        f"- GPU: {gpu['name'] + ', ' + str(gpu['total_memory_mib']) + ' MiB' if gpu else 'none used'}",
        f"- code: git revision `{git.get('revision')}`, working tree dirty: {git.get('dirty')}",
        "",
        "## Reproduce",
        "`python -m frame.train run <this directory>/config.json` (same data, seed and config; checkpoints in `checkpoint/` are not committed).",
        "",
    ]
    return "\n".join(out)
