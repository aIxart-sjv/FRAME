"""Run one experiment end to end and record it (Phase 4). Main environment only.

    config -> refuse if the output directory already holds a run -> manifest gate -> datasets -> model -> trainer
           -> baselines + untrained reference on the validation split -> train (validate, checkpoint) -> summary.json + README.md

Everything about the run that a reader needs to audit it is written into ``summary.json``: model, dataset (scenes, splits,
manifest digest), seed, config digest, environment (software and hardware), git revision, and the measured results, with the
untrained model and each baseline evaluated on the SAME validation patches through the SAME loop. A failed run leaves a
summary with ``status: failed`` and the error, next to whatever metrics and checkpoints it produced.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from frame.train import checkpoint as ckpt
from frame.train.baselines import build_baselines
from frame.train.config import TrainConfig, resolve_path
from frame.train.data import prepare_training_data
from frame.train.errors import TrainError
from frame.train.models import build_model, count_parameters, parameter_megabytes
from frame.train.record import environment_info, git_info, render_readme, write_summary
from frame.train.trainer import Trainer
from frame.train.validate import evaluate, reference_metrics

REPO_ROOT = Path(__file__).resolve().parents[2]
_TRAINING_KEYS = ("steps", "micro_steps", "epochs_completed", "train_seconds", "mean_step_seconds", "final_train_loss", "device", "peak_vram_mib",
                  "precision", "effective_batch_size", "resumed_from_step", "notes")


def _resolve_resume(resume: Optional[Union[str, Path]], output_dir: Path) -> Optional[Path]:
    if resume is None:
        return None
    if str(resume) == "latest":
        latest = ckpt.latest_checkpoint(output_dir / "checkpoint")
        if latest is None:
            raise TrainError(f"No checkpoint found in {output_dir / 'checkpoint'} to resume from.")
        return latest
    return Path(resume)


def execute(config: TrainConfig, *, base_dir: Optional[Path] = None, resume: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    output_dir = resolve_path(config.output_dir, base_dir)
    resume_path = _resolve_resume(resume, output_dir)

    data = prepare_training_data(config, base_dir=base_dir)                     # the leakage gate: refuses before anything is created
    model = build_model(config.model, seed=config.seed)
    git = git_info(REPO_ROOT)
    trainer = Trainer(config, model, data.train, val_data=data.val, metrics_fn=reference_metrics, manifest_digest=data.manifest_digest,
                      scale=data.scale, band_names=data.band_names, git_revision=git["revision"], output_dir=output_dir)
    if resume_path is None:
        trainer.assert_fresh_output()                                          # before a single file is written
    trainer.prepare()

    def score(candidate, dataset) -> Dict[str, Any]:
        return evaluate(candidate, dataset, trainer.loss_fn, device=trainer.device, scale=data.scale, metrics_fn=reference_metrics, band_names=data.band_names)

    baseline_models = build_baselines(config.baselines, device=trainer.device, scale=data.scale)
    baselines = {name: score(candidate, data.val) for name, candidate in baseline_models.items()}
    previous: Dict[str, Any] = {}
    if resume_path is not None and (output_dir / "summary.json").is_file():
        previous = json.loads((output_dir / "summary.json").read_text())
    initial = previous.get("initial_validation") if resume_path is not None else trainer.validate()

    summary: Dict[str, Any] = {
        "name": config.name, "status": "started",
        "model": {"name": config.model.name, "params": config.model.params, "pretrained": config.model.pretrained,
                  "parameters": {"total": count_parameters(model), "trainable": count_parameters(model, trainable_only=True), "megabytes": round(parameter_megabytes(model), 4)}},
        "dataset": data.info, "manifest_digest": data.manifest_digest, "seed": config.seed, "config_digest": config.digest(),
        "resume_signature": config.resume_signature(), "initial_validation": initial, "baselines": baselines,
        "environment": environment_info(config.device), "git": git,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(config.to_json(), encoding="utf-8")
    write_summary(output_dir, summary)

    if trainer.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(trainer.device)                    # peak VRAM then covers training and in-loop validation only
    try:
        trained = trainer.run(resume=resume_path)
    except Exception as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_summary(output_dir, summary)
        raise

    summary.update(
        status="completed", finished_utc=datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat(),
        training={key: trained[key] for key in _TRAINING_KEYS}, validation=trained["validation"],
        checkpoint={"directory": "checkpoint", "final": ckpt.checkpoint_name(trained["steps"])},
        train_evaluation=({"model": score(trainer.model, data.train_eval),
                           "baselines": {name: score(candidate, data.train_eval) for name, candidate in baseline_models.items()}}
                          if data.train_eval is not None else None),
    )
    summary["model"]["parameters"] = trained["parameters"]
    write_summary(output_dir, summary)
    (output_dir / "README.md").write_text(render_readme(summary, config), encoding="utf-8")
    return summary
