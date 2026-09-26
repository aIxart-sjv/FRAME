"""Cached model loading for the FRAME API (Phase 7, extended in Phase 1 for model selection).

Two models are selectable by id:

    "lite"   SEN2SRLite/NonReference_RGBN_x4 -- loaded in-process via `mlstac`
             (the identical pattern every prior phase's experiments use).
    "mamba"  SEN2SR/MambaSR_RGBN_x4 -- served by an isolated worker process in
             the dedicated Mamba environment (frame.models.mamba_client); this
             process never imports the Mamba runtime.

Either way `get_model` returns a plain ``model(x[None]) -> y`` callable, so
`frame.api.services.pipeline` is identical for both. Models are loaded at
most once per (model id, device) per API process via a module-level cache;
that is a Phase 7 requirement ("keep model initialization cached") and matters
most for Mamba, whose worker holds ~0.6 GB of GPU memory.

`ensure_weights` / `load_compiled_model` / `load_mamba` are injectable so this
module is testable without the network, the real weights or a GPU
(frame/tests/test_api_model_service.py).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from frame.api import config
from frame.models import config as models_cfg
from frame.models.errors import ModelError, ModelLoadError, ModelUnavailableError
from frame.models.mamba_client import MambaWorkerClient, check_mamba_availability
from frame.models.selection import MODEL_SPECS, normalize_model_name

_model_cache: Dict[Tuple[str, str], Any] = {}
_cache_lock = threading.Lock()


def _default_ensure_weights(cache_dir: Path) -> Path:
    manifest_path = cache_dir / "mlm.json"
    if manifest_path.exists():
        return cache_dir
    import mlstac

    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    mlstac.download(file=config.MODEL_MANIFEST_URL, output_dir=str(cache_dir))
    return cache_dir


def _default_load_compiled_model(weight_dir: Path, device: str) -> Any:
    import mlstac

    return mlstac.load(str(weight_dir)).compiled_model(device=device)


def _default_load_mamba(device: str) -> Any:
    """Start the isolated SEN2SR-Mamba worker. Raises `ModelUnavailableError`
    (no GPU / runtime / weights) or a worker error; nothing is cached on failure."""
    availability = check_mamba_availability(device)
    if not availability.available:
        raise ModelUnavailableError(availability.message, technical_detail=availability.reason_code)
    client = MambaWorkerClient(device=device)
    client.start()
    return client


def resolve_device(device: str) -> str:
    """"auto" -> cuda if available, else cpu; any other value is returned unchanged."""
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_lite(ensure_weights: Callable[[Path], Path], load_compiled_model: Callable[[Path, str], Any], device: str) -> Any:
    """Fetch (if needed) and load SEN2SR-Lite. A missing or unloadable artifact becomes a named `frame.models` error with a
    user-safe message (503 / 500 at the API) instead of a raw network or library exception; the cause is kept in
    ``technical_detail`` for the server log. Nothing is cached on failure, so the next request retries."""
    try:
        weight_dir = ensure_weights(config.MODEL_WEIGHTS_CACHE_DIR)
    except ModelError:
        raise
    except Exception as exc:  # noqa: BLE001 -- any download / filesystem failure means "the artifact is not available here"
        raise ModelUnavailableError(
            "The SEN2SR-Lite model files are not available on this server and could not be downloaded.",
            technical_detail=f"{type(exc).__name__}: {exc}",
        ) from exc
    try:
        return load_compiled_model(weight_dir, device)
    except ModelError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(
            "The SEN2SR-Lite model files were found but could not be loaded.", technical_detail=f"{type(exc).__name__}: {exc}"
        ) from exc


def get_model(
    device: Optional[str] = None,
    *,
    model_name: str = models_cfg.DEFAULT_MODEL,
    ensure_weights: Callable[[Path], Path] = _default_ensure_weights,
    load_compiled_model: Callable[[Path, str], Any] = _default_load_compiled_model,
    load_mamba: Callable[[str], Any] = _default_load_mamba,
) -> Any:
    """Return the (cached) model callable for ``model_name`` ("lite" or
    "mamba") on ``device``, loading it only on first use.

    Raises `UnknownModelError` for any other id.
    """
    model_id = normalize_model_name(model_name)
    resolved_device = resolve_device(device or config.DEVICE)
    key = (model_id, resolved_device)

    with _cache_lock:  # one loader at a time: two Mamba workers would double its GPU memory
        if key not in _model_cache:
            if model_id == models_cfg.MODEL_MAMBA:
                _model_cache[key] = load_mamba(resolved_device)
            else:
                _model_cache[key] = _load_lite(ensure_weights, load_compiled_model, resolved_device)
        return _model_cache[key]


def list_models(device: Optional[str] = None) -> List[Dict[str, Any]]:
    """Every selectable model with a user-facing availability verdict (does
    not load or start anything)."""
    resolved_device = resolve_device(device or config.DEVICE)
    mamba = check_mamba_availability(resolved_device)
    entries = []
    for model_id, spec in MODEL_SPECS.items():
        available = mamba.available if model_id == models_cfg.MODEL_MAMBA else True
        reason = None if available else mamba.message
        entries.append(
            {"id": model_id, "label": spec.label, "model_name": spec.model_name, "available": available, "reason": reason}
        )
    return entries


def clear_cache() -> None:
    """Drop every cached model (stopping any Mamba worker) -- used by tests, safe to call anytime."""
    with _cache_lock:
        models = list(_model_cache.values())
        _model_cache.clear()
    for model in models:
        close = getattr(model, "close", None)
        if callable(close):
            close()
