"""Isolated SEN2SR-Mamba worker (Phase 1). Runs in the **Mamba environment**.

    <mamba-env python> -m frame.models.mamba_worker --weights-dir DIR [--device cuda]

A long-lived process that loads the model once, then serves requests over its
stdin/stdout using the framed protocol in frame.models.protocol:

    worker -> client   {"type": "ready", ...load report...}            (once, after loading)
                       {"type": "error", "phase": "startup", ...}      (then exit code 2)
    client -> worker   {"op": "infer", "shape": [...], "dtype": "float32"} + tensor bytes
    worker -> client   {"type": "result", "shape": [...], "inference_seconds": .., "peak_memory_mib": ..} + tensor bytes
                       {"type": "error", "error_type": .., "message": ..}    (worker stays up)
    client -> worker   {"op": "ping"}  -> {"type": "pong"}
    client -> worker   {"op": "shutdown"} -> {"type": "bye"}, exit 0

The protocol travels on a private duplicate of the original stdout; file
descriptor 1 (and ``sys.stdout``) are re-pointed at stderr immediately, so a
library that prints cannot corrupt a frame. The worker exits cleanly when its
stdin closes, i.e. when the FRAME process that spawned it goes away.

Import discipline: only frame.models.{config,contract,errors,protocol,
mamba_adapter} plus torch/numpy -- never the API, rasterio, or anything else
absent from the Mamba environment.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from typing import Any, BinaryIO, Dict, Optional

import torch

from frame.models import config
from frame.models.errors import ModelContractError, ModelError
from frame.models.protocol import decode_tensor, encode_tensor, read_message, send_message


def _error_header(exc: BaseException, *, phase: str) -> Dict[str, Any]:
    is_model_error = isinstance(exc, ModelError)
    return {
        "type": "error",
        "phase": phase,
        "error_type": type(exc).__name__ if is_model_error else "InternalError",
        "message": str(exc) if is_model_error else "The SEN2SR-Mamba worker hit an unexpected error.",
        "technical_detail": getattr(exc, "technical_detail", "") or (
            "" if is_model_error else f"{type(exc).__name__}: {exc}"
        ),
    }


def _ready_header(model: Any) -> Dict[str, Any]:
    info = dict(model.describe())
    info.update(
        {
            "type": "ready",
            "worker_python": sys.version.split()[0],
            "worker_torch": torch.__version__,
            "worker_cuda": torch.version.cuda,
            "worker_gpu": torch.cuda.get_device_name(model.device),
            "worker_pid": os.getpid(),
        }
    )
    return info


def _handle_infer(model: Any, header: Dict[str, Any], payload: bytes, out: BinaryIO) -> None:
    try:
        x = decode_tensor(header, payload)
    except (ValueError, KeyError) as exc:
        send_message(out, _error_header(ModelContractError(f"Malformed tensor payload: {exc}"), phase="infer"))
        return

    try:
        model.reset_peak_memory()
        started = time.perf_counter()
        y = model(x)  # validate_output inside forces the GPU sync, so this timing is wall-clock complete
        elapsed = time.perf_counter() - started
        tensor_header, tensor_bytes = encode_tensor(y)
    except ModelError as exc:
        send_message(out, _error_header(exc, phase="infer"))
        return
    except Exception as exc:  # keep serving; the client decides whether to give up
        traceback.print_exc(file=sys.stderr)
        send_message(out, _error_header(exc, phase="infer"))
        return

    send_message(
        out,
        {
            "type": "result",
            **tensor_header,
            "inference_seconds": round(elapsed, 4),
            "peak_memory_mib": round(model.peak_memory_mib(), 1),
        },
        tensor_bytes,
    )


def serve(model: Any, stdin: BinaryIO, out: BinaryIO) -> int:
    """Request loop. Returns the process exit code."""

    def read_exact(n: int) -> bytes:
        return stdin.read(n)

    while True:
        try:
            header, payload = read_message(read_exact)
        except EOFError:
            return 0  # parent closed the pipe
        except ValueError as exc:
            send_message(out, _error_header(ModelContractError(f"Malformed request: {exc}"), phase="protocol"))
            return 3

        op = header.get("op")
        if op == "infer":
            _handle_infer(model, header, payload, out)
        elif op == "ping":
            send_message(out, {"type": "pong"})
        elif op == "shutdown":
            send_message(out, {"type": "bye"})
            return 0
        else:
            send_message(out, _error_header(ModelContractError(f"Unknown operation {op!r}."), phase="protocol"))


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Isolated SEN2SR-Mamba RGBN worker.")
    parser.add_argument("--weights-dir", default=str(config.MAMBA_WEIGHTS_DIR))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    # Claim the real stdout for the protocol, then point fd 1 / sys.stdout at stderr.
    out = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    stdin = sys.stdin.buffer

    from frame.models.mamba_adapter import MambaRGBNModel

    try:
        model = MambaRGBNModel.load(args.weights_dir, args.device)
    except Exception as exc:  # startup failure is reported to the client, never a silent crash
        if not isinstance(exc, ModelError):
            traceback.print_exc(file=sys.stderr)
        send_message(out, _error_header(exc, phase="startup"))
        return 2

    send_message(out, _ready_header(model))
    return serve(model, stdin, out)


if __name__ == "__main__":
    sys.exit(main())
