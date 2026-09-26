"""Main-environment handle to the isolated SEN2SR-Mamba worker (Phase 1).

`MambaWorkerClient` is a plain tensor-in / tensor-out callable -- the same
shape as the compiled Lite model -- so the existing pipeline
(``model(x[None]) -> y``, used by frame.uncertainty's TTA ensemble) runs
unchanged. Behind it, a long-lived subprocess in the dedicated Mamba
environment (`frame.models.mamba_worker`) holds the model.

Why a subprocess and not one merged environment: the main FRAME environment
runs torch 2.14; the prebuilt ``mamba-ssm`` CUDA extension in the Mamba
environment is compiled against torch 2.6.0+cu118, and building it from
source against another torch exhausted this machine's RAM. Two
independent interpreters, one narrow tensor contract between them
(frame.models.protocol), keep both stable. Nothing here imports
``mamba_ssm`` -- frame/tests/test_models_isolation.py enforces that.

Threading: FastAPI runs sync routes in a thread pool, so requests may arrive
concurrently; one lock serialises use of the single worker pipe.
"""

from __future__ import annotations

import atexit
import collections
import logging
import os
import select
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Union

import torch

from frame.models import config
from frame.models.contract import validate_input, validate_output
from frame.models.errors import (
    ModelContractError,
    ModelError,
    ModelInferenceError,
    ModelLoadError,
    ModelUnavailableError,
    ModelWorkerError,
)
from frame.models.protocol import decode_tensor, encode_tensor, read_message, send_message

logger = logging.getLogger("frame.models.mamba")

PathLike = Union[str, Path]

# Worker-reported error_type -> exception raised in this process.
_ERROR_TYPES = {
    "ModelUnavailableError": ModelUnavailableError,
    "ModelLoadError": ModelLoadError,
    "ModelContractError": ModelContractError,
    "ModelInferenceError": ModelInferenceError,
}


# ---------------------------------------------------------------------------
# Availability (cheap: never starts a worker)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Availability:
    available: bool
    reason_code: str        # "ok" | "no_cuda" | "runtime_missing" | "weights_missing"
    message: str            # user-safe


def check_mamba_availability(
    device: str = "cuda",
    *,
    python: Optional[PathLike] = None,
    weights_dir: Optional[PathLike] = None,
) -> Availability:
    """Can SEN2SR-Mamba plausibly run here? Checks the GPU, the isolated
    interpreter and the weight files without spawning anything. It cannot
    prove the runtime imports cleanly -- that is what `MambaWorkerClient.start`
    reports."""
    python = Path(python) if python is not None else config.MAMBA_PYTHON
    weights_dir = Path(weights_dir) if weights_dir is not None else config.MAMBA_WEIGHTS_DIR

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return Availability(False, "no_cuda", "SEN2SR-Mamba requires a CUDA-capable GPU, which is not available.")
    if not python.is_file():
        return Availability(False, "runtime_missing", "The SEN2SR-Mamba runtime is not installed on this server.")
    for name in (config.MAMBA_SR_WEIGHTS_FILENAME, config.MAMBA_HARD_CONSTRAINT_FILENAME):
        if not (weights_dir / name).is_file():
            return Availability(False, "weights_missing", "SEN2SR-Mamba model files were not found on this server.")
    return Availability(True, "ok", "Available.")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class MambaWorkerClient:
    """Callable handle to the isolated Mamba worker. Lazily (re)starts it."""

    def __init__(
        self,
        *,
        python: Optional[PathLike] = None,
        weights_dir: Optional[PathLike] = None,
        device: str = "cuda",
        worker_command: Optional[Sequence[str]] = None,
        startup_timeout: Optional[float] = None,
        request_timeout: Optional[float] = None,
    ):
        self._python = str(python if python is not None else config.MAMBA_PYTHON)
        self._weights_dir = str(weights_dir if weights_dir is not None else config.MAMBA_WEIGHTS_DIR)
        self.device = device
        # `worker_command` exists so tests can substitute a stub worker that
        # speaks the same protocol without a GPU.
        self._command: List[str] = list(worker_command) if worker_command is not None else [
            self._python, "-m", config.MAMBA_WORKER_MODULE, "--weights-dir", self._weights_dir, "--device", device,
        ]
        self._startup_timeout = config.MAMBA_STARTUP_TIMEOUT_S if startup_timeout is None else startup_timeout
        self._request_timeout = config.MAMBA_REQUEST_TIMEOUT_S if request_timeout is None else request_timeout

        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._stderr_tail: Deque[str] = collections.deque(maxlen=100)
        self._ready_info: Dict[str, Any] = {}
        self._last_stats: Dict[str, Any] = {}
        atexit.register(self.close)

    # ------------------------------------------------------------- lifecycle

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> Dict[str, Any]:
        """Start the worker if it is not running; return its load report.

        Raises `ModelUnavailableError` / `ModelLoadError` for problems the
        worker reports about the model, `ModelWorkerError` if the process
        itself fails."""
        with self._lock:
            self._ensure_started()
            return dict(self._ready_info)

    def close(self) -> None:
        """Ask the worker to exit; kill it if it does not. Idempotent."""
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        try:
            send_message(proc.stdin, {"op": "shutdown"})
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
            proc.wait()
        finally:
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    stream.close()
                except Exception:
                    pass

    def __enter__(self) -> "MambaWorkerClient":
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # -------------------------------------------------------------- inference

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        validate_input(x)  # fail in this process, before any IPC
        header, payload = encode_tensor(x)

        with self._lock:
            self._ensure_started()
            proc = self._proc
            try:
                send_message(proc.stdin, {"op": "infer", **header}, payload)
            except (BrokenPipeError, OSError) as exc:
                self._discard_worker()
                raise ModelWorkerError(
                    "The SEN2SR-Mamba worker is not responding.", technical_detail=self._diagnostics(str(exc))
                ) from exc
            reply, reply_payload = self._read_reply(self._request_timeout)

        if reply.get("type") == "error":
            raise self._exception_from_error(reply)
        if reply.get("type") != "result":
            raise ModelWorkerError("The SEN2SR-Mamba worker sent an unexpected reply.", technical_detail=str(reply)[:500])

        try:
            y = decode_tensor(reply, reply_payload)
        except (ValueError, KeyError) as exc:
            raise ModelWorkerError("The SEN2SR-Mamba worker sent a malformed result.", technical_detail=str(exc)) from exc
        validate_output(y, input_shape=tuple(x.shape))
        self._last_stats = {
            "inference_seconds": reply.get("inference_seconds"),
            "peak_memory_mib": reply.get("peak_memory_mib"),
        }
        return y.to(x.device)

    def describe(self) -> Dict[str, Any]:
        """Load report of the running worker plus the last request's timing."""
        return {**self._ready_info, "isolated_worker": True, "last_request": dict(self._last_stats)}

    # ---------------------------------------------------------------- internals

    def _ensure_started(self) -> None:
        if self.is_running:
            return
        self._discard_worker()

        try:
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(config.REPO_ROOT), env.get("PYTHONPATH", "")]))
            env["PYTHONUNBUFFERED"] = "1"
            for inherited in ("PYTHONHOME", "VIRTUAL_ENV"):
                env.pop(inherited, None)
            self._proc = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                cwd=str(config.REPO_ROOT),
                env=env,
            )
        except OSError as exc:
            raise ModelUnavailableError(
                "The SEN2SR-Mamba runtime could not be started on this server.", technical_detail=str(exc)
            ) from exc

        threading.Thread(target=self._drain_stderr, args=(self._proc,), daemon=True).start()

        try:
            reply, _ = self._read_reply(self._startup_timeout)
        except ModelWorkerError:
            self._discard_worker()
            raise
        if reply.get("type") != "ready":
            exc = self._exception_from_error(reply) if reply.get("type") == "error" else ModelWorkerError(
                "The SEN2SR-Mamba worker sent an unexpected startup reply.", technical_detail=str(reply)[:500]
            )
            self._discard_worker()
            raise exc
        self._ready_info = {k: v for k, v in reply.items() if k not in ("type", "payload_bytes")}
        logger.info("SEN2SR-Mamba worker ready (pid=%s, load %.2fs)", self._ready_info.get("worker_pid"),
                    self._ready_info.get("load_seconds", float("nan")))

    def _discard_worker(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        try:
            for raw in iter(proc.stderr.readline, b""):
                self._stderr_tail.append(raw.decode("utf-8", "replace").rstrip())
        except (OSError, ValueError):
            pass

    def _diagnostics(self, extra: str = "") -> str:
        code = self._proc.poll() if self._proc is not None else None
        tail = "\n".join(self._stderr_tail)
        return f"{extra} | exit code: {code} | worker stderr (tail):\n{tail}"

    def _read_reply(self, timeout: float):
        """Read one message with a deadline; a dead/hung worker raises `ModelWorkerError`."""
        proc = self._proc
        fd = proc.stdout.fileno()
        deadline = time.monotonic() + timeout

        def read_exact(n: int) -> bytes:
            chunks: List[bytes] = []
            remaining = n
            while remaining:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError
                ready, _, _ = select.select([fd], [], [], min(left, 0.25))
                if not ready:
                    if proc.poll() is not None:  # died without closing its pipe yet
                        ready, _, _ = select.select([fd], [], [], 0)
                        if not ready:
                            return b"".join(chunks)
                    continue
                chunk = os.read(fd, remaining)
                if not chunk:
                    return b"".join(chunks)
                chunks.append(chunk)
                remaining -= len(chunk)
            return b"".join(chunks)

        try:
            return read_message(read_exact)
        except TimeoutError:
            self._discard_worker()
            raise ModelWorkerError(
                "The SEN2SR-Mamba worker timed out.", technical_detail=self._diagnostics(f"no reply within {timeout}s")
            ) from None
        except (EOFError, ValueError) as exc:
            try:  # EOF can arrive before the process is reaped; the exit code is the best crash clue
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            time.sleep(0.1)  # let the stderr drain thread catch the last lines
            detail = self._diagnostics(str(exc))
            self._discard_worker()
            raise ModelWorkerError("The SEN2SR-Mamba worker stopped unexpectedly.", technical_detail=detail) from exc

    @staticmethod
    def _exception_from_error(reply: Dict[str, Any]) -> ModelError:
        exc_type = _ERROR_TYPES.get(reply.get("error_type", ""), ModelInferenceError)
        return exc_type(reply.get("message") or "SEN2SR-Mamba failed.", technical_detail=reply.get("technical_detail", ""))
