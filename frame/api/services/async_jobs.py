"""A minimal local background-job abstraction (Requirements 142 Requirement 11, Step I Section 6).

`POST /sr/run` remains fully synchronous and unchanged -- every existing test, the entire
frontend, and every prior phase's contract depend on it, and it is not touched here. This module
adds an ADDITIVE path (`POST /jobs/sr`, `GET /jobs/{async_job_id}`) for a caller that explicitly
wants to submit long-running SR work and poll for it instead of blocking one HTTP request, per
Step I's own instruction: "a minimal local asynchronous/background job abstraction is sufficient
... if a full queue is not already present." This is a `ThreadPoolExecutor` and an in-memory
status dict -- not Celery/Redis, not a durable queue; state is lost on process restart, exactly
like `frame.api.services.storage.JobStore` already is.
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, Optional

from frame.api.errors import AsyncJobNotFoundError
from frame.api.services.storage import SRJobRecord


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AsyncJobRecord:
    async_job_id: str
    status: str  # "queued" | "running" | "completed" | "failed"
    sr_job_id: Optional[str]
    error: Optional[str]
    created_at: str


class AsyncJobRegistry:
    """One instance per running API process (like `JobStore`). Thread-safe: every read/write of
    a record happens under `self._lock`, since the background thread pool and the polling HTTP
    handler run concurrently."""

    def __init__(self, max_workers: int = 2) -> None:
        self._records: Dict[str, AsyncJobRecord] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="frame-async-job")

    def submit(self, work: Callable[[], SRJobRecord]) -> AsyncJobRecord:
        """Queue ``work`` (a zero-argument callable that runs the real, synchronous
        `pipeline.run_sr_job` with everything it needs already bound by the caller) and return
        immediately with a "queued" record. The model callable itself must already be resolved
        by the caller BEFORE calling this -- an unavailable model (e.g. Mamba with no CUDA)
        should fail the HTTP request immediately, not silently inside a background thread."""
        async_job_id = uuid.uuid4().hex
        record = AsyncJobRecord(async_job_id=async_job_id, status="queued", sr_job_id=None, error=None, created_at=_now_iso())
        with self._lock:
            self._records[async_job_id] = record
        self._executor.submit(self._run, async_job_id, work)
        return record

    def _run(self, async_job_id: str, work: Callable[[], SRJobRecord]) -> None:
        with self._lock:
            self._records[async_job_id].status = "running"
        try:
            job = work()
        except Exception as exc:  # noqa: BLE001 -- every frame.*/APIError already carries a user-safe message; never a raw traceback
            with self._lock:
                self._records[async_job_id].status = "failed"
                self._records[async_job_id].error = f"{type(exc).__name__}: {exc}"
            return
        with self._lock:
            self._records[async_job_id].status = "completed"
            self._records[async_job_id].sr_job_id = job.job_id

    def get(self, async_job_id: str) -> AsyncJobRecord:
        with self._lock:
            try:
                return self._records[async_job_id]
            except KeyError:
                raise AsyncJobNotFoundError("No such async job", async_job_id=async_job_id) from None
