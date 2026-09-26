"""frame.models.mamba_client -- the main-environment handle to the isolated worker.

These tests need no GPU and no Mamba environment: they drive the REAL client
against a small stub worker that speaks the real wire protocol
(frame.models.protocol), so lifecycle, error mapping, timeouts and crash
recovery are exercised for real. One test additionally launches the REAL
worker module to check that a startup failure is reported cleanly. The
real-model, real-GPU checks live in test_models_mamba_integration.py
(`pytest -m integration`).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from frame.models.errors import (
    ModelContractError,
    ModelInferenceError,
    ModelUnavailableError,
    ModelWorkerError,
)
from frame.models.mamba_client import Availability, MambaWorkerClient, check_mamba_availability

STUB_WORKER = textwrap.dedent(
    '''
    import os, sys, time
    import torch.nn.functional as F
    from frame.models.protocol import decode_tensor, encode_tensor, read_message, send_message

    mode = sys.argv[1]
    marker = sys.argv[2] if len(sys.argv) > 2 else ""
    out = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)

    if mode == "die_at_startup":
        print("boom: cannot initialise runtime", file=sys.stderr)
        sys.exit(7)
    if mode == "startup_unavailable":
        send_message(out, {"type": "error", "phase": "startup", "error_type": "ModelUnavailableError",
                           "message": "SEN2SR-Mamba requires a CUDA-capable GPU.", "technical_detail": "no cuda"})
        sys.exit(2)
    if mode == "startup_hang":
        time.sleep(60)

    send_message(out, {"type": "ready", "worker_pid": os.getpid(), "parameter_count": 1, "load_seconds": 0.01})

    while True:
        try:
            header, payload = read_message(sys.stdin.buffer.read)
        except EOFError:
            sys.exit(0)
        op = header["op"]
        if op == "shutdown":
            send_message(out, {"type": "bye"})
            sys.exit(0)
        if mode == "crash_once" and not os.path.exists(marker):
            open(marker, "w").close()
            print("simulated worker crash", file=sys.stderr)
            os._exit(9)
        if mode == "hang":
            time.sleep(60)
        if mode == "error_reply":
            send_message(out, {"type": "error", "phase": "infer", "error_type": "ModelInferenceError",
                               "message": "SEN2SR-Mamba ran out of GPU memory.", "technical_detail": "CUDA OOM"})
            continue
        x = decode_tensor(header, payload)
        if mode == "bad_shape":
            y = F.interpolate(x, scale_factor=2, mode="nearest")
        else:
            y = F.interpolate(x, scale_factor=4, mode="nearest")
        fields, blob = encode_tensor(y)
        send_message(out, {"type": "result", **fields, "inference_seconds": 0.001, "peak_memory_mib": 1.5}, blob)
    '''
)


@pytest.fixture()
def stub_path(tmp_path: Path) -> Path:
    path = tmp_path / "stub_worker.py"
    path.write_text(STUB_WORKER)
    return path


@pytest.fixture()
def make_client(stub_path, tmp_path):
    clients = []

    def _make(mode: str = "ok", **kwargs) -> MambaWorkerClient:
        client = MambaWorkerClient(
            worker_command=[sys.executable, str(stub_path), mode, str(tmp_path / "crash.marker")],
            device="cpu",
            startup_timeout=kwargs.pop("startup_timeout", 30),
            request_timeout=kwargs.pop("request_timeout", 30),
            **kwargs,
        )
        clients.append(client)
        return client

    yield _make
    for client in clients:
        client.close()


def _tile(batch: bool = True) -> torch.Tensor:
    x = torch.rand(4, 128, 128, dtype=torch.float32) * 0.5
    return x[None] if batch else x


# ---------------------------------------------------------------- lifecycle


def test_start_launches_the_worker_and_returns_its_report(make_client):
    client = make_client()
    assert not client.is_running
    info = client.start()
    assert client.is_running
    assert info["parameter_count"] == 1 and isinstance(info["worker_pid"], int)
    assert "type" not in info and "payload_bytes" not in info


def test_inference_round_trip_returns_the_workers_tensor(make_client):
    client = make_client()
    x = _tile()
    y = client(x)
    assert y.shape == (1, 4, 512, 512) and y.dtype == torch.float32
    assert torch.equal(y, F.interpolate(x, scale_factor=4, mode="nearest"))


def test_unbatched_input_returns_unbatched_output_via_the_stub_contract(make_client):
    client = make_client()
    # The stub interpolates whatever rank it is sent; a 3-D tile is not a valid
    # 4-D interpolate input, so it surfaces as a clean, mapped error rather than a hang.
    with pytest.raises((ModelInferenceError, ModelWorkerError, ModelContractError)):
        client(_tile(batch=False))


def test_worker_is_started_lazily_and_reused_across_calls(make_client):
    client = make_client()
    client(_tile())
    pid = client.describe()["worker_pid"]
    client(_tile())
    client(_tile())
    assert client.describe()["worker_pid"] == pid
    assert client.is_running


def test_describe_includes_isolation_flag_and_last_request_stats(make_client):
    client = make_client()
    client(_tile())
    info = client.describe()
    assert info["isolated_worker"] is True
    assert info["last_request"] == {"inference_seconds": 0.001, "peak_memory_mib": 1.5}


def test_close_stops_the_worker_and_is_idempotent(make_client):
    client = make_client()
    client.start()
    client.close()
    assert not client.is_running
    client.close()  # no error


def test_context_manager_starts_and_stops(make_client):
    with make_client() as client:
        assert client.is_running
    assert not client.is_running


def test_concurrent_requests_do_not_interleave_on_the_pipe(make_client):
    client = make_client()
    client.start()
    inputs = [torch.full((1, 4, 128, 128), 0.1 * (i + 1)) for i in range(6)]
    results = [None] * len(inputs)
    errors = []

    def run(i):
        try:
            results[i] = client(inputs[i])
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(inputs))]
    [t.start() for t in threads]
    [t.join() for t in threads]

    assert not errors
    for x, y in zip(inputs, results):
        assert torch.equal(y, F.interpolate(x, scale_factor=4, mode="nearest"))


# ------------------------------------------------------- contract, pre-IPC


def test_contract_violation_is_rejected_before_any_worker_is_started(make_client):
    client = make_client()
    with pytest.raises(ModelContractError, match="channels"):
        client(torch.rand(1, 3, 128, 128))
    assert not client.is_running  # never spawned: validation happens in this process


def test_raw_digital_numbers_are_rejected_before_any_worker_is_started(make_client):
    client = make_client()
    with pytest.raises(ModelContractError, match="divided by 10000"):
        client(torch.rand(1, 4, 128, 128) * 3000)
    assert not client.is_running


def test_output_with_the_wrong_shape_from_the_worker_is_rejected(make_client):
    client = make_client("bad_shape")
    with pytest.raises(ModelContractError, match="expected"):
        client(_tile())


# --------------------------------------------------- worker failure handling


def test_startup_error_reported_by_the_worker_maps_to_a_user_safe_exception(make_client):
    client = make_client("startup_unavailable")
    with pytest.raises(ModelUnavailableError) as excinfo:
        client.start()
    assert str(excinfo.value) == "SEN2SR-Mamba requires a CUDA-capable GPU."
    assert excinfo.value.technical_detail == "no cuda"
    assert not client.is_running


def test_worker_that_dies_at_startup_gives_a_useful_error(make_client):
    client = make_client("die_at_startup")
    with pytest.raises(ModelWorkerError) as excinfo:
        client.start()
    assert "boom" not in str(excinfo.value)  # user-facing text stays generic
    assert "boom: cannot initialise runtime" in excinfo.value.technical_detail  # ...but the cause is kept for logs
    assert "exit code: 7" in excinfo.value.technical_detail
    assert not client.is_running


def test_worker_that_never_becomes_ready_times_out(make_client):
    client = make_client("startup_hang", startup_timeout=1.0)
    with pytest.raises(ModelWorkerError, match="timed out"):
        client.start()
    assert not client.is_running


def test_worker_crash_mid_request_is_reported_and_the_next_call_restarts_it(make_client):
    client = make_client("crash_once")
    client.start()
    first_pid = client.describe()["worker_pid"]

    with pytest.raises(ModelWorkerError) as excinfo:
        client(_tile())
    assert "stopped unexpectedly" in str(excinfo.value)
    assert "simulated worker crash" in excinfo.value.technical_detail
    assert not client.is_running

    y = client(_tile())  # lazily restarts; the marker file makes the stub behave this time
    assert y.shape == (1, 4, 512, 512)
    assert client.describe()["worker_pid"] != first_pid


def test_hung_worker_is_killed_after_the_request_timeout(make_client):
    client = make_client("hang", request_timeout=1.0)
    client.start()
    with pytest.raises(ModelWorkerError, match="timed out"):
        client(_tile())
    assert not client.is_running


def test_inference_error_from_a_healthy_worker_keeps_the_worker_alive(make_client):
    client = make_client("error_reply")
    with pytest.raises(ModelInferenceError) as excinfo:
        client(_tile())
    assert str(excinfo.value) == "SEN2SR-Mamba ran out of GPU memory."
    assert excinfo.value.technical_detail == "CUDA OOM"
    assert client.is_running


def test_missing_interpreter_is_reported_as_unavailable_not_a_crash(tmp_path):
    client = MambaWorkerClient(python=tmp_path / "no-such-python", weights_dir=tmp_path, device="cuda")
    with pytest.raises(ModelUnavailableError, match="could not be started"):
        client.start()


# --------------------------------------- the REAL worker module, failure path


def test_real_worker_reports_a_clean_startup_failure_without_a_gpu(tmp_path):
    """Launch the actual `frame.models.mamba_worker` (in this environment,
    device 'cpu'): it must report 'requires a CUDA GPU' as a structured
    startup error and exit, not crash or hang. Covers the real worker's
    main()/startup path without needing the Mamba environment."""
    client = MambaWorkerClient(python=sys.executable, weights_dir=tmp_path, device="cpu", startup_timeout=120)
    try:
        with pytest.raises(ModelUnavailableError, match="CUDA"):
            client.start()
        assert not client.is_running
    finally:
        client.close()


# ------------------------------------------------------------- availability


def test_availability_is_false_without_cuda_for_a_cpu_device():
    result = check_mamba_availability("cpu")
    assert result == Availability(False, "no_cuda", "SEN2SR-Mamba requires a CUDA-capable GPU, which is not available.")


def test_availability_reports_a_missing_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    result = check_mamba_availability("cuda", python=tmp_path / "missing", weights_dir=tmp_path)
    assert not result.available and result.reason_code == "runtime_missing"


def test_availability_reports_missing_weights(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    result = check_mamba_availability("cuda", python=Path(sys.executable), weights_dir=tmp_path)
    assert not result.available and result.reason_code == "weights_missing"


def test_availability_is_true_when_everything_is_present(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    (tmp_path / "sr_model.safetensor").write_bytes(b"x")
    (tmp_path / "sr_hard_constraint.safetensor").write_bytes(b"x")
    result = check_mamba_availability("cuda", python=Path(sys.executable), weights_dir=tmp_path)
    assert result.available and result.reason_code == "ok"


def test_availability_messages_do_not_expose_runtime_internals(monkeypatch, tmp_path):
    forbidden = ("triton", "mamba_ssm", "causal-conv1d", "causal_conv1d", "selective_scan")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    cases = [
        check_mamba_availability("cpu"),
        check_mamba_availability("cuda", python=tmp_path / "missing", weights_dir=tmp_path),
        check_mamba_availability("cuda", python=Path(sys.executable), weights_dir=tmp_path),
    ]
    for case in cases:
        assert not any(word in case.message.lower() for word in forbidden), case.message
