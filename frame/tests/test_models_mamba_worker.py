"""frame.models.mamba_worker.serve -- the real worker request loop, driven
in-process with a fake model (no GPU, no Mamba environment)."""

from __future__ import annotations

import io

import torch
import torch.nn.functional as F

from frame.models.errors import ModelContractError
from frame.models.mamba_worker import serve
from frame.models.protocol import decode_tensor, encode_tensor, read_message, send_message


class FakeModel:
    device = "cpu"

    def __init__(self, behaviour=None):
        self.behaviour = behaviour
        self.calls = 0

    def describe(self):
        return {}

    def reset_peak_memory(self):
        pass

    def peak_memory_mib(self):
        return 12.5

    def __call__(self, x):
        self.calls += 1
        if self.behaviour is not None:
            return self.behaviour(x)
        return F.interpolate(x, scale_factor=4, mode="nearest")


def _run(model, *messages) -> tuple:
    request = io.BytesIO()
    for header, payload in messages:
        send_message(request, header, payload)
    request.seek(0)
    out = io.BytesIO()
    code = serve(model, request, out)
    out.seek(0)
    replies = []
    while True:
        try:
            replies.append(read_message(out.read))
        except EOFError:
            break
    return code, replies


def _infer(x):
    fields, blob = encode_tensor(x)
    return {"op": "infer", **fields}, blob


def test_ping_and_shutdown():
    code, replies = _run(FakeModel(), ({"op": "ping"}, b""), ({"op": "shutdown"}, b""))
    assert code == 0
    assert [r[0]["type"] for r in replies] == ["pong", "bye"]


def test_inference_returns_the_tensor_with_timing_and_memory_stats():
    x = torch.rand(1, 4, 128, 128)
    code, replies = _run(FakeModel(), _infer(x), ({"op": "shutdown"}, b""))
    header, payload = replies[0]
    assert header["type"] == "result" and header["peak_memory_mib"] == 12.5 and header["inference_seconds"] >= 0
    assert torch.equal(decode_tensor(header, payload), F.interpolate(x, scale_factor=4, mode="nearest"))


def test_end_of_input_exits_cleanly_when_the_parent_goes_away():
    code, replies = _run(FakeModel())
    assert code == 0 and replies == []


def test_unknown_operation_is_reported_and_the_worker_keeps_serving():
    code, replies = _run(FakeModel(), ({"op": "frobnicate"}, b""), ({"op": "ping"}, b""), ({"op": "shutdown"}, b""))
    assert replies[0][0]["type"] == "error" and "frobnicate" in replies[0][0]["message"]
    assert replies[1][0]["type"] == "pong"


def test_malformed_tensor_payload_is_a_structured_error_not_a_crash():
    fields, blob = encode_tensor(torch.rand(1, 4, 128, 128))
    code, replies = _run(FakeModel(), ({"op": "infer", **fields}, blob[:-4]), ({"op": "ping"}, b""))
    assert replies[0][0]["type"] == "error" and replies[0][0]["error_type"] == "ModelContractError"
    assert replies[1][0]["type"] == "pong"


def test_model_contract_error_is_reported_with_its_type_and_message():
    def reject(x):
        raise ModelContractError("Model input must have 4 channels.")

    _, replies = _run(FakeModel(reject), _infer(torch.rand(1, 4, 128, 128)), ({"op": "ping"}, b""))
    header = replies[0][0]
    assert header["error_type"] == "ModelContractError" and header["message"] == "Model input must have 4 channels."
    assert replies[1][0]["type"] == "pong"  # still alive


def test_unexpected_exception_gets_a_generic_message_and_the_cause_in_technical_detail():
    def explode(x):
        raise RuntimeError("selective_scan_cuda: illegal memory access")

    _, replies = _run(FakeModel(explode), _infer(torch.rand(1, 4, 128, 128)), ({"op": "ping"}, b""))
    header = replies[0][0]
    assert header["error_type"] == "InternalError"
    assert "selective_scan" not in header["message"]  # the user-facing text never names internals
    assert "illegal memory access" in header["technical_detail"]
    assert replies[1][0]["type"] == "pong"


def test_malformed_frame_ends_the_session_with_a_protocol_error():
    garbage = io.BytesIO(b"\xff\xff\xff\xff")
    out = io.BytesIO()
    assert serve(FakeModel(), garbage, out) == 3
    out.seek(0)
    header, _ = read_message(out.read)
    assert header["type"] == "error" and header["phase"] == "protocol"
