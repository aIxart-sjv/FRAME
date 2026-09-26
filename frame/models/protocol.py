"""Wire protocol between FRAME and the isolated Mamba worker (Phase 1).

One message = ``[4-byte big-endian header length][JSON header][payload]``.
The header is a JSON object whose ``payload_bytes`` field says how many raw
payload bytes follow (0 for none). Tensors travel as little-endian float32
with their shape in the header -- no pickle, so neither side ever
deserialises code from the other.

Used identically by frame.models.mamba_client (main environment) and
frame.models.mamba_worker (Mamba environment); only torch/numpy/stdlib, all
present in both.
"""

from __future__ import annotations

import json
import struct
from typing import Any, BinaryIO, Callable, Dict, Tuple

import numpy as np
import torch

from frame.models.errors import ModelContractError

_LENGTH = struct.Struct(">I")
MAX_HEADER_BYTES = 1 << 20          # 1 MiB
MAX_PAYLOAD_BYTES = 64 << 20        # 64 MiB -- far above a 4x512x512 float32 tile (4 MiB)

ReadExact = Callable[[int], bytes]


def send_message(stream: BinaryIO, header: Dict[str, Any], payload: bytes = b"") -> None:
    """Write one framed message and flush."""
    header = dict(header, payload_bytes=len(payload))
    blob = json.dumps(header).encode("utf-8")
    stream.write(_LENGTH.pack(len(blob)))
    stream.write(blob)
    if payload:
        stream.write(payload)
    stream.flush()


def read_message(read_exact: ReadExact) -> Tuple[Dict[str, Any], bytes]:
    """Read one framed message.

    ``read_exact(n)`` must return exactly ``n`` bytes or fewer only at EOF.
    Raises `EOFError` if the stream ends, `ValueError` on a malformed frame.
    """
    prefix = read_exact(_LENGTH.size)
    if len(prefix) < _LENGTH.size:
        raise EOFError("stream closed before a message header")
    (header_len,) = _LENGTH.unpack(prefix)
    if header_len > MAX_HEADER_BYTES:
        raise ValueError(f"implausible header length {header_len}")

    blob = read_exact(header_len)
    if len(blob) < header_len:
        raise EOFError("stream closed inside a message header")
    header = json.loads(blob.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("message header is not a JSON object")

    payload_len = int(header.get("payload_bytes", 0))
    if payload_len < 0 or payload_len > MAX_PAYLOAD_BYTES:
        raise ValueError(f"implausible payload length {payload_len}")
    payload = read_exact(payload_len) if payload_len else b""
    if len(payload) < payload_len:
        raise EOFError("stream closed inside a message payload")
    return header, payload


def encode_tensor(tensor: torch.Tensor) -> Tuple[Dict[str, Any], bytes]:
    """Tensor -> (header fields, bytes). Only float32 is representable, matching the contract."""
    if tensor.dtype != torch.float32:
        raise ModelContractError(f"Only float32 tensors can be sent to the model worker, got {tensor.dtype}.")
    array = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype="<f4")
    return {"shape": list(array.shape), "dtype": "float32"}, array.tobytes()


def decode_tensor(header: Dict[str, Any], payload: bytes) -> torch.Tensor:
    """(header fields, bytes) -> float32 tensor; validates size against the declared shape."""
    if header.get("dtype") != "float32":
        raise ValueError(f"unsupported tensor dtype {header.get('dtype')!r}")
    shape = tuple(int(v) for v in header["shape"])
    expected = int(np.prod(shape, dtype=np.int64)) * 4
    if len(payload) != expected:
        raise ValueError(f"payload has {len(payload)} bytes, shape {shape} needs {expected}")
    return torch.from_numpy(np.frombuffer(payload, dtype="<f4").reshape(shape).copy())
