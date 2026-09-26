"""frame.models.protocol -- the framed message format between FRAME and the worker."""

from __future__ import annotations

import io
import struct

import pytest
import torch

from frame.models.errors import ModelContractError
from frame.models.protocol import MAX_HEADER_BYTES, decode_tensor, encode_tensor, read_message, send_message


def _reader(data: bytes):
    stream = io.BytesIO(data)
    return stream.read


def _roundtrip(header, payload=b""):
    buffer = io.BytesIO()
    send_message(buffer, header, payload)
    return read_message(_reader(buffer.getvalue()))


def test_header_only_message_round_trips():
    header, payload = _roundtrip({"op": "ping"})
    assert header["op"] == "ping" and payload == b""


def test_message_with_payload_round_trips():
    header, payload = _roundtrip({"op": "x"}, b"\x00\x01\x02\xff")
    assert header["payload_bytes"] == 4 and payload == b"\x00\x01\x02\xff"


def test_consecutive_messages_are_read_in_order():
    buffer = io.BytesIO()
    send_message(buffer, {"n": 1}, b"aa")
    send_message(buffer, {"n": 2})
    read = _reader(buffer.getvalue())
    assert read_message(read)[0]["n"] == 1
    assert read_message(read)[0]["n"] == 2


def test_clean_end_of_stream_raises_eof():
    with pytest.raises(EOFError):
        read_message(_reader(b""))


def test_truncated_header_raises_eof():
    with pytest.raises(EOFError):
        read_message(_reader(struct.pack(">I", 50) + b"{"))


def test_truncated_payload_raises_eof():
    buffer = io.BytesIO()
    send_message(buffer, {"op": "x"}, b"12345678")
    with pytest.raises(EOFError):
        read_message(_reader(buffer.getvalue()[:-3]))


def test_implausible_header_length_is_rejected():
    with pytest.raises(ValueError):
        read_message(_reader(struct.pack(">I", MAX_HEADER_BYTES + 1)))


def test_non_object_header_is_rejected():
    body = b"[1, 2]"
    with pytest.raises(ValueError):
        read_message(_reader(struct.pack(">I", len(body)) + body))


def test_tensor_round_trips_exactly():
    x = torch.randn(1, 4, 8, 8)
    fields, blob = encode_tensor(x)
    y = decode_tensor(fields, blob)
    assert y.dtype == torch.float32 and torch.equal(x, y)


def test_tensor_encoding_is_little_endian_float32_regardless_of_input_layout():
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).permute(2, 0, 1)  # non-contiguous
    fields, blob = encode_tensor(x)
    assert fields["shape"] == [4, 2, 3] and len(blob) == 24 * 4
    assert torch.equal(decode_tensor(fields, blob), x)


def test_only_float32_can_be_encoded():
    with pytest.raises(ModelContractError, match="float32"):
        encode_tensor(torch.zeros(2, dtype=torch.float64))


def test_decode_rejects_a_payload_that_does_not_match_the_shape():
    fields, blob = encode_tensor(torch.zeros(2, 2))
    with pytest.raises(ValueError, match="needs"):
        decode_tensor(fields, blob[:-4])


def test_decode_rejects_an_unsupported_dtype():
    with pytest.raises(ValueError, match="dtype"):
        decode_tensor({"shape": [1], "dtype": "float64"}, b"\x00" * 8)
