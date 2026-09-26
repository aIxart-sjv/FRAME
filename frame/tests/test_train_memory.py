"""frame.train.memory -- FRAME-side activation checkpointing for MambaSR (upstream's own flag is unusable)."""

from __future__ import annotations

import importlib.util

import pytest
import torch
import torch.nn as nn

from frame.train.config import ModelConfig
from frame.train.errors import ModelBuildError
from frame.train.memory import enable_block_checkpointing
from frame.train.models import build_model


class VSSBlock(nn.Module):
    """Stands in for sen2sr's VSSBlock: forward(x, x_size) with a non-tensor size argument, and a call counter."""

    def __init__(self, dim=8):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.calls = 0

    def forward(self, x, x_size):
        self.calls += 1
        b, l, c = x.shape
        assert l == x_size[0] * x_size[1]
        return x + torch.tanh(self.lin(x))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([VSSBlock() for _ in range(3)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x, (4, 4))
        return x


def clone_net(net):
    other = Net()
    other.load_state_dict(net.state_dict())
    return other


def test_checkpointed_blocks_compute_exactly_the_same_outputs_and_gradients():
    plain = Net()
    wrapped = clone_net(plain)
    assert enable_block_checkpointing(wrapped) == 3
    x = torch.rand(2, 16, 8)
    for net in (plain, wrapped):
        net(x.clone().requires_grad_()).pow(2).sum().backward()
    assert torch.equal(plain(x), wrapped(x))
    assert all(torch.equal(a.grad, b.grad) for a, b in zip(plain.parameters(), wrapped.parameters()))


def test_activations_really_are_recomputed_in_the_backward_pass():
    plain, wrapped = Net(), clone_net(Net())
    enable_block_checkpointing(wrapped)
    x = torch.rand(2, 16, 8)
    plain(x.clone().requires_grad_()).sum().backward()
    wrapped(x.clone().requires_grad_()).sum().backward()
    assert [b.calls for b in plain.blocks] == [1, 1, 1] and [b.calls for b in wrapped.blocks] == [2, 2, 2]


def test_inference_is_not_recomputed_and_stays_correct():
    wrapped = Net()
    reference = clone_net(wrapped)
    enable_block_checkpointing(wrapped)
    with torch.no_grad():
        out = wrapped(torch.ones(1, 16, 8))
    assert [b.calls for b in wrapped.blocks] == [1, 1, 1] and torch.equal(out, reference(torch.ones(1, 16, 8)))


def test_wrapping_twice_does_not_nest_and_a_model_without_such_blocks_is_left_alone():
    net = Net()
    assert enable_block_checkpointing(net) == 3 and enable_block_checkpointing(net) == 0
    net(torch.rand(1, 16, 8).requires_grad_()).sum().backward()
    assert [b.calls for b in net.blocks] == [2, 2, 2]
    assert enable_block_checkpointing(nn.Linear(2, 2)) == 0


def test_the_block_class_is_selectable():
    class Other(nn.Module):
        def forward(self, x, x_size):
            return x

    holder = nn.ModuleList([Other()])
    assert enable_block_checkpointing(holder) == 0 and enable_block_checkpointing(holder, block_class="Other") == 1


# ============================================================================== model config


def test_the_mamba_model_accepts_the_checkpointing_flag_but_no_other_model_does():
    if importlib.util.find_spec("mamba_ssm") is None:
        with pytest.raises(ModelBuildError, match="mamba_ssm"):                     # reaches the import, so the parameter itself was accepted
            build_model(ModelConfig("sen2sr_mamba", {"activation_checkpointing": True}), seed=0)
    with pytest.raises(ModelBuildError, match="activation_checkpointing"):
        build_model(ModelConfig("tiny_cnn", {"activation_checkpointing": True}), seed=0)
    with pytest.raises(ModelBuildError, match="activation_checkpointing"):
        build_model(ModelConfig("sen2sr_lite", {"activation_checkpointing": True}), seed=0)
