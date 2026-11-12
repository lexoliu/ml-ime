"""Grouped attention is the same attention: cutting a wide projection into groups of whole heads
changes no value, so the width limit the Neural Engine compiler imposes costs nothing in the
arithmetic."""

from __future__ import annotations

import pytest
import torch

from mlime_coreml import spec
from mlime_coreml.modules import Heads, attend_heads, merge_heads, split_heads


@pytest.mark.parametrize(("hidden", "heads", "groups"), [(768, 12, 2), (512, 8, 1), (1536, 12, 3)])
def test_the_groups_keep_every_projection_within_the_limit(
    hidden: int, heads: int, groups: int
) -> None:
    layout = Heads.of(hidden, heads)
    assert layout.groups == groups
    assert layout.width * layout.groups == hidden
    assert layout.width <= spec.MAX_PROJECTION_WIDTH
    assert heads % layout.groups == 0


@pytest.mark.parametrize("length", [5, 16])
def test_grouped_attention_equals_the_whole(length: int, monkeypatch: pytest.MonkeyPatch) -> None:
    hidden, heads = 48, 6
    monkeypatch.setattr(spec, "MAX_PROJECTION_WIDTH", 16)
    layout = Heads.of(hidden, heads)
    assert layout.groups == 3
    generator = torch.Generator().manual_seed(length)
    q, k, v = (torch.randn(1, length, hidden, generator=generator) for _ in range(3))
    bias = torch.triu(torch.full((length, length), spec.NEG), diagonal=1)
    whole = merge_heads(
        attend_heads(
            split_heads(q, heads), split_heads(k, heads), split_heads(v, heads), bias, layout.scale
        )
    )
    torch.testing.assert_close(layout.attend(q, k, v, bias), whole, atol=1e-6, rtol=0.0)
