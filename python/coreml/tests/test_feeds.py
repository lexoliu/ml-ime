"""The host's side of each function: the rows, the masks and the tree's visibility."""

from __future__ import annotations

import numpy as np
import pytest

from mlime_coreml import spec
from mlime_coreml.bundle import Bundle
from mlime_coreml.feeds import TreeNodes, context_row, key_row, tree_row
from mlime_coreml.signature import signature


def test_a_key_row_is_cls_masks_sep_then_padding(bundle: Bundle) -> None:
    row = key_row(bundle.manifest, [3, 4, 5], 8)
    specials = bundle.manifest.specials
    assert row["input_ids"][0].tolist() == [
        specials.cls, specials.mask, specials.mask, specials.mask, specials.sep,
        specials.pad, specials.pad, specials.pad,
    ]  # fmt: skip
    pad = bundle.manifest.letter_pad
    assert row["key_letters"][0].tolist() == [pad, 3, 4, 5, pad, pad, pad, pad]
    assert row["attention_mask"][0].tolist() == [1, 1, 1, 1, 1, 0, 0, 0]
    assert row["input_ids"].dtype == np.int32 and row["attention_mask"].dtype == np.float32
    with pytest.raises(ValueError, match="do not fit"):
        key_row(bundle.manifest, [0] * 7, 8)


def test_a_context_row_pads_and_masks(bundle: Bundle) -> None:
    row = context_row(bundle.manifest, [1, 9, 8, 2], 6)
    assert row["context_ids"][0].tolist() == [1, 9, 8, 2, 0, 0]
    assert row["context_mask"][0].tolist() == [1, 1, 1, 1, 0, 0]


def test_a_node_sees_its_ancestors_and_itself_and_a_padding_row_only_itself(
    bundle: Bundle,
) -> None:
    bos = bundle.manifest.reader_bos
    nodes = TreeNodes(parents=(-1, 0, 0, 1), tokens=(bos, 4, 5, 6))
    row = tree_row(bundle.manifest, nodes, 6)
    bias = row["tree_bias"][0, 0]
    visible = [np.flatnonzero(bias[index] == 0.0).tolist() for index in range(6)]
    assert visible == [[0], [0, 1], [0, 2], [0, 1, 3], [4], [5]]
    assert (bias[bias != 0.0] == spec.NEG).all()
    assert row["positions"][0].tolist() == [0, 1, 1, 2, 0, 0]
    assert row["reader_ids"][0].tolist() == [bos, 4, 5, 6, bos, bos]


def test_each_family_has_one_signature_over_its_symbolic_length(bundle: Bundle) -> None:
    manifest = bundle.manifest
    tree = signature(manifest, "tree")
    assert [tensor.name for tensor in tree.inputs] == [
        "reader_ids", "positions", "tree_bias", "cross_keys", "cross_values", "cross_bias",
    ]  # fmt: skip
    assert tree.inputs[2].resolve(64) == (1, 1, 64, 64)
    assert tree.outputs[0].resolve(64) == (1, 64, manifest.dims.emitted)
    keys = signature(manifest, "keys")
    assert keys.inputs[3].resolve(16) == (1, spec.CONTEXT_BUCKETS[0], manifest.dims.fill.hidden)
    assert keys.outputs[0].resolve(16) == (1, 16, manifest.dims.emitted + 1)
    decoder = manifest.dims.decoder
    assert keys.outputs[1].resolve(16) == (
        decoder.layers,
        decoder.heads,
        spec.N_MAX,
        decoder.head_dim,
    )
    assert not keys.outputs[1].batched and keys.outputs[0].batched
