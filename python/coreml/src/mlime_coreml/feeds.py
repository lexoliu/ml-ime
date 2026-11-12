"""Function inputs built from keys, nodes and contexts: the host's side of each function.

What a consumer assembles before it calls a function, rebuilt here so the verification feeds the
functions the way the host will: a keystroke row is ``[CLS] [MASK]*n [SEP] [PAD]*``, a tree call's
node set carries its ancestors' visibility as an additive bias, and padding is invisible. Arrays
are numpy, masks float32 and ids int32; a caller narrows the masks to float16 for Core ML.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from mlime_coreml import spec
from mlime_coreml.bundle import Manifest

Array = npt.NDArray[np.generic]


@dataclass(frozen=True)
class ContextFeed:
    """The context function's output as the keys function reads it, with its mask."""

    rows: npt.NDArray[np.float32]  # [C, H]
    mask: npt.NDArray[np.float32]  # [C]


@dataclass(frozen=True)
class CrossFeed:
    """The keys function's cross keys and values, with the bias that masks the unused slots."""

    keys: npt.NDArray[np.float32]  # [layers, heads, N_MAX, head_dim]
    values: npt.NDArray[np.float32]
    bias: npt.NDArray[np.float32]  # [1, 1, 1, N_MAX]


def key_row(manifest: Manifest, letters: Sequence[int], width: int) -> dict[str, Array]:
    """The keys function's row inputs for *letters* (ids ``0..25``) at *width*."""
    count = len(letters)
    if count + spec.KEY_SENTINELS > width:
        raise ValueError(f"{count} keys do not fit a row of {width}")
    specials, pad = manifest.specials, manifest.letter_pad
    ids = np.full(width, specials.pad, dtype=np.int32)
    ids[0], ids[count + 1] = specials.cls, specials.sep
    ids[1 : count + 1] = specials.mask
    key_letters = np.full(width, pad, dtype=np.int32)
    key_letters[1 : count + 1] = letters
    attention = np.zeros(width, dtype=np.float32)
    attention[: count + spec.KEY_SENTINELS] = 1.0
    return {
        "input_ids": ids[None],
        "key_letters": key_letters[None],
        "attention_mask": attention[None],
    }


def random_letters(manifest: Manifest, count: int, rng: random.Random) -> list[int]:
    """*count* random key letters."""
    return [rng.randrange(manifest.letter_pad) for _ in range(count)]


def context_row(
    manifest: Manifest, ids: Sequence[int], length: int, tokens: int | None = None
) -> dict[str, Array]:
    """The context function's inputs for *ids* (sentinels included) at *length*."""
    real = len(ids) if tokens is None else tokens
    padded = np.full(length, manifest.specials.pad, dtype=np.int32)
    padded[:real] = list(ids)[:real]
    mask = np.zeros(length, dtype=np.float32)
    mask[:real] = 1.0
    return {"context_ids": padded[None], "context_mask": mask[None]}


def random_context_ids(manifest: Manifest, real: int, rng: random.Random) -> list[int]:
    """``[CLS]``, ``real - 2`` random tokens and ``[SEP]``."""
    specials = manifest.specials
    body = [
        rng.randrange(specials.mask + 1, manifest.dims.context.vocabulary) for _ in range(real - 2)
    ]
    return [specials.cls, *body, specials.sep]


@dataclass(frozen=True)
class TreeNodes:
    """A packed node set: each node's parent (``-1`` for the root) and its reader token."""

    parents: tuple[int, ...]
    tokens: tuple[int, ...]


def random_tree(manifest: Manifest, nodes: int, rng: random.Random) -> TreeNodes:
    """A random ancestor-closed set of *nodes*: a root, then each node under an earlier one."""
    reader_ids = manifest.reader_ids
    parents = [-1, *(rng.randrange(index) for index in range(1, nodes))]
    tokens = [manifest.reader_bos, *(rng.choice(reader_ids) for _ in range(1, nodes))]
    return TreeNodes(tuple(parents), tuple(tokens))


def tree_row(manifest: Manifest, nodes: TreeNodes, bucket: int) -> dict[str, Array]:
    """The tree function's node inputs at *bucket* nodes; the rest is padding, last.

    A node sees its ancestors and itself, a padding row only itself.
    """
    count = len(nodes.parents)
    if count > bucket:
        raise ValueError(f"{count} nodes do not fit a bucket of {bucket}")
    reader_ids = np.full(bucket, manifest.reader_bos, dtype=np.int32)
    positions = np.zeros(bucket, dtype=np.int32)
    bias = np.full((bucket, bucket), spec.NEG, dtype=np.float32)
    for row in range(bucket):
        bias[row, row] = 0.0
    for row, (parent, token) in enumerate(zip(nodes.parents, nodes.tokens, strict=True)):
        reader_ids[row] = token
        if parent >= 0:
            positions[row] = positions[parent] + 1
            bias[row] = np.where(bias[parent] == 0.0, 0.0, bias[row])
            bias[row, row] = 0.0
    return {
        "reader_ids": reader_ids[None],
        "positions": positions[None],
        "tree_bias": bias[None, None],
    }
