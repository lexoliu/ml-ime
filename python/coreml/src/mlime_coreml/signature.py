"""Each function's inputs and outputs: names, dtypes and shapes, from the bundle's dimensions.

One description feeds the conversion (the Core ML feature types), the verification (the feeds it
builds), the benchmark and the per-directory ``manifest.json`` a consumer reads. A dimension is an
int, or ``"length"``: the function's own length (the keys width, the context tokens, the node
count), which differs between a family's functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from mlime_coreml import spec
from mlime_coreml.bundle import Function, Manifest
from mlime_coreml.spec import Family

Dim = int | Literal["length"]
DType = Literal["int32", "float16"]


@dataclass(frozen=True)
class Tensor:
    """One named input or output."""

    name: str
    dtype: DType
    shape: tuple[Dim, ...]
    #: Whether the leading dimension is a batch of one (the cross keys and values have none).
    batched: bool = True

    def resolve(self, length: int) -> tuple[int, ...]:
        """The concrete shape of a function of *length*."""
        return tuple(length if dim == "length" else dim for dim in self.shape)

    @property
    def numpy_dtype(self) -> type[np.int32] | type[np.float16]:
        """The numpy scalar type a Core ML feature of this dtype is fed and read as."""
        return np.int32 if self.dtype == "int32" else np.float16


@dataclass(frozen=True)
class Signature:
    """A function family's inputs and outputs."""

    inputs: tuple[Tensor, ...]
    outputs: tuple[Tensor, ...]


def signature(manifest: Manifest, family: Family) -> Signature:
    """The signature every function of *family* shares, its lengths symbolic."""
    dims = manifest.dims
    decoder = dims.decoder
    cross = (decoder.layers, decoder.heads, spec.N_MAX, decoder.head_dim)
    if family == "context":
        return Signature(
            inputs=(
                Tensor("context_ids", "int32", (1, "length")),
                Tensor("context_mask", "float16", (1, "length")),
            ),
            outputs=(Tensor("context", "float16", (1, "length", dims.context.hidden)),),
        )
    if family == "keys":
        tokens = manifest.context_tokens
        return Signature(
            inputs=(
                Tensor("input_ids", "int32", (1, "length")),
                Tensor("key_letters", "int32", (1, "length")),
                Tensor("attention_mask", "float16", (1, "length")),
                Tensor("context", "float16", (1, tokens, dims.fill.hidden)),
                Tensor("context_mask", "float16", (1, tokens)),
            ),
            outputs=(
                Tensor("ctc_logits", "float16", (1, "length", dims.emitted + 1)),
                Tensor("cross_keys", "float16", cross, batched=False),
                Tensor("cross_values", "float16", cross, batched=False),
            ),
        )
    return Signature(
        inputs=(
            Tensor("reader_ids", "int32", (1, "length")),
            Tensor("positions", "int32", (1, "length")),
            Tensor("tree_bias", "float16", (1, 1, "length", "length")),
            Tensor("cross_keys", "float16", cross, batched=False),
            Tensor("cross_values", "float16", cross, batched=False),
            Tensor("cross_bias", "float16", (1, 1, 1, spec.N_MAX)),
        ),
        outputs=(Tensor("logits", "float16", (1, "length", dims.emitted)),),
    )


def functions(manifest: Manifest, family: Family) -> tuple[Function, ...]:
    """*family*'s functions, smallest first."""
    return manifest.family(family).functions
