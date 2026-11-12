"""What the conversion pins: the buckets, the masks, the tolerances and the placement rules.

The bundle's manifest lists the same buckets and constants; :func:`mlime_coreml.bundle.read_bundle`
refuses a bundle whose own differ, so a bucket changed on one side fails at the door and not in a
shape error in the middle of a conversion.
"""

from __future__ import annotations

from typing import Final, Literal

#: The function families, as the manifest names them, and the directory each converts into.
Family = Literal["context", "keys", "tree"]
FAMILIES: Final[tuple[Family, ...]] = ("context", "keys", "tree")
DIRECTORIES: Final[dict[Family, str]] = {"context": "context", "keys": "keys", "tree": "decoder"}

#: The packing policy this code speaks (the bundle's ``policy_version``).
POLICY_VERSION: Final = 1

#: The context function's static length: the context tower's tokens, sentinels included.
CONTEXT_BUCKETS: Final[tuple[int, ...]] = (64,)

#: The keys functions' static widths: ``[CLS]``, up to ``width - 2`` keys, ``[SEP]``, padding.
KEYS_WIDTHS: Final[tuple[int, ...]] = (16, 32, 64, 128)

#: The tree functions' node counts.
TREE_BUCKETS: Final[tuple[int, ...]] = (32, 64, 128, 256)

#: ``[CLS]`` and ``[SEP]``, the two columns around the keys.
KEY_SENTINELS: Final = 2

#: The cross-attention length of the decoder's key states.
N_MAX: Final = 128

#: The additive mask of every attention: finite in float16, so a fully masked row stays finite
#: (the Neural Engine turns ``-inf`` and float32's minimum into NaN rows).
NEG: Final = -1e4

#: LayerNorm's epsilon in the modules the manifest does not record one for: torch's default,
#: which the key front's blocks, the gated context blocks and the decoder's blocks were built with.
DEFAULT_LAYER_NORM_EPS: Final = 1e-5

#: The function names' prefixes: ``ctx_64``, ``keys_16``, ``tree_32``.
FUNCTION_PREFIX: Final[dict[Family, str]] = {"context": "ctx", "keys": "keys", "tree": "tree"}

# --- tolerances -----------------------------------------------------------------------------

#: How far the fp32 port may stray from the bundle's reference (the same arithmetic, spelled out).
PORT_ATOL: Final = 1e-4

#: How far the fp32 port at a bucket's width may stray from the exact-length run: padding is
#: invisible by construction, so only the order of float32 sums differs.
PADDING_ATOL_FP32: Final = 1e-5

#: float16 Core ML output against the fp32 port, per output: ``|a - b| <= atol + rtol * |b|``
#: elementwise over the real rows. Logits are the head's raw scores, hidden states and the cross
#: keys and values are LayerNorm-scale activations.
FP16_TOLERANCE: Final[dict[str, tuple[float, float]]] = {
    "context": (5e-2, 2e-2),
    "ctc_logits": (1e-1, 2e-2),
    "cross_keys": (5e-2, 2e-2),
    "cross_values": (5e-2, 2e-2),
    "logits": (1e-1, 2e-2),
}

# --- placement ------------------------------------------------------------------------------

#: The ops whose placement matters: the matrix products, the normalisations and the attention
#: softmax. Every one of them must run on the Neural Engine; a CPU fallback of any is a failure.
MAIN_OPS: Final[frozenset[str]] = frozenset(
    {"linear", "matmul", "softmax", "layer_norm", "gelu", "conv"}
)

#: At least this share of a function's non-const ops must run on the Neural Engine. The rest are
#: shape, cast and gather glue, which the plan may place on the CPU.
MIN_ANE_SHARE: Final = 0.85

# --- timing ---------------------------------------------------------------------------------

WARMUP_CALLS: Final = 20
TIMED_CALLS: Final = 200
