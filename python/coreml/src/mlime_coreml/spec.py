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

#: The widest projection whose head split the Neural Engine compiler accepts. Measured on the M1
#: (macOS 27.0.1, coremltools 9.0): a query and key projection of 512 channels or fewer splits
#: into heads and multiplies at every length tried (64 to 256), and 576 and 640 at 64; at 768 the
#: program fails to compile for the engine at every head count (3, 6, 12, 24) and length (32 to
#: 64), and Core ML then fails the predict with a misleading "functionName ... ML Program" error.
#: A wider projection is cut into groups of whole heads (:class:`mlime_coreml.modules.Heads`).
MAX_PROJECTION_WIDTH: Final = 512

#: LayerNorm's epsilon in the modules the manifest does not record one for: torch's default,
#: which the key front's blocks, the gated context blocks and the decoder's blocks were built with.
DEFAULT_LAYER_NORM_EPS: Final = 1e-5

#: The function names' prefixes: ``ctx_64``, ``keys_16``, ``tree_32``.
FUNCTION_PREFIX: Final[dict[Family, str]] = {"context": "ctx", "keys": "keys", "tree": "tree"}

# --- tolerances -----------------------------------------------------------------------------

#: How far the fp32 port may stray from the bundle's reference (the same arithmetic, spelled out).
#: Measured on a random-weight bundle of the real dimensions (M1, 2026-11): 9.2e-5 at most, in
#: the tree's logits, whose scale there is 57 (about 1.6e-6 of it); a tiny bundle's is 7e-7.
PORT_ATOL: Final = 5e-4

#: How far the fp32 port at a bucket's width may stray from the exact-length run: padding is
#: invisible by construction, so only the order of float32 sums differs. Measured the same way:
#: 4.6e-5 at most, in the tree's logits; a tiny bundle's is 1e-6.
PADDING_ATOL_FP32: Final = 2e-4

#: float16 Core ML output against the fp32 port, per output: ``|a - b| <= atol + rtol * |b|``
#: elementwise over the real rows. Logits are the head's raw scores, hidden states and the cross
#: keys and values are LayerNorm-scale activations.
#:
#: Each is about twice the larger of what two bundles of the real dimensions measured on the M1
#: (the largest excess of ``|a - b|`` over ``rtol * |b|``, over every function and probe,
#: against the reference and the exact-length port). Random weights: context 0.025, ctc logits
#: 0.027, cross keys 0.017 and values 0.019, tree logits 0.188 (logits to 57, std 13). The
#: joint stage's trained weights (S2, #257): context 0.012, ctc logits 0.087, cross keys 0.057
#: and values 0.060, tree logits 0.158. Training widens the key tower's outputs and narrows
#: the decoder's logits, so neither bundle bounds the other. A tiny bundle measures an order
#: of magnitude under all of them. The tree's is the loosest in absolute terms; a wrong mask
#: or a mis-sliced weight moves a logit by far more than 0.4 at that scale.
FP16_TOLERANCE: Final[dict[str, tuple[float, float]]] = {
    "context": (5e-2, 2e-2),
    "ctc_logits": (1.8e-1, 2e-2),
    "cross_keys": (1.2e-1, 2e-2),
    "cross_values": (1.2e-1, 2e-2),
    "logits": (4e-1, 2e-2),
}

# --- placement ------------------------------------------------------------------------------

#: The ops whose placement matters: the matrix products, the normalisations and the attention
#: softmax. Every one of them must run on the Neural Engine; a CPU fallback of any is a failure.
MAIN_OPS: Final[frozenset[str]] = frozenset(
    {"linear", "matmul", "softmax", "layer_norm", "gelu", "conv"}
)

#: At least this share of a function's non-const ops must run on the Neural Engine. The rest are
#: glue, which the plan places on the CPU. Measured on the real dimensions (M1): 535 of 542 in
#: the context function, 810 of 822 in every keys function and 497 of 506 in every tree function
#: (98.2% to 98.7%); the CPU ops are the embedding lookups' index handling (``gather``,
#: ``greater_equal``, ``select``, one ``add`` and a few ``cast``) and the keys function's letter
#: mask (``not_equal``, ``cast``, ``expand_dims``), none a main op. A tiny bundle places nothing
#: on the engine, so the gate is the real-size conversion's (``mlime-coreml convert``).
MIN_ANE_SHARE: Final = 0.95

# --- timing ---------------------------------------------------------------------------------

WARMUP_CALLS: Final = 20
TIMED_CALLS: Final = 200
