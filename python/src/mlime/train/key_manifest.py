"""``key-model.json``: what a consumer of a key-model bundle needs to open it and check it (#266).

The key-level model runs through Core ML as three function families, one
multifunction model each (:func:`families`): the context encoder, the keys
function that gives each keystroke's CTC logits and the decoder's cross keys
and values, and the decoder's tree recompute (:mod:`mlime.train.key_tree`).
:class:`KeyManifest` is the bundle's table of contents and its contract: the
emission order, the model's dimensions, each tensor of ``weights.safetensors``
and the families that read it, the families and their buckets, the packing
policy's version, the legality lattices' digests a consumer's own automaton
must reproduce, and the probes ``reference.npz`` was computed over.

It is a tree of typed dataclasses read and written through pydantic, so the
JSON is the dataclasses' serialisation and nothing is assembled as text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, TypeAdapter
from pydantic.dataclasses import dataclass

from mlime.legality import Legality, LegalityOptions, lattice_digest
from mlime.train.key_tree import POLICY_VERSION, TREE_BUCKETS, Prefix
from mlime.train.lexicon import Lexicon

#: The layout string ``key-model.json`` carries.
LAYOUT: Literal["key-bundle"] = "key-bundle"

#: The bundle's files.
MANIFEST = "key-model.json"
WEIGHTS = "weights.safetensors"
TOKENIZER = "tokenizer.json"
REFERENCE = "reference.npz"

#: The context function's static length: the context tower's tokens, sentinels included.
CONTEXT_BUCKETS: tuple[int, ...] = (64,)

#: The keys functions' static widths. A width holds ``width - KEY_SENTINELS`` keys.
KEYS_WIDTHS: tuple[int, ...] = (16, 32, 64, 128)

#: ``[CLS]`` and ``[SEP]``, the two columns around the keys.
KEY_SENTINELS = 2

#: The cross-attention length of the decoder's key states: every keys function
#: pads its cross keys and values to it, and the tree functions read it.
N_MAX = 128

#: The additive mask value of every attention in the Core ML functions: finite
#: in float16, so a fully masked row stays finite.
NEG = -1e4

#: The manifest's strictness: a field it does not know is an error.
_RECORD = ConfigDict(extra="forbid", frozen=True)

#: The key strings whose legality lattices the manifest pins, each with whether
#: its last span may be unfinished: the paper's table (#216 section 3), the
#: initial and tail rules, and a string that reads two ways.
LEGALITY_PROBES: tuple[tuple[str, bool], ...] = (
    ("wmzyq", False),
    ("wmzyq", True),
    ("shuijiap", False),
    ("xian", False),
    ("guozho", True),
    ("zhoguo", False),
)


@dataclass(config=_RECORD)
class LegalityProbe:
    """One probe key string's legality lattice: its arc count and digest."""

    keys: str
    unfinished_tail: bool
    arcs: int
    digest: str

    @classmethod
    def of(
        cls, legality: Legality, keys: str, unfinished_tail: bool, correct: bool
    ) -> LegalityProbe:
        """*legality*'s lattice of *keys* under the options, summarised."""
        lattice = legality.lattice(
            keys, LegalityOptions(unfinished_tail=unfinished_tail, correct_syllables=correct)
        )
        return cls(keys, unfinished_tail, int(lattice.cost.size), lattice_digest(lattice))


def legality_probes(legality: Legality, correct_syllables: bool) -> list[LegalityProbe]:
    """The manifest's probes: :data:`LEGALITY_PROBES` through *legality*."""
    return [
        LegalityProbe.of(legality, keys, tail, correct_syllables) for keys, tail in LEGALITY_PROBES
    ]


@dataclass(config=_RECORD)
class Function:
    """One function of a family: its name, static sequence length and what it holds.

    ``capacity`` is what a call can carry: context tokens (the tower's, less
    its two sentinels), keys (``length`` less the key sentinels) or tree nodes.
    """

    name: str
    length: int
    capacity: int


@dataclass(config=_RECORD)
class Family:
    """A function family: one multifunction Core ML model, its functions smallest first."""

    name: Literal["context", "keys", "tree"]
    functions: tuple[Function, ...]


def families() -> tuple[Family, ...]:
    """The function families and their buckets: what the conversion builds, the host picks from."""
    return (
        Family(
            "context",
            tuple(Function(f"ctx_{size}", size, size - KEY_SENTINELS) for size in CONTEXT_BUCKETS),
        ),
        Family(
            "keys",
            tuple(Function(f"keys_{size}", size, size - KEY_SENTINELS) for size in KEYS_WIDTHS),
        ),
        Family("tree", tuple(Function(f"tree_{size}", size, size) for size in TREE_BUCKETS)),
    )


@dataclass(config=_RECORD)
class Specials:
    """The tokenizer's special ids the key row is built from."""

    pad: int
    cls: int
    sep: int
    mask: int


@dataclass(config=_RECORD)
class TowerDims:
    """A BERT tower's dimensions: the context tower, or the fill tower."""

    layers: int
    heads: int
    hidden: int
    intermediate: int
    positions: int
    vocabulary: int
    layer_norm_eps: float


@dataclass(config=_RECORD)
class KeyFrontDims:
    """The key front's causal stack: its blocks and the positions it embeds."""

    layers: int
    heads: int
    feedforward: int
    positions: int


@dataclass(config=_RECORD)
class DecoderDims:
    """The decoder: the char LM reader's shape and the cross-attention beside each block."""

    layers: int
    heads: int
    head_dim: int
    hidden: int
    embedding: int
    feedforward: int
    positions: int
    vocabulary: int
    layer_norm_eps: float


@dataclass(config=_RECORD)
class ModelDims:
    """Every dimension the three families' modules are built from.

    ``gated`` is how many of the fill tower's top layers a gated context block
    follows. The key front's width and the cross-attention's key states are
    the fill tower's ``hidden``.
    """

    context: TowerDims
    fill: TowerDims
    gated: int
    key_front: KeyFrontDims
    decoder: DecoderDims
    emitted: int


@dataclass(config=_RECORD)
class TensorEntry:
    """One tensor of ``weights.safetensors`` and the families that read it."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    families: tuple[Literal["context", "keys", "tree"], ...]


@dataclass(config=_RECORD)
class ReferenceProbe:
    """One probe of ``reference.npz``: a key string, its context, and the tree sets asked.

    ``tree`` holds one request set per tree reference: the prefixes asked, which
    :func:`mlime.train.key_tree.plan` packs into the calls the reference records.
    """

    name: str
    keys: str
    context: str | None
    tree: tuple[tuple[Prefix, ...], ...]


@dataclass(config=_RECORD)
class KeyManifest:
    """``key-model.json``: the bundle's contract, read and written as typed data."""

    layout: Literal["key-bundle"]
    step: int
    base_model: str
    policy_version: int
    n_max: int
    neg: float
    context_tokens: int
    specials: Specials
    letter_pad: int
    characters: tuple[str, ...]
    reader_ids: tuple[int, ...]
    reader_bos: int
    dims: ModelDims
    correct_syllables: bool
    probes: tuple[LegalityProbe, ...]
    families: tuple[Family, ...]
    tensors: tuple[TensorEntry, ...]
    reference: tuple[ReferenceProbe, ...]

    def __post_init__(self) -> None:
        if len(self.reader_ids) != len(self.characters):
            raise ValueError(
                f"{len(self.reader_ids)} reader ids for {len(self.characters)} emittable characters"
            )
        if self.dims.emitted != len(self.characters):
            raise ValueError(
                f"the head emits {self.dims.emitted} characters, the manifest lists "
                f"{len(self.characters)}"
            )

    @classmethod
    def read(cls, directory: Path) -> KeyManifest:
        """Read *directory*'s manifest, refusing a policy or buckets this code does not speak."""
        path = directory / MANIFEST
        manifest = _ADAPTER.validate_json(path.read_bytes())
        if manifest.policy_version != POLICY_VERSION:
            raise ValueError(
                f"{path} was written for packing policy {manifest.policy_version}; this code "
                f"implements {POLICY_VERSION}"
            )
        if manifest.families != families():
            raise ValueError(f"{path} lists function families other than this code's buckets")
        return manifest

    def write(self, directory: Path) -> Path:
        """Write the manifest into *directory*; its path."""
        path = directory / MANIFEST
        path.write_bytes(_ADAPTER.dump_json(self, indent=2) + b"\n")
        return path

    def check_lexicon(self, lexicon: Lexicon) -> None:
        """Refuse a lexicon whose emission order is not the bundle's."""
        if tuple(lexicon.characters) != self.characters:
            raise ValueError(
                "the lexicon's emittable characters are not the bundle's: an arc's character "
                "would name another row of the head"
            )

    def check_legality(self, legality: Legality) -> None:
        """Refuse an automaton whose probe lattices differ from the bundle's.

        What a consumer's own automaton (the Rust one) must reproduce: the
        same arc count and digest for each probe key string.
        """
        theirs = legality_probes(legality, self.correct_syllables)
        for expected, got in zip(self.probes, theirs, strict=True):
            if expected != got:
                raise ValueError(
                    f"the legality lattice of {expected.keys!r} (tail {expected.unfinished_tail}) "
                    f"has {got.arcs} arcs and digest {got.digest}; the bundle pins "
                    f"{expected.arcs} and {expected.digest}"
                )


_ADAPTER = TypeAdapter(KeyManifest)
