"""Read a key-model bundle: ``key-model.json``, ``weights.safetensors``, ``reference.npz``.

The bundle directory is the only thing this project shares with mlime (whose torch is newer than
the one coremltools is locked with), so the manifest's shape is spelled out here as plain
dataclasses and the JSON is read strictly: a field this code does not know, or one it expects and
does not find, is an error.
"""

from __future__ import annotations

import json
import types
import typing
from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import torch
from safetensors import safe_open

from mlime_coreml import spec
from mlime_coreml.spec import Family

MANIFEST = "key-model.json"
WEIGHTS = "weights.safetensors"
TOKENIZER = "tokenizer.json"
REFERENCE = "reference.npz"

#: A prefix: the emission indices emitted so far.
Prefix = tuple[int, ...]


@dataclass(frozen=True)
class Function:
    """One function of a family: its name, static length and what a call can carry."""

    name: str
    length: int
    capacity: int


@dataclass(frozen=True)
class FamilySpec:
    """A function family: its functions, smallest first."""

    name: Family
    functions: tuple[Function, ...]


@dataclass(frozen=True)
class Specials:
    """The tokenizer's special ids the key row is built from."""

    pad: int
    cls: int
    sep: int
    mask: int


@dataclass(frozen=True)
class TowerDims:
    """A BERT tower's dimensions."""

    layers: int
    heads: int
    hidden: int
    intermediate: int
    positions: int
    vocabulary: int
    layer_norm_eps: float


@dataclass(frozen=True)
class KeyFrontDims:
    """The key front's causal stack."""

    layers: int
    heads: int
    feedforward: int
    positions: int


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class ModelDims:
    """Every dimension the three families' modules are built from."""

    context: TowerDims
    fill: TowerDims
    gated: int
    key_front: KeyFrontDims
    decoder: DecoderDims
    emitted: int


@dataclass(frozen=True)
class TensorEntry:
    """One tensor of ``weights.safetensors`` and the families that read it."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    families: tuple[Family, ...]


@dataclass(frozen=True)
class ReferenceProbe:
    """One probe of ``reference.npz``: a key string, its context, the tree sets asked."""

    name: str
    keys: str
    context: str | None
    tree: tuple[tuple[Prefix, ...], ...]


@dataclass(frozen=True)
class LegalityProbe:
    """One probe lattice of the manifest; opaque here, carried through to the package."""

    keys: str
    unfinished_tail: bool
    arcs: int
    digest: str


@dataclass(frozen=True)
class Manifest:
    """``key-model.json`` as this project reads it."""

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
    families: tuple[FamilySpec, ...]
    tensors: tuple[TensorEntry, ...]
    reference: tuple[ReferenceProbe, ...]

    def family(self, name: Family) -> FamilySpec:
        """The family called *name*."""
        return next(family for family in self.families if family.name == name)


def _build(kind: Any, data: Any, where: str) -> Any:
    """*data* as a *kind*: dataclasses, tuples, literals and scalars, nothing unknown."""
    origin = typing.get_origin(kind)
    if is_dataclass(kind):
        if not isinstance(data, dict):
            raise ValueError(f"{where}: expected an object, found {type(data).__name__}")
        hints = typing.get_type_hints(kind)
        names = {field.name for field in fields(kind)}
        if set(data) != names:
            raise ValueError(
                f"{where}: fields {sorted(set(data) ^ names)} are missing or unknown to this code"
            )
        construct = cast(Callable[..., Any], kind)
        return construct(
            **{name: _build(hints[name], data[name], f"{where}.{name}") for name in names}
        )
    if origin is tuple:
        [item, *_] = typing.get_args(kind)
        if not isinstance(data, list):
            raise ValueError(f"{where}: expected a list, found {type(data).__name__}")
        return tuple(_build(item, value, f"{where}[{index}]") for index, value in enumerate(data))
    if origin is Literal:
        if data not in typing.get_args(kind):
            raise ValueError(f"{where}: {data!r} is not one of {typing.get_args(kind)}")
        return data
    if origin is types.UnionType:
        arms = [arm for arm in typing.get_args(kind) if arm is not type(None)]
        return None if data is None else _build(arms[0], data, where)
    if kind is float and isinstance(data, int | float) and not isinstance(data, bool):
        return float(data)
    if not isinstance(data, kind) or (kind is int and isinstance(data, bool)):
        raise ValueError(f"{where}: expected {kind.__name__}, found {data!r}")
    return data


@dataclass(frozen=True)
class Bundle:
    """A bundle read from disk: its manifest, its float32 weights and where its reference is."""

    directory: Path
    manifest: Manifest
    weights: dict[str, torch.Tensor]

    def family_weights(self, family: Family) -> dict[str, torch.Tensor]:
        """The tensors *family* reads, by their bundle names."""
        return {
            entry.name: self.weights[entry.name]
            for entry in self.manifest.tensors
            if family in entry.families
        }

    def reference(self) -> dict[str, npt.NDArray[np.generic]]:
        """``reference.npz`` as a dict of arrays."""
        with np.load(self.directory / REFERENCE) as archive:
            return {name: archive[name] for name in archive.files}


def expected_families() -> tuple[FamilySpec, ...]:
    """The families and functions this code's buckets define."""
    sizes: dict[Family, tuple[int, ...]] = {
        "context": spec.CONTEXT_BUCKETS,
        "keys": spec.KEYS_WIDTHS,
        "tree": spec.TREE_BUCKETS,
    }
    sentinels = {"context": spec.KEY_SENTINELS, "keys": spec.KEY_SENTINELS, "tree": 0}
    return tuple(
        FamilySpec(
            family,
            tuple(
                Function(f"{spec.FUNCTION_PREFIX[family]}_{size}", size, size - sentinels[family])
                for size in sizes[family]
            ),
        )
        for family in spec.FAMILIES
    )


def read_bundle(directory: Path) -> Bundle:
    """Read and check the bundle in *directory*.

    Refused: a manifest with unknown or missing fields, one written for another packing policy,
    other buckets or another mask than this code's, and weights whose names, shapes or dtypes
    differ from what the manifest lists.
    """
    path = directory / MANIFEST
    manifest = cast(Manifest, _build(Manifest, json.loads(path.read_bytes()), "manifest"))
    if manifest.policy_version != spec.POLICY_VERSION:
        raise ValueError(
            f"{path} was written for packing policy {manifest.policy_version}; this code "
            f"implements {spec.POLICY_VERSION}"
        )
    if manifest.families != expected_families():
        raise ValueError(f"{path} lists function families other than this code's buckets")
    if (manifest.n_max, manifest.neg) != (spec.N_MAX, spec.NEG):
        raise ValueError(
            f"{path} pins n_max {manifest.n_max} and mask {manifest.neg}; this code's are "
            f"{spec.N_MAX} and {spec.NEG}"
        )
    if manifest.context_tokens not in spec.CONTEXT_BUCKETS:
        raise ValueError(f"{path}: the context function holds {spec.CONTEXT_BUCKETS} tokens")
    weights: dict[str, torch.Tensor] = {}
    with safe_open(str(directory / WEIGHTS), framework="pt") as archive:
        held = set(archive.keys())
        listed = {entry.name for entry in manifest.tensors}
        if held != listed:
            raise ValueError(
                f"{WEIGHTS} and the manifest disagree on the tensors: {sorted(held ^ listed)[:6]}"
            )
        for entry in manifest.tensors:
            tensor = archive.get_tensor(entry.name)
            dtype = str(tensor.dtype).removeprefix("torch.")
            if tuple(tensor.shape) != entry.shape or dtype != entry.dtype:
                raise ValueError(
                    f"{entry.name} is {dtype}{list(tensor.shape)}; the manifest lists "
                    f"{entry.dtype}{list(entry.shape)}"
                )
            weights[entry.name] = tensor
    return Bundle(directory, manifest, weights)
