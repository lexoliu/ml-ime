"""The float32 port, run and checked: against the bundle's reference and against itself at padding.

Everything here is plain torch over :mod:`mlime_coreml.modules`; nothing imports coremltools, so
the checks run on any host. They are the ground the Core ML verification stands on: a Core ML
model is compared with this port, and this port is first compared with what the model's own
framework computed (``reference.npz``).
"""

from __future__ import annotations

import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from mlime_coreml import spec
from mlime_coreml.bundle import Bundle
from mlime_coreml.feeds import (
    Array,
    ContextFeed,
    CrossFeed,
    context_row,
    key_row,
    random_context_ids,
    random_letters,
    random_tree,
    tree_row,
)
from mlime_coreml.modules import build_module
from mlime_coreml.spec import Family

Float = npt.NDArray[np.float32]


@dataclass(frozen=True)
class Check:
    """One comparison: what was compared, how far apart the two sides were, what was allowed."""

    name: str
    max_abs: float
    tolerance: float
    #: The largest raw difference, when ``max_abs`` is a tolerance-adjusted excess.
    raw: float | None = None

    @property
    def ok(self) -> bool:
        """Whether the gap is within the tolerance (a NaN gap is not)."""
        return bool(self.max_abs <= self.tolerance)

    def as_json(self) -> dict[str, object]:
        """The check as the verification report records it."""
        return {
            "name": self.name,
            "gap": self.max_abs,
            "tolerance": self.tolerance,
            "max_abs_difference": self.max_abs if self.raw is None else self.raw,
            "ok": self.ok,
        }


def _tensor(array: Array) -> torch.Tensor:
    """A torch tensor that owns a copy of *array*."""
    return torch.from_numpy(np.array(array))


def _gap(got: Array, want: Array) -> float:
    if got.shape != want.shape:
        raise ValueError(f"shapes differ: {got.shape} against {want.shape}")
    if got.size == 0:
        return 0.0
    return float(np.max(np.abs(got.astype(np.float64) - want.astype(np.float64))))


def run_context(module: nn.Module, feed: Mapping[str, Array]) -> Float:
    """The context function over *feed*: ``[L, H]`` rows."""
    with torch.no_grad():
        out = module(_tensor(feed["context_ids"]), _tensor(feed["context_mask"]))
    return np.array(out[0].numpy(), dtype=np.float32)


def run_keys(
    module: nn.Module, feed: Mapping[str, Array], context: ContextFeed
) -> tuple[Float, Float, Float]:
    """The keys function: ``[W, E + 1]`` logits, and the cross keys and values."""
    with torch.no_grad():
        logits, keys, values = module(
            _tensor(feed["input_ids"]),
            _tensor(feed["key_letters"]),
            _tensor(feed["attention_mask"]),
            _tensor(context.rows[None]),
            _tensor(context.mask[None]),
        )
    return (
        np.array(logits[0].numpy(), dtype=np.float32),
        np.array(keys.numpy(), dtype=np.float32),
        np.array(values.numpy(), dtype=np.float32),
    )


def run_tree(module: nn.Module, feed: Mapping[str, Array], cross: CrossFeed) -> Float:
    """The tree function: ``[S, E]`` logits."""
    with torch.no_grad():
        out = module(
            _tensor(feed["reader_ids"]),
            _tensor(feed["positions"]),
            _tensor(feed["tree_bias"]),
            _tensor(cross.keys),
            _tensor(cross.values),
            _tensor(cross.bias),
        )
    return np.array(out[0].numpy(), dtype=np.float32)


def cross_of(reference: Mapping[str, Array], probe: str) -> CrossFeed:
    """A probe's cross keys, values and bias, as the reference recorded them."""
    return CrossFeed(
        np.asarray(reference[f"{probe}.cross_keys"], dtype=np.float32),
        np.asarray(reference[f"{probe}.cross_values"], dtype=np.float32),
        np.asarray(reference[f"{probe}.cross_bias"], dtype=np.float32),
    )


def context_of(reference: Mapping[str, Array], probe: str) -> ContextFeed:
    """A probe's context rows and mask, as the reference recorded them."""
    return ContextFeed(
        np.asarray(reference[f"{probe}.context"], dtype=np.float32),
        np.asarray(reference[f"{probe}.context_mask"], dtype=np.float32),
    )


_CALL = re.compile(r"^(?P<probe>.+)\.tree(?P<set>\d+)\.call(?P<call>\d+)\.reader_ids$")


def tree_calls(reference: Mapping[str, Array], probe: str) -> list[str]:
    """The prefixes (``<probe>.tree<k>.call<c>``) of the tree calls recorded for *probe*."""
    found = []
    for name in reference:
        match = _CALL.match(name)
        if match and match["probe"] == probe:
            found.append(name.removesuffix(".reader_ids"))
    return sorted(found)


def check_port(bundle: Bundle) -> list[Check]:
    """The float32 port against ``reference.npz``: every function over every probe."""
    reference = bundle.reference()
    manifest = bundle.manifest
    context = build_module(bundle, "context", manifest.context_tokens)
    checks: list[Check] = []
    tree = build_module(bundle, "tree", 0)
    for probe in manifest.reference:
        name = probe.name
        mask = np.asarray(reference[f"{name}.context_mask"], dtype=np.float32)
        real = int(mask.sum())
        if real:
            feed = {
                "context_ids": np.asarray(reference[f"{name}.context_ids"])[None],
                "context_mask": mask[None],
            }
            rows = run_context(context, feed)
            want = np.asarray(reference[f"{name}.context"], dtype=np.float32)
            checks.append(Check(f"{name}.context", _gap(rows[:real], want[:real]), spec.PORT_ATOL))
        ids = np.asarray(reference[f"{name}.input_ids"])
        keys = build_module(bundle, "keys", len(ids))
        feed = {
            "input_ids": ids[None],
            "key_letters": np.asarray(reference[f"{name}.key_letters"])[None],
            "attention_mask": np.asarray(reference[f"{name}.attention_mask"])[None],
        }
        logits, cross_keys, cross_values = run_keys(keys, feed, context_of(reference, name))
        wanted = np.asarray(reference[f"{name}.ctc_logits"], dtype=np.float32)
        count = wanted.shape[0]
        checks.append(
            Check(f"{name}.ctc_logits", _gap(logits[1 : count + 1], wanted), spec.PORT_ATOL)
        )
        for what, got in (("cross_keys", cross_keys), ("cross_values", cross_values)):
            checks.append(
                Check(
                    f"{name}.{what}",
                    _gap(got, np.asarray(reference[f"{name}.{what}"], dtype=np.float32)),
                    spec.PORT_ATOL,
                )
            )
        del keys
        cross = cross_of(reference, name)
        for call in tree_calls(reference, name):
            feed = {
                "reader_ids": np.asarray(reference[f"{call}.reader_ids"])[None],
                "positions": np.asarray(reference[f"{call}.positions"])[None],
                "tree_bias": np.asarray(reference[f"{call}.tree_bias"], dtype=np.float32)[
                    None, None
                ],
            }
            got = run_tree(tree, feed, cross)
            checks.append(
                Check(
                    f"{call}.logits",
                    _gap(got, np.asarray(reference[f"{call}.logits"], dtype=np.float32)),
                    spec.PORT_ATOL,
                )
            )
            answers = np.asarray(reference[f"{call}.answers"]).reshape(-1, 2)
            if len(answers):
                rows = torch.from_numpy(got[answers[:, 1]]).log_softmax(dim=-1).numpy()
                checks.append(
                    Check(
                        f"{call}.log_probs",
                        _gap(rows, np.asarray(reference[f"{call}.log_probs"], dtype=np.float32)),
                        spec.PORT_ATOL,
                    )
                )
    return checks


@dataclass(frozen=True)
class Edge:
    """A padding case: how many real rows a function of *length* carries."""

    family: Family
    length: int
    real: int


def edges(bundle: Bundle) -> list[Edge]:
    """The bucket edges of every function: its fullest case and its smallest.

    The smallest real count a bucket takes is one past what the bucket below holds (the first
    bucket's is the least the family supports: one key, two context tokens, one node).
    """
    manifest = bundle.manifest
    found: list[Edge] = []
    for family in spec.FAMILIES:
        previous = 0
        for function in manifest.family(family).functions:
            # A context's real count includes its two sentinels; the manifest's capacity does not.
            capacity = function.length if family == "context" else function.capacity
            least = {"context": 2, "keys": 1, "tree": 1}[family]
            smallest = previous + 1 if previous else least
            found.append(Edge(family, function.length, capacity))
            if smallest != capacity:
                found.append(Edge(family, function.length, smallest))
            previous = capacity
    return found


@dataclass(frozen=True)
class Case:
    """One edge's inputs at its bucket and at its exact length, and what is real in each."""

    edge: Edge
    padded: dict[str, Array]
    exact: dict[str, Array]
    context: ContextFeed | None
    cross: CrossFeed | None
    exact_length: int


def make_case(bundle: Bundle, edge: Edge, seed: int = 0) -> Case:
    """Random inputs for *edge*, built once at its bucket and once at its exact length."""
    manifest = bundle.manifest
    # Not seeded by the length: the same real rows at two bucket lengths are the same inputs.
    rng = random.Random(f"{edge.family}-{edge.real}-{seed}")
    reference = bundle.reference()
    first = manifest.reference[0].name
    if edge.family == "context":
        ids = random_context_ids(manifest, edge.real, rng)
        return Case(
            edge,
            context_row(manifest, ids, edge.length),
            context_row(manifest, ids, edge.real),
            None,
            None,
            edge.real,
        )
    if edge.family == "keys":
        letters = random_letters(manifest, edge.real, rng)
        probe = next(
            (p.name for p in manifest.reference if reference[f"{p.name}.context_mask"].any()),
            first,
        )
        return Case(
            edge,
            key_row(manifest, letters, edge.length),
            key_row(manifest, letters, edge.real + spec.KEY_SENTINELS),
            context_of(reference, probe),
            None,
            edge.real + spec.KEY_SENTINELS,
        )
    nodes = random_tree(manifest, edge.real, rng)
    return Case(
        edge,
        tree_row(manifest, nodes, edge.length),
        tree_row(manifest, nodes, edge.real),
        None,
        cross_of(reference, first),
        edge.real,
    )


def real_parts(family: Family, real: int, outputs: Sequence[Float]) -> dict[str, Float]:
    """A function's outputs (batch dropped) cut to what a host reads, by output name.

    A context's real rows; a keys function's logits for keys ``1..n`` and its cross keys and
    values over the ``n`` real slots; a tree's real node rows. *outputs* are in the family's
    signature order.
    """
    if family == "context":
        return {"context": outputs[0][:real]}
    if family == "keys":
        logits, keys, values = outputs
        return {
            "ctc_logits": logits[1 : real + 1],
            "cross_keys": keys[:, :, :real],
            "cross_values": values[:, :, :real],
        }
    return {"logits": outputs[0][:real]}


def run_case(
    module: nn.Module, family: Family, case: Case, feed: Mapping[str, Array]
) -> dict[str, Float]:
    """The float32 port over *feed* of *case*, cut to the case's real rows (:func:`real_parts`)."""
    edge = case.edge
    if family == "context":
        return real_parts(family, edge.real, [run_context(module, feed)])
    if family == "keys":
        assert case.context is not None
        return real_parts(family, edge.real, run_keys(module, feed, case.context))
    assert case.cross is not None
    return real_parts(family, edge.real, [run_tree(module, feed, case.cross)])


def check_padding(bundle: Bundle) -> list[Check]:
    """The float32 port at each bucket's width against the exact-length run, at every edge."""
    checks: list[Check] = []
    modules: dict[tuple[Family, int], nn.Module] = {}

    def module_for(family: Family, length: int) -> nn.Module:
        key = (family, 0 if family == "tree" else length)
        if key not in modules:
            modules.clear()
            modules[key] = build_module(bundle, family, length)
        return modules[key]

    for edge in edges(bundle):
        case = make_case(bundle, edge)
        padded = run_case(module_for(edge.family, edge.length), edge.family, case, case.padded)
        exact = run_case(module_for(edge.family, case.exact_length), edge.family, case, case.exact)
        for part, rows in padded.items():
            checks.append(
                Check(
                    f"padding.{edge.family}_{edge.length}.real{edge.real}.{part}",
                    _gap(rows, exact[part]),
                    spec.PADDING_ATOL_FP32,
                )
            )
    return checks
