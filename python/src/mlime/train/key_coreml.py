"""The key-level model through Core ML: its functions, the search's transition, a decoder (#266).

``mlime-coreml convert`` (the separate project ``python/coreml/``) turns a key bundle into three
multifunction models, ``context/``, ``keys/`` and ``decoder/``, each a ``.mlmodelc`` beside the
``manifest.json`` that names its functions and tensors. This module is the Python reference for
running them, the role the ONNX runtime had before it: the evaluation of the exported model, and
the contract the Rust session reproduces. It is imported on macOS only, by ``mlime eval keys
--coreml``; ``coremltools`` is a macOS dependency.

**The directory.** The converted models as ``mlime-coreml convert`` leaves them under its work
directory's ``models/`` (or as its archive unpacks), with the bundle's ``key-model.json`` and
``tokenizer.json`` beside them. The functions and their feeds are read from the directories'
manifests, not assumed: a tensor's name, dtype and shape come from there, and the buffers a
function is fed through are allocated from them.

**Per keystroke** (design section B). ``keys_W``, the smallest width that holds the keys, gives the
CTC logits and the decoder's cross keys and values, padded to ``N_MAX``. The decoder is a tree
recompute: ``tree_S`` computes the next-character logits of every node of a packed,
ancestor-closed set of prefixes (:func:`~mlime.train.key_tree.plan` makes the calls; their
packing is the contract with the Rust session). :class:`CoreMlTransition` answers the search from a
cache of those rows. At its construction it runs the **speculative pass** over the prefixes the
previous keystroke queried, one keystroke's search usually asking nearly what the last one did;
a query that is still not cached is a **miss**, and the uncached queries of one call are planned
together, ancestors included. Every node's row of every call is cached and the first value written
wins, so a row is the same wherever the search reads it. The logits are float16 off the engine;
the log-softmax is the host's, in float32, over the rows the search reads.

**The statistics** a transition keeps per keystroke are what the bucket set and the speculation
are judged by: the tree calls by bucket, the rows the speculative pass cached, the calls misses
made, and how many prefixes the search asked and how large their closure was.

**The decoder.** Without replay a record is one keystroke over all of its keys, no speculation.
With replay it is every key prefix in order, each a keystroke whose speculation is the last one's
queries, and the last keystroke's hypotheses are the record's. The prefixes before the last are
live input, so their tail rule is on, as in the product; the last keystroke is the record's
complete input under the decoder's own options, which is what the PyTorch decode searches.
"""

from __future__ import annotations

import statistics
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import coremltools as ct
import numpy as np
import numpy.typing as npt
import torch
from pydantic import TypeAdapter
from pydantic.dataclasses import dataclass as validated

from mlime.legality import Legality, LegalityOptions
from mlime.train.key_eval import (
    DEFAULT_BEAM,
    DEFAULT_RECORDS_PER_CHUNK,
    DEFAULT_TOP_K,
    DecodeTimes,
    KeyRecord,
    KeyRun,
    KeyWeights,
    build_key_collator,
    build_key_run,
    check_automaton,
    search_points,
)
from mlime.train.key_manifest import KeyManifest
from mlime.train.key_search import Hypothesis, Prefix
from mlime.train.key_tree import TreeCall, closure_size, plan
from mlime.train.keys import KeyCollator, KeyInputs
from mlime.train.lexicon import Lexicon
from mlime.train.samples import letter_id

Half = npt.NDArray[np.float16]
Float = npt.NDArray[np.float32]
Ints = npt.NDArray[np.int32]
Array = npt.NDArray[np.generic]

#: The directory each family converts into, and the model it holds (``mlime_coreml.spec``).
DIRECTORIES = {"context": "context", "keys": "keys", "tree": "decoder"}

#: The directory manifest every converted model directory carries.
DIRECTORY_MANIFEST = "manifest.json"

_DTYPES: dict[str, type[np.int32] | type[np.float16]] = {"int32": np.int32, "float16": np.float16}


# --- the converted directory ----------------------------------------------------------------


@validated(frozen=True)
class _TensorSpec:
    dtype: Literal["int32", "float16"]
    shape: tuple[int | Literal["length"], ...]

    def resolve(self, length: int) -> tuple[int, ...]:
        """The concrete shape in a function of *length*."""
        return tuple(length if dim == "length" else dim for dim in self.shape)

    @property
    def numpy_dtype(self) -> type[np.int32] | type[np.float16]:
        return _DTYPES[self.dtype]


@validated(frozen=True)
class _DirectoryManifest:
    """What of a model directory's ``manifest.json`` a loader reads; other fields are ignored."""

    family: Literal["context", "keys", "tree"]
    step: int
    policy_version: int
    functions: dict[str, int]
    inputs: dict[str, _TensorSpec]
    outputs: dict[str, _TensorSpec]
    n_max: int
    mask: float


_DIRECTORY_ADAPTER = TypeAdapter(_DirectoryManifest)


class _Function:
    """One function of a compiled family, loaded for *units*, and the buffers it is fed through.

    Core ML releases a call's input feature values later, on its own queue, and coremltools 9.0
    drops the numpy reference without holding the GIL; an array freed in between corrupts the
    heap. Every input is therefore a buffer this object allocates once and owns for its life,
    overwritten in place for each call, and every output is copied out before it is returned.
    """

    def __init__(
        self,
        compiled: Path,
        name: str,
        length: int,
        manifest: _DirectoryManifest,
        units: ct.ComputeUnit,
    ) -> None:
        self.name = name
        self._outputs = manifest.outputs
        self._inputs: dict[str, Array] = {
            tensor: np.zeros(spec.resolve(length), dtype=spec.numpy_dtype)
            for tensor, spec in manifest.inputs.items()
        }
        self._model = ct.models.CompiledMLModel(
            str(compiled), compute_units=units, function_name=name
        )

    def __call__(self, feeds: Mapping[str, Array]) -> dict[str, Array]:
        """Predict over *feeds*, which name exactly the function's inputs; the outputs, copied."""
        if feeds.keys() != self._inputs.keys():
            raise ValueError(f"{self.name} takes {sorted(self._inputs)}, was fed {sorted(feeds)}")
        for tensor, value in feeds.items():
            buffer = self._inputs[tensor]
            if value.shape != buffer.shape:
                raise ValueError(f"{self.name}.{tensor} is {buffer.shape}, was fed {value.shape}")
            np.copyto(buffer, value, casting="same_kind")
        out = self._model.predict(self._inputs)
        return {
            tensor: np.array(out[tensor], dtype=spec.numpy_dtype)
            for tensor, spec in self._outputs.items()
        }


# --- the feeds -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Encoding:
    """One keystroke's keys function: the CTC logits of its real keys and the decoder's memory.

    ``ctc_logits [n, E + 1]`` is float32, the blank last; ``cross_keys`` and ``cross_values``
    ``[layers, heads, N_MAX, head_dim]`` are the function's float16 outputs, unused slots zero,
    and ``cross_bias [1, 1, 1, N_MAX]`` masks them.
    """

    keys: int
    ctc_logits: Float
    cross_keys: Half
    cross_values: Half
    cross_bias: Half


@dataclass(frozen=True)
class Context:
    """The context function's output as the keys function reads it, with its mask.

    A record without a context is zeros under an all-zero mask, which the function's in-graph
    switch reads as no context.
    """

    rows: Half  # [1, tokens, H]
    mask: Half  # [1, tokens]


def key_row(manifest: KeyManifest, letters: Sequence[int], width: int) -> dict[str, Array]:
    """The keys function's row inputs: ``[CLS] [MASK]*n [SEP] [PAD]*`` at *width*."""
    count = len(letters)
    if count + 2 > width:
        raise ValueError(f"{count} keys do not fit a row of {width}")
    specials = manifest.specials
    ids = np.full(width, specials.pad, dtype=np.int32)
    ids[0], ids[count + 1] = specials.cls, specials.sep
    ids[1 : count + 1] = specials.mask
    key_letters = np.full(width, manifest.letter_pad, dtype=np.int32)
    key_letters[1 : count + 1] = letters
    attention = np.zeros(width, dtype=np.float16)
    attention[: count + 2] = 1.0
    return {
        "input_ids": ids[None],
        "key_letters": key_letters[None],
        "attention_mask": attention[None],
    }


def tree_row(manifest: KeyManifest, call: TreeCall) -> dict[str, Array]:
    """The tree function's node inputs for *call*: the reader ids, depths and additive bias.

    A node's token is its prefix's last character as the reader's id, ``<bos>`` for the root and
    for padding; a node sees its ancestors and itself, a padding row only itself.
    """
    size = call.bucket
    reader_ids = np.full(size, manifest.reader_bos, dtype=np.int32)
    positions = np.zeros(size, dtype=np.int32)
    visible = np.eye(size, dtype=np.bool_)
    for row, node in enumerate(call.nodes):
        if node:
            reader_ids[row] = manifest.reader_ids[node[-1]]
        positions[row] = len(node)
        if call.parents[row] >= 0:
            visible[row] |= visible[call.parents[row]]
    bias = np.where(visible, 0.0, manifest.neg).astype(np.float16)
    return {
        "reader_ids": reader_ids[None],
        "positions": positions[None],
        "tree_bias": bias[None, None],
    }


class CoreMlKeyModel:
    """The converted key model: three families, each function loaded when first called."""

    def __init__(self, directory: Path, units: ct.ComputeUnit = ct.ComputeUnit.CPU_AND_NE) -> None:
        self.directory = directory
        self.units = units
        self.manifest = KeyManifest.read(directory)
        self._manifests: dict[str, _DirectoryManifest] = {}
        for family in self.manifest.families:
            held = _DIRECTORY_ADAPTER.validate_json(
                (directory / DIRECTORIES[family.name] / DIRECTORY_MANIFEST).read_bytes()
            )
            expected = {function.name: function.length for function in family.functions}
            if (
                held.family != family.name
                or held.functions != expected
                or held.step != self.manifest.step
                or held.policy_version != self.manifest.policy_version
                or held.n_max != self.manifest.n_max
                or held.mask != self.manifest.neg
            ):
                raise ValueError(
                    f"{directory / DIRECTORIES[family.name]} was not converted from this "
                    f"bundle: its functions, step, policy or masks differ from key-model.json"
                )
            self._manifests[family.name] = held
        self._loaded: dict[str, _Function] = {}

    @property
    def emittable(self) -> int:
        """Characters the head and the decoder emit."""
        return len(self.manifest.characters)

    def function(self, family: Literal["context", "keys", "tree"], name: str) -> _Function:
        """*family*'s function *name*, loaded on first use."""
        if name not in self._loaded:
            held = self._manifests[family]
            compiled = self.directory / DIRECTORIES[family] / f"{DIRECTORIES[family]}.mlmodelc"
            self._loaded[name] = _Function(compiled, name, held.functions[name], held, self.units)
        return self._loaded[name]

    def context(self, inputs: KeyInputs) -> Context:
        """The context function over a one-row collate's context, padded to its length."""
        tokens = self.manifest.context_tokens
        [function] = self.manifest.families[0].functions
        hidden = self.manifest.dims.fill.hidden
        real = int(inputs.context_ids.shape[1])
        if inputs.size != 1 or real > tokens:
            raise ValueError(
                f"the context function takes one row of at most {tokens} tokens, got "
                f"{inputs.size} of {real}"
            )
        if not bool(inputs.has_context[0]):
            return Context(
                np.zeros((1, tokens, hidden), dtype=np.float16), np.zeros((1, tokens), np.float16)
            )
        ids = np.full((1, tokens), self.manifest.specials.pad, dtype=np.int32)
        ids[0, :real] = inputs.context_ids[0].numpy()
        mask = np.zeros((1, tokens), dtype=np.float16)
        mask[0, :real] = inputs.context_mask[0].numpy()
        out = self.function("context", function.name)({"context_ids": ids, "context_mask": mask})
        return Context(out["context"].astype(np.float16), mask)

    def encode(self, letters: Sequence[int], context: Context) -> Encoding:
        """The keys function, at the smallest width that holds *letters*, over the context."""
        manifest = self.manifest
        keys = manifest.families[1]
        fits = [function for function in keys.functions if function.capacity >= len(letters)]
        if not fits:
            largest = keys.functions[-1].capacity
            raise ValueError(f"{len(letters)} keys do not fit the largest function ({largest})")
        function = fits[0]
        out = self.function("keys", function.name)(
            {
                **key_row(manifest, letters, function.length),
                "context": context.rows,
                "context_mask": context.mask,
            }
        )
        bias = np.full((1, 1, 1, manifest.n_max), manifest.neg, dtype=np.float16)
        bias[..., : len(letters)] = 0.0
        return Encoding(
            keys=len(letters),
            ctc_logits=out["ctc_logits"][0, 1 : len(letters) + 1].astype(np.float32),
            cross_keys=out["cross_keys"].astype(np.float16),
            cross_values=out["cross_values"].astype(np.float16),
            cross_bias=bias,
        )

    def tree(self, call: TreeCall, encoding: Encoding) -> Half:
        """One tree call: every node's next-character logits, ``[bucket, E]`` float16."""
        function = self.function("tree", f"tree_{call.bucket}")
        out = function(
            {
                **tree_row(self.manifest, call),
                "cross_keys": encoding.cross_keys,
                "cross_values": encoding.cross_values,
                "cross_bias": encoding.cross_bias,
            }
        )
        logits: Half = out["logits"][0]
        return logits


# --- the transition --------------------------------------------------------------------------


@dataclass
class KeystrokeStatistics:
    """What one keystroke's transition did."""

    #: The keys the keystroke searched.
    keys: int
    #: Tree calls made, by bucket: the speculative pass's and the misses'.
    calls: Counter[int] = field(default_factory=Counter)
    #: Rows the speculative pass cached: the closure of the previous keystroke's queries.
    speculative_rows: int = 0
    #: Tree calls the misses made.
    miss_calls: int = 0
    #: Distinct prefixes the search asked, and the nodes of their ancestor closure.
    queried: int = 0
    closure: int = 0
    #: Rows handed to the search: every ask, so ``answered == queried`` unless a prefix repeats.
    answered: int = 0

    @property
    def tree_calls(self) -> int:
        """Tree calls of every bucket."""
        return sum(self.calls.values())


def log_softmax(logits: npt.NDArray[np.generic]) -> Float:
    """The row-wise log-softmax of *logits* in float32."""
    shifted: Float = logits.astype(np.float32)
    shifted -= shifted.max(axis=-1, keepdims=True)
    shifted -= np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return shifted


class CoreMlTransition:
    """The search's transition over one keystroke's encoding, answered from a row cache.

    One record: every query's record index is 0. *speculate* are the prefixes the previous
    keystroke queried; their tree calls run before the search asks anything. A query whose prefix
    has no cached row is a miss: the uncached queries of that call are planned together and every
    node of every call is cached, the first value written staying.
    """

    def __init__(
        self,
        model: CoreMlKeyModel,
        encoding: Encoding,
        times: DecodeTimes,
        speculate: Sequence[Prefix] = (),
    ) -> None:
        self.model = model
        self.encoding = encoding
        self.times = times
        self.statistics = KeystrokeStatistics(keys=encoding.keys)
        #: ``prefix -> (the call's logits, the node's row)``.
        self._rows: dict[Prefix, tuple[Half, int]] = {}
        self._queried: dict[Prefix, None] = {}
        started = time.perf_counter()
        self._run(speculate)
        self.statistics.speculative_rows = len(self._rows)
        times.transition += time.perf_counter() - started

    @property
    def queried(self) -> list[Prefix]:
        """The distinct prefixes the search has asked so far, in the order it asked them."""
        return list(self._queried)

    def _run(self, requests: Sequence[Prefix]) -> int:
        """Plan and run *requests*; cache every node. The number of calls made."""
        calls = plan(requests)
        for call in calls:
            logits = self.model.tree(call, self.encoding)
            self.statistics.calls[call.bucket] += 1
            for row, node in enumerate(call.nodes):
                self._rows.setdefault(node, (logits, row))
        return len(calls)

    def __call__(self, records: Sequence[int], prefixes: Sequence[Prefix]) -> Float:
        started = time.perf_counter()
        if any(records):
            raise ValueError("a Core ML transition serves one record")
        missing = [prefix for prefix in prefixes if prefix not in self._rows]
        if missing:
            self.statistics.miss_calls += self._run(missing)
        for prefix in prefixes:
            self._queried.setdefault(prefix)
        self.statistics.answered += len(prefixes)
        rows = log_softmax(
            np.stack([self._rows[prefix][0][self._rows[prefix][1]] for prefix in prefixes])
        )
        self.times.transition += time.perf_counter() - started
        return rows

    def finish(self) -> KeystrokeStatistics:
        """The keystroke's statistics, with what the search queried counted."""
        self.statistics.queried = len(self._queried)
        self.statistics.closure = closure_size(self.queried)
        return self.statistics


# --- the decoder -----------------------------------------------------------------------------


def _spread(values: Sequence[int]) -> str:
    """The median and the p95 (the smallest value at least 95% of them are within)."""
    return (
        f"median {statistics.median(values):g}, "
        f"p95 {float(np.percentile(values, 95, method='higher')):g}"
    )


@dataclass
class DecodeStatistics:
    """Every keystroke's statistics over a decode."""

    keystrokes: list[KeystrokeStatistics] = field(default_factory=list)

    def render(self) -> str:
        """The totals, and the per-keystroke median and p95 of tree calls and miss calls."""
        count = len(self.keystrokes)
        if not count:
            return "core ml transition: no keystroke decoded"
        buckets: Counter[int] = Counter()
        for each in self.keystrokes:
            buckets.update(each.calls)
        calls = [each.tree_calls for each in self.keystrokes]
        misses = [each.miss_calls for each in self.keystrokes]
        by_bucket = ", ".join(f"{size}: {buckets[size]}" for size in sorted(buckets))
        return "\n".join(
            [
                f"core ml transition: {count} keystrokes",
                f"  tree calls: {sum(calls)} total ({by_bucket}); per keystroke {_spread(calls)}",
                f"  miss calls: {sum(misses)} total; per keystroke {_spread(misses)}",
                f"  speculative rows cached: {sum(e.speculative_rows for e in self.keystrokes)}",
                f"  queried prefixes: {sum(e.queried for e in self.keystrokes)}, "
                f"closure nodes: {sum(e.closure for e in self.keystrokes)}, "
                f"rows answered: {sum(e.answered for e in self.keystrokes)}",
            ]
        )


@dataclass(frozen=True)
class CoreMlKeyDecoder:
    """:class:`~mlime.train.key_eval.KeyDecoder`'s counterpart over the converted model.

    Each record is decoded on its own, as the product decodes: the keys function and the search
    per keystroke. *replay* decodes every key prefix as a keystroke, speculation carried from one
    to the next, and takes the last keystroke's hypotheses; without it the record is one
    keystroke over all its keys.
    """

    model: CoreMlKeyModel
    lexicon: Lexicon
    collator: KeyCollator
    legality: Legality
    options: LegalityOptions
    replay: bool = False
    beam: int = DEFAULT_BEAM
    top_k: int = DEFAULT_TOP_K
    times: DecodeTimes = field(default_factory=DecodeTimes)
    statistics: DecodeStatistics = field(default_factory=DecodeStatistics)

    def __post_init__(self) -> None:
        self.model.manifest.check_lexicon(self.lexicon)
        check_automaton(self.legality, self.lexicon, self.options)

    def decode(
        self, records: Sequence[KeyRecord], grid: Sequence[KeyWeights]
    ) -> list[list[list[Hypothesis]]]:
        """Each grid point's hypotheses for each record, best first: ``[point][record]``."""
        found: list[list[list[Hypothesis]]] = [[] for _ in grid]
        for record in records:
            for point, hypotheses in enumerate(self.record(record, grid)):
                found[point].append(hypotheses)
            self.times.records += 1
            self.times.searches += len(grid)
        return found

    def record(self, record: KeyRecord, grid: Sequence[KeyWeights]) -> list[list[Hypothesis]]:
        """One record's hypotheses at each grid point.

        The keys function of a keystroke does not depend on the weights, so it runs once and
        every point searches it; a point's speculation is its own last keystroke's queries.
        """
        started = time.perf_counter()
        context = self.model.context(self.collator.inputs([record.keys], [record.context]))
        self.times.encode += time.perf_counter() - started
        letters = [letter_id(key) for key in record.keys]
        counts = range(1, len(letters) + 1) if self.replay else [len(letters)]
        asked: list[list[Prefix]] = [[] for _ in grid]
        results: list[list[Hypothesis]] = [[] for _ in grid]
        for count in counts:
            options = (
                self.options
                if count == len(letters)
                else replace(self.options, unfinished_tail=True)
            )
            started = time.perf_counter()
            lattice = self.legality.lattice(record.keys[:count], options)
            encoding = self.model.encode(letters[:count], context)
            log_probs = log_softmax(encoding.ctc_logits)
            self.times.lattice += time.perf_counter() - started
            for point, weights in enumerate(grid):
                transition = CoreMlTransition(self.model, encoding, self.times, asked[point])
                [[results[point]]] = search_points(
                    [lattice],
                    torch.from_numpy(log_probs).unsqueeze(0),
                    transition,
                    [weights],
                    beam=self.beam,
                    top_k=self.top_k,
                    times=self.times,
                )
                self.statistics.keystrokes.append(transition.finish())
                asked[point] = transition.queried
        return results


def load_coreml_key_run(
    directory: Path,
    eval_set: Path,
    *,
    char_table: Path,
    typo_table: Path,
    syllables: Path,
    with_context: bool,
    dev_share: float,
    replay: bool,
    beam: int = DEFAULT_BEAM,
    top_k: int = DEFAULT_TOP_K,
    records_per_chunk: int = DEFAULT_RECORDS_PER_CHUNK,
    limit: int | None = None,
) -> tuple[KeyRun, CoreMlKeyDecoder, int]:
    """An evaluation of the converted model in *directory* on *eval_set*: run, decoder and step.

    The lexicon and the automaton are built from the tables the way
    :func:`~mlime.train.key_eval.load_key_run` builds them, and the bundle's manifest is checked
    against them: the emission order, and the legality lattices of its probe key strings.
    """
    from mlime.train.run import Vocabularies

    model = CoreMlKeyModel(directory)
    manifest = model.manifest
    vocabularies = Vocabularies.load(char_table, manifest.base_model, None)
    legality = vocabularies.legality()
    manifest.check_lexicon(vocabularies.lexicon)
    manifest.check_legality(legality)
    decoder = CoreMlKeyDecoder(
        model=model,
        lexicon=vocabularies.lexicon,
        collator=build_key_collator(
            vocabularies.spans,
            vocabularies.lexicon,
            vocabularies.tokenizer,
            vocabularies.context_tokenizer,
            with_context=with_context,
            max_context_tokens=manifest.context_tokens,
        ),
        legality=legality,
        options=LegalityOptions(
            unfinished_tail=False, correct_syllables=manifest.correct_syllables
        ),
        replay=replay,
        beam=beam,
        top_k=top_k,
    )
    run = build_key_run(
        decoder,
        eval_set,
        char_table=char_table,
        typo_table=typo_table,
        syllables=syllables,
        dev_share=dev_share,
        records_per_chunk=records_per_chunk,
        limit=limit,
    )
    return run, decoder, manifest.step
