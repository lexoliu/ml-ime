"""The exported key-level model, run with onnxruntime: the reference ``ime-neural`` is held to.

:func:`~mlime.train.key_export.export_key_model` writes three graphs and
``key-model.json``. This module is the consumer of that directory in Python:
:class:`KeyManifest` reads and checks the manifest, :class:`OnnxKeyModel` runs
the graphs, and :class:`OnnxTransition` is the search's transition
(:class:`~mlime.train.key_search.Transition`) over a page pool of the decoder's
keys and values, so :func:`~mlime.train.key_search.search` -- the very function
the PyTorch evaluation calls -- decodes over the exported graphs.

**A keystroke.** ``context.onnx`` runs once per context text; ``keys.onnx`` runs
once per keystroke over one row, and gives each key's CTC log-probabilities
and the decoder's per-layer cross-attention keys and values of the key states
(:class:`KeyEncoding`). The fill tower is bidirectional over the keys, so every
key state changes with a keystroke and no decoder state survives one: a new
:class:`OnnxTransition` starts an empty pool.

**The page pool.** ``decoder.onnx`` runs one row per queried prefix. A prefix
of ``L`` characters is the decoder's token ``c_L`` (``<bos>``, the emission
index ``E``, for the empty prefix) at position ``L``; the row produces that
position's key and value, one page of ``[layers, heads, head_dim]``, and the
distribution of the character after the prefix. Its history is the pages of
its ``L`` proper prefixes. The pool is ``[pages, layers, heads, head_dim]`` with
page 0 the always-masked scratch (the char LM's layout), and a dict maps each
prefix whose row ran to its page. A prefix whose ancestors have no page yet is
carried in the same call as them: the ``tree_mask`` lets a row read the fresh
positions of the rows before it in the call, so a chain of prefixes costs one
call.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
import torch

from mlime.legality import KeyLattice, Legality, LegalityOptions, lattice_digest
from mlime.train.key_eval import (
    DEFAULT_BEAM,
    DEFAULT_RECORDS_PER_CHUNK,
    DEFAULT_TOP_K,
    DecodeTimes,
    KeyRecord,
    KeyRun,
    KeyWeights,
    build_key_collator,
    search_points,
)
from mlime.train.key_search import Hypothesis, LegalityVerifier, Prefix
from mlime.train.keys import KeyCollator, KeyInputs
from mlime.train.lexicon import Lexicon

#: The layout string ``key-model.json`` carries and ``ime-neural`` requires.
LAYOUT = "keys-pages"

#: The exported files' names.
MANIFEST = "key-model.json"
CONTEXT_GRAPH = "context.onnx"
KEYS_GRAPH = "keys.onnx"
DECODER_GRAPH = "decoder.onnx"
TOKENIZER = "tokenizer.json"

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

Floats = npt.NDArray[np.float32]


@dataclass(frozen=True)
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

    def as_record(self) -> dict[str, Any]:
        """The probe as the manifest writes it."""
        return {
            "keys": self.keys,
            "unfinished_tail": self.unfinished_tail,
            "arcs": self.arcs,
            "digest": self.digest,
        }


def legality_probes(legality: Legality, correct_syllables: bool) -> list[LegalityProbe]:
    """The manifest's probes: :data:`LEGALITY_PROBES` through *legality*."""
    return [
        LegalityProbe.of(legality, keys, tail, correct_syllables) for keys, tail in LEGALITY_PROBES
    ]


@dataclass(frozen=True)
class DecoderShape:
    """The decoder's pages: one position is ``[layers, heads, head_dim]`` per tensor."""

    layers: int
    heads: int
    head_dim: int


@dataclass(frozen=True)
class KeyManifest:
    """``key-model.json``: what a loader needs to open the export and check it."""

    step: int
    base_model: str
    hidden: int
    context_tokens: int
    specials: Mapping[str, int]
    characters: tuple[str, ...]
    bos: int
    decoder: DecoderShape
    correct_syllables: bool
    probes: tuple[LegalityProbe, ...]
    precision: Mapping[str, Mapping[str, Any]]
    graphs: Mapping[str, str]

    @classmethod
    def load(cls, directory: Path) -> KeyManifest:
        """Read *directory*'s manifest, refusing any other layout."""
        path = directory / MANIFEST
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["layout"] != LAYOUT:
            raise ValueError(f"{path} has layout {record['layout']!r}, not {LAYOUT!r}")
        if record["bos"] != len(record["characters"]):
            raise ValueError(f"{path}: <bos> is {record['bos']} for {len(record['characters'])}")
        legality = record["legality"]
        return cls(
            step=int(record["step"]),
            base_model=str(record["base_model"]),
            hidden=int(record["hidden"]),
            context_tokens=int(record["context_tokens"]),
            specials=dict(record["specials"]),
            characters=tuple(record["characters"]),
            bos=int(record["bos"]),
            decoder=DecoderShape(**record["decoder"]),
            correct_syllables=bool(legality["correct_syllables"]),
            probes=tuple(LegalityProbe(**probe) for probe in legality["probes"]),
            precision=dict(record["precision"]),
            graphs=dict(record["graphs"]),
        )

    def check_lexicon(self, lexicon: Lexicon) -> None:
        """Refuse a lexicon whose emission order is not the export's."""
        if tuple(lexicon.characters) != self.characters:
            raise ValueError(
                "the lexicon's emittable characters are not the export's: an arc's character "
                "would name another row of the head"
            )

    def check_legality(self, legality: Legality) -> None:
        """Refuse an automaton whose probe lattices differ from the export's.

        What a consumer's own automaton (the Rust one) must reproduce: the
        same arc count and digest for each probe key string.
        """
        theirs = legality_probes(legality, self.correct_syllables)
        for expected, got in zip(self.probes, theirs, strict=True):
            if expected != got:
                raise ValueError(
                    f"the legality lattice of {expected.keys!r} (tail {expected.unfinished_tail}) "
                    f"has {got.arcs} arcs and digest {got.digest}; the export pins "
                    f"{expected.arcs} and {expected.digest}"
                )


@dataclass(frozen=True)
class KeyEncoding:
    """One keystroke's ``keys.onnx`` output.

    ``log_probs [n, E + 1]`` is each key's CTC log-softmax, the blank last;
    ``cross_keys`` and ``cross_values`` ``[layers, heads, n, head_dim]`` are the
    decoder's cross-attention projections of the key states.
    """

    log_probs: Floats
    cross_keys: Floats
    cross_values: Floats


class OnnxKeyModel:
    """The three graphs of an export, each in its own onnxruntime session on the CPU."""

    def __init__(self, directory: Path, threads: int | None = None) -> None:
        self.directory = directory
        self.manifest = KeyManifest.load(directory)
        options = ort.SessionOptions()
        if threads is not None:
            options.intra_op_num_threads = threads
        graphs = self.manifest.graphs
        providers = ["CPUExecutionProvider"]
        self.context = ort.InferenceSession(
            str(directory / graphs["context"]), options, providers=providers
        )
        self.keys = ort.InferenceSession(
            str(directory / graphs["keys"]), options, providers=providers
        )
        self.decoder = ort.InferenceSession(
            str(directory / graphs["decoder"]), options, providers=providers
        )

    @property
    def emittable(self) -> int:
        """Characters the head and the decoder emit."""
        return len(self.manifest.characters)

    def encode(self, inputs: KeyInputs) -> KeyEncoding:
        """One record's keys and context through ``context.onnx`` and ``keys.onnx``.

        *inputs* is a one-row :meth:`~mlime.train.keys.KeyCollator.inputs`.
        A record without a context binds zeros, which the gate's switch
        multiplies away, and skips the context graph.
        """
        if inputs.size != 1:
            raise ValueError(f"the keys graph runs one row per keystroke, got {inputs.size}")
        context_ids = inputs.context_ids.numpy()
        context_mask = inputs.context_mask.numpy()
        has_context = inputs.has_context.numpy().astype(np.float32)
        if bool(has_context[0]):
            context = self.context.run(
                None, {"context_ids": context_ids, "context_mask": context_mask}
            )[0]
        else:
            context = np.zeros((1, context_ids.shape[1], self.manifest.hidden), dtype=np.float32)
        log_probs, cross_keys, cross_values = self.keys.run(
            None,
            {
                "input_ids": inputs.key_input_ids.numpy(),
                "key_letters": inputs.key_letters.numpy(),
                "attention_mask": inputs.key_attention.numpy(),
                "context": context,
                "context_mask": context_mask,
                "has_context": has_context,
            },
        )
        return KeyEncoding(log_probs=log_probs, cross_keys=cross_keys, cross_values=cross_values)

    def step(
        self,
        encoding: KeyEncoding,
        token: npt.NDArray[np.int64],
        position: npt.NDArray[np.int64],
        page_row: npt.NDArray[np.int64],
        page_mask: npt.NDArray[np.bool_],
        tree_mask: npt.NDArray[np.bool_],
        pool_keys: Floats,
        pool_values: Floats,
    ) -> tuple[Floats, Floats, Floats]:
        """One decoder call: ``(log_probs [R, E], next_keys, next_values [R, L, H, D])``."""
        log_probs, next_keys, next_values = self.decoder.run(
            None,
            {
                "token": token,
                "position": position,
                "page_row": page_row,
                "page_mask": page_mask,
                "tree_mask": tree_mask,
                "pool_keys": pool_keys,
                "pool_values": pool_values,
                "cross_keys": encoding.cross_keys,
                "cross_values": encoding.cross_values,
            },
        )
        return log_probs, next_keys, next_values


class PagePool:
    """The decoder's resident pages: ``[capacity, layers, heads, head_dim]`` keys and values.

    Page 0 is the masked scratch and stays zero; claims start at 1.
    """

    def __init__(self, shape: DecoderShape) -> None:
        self.shape = shape
        self.used = 1
        self.keys = self._zeros(16)
        self.values = self._zeros(16)

    def _zeros(self, pages: int) -> Floats:
        shape = self.shape
        return np.zeros((pages, shape.layers, shape.heads, shape.head_dim), dtype=np.float32)

    def claim(self, keys: Floats, values: Floats) -> int:
        """Write one position's ``[layers, heads, head_dim]`` into a new page; its id."""
        if self.used == self.keys.shape[0]:
            self.keys = np.concatenate([self.keys, self._zeros(self.used)])
            self.values = np.concatenate([self.values, self._zeros(self.used)])
        page = self.used
        self.keys[page] = keys
        self.values[page] = values
        self.used += 1
        return page

    def resident(self) -> tuple[Floats, Floats]:
        """The claimed pages, scratch included: what a call binds."""
        return self.keys[: self.used], self.values[: self.used]


class OnnxTransition:
    """The search's transition over ``decoder.onnx`` and a page pool, per record.

    ``encodings[i]`` is record ``i``'s :class:`KeyEncoding`. A call answers
    each record's queried prefixes in one decoder call, together with any of
    their ancestors that have no page yet.
    """

    def __init__(self, model: OnnxKeyModel, encodings: Sequence[KeyEncoding]) -> None:
        self.model = model
        self.encodings = list(encodings)
        self.pool = PagePool(model.manifest.decoder)
        self.pages: list[dict[Prefix, int]] = [{} for _ in self.encodings]
        #: Rows of each decoder call made, for the callers that count them.
        self.calls: list[int] = []

    def __call__(self, records: Sequence[int], prefixes: Sequence[Prefix]) -> Floats:
        out = np.empty((len(prefixes), self.model.emittable), dtype=np.float32)
        by_record: dict[int, list[int]] = {}
        for index, record in enumerate(records):
            by_record.setdefault(record, []).append(index)
        for record, indices in by_record.items():
            rows = self._call(record, [prefixes[index] for index in indices])
            for index in indices:
                out[index] = rows[prefixes[index]]
        return out

    def _call(self, record: int, queried: Sequence[Prefix]) -> dict[Prefix, Floats]:
        """One decoder call over *queried* and their pageless ancestors, rows by prefix."""
        pages = self.pages[record]
        rows = list(dict.fromkeys(queried))
        seen = set(rows)
        for prefix in queried:
            for length in range(len(prefix)):
                ancestor = prefix[:length]
                if ancestor not in pages and ancestor not in seen:
                    seen.add(ancestor)
                    rows.append(ancestor)
        rows.sort(key=len)
        count = len(rows)
        row_of = {prefix: index for index, prefix in enumerate(rows)}
        depth = max(1, max(len(prefix) for prefix in rows))
        bos = self.model.manifest.bos
        token = np.array([prefix[-1] if prefix else bos for prefix in rows], dtype=np.int64)
        position = np.array([len(prefix) for prefix in rows], dtype=np.int64)
        page_row = np.zeros((count, depth), dtype=np.int64)
        page_mask = np.zeros((count, depth), dtype=np.bool_)
        tree_mask = np.eye(count, dtype=np.bool_)
        for index, prefix in enumerate(rows):
            for length in range(len(prefix)):
                ancestor = prefix[:length]
                if ancestor in pages:
                    page_row[index, length] = pages[ancestor]
                    page_mask[index, length] = True
                else:
                    tree_mask[index, row_of[ancestor]] = True
        pool_keys, pool_values = self.pool.resident()
        log_probs, next_keys, next_values = self.model.step(
            self.encodings[record],
            token,
            position,
            page_row,
            page_mask,
            tree_mask,
            pool_keys,
            pool_values,
        )
        for index, prefix in enumerate(rows):
            if prefix not in pages:
                pages[prefix] = self.pool.claim(next_keys[index], next_values[index])
        self.calls.append(count)
        return {prefix: log_probs[row_of[prefix]] for prefix in queried}


class TimedTransition:
    """A transition whose wall time accumulates in :attr:`DecodeTimes.transition`."""

    def __init__(self, inner: OnnxTransition, times: DecodeTimes) -> None:
        self.inner = inner
        self.times = times

    def __call__(self, records: Sequence[int], prefixes: Sequence[Prefix]) -> Floats:
        started = time.perf_counter()
        rows = self.inner(records, prefixes)
        self.times.transition += time.perf_counter() - started
        return rows


@dataclass(frozen=True)
class OnnxKeyDecoder:
    """:class:`~mlime.train.key_eval.KeyDecoder`'s counterpart over an export.

    Each record is a keystroke of its own: one ``keys.onnx`` run, then the
    constrained search over its lattice with an :class:`OnnxTransition` -- the
    product's shape -- rather than a batch of records.
    """

    model: OnnxKeyModel
    lexicon: Lexicon
    collator: KeyCollator
    legality: Legality
    options: LegalityOptions
    beam: int = DEFAULT_BEAM
    top_k: int = DEFAULT_TOP_K
    times: DecodeTimes = field(default_factory=DecodeTimes)

    def __post_init__(self) -> None:
        self.model.manifest.check_lexicon(self.lexicon)
        if self.legality.characters != self.lexicon.characters:
            raise ValueError(
                "the legality automaton emits over a different set than the export's characters"
            )
        if self.options.unfinished_tail:
            raise ValueError("an evaluation record is complete input; the tail rule is off")

    def decode(
        self, records: Sequence[KeyRecord], grid: Sequence[KeyWeights]
    ) -> list[list[list[Hypothesis]]]:
        """Each grid point's hypotheses for each record, best first: ``[point][record]``."""
        found: list[list[list[Hypothesis]]] = [[] for _ in grid]
        for record in records:
            started = time.perf_counter()
            lattice = self.legality.lattice(record.keys, self.options)
            lattice_done = time.perf_counter()
            inputs = self.collator.inputs([record.keys], [record.context])
            encoding = self.model.encode(inputs)
            encode_done = time.perf_counter()
            self.times.lattice += lattice_done - started
            self.times.encode += encode_done - lattice_done
            for point, hypotheses in enumerate(self.search(lattice, encoding, grid)):
                found[point].extend(hypotheses)
            self.times.records += 1
            self.times.searches += len(grid)
        return found

    def search(
        self, lattice: KeyLattice, encoding: KeyEncoding, grid: Sequence[KeyWeights]
    ) -> list[list[list[Hypothesis]]]:
        """The search at every grid point over one record's encoding: ``[point][1]``."""
        transition = TimedTransition(OnnxTransition(self.model, [encoding]), self.times)
        return search_points(
            [lattice],
            torch.from_numpy(encoding.log_probs).unsqueeze(0),
            transition,
            grid,
            beam=self.beam,
            top_k=self.top_k,
            times=self.times,
        )


def load_onnx_key_run(
    directory: Path,
    eval_set: Path,
    *,
    char_table: Path,
    typo_table: Path,
    syllables: Path,
    with_context: bool,
    dev_share: float,
    beam: int = DEFAULT_BEAM,
    top_k: int = DEFAULT_TOP_K,
    records_per_chunk: int = DEFAULT_RECORDS_PER_CHUNK,
    limit: int | None = None,
    threads: int | None = None,
) -> tuple[KeyRun, int]:
    """Everything an evaluation of the export in *directory* on *eval_set* needs, and its step.

    The lexicon and the automaton are built from the tables the way
    :func:`~mlime.train.key_eval.load_key_run` builds them, and the export's
    manifest is checked against them: the emission order, and the legality
    lattices of its probe key strings.
    """
    from mlime.train.e2e_eval import read_eval_records
    from mlime.train.run import Vocabularies

    model = OnnxKeyModel(directory, threads)
    manifest = model.manifest
    vocabularies = Vocabularies.load(char_table, manifest.base_model, None)
    legality = vocabularies.legality()
    manifest.check_lexicon(vocabularies.lexicon)
    manifest.check_legality(legality)
    collator = build_key_collator(
        vocabularies.spans,
        vocabularies.lexicon,
        vocabularies.tokenizer,
        vocabularies.context_tokenizer,
        with_context=with_context,
        max_context_tokens=manifest.context_tokens,
    )
    decoder = OnnxKeyDecoder(
        model=model,
        lexicon=vocabularies.lexicon,
        collator=collator,
        legality=legality,
        options=LegalityOptions(
            unfinished_tail=False, correct_syllables=manifest.correct_syllables
        ),
        beam=beam,
        top_k=top_k,
        times=DecodeTimes(),
    )
    verifier = LegalityVerifier.load(
        char_table=char_table, typo_table=typo_table, syllables=syllables
    )
    run = KeyRun(
        decoder=decoder,
        verifier=verifier,
        eval_rows=read_eval_records(eval_set),
        dev_share=dev_share,
        records_per_chunk=records_per_chunk,
        limit=limit,
    )
    return run, manifest.step
