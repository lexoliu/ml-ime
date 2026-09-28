"""Decoding an end-to-end checkpoint over an evaluation lattice, and its report.

The fused evaluation scores a lattice's every candidate and lets a beam search
in Rust weigh a neural emission against a transition. This model needs none of
that: the transition *is* in the decoder, so the lattice is only the menu of
segmentations, and the search is one beam over the model's own step
log-probabilities -- no trigram, no fusion weight, nothing external.

Each record's lattice lists one path per segmentation of its keystrokes, and
per position the characters that span admits. The encoder reads every path's
typed spans once; each path is then decoded autoregressively against its own
encoder output, sixteen beams wide, candidates restricted to the admitted
characters and renormalised. The surviving hypotheses merge over the record's
paths -- a hypothesis is a full sentence, and where two segmentations of one
sentence meet, the higher sum of step log-probabilities wins. That merge is
what turns "the lattice's best reading" into "the model's best answer".

The report counts what ``ime-eval`` counts and means the same by it: sentence
top-1 and top-8, character accuracy against the expected text position by
position, MRR over the top *k*, and how many records produced nothing. The dev
and test slices are the same records as everywhere else -- the BLAKE2b record
digest in :mod:`mlime.gui`, verified byte for byte against the Rust oracle.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from mlime.gui import DEV_SHARE, digest, slice_indices
from mlime.logging import log
from mlime.train.charlm_vocab import BOS
from mlime.train.e2e import load_model
from mlime.train.e2e_model import E2EModel
from mlime.train.emit import CandidateIndex, LatticeRecord, examples_for, read_lattice
from mlime.train.lexicon import Lexicon
from mlime.train.model import mark_dynamic
from mlime.train.samples import (
    DEFAULT_CONTEXT_TOKENS,
    BaseTokenizer,
    Collator,
    token_budget_batches,
)
from mlime.train.spans import SpanVocab

#: The search's width and the report's depth, as `fused-eval` runs them.
DEFAULT_BEAM_WIDTH = 16
DEFAULT_TOP_K = 8

#: The fixed-point scale the reciprocal ranks accumulate in, as in Rust.
RANK_SCALE = 1 << 20


def decode_path(
    model: E2EModel,
    span_vectors: torch.Tensor,
    candidates: Sequence[torch.Tensor],
    beam_width: int,
) -> list[tuple[list[int], float]]:
    """Beam-search one lattice path; return emission-index hypotheses and scores.

    ``span_vectors [n, enc]`` is the encoder's hidden state at each of the
    path's typed positions and ``candidates`` the emission indices each admits.
    The beam state is the decoder's key/value cache, so the step is the same
    arithmetic training ran, advanced one position at a time. A position that
    admits nothing kills the path -- the beam cannot complete a sentence
    through it -- which is what an empty hypothesis list means here.
    """
    if any(candidates_for.numel() == 0 for candidates_for in candidates):
        return []
    decoder = model.decoder
    encoder_kv = [cross.kv(span_vectors.unsqueeze(0)) for cross in decoder.gated_layers()]
    layers, heads = decoder.reader.config.layers, decoder.reader.config.heads
    head_dim = decoder.reader.config.hidden // heads
    device = span_vectors.device
    keys = torch.zeros(1, layers, heads, 0, head_dim, device=device)
    values = keys.clone()
    scores = torch.zeros(1, device=device)
    sequences: list[list[int]] = [[]]
    tokens = torch.full((1,), BOS, dtype=torch.long, device=device)
    for position, admitted in enumerate(candidates):
        logits, (keys, values) = decoder.step(
            tokens,
            span_vectors[position].expand(len(sequences), -1),
            (keys, values),
            encoder_kv,
            candidates=admitted.to(device),
        )
        log_probs = logits.log_softmax(dim=-1)
        total = scores.unsqueeze(1) + log_probs
        keep = min(beam_width, total.numel())
        top = total.flatten().topk(keep)
        parents = top.indices // admitted.numel()
        choices = top.indices % admitted.numel()
        emitted = admitted.to(device)[choices]
        parent_list = parents.tolist()
        sequences = [
            sequences[parent] + [int(emitted[index])] for index, parent in enumerate(parent_list)
        ]
        scores = top.values
        keys, values = keys[parents], values[parents]
        tokens = decoder.emit_ids[emitted]
    return list(zip(sequences, scores.tolist(), strict=True))


@torch.no_grad()
def decode_record(
    model: E2EModel,
    record: LatticeRecord,
    lexicon: Lexicon,
    spans: SpanVocab,
    index: CandidateIndex,
    collator: Collator,
    device: torch.device,
    token_budget: int,
    beam_width: int,
    top_k: int,
) -> list[tuple[str, float]]:
    """A record's best hypotheses over all its lattice paths, best first.

    One encoder pass covers every reading of the record; each reading's beam
    then walks it position by position. Hypotheses colliding on text keep the
    higher score -- identical sentences from different segmentations are one
    answer, as the fused search also says.
    """
    examples = [example for _, example in examples_for(record, spans)]
    hidden: dict[str, torch.Tensor] = {}
    for group in token_budget_batches(iter(examples), token_budget, collator.max_context_tokens):
        batch = collator(group).to(device)
        mark_dynamic(batch)
        encoded = model.encode(batch).float()
        for row, example in enumerate(group):
            hidden[example.id] = encoded[row, 1 : len(example) + 1]

    merged: dict[str, float] = {}
    for path_index, path in enumerate(record.paths):
        example_id = f"{record.record}:{path_index}"
        span_vectors = hidden[example_id]
        admitted = [
            index.resolve(spans.id(span), candidates)
            for span, candidates in zip(path.spans, path.candidates, strict=True)
        ]
        for sequence, score in decode_path(model, span_vectors, admitted, beam_width):
            text = "".join(lexicon.characters[emission] for emission in sequence)
            if score > merged.get(text, float("-inf")):
                merged[text] = score
    ranked = sorted(merged.items(), key=lambda pair: pair[1], reverse=True)
    return ranked[:top_k]


@dataclass(frozen=True)
class Observation:
    """What one record's decode contributed -- the Rust ``Observation``, in Python."""

    characters: int
    character_hits: int
    unanswered: bool
    rank: int | None

    @classmethod
    def new(cls, expected: str, hypotheses: Sequence[str], top_k: int) -> Observation:
        """The same fields :func:`Report.fold` consumes, counted as metrics.rs does."""
        if not hypotheses:
            return cls(characters=len(expected), character_hits=0, unanswered=True, rank=None)
        top = hypotheses[0]
        character_hits = sum(1 for found, want in zip(top, expected, strict=False) if found == want)
        rank = next((i for i, found in enumerate(hypotheses[:top_k]) if found == expected), None)
        return cls(
            characters=len(expected), character_hits=character_hits, unanswered=False, rank=rank
        )


@dataclass
class Report:
    """The counters ``ime-eval``'s Report carries, with its rendering."""

    top_k: int
    records: int = 0
    top1_hits: int = 0
    topk_hits: int = 0
    characters: int = 0
    character_hits: int = 0
    reciprocal_ranks: int = 0
    unanswered: int = 0

    def fold(self, observation: Observation) -> None:
        """Merge one observation; the order-free fold the Rust report defines."""
        self.records += 1
        self.characters += observation.characters
        if observation.unanswered:
            self.unanswered += 1
            return
        self.character_hits += observation.character_hits
        if observation.rank is not None:
            if observation.rank == 0:
                self.top1_hits += 1
            self.topk_hits += 1
            self.reciprocal_ranks += RANK_SCALE // (observation.rank + 1)

    def observe(self, expected: str, hypotheses: Sequence[str]) -> None:
        """Fold the observation an expected text and its hypotheses make."""
        self.fold(Observation.new(expected, hypotheses, self.top_k))

    def render(self) -> str:
        """The metrics table, in ``ime-eval``'s column layout."""
        top1 = self.top1_hits / self.records if self.records else 0.0
        topk = self.topk_hits / self.records if self.records else 0.0
        characters = self.character_hits / self.characters if self.characters else 0.0
        mrr = self.reciprocal_ranks / RANK_SCALE / self.records if self.records else 0.0
        return "\n".join(
            [
                f"{'metric':<22}value   detail",
                f"{'-' * 20}  {'-' * 6}  {'-' * 20}",
                f"{'records':<22}{self.records}",
                f"{'sentence, top-1':<22}{top1:.4f}  {self.top1_hits} / {self.records}",
                f"sentence, top-{self.top_k:<8}{topk:.4f}  {self.topk_hits} / {self.records}",
                f"{'character':<22}{characters:.4f}  {self.character_hits} / {self.characters}",
                f"MRR@{self.top_k:<18}{mrr:.4f}",
                f"{'unanswered':<22}{self.unanswered}",
            ]
        )


def read_eval_records(path: Path) -> list[tuple[str, str, str | None]]:
    """The eval set's ``(pinyin, text, context)`` rows, validated as Rust validates.

    ``deny_unknown_fields`` holds here too: a record that grew a field means a
    file from a different schema, and reading past it silently is how a digest
    over stale bytes gets scored as current.
    """
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw: dict[str, Any] = json.loads(line)
        unknown = set(raw) - {"pinyin", "text", "context"}
        if unknown:
            raise ValueError(f"{path}:{number} carries unknown fields {sorted(unknown)}")
        pinyin, text = raw.get("pinyin"), raw.get("text")
        if not isinstance(pinyin, str) or not pinyin:
            raise ValueError(f"{path}:{number} has an empty or missing pinyin")
        if not isinstance(text, str) or not text:
            raise ValueError(f"{path}:{number} has an empty or missing text")
        context = raw.get("context")
        if context is not None and not isinstance(context, str):
            raise ValueError(f"{path}:{number} has a non-string context")
        records.append((pinyin, text, context))
    if not records:
        raise ValueError(f"{path} holds no records")
    return records


def evaluate(
    checkpoint: Path,
    lattice: Path,
    eval_set: Path,
    tokenizer: BaseTokenizer,
    lexicon: Lexicon,
    spans: SpanVocab,
    slice_: str,
    dev_share: float = DEV_SHARE,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    top_k: int = DEFAULT_TOP_K,
    with_context: bool = True,
    token_budget: int = 8192,
    max_context_tokens: int = DEFAULT_CONTEXT_TOKENS,
    dump: Path | None = None,
) -> list[tuple[str, Report]]:
    """Decode *lattice*'s records of *slice_* and report the same way ime-eval does.

    Returns ``(name, report)`` pairs: the chosen slice's report alone, unless
    the run covered every record -- then dev, test and all, since the three
    cost one decode. *dump* writes every decoded record's hypotheses as JSON
    Lines in the ``fused-eval --dump`` shape, so the rescore and generate
    tooling reads them unchanged.
    """
    if slice_ not in ("dev", "test", "all"):
        raise ValueError(f"slice {slice_!r} is not one of 'dev', 'test', 'all'")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, step = load_model(checkpoint, lexicon, device)
    collator = Collator(
        tokenizer,
        lexicon.candidate_mask,
        context_dropout=0.0 if with_context else 1.0,
        max_context_tokens=max_context_tokens,
    )
    index = CandidateIndex(lexicon)

    eval_rows = read_eval_records(eval_set)
    rows = [
        {"pinyin": pinyin, "text": text, "context": context} for pinyin, text, context in eval_rows
    ]
    u64_max = float(2**64 - 1)
    is_dev = [
        digest(pinyin, text, context) / u64_max < dev_share for pinyin, text, context in eval_rows
    ]
    chosen = set(slice_indices(rows, slice_, dev_share))

    names = ("dev", "test", "all") if slice_ == "all" else (slice_,)
    reports = {name: Report(top_k) for name in names}

    decoded = 0
    sink = dump.open("w", encoding="utf-8") if dump is not None else None
    try:
        for record in read_lattice(lattice):
            if record.record not in chosen:
                continue
            pinyin, text, context = eval_rows[record.record]
            if pinyin != record.pinyin or (context or None) != (record.context or None):
                raise ValueError(
                    f"record {record.record}: the lattice and the eval set disagree "
                    f"({record.pinyin!r} / {record.context!r} vs {pinyin!r} / {context!r}); "
                    "they were not written for the same run"
                )
            hypotheses = decode_record(
                model,
                record,
                lexicon,
                spans,
                index,
                collator,
                device,
                token_budget,
                beam_width,
                top_k,
            )
            texts = [hypothesis for hypothesis, _ in hypotheses]
            observation = Observation.new(text, texts, top_k)
            if "all" in reports:
                reports["all"].fold(observation)
                reports["dev" if is_dev[record.record] else "test"].fold(observation)
            else:
                reports[slice_].fold(observation)
            if sink is not None:
                sink.write(
                    json.dumps(
                        {
                            "record": record.record,
                            "text": text,
                            "hypotheses": [
                                {"text": hypothesis, "score": score}
                                for hypothesis, score in hypotheses
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            decoded += 1
            if decoded % 200 == 0:
                log.debug("records decoded", records=decoded)
    finally:
        if sink is not None:
            sink.close()

    log.info(
        "e2e evaluation finished",
        checkpoint=str(checkpoint),
        step=step,
        lattice=str(lattice),
        slice=slice_,
        records=decoded,
        with_context=with_context,
        beam_width=beam_width,
    )
    return [(name, reports[name]) for name in names]
