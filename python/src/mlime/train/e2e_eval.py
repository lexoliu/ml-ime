"""Decoding an end-to-end checkpoint over an evaluation lattice, and its report.

The fused evaluation scores a lattice's every candidate and lets a beam search
in Rust weigh a neural emission against a transition. This model needs none of
that: the transition *is* in the decoder, so the lattice is only the menu of
segmentations, and the search is one beam over the model's own step
log-probabilities -- no trigram, no fusion weight, nothing external.

Each record's lattice lists one path per segmentation of its keystrokes, and
per position the characters that span admits. The encoder reads every path's
typed spans once; each path is then decoded autoregressively against its own
encoder output, sixteen beams wide, the search restricted to the admitted
characters but each step scored by its log-probability over the whole
emittable set. Renormalising over the admitted characters instead makes the
scores of two segmentations incomparable: a narrow span such as ``o`` admits a
handful of characters, so splitting ``zao`` into ``za o`` costs nearly
nothing, and the over-segmented reading wins (on the full-pinyin dev slice,
69.88% top-1 renormalised against 73.90% over the emittable set). Paths of one
length are searched together -- across records -- so the step runs over
hundreds of beam rows at once rather than sixteen; batching changes the
arithmetic's shape, not what any one path computes. The surviving hypotheses
merge over the record's paths -- a hypothesis is a full sentence, and where
two segmentations of one sentence meet, the higher sum of step
log-probabilities wins. That merge is what turns "the lattice's best reading"
into "the model's best answer".

The report counts what ``ime-eval`` counts and means the same by it: sentence
top-1 and top-8, character accuracy against the expected text position by
position, MRR over the top *k*, and how many records produced nothing. The dev
and test slices are the same records as everywhere else -- the BLAKE2b record
digest in :mod:`mlime.gui`, verified byte for byte against the Rust oracle.

The ambiguity slice (#150) -- the characters such as 他 / 她 / 它 that later
input or the context decides -- is not computed here. Its selection and
metrics live in ``ime-eval``, where ``fused-eval`` measures its own beams with
them, and :func:`ambiguity_report` hands this evaluation's dump to the same
code through the ``ime-ambiguity`` binary, so the fused decoder and every
end-to-end checkpoint are measured on the same positions the same way.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import batched
from pathlib import Path
from typing import Any

import torch

from mlime.gui import DEV_SHARE, digest, slice_indices
from mlime.logging import log
from mlime.train.e2e import load_model
from mlime.train.e2e_model import E2EModel
from mlime.train.e2e_search import Lane, decode_lanes
from mlime.train.emit import CandidateIndex, LatticeRecord, examples_for, read_lattice
from mlime.train.lexicon import CandidateSpace, Lexicon
from mlime.train.model import mark_dynamic
from mlime.train.samples import (
    DEFAULT_CONTEXT_TOKENS,
    BaseTokenizer,
    Collator,
    ContextTokenizer,
    token_budget_batches,
)
from mlime.train.spans import SpanVocab

#: The search's width and the report's depth, as `fused-eval` runs them.
DEFAULT_BEAM_WIDTH = 16
DEFAULT_TOP_K = 8

#: Records decoded between two writes of the dump: the progress a stopped run keeps.
DEFAULT_RECORDS_PER_CHUNK = 256

#: The fixed-point scale the reciprocal ranks accumulate in, as in Rust.
RANK_SCALE = 1 << 20


#: Beam rows times positions one decode batch may hold: the key/value cache
#: grows with both, so a batch of long paths carries fewer of them.
DEFAULT_ROW_POSITIONS = 16384


@torch.no_grad()
def decode_records(
    model: E2EModel,
    records: Sequence[LatticeRecord],
    lexicon: Lexicon,
    spans: SpanVocab,
    index: CandidateIndex,
    collator: Collator,
    device: torch.device,
    token_budget: int,
    beam_width: int,
    top_k: int,
    row_positions: int = DEFAULT_ROW_POSITIONS,
) -> list[list[tuple[str, float]]]:
    """Each record's best hypotheses over all its lattice paths, best first.

    The encoder reads every path of every record in token-budget batches; the
    paths are then bucketed by length and searched together, as many a batch
    as *row_positions* allows. A path with a position that admits nothing
    cannot complete a sentence and is not searched. Hypotheses colliding on
    text keep the higher score -- identical sentences from different
    segmentations are one answer, as the fused search also says.
    """
    examples = [example for record in records for _, example in examples_for(record, spans)]
    hidden: dict[str, torch.Tensor] = {}
    for group in token_budget_batches(iter(examples), token_budget, collator.max_context_tokens):
        batch = collator(group).to(device)
        mark_dynamic(batch)
        encoded = model.encode(batch).float()
        for row, example in enumerate(group):
            hidden[example.id] = encoded[row, 1 : len(example) + 1]

    by_length: dict[int, list[Lane]] = {}
    for owner, record in enumerate(records):
        for path_index, path in enumerate(record.paths):
            admitted = [
                index.resolve(spans.id(span), candidates)
                for span, candidates in zip(path.spans, path.candidates, strict=True)
            ]
            if any(indices.numel() == 0 for indices in admitted):
                continue
            mask = torch.zeros(len(admitted), lexicon.size, dtype=torch.bool)
            for position, indices in enumerate(admitted):
                mask[position, indices] = True
            lane = Lane(owner, hidden[f"{record.record}:{path_index}"], mask)
            by_length.setdefault(lane.length, []).append(lane)

    merged: list[dict[str, float]] = [{} for _ in records]
    for length, lanes in sorted(by_length.items()):
        per_batch = max(1, row_positions // (beam_width * length))
        for start in range(0, len(lanes), per_batch):
            batch_lanes = lanes[start : start + per_batch]
            for lane, hypotheses in zip(
                batch_lanes, decode_lanes(model, batch_lanes, beam_width), strict=True
            ):
                found = merged[lane.owner]
                for sequence, score in hypotheses:
                    text = "".join(lexicon.characters[emission] for emission in sequence)
                    if score > found.get(text, float("-inf")):
                        found[text] = score
    return [
        sorted(found.items(), key=lambda pair: pair[1], reverse=True)[:top_k] for found in merged
    ]


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
    context_tokenizer: ContextTokenizer,
    lexicon: Lexicon,
    spans: SpanVocab,
    slice_: str,
    dump: Path,
    dev_share: float = DEV_SHARE,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    top_k: int = DEFAULT_TOP_K,
    with_context: bool = True,
    token_budget: int = 8192,
    max_context_tokens: int = DEFAULT_CONTEXT_TOKENS,
    records_per_chunk: int = DEFAULT_RECORDS_PER_CHUNK,
    row_positions: int = DEFAULT_ROW_POSITIONS,
) -> list[tuple[str, Report]]:
    """Decode *lattice*'s records of *slice_* and report the same way ime-eval does.

    Returns ``(name, report)`` pairs: the chosen slice's report alone, unless
    the run covered every record -- then dev, test and all, since the three
    cost one decode. Every decoded record's hypotheses go to *dump* as JSON
    Lines in the ``fused-eval --dump`` shape, so the rescore and generate
    tooling reads them unchanged. The dump is also the run's progress: started
    again on the same dump, the evaluation folds the records already there and
    decodes only the rest.
    """
    if slice_ not in ("dev", "test", "all"):
        raise ValueError(f"slice {slice_!r} is not one of 'dev', 'test', 'all'")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, step = load_model(checkpoint, lexicon, device)
    collator = Collator(
        tokenizer,
        context_tokenizer,
        CandidateSpace(spans, lexicon.candidate_mask),
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

    def fold(record: int, hypotheses: Sequence[str]) -> None:
        observation = Observation.new(eval_rows[record][1], hypotheses, top_k)
        if "all" in reports:
            reports["all"].fold(observation)
            reports["dev" if is_dev[record] else "test"].fold(observation)
        else:
            reports[slice_].fold(observation)

    done = resume_dump(dump)
    for record, hypotheses in done.items():
        if record not in chosen:
            raise ValueError(
                f"{dump} holds record {record}, which slice {slice_!r} does not; "
                "it was written by a different evaluation"
            )
        fold(record, hypotheses)

    def pending() -> Iterator[LatticeRecord]:
        for record in read_lattice(lattice):
            if record.record not in chosen or record.record in done:
                continue
            pinyin, _, context = eval_rows[record.record]
            if pinyin != record.pinyin or (context or None) != (record.context or None):
                raise ValueError(
                    f"record {record.record}: the lattice and the eval set disagree "
                    f"({record.pinyin!r} / {record.context!r} vs {pinyin!r} / {context!r}); "
                    "they were not written for the same run"
                )
            yield record

    decoded = 0
    started = time.monotonic()
    with dump.open("a", encoding="utf-8") as sink:
        for chunk in batched(pending(), records_per_chunk):
            results = decode_records(
                model,
                chunk,
                lexicon,
                spans,
                index,
                collator,
                device,
                token_budget,
                beam_width,
                top_k,
                row_positions,
            )
            for decoded_record, found in zip(chunk, results, strict=True):
                fold(decoded_record.record, [text for text, _ in found])
                sink.write(
                    json.dumps(
                        {
                            "record": decoded_record.record,
                            "text": eval_rows[decoded_record.record][1],
                            "hypotheses": [{"text": text, "score": score} for text, score in found],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            sink.flush()
            decoded += len(chunk)
            log.info(
                "records decoded",
                records=decoded,
                resumed=len(done),
                records_per_second=round(decoded / (time.monotonic() - started), 2),
            )

    log.info(
        "e2e evaluation finished",
        checkpoint=str(checkpoint),
        step=step,
        lattice=str(lattice),
        slice=slice_,
        records=decoded + len(done),
        resumed=len(done),
        with_context=with_context,
        beam_width=beam_width,
    )
    return [(name, reports[name]) for name in names]


def ambiguity_report(
    binary: Path, eval_set: Path, dump: Path, slice_: str, dev_share: float = DEV_SHARE
) -> str:
    """The ambiguity slice of *dump*'s records of *slice_*, as ``ime-ambiguity`` reports it.

    *binary* is ``cargo build -p ime-eval --features cli --bin ime-ambiguity``.
    A dump that lacks a record the slice holds, or was decoded from another
    eval set, is refused there, and the refusal is raised here with its reason.
    """
    result = subprocess.run(
        [
            str(binary),
            "--eval-set",
            str(eval_set),
            "--dump",
            str(dump),
            "--slice",
            slice_,
            "--dev-share",
            str(dev_share),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ime-ambiguity exited {result.returncode} on {dump}: {result.stderr.strip()}"
        )
    return result.stdout


def resume_dump(dump: Path) -> dict[int, list[str]]:
    """The records an earlier run of this evaluation already wrote to *dump*.

    A run stopped mid-write leaves a last line without its newline; that line
    is cut off here, so the record it was writing is decoded again rather than
    folded from half a line.
    """
    if not dump.exists():
        return {}
    data = dump.read_bytes()
    complete = data[: data.rfind(b"\n") + 1]
    if len(complete) != len(data):
        dump.write_bytes(complete)
    done: dict[int, list[str]] = {}
    for line in complete.decode("utf-8").splitlines():
        raw: dict[str, Any] = json.loads(line)
        record = int(raw["record"])
        if record in done:
            raise ValueError(f"{dump} holds record {record} twice")
        done[record] = [str(hypothesis["text"]) for hypothesis in raw["hypotheses"]]
    return done
