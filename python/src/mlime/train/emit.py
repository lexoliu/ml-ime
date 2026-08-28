"""Scoring a decoder's lattice with a trained route A model.

The model cannot decode a sentence on its own. It is non-autoregressive: it
emits every position at once, so it has no way to say that 中国 is likelier than
中过 -- the two differ only in how the second character agrees with the first,
which is exactly what conditional independence throws away. The search that puts
that back lives in Rust, and it needs one number per candidate per position.

So this module answers a lattice rather than a sentence. `ime-cli emit-lattice`
writes, for every evaluation record, every reading of the keystrokes and, for
every position of every reading, the letters typed there and the characters that
position admits. Here each reading becomes one forward pass of the fill tower --
the same spans, the same collator, the same restricted softmax as training, with
no targets, so the same code that learned the distribution is the code that
reports it -- and the answer goes back as log probabilities in exactly the order
the lattice listed the candidates.

Two things the lattice may ask about that the model cannot answer, and they are
not the same thing:

* A character outside the base tokenizer's vocabulary has no row in the output
  head at all. The Rust lexicon holds 41,923 characters and MacBERT covers 7,322
  of them, so this is common, expected, and gets the floor -- a finite number,
  so that the fused score of an emission weight of zero is still the transition
  score and not a NaN. How many slots took the floor is recorded, because it is
  a ceiling on what the neural route can do.
* A character the model *can* emit, at a position its own mask says that span
  does not admit, means the pinyin tables the two sides were built from have
  drifted apart. That is not a missing number, it is two programs disagreeing
  about what the user typed, and it raises.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from mlime.logging import log
from mlime.train.lexicon import Lexicon
from mlime.train.model import RouteAConfig, RouteAModel
from mlime.train.samples import (
    IGNORE_INDEX,
    BaseTokenizer,
    Collator,
    TrainingExample,
    token_budget_batches,
)
from mlime.train.spans import SpanVocab


@dataclass(frozen=True)
class LatticePath:
    """One reading of a typed string, as the decoder asks about it."""

    spans: tuple[str, ...]
    candidates: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.spans) != len(self.candidates):
            raise ValueError(
                f"a reading typed {len(self.spans)} spans but admits candidates at "
                f"{len(self.candidates)} positions"
            )
        if not self.spans:
            raise ValueError("a reading with no positions cannot be scored")


@dataclass(frozen=True)
class LatticeRecord:
    """Everything the model is asked about one evaluation record."""

    record: int
    pinyin: str
    context: str | None
    paths: tuple[LatticePath, ...]

    def __post_init__(self) -> None:
        if not self.paths:
            raise ValueError(f"record {self.record} has no reading to score")

    @classmethod
    def parse(cls, line: str) -> LatticeRecord:
        """Read one JSON Lines lattice record."""
        raw: dict[str, Any] = json.loads(line)
        expected = {"record", "pinyin", "context", "paths"}
        unknown = set(raw) - expected
        if unknown:
            raise ValueError(f"a lattice record carries unknown fields {sorted(unknown)}")
        missing = expected - set(raw) - {"context"}
        if missing:
            raise ValueError(f"a lattice record is missing {sorted(missing)}")
        return cls(
            record=int(raw["record"]),
            pinyin=str(raw["pinyin"]),
            context=raw.get("context"),
            paths=tuple(
                LatticePath(spans=tuple(path["spans"]), candidates=tuple(path["candidates"]))
                for path in raw["paths"]
            ),
        )


def read_lattice(path: Path) -> Iterator[LatticeRecord]:
    """Stream a lattice file, one record per line."""
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield LatticeRecord.parse(line)
            except (ValueError, KeyError, json.JSONDecodeError) as error:
                raise ValueError(f"{path}:{number} is not a lattice record: {error}") from error


class CandidateIndex:
    """Maps a position's candidate characters onto the model's emission indices.

    The same (span, candidate set) pair recurs at almost every position of every
    record -- there are only so many syllables -- so the resolution is cached.
    Without the cache this is twenty-one million dictionary lookups; with it, a
    few thousand.
    """

    def __init__(self, lexicon: Lexicon):
        self._lexicon = lexicon
        self._cache: dict[tuple[int, str], torch.Tensor] = {}
        self.slots = 0

    def resolve(self, span_id: int, candidates: str) -> torch.Tensor:
        """Emission indices for *candidates*, in the order the lattice listed them.

        Raises on anything it cannot resolve. The lattice was written against
        this model's own emittable set, so a character it does not hold, or one
        the model's mask rules out at this span, means the lattice and the
        checkpoint came from different runs -- and scoring through that would
        hide it behind a plausible number.
        """
        key = (span_id, candidates)
        cached = self._cache.get(key)
        if cached is None:
            indices = []
            for character in candidates:
                if not self._lexicon.contains(character):
                    raise ValueError(
                        f"the lattice asks about {character!r}, which this model cannot "
                        "emit; the lattice was written against a different emittable set"
                    )
                index = self._lexicon.index(character)
                if not bool(self._lexicon.candidate_mask[span_id, index]):
                    raise ValueError(
                        f"the lattice admits {character!r} after typed span {span_id} "
                        "and the model's mask does not; the two pinyin tables were not "
                        "generated by the same run"
                    )
                indices.append(index)
            cached = torch.tensor(indices, dtype=torch.long)
            self._cache[key] = cached
        self.slots += int(cached.numel())
        return cached


def examples_for(record: LatticeRecord, spans: SpanVocab) -> Iterator[tuple[int, TrainingExample]]:
    """One example per reading, carrying no targets because nothing is scored.

    ``IGNORE_INDEX`` everywhere means the model's own loss is undefined and comes
    back as ``None``; the logits are read straight off instead. The alternative
    -- a second forward path for inference -- is how the thing that is measured
    stops being the thing that was trained.
    """
    for index, path in enumerate(record.paths):
        for span in path.spans:
            if span not in spans:
                raise ValueError(
                    f"record {record.record} typed the span {span!r}, which is not a prefix "
                    "of any syllable; the span table and the syllable table disagree"
                )
        yield (
            index,
            TrainingExample(
                id=f"{record.record}:{index}",
                spans=tuple(path.spans),
                span_ids=tuple(spans.id(span) for span in path.spans),
                targets=tuple(IGNORE_INDEX for _ in path.spans),
                context=record.context,
            ),
        )


@dataclass(frozen=True)
class ScoreRecord:
    """The model's answer for one evaluation record."""

    record: int
    paths: list[list[list[float]]]

    def line(self) -> str:
        """The record as one JSON Lines row."""
        return json.dumps({"record": self.record, "paths": self.paths}, ensure_ascii=False)


def score_chunk(
    model: RouteAModel,
    collator: Collator,
    index: CandidateIndex,
    chunk: Sequence[LatticeRecord],
    spans: SpanVocab,
    device: torch.device,
    token_budget: int,
) -> list[ScoreRecord]:
    """Score every reading of every record in *chunk*."""
    examples: list[TrainingExample] = []
    owners: dict[str, tuple[int, int]] = {}
    for record in chunk:
        for path_index, example in examples_for(record, spans):
            owners[example.id] = (record.record, path_index)
            examples.append(example)
    scores: dict[int, dict[int, list[list[float]]]] = {record.record: {} for record in chunk}
    admitted = {record.record: record.paths for record in chunk}

    with torch.no_grad():
        for group in token_budget_batches(iter(examples), token_budget):
            batch = collator(group).to(device)
            logits = model(batch).logits.float()
            for row, example in enumerate(group):
                record_id, path_index = owners[example.id]
                candidates = admitted[record_id][path_index].candidates
                masks = model.candidate_mask.index_select(
                    0, batch.span_ids[row, 1 : len(example) + 1]
                )
                position_logits = logits[row, 1 : len(example) + 1]
                floor = torch.finfo(position_logits.dtype).min
                log_probabilities = position_logits.masked_fill(~masks, floor).log_softmax(dim=-1)
                path_scores: list[list[float]] = []
                for position, admitted_here in enumerate(candidates):
                    if not admitted_here:
                        path_scores.append([])
                        continue
                    slots = index.resolve(example.span_ids[position], admitted_here).to(device)
                    values = log_probabilities[position].index_select(0, slots)
                    path_scores.append([round(value, 4) for value in values.tolist()])
                scores[record_id][path_index] = path_scores

    written = []
    for record in chunk:
        by_path = scores[record.record]
        if len(by_path) != len(record.paths):
            raise RuntimeError(
                f"record {record.record} has {len(record.paths)} readings but "
                f"{len(by_path)} were scored"
            )
        written.append(
            ScoreRecord(
                record=record.record,
                paths=[by_path[index] for index in range(len(record.paths))],
            )
        )
    return written


def load_model(
    checkpoint: Path, lexicon: Lexicon, device: torch.device
) -> tuple[RouteAModel, RouteAConfig, int]:
    """Rebuild the trained model from a checkpoint, weights and all.

    Built from the base model's *configuration* rather than its weights: every
    parameter is about to be overwritten by the checkpoint, so downloading the
    pretrained tensors would be a few hundred megabytes spent on values that
    survive one line. The load is strict, which also checks the lexicon: the
    candidate mask is a persistent buffer, so a mask built from a different
    character table fails to load rather than scoring the wrong characters.
    """
    from transformers import BertConfig

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    route = RouteAConfig(**state["route_a"])
    bert = BertConfig.from_pretrained(route.base_model)
    model = RouteAModel.from_config(bert, lexicon, route)
    model.load_state_dict(state["model"], strict=True)
    model.to(device)
    model.eval()
    step = int(state["step"])
    log.info(
        "checkpoint loaded",
        path=str(checkpoint),
        step=step,
        base=route.base_model,
        gates=[round(gate, 5) for gate in model.gates()],
    )
    return model, route, step


def emit(
    checkpoint: Path,
    lattice: Path,
    out: Path,
    tokenizer: BaseTokenizer,
    lexicon: Lexicon,
    spans: SpanVocab,
    with_context: bool,
    token_budget: int = 8192,
    max_context_tokens: int = 128,
    records_per_chunk: int = 256,
) -> Path:
    """Score *lattice* with *checkpoint* and write the score file beside a meta file.

    Context is switched by the collator's dropout being 0 or 1, exactly as the
    in-training evaluation switches it, so the with-context and without-context
    runs differ in one thing and not in a code path.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, route, step = load_model(checkpoint, lexicon, device)
    collator = Collator(
        tokenizer,
        context_dropout=0.0 if with_context else 1.0,
        max_context_tokens=max_context_tokens,
    )
    index = CandidateIndex(lexicon)
    out.parent.mkdir(parents=True, exist_ok=True)

    records = 0
    chunk: list[LatticeRecord] = []
    with gzip.open(out, "wt", encoding="utf-8") as sink:

        def flush(chunk: list[LatticeRecord]) -> None:
            nonlocal records
            if not chunk:
                return
            for scored in score_chunk(model, collator, index, chunk, spans, device, token_budget):
                sink.write(scored.line() + "\n")
            records += len(chunk)
            log.debug("chunk scored", records=records)

        for record in read_lattice(lattice):
            chunk.append(record)
            if len(chunk) >= records_per_chunk:
                flush(chunk)
                chunk = []
        flush(chunk)

    meta = Path(f"{out}.meta.json")
    meta.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "step": step,
                "base_model": route.base_model,
                "with_context": with_context,
                "records": records,
                "candidate_slots": index.slots,
                "emittable_characters": lexicon.size,
                "typed_spans": lexicon.spans,
                "gates": model.gates(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log.info(
        "lattice scored",
        records=records,
        slots=index.slots,
        with_context=with_context,
        path=str(out),
    )
    return out
