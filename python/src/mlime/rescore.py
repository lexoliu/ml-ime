"""Rescoring the beam's hypotheses with a pretrained language model.

The fused decoder ranks a handful of hypotheses per record, and the route A v2
evaluation found that the ranking, not the candidate set, is not the problem:
the sentence the user meant is often not among the eight the beam kept. Before
building a language model into the search, this module measures what a language
model *after* the search can still recover. It reads the per-record dump
``ime-cli fused-eval --dump`` writes, scores every hypothesis with a pretrained
causal LM conditioned on the record's context, and interpolates that score with
the beam's own. The interpolation weight is tuned on the dev slice and reported
on the test slice, the same protocol the fusion weight follows.

The LM score is ``log p(text | context)``: the context and the text are
tokenised apart and only the text's tokens are summed, so every hypothesis of a
record is conditioned on the same context tokens and a tokenizer that merges
across the boundary cannot bias one hypothesis against another.

Three numbers come out for each slice: the beam's own top-1 (what the dump
already ranks first), the top-1 after rescoring at the chosen weight, and the
oracle -- how often the expected sentence is among the hypotheses at all -- which
bounds anything a rescorer can do and says how much has to come from the search
itself.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

log = structlog.get_logger()

#: The interpolation weights swept on the dev slice: ``beam + weight * lm``.
WEIGHTS: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)


@dataclass(frozen=True)
class Hypothesis:
    """One beam output for a record: its text and the decoder's fused score."""

    text: str
    score: float


@dataclass(frozen=True)
class DumpedRecord:
    """One line of a ``fused-eval --dump`` file, joined with its eval-set context."""

    index: int
    text: str
    context: str | None
    hypotheses: tuple[Hypothesis, ...]


@dataclass(frozen=True)
class ScoredRecord:
    """A dumped record with a language-model log probability per hypothesis."""

    record: DumpedRecord
    lm: tuple[float, ...]

    def top1(self, weight: float) -> str:
        """The hypothesis ranked first when the LM score is added at *weight*."""
        best = max(
            range(len(self.record.hypotheses)),
            key=lambda i: self.record.hypotheses[i].score + weight * self.lm[i],
        )
        return self.record.hypotheses[best].text


@dataclass(frozen=True)
class SliceResult:
    """Top-1 accuracies of one slice at every weight, plus its oracle."""

    records: int
    top1_by_weight: dict[float, float]
    oracle: float

    def best_weight(self) -> float:
        """The first weight, in ``WEIGHTS`` order, with the highest top-1."""
        best = WEIGHTS[0]
        for weight in WEIGHTS[1:]:
            if self.top1_by_weight[weight] > self.top1_by_weight[best]:
                best = weight
        return best


def read_dump(dump: Path, eval_set: Path) -> list[DumpedRecord]:
    """Read a dump file and attach each record's context from the eval set."""
    contexts: list[str | None] = []
    with eval_set.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                contexts.append(json.loads(line).get("context"))
    records: list[DumpedRecord] = []
    with dump.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            records.append(
                DumpedRecord(
                    index=row["record"],
                    text=row["text"],
                    context=contexts[row["record"]],
                    hypotheses=tuple(
                        Hypothesis(text=h["text"], score=float(h["score"]))
                        for h in row["hypotheses"]
                    ),
                )
            )
    log.info("read dump", path=str(dump), records=len(records))
    return records


class LanguageModel:
    """A pretrained causal LM that scores sentences given their context."""

    def __init__(self, name: str, device: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(name)
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.float16 if device != "cpu" else torch.float32
        ).to(device)
        self.model.eval()
        self.device = device
        self.bos = self.tokenizer.bos_token_id
        if self.bos is None:
            self.bos = self.tokenizer.eos_token_id
        log.info("loaded language model", name=name, device=device)

    def _encode(self, text: str) -> list[int]:
        ids: list[int] = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return ids

    def log_probabilities(self, prefix: str, texts: Sequence[str]) -> list[float]:
        """``log p(text | prefix)`` of every text, one padded forward pass.

        The prefix and each text are tokenised apart, so every row scores the
        same prefix tokens and only its own continuation counts. Rows are padded
        on the left, which a rotary model does not notice, so that the
        continuations end together and the model only has to produce logits for
        the longest of them: the full vocabulary over the whole prefix would not
        fit an 8 GB machine.
        """
        import torch

        prefix_ids = [self.bos, *self._encode(prefix)]
        tails = [self._encode(text) for text in texts]
        keep = max(len(tail) for tail in tails)
        rows = [prefix_ids + tail for tail in tails]
        width = max(len(row) for row in rows)
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.bos
        input_ids = torch.full((len(rows), width), pad, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        counted = torch.zeros((len(rows), keep), dtype=torch.float32)
        for row, (ids, tail) in enumerate(zip(rows, tails, strict=True)):
            input_ids[row, width - len(ids) :] = torch.tensor(ids)
            mask[row, width - len(ids) :] = 1
            counted[row, keep - len(tail) :] = 1.0
        input_ids = input_ids.to(self.device)
        mask = mask.to(self.device)
        counted = counted.to(self.device)
        with torch.no_grad():
            # Logits at the last keep + 1 positions; the one at position p
            # predicts the token at p + 1, so dropping the final one leaves the
            # predictions for the last keep tokens.
            logits = self.model(
                input_ids=input_ids, attention_mask=mask, logits_to_keep=keep + 1
            ).logits[:, :-1]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        targets = input_ids[:, width - keep :]
        picked = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return (picked * counted).sum(dim=-1).tolist()

    def score(self, record: DumpedRecord) -> tuple[float, ...]:
        """``log p(hypothesis | context)`` for every hypothesis of *record*."""
        return tuple(
            self.log_probabilities(record.context or "", [h.text for h in record.hypotheses])
        )


def score_records(
    lm: LanguageModel, records: Sequence[DumpedRecord], every: int = 500
) -> Iterator[ScoredRecord]:
    """Score records one at a time, logging progress every *every* records."""
    for n, record in enumerate(records, 1):
        yield ScoredRecord(record=record, lm=lm.score(record))
        if n % every == 0:
            log.info("scored", records=n, of=len(records))


def evaluate(scored: Sequence[ScoredRecord]) -> SliceResult:
    """Top-1 at every weight, and how often the expected text was in the beam."""
    top1 = {w: sum(s.top1(w) == s.record.text for s in scored) / len(scored) for w in WEIGHTS}
    oracle = sum(any(h.text == s.record.text for h in s.record.hypotheses) for s in scored)
    return SliceResult(records=len(scored), top1_by_weight=top1, oracle=oracle / len(scored))


@dataclass(frozen=True)
class RescoreReport:
    """What the experiment found on one section of one eval set."""

    model: str
    dev: SliceResult
    test: SliceResult
    weight: float

    def as_dict(self) -> dict[str, object]:
        """A JSON-friendly view."""
        return {
            "model": self.model,
            "weight": self.weight,
            "dev": {
                "records": self.dev.records,
                "oracle": self.dev.oracle,
                "top1_by_weight": {str(w): v for w, v in self.dev.top1_by_weight.items()},
            },
            "test": {
                "records": self.test.records,
                "oracle": self.test.oracle,
                "beam_top1": self.test.top1_by_weight[0.0],
                "rescored_top1": self.test.top1_by_weight[self.weight],
                "top1_by_weight": {str(w): v for w, v in self.test.top1_by_weight.items()},
            },
        }

    def render(self) -> str:
        """A few lines for the terminal."""
        t = self.test
        return "\n".join(
            [
                f"model {self.model}, weight {self.weight} "
                f"(chosen on dev, {self.dev.records} records)",
                f"test ({t.records} records): beam top-1 {t.top1_by_weight[0.0]:.4f}, "
                f"rescored top-1 {t.top1_by_weight[self.weight]:.4f}, oracle {t.oracle:.4f}",
                "dev sweep: " + "  ".join(f"{w}:{self.dev.top1_by_weight[w]:.4f}" for w in WEIGHTS),
            ]
        )


def rescore(
    dev_dump: Path, test_dump: Path, eval_set: Path, model: str, device: str
) -> RescoreReport:
    """Tune the interpolation on the dev dump and report it on the test dump."""
    lm = LanguageModel(model, device)
    dev = evaluate(list(score_records(lm, read_dump(dev_dump, eval_set))))
    weight = dev.best_weight()
    log.info("chose weight on dev", weight=weight, top1=dev.top1_by_weight[weight])
    test = evaluate(list(score_records(lm, read_dump(test_dump, eval_set))))
    return RescoreReport(model=model, dev=dev, test=test, weight=weight)


def default_device() -> str:
    """Apple GPU when there is one, else CUDA, else the CPU."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
