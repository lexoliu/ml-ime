"""Rerank the beam's hypotheses with a strong language model behind an API.

Route A's fused decode reaches its ceiling before its top-1 does: the neural
top-8 sits within a few points of its top-1, and the trigram it is fused with
scores a character against the two before it. The question this module
answers is how much a much stronger language model, reading the context and
the keystrokes, could recover from the beam's own hypotheses *without*
touching the search -- the ceiling of rescoring -- and therefore how much has
to come from the search instead.

`fused-eval --dump` writes each record's hypotheses with the decoder's own
score; this module joins them with the eval set's context and pinyin and asks
the ``MLIME_LLM_*`` endpoint to pick the hypothesis the user most likely
meant. The model never sees the expected text.

Three numbers come out for a slice: the beam's own top-1 (what the dump
already ranks first), the top-1 after the model's choice, and the oracle -- how
often the expected sentence is among the hypotheses at all -- which bounds
anything a reranker can do and says how much has to come from the search.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import regex
from jinja2 import Template
from openai.types.shared import ReasoningEffort

from mlime.endpoint import Answer, Endpoint, answer_all
from mlime.logging import log
from mlime.settings import LlmSettings

PROMPT = "templates/rerank_prompt.txt"


@dataclass(frozen=True)
class Hypothesis:
    """One beam output for a record: its text and the decoder's fused score."""

    text: str
    score: float


@dataclass(frozen=True)
class DumpedRecord:
    """One line of a ``fused-eval --dump`` file, joined with its eval-set record."""

    index: int
    text: str
    pinyin: str
    context: str | None
    hypotheses: tuple[Hypothesis, ...]


@dataclass(frozen=True)
class Choice:
    """What the model picked for a record, or why it picked nothing."""

    record: DumpedRecord
    picked: int | None
    reason: str = ""

    @property
    def top1(self) -> str:
        """The reranked answer: the pick, or the beam's own first when there is none."""
        index = 0 if self.picked is None else self.picked
        return self.record.hypotheses[index].text


@dataclass(frozen=True)
class SliceResult:
    """What reranking did on one slice."""

    records: int
    beam_top1: float
    reranked_top1: float
    oracle: float
    unanswered: int

    def as_dict(self) -> dict[str, object]:
        """A JSON-friendly view."""
        return {
            "records": self.records,
            "beam_top1": self.beam_top1,
            "reranked_top1": self.reranked_top1,
            "oracle": self.oracle,
            "unanswered": self.unanswered,
        }


def read_dump(dump: Path, eval_set: Path) -> list[DumpedRecord]:
    """Read a dump file and attach each record's pinyin and context from the eval set."""
    rows: list[dict[str, object]] = []
    with eval_set.open(encoding="utf-8") as handle:
        rows.extend(json.loads(line) for line in handle if line.strip())
    records: list[DumpedRecord] = []
    with dump.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source = rows[row["record"]]
            if source["text"] != row["text"]:
                raise ValueError(
                    f"record {row['record']} of {dump} is {row['text']!r} but the eval set "
                    f"holds {source['text']!r}; the dump was decoded from another set"
                )
            records.append(
                DumpedRecord(
                    index=row["record"],
                    text=row["text"],
                    pinyin=str(source["pinyin"]),
                    context=source.get("context"),
                    hypotheses=tuple(
                        Hypothesis(text=h["text"], score=float(h["score"]))
                        for h in row["hypotheses"]
                    ),
                )
            )
    log.info("read dump", path=str(dump), records=len(records))
    return records


def prompt_template() -> Template:
    """The reranking prompt, kept in a file so it can be diffed and reviewed."""
    source: str = resources.files(__spec__.parent).joinpath(PROMPT).read_text(encoding="utf-8")
    # `jinja2.Template.__new__` is declared to return `Any`, so pin the type here.
    template: Template = Template(source, keep_trailing_newline=True)
    return template


def parse_choice(content: str, count: int) -> int:
    """The zero-based index the model answered with, or a ``ValueError``."""
    numbers = regex.findall(r"\d+", content)
    if len(numbers) != 1:
        raise ValueError(f"expected one number in the answer, got {content!r}")
    picked = int(numbers[0])
    if not 1 <= picked <= count:
        raise ValueError(f"the answer {picked} is outside 1..{count}")
    return picked - 1


class Reranker:
    """Picks the likeliest hypothesis of each record through a chat completion endpoint."""

    def __init__(self, endpoint: Endpoint):
        self._endpoint = endpoint
        self._template = prompt_template()

    @classmethod
    def from_settings(
        cls, settings: LlmSettings, concurrency: int, reasoning_effort: ReasoningEffort
    ) -> Reranker:
        """Build a client from the ``MLIME_LLM_*`` environment."""
        return cls(Endpoint.from_settings(settings, concurrency, reasoning_effort))

    @property
    def model(self) -> str:
        """The model id the picks come from."""
        return self._endpoint.model

    def prompt(self, record: DumpedRecord) -> str:
        """Render the prompt for *record*."""
        return self._template.render(
            context=record.context,
            pinyin=record.pinyin,
            hypotheses=[h.text for h in record.hypotheses],
        )

    async def choose_all(self, records: Sequence[DumpedRecord], picks: Path) -> list[Choice]:
        """Choose for every record concurrently, keeping every answer in *picks*."""

        async def one(record: DumpedRecord) -> Answer[int]:
            return await self._endpoint.ask(
                self.prompt(record),
                lambda content: parse_choice(content, len(record.hypotheses)),
                record.index,
            )

        answers = await answer_all(records, picks, lambda r: r.index, one, int)
        return [
            Choice(record=record, picked=answer.value, reason=answer.reason)
            for record, answer in zip(records, answers, strict=True)
        ]


def evaluate(choices: Sequence[Choice]) -> SliceResult:
    """Beam top-1, reranked top-1, and how often the expected text was in the beam."""
    total = len(choices)
    return SliceResult(
        records=total,
        beam_top1=sum(c.record.hypotheses[0].text == c.record.text for c in choices) / total,
        reranked_top1=sum(c.top1 == c.record.text for c in choices) / total,
        oracle=sum(any(h.text == c.record.text for h in c.record.hypotheses) for c in choices)
        / total,
        unanswered=sum(c.picked is None for c in choices),
    )


@dataclass(frozen=True)
class RescoreReport:
    """What the experiment found on one slice of one eval set."""

    model: str
    reasoning_effort: str
    result: SliceResult
    choices: tuple[Choice, ...]

    def as_dict(self) -> dict[str, object]:
        """A JSON-friendly view, with every pick so the picks can be inspected."""
        return {
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "result": self.result.as_dict(),
            "choices": [
                {"record": c.record.index, "picked": c.picked, "reason": c.reason}
                for c in self.choices
            ],
        }

    def render(self) -> str:
        """A few lines for the terminal."""
        r = self.result
        return "\n".join(
            [
                f"model {self.model}, reasoning effort {self.reasoning_effort}",
                f"{r.records} records: beam top-1 {r.beam_top1:.4f}, "
                f"reranked top-1 {r.reranked_top1:.4f}, oracle {r.oracle:.4f}, "
                f"unanswered {r.unanswered}",
            ]
        )


def rescore(
    dump: Path, eval_set: Path, picks: Path, concurrency: int, reasoning_effort: ReasoningEffort
) -> RescoreReport:
    """Rerank every record of *dump*, resuming from *picks*, and report the slice."""
    reranker = Reranker.from_settings(LlmSettings.load(), concurrency, reasoning_effort)
    choices = asyncio.run(reranker.choose_all(read_dump(dump, eval_set), picks))
    return RescoreReport(
        model=reranker.model,
        reasoning_effort=str(reasoning_effort),
        result=evaluate(choices),
        choices=tuple(choices),
    )
