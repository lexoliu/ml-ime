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
from openai import AsyncOpenAI, OpenAIError
from openai.types.shared import ReasoningEffort

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

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        concurrency: int = 16,
        retries: int = 2,
        reasoning_effort: ReasoningEffort = "medium",
    ):
        self._client = client
        self._model = model
        self._retries = retries
        self._reasoning_effort = reasoning_effort
        self._template = prompt_template()
        self._gate = asyncio.Semaphore(concurrency)

    @classmethod
    def from_settings(
        cls, settings: LlmSettings, concurrency: int, reasoning_effort: ReasoningEffort
    ) -> Reranker:
        """Build a client from the ``MLIME_LLM_*`` environment."""
        client = AsyncOpenAI(base_url=settings.base_url, api_key=settings.api_key)
        return cls(
            client, settings.model, concurrency=concurrency, reasoning_effort=reasoning_effort
        )

    @property
    def model(self) -> str:
        """The model id the picks come from."""
        return self._model

    def prompt(self, record: DumpedRecord) -> str:
        """Render the prompt for *record*."""
        return self._template.render(
            context=record.context,
            pinyin=record.pinyin,
            hypotheses=[h.text for h in record.hypotheses],
        )

    async def choose_all(self, records: Sequence[DumpedRecord], every: int = 500) -> list[Choice]:
        """Choose for every record concurrently, logging progress every *every* answers."""
        done = 0

        async def one(record: DumpedRecord) -> Choice:
            nonlocal done
            choice = await self._one(record)
            done += 1
            if done % every == 0:
                log.info("chosen", records=done, of=len(records))
            return choice

        return list(await asyncio.gather(*(one(record) for record in records)))

    async def _one(self, record: DumpedRecord) -> Choice:
        """Choose for one record, retrying a malformed or failed answer."""
        last = ""
        for attempt in range(self._retries + 1):
            async with self._gate:
                try:
                    content = await self._complete(record)
                    return Choice(
                        record=record, picked=parse_choice(content, len(record.hypotheses))
                    )
                except (ValueError, OpenAIError) as error:
                    last = f"{type(error).__name__}: {error}"
            log.debug("rerank retry", record=record.index, attempt=attempt, reason=last)
        log.warning("rerank unanswered", record=record.index, reason=last)
        return Choice(record=record, picked=None, reason=last)

    async def _complete(self, record: DumpedRecord) -> str:
        """One chat completion, at the pinned reasoning effort."""
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": self.prompt(record)}],
            reasoning_effort=self._reasoning_effort,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("the endpoint returned a message with no content")
        return content


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
    dump: Path, eval_set: Path, concurrency: int, reasoning_effort: ReasoningEffort
) -> RescoreReport:
    """Rerank every record of *dump* and report the slice."""
    reranker = Reranker.from_settings(LlmSettings.load(), concurrency, reasoning_effort)
    choices = asyncio.run(reranker.choose_all(read_dump(dump, eval_set)))
    return RescoreReport(
        model=reranker.model,
        reasoning_effort=str(reasoning_effort),
        result=evaluate(choices),
        choices=tuple(choices),
    )
