"""Generate the sentence from context and keystrokes with a strong language model.

Every measurement of #40 says the beam does not contain the answer: the
reranker (`mlime eval rescore`) is capped by the oracle, the character LM
inside the beam moves the oracle by a point, and 128 candidates hold the
expected abbreviated sentence a third of the time. This module asks the
opposite question: given the context and the typed keystrokes and nothing
else, how often does a far stronger model *write* the sentence the user
meant? That bounds what any sentence-level model can reach from the same
evidence, and says what a different decoder would be aiming at.

The prompt explains the typing conventions (full pinyin, initials, or a mix,
without separators) and asks for the sentence alone, with as many characters
as the keystrokes have syllables. A variant also shows the beam's hypotheses
as hints. The expected text is never shown.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from jinja2 import Template
from openai.types.shared import ReasoningEffort
from rapidfuzz.distance import Levenshtein

from mlime.endpoint import Answer, Endpoint, answer_all
from mlime.rescore import DumpedRecord, read_dump
from mlime.settings import LlmSettings

PROMPT = "templates/generate_prompt.txt"

#: What the model may wrap or end the sentence with and still be read as it.
QUOTES = "\"'“”‘’「」『』《》"
TRAILING = "。！？!?.,，；;"


@dataclass(frozen=True)
class Generation:
    """What the model wrote for a record, or why it wrote nothing."""

    record: DumpedRecord
    text: str | None
    reason: str = ""


@dataclass(frozen=True)
class GenerateResult:
    """What generating did on one slice, next to the beam it was measured against."""

    records: int
    generated_top1: float
    beam_top1: float
    oracle: float
    characters: int
    characters_right: float
    length_mismatches: int
    unanswered: int

    def as_dict(self) -> dict[str, object]:
        """A JSON-friendly view."""
        return {
            "records": self.records,
            "generated_top1": self.generated_top1,
            "beam_top1": self.beam_top1,
            "oracle": self.oracle,
            "characters": self.characters,
            "characters_right": self.characters_right,
            "length_mismatches": self.length_mismatches,
            "unanswered": self.unanswered,
        }


def prompt_template() -> Template:
    """The generation prompt, kept in a file so it can be diffed and reviewed."""
    source: str = resources.files(__spec__.parent).joinpath(PROMPT).read_text(encoding="utf-8")
    # `jinja2.Template.__new__` is declared to return `Any`, so pin the type here.
    template: Template = Template(source, keep_trailing_newline=True)
    return template


def parse_sentence(content: str) -> str:
    """The sentence the model answered with: its first non-empty line, unquoted, unpunctuated."""
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        raise ValueError("the answer is empty")
    sentence = lines[0].strip(QUOTES + TRAILING).strip()
    if not sentence:
        raise ValueError(f"the answer holds no sentence: {content!r}")
    if any(char.isspace() for char in sentence):
        raise ValueError(f"the answer is not one unbroken sentence: {content!r}")
    return sentence


def characters_right(expected: str, generated: str) -> int:
    """How many of *expected*'s characters *generated* gets right, by edit distance.

    The decoder's character metric is positional over hypotheses that always
    have the expected length; a generated sentence may not, so the count is
    the length less the Levenshtein distance, floored at zero. For a sentence
    of the right length that is at least the positional count.
    """
    return max(0, len(expected) - Levenshtein.distance(expected, generated))


class Generator:
    """Writes the sentence for each record through a chat completion endpoint."""

    def __init__(self, endpoint: Endpoint, with_hypotheses: bool):
        self._endpoint = endpoint
        self._with_hypotheses = with_hypotheses
        self._template = prompt_template()

    @classmethod
    def from_settings(
        cls,
        settings: LlmSettings,
        concurrency: int,
        reasoning_effort: ReasoningEffort,
        with_hypotheses: bool,
    ) -> Generator:
        """Build a client from the ``MLIME_LLM_*`` environment."""
        return cls(Endpoint.from_settings(settings, concurrency, reasoning_effort), with_hypotheses)

    @property
    def model(self) -> str:
        """The model id the sentences come from."""
        return self._endpoint.model

    @property
    def reasoning_effort(self) -> str:
        """The reasoning effort every request is sent at."""
        return self._endpoint.reasoning_effort

    @property
    def with_hypotheses(self) -> bool:
        """Whether the prompt shows the beam's hypotheses as hints."""
        return self._with_hypotheses

    def prompt(self, record: DumpedRecord) -> str:
        """Render the prompt for *record*."""
        return self._template.render(
            context=record.context,
            pinyin=record.pinyin,
            hypotheses=[h.text for h in record.hypotheses] if self._with_hypotheses else [],
        )

    async def generate_all(
        self, records: Sequence[DumpedRecord], answers: Path
    ) -> list[Generation]:
        """Generate for every record concurrently, keeping every answer in *answers*."""

        async def one(record: DumpedRecord) -> Answer[str]:
            return await self._endpoint.ask(self.prompt(record), parse_sentence, record.index)

        results = await answer_all(records, answers, lambda r: r.index, one, str)
        return [
            Generation(record=record, text=answer.value, reason=answer.reason)
            for record, answer in zip(records, results, strict=True)
        ]


def evaluate(generations: Sequence[Generation]) -> GenerateResult:
    """Generated top-1 and characters right, next to the beam's top-1 and oracle."""
    total = len(generations)
    characters = sum(len(g.record.text) for g in generations)
    return GenerateResult(
        records=total,
        generated_top1=sum(g.text == g.record.text for g in generations) / total,
        beam_top1=sum(g.record.hypotheses[0].text == g.record.text for g in generations) / total,
        oracle=sum(any(h.text == g.record.text for h in g.record.hypotheses) for g in generations)
        / total,
        characters=characters,
        characters_right=sum(
            characters_right(g.record.text, g.text) for g in generations if g.text is not None
        )
        / characters,
        length_mismatches=sum(
            g.text is not None and len(g.text) != len(g.record.text) for g in generations
        ),
        unanswered=sum(g.text is None for g in generations),
    )


@dataclass(frozen=True)
class GenerateReport:
    """What the experiment found on one slice of one eval set."""

    model: str
    reasoning_effort: str
    with_hypotheses: bool
    result: GenerateResult
    generations: tuple[Generation, ...]

    def as_dict(self) -> dict[str, object]:
        """A JSON-friendly view, with every sentence so the answers can be inspected."""
        return {
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "with_hypotheses": self.with_hypotheses,
            "result": self.result.as_dict(),
            "generations": [
                {
                    "record": g.record.index,
                    "expected": g.record.text,
                    "generated": g.text,
                    "reason": g.reason,
                }
                for g in self.generations
            ],
        }

    def render(self) -> str:
        """A few lines for the terminal."""
        r = self.result
        hints = (
            "with the beam's hypotheses" if self.with_hypotheses else "from context and keystrokes"
        )
        return "\n".join(
            [
                f"model {self.model}, reasoning effort {self.reasoning_effort}, {hints}",
                f"{r.records} records: generated top-1 {r.generated_top1:.4f}, "
                f"beam top-1 {r.beam_top1:.4f}, oracle {r.oracle:.4f}",
                f"characters right {r.characters_right:.4f} of {r.characters}, "
                f"length mismatches {r.length_mismatches}, unanswered {r.unanswered}",
            ]
        )


def generate(
    dump: Path,
    eval_set: Path,
    answers: Path,
    concurrency: int,
    reasoning_effort: ReasoningEffort,
    with_hypotheses: bool,
) -> GenerateReport:
    """Generate for every record of *dump*, resuming from *answers*, and report the slice."""
    generator = Generator.from_settings(
        LlmSettings.load(), concurrency, reasoning_effort, with_hypotheses
    )
    generations = asyncio.run(generator.generate_all(read_dump(dump, eval_set), answers))
    return GenerateReport(
        model=generator.model,
        reasoning_effort=generator.reasoning_effort,
        with_hypotheses=generator.with_hypotheses,
        result=evaluate(generations),
        generations=tuple(generations),
    )
