"""The LLM annotator: one prompted request per sentence, over an OpenAI-compatible API.

This is the second, independent opinion that makes the agreement filter worth
anything, so it must not share g2pW's failure modes. Three things make that hold:

* ``reasoning_effort`` is pinned to ``high``. Measured on a polyphone probe, the
  same model scores 5/6 at its default effort and 10/10 at high, and the items
  that flip are precisely the hard ones (得 dé/děi, 朝 cháo/zhāo) -- the ones this
  whole stage exists to catch.
* The prompt hands the model an *enumerated* character list and demands an array
  of the same length, so a reply can be checked against the sentence instead of
  being aligned by guesswork.
* A reply that will not parse, or comes back the wrong length, is retried once
  and then recorded as a refusal. It is never patched up, because a repaired
  label is indistinguishable from a correct one downstream.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from importlib import resources

import regex
from jinja2 import Template
from openai import AsyncOpenAI, OpenAIError
from openai.types.shared import ReasoningEffort
from pydantic import BaseModel, ValidationError

from mlime.logging import log
from mlime.settings import LlmSettings

from .g2p import Annotator, Outcome, Reading, Refusal
from .text import han_characters

#: A tone-numbered syllable in typing spelling: letters only, then one tone digit.
SYLLABLE = regex.compile(r"\A[a-z]+[1-5]\Z")

_FENCE = regex.compile(r"\A```[a-z]*\n(?P<body>.*)\n```\Z", regex.DOTALL)

#: The prompt file, resolved as package data so it survives being installed as a wheel.
PROMPT = "templates/g2p_prompt.txt"


class LlmReadings(BaseModel):
    """The JSON body the prompt asks for."""

    readings: list[str]


def prompt_template() -> Template:
    """The annotation prompt, kept in a file so it can be diffed and reviewed."""
    source: str = resources.files(__spec__.parent).joinpath(PROMPT).read_text(encoding="utf-8")
    # `jinja2.Template.__new__` is declared to return `Any`, so pin the type here.
    template: Template = Template(source, keep_trailing_newline=True)
    return template


def parse_readings(content: str, expected: Sequence[str]) -> Reading:
    """Parse a reply into one syllable per expected character, raising on any deviation."""
    stripped = content.strip()
    fenced = _FENCE.match(stripped)
    if fenced:
        stripped = fenced.group("body")
    try:
        payload = LlmReadings.model_validate_json(stripped)
    except ValidationError as error:
        raise ValueError(f"reply is not the expected JSON object: {stripped[:200]!r}") from error
    if len(payload.readings) != len(expected):
        raise ValueError(
            f"got {len(payload.readings)} readings for {len(expected)} characters: "
            f"{payload.readings}"
        )
    for character, syllable in zip(expected, payload.readings, strict=True):
        if not SYLLABLE.match(syllable):
            raise ValueError(f"{syllable!r} is not a tone-numbered syllable (for {character!r})")
    return Reading(tuple(payload.readings))


class LlmAnnotator(Annotator):
    """Annotates sentences through an OpenAI-compatible chat completion endpoint."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        concurrency: int = 8,
        retries: int = 1,
        reasoning_effort: ReasoningEffort = "high",
    ):
        self._client = client
        self._model = model
        self._retries = retries
        self._reasoning_effort = reasoning_effort
        self._template = prompt_template()
        self._gate = asyncio.Semaphore(concurrency)

    @classmethod
    def from_settings(cls, settings: LlmSettings, concurrency: int = 8) -> LlmAnnotator:
        """Build a client from the ``MLIME_LLM_*`` environment."""
        client = AsyncOpenAI(base_url=settings.base_url, api_key=settings.api_key)
        return cls(client, settings.model, concurrency=concurrency)

    @property
    def name(self) -> str:
        """Column name this annotator's readings are stored under."""
        return "llm"

    def prompt(self, text: str) -> str:
        """Render the annotation prompt for *text*."""
        return self._template.render(sentence=text, characters=han_characters(text))

    async def annotate(self, texts: Sequence[str]) -> list[Outcome]:
        """Annotate the batch concurrently, bounded by the configured semaphore."""
        return list(await asyncio.gather(*(self._one(text) for text in texts)))

    async def _one(self, text: str) -> Outcome:
        """Annotate one sentence, retrying once before giving up on it."""
        characters = han_characters(text)
        if not characters:
            return Refusal("no Han characters to read")
        last = ""
        for attempt in range(self._retries + 1):
            async with self._gate:
                try:
                    return parse_readings(await self._complete(text), characters)
                except (ValueError, OpenAIError) as error:
                    last = f"{type(error).__name__}: {error}"
            log.debug("llm annotation retry", text=text, attempt=attempt, reason=last)
        return Refusal(last)

    async def _complete(self, text: str) -> str:
        """One chat completion, at the pinned reasoning effort."""
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": self.prompt(text)}],
            reasoning_effort=self._reasoning_effort,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("the endpoint returned a message with no content")
        return content
