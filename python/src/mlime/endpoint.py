"""A chat-completion endpoint asked one question per record, resumably.

The reranking and generation experiments differ only in the prompt they
send and the answer they parse; the endpoint, the bounded concurrency, the
back-off on the subscription's burst limit, the retries on a malformed
answer, and the JSONL file that lets a cut-off run resume are the same. They
live here once.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, OpenAIError, RateLimitError
from openai.types.shared import ReasoningEffort

from mlime.logging import log
from mlime.settings import LlmSettings


@dataclass(frozen=True)
class Answer[T]:
    """What the model answered for one record, or why it answered nothing."""

    value: T | None
    reason: str = ""


class Endpoint:
    """One ``MLIME_LLM_*`` endpoint at a pinned reasoning effort."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        concurrency: int = 8,
        retries: int = 12,
        reasoning_effort: ReasoningEffort = "high",
    ):
        self._client = client
        self._model = model
        self._retries = retries
        self._reasoning_effort = reasoning_effort
        self._gate = asyncio.Semaphore(concurrency)

    @classmethod
    def from_settings(
        cls, settings: LlmSettings, concurrency: int, reasoning_effort: ReasoningEffort
    ) -> Endpoint:
        """Build a client from the ``MLIME_LLM_*`` environment."""
        client = AsyncOpenAI(base_url=settings.base_url, api_key=settings.api_key)
        return cls(
            client, settings.model, concurrency=concurrency, reasoning_effort=reasoning_effort
        )

    @property
    def model(self) -> str:
        """The model id the answers come from."""
        return self._model

    @property
    def reasoning_effort(self) -> str:
        """The reasoning effort every request is sent at."""
        return str(self._reasoning_effort)

    async def ask[T](self, prompt: str, parse: Callable[[str], T], record: int) -> Answer[T]:
        """Send *prompt* until *parse* accepts the answer, or give up after the retries."""
        last = ""
        for attempt in range(self._retries + 1):
            async with self._gate:
                try:
                    return Answer(value=parse(await self._complete(prompt)))
                except RateLimitError as error:
                    last = f"{type(error).__name__}: {error}"
                    # The subscription meters bursts; back off exponentially,
                    # jittered so the in-flight requests do not retry as one.
                    pause = min(60.0, 2.0**attempt) * (0.5 + random.random())
                    log.debug("rate limited", record=record, pause=round(pause, 1))
                    await asyncio.sleep(pause)
                    continue
                except (ValueError, OpenAIError) as error:
                    last = f"{type(error).__name__}: {error}"
            log.debug("retry", record=record, attempt=attempt, reason=last)
        log.warning("unanswered", record=record, reason=last)
        return Answer(value=None, reason=last)

    async def _complete(self, prompt: str) -> str:
        """One chat completion."""
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort=self._reasoning_effort,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("the endpoint returned a message with no content")
        return content


async def answer_all[R, T](
    records: Sequence[R],
    answers: Path,
    key: Callable[[R], int],
    ask: Callable[[R], Awaitable[Answer[T]]],
    decode: Callable[[Any], T],
    every: int = 500,
) -> list[Answer[T]]:
    """Ask for every record concurrently, keeping every answer in *answers*.

    *answers* is a JSONL file of ``{"record", "answer"}`` rows appended as
    answers arrive, so a run cut off by the endpoint's limits resumes from
    where it stopped instead of paying for the answered records again. The
    answer value has to be JSON as written; *decode* turns it back.
    """
    known: dict[int, T] = {}
    if answers.is_file():
        with answers.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    known[int(row["record"])] = decode(row["answer"])
        log.info("resuming", answers=str(answers), answered=len(known))
    done = 0
    lock = asyncio.Lock()

    async def one(record: R) -> Answer[T]:
        nonlocal done
        index = key(record)
        if index in known:
            return Answer(value=known[index])
        answer = await ask(record)
        async with lock:
            if answer.value is not None:
                with answers.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"record": index, "answer": answer.value}) + "\n")
            done += 1
            if done % every == 0:
                log.info("answered", records=done, of=len(records) - len(known))
        return answer

    return list(await asyncio.gather(*(one(record) for record in records)))
