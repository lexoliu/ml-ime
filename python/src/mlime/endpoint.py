"""A chat-completion endpoint asked one question per record, resumably.

The reranking and generation experiments differ only in the prompt they
send and the answer they parse; the endpoint, the bounded concurrency, the
back-off on the subscription's burst limit, the retries on a malformed
answer, and the JSONL file that lets a cut-off run resume are the same. They
live here once.

A record the endpoint does not answer is not a wrong answer: the slice is
unfinished, and a summary over it would measure the endpoint's failures
rather than the model. So a request that still fails after its retries stops
the run, the answers written so far stay in the file, and the next run
resumes from them.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from openai.types.shared import ReasoningEffort

from mlime.logging import log
from mlime.settings import LlmSettings

#: The longest pause between two attempts at one request, in seconds.
MAX_PAUSE = 60.0


class UnansweredError(RuntimeError):
    """The endpoint gave no usable answer for one record."""

    def __init__(self, record: int, reason: str):
        super().__init__(f"record {record}: {reason}")
        self.record = record
        self.reason = reason


class UnfinishedError(RuntimeError):
    """A slice stopped before every record was answered; the answers so far are kept."""

    def __init__(
        self, failures: Sequence[UnansweredError], answers: Path, answered: int, total: int
    ):
        first = failures[0]
        super().__init__(
            f"{len(failures)} request(s) failed, the first for record {first.record}: "
            f"{first.reason}\n{answered} of {total} records are answered in {answers}; "
            "run again to resume from them"
        )
        self.failures = tuple(failures)
        self.answered = answered
        self.total = total


def exhausted(error: RateLimitError) -> bool:
    """Whether a 429 says the account is out of quota rather than over its burst rate.

    OpenAI-compatible endpoints send both as 429 and tell them apart by the
    error code; only the burst limit clears by waiting.
    """
    return error.code == "insufficient_quota"


def transient(error: APIConnectionError | APIStatusError) -> bool:
    """Whether the same request, sent again after a pause, can succeed.

    A dropped connection, a server error, a request or lock timeout and a burst
    limit can; any other refusal (bad request, authentication, permission,
    payment, an exhausted quota) is the same refusal on every attempt.
    """
    if isinstance(error, APIConnectionError | InternalServerError):
        return True
    if isinstance(error, RateLimitError):
        return not exhausted(error)
    return error.status_code in (408, 409)


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

    async def ask[T](self, prompt: str, parse: Callable[[str], T], record: int) -> T:
        """Send *prompt* until *parse* accepts the answer.

        A malformed answer is asked again at once and a transient endpoint
        error after a jittered exponential pause. A refusal no retry can
        change raises `UnansweredError` at once, and so does the last failed attempt.
        """
        last = ""
        for attempt in range(self._retries + 1):
            async with self._gate:
                try:
                    return parse(await self._complete(prompt))
                except ValueError as error:
                    last = f"{type(error).__name__}: {error}"
                    pause = 0.0
                except (APIConnectionError, APIStatusError) as error:
                    code = "" if error.code is None else f" ({error.code})"
                    last = f"{type(error).__name__}{code}: {error}"
                    if not transient(error):
                        raise UnansweredError(record, last) from error
                    # Jittered so the in-flight requests do not retry as one.
                    pause = min(MAX_PAUSE, 2.0**attempt) * (0.5 + random.random())
            log.debug("retry", record=record, attempt=attempt, pause=round(pause, 1), reason=last)
            await asyncio.sleep(pause)
        raise UnansweredError(record, f"{last} (after {self._retries} retries)")

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
    ask: Callable[[R], Awaitable[T]],
    decode: Callable[[Any], T],
    every: int = 500,
) -> list[T]:
    """Ask for every record concurrently, keeping every answer in *answers*.

    *answers* is a JSONL file of ``{"record", "answer"}`` rows appended as
    answers arrive, so a run cut off by the endpoint's limits resumes from
    where it stopped instead of paying for the answered records again. The
    answer value has to be JSON as written; *decode* turns it back.

    The first record *ask* fails with `UnansweredError` cancels the requests still
    in flight and raises `UnfinishedError`; every answer that arrived before it is
    already in *answers*. The answers come back in the order of *records*.
    """
    known: dict[int, T] = {}
    if answers.is_file():
        with answers.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    known[int(row["record"])] = decode(row["answer"])
        log.info("resuming", answers=str(answers), answered=len(known))
    asked = [record for record in records if key(record) not in known]
    done = 0

    async def one(record: R) -> None:
        nonlocal done
        answer = await ask(record)
        # Nothing awaits between the answer and its row, so a cancelled
        # request never leaves a partial row and rows never interleave.
        with answers.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"record": key(record), "answer": answer}) + "\n")
        known[key(record)] = answer
        done += 1
        if done % every == 0:
            log.info("answered", records=done, of=len(asked))

    try:
        async with asyncio.TaskGroup() as group:
            for record in asked:
                group.create_task(one(record))
    except* UnansweredError as failed:
        failures = [error for error in failed.exceptions if isinstance(error, UnansweredError)]
        answered = sum(key(record) in known for record in records)
        raise UnfinishedError(failures, answers, answered, len(records)) from failed
    return [known[key(record)] for record in records]
