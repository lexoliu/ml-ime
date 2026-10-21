"""A request the endpoint will not answer stops the slice; the answers so far resume it.

The endpoint is faked at the client: every request is routed by the record
its prompt names (each record's pinyin is unique), so a test says per record
what the endpoint does -- answer, refuse, or hang until it is cancelled.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2
import pytest
from openai import InternalServerError, RateLimitError
from typer.testing import CliRunner

from mlime import endpoint
from mlime.__main__ import app
from mlime.endpoint import Endpoint, UnansweredError

#: The expected sentence of each record of the slice.
TEXTS = ["你好", "再见", "谢谢", "晚安", "早安"]


def _status(kind: type[RateLimitError | InternalServerError], code: str | None) -> Any:
    status = 429 if kind is RateLimitError else 500
    request = httpx2.Request("POST", "http://endpoint.test/v1/chat/completions")
    body = {"message": "refused", "code": code}
    return kind("refused", response=httpx2.Response(status, request=request), body=body)


def quota() -> RateLimitError:
    """The 429 an account out of quota answers with."""
    error: RateLimitError = _status(RateLimitError, "insufficient_quota")
    return error


def burst() -> RateLimitError:
    """The 429 a burst over the rate limit answers with."""
    error: RateLimitError = _status(RateLimitError, "rate_limit_exceeded")
    return error


def server() -> InternalServerError:
    """A 500."""
    error: InternalServerError = _status(InternalServerError, None)
    return error


#: What the fake endpoint does for a record: the content it answers, an
#: error to raise, or None to hang until the request is cancelled.
Behaviour = str | Exception | None

#: Installs a client scripted per record and attempt for the CLI to build.
type Install = Callable[[Callable[[int, int], Behaviour]], FakeClient]


@dataclass
class FakeClient:
    """An ``AsyncOpenAI`` stand-in whose answers are scripted per record."""

    script: Callable[[int, int], Behaviour]
    calls: dict[int, int] = field(default_factory=dict)
    cancelled: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, *, model: str, messages: list[dict[str, str]], **_: object) -> Any:
        record = next(i for i in range(len(TEXTS)) if f"pin{i}x" in messages[0]["content"])
        attempt = self.calls.get(record, 0)
        self.calls[record] = attempt + 1
        behaviour = self.script(record, attempt)
        if isinstance(behaviour, Exception):
            raise behaviour
        if behaviour is None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.append(record)
                raise
        message = SimpleNamespace(content=behaviour)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@dataclass
class Slice:
    """A five-record slice on disk: its dump, its eval set, and where a run writes."""

    dump: Path
    eval_set: Path
    answers: Path
    out: Path


@pytest.fixture
def slice_(tmp_path: Path) -> Slice:
    eval_set = tmp_path / "eval.jsonl"
    dump = tmp_path / "dump.jsonl"
    eval_set.write_text(
        "".join(
            json.dumps({"text": text, "pinyin": f"pin{i}x", "context": None}) + "\n"
            for i, text in enumerate(TEXTS)
        ),
        encoding="utf-8",
    )
    dump.write_text(
        "".join(
            json.dumps(
                {
                    "record": i,
                    "text": text,
                    "hypotheses": [{"text": text, "score": 0.0}, {"text": "错误", "score": -1.0}],
                }
            )
            + "\n"
            for i, text in enumerate(TEXTS)
        ),
        encoding="utf-8",
    )
    return Slice(dump, eval_set, tmp_path / "answers.jsonl", tmp_path / "report.json")


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> Iterator[Install]:
    """Install a scripted client in place of the endpoint's for the CLI to build."""
    monkeypatch.setenv("MLIME_LLM_BASE_URL", "http://endpoint.test/v1")
    monkeypatch.setenv("MLIME_LLM_API_KEY", "test")
    monkeypatch.setenv("MLIME_LLM_MODEL", "test-model")
    monkeypatch.setattr(endpoint, "MAX_PAUSE", 0.0)

    def install(script: Callable[[int, int], Behaviour]) -> FakeClient:
        client = FakeClient(script)
        monkeypatch.setattr(endpoint, "AsyncOpenAI", lambda **_: client)
        return client

    yield install


def generate(slice_: Slice) -> Any:
    argv = [
        "eval",
        "generate",
        "--dump",
        str(slice_.dump),
        "--eval-set",
        str(slice_.eval_set),
        "--answers",
        str(slice_.answers),
        "--out",
        str(slice_.out),
    ]
    return CliRunner().invoke(app, argv)


def answered(answers: Path) -> dict[int, object]:
    rows = [json.loads(line) for line in answers.read_text(encoding="utf-8").splitlines()]
    return {row["record"]: row["answer"] for row in rows}


def test_a_failed_request_stops_the_slice_and_a_rerun_resumes_it(
    slice_: Slice, fake: Install
) -> None:
    # Records 0-2 answer, 3 is refused for an exhausted quota, 4 is still in
    # flight when the run stops.
    first = fake(
        lambda record, _: TEXTS[record] if record < 3 else quota() if record == 3 else None
    )
    result = generate(slice_)
    assert result.exit_code == 1
    assert "record 3" in result.output
    assert "insufficient_quota" in result.output
    assert "records:" not in result.output
    assert not slice_.out.exists()
    assert answered(slice_.answers) == {0: TEXTS[0], 1: TEXTS[1], 2: TEXTS[2]}
    # The quota error is not retried, and the request in flight is cancelled.
    assert first.calls[3] == 1
    assert first.cancelled == [4]

    second = fake(lambda record, _: TEXTS[record] if record != 4 else "错误")
    result = generate(slice_)
    assert result.exit_code == 0, result.output
    assert sorted(second.calls) == [3, 4]
    report = json.loads(slice_.out.read_text(encoding="utf-8"))
    assert report["result"]["records"] == 5
    assert report["result"]["generated_top1"] == pytest.approx(4 / 5)
    assert "unanswered" not in report["result"]


def test_rescore_stops_on_a_failed_request_without_a_report(slice_: Slice, fake: Install) -> None:
    client = fake(lambda record, _: "1" if record != 2 else "no number at all")
    argv = [
        "eval",
        "rescore",
        "--dump",
        str(slice_.dump),
        "--eval-set",
        str(slice_.eval_set),
        "--picks",
        str(slice_.answers),
        "--out",
        str(slice_.out),
    ]
    result = CliRunner().invoke(app, argv)
    assert result.exit_code == 1
    assert "record 2" in result.output
    assert "after 12 retries" in result.output
    assert not slice_.out.exists()
    assert client.calls[2] == 13
    assert answered(slice_.answers) == {0: 0, 1: 0, 3: 0, 4: 0}


def _endpoint(script: Callable[[int, int], Behaviour], retries: int) -> tuple[Endpoint, FakeClient]:
    client = FakeClient(script)
    return Endpoint(client, "test-model", retries=retries), client  # type: ignore[arg-type]


def _ask(endpoint_: Endpoint) -> str:
    return asyncio.run(endpoint_.ask("pin0x", str, 0))


def test_transient_errors_are_retried_until_one_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(endpoint, "MAX_PAUSE", 0.0)
    errors = [burst(), server()]
    endpoint_, client = _endpoint(lambda _, attempt: errors[attempt] if attempt < 2 else "好", 2)
    assert _ask(endpoint_) == "好"
    assert client.calls[0] == 3


def test_transient_errors_past_the_retries_leave_the_record_unanswered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(endpoint, "MAX_PAUSE", 0.0)
    endpoint_, client = _endpoint(lambda _, __: server(), 2)
    with pytest.raises(UnansweredError, match=r"record 0: InternalServerError.*after 2 retries"):
        _ask(endpoint_)
    assert client.calls[0] == 3


def test_an_exhausted_quota_is_not_retried() -> None:
    endpoint_, client = _endpoint(lambda _, __: quota(), 12)
    with pytest.raises(UnansweredError, match="insufficient_quota"):
        _ask(endpoint_)
    assert client.calls[0] == 1
