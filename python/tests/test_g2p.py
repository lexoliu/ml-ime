"""Dual annotation is the project's honesty mechanism, so its edges are tested first.

Two things must hold. Agreement has to be judged on the toneless syllable, because
that is what a keyboard produces and what the masks are keyed on -- a tone-only
difference must not cost a training pair. And an annotator that fails must produce
a recorded refusal, never a shortened list that would silently misalign every
character after it.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import polars as pl
import pytest
from openai import AsyncOpenAI

from mlime.data.corpus import Sample
from mlime.data.g2p import (
    ANNOTATED,
    HARD,
    REFUSED,
    Annotator,
    Comparison,
    Outcome,
    Reading,
    Refusal,
    annotate,
    compare,
)
from mlime.data.g2p_report import by_frequency, summarise, worst_characters
from mlime.data.g2pw_annotator import default_workers
from mlime.data.llm_annotator import LlmAnnotator, parse_readings, prompt_template
from mlime.data.shards import read_shards


class FakeAnnotator(Annotator):
    """Replays canned outcomes, so the agreement logic is tested without a model."""

    def __init__(self, name: str, outcomes: dict[str, Outcome]):
        self._name = name
        self.outcomes = outcomes

    @property
    def name(self) -> str:
        return self._name

    async def annotate(self, texts: Sequence[str]) -> list[Outcome]:
        return [self.outcomes[text] for text in texts]


def test_tone_only_differences_still_agree() -> None:
    """A tone key does not exist on a pinyin keyboard, so a tone clash costs nothing."""
    comparison = Comparison(("重", "要"), ("zhong4", "yao4"), ("zhong2", "yao4"))
    assert comparison.agree == (True, True)
    assert comparison.unanimous


def test_a_different_syllable_is_a_disagreement() -> None:
    comparison = Comparison(("还",), ("hai2",), ("huan2",))
    assert comparison.agree == (False,)
    assert not comparison.unanimous


def test_umlaut_spellings_agree_with_their_typed_form() -> None:
    assert Comparison(("绿",), ("lv4",), ("lü4",)).unanimous


def test_readings_of_different_lengths_cannot_be_compared() -> None:
    """A silent truncation here would misalign every character after it."""
    with pytest.raises(ValueError, match="different lengths"):
        Comparison(("重", "要"), ("zhong4",), ("zhong4", "yao4"))


def test_comparison_lines_up_with_the_han_characters_only() -> None:
    comparison = compare(
        "他在2026年", Reading(("ta1", "zai4", "nian2")), Reading(("ta1", "zai4", "nian2"))
    )
    assert comparison.characters == ("他", "在", "年")


def _samples() -> list[Sample]:
    return [
        Sample("a", "wiki", "重要", "上下文"),
        Sample("b", "news", "还钱", None),
        Sample("c", "dialogue", "你好", None),
    ]


def test_annotate_splits_agreement_from_disagreement_and_refusal(tmp_path: Path) -> None:
    first = FakeAnnotator(
        "g2pw",
        {
            "重要": Reading(("zhong4", "yao4")),
            "还钱": Reading(("hai2", "qian2")),
            "你好": Refusal("g2pw has no reading for '你'"),
        },
    )
    second = FakeAnnotator(
        "llm",
        {
            "重要": Reading(("zhong4", "yao4")),
            "还钱": Reading(("huan2", "qian2")),
            "你好": Reading(("ni3", "hao3")),
        },
    )
    counts = asyncio.run(annotate(_samples(), first, second, tmp_path, batch_size=2))
    assert (counts.agreed, counts.disagreed, counts.refused) == (1, 1, 1)
    assert counts.agreement_rate == pytest.approx(0.5)

    annotated = read_shards(tmp_path, ANNOTATED)
    assert annotated.height == 2
    assert annotated.columns == [
        "id",
        "source",
        "text",
        "context",
        "characters",
        "g2pw",
        "llm",
        "agree",
        "agree_all",
    ]
    assert annotated.filter(pl.col("id") == "a")["context"].item() == "上下文"

    hard = read_shards(tmp_path, HARD)
    assert hard["text"].to_list() == ["还钱"]

    refused = read_shards(tmp_path, REFUSED)
    assert refused["annotator"].to_list() == ["g2pw"]
    assert "你" in refused["reason"].item()


def test_a_refused_sentence_is_never_written_as_annotated(tmp_path: Path) -> None:
    """Half an annotation is not an annotation; it must not reach the training set."""
    first = FakeAnnotator("g2pw", {"重要": Refusal("down")})
    second = FakeAnnotator("llm", {"重要": Reading(("zhong4", "yao4"))})
    counts = asyncio.run(annotate([Sample("a", "wiki", "重要", None)], first, second, tmp_path))
    assert counts == type(counts)(agreed=0, disagreed=0, refused=1)
    assert not list(tmp_path.glob(f"{ANNOTATED}-*.parquet"))


def test_the_prompt_enumerates_the_characters_to_annotate() -> None:
    """Alignment is the LLM's main failure mode; numbering the input removes the guesswork."""
    rendered = prompt_template().render(sentence="重要", characters=["重", "要"])
    assert "0: 重" in rendered
    assert "1: 要" in rendered
    assert "2" in rendered


def test_a_well_formed_reply_parses() -> None:
    reply = json.dumps({"readings": ["zhong4", "yao4"]})
    assert parse_readings(reply, ["重", "要"]).syllables == ("zhong4", "yao4")


def test_a_fenced_reply_parses() -> None:
    """Chat models wrap JSON in a code fence however firmly they are told not to."""
    reply = '```json\n{"readings": ["zhong4", "yao4"]}\n```'
    assert parse_readings(reply, ["重", "要"]).syllables == ("zhong4", "yao4")


@pytest.mark.parametrize(
    ("reply", "match"),
    [
        ("not json at all", "not the expected JSON"),
        ('{"readings": ["zhong4"]}', "1 readings for 2 characters"),
        ('{"readings": ["zhong4", "yao"]}', "not a tone-numbered syllable"),
        ('{"readings": ["zhong4", "yào"]}', "not a tone-numbered syllable"),
        ('{"pinyin": ["zhong4", "yao4"]}', "not the expected JSON"),
    ],
)
def test_a_malformed_reply_raises_rather_than_being_repaired(reply: str, match: str) -> None:
    """A repaired label is indistinguishable from a correct one downstream."""
    with pytest.raises(ValueError, match=match):
        parse_readings(reply, ["重", "要"])


def _completion(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "1",
            "object": "chat.completion",
            "created": 0,
            "model": "fake",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        },
    )


def _annotator(replies: list[str], seen: list[dict[str, Any]]) -> LlmAnnotator:
    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return _completion(replies[len(seen) - 1])

    client = AsyncOpenAI(
        base_url="http://annotator.invalid/v1",
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    return LlmAnnotator(client, "fake-model", concurrency=1)


def test_the_llm_annotator_pins_reasoning_effort_high() -> None:
    """Measured: this model reads 得 and 朝 correctly at high effort and not below it."""
    seen: list[dict[str, Any]] = []
    outcomes = asyncio.run(
        _annotator(['{"readings": ["zhong4", "yao4"]}'], seen).annotate(["重要"])
    )
    assert outcomes == [Reading(("zhong4", "yao4"))]
    assert seen[0]["reasoning_effort"] == "high"
    assert seen[0]["model"] == "fake-model"


def test_a_malformed_reply_is_retried_once() -> None:
    seen: list[dict[str, Any]] = []
    annotator = _annotator(["nonsense", '{"readings": ["zhong4", "yao4"]}'], seen)
    assert asyncio.run(annotator.annotate(["重要"])) == [Reading(("zhong4", "yao4"))]
    assert len(seen) == 2


def test_a_reply_that_stays_malformed_becomes_a_refusal() -> None:
    seen: list[dict[str, Any]] = []
    annotator = _annotator(["nonsense", "still nonsense"], seen)
    outcome = asyncio.run(annotator.annotate(["重要"]))[0]
    assert isinstance(outcome, Refusal)
    assert "not the expected JSON" in outcome.reason
    assert len(seen) == 2


def test_a_sentence_with_no_han_is_refused_without_a_request() -> None:
    seen: list[dict[str, Any]] = []
    assert asyncio.run(_annotator([], seen).annotate(["2026"])) == [
        Refusal("no Han characters to read")
    ]
    assert seen == []


def _annotated() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": ["a", "b", "c"],
            "source": ["wiki", "news", "wiki"],
            "text": ["重要还", "重要还", "重要好"],
            "context": [None, None, None],
            "characters": [["重", "要", "还"], ["重", "要", "还"], ["重", "要", "好"]],
            "g2pw": [
                ["zhong4", "yao4", "hai2"],
                ["zhong4", "yao4", "hai2"],
                ["zhong4", "yao4", "hao3"],
            ],
            "llm": [
                ["zhong4", "yao4", "huan2"],
                ["zhong4", "yao4", "hai2"],
                ["zhong4", "yao4", "hao3"],
            ],
            "agree": [[True, True, False], [True, True, True], [True, True, True]],
            "agree_all": [False, True, True],
        }
    )


def test_the_report_measures_both_rates() -> None:
    """This pair of numbers is the ceiling on every accuracy the project reports."""
    summary = summarise(_annotated(), refusals=4)
    assert summary.sentences == 3
    assert summary.characters == 9
    assert summary.sentence_rate == pytest.approx(2 / 3)
    assert summary.character_rate == pytest.approx(8 / 9)
    assert summary.refusals == 4


def test_the_report_names_the_characters_that_disagree() -> None:
    from mlime.data.g2p_report import per_character

    worst = worst_characters(per_character(_annotated()))
    assert worst["characters"].to_list() == ["还"]
    assert worst["disagreements"].to_list() == [1]
    assert worst["g2pw_example"].to_list() == ["hai2"]
    assert worst["llm_example"].to_list() == ["huan2"]


def test_the_report_buckets_by_character_frequency() -> None:
    from mlime.data.g2p_report import per_character

    bands = by_frequency(per_character(_annotated()))
    assert bands["positions"].sum() == 9
    assert set(bands["band"].to_list()) == {"top 500"}


@pytest.mark.parametrize(
    ("platform", "cores", "expected"),
    [("darwin", 12, 0), ("win32", 12, 0), ("linux", 4, 2), ("linux", 1, 1), ("linux2", 8, 4)],
)
def test_dataloader_workers_are_linux_only(
    monkeypatch: pytest.MonkeyPatch, platform: str, cores: int, expected: int
) -> None:
    """g2pW's spawned loader workers deadlock on macOS, so only Linux gets any."""
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(os, "cpu_count", lambda: cores)
    assert default_workers() == expected


def test_an_unknowable_core_count_still_leaves_a_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """`os.cpu_count()` may return None; that must not turn into zero workers on Linux."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert default_workers() == 1
