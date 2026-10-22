"""The judge keeps a mined record only when every rule passes it, and resumes where it stopped.

The two endpoints are faked: the rules are tested through fake askers keyed by
sentence, and the HTTP side through ``httpx.MockTransport``. Every sentence here
is written for the test.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx
import polars as pl
import pytest

from mlime.data import judge
from mlime.data.judge import (
    BoundaryLexicon,
    JevAnswer,
    JudgeError,
    MinedLayout,
    Poster,
    ReviewAnswer,
    Unit,
)

LEXICON = BoundaryLexicon(["一丁不识", "花好月圆", "月圆", "识字", "鲜花", "画蛇添足"])

ENTRY_TABLE_HEADER = (
    "entry\tcategory\toccurrences\tsentences\tall_initials_per_epoch\tany_initial_per_epoch"
    "\tblocked\n"
)


def unit(text: str, entry: str = "一丁不识", context: str = "", key: str = "a") -> Unit:
    return Unit(ordinal=0, id=key, entry=entry, text=text, context=context)


# --- rule 1 -----------------------------------------------------------------


def test_an_entry_used_as_itself_is_no_false_match() -> None:
    assert LEXICON.false_match("他小时候家里穷得一丁不识", "一丁不识") is None


def test_a_word_running_on_past_the_entry_takes_its_characters() -> None:
    assert LEXICON.false_match("他小时候一丁不识字", "一丁不识") == ["识字"]


def test_a_word_running_in_from_before_the_entry_takes_its_characters() -> None:
    assert LEXICON.false_match("我送她一束鲜花好月圆的晚上", "花好月圆") == ["鲜花"]


def test_a_word_inside_the_entry_is_part_of_it() -> None:
    assert LEXICON.false_match("中秋那天果然花好月圆了", "花好月圆") is None


def test_one_free_occurrence_is_enough() -> None:
    assert LEXICON.false_match("一丁不识字的人也不是一辈子一丁不识", "一丁不识") is None


def test_a_sentence_without_its_entry_breaks_the_invariant() -> None:
    with pytest.raises(JudgeError, match="does not hold its entry"):
        LEXICON.false_match("今天天气很好", "一丁不识")


def test_the_lexicon_joins_the_entry_table_and_the_ordinary_vocabulary(tmp_path: Path) -> None:
    table = tmp_path / "entries.tsv"
    table.write_text(
        ENTRY_TABLE_HEADER + "一丁不识\tidiom\t0\t0\t0.000\t0.000\t0\n", encoding="utf-8"
    )
    lexicon = BoundaryLexicon.load(table, {"识字": "v", "人": "n"})
    assert len(lexicon) == 2
    assert lexicon.false_match("他一丁不识字", "一丁不识") == ["识字"]


def test_a_table_of_another_shape_is_refused(tmp_path: Path) -> None:
    table = tmp_path / "entries.tsv"
    table.write_text("word\tcategory\n一丁不识\tidiom\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not #215's entry table"):
        BoundaryLexicon.load(table, {})


# --- the HTTP side ------------------------------------------------------------


@dataclass
class Sleeps:
    pauses: list[float] = field(default_factory=list)

    async def __call__(self, seconds: float) -> None:
        self.pauses.append(seconds)


def poster(handler: Callable[[httpx.Request], httpx.Response], sleeps: Sleeps) -> Poster:
    return Poster(httpx.AsyncClient(transport=httpx.MockTransport(handler)), sleep=sleeps)


def test_a_rate_limit_is_retried_after_the_pause_the_server_names() -> None:
    answers = iter(
        [
            httpx.Response(429, headers={"Retry-After": "7"}),
            httpx.Response(529),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    sleeps = Sleeps()
    result = asyncio.run(
        poster(lambda _: next(answers), sleeps).post("https://judge.test/", {}, {})
    )
    assert result == {"ok": True}
    assert sleeps.pauses[0] == 7.0
    assert 0.5 <= sleeps.pauses[1] <= 2.0


def test_a_dropped_connection_is_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json=[1])

    assert asyncio.run(poster(handler, Sleeps()).post("https://judge.test/", {}, {})) == [1]


def test_any_other_failure_fails_at_once() -> None:
    sleeps = Sleeps()
    with pytest.raises(JudgeError, match="answered 401"):
        asyncio.run(
            poster(lambda _: httpx.Response(401, text="bad key"), sleeps).post(
                "https://judge.test/", {}, {}
            )
        )
    assert sleeps.pauses == []


def test_a_transient_failure_that_outlasts_the_attempts_fails() -> None:
    sleeps = Sleeps()
    with pytest.raises(JudgeError, match="after 8 attempts"):
        asyncio.run(
            poster(lambda _: httpx.Response(503), sleeps).post("https://judge.test/", {}, {})
        )
    assert len(sleeps.pauses) == judge.MAX_ATTEMPTS - 1
    assert max(sleeps.pauses) <= judge.MAX_BACKOFF_SECONDS


def test_retry_after_may_be_a_date() -> None:
    response = httpx.Response(
        429,
        headers={"Retry-After": "Wed, 21 Oct 2026 07:28:30 GMT"},
        request=httpx.Request("POST", "https://judge.test/"),
    )
    now = datetime(2026, 10, 21, 7, 28, 0, tzinfo=UTC)
    assert judge.retry_after_seconds(response, now) == 30.0


def test_jev_is_asked_the_labelled_question_about_the_unit() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "model": "jev-test",
                "answers": {"natural_usage": {"noul": 0.81}},
                "usage": {"input_tokens": 10},
            },
        )

    jev = judge.Jev(poster(handler, Sleeps()), api_key="secret")
    answer = asyncio.run(jev.ask(unit("他小时候家里穷得一丁不识", context="上文")))
    assert answer == JevAnswer(p=0.81, model="jev-test")
    body = json.loads(seen[0].content)
    assert str(seen[0].url) == judge.JEV_URL
    assert seen[0].headers["authorization"] == "Bearer secret"
    assert body == {
        "model": "jev-latest",
        "state": {
            "entry": "一丁不识",
            "sentence": "他小时候家里穷得一丁不识",
            "preceding_text": "上文",
        },
        "questions": {"natural_usage": judge.QUESTION},
    }


def test_the_review_model_gets_the_prompt_with_the_unit_filled_in() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        content = '好的。{"context_spun": false, "sentence_natural": true}'
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    review = judge.Review(
        poster(handler, Sleeps()), "http://review.test/v1/", "review-model", api_key="k"
    )
    answer = asyncio.run(review.ask(unit("他小时候家里穷得一丁不识", context="他说起往事。")))
    assert answer == ReviewAnswer(context_spun=False, sentence_natural=True)
    assert str(seen[0].url) == "http://review.test/v1/chat/completions"
    body = json.loads(seen[0].content)
    assert body["model"] == "review-model"
    assert body["temperature"] == 0
    prompt = body["messages"][0]["content"]
    assert "包含词条「一丁不识」" in prompt
    assert "上文：他说起往事。\n句子：他小时候家里穷得一丁不识\n" in prompt
    assert "色情或成人内容一律算不自然。" in prompt
    assert prompt.endswith('{"context_spun": true 或 false, "sentence_natural": true 或 false}')


@pytest.mark.parametrize(
    "reply",
    [
        "我觉得是自然的",
        '{"context_spun": "no", "sentence_natural": true}',
        '{"context_spun": false}',
    ],
)
def test_a_review_reply_without_the_two_booleans_fails(reply: str) -> None:
    with pytest.raises(JudgeError):
        judge.parse_review(reply)


def test_verdicts_out_of_order_are_refused() -> None:
    with pytest.raises(JudgeError, match="out of order"):
        judge.next_stage([{"id": "a", "entry": "e", "stage": "jev", "keep": True}])


# --- the run ------------------------------------------------------------------

#: Sentence, entry, Jev's probability, and the review's (context_spun, sentence_natural).
RECORDS: list[tuple[str, str, float, tuple[bool, bool]]] = [
    ("他小时候家里穷得一丁不识", "一丁不识", 0.9, (False, True)),  # kept
    ("他小时候一丁不识字", "一丁不识", 0.9, (False, True)),  # rule 1
    ("中秋那天果然花好月圆了", "花好月圆", 0.5, (False, True)),  # Jev
    ("这一家人从此花好月圆", "花好月圆", 0.95, (True, True)),  # spun page
    ("那幅画画得一丁不识的样子", "一丁不识", 0.75, (False, False)),  # unnatural
    ("村里的老人大多一丁不识", "一丁不识", 0.7, (False, True)),  # kept at the threshold
]


@dataclass
class FakeJev:
    asked: list[str] = field(default_factory=list)

    async def ask(self, unit: Unit) -> JevAnswer:
        self.asked.append(unit.text)
        return JevAnswer(p=next(r[2] for r in RECORDS if r[0] == unit.text), model="jev-fake")


@dataclass
class FakeReview:
    fail_after: int | None = None
    asked: list[str] = field(default_factory=list)

    async def ask(self, unit: Unit) -> ReviewAnswer:
        if self.fail_after is not None and len(self.asked) >= self.fail_after:
            raise JudgeError("the review endpoint went away")
        self.asked.append(unit.text)
        spun, natural = next(r[3] for r in RECORDS if r[0] == unit.text)
        return ReviewAnswer(context_spun=spun, sentence_natural=natural)


def mined_root(root: Path, records: list[tuple[str, str]], ordinal: int = 0) -> MinedLayout:
    """A mining run's state for one upstream file holding *records* (sentence, entry)."""
    layout = MinedLayout(root)
    for directory in (layout.records, layout.unjudged, layout.entry_rows):
        directory.mkdir(parents=True, exist_ok=True)
    ids = [f"{ordinal:04d}{index:08x}" for index in range(len(records))]
    pl.DataFrame(
        {
            "id": ids,
            "source": [judge.SOURCE] * len(records),
            "text": [text for text, _ in records],
            "context": ["前文" * 100 if index == 0 else None for index in range(len(records))],
        }
    ).write_parquet(layout.unjudged / f"{judge.SOURCE}-{ordinal:04d}000.parquet")
    (layout.entry_rows / f"{ordinal:04d}.tsv").write_text(
        "".join(f"{key}\t{entry}\n" for key, (_, entry) in zip(ids, records, strict=True)),
        encoding="utf-8",
    )
    (layout.records / f"{ordinal:04d}.json").write_text(
        json.dumps({"ordinal": ordinal, "census": {"written": len(records)}}), encoding="utf-8"
    )
    layout.mined_entries.write_text(
        "entry\tcategory\tminable\tkept\n一丁不识\tidiom\ttrue\t4\n花好月圆\tidiom\ttrue\t2\n"
        "画蛇添足\tidiom\ttrue\t0\n",
        encoding="utf-8",
    )
    return layout


def run(layout: MinedLayout, jev: FakeJev, review: FakeReview) -> judge.JudgeCensus:
    return asyncio.run(judge.judge_source(layout, LEXICON, jev, review, 3, 2))


def test_only_records_every_rule_passes_reach_the_samples(tmp_path: Path) -> None:
    layout = mined_root(tmp_path, [(text, entry) for text, entry, _, _ in RECORDS])
    jev, review = FakeJev(), FakeReview()
    census = run(layout, jev, review)

    assert census == judge.JudgeCensus(
        records=6,
        units=6,
        boundary=1,
        jev=1,
        review_spun=1,
        review_unnatural=1,
        kept_units=2,
        kept_records=2,
    )
    # Each rule runs only on what passed the one before.
    assert "他小时候一丁不识字" not in jev.asked
    assert sorted(review.asked) == sorted(
        [RECORDS[0][0], RECORDS[3][0], RECORDS[4][0], RECORDS[5][0]]
    )
    samples = pl.read_parquet(layout.samples / f"{judge.SOURCE}-0000000.parquet")
    assert samples.get_column("text").to_list() == [RECORDS[0][0], RECORDS[5][0]]
    assert samples.columns == ["id", "source", "text", "context"]
    entries = pl.read_csv(layout.judged_entries, separator="\t")
    assert entries.select("entry", "mined", "kept").rows() == [
        ("一丁不识", 4, 2),
        ("花好月圆", 2, 0),
        ("画蛇添足", 0, 0),
    ]
    summary = json.loads(layout.census.read_text(encoding="utf-8"))
    assert summary["files_judged"] == 1
    assert summary["census"]["kept_records"] == 2
    kept_rows = (layout.kept_rows(0)).read_text(encoding="utf-8").splitlines()
    assert [row.split("\t")[1] for row in kept_rows] == ["一丁不识", "一丁不识"]


def test_the_judges_see_the_end_of_the_context(tmp_path: Path) -> None:
    layout = mined_root(tmp_path, [(RECORDS[0][0], RECORDS[0][1])])
    units, _ = judge.load_units(layout, 0)
    assert units[0].context == "前文" * (judge.CONTEXT_CHARACTERS // 2)


def test_a_stopped_run_resumes_without_asking_again(tmp_path: Path) -> None:
    layout = mined_root(tmp_path, [(text, entry) for text, entry, _, _ in RECORDS])
    with pytest.raises(ExceptionGroup) as stopped:
        run(layout, FakeJev(), FakeReview(fail_after=1))
    assert stopped.group_contains(JudgeError)
    assert not layout.judged(0).exists()
    assert not (layout.samples / f"{judge.SOURCE}-0000000.parquet").exists()
    logged = [json.loads(line) for line in layout.verdicts(0).read_text("utf-8").splitlines()]
    judged_by_jev = {line["id"] for line in logged if line["stage"] == "jev"}
    # A crash in the middle of a write leaves half a line; the resumed run drops it.
    with layout.verdicts(0).open("a", encoding="utf-8") as handle:
        handle.write('{"id": "00')

    jev, review = FakeJev(), FakeReview()
    census = run(layout, jev, review)
    assert census.kept_records == 2
    assert len(jev.asked) == 5 - len(judged_by_jev)
    assert len(review.asked) == 3
    stages = [
        json.loads(line)["stage"] for line in layout.verdicts(0).read_text("utf-8").splitlines()
    ]
    assert stages.count("boundary") == 6
    assert stages.count("jev") == 5
    assert stages.count("review") == 4

    # A judged file is not read again.
    again_jev, again_review = FakeJev(), FakeReview()
    assert run(layout, again_jev, again_review) == census
    assert again_jev.asked == [] and again_review.asked == []


def test_a_file_whose_shards_moved_away_is_refused(tmp_path: Path) -> None:
    layout = mined_root(tmp_path, [(RECORDS[0][0], RECORDS[0][1])])
    (layout.unjudged / f"{judge.SOURCE}-0000000.parquet").unlink()
    with pytest.raises(JudgeError, match="wrote 1 records"):
        run(layout, FakeJev(), FakeReview())


def test_a_record_with_two_entries_keeps_the_one_that_passes(tmp_path: Path) -> None:
    text = "他小时候一丁不识字，画蛇添足的事却做得不少"
    layout = mined_root(tmp_path, [(text, "一丁不识")])
    with (layout.entry_rows / "0000.tsv").open("a", encoding="utf-8") as handle:
        handle.write("000000000000\t画蛇添足\n")

    @dataclass
    class Natural:
        async def ask(self, unit: Unit) -> JevAnswer:
            return JevAnswer(p=0.9, model="jev-fake")

    @dataclass
    class Fine:
        async def ask(self, unit: Unit) -> ReviewAnswer:
            return ReviewAnswer(context_spun=False, sentence_natural=True)

    census = asyncio.run(judge.judge_source(layout, LEXICON, Natural(), Fine(), 1, 1))
    assert (census.units, census.boundary, census.kept_units, census.kept_records) == (2, 1, 1, 1)
    assert layout.kept_rows(0).read_text(encoding="utf-8") == "000000000000\t画蛇添足\n"
