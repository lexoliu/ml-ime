"""The corpus stage decides what the model ever sees, so its filters are the contract.

Each test names the thing a downstream stage would break on if the filter let it
through: a target longer than the encoder's positions, a target holding a
character `ime-pinyin` has no reading for, or the same sentence twice with two
different contexts inflating one phrasing's weight.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from mlime.data.corpus import (
    DIALOGUE,
    DOCUMENT_SCHEMA,
    NEWS,
    SOURCES,
    WIKIPEDIA,
    Document,
    RawDocument,
    Sample,
    SampleFilter,
    build_samples,
    content_id,
    load_reference_characters,
    prepare,
)
from mlime.data.shards import ShardWriter
from mlime.data.text import Normalizer

KNOWN = frozenset("数学是研究量结构以及空间的一门科你好世界火锅重庆成都吃了七八顿今天气很晴朗")


@pytest.fixture(scope="module")
def normalize() -> Normalizer:
    return Normalizer()


@pytest.fixture
def sample_filter() -> SampleFilter:
    return SampleFilter(KNOWN)


def test_short_targets_are_rejected(sample_filter: SampleFilter) -> None:
    assert not sample_filter.accepts("你好")
    assert sample_filter.counts.too_short == 1


def test_long_targets_are_rejected(sample_filter: SampleFilter) -> None:
    """A target longer than the bound has more positions than the decoder fills."""
    assert not sample_filter.accepts("你好世界" * 20)
    assert sample_filter.counts.too_long == 1


def test_targets_that_are_mostly_not_chinese_are_rejected(sample_filter: SampleFilter) -> None:
    assert not sample_filter.accepts("你好 ABCDEFGHIJ")
    assert sample_filter.counts.not_chinese_enough == 1


def test_targets_using_characters_with_no_reading_are_rejected(
    sample_filter: SampleFilter,
) -> None:
    """`ime-pinyin` cannot mask a character it has no reading for, so it cannot be a target."""
    assert not sample_filter.accepts("你好世界𠀀")
    assert sample_filter.counts.unknown_character == 1


def test_repeated_targets_are_dropped(sample_filter: SampleFilter) -> None:
    assert sample_filter.accepts("你好世界")
    assert not sample_filter.accepts("你好世界")
    assert sample_filter.counts == type(sample_filter.counts)(kept=1, duplicate=1)


def test_counts_add_up(sample_filter: SampleFilter) -> None:
    for text in ("你好", "你好世界", "你好世界", "你好 ABCDEFGHIJ"):
        sample_filter.accepts(text)
    assert sample_filter.counts.considered == 4


def test_context_is_the_preceding_sentences(sample_filter: SampleFilter) -> None:
    document = Document("d", ("今天天气很晴朗。", "你好世界。", "火锅重庆成都。"))
    samples = list(build_samples(document, "wiki", 2, 256, sample_filter))
    assert [s.text for s in samples] == ["今天天气很晴朗", "你好世界", "火锅重庆成都"]
    assert [s.context for s in samples] == [
        None,
        "今天天气很晴朗。",
        "今天天气很晴朗。你好世界。",
    ]


def test_dialogue_context_keeps_turns_apart(sample_filter: SampleFilter) -> None:
    """Turns carry no terminal punctuation, so running them together would fuse them."""
    document = Document("d", ("你好世界", "今天天气很晴朗"), joiner="\n")
    samples = list(build_samples(document, "dialogue", 2, 256, sample_filter))
    assert samples[1].context == "你好世界"


def test_context_is_trimmed_to_its_budget(sample_filter: SampleFilter) -> None:
    """The context encoder is cached per keystroke; an unbounded context is not free."""
    document = Document("d", ("今天天气很晴朗。", "你好世界。"))
    samples = list(build_samples(document, "wiki", 5, 4, sample_filter))
    assert samples[1].context == "很晴朗。"


def test_identifiers_follow_content_not_position() -> None:
    """Shards are rebuilt piecemeal, so an identifier has to be reproducible."""
    assert content_id("wiki", "你好", None) == content_id("wiki", "你好", None)
    assert content_id("wiki", "你好", None) != content_id("news", "你好", None)
    assert content_id("wiki", "你好", None) != content_id("wiki", "你好", "世界")


def test_fetching_keeps_the_upstream_text_untouched() -> None:
    """Normalising here would mean re-downloading every time a rule changes."""
    assert WIKIPEDIA.parts({"id": "13", "text": "數學。是研究數量"}) == ("數學。是研究數量",)


def test_wikipedia_records_become_sentences(normalize: Normalizer) -> None:
    raw = RawDocument("13", WIKIPEDIA.parts({"id": "13", "text": "數學。是研究數量"}))
    document = WIKIPEDIA.document(raw, normalize)
    assert document.document_id == "13"
    assert document.segments == ("数学。", "是研究数量")


def test_news_records_lead_with_their_headline(normalize: Normalizer) -> None:
    parts = NEWS.parts({"title": "国奥队训练", "text": "天就是阴沉沉的。"})
    assert NEWS.document(RawDocument("d", parts), normalize).segments == (
        "国奥队训练",
        "天就是阴沉沉的。",
    )


def test_dialogue_records_become_one_segment_per_turn(normalize: Normalizer) -> None:
    parts = DIALOGUE.parts({"dialog": ["自贡 哪里 有 好吃 的 鱼", "服务 相当 好"]})
    document = DIALOGUE.document(RawDocument("d", parts), normalize)
    assert document.segments == ("自贡哪里有好吃的鱼", "服务相当好")
    assert document.joiner == "\n"


def test_a_record_missing_the_field_the_adapter_reads_raises() -> None:
    """Upstream schemas drift; the pipeline must say so instead of writing empty rows."""
    with pytest.raises(KeyError):
        DIALOGUE.parts({"conversation": ["你好"]})


def test_every_source_is_registered_under_its_own_name() -> None:
    assert set(SOURCES) == {"wiki", "dialogue", "news"}
    assert all(name == source.name for name, source in SOURCES.items())


def test_reference_characters_are_read_from_the_rust_table(tmp_path: Path) -> None:
    table = tmp_path / "char_pinyin.tsv"
    table.write_text("重\tzhong,chong\n行\txing,hang\n", encoding="utf-8")
    assert load_reference_characters(table) == frozenset("重行")


@pytest.mark.parametrize("content", ["", "重\n", "重\t\n", "重要\tzhong\n"])
def test_a_malformed_character_table_raises(tmp_path: Path, content: str) -> None:
    table = tmp_path / "char_pinyin.tsv"
    table.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=str(table.name)):
        load_reference_characters(table)


def test_a_missing_character_table_names_the_option(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="--char-table"):
        load_reference_characters(tmp_path / "absent.tsv")


def test_prepare_writes_samples_that_round_trip(tmp_path: Path) -> None:
    """The end-to-end local half: documents in, filtered samples with context out."""
    raw, out = tmp_path / "documents", tmp_path / "samples"
    with ShardWriter(raw, "wiki", DOCUMENT_SCHEMA, 10) as writer:
        writer.write(
            {"document_id": "d", "source": "wiki", "parts": ["今天天气很晴朗。你好\n你好世界。"]}
        )
    counts = prepare(raw, out, KNOWN, ("wiki",))
    assert counts["wiki"].kept == 2
    assert counts["wiki"].too_short == 1

    samples = list(Sample.read(out))
    assert [s.text for s in samples] == ["今天天气很晴朗", "你好世界"]
    assert samples[1].context == "今天天气很晴朗。你好"
    frame = pl.read_parquet(out / "wiki-00000.parquet")
    assert frame.columns == ["id", "source", "text", "context"]
