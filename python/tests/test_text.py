"""The normaliser decides what "the same sentence" means for every later stage.

Deduplication, the character-coverage filter and the per-character alignment
between the two g2p annotators all assume the text has already been reduced to
one canonical form. These tests pin that form against the three shapes the real
corpora actually arrive in: mixed-script Wikipedia, whitespace-tokenised LCCC
turns, and news prose padded with ideographic spaces.
"""

from __future__ import annotations

import pytest

from mlime.data.text import (
    Normalizer,
    han_characters,
    han_ratio,
    split_sentences,
    strip_terminal_delimiter,
    toneless,
)


@pytest.fixture(scope="module")
def normalize() -> Normalizer:
    return Normalizer()


def test_traditional_characters_become_simplified(normalize: Normalizer) -> None:
    """Chinese Wikipedia mixes both scripts inside one article; the IME types one."""
    assert (
        normalize("數學，是研究數量、结构以及空间的一門学科")
        == "数学，是研究数量、结构以及空间的一门学科"
    )


def test_tokenised_turns_are_rejoined(normalize: Normalizer) -> None:
    """LCCC ships its turns pre-segmented; nobody types those spaces."""
    assert normalize("自贡 哪里 有 好吃 的 鱼") == "自贡哪里有好吃的鱼"


def test_spaces_before_punctuation_go_but_spaces_beside_latin_stay(normalize: Normalizer) -> None:
    """The rule is full-width on both sides; `使用 Python` is spaced on purpose."""
    assert normalize("哈哈哈哈 ！ 那 我 的 嘴巴 要 烂掉") == "哈哈哈哈！那我的嘴巴要烂掉"
    assert normalize("我们 使用 Python 编程") == "我们使用 Python 编程"


def test_ideographic_space_and_fullwidth_latin_fold(normalize: Normalizer) -> None:
    """THUCNews pads with U+3000, and full-width Latin is not what a keyboard emits."""
    assert normalize("新浪体育讯　12月27日晚，ＡＢＣ１２３") == "新浪体育讯 12月27日晚，ABC123"


def test_chinese_punctuation_survives_the_fold(normalize: Normalizer) -> None:
    """The sentence splitter runs on these, so folding them to ASCII would break it."""
    assert normalize("你好，世界！真的吗？是的；好") == "你好，世界！真的吗？是的；好"


def test_brackets_emptied_by_template_stripping_are_removed(normalize: Normalizer) -> None:
    """`wikimedia/wikipedia` leaves `（）` behind where a language template was."""
    assert normalize("文學（），在狭义上") == "文学，在狭义上"


def test_sentences_split_on_delimiters_and_newlines() -> None:
    text = "第一句。第二句！第三句？\n第四句"
    assert list(split_sentences(text)) == ["第一句。", "第二句！", "第三句？", "第四句"]


def test_sentences_keep_their_delimiter_for_context_reassembly() -> None:
    """Context is preceding sentences run together, so they must carry their punctuation."""
    assert "".join(split_sentences("甲。乙。丙。")) == "甲。乙。丙。"


def test_split_ignores_blank_runs() -> None:
    assert list(split_sentences("甲。\n\n  \n乙。")) == ["甲。", "乙。"]


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [("你好。", "你好"), ("你好！", "你好"), ("你好", "你好"), ("", "")],
)
def test_terminal_delimiter_is_stripped(sentence: str, expected: str) -> None:
    """A target's trailing punctuation is typed as a key, not decoded from pinyin."""
    assert strip_terminal_delimiter(sentence) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [("你好世界", 1.0), ("你好，世界", 0.8), ("abcd", 0.0), ("", 0.0), ("你 好", 1.0)],
)
def test_han_ratio_counts_only_visible_characters(text: str, expected: float) -> None:
    assert han_ratio(text) == pytest.approx(expected)


def test_han_characters_drops_everything_else() -> None:
    """The g2p annotators are aligned against exactly this list."""
    assert han_characters("他在2026年买了iPhone。") == list("他在年买了")


@pytest.mark.parametrize(
    ("syllable", "expected"),
    [("zhong1", "zhong"), ("de5", "de"), ("lv4", "lv"), ("lü4", "lv"), ("LÜ3", "lv"), ("yi", "yi")],
)
def test_toneless_folds_tone_and_umlaut(syllable: str, expected: str) -> None:
    """Comparison is on what a keyboard can produce: no tone key, no `ü` key."""
    assert toneless(syllable) == expected
