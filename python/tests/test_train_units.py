"""Unit initials: the idioms and internet phrases a typist presses as one word.

The typing augmentation gains one switch, and the invariant that matters most is
that off it changes nothing: the same generator must draw the same keystrokes,
and leave the same state behind, as before the switch existed. The rest is the
rule itself -- every unit of a sentence is typed as initials whole, the rest of
the sentence keeps its drawn style, and a mixed cut inside a unit moves to the
unit's start.
"""

from __future__ import annotations

import pickle
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mlime.__main__ import app
from mlime.data.corpus import Sample
from mlime.legality import Legality
from mlime.train.arbitration import ReadingArbitration
from mlime.train.keys import KeyFrontConfig, KeySampleBuilder
from mlime.train.lexicon import Lexicon
from mlime.train.run import Vocabularies
from mlime.train.samples import Augmentation, SampleBuilder, example_rng, type_syllables
from mlime.train.spans import SpanVocab, initial
from mlime.train.units import UnitMatcher, read_unit_entries
from mlime.typo import load_syllables

DATA = Path(__file__).parent.parent.parent / "crates/ime-pinyin/data"

ALWAYS_FULL = Augmentation(full=1.0, abbreviated=0.0, mixed=0.0, unit_initials=1.0)
ALWAYS_ABBREVIATED = Augmentation(full=0.0, abbreviated=1.0, mixed=0.0, unit_initials=1.0)
ALWAYS_MIXED = Augmentation(full=0.0, abbreviated=0.0, mixed=1.0, unit_initials=1.0)

SYLLABLES = ("bei", "jing", "da", "xue", "zhong", "wo")


def _sample(text: str) -> Sample:
    return Sample(id=f"id-{text}", source="test", text=text, context=None)


def test_a_unit_is_found_wherever_it_stands() -> None:
    matcher = UnitMatcher(["一帆风顺", "龙马精神"])
    assert matcher.spans("祝你一帆风顺龙马精神") == [(2, 6), (6, 10)]
    assert matcher.spans("没有成语") == []
    assert matcher.spans("一") == []


def test_the_leftmost_longest_unit_wins_and_units_do_not_overlap() -> None:
    matcher = UnitMatcher(["一帆", "一帆风顺", "风顺", "顺利"])
    # 一帆风顺 beats its prefix 一帆, and the 风顺 and 顺利 inside or against it are not taken.
    assert matcher.spans("一帆风顺利") == [(0, 4)]
    # Leftmost beats longest: 一帆风 starts first, so 风顺 after it is the next unit.
    assert UnitMatcher(["一帆风", "帆风顺利"]).spans("一帆风顺利") == [(0, 3)]
    assert matcher.spans("顺利一帆") == [(0, 2), (2, 4)]


def test_every_occurrence_counts_nested_and_overlapping_ones_too() -> None:
    matcher = UnitMatcher(["一帆", "一帆风顺", "风顺"])
    assert matcher.occurrences("一帆风顺") == [(0, 4), (0, 2), (2, 4)]


def test_an_entry_of_one_character_is_no_unit() -> None:
    with pytest.raises(ValueError, match="at least two characters"):
        UnitMatcher(["一", "一帆风顺"])
    with pytest.raises(ValueError, match="at least one entry"):
        UnitMatcher([])


def test_the_lexicon_file_yields_its_idioms_and_internet_phrases(tmp_path: Path) -> None:
    path = tmp_path / "knowledge.tsv"
    path.write_text(
        "一帆风顺\tidiom\n绝绝子\tinternet\n张三\tperson\n北京\tplace\n好\tidiom\n",
        encoding="utf-8",
    )
    assert read_unit_entries(path) == ["一帆风顺", "绝绝子"]
    matcher = UnitMatcher.load(path)
    assert len(matcher) == 2
    assert matcher.spans("这也太绝绝子了") == [(3, 6)]


def test_a_malformed_or_absent_lexicon_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no unit lexicon"):
        read_unit_entries(tmp_path / "missing.tsv")
    bad = tmp_path / "bad.tsv"
    bad.write_text("一帆风顺\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a `<word>"):
        read_unit_entries(bad)
    people = tmp_path / "people.tsv"
    people.write_text("张三\tperson\n", encoding="utf-8")
    with pytest.raises(ValueError, match="holds no entry"):
        read_unit_entries(people)


def test_the_digest_names_the_entry_set_and_not_its_order() -> None:
    assert (
        UnitMatcher(["一帆风顺", "龙马精神"]).digest == UnitMatcher(["龙马精神", "一帆风顺"]).digest
    )
    assert UnitMatcher(["一帆风顺"]).digest != UnitMatcher(["龙马精神"]).digest


def test_a_matcher_survives_the_pickle_the_count_workers_receive() -> None:
    matcher = pickle.loads(pickle.dumps(UnitMatcher(["一帆风顺"])))
    assert matcher.spans("祝一帆风顺") == [(1, 5)]


@pytest.mark.parametrize("augmentation", [ALWAYS_FULL, ALWAYS_ABBREVIATED, ALWAYS_MIXED])
def test_a_unit_is_typed_as_initials_whole_in_every_style(augmentation: Augmentation) -> None:
    rng = random.Random(4)
    for _ in range(200):
        typed = type_syllables(SYLLABLES, rng, augmentation, [(1, 4)])
        assert typed[1:4] == tuple(initial(syllable) for syllable in SYLLABLES[1:4])


def test_the_rest_of_the_sentence_keeps_its_drawn_style() -> None:
    full = type_syllables(SYLLABLES, random.Random(0), ALWAYS_FULL, [(1, 3)])
    assert full == ("bei", "j", "d", "xue", "zhong", "wo")
    rng = random.Random(9)
    outside = Counter()
    for _ in range(20_000):
        typed = type_syllables(SYLLABLES, rng, ALWAYS_ABBREVIATED, [(1, 3)])
        outside.update(
            span == initial(syllable)
            for span, syllable in list(zip(typed, SYLLABLES, strict=True))[3:]
        )
    assert outside[True] / sum(outside.values()) == pytest.approx(0.7, abs=0.01)


def test_a_mixed_cut_inside_a_unit_moves_to_its_start() -> None:
    rng = random.Random(11)
    typings = {type_syllables(SYLLABLES, rng, ALWAYS_MIXED, [(2, 5)]) for _ in range(2_000)}
    # A cut at 1 abbreviates from the second syllable on. The cuts 2, 3, 4 and 5
    # all leave "bei", "jing" in full -- 3 and 4 fall inside the unit (2, 5) and
    # land on its start -- and abbreviate everything after.
    assert typings == {
        ("bei", "j", "d", "x", "zh", "w"),
        ("bei", "jing", "d", "x", "zh", "w"),
    }


def test_a_unit_at_the_start_pulls_a_mixed_cut_to_the_whole_sentence() -> None:
    rng = random.Random(1)
    for _ in range(100):
        typed = type_syllables(SYLLABLES, rng, ALWAYS_MIXED, [(0, 3)])
        assert typed[:3] == tuple(initial(syllable) for syllable in SYLLABLES[:3])


def test_the_coin_decides_whether_a_unit_is_typed_whole() -> None:
    augmentation = replace(ALWAYS_FULL, unit_initials=0.5)
    rng = random.Random(2)
    whole = sum(
        type_syllables(SYLLABLES, rng, augmentation, [(1, 3)])[1] == "j" for _ in range(20_000)
    )
    assert whole / 20_000 == pytest.approx(0.5, abs=0.01)


@pytest.mark.parametrize("seed", range(20))
def test_without_unit_initials_the_typing_and_the_generator_are_unchanged(seed: int) -> None:
    """The draw order of a run that never asked for units is exactly what it was."""
    plain = Augmentation()
    assert plain.unit_initials == 0.0
    expected, found = random.Random(seed), random.Random(seed)
    for _ in range(50):
        before = type_syllables(SYLLABLES, expected, plain)
        # Units present but the coin never thrown: the same keystrokes, the same state.
        after = type_syllables(SYLLABLES, found, plain, [(1, 3)])
        assert after == before
        assert found.getstate() == expected.getstate()


def test_a_sentence_without_a_unit_draws_what_it_drew_before_the_coin_existed() -> None:
    expected, found = random.Random(3), random.Random(3)
    for _ in range(50):
        before = type_syllables(SYLLABLES, expected, Augmentation())
        after = type_syllables(SYLLABLES, found, replace(Augmentation(), unit_initials=0.5), [])
        assert after == before
        assert found.getstate() == expected.getstate()


def test_the_default_style_mix_is_untouched() -> None:
    augmentation = Augmentation(unit_initials=0.5)
    assert (augmentation.full, augmentation.abbreviated, augmentation.mixed) == (0.55, 0.25, 0.20)


def test_unit_initials_is_a_probability() -> None:
    with pytest.raises(ValueError, match="unit_initials must be a probability"):
        Augmentation(unit_initials=1.5)


def test_the_builder_types_the_unit_whole_and_the_rest_in_full(
    lexicon: Lexicon, spans: SpanVocab, arbitration: ReadingArbitration
) -> None:
    builder = SampleBuilder(
        lexicon, spans, arbitration, ALWAYS_FULL, seed=5, units=UnitMatcher(["北京"])
    )
    example = builder.build(_sample("我爱北京"), ["wo3", "ai4", "bei3", "jing1"], epoch=0)
    assert example is not None
    assert example.spans == ("wo", "ai", "b", "j")
    assert example.targets == tuple(lexicon.index(character) for character in "我爱北京")


def test_the_builder_with_the_switch_off_builds_what_it_always_built(
    lexicon: Lexicon, spans: SpanVocab, arbitration: ReadingArbitration
) -> None:
    plain = SampleBuilder(lexicon, spans, arbitration, seed=5)
    again = SampleBuilder(lexicon, spans, arbitration, Augmentation(unit_initials=0.0), seed=5)
    syllables = ["wo3", "ai4", "bei3", "jing1"]
    for epoch in range(30):
        assert plain.build(_sample("我爱北京"), syllables, epoch) == again.build(
            _sample("我爱北京"), syllables, epoch
        )


def test_each_repeat_of_a_sentence_draws_its_own_typing(
    lexicon: Lexicon, spans: SpanVocab, arbitration: ReadingArbitration
) -> None:
    builder = SampleBuilder(
        lexicon,
        spans,
        arbitration,
        Augmentation(unit_initials=0.5),
        seed=5,
        units=UnitMatcher(["北京"]),
    )
    syllables = ["wo3", "ai4", "bei3", "jing1"]
    typed = {
        example.spans
        for repeat in range(8)
        if (
            example := builder.build(
                Sample(id=f"abc+{repeat}", source="test", text="我爱北京", context=None),
                syllables,
                epoch=0,
            )
        )
        is not None
    }
    assert len(typed) > 1
    assert example_rng(5, 0, "abc+1").random() != example_rng(5, 0, "abc+2").random()


def test_the_matcher_and_the_switch_come_together(
    lexicon: Lexicon, spans: SpanVocab, arbitration: ReadingArbitration
) -> None:
    with pytest.raises(ValueError, match="must come together"):
        SampleBuilder(lexicon, spans, arbitration, Augmentation(unit_initials=0.5))
    with pytest.raises(ValueError, match="must come together"):
        SampleBuilder(lexicon, spans, arbitration, units=UnitMatcher(["北京"]))


@pytest.fixture(name="legality")
def legality_fixture(
    lexicon: Lexicon, spans: SpanVocab, arbitration: ReadingArbitration, tokenizer: object
) -> Legality:
    vocabularies = Vocabularies(
        spans=spans,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        context_tokenizer=tokenizer,  # type: ignore[arg-type]
        lexicon=lexicon,
        arbitration=arbitration,
        syllables=frozenset(load_syllables(DATA / "syllables.txt")),
    )
    return vocabularies.legality()


def test_the_key_builder_presses_a_unit_as_its_initials(
    lexicon: Lexicon, spans: SpanVocab, arbitration: ReadingArbitration, legality: Legality
) -> None:
    builder = KeySampleBuilder(
        lexicon,
        spans,
        arbitration,
        legality,
        KeyFrontConfig(layers=1, max_keys=62),
        ALWAYS_FULL,
        seed=5,
        units=UnitMatcher(["北京"]),
    )
    example = builder.build(_sample("我爱北京"), ["wo3", "ai4", "bei3", "jing1"], epoch=0)
    assert example is not None
    assert example.spans == ("wo", "ai", "b", "j")
    assert example.keys == "woaibj"
    with pytest.raises(ValueError, match="must come together"):
        KeySampleBuilder(
            lexicon,
            spans,
            arbitration,
            legality,
            KeyFrontConfig(layers=1, max_keys=62),
            ALWAYS_FULL,
        )


def test_the_stamp_refuses_unit_initials_without_their_lexicon(tmp_path: Path) -> None:
    runner = CliRunner()
    out = tmp_path / "kernel"
    stage = ["e2e", "stamp", "mlime-e2e-v2-s3", "--out", str(out), "--init-from", "mlime-e2e-v2-g4"]
    stamped = runner.invoke(app, [*stage, "--unit-initials", "0.5"])
    assert stamped.exit_code == 2
    stamped = runner.invoke(app, [*stage, "--unit-lexicon", "knowledge.tsv"])
    assert stamped.exit_code == 2
    assert not out.exists()
