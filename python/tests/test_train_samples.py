"""The sample builder decides what the model ever sees, so its statistics are tested.

An augmentation bug does not crash: it quietly trains the model on the wrong
distribution of keystrokes and shows up months later as "abbreviations are weak".
So the probabilities are measured on a fixed seed rather than assumed, and the
invariants the model relies on -- every span is in the inventory, every target is
admitted by its span, the two sequences are the same length -- are asserted on
built examples.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import polars as pl
import pytest
import torch

from mlime.data.corpus import SAMPLE_SCHEMA, Sample
from mlime.train.labels import LABEL_SCHEMA, keyboard_form
from mlime.train.lexicon import Lexicon
from mlime.train.samples import (
    IGNORE_INDEX,
    Augmentation,
    BaseTokenizer,
    Batch,
    Collator,
    CorpusStream,
    SampleBuilder,
    TrainingExample,
    context_tail,
    example_rng,
    token_budget_batches,
    type_syllables,
)
from mlime.train.spans import SpanVocab, initial

ALWAYS_FULL = Augmentation(full=1.0, abbreviated=0.0, mixed=0.0)
ALWAYS_ABBREVIATED = Augmentation(full=0.0, abbreviated=1.0, mixed=0.0)
ALWAYS_MIXED = Augmentation(full=0.0, abbreviated=0.0, mixed=1.0)


def test_style_probabilities_hold_on_a_fixed_seed() -> None:
    """0.55 / 0.25 / 0.2, measured rather than asserted by reading the constants."""
    augmentation = Augmentation()
    rng = random.Random(20260826)
    draws = Counter(augmentation.style(rng) for _ in range(200_000))
    total = sum(draws.values())
    assert draws["full"] / total == pytest.approx(0.55, abs=0.005)
    assert draws["abbreviated"] / total == pytest.approx(0.25, abs=0.005)
    assert draws["mixed"] / total == pytest.approx(0.20, abs=0.005)


def test_abbreviation_drops_seven_syllables_in_ten() -> None:
    syllables = ("bei", "jing", "da", "xue", "de", "jue", "ding", "wo")
    rng = random.Random(7)
    dropped = full = 0
    for _ in range(20_000):
        typed = type_syllables(syllables, rng, ALWAYS_ABBREVIATED)
        for span, syllable in zip(typed, syllables, strict=True):
            if span == syllable:
                full += 1
            else:
                assert span == initial(syllable)
                dropped += 1
    assert dropped / (dropped + full) == pytest.approx(0.7, abs=0.01)


def test_zh_initial_syllables_abbreviate_whole() -> None:
    """`zhong` abbreviates to `zh`, never to `z`."""
    rng = random.Random(3)
    seen = set()
    for _ in range(500):
        seen.update(type_syllables(("zhong", "chi", "shi"), rng, ALWAYS_ABBREVIATED))
    assert seen <= {"zhong", "chi", "shi", "zh", "ch", "sh"}
    assert {"zh", "ch", "sh"} <= seen


def test_a_mixed_example_is_full_then_abbreviated() -> None:
    syllables = ("bei", "jing", "da", "xue", "zhong")
    rng = random.Random(11)
    cuts = set()
    for _ in range(2_000):
        typed = type_syllables(syllables, rng, ALWAYS_MIXED)
        cut = next(index for index, span in enumerate(typed) if span != syllables[index])
        cuts.add(cut)
        assert typed[:cut] == syllables[:cut]
        assert typed[cut:] == tuple(initial(s) for s in syllables[cut:])
    assert cuts == {1, 2, 3, 4}


def test_a_single_syllable_sentence_is_abbreviated_whole() -> None:
    assert type_syllables(("zhong",), random.Random(0), ALWAYS_MIXED) == ("zh",)


def test_full_typing_is_the_readings_unchanged() -> None:
    syllables = ("wo", "ai", "lv")
    assert type_syllables(syllables, random.Random(0), ALWAYS_FULL) == syllables


def test_the_probabilities_must_be_a_distribution() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        Augmentation(full=0.5, abbreviated=0.5, mixed=0.5)


def test_the_same_example_redraws_only_when_the_epoch_moves() -> None:
    first = type_syllables(("bei", "jing"), example_rng(7, 0, "abc"), Augmentation())
    again = type_syllables(("bei", "jing"), example_rng(7, 0, "abc"), Augmentation())
    assert first == again
    styles = {
        type_syllables(("bei", "jing", "da", "xue"), example_rng(7, epoch, "abc"), Augmentation())
        for epoch in range(50)
    }
    assert len(styles) > 1


def _sample(text: str, context: str | None = None) -> Sample:
    return Sample(id=f"id-{text}", source="test", text=text, context=context)


def test_a_built_example_is_aligned_and_resolvable(lexicon: Lexicon, spans: SpanVocab) -> None:
    builder = SampleBuilder(lexicon, spans, seed=5)
    example = builder.build(_sample("我爱北京"), ["wo3", "ai4", "bei3", "jing1"], epoch=0)
    assert example is not None
    assert len(example.spans) == len(example.targets) == len(example.span_ids) == 4
    for span, span_id in zip(example.spans, example.span_ids, strict=True):
        assert span in spans
        assert spans.id(span) == span_id
    assert [lexicon.characters[target] for target in example.targets] == list("我爱北京")


def test_every_target_is_admitted_by_the_span_that_was_typed(
    lexicon: Lexicon, spans: SpanVocab
) -> None:
    builder = SampleBuilder(lexicon, spans, seed=1)
    for epoch in range(20):
        example = builder.build(_sample("中重我绿"), ["zhong1", "chong2", "wo3", "lv4"], epoch)
        assert example is not None
        for span_id, target in zip(example.span_ids, example.targets, strict=True):
            assert bool(lexicon.candidate_mask[span_id, target])


def test_unusable_samples_are_counted_not_patched(lexicon: Lexicon, spans: SpanVocab) -> None:
    builder = SampleBuilder(lexicon, spans)
    assert builder.build(_sample("我爱Python"), ["wo3", "ai4"], 0) is None
    assert builder.counts.not_all_han == 1
    assert builder.build(_sample("我爱"), ["wo3"], 0) is None
    assert builder.counts.length_mismatch == 1
    assert builder.build(_sample("我爱"), None, 0) is None
    assert builder.counts.unlabelled == 1
    assert builder.build(_sample("我兲"), ["wo3", "tian1"], 0) is None
    assert builder.counts.unemittable_character == 1
    assert builder.counts.kept == 0


def test_context_is_dropped_at_the_stated_rate(
    lexicon: Lexicon, spans: SpanVocab, tokenizer: BaseTokenizer
) -> None:
    builder = SampleBuilder(lexicon, spans)
    example = builder.build(_sample("我爱北京", context="北京"), ["wo3", "ai4", "bei3", "jing1"], 0)
    assert example is not None
    collator = Collator(tokenizer, context_dropout=0.3, seed=42)
    kept = sum(float(collator([example]).has_context[0]) for _ in range(4_000))
    assert kept / 4_000 == pytest.approx(0.7, abs=0.02)


def test_a_sample_with_no_context_never_claims_one(
    lexicon: Lexicon, spans: SpanVocab, tokenizer: BaseTokenizer
) -> None:
    builder = SampleBuilder(lexicon, spans)
    example = builder.build(_sample("我爱北京"), ["wo3", "ai4", "bei3", "jing1"], 0)
    assert example is not None
    batch = Collator(tokenizer, context_dropout=0.0)([example])
    assert float(batch.has_context[0]) == 0.0


def test_the_collated_batch_masks_and_targets_line_up(
    lexicon: Lexicon, spans: SpanVocab, tokenizer: BaseTokenizer
) -> None:
    builder = SampleBuilder(lexicon, spans)
    examples = [
        builder.build(_sample("我爱北京"), ["wo3", "ai4", "bei3", "jing1"], 0),
        builder.build(_sample("中重"), ["zhong1", "chong2"], 0),
    ]
    assert all(example is not None for example in examples)
    batch = Collator(tokenizer)([example for example in examples if example is not None])
    assert batch.input_ids.shape == (2, 6)
    assert int(batch.span_positions.sum()) == 6
    assert int((batch.targets != IGNORE_INDEX).sum()) == 6
    assert torch.equal(batch.span_positions, batch.targets != IGNORE_INDEX)
    assert int(batch.input_ids[0, 0]) == tokenizer.cls_token_id
    assert int(batch.input_ids[0, 5]) == tokenizer.sep_token_id
    assert int(batch.input_ids[1, 3]) == tokenizer.sep_token_id
    assert int(batch.input_ids[1, 5]) == tokenizer.pad_token_id
    assert torch.equal(
        batch.input_ids[0, 1:5], torch.full((4,), tokenizer.mask_token_id, dtype=torch.long)
    )


def test_batches_respect_the_token_budget(lexicon: Lexicon, spans: SpanVocab) -> None:
    builder = SampleBuilder(lexicon, spans)
    examples = []
    for index in range(20):
        text = f"我爱北京{'中' * (index % 3)}"[:4]
        example = builder.build(_sample(text), ["wo3", "ai4", "bei3", "jing1"], 0)
        if example is not None:
            examples.append(example)
    groups = list(token_budget_batches(iter(examples), budget=24))
    assert sum(len(group) for group in groups) == len(examples)
    for group in groups:
        width = max(len(example) for example in group) + 2
        assert width * len(group) <= 24


def test_a_batch_pays_for_its_context_rectangle_too(lexicon: Lexicon, spans: SpanVocab) -> None:
    """A batch of short sentences with long contexts is a *small* batch.

    Bounding the fill tower alone, the sentences here are four characters and the
    budget would admit ten of them; their contexts are the expensive half, and it
    is that half that took a two-rank run out of memory.
    """
    builder = SampleBuilder(lexicon, spans, ALWAYS_FULL)
    examples = [
        builder.build(_sample("我爱北京", context="北" * 40), ["wo", "ai", "bei", "jing"], 0)
        for _ in range(20)
    ]
    kept = [example for example in examples if example is not None]
    assert kept, "the fixture sentence must build"
    groups = list(token_budget_batches(iter(kept), budget=64, max_context_tokens=16))
    assert sum(len(group) for group in groups) == len(kept)
    for group in groups:
        fill = max(len(example) for example in group) + 2
        context = max(min(len(example.context or ""), 14) + 2 for example in group)
        assert (fill + context) * len(group) <= 64
    # Six positions of fill plus sixteen of context is twenty-two per example.
    assert max(len(group) for group in groups) == 2


def test_a_batch_with_no_context_tower_pays_for_the_fill_alone(
    lexicon: Lexicon, spans: SpanVocab
) -> None:
    builder = SampleBuilder(lexicon, spans, ALWAYS_FULL)
    kept = [
        example
        for example in (
            builder.build(_sample("我爱北京", context="北" * 40), ["wo", "ai", "bei", "jing"], 0)
            for _ in range(20)
        )
        if example is not None
    ]
    groups = list(token_budget_batches(iter(kept), budget=64, max_context_tokens=0))
    assert max(len(group) for group in groups) == 10


def test_the_context_kept_is_the_end_of_it() -> None:
    """The cursor sits after the context, so its last characters are the useful ones."""
    assert context_tail("一二三四五六", 5) == "四五六"
    assert context_tail("一二", 64) == "一二"
    with pytest.raises(ValueError, match="two sentinels"):
        context_tail("一二", 2)


def test_a_stream_reads_its_shards_and_reaugments(
    tmp_path: Path, lexicon: Lexicon, spans: SpanVocab
) -> None:
    samples_dir, labels_dir = tmp_path / "samples", tmp_path / "labels"
    samples_dir.mkdir()
    labels_dir.mkdir()
    rows = [_sample("我爱北京", context="北京"), _sample("中重我绿")]
    readings = {
        "id-我爱北京": ["wo3", "ai4", "bei3", "jing1"],
        "id-中重我绿": ["zhong1", "chong2", "wo3", "lv4"],
    }
    pl.DataFrame([row.row() for row in rows], schema=SAMPLE_SCHEMA).write_parquet(
        samples_dir / "test-00000.parquet"
    )
    pl.DataFrame(
        [{"id": row.id, "syllables": readings[row.id], "refusal": None} for row in rows],
        schema=LABEL_SCHEMA,
    ).write_parquet(labels_dir / "test-00000.parquet")

    stream = CorpusStream(samples_dir, labels_dir, SampleBuilder(lexicon, spans, seed=3))
    first = [example.spans for example in stream]
    assert len(first) == 2
    stream.set_epoch(1)
    assert len(list(stream)) == 2
    assert stream.builder.counts.kept == 4


def test_a_missing_label_shard_is_an_error(
    tmp_path: Path, lexicon: Lexicon, spans: SpanVocab
) -> None:
    samples_dir, labels_dir = tmp_path / "samples", tmp_path / "labels"
    samples_dir.mkdir()
    labels_dir.mkdir()
    pl.DataFrame([_sample("我爱北京").row()], schema=SAMPLE_SCHEMA).write_parquet(
        samples_dir / "test-00000.parquet"
    )
    stream = CorpusStream(samples_dir, labels_dir, SampleBuilder(lexicon, spans))
    with pytest.raises(FileNotFoundError, match="no label shard"):
        list(stream)


@pytest.mark.parametrize(
    ("reading", "expected"),
    [
        ("zhong1", "zhong"),
        ("lv4", "lv"),
        ("lü4", "lv"),
        ("jü1", "ju"),
        ("nv3", "nv"),
        ("jv1", "ju"),
        ("yv2", "yu"),
        ("xve4", "xue"),
        ("ju1", "ju"),
    ],
)
def test_readings_fold_to_what_a_keyboard_types(reading: str, expected: str) -> None:
    assert keyboard_form(reading) == expected


def test_an_unaligned_example_is_refused() -> None:
    with pytest.raises(ValueError, match="not aligned"):
        TrainingExample(id="x", spans=("wo",), span_ids=(1, 2), targets=(3,), context=None)


def test_a_batch_moves_wholesale(
    lexicon: Lexicon, spans: SpanVocab, tokenizer: BaseTokenizer
) -> None:
    builder = SampleBuilder(lexicon, spans)
    example = builder.build(_sample("我爱北京"), ["wo3", "ai4", "bei3", "jing1"], 0)
    assert example is not None
    batch = Collator(tokenizer)([example])
    moved = batch.to(torch.device("cpu"))
    assert isinstance(moved, Batch)
    assert moved.ids == batch.ids
    assert torch.equal(moved.targets, batch.targets)
