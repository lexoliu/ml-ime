"""The collator builds the very batches its per-position predecessor built.

Collation moved from a loop that wrote each position's candidate row and
letters as its own small tensor to one gather per field over the batch's
distinct spans, and the context rows from transformers' tensor conversion
to one ``torch.tensor`` over the tokenizer's rows. The batches a run trains on
must not move with either: every field, its dtype and its shape, the
generator state the context coin leaves behind, and the refusals. The
predecessor is kept here verbatim as the oracle, both run a real transformers
fast tokenizer, and both run over batches shaped like the corpus's --
ninety-odd rows, full, abbreviated and mixed typing over a lexicon wide enough
that an initial admits hundreds of characters -- clean and under the typo
model.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from mlime.train.lexicon import CandidateSpace, Lexicon, build_lexicon
from mlime.train.samples import (
    IGNORE_INDEX,
    LETTER_PAD,
    MAX_SPAN_LETTERS,
    Augmentation,
    Batch,
    Collator,
    TrainingExample,
    context_tail,
    letter_id,
    type_syllables,
)
from mlime.train.spans import SpanVocab
from mlime.typo import CorrectionTable, NoiseModel, load_syllables

MODEL_PATH = Path(__file__).parent.parent / "src/mlime/data/typo_model.json"
SYLLABLES = Path(__file__).parent.parent.parent / "crates/ime-pinyin/data/syllables.txt"

BATCH_FIELDS = (
    "input_ids",
    "attention_mask",
    "span_ids",
    "span_positions",
    "targets",
    "scored",
    "context_ids",
    "context_mask",
    "has_context",
    "context_rows",
    "candidate_ids",
    "candidate_counts",
    "span_letters",
)


def fast_tokenizer(characters: Sequence[str]) -> PreTrainedTokenizerFast:
    """A BERT-shaped transformers fast tokenizer over *characters*, one token each."""
    vocabulary = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    vocabulary.update({character: index + 5 for index, character in enumerate(characters)})
    inner = Tokenizer(models.WordPiece(vocabulary, unk_token="[UNK]"))
    inner.normalizer = normalizers.BertNormalizer(
        clean_text=True, handle_chinese_chars=True, strip_accents=False, lowercase=False
    )
    inner.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    inner.post_processor = processors.BertProcessing(("[SEP]", 3), ("[CLS]", 2))
    return PreTrainedTokenizerFast(
        tokenizer_object=inner,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


def legacy_collate(
    collator: Collator,
    tokenizer: PreTrainedTokenizerFast,
    examples: Sequence[TrainingExample],
    contexts: Sequence[str | None],
    context_rows: Sequence[int],
) -> Batch:
    """The per-position collation this module's ``Collator.collate`` replaced, verbatim."""
    if not examples:
        raise ValueError("cannot collate an empty batch")
    if len(context_rows) != len(examples):
        raise ValueError(f"{len(examples)} fill rows but {len(context_rows)} context assignments")
    width = max(len(example) for example in examples) + 2
    size = len(examples)
    input_ids = torch.full((size, width), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((size, width), dtype=torch.long)
    span_ids = torch.zeros((size, width), dtype=torch.long)
    span_positions = torch.zeros((size, width), dtype=torch.bool)
    targets = torch.full((size, width), IGNORE_INDEX, dtype=torch.long)

    candidate_lists = [
        [collator.candidates.resolve(span) for span in example.spans] for example in examples
    ]
    candidate_width = max(
        (len(row) for example_lists in candidate_lists for row in example_lists),
        default=1,
    )
    candidate_ids = torch.zeros((size, width, candidate_width), dtype=torch.long)
    candidate_counts = torch.zeros((size, width), dtype=torch.long)
    span_letters = torch.full((size, width, MAX_SPAN_LETTERS), LETTER_PAD, dtype=torch.long)

    for row, example in enumerate(examples):
        length = len(example)
        input_ids[row, 0] = tokenizer.cls_token_id
        input_ids[row, 1 : length + 1] = tokenizer.mask_token_id
        input_ids[row, length + 1] = tokenizer.sep_token_id
        attention_mask[row, : length + 2] = 1
        row_spans = torch.tensor(example.span_ids, dtype=torch.long)
        row_targets = torch.tensor(example.targets, dtype=torch.long)
        for column, (span, listed) in enumerate(
            zip(example.spans, candidate_lists[row], strict=True)
        ):
            if len(span) > MAX_SPAN_LETTERS:
                raise ValueError(
                    f"example {example.id} holds a {len(span)}-letter span {span!r}; "
                    f"the letter encoding reads at most {MAX_SPAN_LETTERS}"
                )
            candidate_ids[row, column + 1, : len(listed)] = torch.tensor(listed, dtype=torch.long)
            candidate_counts[row, column + 1] = len(listed)
            span_letters[row, column + 1, : len(span)] = torch.tensor(
                [letter_id(letter) for letter in span], dtype=torch.long
            )
        scored = row_targets != IGNORE_INDEX
        admitted = torch.tensor(
            [
                collator.candidates.admits(span, int(target))
                for span, target in zip(example.spans, example.targets, strict=True)
            ]
        )
        if not bool(admitted[scored].all()):
            raise ValueError(f"example {example.id} has a target its span does not admit")
        span_ids[row, 1 : length + 1] = row_spans
        span_positions[row, 1 : length + 1] = True
        targets[row, 1 : length + 1] = row_targets

    kept: list[str] = []
    has_context = torch.zeros(len(contexts), dtype=torch.float)
    for index, context in enumerate(contexts):
        keep = context is not None and collator.rng.random() >= collator.context_dropout
        kept.append(context_tail(context, collator.max_context_tokens) if keep and context else "")
        has_context[index] = 1.0 if keep else 0.0
    encoded = tokenizer(
        kept,
        padding=True,
        truncation=True,
        max_length=collator.max_context_tokens,
        return_tensors="pt",
    )
    scored_positions = (targets != IGNORE_INDEX).reshape(-1).nonzero(as_tuple=True)[0]
    columns = scored_positions % width
    if bool(((columns == 0) | (columns == width - 1)).any()):
        raise ValueError("a scored position sits outside the span columns")
    return Batch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        span_ids=span_ids,
        span_positions=span_positions,
        targets=targets,
        scored=scored_positions,
        context_ids=encoded["input_ids"],
        context_mask=encoded["attention_mask"],
        has_context=has_context,
        context_rows=torch.tensor(context_rows, dtype=torch.long),
        candidate_ids=candidate_ids,
        candidate_counts=candidate_counts,
        span_letters=span_letters,
        ids=tuple(example.id for example in examples),
    )


def assert_same_batch(new: Batch, old: Batch) -> None:
    """Every field equal, dtype for dtype and shape for shape."""
    assert new.ids == old.ids
    for name in BATCH_FIELDS:
        got, want = getattr(new, name), getattr(old, name)
        assert got.dtype == want.dtype, name
        assert got.shape == want.shape, name
        assert torch.equal(got, want), name


@pytest.fixture(name="wide")
def wide_fixture(spans: SpanVocab) -> tuple[Lexicon, dict[str, tuple[str, ...]]]:
    """Three thousand characters over the real syllables: an initial admits hundreds."""
    syllables = sorted(load_syllables(SYLLABLES))
    rng = random.Random(7)
    readings = {
        chr(0x4E00 + index): tuple(rng.sample(syllables, rng.choice((1, 1, 2))))
        for index in range(3000)
    }
    vocabulary = {character: index + 100 for index, character in enumerate(readings)}
    return build_lexicon(readings, vocabulary, spans), readings


def corpus_examples(
    lexicon: Lexicon,
    readings: dict[str, tuple[str, ...]],
    spans: SpanVocab,
    candidates: CandidateSpace,
    count: int,
    seed: int,
    noise: NoiseModel | None = None,
) -> list[tuple[TrainingExample, str | None]]:
    """Sentences typed the three ways the corpus is, kept only when every target is admitted."""
    rng = random.Random(seed)
    characters = list(readings)
    built: list[tuple[TrainingExample, str | None]] = []
    while len(built) < count:
        text = "".join(rng.choice(characters) for _ in range(rng.randint(1, 40)))
        typed = type_syllables(
            tuple(rng.choice(readings[character]) for character in text), rng, Augmentation()
        )
        if noise is not None:
            typed = noise.maybe_corrupt(typed, rng).spans
        targets = lexicon.indices(text)
        assert targets is not None
        if not candidates.admits_all(typed, targets):
            continue
        context = (
            None
            if rng.random() < 0.2
            else "".join(rng.choice(characters) for _ in range(rng.randint(0, 90)))
        )
        built.append(
            (
                TrainingExample(
                    id=f"s{len(built)}",
                    spans=typed,
                    span_ids=tuple(spans.id_or_unknown(span) for span in typed),
                    targets=targets,
                    context=context,
                ),
                context,
            )
        )
    return built


def run_both(
    candidates: CandidateSpace,
    groups: list[list[tuple[TrainingExample, str | None]]],
    shared_contexts: bool,
    characters: Sequence[str],
) -> None:
    """Collate every group with both implementations, from twin generators.

    The tokenizer knows every other character, so contexts carry unknown
    tokens as well as known ones.
    """
    tokenizer = fast_tokenizer(characters[::2])
    new = Collator(tokenizer, candidates, context_dropout=0.3, max_context_tokens=64, seed=11)
    old = Collator(tokenizer, candidates, context_dropout=0.3, max_context_tokens=64, seed=11)
    for group in groups:
        examples = [example for example, _ in group]
        if shared_contexts:
            # A ranking batch's shape: several fill rows read one context row.
            contexts = [context for _, context in group[::3]]
            rows = [index // 3 for index in range(len(group))]
        else:
            contexts = [context for _, context in group]
            rows = list(range(len(group)))
        assert_same_batch(
            new.collate(examples, contexts, rows),
            legacy_collate(old, tokenizer, examples, contexts, rows),
        )
        assert new.rng.getstate() == old.rng.getstate()


def groups_of(
    examples: list[tuple[TrainingExample, str | None]], seed: int
) -> list[list[tuple[TrainingExample, str | None]]]:
    """Corpus-sized batches and the edge sizes: one row, and one row of one span."""
    rng = random.Random(seed)
    groups = []
    start = 0
    while start < len(examples):
        size = rng.randint(60, 100)
        groups.append(examples[start : start + size])
        start += size
    groups.append(examples[:1])
    single = [pair for pair in examples if len(pair[0]) == 1]
    assert single, "the fixture corpus must hold a one-span sentence"
    groups.append(single[:1])
    return groups


def test_clean_batches_match_the_per_position_collation(
    wide: tuple[Lexicon, dict[str, tuple[str, ...]]], spans: SpanVocab
) -> None:
    lexicon, readings = wide
    candidates = CandidateSpace(spans, lexicon.candidate_mask)
    examples = corpus_examples(lexicon, readings, spans, candidates, 600, seed=1)
    widest = max(len(candidates.resolve(span)) for example, _ in examples for span in example.spans)
    assert widest > 300, "an initial must admit hundreds of characters for the test to bite"
    groups = groups_of(examples, seed=2)
    run_both(candidates, groups, shared_contexts=False, characters=list(readings))
    run_both(candidates, groups, shared_contexts=True, characters=list(readings))


def test_typo_batches_match_the_per_position_collation(
    wide: tuple[Lexicon, dict[str, tuple[str, ...]]], spans: SpanVocab
) -> None:
    lexicon, readings = wide
    data = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    data["per_sentence"] = 0.6
    noise = NoiseModel.parse(json.dumps(data))
    table = CorrectionTable(noise, load_syllables(SYLLABLES))
    candidates = CandidateSpace(spans, lexicon.candidate_mask, table, lexicon.homophones)
    examples = corpus_examples(lexicon, readings, spans, candidates, 300, seed=3, noise=noise)
    assert any(span not in spans for example, _ in examples for span in example.spans), (
        "the typo corpus must hold off-inventory spans"
    )
    run_both(
        candidates, groups_of(examples, seed=4), shared_contexts=False, characters=list(readings)
    )


def test_refusals_are_the_per_position_collations(lexicon: Lexicon, spans: SpanVocab) -> None:
    tokenizer = fast_tokenizer(lexicon.characters)
    candidates = CandidateSpace(spans, lexicon.candidate_mask)
    zhong = lexicon.index("中")
    wo = lexicon.index("我")
    fine = TrainingExample(
        id="a", spans=("wo",), span_ids=(spans.id("wo"),), targets=(wo,), context=None
    )
    cases = [
        TrainingExample(
            id="b", spans=("wo",), span_ids=(spans.id("wo"),), targets=(zhong,), context=None
        ),
        TrainingExample(
            id="c",
            spans=("w" * 13,),
            span_ids=(spans.id_or_unknown("w" * 13),),
            targets=(wo,),
            context=None,
        ),
        TrainingExample(
            id="d",
            spans=("w0",),
            span_ids=(spans.id_or_unknown("w0"),),
            targets=(wo,),
            context=None,
        ),
    ]
    for bad in cases:
        group = [fine, bad]
        with pytest.raises(ValueError) as old:
            legacy_collate(Collator(tokenizer, candidates), tokenizer, group, [None, None], [0, 1])
        with pytest.raises(ValueError) as new:
            Collator(tokenizer, candidates)(group)
        assert str(new.value) == str(old.value)
