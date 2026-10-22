"""The key front, its examples and batches, and the S1 warm-up (#216 §1-§2, §4, §6-§7).

Data first: the QWERTY layout the front reads is the one the noise model's
neighbour table was drawn from; a key example's keys are its typed spans run
together, its gold alignments are the legality automaton's; the typo
augmentation hits its share of the sentences, never an initial, and only with
a slip the corrections read back; a gold the automaton rejects is counted and
dropped. Then the model: a fresh front hands the tower each key's letter, S1
trains the front and the head only, the parent loads whole, the gate's frame
is the parent's own when the student matches the teacher, and the run --
through ``e2e()`` itself -- logs every loss component, resumes exactly and
compiles once.
"""

from __future__ import annotations

import copy
import json
import math
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
import torch
from conftest import CORPUS, READINGS, StubTokenizer
from test_e2e_model import READER_TINY, TINY
from torch import nn
from transformers import BertConfig

from mlime.data.corpus import Sample
from mlime.legality import Legality
from mlime.train.arbitration import ReadingArbitration
from mlime.train.charlm_vocab import SPECIALS, CharVocab
from mlime.train.count import count_batches
from mlime.train.ctc import constrained_ctc
from mlime.train.e2e import e2e, load_model
from mlime.train.e2e_model import E2EConfig, E2EModel, emittable_ids
from mlime.train.key_model import (
    KEY_PARAMETER_PREFIXES,
    KeyE2EModel,
    key_evaluate,
    letter_token_ids,
)
from mlime.train.keys import (
    QWERTY_ROWS,
    FillTowerTraining,
    KeyBatch,
    KeyCollator,
    KeyExample,
    KeyFrontConfig,
    KeySampleBuilder,
    key_coordinates,
    typed_in_full,
)
from mlime.train.lexicon import CandidateSpace, Lexicon
from mlime.train.loop import TrainingConfig
from mlime.train.producer import EpochBatches
from mlime.train.run import RunPaths, Slices, Vocabularies
from mlime.train.samples import LETTER_PAD, Augmentation, Collator, CorpusStream, letter_id
from mlime.train.spans import SpanVocab
from mlime.typo import MULTI_LETTER_INITIALS, NoiseModel, load_syllables

ROOT = Path(__file__).parent.parent.parent
DATA = ROOT / "crates/ime-pinyin/data"

#: The tiny tower has 64 positions; two are the sentinels.
KEYS = KeyFrontConfig(layers=2, max_keys=62)

#: What S1 trains of the fill path beside the front and the head (#241).
TOWER = FillTowerTraining()
CROSS_ATTENTION = FillTowerTraining(cross_attention=True)
TRAINED_PARTS = [
    pytest.param(None, id="frozen"),
    pytest.param(TOWER, id="tower"),
    pytest.param(CROSS_ATTENTION, id="cross-attention"),
]
TRAINED_TOWERS = TRAINED_PARTS[1:]


class KeyTokenizer(StubTokenizer):
    """The stub tokenizer with the 26 letters in its vocabulary, as the base model's has."""

    def get_vocab(self) -> dict[str, int]:
        return {chr(ord("a") + offset): 200 + offset for offset in range(LETTER_PAD)}


@pytest.fixture(name="vocabularies")
def vocabularies_fixture(
    spans: SpanVocab, lexicon: Lexicon, arbitration: ReadingArbitration
) -> Vocabularies:
    tokenizer = KeyTokenizer()
    return Vocabularies(
        spans=spans,
        tokenizer=tokenizer,
        context_tokenizer=tokenizer,
        lexicon=lexicon,
        arbitration=arbitration,
        syllables=frozenset(load_syllables(DATA / "syllables.txt")),
    )


@pytest.fixture(name="legality")
def legality_fixture(vocabularies: Vocabularies) -> Legality:
    return vocabularies.legality()


@pytest.fixture(name="reader_vocab")
def reader_vocab_fixture(lexicon: Lexicon) -> CharVocab:
    return CharVocab(chars=SPECIALS + tuple(lexicon.characters))


def key_config(
    train_fill_tower: FillTowerTraining | None = None, cross_attention_dropout: float | None = None
) -> E2EConfig:
    config = E2EConfig(
        cross_attention_layers=1,
        decoder=READER_TINY,
        keys=KEYS,
        key_warm_up=True,
        train_fill_tower=train_fill_tower,
    )
    if cross_attention_dropout is None:
        return config
    return replace(config, cross_attention_dropout=cross_attention_dropout)


def tiny_key_model(
    lexicon: Lexicon,
    vocab: CharVocab,
    train_fill_tower: FillTowerTraining | None = None,
    tower: BertConfig = TINY,
    cross_attention_dropout: float | None = None,
) -> KeyE2EModel:
    torch.manual_seed(0)
    return KeyE2EModel.compose(
        tower,
        lexicon,
        key_config(train_fill_tower, cross_attention_dropout),
        len(vocab),
        emittable_ids(lexicon, vocab),
    )


def open_gates(model: E2EModel, value: float = 0.5) -> None:
    """Open the fill tower's context gates, as a trained parent's are.

    A fresh block's gate is zero, which multiplies its attention away: the
    attention's weights would get an exactly zero gradient and the teacher's
    states would not depend on them at all, so a test of either would pass
    whatever the blocks did.
    """
    with torch.no_grad():
        for layer in model.gated_layers():
            layer.gate.fill_(value)


def tiny_parent(lexicon: Lexicon, vocab: CharVocab, seed: int) -> E2EModel:
    """A span model to warm a key front up over, its weights apart from the key model's."""
    torch.manual_seed(seed)
    return E2EModel.compose(
        TINY,
        lexicon,
        E2EConfig(cross_attention_layers=1, decoder=READER_TINY),
        len(vocab),
        emittable_ids(lexicon, vocab),
    )


def samples() -> list[tuple[Sample, list[str]]]:
    """The fixture corpus as samples with their labels."""
    return [
        (Sample(id=f"k-{index}", source="test", text=text, context=context), readings)
        for index, (text, readings, context) in enumerate(CORPUS)
    ]


def build_examples(
    vocabularies: Vocabularies, epochs: int = 4, typos: bool = False
) -> list[KeyExample]:
    builder = vocabularies.key_builder(Augmentation(typos=typos), 5, KEYS)
    built = [
        builder.build(sample, readings, epoch)
        for epoch in range(epochs)
        for sample, readings in samples()
    ]
    return [example for example in built if example is not None]


def pressed_spans(example: KeyExample) -> list[str]:
    starts = (0, *(end + 1 for end in example.ends[:-1]))
    return [example.keys[start : end + 1] for start, end in zip(starts, example.ends, strict=True)]


def key_collator(spans: SpanVocab, lexicon: Lexicon, dropout: float = 0.0) -> KeyCollator:
    tokenizer = KeyTokenizer()
    return KeyCollator(
        Collator(
            tokenizer,
            tokenizer,
            CandidateSpace(spans, lexicon.candidate_mask),
            context_dropout=dropout,
        )
    )


# --- the layout -----------------------------------------------------------------


def test_the_coordinates_are_the_staggered_qwerty_rows() -> None:
    table = key_coordinates()
    assert table.shape == (LETTER_PAD, 2)
    assert table[letter_id("q")].tolist() == [0.0, 0.0]
    assert table[letter_id("a")].tolist() == [1.0, 0.25]
    assert table[letter_id("z")].tolist() == [2.0, 0.75]
    assert table[letter_id("p")].tolist() == [0.0, 9.0]
    assert sorted("".join(QWERTY_ROWS)) == [chr(ord("a") + i) for i in range(LETTER_PAD)]


def test_the_noise_models_neighbours_are_the_keys_within_a_key_and_a_quarter() -> None:
    """The front's geometry and the slips it is trained on describe one keyboard."""
    model = json.loads((DATA / "typo.json").read_text(encoding="utf-8"))
    table = key_coordinates()
    letters = [chr(ord("a") + i) for i in range(LETTER_PAD)]

    def distance(a: str, b: str) -> float:
        return float(torch.dist(table[letter_id(a)], table[letter_id(b)]))

    listed = {(a, b) for a, near in model["neighbours"].items() for b in near}
    within = {(a, b) for a in letters for b in letters if a != b and distance(a, b) <= 1.25}
    assert listed == within


# --- examples --------------------------------------------------------------------


def test_a_clean_example_is_its_typed_spans_run_together_with_legal_alignments(
    vocabularies: Vocabularies, legality: Legality
) -> None:
    examples = build_examples(vocabularies)
    assert len(examples) == 4 * len(CORPUS)
    styles = set()
    for example in examples:
        assert pressed_spans(example) == list(example.spans)
        assert not any(example.corrupted)
        gold = example.arcs()
        assert gold.keys == len(example.keys) and gold.length == len(example.targets)
        assert gold.best == 0.0
        again = legality.gold_ids(example.keys, example.targets, KEYS.options)
        assert again is not None and (again.dense() == gold.dense()).all()
        assert example.fill_width() == len(example.keys) + len(example.spans) + 4
        styles.add(tuple(len(span) for span in example.spans))
    assert len(styles) > 4, "the typing styles never varied"


def test_the_typo_augmentation_hits_its_share_and_only_readable_full_spans(
    vocabularies: Vocabularies, legality: Legality
) -> None:
    noise = NoiseModel.load(DATA / "typo.json")
    builder = vocabularies.key_builder(Augmentation(typos=True), 9, KEYS)
    sentence, readings = "我爱北京", ["wo3", "ai4", "bei3", "jing1"]
    meant_readings = ("wo", "ai", "bei", "jing")
    eligible = slipped = 0
    edits: Counter[str] = Counter()
    for index in range(3000):
        example = builder.build(
            Sample(id=f"t-{index}", source="test", text=sentence, context=None), readings, 0
        )
        assert example is not None
        pressed = pressed_spans(example)
        if any(
            typed_in_full(span, reading)
            for span, reading in zip(example.spans, meant_readings, strict=True)
        ):
            eligible += 1
        if any(example.corrupted):
            slipped += 1
        for span, meant, reading, target, corrupted in zip(
            pressed,
            example.spans,
            meant_readings,
            example.targets,
            example.corrupted,
            strict=True,
        ):
            if span != meant:
                assert corrupted
            if not corrupted:
                continue
            edits[f"{meant}>{span}"] += 1
            # Never an initial, never into one, and always read back.
            assert typed_in_full(meant, reading)
            assert len(span) >= 2 and span not in MULTI_LETTER_INITIALS
            assert math.isfinite(legality.char_cost(span, False, target, KEYS.options))
    counts = builder.counts
    assert counts.illegal_gold == 0
    assert counts.typo_sentences / eligible == pytest.approx(noise.per_sentence, abs=0.04)
    assert slipped <= counts.typo_sentences
    assert counts.typo_edits > counts.typo_sentences
    assert len(edits) > 20, "the slips never varied"


def test_a_gold_the_automaton_rejects_is_counted_and_dropped(
    spans: SpanVocab,
    lexicon: Lexicon,
    arbitration: ReadingArbitration,
    legality: Legality,
) -> None:
    # The automaton is told 北 reads only "nan": every 北京 the corpus types is illegal.
    readings = {character: list(READINGS[character]) for character in lexicon.characters}
    readings["北"] = ["nan"]
    wrong = Legality(lexicon.characters, readings, legality.corrections)
    builder = KeySampleBuilder(lexicon, spans, arbitration, wrong, KEYS, seed=5)
    built = [builder.build(sample, labels, 0) for sample, labels in samples()]
    assert [example is None for example in built] == [
        "北" in sample.text for sample, _ in samples()
    ]
    assert builder.counts.illegal_gold == sum("北" in sample.text for sample, _ in samples())

    short = KeySampleBuilder(
        lexicon, spans, arbitration, legality, replace(KEYS, max_keys=3), seed=5
    )
    assert all(short.build(sample, labels, 0) is None for sample, labels in samples())
    assert short.counts.keys_too_long == len(CORPUS)

    with pytest.raises(ValueError, match="come together"):
        KeySampleBuilder(lexicon, spans, arbitration, legality, KEYS, Augmentation(typos=True))


def test_the_collated_keys_line_up_with_the_examples(
    vocabularies: Vocabularies, spans: SpanVocab, lexicon: Lexicon
) -> None:
    examples = build_examples(vocabularies, epochs=2, typos=False)[:6]
    batch = key_collator(spans, lexicon)(examples)
    width = batch.key_input_ids.shape[1]
    flat_letters = batch.key_letters.reshape(-1)
    assert batch.key_positions.numel() == sum(len(example.keys) for example in examples)
    for row, example in enumerate(examples):
        letters = [letter_id(letter) for letter in example.keys]
        assert batch.key_letters[row, 1 : len(letters) + 1].tolist() == letters
        own = flat_letters.index_select(0, batch.key_positions)[
            batch.key_index[row, : len(letters)]
        ]
        assert own.tolist() == letters
        assert batch.key_input_ids[row, 0] == StubTokenizer.cls_token_id
        assert batch.key_input_ids[row, len(letters) + 1] == StubTokenizer.sep_token_id
        assert int(batch.key_attention[row].sum()) == len(letters) + 2
        assert batch.span_end[row, : len(example.spans)].tolist() == list(example.ends)
        assert batch.ends[row].nonzero().flatten().tolist() == list(example.ends)
        dense = torch.from_numpy(example.arcs().dense()).float()
        assert torch.equal(batch.cost[row, : len(example.spans), : len(example.keys)], dense)
        assert torch.isinf(batch.cost[row, len(example.spans) :]).all()
        assert torch.isinf(batch.cost[row, :, len(example.keys) :]).all()
    # The span ends, in the span batch's scored order, are the keys that close each span.
    rows = batch.end_positions // width
    assert torch.equal(rows, batch.spans.scored // batch.spans.targets.shape[1])
    assert flat_letters.index_select(0, batch.end_positions).tolist() == [
        letter_id(example.keys[end]) for example in examples for end in example.ends
    ]
    # Any key-level distribution over the batch has every gold finite.
    uniform = torch.full((batch.key_positions.numel(), lexicon.size + 1), -1.0)
    blank = uniform[:, -1][batch.key_index]
    emit = uniform[:, 0][batch.key_index].unsqueeze(1).expand(-1, batch.gold.shape[1], -1)
    nll = constrained_ctc(blank, emit, batch.cost, batch.keys, batch.lengths)
    assert torch.isfinite(nll).all() and (nll < 1e3).all()


# --- the model -------------------------------------------------------------------


def test_a_fresh_front_hands_the_tower_each_keys_letter(
    lexicon: Lexicon, reader_vocab: CharVocab
) -> None:
    model = tiny_key_model(lexicon, reader_vocab).eval()
    letters = torch.tensor([[LETTER_PAD, 3, 0, 25, LETTER_PAD], [LETTER_PAD, 7, 7, LETTER_PAD, 0]])
    with torch.no_grad():
        torch.testing.assert_close(model.key_front(letters), model.key_front.letters(letters))
    assert not model.blank.weight.any() and not model.blank.bias.any()


def test_the_parent_loads_whole_and_seeds_the_letters(
    lexicon: Lexicon, reader_vocab: CharVocab, tmp_path: Path
) -> None:
    torch.manual_seed(1)
    parent = E2EModel.compose(
        TINY,
        lexicon,
        E2EConfig(cross_attention_layers=1, decoder=READER_TINY),
        len(reader_vocab),
        emittable_ids(lexicon, reader_vocab),
    )
    model = tiny_key_model(lexicon, reader_vocab)
    ids = letter_token_ids(KeyTokenizer().get_vocab())
    model.load_parent(parent.state_dict(), tmp_path / "parent.pt", ids)
    own = model.state_dict()
    for name, tensor in parent.state_dict().items():
        assert torch.equal(own[name], tensor), name
    words = parent.fill.embeddings.word_embeddings.weight
    assert torch.equal(model.key_front.letters.weight[:LETTER_PAD], words[ids])
    assert not model.key_front.letters.weight[LETTER_PAD].any()
    with pytest.raises(ValueError, match="already holds key-level weights"):
        model.load_parent(model.state_dict(), tmp_path / "keys.pt", ids)
    with pytest.raises(KeyError, match="letters"):
        letter_token_ids({"a": 1})


@pytest.mark.parametrize("train_fill_tower", TRAINED_PARTS)
def test_s1_trains_only_the_front_the_head_and_the_parts_it_names_and_keeps_the_rest_in_eval(
    vocabularies: Vocabularies,
    spans: SpanVocab,
    lexicon: Lexicon,
    reader_vocab: CharVocab,
    train_fill_tower: FillTowerTraining | None,
) -> None:
    """Exactly the named parts get a gradient -- the parent's a nonzero one -- the copies none.

    The context gates are open, so a trained block's attention has a
    gradient to get; the decoder's own cross-attention is not the fill
    tower's and stays frozen whatever trains.
    """
    model = tiny_key_model(lexicon, reader_vocab, train_fill_tower)
    open_gates(model)
    model.freeze_for_key_warm_up()
    parts = {
        None: (),
        TOWER: ("fill",),
        CROSS_ATTENTION: ("fill", "cross_attention"),
    }[train_fill_tower]
    assert model.trained_parts() == parts
    trained = (*KEY_PARAMETER_PREFIXES, "head.", *(f"{part}." for part in parts))
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == {name for name, _ in model.named_parameters() if name.startswith(trained)}
    assert any(name.startswith("key_front.blocks.") for name in trainable)
    assert "blank.weight" in trainable
    assert any(name.startswith("fill.encoder.") for name in trainable) == ("fill" in parts)
    crossing = "cross_attention" in parts
    assert ("cross_attention.0.attention.in_proj_weight" in trainable) == crossing
    assert not any(name.startswith(("decoder.", "teacher_")) for name in trainable)
    assert (model.teacher_fill is not None) == ("fill" in parts)
    assert (model.teacher_cross_attention is not None) == crossing
    model.train()
    frozen = ["context", "decoder", "letter_encoder"]
    for part in ("fill", "cross_attention"):
        if part in parts:
            frozen.append(f"teacher_{part}")
            assert all(module.training for module in getattr(model, part).modules()), part
        else:
            frozen.append(part)
    for name in frozen:
        assert not any(module.training for module in getattr(model, name).modules()), name
    assert model.key_front.training

    batch = key_collator(spans, lexicon, dropout=0.3)(build_examples(vocabularies)[:8])
    assert bool(batch.spans.has_context.any())
    output = model(batch)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    for name, parameter in model.named_parameters():
        if name not in trainable:
            assert parameter.grad is None, name
            continue
        assert parameter.grad is not None, name
        # A fresh front's blocks sit behind zero output projections, so only
        # the parent's parts are sure of a gradient that is not zero.
        if name.startswith(tuple(f"{part}." for part in parts)):
            assert bool(parameter.grad.any()), name
    for component in (
        "ctc_loss",
        "distill_loss",
        "segmentation_loss",
        "clean_span_loss",
        "typo_span_loss",
        "typo_share",
    ):
        assert torch.isfinite(output.extras[component]), component
    total = output.extras["ctc_loss"] + KEYS.distill_weight * output.extras["distill_loss"]
    torch.testing.assert_close(output.loss.detach(), total)

    # The tower trains in the pretrained group, at the head's rate.
    groups = model.parameter_groups(1e-4, 1e-3)
    base = {id(p) for p in groups[0]["params"]}  # type: ignore[attr-defined]
    new = {id(p) for p in groups[1]["params"]}  # type: ignore[attr-defined]
    named = dict(model.named_parameters())
    assert base == {
        id(named[name]) for name in trainable if not name.startswith(KEY_PARAMETER_PREFIXES)
    }
    assert base == {
        id(named[name])
        for name in trainable
        if name.startswith(("head.", *(f"{part}." for part in parts)))
    }
    assert new == {id(named[name]) for name in trainable if name.startswith(KEY_PARAMETER_PREFIXES)}


@pytest.mark.parametrize("train_fill_tower", TRAINED_PARTS)
def test_the_s1_loss_falls_on_a_fixed_batch(
    vocabularies: Vocabularies,
    spans: SpanVocab,
    lexicon: Lexicon,
    reader_vocab: CharVocab,
    train_fill_tower: FillTowerTraining | None,
) -> None:
    model = tiny_key_model(lexicon, reader_vocab, train_fill_tower)
    model.freeze_for_key_warm_up()
    model.train()
    batch = key_collator(spans, lexicon)(build_examples(vocabularies)[:8])
    optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-3)
    losses = []
    totals = []
    for _ in range(40):
        optimiser.zero_grad()
        output = model(batch)
        assert output.loss is not None
        output.loss.backward()
        optimiser.step()
        losses.append(float(output.extras["ctc_loss"]))
        totals.append(float(output.loss.detach()))
    assert losses[-1] < 0.5 * losses[0]
    assert totals[-1] < 0.5 * totals[0]


@pytest.mark.parametrize("train_fill_tower", TRAINED_TOWERS)
def test_the_teacher_stays_the_parents_while_the_trained_parts_move(
    vocabularies: Vocabularies,
    spans: SpanVocab,
    lexicon: Lexicon,
    reader_vocab: CharVocab,
    tmp_path: Path,
    train_fill_tower: FillTowerTraining,
) -> None:
    """The copies are seeded from the parent, get no gradient and no update, and their states hold.

    The teacher's states are the frozen parent's own encoder pass over the
    spans before any step and after many, bit for bit, while the parts the front
    feeds have moved, and the gate's parent frame is the parent's frame
    throughout. The parent's context gates are open, so its states read its
    blocks; once the student's blocks have moved, the teacher's tower run
    through them is no longer the parent, so the copy is what holds it.
    """
    parent = tiny_parent(lexicon, reader_vocab, seed=1)
    open_gates(parent)
    # Frozen as a parent is: nn.MultiheadAttention takes another kernel when
    # none of its weights wants a gradient, a rounding apart from the one it
    # takes otherwise, and the teacher's copies want none.
    parent.requires_grad_(False).eval()
    model = tiny_key_model(lexicon, reader_vocab, train_fill_tower)
    model.load_parent(
        parent.state_dict(), tmp_path / "parent.pt", letter_token_ids(KeyTokenizer().get_vocab())
    )
    teachers: list[tuple[nn.Module, nn.Module, nn.Module]] = []
    assert model.teacher_fill is not None
    teachers.append((model.teacher_fill, parent.fill, model.fill))
    if train_fill_tower.cross_attention:
        assert model.teacher_cross_attention is not None
        teachers.append(
            (model.teacher_cross_attention, parent.cross_attention, model.cross_attention)
        )
    else:
        assert model.teacher_cross_attention is None
    for teacher, held, _ in teachers:
        for name, tensor in held.state_dict().items():
            assert torch.equal(teacher.state_dict()[name], tensor), name
    model.freeze_for_key_warm_up()
    batch = key_collator(spans, lexicon)(build_examples(vocabularies)[:8])
    assert bool(batch.spans.has_context.any())
    with torch.no_grad():
        expected = parent.encode(batch.spans)
        frame = parent.scores(batch.spans)

    def teacher_states() -> torch.Tensor:
        with torch.no_grad():
            return model.span_states(batch, model.encode_context(batch.spans))

    model.train()
    torch.testing.assert_close(teacher_states(), expected, rtol=0, atol=0)
    before = [
        {name: tensor.clone() for name, tensor in part.state_dict().items()}
        for _, _, part in teachers
    ]
    optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-3)
    for _ in range(10):
        optimiser.zero_grad()
        output = model(batch)
        assert output.loss is not None
        output.loss.backward()
        for teacher, _, _ in teachers:
            assert all(parameter.grad is None for parameter in teacher.parameters())
        optimiser.step()
    for (teacher, held, part), started in zip(teachers, before, strict=True):
        assert teacher.training is False
        for name, tensor in teacher.state_dict().items():
            assert torch.equal(tensor, held.state_dict()[name]), name
        moved = [
            name
            for name, tensor in part.state_dict().items()
            if not torch.equal(tensor, started[name])
        ]
        assert moved, type(part).__name__
    if train_fill_tower.cross_attention:
        block = model.cross_attention[0]
        assert not torch.equal(
            block.attention.in_proj_weight, parent.cross_attention[0].attention.in_proj_weight
        )
        assert not torch.equal(block.gate, parent.cross_attention[0].gate)
        with torch.no_grad():
            spans_ = batch.spans
            memory = model.encode_context(spans_)
            tower = model.teacher_tower
            through_student = model.fill_pass(
                tower,
                model.cross_attention,
                model._fill_inputs(spans_, tower),
                spans_.attention_mask,
                memory,
                spans_.context_rows,
                spans_.has_context,
            )
        assert not torch.allclose(through_student, expected)
    torch.testing.assert_close(teacher_states(), expected, rtol=0, atol=0)
    model.eval()
    torch.testing.assert_close(teacher_states(), expected, rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(model.parent_frame(batch), frame, rtol=0, atol=0)


def test_the_teacher_loads_as_saved_or_from_parts_that_never_trained(
    lexicon: Lexicon, reader_vocab: CharVocab, tmp_path: Path
) -> None:
    """A checkpoint without a copy of a part seeds it from the part; one with the copy keeps it."""
    frozen = tiny_key_model(lexicon, reader_vocab)
    frozen.load_parent(
        tiny_parent(lexicon, reader_vocab, seed=1).state_dict(),
        tmp_path / "parent.pt",
        letter_token_ids(KeyTokenizer().get_vocab()),
    )
    tower = tiny_key_model(lexicon, reader_vocab, TOWER)
    tower.load_resumed(frozen.state_dict(), tmp_path / "frozen.pt")
    assert tower.teacher_fill is not None
    for name, tensor in frozen.fill.state_dict().items():
        assert torch.equal(tower.teacher_fill.state_dict()[name], tensor), name

    def moved(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
        """*state* as a run that trained the part under *prefix* saves it: the part moved."""
        return {
            name: tensor + 1.0 if name.startswith(prefix) else tensor
            for name, tensor in state.items()
        }

    # A run that trained its tower saved the teacher beside it, and resumes it as saved.
    saved = moved(tower.state_dict(), "fill.encoder.")
    resumed = tiny_key_model(lexicon, reader_vocab, TOWER)
    resumed.load_resumed(saved, tmp_path / "tower.pt")
    assert resumed.teacher_fill is not None
    for name, tensor in frozen.fill.state_dict().items():
        assert torch.equal(resumed.teacher_fill.state_dict()[name], tensor), name
    assert not torch.equal(
        resumed.fill.encoder.layer[0].output.dense.weight,
        frozen.fill.encoder.layer[0].output.dense.weight,
    )
    with pytest.raises(ValueError, match="unexpected"):
        tiny_key_model(lexicon, reader_vocab).load_resumed(saved, tmp_path / "tower.pt")

    # That run never trained its blocks, so they are its parent's, and a run
    # that trains them seeds their copy from them; its tower's copy loads as saved.
    crossing = tiny_key_model(lexicon, reader_vocab, CROSS_ATTENTION)
    crossing.load_resumed(saved, tmp_path / "tower.pt")
    assert crossing.teacher_fill is not None and crossing.teacher_cross_attention is not None
    for name, tensor in frozen.fill.state_dict().items():
        assert torch.equal(crossing.teacher_fill.state_dict()[name], tensor), name
    for name, tensor in frozen.cross_attention.state_dict().items():
        assert torch.equal(crossing.teacher_cross_attention.state_dict()[name], tensor), name

    # A run that trained its blocks saved their copy too, and resumes both as saved.
    saved = moved(crossing.state_dict(), "cross_attention.")
    resumed = tiny_key_model(lexicon, reader_vocab, CROSS_ATTENTION)
    resumed.load_resumed(saved, tmp_path / "cross.pt")
    assert resumed.teacher_cross_attention is not None
    for name, tensor in frozen.cross_attention.state_dict().items():
        assert torch.equal(resumed.teacher_cross_attention.state_dict()[name], tensor), name
        assert not torch.equal(resumed.cross_attention.state_dict()[name], tensor), name
    for name, tensor in frozen.fill.state_dict().items():
        assert torch.equal(resumed.teacher_fill.state_dict()[name], tensor), name
    # Its blocks moved: a model that would distil through them refuses the run.
    with pytest.raises(ValueError, match="unexpected"):
        tiny_key_model(lexicon, reader_vocab, TOWER).load_resumed(saved, tmp_path / "cross.pt")
    with pytest.raises(ValueError, match="fill tower trains"):
        E2EConfig(cross_attention_layers=1, decoder=READER_TINY, train_fill_tower=TOWER)


def test_the_gates_frame_is_the_parents_when_the_student_matches_the_teacher(
    vocabularies: Vocabularies,
    spans: SpanVocab,
    lexicon: Lexicon,
    reader_vocab: CharVocab,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tiny_key_model(lexicon, reader_vocab).eval()
    examples = build_examples(vocabularies)[:6]
    batch = key_collator(spans, lexicon)(examples)
    hidden = TINY.hidden_size

    def matching(batch_: KeyBatch, memory_: Any) -> torch.Tensor:
        """The teacher's state at each span, placed at the key that closes it."""
        teacher = model.span_states(batch_, memory_).reshape(-1, hidden)
        placed = teacher.new_zeros((batch_.key_input_ids.numel(), hidden)).index_copy(
            0, batch_.end_positions, teacher.index_select(0, batch_.spans.scored)
        )
        return placed.view(*batch_.key_input_ids.shape, hidden)

    monkeypatch.setattr(model, "encode_keys", matching)
    with torch.no_grad():
        torch.testing.assert_close(model.student_frame(batch), model.parent_frame(batch))
    student, parent = key_evaluate(
        model, examples, key_collator(spans, lexicon), torch.device("cpu"), 64
    )
    assert student == parent
    assert student.scored == sum(len(example.targets) for example in examples)


# --- the run ---------------------------------------------------------------------


def parent_checkpoint(lexicon: Lexicon, reader_vocab: CharVocab, path: Path) -> Path:
    torch.manual_seed(2)
    parent = E2EModel.compose(
        TINY,
        lexicon,
        E2EConfig(cross_attention_layers=1, decoder=READER_TINY),
        len(reader_vocab),
        emittable_ids(lexicon, reader_vocab),
    )
    torch.save({"e2e": asdict(parent.config), "model": parent.state_dict(), "step": 9}, path)
    return path


def run_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("train_fill_tower", TRAINED_PARTS)
def test_an_s1_run_logs_every_component_resumes_exactly_and_is_scored_against_its_parent(
    corpus: tuple[Path, Path],
    vocabularies: Vocabularies,
    lexicon: Lexicon,
    reader_vocab: CharVocab,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    train_fill_tower: FillTowerTraining | None,
) -> None:
    monkeypatch.setattr(Vocabularies, "load", classmethod(lambda cls, *_: vocabularies))
    monkeypatch.setattr(BertConfig, "from_pretrained", classmethod(lambda cls, *_: TINY))
    parent = parent_checkpoint(lexicon, reader_vocab, tmp_path / "parent.pt")
    samples_dir, labels_dir = corpus
    slices = Slices(
        train=("test-00000.parquet",), held_out=("test-00001.parquet",), max_held_out_examples=8
    )
    training = TrainingConfig(
        max_steps=4, token_budget=96, log_every=1, checkpoint_every=2, fp16=False, seed=3
    )

    def paths(name: str) -> RunPaths:
        return RunPaths(
            samples=samples_dir,
            labels=labels_dir,
            char_table=tmp_path / "unread",
            out=tmp_path / name,
        )

    whole = e2e(
        paths("whole"),
        slices,
        training,
        init_e2e=parent,
        keys=KEYS,
        key_warm_up=True,
        train_fill_tower=train_fill_tower,
    )
    records = run_records(whole.metrics)
    steps = [record for record in records if record.get("event") == "step"]
    assert len(steps) == 4
    for record in steps:
        for component in (
            "ctc_loss",
            "distill_loss",
            "segmentation_loss",
            "clean_span_loss",
            "typo_span_loss",
            "typo_share",
        ):
            assert math.isfinite(record[component]), component
    summary = next(record for record in records if record.get("event") == "summary")
    for name in ("parent_accuracy_with_context", "parent_accuracy_without_context"):
        assert 0.0 <= summary[name] <= 1.0
    assert summary["build_counts"]["illegal_gold"] == 0
    provenance = next(record for record in records if record.get("event") == "provenance")
    assert provenance["e2e"]["keys"] == asdict(KEYS)
    assert provenance["e2e"]["train_fill_tower"] == (
        None if train_fill_tower is None else asdict(train_fill_tower)
    )

    resumed = e2e(
        paths("resumed"),
        slices,
        training,
        resume=tmp_path / "whole" / "checkpoint-000002.pt",
        keys=KEYS,
        key_warm_up=True,
        train_fill_tower=train_fill_tower,
    )
    resumed_losses = [
        record["loss"] for record in run_records(resumed.metrics) if record.get("event") == "step"
    ]
    assert resumed_losses == [record["loss"] for record in steps[2:]]
    # The resumed run is this one: what of the tower trains is part of its record.
    switched = {None: TOWER, TOWER: CROSS_ATTENTION, CROSS_ATTENTION: TOWER}[train_fill_tower]
    with pytest.raises(ValueError, match="train_fill_tower"):
        e2e(
            paths("switched"),
            slices,
            training,
            resume=tmp_path / "whole" / "checkpoint-000002.pt",
            keys=KEYS,
            key_warm_up=True,
            train_fill_tower=switched,
        )

    # A key checkpoint is not a span model the lattice search can decode.
    with pytest.raises(ValueError, match="key-level model"):
        load_model(tmp_path / "whole" / "checkpoint-000004.pt", lexicon, torch.device("cpu"))
    # And a key front over a fresh composition is refused before anything loads.
    with pytest.raises(ValueError, match="--init-e2e"):
        e2e(
            paths("fresh"),
            slices,
            training,
            init_encoder=tmp_path / "unread",
            init_decoder=tmp_path / "unread",
            keys=KEYS,
            key_warm_up=True,
        )


def test_the_count_is_the_key_loops_own(
    corpus: tuple[Path, Path], vocabularies: Vocabularies, tmp_path: Path
) -> None:
    samples_dir, labels_dir = corpus
    budget = 96
    census = count_batches(
        vocabularies,
        RunPaths(
            samples=samples_dir, labels=labels_dir, char_table=tmp_path / "u", out=tmp_path / "c"
        ),
        Slices(train=("test-00000.parquet", "test-00001.parquet"), held_out=()),
        token_budget=budget,
        seed=1,
        keys=KEYS,
    )
    batches = EpochBatches(
        CorpusStream(samples_dir, labels_dir, vocabularies.key_builder(None, 1, KEYS)),
        vocabularies.key_collator(),
        budget,
    )
    taken = 0
    while True:
        next(batches)
        if batches.epoch != 0:
            break
        taken += 1
    assert taken > 1
    assert census.ranks[0].epochs == [taken]
    assert census.keys == KEYS.sampling()
    assert census.build_counts["kept"] == 128
    plain = count_batches(
        vocabularies,
        RunPaths(
            samples=samples_dir, labels=labels_dir, char_table=tmp_path / "u", out=tmp_path / "p"
        ),
        Slices(train=("test-00000.parquet", "test-00001.parquet"), held_out=()),
        token_budget=budget,
        seed=1,
    )
    # A key example costs its keys beside its spans: fewer fit a budget.
    assert plain.ranks[0].epochs[0] < taken


@pytest.mark.parametrize("train_fill_tower", TRAINED_PARTS)
def test_the_s1_forward_compiles_once_across_key_lengths(
    vocabularies: Vocabularies,
    spans: SpanVocab,
    lexicon: Lexicon,
    reader_vocab: CharVocab,
    train_fill_tower: FillTowerTraining | None,
) -> None:
    """Marked dims make every compiled callable dynamic from its first call.

    After the first batch, batches of other key widths, span widths and row
    counts compile nothing new and break no graph.
    """
    from torch._dynamo.utils import counters

    model = tiny_key_model(lexicon, reader_vocab, train_fill_tower)
    model.freeze_for_key_warm_up()
    model.fill_pass = torch.compile(  # type: ignore[method-assign]
        model.fill_pass, dynamic=True, backend="eager"
    )
    model.key_front.forward = torch.compile(  # type: ignore[method-assign]
        model.key_front.forward, dynamic=True, backend="eager"
    )
    collator = key_collator(spans, lexicon)
    examples = sorted(build_examples(vocabularies), key=lambda example: len(example.keys))
    groups = [examples[:2], examples[-3:], examples[4:9], examples[::3]]
    widths = {collator(group).key_input_ids.shape for group in groups}
    assert len(widths) == len(groups)
    model.train()
    breaks = sum(counters["graph_breaks"].values())
    first = model(collator(groups[0])).loss
    assert first is not None
    first.backward()
    graphs = int(counters["stats"].get("unique_graphs") or 0)
    for group in groups[1:]:
        loss = model(collator(group)).loss
        assert loss is not None
        loss.backward()
    assert int(counters["stats"].get("unique_graphs") or 0) == graphs
    assert int(sum(counters["graph_breaks"].values())) == breaks


def test_compiling_the_key_model_keeps_its_names(
    vocabularies: Vocabularies, spans: SpanVocab, lexicon: Lexicon, reader_vocab: CharVocab
) -> None:
    from mlime.train.e2e import compile_modules

    model = tiny_key_model(lexicon, reader_vocab)
    names = list(model.state_dict())
    compile_modules(model, "default")
    assert list(model.state_dict()) == names
    model.freeze_for_key_warm_up()
    model.train()
    loss = model(key_collator(spans, lexicon)(build_examples(vocabularies)[:4])).loss
    assert loss is not None and torch.isfinite(loss)
    loss.backward()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sync detection needs CUDA")
def test_an_s1_step_issues_no_host_syncs(
    vocabularies: Vocabularies, spans: SpanVocab, lexicon: Lexicon, reader_vocab: CharVocab
) -> None:
    """set_sync_debug_mode("error") fails the test if forward or backward syncs."""
    model = tiny_key_model(lexicon, reader_vocab).cuda()
    model.freeze_for_key_warm_up()
    model.train()
    batch = key_collator(spans, lexicon, dropout=0.3)(build_examples(vocabularies)[:8]).to(
        torch.device("cuda")
    )
    warm = model(batch).loss
    assert warm is not None
    warm.backward()
    torch.cuda.set_sync_debug_mode("error")
    try:
        loss = model(batch).loss
        assert loss is not None
        loss.backward()
    finally:
        torch.cuda.set_sync_debug_mode("default")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="inductor's CUDA graphs need CUDA")
def test_the_compiled_s1_step_matches_the_eager_one_on_cuda(
    vocabularies: Vocabularies, spans: SpanVocab, lexicon: Lexicon, reader_vocab: CharVocab
) -> None:
    from mlime.train.e2e import compile_modules

    batch = key_collator(spans, lexicon)(build_examples(vocabularies)[:8]).to(torch.device("cuda"))
    losses = []
    for compiled in (False, True):
        model = tiny_key_model(lexicon, reader_vocab).cuda()
        if compiled:
            compile_modules(model, "default")
        model.freeze_for_key_warm_up()
        model.train()
        output = model(batch)
        assert output.loss is not None
        output.loss.backward()
        losses.append(output.loss.detach().cpu())
    torch.testing.assert_close(losses[1], losses[0], rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sync detection needs CUDA")
@pytest.mark.parametrize("train_fill_tower", TRAINED_TOWERS)
def test_a_compiled_s1_step_training_the_tower_issues_no_host_syncs(
    vocabularies: Vocabularies,
    spans: SpanVocab,
    lexicon: Lexicon,
    reader_vocab: CharVocab,
    train_fill_tower: FillTowerTraining,
) -> None:
    """Forward, backward and the loop's fused AdamW step, compiled as a run compiles them.

    The context gates are open, so the blocks' attention -- trained, or the
    frozen path the tower's gradient crosses -- does the work a parent's does.
    """
    from mlime.train.e2e import compile_modules

    model = tiny_key_model(lexicon, reader_vocab, train_fill_tower).cuda()
    open_gates(model)
    compile_modules(model, "default")
    model.freeze_for_key_warm_up()
    model.train()
    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-5, fused=True
    )
    batch = key_collator(spans, lexicon, dropout=0.3)(build_examples(vocabularies)[:8]).to(
        torch.device("cuda")
    )

    def step() -> None:
        optimiser.zero_grad()
        loss = model(batch).loss
        assert loss is not None
        loss.backward()
        optimiser.step()

    for _ in range(2):
        step()
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        step()
    finally:
        torch.cuda.set_sync_debug_mode("default")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="inductor's CUDA graphs need CUDA")
@pytest.mark.parametrize("train_fill_tower", TRAINED_TOWERS)
def test_the_compiled_s1_step_training_the_tower_matches_the_eager_one_on_cuda(
    vocabularies: Vocabularies,
    spans: SpanVocab,
    lexicon: Lexicon,
    reader_vocab: CharVocab,
    train_fill_tower: FillTowerTraining,
) -> None:
    """The loss and every trained gradient agree; the towers drop nothing, so both draw alike.

    The trained parts run in training mode, and eager and inductor draw
    their dropout masks from different generators, so the comparison is made
    on a tower and context blocks whose dropout is zero. The context gates
    are open: at a fresh block's zero gate the attention's gradient is
    exactly zero both ways, and its comparison would pass whatever the
    compiled blocks computed.
    """
    from mlime.train.e2e import compile_modules

    tower = copy.deepcopy(TINY)
    tower.hidden_dropout_prob = 0.0
    tower.attention_probs_dropout_prob = 0.0
    batch = key_collator(spans, lexicon)(build_examples(vocabularies)[:8]).to(torch.device("cuda"))
    assert bool(batch.spans.has_context.any())
    losses = []
    gradients = []
    for compiled in (False, True):
        model = tiny_key_model(
            lexicon, reader_vocab, train_fill_tower, tower=tower, cross_attention_dropout=0.0
        ).cuda()
        open_gates(model)
        if compiled:
            compile_modules(model, "default")
        model.freeze_for_key_warm_up()
        model.train()
        output = model(batch)
        assert output.loss is not None
        output.loss.backward()
        losses.append(output.loss.detach().cpu())
        gradients.append(
            {
                name: parameter.grad.detach().cpu()
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
        )
    torch.testing.assert_close(losses[1], losses[0], rtol=1e-3, atol=1e-3)
    assert gradients[1].keys() == gradients[0].keys()
    assert any(name.startswith("fill.encoder.") for name in gradients[0])
    crossing = {name for name in gradients[0] if name.startswith("cross_attention.")}
    if train_fill_tower.cross_attention:
        assert crossing == {
            f"cross_attention.0.{name}" for name, _ in model.cross_attention[0].named_parameters()
        }
        for name in crossing:
            assert bool(gradients[0][name].any()), name
    else:
        assert not crossing
    assert not any(name.startswith(("teacher_", "decoder.")) for name in gradients[0])
    for name, gradient in gradients[0].items():
        torch.testing.assert_close(gradients[1][name], gradient, rtol=1e-3, atol=1e-4, msg=name)
