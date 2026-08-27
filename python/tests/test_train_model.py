"""Route A's two load-bearing claims, checked on a tiny model on the CPU.

The claims are that the homophone mask really excludes non-homophones from the
loss, and that a zero gate really makes the context tower a no-op. Both are the
kind of thing that is *almost* true when implemented carelessly -- a mask that
lowers a logit instead of removing it, a gate that starts at 1e-3 -- and almost
true is indistinguishable from true until the numbers stop making sense months
later. So they are asserted as exact equalities on a model small enough to build
without a download.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from transformers import BertConfig

from mlime.data.corpus import Sample
from mlime.train.lexicon import Lexicon
from mlime.train.model import (
    RouteAConfig,
    RouteAModel,
    count_correct,
    letter_token_ids,
    restricted_cross_entropy,
)
from mlime.train.samples import IGNORE_INDEX, BaseTokenizer, Batch, Collator, SampleBuilder
from mlime.train.spans import SpanVocab

#: Small enough to build in a test, wide enough that the head is a real matmul.
TINY = BertConfig(
    vocab_size=256,
    hidden_size=32,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=64,
    max_position_embeddings=64,
)


@pytest.fixture(name="model")
def model_fixture(lexicon: Lexicon) -> RouteAModel:
    torch.manual_seed(0)
    model = RouteAModel.from_config(TINY, lexicon, RouteAConfig(cross_attention_layers=2))
    return model.eval()


@pytest.fixture(name="make_batch")
def make_batch_fixture(
    lexicon: Lexicon, spans: SpanVocab, tokenizer: BaseTokenizer
) -> Callable[..., Batch]:
    """A two-example batch, with or without context, collated the way training does."""

    def build(context: str | None = "北京大学") -> Batch:
        builder = SampleBuilder(lexicon, spans, seed=0)
        examples = [
            builder.build(
                Sample(id="a", source="test", text="我爱北京", context=context),
                ["wo3", "ai4", "bei3", "jing1"],
                0,
            ),
            builder.build(
                Sample(id="b", source="test", text="中重", context=context),
                ["zhong1", "chong2"],
                0,
            ),
        ]
        kept = [example for example in examples if example is not None]
        assert len(kept) == 2
        return Collator(tokenizer, context_dropout=0.0)(kept)

    return build


def test_forward_emits_one_distribution_per_position(
    model: RouteAModel, lexicon: Lexicon, make_batch: Callable[..., Batch]
) -> None:
    batch = make_batch()
    output = model(batch)
    assert output.logits.shape == (2, 6, lexicon.size)
    assert output.loss is not None
    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)


def test_a_batch_with_no_targets_has_no_loss(
    model: RouteAModel, make_batch: Callable[..., Batch]
) -> None:
    batch = make_batch()
    blanked = Batch(**{**vars(batch), "targets": torch.full_like(batch.targets, IGNORE_INDEX)})
    assert model(blanked).loss is None


def test_the_mask_removes_non_homophones_from_the_loss(
    model: RouteAModel, lexicon: Lexicon, spans: SpanVocab, make_batch: Callable[..., Batch]
) -> None:
    """Moving a ruled-out character's logit must not move the loss at all."""
    batch = make_batch()
    logits = model(batch).logits
    reference = model.loss(logits, batch)
    assert reference is not None

    zhong = spans.id("zhong")
    ruled_out = [
        index for index in range(lexicon.size) if not bool(lexicon.candidate_mask[zhong, index])
    ]
    assert ruled_out, "the fixture must have a character zhong cannot mean"
    perturbed = logits.clone()
    perturbed[:, :, ruled_out] += 100.0
    assert torch.equal(model.loss(perturbed, batch), reference)


def test_a_ruled_out_character_is_never_predicted(
    model: RouteAModel, lexicon: Lexicon, make_batch: Callable[..., Batch]
) -> None:
    batch = make_batch()
    logits = torch.zeros((2, 6, lexicon.size))
    forbidden = lexicon.index("我")
    logits[:, :, forbidden] = 50.0
    predictions = model.predictions(logits, batch)
    scored = batch.targets != IGNORE_INDEX
    for row, column in scored.nonzero().tolist():
        span_id = int(batch.span_ids[row, column])
        predicted = int(predictions[row, column])
        assert bool(lexicon.candidate_mask[span_id, predicted])


def test_restricted_cross_entropy_matches_the_hand_computation() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    candidates = torch.tensor([[True, True, False, False]])
    targets = torch.tensor([0])
    loss = restricted_cross_entropy(logits, targets, candidates, label_smoothing=0.0)
    expected = -torch.tensor([1.0, 2.0]).log_softmax(dim=-1)[0]
    assert loss == pytest.approx(float(expected), abs=1e-6)


def test_label_smoothing_spreads_over_candidates_only() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    candidates = torch.tensor([[True, True, False, False]])
    targets = torch.tensor([0])
    smoothed = restricted_cross_entropy(logits, targets, candidates, label_smoothing=0.05)
    log_probabilities = torch.tensor([1.0, 2.0]).log_softmax(dim=-1)
    expected = -(0.95 * log_probabilities[0] + 0.05 * log_probabilities.mean())
    assert smoothed == pytest.approx(float(expected), abs=1e-6)
    assert torch.isfinite(smoothed)


def test_a_position_admitting_nothing_is_an_error() -> None:
    with pytest.raises(ValueError, match="admits no character"):
        restricted_cross_entropy(
            torch.zeros((1, 4)),
            torch.tensor([0]),
            torch.zeros((1, 4), dtype=torch.bool),
            label_smoothing=0.0,
        )


def test_the_gates_start_shut(model: RouteAModel) -> None:
    assert model.gates() == [0.0, 0.0]


def test_a_shut_gate_makes_the_context_tower_a_no_op(
    model: RouteAModel, make_batch: Callable[..., Batch]
) -> None:
    """With the gates at zero, context and no context are the same numbers."""
    with_context = make_batch()
    without_context = make_batch(context=None)
    with torch.no_grad():
        assert torch.equal(model(with_context).logits, model(without_context).logits)


def test_an_open_gate_makes_the_context_matter(
    model: RouteAModel, make_batch: Callable[..., Batch]
) -> None:
    """The no-op has to be the gate's doing, not the context being ignored."""
    for layer in model.gated_layers():
        with torch.no_grad():
            layer.gate.fill_(1.0)
    with_context = make_batch()
    without_context = make_batch(context=None)
    with torch.no_grad():
        assert not torch.equal(model(with_context).logits, model(without_context).logits)


def test_the_gate_still_learns_from_a_zero_start(
    model: RouteAModel, make_batch: Callable[..., Batch]
) -> None:
    """A zero gate must have a gradient, or the context tower could never turn on."""
    batch = make_batch()
    output = model(batch)
    assert output.loss is not None
    output.loss.backward()
    gradients = [layer.gate.grad for layer in model.gated_layers()]
    assert all(gradient is not None and float(gradient.abs()) > 0.0 for gradient in gradients)


def test_the_new_parameters_are_the_ones_we_added(model: RouteAModel) -> None:
    base, new = model.parameter_groups(3e-5, 1e-4)
    assert base["lr"] == 3e-5
    assert new["lr"] == 1e-4
    added = sum(parameter.numel() for parameter in new["params"])
    expected = model.span_embeddings.weight.numel() + sum(
        parameter.numel() for layer in model.gated_layers() for parameter in layer.parameters()
    )
    assert added == expected


def test_span_embeddings_start_at_the_mean_of_their_letters(
    lexicon: Lexicon, spans: SpanVocab
) -> None:
    torch.manual_seed(1)
    model = RouteAModel.from_config(TINY, lexicon, RouteAConfig(cross_attention_layers=1))
    vocabulary = {letter: index + 10 for index, letter in enumerate("abcdefghijklmnopqrstuvwxyz")}
    letters = letter_token_ids(spans, vocabulary)
    model.initialise_span_embeddings(letters)
    words = model.fill.embeddings.word_embeddings.weight
    for span in ("a", "zh", "zhong"):
        ids = torch.tensor([vocabulary[letter] for letter in span])
        expected = words.index_select(0, ids).mean(dim=0)
        assert torch.allclose(model.span_embeddings.weight[spans.id(span)], expected)


def test_anagram_spans_are_separate_parameters(lexicon: Lexicon, spans: SpanVocab) -> None:
    """`na` and `an` start equal, which is only acceptable because they can diverge."""
    model = RouteAModel.from_config(TINY, lexicon, RouteAConfig(cross_attention_layers=1))
    vocabulary = {letter: index + 10 for index, letter in enumerate("abcdefghijklmnopqrstuvwxyz")}
    model.initialise_span_embeddings(letter_token_ids(spans, vocabulary))
    table = model.span_embeddings.weight
    assert torch.allclose(table[spans.id("na")], table[spans.id("an")])
    with torch.no_grad():
        table[spans.id("na")] += 1.0
    assert not torch.allclose(table[spans.id("na")], table[spans.id("an")])


def test_a_letter_outside_the_vocabulary_is_an_error(spans: SpanVocab) -> None:
    with pytest.raises(KeyError, match="no token for the letter"):
        letter_token_ids(spans, {"a": 1})


def test_counting_ignores_unscored_positions() -> None:
    predictions = torch.tensor([[1, 2, IGNORE_INDEX]])
    targets = torch.tensor([[1, 3, IGNORE_INDEX]])
    assert count_correct(predictions, targets) == (1, 2)


def test_more_gated_layers_than_layers_is_refused(lexicon: Lexicon) -> None:
    with pytest.raises(ValueError, match="cannot gate"):
        RouteAModel.from_config(TINY, lexicon, RouteAConfig(cross_attention_layers=99))
