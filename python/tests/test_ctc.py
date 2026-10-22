"""The constrained CTC against a brute force over every legal alignment (#216 §2, §7).

The batched recursion in :func:`~mlime.train.ctc.constrained_ctc` is checked
against the definition it implements: enumerate every cut of the keys into as
many contiguous spans as the sentence has characters, keep the cuts whose
every span the arc table allows, score each as its blanks, its emissions and
its correction costs, and log-sum-exp them. Values and gradients both, on
random tables with holes and padding, and on the legality automaton's own
arcs for sentences typed in full, abbreviated, mixed and with a slip -- there
the span costs come from :meth:`~mlime.legality.Legality.char_cost` directly,
not from the arcs the loss reads, so the two sides share no code.
"""

from __future__ import annotations

import itertools
import math
import random
from pathlib import Path

import pytest
import torch

from mlime.legality import MAX_ARC_KEYS, OFFLINE, Legality, LegalityOptions
from mlime.train.ctc import alignment_components, alignment_scores, constrained_ctc

ROOT = Path(__file__).parent.parent.parent
DATA = ROOT / "crates/ime-pinyin/data"

#: Offline, with corrections kept off spans that are syllables.
STRICT = LegalityOptions(unfinished_tail=False, correct_syllables=False)


@pytest.fixture(scope="module")
def legality() -> Legality:
    return Legality.load(
        None,
        char_table=DATA / "char_pinyin.tsv",
        typo_table=DATA / "typo.json",
        syllables=DATA / "syllables.txt",
    )


def cuts(keys: int, length: int) -> list[tuple[int, ...]]:
    """Every way to end *length* contiguous spans covering *keys* keys: the end keys."""
    return [(*inner, keys - 1) for inner in itertools.combinations(range(keys - 1), length - 1)]


def brute_force(
    blank: torch.Tensor,
    emit: torch.Tensor,
    span_cost: dict[tuple[int, int, int], float],
    keys: int,
    length: int,
) -> torch.Tensor:
    """``-log`` of the summed probability of every alignment *span_cost* allows.

    *span_cost* maps (position, first key, last key) to the arc's correction
    cost; a span it does not name has no arc.
    """
    scores = []
    for ends in cuts(keys, length):
        starts = (0, *(end + 1 for end in ends[:-1]))
        if any(
            (position, start, end) not in span_cost
            for position, (start, end) in enumerate(zip(starts, ends, strict=True))
        ):
            continue
        score = blank[:keys].sum()
        for position, (start, end) in enumerate(zip(starts, ends, strict=True)):
            score = score - blank[end] + emit[position, end] - span_cost[(position, start, end)]
        scores.append(score)
    assert scores, "the brute force was handed a sentence with no legal alignment"
    return -torch.logsumexp(torch.stack(scores), dim=0)


def dense_cost(
    span_cost: dict[tuple[int, int, int], float], length: int, keys: int, width: int
) -> torch.Tensor:
    """The arc table :func:`constrained_ctc` reads, ``[length, keys, width]``."""
    table = torch.full((length, keys, width), math.inf, dtype=torch.float64)
    for (position, start, end), cost in span_cost.items():
        table[position, end, end - start] = cost
    return table


def random_row(
    rng: random.Random, keys: int, length: int
) -> tuple[torch.Tensor, torch.Tensor, dict[tuple[int, int, int], float]]:
    """Log-probabilities over keys, and a random arc table with holes and a legal path."""
    logits = torch.randn(keys, 5, dtype=torch.float64)
    log_probs = logits.log_softmax(-1)
    blank = log_probs[:, -1].clone()
    emit = torch.stack([log_probs[:, rng.randrange(4)] for _ in range(length)])
    while True:
        span_cost = {
            (position, start, end): rng.choice([0.0, 0.0, rng.uniform(0.1, 3.0)])
            for position in range(length)
            for end in range(keys)
            for start in range(max(0, end - MAX_ARC_KEYS + 1), end + 1)
            if rng.random() < 0.6
        }
        legal = any(
            all(
                (position, start, end) in span_cost
                for position, (start, end) in enumerate(
                    zip((0, *(end + 1 for end in ends[:-1])), ends, strict=True)
                )
            )
            for ends in cuts(keys, length)
        )
        if legal:
            return blank, emit, span_cost


def test_the_loss_is_the_log_sum_over_every_legal_alignment_with_its_gradients() -> None:
    rng = random.Random(7)
    torch.manual_seed(7)
    shapes = [(1, 1), (3, 1), (4, 2), (6, 3), (9, 2), (10, 4), (11, 5), (12, 3)]
    rows = [random_row(rng, keys, length) for keys, length in shapes]
    width = max(keys for keys, _ in shapes)
    positions = max(length for _, length in shapes)

    # The batch: every row padded to the widest, padding filled with noise the
    # arc table rules out, so a leak shows as a wrong value.
    blank = torch.randn(len(rows), width, dtype=torch.float64)
    emit = torch.randn(len(rows), positions, width, dtype=torch.float64)
    cost = torch.full((len(rows), positions, width, MAX_ARC_KEYS), math.inf, dtype=torch.float64)
    for row, ((keys, length), (own_blank, own_emit, span_cost)) in enumerate(
        zip(shapes, rows, strict=True)
    ):
        blank[row, :keys] = own_blank
        emit[row, :length, :keys] = own_emit
        cost[row, :length, :keys] = dense_cost(span_cost, length, keys, MAX_ARC_KEYS)
    blank.requires_grad_(True)
    emit.requires_grad_(True)
    keys_tensor = torch.tensor([keys for keys, _ in shapes])
    lengths = torch.tensor([length for _, length in shapes])
    nll = constrained_ctc(blank, emit, cost, keys_tensor, lengths)
    nll.sum().backward()
    assert blank.grad is not None and emit.grad is not None

    for row, ((keys, length), (_, _, span_cost)) in enumerate(zip(shapes, rows, strict=True)):
        own_blank = blank.detach()[row].clone().requires_grad_(True)
        own_emit = emit.detach()[row].clone().requires_grad_(True)
        expected = brute_force(own_blank, own_emit, span_cost, keys, length)
        expected.backward()
        assert own_blank.grad is not None and own_emit.grad is not None
        torch.testing.assert_close(nll[row].detach(), expected.detach(), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(blank.grad[row], own_blank.grad, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(emit.grad[row], own_emit.grad, rtol=1e-10, atol=1e-12)
        # Nothing reaches the padding.
        assert not blank.grad[row, keys:].any()
        assert not emit.grad[row, length:].any()
        assert not emit.grad[row, :, keys:].any()


def test_a_single_alignment_is_its_own_score_and_more_alignments_only_add_mass() -> None:
    log_probs = torch.randn(4, 3, dtype=torch.float64).log_softmax(-1)
    blank, emit = log_probs[:, -1], log_probs[:, :2].T.contiguous()
    only = {(0, 0, 1): 0.0, (1, 2, 3): 0.5}
    nll = constrained_ctc(
        blank[None],
        emit[None],
        dense_cost(only, 2, 4, MAX_ARC_KEYS)[None],
        torch.tensor([4]),
        torch.tensor([2]),
    )
    expected = -(blank[0] + emit[0, 1] + blank[2] + emit[1, 3] - 0.5)
    torch.testing.assert_close(nll[0], expected, rtol=1e-12, atol=1e-12)
    wider = {**only, (0, 0, 0): 0.0, (1, 1, 3): 0.0}
    more = constrained_ctc(
        blank[None],
        emit[None],
        dense_cost(wider, 2, 4, MAX_ARC_KEYS)[None],
        torch.tensor([4]),
        torch.tensor([2]),
    )
    assert float(more[0]) < float(nll[0])


def test_float32_agrees_with_float64() -> None:
    rng = random.Random(3)
    blank, emit, span_cost = random_row(rng, 12, 4)
    table = dense_cost(span_cost, 4, 12, MAX_ARC_KEYS)
    wide = constrained_ctc(
        blank[None], emit[None], table[None], torch.tensor([12]), torch.tensor([4])
    )
    narrow = constrained_ctc(
        blank[None].float(),
        emit[None].float(),
        table[None].float(),
        torch.tensor([12]),
        torch.tensor([4]),
    )
    torch.testing.assert_close(narrow.double(), wide, rtol=1e-5, atol=1e-5)


def legality_span_costs(
    legality: Legality, keys: str, sentence: str, options: LegalityOptions
) -> dict[tuple[int, int, int], float]:
    """Every (position, first key, last key) the automaton's rules price, read span by span."""
    ids = legality.ids(sentence)
    assert ids is not None
    costs: dict[tuple[int, int, int], float] = {}
    for position, emit in enumerate(ids):
        for start in range(len(keys)):
            for end in range(start, min(start + MAX_ARC_KEYS, len(keys))):
                cost = legality.char_cost(
                    keys[start : end + 1], end == len(keys) - 1, emit, options
                )
                if math.isfinite(cost):
                    costs[(position, start, end)] = cost
    return costs


@pytest.mark.parametrize(
    ("keys", "sentence", "options"),
    [
        ("xianzai", "现在", OFFLINE),
        ("beijingdaxue", "北京大学", OFFLINE),
        ("bjdx", "北京大学", OFFLINE),
        ("beijdaxue", "北京大学", OFFLINE),
        ("shuijiap", "水饺", OFFLINE),
        ("shuijiap", "水饺", STRICT),
        ("zhongguoren", "中国人", OFFLINE),
        ("wmchishuijiao", "我们吃水饺", STRICT),
    ],
)
def test_the_automatons_arcs_carry_exactly_the_brute_force_mass(
    legality: Legality, keys: str, sentence: str, options: LegalityOptions
) -> None:
    gold = legality.gold(keys, sentence, options)
    assert gold is not None
    length, width = len(sentence), len(keys)
    torch.manual_seed(len(keys))
    log_probs = torch.randn(width, length + 1, dtype=torch.float64).log_softmax(-1)
    blank = log_probs[:, -1].clone().requires_grad_(True)
    emit = log_probs[:, :length].T.contiguous().requires_grad_(True)
    nll = constrained_ctc(
        blank[None],
        emit[None],
        torch.from_numpy(gold.dense())[None],
        torch.tensor([width]),
        torch.tensor([length]),
    )
    nll.sum().backward()
    assert blank.grad is not None and emit.grad is not None
    got = (nll[0].detach(), blank.grad.clone(), emit.grad.clone())

    blank.grad = None
    emit.grad = None
    expected = brute_force(
        blank, emit, legality_span_costs(legality, keys, sentence, options), width, length
    )
    expected.backward()
    torch.testing.assert_close(got[0], expected.detach(), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(got[1], blank.grad, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(got[2], emit.grad, rtol=1e-10, atol=1e-12)


def test_the_gathers_read_each_rows_keys_and_gold() -> None:
    log_probs = torch.randn(7, 6).log_softmax(-1)
    key_index = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 4]])
    gold = torch.tensor([[2, 0], [4, 1]])
    blank, emit = alignment_scores(log_probs, key_index, gold)
    assert torch.equal(blank, log_probs[key_index, -1])
    for row in range(2):
        for position in range(2):
            for key in range(4):
                assert (
                    emit[row, position, key] == log_probs[key_index[row, key], gold[row, position]]
                )


def test_the_components_read_the_typed_cover() -> None:
    blank = torch.log(torch.tensor([[0.9, 0.2, 0.7, 0.4], [0.5, 0.25, 0.5, 0.5]]))
    emit = torch.log(torch.full((2, 2, 4), 0.05))
    emit[0, 0, 1] = math.log(0.6)
    emit[0, 1, 3] = math.log(0.3)
    emit[1, 0, 1] = math.log(0.375)
    span_end = torch.tensor([[1, 3], [1, 0]])
    ends = torch.tensor([[False, True, False, True], [False, True, False, False]])
    key_mask = torch.tensor([[True] * 4, [True, True, False, False]])
    position_mask = torch.tensor([[True, True], [True, False]])
    corrupted = torch.tensor([[False, True], [False, False]])
    parts = alignment_components(blank, emit, span_end, ends, key_mask, position_mask, corrupted)
    decisions = [0.9, 0.8, 0.7, 0.6, 0.5, 0.75]
    segmentation = -sum(math.log(p) for p in decisions) / 6
    torch.testing.assert_close(parts["segmentation_loss"], torch.tensor(segmentation))
    # -log p(c | emit) = -log(p(c) / (1 - p(blank))).
    clean = (-math.log(0.6 / 0.8) - math.log(0.375 / 0.75)) / 2
    torch.testing.assert_close(parts["clean_span_loss"], torch.tensor(clean))
    torch.testing.assert_close(parts["typo_span_loss"], torch.tensor(-math.log(0.3 / 0.6)))
    torch.testing.assert_close(parts["typo_share"], torch.tensor(1 / 3))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the CUDA forward needs a CUDA device")
def test_the_cuda_loss_and_gradients_match_the_cpus() -> None:
    rng = random.Random(11)
    shapes = [(5, 2), (12, 4), (9, 3), (16, 6)]
    rows = [random_row(rng, keys, length) for keys, length in shapes]
    width = max(keys for keys, _ in shapes)
    positions = max(length for _, length in shapes)
    blank = torch.randn(len(rows), width)
    emit = torch.randn(len(rows), positions, width)
    cost = torch.full((len(rows), positions, width, MAX_ARC_KEYS), math.inf)
    for row, ((keys, length), (own_blank, own_emit, span_cost)) in enumerate(
        zip(shapes, rows, strict=True)
    ):
        blank[row, :keys] = own_blank.float()
        emit[row, :length, :keys] = own_emit.float()
        cost[row, :length, :keys] = dense_cost(span_cost, length, keys, MAX_ARC_KEYS).float()
    keys_tensor = torch.tensor([keys for keys, _ in shapes])
    lengths = torch.tensor([length for _, length in shapes])
    results = []
    for device in ("cpu", "cuda"):
        own_blank = blank.to(device).requires_grad_(True)
        own_emit = emit.to(device).requires_grad_(True)
        nll = constrained_ctc(
            own_blank,
            own_emit,
            cost.to(device),
            keys_tensor.to(device),
            lengths.to(device),
        )
        nll.sum().backward()
        assert own_blank.grad is not None and own_emit.grad is not None
        results.append((nll.detach().cpu(), own_blank.grad.cpu(), own_emit.grad.cpu()))
    for cpu, cuda in zip(results[0], results[1], strict=True):
        torch.testing.assert_close(cuda, cpu, rtol=1e-4, atol=1e-5)
