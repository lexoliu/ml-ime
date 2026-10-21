"""The sequence-level ranking loss (#116): its batch, its search, its score, its loss.

The loss is only worth anything if the number it trains is the number the
evaluation ranks by, so the central test here is that a teacher-forced path
score equals the beam's own score for the same sentence. Around it: the batch
leads every example with its gold reading and runs one context per example,
the ragged search is the per-lane search, the item selection merges readings
as the evaluation merges them, and a few steps of the loss move the gold
sentence ahead of its negatives.
"""

from __future__ import annotations

import contextlib
import json
import random
import stat
import sys
from collections import Counter
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pytest
import torch
from conftest import BuiltLane, StubTokenizer
from test_e2e_model import READER_TINY, TINY, example

from mlime.train.arbitration import ReadingArbitration
from mlime.train.charlm_vocab import BOS, SPECIALS, CharVocab
from mlime.train.e2e import from_checkpoint
from mlime.train.e2e_model import E2EConfig, E2EModel, RankingConfig, emittable_ids
from mlime.train.e2e_search import (
    Lane,
    SearchGraphs,
    SearchInputs,
    bucket,
    decode_lanes,
    run_search,
    search_inputs,
    walk_back,
)
from mlime.train.e2e_search import hypotheses as read_back
from mlime.train.lexicon import CandidateSpace, Lexicon
from mlime.train.loop import TrainingConfig, train
from mlime.train.model import restricted_cross_entropy
from mlime.train.ranking import (
    STEP_CHUNK,
    Items,
    RankBatch,
    RankCollator,
    RankingE2EModel,
    _step_log_probabilities,
    select_items,
)
from mlime.train.samples import (
    IGNORE_INDEX,
    Collator,
    CorpusStream,
    SampleBuilder,
    TrainingExample,
)
from mlime.train.segmenter import Segmentations, Segmenter
from mlime.train.spans import SpanVocab

RANKING = RankingConfig(segmentations=8, beam=4, negatives=4, weight=1.0)

#: Two sentences and the readings a segmenter would give their keystrokes.
READINGS = {
    "woaibeijing": [("wo", "ai", "bei", "jing"), ("wo", "ai", "bei", "ji", "n", "g")],
    "zhongaibeijing": [("zhong", "ai", "bei", "jin", "g"), ("zhong", "ai", "bei", "jing")],
}


def ranking_model(lexicon: Lexicon) -> RankingE2EModel:
    """A tiny ranking model over the fixture lexicon, seeded."""
    torch.manual_seed(0)
    vocab = CharVocab(chars=SPECIALS + tuple(lexicon.characters))
    config = E2EConfig(cross_attention_layers=1, decoder=READER_TINY, ranking=RANKING)
    model = RankingE2EModel.compose(
        TINY, lexicon, config, len(vocab), emittable_ids(lexicon, vocab)
    )
    assert isinstance(model, RankingE2EModel)
    return model


#: The same two sentences with a second reading of the first one's length.
EVEN_READINGS = {
    "woaibeijing": [("wo", "ai", "bei", "jing"), ("wo", "ai", "bei", "jin")],
    "zhongaibeijing": [("zhong", "ai", "bei", "jin", "g"), ("zhong", "ai", "bei", "jing")],
}


def rank_batch(
    lexicon: Lexicon,
    spans: SpanVocab,
    tokenizer: StubTokenizer,
    readings: dict[str, list[tuple[str, ...]]] = READINGS,
) -> RankBatch:
    """The two sentences, typed out, with the first one's context kept."""
    collator = Collator(
        tokenizer, CandidateSpace(spans, lexicon.candidate_mask), context_dropout=0.0
    )
    first = example(spans, lexicon, "我爱北京", ("wo", "ai", "bei", "jing"))
    first = type(first)(**{**vars(first), "context": "北京"})
    second = example(spans, lexicon, "钟爱北京", ("zhong", "ai", "bei", "jing"))
    return RankCollator(collator, lambda keys: readings[keys], spans, RANKING)([first, second])


def test_an_example_costs_its_readings_and_the_paths_it_ranks(
    lexicon: Lexicon, spans: SpanVocab, tokenizer: StubTokenizer
) -> None:
    collator = Collator(tokenizer, CandidateSpace(spans, lexicon.candidate_mask))
    typed = example(spans, lexicon, "我爱北京", ("wo", "ai", "bei", "jing"))
    two = RankCollator(collator, lambda keys: READINGS[keys], spans, RANKING)
    # Two readings, then the gold path, the gold sentence on the other
    # reading and four negatives.
    assert two.rows(typed) == 2 + 2 + 4
    one = RankCollator(collator, lambda keys: [typed.spans], spans, RANKING)
    assert one.rows(typed) == 1 + 2 + 4
    narrow = replace(RANKING, beam=1, negatives=8)
    # One reading under a beam of one finds one sentence at most.
    assert (
        RankCollator(collator, lambda keys: [typed.spans], spans, narrow).rows(typed) == 1 + 2 + 1
    )


def test_the_batch_leads_with_gold_and_runs_one_context_per_example(
    lexicon: Lexicon, spans: SpanVocab, tokenizer: StubTokenizer
) -> None:
    batch = rank_batch(lexicon, spans, tokenizer)
    assert batch.owners == (0, 0, 1, 1)
    assert batch.gold_rows == (0, 2)
    assert batch.lengths == (4, 6, 4, 5)
    assert batch.size == 2
    rows = batch.rows
    scored_rows = {int(row) for row in (rows.targets != IGNORE_INDEX).nonzero()[:, 0]}
    assert scored_rows == {0, 2}
    assert rows.context_rows.tolist() == [0, 0, 1, 1]
    assert rows.context_ids.shape[0] == 2
    assert rows.has_context.tolist() == [1.0, 0.0]
    assert torch.equal(batch.scored, rows.scored)


def test_items_merge_readings_and_keep_the_gold_sentence_positive(
    lexicon: Lexicon, spans: SpanVocab, tokenizer: StubTokenizer
) -> None:
    batch = rank_batch(lexicon, spans, tokenizer, EVEN_READINGS)
    assert batch.lengths == (4, 4, 4, 5)
    gold, other = gold_sentences(batch)
    a, b = (7, 7, 7, 7), (8, 8, 8, 8)
    found = [
        [(list(gold), -1.0), (list(a), -0.5)],
        [(list(a), -0.7), (list(gold), -0.2), (list(b), -2.0)],
        [(list(other), -0.3)],
        [],
    ]
    sequences, scores = beams(found, batch.lengths, width=4)
    # Example 0: its gold path, the gold sentence again from the reading that
    # scored it best, then the negatives best first, each from its best reading.
    assert listed(select_items(batch, sequences, scores, negatives=8)) == [
        [(0, gold, True), (1, gold, True), (0, a, False), (1, b, False)],
        [(2, other, True)],
    ]
    assert listed(select_items(batch, sequences, scores, negatives=1)) == [
        [(0, gold, True), (1, gold, True), (0, a, False)],
        [(2, other, True)],
    ]


def gold_sentences(batch: RankBatch) -> list[tuple[int, ...]]:
    """Each example's gold sentence, read off its gold row's targets."""
    targets = batch.rows.targets
    return [tuple(targets[row, 1 : batch.lengths[row] + 1].tolist()) for row in batch.gold_rows]


Found = list[list[tuple[list[int], float]]]


def beams(
    found: Found, lengths: tuple[int, ...], width: int, lanes: int = 0, extra: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """A search's output holding *found*: ``[L, W, T]`` sentences and ``[L, W]`` scores.

    Unfilled slots score minus infinity; every position past a row's length,
    every slot's sentence and every lane past the rows (*lanes* of them, a
    bucket's padding) hold noise the selection must not read.
    """
    noise = torch.Generator().manual_seed(len(found) * 31 + width)
    count = len(found) + lanes
    span = max(lengths) + extra
    sequences = torch.randint(0, 9, (count, width, span), generator=noise)
    scores = torch.randn((count, width), generator=noise)
    scores[: len(found)] = float("-inf")
    for row, hypotheses in enumerate(found):
        for beam, (sequence, score) in enumerate(hypotheses):
            sequences[row, beam, : len(sequence)] = torch.tensor(sequence)
            scores[row, beam] = score
    return sequences, scores


def listed(items: Items) -> list[list[tuple[int, tuple[int, ...], bool]]]:
    """Each example's ranked paths as (row, sentence, positive), in slot order.

    Every listed slot sits before every unlisted one, as the loss's compact
    order has it.
    """
    rows = items.rows.tolist()
    sequences = items.sequences.tolist()
    lengths = items.steps.sum(1).tolist()
    out = []
    for slots, marks in zip(items.slots.tolist(), items.positive.tolist(), strict=True):
        used = [slot for slot in slots if slot >= 0]
        assert slots == used + [-1] * (len(slots) - len(used))
        assert not any(marks[len(used) :])
        out.append(
            [
                (rows[slot], tuple(sequences[slot][: lengths[slot]]), mark)
                for slot, mark in zip(used, marks, strict=False)
            ]
        )
    return out


# ---------------------------------------------------------------------------
# The host-side selection the device one replaces, kept here as the reference
# it must agree with: #116's `select_items` and `walk_back`, verbatim but for
# taking the batch's fields as arguments.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceItems:
    rows: list[int]
    sequences: list[tuple[int, ...]]
    slots: list[list[int]]
    positive: list[list[bool]]


def reference_select(
    owners: tuple[int, ...],
    gold_rows: tuple[int, ...],
    gold: list[tuple[int, ...]],
    found: Found,
    negatives: int,
) -> ReferenceItems:
    best: list[dict[tuple[int, ...], tuple[float, int]]] = [{} for _ in gold]
    for row, hypotheses in enumerate(found):
        merged = best[owners[row]]
        for sequence, score in hypotheses:
            key = tuple(sequence)
            if score > merged.get(key, (float("-inf"), -1))[0]:
                merged[key] = (score, row)

    rows = list(gold_rows)
    sequences = list(gold)
    slots: list[list[int]] = [[example] for example in range(len(gold))]
    positive: list[list[bool]] = [[True] for _ in gold]
    for index, merged in enumerate(best):
        target = gold[index]
        if target in merged and merged[target][1] != gold_rows[index]:
            slots[index].append(len(rows))
            positive[index].append(True)
            rows.append(merged[target][1])
            sequences.append(target)
        ranked = sorted(
            (
                (score, row, sequence)
                for sequence, (score, row) in merged.items()
                if sequence != target
            ),
            reverse=True,
        )
        for _, row, sequence in ranked[:negatives]:
            slots[index].append(len(rows))
            positive[index].append(False)
            rows.append(row)
            sequences.append(sequence)
    width = max(len(listed) for listed in slots)
    return ReferenceItems(
        rows=rows,
        sequences=sequences,
        slots=[listed + [-1] * (width - len(listed)) for listed in slots],
        positive=[marks + [False] * (width - len(marks)) for marks in positive],
    )


def reference_listed(items: ReferenceItems) -> list[list[tuple[int, tuple[int, ...], bool]]]:
    """:func:`listed` for the reference's items."""
    return [
        [
            (items.rows[slot], items.sequences[slot], mark)
            for slot, mark in zip(slots, marks, strict=True)
            if slot >= 0
        ]
        for slots, marks in zip(items.slots, items.positive, strict=True)
    ]


def reference_walk_back(
    parents: torch.Tensor, choices: torch.Tensor, scores: torch.Tensor, lengths: list[int]
) -> Found:
    parent_steps: list[list[list[int]]] = parents.tolist()
    choice_steps: list[list[list[int]]] = choices.tolist()
    final: list[list[float]] = scores.tolist()
    length = len(parent_steps)
    decoded: Found = []
    for lane, own in enumerate(lengths):
        hypotheses = []
        for beam, score in enumerate(final[lane]):
            if score == float("-inf"):
                continue
            sequence = []
            row = beam
            for position in range(length - 1, -1, -1):
                sequence.append(choice_steps[position][lane][row])
                row = parent_steps[position][lane][row]
            sequence.reverse()
            hypotheses.append((sequence[:own], score))
        decoded.append(hypotheses)
    return decoded


def test_the_walk_back_on_the_device_is_the_host_walk_back() -> None:
    noise = torch.Generator().manual_seed(5)
    for _ in range(50):
        length, lanes, width = (int(n) for n in torch.randint(1, 9, (3,), generator=noise))
        parents = torch.randint(0, width, (length, lanes, width), generator=noise)
        choices = torch.randint(0, 50, (length, lanes, width), generator=noise)
        scores = torch.randn((lanes, width), generator=noise)
        scores[torch.rand((lanes, width), generator=noise) < 0.3] = float("-inf")
        own = torch.randint(1, length + 1, (lanes,), generator=noise).tolist()
        assert read_back(walk_back(parents, choices), scores, own) == reference_walk_back(
            parents, choices, scores, own
        )


#: The three characters ``zhong`` admits, and a fourth that no gold says.
def alphabet(lexicon: Lexicon) -> list[int]:
    return [lexicon.index(character) for character in "中钟重"] + [lexicon.index("我")]


def random_case(
    rng: random.Random, lexicon: Lexicon, spans: SpanVocab, tokenizer: StubTokenizer
) -> tuple[RankBatch, Found, int, int]:
    """A random ranking batch and beams built to collide: a batch, its hypotheses, beam, negatives.

    Readings are often the gold length, so a sentence turns up on several of
    an example's rows, the gold one among them; scores come from a short list
    with minus infinity in it, so ties and unfilled slots are common, and a
    beam sometimes repeats a sentence of its own lane.
    """
    letters = alphabet(lexicon)
    width = rng.choice([1, 2, 4])
    negatives = rng.choice([1, 2, 3, 5, 8])
    examples = []
    plan = []
    readings_spans = ["wo", "ai", "bei", "jing", "lv"]
    for index in range(rng.randint(1, 4)):
        size = rng.randint(1, 4)
        examples.append(
            TrainingExample(
                id=f"case-{index}",
                spans=("zhong",) * size,
                span_ids=(spans.id("zhong"),) * size,
                targets=tuple(rng.choice(letters[:3]) for _ in range(size)),
                context=None,
            )
        )
        plan.append(
            [
                (readings_spans[reading],) * rng.choice([size, size, rng.randint(1, 5)])
                for reading in range(rng.randint(0, 4))
            ]
        )
    readings = iter(plan)
    collator = Collator(
        tokenizer, CandidateSpace(spans, lexicon.candidate_mask), context_dropout=0.0
    )
    ranking = replace(RANKING, beam=width, negatives=negatives)
    batch = RankCollator(collator, lambda _: next(readings), spans, ranking)(examples)
    gold = gold_sentences(batch)
    found: Found = []
    for row, length in enumerate(batch.lengths):
        owner = batch.owners[row]
        hypotheses: list[tuple[list[int], float]] = []
        for _ in range(width):
            draw = rng.random()
            same_length = [
                sequence
                for other, hypotheses_there in enumerate(found)
                if batch.owners[other] == owner and batch.lengths[other] == length
                for sequence, _ in hypotheses_there
            ]
            if draw < 0.3 and len(gold[owner]) == length:
                sequence = list(gold[owner])
            elif draw < 0.6 and same_length:
                sequence = list(rng.choice(same_length))
            elif draw < 0.7 and hypotheses:
                sequence = list(rng.choice(hypotheses)[0])
            else:
                sequence = [rng.choice(letters) for _ in range(length)]
            score = rng.choice([-0.5, -1.0, -1.0, -2.0, float("-inf"), rng.uniform(-3, 0)])
            hypotheses.append((sequence, score))
        found.append(hypotheses)
    return batch, found, width, negatives


def test_the_device_selection_is_the_host_selection(
    lexicon: Lexicon, spans: SpanVocab, tokenizer: StubTokenizer
) -> None:
    """Random colliding beams pick the same paths in the same order as #116's selection.

    The cases are drawn to collide, and the test counts that they did: a
    sentence on two readings, the gold sentence found on another reading or
    not found at all, fewer candidates than negatives, a tie at the cut.
    """
    rng = random.Random(7)
    seen: Counter[str] = Counter()
    for _ in range(400):
        batch, found, width, negatives = random_case(rng, lexicon, spans, tokenizer)
        gold = gold_sentences(batch)
        sequences, scores = beams(
            [[(sequence, score) for sequence, score in row] for row in found],
            batch.lengths,
            width,
            lanes=rng.randint(0, 3),
            extra=rng.randint(0, 2),
        )
        # Hypotheses as a search reports them: unfilled slots dropped.
        reported = [[(s, v) for s, v in row if v != float("-inf")] for row in found]
        expected = reference_select(batch.owners, batch.gold_rows, gold, reported, negatives)
        assert listed(select_items(batch, sequences, scores, negatives)) == reference_listed(
            expected
        )

        for index, gold_row in enumerate(batch.gold_rows):
            rows = [row for row, owner in enumerate(batch.owners) if owner == index]
            said: dict[tuple[int, ...], set[int]] = {}
            for row in rows:
                for sequence, _ in reported[row]:
                    said.setdefault(tuple(sequence), set()).add(row)
            seen["sentence on two readings"] += any(len(on) > 1 for on in said.values())
            on_gold = said.get(gold[index], set())
            seen["gold absent"] += not on_gold
            seen["gold on another reading"] += any(
                (row, gold[index], True) in reference_listed(expected)[index]
                for row in on_gold - {gold_row}
            )
            candidates = [sentence for sentence in said if sentence != gold[index]]
            seen["fewer candidates than negatives"] += len(candidates) < negatives
            best = sorted(
                (
                    max(
                        v
                        for row in said[sentence]
                        for s, v in reported[row]
                        if tuple(s) == sentence
                    )
                    for sentence in candidates
                ),
                reverse=True,
            )
            seen["tie at the cut"] += (
                len(best) > negatives and best[negatives - 1] == best[negatives]
            )
            seen["unfilled slot"] += any(v == float("-inf") for _, v in found[gold_row])
    assert all(seen[case] >= 20 for case in CASES), seen


CASES = (
    "sentence on two readings",
    "gold absent",
    "gold on another reading",
    "fewer candidates than negatives",
    "tie at the cut",
    "unfilled slot",
)


def admitted_lane(
    lexicon: Lexicon, spans: SpanVocab, owner: int, reading: tuple[str, ...], width: int
) -> Lane:
    """A lane over *reading* with random encoder vectors and the lexicon's masks."""
    mask = torch.zeros(len(reading), lexicon.size, dtype=torch.bool)
    for position, span in enumerate(reading):
        mask[position] = lexicon.candidate_mask[spans.id(span)]
    return Lane(owner, torch.randn(len(reading), width), mask)


def test_ragged_lanes_search_as_each_lane_alone(lexicon: Lexicon, spans: SpanVocab) -> None:
    model = ranking_model(lexicon).eval()
    torch.manual_seed(1)
    lanes = [
        admitted_lane(lexicon, spans, 0, ("wo", "ai", "bei", "jing"), TINY.hidden_size),
        admitted_lane(lexicon, spans, 1, ("zhong", "ai"), TINY.hidden_size),
        admitted_lane(lexicon, spans, 2, ("bei", "jing", "wo", "ai", "zhong"), TINY.hidden_size),
    ]
    with torch.no_grad():
        together = decode_lanes(model, lanes, 4)
        alone = [decode_lanes(model, [lane], 4)[0] for lane in lanes]
    for joint, single in zip(together, alone, strict=True):
        assert [sequence for sequence, _ in joint] == [sequence for sequence, _ in single]
        assert [score for _, score in joint] == pytest.approx(
            [score for _, score in single], abs=1e-4
        )


def test_buckets_pad_at_most_a_quarter() -> None:
    assert [bucket(n) for n in (1, 8, 9, 11, 16, 17, 33, 100, 384)] == [
        8,
        8,
        10,
        12,
        16,
        20,
        40,
        112,
        384,
    ]
    assert all(n <= bucket(n) <= max(8, n * 1.25) for n in range(1, 2000))


def search_lanes(lexicon: Lexicon, spans: SpanVocab) -> list[Lane]:
    """Three ragged lanes, none of a bucket's size."""
    torch.manual_seed(1)
    return [
        admitted_lane(lexicon, spans, 0, ("wo", "ai", "bei", "jing"), TINY.hidden_size),
        admitted_lane(lexicon, spans, 1, ("zhong", "ai"), TINY.hidden_size),
        admitted_lane(lexicon, spans, 2, ("bei", "jing", "wo", "ai", "zhong"), TINY.hidden_size),
    ]


def test_a_bucketed_search_finds_what_the_exact_one_finds(
    lexicon: Lexicon, spans: SpanVocab
) -> None:
    """Padding lanes and positions to a bucket changes no lane's hypotheses."""
    model = ranking_model(lexicon).eval()
    lanes = search_lanes(lexicon, spans)
    device = model.decoder.emit_ids.device
    lengths = [lane.length for lane in lanes]
    with torch.no_grad():
        exact = read_back(
            *run_search(model.decoder, search_inputs(lanes, 4, device, bucketed=False)), lengths
        )
        padded = search_inputs(lanes, 4, device, bucketed=True)
        assert padded.masks.shape[:2] == (8, 8)
        bucketed = read_back(*run_search(model.decoder, padded), lengths)
    for joint, single in zip(exact, bucketed, strict=True):
        assert [sequence for sequence, _ in joint] == [sequence for sequence, _ in single]
        assert [score for _, score in joint] == pytest.approx(
            [score for _, score in single], abs=1e-5
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a CUDA device")
def test_a_replayed_search_is_the_search_it_captured(lexicon: Lexicon, spans: SpanVocab) -> None:
    """The graph replays the bucketed search bit for bit, for new inputs of its shape too."""
    model = ranking_model(lexicon).cuda().eval()
    graphs = SearchGraphs()
    first = search_lanes(lexicon, spans)
    second = [replace(lane, vectors=lane.vectors.flip(0)) for lane in reversed(first)]
    with torch.no_grad():
        for lanes in (first, second, first):
            eager = run_search(
                model.decoder, search_inputs(lanes, 4, torch.device("cuda"), bucketed=True)
            )
            replayed = graphs.run(
                model.decoder, search_inputs(lanes, 4, torch.device("cuda"), bucketed=True)
            )
            for expected, found in zip(eager, replayed, strict=True):
                assert torch.equal(expected, found)
    assert len(graphs.captured) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a CUDA device")
def test_a_search_past_the_buffers_is_captured_anew(lexicon: Lexicon, spans: SpanVocab) -> None:
    """A larger search grows the buffers, drops the graphs, and captures its own.

    The dropped graph's outputs are still held, so its pool outlives every
    graph that used it.
    """
    model = ranking_model(lexicon).cuda().eval()
    graphs = SearchGraphs()
    small = search_lanes(lexicon, spans)
    torch.manual_seed(2)
    reading = ("wo", "ai", "bei", "jing", "zhong", "ai", "wo", "ai", "bei", "jing")
    large = [*small, admitted_lane(lexicon, spans, 3, reading, TINY.hidden_size)]

    def inputs(lanes: list[Lane]) -> SearchInputs:
        return search_inputs(lanes, 4, torch.device("cuda"), bucketed=True)

    with torch.no_grad():
        held = graphs.run(model.decoder, inputs(small))
        eager = run_search(model.decoder, inputs(large))
        replayed = graphs.run(model.decoder, inputs(large))
    # The sequences are [lanes, beam, positions]: the longer lane grows the positions.
    assert replayed[0].shape[-1] > held[0].shape[-1]
    for expected, found in zip(eager, replayed, strict=True):
        assert torch.equal(expected, found)
    assert len(graphs.captured) == 1


def test_a_compiled_step_stops_compiling_once_warm(lexicon: Lexicon, spans: SpanVocab) -> None:
    """The compiled search equals the eager one, and new shapes add no graphs.

    Each new cache length would otherwise be a new specialisation, and past
    dynamo's recompile limit the step falls back to eager for the rest of
    the run -- the dispatch cost compiling it was for.
    """
    from torch._dynamo.utils import counters

    from mlime.train.e2e import compile_modules

    model = ranking_model(lexicon).eval()
    torch.manual_seed(1)
    width = TINY.hidden_size
    readings = [
        ("wo", "ai", "bei", "jing"),
        ("zhong", "ai"),
        ("bei", "jing", "wo", "ai", "zhong"),
        ("wo", "ai", "bei"),
        ("zhong", "ai", "bei", "jing", "wo"),
        ("wo", "ai", "bei", "jing", "zhong", "ai"),
        ("bei", "jing"),
    ]

    def batch(*picks: int) -> list[Lane]:
        return [
            admitted_lane(lexicon, spans, owner, readings[pick], width)
            for owner, pick in enumerate(picks)
        ]

    # Ragged and even, each at two lengths, three lanes apiece: sizes that
    # equal another dimension of the tiny model on the first trace (two
    # layers, four heads) would be guarded equal to it, and a run's first
    # batches do not have such coincidences to offer. Then new rows,
    # lengths, and the coincident sizes themselves.
    warming = [batch(2, 3, 1), batch(5, 0, 3), batch(3, 3, 3), batch(2, 4, 2)]
    warm = [batch(0, 1), batch(6, 6), batch(2, 3, 4, 1), batch(5, 5, 5, 5, 5), batch(0, 0, 0, 0)]
    torch._dynamo.reset()
    with torch.no_grad():
        eager = [decode_lanes(model, lanes, 4) for lanes in warming + warm]
        compile_modules(model, "default")
        compiled = [decode_lanes(model, lanes, 4) for lanes in warming]
        graphs = counters["stats"]["unique_graphs"]
        compiled += [decode_lanes(model, lanes, 4) for lanes in warm]
    assert counters["stats"]["unique_graphs"] == graphs
    for expected, found in zip(eager, compiled, strict=True):
        for lane, lane_found in zip(expected, found, strict=True):
            assert [sequence for sequence, _ in lane] == [sequence for sequence, _ in lane_found]
            assert [score for _, score in lane] == pytest.approx(
                [score for _, score in lane_found], abs=1e-4
            )


def test_a_compiled_step_inside_ddp_stays_one_graph(
    lexicon: Lexicon, spans: SpanVocab, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The search's step is not cut at DDP's buckets.

    Dynamo splits every graph it compiles inside a DDP forward at the
    gradient buckets, so each piece's allreduce can start while backward is
    still running. The step has no backward; split, a position became nine
    compiled calls, and on a T4 their launches took four times the device
    time they bought.
    """
    from torch._dynamo.utils import counters
    from torch.nn.parallel import DistributedDataParallel

    from mlime.train.e2e import compile_modules

    width = TINY.hidden_size
    lanes = [
        admitted_lane(lexicon, spans, 0, ("wo", "ai", "bei"), width),
        admitted_lane(lexicon, spans, 1, ("zhong", "ai", "bei", "jing", "wo"), width),
    ]

    def compiled_graphs(wrap: Callable[[RankingE2EModel], AbstractContextManager[object]]) -> int:
        """Graphs one search compiles from a cold cache, inside *wrap*'s context."""
        torch._dynamo.reset()
        model = ranking_model(lexicon)
        compile_modules(model, "default")
        before = counters["aot_autograd"]["total"]
        with torch.no_grad(), wrap(model):
            decode_lanes(model, lanes, 4)
        return int(counters["aot_autograd"]["total"] - before)

    alone = compiled_graphs(lambda _: contextlib.nullcontext())
    torch.distributed.init_process_group(
        "gloo", store=torch.distributed.HashStore(), rank=0, world_size=1
    )
    try:
        # Buckets far smaller than any one weight -- the first one too, which
        # dynamo takes from c10d's default: split, every weight would end a
        # graph.
        monkeypatch.setattr(torch.distributed, "_DEFAULT_FIRST_BUCKET_BYTES", 64)
        inside = compiled_graphs(
            lambda model: DistributedDataParallel(
                model, bucket_cap_mb=64 / 2**20
            )._inside_ddp_forward()
        )
        assert inside == alone
    finally:
        torch.distributed.destroy_process_group()


def test_the_path_score_is_the_search_score(
    lexicon: Lexicon, spans: SpanVocab, tokenizer: StubTokenizer
) -> None:
    """What the loss trains is what the evaluation ranks by, sentence for sentence."""
    model = ranking_model(lexicon).eval()
    # Open the gates so the encoder and the cross-attention take part.
    with torch.no_grad():
        for layer in model.decoder.gated_layers():
            layer.gate.fill_(0.5)
        model.decoder.span_input.weight.normal_(std=0.1)
    batch = rank_batch(lexicon, spans, tokenizer)
    with torch.no_grad():
        hidden = model.encode(batch.rows)
        width = batch.rows.targets.shape[1]
        rows = hidden[:, 1 : width - 1]
        admitted = model._admitted(batch.rows)
        lanes = [
            Lane(row, rows[row, :length], admitted[row, :length])
            for row, length in enumerate(batch.lengths)
        ]
        found = decode_lanes(model, lanes, 4)
        pairs = [
            (row, tuple(sequence), score)
            for row, hypotheses in enumerate(found)
            for sequence, score in hypotheses
        ]
        assert pairs
        read = torch.tensor([row for row, _, _ in pairs])
        said = torch.full((len(pairs), rows.shape[1]), -1)
        for item, (_, sequence, _) in enumerate(pairs):
            said[item, : len(sequence)] = torch.tensor(sequence)
        steps = said >= 0
        bound = int(steps.sum()) + 3
        paths, _ = model.path_scores(rows, read, said, steps, bound)
    assert paths.tolist() == pytest.approx([score for _, _, score in pairs], abs=1e-4)


def test_the_ranking_loss_moves_the_gold_sentence_first(
    lexicon: Lexicon, spans: SpanVocab, tokenizer: StubTokenizer
) -> None:
    model = ranking_model(lexicon).train()
    batch = rank_batch(lexicon, spans, tokenizer)
    first = model(batch)
    assert first.loss is not None and torch.isfinite(first.loss)
    optimiser = torch.optim.AdamW(model.parameter_groups(1e-3, 3e-3))
    for _ in range(60):
        loss = model(batch).loss
        assert loss is not None
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    last = model(batch)
    assert float(last.extras["rank_loss"]) < 0.5 * float(first.extras["rank_loss"])
    assert float(last.extras["rank_accuracy"]) == 1.0


def reference_path_scores(
    model: RankingE2EModel,
    spans: torch.Tensor,
    rows: list[int],
    sequences: list[tuple[int, ...]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """#116's `path_scores`, verbatim: the padding and the real steps built on the host."""
    decoder = model.decoder
    device = spans.device
    length = max(len(sequence) for sequence in sequences)
    padded = torch.zeros((len(sequences), length), dtype=torch.long)
    steps = torch.zeros((len(sequences), length), dtype=torch.bool)
    for item, sequence in enumerate(sequences):
        padded[item, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        steps[item, : len(sequence)] = True
    taken = steps.reshape(-1).nonzero().squeeze(1)
    owner = taken // length
    said = padded.to(device, non_blocking=True)
    mask = steps.to(device, non_blocking=True)
    read = torch.tensor(rows, dtype=torch.long).to(device, non_blocking=True)
    reader_ids = decoder.emit_ids[said]
    previous = torch.cat([torch.full_like(reader_ids[:, :1], BOS), reader_ids[:, :-1]], dim=1)
    read_spans = spans[read, :length]
    features = decoder(previous, read_spans, mask)
    reader = decoder.reader
    projected: torch.Tensor = reader.project(reader.dropout(features))
    flat = projected.reshape(-1, projected.shape[-1]).index_select(
        0, taken.to(device, non_blocking=True)
    )
    targets = said.reshape(-1).index_select(0, taken.to(device, non_blocking=True))
    emitted = model._emitted()
    stepped = torch.cat(
        [
            torch.as_tensor(
                torch.utils.checkpoint.checkpoint(
                    _step_log_probabilities,
                    flat[start : start + STEP_CHUNK],
                    emitted,
                    targets[start : start + STEP_CHUNK],
                    use_reentrant=False,
                )
            )
            for start in range(0, flat.shape[0], STEP_CHUNK)
        ]
    )
    paths = torch.zeros(len(sequences), dtype=stepped.dtype, device=device).index_add(
        0, owner.to(device, non_blocking=True), stepped
    )
    return paths, projected


def reference_forward(model: RankingE2EModel, batch: RankBatch) -> dict[str, torch.Tensor]:
    """#116's ranking forward, verbatim: beams read back, selected and padded on the host."""
    ranking = model.ranking()
    rows = batch.rows
    hidden = model.encode(rows)
    width = rows.targets.shape[1]
    spans = hidden[:, 1 : width - 1]
    admitted = model._admitted(rows)
    decoder = model.decoder
    training = decoder.training
    decoder.eval()
    with torch.no_grad():
        detached = spans.detach()
        found = decode_lanes(
            model,
            [
                Lane(row, detached[row, :length], admitted[row, :length])
                for row, length in enumerate(batch.lengths)
            ],
            ranking.beam,
        )
    decoder.train(training)
    torch.clear_autocast_cache()
    items = reference_select(
        batch.owners, batch.gold_rows, gold_sentences(batch), found, ranking.negatives
    )
    paths, projected = reference_path_scores(model, spans, items.rows, items.sequences)
    length = projected.shape[1]
    device = spans.device
    slots = torch.tensor(items.slots, dtype=torch.long).to(device, non_blocking=True)
    positive = torch.tensor(items.positive, dtype=torch.bool).to(device, non_blocking=True)
    listed = slots >= 0
    floor = torch.tensor(float("-inf"), device=device)
    scores = torch.where(listed, paths[slots.clamp(min=0)], floor)
    rank_loss = (
        scores.logsumexp(dim=1) - torch.where(positive, scores, floor).logsumexp(dim=1)
    ).mean()
    first_negative = torch.where(listed & ~positive, scores, floor).amax(dim=1)
    rank_accuracy = (scores[:, 0] > first_negative).float().mean()
    positions = rows.scored
    owners = torch.tensor(batch.owners, dtype=torch.long).to(device, non_blocking=True)
    region = owners[positions // width] * length + positions % width - 1
    decoder_logits = (
        projected.reshape(-1, projected.shape[-1]).index_select(0, region) @ model._emitted().T
    )
    candidates = model.candidates_at(rows, positions)
    targets = rows.targets.reshape(-1).index_select(0, positions)
    encoder_logits = model.head.decoder(
        model.head.transform(hidden.reshape(-1, hidden.shape[-1]).index_select(0, positions))
    )
    decoder_loss = model.position_loss(decoder_logits, targets, candidates)
    encoder_loss = restricted_cross_entropy(
        encoder_logits, targets, candidates, model.config.label_smoothing, check=False
    )
    return {
        "loss": decoder_loss
        + model.e2e_config.aux_weight * encoder_loss
        + ranking.weight * rank_loss,
        "decoder_loss": decoder_loss,
        "encoder_loss": encoder_loss,
        "rank_loss": rank_loss,
        "rank_accuracy": rank_accuracy,
    }


#: Readings that collide: ``zh``/``z`` admit what ``zhong`` does and ``hu``/``huo``
#: what ``he`` does, so one sentence turns up on several readings of one length,
#: beside a reading of another length whose ``g`` admits nothing at all; the
#: last example's 81 sentences leave its gold one to chance.
COLLIDING = {
    "zhongzhongzhong": [
        ("zh", "zhong", "zhong"),
        ("zhong", "z", "zhong"),
        ("zh", "zh", "zh"),
        ("zhong", "zhong", "zh", "g"),
    ],
    "woaihe": [("wo", "ai", "hu"), ("wo", "ai", "huo"), ("w", "ai", "h")],
    "zhonglv": [("zh", "lu"), ("z", "l"), ("zho", "l")],
    "zhongzhongzhongzhong": [("z", "z", "z", "z")],
}


def colliding_batch(lexicon: Lexicon, spans: SpanVocab, tokenizer: StubTokenizer) -> RankBatch:
    """Four examples whose readings collide, for a forward that exercises every merge."""
    collator = Collator(
        tokenizer, CandidateSpace(spans, lexicon.candidate_mask), context_dropout=0.0
    )
    examples = [
        example(spans, lexicon, "中钟重", ("zhong", "zhong", "zhong")),
        example(spans, lexicon, "我爱和", ("wo", "ai", "he")),
        example(spans, lexicon, "钟绿", ("zhong", "lv")),
        example(spans, lexicon, "重中重钟", ("zhong", "zhong", "zhong", "zhong")),
    ]
    examples = [replace(typed, id=f"{typed.id}-{index}") for index, typed in enumerate(examples)]
    ranking = replace(RANKING, negatives=6)
    return RankCollator(collator, lambda keys: COLLIDING[keys], spans, ranking)(examples)


def forward_model(lexicon: Lexicon, seed: int) -> RankingE2EModel:
    """A tiny ranking model from *seed*, gates open, ranking six negatives, dropout off."""
    torch.manual_seed(seed)
    model = ranking_model(lexicon)
    model.e2e_config = replace(model.e2e_config, ranking=replace(RANKING, negatives=6))
    with torch.no_grad():
        for layer in model.decoder.gated_layers():
            layer.gate.fill_(0.5)
        model.decoder.span_input.weight.normal_(std=0.1)
    return model.eval()


SEEDS = (0, 1, 2)


DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
    ),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEEDS)
def test_the_ranking_forward_is_116s_forward(
    lexicon: Lexicon, spans: SpanVocab, tokenizer: StubTokenizer, seed: int, device: str
) -> None:
    """The device step trains what the host step trained: the same losses, the same gradients.

    Dropout is off -- the two lay their paths out differently, so they would
    draw different masks -- and everything else is the same model and batch.
    On CUDA the path sums are atomic adds in whatever order they land, so
    the two agree to rounding rather than bit for bit.
    """
    model = forward_model(lexicon, seed).to(device)
    batch = colliding_batch(lexicon, spans, tokenizer).to(torch.device(device))

    output = model(batch)
    assert output.loss is not None
    output.loss.backward()
    grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    expected = reference_forward(model, batch)
    expected["loss"].backward()

    torch.testing.assert_close(output.loss.detach(), expected["loss"].detach())
    for name, value in output.extras.items():
        torch.testing.assert_close(value, expected[name].detach(), msg=name)
    assert grads.keys() == {name for name, p in model.named_parameters() if p.grad is not None}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(grads[name], parameter.grad, msg=name)


def test_the_colliding_batch_collides(
    lexicon: Lexicon, spans: SpanVocab, tokenizer: StubTokenizer
) -> None:
    """The forward test's batch is worth its name, over the seeds it runs.

    A sentence turns up on two readings, the gold sentence on a reading not
    its own and nowhere at all, an example has fewer candidates than
    negatives, and a lane finds nothing.
    """
    seen: Counter[str] = Counter()
    batch = colliding_batch(lexicon, spans, tokenizer)
    gold = gold_sentences(batch)
    for seed in SEEDS:
        model = forward_model(lexicon, seed)
        with torch.no_grad():
            hidden = model.encode(batch.rows)
            width = batch.rows.targets.shape[1]
            rows = hidden[:, 1 : width - 1]
            admitted = model._admitted(batch.rows)
            found = decode_lanes(
                model,
                [
                    Lane(row, rows[row, :length], admitted[row, :length])
                    for row, length in enumerate(batch.lengths)
                ],
                RANKING.beam,
            )
        items = reference_listed(reference_select(batch.owners, batch.gold_rows, gold, found, 6))
        seen["lane finds nothing"] += any(not there for there in found)
        for index, gold_row in enumerate(batch.gold_rows):
            said: Counter[tuple[int, ...]] = Counter()
            for row, owner in enumerate(batch.owners):
                if owner == index:
                    said.update({tuple(sequence) for sequence, _ in found[row]})
            seen["sentence on two readings"] += any(count > 1 for count in said.values())
            seen["gold absent"] += gold[index] not in said
            seen["gold on another reading"] += any(
                row != gold_row and mark for row, _, mark in items[index][1:]
            )
            seen["fewer candidates than negatives"] += (
                sum(not mark for _, _, mark in items[index]) < 6
            )
    assert all(seen[case] for case in CASES if case not in ("tie at the cut", "unfilled slot")), (
        seen
    )
    assert seen["lane finds nothing"], seen


@pytest.mark.skipif(not torch.cuda.is_available(), reason="device syncs need a CUDA device")
@pytest.mark.parametrize("graphs", [False, True], ids=["eager", "graphs"])
def test_a_ranking_step_never_waits_on_the_device(
    lexicon: Lexicon, spans: SpanVocab, tokenizer: StubTokenizer, graphs: bool
) -> None:
    """A whole training step -- copy in, forward, backward, clip, update -- queues without a sync.

    The loop's own step under fp16 autocast and loss scaling with the fused
    AdamW, on the colliding batch. The first step warms up what is lazily made
    once (the cuBLAS handles, the scaler's state, a captured search graph);
    the second runs with every synchronising call an error.
    """
    device = torch.device("cuda")
    model = forward_model(lexicon, 0).to(device).train()
    if graphs:
        model.search_graphs = SearchGraphs()
    held = colliding_batch(lexicon, spans, tokenizer).pin_memory()
    optimiser = torch.optim.AdamW(model.parameter_groups(1e-3, 3e-3), fused=True)
    scaler = torch.amp.GradScaler("cuda")

    def step() -> None:
        batch = held.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(batch)
        assert output.loss is not None
        scaler.scale(output.loss).backward()
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimiser)
        scaler.update()
        optimiser.zero_grad(set_to_none=True)

    step()
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        step()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()


def test_a_ranking_checkpoint_rebuilds_the_ranking_model(lexicon: Lexicon, tmp_path: Path) -> None:
    model = ranking_model(lexicon)
    path = tmp_path / "ranking.pt"
    torch.save({"e2e": asdict(model.config), "model": model.state_dict(), "step": 3}, path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert E2EConfig.from_record(state["e2e"]).ranking == RANKING
    rebuilt = from_checkpoint(state, lexicon, bert_config=TINY)
    assert isinstance(rebuilt, RankingE2EModel)
    rebuilt.load_state_dict(state["model"], strict=True)

    # A v1 record has no ranking field, and builds the plain model.
    plain = {key: value for key, value in state["e2e"].items() if key != "ranking"}
    assert E2EConfig.from_record(plain).ranking is None
    rebuilt = from_checkpoint({**state, "e2e": plain}, lexicon, bert_config=TINY)
    assert type(rebuilt) is E2EModel


def greedy_readings(spans: SpanVocab) -> Segmentations:
    """A stand-in segmenter: the longest-span reading of the keys."""

    def segment(keys: str) -> list[tuple[str, ...]]:
        reading: list[str] = []
        start = 0
        while start < len(keys):
            end = max(stop for stop in range(start + 1, len(keys) + 1) if keys[start:stop] in spans)
            reading.append(keys[start:end])
            start = end
        return [tuple(reading)]

    return segment


def test_a_resumed_ranking_run_is_the_run_that_was_not_interrupted(
    short_corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    tokenizer: StubTokenizer,
    tmp_path: Path,
) -> None:
    config = TrainingConfig(
        max_steps=4,
        base_lr=1e-3,
        new_lr=3e-3,
        token_budget=24,
        log_every=1,
        checkpoint_every=2,
        fp16=False,
        seed=3,
    )

    def lane() -> BuiltLane:
        stream = CorpusStream(
            short_corpus[0], short_corpus[1], SampleBuilder(lexicon, spans, arbitration, seed=1)
        )
        collator = Collator(tokenizer, CandidateSpace(spans, lexicon.candidate_mask))
        return BuiltLane(stream, RankCollator(collator, greedy_readings(spans), spans, RANKING))

    def losses(metrics: Path) -> list[float]:
        return [
            float(record["rank_loss"])
            for record in (json.loads(line) for line in metrics.read_text().splitlines())
            if record.get("event") == "step"
        ]

    whole_model = ranking_model(lexicon)
    whole = train(whole_model, [lane()], config, tmp_path / "whole").metrics
    resumed_model = ranking_model(lexicon)
    resumed = train(
        resumed_model,
        [lane()],
        config,
        tmp_path / "resumed",
        resume=tmp_path / "whole" / "checkpoint-000002.pt",
    ).metrics
    assert losses(resumed) == losses(whole)[2:]
    weights = whole_model.state_dict()
    for name, tensor in resumed_model.state_dict().items():
        assert torch.equal(tensor, weights[name]), name


def script(path: Path, body: str) -> Path:
    """An executable Python script at *path*."""
    path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_the_segmenter_speaks_one_line_each_way(tmp_path: Path) -> None:
    echo = script(
        tmp_path / "segment",
        "import json, sys\n"
        "assert sys.argv[1:] == ['--max-paths', '8']\n"
        "for line in sys.stdin:\n"
        "    keys = line.strip()\n"
        "    print(json.dumps([[keys[:2], keys[2:]], [keys]]), flush=True)\n",
    )
    segment = Segmenter(echo, 8)
    assert segment("woai") == [("wo", "ai"), ("woai",)]
    assert segment("zhong") == [("zh", "ong"), ("zhong",)]

    dead = script(tmp_path / "dead", "import sys\nsys.exit(4)\n")
    with pytest.raises(RuntimeError, match=r"exited \(4\)"):
        Segmenter(dead, 8)("woai")


def test_every_parameter_learns_under_autocast(
    lexicon: Lexicon, spans: SpanVocab, tokenizer: StubTokenizer
) -> None:
    """The search's gradient-free casts must not stand in for the scored path's.

    Autocast caches the low-precision copy of a weight within its region; a
    copy first made under ``no_grad`` carries no graph, and a later use that
    found it in the cache would cut the weight out of the loss -- which DDP
    reports as a parameter that received no gradient. Torch builds whose
    cache keeps such copies (the Kaggle image's) fail this without the
    forward's cache clear; newer ones do not cache them at all.
    """
    model = ranking_model(lexicon).train()
    batch = rank_batch(lexicon, spans, tokenizer)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = model(batch).loss
    assert loss is not None
    loss.backward()
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None and "letter_encoder" not in name
    ]
    assert missing == []
