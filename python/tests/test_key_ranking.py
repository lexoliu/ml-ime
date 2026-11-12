"""The key-level model's sequence-level training (S4, #278), on tiny CPU models.

The tests pin the stage's pieces against the search they stand for: the
training path's score of a sentence against ``key_search.search`` without a
beam, the gathered arc tables against the gold's own, the decoder's cached
step against its teacher-forced query, the cover DP and the lane beam against
brute force, and the candidates and the term against hand computation.
"""

from __future__ import annotations

import gzip
import itertools
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pytest
import torch
from test_e2e_model import TINY
from test_keys import (
    CPU,
    FRAMED,
    JOINT_KEYS,
    build_examples,
    key_collator,
    legality_fixture,  # noqa: F401 -- the fixture, by name
    parent_checkpoint,
    reader_vocab_fixture,  # noqa: F401
    run_records,
    tiny_joint_model,
    vocabularies_fixture,  # noqa: F401
)
from transformers import BertConfig

from mlime.legality import Legality
from mlime.train.charlm_vocab import BOS, CharVocab
from mlime.train.ctc import alignment_scores, constrained_ctc
from mlime.train.e2e import e2e, load_key_model
from mlime.train.e2e_model import E2EConfig, KeyRankingConfig, emittable_ids
from mlime.train.e2e_search import (
    KeySearchInputs,
    padded_key_inputs,
    run_key_search,
)
from mlime.train.key_eval import DecodeTimes, KeyDecoder, KeyRecord, KeyWeights, ModelTransition
from mlime.train.key_model import KeyE2EModel, KeyStates
from mlime.train.key_ranking import (
    KeyItems,
    KeyRankBatch,
    KeyRankCollator,
    KeyRankingModel,
    Lanes,
    cover_dp,
    cover_lanes,
    dense_span_table,
    edit_distance,
    item_arc_costs,
    lane_gains,
    select_key_items,
    sentence_scores,
    span_gains,
    span_rows_of,
)
from mlime.train.key_rerank import rerank_dump
from mlime.train.key_search import Hypothesis, LegalityVerifier, search
from mlime.train.keys import FillTowerTraining
from mlime.train.lexicon import Lexicon
from mlime.train.loop import TrainingConfig
from mlime.train.run import RunPaths, Slices, Vocabularies
from mlime.train.spans import SpanVocab

ROOT = Path(__file__).parent.parent.parent
DATA = ROOT / "crates/ime-pinyin/data"
BETA = 2.0
RANKING = KeyRankingConfig(beam=4, negatives=3, other_lengths=1, weight=0.5, beta=BETA, gamma=1.0)


def ranked_model(lexicon: Lexicon, vocab: CharVocab, ranking: KeyRankingConfig) -> KeyRankingModel:
    """The tiny joint model as the stage trains it: its gates open, its weights S3's shape."""
    model = tiny_joint_model(lexicon, vocab)
    config: E2EConfig = replace(model.e2e_config, key_ranking=ranking)
    ranked = KeyRankingModel.compose(
        TINY, lexicon, config, len(vocab), emittable_ids(lexicon, vocab)
    )
    assert isinstance(ranked, KeyRankingModel)
    ranked.load_state_dict(model.state_dict())
    return ranked


@dataclass
class Rig:
    """A batch, a model in evaluation mode and everything the scoring path reads off them."""

    model: KeyRankingModel
    batch: KeyRankBatch
    states: KeyStates
    flat: torch.Tensor
    table: torch.Tensor
    spans: torch.Tensor
    keys: list[str]


def rig(
    vocabularies: Vocabularies,
    lexicon: Lexicon,
    vocab: CharVocab,
    ranking: KeyRankingConfig = RANKING,
    count: int = 6,
    typos: bool = True,
) -> Rig:
    model = ranked_model(lexicon, vocab, ranking)
    model.eval()
    examples = build_examples(vocabularies, typos=typos)[:count]
    collator = vocabularies.key_rank_collator(JOINT_KEYS, ranking, context_dropout=0.0)
    batch = collator(examples)
    keyed = batch.keys
    with torch.no_grad():
        encoded = model.encode_keys(keyed, model.encode_context(keyed.spans))
        flat = model.ctc_log_probs(
            encoded.reshape(-1, encoded.shape[-1]).index_select(0, keyed.key_positions)
        )
        states = model.states_at_keys(keyed.key_mask, encoded)
        table = dense_span_table(batch, model.decoder.emit_ids.numel())
    return Rig(model, batch, states, flat, table, span_rows_of(batch), [e.keys for e in examples])


def padded(sequences: list[tuple[int, ...]]) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(map(len, sequences))
    ids = torch.zeros(len(sequences), width, dtype=torch.long)
    for row, each in enumerate(sequences):
        ids[row, : len(each)] = torch.tensor(each)
    return ids, torch.tensor([len(each) for each in sequences])


def every_output(
    r: Rig, legality: Legality, gamma: float
) -> tuple[list[int], list[Hypothesis], list[list[Hypothesis]]]:
    """The unbounded search's outputs of every record, flattened with their record."""
    keyed = r.batch.keys
    lattices = [legality.lattice(each, JOINT_KEYS.options) for each in r.keys]
    with torch.no_grad():
        log_probs = r.model.ctc_log_probs(r.states.states)
        found = search(
            lattices,
            log_probs,
            ModelTransition(r.model, r.states, DecodeTimes()),
            beta=BETA,
            gamma=gamma,
            beam=None,
            viterbi=False,
        )
    assert keyed.size == len(found)
    owners = [record for record, each in enumerate(found) for _ in each]
    return owners, [h for each in found for h in each], found


@pytest.mark.parametrize("gamma", [0.5, 1.0])
def test_a_sentences_training_score_is_the_unbounded_searchs(
    vocabularies: Vocabularies,
    legality: Legality,
    lexicon: Lexicon,
    reader_vocab: CharVocab,
    gamma: float,
) -> None:
    r = rig(vocabularies, lexicon, reader_vocab)
    owners, outputs, _ = every_output(r, legality, gamma)
    assert len(outputs) > 20
    sequences, lengths = padded([h.ids for h in outputs])
    owner = torch.tensor(owners)
    keyed = r.batch.keys
    with torch.no_grad():
        parts = sentence_scores(
            r.model,
            r.states,
            r.flat,
            keyed.key_index,
            keyed.keys,
            owner,
            sequences,
            lengths,
            item_arc_costs(r.table, r.spans, owner, sequences),
            gamma,
            int(lengths.sum()),
        )
    got = parts.score(BETA)
    expected = torch.tensor([h.score for h in outputs], dtype=got.dtype)
    assert torch.allclose(got, expected, atol=1e-4, rtol=1e-5), (got - expected).abs().max()
    # The parts are the search's own terms.
    assert torch.allclose(
        parts.transition, torch.tensor([h.transition for h in outputs], dtype=got.dtype), atol=1e-4
    )


def test_the_scores_gradient_reaches_the_encoder_and_the_decoder(
    vocabularies: Vocabularies,
    legality: Legality,
    lexicon: Lexicon,
    reader_vocab: CharVocab,
) -> None:
    r = rig(vocabularies, lexicon, reader_vocab)
    model = r.model
    model.train()
    keyed = r.batch.keys
    encoded = model.encode_keys(keyed, model.encode_context(keyed.spans))
    flat = model.ctc_log_probs(
        encoded.reshape(-1, encoded.shape[-1]).index_select(0, keyed.key_positions)
    )
    states = model.states_at_keys(keyed.key_mask, encoded)
    owner = torch.arange(keyed.size)
    parts = sentence_scores(
        model,
        states,
        flat,
        keyed.key_index,
        keyed.keys,
        owner,
        keyed.gold,
        keyed.lengths,
        item_arc_costs(r.table, r.spans, owner, keyed.gold),
        1.0,
        int(keyed.lengths.sum()),
    )
    parts.score(BETA).sum().backward()
    for name in ("decoder", "key_front", "ctc_head"):
        grads = [
            p.grad for n, p in model.named_parameters() if n.startswith(name) and p.requires_grad
        ]
        if grads:
            assert any(g is not None and g.abs().sum() > 0 for g in grads), name


def test_the_gathered_arc_table_prices_a_sentence_as_its_alignments_do(
    vocabularies: Vocabularies,
    legality: Legality,
    lexicon: Lexicon,
    reader_vocab: CharVocab,
) -> None:
    r = rig(vocabularies, lexicon, reader_vocab)
    owners, outputs, _ = every_output(r, legality, 1.0)
    keyed = r.batch.keys
    # The gold sentences, then the search's legal outputs.
    sentences = [
        tuple(row[:n]) for row, n in zip(keyed.gold.tolist(), keyed.lengths.tolist(), strict=True)
    ]
    sentences += [h.ids for h in outputs]
    owner = torch.tensor(list(range(keyed.size)) + owners)
    sequences, lengths = padded(sentences)
    width = keyed.key_index.shape[1]
    exact = torch.full((len(sentences), sequences.shape[1], width, 8), math.inf)
    for item, (record, ids) in enumerate(zip(owner.tolist(), sentences, strict=True)):
        arcs = legality.gold_ids(r.keys[record], ids, JOINT_KEYS.options)
        assert arcs is not None
        dense = torch.from_numpy(arcs.dense())
        exact[item, : dense.shape[0], : dense.shape[1]] = dense.float()
    gathered = item_arc_costs(r.table, r.spans, owner, sequences)
    blank, emit = alignment_scores(r.flat, keyed.key_index.index_select(0, owner), sequences)
    keys = keyed.keys.index_select(0, owner)
    with torch.no_grad():
        want = constrained_ctc(blank, emit, exact, keys, lengths)
        got = constrained_ctc(blank, emit, gathered, keys, lengths)
    assert torch.allclose(got, want, atol=1e-4, rtol=1e-5), (got - want).abs().max()
    # A typo example's sentences pay a correction somewhere in the batch.
    assert (gathered.isfinite() & (gathered > 0)).any()


# --- the decoder's cached step ---------------------------------------------------


def test_the_cached_step_equals_the_teacher_forced_query_at_every_prefix(
    vocabularies: Vocabularies, lexicon: Lexicon, reader_vocab: CharVocab
) -> None:
    r = rig(vocabularies, lexicon, reader_vocab, count=4)
    decoder = r.model.decoder
    keyed = r.batch.keys
    rows = keyed.size
    length = int(keyed.lengths.max())
    encoder_kv = [cross.kv(r.states.states) for cross in decoder.gated_layers()]
    layers, heads = decoder.reader.config.layers, decoder.reader.config.heads
    head_dim = decoder.reader.config.hidden // heads
    cache = (
        torch.zeros(layers, rows, heads, length, head_dim),
        torch.zeros(layers, rows, heads, length, head_dim),
    )
    tokens = torch.full((rows,), BOS, dtype=torch.long)
    with torch.no_grad():
        for position in range(length):
            logits, (new_keys, new_values) = decoder.step_keys(
                tokens, torch.tensor(position), cache, encoder_kv, r.states.mask
            )
            prefixes = keyed.gold[:, :position]
            want = r.model.transition_log_probs(
                r.states,
                torch.arange(rows),
                prefixes,
                torch.full((rows,), position),
            )
            assert torch.allclose(logits.float().log_softmax(-1), want, atol=1e-4)
            cache[0][:, :, :, position : position + 1] = new_keys
            cache[1][:, :, :, position : position + 1] = new_values
            tokens = decoder.emit_ids[keyed.gold[:, position]]


# --- the covers and the lane beam -------------------------------------------------


def covers(keys: int, spans: int, limit: int) -> list[list[tuple[int, int]]]:
    """Every cover of ``keys`` keys by ``spans`` spans of at most *limit* keys, as (end, width)."""
    found = []
    for widths in itertools.product(range(1, limit + 1), repeat=spans):
        if sum(widths) == keys:
            ends = list(itertools.accumulate(widths))
            found.append(list(zip(ends, widths, strict=True)))
    return found


def test_the_length_indexed_dp_is_the_best_cover_of_each_length() -> None:
    generator = torch.Generator().manual_seed(3)
    rows, width, limit, cap = 5, 7, 4, 8
    gain = torch.randn(rows, width, limit, generator=generator)
    gain[torch.rand(rows, width, limit, generator=generator) < 0.3] = -math.inf
    keys = torch.tensor([7, 5, 3, 1, 6])
    value, back = cover_dp(gain, keys, cap)
    for row, count in enumerate(keys.tolist()):
        for spans in range(cap + 1):
            options = covers(count, spans, limit) if spans else []
            totals = [sum(float(gain[row, end - 1, w - 1]) for end, w in c) for c in options]
            best = max(totals, default=-math.inf)
            assert float(value[row, spans]) == pytest.approx(best, abs=1e-5), (row, spans)
            if best > -math.inf:
                # The back pointers walk to a cover of that value.
                cursor, left, walked = count, spans, 0.0
                while left:
                    step = int(back[row, cursor, left])
                    walked += float(gain[row, cursor - 1, step - 1])
                    cursor, left = cursor - step, left - 1
                assert cursor == 0
                assert walked == pytest.approx(best, abs=1e-5)


def lane_rig(r: Rig, ranking: KeyRankingConfig) -> tuple[Lanes, torch.Tensor, int]:
    keyed = r.batch.keys
    cap = r.batch.cover_cap
    with torch.no_grad():
        gain = span_gains(r.flat, keyed.key_index, r.table, r.spans, ranking.gamma)
        value, back = cover_dp(gain, keyed.keys, cap)
        lanes = cover_lanes(
            value, back, keyed.keys, keyed.span_end, keyed.lengths, ranking.other_lengths, cap
        )
        gains = lane_gains(r.flat, keyed.key_index, r.table, r.spans, lanes, ranking.gamma)
    return lanes, gains, cap


def test_the_lanes_are_covers_the_first_the_gold_typed_one(
    vocabularies: Vocabularies, lexicon: Lexicon, reader_vocab: CharVocab
) -> None:
    ranking = replace(RANKING, other_lengths=2)
    r = rig(vocabularies, lexicon, reader_vocab, ranking, count=12)
    keyed = r.batch.keys
    lanes, gains, _ = lane_rig(r, ranking)
    count = keyed.size
    assert lanes.owner.shape == (count * 3,)
    for lane in range(count * 3):
        record = int(lanes.owner[lane])
        length = int(lanes.lengths[lane])
        if lane < count:
            assert length == int(keyed.lengths[record])
            assert lanes.usable[lane]
            assert (
                lanes.ends[lane, :length].tolist() == (keyed.span_end[record, :length] + 1).tolist()
            )
        else:
            assert length != int(keyed.lengths[record]) or not lanes.usable[lane]
        if not lanes.usable[lane]:
            assert not (gains[lane] > -math.inf).any()
            continue
        ends, widths = lanes.ends[lane, :length], lanes.widths[lane, :length]
        assert int(ends[-1]) == int(keyed.keys[record])
        assert (ends - widths).tolist() == [0, *ends[:-1].tolist()]
        assert int(widths.min()) >= 1
    # Some example has a cover of another length.
    assert lanes.usable[count:].any()


def test_the_gold_lane_carries_the_cost_of_a_typo(
    vocabularies: Vocabularies, lexicon: Lexicon, reader_vocab: CharVocab
) -> None:
    r = rig(vocabularies, lexicon, reader_vocab, count=16)
    keyed = r.batch.keys
    lanes, gains, _ = lane_rig(r, RANKING)
    found = 0
    for record in range(keyed.size):
        for position in range(int(keyed.lengths[record])):
            slip = bool(keyed.corrupted[record, position])
            end = int(lanes.ends[record, position])
            character = int(keyed.gold[record, position])
            key = int(keyed.key_index[record, end - 1])
            emit = float(r.flat[key, character] - r.flat[key, -1])
            span = int(r.spans[record, end - 1, int(lanes.widths[record, position]) - 1])
            cost = float(r.table[span, character])
            assert math.isfinite(cost)
            assert float(gains[record, position, character]) == pytest.approx(
                emit - RANKING.gamma * cost, abs=1e-5
            )
            found += slip and cost > 0
    assert found > 0


def every_sentence(
    model: KeyE2EModel, states: KeyStates, record: int, gains: torch.Tensor, beta: float
) -> dict[tuple[int, ...], float]:
    """Every sentence a lane admits, with its score: the transition plus the lane's gains."""
    found: dict[tuple[int, ...], float] = {}
    stack: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
    while stack:
        prefix, score = stack.pop()
        position = len(prefix)
        if position == gains.shape[0]:
            found[prefix] = score
            continue
        row = model.transition_log_probs(
            states,
            torch.tensor([record]),
            torch.tensor([prefix], dtype=torch.long).view(1, position),
            torch.tensor([position]),
        )[0]
        for character in torch.nonzero(gains[position] > -math.inf).flatten().tolist():
            step = beta * float(row[character]) + float(gains[position, character])
            stack.append(((*prefix, character), score + step))
    return found


def test_the_lane_beam_is_brute_force_over_its_cover(
    vocabularies: Vocabularies, lexicon: Lexicon, reader_vocab: CharVocab
) -> None:
    ranking = replace(RANKING, beam=64)
    r = rig(vocabularies, lexicon, reader_vocab, ranking, count=6)
    keyed = r.batch.keys
    model = r.model
    lanes, gains, cap = lane_rig(r, ranking)
    inputs = padded_key_inputs(
        KeySearchInputs(
            states=r.states.states,
            key_mask=r.states.mask,
            lane_owner=lanes.owner,
            gains=gains,
            lane_lengths=lanes.lengths,
            width=ranking.beam,
            beta=ranking.beta,
        ),
        bucketed=False,
    )
    with torch.no_grad():
        sequences, scores = run_key_search(model.decoder, inputs)
        checked = 0
        for lane in range(lanes.owner.numel()):
            if not lanes.usable[lane]:
                assert torch.isinf(scores[lane]).all()
                continue
            record, length = int(lanes.owner[lane]), int(lanes.lengths[lane])
            expected = every_sentence(model, r.states, record, gains[lane, :length], ranking.beta)
            top = sorted(expected.items(), key=lambda item: -item[1])[: ranking.beam]
            got = {
                tuple(sequences[lane, beam, :length].tolist()): float(scores[lane, beam])
                for beam in range(ranking.beam)
                if scores[lane, beam] > -math.inf
            }
            assert got.keys() == {ids for ids, _ in top}
            for ids, score in top:
                assert got[ids] == pytest.approx(score, abs=1e-3)
            checked += 1
    assert checked >= keyed.size
    assert cap >= 1


# --- the candidates -----------------------------------------------------------------


def items_of(r: Rig, ranking: KeyRankingConfig) -> KeyItems:
    log_probs = r.flat
    return r.model.candidates(r.batch, r.states, log_probs, r.table, r.spans)


def test_every_negative_is_legal_and_not_the_gold_and_padding_is_flagged(
    vocabularies: Vocabularies,
    legality: Legality,
    lexicon: Lexicon,
    reader_vocab: CharVocab,
) -> None:
    verifier = LegalityVerifier.load(
        char_table=DATA / "char_pinyin.tsv",
        typo_table=DATA / "typo.json",
        syllables=DATA / "syllables.txt",
    )
    ranking = replace(RANKING, beam=6, negatives=5, other_lengths=2)
    r = rig(vocabularies, lexicon, reader_vocab, ranking, count=12)
    keyed = r.batch.keys
    items = items_of(r, ranking)
    count, negatives = keyed.size, ranking.negatives
    assert items.sequences.shape[0] == count * (1 + negatives)
    assert items.valid.shape == (count, negatives)
    found = 0
    for record in range(count):
        gold = tuple(keyed.gold[record, : int(keyed.lengths[record])].tolist())
        assert tuple(items.sequences[record, : int(items.lengths[record])].tolist()) == gold
        seen = set()
        for slot in range(negatives):
            item = count + record * negatives + slot
            ids = tuple(items.sequences[item, : int(items.lengths[item])].tolist())
            if not items.valid[record, slot]:
                # Padding is the gold again, and every later slot is padding too.
                assert ids == gold
                assert not items.valid[record, slot:].any()
                continue
            found += 1
            assert ids != gold
            assert ids not in seen
            seen.add(ids)
            sentence = "".join(lexicon.characters[i] for i in ids)
            assert verifier.cost(r.keys[record], sentence, JOINT_KEYS.options) is not None
    assert found > 0
    assert (~items.valid).any()


def test_selection_merges_lanes_by_the_better_score_and_drops_the_gold() -> None:
    # Two examples, two lanes each, three beams; the lanes are lane-major.
    sequences = torch.tensor(
        [
            # lane 0, example 0
            [[1, 2, 0], [1, 3, 0], [4, 4, 0]],
            # lane 0, example 1
            [[5, 6, 0], [5, 7, 0], [6, 6, 0]],
            # lane 1, example 0
            [[1, 2, 0], [1, 5, 0], [9, 9, 0]],
            # lane 1, example 1 (length 1)
            [[8, 0, 0], [7, 0, 0], [0, 0, 0]],
        ]
    )
    neg = -math.inf
    scores = torch.tensor(
        [[-1.0, -2.0, -3.0], [-1.0, -2.0, -3.0], [-0.5, -1.5, neg], [-4.0, -0.2, neg]]
    )
    lanes = Lanes(
        owner=torch.tensor([0, 1, 0, 1]),
        lengths=torch.tensor([2, 2, 2, 1]),
        ends=torch.ones(4, 3, dtype=torch.long),
        widths=torch.ones(4, 3, dtype=torch.long),
        usable=torch.ones(4, dtype=torch.bool),
    )
    gold = torch.tensor([[1, 3], [5, 6]])
    items = select_key_items(sequences, scores, lanes, gold, torch.tensor([2, 2]), negatives=4)
    # Example 0: (1,2) was found by lane 0 at -1.0 and by lane 1 at -0.5, one answer at -0.5;
    # (1,3) is the gold; (4,4) -3.0; (1,5) -1.5; (9,9) was never filled.
    first = [
        tuple(items.sequences[2 + slot, : int(items.lengths[2 + slot])].tolist())
        for slot in range(4)
    ]
    assert first[:3] == [(1, 2), (1, 5), (4, 4)]
    assert items.valid[0].tolist() == [True, True, True, False]
    assert first[3] == (1, 3)
    # Example 1: (7,) -0.2, (5,6) is the gold, (5,7) -2.0, (6,6) -3.0, (8,) -4.0.
    second = [
        tuple(items.sequences[2 + 4 + slot, : int(items.lengths[2 + 4 + slot])].tolist())
        for slot in range(4)
    ]
    assert second == [(7,), (5, 7), (6, 6), (8,)]
    assert items.valid[1].tolist() == [True] * 4
    assert items.owner.tolist() == [0, 1, 0, 0, 0, 0, 1, 1, 1, 1]


# --- the collator -------------------------------------------------------------------


def test_the_collator_is_deterministic_counts_rows_and_ships_the_automatons_spans(
    vocabularies: Vocabularies, legality: Legality, lexicon: Lexicon
) -> None:
    examples = build_examples(vocabularies, typos=True)[:10]
    collator = vocabularies.key_rank_collator(JOINT_KEYS, RANKING, context_dropout=0.0)
    first, second = collator(examples), collator(examples)
    for name in ("span_rows", "span_ids", "span_costs", "span_index"):
        assert torch.equal(getattr(first, name), getattr(second, name)), name
    assert collator.rows(examples[0]) == 2 + RANKING.negatives
    assert first.size == len(examples)
    assert first.cover_cap == int(first.keys.lengths.max()) + 2
    lengths = first.keys.lengths.tolist()
    assert first.steps == sum(n + RANKING.negatives * (n + 2) for n in lengths)
    table = dense_span_table(first, len(lexicon.characters))
    assert table.shape == (first.unique_spans + 1, len(lexicon.characters))
    assert torch.isinf(table[-1]).all()
    for record, example in enumerate(examples):
        for end in range(1, len(example.keys) + 1):
            for width in range(1, min(end, 8) + 1):
                span = example.keys[end - width : end]
                matches = legality.span_matches(span, False, JOINT_KEYS.options)
                row = int(first.span_index[record, end - 1, width - 1])
                if not matches.ids.size:
                    assert row == -1
                    continue
                finite = torch.isfinite(table[row])
                assert finite.nonzero().flatten().tolist() == sorted(matches.ids.tolist())
                for character, cost in zip(
                    matches.ids.tolist(), matches.costs.tolist(), strict=True
                ):
                    assert float(table[row, character]) == pytest.approx(cost, abs=1e-6)
    # Positions past a record's keys, and spans reaching before its first key, name nothing.
    assert (first.span_index[:, :, 0][~first.keys.key_mask] == -1).all()


def test_a_collator_refuses_a_tail_rule(
    vocabularies: Vocabularies, legality: Legality, spans: SpanVocab, lexicon: Lexicon
) -> None:
    from mlime.legality import LegalityOptions

    live = LegalityOptions(unfinished_tail=True, correct_syllables=True)
    with pytest.raises(ValueError, match="tail"):
        KeyRankCollator(vocabularies.key_collator(0.0), legality, live, RANKING)


# --- the term ---------------------------------------------------------------------------


def test_the_edit_distance_is_levenshtein_over_the_longer() -> None:
    said = torch.tensor([[1, 2, 3, 0], [1, 3, 0, 0], [4, 5, 6, 0], [1, 2, 3, 4], [2, 1, 3, 0]])
    lengths = torch.tensor([3, 2, 3, 4, 3])
    gold = torch.tensor([[1, 2, 3, 0]]).expand(5, 4)
    distance = edit_distance(said, lengths, gold, torch.full((5,), 3))
    assert distance.tolist() == pytest.approx([0.0, 1 / 3, 1.0, 1 / 4, 2 / 3])


def hand_items() -> tuple[KeyItems, torch.Tensor, torch.Tensor]:
    """Two examples, two negatives each, the second example's last negative padding."""
    sequences = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
            [1, 2, 3],
            [1, 3, 3],
            [4, 6, 6],
            [4, 5, 6],
            [4, 5, 6],
            [4, 5, 6],
        ][:6]
    )
    sequences = torch.tensor([[1, 2, 3], [4, 5, 6], [1, 3, 3], [7, 8, 9], [4, 6, 6], [4, 5, 6]])
    items = KeyItems(
        owner=torch.tensor([0, 1, 0, 0, 1, 1]),
        sequences=sequences,
        lengths=torch.full((6,), 3),
        valid=torch.tensor([[True, True], [True, False]]),
    )
    score = torch.tensor([-1.0, -2.0, -0.5, -3.0, -2.5, -9.0])
    return items, score, torch.tensor([0.5, 0.25, 0.1, 0.2, 0.3, 0.4])


def test_the_rank_and_risk_terms_are_computed_by_hand(
    vocabularies: Vocabularies, lexicon: Lexicon, reader_vocab: CharVocab
) -> None:
    from mlime.train.key_ranking import SentenceScores

    items, score, transition = hand_items()
    parts = SentenceScores(nll=-(score - 0.0) + BETA * transition, transition=transition)
    assert torch.allclose(parts.score(BETA), score)
    model = ranked_model(lexicon, reader_vocab, RANKING)
    term, extras = model.sequence_term(items, parts, RANKING)
    rows = [[-1.0, -0.5, -3.0], [-2.0, -2.5]]
    expected = sum(math.log(sum(math.exp(s) for s in row)) - row[0] for row in rows) / 2
    assert float(term) == pytest.approx(expected, abs=1e-5)
    assert float(extras["rank_accuracy"]) == pytest.approx(0.5)
    assert float(extras["negatives_found"]) == pytest.approx(1.5)

    risk = replace(RANKING, objective="risk")
    term, _ = model.sequence_term(items, parts, risk)
    distances = [[0.0, 1 / 3, 1.0], [0.0, 1 / 3]]
    total = 0.0
    for scores_row, risk_row in zip(rows, distances, strict=True):
        mean = sum(risk_row) / len(risk_row)
        norm = sum(math.exp(s) for s in scores_row)
        total += sum(
            math.exp(s) / norm * (d - mean) for s, d in zip(scores_row, risk_row, strict=True)
        )
    assert float(term) == pytest.approx(total / 2, abs=1e-5)


# --- the model -----------------------------------------------------------------------------


def test_without_a_weight_the_stage_is_the_joint_stage(
    vocabularies: Vocabularies, lexicon: Lexicon, reader_vocab: CharVocab
) -> None:
    ranking = replace(RANKING, weight=0.0)
    r = rig(vocabularies, lexicon, reader_vocab, ranking)
    with torch.no_grad():
        ranked = r.model(r.batch)
        joint = r.model.joint_loss(r.batch.keys)
    assert ranked.loss is not None and joint.loss is not None
    assert torch.allclose(ranked.loss, joint.loss)


def test_the_ranking_step_trains_and_its_loss_falls_on_a_fixed_batch(
    vocabularies: Vocabularies, lexicon: Lexicon, reader_vocab: CharVocab
) -> None:
    r = rig(vocabularies, lexicon, reader_vocab, count=8)
    model = r.model
    model.freeze_for_joint()
    model.train()
    optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-3)
    records = []
    for _ in range(40):
        optimiser.zero_grad()
        output = model(r.batch)
        assert output.loss is not None
        assert torch.isfinite(output.loss)
        output.loss.backward()
        optimiser.step()
        records.append(
            (
                float(output.loss.detach()),
                float(output.extras["ctc_loss"]),
                float(output.extras["rank_loss"]),
            )
        )
    for extra in ("transition_loss", "rank_accuracy", "negatives_found", "beta_probe"):
        assert extra in output.extras and torch.isfinite(output.extras[extra]), extra
    assert records[-1][0] < 0.5 * records[0][0]
    assert records[-1][1] < 0.5 * records[0][1]
    # The scored term moves with the model: it is a gradient path, not a constant.
    assert records[-1][2] != records[0][2]


def test_a_risk_step_trains_too(
    vocabularies: Vocabularies, lexicon: Lexicon, reader_vocab: CharVocab
) -> None:
    ranking = replace(RANKING, objective="risk")
    r = rig(vocabularies, lexicon, reader_vocab, ranking, count=8)
    model = r.model
    model.freeze_for_joint()
    model.train()
    output = model(r.batch)
    assert output.loss is not None
    output.loss.backward()
    assert "risk_loss" in output.extras
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.decoder.parameters()
        if p.requires_grad
    )


# --- the evaluation and the run ------------------------------------------------------------


def test_reranking_a_dump_by_the_searchs_own_weights_reproduces_its_order(
    vocabularies: Vocabularies,
    legality: Legality,
    lexicon: Lexicon,
    reader_vocab: CharVocab,
    spans: SpanVocab,
    tmp_path: Path,
) -> None:
    r = rig(vocabularies, lexicon, reader_vocab, count=8)
    decoder = KeyDecoder(
        model=r.model,
        lexicon=lexicon,
        collator=key_collator(spans, lexicon),
        legality=legality,
        options=JOINT_KEYS.options,
        device=CPU,
        token_budget=96,
        beam=10_000,
        top_k=10_000,
    )
    examples = build_examples(vocabularies, typos=True)[:8]
    records = [
        KeyRecord(
            record=number,
            keys=example.keys,
            text="".join(lexicon.characters[i] for i in example.targets),
            context=None,
        )
        for number, example in enumerate(examples)
    ]
    found = decoder.decode(records, [KeyWeights(beta=BETA, gamma=RANKING.gamma, viterbi=False)])[0]
    dump = tmp_path / "dump.jsonl"
    dump.write_text(
        "".join(
            json.dumps(
                {
                    "record": record.record,
                    "text": record.text,
                    "hypotheses": [
                        {"text": "".join(lexicon.characters[i] for i in h.ids)} for h in own[:6]
                    ],
                },
                ensure_ascii=False,
            )
            + "\n"
            for record, own in zip(records, found, strict=True)
        ),
        encoding="utf-8",
    )
    report = rerank_dump(decoder, records, dump, RANKING)
    assert report.records == len(records)
    assert report.unreachable == 0
    # S is the search's score, so the order it gives the listed hypotheses is the search's.
    assert report.reranked == report.searched
    assert report.reranked <= report.listed <= report.records
    assert report.outscored >= report.searched
    assert "gold listed" in report.render()
    # The published form of a dump is gzipped.
    packed = tmp_path / "dump.jsonl.gz"
    packed.write_bytes(gzip.compress(dump.read_bytes()))
    assert rerank_dump(decoder, records, packed, RANKING) == report


def test_a_ranking_run_trains_resumes_and_loads_for_the_search(
    corpus: tuple[Path, Path],
    vocabularies: Vocabularies,
    lexicon: Lexicon,
    reader_vocab: CharVocab,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Through ``e2e()``: S1, S2, then S4 over S2 and its resume, loaded as the search loads it."""
    monkeypatch.setattr(Vocabularies, "load", classmethod(lambda cls, *_: vocabularies))
    monkeypatch.setattr(BertConfig, "from_pretrained", classmethod(lambda cls, *_: TINY))
    parent = parent_checkpoint(lexicon, reader_vocab, tmp_path / "parent.pt")
    samples_dir, labels_dir = corpus
    slices = Slices(
        train=("test-00000.parquet",), held_out=("test-00001.parquet",), max_held_out_examples=8
    )
    training = TrainingConfig(
        max_steps=4, token_budget=160, log_every=1, checkpoint_every=2, fp16=False, seed=3
    )

    def paths(name: str) -> RunPaths:
        return RunPaths(
            samples=samples_dir,
            labels=labels_dir,
            char_table=tmp_path / "unread",
            out=tmp_path / name,
        )

    e2e(
        paths("s1"),
        slices,
        training,
        init_e2e=parent,
        keys=FRAMED,
        key_warm_up=True,
        train_fill_tower=FillTowerTraining(cross_attention=True),
    )
    warmed = tmp_path / "s1" / "checkpoint-000004.pt"
    e2e(paths("s2"), slices, training, init_e2e=warmed, keys=JOINT_KEYS)
    joint = tmp_path / "s2" / "checkpoint-000004.pt"
    ranked = e2e(
        paths("s4"), slices, training, init_e2e=joint, keys=JOINT_KEYS, key_ranking=RANKING
    )
    records = run_records(ranked.metrics)
    steps = [r for r in records if r.get("event") == "step"]
    assert len(steps) == 4
    for record in steps:
        for name in ("ctc_loss", "transition_loss", "rank_loss", "rank_accuracy", "beta_probe"):
            assert math.isfinite(record[name]), name
    provenance = next(r for r in records if r.get("event") == "provenance")
    assert provenance["e2e"]["key_ranking"] == asdict(RANKING)

    resumed = e2e(
        paths("s4-resumed"),
        slices,
        training,
        resume=tmp_path / "s4" / "checkpoint-000002.pt",
        keys=JOINT_KEYS,
        key_ranking=RANKING,
    )
    again = [r["loss"] for r in run_records(resumed.metrics) if r.get("event") == "step"]
    assert again == [r["loss"] for r in steps[2:]]

    model, step = load_key_model(tmp_path / "s4" / "checkpoint-000004.pt", lexicon, CPU, TINY)
    assert step == 4 and not model.training
    assert model.e2e_config.key_ranking == RANKING
