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
import stat
import sys
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from conftest import BuiltLane, StubTokenizer
from test_e2e_model import READER_TINY, TINY, example

from mlime.train.arbitration import ReadingArbitration
from mlime.train.charlm_vocab import SPECIALS, CharVocab
from mlime.train.e2e import from_checkpoint
from mlime.train.e2e_model import E2EConfig, E2EModel, RankingConfig, emittable_ids
from mlime.train.e2e_search import Lane, decode_lanes
from mlime.train.lexicon import CandidateSpace, Lexicon
from mlime.train.loop import TrainingConfig, train
from mlime.train.ranking import RankBatch, RankCollator, RankingE2EModel, select_items
from mlime.train.samples import IGNORE_INDEX, Collator, CorpusStream, SampleBuilder
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


def rank_batch(lexicon: Lexicon, spans: SpanVocab, tokenizer: StubTokenizer) -> RankBatch:
    """The two sentences, typed out, with the first one's context kept."""
    collator = Collator(
        tokenizer, CandidateSpace(spans, lexicon.candidate_mask), context_dropout=0.0
    )
    first = example(spans, lexicon, "我爱北京", ("wo", "ai", "bei", "jing"))
    first = type(first)(**{**vars(first), "context": "北京"})
    second = example(spans, lexicon, "钟爱北京", ("zhong", "ai", "bei", "jing"))
    return RankCollator(collator, lambda keys: READINGS[keys], spans, RANKING)([first, second])


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
    batch = rank_batch(lexicon, spans, tokenizer)
    gold, other = batch.gold
    a, b = (9, 9, 9, 9), (8, 8, 8, 8)
    found = [
        [(list(gold), -1.0), (list(a), -0.5)],
        [(list(a), -0.7), (list(gold), -0.2), (list(b), -2.0)],
        [(list(other), -0.3)],
        [],
    ]
    items = select_items(batch, found, negatives=8)
    # Example 0: its gold path, the gold sentence again from the reading that
    # scored it best, then the negatives best first, each from its best reading.
    assert items.rows == [0, 2, 1, 0, 1]
    assert items.sequences == [gold, other, gold, a, b]
    assert items.slots == [[0, 2, 3, 4], [1, -1, -1, -1]]
    assert items.positive == [[True, True, False, False], [True, False, False, False]]
    assert select_items(batch, found, negatives=1).sequences == [gold, other, gold, a]


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
        paths, _ = model.path_scores(
            rows, [row for row, _, _ in pairs], [sequence for _, sequence, _ in pairs]
        )
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
