"""The key-level evaluation: decoding typed records from their keys alone (#257).

A tiny randomly initialised joint-stage model decodes real sentences of the
fixture corpus typed in full, abbreviated and mixed, end to end through the
report: every dumped hypothesis is legal by the independent verifier, the
dump round-trips, and a run stopped partway resumes to the report the
uninterrupted run gave. The gold-free collate hands the model the very states
a training batch of the same records does. An oracle stub -- emissions and
transition boosted toward the gold, in the lexicon's emission indices -- puts
the gold first, which only holds if the search's arcs name the head's rows.
The dev selection decodes every test record only after it has chosen, and
only at the choice.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import click
import pytest
import torch
from test_keys import (
    build_examples,
    key_collator,
    legality_fixture,  # noqa: F401 -- the fixture, by name
    reader_vocab_fixture,  # noqa: F401
    tiny_joint_model,
    vocabularies_fixture,  # noqa: F401
)

from mlime.legality import Legality
from mlime.train.charlm_vocab import CharVocab
from mlime.train.e2e_eval import EvalRow, Report, SliceReports, resume_dump
from mlime.train.key_eval import (
    KeyDecoder,
    KeyRecord,
    KeyRun,
    KeyWeights,
    choose,
    dump_path,
    report_keys,
    select_on_dev,
    typed_keys,
    weight_grid,
)
from mlime.train.key_model import KeyE2EModel, KeyStates
from mlime.train.key_search import LegalityVerifier
from mlime.train.keys import KeyCollator
from mlime.train.lexicon import Lexicon
from mlime.train.run import Vocabularies
from mlime.train.spans import SpanVocab

ROOT = Path(__file__).parent.parent.parent
DATA = ROOT / "crates/ime-pinyin/data"
CPU = torch.device("cpu")

#: The fixture corpus's sentences, each typed the three ways the eval twins
#: type them -- in full, as initials, and mixed -- with its context.
TYPED = [
    ("我爱北京", "北京", ("woaibeijing", "wabj", "woabeij")),
    ("钟爱北京", "钟", ("zhongaibeijing", "zabj", "zhongabj")),
    ("我爱绿钟", "绿", ("woailvzhong", "walz", "woalvzh")),
    ("中重我绿", None, ("zhongchongwolv", "zcwl", "zhongcwolv")),
]

#: Half the records are dev, so a handful of them fills both slices.
SHARE = 0.5


@pytest.fixture(name="verifier", scope="module")
def verifier_fixture() -> LegalityVerifier:
    return LegalityVerifier.load(
        char_table=DATA / "char_pinyin.tsv",
        typo_table=DATA / "typo.json",
        syllables=DATA / "syllables.txt",
    )


def eval_rows() -> list[EvalRow]:
    return [EvalRow(keys, text, context) for text, context, styles in TYPED for keys in styles]


def write_eval_set(path: Path) -> Path:
    path.write_text(
        "".join(
            json.dumps(
                {"pinyin": row.pinyin, "text": row.text, "context": row.context},
                ensure_ascii=False,
            )
            + "\n"
            for row in eval_rows()
        ),
        encoding="utf-8",
    )
    return path


def decoder(
    model: KeyE2EModel,
    lexicon: Lexicon,
    spans: SpanVocab,
    legality: Legality,
    token_budget: int = 96,
) -> KeyDecoder:
    keys = model.e2e_config.keys
    assert keys is not None
    return KeyDecoder(
        model=model.eval(),
        lexicon=lexicon,
        collator=key_collator(spans, lexicon),
        legality=legality,
        options=keys.options,
        device=CPU,
        token_budget=token_budget,
    )


def key_run(
    model: KeyE2EModel,
    lexicon: Lexicon,
    spans: SpanVocab,
    legality: Legality,
    verifier: LegalityVerifier,
    records_per_chunk: int = 5,
) -> KeyRun:
    return KeyRun(
        decoder=decoder(model, lexicon, spans, legality),
        verifier=verifier,
        eval_rows=eval_rows(),
        dev_share=SHARE,
        records_per_chunk=records_per_chunk,
    )


def dumped(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_keys_are_the_pinyin_without_separators() -> None:
    assert typed_keys("xi'an") == "xian"
    assert typed_keys("woaibeijing") == "woaibeijing"
    for bad in ("", "'", "Wo", "wo ai", "lü"):
        with pytest.raises(ValueError, match="keystrokes"):
            typed_keys(bad)


def test_a_tiny_model_decodes_real_records_and_resumes_to_the_same_report(
    lexicon: Lexicon,
    spans: SpanVocab,
    legality: Legality,
    reader_vocab: CharVocab,
    verifier: LegalityVerifier,
    tmp_path: Path,
) -> None:
    model = tiny_joint_model(lexicon, reader_vocab)
    weights = KeyWeights(beta=0.7, gamma=1.3, viterbi=False)
    whole = tmp_path / "whole"
    text = report_keys(
        key_run(model, lexicon, spans, legality, verifier), whole, [weights], slice_="all"
    )
    assert "slice: dev" in text and "slice: test" in text and "slice: all" in text
    dump = dump_path(whole, "all", weights)
    rows = dumped(dump)
    assert [row["record"] for row in rows] == list(range(len(eval_rows())))
    for row, (pinyin, expected, _, _) in zip(rows, eval_rows(), strict=True):
        assert row["text"] == expected
        assert 1 <= len(row["hypotheses"]) <= 8
        scores = [hypothesis["score"] for hypothesis in row["hypotheses"]]
        assert scores == sorted(scores, reverse=True)
        for hypothesis in row["hypotheses"]:
            assert set(hypothesis) == {
                "text",
                "score",
                "ctc",
                "transition",
                "correction",
                "cuts",
                "arcs",
            }
            assert hypothesis["cuts"][-1] == len(pinyin)
            assert len(hypothesis["cuts"]) == len(hypothesis["text"])
            cost = verifier.cost(pinyin, hypothesis["text"], legality_options(model))
            assert cost is not None, (pinyin, hypothesis["text"])
            assert cost <= hypothesis["correction"] + 1e-6
            assert math.isfinite(hypothesis["ctc"]) and hypothesis["transition"] <= 0.0
    # The dump round-trips: what a resume reads back is what was written.
    assert resume_dump(dump) == {
        row["record"]: [hypothesis["text"] for hypothesis in row["hypotheses"]] for row in rows
    }

    # A run stopped after a few records, mid-line, resumes to the same report.
    halted = tmp_path / "halted"
    halted.mkdir()
    lines = dump.read_text(encoding="utf-8").splitlines(keepends=True)
    dump_path(halted, "all", weights).write_text(
        "".join(lines[:5]) + lines[5][: len(lines[5]) // 2], encoding="utf-8"
    )
    resumed = report_keys(
        key_run(model, lexicon, spans, legality, verifier), halted, [weights], slice_="all"
    )
    assert resumed == text
    again = dumped(dump_path(halted, "all", weights))
    assert [row["record"] for row in again] == [row["record"] for row in rows]
    for first, second in zip(rows, again, strict=True):
        assert [h["text"] for h in first["hypotheses"]] == [h["text"] for h in second["hypotheses"]]
        for a, b in zip(first["hypotheses"], second["hypotheses"], strict=True):
            assert a["score"] == pytest.approx(b["score"], abs=1e-4)


def legality_options(model: KeyE2EModel) -> Any:
    keys = model.e2e_config.keys
    assert keys is not None
    return keys.options


def test_an_illegal_hypothesis_is_refused_with_its_record(
    lexicon: Lexicon,
    spans: SpanVocab,
    legality: Legality,
    reader_vocab: CharVocab,
    verifier: LegalityVerifier,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tiny_joint_model(lexicon, reader_vocab)
    run = key_run(model, lexicon, spans, legality, verifier)
    original = KeyDecoder.decode

    def swapped(self: KeyDecoder, records: Sequence[KeyRecord], grid: Sequence[KeyWeights]):
        found = original(self, records, grid)
        # Every hypothesis of the first record reads 中中中, which wabj is not.
        wrong = lexicon.indices("中中中")
        assert wrong is not None
        for point in found:
            point[0] = [
                type(hypothesis)(**{**vars(hypothesis), "ids": wrong}) for hypothesis in point[0]
            ]
        return found

    monkeypatch.setattr(KeyDecoder, "decode", swapped)
    with pytest.raises(ValueError, match=r"record 0 \('woaibeijing'.*'中中中'.*not a legal"):
        run.evaluate("all", [KeyWeights(1.0, 1.0, False)], [tmp_path / "dump.jsonl"])


def test_the_gold_free_collate_gives_the_training_batchs_key_states(
    vocabularies: Vocabularies,
    lexicon: Lexicon,
    spans: SpanVocab,
    reader_vocab: CharVocab,
) -> None:
    model = tiny_joint_model(lexicon, reader_vocab).eval()
    examples = build_examples(vocabularies)[:8]
    assert any(example.context for example in examples)
    assert any(not example.context for example in examples)
    trained = key_collator(spans, lexicon)(examples)
    searched = key_collator(spans, lexicon).inputs(
        [example.keys for example in examples], [example.context for example in examples]
    )
    for name, tensor in vars(trained.inputs).items():
        assert torch.equal(tensor, getattr(searched, name)), name
    with torch.no_grad():
        held = model.key_states(trained.inputs)
        free = model.key_states(searched)
    torch.testing.assert_close(free.states, held.states)
    assert torch.equal(free.mask, held.mask)
    # Context off drops every context, as a dropout of one does in training.
    dropped = key_collator(spans, lexicon, dropout=1.0).inputs(
        [example.keys for example in examples], [example.context for example in examples]
    )
    assert not bool(dropped.has_context.any())
    with pytest.raises(ValueError, match="no keys"):
        key_collator(spans, lexicon).inputs(["wo", ""], [None, None])


def test_an_oracle_puts_the_gold_first_in_the_heads_own_indices(
    lexicon: Lexicon,
    spans: SpanVocab,
    legality: Legality,
    reader_vocab: CharVocab,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Emissions and transition boosted toward each record's gold, by lexicon emission index.

    The CTC stub favours the blank and every gold character at every key, with
    no position in it; the transition stub favours the gold's next character
    after each gold prefix. Only a search whose arcs are the head's rows can
    assemble the gold from that.
    """
    model = tiny_joint_model(lexicon, reader_vocab)
    run = decoder(model, lexicon, spans, legality, token_budget=10_000)
    records = [
        KeyRecord(record=index, keys=keys, text=text, context=context)
        for index, (keys, text, context, _) in enumerate(eval_rows())
    ]
    golds = [lexicon.indices(record.text) for record in records]
    assert all(gold is not None for gold in golds)
    width = lexicon.size

    def emissions(states: torch.Tensor) -> torch.Tensor:
        assert states.shape[0] == len(records), "the oracle reads one group of every record"
        logits = torch.zeros(states.shape[0], states.shape[1], width + 1)
        logits[..., width] = 4.0
        for row, gold in enumerate(golds):
            assert gold is not None
            logits[row, :, list(gold)] = 4.0
        return logits.log_softmax(dim=-1)

    def transition(
        keys: KeyStates, rows: torch.Tensor, prefixes: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        logits = torch.zeros(rows.shape[0], width)
        for slot, (row, length) in enumerate(zip(rows.tolist(), lengths.tolist(), strict=True)):
            gold = golds[row]
            assert gold is not None
            if prefixes[slot, :length].tolist() == list(gold[:length]) and length < len(gold):
                logits[slot, gold[length]] = 10.0
        return logits.log_softmax(dim=-1)

    monkeypatch.setattr(model, "ctc_log_probs", emissions)
    monkeypatch.setattr(model, "transition_log_probs", transition)
    [found] = run.decode(records, [KeyWeights(beta=1.0, gamma=1.0, viterbi=False)])
    for record, gold, hypotheses in zip(records, golds, found, strict=True):
        assert hypotheses, record
        assert hypotheses[0].ids == gold, (record.keys, record.text)


def test_the_dev_winner_is_top1_then_topk_then_the_smaller_beta() -> None:
    def report(top1: int, topk: int) -> Report:
        return Report(top_k=8, records=10, top1_hits=top1, topk_hits=topk)

    low, high = KeyWeights(0.5, 1.0, False), KeyWeights(2.0, 1.0, False)
    viterbi = KeyWeights(0.5, 1.0, True)
    assert choose([(low, report(3, 5)), (high, report(4, 4))]) == high
    assert choose([(high, report(4, 6)), (low, report(4, 5))]) == high
    assert choose([(high, report(4, 5)), (low, report(4, 5))]) == low
    assert choose([(low, report(4, 5)), (viterbi, report(4, 5))]) == low
    with pytest.raises(ValueError, match="repeats"):
        weight_grid([1.0, 1.0], [1.0], [False])
    with pytest.raises(ValueError, match="not negative"):
        KeyWeights(-0.1, 1.0, False)


def test_the_selection_decodes_test_only_after_choosing_and_only_at_the_choice(
    lexicon: Lexicon,
    spans: SpanVocab,
    legality: Legality,
    reader_vocab: CharVocab,
    verifier: LegalityVerifier,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tiny_joint_model(lexicon, reader_vocab)
    run = key_run(model, lexicon, spans, legality, verifier)
    dev = set(SliceReports(run.eval_rows, "dev", SHARE, 8).indices)
    test = set(SliceReports(run.eval_rows, "test", SHARE, 8).indices)
    assert dev and test
    calls: list[tuple[set[int], tuple[KeyWeights, ...]]] = []
    original = KeyDecoder.decode

    def spy(self: KeyDecoder, records: Sequence[KeyRecord], grid: Sequence[KeyWeights]):
        calls.append(({record.record for record in records}, tuple(grid)))
        return original(self, records, grid)

    monkeypatch.setattr(KeyDecoder, "decode", spy)
    grid = weight_grid([0.5, 2.0], [0.5, 1.0], [False, True])
    text = report_keys(run, tmp_path, grid, slice_=None)
    selection = json.loads((tmp_path / "selection.json").read_text(encoding="utf-8"))
    chosen = KeyWeights(
        selection["chosen"]["beta"],
        selection["chosen"]["gamma"],
        selection["chosen"]["combination"] == "viterbi",
    )
    reports = [
        (
            KeyWeights(point["beta"], point["gamma"], point["combination"] == "viterbi"),
            Report(**{key: point[key] for key in vars(Report(8))}),
        )
        for point in selection["grid"]
    ]
    assert choose(reports) == chosen
    first_test = next(index for index, (records, _) in enumerate(calls) if records & test)
    assert set().union(*(records for records, _ in calls[:first_test])) == dev
    for records, points in calls[:first_test]:
        assert records <= dev and points == tuple(grid)
    for records, points in calls[first_test:]:
        assert records <= test and points == (chosen,)
    assert set().union(*(records for records, _ in calls[first_test:])) == test
    assert "<- chosen" in text and "slice: test" in text
    assert selection["test_records"] == len(test) and selection["dev_records"] == len(dev)
    assert selection["test_seconds_estimate"] > 0.0
    assert not dump_path(tmp_path, "test", grid[0] if grid[0] != chosen else grid[1]).exists()

    # Selection alone never touches test, and a resumed sweep decodes nothing again.
    calls.clear()
    alone = tmp_path / "alone"
    first = select_on_dev(run, grid, alone)
    assert calls and all(records <= dev for records, _ in calls)
    assert not dump_path(alone, "test", chosen).exists()
    calls.clear()
    again = select_on_dev(run, grid, alone)
    assert not calls
    assert again.chosen == first.chosen == chosen
    # Nothing was decoded to measure; the estimate is the earlier sweep's.
    assert again.test_seconds_estimate == first.test_seconds_estimate


def test_the_collator_refuses_mismatched_contexts(lexicon: Lexicon, spans: SpanVocab) -> None:
    collator: KeyCollator = key_collator(spans, lexicon)
    with pytest.raises(ValueError, match="contexts"):
        collator.inputs(["wo"], [None, None])
    with pytest.raises(ValueError, match="empty"):
        collator.inputs([], [])


def test_a_limit_cuts_each_slice_to_its_first_records(
    lexicon: Lexicon,
    spans: SpanVocab,
    legality: Legality,
    reader_vocab: CharVocab,
    verifier: LegalityVerifier,
    tmp_path: Path,
) -> None:
    model = tiny_joint_model(lexicon, reader_vocab)
    run = replace(key_run(model, lexicon, spans, legality, verifier), limit=2)
    for name in ("dev", "test"):
        held = run.indices(name)
        assert len(held) == 2 and held == sorted(held)
    report_keys(run, tmp_path, [KeyWeights(1.0, 1.0, False)], slice_="all")
    written = (tmp_path / "all-beta1-gamma1-sum.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(written) == 2
    assert run.indices("all") == [0, 1]
    assert run.decoder.times.records == 2


def test_a_decoder_refuses_an_automaton_over_another_emission_order(
    lexicon: Lexicon,
    spans: SpanVocab,
    legality: Legality,
    reader_vocab: CharVocab,
) -> None:
    """The index-space decision: the automaton must be built over the lexicon's own order."""
    model = tiny_joint_model(lexicon, reader_vocab)
    reordered = Legality(
        tuple(reversed(lexicon.characters)),
        {
            character: [syllable for syllable, ids in lexicon.homophones.items() if index in ids]
            for index, character in enumerate(lexicon.characters)
        },
        legality.corrections,
    )
    with pytest.raises(ValueError, match="different set"):
        decoder(model, lexicon, spans, reordered)


def test_only_a_joint_stage_checkpoint_loads_for_the_search(
    lexicon: Lexicon, reader_vocab: CharVocab, tmp_path: Path
) -> None:
    from dataclasses import asdict

    from test_e2e_model import TINY
    from test_keys import parent_checkpoint, tiny_key_model

    from mlime.train.e2e import load_key_model, recorded_config

    joint = tiny_joint_model(lexicon, reader_vocab)
    path = tmp_path / "s2.pt"
    torch.save({"e2e": asdict(joint.config), "model": joint.state_dict(), "step": 11}, path)
    assert recorded_config(path) == joint.e2e_config
    loaded, step = load_key_model(path, lexicon, CPU, bert_config=TINY)
    assert step == 11 and not loaded.training
    for name, tensor in joint.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor), name

    warm = tiny_key_model(lexicon, reader_vocab)
    s1 = tmp_path / "s1.pt"
    torch.save({"e2e": asdict(warm.config), "model": warm.state_dict(), "step": 3}, s1)
    with pytest.raises(ValueError, match="warm-up"):
        load_key_model(s1, lexicon, CPU, bert_config=TINY)
    with pytest.raises(ValueError, match="span model"):
        load_key_model(
            parent_checkpoint(lexicon, reader_vocab, tmp_path / "span.pt"),
            lexicon,
            CPU,
            bert_config=TINY,
        )


def test_a_prior_keeps_a_points_old_name_at_zero_and_old_selections_read_back() -> None:
    plain = KeyWeights(1.0, 2.0, False)
    assert plain.prior == 0.0
    assert plain.name == "beta1-gamma2-sum"
    assert KeyWeights(1.0, 2.0, True).name == "beta1-gamma2-viterbi"
    assert KeyWeights(1.0, 2.0, False, 4.0).name == "beta1-gamma2-sum-prior4"
    assert KeyWeights(1.0, 2.0, True, 0.5).name == "beta1-gamma2-viterbi-prior0.5"
    assert plain.as_record() == {"beta": 1.0, "gamma": 2.0, "combination": "sum", "prior": 0.0}
    # A selection written before the arcs has no prior in any row: 0.
    old = {"beta": 1.0, "gamma": 2.0, "combination": "viterbi"}
    assert KeyWeights.from_record(old) == KeyWeights(1.0, 2.0, True)
    for point in (plain, KeyWeights(0.5, 1.0, True, 12.0)):
        assert KeyWeights.from_record(point.as_record()) == point
    with pytest.raises(ValueError, match="prior"):
        KeyWeights(1.0, 1.0, False, -1.0)
    with pytest.raises(ValueError, match="prior"):
        KeyWeights(1.0, 1.0, False, math.inf)
    # The grid sweeps the prior between gamma and the combination; none given is today's grid.
    assert weight_grid([1.0, 2.0], [1.0], [False, True]) == [
        KeyWeights(beta, 1.0, viterbi) for beta in (1.0, 2.0) for viterbi in (False, True)
    ]
    grid = weight_grid([1.0], [1.0], [False], [0.0, 4.0, 8.0])
    assert [point.name for point in grid] == [
        "beta1-gamma1-sum",
        "beta1-gamma1-sum-prior4",
        "beta1-gamma1-sum-prior8",
    ]
    with pytest.raises(ValueError, match="repeats"):
        weight_grid([1.0], [1.0], [False], [4.0, 4.0])


def test_the_dev_sweep_over_the_prior_decodes_with_arcs_and_counts_them(
    lexicon: Lexicon,
    spans: SpanVocab,
    legality: Legality,
    reader_vocab: CharVocab,
    verifier: LegalityVerifier,
    tmp_path: Path,
) -> None:
    from mlime.train.key_arcs import ArcLexicon

    model = tiny_joint_model(lexicon, reader_vocab)
    table = tmp_path / "knowledge.tsv"
    bigrams = ["北京", "爱北", "我爱", "钟爱", "绿钟", "我绿", "中重", "重我"]
    table.write_text(
        "".join(f"{word}\tidiom\n" for word in bigrams) + "我\tidiom\n龘龘\tidiom\n北京\tperson\n",
        encoding="utf-8",
    )
    arcs = ArcLexicon.load(table, ["idiom"], lexicon.characters)
    assert (arcs.size, arcs.skipped_short, arcs.skipped_outside) == (len(bigrams), 1, 1)
    plain_run = key_run(model, lexicon, spans, legality, verifier)
    run = replace(plain_run, decoder=replace(plain_run.decoder, arcs=arcs, arc_width=4))
    grid = weight_grid([1.0], [1.0], [False], [0.0, 6.0])

    # A prior without a lexicon is refused, not ignored.
    with pytest.raises(ValueError, match="word arcs"):
        select_on_dev(plain_run, grid, tmp_path / "refused")

    selection = select_on_dev(run, grid, tmp_path / "sweep")
    assert [weights.name for weights, _ in selection.grid] == [
        "beta1-gamma1-sum",
        "beta1-gamma1-sum-prior6",
    ]
    assert dump_path(tmp_path / "sweep", "dev", grid[0]).name == "dev-beta1-gamma1-sum.jsonl"
    record = json.loads((tmp_path / "sweep" / "selection.json").read_text(encoding="utf-8"))
    assert [point["prior"] for point in record["grid"]] == [0.0, 6.0]
    assert record["chosen"]["prior"] in (0.0, 6.0)
    assert KeyWeights.from_record(record["chosen"]) == selection.chosen
    times = run.decoder.times
    assert times.arcs_offered > 0 and times.arcs_landed > 0 and times.arcs > 0.0
    assert times.arc_rows >= 0 and "arcs" in times.seconds and "arcs_offered" not in times.seconds
    # The prior-0 point is the decode without arcs, row for row; the other lists its arcs.
    plain = dumped(dump_path(tmp_path / "sweep", "dev", grid[0]))
    reference = report_keys(plain_run, tmp_path / "ref", [grid[0]], slice_="dev")
    assert dumped(dump_path(tmp_path / "ref", "dev", grid[0])) == plain
    assert "slice: dev" in reference
    assert all(h["arcs"] == 0 for row in plain for h in row["hypotheses"])
    boosted = dumped(dump_path(tmp_path / "sweep", "dev", grid[1]))
    assert any(h["arcs"] > 0 for row in boosted for h in row["hypotheses"])
    table_text = selection.render()
    assert "prior" in table_text and "6" in table_text

    # An old selection (no prior in its rows) still resumes: the dumps are the progress.
    for point in record["grid"] + [record["chosen"]]:
        del point["prior"]
    (tmp_path / "sweep" / "selection.json").write_text(json.dumps(record), encoding="utf-8")
    again = select_on_dev(run, grid, tmp_path / "sweep")
    assert again.chosen == selection.chosen
    assert again.test_seconds_estimate == selection.test_seconds_estimate


def error(output: str) -> str:
    """A refusal's text with its styling and the error box's borders and wrapping taken out."""
    return " ".join(click.unstyle(output).replace("│", " ").split())


def test_the_keys_command_sweeps_the_prior_and_refuses_it_without_a_lexicon(
    tmp_path: Path,
) -> None:
    from typer.testing import CliRunner

    from mlime.__main__ import app

    base = [
        "eval",
        "keys",
        "--checkpoint",
        str(tmp_path / "missing.pt"),
        "--eval-set",
        str(tmp_path / "set.jsonl"),
        "--dump-dir",
        str(tmp_path / "dump"),
        "--select-on-dev",
    ]
    refused = CliRunner().invoke(app, [*base, "--arc-prior", "0", "--arc-prior", "4"])
    assert refused.exit_code == 2, refused.output
    assert "--arcs" in error(refused.output)
    fixed = [*base[:-1], "--slice", "dev", "--arcs", str(tmp_path / "k.tsv")]
    sweeping = CliRunner().invoke(app, [*fixed, "--arc-prior", "4", "--arc-prior", "8"])
    assert sweeping.exit_code == 2 and "one --arc-prior" in error(sweeping.output)
    narrow = CliRunner().invoke(app, [*base, "--arc-prior", "0", "--arc-width", "0"])
    assert narrow.exit_code == 2 and "--arc-width" in error(narrow.output)
    # Prior 0 needs no lexicon to parse; the missing checkpoint is what stops it.
    parsed = CliRunner().invoke(app, [*base, "--arc-prior", "0"])
    assert "--arc-prior" not in error(parsed.output) and "--arcs" not in error(parsed.output)
