"""The character LM as the key ranking stage's teacher (S5, #298), on tiny CPU models.

The tests pin the term against hand computation -- the LM's conditioning and
scores built position by position, the KL and its gradient -- and the stage
around it: off, it is S4's loss and gradients; on, it trains through ``e2e()``
and a resumed leg that drops it is refused.
"""

from __future__ import annotations

import math
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from test_e2e_kernel import load_kernel
from test_e2e_model import TINY
from test_e2e_stamp import TEMPLATE, changed_lines
from test_key_ranking import RANKING, rig
from test_keys import (
    CPU,
    FRAMED,
    JOINT_KEYS,
    build_examples,
    legality_fixture,  # noqa: F401 -- the fixture, by name
    parent_checkpoint,
    reader_vocab_fixture,  # noqa: F401
    run_records,
    vocabularies_fixture,  # noqa: F401
)
from transformers import BertConfig
from typer.testing import CliRunner

from mlime.__main__ import app
from mlime.stamp import literal
from mlime.train.charlm_model import CharLm, CharLmConfig, build
from mlime.train.charlm_vocab import BOS, EOS, SEP, SPECIALS, UNK, CharVocab
from mlime.train.e2e import e2e
from mlime.train.e2e_model import KeyRankingConfig
from mlime.train.key_ranking import (
    KeyItems,
    KeyRankBatch,
    KeyRankingModel,
    by_example,
    dense_span_table,
    item_arc_costs,
    sentence_scores,
    span_rows_of,
)
from mlime.train.key_teacher import (
    Teacher,
    TeacherContext,
    TeacherVocab,
    distillation,
    teacher_context,
)
from mlime.train.keys import FillTowerTraining
from mlime.train.lexicon import Lexicon
from mlime.train.loop import TrainingConfig
from mlime.train.run import RunPaths, Slices, Vocabularies

CONTEXT_CHARS = 5
#: Chinese, Japanese and Korean, so the conditioning is checked on all three scripts.
CHARACTERS = tuple("你好吗今天气") + tuple("こんにちは") + tuple("안녕하세요")
SENTENCE_CHARACTERS = tuple("你好吗今天气")
TEACHER = TeacherVocab(
    index={character: i for i, character in enumerate(SPECIALS + tuple(sorted(CHARACTERS)))},
    context_chars=CONTEXT_CHARS,
)


def tiny_lm(vocab_size: int, seed: int = 0) -> CharLm:
    torch.manual_seed(seed)
    config = CharLmConfig(
        arch="transformer",
        embedding=16,
        hidden=32,
        layers=2,
        heads=2,
        feedforward=64,
        max_positions=64,
        dropout=0.0,
        context_chars=CONTEXT_CHARS,
    )
    return build(vocab_size, config).eval()


def reference(model: CharLm, index: dict[str, int], context: str | None, sentence: str) -> float:
    """``log p(sentence <eos> | <bos> context <sep>)``, position by position, whole alphabet."""
    ids = [BOS]
    if context:
        ids += [index.get(character, UNK) for character in context[-CONTEXT_CHARS:]]
    ids.append(SEP)
    prelude = len(ids)
    ids += [index[character] for character in sentence] + [EOS]
    with torch.no_grad():
        logp = model(torch.tensor([ids[:-1]])).log_softmax(-1)[0]
    return sum(
        float(logp[position, ids[position + 1]]) for position in range(prelude - 1, len(ids) - 1)
    )


def checkpoint_of(model: CharLm, vocab_chars: tuple[str, ...], path: Path) -> Path:
    """*model* in the char LM trainer's checkpoint layout, as ``load_model`` reads it."""
    torch.save(
        {
            "vocab": list(vocab_chars),
            "config": asdict(model.config),
            "model": model.state_dict(),
            "step": 7,
        },
        path,
    )
    return path


def sentence_rows(sentences: list[str], emit: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(map(len, sentences))
    rows = torch.zeros(len(sentences), width, dtype=torch.long)
    for row, sentence in enumerate(sentences):
        rows[row, : len(sentence)] = torch.tensor([emit[character] for character in sentence])
    return rows, torch.tensor([len(sentence) for sentence in sentences])


def test_the_lm_scores_each_sentence_under_its_context_as_ime_lm_conditions_it(
    tmp_path: Path,
) -> None:
    vocab = CharVocab(chars=SPECIALS + tuple(sorted(CHARACTERS)))
    model = tiny_lm(len(vocab))
    # A context is its last five characters (Korean, Japanese then Chinese, Japanese then
    # Chinese, and Korean again); a dropped one (None) and an empty one are <bos> <sep>;
    # a character the LM has no id for (the Korean 국 and 어) is <unk>.
    contexts = ["안녕하세요こんにちは今天", None, "こんにちは你好吗", "한국어하세요", ""]
    ids = TEACHER.index
    probe = teacher_context(contexts, TEACHER, [0], 1)
    assert probe.prelude.tolist() == [
        [BOS, *(ids[c] for c in "にちは今天"), SEP],
        [BOS, SEP, 0, 0, 0, 0, 0],
        [BOS, *(ids[c] for c in "ちは你好吗"), SEP],
        [BOS, UNK, UNK, *(ids[c] for c in "하세요"), SEP],
        [BOS, SEP, 0, 0, 0, 0, 0],
    ]
    assert probe.lengths.tolist() == [7, 2, 7, 7, 2]

    emit = {character: i for i, character in enumerate(SENTENCE_CHARACTERS)}
    owner_sentences = [
        (0, "你好"),
        (0, "今天气"),
        (1, "你好吗"),
        (2, "好"),
        (2, "今天气好吗你"),
        (3, "你好"),
        (4, "今天"),
        (1, "你"),
    ]
    sequences, lengths = sentence_rows([text for _, text in owner_sentences], emit)
    owner = torch.tensor([who for who, _ in owner_sentences])
    index = vocab.index
    teacher = Teacher(model, torch.tensor([index[c] for c in SENTENCE_CHARACTERS]))
    vocabulary = replace(TEACHER, index=index)
    span = sequences.shape[1]
    # A small chunk, so the scoring runs in several, each padded to its own longest prelude.
    context = teacher_context(contexts, vocabulary, owner.tolist(), span, positions=40)
    assert len(context.chunks) > 2
    assert sorted(context.order.tolist()) == list(range(len(owner)))
    assert context.ranks[context.order].tolist() == list(range(len(owner)))
    assert {longest for _, _, longest in context.chunks} == {2, 7}
    got = teacher.log_probs(context, owner, sequences, lengths)
    expected = torch.tensor(
        [reference(model, index, contexts[who], text) for who, text in owner_sentences]
    )
    assert torch.allclose(got, expected, atol=1e-5, rtol=1e-5), (got - expected).abs().max()
    # The padding of a sentence and the batch's other rows change nothing.
    one = teacher_context(contexts, vocabulary, [1], 3)
    alone = teacher.log_probs(one, owner[2:3], sequences[2:3, :3], lengths[2:3])
    assert torch.allclose(alone, expected[2:3], atol=1e-5)
    assert not got.requires_grad
    with pytest.raises(ValueError, match="planned for"):
        teacher.log_probs(one, owner, sequences, lengths)

    # The checkpoint loads under the same names, and its alphabet is what the collator reads.
    path = checkpoint_of(model, vocab.chars, tmp_path / "charlm.pt")
    loaded = Teacher.load(path, SENTENCE_CHARACTERS, CPU)
    assert torch.allclose(loaded.log_probs(context, owner, sequences, lengths), expected, atol=1e-5)
    read = TeacherVocab.read(path)
    assert read.index == index and read.context_chars == CONTEXT_CHARS
    assert not any(p.requires_grad for p in loaded.model.parameters()) and not loaded.model.training


def test_the_teacher_loads_on_a_runtime_without_onnx(tmp_path: Path) -> None:
    vocab = CharVocab(chars=SPECIALS + tuple(sorted(CHARACTERS)))
    path = checkpoint_of(tiny_lm(len(vocab)), vocab.chars, tmp_path / "charlm.pt")
    # A training runtime has no onnx; a `None` entry makes its import fail in the child.
    script = (
        "import sys; sys.modules['onnx'] = None\n"
        "import torch\n"
        "from mlime.train.key_teacher import Teacher\n"
        "Teacher.load(sys.argv[1], tuple(sys.argv[2]), torch.device('cpu'))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script, str(path), "".join(SENTENCE_CHARACTERS)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]


def test_a_character_the_lm_cannot_score_fails_when_the_teacher_loads(tmp_path: Path) -> None:
    vocab = CharVocab(chars=SPECIALS + tuple(sorted(CHARACTERS)))
    path = checkpoint_of(tiny_lm(len(vocab)), vocab.chars, tmp_path / "charlm.pt")
    with pytest.raises(ValueError, match="龘"):
        Teacher.load(path, (*SENTENCE_CHARACTERS, "龘"), CPU)


def test_the_distillation_is_zero_without_a_scale_and_its_gradient_is_p_minus_q() -> None:
    scores = torch.tensor([[1.0, 0.0, 0.0, -math.inf], [0.5, 0.2, -0.3, 0.1]])
    listed = torch.tensor([[True, True, True, False], [True, True, True, True]])
    teacher = torch.tensor([[-5.0, -1.0, -3.0, -2.0], [-2.0, -2.5, -0.5, -4.0]])

    slots = scores.clone().requires_grad_()
    kl, extras = distillation(slots, listed, teacher, 0.0)
    assert kl.tolist() == [0.0, 0.0] and extras["teacher_flip"] == 0.0
    kl.sum().backward()
    # p - q, where q is p to the rounding of its exp and log.
    assert slots.grad is not None and bool((slots.grad.abs() < 1e-6).all())

    slots = scores.clone().requires_grad_()
    kl, extras = distillation(slots, listed, teacher, 1.0)
    p = scores.masked_fill(~listed, -math.inf).softmax(1)
    q = (scores + teacher).masked_fill(~listed, -math.inf).softmax(1)
    by_hand = (q * (q.log() - p.log())).nan_to_num(0.0).sum(1)
    assert torch.allclose(kl, by_hand, atol=1e-6) and bool((kl > 0).all())
    kl.sum().backward()
    assert slots.grad is not None
    assert torch.isfinite(slots.grad).all()
    assert torch.allclose(slots.grad, p - q, atol=1e-6)
    assert slots.grad[0, 3] == 0.0
    # Example 0's best sentence moves from slot 0 to 1 under q, example 1's moves to slot 2:
    # both flip; and the LM alone puts the gold first in neither.
    assert float(extras["teacher_flip"]) == 1.0
    assert float(extras["teacher_gold_first"]) == 0.0
    assert float(extras["teacher_kl"]) == pytest.approx(float(kl.detach().mean()))


# --- the stage -------------------------------------------------------------------------------


def teaching(weight: float = 1.0, scale: float = 0.6) -> KeyRankingConfig:
    return replace(RANKING, teacher_weight=weight, teacher_scale=scale)


class Stub:
    """A teacher that prefers sentences of lower character ids: a fixed function of the sentence."""

    def __init__(self) -> None:
        self.calls = 0

    def log_probs(
        self,
        context: TeacherContext | None,
        owner: torch.Tensor,
        sequences: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        self.calls += 1
        live = torch.arange(sequences.shape[1]) < lengths.unsqueeze(1)
        return -(sequences * live).sum(1).float() / 4.0


def grads_of(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: p.grad.clone()
        for name, p in model.named_parameters()
        if p.requires_grad and p.grad is not None
    }


def test_without_a_teacher_weight_the_stage_is_s4s_and_never_runs_the_lm(
    vocabularies: Vocabularies, lexicon: Lexicon, reader_vocab: CharVocab
) -> None:
    off = replace(RANKING, teacher_weight=0.0, teacher_scale=9.0)
    r = rig(vocabularies, lexicon, reader_vocab, off, count=6)
    stub = Stub()
    r.model.teacher = stub  # type: ignore[assignment]
    model = r.model
    output = model(r.batch)
    assert output.loss is not None
    output.loss.backward()
    assert stub.calls == 0
    assert not any(name.startswith("teacher") for name in output.extras)
    # S4's loss is its three terms and nothing else.
    extras = output.extras
    expected = (
        extras["ctc_loss"]
        + model.keys_config.transition_weight * extras["transition_loss"]
        + off.weight * extras["rank_loss"]
    )
    assert torch.allclose(output.loss, expected, atol=1e-5)

    # The same weights under the default config give the same loss and gradients.
    plain = rig(vocabularies, lexicon, reader_vocab, RANKING, count=6)
    reference_output = plain.model(plain.batch)
    assert reference_output.loss is not None
    reference_output.loss.backward()
    assert torch.equal(output.loss, reference_output.loss)
    mine, theirs = grads_of(model), grads_of(plain.model)
    assert mine.keys() == theirs.keys() and mine
    assert all(torch.equal(mine[name], theirs[name]) for name in mine)


def test_the_teacher_term_is_zero_at_scale_zero_and_adds_nothing_to_the_gradient(
    vocabularies: Vocabularies, lexicon: Lexicon, reader_vocab: CharVocab
) -> None:
    quiet = rig(vocabularies, lexicon, reader_vocab, teaching(2.0, 0.0), count=6)
    base = rig(vocabularies, lexicon, reader_vocab, RANKING, count=6)
    stub = Stub()
    quiet.model.teacher = stub  # type: ignore[assignment]
    quiet.batch = replace(
        quiet.batch,
        teacher=teacher_context([None] * quiet.batch.size, TEACHER, [0], 1),
    )
    outputs = []
    for r in (quiet, base):
        output = r.model(r.batch)
        assert output.loss is not None
        output.loss.backward()
        outputs.append(output)
    assert stub.calls == 1
    assert float(outputs[0].extras["teacher_kl"]) == 0.0
    assert float(outputs[0].extras["teacher_flip"]) == 0.0
    assert torch.allclose(outputs[0].loss, outputs[1].loss, atol=1e-6)
    mine, theirs = grads_of(quiet.model), grads_of(base.model)
    assert all(torch.allclose(mine[name], theirs[name], rtol=1e-4, atol=1e-5) for name in theirs)


def test_a_stage_with_a_teacher_weight_and_no_teacher_or_no_context_refuses(
    vocabularies: Vocabularies, lexicon: Lexicon, reader_vocab: CharVocab
) -> None:
    r = rig(vocabularies, lexicon, reader_vocab, teaching(), count=4)
    with pytest.raises(ValueError, match="no teacher is loaded"):
        r.model(r.batch)
    r.model.teacher = Stub()  # type: ignore[assignment]
    with pytest.raises(ValueError, match="no teacher context"):
        r.model(r.batch)
    with pytest.raises(ValueError, match="positive ranking weight"):
        replace(teaching(), weight=0.0)


def fixed_list(
    r_model: KeyRankingModel, batch: KeyRankBatch
) -> tuple[KeyItems, Callable[[], torch.Tensor]]:
    """The list the step would score, drawn once, and a function of its scores, with gradient."""
    keyed = batch.keys
    with torch.no_grad():
        encoded = r_model.encode_keys(keyed, r_model.encode_context(keyed.spans))
        log_probs = r_model.ctc_log_probs(
            encoded.reshape(-1, encoded.shape[-1]).index_select(0, keyed.key_positions)
        )
        states = r_model.states_at_keys(keyed.key_mask, encoded)
        table = dense_span_table(batch, r_model.decoder.emit_ids.numel())
        spans = span_rows_of(batch)
        items = r_model.candidates(batch, states, log_probs, table, spans)

    def scores() -> torch.Tensor:
        encoded = r_model.encode_keys(keyed, r_model.encode_context(keyed.spans))
        flat = r_model.ctc_log_probs(
            encoded.reshape(-1, encoded.shape[-1]).index_select(0, keyed.key_positions)
        )
        parts = sentence_scores(
            r_model,
            r_model.states_at_keys(keyed.key_mask, encoded),
            flat,
            keyed.key_index,
            keyed.keys,
            items.owner,
            items.sequences,
            items.lengths,
            item_arc_costs(table, spans, items.owner, items.sequences),
            RANKING.gamma,
            batch.steps,
        )
        return parts.score(RANKING.beta)

    return items, scores


def test_the_term_moves_the_lists_distribution_toward_the_teachers_ordering(
    vocabularies: Vocabularies, lexicon: Lexicon, reader_vocab: CharVocab
) -> None:
    config = teaching(1.0, 2.0)
    r = rig(vocabularies, lexicon, reader_vocab, config, count=8)
    model = r.model
    model.freeze_for_joint()
    model.eval()
    stub = Stub()
    model.teacher = stub  # type: ignore[assignment]
    items, scores = fixed_list(model, r.batch)
    examples, negatives = items.valid.shape
    # The collator plans the LM's chunks for this layout of items, before the search runs.
    assert items.owner.tolist() == [
        *range(examples),
        *(owner for owner in range(examples) for _ in range(negatives)),
    ]
    listed = torch.cat([torch.ones_like(items.valid[:, :1]), items.valid], dim=1)
    lm = by_example(
        stub.log_probs(None, items.owner, items.sequences, items.lengths), examples, negatives
    )

    def listed_scores() -> torch.Tensor:
        return by_example(scores(), examples, negatives).masked_fill(~listed, -math.inf)

    before = listed_scores().detach()
    q = (before + config.teacher_scale * lm).masked_fill(~listed, -math.inf).softmax(1)
    p0 = before.softmax(1)
    assert not torch.allclose(p0, q, atol=1e-3)

    def cross_entropy() -> float:
        with torch.no_grad():
            log_p = listed_scores().log_softmax(1).masked_fill(~listed, 0.0)
        return float(-(q * log_p).sum(1).mean())

    start = cross_entropy()
    expected = (lm.masked_fill(~listed, 0.0) * p0).sum(1).mean()
    optimiser = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    for _ in range(3):
        optimiser.zero_grad()
        slots = listed_scores()
        kl, _ = distillation(slots, listed, lm, config.teacher_scale)
        kl.mean().backward()
        optimiser.step()
    assert cross_entropy() < start
    after = listed_scores().detach().softmax(1)
    moved = (lm.masked_fill(~listed, 0.0) * after).sum(1).mean()
    # The distribution's expected LM score rises: it leans toward what the LM prefers.
    assert float(moved) > float(expected)


def test_the_collator_ships_each_examples_context_as_the_lm_reads_it(
    vocabularies: Vocabularies, lexicon: Lexicon
) -> None:
    examples = build_examples(vocabularies)[:6]
    assert any(example.context for example in examples)
    vocab = TeacherVocab(
        index={c: i for i, c in enumerate(SPECIALS + tuple(lexicon.characters))}, context_chars=62
    )

    def collate(dropout: float) -> TeacherContext:
        collator = vocabularies.key_rank_collator(
            JOINT_KEYS, RANKING, context_dropout=dropout, teacher=vocab
        )
        batch = collator(examples)
        assert batch.teacher is not None
        return batch.teacher.to(CPU)

    kept = collate(0.0)
    for row, example in enumerate(examples):
        want = teacher_context([example.context], vocab, [0], 1)
        assert kept.prelude[row, : int(kept.lengths[row])].tolist() == want.prelude[0].tolist()
    dropped = collate(1.0)
    assert dropped.lengths.tolist() == [2] * len(examples)
    assert dropped.prelude.tolist() == [[BOS, SEP]] * len(examples)
    plain = vocabularies.key_rank_collator(JOINT_KEYS, RANKING, context_dropout=0.0)(examples)
    assert plain.teacher is None


# --- the run ---------------------------------------------------------------------------------


def test_the_teacher_is_on_both_ways_through_the_cli_and_the_kernel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "s5"
    argv = ["e2e", "stamp", "mlime-e2e-keys-s5-check", "--out", str(out)]
    stage = ["--init-from", "mlime-e2e-keys-s4-check", "--keys", "--typos", "--key-ranking"]
    knobs = ["--key-rank-teacher-weight", "1.0", "--key-rank-teacher-scale", "0.6"]
    result = CliRunner().invoke(app, [*argv, *stage, *knobs])
    assert result.exit_code == 0, result.output
    kernel = (out / "kernel.py").read_text()
    stamped = literal(kernel, "KEY_RANKING")
    assert stamped["teacher_weight"] == 1.0 and stamped["teacher_scale"] == 0.6
    assert "KEY_RANKING" in " ".join(changed_lines((TEMPLATE / "kernel.py").read_text(), kernel))
    refused = CliRunner().invoke(
        app, [*argv, "--init-from", "x", "--keys", "--key-rank-teacher-weight", "1"]
    )
    assert refused.exit_code != 0

    module = load_kernel(monkeypatch, tmp_path / "inputs", segment=0)
    monkeypatch.setattr(module, "KEYS", {"layers": 2, "transition_weight": 1.0})
    monkeypatch.setattr(module, "KEY_RANKING", stamped)
    # The checkpoint is the decoder's own initial one, in the mount it starts from.
    mount = tmp_path / "inputs" / "mlime-e2e-init"
    mount.mkdir(parents=True)
    (mount / module.INIT_DECODER).write_bytes(b"")
    argv_trainer = module.key_teacher_argv()
    assert argv_trainer == ["--key-rank-teacher", str(mount / module.INIT_DECODER)]
    assert "--key-rank-teacher" not in module.key_ranking_argv()
    assert module.key_ranking_argv()[-4:] == [
        "--key-rank-teacher-weight",
        "1.0",
        "--key-rank-teacher-scale",
        "0.6",
    ]
    monkeypatch.setattr(module, "KEY_RANKING", {**stamped, "teacher_weight": 0.0})
    assert module.key_teacher_argv() == []


def test_a_teacher_run_trains_resumes_and_refuses_a_leg_without_it(
    corpus: tuple[Path, Path],
    vocabularies: Vocabularies,
    lexicon: Lexicon,
    reader_vocab: CharVocab,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    e2e(
        paths("s2"),
        slices,
        training,
        init_e2e=tmp_path / "s1" / "checkpoint-000004.pt",
        keys=JOINT_KEYS,
    )
    joint = tmp_path / "s2" / "checkpoint-000004.pt"
    lm = checkpoint_of(
        tiny_lm(len(reader_vocab), seed=1), reader_vocab.chars, tmp_path / "charlm.pt"
    )
    on = teaching(0.5, 0.6)

    # The two options come together.
    with pytest.raises(ValueError, match="come together"):
        e2e(paths("x"), slices, training, init_e2e=joint, keys=JOINT_KEYS, key_ranking=on)
    with pytest.raises(ValueError, match="come together"):
        e2e(
            paths("x"),
            slices,
            training,
            init_e2e=joint,
            keys=JOINT_KEYS,
            key_ranking=RANKING,
            key_teacher=lm,
        )

    taught = e2e(
        paths("s5"),
        slices,
        training,
        init_e2e=joint,
        keys=JOINT_KEYS,
        key_ranking=on,
        key_teacher=lm,
    )
    records = run_records(taught.metrics)
    steps = [r for r in records if r.get("event") == "step"]
    assert len(steps) == 4
    for record in steps:
        for name in ("teacher_kl", "teacher_flip", "teacher_gold_first", "rank_loss"):
            assert math.isfinite(record[name]), name
    provenance = next(r for r in records if r.get("event") == "provenance")
    assert provenance["e2e"]["key_ranking"] == asdict(on)
    assert provenance["key_teacher"] == str(lm)

    # The checkpoint holds none of the LM, so S4's weights are the stage's weights.
    state = torch.load(
        tmp_path / "s5" / "checkpoint-000004.pt", map_location="cpu", weights_only=False
    )
    assert not any("teacher" in name for name in state["model"])

    resume = tmp_path / "s5" / "checkpoint-000002.pt"
    resumed = e2e(
        paths("s5-resumed"),
        slices,
        training,
        resume=resume,
        keys=JOINT_KEYS,
        key_ranking=on,
        key_teacher=lm,
    )
    again = [r["loss"] for r in run_records(resumed.metrics) if r.get("event") == "step"]
    assert again == [r["loss"] for r in steps[2:]]
    # A leg that drops the teacher is another run, and the checkpoint says which fields differ.
    with pytest.raises(ValueError, match="teacher_weight"):
        e2e(
            paths("s5-bare"),
            slices,
            training,
            resume=resume,
            keys=JOINT_KEYS,
            key_ranking=replace(on, teacher_weight=0.0),
        )
