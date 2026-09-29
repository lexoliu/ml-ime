"""The end-to-end model: identity at initialisation, then the things it must learn.

The central fact the design leans on is that a fresh e2e model *is* its parts:
with every gate and the span-input projection at zero, the decoder's step
log-probabilities are exactly the reader's renormalised over the candidates,
and the encoder's head is exactly route A's. That is testable, and it is the
first test here, because everything after it -- overfitting, decoding, resuming
-- rests on the composition being exact rather than approximate.

The tiny models are the same shape as the loop tests': a four-layer BERT the
size of a hand and a two-block transformer reader over the fixture lexicon's
nine characters.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from conftest import READINGS, StubTokenizer
from transformers import BertConfig

from mlime.gui import DEV_SHARE, digest
from mlime.train.arbitration import ReadingArbitration
from mlime.train.charlm_model import CharLmConfig, TransformerCharLm
from mlime.train.charlm_vocab import BOS, SPECIALS, CharVocab
from mlime.train.e2e import from_checkpoint, initialise
from mlime.train.e2e_eval import (
    RANK_SCALE,
    Observation,
    Report,
    decode_record,
    read_eval_records,
)
from mlime.train.e2e_model import E2EConfig, E2EModel, emittable_ids
from mlime.train.emit import CandidateIndex, LatticePath, LatticeRecord
from mlime.train.lexicon import CandidateSpace, Lexicon, build_lexicon
from mlime.train.loop import TrainingConfig, train
from mlime.train.model import RouteAConfig, RouteAModel, restricted_cross_entropy
from mlime.train.samples import (
    IGNORE_INDEX,
    BaseTokenizer,
    Collator,
    CorpusStream,
    SampleBuilder,
    TrainingExample,
)
from mlime.train.spans import SpanVocab

TINY = BertConfig(
    vocab_size=256,
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=64,
    max_position_embeddings=64,
)

#: A reader the size of the fixture: two blocks over a sixteen-wide model.
READER_TINY = CharLmConfig(
    arch="transformer",
    embedding=8,
    hidden=16,
    layers=2,
    heads=4,
    feedforward=32,
    max_positions=64,
    dropout=0.0,
)


@pytest.fixture(name="reader_vocab")
def reader_vocab_fixture(lexicon: Lexicon) -> CharVocab:
    """The reader's alphabet over the fixture's characters."""
    return CharVocab(chars=SPECIALS + tuple(lexicon.characters))


@pytest.fixture(name="e2e_config")
def e2e_config_fixture() -> E2EConfig:
    """The tiny reader, one cross-attended encoder layer, the default aux weight."""
    return E2EConfig(cross_attention_layers=1, decoder=READER_TINY)


def tiny_e2e(lexicon: Lexicon, vocab: CharVocab, config: E2EConfig) -> E2EModel:
    """A randomly initialised e2e model, seeded so tests can be compared."""
    torch.manual_seed(0)
    return E2EModel.compose(TINY, lexicon, config, len(vocab), emittable_ids(lexicon, vocab))


def example(
    spans: SpanVocab, lexicon: Lexicon, text: str, pinyin: tuple[str, ...]
) -> TrainingExample:
    """One training example: the typed spans, their ids, and the targets."""
    return TrainingExample(
        id=f"ex-{pinyin[0]}",
        spans=pinyin,
        span_ids=tuple(spans.id(span) for span in pinyin),
        targets=tuple(lexicon.index(character) for character in text),
        context=None,
    )


def test_the_emission_logits_have_the_frame_route_a_uses(
    lexicon: Lexicon,
    spans: SpanVocab,
    tokenizer: StubTokenizer,
    reader_vocab: CharVocab,
    e2e_config: E2EConfig,
) -> None:
    model = tiny_e2e(lexicon, reader_vocab, e2e_config).eval()
    batch = Collator(tokenizer, CandidateSpace(spans, lexicon.candidate_mask))(
        [example(spans, lexicon, "我爱北京", ("wo", "ai", "bei", "jing"))]
    )
    with torch.no_grad():
        output = model(batch)
        frame = model.scores(batch)
    # The frame's last axis is the position's own candidate list: its columns
    # count is the widest span's, not the emittable alphabet's.
    assert frame.shape == (1, batch.input_ids.shape[1], batch.candidate_ids.shape[2])
    assert output.loss is not None and output.loss.ndim == 0
    assert set(output.extras) == {"decoder_loss", "encoder_loss"}
    scored = batch.targets != IGNORE_INDEX
    predicted = model.predictions(frame, batch)
    assert bool(model.candidate_mask[batch.span_ids[scored], predicted[scored]].all())
    assert int((predicted == IGNORE_INDEX).sum()) == int((~scored).sum())
    # Training reads only loss and extras; there is no logits field to misread.
    assert not hasattr(output, "logits")


def test_the_vectorised_loss_is_the_masked_full_vocabulary_loss(
    lexicon: Lexicon,
    spans: SpanVocab,
    tokenizer: StubTokenizer,
    reader_vocab: CharVocab,
    e2e_config: E2EConfig,
) -> None:
    """forward()'s [P, E] losses equal restricted CE on wide logits."""
    model = tiny_e2e(lexicon, reader_vocab, e2e_config).eval()
    batch = Collator(tokenizer, CandidateSpace(spans, lexicon.candidate_mask))(
        [
            example(spans, lexicon, "我爱北京", ("wo", "ai", "bei", "jing")),
            example(spans, lexicon, "钟爱北京", ("zhong", "ai", "bei", "jing")),
        ]
    )
    with torch.no_grad():
        output = model(batch)
        hidden = model.encode(batch)
        # The encoder side, exactly as route A computes it.
        wide_encoder = model.head(hidden)
        encoder_loss = model.loss(wide_encoder, batch)
        # The decoder side: the teacher-forced features over the whole
        # emittable alphabet at once, scored by the masked loss.
        width = batch.targets.shape[1]
        spans_v = hidden[:, 1 : width - 1]
        positions = batch.span_positions[:, 1 : width - 1]
        previous = torch.cat(
            [
                torch.full((batch.size, 1), BOS, dtype=torch.long),
                reader_ids(model, batch.targets[:, 1 : width - 1])[:, :-1],
            ],
            dim=1,
        )
        features = model.decoder(previous, spans_v, positions)
        emit_weight = model.decoder.reader.embed.weight.index_select(0, model.decoder.emit_ids)
        wide_decoder = model.decoder.reader.project(features) @ emit_weight.T
        where = (batch.targets[:, 1 : width - 1] != IGNORE_INDEX).reshape(-1).nonzero()
        where = where.squeeze(1)
        decoder_loss = restricted_cross_entropy(
            wide_decoder.reshape(-1, lexicon.size).index_select(0, where),
            batch.targets[:, 1 : width - 1].reshape(-1).index_select(0, where),
            model.candidate_mask.index_select(
                0, batch.span_ids[:, 1 : width - 1].reshape(-1).index_select(0, where)
            ),
            model.config.label_smoothing,
        )
    assert output.extras is not None
    assert torch.allclose(output.extras["decoder_loss"], decoder_loss, atol=1e-6)
    assert torch.allclose(output.extras["encoder_loss"], encoder_loss, atol=1e-6)


def test_with_zero_gates_the_decoder_is_the_reader(
    lexicon: Lexicon,
    spans: SpanVocab,
    tokenizer: StubTokenizer,
    reader_vocab: CharVocab,
    e2e_config: E2EConfig,
) -> None:
    model = tiny_e2e(lexicon, reader_vocab, e2e_config).eval()
    # The reader under test is the decoder's own transformer, weight for weight.
    reader = TransformerCharLm(len(reader_vocab), READER_TINY).eval()
    reader.load_state_dict(model.decoder.reader.state_dict())

    batch = Collator(tokenizer, CandidateSpace(spans, lexicon.candidate_mask))(
        [example(spans, lexicon, "我爱北京", ("wo", "ai", "bei", "jing"))]
    )
    with torch.no_grad():
        # The reader's own log-probabilities over the same teacher-forced
        # prefix, restricted to the emission ids and renormalised over each
        # position's candidates.
        targets = batch.targets[:, 1 : batch.targets.shape[1] - 1]
        tokens = torch.cat(
            [torch.full((1, 1), BOS, dtype=torch.long), reader_ids(model, targets)[:, :-1]],
            dim=1,
        )
        reader_logits = reader.logits(reader.features(tokens))
        emit = model.decoder.emit_ids
        frame = model.scores(batch)
        floor = torch.finfo(frame.dtype).min
        for position in range(targets.shape[1]):
            count = int(batch.candidate_counts[0, position + 1])
            cand_ids = batch.candidate_ids[0, position + 1, :count]
            wanted = reader_logits[0, position].index_select(0, emit.index_select(0, cand_ids))
            got = frame[0, position + 1]
            # Raw equality over the position's candidates, and therefore
            # equality of the log-probabilities renormalised over them.
            assert torch.allclose(wanted, got[:count], atol=1e-5), (
                f"position {position} differs from the reader's"
            )
            assert bool((got[count:] == floor).all())
            assert torch.allclose(
                wanted.log_softmax(dim=-1), got[:count].log_softmax(dim=-1), atol=1e-5
            )


def reader_ids(model: E2EModel, targets: torch.Tensor) -> torch.Tensor:
    """The reader-alphabet ids of a targets row, padding where there is no target."""
    return model.decoder.emit_ids[targets.clamp(min=0)]


def test_with_zero_gates_the_encoder_is_route_a(
    lexicon: Lexicon,
    spans: SpanVocab,
    tokenizer: StubTokenizer,
    reader_vocab: CharVocab,
    e2e_config: E2EConfig,
) -> None:
    route = RouteAModel.from_config(TINY, lexicon, RouteAConfig(cross_attention_layers=1)).eval()
    model = tiny_e2e(lexicon, reader_vocab, e2e_config).eval()
    result = model.load_state_dict(route.state_dict(), strict=False)
    assert result.unexpected_keys == []
    assert all(name.startswith("decoder.") for name in result.missing_keys)

    batch = Collator(tokenizer, CandidateSpace(spans, lexicon.candidate_mask))(
        [example(spans, lexicon, "我爱北京", ("wo", "ai", "bei", "jing"))]
    )
    with torch.no_grad():
        wanted = route.scores(batch)
        got = model.head(model.encode(batch))
    assert torch.equal(wanted, got)


def test_the_step_path_matches_the_teacher_forced_path(
    lexicon: Lexicon,
    spans: SpanVocab,
    tokenizer: StubTokenizer,
    reader_vocab: CharVocab,
    e2e_config: E2EConfig,
) -> None:
    model = tiny_e2e(lexicon, reader_vocab, e2e_config).eval()
    decoder = model.decoder
    for cross in decoder.gated_layers():
        cross.gate.data.fill_(0.25)
    decoder.span_input.weight.data.normal_(0, 0.1)

    batch = Collator(tokenizer, CandidateSpace(spans, lexicon.candidate_mask))(
        [example(spans, lexicon, "我爱北京", ("wo", "ai", "bei", "jing"))]
    )
    with torch.no_grad():
        hidden = model.encode(batch)[0]
        n = batch.targets.shape[1] - 2
        spans_v, mask = hidden[1 : n + 1], batch.span_positions[0, 1 : n + 1]
        tokens = torch.cat(
            [
                torch.full((1, 1), BOS, dtype=torch.long),
                reader_ids(model, batch.targets[:, 1 : n + 1])[:, :-1],
            ],
            dim=1,
        )
        features = decoder(tokens, spans_v.unsqueeze(0), mask.unsqueeze(0))
        # Per position the two paths agree over that span's candidates.
        candidates = [
            batch.candidate_ids[
                0,
                position + 1,
                : int(batch.candidate_counts[0, position + 1]),
            ]
            for position in range(n)
        ]
        wanted = [
            decoder.logits_over(features, cand)[0, position]
            for position, cand in enumerate(candidates)
        ]

        encoder_kv = [cross.kv(spans_v.unsqueeze(0)) for cross in decoder.gated_layers()]
        layers, heads = decoder.reader.config.layers, decoder.reader.config.heads
        head_dim = decoder.reader.config.hidden // heads
        keys = torch.zeros(1, layers, heads, 0, head_dim)
        values = keys.clone()
        got = []
        for position in range(n):
            step_logits, (keys, values) = decoder.step(
                tokens[:, position],
                spans_v[position].unsqueeze(0),
                (keys, values),
                encoder_kv,
                candidates=candidates[position],
            )
            got.append(step_logits[0])
        for position, (g, w) in enumerate(zip(got, wanted, strict=True)):
            assert torch.allclose(g, w, atol=1e-5), f"position {position} differs"


def test_a_missing_character_in_the_reader_alphabet_fails_loudly(lexicon: Lexicon) -> None:
    vocab = CharVocab(chars=SPECIALS + tuple(lexicon.characters[:-1]))
    with pytest.raises(ValueError, match="no ids"):
        emittable_ids(lexicon, vocab)


def test_two_records_overfit_to_near_zero_loss(
    lexicon: Lexicon,
    spans: SpanVocab,
    tokenizer: StubTokenizer,
    reader_vocab: CharVocab,
    e2e_config: E2EConfig,
) -> None:
    model = tiny_e2e(lexicon, reader_vocab, e2e_config).train()
    collator = Collator(tokenizer, CandidateSpace(spans, lexicon.candidate_mask))
    batch = collator(
        [
            example(spans, lexicon, "我爱北京", ("wo", "ai", "bei", "jing")),
            example(spans, lexicon, "钟爱北京", ("zhong", "ai", "bei", "jing")),
        ]
    )
    optimiser = torch.optim.AdamW(model.parameter_groups(1e-3, 3e-3))
    for _ in range(400):
        loss = model(batch).loss
        assert loss is not None
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    assert model(batch).loss < 0.1


#: The two sentences of the #86 pronoun failure, plus what they need: 她/他
#: share ``ta``, 是 is ``shi``, 姐 ``jie`` and 哥 ``ge`` -- so the only
#: difference between the records is the tail of the typed span.
PRONOUN_READINGS = {
    **READINGS,
    "她": ("ta",),
    "他": ("ta",),
    "是": ("shi",),
    "姐": ("jie",),
    "哥": ("ge",),
}

PRONOUN_SENTENCES = (
    ("她是我姐姐", ("ta", "shi", "wo", "jie", "jie"), "tashiwojiejie"),
    ("他是我哥哥", ("ta", "shi", "wo", "ge", "ge"), "tashiwogege"),
)


def pronoun_lexicon(spans: SpanVocab) -> Lexicon:
    """The fixture lexicon plus the five characters the pronoun records use."""
    vocabulary = {
        character: index + 100 for index, character in enumerate(sorted(PRONOUN_READINGS))
    }
    return build_lexicon(PRONOUN_READINGS, vocabulary, spans)


def pronoun_lattice(spans: SpanVocab, lexicon: Lexicon) -> list[LatticeRecord]:
    """One single-path lattice record per pronoun sentence."""
    records = []
    for index, (_text, pinyin_spans, pinyin) in enumerate(PRONOUN_SENTENCES):
        candidates = tuple(
            "".join(
                character
                for character in lexicon.characters
                if lexicon.admits(spans.id(span), character)
            )
            for span in pinyin_spans
        )
        records.append(
            LatticeRecord(
                record=index,
                pinyin=pinyin,
                context=None,
                paths=(LatticePath(spans=pinyin_spans, candidates=candidates),),
            )
        )
    return records


def test_an_overfit_model_resolves_the_pronoun(spans: SpanVocab, tmp_path: Path) -> None:
    lexicon = pronoun_lexicon(spans)
    vocab = CharVocab(chars=SPECIALS + tuple(lexicon.characters))
    model = tiny_e2e(lexicon, vocab, E2EConfig(cross_attention_layers=1, decoder=READER_TINY))
    tokenizer = StubTokenizer()
    collator = Collator(tokenizer, CandidateSpace(spans, lexicon.candidate_mask))
    batch = collator(
        [
            TrainingExample(
                id=f"pr-{index}",
                spans=pinyin_spans,
                span_ids=tuple(spans.id(span) for span in pinyin_spans),
                targets=tuple(lexicon.index(character) for character in text),
                context=None,
            )
            for index, (text, pinyin_spans, _) in enumerate(PRONOUN_SENTENCES)
        ]
    )
    optimiser = torch.optim.AdamW(model.parameter_groups(1e-3, 3e-3))
    for _ in range(400):
        loss = model(batch).loss
        assert loss is not None
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    model.eval()
    index = CandidateIndex(lexicon)
    for record, (text, _, _) in zip(
        pronoun_lattice(spans, lexicon), PRONOUN_SENTENCES, strict=True
    ):
        hypotheses = decode_record(
            model, record, lexicon, spans, index, collator, torch.device("cpu"), 4096, 16, 8
        )
        assert hypotheses
        assert hypotheses[0][0] == text


def test_decode_returns_ranked_hypotheses_within_top_k(
    lexicon: Lexicon,
    spans: SpanVocab,
    tokenizer: StubTokenizer,
    reader_vocab: CharVocab,
    e2e_config: E2EConfig,
) -> None:
    model = tiny_e2e(lexicon, reader_vocab, e2e_config).eval()
    collator = Collator(tokenizer, CandidateSpace(spans, lexicon.candidate_mask))
    record = LatticeRecord(
        record=0,
        pinyin="woaibeijing",
        context=None,
        paths=(
            LatticePath(
                spans=("wo", "ai", "bei", "jing"),
                candidates=tuple(
                    "".join(
                        character
                        for character in lexicon.characters
                        if lexicon.admits(spans.id(span), character)
                    )
                    for span in ("wo", "ai", "bei", "jing")
                ),
            ),
        ),
    )
    hypotheses = decode_record(
        model,
        record,
        lexicon,
        spans,
        CandidateIndex(lexicon),
        collator,
        torch.device("cpu"),
        4096,
        16,
        8,
    )
    assert 0 < len(hypotheses) <= 8
    scores = [score for _, score in hypotheses]
    assert scores == sorted(scores, reverse=True)
    assert all(len(text) == 4 for text, _ in hypotheses)


def test_a_checkpoint_round_trips_and_a_route_a_one_is_refused(
    lexicon: Lexicon,
    reader_vocab: CharVocab,
    e2e_config: E2EConfig,
    tmp_path: Path,
) -> None:
    model = tiny_e2e(lexicon, reader_vocab, e2e_config)
    path = tmp_path / "e2e.pt"
    torch.save({"e2e": asdict(model.config), "model": model.state_dict(), "step": 7}, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    rebuilt = from_checkpoint(loaded, lexicon, bert_config=TINY)
    rebuilt.load_state_dict(loaded["model"], strict=True)

    wrong = tmp_path / "route-a.pt"
    torch.save({"route_a": {"base_model": "stub"}, "model": {}, "step": 1}, wrong)
    with pytest.raises(ValueError, match="no e2e record"):
        from_checkpoint(torch.load(wrong, map_location="cpu", weights_only=False), lexicon)


def test_initialise_composes_the_two_checkpoints(
    lexicon: Lexicon,
    reader_vocab: CharVocab,
    e2e_config: E2EConfig,
) -> None:
    encoder = RouteAModel.from_config(TINY, lexicon, RouteAConfig(cross_attention_layers=1))
    reader = TransformerCharLm(len(reader_vocab), READER_TINY)
    model = initialise(
        {"route_a": asdict(encoder.config), "model": encoder.state_dict(), "step": 9},
        {
            "config": asdict(READER_TINY),
            "vocab": list(reader_vocab.chars),
            "model": reader.state_dict(),
            "step": 12,
        },
        lexicon,
        0.3,
        bert_config=TINY,
    )
    for gate in model.decoder.gated_layers():
        assert float(gate.gate.detach()) == 0.0
    assert torch.equal(
        model.decoder.span_input.weight,
        torch.zeros_like(model.decoder.span_input.weight),
    )


def test_a_resumed_e2e_run_is_the_run_that_was_not_interrupted(
    short_corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    tokenizer: BaseTokenizer,
    reader_vocab: CharVocab,
    e2e_config: E2EConfig,
    tmp_path: Path,
) -> None:
    resumable = TrainingConfig(
        max_steps=6,
        base_lr=1e-3,
        new_lr=3e-3,
        token_budget=24,
        log_every=1,
        checkpoint_every=3,
        fp16=False,
        seed=3,
    )
    whole_dir = tmp_path / "whole"
    uninterrupted = tiny_e2e(lexicon, reader_vocab, e2e_config)
    stream = CorpusStream(
        short_corpus[0], short_corpus[1], SampleBuilder(lexicon, spans, arbitration, seed=1)
    )
    collator = Collator(tokenizer, CandidateSpace(spans, lexicon.candidate_mask))
    whole = train(uninterrupted, [(stream, collator)], resumable, whole_dir).metrics
    checkpoint = torch.load(whole_dir / "checkpoint-000003.pt", weights_only=False)
    assert "e2e" in checkpoint and "route_a" not in checkpoint
    assert checkpoint["e2e"]["aux_weight"] == e2e_config.aux_weight

    resumed_model = tiny_e2e(lexicon, reader_vocab, e2e_config)
    stream = CorpusStream(
        short_corpus[0], short_corpus[1], SampleBuilder(lexicon, spans, arbitration, seed=1)
    )
    resumed = train(
        resumed_model,
        [(stream, Collator(tokenizer, CandidateSpace(spans, lexicon.candidate_mask)))],
        resumable,
        tmp_path / "resumed",
        resume=whole_dir / "checkpoint-000003.pt",
    ).metrics
    losses = [
        float(record["loss"])
        for record in (json.loads(line) for line in resumed.read_text().splitlines())
        if record.get("event") == "step"
    ]
    whole_losses = [
        float(record["loss"])
        for record in (json.loads(line) for line in whole.read_text().splitlines())
        if record.get("event") == "step"
    ]
    assert losses == whole_losses[3:]
    weights = uninterrupted.state_dict()
    for name, tensor in resumed_model.state_dict().items():
        assert torch.equal(tensor, weights[name]), name

    # A checkpoint of the other model is a different run, not a resume.
    route = RouteAModel.from_config(TINY, lexicon, RouteAConfig(cross_attention_layers=1))
    stream = CorpusStream(
        short_corpus[0], short_corpus[1], SampleBuilder(lexicon, spans, arbitration, seed=1)
    )
    train(
        route,
        [(stream, Collator(tokenizer, CandidateSpace(spans, lexicon.candidate_mask)))],
        resumable,
        tmp_path / "route-a",
    )
    with pytest.raises(ValueError, match="no e2e"):
        train(
            tiny_e2e(lexicon, reader_vocab, e2e_config),
            [
                (
                    CorpusStream(
                        short_corpus[0],
                        short_corpus[1],
                        SampleBuilder(lexicon, spans, arbitration, seed=1),
                    ),
                    Collator(tokenizer, CandidateSpace(spans, lexicon.candidate_mask)),
                )
            ],
            resumable,
            tmp_path / "wrong",
            resume=tmp_path / "route-a" / "checkpoint-000003.pt",
        )


def test_the_slice_digest_matches_the_rust_oracle() -> None:
    """The dev/test hash, against values read from ``ime-eval``'s own code."""
    oracle = {
        ("zhongguo", "中国", None): 1483103352520586474,
        ("renmin", "人民", None): 5258366442535009459,
        ("yinhang", "银行", "我要去"): 9404986330596425458,
        ("beijing", "北京", None): 8823118165217649467,
        ("tianqi", "天气", "今天的"): 17299626687068035301,
        ("wo", "我", None): 6675338376534104286,
        ("wo", "我", ""): 6675338376534104286,
        ("wo", "我", "你好"): 8180375640380387211,
        ("nimenhao", "你们好", "吧"): 236004725558228434,
        ("tashiwojiejie", "她是我姐姐", None): 12806792326297396741,
    }
    u64_max = float(2**64 - 1)
    for (pinyin, text, context), wanted in oracle.items():
        got = digest(pinyin, text, context)
        assert got == wanted, f"{pinyin}/{text}: {got} != {wanted}"
        assert (got / u64_max < DEV_SHARE) == (
            (pinyin, text) in {("zhongguo", "中国"), ("nimenhao", "你们好")}
        )


def test_the_report_counts_like_rust() -> None:
    report = Report(top_k=8)
    report.observe("她是我姐姐", ["他是我姐姐", "她是我姐姐"])
    report.observe("他是我哥哥", [])
    assert report.records == 2
    assert report.top1_hits == 0
    assert report.topk_hits == 1
    assert report.unanswered == 1
    assert report.characters == 10
    assert report.character_hits == 4  # 他是我姐姐 vs 她是我姐姐: four of five
    assert report.reciprocal_ranks == RANK_SCALE // 2
    rendered = report.render()
    assert "sentence, top-8" in rendered and "MRR@8" in rendered


def test_the_eval_set_parse_is_strict(tmp_path: Path) -> None:
    good = tmp_path / "eval.jsonl"
    good.write_text(
        '{"pinyin": "wo", "text": "我", "context": null}\n'
        '{"pinyin": "ni", "text": "你", "context": "来"}\n',
        encoding="utf-8",
    )
    assert len(read_eval_records(good)) == 2
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"pinyin": "wo", "text": "我", "extra": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown fields"):
        read_eval_records(bad)
    empty = tmp_path / "empty.jsonl"
    empty.write_text('{"pinyin": "", "text": "我"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="pinyin"):
        read_eval_records(empty)


def test_the_report_fold_rebuilds_identically() -> None:
    """A report folded from observations equals one observed directly."""
    direct = Report(top_k=8)
    rebuilt = Report(top_k=8)
    observations = [Observation.new("中国", ["中国"], 8), Observation.new("人民", [], 8)]
    for observation in observations:
        direct.fold(observation)
        rebuilt.fold(observation)
    assert direct == rebuilt


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sync detection needs CUDA")
def test_a_training_step_issues_no_host_syncs(
    lexicon: Lexicon,
    spans: SpanVocab,
    tokenizer: StubTokenizer,
    reader_vocab: CharVocab,
    e2e_config: E2EConfig,
) -> None:
    """set_sync_debug_mode("error") fails the test if forward or backward syncs."""
    model = tiny_e2e(lexicon, reader_vocab, e2e_config).cuda()
    batch = Collator(tokenizer, CandidateSpace(spans, lexicon.candidate_mask))(
        [
            example(spans, lexicon, "我爱北京", ("wo", "ai", "bei", "jing")),
            example(spans, lexicon, "钟爱北京", ("zhong", "ai", "bei", "jing")),
        ]
    ).to(torch.device("cuda"))
    model(batch).loss.backward()  # warm the allocator outside the check
    torch.cuda.set_sync_debug_mode("error")
    try:
        loss = model(batch).loss
        assert loss is not None
        loss.backward()
    finally:
        torch.cuda.set_sync_debug_mode("default")


def test_compiling_changes_no_checkpoint_or_parameter_names(
    lexicon: Lexicon,
    spans: SpanVocab,
    tokenizer: StubTokenizer,
    reader_vocab: CharVocab,
    e2e_config: E2EConfig,
) -> None:
    """Compiled methods leave the module tree alone: a checkpoint written while
    compiled loads into an uncompiled model, so a resumed segment recompiles
    instead of refusing the file."""
    from mlime.train.e2e import compile_modules

    model = tiny_e2e(lexicon, reader_vocab, e2e_config)
    keys_before = list(model.state_dict())
    params_before = [name for name, _ in model.named_parameters()]
    compile_modules(model, "default")
    assert list(model.state_dict()) == keys_before
    assert [name for name, _ in model.named_parameters()] == params_before
    batch = Collator(tokenizer, CandidateSpace(spans, lexicon.candidate_mask))(
        [example(spans, lexicon, "我爱北京", ("wo", "ai", "bei", "jing"))]
    )
    compiled = model(batch).loss
    assert compiled is not None and torch.isfinite(compiled)
    compiled.backward()

    fresh = tiny_e2e(lexicon, reader_vocab, e2e_config)
    fresh.load_state_dict(model.state_dict())


def test_the_prefilled_masks_skip_the_builder(
    lexicon: Lexicon,
    spans: SpanVocab,
    tokenizer: StubTokenizer,
    reader_vocab: CharVocab,
    e2e_config: E2EConfig,
) -> None:
    """The towers' 4D boolean masks early-exit ``_create_attention_masks``.

    The 2D path's padding check, ``padding_mask.all()``, is a device sync a
    step cannot pay; a 4D mask is returned as passed in, and sdpa reads it as
    the same mask the builder would have made -- ``True`` attends.
    """
    model = tiny_e2e(lexicon, reader_vocab, e2e_config).eval()
    batch = Collator(tokenizer, CandidateSpace(spans, lexicon.candidate_mask))(
        [example(spans, lexicon, "我爱北京", ("wo", "ai", "bei", "jing"))]
    )
    embeddings = model.fill.embeddings(inputs_embeds=model._fill_inputs(batch))
    fill_mask = batch.attention_mask[:, None, None, :].bool()
    returned, extra = model.fill._create_attention_masks(
        attention_mask=fill_mask,
        encoder_attention_mask=None,
        embedding_output=embeddings,
        encoder_hidden_states=None,
        past_key_values=None,
    )
    assert returned is fill_mask and extra is None
    context_mask = batch.context_mask[:, None, None, :].bool()
    with torch.no_grad():
        hidden = embeddings
        for layer in model.fill.encoder.layer:
            hidden = layer(hidden, attention_mask=returned)
        # ``embeddings`` is post-LayerNorm, so the reference re-derives it from
        # the raw fill inputs -- ``BertModel`` embeds ``inputs_embeds`` itself.
        reference: torch.Tensor = model.fill(
            inputs_embeds=model._fill_inputs(batch), attention_mask=batch.attention_mask
        ).last_hidden_state
        context: torch.Tensor = model.context(
            input_ids=batch.context_ids, attention_mask=context_mask
        ).last_hidden_state
        context_reference: torch.Tensor = model.context(
            input_ids=batch.context_ids, attention_mask=batch.context_mask
        ).last_hidden_state
    torch.testing.assert_close(hidden, reference)
    torch.testing.assert_close(context, context_reference)


def test_the_first_compile_is_already_dynamic(
    lexicon: Lexicon,
    spans: SpanVocab,
    tokenizer: StubTokenizer,
    reader_vocab: CharVocab,
    e2e_config: E2EConfig,
) -> None:
    """Marked batch and sequence dims compile one graph per callable.

    Without the marks dynamo would compile each callable static, then again
    dynamically on the second shape; three widths produce exactly two graphs
    -- one for ``encode``, one for the decoder -- and no graph breaks.
    """
    from torch._dynamo.utils import counters

    model = tiny_e2e(lexicon, reader_vocab, e2e_config)
    # The eager backend keeps dynamo's guard machinery without inductor's
    # compile time, which buys a CPU test nothing.
    model.encode = torch.compile(  # type: ignore[method-assign]
        model.encode, dynamic=True, backend="eager"
    )
    model.decoder.forward = torch.compile(  # type: ignore[method-assign]
        model.decoder.forward, dynamic=True, backend="eager"
    )
    collator = Collator(tokenizer, CandidateSpace(spans, lexicon.candidate_mask))
    batches = [
        collator([example(spans, lexicon, text, pinyin)])
        for text, pinyin in (
            ("我爱", ("wo", "ai")),
            ("我爱北", ("wo", "ai", "bei")),
            ("我爱北京", ("wo", "ai", "bei", "jing")),
        )
    ]
    graphs = int(counters["stats"].get("unique_graphs") or 0)
    breaks = sum(counters["graph_breaks"].values())
    model.train()
    for batch in batches:
        loss = model(batch).loss
        assert loss is not None
        loss.backward()
    assert int(counters["stats"].get("unique_graphs") or 0) - graphs == 2
    assert int(sum(counters["graph_breaks"].values())) == breaks
