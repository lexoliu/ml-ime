"""The training side of reading typos: augmentation, letters, candidates, resume.

The shared noise model corrupts what was pressed, never the target; the letters
reach the fill tower through an input that adds nothing at init; the candidate
set a typoed span admits is the union of its corrections' homophones; and a
checkpoint written before any of it resumes with the new parameters arriving
at the new-parameter rate and the old state still mapped to the old tensors.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from conftest import READINGS, StubTokenizer
from transformers import BertConfig

from mlime.data.corpus import Sample
from mlime.train.arbitration import ReadingArbitration
from mlime.train.lexicon import CandidateSpace, Lexicon
from mlime.train.loop import TrainingConfig, _load_resumed_optimiser, train
from mlime.train.model import RouteAConfig, RouteAModel
from mlime.train.samples import (
    LETTER_PAD,
    Augmentation,
    Collator,
    CorpusStream,
    SampleBuilder,
    TrainingExample,
)
from mlime.train.spans import UNKNOWN_SPAN, SpanVocab
from mlime.typo import CorrectionTable, NoiseModel, load_syllables

TINY = BertConfig(
    vocab_size=256,
    hidden_size=32,
    num_attention_heads=4,
    num_hidden_layers=2,
    intermediate_size=64,
    max_position_embeddings=64,
)

#: Six steps over :func:`short_corpus_fixture`: a checkpoint mid-epoch, so a
#: resume exercises the epoch and the index it saved, not a boundary.
RESUMABLE = TrainingConfig(
    max_steps=6,
    base_lr=1e-3,
    new_lr=3e-3,
    token_budget=24,
    log_every=1,
    checkpoint_every=3,
    fp16=False,
    seed=3,
)

MODEL_PATH = Path(__file__).parent.parent / "src/mlime/data/typo_model.json"
SYLLABLES = Path(__file__).parent.parent.parent / "crates/ime-pinyin/data/syllables.txt"


def always_typos() -> NoiseModel:
    """The shipped table with the per-sentence coin forced: one edit, always."""
    data = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    data["per_sentence"] = 1.0
    data["edit_count"] = {"1": 1.0}
    return NoiseModel.parse(json.dumps(data))


@pytest.fixture(name="noise")
def noise_fixture() -> NoiseModel:
    return NoiseModel.load(MODEL_PATH)


@pytest.fixture(name="corrections")
def corrections_fixture(noise: NoiseModel) -> CorrectionTable:
    """The real model against the real syllable inventory."""
    return CorrectionTable(noise, load_syllables(SYLLABLES))


def tiny_model(lexicon: Lexicon) -> RouteAModel:
    """The same randomly initialised model every time it is called."""
    torch.manual_seed(0)
    return RouteAModel.from_config(TINY, lexicon, RouteAConfig(cross_attention_layers=1)).eval()


def example(
    lexicon: Lexicon, spans: SpanVocab, text: str, pinyin: tuple[str, ...]
) -> TrainingExample:
    """One training example: the typed spans, their ids, and the targets."""
    return TrainingExample(
        id=f"ex-{pinyin[0]}",
        spans=pinyin,
        span_ids=tuple(spans.id_or_unknown(span) for span in pinyin),
        targets=tuple(lexicon.index(character) for character in text),
        context=None,
    )


def test_noise_and_the_switch_come_together(
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    noise: NoiseModel,
    corrections: CorrectionTable,
) -> None:
    space = CandidateSpace(spans, lexicon.candidate_mask, corrections)
    clean = CandidateSpace(spans, lexicon.candidate_mask)
    with pytest.raises(ValueError, match="come together"):
        SampleBuilder(lexicon, spans, arbitration, Augmentation(typos=True), noise=None)
    with pytest.raises(ValueError, match="come together"):
        SampleBuilder(lexicon, spans, arbitration, Augmentation(), noise=noise, candidates=space)
    with pytest.raises(ValueError, match="come together"):
        SampleBuilder(
            lexicon, spans, arbitration, Augmentation(typos=True), noise=noise, candidates=clean
        )
    with pytest.raises(ValueError, match="come together"):
        SampleBuilder(
            lexicon, spans, arbitration, Augmentation(typos=True), noise=noise, candidates=None
        )


def test_a_typoed_example_keeps_alignment_and_target(
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    corrections: CorrectionTable,
) -> None:
    clean = SampleBuilder(lexicon, spans, arbitration, seed=0)
    builder = SampleBuilder(
        lexicon,
        spans,
        arbitration,
        Augmentation(typos=True),
        seed=0,
        noise=always_typos(),
        candidates=CandidateSpace(spans, lexicon.candidate_mask, corrections),
    )
    syllables = ["wo3", "ai4", "bei3", "jing1"]
    # A slip whose span admits nothing is a counted drop; walk the ids until a
    # kept example shows what the corruption did to the kept ones.
    kept = []
    for index in range(8):
        sample = Sample(id=f"s{index}", source="test", text="我爱北京", context=None)
        clean_example = clean.build(sample, syllables, 0)
        typoed = builder.build(sample, syllables, 0)
        assert clean_example is not None
        if typoed is not None:
            kept.append((clean_example, typoed))
    assert kept
    for clean_example, typoed in kept:
        # The target is the clean sentence; the styles ran first, so the spans
        # carry the slip the noise model added to them.
        assert typoed.targets == clean_example.targets
        assert len(typoed.spans) == len(typoed.targets)
        assert typoed.spans != clean_example.spans


def test_typo_corruption_is_seeded_per_example(
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    corrections: CorrectionTable,
) -> None:
    def build(seed: int, epoch: int, sample_id: str) -> tuple[str, ...]:
        builder = SampleBuilder(
            lexicon,
            spans,
            arbitration,
            Augmentation(typos=True),
            seed=seed,
            noise=always_typos(),
            candidates=CandidateSpace(spans, lexicon.candidate_mask, corrections),
        )
        built = builder.build(
            Sample(id=sample_id, source="test", text="中重我绿", context=None),
            ["zhong1", "chong2", "wo3", "lv4"],
            epoch,
        )
        # A slip whose span admits nothing is a counted drop, not an error.
        return built.spans if built is not None else ()

    assert build(1, 0, "s1") == build(1, 0, "s1")
    # The draw comes out of the example's own generator: another id or epoch is
    # a different slip, not the same one replayed.
    variants = {build(1, 0, f"s{index}") for index in range(8)}
    assert len(variants) > 1
    epochs = {build(1, epoch, "s1") for epoch in range(8)}
    assert len(epochs) > 1


def test_off_inventory_spans_map_to_the_reserved_id(spans: SpanVocab) -> None:
    unknown = spans.unknown_id
    assert UNKNOWN_SPAN in spans
    assert spans.id(UNKNOWN_SPAN) == unknown == len(spans) - 1
    assert spans.id_or_unknown("zhong") == spans.id("zhong")
    assert spans.id_or_unknown("zhpng") == unknown
    assert spans.id_or_unknown("") == unknown


def test_candidate_space_widens_over_corrections(
    lexicon: Lexicon, spans: SpanVocab, corrections: CorrectionTable
) -> None:
    clean = CandidateSpace(spans, lexicon.candidate_mask)
    space = CandidateSpace(spans, lexicon.candidate_mask, corrections)

    def expected_union(span: str) -> set[int]:
        # The homophones of every correction: a character is admitted when one
        # of the span's corrections prefixes one of its readings -- the same
        # relation the lexicon's mask rows already encode.
        return {
            lexicon.index(character)
            for character, readings in READINGS.items()
            if any(
                reading.startswith(entry)
                for entry, _cost in corrections.corrections(span)
                for reading in readings
            )
        }

    # An untouched valid span keeps exactly its old candidates plus what its
    # corrections add: the union spec applied to the miniature lexicon.
    for span in ("zhong", "wo", "lv", "zh"):
        assert set(space.resolve(span)) == expected_union(span)
        assert set(space.resolve(span)) >= set(clean.resolve(span))
    # A typoed span admits the union of its corrections' homophones -- "zhon"
    # is one omission from "zhong", so the zhong-reading characters are in.
    zhong = {lexicon.index("中"), lexicon.index("钟")}
    assert zhong <= set(space.resolve("zhon"))
    assert set(space.resolve("zhon")) == expected_union("zhon")
    assert zhong <= set(space.resolve("zhpng"))
    # Under the clean space the same typo admits nothing -- the <unk> row.
    assert clean.resolve("zhpng") == ()
    assert space.admits("zhon", lexicon.index("中"))
    assert not space.admits("zhon", lexicon.index("绿"))


def test_the_batch_carries_letters_and_candidate_lists(
    lexicon: Lexicon, spans: SpanVocab, tokenizer: StubTokenizer
) -> None:
    batch = Collator(tokenizer, CandidateSpace(spans, lexicon.candidate_mask))(
        [example(lexicon, spans, "我爱", ("wo", "ai"))]
    )
    letters = batch.span_letters[0]
    assert letters[1, :2].tolist() == [22, 14]  # w, o
    assert bool((letters[1, 2:] == LETTER_PAD).all())
    assert letters[2, :2].tolist() == [0, 8]  # a, i
    assert bool((letters[0] == LETTER_PAD).all())
    row = lexicon.candidate_mask[spans.id("wo")].nonzero(as_tuple=True)[0].tolist()
    assert batch.candidate_ids[0, 1, : len(row)].tolist() == row
    assert int(batch.candidate_counts[0, 1]) == len(row)


def test_a_typoed_span_collates_under_the_typo_space(
    lexicon: Lexicon,
    spans: SpanVocab,
    tokenizer: StubTokenizer,
    corrections: CorrectionTable,
) -> None:
    typoed = example(lexicon, spans, "中", ("zhpng",))
    space = CandidateSpace(spans, lexicon.candidate_mask, corrections)
    batch = Collator(tokenizer, space)([typoed])
    assert batch.span_letters[0, 1, :5].tolist() == [25, 7, 15, 13, 6]
    count = int(batch.candidate_counts[0, 1])
    assert count == len(space.resolve("zhpng")) > 0
    assert lexicon.index("中") in batch.candidate_ids[0, 1, :count].tolist()
    # Under the clean space the same example is refused, as typos always were.
    with pytest.raises(ValueError, match="target its span does not admit"):
        Collator(tokenizer, CandidateSpace(spans, lexicon.candidate_mask))([typoed])


def test_candidates_at_rebuilds_the_mask(
    lexicon: Lexicon, spans: SpanVocab, tokenizer: StubTokenizer
) -> None:
    model = tiny_model(lexicon)
    batch = Collator(tokenizer, CandidateSpace(spans, lexicon.candidate_mask))(
        [example(lexicon, spans, "中我", ("zhong", "wo"))]
    )
    positions = torch.arange(batch.span_ids.numel())
    mask = model.candidates_at(batch, positions).reshape(*batch.span_ids.shape, -1)
    wanted = torch.zeros_like(mask)
    wanted[:, 1:-1] = lexicon.candidate_mask[batch.span_ids[:, 1:-1]]
    assert torch.equal(mask, wanted)


def test_the_letter_encoder_separates_anagram_spans(lexicon: Lexicon) -> None:
    model = tiny_model(lexicon)
    encoder = model.letter_encoder
    assert bool((encoder.project.weight == 0).all())
    letters = torch.tensor(
        [
            [[ord(letter) - ord("a") for letter in "na"] + [LETTER_PAD] * 10],
            [[ord(letter) - ord("a") for letter in "an"] + [LETTER_PAD] * 10],
        ]
    )
    # With a non-zero projection the position table's order difference reaches
    # the tower's input; at init the projection is zero and the term is absent.
    encoder.project.weight.data.normal_(0, 0.1)
    assert not torch.equal(encoder(letters)[0, 0], encoder(letters)[1, 0])


def test_a_pre_letter_checkpoint_loads_and_is_bit_identical(
    lexicon: Lexicon, spans: SpanVocab, tokenizer: StubTokenizer, tmp_path: Path
) -> None:
    fresh = tiny_model(lexicon)
    # A checkpoint the pre-#103 code would have written: no letter parameters,
    # and the span table and mask one row shorter -- before the <unk> tail.
    weights = dict(fresh.state_dict())
    for name in [name for name in weights if name.startswith("letter_encoder.")]:
        del weights[name]
    weights["span_embeddings.weight"] = weights["span_embeddings.weight"][:-1]
    weights["candidate_mask"] = weights["candidate_mask"][:-1]
    resumed = tiny_model(lexicon)
    resumed.load_resumed(weights, tmp_path / "pre-letter.pt")
    # The new parameters loaded as missing keys keep their initialisation: the
    # projection is zero, so the resumed model is the old model, bit for bit.
    assert bool((resumed.letter_encoder.project.weight == 0).all())
    assert bool((resumed.span_embeddings.weight[spans.unknown_id] == 0).all())

    batch = Collator(tokenizer, CandidateSpace(spans, lexicon.candidate_mask))(
        [example(lexicon, spans, "我爱北京", ("wo", "ai", "bei", "jing"))]
    )
    with torch.no_grad():
        assert torch.equal(fresh.scores(batch), resumed.scores(batch))
        assert fresh(batch).loss == resumed(batch).loss


def test_letter_parameters_trail_the_new_group(lexicon: Lexicon) -> None:
    model = tiny_model(lexicon)
    base, new = model.parameter_groups(1e-3, 3e-3)
    letter_ids = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("letter_encoder.")
    }
    assert {id(parameter) for parameter in new["params"][-3:]} == letter_ids
    assert not letter_ids & {id(parameter) for parameter in base["params"]}
    assert not letter_ids & {id(parameter) for parameter in new["params"][:-3]}
    assert new["lr"] == 3e-3
    # Frozen -- which is what the clean path does -- they leave the groups.
    model.letter_encoder.requires_grad_(False)
    _, frozen_new = model.parameter_groups(1e-3, 3e-3)
    assert not letter_ids & {id(parameter) for parameter in frozen_new["params"]}
    model.letter_encoder.requires_grad_(True)


def test_a_pre_letter_optimiser_resumes_onto_the_grown_group(
    lexicon: Lexicon, tmp_path: Path
) -> None:
    model = tiny_model(lexicon)
    optimiser = torch.optim.AdamW(model.parameter_groups(1e-3, 3e-3))
    saved = optimiser.state_dict()
    # The old checkpoint's groups were shorter: drop the letter parameters'
    # trailing ids, the way an #88 file reads.
    group = saved["param_groups"][1]
    dropped = group["params"][-3:]
    group["params"] = group["params"][:-3]
    for identifier in dropped:
        saved["state"].pop(identifier, None)
    before = len(saved["state"])
    _load_resumed_optimiser(optimiser, saved, tmp_path / "old.pt")
    state = optimiser.state_dict()
    # The letter parameters got fresh ids at the group's tail and no state.
    assert len(state["param_groups"][1]["params"]) == len(group["params"]) + 3
    assert len(state["state"]) == before


def test_a_pre_letter_checkpoint_resumes_a_typo_run(
    short_corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    tokenizer: StubTokenizer,
    tmp_path: Path,
) -> None:
    samples_dir, labels_dir = short_corpus

    def lane(typos: bool) -> tuple[CorpusStream, Collator]:
        noise = always_typos() if typos else None
        space = CandidateSpace(
            spans,
            lexicon.candidate_mask,
            CorrectionTable(noise, load_syllables(SYLLABLES)) if noise is not None else None,
        )
        return (
            CorpusStream(
                samples_dir,
                labels_dir,
                SampleBuilder(
                    lexicon,
                    spans,
                    arbitration,
                    Augmentation(typos=typos),
                    seed=1,
                    noise=noise,
                    candidates=space,
                ),
            ),
            Collator(tokenizer, space),
        )

    # Three clean steps, then the checkpoint rewritten the way the pre-#103
    # code would have left it: no letter parameters, one span row shorter.
    first = train(tiny_model(lexicon), [lane(False)], RESUMABLE, tmp_path / "first")
    assert first.metrics.is_file()
    checkpoint = tmp_path / "first" / "checkpoint-000003.pt"
    state = torch.load(checkpoint, weights_only=False)
    for name in [name for name in state["model"] if name.startswith("letter_encoder.")]:
        del state["model"][name]
    state["model"]["span_embeddings.weight"] = state["model"]["span_embeddings.weight"][:-1]
    state["model"]["candidate_mask"] = state["model"]["candidate_mask"][:-1]
    saved_group = state["optimiser"]["param_groups"][1]
    dropped = saved_group["params"][-3:]
    saved_group["params"] = saved_group["params"][:-3]
    for identifier in dropped:
        state["optimiser"]["state"].pop(identifier, None)
    old = tmp_path / "old.pt"
    torch.save(state, old)

    losses = []
    for run in ("resume-a", "resume-b"):
        model = tiny_model(lexicon)
        model.letter_encoder.requires_grad_(True)
        segment = train(model, [lane(True)], RESUMABLE, tmp_path / run, resume=old)
        written = [
            record
            for record in (json.loads(line) for line in segment.metrics.read_text().splitlines())
            if record["event"] == "resume"
        ]
        assert len(written) == 1
        assert written[0]["step"] == 3 and written[0]["epoch"] == 0
        losses.append(
            [
                float(record["loss"])
                for record in (
                    json.loads(line) for line in segment.metrics.read_text().splitlines()
                )
                if record["event"] == "step"
            ]
        )
    # Positions and RNG carried: two resumes of the same old checkpoint take
    # the same typoed batches and land on the same losses.
    assert losses[0] == losses[1]
    assert len(losses[0]) == RESUMABLE.max_steps - 3
