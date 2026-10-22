"""Qwen as the context tower (#222): the contract, the head, the freezing, the swap.

The tower is tested on a Qwen3 the size of a hand -- two layers, sixty-four
wide, grouped-query attention like the real one -- over a byte-level BPE with
no merges and Qwen's end-of-text token, saved to a directory so every path
that loads Qwen by name loads this one. The fill side is the e2e tests' tiny
BERT, thirty-two wide, so every cross-attention block has a real width to fold.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from conftest import BuiltLane, StubTokenizer

# The e2e tests' tiny model and its fixtures, which pytest finds by import.
from test_e2e_model import (  # noqa: F401
    TINY,
    e2e_config_fixture,
    example,
    reader_vocab_fixture,
    tiny_e2e,
)
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model

from mlime.train.arbitration import ReadingArbitration
from mlime.train.charlm_vocab import CharVocab
from mlime.train.e2e import from_checkpoint, swap_in_qwen
from mlime.train.e2e_model import E2EConfig, E2EModel
from mlime.train.lexicon import CandidateSpace, Lexicon
from mlime.train.loop import TrainingConfig, train
from mlime.train.model import ContextMemory, GatedCrossAttention
from mlime.train.qwen_tower import (
    CONSTANT_HEAD,
    LoraConfig,
    LoraLinear,
    ProjectedCrossAttention,
    QwenContextConfig,
    QwenContextTokenizer,
    QwenContextTower,
    TowerGraphs,
    build_tower,
    length_bucket,
    load_context_tokenizer,
    load_pretrained,
)
from mlime.train.samples import Collator, CorpusStream, SampleBuilder
from mlime.train.spans import SpanVocab

END_OF_TEXT = "<|endoftext|>"

TINY_QWEN = Qwen3Config(
    vocab_size=264,
    hidden_size=64,
    intermediate_size=96,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    max_position_embeddings=128,
    tie_word_embeddings=True,
)


@pytest.fixture(name="qwen_dir", scope="module")
def qwen_dir_fixture(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny Qwen3 release on disk: config, weights and a byte-level tokenizer."""
    directory = tmp_path_factory.mktemp("tiny-qwen")
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    inner = Tokenizer(models.BPE(vocab={byte: i for i, byte in enumerate(alphabet)}, merges=[]))
    inner.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    inner.decoder = decoders.ByteLevel()
    inner.add_special_tokens([END_OF_TEXT])
    PreTrainedTokenizerFast(
        tokenizer_object=inner, eos_token=END_OF_TEXT, pad_token=END_OF_TEXT
    ).save_pretrained(directory)
    torch.manual_seed(11)
    Qwen3Model(TINY_QWEN).save_pretrained(directory)
    return directory


@pytest.fixture(name="qwen")
def qwen_fixture(qwen_dir: Path) -> QwenContextConfig:
    """The tiny tower, frozen, in fp32 so comparisons are exact to rounding."""
    return QwenContextConfig(model=str(qwen_dir), revision=None, dtype="float32")


def tower_of(config: QwenContextConfig) -> QwenContextTower:
    """The tower *config* names, with the saved release's weights."""
    tower = build_tower(config)
    load_pretrained(tower)
    return tower.eval()


def contexts(config: QwenContextConfig, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """*texts* tokenised as the collator tokenises them, as tensors."""
    encoded = load_context_tokenizer(config)(texts, padding=True, truncation=True, max_length=64)
    return (
        torch.tensor(encoded["input_ids"], dtype=torch.long),
        torch.tensor(encoded["attention_mask"], dtype=torch.long),
    )


def test_the_head_is_end_of_text_then_a_line_break(qwen: QwenContextConfig) -> None:
    tokenizer = load_context_tokenizer(qwen)
    end, newline = tokenizer.head_ids(CONSTANT_HEAD)
    assert end == tokenizer.tokenizer.convert_tokens_to_ids(END_OF_TEXT)
    assert tokenizer.tokenizer.decode([newline]) == "\n"
    assert tower_of(qwen).head_ids.tolist() == [end, newline]


def test_the_tokenizer_keeps_the_end_of_a_long_context(qwen: QwenContextConfig) -> None:
    tokenizer = load_context_tokenizer(qwen)
    encoded = tokenizer(["abcdefgh", "xy"], padding=True, truncation=True, max_length=4)
    assert tokenizer.tokenizer.decode(encoded["input_ids"][0]) == "efgh"
    # No special tokens, and padding on the right.
    pad = tokenizer.tokenizer.pad_token_id
    assert encoded["input_ids"][1][2:] == [pad, pad]
    assert encoded["attention_mask"] == [[1, 1, 1, 1], [1, 1, 0, 0]]


def test_each_row_reads_as_the_head_and_the_row_in_one_sequence(
    qwen: QwenContextConfig,
) -> None:
    """The constant-head equivalence: a cached prefix is the sequence it was cut from."""
    tower = tower_of(qwen)
    texts = ["我爱北京天安门", "钟", ""]
    ids, mask = contexts(qwen, texts)
    head = tower.head_state()
    with torch.no_grad():
        states = tower.frozen_states(ids, mask)
        for row, length in enumerate(mask.sum(dim=1).tolist()):
            joined = torch.cat([tower.head_ids, ids[row, :length]]).unsqueeze(0)
            whole = tower.model(input_ids=joined, use_cache=False).last_hidden_state[0]
            torch.testing.assert_close(head.rows, whole[: head.length], rtol=0, atol=1e-5)
            torch.testing.assert_close(
                states[row, :length], whole[head.length :], rtol=0, atol=1e-5
            )
    assert torch.isfinite(states).all()


def test_the_memory_is_the_head_then_the_context_normed_and_masked(
    qwen: QwenContextConfig,
) -> None:
    tower = tower_of(qwen)
    ids, mask = contexts(qwen, ["北京", "我爱绿钟"])
    memory = tower.memory(ids, mask)
    head = tower.head_state()
    rows, length = ids.shape
    assert memory.states.shape == (rows, head.length + length, TINY_QWEN.hidden_size)
    assert memory.states.dtype == torch.float32
    assert memory.mask[:, : head.length].all()
    assert torch.equal(memory.mask[:, head.length :], mask.bool())
    # The head is computed once while the weights stand, and again once they move.
    assert tower.head_state() is head
    tower.load_state_dict(tower.state_dict())
    assert tower.head_state() is not head

    # No context anywhere in the batch: the memory is the head alone.
    empty = tower.memory(torch.zeros((2, 0), dtype=torch.long), torch.zeros((2, 0)))
    assert empty.states.shape == (2, head.length, TINY_QWEN.hidden_size)
    assert empty.mask.all()


def test_a_projected_block_reads_shared_rows_and_switches_per_example() -> None:
    torch.manual_seed(5)
    block = ProjectedCrossAttention(32, 4, 0.0, 64).eval()
    hidden = torch.randn(3, 5, 32)
    memory = ContextMemory(states=torch.randn(2, 6, 64), mask=torch.ones(2, 6, dtype=torch.bool))
    memory.mask[1, 4:] = False
    on = torch.ones(3)
    # A zero gate is the identity, whatever the memory holds.
    assert torch.equal(block(hidden, memory, torch.tensor([0, 1, 1]), on), hidden)
    with torch.no_grad():
        block.gate.fill_(0.7)
    rows = torch.tensor([0, 1, 1])
    together = block(hidden, memory, rows, on)
    for index, row in enumerate(rows.tolist()):
        alone = block(
            hidden[index : index + 1],
            ContextMemory(memory.states[row : row + 1], memory.mask[row : row + 1]),
            torch.tensor([0]),
            on[:1],
        )
        torch.testing.assert_close(together[index : index + 1], alone)
    # A masked row changes nothing: the padding is not read.
    shuffled = memory.states.clone()
    shuffled[1, 4:] = torch.randn(2, 64)
    torch.testing.assert_close(
        block(hidden, ContextMemory(shuffled, memory.mask), rows, on), together
    )
    off = torch.tensor([1.0, 0.0, 1.0])
    assert torch.equal(block(hidden, memory, rows, off)[1], hidden[1])


def test_a_frozen_tower_gives_no_gradient_but_to_its_layer_norm(
    qwen: QwenContextConfig,
) -> None:
    tower = tower_of(qwen).train()
    assert tower.frozen
    assert not any(parameter.requires_grad for parameter in tower.model.parameters())
    ids, mask = contexts(qwen, ["北京", "钟"])
    memory = tower.memory(ids, mask)
    memory.states.square().sum().backward()
    assert all(parameter.grad is None for parameter in tower.model.parameters())
    assert tower.norm.weight.grad is not None and tower.norm.bias.grad is not None
    assert set(tower.new_parameters()) == {"norm.weight", "norm.bias"}


def test_an_adapted_tower_starts_as_the_frozen_one_and_trains_only_its_adapters(
    qwen: QwenContextConfig,
) -> None:
    frozen = tower_of(qwen)
    adapted = tower_of(replace(qwen, lora=LoraConfig(rank=2, alpha=4.0)))
    adapted.load_state_dict(frozen.state_dict(), strict=False)
    assert not adapted.frozen
    adapters = [name for name in adapted.new_parameters() if "lora" in name]
    # Four projections of two layers, two matrices each.
    assert len(adapters) == 2 * 4 * 2
    assert all(isinstance(layer.self_attn.q_proj, LoraLinear) for layer in adapted.model.layers)
    # The base weights keep their names, so a frozen tower's checkpoint fits.
    assert set(frozen.state_dict()) <= set(adapted.state_dict())

    ids, mask = contexts(qwen, ["我爱北京", "绿"])
    with torch.no_grad():
        torch.testing.assert_close(
            adapted.memory(ids, mask).states, frozen.memory(ids, mask).states
        )
    adapted.train()
    adapted.memory(ids, mask).states.square().sum().backward()
    trained = {name for name, p in adapted.named_parameters() if p.grad is not None}
    # lora_b starts at zero, so only it moves the output at the first step; A
    # receives its gradient through B and is zero until B is not.
    assert {name for name in trained if name.endswith("lora_b")}
    assert trained <= set(adapted.new_parameters())


def qwen_e2e(
    lexicon: Lexicon, vocab: CharVocab, config: E2EConfig, qwen: QwenContextConfig
) -> E2EModel:
    """The tiny e2e model with Qwen reading its context, seeded like :func:`tiny_e2e`."""
    model = tiny_e2e(lexicon, vocab, replace(config, context=qwen))
    assert isinstance(model.context, QwenContextTower)
    load_pretrained(model.context)
    return model


def test_the_swap_carries_everything_but_the_context_path(
    lexicon: Lexicon,
    reader_vocab: CharVocab,
    e2e_config: E2EConfig,
    qwen: QwenContextConfig,
) -> None:
    source = tiny_e2e(lexicon, reader_vocab, e2e_config)
    for layer in source.gated_layers():
        assert isinstance(layer, GatedCrossAttention)
        with torch.no_grad():
            layer.gate.fill_(0.25)
    weights = source.state_dict()

    torch.manual_seed(1)
    model = tiny_e2e(lexicon, reader_vocab, replace(e2e_config, context=qwen))
    swap_in_qwen(model, weights, "source")
    swapped = model.state_dict()
    hidden = TINY.hidden_size
    block = "cross_attention.0."
    packed = weights[f"{block}attention.in_proj_weight"]
    assert torch.equal(swapped[f"{block}query.weight"], packed[:hidden])
    assert torch.equal(swapped[f"{block}gate"], weights[f"{block}gate"])
    assert swapped[f"{block}key.weight"].shape == (hidden, TINY_QWEN.hidden_size)
    released = Qwen3Model.from_pretrained(qwen.model).state_dict()
    for name, tensor in released.items():
        assert torch.equal(swapped[f"context.model.{name}"], tensor), name
    for name, tensor in weights.items():
        if not name.startswith(("context.", "cross_attention.")):
            assert torch.equal(swapped[name], tensor), name

    model.freeze_for_context_warm_up()
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == {
        name for name, _ in model.named_parameters() if name.startswith("cross_attention.")
    } | {"context.norm.weight", "context.norm.bias"}
    groups = model.parameter_groups(1e-5, 1e-4)
    added = {
        id(p) for name, p in model.named_parameters() if name in model.new_context_parameters()
    }
    assert {id(p) for p in groups[1]["params"]} == added  # type: ignore[attr-defined]

    # A MacBERT checkpoint carrying a weight the swap does not know is refused.
    with pytest.raises(ValueError, match="does not read"):
        swap_in_qwen(model, {**weights, f"{block}stray": torch.zeros(1)}, "source")


def test_a_resumed_warm_up_is_the_warm_up_that_was_not_interrupted(
    short_corpus: tuple[Path, Path],
    lexicon: Lexicon,
    spans: SpanVocab,
    arbitration: ReadingArbitration,
    tokenizer: StubTokenizer,
    reader_vocab: CharVocab,
    e2e_config: E2EConfig,
    qwen: QwenContextConfig,
    tmp_path: Path,
) -> None:
    config = replace(e2e_config, context_warm_up=True)
    context_tokenizer = load_context_tokenizer(qwen)
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

    def lane() -> BuiltLane:
        stream = CorpusStream(
            short_corpus[0], short_corpus[1], SampleBuilder(lexicon, spans, arbitration, seed=1)
        )
        collator = Collator(
            tokenizer,
            context_tokenizer,
            CandidateSpace(spans, lexicon.candidate_mask),
            context_dropout=0.0,
        )
        return BuiltLane(stream, collator)

    def model() -> E2EModel:
        built = qwen_e2e(lexicon, reader_vocab, config, qwen)
        built.freeze_for_context_warm_up()
        return built

    whole_dir = tmp_path / "whole"
    uninterrupted = model()
    frozen = {
        name: p.detach().clone() for name, p in uninterrupted.context.model.named_parameters()
    }
    whole = train(uninterrupted, [lane()], resumable, whole_dir).metrics
    for name, parameter in uninterrupted.context.model.named_parameters():
        assert torch.equal(parameter, frozen[name]), name

    checkpoint = torch.load(whole_dir / "checkpoint-000003.pt", weights_only=False)
    assert checkpoint["e2e"]["context"] == asdict(qwen)
    rebuilt = from_checkpoint(checkpoint, lexicon, bert_config=TINY)
    assert isinstance(rebuilt.context, QwenContextTower)
    rebuilt.load_state_dict(checkpoint["model"], strict=True)

    resumed_model = model()
    resumed = train(
        resumed_model,
        [lane()],
        resumable,
        tmp_path / "resumed",
        resume=whole_dir / "checkpoint-000003.pt",
    ).metrics

    def losses(path: Path) -> list[float]:
        return [
            float(record["loss"])
            for record in (json.loads(line) for line in path.read_text().splitlines())
            if record.get("event") == "step"
        ]

    assert losses(resumed) == losses(whole)[3:]
    weights = uninterrupted.state_dict()
    for name, tensor in resumed_model.state_dict().items():
        assert torch.equal(tensor, weights[name]), name


def test_length_buckets_are_quanta() -> None:
    assert [length_bucket(n) for n in (0, 1, 16, 17, 64)] == [16, 16, 16, 32, 64]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_a_replayed_tower_is_the_eager_tower(qwen: QwenContextConfig) -> None:
    """The cheap-card check: graphs at every bucket a step can hit, against eager."""
    tower = tower_of(replace(qwen, dtype="bfloat16")).cuda()
    graphs = TowerGraphs()
    for texts in (["北京"], ["我爱北京天安门", "钟", ""], ["绿"] * 9, ["我爱北京天安门" * 4] * 3):
        ids, mask = (tensor.cuda() for tensor in contexts(qwen, texts))
        eager = tower.frozen_states(ids, mask)
        replayed = graphs.states(tower, ids, mask)
        real = mask.bool()
        # The graph runs the padded shape, so its bf16 sums round differently.
        torch.testing.assert_close(replayed[real], eager[real], rtol=2e-2, atol=2e-2)
        # A second replay of the same shape is the same graph.
        before = len(graphs.captured)
        assert torch.equal(graphs.states(tower, ids, mask), replayed)
        assert len(graphs.captured) == before
    tower.load_state_dict(tower.state_dict())
    ids, mask = (tensor.cuda() for tensor in contexts(qwen, ["北京"]))
    graphs.states(tower, ids, mask)
    assert len(graphs.captured) == 1


def test_the_tokenizer_wrapper_refuses_a_tokenizer_without_a_pad(qwen_dir: Path) -> None:
    from transformers import AutoTokenizer

    bare = AutoTokenizer.from_pretrained(qwen_dir)
    bare.pad_token = None
    with pytest.raises(ValueError, match="no pad token"):
        QwenContextTokenizer(bare)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_a_warm_up_step_over_replayed_towers_issues_no_host_syncs(
    lexicon: Lexicon,
    spans: SpanVocab,
    tokenizer: StubTokenizer,
    reader_vocab: CharVocab,
    e2e_config: E2EConfig,
    qwen: QwenContextConfig,
) -> None:
    """S0's step as it trains: fp16 autocast over a bf16 tower replayed from graphs."""
    model = qwen_e2e(lexicon, reader_vocab, e2e_config, replace(qwen, dtype="bfloat16")).cuda()
    # Dropout off, so the eager and the replayed losses compare.
    model.eval()
    model.freeze_for_context_warm_up()
    with torch.no_grad():
        for layer in model.gated_layers():
            layer.gate.fill_(0.5)
    collator = Collator(
        tokenizer,
        load_context_tokenizer(qwen),
        CandidateSpace(spans, lexicon.candidate_mask),
        context_dropout=0.0,
    )
    batch = collator(
        [
            replace(example(spans, lexicon, "我爱北京", ("wo", "ai", "bei", "jing")), context="钟"),
            replace(
                example(spans, lexicon, "钟爱北京", ("zhong", "ai", "bei", "jing")),
                context="我爱绿钟北京",
            ),
        ]
    ).to(torch.device("cuda"))

    def loss() -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.float16):
            value = model(batch).loss
        assert value is not None
        return value

    eager = loss()
    eager.backward()
    tower = model.context
    assert isinstance(tower, QwenContextTower)
    tower.graphs = TowerGraphs()
    loss().backward()  # capture, and warm the allocator, outside the check
    torch.cuda.set_sync_debug_mode("error")
    try:
        replayed = loss()
        replayed.backward()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.testing.assert_close(replayed, eager, rtol=2e-2, atol=2e-2)
    assert all(p.grad is None for p in tower.model.parameters())
