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
from mlime.train.samples import Batch, Collator, CorpusStream, SampleBuilder
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
    assert {id(p) for p in groups[1]["params"]} == added

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
@pytest.mark.parametrize("layers", [None, 1], ids=["whole", "cut"])
def test_a_replayed_tower_is_the_eager_tower(qwen: QwenContextConfig, layers: int | None) -> None:
    """The cheap-card check: graphs at every bucket a step can hit, against eager."""
    tower = tower_of(replace(qwen, dtype="bfloat16", layers=layers)).cuda()
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


def test_a_cut_tower_is_the_whole_towers_stream_after_its_last_block_normed(
    qwen: QwenContextConfig,
) -> None:
    """#237: a tower cut at N builds N blocks and reads what the whole one has after block N."""
    whole = tower_of(qwen)
    cut = tower_of(replace(qwen, layers=1))
    assert len(cut.model.layers) == 1
    released = whole.model.state_dict()
    kept = cut.model.state_dict()
    assert set(kept) == {name for name in released if not name.startswith("layers.1.")}
    for name, tensor in kept.items():
        assert torch.equal(tensor, released[name]), name
    head = cut.head_state()
    assert len(head.keys) == len(head.values) == 1

    ids, mask = contexts(qwen, ["今天下午下雨了", "好", ""])
    with torch.no_grad():
        states = cut.frozen_states(ids, mask)
        for row, length in enumerate(mask.sum(dim=1).tolist()):
            joined = torch.cat([whole.head_ids, ids[row, :length]]).unsqueeze(0)
            output = whole.model(input_ids=joined, use_cache=False, output_hidden_states=True)
            # hidden_states[1] is the stream after the first block, before any norm.
            after = whole.model.norm(output.hidden_states[1][0])
            torch.testing.assert_close(head.rows, after[: head.length], rtol=0, atol=1e-5)
            torch.testing.assert_close(
                states[row, :length], after[head.length :], rtol=0, atol=1e-5
            )

    with pytest.raises(ValueError, match="cannot keep 3"):
        build_tower(replace(qwen, layers=3))
    with pytest.raises(ValueError, match="at least one layer"):
        replace(qwen, layers=0)


#: #237's two S0 variants, on the two-block tiny tower: cut after its first
#: block, and LoRA on every block's attention.
VARIANTS: dict[str, dict[str, object]] = {
    "layer-cut": {"layers": 1},
    "lora": {"lora": LoraConfig(rank=2, alpha=4.0)},
}

ADAPTER_NAMES = ("lora_a", "lora_b")


def s0_model(
    lexicon: Lexicon, vocab: CharVocab, config: E2EConfig, qwen: QwenContextConfig
) -> E2EModel:
    """S0 as a stage builds it: a MacBERT run's weights, Qwen swapped in, the warm-up's freeze.

    The source's gates are opened, as a trained run's are, so the context
    path has a gradient to train on.
    """
    source = tiny_e2e(lexicon, vocab, config)
    with torch.no_grad():
        for layer in source.gated_layers():
            layer.gate.fill_(0.5)
    weights = source.state_dict()
    model = tiny_e2e(lexicon, vocab, replace(config, context=qwen, context_warm_up=True))
    swap_in_qwen(model, weights, "source")
    model.freeze_for_context_warm_up()
    return model


def s0_batch(
    spans: SpanVocab,
    lexicon: Lexicon,
    tokenizer: StubTokenizer,
    qwen: QwenContextConfig,
    contexts: tuple[str, str] = ("明天下午三点见", "他说那个颜色"),
) -> Batch:
    """Two examples with *contexts*, collated for the Qwen tower."""
    collator = Collator(
        tokenizer,
        load_context_tokenizer(qwen),
        CandidateSpace(spans, lexicon.candidate_mask),
        context_dropout=0.0,
    )
    return collator(
        [
            replace(
                example(spans, lexicon, "中和北京", ("zhong", "he", "bei", "jing")),
                context=contexts[0],
            ),
            replace(example(spans, lexicon, "绿钟", ("lv", "zhong")), context=contexts[1]),
        ]
    )


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_s0_trains_the_gated_blocks_and_the_towers_additions_and_nothing_inherited(
    lexicon: Lexicon,
    spans: SpanVocab,
    tokenizer: StubTokenizer,
    reader_vocab: CharVocab,
    e2e_config: E2EConfig,
    qwen: QwenContextConfig,
    variant: str,
) -> None:
    """Each variant's warm-up moves the four blocks, the context norm and any adapters alone."""
    tower = replace(qwen, **VARIANTS[variant])
    model = s0_model(lexicon, reader_vocab, e2e_config, tower)
    names = {name for name, _ in model.named_parameters()}
    blocks = {name for name in names if name.startswith("cross_attention.")}
    adapters = {name for name in names if name.rsplit(".", 1)[-1] in ADAPTER_NAMES}
    layers = 1 if tower.layers is not None else TINY_QWEN.num_hidden_layers
    # Every targeted projection of every kept block, two matrices each.
    assert len(adapters) == (0 if tower.lora is None else layers * len(tower.lora.targets) * 2)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == blocks | adapters | {"context.norm.weight", "context.norm.bias"}
    # The adapters, the norm and the blocks' key and value weights are new and
    # take the new rate; what the blocks carry over from #132 takes the base one.
    groups = model.parameter_groups(1e-5, 1e-4)
    by_id = {id(p): name for name, p in model.named_parameters()}
    new = {by_id[id(p)] for p in groups[1]["params"]}
    base = {by_id[id(p)] for p in groups[0]["params"]}
    assert new == set(model.new_context_parameters())
    assert adapters <= new
    assert base == blocks - new

    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    optimiser = torch.optim.AdamW(model.parameter_groups(1e-3, 1e-3))
    batch = s0_batch(spans, lexicon, tokenizer, tower)
    model.train()
    for _ in range(2):
        loss = model(batch).loss
        assert loss is not None
        loss.backward()
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
    after = model.state_dict()
    moved = {name for name, tensor in after.items() if not torch.equal(tensor, before[name])}
    assert moved <= trainable
    assert adapters <= moved
    assert {f"cross_attention.{i}.key.weight" for i in range(len(model.gated_layers()))} <= moved

    # The run's checkpoint record names the variant, and rebuilds it exactly.
    record = asdict(model.config)
    assert record["context"] == asdict(tower)
    rebuilt = from_checkpoint({"e2e": record, "model": after}, lexicon, bert_config=TINY)
    rebuilt_tower = rebuilt.context
    assert isinstance(rebuilt_tower, QwenContextTower)
    assert len(rebuilt_tower.model.layers) == layers
    rebuilt.load_state_dict(after, strict=True)


def test_an_adapted_tower_trains_inside_the_compiled_pass_and_computes_the_eager_step(
    lexicon: Lexicon,
    spans: SpanVocab,
    tokenizer: StubTokenizer,
    reader_vocab: CharVocab,
    e2e_config: E2EConfig,
    qwen: QwenContextConfig,
) -> None:
    """#237: the adapted tower runs inside the compiled ``encode``, not eager before it.

    Eager, its forward and backward are a launch for every RMSNorm, rotary and
    adapter op of every block, and the L4 sat idle half the step issuing them.
    Compiled, it is one graph with no break, whose loss and gradients are the
    eager step's. The adapters' B is moved off zero first, so A has a gradient
    to compare too.
    """
    from mlime.train.e2e import compile_modules

    tower = replace(qwen, lora=LoraConfig(rank=2, alpha=4.0))
    eager = s0_model(lexicon, reader_vocab, e2e_config, tower).eval()
    torch.manual_seed(5)
    with torch.no_grad():
        for name, parameter in eager.named_parameters():
            if name.endswith("lora_b"):
                parameter.normal_(std=0.1)
    compiled = s0_model(lexicon, reader_vocab, e2e_config, tower).eval()
    compiled.load_state_dict(eager.state_dict())
    assert isinstance(compiled.context, QwenContextTower)
    assert not compiled.context.separate
    compile_modules(compiled, "default")
    batch = s0_batch(spans, lexicon, tokenizer, tower)

    torch._dynamo.utils.counters.clear()
    losses = []
    for model in (eager, compiled):
        loss = model(batch).loss
        assert loss is not None
        loss.backward()
        losses.append(loss.detach())
    assert not torch._dynamo.utils.counters["graph_break"]
    torch.testing.assert_close(losses[1], losses[0])
    trained = [
        (name, p, dict(compiled.named_parameters())[name])
        for name, p in eager.named_parameters()
        if p.requires_grad
    ]
    assert any(name.endswith("lora_a") for name, _, _ in trained)
    for name, before, after in trained:
        assert before.grad is not None and after.grad is not None, name
        torch.testing.assert_close(after.grad, before.grad, msg=name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton scheduling needs a CUDA device")
def test_an_adapted_tower_compiled_on_cuda_computes_the_eager_step(
    lexicon: Lexicon,
    spans: SpanVocab,
    tokenizer: StubTokenizer,
    reader_vocab: CharVocab,
    e2e_config: E2EConfig,
    qwen: QwenContextConfig,
) -> None:
    """#237: the adapted tower's compiled ``encode`` on CUDA, as the L4 leg runs it.

    The CPU check above never reaches Triton's scheduler, and that is where
    the leg failed: Inductor's tiling analysis asserted on the context
    LayerNorm, whose rows were the head and the context together, ``C * (H + T)``
    with ``T`` symbolic. Here the bf16 tower under fp16 autocast compiles
    dynamic on its first batch and is reused on a batch of another context
    width; each step's loss and gradients are the eager step's to the
    precision's rounding.
    """
    from mlime.train.e2e import compile_modules

    device = torch.device("cuda")
    tower = replace(qwen, dtype="bfloat16", lora=LoraConfig(rank=2, alpha=4.0))
    eager = s0_model(lexicon, reader_vocab, e2e_config, tower).to(device).eval()
    torch.manual_seed(5)
    with torch.no_grad():
        for name, parameter in eager.named_parameters():
            if name.endswith("lora_b"):
                parameter.normal_(std=0.1)
    compiled = s0_model(lexicon, reader_vocab, e2e_config, tower).to(device).eval()
    compiled.load_state_dict(eager.state_dict())
    compile_modules(compiled, "default")
    batches = [
        s0_batch(spans, lexicon, tokenizer, tower),
        s0_batch(spans, lexicon, tokenizer, tower, ("昨天晚上我们一起去看了电影", "好")),
    ]
    assert batches[0].context_ids.shape[1] != batches[1].context_ids.shape[1]

    torch._dynamo.utils.counters.clear()
    for batch in (batch.to(device) for batch in batches):
        losses = []
        for model in (eager, compiled):
            model.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(batch).loss
            assert loss is not None
            loss.backward()
            losses.append(loss.detach().float())
        torch.testing.assert_close(losses[1], losses[0], rtol=1e-2, atol=1e-3)
        parameters = dict(compiled.named_parameters())
        for name, before in eager.named_parameters():
            if not before.requires_grad:
                continue
            after = parameters[name]
            assert before.grad is not None and after.grad is not None, name
            torch.testing.assert_close(
                after.grad.float(), before.grad.float(), rtol=5e-2, atol=1e-3, msg=name
            )
    assert not torch._dynamo.utils.counters["graph_break"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="device syncs need a CUDA device")
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_an_s0_step_of_each_variant_never_waits_on_the_device(
    lexicon: Lexicon,
    spans: SpanVocab,
    tokenizer: StubTokenizer,
    reader_vocab: CharVocab,
    e2e_config: E2EConfig,
    qwen: QwenContextConfig,
    variant: str,
) -> None:
    """#237's cheap-card check: a whole warm-up step queues without a sync.

    The loop's step -- copy in, forward under fp16 autocast over the bf16
    tower, scaled backward, clip, fused AdamW, scaler update -- as the
    ranking step's check runs it. The cut tower is frozen and replays its
    forward from graphs; the adapted one runs eager with autograd through
    every block. The first step warms up what is made once; the second runs
    with every synchronising call an error. Nothing inherited moves.
    """
    device = torch.device("cuda")
    tower = replace(qwen, dtype="bfloat16", **VARIANTS[variant])
    model = s0_model(lexicon, reader_vocab, e2e_config, tower).to(device).train()
    context = model.context
    assert isinstance(context, QwenContextTower)
    if context.frozen:
        context.graphs = TowerGraphs()
    inherited = {
        name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad
    }
    held = s0_batch(spans, lexicon, tokenizer, tower).pin_memory()
    optimiser = torch.optim.AdamW(model.parameter_groups(2e-5, 1e-4), fused=True)
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
    for name, parameter in model.named_parameters():
        if name in inherited:
            assert torch.equal(parameter, inherited[name]), name
