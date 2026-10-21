"""Qwen3-0.6B-Base as the context tower (#219, #222).

MacBERT read the text left of the cursor as 64 BERT tokens; this tower reads it
with a causal model that has seen several orders of magnitude more text. It
serves the interface MacBERT's tower serves -- a :class:`ContextMemory` per
context row, read by the fill tower's top layers through gated cross-attention
-- so nothing downstream of the context memory changes.

Three things make it a context tower rather than a language model:

- **The constant head.** Every context is read after ``<|endoftext|>\\n``. Its
  keys and values in every layer, and its two output rows, depend on nothing
  but the weights, so they are computed once (:class:`HeadState`) and every
  context row attends to them as a prefix, at positions 0 and 1, with its own
  tokens from position 2. The head is the attention sink Qwen expects at the
  start of a sequence, and it is what the product ships precomputed: without it
  the int8 tower breaks its first two positions (#219).
- **The width fold.** Qwen's rows are 1024 wide and the fill tower's 768. The
  four gated cross-attention blocks take the difference in their key and value
  projections (:class:`ProjectedCrossAttention`), behind one new LayerNorm on
  the context rows: the query, the output projection and the gates are the
  trained ones carried over from #132, and only the key and value weights are
  new.
- **Frozen, or LoRA.** The tower's own weights never train. With
  :class:`LoraConfig` its attention projections carry low-rank adapters that
  do; without, it runs under ``no_grad`` -- no autograd graph through 28
  layers -- and on CUDA it can be replayed from captured graphs
  (:class:`TowerGraphs`), one launch for the whole forward.

The context side is tokenised with Qwen's own BPE (:class:`QwenContextTokenizer`)
and the fill side keeps MacBERT's one character a position.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from torch import nn

from mlime.logging import log
from mlime.train.model import ContextCrossAttention, ContextMemory, ContextTower
from mlime.train.shapes import bucket

#: Qwen's first two positions, read before every context: the end-of-text
#: token Qwen trained documents to follow, and a line break.
CONSTANT_HEAD = "<|endoftext|>\n"

#: The tower the context moves to (#219): the base model, pinned to the
#: commit whose config, tokenizer and weights the checks were made against.
QWEN_MODEL = "Qwen/Qwen3-0.6B-Base"
QWEN_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"

#: The context length a captured tower graph pads to a multiple of. Contexts
#: are at most 64 tokens, so a run captures at most four lengths a row bucket.
LENGTH_QUANTUM = 16


@dataclass(frozen=True)
class LoraConfig:
    """Low-rank adapters on the tower's attention projections.

    Each adapted projection computes ``W x + (alpha / rank) * B A x`` with ``A``
    random and ``B`` zero, so an adapted tower starts as exactly the frozen one.
    """

    rank: int
    alpha: float
    targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError(f"a LoRA rank is positive, got {self.rank}")
        if self.alpha <= 0.0:
            raise ValueError(f"a LoRA alpha is positive, got {self.alpha}")
        if not self.targets:
            raise ValueError("LoRA adapts at least one projection")


@dataclass(frozen=True)
class QwenContextConfig:
    """Which Qwen reads the context, at which commit, in which precision.

    ``model`` and ``revision`` name what ``from_pretrained`` loads: the config
    and tokenizer for every build, the weights only when a run swaps the tower
    in (a checkpoint carries them after that). ``dtype`` is the tower's own
    precision, outside the training's fp16 autocast: Qwen's activations do not
    fit fp16's range. ``lora`` None keeps the tower frozen.
    """

    model: str = QWEN_MODEL
    revision: str | None = QWEN_REVISION
    head: str = CONSTANT_HEAD
    dtype: str = "bfloat16"
    lora: LoraConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(getattr(torch, self.dtype, None), torch.dtype):
            raise ValueError(f"{self.dtype!r} is not a torch dtype")
        if not self.head:
            raise ValueError("the constant head holds at least one token")

    @property
    def torch_dtype(self) -> torch.dtype:
        """The tower's precision as a torch dtype."""
        dtype: torch.dtype = getattr(torch, self.dtype)
        return dtype

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> QwenContextConfig:
        """Rebuild the config from a checkpoint's record of it."""
        fields = dict(record)
        lora = fields.get("lora")
        if lora is not None:
            fields["lora"] = LoraConfig(**{**lora, "targets": tuple(lora["targets"])})
        return cls(**fields)


class LoraLinear(nn.Linear):
    """A frozen linear layer plus a trainable low-rank update.

    A subclass rather than a wrapper, so the base weight keeps its name in the
    state dict: a frozen tower's checkpoint loads into an adapted one with only
    the adapters missing, and they start at zero contribution.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__(
            base.in_features,
            base.out_features,
            bias=base.bias is not None,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        self.weight = base.weight
        self.bias = base.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        # The adapters stay in fp32 whatever the tower's precision: they are
        # what AdamW moves, and bf16 cannot hold its small updates.
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base: torch.Tensor = super().forward(x)
        update = (x.float() @ self.lora_a.T) @ self.lora_b.T
        result: torch.Tensor = base + (self.scale * update).to(base.dtype)
        return result


@dataclass(frozen=True)
class HeadState:
    """The constant head through the tower: each layer's keys and values, and its rows.

    ``keys[i]`` and ``values[i]`` are layer *i*'s ``[1, kv_heads, H, head_dim]``,
    and ``rows`` the head's ``[H, width]`` output, the final norm applied.
    """

    keys: tuple[torch.Tensor, ...]
    values: tuple[torch.Tensor, ...]
    rows: torch.Tensor

    @property
    def length(self) -> int:
        """The head's token count."""
        return int(self.rows.shape[0])


class _Recorder:
    """The cache a head forward writes: it keeps each layer's keys and values."""

    def __init__(self) -> None:
        self.keys: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []

    def update(
        self, keys: torch.Tensor, values: torch.Tensor, layer: int, *_: object, **__: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer != len(self.keys):
            raise RuntimeError(
                f"the head's layers ran out of order: {layer} after {len(self.keys)}"
            )
        self.keys.append(keys)
        self.values.append(values)
        return keys, values


class _HeadPrefix:
    """The cache a context forward reads: the head's keys and values before its own.

    Qwen's attention hands each layer's new keys and values to its cache and
    attends to what comes back; this one puts the head in front, broadcast to
    every row, and keeps nothing.
    """

    def __init__(self, head: HeadState):
        self.head = head

    def update(
        self, keys: torch.Tensor, values: torch.Tensor, layer: int, *_: object, **__: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = keys.shape[0]
        head_keys = self.head.keys[layer].expand(rows, -1, -1, -1)
        head_values = self.head.values[layer].expand(rows, -1, -1, -1)
        return torch.cat([head_keys, keys], dim=2), torch.cat([head_values, values], dim=2)


class ProjectedCrossAttention(ContextCrossAttention):
    """A fill layer's gated cross-attention into context rows of another width.

    The arithmetic is :class:`~mlime.train.model.GatedCrossAttention`'s -- a
    pre-norm query, scaled dot-product attention over the real context rows,
    an output projection behind a scalar gate switched per example -- with the
    key and value projections taking the context's width to the fill tower's.
    Keys and values are projected once per *context* row and then gathered for
    the fill rows that read it, so a ranking batch's readings of one example
    share them.
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float, width: int):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError(f"hidden {hidden_size} does not split over {num_heads} heads")
        self.heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(hidden_size)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(width, hidden_size)
        self.value = nn.Linear(width, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        # nn.MultiheadAttention's initialisation, which the carried blocks had.
        for projection in (self.query, self.key, self.value):
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)
        nn.init.zeros_(self.out.bias)
        self.gate = nn.Parameter(torch.zeros(()))

    def new_parameters(self) -> tuple[str, ...]:
        """The key and value weights: the only part no earlier checkpoint holds."""
        return ("key.weight", "value.weight")

    def _heads(self, projected: torch.Tensor) -> torch.Tensor:
        rows, length, _ = projected.shape
        return projected.view(rows, length, self.heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory: ContextMemory,
        rows: torch.Tensor,
        has_context: torch.Tensor,
    ) -> torch.Tensor:
        """Add the gated, per-example-switched context term to *hidden_states*."""
        keys = self._heads(self.key(memory.states)).index_select(0, rows)
        values = self._heads(self.value(memory.states)).index_select(0, rows)
        query = self._heads(self.query(self.norm(hidden_states)))
        attended = nn.functional.scaled_dot_product_attention(
            query,
            keys,
            values,
            attn_mask=memory.mask.index_select(0, rows)[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0,
        )
        merged = attended.transpose(1, 2).reshape(hidden_states.shape)
        gated: torch.Tensor = hidden_states + self.gate * has_context[:, None, None] * self.out(
            merged
        )
        return gated


def carried_cross_attention(
    weights: Mapping[str, torch.Tensor], prefix: str, hidden_size: int
) -> dict[str, torch.Tensor]:
    """A :class:`GatedCrossAttention` block's weights as a projected block's.

    The packed ``in_proj`` splits into query, key and value; the query and
    every bias carry over, as do the norm, the output projection and the
    gate. The key and value *weights* read MacBERT's 768-wide rows and have no
    counterpart over Qwen's, so they are not returned: the projected block's
    own initialisation stands for them.
    """
    packed = weights[f"{prefix}attention.in_proj_weight"]
    bias = weights[f"{prefix}attention.in_proj_bias"]
    if packed.shape != (3 * hidden_size, hidden_size):
        raise ValueError(f"{prefix} packs {tuple(packed.shape)}, not three {hidden_size}-wide maps")
    return {
        "norm.weight": weights[f"{prefix}norm.weight"],
        "norm.bias": weights[f"{prefix}norm.bias"],
        "query.weight": packed[:hidden_size],
        "query.bias": bias[:hidden_size],
        "key.bias": bias[hidden_size : 2 * hidden_size],
        "value.bias": bias[2 * hidden_size :],
        "out.weight": weights[f"{prefix}attention.out_proj.weight"],
        "out.bias": weights[f"{prefix}attention.out_proj.bias"],
        "gate": weights[f"{prefix}gate"],
    }


class QwenContextTower(ContextTower):
    """Qwen3 reading the context after its constant head, frozen or with LoRA.

    ``model`` is the Qwen3 decoder stack without its LM head; its layers are
    run one by one here, as the fill tower's are, so the head's keys and values
    can be handed to each layer as a prefix and the masks built without the
    host checks transformers' own mask builder makes. ``norm`` is the new
    LayerNorm on the context rows, the only part of the tower that trains in
    S0. ``graphs``, set by the trainer on a CUDA run that asks for it, replays
    a frozen tower's forward from captured CUDA graphs.
    """

    separate: ClassVar[bool] = True

    model: Any
    norm: nn.LayerNorm
    head_ids: torch.Tensor

    def __init__(self, model: Any, head_ids: Sequence[int], config: QwenContextConfig):
        super().__init__()
        implementation = getattr(model.config, "_attn_implementation", None)
        if implementation != "sdpa":
            raise RuntimeError(
                f"the tower is fed 4D boolean masks, which only sdpa reads as 'True attends'; "
                f"this tower's attention is {implementation!r}"
            )
        if not head_ids:
            raise ValueError("the constant head tokenised to nothing")
        self.tower_config = config
        self.model = model.to(config.torch_dtype)
        self.model.requires_grad_(False)
        if config.lora is not None:
            self._adapt(config.lora)
        self.norm = nn.LayerNorm(model.config.hidden_size)
        self.register_buffer("head_ids", torch.tensor(head_ids, dtype=torch.long), persistent=False)
        self.graphs: TowerGraphs | None = None
        self.checkpoint_layers = False
        self._head: HeadState | None = None
        self.register_load_state_dict_post_hook(  # type: ignore[no-untyped-call]  # torch ships it untyped
            QwenContextTower._forget_head
        )

    def _adapt(self, lora: LoraConfig) -> None:
        """Replace each targeted projection of every layer with its LoRA form."""
        for layer in self.model.layers:
            attention = layer.self_attn
            for name in lora.targets:
                base = getattr(attention, name, None)
                if not isinstance(base, nn.Linear):
                    raise ValueError(f"the tower's attention has no {name!r} projection to adapt")
                setattr(attention, name, LoraLinear(base, lora.rank, lora.alpha))

    @staticmethod
    def _forget_head(module: nn.Module, _: object) -> None:
        if isinstance(module, QwenContextTower):
            module._head = None

    @property
    def width(self) -> int:
        """The width of the rows the tower emits."""
        return int(self.model.config.hidden_size)

    @property
    def frozen(self) -> bool:
        """Whether nothing of the tower's own trains (no LoRA)."""
        return self.tower_config.lora is None

    def new_parameters(self) -> tuple[str, ...]:
        """The LayerNorm on the context rows and every LoRA adapter."""
        adapters = tuple(
            name for name, _ in self.named_parameters() if name.rsplit(".", 1)[-1] in LORA_NAMES
        )
        return ("norm.weight", "norm.bias", *adapters)

    def enable_checkpointing(self) -> None:
        """Recompute each layer in backward; only an adapted tower has a backward."""
        self.checkpoint_layers = True

    def cross_attention(
        self, hidden_size: int, num_heads: int, dropout: float
    ) -> ContextCrossAttention:
        """A projected block from this tower's width to the fill tower's."""
        return ProjectedCrossAttention(hidden_size, num_heads, dropout, self.width)

    def _layers(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
        positions: torch.Tensor,
        cache: _Recorder | _HeadPrefix,
    ) -> torch.Tensor:
        """Qwen's stack over *ids* with the 4D *mask*, writing to and reading *cache*."""
        model = self.model
        hidden: torch.Tensor = model.embed_tokens(ids)
        rotary = model.rotary_emb(hidden, positions)
        for layer in model.layers:
            if self.checkpoint_layers and torch.is_grad_enabled():
                hidden = torch.as_tensor(
                    torch.utils.checkpoint.checkpoint(
                        layer,
                        hidden,
                        attention_mask=mask,
                        position_embeddings=rotary,
                        position_ids=positions,
                        past_key_values=cache,
                        use_reentrant=False,
                    )
                )
            else:
                hidden = torch.as_tensor(
                    layer(
                        hidden,
                        attention_mask=mask,
                        position_embeddings=rotary,
                        position_ids=positions,
                        past_key_values=cache,
                    )
                )
        normed: torch.Tensor = model.norm(hidden)
        return normed

    def compute_head(self) -> HeadState:
        """The constant head through every layer, from the current weights."""
        length = self.head_ids.numel()
        recorder = _Recorder()
        causal = torch.ones(length, length, dtype=torch.bool, device=self.head_ids.device).tril()
        with torch.autocast(self.head_ids.device.type, enabled=False):
            rows = self._layers(
                self.head_ids.unsqueeze(0),
                causal[None, None],
                torch.arange(length, device=self.head_ids.device).unsqueeze(0),
                recorder,
            )
        return HeadState(keys=tuple(recorder.keys), values=tuple(recorder.values), rows=rows[0])

    def head_state(self) -> HeadState:
        """The head's state: computed once for a frozen tower, every call for an adapted one.

        A frozen tower's head depends on weights that never move, so it is
        computed once, without a graph, and kept until the weights are loaded
        again or the tower moves device. An adapted tower's depends on the
        adapters, so it is recomputed each forward and trains with them.
        """
        if not self.frozen:
            return self.compute_head()
        head = self._head
        if head is None or head.rows.device != self.head_ids.device:
            with torch.no_grad():
                head = self._head = self.compute_head()
        return head

    def states(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``[C, T]`` context ids, right-padded under *mask*, to ``[C, T, width]`` rows.

        Each row attends to the head and to its own tokens up to itself; its
        tokens sit at positions ``H..H+T-1`` after the head's ``0..H-1``, so a
        row reads exactly as the head and the row in one sequence would. The
        tower runs outside the training's autocast, in its own precision.
        """
        head = self.head_state()
        length = ids.shape[1]
        own = torch.ones(length, length, dtype=torch.bool, device=ids.device).tril()
        visible = torch.cat(
            [
                torch.ones(length, head.length, dtype=torch.bool, device=ids.device).expand(
                    ids.shape[0], -1, -1
                ),
                own & mask.bool()[:, None, :],
            ],
            dim=2,
        )
        positions = (torch.arange(length, device=ids.device) + head.length).unsqueeze(0)
        with torch.autocast(ids.device.type, enabled=False):
            return self._layers(ids, visible[:, None], positions, _HeadPrefix(head))

    def frozen_states(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """:meth:`states` without autograd: what a frozen tower runs and a graph captures."""
        with torch.no_grad():
            return self.states(ids, mask)

    def memory(self, ids: torch.Tensor, mask: torch.Tensor) -> ContextMemory:
        """The rows the cross-attention reads: the head's, then the context's, normed.

        The head's rows are real for every context row, so a row whose context
        was dropped, or is empty, still attends to something; its gate switch
        is what keeps it out of the fill tower. A batch with no context at all
        -- every one dropped, or a held-out pass without context -- is zero
        tokens wide and reads the head alone.
        """
        head = self.head_state()
        rows = ids.shape[0]
        if ids.shape[1] == 0:
            states = head.rows.new_empty((rows, 0, head.rows.shape[-1]))
        elif self.frozen:
            states = (
                self.frozen_states(ids, mask)
                if self.graphs is None
                else self.graphs.states(self, ids, mask)
            )
        else:
            states = self.states(ids, mask)
        joined = torch.cat([head.rows.expand(rows, -1, -1), states], dim=1)
        real = torch.cat(
            [torch.ones(rows, head.length, dtype=torch.bool, device=ids.device), mask.bool()],
            dim=1,
        )
        return ContextMemory(states=self.norm(joined.float()), mask=real)


#: The parameter names a :class:`LoraLinear` adds.
LORA_NAMES = frozenset({"lora_a", "lora_b"})


def length_bucket(length: int) -> int:
    """*length* rounded up to a multiple of :data:`LENGTH_QUANTUM`, at least one quantum."""
    return max(LENGTH_QUANTUM, -(-length // LENGTH_QUANTUM) * LENGTH_QUANTUM)


class TowerGraphs:
    """A frozen tower's forwards captured as CUDA graphs, one a bucketed shape.

    Qwen is 28 layers of some forty kernels each; issued from the host one by
    one they are a step's worth of launches in a step that is already bound by
    launches. Replayed, a forward is one launch. Contexts are padded to
    :func:`~mlime.train.shapes.bucket` rows and :func:`length_bucket`
    tokens -- padded rows attend to the head alone and padded tokens are
    masked, so the real rows' outputs are the eager tower's -- and every
    graph's inputs are views of one pair of flat buffers, grown when a larger
    shape arrives (which drops the graphs that baked the old addresses), with
    every graph's working memory in one shared pool, as
    :class:`~mlime.train.e2e_search.SearchGraphs` keeps them.

    The graphs bake the head's tensors in, so a head computed afresh -- the
    weights loaded again -- drops them too.
    """

    def __init__(self) -> None:
        self.pool = torch.cuda.graph_pool_handle()
        self.flat: dict[str, torch.Tensor] = {}
        self.captured: dict[tuple[int, int], tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}
        self.head: HeadState | None = None

    def _drop(self, reason: str) -> None:
        if self.captured:
            log.info("tower graphs dropped", reason=reason, graphs=len(self.captured))
            self.captured.clear()
            # A dropped graph's pool lives on until its memory is freed; the
            # next graphs take a pool of their own.
            self.pool = torch.cuda.graph_pool_handle()

    def _view(self, name: str, shape: tuple[int, int], like: torch.Tensor) -> torch.Tensor:
        """A ``shape`` view of the flat buffer *name*, grown when too small."""
        count = shape[0] * shape[1]
        flat = self.flat.get(name)
        if flat is None or flat.numel() < count:
            self.flat[name] = torch.empty(
                max(count, 0 if flat is None else flat.numel()),
                dtype=like.dtype,
                device=like.device,
            )
            self._drop(f"{name} buffer grown")
            flat = self.flat[name]
        return flat[:count].view(shape)

    def states(
        self, tower: QwenContextTower, ids: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """:meth:`QwenContextTower.frozen_states` over *ids*, from its shape's graph."""
        if not tower.frozen:
            raise ValueError("only a frozen tower has a forward to replay; an adapted one trains")
        head = tower.head_state()
        if self.head is not head:
            self._drop("head recomputed")
            self.head = head
        rows, length = ids.shape
        shape = (bucket(rows), length_bucket(length))
        static_ids = self._view("ids", shape, ids)
        static_mask = self._view("mask", shape, mask)
        static_ids.zero_()
        static_mask.zero_()
        static_ids[:rows, :length].copy_(ids)
        static_mask[:rows, :length].copy_(mask)
        captured = self.captured.get(shape)
        if captured is None:
            captured = self.captured[shape] = self._capture(tower, static_ids, static_mask)
            log.info(
                "tower graph captured", rows=shape[0], length=shape[1], graphs=len(self.captured)
            )
        graph, output = captured
        graph.replay()
        # A copy, because the next replay writes the same memory and the
        # context rows are saved for the backward of the step that reads them.
        return output[:rows, :length].clone()

    def _capture(
        self, tower: QwenContextTower, ids: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
        """Warm the forward up on a side stream, then capture it over the static inputs."""
        side = torch.cuda.Stream()  # type: ignore[no-untyped-call]  # torch ships it untyped
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            tower.frozen_states(ids, mask)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            output = tower.frozen_states(ids, mask)
        return graph, output


class QwenContextTokenizer:
    """Qwen's BPE over the context, in the shape the collator reads.

    No special tokens -- the head is the tower's, not the batch's -- padded on
    the right, and truncated on the *left*: the context runs up to the cursor,
    so a context longer than the budget keeps its end.
    """

    def __init__(self, tokenizer: Any):
        if tokenizer.pad_token_id is None:
            raise ValueError("the context tokenizer has no pad token; Qwen's base model names one")
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "left"
        self.tokenizer = tokenizer

    def __call__(
        self, text: list[str], padding: bool, truncation: bool, max_length: int
    ) -> Mapping[str, list[list[int]]]:
        encoded = self.tokenizer(
            text,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            add_special_tokens=False,
        )
        return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}

    def head_ids(self, head: str) -> list[int]:
        """The ids of the constant head *head*."""
        ids: list[int] = self.tokenizer(head, add_special_tokens=False)["input_ids"]
        return ids


def load_context_tokenizer(config: QwenContextConfig) -> QwenContextTokenizer:
    """The tokenizer of the Qwen *config* names, at its revision."""
    from transformers import AutoTokenizer

    return QwenContextTokenizer(
        AutoTokenizer.from_pretrained(config.model, revision=config.revision)
    )


def build_tower(config: QwenContextConfig) -> QwenContextTower:
    """The tower *config* names, its shape from Qwen's config and its weights random.

    Every build takes the weights from somewhere else: a run that swaps the
    tower in from Qwen's release (:func:`load_pretrained`), and every later
    build -- a resume, an evaluation -- from the run's own checkpoint.
    """
    from transformers import AutoConfig, Qwen3Model

    tokenizer = load_context_tokenizer(config)
    qwen = AutoConfig.from_pretrained(config.model, revision=config.revision)
    qwen._attn_implementation = "sdpa"
    tower = QwenContextTower(Qwen3Model(qwen), tokenizer.head_ids(config.head), config)
    log.info(
        "qwen context tower built",
        model=config.model,
        revision=config.revision,
        head=tower.head_ids.tolist(),
        lora=config.lora is not None,
    )
    return tower


def load_pretrained(tower: QwenContextTower) -> None:
    """Load Qwen's released weights into *tower*, every one of them.

    Only the adapters an adapted tower adds may be left at their
    initialisation; anything else missing or left over is a different model.
    """
    from transformers import Qwen3Model

    config = tower.tower_config
    released = Qwen3Model.from_pretrained(
        config.model,
        revision=config.revision,
        dtype=config.torch_dtype,
        attn_implementation="sdpa",
    )
    result = tower.model.load_state_dict(released.state_dict(), strict=False)
    missing = [name for name in result.missing_keys if name.rsplit(".", 1)[-1] not in LORA_NAMES]
    if missing or result.unexpected_keys:
        raise ValueError(
            f"{config.model}@{config.revision} does not fit the tower: "
            f"missing {missing[:4]}, unexpected {result.unexpected_keys[:4]}"
        )
    log.info("qwen context tower loaded", model=config.model, revision=config.revision)
