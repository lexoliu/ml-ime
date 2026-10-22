"""Route A: two MacBERT-initialised towers, one filling and one reading context.

The fill tower is an encoder over typed *spans* rather than over text. Every
input position is the base ``[MASK]`` embedding plus the additive embedding of
the span typed there, so the tower sees a sentence-shaped hole and knows what was
pressed at each position. Its head is the base MLM head with its rows cut down to
the characters the lexicon and the vocabulary share, and its softmax at each
position is restricted further to the characters that position's span admits.
That restriction is the difference between a language model and an input method:
the model never spends probability on a character the user could not have meant.

The context tower is the same base with its own weights, encoding whatever was on
screen before. It reaches the fill tower only through gated cross-attention added
to the top layers, and the gates start at zero -- so at initialisation route A is
exactly the fill tower, and any accuracy the context buys is accuracy the model
chose to learn rather than an artefact of the wiring. Because the gate is a
multiplier, an example whose context was dropped costs nothing: the same forward
pass runs with the term scaled to zero.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import torch
import torch.utils.checkpoint
from torch import nn
from transformers import BertConfig, BertForMaskedLM, BertModel

from mlime.logging import log
from mlime.train.lexicon import Lexicon
from mlime.train.samples import IGNORE_INDEX, LETTER_PAD, Batch
from mlime.train.spans import UNKNOWN_SPAN, SpanVocab

#: Parameters under these prefixes did not come from the pretrained checkpoint and
#: train at the higher learning rate.
NEW_PARAMETER_PREFIXES = ("span_embeddings", "cross_attention", "letter_encoder")

#: Width of the letter encoder's internal space before the projection to the
#: tower width. Sixty-four is plenty to hold six letters' identities.
LETTER_WIDTH = 64


class LetterEncoder(nn.Module):
    """A span vector out of the keys actually pressed, added through a zero gate.

    ``letters`` and ``positions`` summed per key, meaned over the span, then a
    projection that starts at zero -- so the encoder separates ``na`` from
    ``an`` (the position table carries the order), and a fresh model's output
    is bit-identical to one without the path, because the projection's zero
    keeps the whole term exactly 0.
    """

    def __init__(self, hidden_size: int, width: int = LETTER_WIDTH, max_letters: int = 12):
        super().__init__()
        self.letters = nn.Embedding(LETTER_PAD + 1, width, padding_idx=LETTER_PAD)
        self.positions = nn.Embedding(max_letters, width)
        self.project = nn.Linear(width, hidden_size, bias=False)
        nn.init.zeros_(self.project.weight)

    def forward(self, span_letters: torch.Tensor) -> torch.Tensor:
        """``[B, W, L]`` letter ids to ``[B, W, hidden]`` additive span terms."""
        valid = span_letters != LETTER_PAD
        positions = self.positions.weight[: span_letters.shape[-1]]
        letters = (self.letters(span_letters) + positions) * valid[..., None]
        pooled = letters.sum(dim=2) / valid.sum(dim=2, keepdim=True).clamp(min=1)
        encoded: torch.Tensor = self.project(pooled)
        return encoded


@dataclass(frozen=True)
class RouteAConfig:
    """Everything about route A's shape that is not the base checkpoint's own."""

    base_model: str = "hfl/chinese-macbert-base"
    cross_attention_layers: int = 4
    label_smoothing: float = 0.05
    cross_attention_dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.cross_attention_layers < 0:
            raise ValueError(
                f"cross_attention_layers must not be negative, got {self.cross_attention_layers}"
            )
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0, 1), got {self.label_smoothing}")


@dataclass(frozen=True)
class ContextMemory:
    """What the fill tower's cross-attention reads of the context: rows, and which are real.

    ``states [C, S, width]`` and ``mask [C, S]`` (true where real) hold one row
    per *context*; a batch's ``context_rows`` names the one each fill row reads.
    """

    states: torch.Tensor
    mask: torch.Tensor


class ContextCrossAttention(nn.Module):
    """A fill layer's gated attention into the context memory.

    The gate is a single scalar rather than a vector so that "the context is off"
    is one number per layer, readable straight out of a checkpoint. At zero the
    module returns its input unchanged -- not approximately, exactly -- which is
    what makes "does context help?" answerable by training rather than by
    argument.
    """

    gate: nn.Parameter

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory: ContextMemory,
        rows: torch.Tensor,
        has_context: torch.Tensor,
    ) -> torch.Tensor:
        """*hidden_states* ``[B, W, H]`` plus the gated term read from *memory*.

        Fill row *i* reads context row ``rows[i]``; ``has_context [B]`` switches
        the term per fill row.
        """
        raise NotImplementedError

    def new_parameters(self) -> tuple[str, ...]:
        """Names of the block's parameters no earlier checkpoint of its model holds."""
        return ()


class ContextTower(nn.Module):
    """What reads the context, and how the fill tower's blocks read it back.

    ``separate`` says where the tower runs: inside the compiled encoder pass
    (a tower that trains with it), or before it, outside the compiled region
    (a frozen tower, which may run from captured graphs of its own).
    """

    separate: ClassVar[bool]

    def memory(self, ids: torch.Tensor, mask: torch.Tensor) -> ContextMemory:
        """The context rows of ``[C, T]`` token *ids* under their padding *mask*."""
        raise NotImplementedError

    def cross_attention(
        self, hidden_size: int, num_heads: int, dropout: float
    ) -> ContextCrossAttention:
        """A fill-layer block that reads this tower's memory."""
        raise NotImplementedError

    def new_parameters(self) -> tuple[str, ...]:
        """Names of the tower's parameters no earlier checkpoint of its model holds."""
        return ()

    def enable_checkpointing(self) -> None:
        """Recompute the tower's layers in backward instead of storing them."""
        raise NotImplementedError


class GatedCrossAttention(ContextCrossAttention):
    """Multi-head attention from the fill tower into same-width context rows."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory: ContextMemory,
        rows: torch.Tensor,
        has_context: torch.Tensor,
    ) -> torch.Tensor:
        """Add the gated, per-example-switched context term to *hidden_states*."""
        context = memory.states[rows]
        attended, _ = self.attention(
            self.norm(hidden_states),
            context,
            context,
            key_padding_mask=~memory.mask[rows],
            need_weights=False,
        )
        gated: torch.Tensor = hidden_states + self.gate * has_context[:, None, None] * attended
        return gated


class BertContextTower(BertModel, ContextTower):  # type: ignore[misc]  # transformers is untyped
    """MacBERT's context tower: a BERT encoder over the context's base-vocabulary tokens.

    A subclass of :class:`BertModel` rather than a wrapper, so its parameters
    keep the ``context.*`` names every route A and e2e checkpoint stores them
    under. It trains with the fill tower, inside the same compiled pass.
    """

    separate: ClassVar[bool] = False

    # BertModel's attributes, which nn.Module's typed __getattr__ would
    # otherwise shadow with ``Tensor | Module``.
    config: Any
    embeddings: Any
    encoder: Any
    gradient_checkpointing_enable: Any

    def memory(self, ids: torch.Tensor, mask: torch.Tensor) -> ContextMemory:
        """The last hidden states, under the same host-check-free mask as the fill tower's."""
        encoded: torch.Tensor = self(
            input_ids=ids, attention_mask=mask[:, None, None, :].bool()
        ).last_hidden_state
        return ContextMemory(states=encoded, mask=mask.bool())

    def cross_attention(
        self, hidden_size: int, num_heads: int, dropout: float
    ) -> ContextCrossAttention:
        """MacBERT's own width is the fill tower's, so the block projects nothing across."""
        if self.config.hidden_size != hidden_size:
            raise ValueError(
                f"the towers disagree on hidden size: {hidden_size} and {self.config.hidden_size}"
            )
        return GatedCrossAttention(hidden_size, num_heads, dropout)

    def enable_checkpointing(self) -> None:
        """transformers' own gradient checkpointing, which the tower's forward reads."""
        self.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})


class RestrictedMlmHead(nn.Module):
    """The base MLM head with its output rows cut to the emittable characters.

    Slicing rather than masking after the fact is a compute decision: the full
    head scores 21,128 tokens of which 14,000 are characters nobody types, and
    the restricted head is a third of the matmul. The rows are copied out of the
    pretrained decoder, so the head starts as the base head's restriction and not
    as a fresh layer.
    """

    def __init__(self, transform: nn.Module, decoder: nn.Linear, token_ids: torch.Tensor):
        super().__init__()
        self.transform = transform
        self.decoder = nn.Linear(
            decoder.in_features, token_ids.numel(), bias=decoder.bias is not None
        )
        with torch.no_grad():
            self.decoder.weight.copy_(decoder.weight.index_select(0, token_ids))
            if decoder.bias is not None and self.decoder.bias is not None:
                self.decoder.bias.copy_(decoder.bias.index_select(0, token_ids))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Emission-space logits for every position."""
        logits: torch.Tensor = self.decoder(self.transform(hidden_states))
        return logits


@dataclass(frozen=True)
class RouteAOutput:
    """What a forward pass produced for the training loop.

    The score frame is deliberately not a field: the loop reads ``loss`` and
    ``extras`` only, and a logits field that meant nothing at training time
    would be a value read by mistake later. Evaluation and emission call
    ``scores()`` for the frame instead.
    """

    loss: torch.Tensor | None = None
    #: Extra scalars for the metrics record (the end-to-end model reports its
    #: decoder and auxiliary losses apart). Tensors, floated at write time.
    extras: Mapping[str, torch.Tensor] = field(default_factory=dict)


def restricted_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    candidates: torch.Tensor,
    label_smoothing: float,
    *,
    check: bool = True,
) -> torch.Tensor:
    """Cross-entropy over each position's candidate set alone.

    The softmax is taken after the non-candidates are pushed to the dtype's
    minimum, so they hold no probability and their logits have no gradient: the
    loss is provably independent of what the model thinks about a character the
    typed span rules out. Smoothing then spreads its mass over *the candidates*
    rather than over the vocabulary -- torch's own ``label_smoothing`` would put
    mass on the ruled-out characters, whose log-probability is minus infinity,
    and the loss would stop being a number.

    ``check`` verifies the mask admits a character at every position -- reading
    that verdict back synchronises the device, so a caller whose batch a
    validating collator built passes ``False`` and keeps the step asynchronous.
    """
    if targets.numel() == 0:
        raise ValueError("no positions to take a loss at")
    if check and bool((candidates.sum(dim=-1) == 0).any()):
        raise ValueError("a position admits no character at all; the mask and the target disagree")
    floor = torch.finfo(logits.dtype).min
    return smoothed_cross_entropy(
        logits.masked_fill(~candidates, floor), targets, candidates, label_smoothing
    )


def smoothed_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, candidates: torch.Tensor, label_smoothing: float
) -> torch.Tensor:
    """Cross-entropy normalised over every column of *logits*, smoothed over *candidates*.

    :func:`restricted_cross_entropy` is this over logits whose non-candidates
    were pushed to the floor. Called on whole-alphabet logits it is the loss
    of a model whose scores are normalised over the whole alphabet -- the
    ranking model's path score (#121) -- where every character the span rules
    out still holds probability, and still gets the gradient that takes it
    away. The smoothing mass stays on the candidates either way.
    """
    log_probabilities = logits.log_softmax(dim=-1)
    gold = log_probabilities.gather(1, targets[:, None]).squeeze(1)
    if label_smoothing == 0.0:
        return -gold.mean()
    zero = torch.zeros((), dtype=log_probabilities.dtype, device=log_probabilities.device)
    spread = torch.where(candidates, log_probabilities, zero).sum(dim=-1) / candidates.sum(dim=-1)
    return -((1.0 - label_smoothing) * gold + label_smoothing * spread).mean()


def mark_dynamic(batch: Batch) -> None:
    """Tell dynamo which dimensions of the batch vary between calls.

    ``torch.compile(dynamic=True)`` still compiles its first call static and
    only recompiles a dynamic graph on the second shape -- marking the batch
    and sequence dimensions at the call site makes the first compile the
    dynamic one. The marks live outside the compiled callables because dynamo
    refuses to trace the marker itself. ``maybe_`` rather than
    ``mark_dynamic``: a size-1 dimension is not markable and a tiny batch must
    still run. Unmarked fields -- ``targets``, ``scored`` -- are read only by
    the uncompiled loss path, so they need no mark.
    """
    for tensor in (
        batch.input_ids,
        batch.attention_mask,
        batch.span_ids,
        batch.span_positions,
        batch.context_ids,
        batch.context_mask,
    ):
        torch._dynamo.maybe_mark_dynamic(tensor, (0, 1))
    torch._dynamo.maybe_mark_dynamic(batch.has_context, 0)
    torch._dynamo.maybe_mark_dynamic(batch.context_rows, 0)
    torch._dynamo.maybe_mark_dynamic(batch.candidate_ids, (0, 1, 2))
    torch._dynamo.maybe_mark_dynamic(batch.candidate_counts, (0, 1))
    torch._dynamo.maybe_mark_dynamic(batch.span_letters, (0, 1, 2))


class RouteAModel(nn.Module):
    """The fill tower, the context tower, and the gates between them.

    The submodules are annotated at class level because ``nn.Module.__getattr__``
    is typed as returning a module *or* a tensor; without the annotations every
    call through one of them is a call on a possible tensor.
    """

    #: The checkpoint record a run of this model stores its config under, and
    #: refuses a resume against any other.
    config_key: ClassVar[str] = "route_a"

    fill: BertModel
    context: ContextTower
    head: RestrictedMlmHead
    span_embeddings: nn.Embedding
    cross_attention: nn.ModuleList
    candidate_mask: torch.Tensor
    emittable_token_ids: torch.Tensor
    #: When set, each fill-tower layer is recomputed in backward rather than
    #: stored -- the knob a memory-tight run turns, off by default.
    checkpoint_encoder: bool

    def __init__(
        self,
        fill: BertForMaskedLM,
        context: ContextTower,
        lexicon: Lexicon,
        config: RouteAConfig,
    ):
        super().__init__()
        bert_config = fill.config
        if config.cross_attention_layers > bert_config.num_hidden_layers:
            raise ValueError(
                f"cannot gate {config.cross_attention_layers} of "
                f"{bert_config.num_hidden_layers} layers"
            )
        if not hasattr(fill.bert, "_create_attention_masks"):
            raise RuntimeError(
                "this transformers build has no BertModel._create_attention_masks; the "
                "fill tower's layer-by-layer pass needs it to build the same masks the "
                "model's own forward would"
            )
        towers = (fill.bert, context) if isinstance(context, BertContextTower) else (fill.bert,)
        for tower in towers:
            implementation = getattr(tower.config, "_attn_implementation", None)
            if implementation != "sdpa":
                raise RuntimeError(
                    f"the towers are fed 4D boolean masks, which only sdpa reads as "
                    f"'True attends'; this tower's attention is {implementation!r}"
                )
        self.config = config
        self.fill = fill.bert
        self.context = context
        self.head = RestrictedMlmHead(
            fill.cls.predictions.transform, fill.cls.predictions.decoder, lexicon.token_ids
        )
        self.span_embeddings = nn.Embedding(lexicon.spans, bert_config.hidden_size)
        self.letter_encoder = LetterEncoder(bert_config.hidden_size)
        self.cross_attention = nn.ModuleList(
            context.cross_attention(
                bert_config.hidden_size,
                bert_config.num_attention_heads,
                config.cross_attention_dropout,
            )
            for _ in range(config.cross_attention_layers)
        )
        self.register_buffer("candidate_mask", lexicon.candidate_mask, persistent=True)
        self.register_buffer("emittable_token_ids", lexicon.token_ids, persistent=True)
        self.checkpoint_encoder = False

    @classmethod
    def from_pretrained(
        cls, config: RouteAConfig, lexicon: Lexicon, spans: SpanVocab, vocabulary: dict[str, int]
    ) -> RouteAModel:
        """Build route A on top of the base checkpoint, initialising the new tables."""
        fill = BertForMaskedLM.from_pretrained(config.base_model)
        context = BertContextTower.from_pretrained(config.base_model, add_pooling_layer=False)
        model = cls(fill, context, lexicon, config)
        model.initialise_span_embeddings(letter_token_ids(spans, vocabulary))
        log.info(
            "route A built",
            base=config.base_model,
            emittable=lexicon.size,
            spans=lexicon.spans,
            gated_layers=config.cross_attention_layers,
            parameters=sum(parameter.numel() for parameter in model.parameters()),
        )
        return model

    @classmethod
    def from_config(
        cls, bert_config: BertConfig, lexicon: Lexicon, config: RouteAConfig
    ) -> RouteAModel:
        """Build route A on randomly initialised towers, for tests."""
        return cls(
            BertForMaskedLM(bert_config),
            BertContextTower(bert_config, add_pooling_layer=False),
            lexicon,
            config,
        )

    def initialise_span_embeddings(self, letter_ids: Sequence[Sequence[int]]) -> None:
        """Seed each span's embedding with the mean of its letters' base embeddings.

        Order is lost at initialisation -- ``na`` and ``an`` start equal -- but
        the entries are separate parameters from the first step, so training
        separates them. The alternative, keeping the mean as the encoding, never
        could.
        """
        if len(letter_ids) != self.span_embeddings.num_embeddings:
            raise ValueError(
                f"got letters for {len(letter_ids)} spans, table holds "
                f"{self.span_embeddings.num_embeddings}"
            )
        words = self.fill.embeddings.word_embeddings.weight
        with torch.no_grad():
            for index, letters in enumerate(letter_ids):
                if not letters:
                    # The reserved ``<unk>`` tail: a typoed span's signal is its
                    # letters, which the letter encoder supplies -- the table
                    # row itself starts at zero rather than a fabricated mean.
                    self.span_embeddings.weight[index].zero_()
                    continue
                ids = torch.tensor(letters, dtype=torch.long, device=words.device)
                self.span_embeddings.weight[index] = words.index_select(0, ids).mean(dim=0)

    def parameter_groups(self, base_lr: float, new_lr: float) -> list[dict[str, object]]:
        """Split the parameters into the pretrained ones and the ones we added.

        The letter encoder trails the new group on purpose: a checkpoint that
        predates it holds the rest of this group in the same order, so the
        resumed optimiser's states land on the parameters they were trained
        beside and the letters simply arrive with none.
        """
        base: list[nn.Parameter] = []
        new: list[nn.Parameter] = []
        letters: list[nn.Parameter] = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("letter_encoder"):
                letters.append(parameter)
                continue
            target = new if name.startswith(NEW_PARAMETER_PREFIXES) else base
            target.append(parameter)
        new += letters
        if not new:
            raise ValueError("no new parameters found; the naming convention has drifted")
        return [{"params": base, "lr": base_lr}, {"params": new, "lr": new_lr}]

    def gated_layers(self) -> list[ContextCrossAttention]:
        """The cross-attention layers, narrowed out of the untyped module list."""
        layers = []
        for layer in self.cross_attention:
            if not isinstance(layer, ContextCrossAttention):
                raise TypeError(f"the cross-attention list holds a {type(layer).__name__}")
            layers.append(layer)
        return layers

    def gate_parameters(self) -> list[nn.Parameter]:
        """Each context gate the model learns, in the order :meth:`gates` reports them."""
        return [layer.gate for layer in self.gated_layers()]

    def gates(self) -> list[float]:
        """The current value of each gate, read back in one transfer."""
        gates = [gate.detach() for gate in self.gate_parameters()]
        values: list[float] = torch.stack(gates).tolist() if gates else []
        return values

    def enable_encoder_checkpointing(self) -> None:
        """Recompute each encoder layer in backward instead of storing it.

        The fill tower's layers run one at a time from :meth:`encode`, so the
        checkpoint wraps each call there; the context tower takes the same
        switch its own way.
        """
        self.checkpoint_encoder = True
        self.context.enable_checkpointing()

    def bert_context(self) -> BertContextTower:
        """The context tower as MacBERT's, for the callers that read its BERT layers."""
        if not isinstance(self.context, BertContextTower):
            raise TypeError(f"this model's context tower is a {type(self.context).__name__}")
        return self.context

    def new_context_parameters(self) -> frozenset[str]:
        """The full names of every parameter the context path adds over its parent model."""
        names = {f"context.{name}" for name in self.context.new_parameters()}
        for index, layer in enumerate(self.gated_layers()):
            names.update(f"cross_attention.{index}.{name}" for name in layer.new_parameters())
        return frozenset(names)

    def _fill_inputs(self, batch: Batch) -> torch.Tensor:
        """``[MASK]`` everywhere a span was typed, plus that span's embedding and letters.

        The letter term is gated by the same ``span_positions`` the table row
        is, and its projection is zero-initialised, so at init this is exactly
        the old sum.
        """
        words: torch.Tensor = self.fill.embeddings.word_embeddings(batch.input_ids)
        spans: torch.Tensor = self.span_embeddings(batch.span_ids) + self.letter_encoder(
            batch.span_letters
        )
        return words + spans * batch.span_positions[..., None]

    def encode(self, batch: Batch) -> torch.Tensor:
        """The fill tower's last hidden states, context-attended (``[B, W, H]``).

        Split out of :meth:`forward` because the end-to-end model's decoder
        attends to the per-span hidden states rather than to their logits.
        The context tower runs once per context row (:meth:`encode_context`),
        and each fill row then reads the one ``context_rows`` names
        (:meth:`encode_fill`).
        """
        memory = self.encode_context(batch)
        if memory is not None and self.context.separate:
            # A separate tower's memory enters the compiled fill pass as an
            # input, so its varying dims are marked as the batch's are.
            torch._dynamo.maybe_mark_dynamic(memory.states, (0, 1))
            torch._dynamo.maybe_mark_dynamic(memory.mask, (0, 1))
        return self.encode_fill(batch, memory)

    def encode_context(self, batch: Batch) -> ContextMemory | None:
        """The context tower's memory of the batch's contexts; None when no layer reads it."""
        if not self.gated_layers():
            return None
        return self.context.memory(batch.context_ids, batch.context_mask)

    def encode_fill(self, batch: Batch, memory: ContextMemory | None) -> torch.Tensor:
        """The fill tower over the typed spans, its top layers reading *memory*."""
        embeddings: torch.Tensor = self.fill.embeddings(inputs_embeds=self._fill_inputs(batch))
        # A 4D mask early-exits ``_create_attention_masks`` before its padding
        # check -- ``padding_mask.all()`` is a device-to-host sync the step
        # cannot afford. The ``[B, 1, 1, W]`` boolean is what the built mask
        # becomes anyway under sdpa (True attends), so the towers see the same
        # values; eager attention would add it to the scores instead, which is
        # why the constructor refuses anything but sdpa.
        attention_mask, _ = self.fill._create_attention_masks(
            attention_mask=batch.attention_mask[:, None, None, :].bool(),
            encoder_attention_mask=None,
            embedding_output=embeddings,
            encoder_hidden_states=None,
            past_key_values=None,
        )
        gated = self.gated_layers()
        has_context = batch.has_context[batch.context_rows]
        gated_from = len(self.fill.encoder.layer) - len(gated)
        hidden = embeddings
        for depth, layer in enumerate(self.fill.encoder.layer):
            if self.checkpoint_encoder:
                hidden = torch.as_tensor(
                    torch.utils.checkpoint.checkpoint(
                        layer, hidden, attention_mask=attention_mask, use_reentrant=False
                    )
                )
            else:
                hidden = torch.as_tensor(layer(hidden, attention_mask=attention_mask))
            if memory is not None and depth >= gated_from:
                hidden = gated[depth - gated_from](hidden, memory, batch.context_rows, has_context)
        return hidden

    def forward(self, batch: Batch) -> RouteAOutput:
        """The loss at the scored positions; the frame belongs to ``scores``."""
        mark_dynamic(batch)
        return RouteAOutput(loss=self.loss(self.head(self.encode(batch)), batch))

    def scores(self, batch: Batch) -> torch.Tensor:
        """The ``[B, W, E]`` emission frame, for the eval-side callers."""
        mark_dynamic(batch)
        logits: torch.Tensor = self.head(self.encode(batch))
        return logits

    def candidates_at(self, batch: Batch, positions: torch.Tensor) -> torch.Tensor:
        """The ``[P, E]`` mask the batch's own candidate lists describe.

        The batch carries each position's list because a typoed span admits the
        union of its corrections' homophones -- a per-position set no span-table
        row can name. Reassembling the mask here keeps the restricted loss's
        ``[P, E]`` contract while the candidate source stays per-example.
        """
        ids = batch.candidate_ids.reshape(-1, batch.candidate_ids.shape[-1]).index_select(
            0, positions
        )
        counts = batch.candidate_counts.reshape(-1).index_select(0, positions)
        size = self.candidate_mask.shape[1]
        within = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0) < counts.unsqueeze(1)
        mask = torch.zeros((ids.shape[0], size + 1), dtype=torch.bool, device=ids.device)
        mask.scatter_(1, ids.masked_fill(~within, size), within)
        return mask[:, :-1]

    def loss(self, logits: torch.Tensor, batch: Batch) -> torch.Tensor | None:
        """Restricted, smoothed cross-entropy at the positions a span was typed.

        The positions come from the batch's own ``scored`` index, found by the
        collator on the host -- mask-indexing ``logits`` here would synchronise
        the device for the same answer the caller already computed.
        """
        scored = batch.scored
        if not scored.numel():
            return None
        return restricted_cross_entropy(
            logits.reshape(-1, logits.shape[-1]).index_select(0, scored),
            batch.targets.reshape(-1).index_select(0, scored),
            self.candidates_at(batch, scored),
            self.config.label_smoothing,
            check=False,
        )

    def predictions(self, logits: torch.Tensor, batch: Batch) -> torch.Tensor:
        """The most likely admitted character at every position, in emission space.

        Positions with no target are left at :data:`IGNORE_INDEX` so a caller can
        compare against ``batch.targets`` directly.
        """
        flat = torch.arange(batch.span_ids.numel(), device=logits.device)
        candidates = self.candidates_at(batch, flat).reshape(*batch.span_ids.shape, -1)
        floor = torch.finfo(logits.dtype).min
        best = logits.masked_fill(~candidates, floor).argmax(dim=-1)
        return best.masked_fill(batch.targets == IGNORE_INDEX, IGNORE_INDEX)

    def load_resumed(self, weights: Mapping[str, torch.Tensor], path: Path | str) -> None:
        """Load a checkpoint that may predate the typo inputs, tolerantly.

        A run checkpointed before this stage lacks ``letter_encoder.*`` (the
        zero projection is its initialisation) and carries one row fewer of
        ``span_embeddings`` and ``candidate_mask`` -- the ``<unk>`` tail is
        appended, so the old rows prefix the new. The decoder's derived
        ``span_candidates`` buffers are gone now that the batch carries them.
        Anything else missing or unexpected is still a refusal, not a fold.
        """
        own = self.state_dict()
        grown = dict(weights)
        popped: set[str] = set()
        for name in ("decoder.span_candidates", "decoder.span_candidate_counts"):
            if name not in own and grown.pop(name, None) is not None:
                popped.add(name)
        for name in ("span_embeddings.weight", "candidate_mask"):
            stored = grown.get(name)
            if stored is None or name not in own or stored.shape == own[name].shape:
                continue
            if (
                stored.shape[0] + 1 == own[name].shape[0]
                and stored.shape[1:] == own[name].shape[1:]
            ):
                if name == "candidate_mask":
                    if not torch.equal(stored, own[name][: stored.shape[0]]):
                        raise ValueError(
                            f"{path}'s candidate mask disagrees with the lexicon's prefix rows"
                        )
                    grown.pop(name)  # the built table already carries the tail row
                    popped.add(name)
                else:
                    merged = own[name].clone()
                    merged[: stored.shape[0]] = stored
                    # The appended ``<unk>`` tail resumes at zero, matching what
                    # ``initialise_span_embeddings`` writes into a fresh table.
                    merged[stored.shape[0] :] = 0
                    grown[name] = merged
            else:
                raise ValueError(
                    f"{path}'s {name} is {tuple(stored.shape)} against this model's "
                    f"{tuple(own[name].shape)}; the span table drifted"
                )
        result = self.load_state_dict(grown, strict=False)
        missing = [
            name
            for name in result.missing_keys
            if name not in popped and not name.startswith(("decoder.", "letter_encoder."))
        ]
        if missing or result.unexpected_keys:
            raise ValueError(
                f"{path} does not fit this model: "
                f"missing {missing[:4]}, unexpected {result.unexpected_keys[:4]}"
            )


def letter_token_ids(spans: SpanVocab, vocabulary: dict[str, int]) -> list[list[int]]:
    """The base-vocabulary ids of each span's letters, in order.

    A pinyin span is ASCII, and the base vocabulary holds every ASCII letter as
    its own token, so a miss here means the tokenizer is not the one the table
    was built for. The reserved ``<unk>`` tail carries no letters at all -- the
    caller seeds its row to zeros.
    """
    ids: list[list[int]] = []
    for span in spans:
        if span == UNKNOWN_SPAN:
            ids.append([])
            continue
        letters = []
        for letter in span:
            if letter not in vocabulary:
                raise KeyError(f"the base vocabulary has no token for the letter {letter!r}")
            letters.append(vocabulary[letter])
        ids.append(letters)
    return ids


def count_correct(predictions: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
    """Characters predicted correctly, and characters scored at all."""
    scored = targets != IGNORE_INDEX
    return int((predictions.eq(targets) & scored).sum()), int(scored.sum())


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    """Every parameter an optimiser step would move."""
    return (parameter for parameter in model.parameters() if parameter.requires_grad)
