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

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import nn
from transformers import BertConfig, BertForMaskedLM, BertModel

from mlime.logging import log
from mlime.train.lexicon import Lexicon
from mlime.train.samples import IGNORE_INDEX, Batch
from mlime.train.spans import SpanVocab

#: Parameters under these prefixes did not come from the pretrained checkpoint and
#: train at the higher learning rate.
NEW_PARAMETER_PREFIXES = ("span_embeddings", "cross_attention")


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


class GatedCrossAttention(nn.Module):
    """Multi-head attention from the fill tower into the context, behind a zero gate.

    The gate is a single scalar rather than a vector so that "the context is off"
    is one number per layer, readable straight out of a checkpoint. At zero the
    module returns its input unchanged -- not approximately, exactly -- which is
    what makes "does context help?" answerable by training rather than by
    argument.
    """

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
        context: torch.Tensor,
        context_mask: torch.Tensor,
        has_context: torch.Tensor,
    ) -> torch.Tensor:
        """Add the gated, per-example-switched context term to *hidden_states*."""
        attended, _ = self.attention(
            self.norm(hidden_states),
            context,
            context,
            key_padding_mask=context_mask == 0,
            need_weights=False,
        )
        gated: torch.Tensor = hidden_states + self.gate * has_context[:, None, None] * attended
        return gated


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
    """What a forward pass produced."""

    logits: torch.Tensor
    loss: torch.Tensor | None = None


def restricted_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    candidates: torch.Tensor,
    label_smoothing: float,
) -> torch.Tensor:
    """Cross-entropy over each position's candidate set alone.

    The softmax is taken after the non-candidates are pushed to the dtype's
    minimum, so they hold no probability and their logits have no gradient: the
    loss is provably independent of what the model thinks about a character the
    typed span rules out. Smoothing then spreads its mass over *the candidates*
    rather than over the vocabulary -- torch's own ``label_smoothing`` would put
    mass on the ruled-out characters, whose log-probability is minus infinity,
    and the loss would stop being a number.
    """
    if targets.numel() == 0:
        raise ValueError("no positions to take a loss at")
    counts = candidates.sum(dim=-1)
    if bool((counts == 0).any()):
        raise ValueError("a position admits no character at all; the mask and the target disagree")
    floor = torch.finfo(logits.dtype).min
    log_probabilities = logits.masked_fill(~candidates, floor).log_softmax(dim=-1)
    gold = log_probabilities.gather(1, targets[:, None]).squeeze(1)
    if label_smoothing == 0.0:
        return -gold.mean()
    zero = torch.zeros((), dtype=log_probabilities.dtype, device=log_probabilities.device)
    spread = torch.where(candidates, log_probabilities, zero).sum(dim=-1) / counts
    return -((1.0 - label_smoothing) * gold + label_smoothing * spread).mean()


class RouteAModel(nn.Module):
    """The fill tower, the context tower, and the gates between them.

    The submodules are annotated at class level because ``nn.Module.__getattr__``
    is typed as returning a module *or* a tensor; without the annotations every
    call through one of them is a call on a possible tensor.
    """

    fill: BertModel
    context: BertModel
    head: RestrictedMlmHead
    span_embeddings: nn.Embedding
    cross_attention: nn.ModuleList
    candidate_mask: torch.Tensor
    emittable_token_ids: torch.Tensor

    def __init__(
        self,
        fill: BertForMaskedLM,
        context: BertModel,
        lexicon: Lexicon,
        config: RouteAConfig,
    ):
        super().__init__()
        bert_config = fill.config
        if context.config.hidden_size != bert_config.hidden_size:
            raise ValueError(
                f"the towers disagree on hidden size: {bert_config.hidden_size} and "
                f"{context.config.hidden_size}"
            )
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
        self.config = config
        self.fill = fill.bert
        self.context = context
        self.head = RestrictedMlmHead(
            fill.cls.predictions.transform, fill.cls.predictions.decoder, lexicon.token_ids
        )
        self.span_embeddings = nn.Embedding(lexicon.spans, bert_config.hidden_size)
        self.cross_attention = nn.ModuleList(
            GatedCrossAttention(
                bert_config.hidden_size,
                bert_config.num_attention_heads,
                config.cross_attention_dropout,
            )
            for _ in range(config.cross_attention_layers)
        )
        self.register_buffer("candidate_mask", lexicon.candidate_mask, persistent=True)
        self.register_buffer("emittable_token_ids", lexicon.token_ids, persistent=True)

    @classmethod
    def from_pretrained(
        cls, config: RouteAConfig, lexicon: Lexicon, spans: SpanVocab, vocabulary: dict[str, int]
    ) -> RouteAModel:
        """Build route A on top of the base checkpoint, initialising the new tables."""
        fill = BertForMaskedLM.from_pretrained(config.base_model)
        context = BertModel.from_pretrained(config.base_model, add_pooling_layer=False)
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
            BertModel(bert_config, add_pooling_layer=False),
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
                    raise ValueError(f"span {index} has no letters")
                ids = torch.tensor(letters, dtype=torch.long, device=words.device)
                self.span_embeddings.weight[index] = words.index_select(0, ids).mean(dim=0)

    def parameter_groups(self, base_lr: float, new_lr: float) -> list[dict[str, object]]:
        """Split the parameters into the pretrained ones and the ones we added."""
        base: list[nn.Parameter] = []
        new: list[nn.Parameter] = []
        for name, parameter in self.named_parameters():
            target = new if name.startswith(NEW_PARAMETER_PREFIXES) else base
            target.append(parameter)
        if not new:
            raise ValueError("no new parameters found; the naming convention has drifted")
        return [{"params": base, "lr": base_lr}, {"params": new, "lr": new_lr}]

    def gated_layers(self) -> list[GatedCrossAttention]:
        """The cross-attention layers, narrowed out of the untyped module list."""
        layers = []
        for layer in self.cross_attention:
            if not isinstance(layer, GatedCrossAttention):
                raise TypeError(f"the cross-attention list holds a {type(layer).__name__}")
            layers.append(layer)
        return layers

    def gates(self) -> list[float]:
        """The current gate value of each cross-attention layer."""
        return [float(layer.gate.detach()) for layer in self.gated_layers()]

    def _encode_context(self, batch: Batch) -> torch.Tensor:
        """Run the context tower."""
        encoded: torch.Tensor = self.context(
            input_ids=batch.context_ids, attention_mask=batch.context_mask
        ).last_hidden_state
        return encoded

    def _fill_inputs(self, batch: Batch) -> torch.Tensor:
        """``[MASK]`` everywhere a span was typed, plus that span's own embedding."""
        words: torch.Tensor = self.fill.embeddings.word_embeddings(batch.input_ids)
        spans: torch.Tensor = self.span_embeddings(batch.span_ids)
        return words + spans * batch.span_positions[..., None]

    def forward(self, batch: Batch) -> RouteAOutput:
        """Emission-space logits for every position, and the loss where targets are."""
        embeddings: torch.Tensor = self.fill.embeddings(inputs_embeds=self._fill_inputs(batch))
        attention_mask, _ = self.fill._create_attention_masks(
            attention_mask=batch.attention_mask,
            encoder_attention_mask=None,
            embedding_output=embeddings,
            encoder_hidden_states=None,
            past_key_values=None,
        )
        gated = self.gated_layers()
        context = self._encode_context(batch) if gated else None
        gated_from = len(self.fill.encoder.layer) - len(gated)
        hidden = embeddings
        for depth, layer in enumerate(self.fill.encoder.layer):
            hidden = torch.as_tensor(layer(hidden, attention_mask=attention_mask))
            if context is not None and depth >= gated_from:
                hidden = gated[depth - gated_from](
                    hidden, context, batch.context_mask, batch.has_context
                )
        logits = self.head(hidden)
        return RouteAOutput(logits=logits, loss=self.loss(logits, batch))

    def loss(self, logits: torch.Tensor, batch: Batch) -> torch.Tensor | None:
        """Restricted, smoothed cross-entropy at the positions a span was typed."""
        positions = batch.targets != IGNORE_INDEX
        if not bool(positions.any()):
            return None
        span_ids = batch.span_ids[positions]
        return restricted_cross_entropy(
            logits[positions],
            batch.targets[positions],
            self.candidate_mask.index_select(0, span_ids),
            self.config.label_smoothing,
        )

    def predictions(self, logits: torch.Tensor, batch: Batch) -> torch.Tensor:
        """The most likely admitted character at every position, in emission space.

        Positions with no target are left at :data:`IGNORE_INDEX` so a caller can
        compare against ``batch.targets`` directly.
        """
        candidates = self.candidate_mask.index_select(0, batch.span_ids.reshape(-1))
        candidates = candidates.reshape(*batch.span_ids.shape, -1)
        floor = torch.finfo(logits.dtype).min
        best = logits.masked_fill(~candidates, floor).argmax(dim=-1)
        return best.masked_fill(batch.targets == IGNORE_INDEX, IGNORE_INDEX)


def letter_token_ids(spans: SpanVocab, vocabulary: dict[str, int]) -> list[list[int]]:
    """The base-vocabulary ids of each span's letters, in order.

    A pinyin span is ASCII, and the base vocabulary holds every ASCII letter as
    its own token, so a miss here means the tokenizer is not the one the table
    was built for.
    """
    ids: list[list[int]] = []
    for span in spans:
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
