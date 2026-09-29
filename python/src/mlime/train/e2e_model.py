"""The end-to-end model of issue #88: route A's encoder, the reader's decoder.

Route A commits every position at once and lets the fused trigram sort out what
the characters mean to each other. This model moves that agreement *inside* the
network: an autoregressive decoder reads the encoder's per-span hidden states
and emits the sentence one character at a time, so each position's distribution
is conditioned on the characters already chosen, not only on what was typed.

Nothing here is reimplemented. The encoder *is* :class:`RouteAModel` -- both
towers, the span embeddings, the gated context, the restricted head, kept as the
auxiliary loss that anchors training -- and the decoder *is* the trained
:class:`TransformerCharLm` reader: its embedding, position table, blocks, norm
and tied output projection, iterated by hand so a gated cross-attention into the
encoder can sit after each block. Because every new path is behind a
zero-initialised scalar gate -- and the per-step span projection is
zero-initialised too -- the fresh model's step log-probabilities are exactly the
reader's renormalised over each position's candidates, and the encoder's head
is exactly route A's.

The decoder's step *i* input is ``input(embed(char_{i-1})) + span_input(h_i) +
positions(i)`` -- ``<bos>`` at step 0, ``h_i`` the encoder's hidden state at
span *i*, ``span_input`` a 768->512 projection that starts at zero. Its output
is the tied reader head restricted to the characters the lexicon admits
(``emit_ids`` maps emission index to reader id), renormalised over the position's
homophone candidates -- the same mask route A's loss uses.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch
from torch import nn
from transformers import BertConfig, BertForMaskedLM, BertModel

from mlime.train.charlm_model import Block, CharLmConfig, TransformerCharLm
from mlime.train.charlm_vocab import BOS, CharVocab
from mlime.train.lexicon import Lexicon
from mlime.train.model import (
    RouteAConfig,
    RouteAModel,
    RouteAOutput,
    mark_dynamic,
    restricted_cross_entropy,
)
from mlime.train.samples import IGNORE_INDEX, Batch

#: Decoder-side parameter name prefixes that train at the higher learning rate,
#: plus the encoder's letter encoder, which no earlier checkpoint carries.
#: The towers, the span table, the context gates and the reader all arrive
#: trained; only the span-input projection and the cross-attention are new.
NEW_PARAMETER_PREFIXES = ("decoder.span_input", "decoder.cross_attention", "letter_encoder")


def _default_reader() -> CharLmConfig:
    """The transformer reader's shape, as `char-lm` trained it on Kaggle."""
    return CharLmConfig(
        arch="transformer",
        embedding=512,
        hidden=512,
        layers=12,
        heads=8,
        feedforward=2048,
        max_positions=512,
    )


@dataclass(frozen=True)
class E2EConfig(RouteAConfig):
    """Route A's config, plus what the decoder adds.

    Subclassed rather than nested so the checkpoint record a fresh run writes
    still carries the encoder's fields verbatim; the decoder's shape and the
    auxiliary-loss weight are recorded the same way and a resume under a
    different one is refused.
    """

    #: The reader the decoder is built from. Always the transformer -- the LSTM
    #: has no blocks for the cross-attention to follow.
    decoder: CharLmConfig = field(default_factory=_default_reader)
    #: Weight of the encoder's own restricted loss inside the total.
    aux_weight: float = 0.3

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.decoder.arch != "transformer":
            raise ValueError(
                f"the e2e decoder must be the transformer reader, got {self.decoder!r}"
            )
        if self.aux_weight < 0.0:
            raise ValueError(f"aux_weight must not be negative, got {self.aux_weight}")

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> E2EConfig:
        """Rebuild a config from a checkpoint's ``e2e`` record."""
        fields = dict(record)
        fields["decoder"] = CharLmConfig(**fields["decoder"])
        return cls(**fields)


def emittable_ids(lexicon: Lexicon, vocab: CharVocab) -> torch.Tensor:
    """The reader-alphabet id of every emittable character, in emission order.

    Explicit rather than assumed because the two vocabularies are built
    differently: the lexicon's order is code points, the reader's is the
    character table's. A character the reader was never trained on has no
    embedding to inherit, so a miss is a hard error, not a silent UNK.
    """
    index = vocab.index
    missing = [character for character in lexicon.characters if character not in index]
    if missing:
        raise ValueError(
            f"the reader's alphabet has no ids for {len(missing)} emittable characters "
            f"({missing[:8]}...); it was trained on a different character table"
        )
    return torch.tensor([index[character] for character in lexicon.characters], dtype=torch.long)


class EncoderAttention(nn.Module):
    """One decoder block's gated cross-attention into the encoder's span vectors.

    The encoder's hidden width is not the decoder's (768 against 512), so the
    key and value projections carry the width change. The gate is a scalar
    starting at zero, exactly as route A's: at initialisation the module is the
    identity -- not approximately, exactly -- and how far training opened it is
    one number per layer in the checkpoint.
    """

    def __init__(self, config: CharLmConfig, encoder_width: int):
        super().__init__()
        if config.hidden % config.heads:
            raise ValueError(f"hidden {config.hidden} does not split over {config.heads} heads")
        self.heads = config.heads
        self.head_dim = config.hidden // config.heads
        self.norm = nn.LayerNorm(config.hidden)
        self.query = nn.Linear(config.hidden, config.hidden)
        self.key = nn.Linear(encoder_width, config.hidden)
        self.value = nn.Linear(encoder_width, config.hidden)
        self.out = nn.Linear(config.hidden, config.hidden)
        self.attention_dropout = config.dropout
        self.gate = nn.Parameter(torch.zeros(()))

    def kv(self, encoder: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The encoder's span outputs as keys and values, ``[B, heads, S, head_dim]``."""
        batch, length, _ = encoder.shape
        shape = (batch, length, self.heads, self.head_dim)
        return (
            self.key(encoder).view(shape).transpose(1, 2),
            self.value(encoder).view(shape).transpose(1, 2),
        )

    def attend(
        self,
        x: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """``x [B, T, hidden]`` attends to the encoder and comes back gated.

        *mask* ``[B, S]`` marks the span positions that are real; padded ones
        are weightless. ``None`` means all real, which is what a decoded path
        is. Keys and values broadcast over the batch, so a beam's rows share
        the one encoder pass.
        """
        batch, length, _ = x.shape
        q = self.query(self.norm(x)).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        attended = nn.functional.scaled_dot_product_attention(
            q,
            keys,
            values,
            attn_mask=None if mask is None else mask[:, None, None, :],
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        merged = attended.transpose(1, 2).reshape(batch, length, -1)
        result: torch.Tensor = x + self.gate * self.out(merged)
        return result


class Decoder(nn.Module):
    """The reader's transformer plus the wiring that lets it see the encoder.

    ``reader`` is the trained TransformerCharLm, held whole rather than copied:
    every distribution it learned over run3 is the e2e decoder's starting point.
    ``span_input`` is the zero-initialised projection that adds the current
    span's encoder vector to the step's input, and ``cross_attention`` holds one
    :class:`EncoderAttention` per block. ``emit_ids`` is the emission index of
    every character the model may produce mapped onto the reader's alphabet;
    persistent, so a checkpoint carries the mapping it was written with.
    """

    reader: TransformerCharLm
    emit_ids: torch.Tensor

    def __init__(
        self,
        reader: TransformerCharLm,
        encoder_width: int,
        emit_ids: torch.Tensor,
    ):
        super().__init__()
        if emit_ids.numel() == 0:
            raise ValueError("the decoder needs at least one emittable character")
        if int(emit_ids.max()) >= reader.embed.num_embeddings or int(emit_ids.min()) < 0:
            raise ValueError("emit_ids names ids outside the reader's alphabet")
        self.reader = reader
        self.span_input = nn.Linear(encoder_width, reader.config.hidden, bias=False)
        nn.init.zeros_(self.span_input.weight)
        self.cross_attention = nn.ModuleList(
            EncoderAttention(reader.config, encoder_width) for _ in reader.layers()
        )
        self.register_buffer("emit_ids", emit_ids, persistent=True)
        # The per-position candidate lists the decode and the score frame
        # restrict to travel with the batch itself: a typoed span admits the
        # union of its corrections' homophones, a set no span-table row can
        # name, so the collator writes the lists and the decoder reads them.

    def gated_layers(self) -> list[EncoderAttention]:
        """The cross-attention layers, narrowed out of the untyped module list."""
        layers = []
        for layer in self.cross_attention:
            if not isinstance(layer, EncoderAttention):
                raise TypeError(f"the cross-attention list holds a {type(layer).__name__}")
            layers.append(layer)
        return layers

    def embed(self, tokens: torch.Tensor, span_vectors: torch.Tensor, first: int) -> torch.Tensor:
        """``input(embed(token)) + span_input(span) + position``, ``[B, T] -> [B, T, H]``."""
        positions = torch.arange(tokens.shape[1], device=tokens.device) + first
        embedded = (
            self.reader.input(self.reader.embed(tokens))
            + self.span_input(span_vectors)
            + self.reader.positions(positions)
        )
        dropped: torch.Tensor = self.reader.dropout(embedded)
        return dropped

    def forward(
        self,
        tokens: torch.Tensor,
        span_vectors: torch.Tensor,
        span_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced features over a whole row of typed spans.

        ``tokens [B, T]`` are the reader ids of the previous characters (``<bos>``
        first), ``span_vectors [B, T, enc]`` the encoder's hidden state at each
        span position, and ``span_mask`` marks the real ones. Returns the normed
        features; :meth:`logits` turns them into emission-space scores.
        """
        x = self.embed(tokens, span_vectors, 0)
        for block, cross in zip(self.reader.layers(), self.gated_layers(), strict=True):
            keys, values = cross.kv(span_vectors)
            x = cross.attend(block(x), keys, values, span_mask)
        normed: torch.Tensor = self.reader.norm(x)
        return normed

    def logits_over(self, features: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """Logits over *candidates* alone: ``[B, T, H] -> [B, T, K]`` (or ``[B, H] -> [B, K]``).

        ``candidates`` are emission indices. The weight rows are the reader
        embeddings of the characters they name, so each value is the same score
        a full emittable-wide row would carry -- gathered without materialising
        it.
        """
        emit = self.emit_ids.index_select(0, candidates)
        weight = self.reader.embed.weight.index_select(0, emit)
        scores: torch.Tensor = self.reader.project(self.reader.dropout(features)) @ weight.T
        return scores

    def step(
        self,
        tokens: torch.Tensor,
        span_vector: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
        encoder_kv: Sequence[tuple[torch.Tensor, torch.Tensor]],
        encoder_mask: torch.Tensor | None = None,
        *,
        candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Advance one position: logits over the position's candidates, and the cache.

        ``tokens [B]`` are the reader ids of each row's previous character
        (``<bos>`` to open a path), ``span_vector [B, enc]`` the encoder vector
        of the position being predicted, and ``candidates`` the emission
        indices the position's span admits -- the only logits the search needs
        and the only ones computed. ``state`` is the key/value cache,
        ``[B, layers, heads, T, head_dim]`` each, and ``encoder_kv`` the
        per-block encoder keys and values computed once for the path.
        """
        keys, values = state
        x = self.embed(tokens.unsqueeze(1), span_vector.unsqueeze(1), keys.shape[3])
        new_keys, new_values = [], []
        for layer, (block, cross) in enumerate(
            zip(self.reader.layers(), self.gated_layers(), strict=True)
        ):
            x, k, v = _attend_step(block, x, keys[:, layer], values[:, layer])
            enc_keys, enc_values = encoder_kv[layer]
            x = cross.attend(x, enc_keys, enc_values, encoder_mask)
            new_keys.append(k)
            new_values.append(v)
        keys = torch.cat([keys, torch.stack(new_keys, dim=1)], dim=3)
        values = torch.cat([values, torch.stack(new_values, dim=1)], dim=3)
        features: torch.Tensor = self.reader.norm(x).squeeze(1)
        return self.logits_over(features, candidates), (keys, values)


def _attend_step(
    block: Block, x: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One position of *x* against the block's cache and itself.

    The reader's own :meth:`Block.attend` folds the prefix machinery in for the
    decoder's context prelude; this model has no prelude, so the step is the
    same split/merge arithmetic without it -- the block itself is reused, not
    copied. Returns the merged output and the new key/value.
    """
    q, k, v = block.split(x)
    keys = torch.cat([keys, k], dim=2)
    values = torch.cat([values, v], dim=2)
    scores = (q @ keys.transpose(-1, -2)) / math.sqrt(block.head_dim)
    attended = torch.softmax(scores, dim=-1) @ values
    return block.merge(x, attended), k, v


@dataclass(frozen=True)
class _Scored:
    """The scored positions' two heads of logits, and what they are scored against.

    ``decoder`` and ``encoder`` are ``[P, E]`` rows at the scored positions;
    ``candidates`` is the ``[P, E]`` mask the restriction applies on both, and
    ``targets`` the ``[P]`` emission ids. ``region`` keeps the scored
    positions' coordinates in span-column space for the ``scores`` frame.
    """

    decoder: torch.Tensor
    encoder: torch.Tensor
    candidates: torch.Tensor
    targets: torch.Tensor
    region: torch.Tensor


class E2EModel(RouteAModel):
    """Route A's towers as the encoder, the reader's transformer as the decoder.

    The class subclasses :class:`RouteAModel` rather than wrapping one because
    the encoder side is literally the same module tree -- the fill tower, the
    context tower, the span embeddings, the cross-attention gates and the
    restricted head -- and the training loop, the checkpoint format and the
    held-out evaluation all speak to that tree unchanged. :meth:`forward`
    trains the decoder against the positions the spans mark and keeps the head
    loss as the auxiliary; :meth:`encode` is the encoder pass the decoder's
    cross-attention reads.
    """

    config_key: ClassVar[str] = "e2e"

    decoder: Decoder

    def __init__(
        self,
        fill: BertForMaskedLM,
        context: BertModel,
        lexicon: Lexicon,
        config: E2EConfig,
        reader: TransformerCharLm,
        emit_ids: torch.Tensor,
    ):
        super().__init__(fill, context, lexicon, config)
        self.e2e_config = config
        self.decoder = Decoder(reader, fill.config.hidden_size, emit_ids)

    @classmethod
    def compose(
        cls,
        bert_config: BertConfig,
        lexicon: Lexicon,
        config: E2EConfig,
        reader_vocab: int,
        emit_ids: torch.Tensor,
    ) -> E2EModel:
        """Build the model on randomly initialised towers, for tests and loading.

        Not named ``from_config`` on purpose: that method's signature on
        :class:`RouteAModel` takes only the three arguments a route A model
        needs, and this one needs the reader's alphabet size and the emission
        mapping as well.
        """
        return cls(
            BertForMaskedLM(bert_config),
            BertModel(bert_config, add_pooling_layer=False),
            lexicon,
            config,
            TransformerCharLm(reader_vocab, config.decoder),
            emit_ids,
        )

    def parameter_groups(self, base_lr: float, new_lr: float) -> list[dict[str, object]]:
        """Split the parameters into the pretrained ones and the ones e2e adds.

        The towers, the span table, the context gates and the reader all arrive
        trained; only the span-input projection and the decoder's
        cross-attention are new. The letter encoder trails the group so a
        checkpoint that predates it resumes with its saved states landing on
        the parameters they were trained beside.
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

    def gates(self) -> list[float]:
        """The encoder's context gates, then each decoder block's cross gate."""
        return super().gates() + [
            float(layer.gate.detach()) for layer in self.decoder.gated_layers()
        ]

    def _scored(self, batch: Batch) -> _Scored | None:
        """The scored positions' logits for both heads, ``[P, E]`` apiece.

        One matmul per head against the whole emittable alphabet at the scored
        positions only -- the same tensors route A's loss is computed on, so a
        step carries no per-span loop. The positions themselves arrive in
        ``batch.scored`` from the collator, so nothing here waits on the device:
        no ``nonzero``, no mask-indexing, no boolean read back to the host.
        ``targets`` and ``candidates`` are shared by the two heads; ``region``
        indexes the span-column space for ``scores``.
        ``None`` when the batch holds no scored position.
        """
        decoder = self.decoder
        mark_dynamic(batch)
        hidden = self.encode(batch)
        width = batch.targets.shape[1]
        # The spans sit at fill positions 1..n and the sentinel pair frames
        # them, so the decoder's sequence is the batch's width less the two.
        spans = hidden[:, 1 : width - 1]
        mask = batch.span_positions[:, 1 : width - 1]
        targets = batch.targets[:, 1 : width - 1]
        target_ids = decoder.emit_ids[targets.clamp(min=0)]
        previous = torch.cat(
            [
                torch.full((batch.size, 1), BOS, dtype=torch.long, device=target_ids.device),
                target_ids[:, :-1],
            ],
            dim=1,
        )
        # The decoder's row count and length vary with the batch, so its
        # compiled forward wants the same first-call-dynamic marks as the
        # batch. The column dimension of ``spans`` and ``mask`` is the batch's
        # width less the sentinels -- a different dynamic dim than ``width``'s.
        for tensor in (previous, spans, mask):
            torch._dynamo.maybe_mark_dynamic(tensor, (0, 1))
        features: torch.Tensor = decoder(previous, spans, mask)

        positions = batch.scored
        if not positions.numel():
            return None
        columns = positions % width
        # The flat index of a scored position inside the span-column slice is
        # its row times the slice's width plus its column offset into it. That
        # positions sit inside those columns is the collator's checked
        # invariant, not something a step may afford to verify.
        region = (positions // width) * (width - 2) + columns - 1
        candidates = self.candidates_at(batch, positions)
        scored_targets = batch.targets.reshape(-1).index_select(0, positions)

        reader = decoder.reader
        projected = reader.project(reader.dropout(features))
        projected = projected.reshape(-1, projected.shape[-1]).index_select(0, region)
        emitted = reader.embed.weight.index_select(0, decoder.emit_ids)
        decoder_logits = projected @ emitted.T
        transformed = self.head.transform(
            hidden.reshape(-1, hidden.shape[-1]).index_select(0, positions)
        )
        encoder_logits = self.head.decoder(transformed)
        return _Scored(
            decoder=decoder_logits,
            encoder=encoder_logits,
            candidates=candidates,
            targets=scored_targets,
            region=region,
        )

    def forward(self, batch: Batch) -> RouteAOutput:
        """The training halves: the decoder's loss plus the encoder head's own.

        Decoder position *i* predicts the target of fill position ``i + 1``,
        conditioned on the ids of the characters before it and on the encoder's
        hidden states -- per span in the input projection, and over all of them
        in the cross-attention. The loss is the decoder's restricted,
        label-smoothed cross-entropy plus ``aux_weight`` times the encoder
        head's own, so the fill tower's anchor does not drift while the decoder
        learns. The score frame is ``scores``' job -- forward never builds it.
        """
        scored = self._scored(batch)
        if scored is None:
            return RouteAOutput()
        smoothing = self.config.label_smoothing
        decoder_loss = restricted_cross_entropy(
            scored.decoder, scored.targets, scored.candidates, smoothing, check=False
        )
        encoder_loss = restricted_cross_entropy(
            scored.encoder, scored.targets, scored.candidates, smoothing, check=False
        )
        return RouteAOutput(
            loss=decoder_loss + self.e2e_config.aux_weight * encoder_loss,
            extras={
                "decoder_loss": decoder_loss.detach(),
                "encoder_loss": encoder_loss.detach(),
            },
        )

    def scores(self, batch: Batch) -> torch.Tensor:
        """The ``[B, W, K]`` candidate frame, for the eval-side callers.

        A row's columns are that position's candidate list from the batch
        itself (``candidate_ids``), padded at the dtype's minimum;
        ``predictions`` maps an argmax back to its emission index. A position
        no span was typed at keeps the floor.
        """
        scored = self._scored(batch)
        width = batch.targets.shape[1]
        ids = batch.candidate_ids[:, 1 : width - 1]
        counts = batch.candidate_counts[:, 1 : width - 1]
        k = ids.shape[-1]
        weight = self.head.decoder.weight
        floor = torch.finfo(weight.dtype).min
        frame = weight.new_full((batch.size, width, k), floor)
        if scored is not None:
            gathered = scored.decoder.gather(1, ids.reshape(-1, k).index_select(0, scored.region))
            counts_here = counts.reshape(-1).index_select(0, scored.region)
            within = torch.arange(k, device=ids.device).unsqueeze(0) < counts_here.unsqueeze(1)
            flat = frame.new_full((batch.size * (width - 2), k), floor)
            flat[scored.region] = gathered.masked_fill(~within, floor)
            frame[:, 1 : width - 1] = flat.view(batch.size, width - 2, -1)
        return frame

    def predictions(self, logits: torch.Tensor, batch: Batch) -> torch.Tensor:
        """The most likely candidate at every position, as an emission index.

        ``logits`` is the ``[B, W, K]`` frame :meth:`scores` fills, each row
        scored over that position's own candidate list; the argmax's index
        selects the emission id out of the batch's ``candidate_ids``.
        """
        candidates = batch.candidate_ids
        best = candidates.gather(-1, logits.argmax(dim=-1, keepdim=True)).squeeze(-1)
        return best.masked_fill(batch.targets == IGNORE_INDEX, IGNORE_INDEX)
