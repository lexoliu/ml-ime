"""The key-level model of #216: a thin key front, the CTC head, and the S1 warm-up.

The span model's fill tower reads ``[CLS]``, one ``[MASK]`` plus a span-table
row per typed span, and ``[SEP]``. The key model's reads ``[CLS]``, one
``[MASK]`` plus a key-front vector per keystroke, and ``[SEP]``: the towers,
their gated context and the decoder are the parent's, and only the front and
the head's blank row are new (#216 §6).

**The key front** embeds each key as its letter -- initialised from the base
model's own letter tokens, the rows the span table was seeded from -- plus a
projection of its QWERTY coordinates and its position, then runs a few causal
pre-norm transformer blocks at the tower's width. Key ``t`` reads keys up to
``t`` only, so a live session's front caches exactly and a keystroke adds one
position. The coordinate projection, the position table and every block's
output projections start at zero: a fresh front hands the tower each key's
letter embedding, as a one-letter span row would be, and every new path is
there for training to open.

**The CTC head** is the parent's restricted MLM head -- its emittable rows are
the characters a key may emit -- plus a blank row, new and starting at zero.
Each key's log-softmax runs over the whole emittable set and the blank, and
the constrained CTC (:mod:`mlime.train.ctc`) sums over the gold's legal
alignments.

**S1** trains the front and the head with everything else frozen. The parent
-- the same towers fed the spans the typist meant -- is the teacher: at each
typed span's last key, the frozen tower's state over the keys is pulled
towards its state at that span over the spans. The loss is the constrained CTC
plus the distillation's mean squared error, weighted. The frozen parts run in
evaluation mode, so the teacher's states are the parent's own and the tower
the front feeds adds no dropout the front cannot see. The gate feeds the
parent's frozen decoder the front's states at the span ends, in place of its
encoder's (:meth:`KeyE2EModel.student_frame`).

**The frame term** (``frame_weight``, #255) trains on what that gate reads.
The squared error weighs every direction of a state alike and the decoder
does not, so a run can fit the parent's states more closely and still decode
worse. The term is the parent's own per-position loss, computed by its frozen
decoder over the student's states at the span ends: the decoder's weights
take no step, and its gradient reaches the front and the tower through them.

**S1 with the tower** (``train_fill_tower``, #241) trains the fill tower too,
beside the front and the head and at the head's rate, in training mode, and
with ``cross_attention`` the tower's gated context blocks as well. The teacher
must stay the parent, so the model keeps a frozen copy of each parent part
that trains -- ``teacher_fill`` of the tower, ``teacher_cross_attention`` of
the blocks, always in evaluation mode -- and the teacher's pass and the gate's
parent frame run through the copies; the span tables and the context tower
they share with the student stay frozen, so the copies' states are the
parent's exactly. Each copy is a parameter set the size of its part and is
saved with the run, so a resume distils towards the same targets.

**S2, the joint stage** (``key_warm_up`` off, #257) starts from a finished
S1 and trains everything the key path reads: the front, the towers, their
gated context, the CTC head and the decoder. There is no teacher, so no
copies, no distillation and no frame term. The loss is the constrained CTC,
exactly as S1 computes it, plus ``transition_weight`` times the transition's
term (#216 §5a): the decoder runs teacher-forced over each row's gold
characters, ``<bos>`` then the gold prefix, and every gated block
cross-attends to all of the row's key states -- the fill tower's output at
each real key, padding masked -- rather than to span-end states. No span is
aligned to a step, so the decoder has no span input (#216 §6): an S2 model
builds none, and starting from S1 drops the S1 decoder's
(:meth:`KeyE2EModel.load_warmed_up`). The term is the cross-entropy of the
gold character under a log-softmax over the whole emittable set. The search
masks the characters a key cannot emit without renormalising, the rule #216
§3 sets for the CTC head, so training and search read the same
log-probabilities and no per-step legal set is built on the host.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch
from torch import nn
from transformers import BertConfig, BertForMaskedLM, BertModel

from mlime.logging import log
from mlime.train.charlm_model import Block, TransformerCharLm
from mlime.train.charlm_vocab import BOS
from mlime.train.ctc import alignment_components, alignment_scores, constrained_ctc
from mlime.train.e2e_model import E2EConfig, E2EModel, EncoderAttention, context_tower
from mlime.train.key_manifest import N_MAX, NEG
from mlime.train.key_tree import TreeCall
from mlime.train.keys import (
    KeyBatch,
    KeyCollator,
    KeyExample,
    KeyFrontConfig,
    KeyInputs,
    key_coordinates,
)
from mlime.train.lexicon import Lexicon
from mlime.train.loop import Accuracy
from mlime.train.model import ContextTower, RouteAOutput, count_correct, mark_dynamic
from mlime.train.samples import LETTER_PAD, token_budget_batches

#: Parameter prefixes of the parts the key model adds: they train at the new rate.
KEY_PARAMETER_PREFIXES = ("key_front.", "blank.")

#: The decoder's gated cross-attention: it trains at the new rate too, since in
#: the joint stage it reads every key's state where the parent's read span
#: vectors, and has to relearn what it attends to (#216 §6). S1 freezes it.
RELEARNED_PREFIXES = ("decoder.cross_attention.",)

#: The submodules S1 leaves frozen and runs in evaluation mode.
FROZEN_IN_WARM_UP = ("fill", "context", "cross_attention", "decoder", "letter_encoder")

#: Parameter prefixes of the frozen copies of the parent's fill tower and of its
#: gated context blocks: the teacher of an S1 that trains them.
TEACHER_FILL_PREFIX = "teacher_fill."
TEACHER_CROSS_ATTENTION_PREFIX = "teacher_cross_attention."

#: What only S1 holds: the teacher's copies and the parent decoder's span
#: input, which the joint stage has no use for (#216 §6). Starting S2 from a
#: finished S1 drops exactly these.
WARM_UP_ONLY_PREFIXES = (
    TEACHER_FILL_PREFIX,
    TEACHER_CROSS_ATTENTION_PREFIX,
    "decoder.span_input.",
)

#: The parent's parts the key path never reads -- the span table and the
#: letters behind it, which only S1's teacher fed -- frozen in the joint stage.
UNREAD_BY_KEYS = ("span_embeddings.", "letter_encoder.")


@dataclass(frozen=True)
class KeyStates:
    """The fill tower's state at every key of a batch, the search's encoder output.

    ``states [B, Wk, H]`` holds row ``b``'s key ``t`` at ``[b, t]`` -- the
    sentinels' columns are cut away -- and ``mask [B, Wk]`` (bool) marks the
    real keys; a padded key's state is whatever the tower computed there and
    is read by nothing the mask excludes.
    """

    states: torch.Tensor
    mask: torch.Tensor


@dataclass(frozen=True)
class CrossMemory:
    """One record's key states as the decoder's cross-attention reads them, at a fixed length.

    ``keys`` and ``values`` are ``[layers, heads, N, head_dim]``: key ``j``'s
    projection at slot ``j``, zeros after the real keys. ``bias``
    ``[1, 1, 1, N]`` is additive: 0 at a real key, :data:`~mlime.train.key_manifest.NEG`
    at a padded slot.
    """

    keys: torch.Tensor
    values: torch.Tensor
    bias: torch.Tensor


@dataclass(frozen=True)
class TreeFeeds:
    """A tree call's node inputs: ``[1, S]`` reader ids and depths, and ``[1, 1, S, S]`` bias.

    ``tree_bias`` is additive: 0 where the column is an ancestor of the row or
    the row itself, :data:`~mlime.train.key_manifest.NEG` elsewhere.
    """

    reader_ids: torch.Tensor
    positions: torch.Tensor
    tree_bias: torch.Tensor


def _tree_attention(block: Block, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """One decoder block's self-attention under an additive *bias* ``[1, 1, S, S]``.

    The arithmetic of :meth:`~mlime.train.charlm_model.Block.forward` with
    the mask a tree needs, attention written out as a matrix product, an
    additive mask, a softmax and a matrix product so the same module converts
    for the Neural Engine, which has no fused attention.
    """
    q, k, v = block.split(x)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(block.head_dim) + bias
    return block.merge(x, scores.softmax(dim=-1) @ v)


def _cross_attention(
    cross: EncoderAttention,
    x: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """One gated cross-attention of nodes *x* ``[1, S, hidden]`` into a layer's key states.

    *keys* and *values* ``[heads, N, head_dim]`` are the layer's slice of a
    :class:`CrossMemory`, *bias* its additive ``[1, 1, 1, N]``. Spelled out as
    :func:`_tree_attention` is.
    """
    length = x.shape[1]
    q = cross.query(cross.norm(x)).view(1, length, cross.heads, cross.head_dim).transpose(1, 2)
    scores = (q @ keys.transpose(-1, -2)) / math.sqrt(cross.head_dim) + bias
    attended = (scores.softmax(dim=-1) @ values).transpose(1, 2).reshape(x.shape)
    result: torch.Tensor = x + cross.gate * cross.out(attended)
    return result


@dataclass(frozen=True)
class _JointTerms:
    """One joint pass: the constrained CTC's per-row terms and the transition's logits.

    ``nll [B]`` is each row's constrained-CTC loss, ``blank`` and ``emit``
    the alignment scores it was summed from, ``transition [P, E]`` the
    decoder's teacher-forced logits at each gold character over the whole
    emittable set, and ``targets [P]`` those characters, in the span batch's
    ``scored`` order.
    """

    nll: torch.Tensor
    blank: torch.Tensor
    emit: torch.Tensor
    transition: torch.Tensor
    targets: torch.Tensor


class KeyBlock(nn.Module):
    """One causal pre-norm transformer block, its output projections starting at zero."""

    def __init__(self, width: int, heads: int, feedforward: int, dropout: float):
        super().__init__()
        if width % heads:
            raise ValueError(f"width {width} does not split over {heads} heads")
        self.heads = heads
        self.attention_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.feedforward_norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, feedforward)
        self.down = nn.Linear(feedforward, width)
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = dropout
        for projection in (self.out, self.down):
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, T, width]``, each position reading itself and the positions before it."""
        batch, length, width = x.shape
        q, k, v = (
            self.qkv(self.attention_norm(x))
            .view(batch, length, 3, self.heads, width // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        attended = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        x = x + self.dropout(self.out(attended.transpose(1, 2).reshape(batch, length, width)))
        lifted = nn.functional.gelu(self.up(self.feedforward_norm(x)))
        result: torch.Tensor = x + self.dropout(self.down(lifted))
        return result


class KeyFront(nn.Module):
    """Keys to tower inputs: letter, QWERTY coordinates and position, then causal blocks.

    ``letters`` holds the 26 keys and the padding id, whose row is zero.
    ``coordinates`` is each key's (row, staggered column), a fact of the layout
    rebuilt at construction rather than stored.
    """

    coordinates: torch.Tensor

    def __init__(self, width: int, positions: int, config: KeyFrontConfig, tower: BertConfig):
        super().__init__()
        self.letters = nn.Embedding(LETTER_PAD + 1, width, padding_idx=LETTER_PAD)
        self.place = nn.Linear(2, width, bias=False)
        self.positions = nn.Embedding(positions, width)
        for zeroed in (self.place.weight, self.positions.weight):
            nn.init.zeros_(zeroed)
        self.blocks = nn.ModuleList(
            KeyBlock(
                width,
                int(tower.num_attention_heads),
                int(tower.intermediate_size),
                float(tower.hidden_dropout_prob),
            )
            for _ in range(config.layers)
        )
        padded = torch.cat([key_coordinates(), torch.zeros((1, 2))])
        self.register_buffer("coordinates", padded, persistent=False)

    def initialise_letters(self, words: torch.Tensor, letter_ids: Sequence[int]) -> None:
        """Seed each key's row with its letter's base-vocabulary embedding, ``a`` first."""
        if len(letter_ids) != LETTER_PAD:
            raise ValueError(f"expected the 26 letters' token ids, got {len(letter_ids)}")
        with torch.no_grad():
            ids = torch.tensor(list(letter_ids), dtype=torch.long, device=words.device)
            self.letters.weight[:LETTER_PAD] = words.index_select(0, ids)

    def forward(self, letters: torch.Tensor) -> torch.Tensor:
        """``[B, W]`` letter ids, laid out at the tower's columns, to ``[B, W, width]``."""
        columns = torch.arange(letters.shape[1], device=letters.device)
        x: torch.Tensor = (
            self.letters(letters)
            + self.place(self.coordinates.index_select(0, letters.reshape(-1))).view(
                *letters.shape, -1
            )
            + self.positions(columns)
        )
        for block in self.blocks:
            x = block(x)
        return x


class KeyE2EModel(E2EModel):
    """The e2e model reading keystrokes through a key front, with a per-key CTC head."""

    config_key: ClassVar[str] = "e2e"

    key_front: KeyFront
    blank: nn.Linear
    teacher_fill: BertModel | None
    teacher_cross_attention: nn.ModuleList | None

    def __init__(
        self,
        fill: BertForMaskedLM,
        context: ContextTower,
        lexicon: Lexicon,
        config: E2EConfig,
        reader: TransformerCharLm,
        emit_ids: torch.Tensor,
    ):
        super().__init__(fill, context, lexicon, config, reader, emit_ids)
        keys = config.keys
        if keys is None:
            raise ValueError("a key-level model needs a key front config")
        positions = int(fill.config.max_position_embeddings)
        if keys.max_keys + 2 > positions:
            raise ValueError(
                f"{keys.max_keys} keys and two sentinels do not fit the tower's {positions} "
                "positions"
            )
        self.keys_config = keys
        width = int(fill.config.hidden_size)
        self.key_front = KeyFront(width, keys.max_keys + 2, keys, fill.config)
        self.blank = nn.Linear(width, 1)
        nn.init.zeros_(self.blank.weight)
        nn.init.zeros_(self.blank.bias)
        # Seeded with the parent's parts when the weights load (load_resumed).
        trains = config.train_fill_tower
        self.teacher_fill = (
            None if trains is None else copy.deepcopy(self.fill).requires_grad_(False)
        )
        self.teacher_cross_attention = (
            copy.deepcopy(self.cross_attention).requires_grad_(False)
            if trains is not None and trains.cross_attention
            else None
        )

    @classmethod
    def compose(
        cls,
        bert_config: BertConfig,
        lexicon: Lexicon,
        config: E2EConfig,
        reader_vocab: int,
        emit_ids: torch.Tensor,
    ) -> KeyE2EModel:
        """Build the model on randomly initialised towers, for tests and loading."""
        return cls(
            BertForMaskedLM(bert_config),
            context_tower(config, bert_config),
            lexicon,
            config,
            TransformerCharLm(reader_vocab, config.decoder),
            emit_ids,
        )

    def load_parent(
        self,
        weights: Mapping[str, torch.Tensor],
        path: Path | str,
        letter_ids: Sequence[int],
    ) -> None:
        """Load a span model's weights, then seed the front's letters from its word embeddings.

        Everything the parent holds loads as it was trained; what is new --
        the key front and the blank row -- keeps its initialisation, and the
        front's letter rows take the base model's letter-token embeddings. A
        checkpoint that already holds a key front is a key run, which resumes
        rather than starts.
        """
        held = sorted(name for name in weights if name.startswith(KEY_PARAMETER_PREFIXES))
        if held:
            raise ValueError(f"{path} already holds key-level weights ({held[:3]}); resume it")
        self.load_resumed(weights, path, fresh=KEY_PARAMETER_PREFIXES)
        self.key_front.initialise_letters(self.fill.embeddings.word_embeddings.weight, letter_ids)

    def load_resumed(
        self,
        weights: Mapping[str, torch.Tensor],
        path: Path | str,
        fresh: tuple[str, ...] = (),
    ) -> None:
        """Load *weights*, seeding each teacher copy from its part when they hold none of it.

        A checkpoint without a copy of a part never trained that part -- a run
        that trains it keeps and saves the frozen copy -- so the part is its
        parent's, and the copy starts as that. A copy the checkpoint holds
        loads as saved; one loaded by a model that keeps none is refused as
        unexpected weights.

        The decoder must load whole. A key model is always built from an e2e
        checkpoint, which holds its decoder, so a decoder weight the
        checkpoint lacks is a different decoder -- an S1 model over an S2
        run, whose decoder has no span input -- and is refused rather than
        left at its initialisation.
        """
        absent = sorted(
            name
            for name in self.state_dict()
            if name.startswith("decoder.") and name not in weights
        )
        if absent:
            raise ValueError(f"{path} does not hold this model's decoder: missing {absent[:4]}")
        seeded = [
            (prefix, teacher, part)
            for prefix, teacher, part in (
                (TEACHER_FILL_PREFIX, self.teacher_fill, self.fill),
                (
                    TEACHER_CROSS_ATTENTION_PREFIX,
                    self.teacher_cross_attention,
                    self.cross_attention,
                ),
            )
            if teacher is not None and not any(name.startswith(prefix) for name in weights)
        ]
        super().load_resumed(weights, path, (*fresh, *(prefix for prefix, _, _ in seeded)))
        for _, teacher, part in seeded:
            teacher.load_state_dict(part.state_dict())

    def load_warmed_up(self, weights: Mapping[str, torch.Tensor], path: Path | str) -> None:
        """Start the joint stage from a finished S1 run's weights.

        What only S1 holds (:data:`WARM_UP_ONLY_PREFIXES`) -- the teacher's
        copies, when that run trained its tower, and the parent decoder's
        span input -- is dropped by name; everything else loads as it was
        trained, and a weight this model does not hold, or one of its own
        the checkpoint lacks, is still refused (:meth:`load_resumed`). A
        checkpoint without a key front is not a warm-up, and S1 itself
        resumes rather than starts from one.
        """
        if self.e2e_config.key_warm_up:
            raise ValueError("the warm-up resumes its own weights; the joint stage starts from it")
        if not any(name.startswith(KEY_PARAMETER_PREFIXES) for name in weights):
            raise ValueError(f"{path} holds no key front; the joint stage starts from a warm-up")
        dropped = {name for name in weights if name.startswith(WARM_UP_ONLY_PREFIXES)}
        self.load_resumed(
            {name: tensor for name, tensor in weights.items() if name not in dropped}, path
        )
        log.info(
            "joint stage started from a warm-up",
            dropped=len(dropped),
            teacher_copies=sum(name.startswith("teacher_") for name in dropped),
        )

    @property
    def teacher_tower(self) -> BertModel:
        """The parent's fill tower: the frozen copy while ``fill`` trains, ``fill`` otherwise."""
        return self.fill if self.teacher_fill is None else self.teacher_fill

    @property
    def teacher_blocks(self) -> nn.ModuleList:
        """The parent's gated context blocks: the frozen copy while they train, else the blocks."""
        if self.teacher_cross_attention is None:
            return self.cross_attention
        return self.teacher_cross_attention

    def trained_parts(self) -> tuple[str, ...]:
        """The parent's submodules S1 trains beside the front and the head."""
        trains = self.e2e_config.train_fill_tower
        if trains is None:
            return ()
        return ("fill", "cross_attention") if trains.cross_attention else ("fill",)

    def freeze_for_key_warm_up(self) -> None:
        """Leave only the key front and the CTC head -- the MLM head and the blank -- trainable.

        With ``train_fill_tower`` the fill tower trains as well, and its gated
        context blocks when it names them; their frozen copies, the teacher,
        never do.
        """
        trained = (
            *KEY_PARAMETER_PREFIXES,
            "head.",
            *(f"{part}." for part in self.trained_parts()),
        )
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith(trained))

    def freeze_for_joint(self) -> None:
        """Leave everything the key path reads trainable: the joint stage (S2).

        Only the span table and the letter encoder stay frozen
        (:data:`UNREAD_BY_KEYS`): nothing in the stage reads them, and a
        parameter that wants a gradient and never gets one is an error under
        DDP's static bucketing.
        """
        if self.e2e_config.key_warm_up:
            raise ValueError("the warm-up freezes its own way (freeze_for_key_warm_up)")
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(not name.startswith(UNREAD_BY_KEYS))

    def train(self, mode: bool = True) -> KeyE2EModel:
        """Train mode for what trains; in the warm-up, the frozen parts stay in evaluation mode.

        The fill tower and its gated context blocks run in training mode when
        the warm-up trains them, and the teacher's copies of them always run in
        evaluation mode.
        """
        super().train(mode)
        if self.e2e_config.key_warm_up:
            trained = self.trained_parts()
            for name in FROZEN_IN_WARM_UP:
                module = getattr(self, name)
                if name not in trained and isinstance(module, nn.Module):
                    module.eval()
        for teacher in (self.teacher_fill, self.teacher_cross_attention):
            if teacher is not None:
                teacher.eval()
        return self

    def parameter_groups(self, base_lr: float, new_lr: float) -> list[dict[str, object]]:
        """Trainable parent parameters at the base rate; the front, the blank and the
        decoder's cross-attention, where it trains, at the new."""
        base: list[nn.Parameter] = []
        new: list[nn.Parameter] = []
        at_new_rate = (*KEY_PARAMETER_PREFIXES, *RELEARNED_PREFIXES)
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            (new if name.startswith(at_new_rate) else base).append(parameter)
        if not new:
            raise ValueError("no key-level parameter trains; the naming convention has drifted")
        return [{"params": base, "lr": base_lr}, {"params": new, "lr": new_lr}]

    def encode_keys(self, batch: KeyBatch, memory: Any) -> torch.Tensor:
        """The fill tower over a training batch's keys (:meth:`encode_inputs`)."""
        return self.encode_inputs(batch.inputs, memory)

    def encode_inputs(self, keys: KeyInputs, memory: Any) -> torch.Tensor:
        """The fill tower over the keys, ``[B, Wk, H]``: ``[MASK]`` plus the front at each key."""
        words: torch.Tensor = self.fill.embeddings.word_embeddings(keys.key_input_ids)
        at_key = (keys.key_letters != LETTER_PAD).unsqueeze(-1)
        front = self.key_front(keys.key_letters)
        inputs = words + front * at_key
        _mark_fill(inputs, keys.key_attention, memory, keys.context_rows, keys.has_context)
        return self.fill_pass(
            self.fill,
            self.cross_attention,
            inputs,
            keys.key_attention,
            memory,
            keys.context_rows,
            keys.has_context,
        )

    def span_states(self, batch: KeyBatch, memory: Any) -> torch.Tensor:
        """The parent's fill path over the spans the typist meant, ``[B, W, H]``."""
        spans = batch.spans
        tower = self.teacher_tower
        inputs = self._fill_inputs(spans, tower)
        _mark_fill(inputs, spans.attention_mask, memory, spans.context_rows, spans.has_context)
        return self.fill_pass(
            tower,
            self.teacher_blocks,
            inputs,
            spans.attention_mask,
            memory,
            spans.context_rows,
            spans.has_context,
        )

    def ctc_logits(self, states: torch.Tensor) -> torch.Tensor:
        """``[..., H]`` key states to ``[..., E + 1]`` logits, the blank last."""
        transformed = self.head.transform(states)
        logits: torch.Tensor = torch.cat(
            [self.head.decoder(transformed), self.blank(transformed)], dim=-1
        )
        return logits

    def ctc_log_probs(self, states: torch.Tensor) -> torch.Tensor:
        """``[K, H]`` key states to ``[K, E + 1]`` log-probabilities, the blank last, in float32."""
        return self.ctc_logits(states).float().log_softmax(dim=-1)

    def at_span_ends(self, batch: KeyBatch, ends: torch.Tensor) -> torch.Tensor:
        """The ``[P, H]`` span-end states in the span batch's columns, ``[B, W, H]``.

        They land where the parent's encoder states would be, which is how its
        decoder reads them.
        """
        spans = batch.spans
        width = spans.targets.shape[1]
        hidden = ends.shape[-1]
        placed = ends.new_zeros((spans.size * width, hidden)).index_copy(0, spans.scored, ends)
        return placed.view(spans.size, width, hidden)

    def frame_loss(self, batch: KeyBatch, ends: torch.Tensor) -> torch.Tensor:
        """The parent's per-position loss, its decoder reading the student's span-end states."""
        scored = self._scored(batch.spans, self.at_span_ends(batch, ends))
        if scored is None:
            raise ValueError("a key batch scores at least one span")
        return self.position_loss(scored.decoder, scored.targets, scored.candidates)

    def forward(self, batch: KeyBatch) -> RouteAOutput:  # type: ignore[override]
        """The stage's loss: S1's in the warm-up (:meth:`warm_up_loss`), else S2's."""
        if not isinstance(batch, KeyBatch):
            raise TypeError(f"the key model trains on key batches, got a {type(batch).__name__}")
        if self.e2e_config.key_warm_up:
            return self.warm_up_loss(batch)
        return self.joint_loss(batch)

    def ctc_alignment(
        self, batch: KeyBatch, student: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Each row's constrained-CTC loss ``[B]``, and the blank and emit scores behind it.

        *student* is the fill tower over the keys, flat ``[B * (Wk + 2), H]``.
        """
        log_probs = self.ctc_log_probs(student.index_select(0, batch.key_positions))
        blank, emit = alignment_scores(log_probs, batch.key_index, batch.gold)
        nll = constrained_ctc(blank, emit, batch.cost, batch.keys, batch.lengths)
        return nll, blank, emit

    def components(
        self, batch: KeyBatch, blank: torch.Tensor, emit: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """The CTC's per-component losses (#216 §7), detached: what every key stage logs."""
        with torch.no_grad():
            return alignment_components(
                blank,
                emit,
                batch.span_end,
                batch.ends,
                batch.key_mask,
                batch.position_mask,
                batch.corrupted,
            )

    def warm_up_loss(self, batch: KeyBatch) -> RouteAOutput:
        """S1's loss: the constrained CTC, the distillation of the parent's states, the frame term.

        The last two are weighted, and a frame weight of 0 skips the decoder.
        """
        spans = batch.spans
        mark_dynamic(spans)
        hidden = self.fill.config.hidden_size
        with torch.no_grad():
            memory = self.encode_context(spans)
            teacher = self.span_states(batch, memory)
            target = teacher.reshape(-1, hidden).index_select(0, spans.scored)
        student = self.encode_keys(batch, memory).reshape(-1, hidden)
        ends = student.index_select(0, batch.end_positions)
        distill = (ends.float() - target.float()).pow(2).mean()
        nll, blank, emit = self.ctc_alignment(batch, student)
        ctc = nll.sum() / spans.scored.numel()
        loss = ctc + self.keys_config.distill_weight * distill
        extras = {"ctc_loss": ctc.detach(), "distill_loss": distill.detach()}
        if self.keys_config.frame_weight:
            frame = self.frame_loss(batch, ends)
            loss = loss + self.keys_config.frame_weight * frame
            extras["frame_loss"] = frame.detach()
        return RouteAOutput(loss=loss, extras={**extras, **self.components(batch, blank, emit)})

    def joint_loss(self, batch: KeyBatch) -> RouteAOutput:
        """S2's loss: the constrained CTC plus ``transition_weight`` times the transition's term.

        Both are normalised per gold character, the CTC exactly as S1's. The
        transition's term is the cross-entropy of each gold character under
        the decoder's log-softmax over the whole emittable set, teacher-forced
        over the gold prefix and reading every key (:meth:`joint_terms`).
        """
        terms = self.joint_terms(batch)
        scored = terms.targets.numel()
        ctc = terms.nll.sum() / scored
        log_probs = terms.transition.float().log_softmax(dim=-1)
        transition = -log_probs.gather(1, terms.targets.unsqueeze(1)).sum() / scored
        loss = ctc + self.keys_config.transition_weight * transition
        extras = {"ctc_loss": ctc.detach(), "transition_loss": transition.detach()}
        return RouteAOutput(
            loss=loss, extras={**extras, **self.components(batch, terms.blank, terms.emit)}
        )

    def joint_terms(self, batch: KeyBatch) -> _JointTerms:
        """One pass of the joint stage over *batch*: the CTC's terms and the transition's logits.

        The context is read once and the fill tower runs once, over the keys;
        the CTC head reads every key's state and the decoder cross-attends to
        all of them. Every step is a batched tensor op over the whole batch.
        """
        spans = batch.spans
        mark_dynamic(spans)
        encoded = self.encode_keys(batch, self.encode_context(spans))
        nll, blank, emit = self.ctc_alignment(batch, encoded.reshape(-1, encoded.shape[-1]))
        at = batch.gold_positions
        return _JointTerms(
            nll=nll,
            blank=blank,
            emit=emit,
            transition=self.transition_logits(
                self.states_at_keys(batch.key_mask, encoded), batch.gold, at
            ),
            targets=batch.gold.reshape(-1).index_select(0, at),
        )

    def states_at_keys(self, mask: torch.Tensor, encoded: torch.Tensor) -> KeyStates:
        """*encoded* ``[B, Wk + 2, H]``, the fill tower over the keys, without its sentinels.

        *mask* ``[B, Wk]`` marks each row's real keys.
        """
        return KeyStates(states=encoded[:, 1 : mask.shape[1] + 1], mask=mask)

    def key_states(self, keys: KeyInputs) -> KeyStates:
        """Encode the keys once: the search's encoder pass (6a of #257).

        Reads only the keys and the context (:class:`KeyInputs`), never a
        gold or its segmentation: a training batch's :attr:`KeyBatch.inputs`,
        or :meth:`KeyCollator.inputs` over typed records. Returns
        :class:`KeyStates`: ``states [B, Wk, H]`` with ``[b, t]`` row ``b``'s
        key ``t``, and ``mask [B, Wk]``. :meth:`ctc_log_probs` turns
        ``states`` into each key's ``[B, Wk, E + 1]`` CTC log-probabilities
        (the blank last), and :meth:`transition_log_probs` queries the
        decoder over them. The caller sets the mode and the grad context: a
        search runs this under ``eval()`` and ``torch.no_grad()``.
        """
        for tensor in (keys.context_ids, keys.context_mask):
            torch._dynamo.maybe_mark_dynamic(tensor, (0, 1))
        memory = self.context_memory(keys.context_ids, keys.context_mask)
        return self.states_at_keys(keys.key_mask, self.encode_inputs(keys, memory))

    def transition_logits(
        self, keys: KeyStates, gold: torch.Tensor, at: torch.Tensor
    ) -> torch.Tensor:
        """The decoder's teacher-forced logits at the gold characters, ``[P, E]``.

        *gold* ``[B, M]`` holds each row's sentence as emission indices, any
        valid index past its end; step ``j`` reads ``<bos>`` and the gold's
        first ``j`` characters, and every block cross-attends to *keys*. *at*
        ``[P]`` picks the real steps as flat indices into ``[B, M]``
        (:attr:`KeyBatch.gold_positions`), so only those rows meet the head.
        """
        decoder = self.decoder
        ids = decoder.emit_ids[gold]
        previous = torch.cat(
            [torch.full((ids.shape[0], 1), BOS, dtype=torch.long, device=ids.device), ids[:, :-1]],
            dim=1,
        )
        for tensor in (previous, keys.states, keys.mask):
            torch._dynamo.maybe_mark_dynamic(tensor, (0, 1))
        features = decoder.over_keys(previous, keys.states, keys.mask)
        return decoder.emission_logits(features.reshape(-1, features.shape[-1]).index_select(0, at))

    def transition_log_probs(
        self,
        keys: KeyStates,
        rows: torch.Tensor,
        prefixes: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """The next character's log-probabilities after each hypothesis, ``[N, E]`` float32.

        The search's batched transition query (6c of #257). *keys* are
        :meth:`key_states`' output for a set of records; hypothesis ``n``
        belongs to record ``rows[n]`` (``[N]`` long, indices into
        ``keys.states``' first dim) and has emitted ``prefixes[n, :lengths[n]]``
        (``prefixes [N, L]`` long emission indices, any valid emission index
        past ``lengths[n]``; ``lengths [N]`` long, each at most ``L``; ``L``
        may be 0). Row ``n`` is the log-softmax over the whole emittable set,
        emission order, of the character that follows the prefix: the
        distribution the training term reads at that step, which the search
        masks without renormalising. Each query re-runs the decoder over
        ``<bos>`` and the whole prefix, batched over the hypotheses; the
        decoder is causal, so what follows a prefix's end does not reach it.
        Call under ``eval()`` and ``torch.no_grad()``.
        """
        decoder = self.decoder
        tokens = torch.cat(
            [
                torch.full((prefixes.shape[0], 1), BOS, dtype=torch.long, device=prefixes.device),
                decoder.emit_ids[prefixes],
            ],
            dim=1,
        )
        features = decoder.over_keys(tokens, keys.states, keys.mask, rows)
        steps = tokens.shape[1]
        at = torch.arange(tokens.shape[0], device=tokens.device) * steps + lengths
        last = features.reshape(-1, features.shape[-1]).index_select(0, at)
        return decoder.emission_logits(last).float().log_softmax(dim=-1)

    def cross_memory(self, keys: KeyStates, row: int = 0, length: int = N_MAX) -> CrossMemory:
        """Record *row*'s key states as the decoder's cross keys and values, padded to *length*.

        The projections are what the tree recompute reads at every call, made
        once per keystroke: :meth:`~mlime.train.e2e_model.EncoderAttention.kv`
        of each gated block over the row's real keys.
        """
        real = int(keys.mask[row].sum())
        if real == 0 or not bool(keys.mask[row, :real].all()):
            raise ValueError(f"record {row}'s real keys are not a non-empty prefix of its row")
        if real > length:
            raise ValueError(f"{real} keys do not fit a cross length of {length}")
        states = keys.states[row : row + 1, :real]
        projected = [cross.kv(states) for cross in self.decoder.gated_layers()]
        heads, head_dim = int(projected[0][0].shape[1]), int(projected[0][0].shape[3])
        shape = (len(projected), heads, length, head_dim)
        key_slots = states.new_zeros(shape)
        value_slots = states.new_zeros(shape)
        for layer, (layer_keys, layer_values) in enumerate(projected):
            key_slots[layer, :, :real] = layer_keys[0]
            value_slots[layer, :, :real] = layer_values[0]
        bias = states.new_full((1, 1, 1, length), NEG)
        bias[..., :real] = 0.0
        return CrossMemory(keys=key_slots, values=value_slots, bias=bias)

    def tree_feeds(self, call: TreeCall) -> TreeFeeds:
        """A packed call's inputs: reader ids, depths and the additive tree bias.

        A node's token is its prefix's last character mapped to the reader's
        id, ``<bos>`` for the root; a padding row is ``<bos>`` at depth 0 and
        sees only itself.
        """
        size = call.bucket
        device = self.decoder.emit_ids.device
        reader_ids = torch.full((1, size), BOS, dtype=torch.long, device=device)
        positions = torch.zeros((1, size), dtype=torch.long, device=device)
        for row, node in enumerate(call.nodes):
            if node:
                reader_ids[0, row] = self.decoder.emit_ids[node[-1]]
            positions[0, row] = len(node)
        bias = torch.full((1, 1, size, size), NEG, device=device)
        for row in range(size):
            bias[0, 0, row, list(call.visible(row))] = 0.0
        return TreeFeeds(reader_ids=reader_ids, positions=positions, tree_bias=bias)

    def tree_logits(
        self,
        reader_ids: torch.Tensor,
        positions: torch.Tensor,
        tree_bias: torch.Tensor,
        cross: CrossMemory,
    ) -> torch.Tensor:
        """Every node's next-character logits over a packed prefix tree, ``[1, S, E]``.

        ``reader_ids [1, S]`` is each node's last token as a reader id and
        ``positions [1, S]`` its depth; ``tree_bias [1, 1, S, S]`` is additive,
        0 where the column is an ancestor of the row or the row itself and
        :data:`~mlime.train.key_manifest.NEG` elsewhere; *cross* is the
        record's key states. Row ``i`` is what :meth:`transition_logits` gives
        after the prefix ending at node ``i``: a node attending to exactly its
        ancestors at their depths is causal attention over its prefix. A
        padding row that sees only itself changes no other row. Attention is
        a matrix product, an additive mask, a softmax and a matrix product,
        and no row is fully masked, so the module converts for the Neural
        Engine as written. Call under ``eval()`` and ``torch.no_grad()``.
        """
        decoder = self.decoder
        reader = decoder.reader
        x = reader.dropout(reader.input(reader.embed(reader_ids)) + reader.positions(positions))
        for layer, (block, gate) in enumerate(
            zip(reader.layers(), decoder.gated_layers(), strict=True)
        ):
            x = _tree_attention(block, x, tree_bias)
            x = _cross_attention(
                gate,
                x,
                cross.keys[layer].unsqueeze(0),
                cross.values[layer].unsqueeze(0),
                cross.bias,
            )
        return decoder.emission_logits(reader.norm(x))

    def tree_log_probs(
        self,
        reader_ids: torch.Tensor,
        positions: torch.Tensor,
        tree_bias: torch.Tensor,
        cross: CrossMemory,
    ) -> torch.Tensor:
        """:meth:`tree_logits` log-softmaxed over the emittable set in float32, ``[1, S, E]``."""
        return self.tree_logits(reader_ids, positions, tree_bias, cross).float().log_softmax(-1)

    def student_frame(self, batch: KeyBatch) -> torch.Tensor:
        """The parent's decoder frame fed the key front's states at the span ends.

        The states land in the span columns of the span batch
        (:meth:`at_span_ends`), and the frame is the span model's own
        (:meth:`~mlime.train.e2e_model.E2EModel.frame`): the S1 gate.
        """
        spans = batch.spans
        mark_dynamic(spans)
        memory = self.encode_context(spans)
        student = self.encode_keys(batch, memory)
        ends = student.reshape(-1, student.shape[-1]).index_select(0, batch.end_positions)
        return self.frame(spans, self.at_span_ends(batch, ends))

    def parent_frame(self, batch: KeyBatch) -> torch.Tensor:
        """The parent's own frame over the spans: what the gate compares the student against.

        Its encoder states are the teacher's (:meth:`span_states`), the span
        model's own pass through the parent's tower and context blocks even
        while the student's train.
        """
        spans = batch.spans
        mark_dynamic(spans)
        return self.frame(spans, self.span_states(batch, self.encode_context(spans)))


def letter_token_ids(vocabulary: Mapping[str, int]) -> list[int]:
    """The base vocabulary's token id of each key's letter, ``a`` first.

    The base vocabulary holds every ASCII letter as a token of its own -- the
    span table was seeded from the same rows -- so a miss means the tokenizer
    is not the one the model was trained with.
    """
    letters = [chr(ord("a") + offset) for offset in range(LETTER_PAD)]
    missing = [letter for letter in letters if letter not in vocabulary]
    if missing:
        raise KeyError(f"the base vocabulary has no token for the letters {missing}")
    return [vocabulary[letter] for letter in letters]


def _mark_fill(
    inputs: torch.Tensor,
    attention: torch.Tensor,
    memory: Any,
    rows: torch.Tensor,
    kept: torch.Tensor,
) -> None:
    """Mark the fill pass's varying dims, so its compiled form is dynamic from the first call."""
    torch._dynamo.maybe_mark_dynamic(inputs, (0, 1))
    torch._dynamo.maybe_mark_dynamic(attention, (0, 1))
    torch._dynamo.maybe_mark_dynamic(rows, 0)
    torch._dynamo.maybe_mark_dynamic(kept, 0)
    if memory is not None:
        torch._dynamo.maybe_mark_dynamic(memory.states, (0, 1))
        torch._dynamo.maybe_mark_dynamic(memory.mask, (0, 1))


def key_evaluate(
    model: KeyE2EModel,
    examples: Sequence[KeyExample],
    collator: KeyCollator,
    device: torch.device,
    token_budget: int,
) -> tuple[Accuracy, Accuracy]:
    """Per-position accuracy of the parent's decoder fed the front's states, and fed its own.

    The first is the S1 gate's number; the second is the parent's accuracy on
    the same examples, typed the same way, under the same context switch --
    *collator*'s context dropout of 0 or 1 is that switch.
    """
    student = parent = scored = 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for group in token_budget_batches(
            iter(examples), token_budget, collator.max_context_tokens
        ):
            batch = collator(group).to(device)
            targets = batch.spans.targets
            hit, total = count_correct(
                model.predictions(model.student_frame(batch), batch.spans), targets
            )
            own, _ = count_correct(
                model.predictions(model.parent_frame(batch), batch.spans), targets
            )
            student += hit
            parent += own
            scored += total
    if was_training:
        model.train()
    return Accuracy(correct=student, scored=scored), Accuracy(correct=parent, scored=scored)


def joint_evaluate(
    model: KeyE2EModel,
    examples: Sequence[KeyExample],
    collator: KeyCollator,
    device: torch.device,
    token_budget: int,
) -> tuple[Accuracy, float]:
    """The joint stage's held-out record: teacher-forced transition accuracy, and the CTC loss.

    The accuracy counts the gold characters at which the decoder's argmax
    over the whole emittable set, teacher-forced over the gold prefix, is
    the gold; the loss is the constrained CTC per gold character, as the
    step reports it. A per-key greedy read means nothing here, and the
    stage's gate is the constrained decode. *collator*'s context dropout of
    0 or 1 is the context switch.
    """
    scored = 0
    correct = torch.zeros((), dtype=torch.long, device=device)
    nll = torch.zeros((), dtype=torch.float64, device=device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for group in token_budget_batches(
            iter(examples), token_budget, collator.max_context_tokens
        ):
            terms = model.joint_terms(collator(group).to(device))
            correct += (terms.transition.argmax(dim=-1) == terms.targets).sum()
            scored += terms.targets.numel()
            nll += terms.nll.double().sum()
    if was_training:
        model.train()
    if not scored:
        raise ValueError("no held-out example scored a character")
    return Accuracy(correct=int(correct), scored=scored), float(nll) / scored
