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

**S1 with the tower** (``train_fill_tower``, #241) trains the fill tower too,
beside the front and the head and at the head's rate, in training mode. The
teacher must stay the parent, so the model keeps a frozen copy of the parent's
tower -- ``teacher_fill``, always in evaluation mode -- and the teacher's pass
and the gate's parent frame run through it; the gated context blocks, the span
tables and the context tower they share stay frozen, so the copy's states are
the parent's exactly. The copy is a parameter set the size of the tower and is
saved with the run, so a resume distils towards the same targets.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import torch
from torch import nn
from transformers import BertConfig, BertForMaskedLM, BertModel

from mlime.train.charlm_model import TransformerCharLm
from mlime.train.ctc import alignment_components, alignment_scores, constrained_ctc
from mlime.train.e2e_model import E2EConfig, E2EModel, context_tower
from mlime.train.keys import KeyBatch, KeyCollator, KeyExample, KeyFrontConfig, key_coordinates
from mlime.train.lexicon import Lexicon
from mlime.train.loop import Accuracy
from mlime.train.model import ContextTower, RouteAOutput, count_correct, mark_dynamic
from mlime.train.samples import LETTER_PAD, token_budget_batches

#: Parameter prefixes of the parts the key model adds: they train at the new rate.
KEY_PARAMETER_PREFIXES = ("key_front.", "blank.")

#: The submodules S1 leaves frozen and runs in evaluation mode.
FROZEN_IN_WARM_UP = ("fill", "context", "cross_attention", "decoder", "letter_encoder")

#: Parameter prefix of the frozen copy of the parent's fill tower, the teacher
#: of an S1 that trains the tower.
TEACHER_PREFIX = "teacher_fill."


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
        # Seeded with the parent's tower when the weights load (load_resumed).
        self.teacher_fill = (
            copy.deepcopy(self.fill).requires_grad_(False) if config.train_fill_tower else None
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
        """Load *weights*, seeding the teacher's tower from them when they hold none.

        A checkpoint without a teacher never trained its fill tower -- a run
        that trains it keeps and saves the frozen copy -- so its tower is its
        parent's, and the copy starts as that. A checkpoint that holds a
        teacher loads it as saved; one loaded by a model that keeps none is
        refused as unexpected weights.
        """
        teacher = self.teacher_fill
        if teacher is None or any(name.startswith(TEACHER_PREFIX) for name in weights):
            super().load_resumed(weights, path, fresh)
            return
        super().load_resumed(weights, path, (*fresh, TEACHER_PREFIX))
        teacher.load_state_dict(self.fill.state_dict())

    @property
    def teacher_tower(self) -> BertModel:
        """The parent's fill tower: the frozen copy while ``fill`` trains, ``fill`` otherwise."""
        return self.fill if self.teacher_fill is None else self.teacher_fill

    def freeze_for_key_warm_up(self) -> None:
        """Leave only the key front and the CTC head -- the MLM head and the blank -- trainable.

        With ``train_fill_tower`` the fill tower trains as well; its frozen
        copy, the teacher, never does.
        """
        trained: tuple[str, ...] = (*KEY_PARAMETER_PREFIXES, "head.")
        if self.e2e_config.train_fill_tower:
            trained = (*trained, "fill.")
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith(trained))

    def train(self, mode: bool = True) -> KeyE2EModel:
        """Train mode for what trains; in the warm-up, the frozen parts stay in evaluation mode.

        The fill tower runs in training mode when the warm-up trains it, and
        the teacher's copy of it always runs in evaluation mode.
        """
        super().train(mode)
        if self.e2e_config.key_warm_up:
            for name in FROZEN_IN_WARM_UP:
                if name == "fill" and self.e2e_config.train_fill_tower:
                    continue
                module = getattr(self, name)
                if isinstance(module, nn.Module):
                    module.eval()
        if self.teacher_fill is not None:
            self.teacher_fill.eval()
        return self

    def parameter_groups(self, base_lr: float, new_lr: float) -> list[dict[str, object]]:
        """Trainable parent parameters at the base rate, the front's and the blank's at the new."""
        base: list[nn.Parameter] = []
        new: list[nn.Parameter] = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            (new if name.startswith(KEY_PARAMETER_PREFIXES) else base).append(parameter)
        if not new:
            raise ValueError("no key-level parameter trains; the naming convention has drifted")
        return [{"params": base, "lr": base_lr}, {"params": new, "lr": new_lr}]

    def encode_keys(self, batch: KeyBatch, memory: Any) -> torch.Tensor:
        """The fill tower over the keys, ``[B, Wk, H]``: ``[MASK]`` plus the front at each key."""
        spans = batch.spans
        words: torch.Tensor = self.fill.embeddings.word_embeddings(batch.key_input_ids)
        at_key = (batch.key_letters != LETTER_PAD).unsqueeze(-1)
        front = self.key_front(batch.key_letters)
        inputs = words + front * at_key
        _mark_fill(inputs, batch.key_attention, memory, spans.context_rows, spans.has_context)
        return self.fill_pass(
            self.fill, inputs, batch.key_attention, memory, spans.context_rows, spans.has_context
        )

    def span_states(self, batch: KeyBatch, memory: Any) -> torch.Tensor:
        """The parent's fill tower over the spans the typist meant, ``[B, W, H]``."""
        spans = batch.spans
        tower = self.teacher_tower
        inputs = self._fill_inputs(spans, tower)
        _mark_fill(inputs, spans.attention_mask, memory, spans.context_rows, spans.has_context)
        return self.fill_pass(
            tower, inputs, spans.attention_mask, memory, spans.context_rows, spans.has_context
        )

    def ctc_log_probs(self, states: torch.Tensor) -> torch.Tensor:
        """``[K, H]`` key states to ``[K, E + 1]`` log-probabilities, the blank last, in float32."""
        transformed = self.head.transform(states)
        logits = torch.cat([self.head.decoder(transformed), self.blank(transformed)], dim=-1)
        return logits.float().log_softmax(dim=-1)

    def forward(self, batch: KeyBatch) -> RouteAOutput:  # type: ignore[override]
        """S1's loss: the constrained CTC plus the weighted distillation of the parent's states."""
        if not isinstance(batch, KeyBatch):
            raise TypeError(f"the key model trains on key batches, got a {type(batch).__name__}")
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
        log_probs = self.ctc_log_probs(student.index_select(0, batch.key_positions))
        blank, emit = alignment_scores(log_probs, batch.key_index, batch.gold)
        nll = constrained_ctc(blank, emit, batch.cost, batch.keys, batch.lengths)
        ctc = nll.sum() / spans.scored.numel()
        loss = ctc + self.keys_config.distill_weight * distill
        with torch.no_grad():
            components = alignment_components(
                blank,
                emit,
                batch.span_end,
                batch.ends,
                batch.key_mask,
                batch.position_mask,
                batch.corrupted,
            )
        return RouteAOutput(
            loss=loss,
            extras={
                "ctc_loss": ctc.detach(),
                "distill_loss": distill.detach(),
                **components,
            },
        )

    def student_frame(self, batch: KeyBatch) -> torch.Tensor:
        """The parent's decoder frame fed the key front's states at the span ends.

        The states land in the span columns of the span batch, where the
        parent's encoder states would be, and the frame is the span model's
        own (:meth:`~mlime.train.e2e_model.E2EModel.frame`): the S1 gate.
        """
        spans = batch.spans
        mark_dynamic(spans)
        memory = self.encode_context(spans)
        student = self.encode_keys(batch, memory)
        hidden = student.shape[-1]
        ends = student.reshape(-1, hidden).index_select(0, batch.end_positions)
        width = spans.targets.shape[1]
        placed = student.new_zeros((spans.size * width, hidden)).index_copy(0, spans.scored, ends)
        return self.frame(spans, placed.view(spans.size, width, hidden))

    def parent_frame(self, batch: KeyBatch) -> torch.Tensor:
        """The parent's own frame over the spans: what the gate compares the student against.

        Its encoder states are the teacher's (:meth:`span_states`), the span
        model's own pass through the parent's tower even while ``fill`` trains.
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
