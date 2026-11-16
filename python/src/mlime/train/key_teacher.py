"""The character LM as the key ranking stage's teacher (S5, #298).

S4's list -- the gold sentence and the eight negatives of the no-grad beam --
is ordered by the model's own score ``S``. A character LM orders it better
than ``S`` does on full pinyin, and the design keeps one network at inference,
so the LM's ordering is taught to the score instead of being fused in the
search: for each example the frozen LM scores every sentence of the list under
the example's context, and the stage adds ``lambda_d * KL(q || p)`` over the
list, ``p = softmax(S)`` and ``q = softmax(stop_grad(S) + w * log p_LM)``.
The gradient of the term with respect to the list's scores is ``p - q``.

**The conditioning** is ``ime-lm``'s: ``<bos>``, the last ``context_chars``
characters of the example's context, ``<sep>``, the sentence, ``<eos>``, and
a context the key model dropped is dropped for the LM too (``<bos> <sep>``).
``log p_LM`` is the sum over the sentence's characters and the ``<eos>``, each
a log-softmax over the LM's whole alphabet, taken at the sentence's positions
only and a chunk of sentences at a time, so no tensor of a whole batch times
the alphabet is ever held.

**The alphabets** differ: the key model emits the lexicon's characters and the
LM is indexed by its own table. A sentence is a row of emission indices, which
:attr:`Teacher.emit_to_lm` maps to LM ids by character. A sentence can be made
of any emittable character, so the whole set is checked when the teacher is
loaded: a character the LM has no id for fails there, by name, and nothing is
read back from the device in a step.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn.functional as functional

from mlime.logging import log
from mlime.train.charlm_model import CharLm, CharLmConfig
from mlime.train.charlm_vocab import BOS, EOS, PAD, SEP, UNK

#: Positions of LM input one chunk of sentences holds, padding included. The
#: chunk's logits are ``sentences * (span + 1) * alphabet`` floats, so this
#: bounds the teacher's peak whatever the step's batch is.
CHUNK_POSITIONS = 4096


@dataclass(frozen=True)
class TeacherVocab:
    """What the collator needs of the LM: its alphabet and how much context it reads."""

    index: dict[str, int]
    context_chars: int

    @classmethod
    def read(cls, path: Path) -> TeacherVocab:
        """The alphabet and context window of the LM checkpoint at *path*, without its weights."""
        state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        return cls(
            index={character: i for i, character in enumerate(state["vocab"])},
            context_chars=CharLmConfig(**state["config"]).context_chars,
        )


@dataclass(frozen=True)
class TeacherContext:
    """What the LM reads of a batch besides its sentences, and how it will read it.

    ``prelude [B, P]`` and ``lengths [B]`` are each example's ``<bos> context
    <sep>`` as LM ids. A sentence's cost is its prelude plus itself, and a
    prelude is anywhere from two to sixty-four positions, so the sentences are
    scored longest-prelude-last in chunks that each pad only to their own
    longest prelude: ``order [I]`` lists the items in that order, ``ranks
    [I]`` undoes it, and ``chunks`` are host ``(start, stop, prelude width)``
    ranges over it, planned where the batch is collated so the step never
    reads a length back. ``span`` is the sentence width the plan was made for.
    """

    prelude: torch.Tensor
    lengths: torch.Tensor
    order: torch.Tensor
    ranks: torch.Tensor
    chunks: tuple[tuple[int, int, int], ...]
    span: int

    def to(self, device: torch.device) -> TeacherContext:
        """The same context on *device*."""
        return replace(
            self,
            prelude=self.prelude.to(device, non_blocking=True),
            lengths=self.lengths.to(device, non_blocking=True),
            order=self.order.to(device, non_blocking=True),
            ranks=self.ranks.to(device, non_blocking=True),
        )

    def pin_memory(self) -> TeacherContext:
        """The same context in page-locked host memory."""
        return replace(
            self,
            prelude=self.prelude.pin_memory(),
            lengths=self.lengths.pin_memory(),
            order=self.order.pin_memory(),
            ranks=self.ranks.pin_memory(),
        )


def teacher_context(
    contexts: Sequence[str | None],
    vocab: TeacherVocab,
    owners: Sequence[int],
    span: int,
    positions: int = CHUNK_POSITIONS,
) -> TeacherContext:
    """The preludes of *contexts*, ``None`` standing for one the key model dropped, and the plan.

    A character outside the LM's alphabet in the context is ``<unk>``, as
    ``ime-lm`` reads it. *owners* is the example each item to be scored reads
    (in the order of the items), *span* their sentences' padded width, and
    *positions* the most LM input positions a chunk holds.
    """
    rows = []
    for context in contexts:
        tail = "" if not context else context[max(0, len(context) - vocab.context_chars) :]
        rows.append([BOS, *(vocab.index.get(character, UNK) for character in tail), SEP])
    width = max(map(len, rows))
    prelude = torch.full((len(rows), width), PAD, dtype=torch.long)
    for row, ids in enumerate(rows):
        prelude[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)

    needs = [len(rows[owner]) for owner in owners]
    order = sorted(range(len(owners)), key=needs.__getitem__)
    chunks: list[tuple[int, int, int]] = []
    start = 0
    while start < len(order):
        stop = start + 1
        while (
            stop < len(order) and (stop + 1 - start) * (needs[order[stop]] + span + 1) <= positions
        ):
            stop += 1
        chunks.append((start, stop, needs[order[stop - 1]]))
        start = stop
    ranks = [0] * len(order)
    for place, item in enumerate(order):
        ranks[item] = place
    return TeacherContext(
        prelude=prelude,
        lengths=torch.tensor([len(ids) for ids in rows], dtype=torch.long),
        order=torch.tensor(order, dtype=torch.long),
        ranks=torch.tensor(ranks, dtype=torch.long),
        chunks=tuple(chunks),
        span=span,
    )


class Teacher:
    """A frozen character LM that scores the ranking list's sentences under their contexts.

    Deliberately not an ``nn.Module``: it is an asset of the run, so it must
    stay out of the model's ``state_dict`` (a checkpoint would carry it, and
    S4's would no longer load) and out of the optimiser.
    """

    def __init__(self, model: CharLm, emit_to_lm: torch.Tensor):
        self.model = model.eval().requires_grad_(False)
        self.emit_to_lm = emit_to_lm

    @classmethod
    def load(cls, path: Path, characters: Sequence[str], device: torch.device) -> Teacher:
        """The LM checkpoint at *path* on *device*, mapped onto the emittable *characters*."""
        # Imported here: `charlm` reads `mlime.train.run`, which reads the ranking stage.
        from mlime.train.charlm import load_model

        model, vocab, step = load_model(path, torch.device("cpu"))
        index = vocab.index
        missing = [character for character in characters if character not in index]
        if missing:
            raise ValueError(
                f"the teacher at {path} has no ids for {len(missing)} of the model's emittable "
                f"characters ({missing[:8]}...); it cannot score every sentence of a list"
            )
        log.info("key ranking teacher loaded", path=str(path), step=step, alphabet=len(vocab))
        emit_to_lm = torch.tensor([index[character] for character in characters])
        return cls(model.to(device), emit_to_lm.to(device))

    @torch.no_grad()
    def log_probs(
        self,
        context: TeacherContext,
        owner: torch.Tensor,
        sequences: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """``[I]``: each sentence's log-probability with its ``<eos>``, under its example's context.

        ``sequences [I, M]`` are emission indices (any valid one past
        ``lengths [I]``) and sentence ``i`` reads the prelude of example
        ``owner[i]``, the owners *context* was planned for. Every shape is a
        host number: nothing is read back.
        """
        items, span = sequences.shape
        if span != context.span or len(context.order) != items:
            raise ValueError(
                f"the teacher's chunks were planned for {len(context.order)} sentences of "
                f"{context.span} characters, not {items} of {span}"
            )
        device = sequences.device
        prelude_width = context.prelude.shape[1]
        width = prelude_width + span + 1
        prelude_lengths = context.lengths[owner]

        sentence = functional.pad(self.emit_to_lm[sequences], (0, 1))
        sentence = sentence.scatter(1, lengths.unsqueeze(1), torch.full_like(sentence, EOS))
        position = torch.arange(width, device=device)
        offset = position - prelude_lengths.unsqueeze(1)
        prelude = functional.pad(context.prelude, (0, width - prelude_width), value=PAD)[owner]
        tokens = torch.where(
            offset < 0,
            prelude,
            torch.where(
                offset <= lengths.unsqueeze(1), sentence.gather(1, offset.clamp(0, span)), PAD
            ),
        )

        # Input position ``plen - 1 + j`` predicts sentence character ``j``; step
        # ``length`` predicts the ``<eos>`` and later steps score nothing.
        step = torch.arange(span + 1, device=device)
        at = prelude_lengths.unsqueeze(1) - 1 + step
        valid = step <= lengths.unsqueeze(1)
        target = tokens.gather(1, at + 1)

        totals = []
        for start, stop, longest in context.chunks:
            rows = context.order[start:stop]
            features = self.model.features(tokens[rows, : longest + span])
            picked = features.gather(1, at[rows].unsqueeze(2).expand(-1, -1, features.shape[2]))
            logits = self.model.logits(picked).float()
            chosen = logits.gather(2, target[rows].unsqueeze(2)).squeeze(2)
            scored = chosen - logits.logsumexp(dim=2)
            totals.append(torch.where(valid[rows], scored, 0.0).sum(dim=1))
        return torch.cat(totals)[context.ranks]


def distillation(
    slots: torch.Tensor, listed: torch.Tensor, teacher: torch.Tensor, scale: float
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """``KL(q || p)`` of each example's list, and the diagnostics a run is judged by.

    *slots* ``[B, 1 + K]`` are the list's scores, minus infinity where *listed*
    is false, and *teacher* the LM's log-probability of each slot. ``p`` is the
    softmax of the scores and ``q`` that of the same scores held constant plus
    *scale* times the LM's, so the gradient of the term with respect to the
    scores is ``p - q`` and a scale of zero makes the term exactly zero.
    Returns the per-example KL ``[B]`` and ``teacher_kl`` (its mean),
    ``teacher_flip`` (the share of examples whose best sentence under ``q``
    is not the one under ``p``) and ``teacher_gold_first`` (the share whose
    gold the LM alone ranks first).
    """
    log_p = slots.log_softmax(dim=1)
    with torch.no_grad():
        shifted = (slots.detach() + scale * teacher).masked_fill(~listed, float("-inf"))
        log_q = shifted.log_softmax(dim=1)
        q = log_q.exp()
    kl = (q * (log_q.masked_fill(~listed, 0.0) - log_p.masked_fill(~listed, 0.0))).sum(dim=1)
    with torch.no_grad():
        alone = teacher.masked_fill(~listed, float("-inf"))
        extras = {
            "teacher_kl": kl.detach().mean(),
            "teacher_flip": (log_q.argmax(dim=1) != log_p.detach().argmax(dim=1)).float().mean(),
            "teacher_gold_first": (alone.argmax(dim=1) == 0).float().mean(),
        }
    return kl, extras
