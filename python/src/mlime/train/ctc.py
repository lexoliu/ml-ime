"""Constrained CTC: the gold sentence's likelihood summed over its legal alignments only.

#216 §2-§3. Every key emits a character or a blank; a character lands on the
last key of the span it consumes, and the spans of one alignment cover the
keys contiguously, in order. Which spans may carry which character is the
legality automaton's (:class:`~mlime.legality.Legality`), and an arc that
reads its character through a correction carries the correction's cost. The
loss is

    -log  sum over legal alignments of  prod_t p_t(blank or c_j) * exp(-w(s_j, c_j))

with each key's distribution normalised over the whole emittable set plus the
blank -- inference masks with the same log-probabilities and never
renormalises, so covers of different lengths stay comparable (#115).

The forward recursion runs over the sentence's positions. With ``alpha_j(e)``
the log-mass of the alignments of characters ``0..j`` whose last emission is
at key ``e - 1``, and ``prefix(e)`` the summed blank log-probabilities of keys
``0..e-1``, the variable ``beta_j(e) = alpha_j(e) - prefix(e)`` obeys

    beta_j(e) = emit_j(e - 1) - blank(e - 1) + logsumexp_d [ beta_{j-1}(e - d) - w_j(e - 1, d) ]

over span widths ``d`` of at most :data:`~mlime.legality.MAX_ARC_KEYS` keys, and
the sentence's log-likelihood is ``beta_{m-1}(n) + prefix(n)``. Each position
is a handful of batched ops over every row, key and width at once; the arc
table comes from the collator, built on the host, so nothing here reads the
device back.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional

#: The floor an unreachable state and an absent arc sit at. Finite, so a
#: logsumexp over nothing but unreachable states has a gradient of zero rather
#: than NaN; far enough below any real score that adding a few of them is
#: still unreachable, and far enough from float32's limit that sums of them do
#: not overflow.
UNREACHABLE = -1e30


def constrained_ctc(
    blank: torch.Tensor,
    emit: torch.Tensor,
    cost: torch.Tensor,
    keys: torch.Tensor,
    length: torch.Tensor,
) -> torch.Tensor:
    """Each row's negative log-likelihood of its gold sentence over its legal alignments.

    *blank* ``[B, T]`` is each key's blank log-probability and *emit* ``[B, M,
    T]`` the log-probability key ``t`` gives character ``j`` of the row's gold
    sentence. *cost* ``[B, M, T, W]`` is the arc table: entry ``[b, j, t, d -
    1]`` is the correction cost of emitting character ``j`` at key ``t`` after
    consuming the ``d`` keys ``t - d + 1 .. t``, and ``inf`` where the legality
    automaton has no such arc on any legal alignment
    (:meth:`~mlime.legality.GoldArcs.dense`). *keys* ``[B]`` and *length*
    ``[B]`` are each row's key count and sentence length; everything past them
    is padding the arc table already rules out. Returns ``[B]``.

    A row whose gold has no legal alignment comes out near ``-UNREACHABLE``;
    the builder never lets one reach a batch.
    """
    rows, positions, width = emit.shape
    if blank.shape != (rows, width):
        raise ValueError(f"blank is {tuple(blank.shape)} against emit's {tuple(emit.shape)}")
    if cost.shape[:3] != emit.shape:
        raise ValueError(f"cost is {tuple(cost.shape)} against emit's {tuple(emit.shape)}")
    spans = cost.shape[3]
    # Absent arcs priced at the floor, and the widths reversed so a window of
    # the shifted state lines up with them: window slot k holds the state
    # ``spans - k`` keys back, which pairs with width ``spans - k``.
    priced = torch.where(torch.isinf(cost), -UNREACHABLE, cost).flip(-1)
    step = emit - blank.unsqueeze(1)
    prefix = functional.pad(blank.cumsum(-1), (1, 0))
    floor = blank.new_full((rows, 1), UNREACHABLE)
    beta = torch.cat([blank.new_zeros((rows, 1)), blank.new_full((rows, width), UNREACHABLE)], 1)
    betas = []
    for position in range(positions):
        windows = functional.pad(beta, (spans, 0), value=UNREACHABLE).unfold(1, spans, 1)
        inner = windows[:, 1 : width + 1] - priced[:, position]
        beta = torch.cat([floor, step[:, position] + torch.logsumexp(inner, dim=-1)], 1)
        betas.append(beta)
    stacked = torch.stack(betas, dim=1)
    row = torch.arange(rows, device=blank.device)
    total = stacked[row, length - 1, keys] + prefix[row, keys]
    return -total


def alignment_scores(
    log_probs: torch.Tensor, key_index: torch.Tensor, gold: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The blank and gold-character log-probabilities :func:`constrained_ctc` reads.

    *log_probs* ``[K, E + 1]`` holds every real key of the batch, the blank in
    the last column; *key_index* ``[B, T]`` names each row's key ``t`` among
    them (padding names any real one, which the arc table rules out), and
    *gold* ``[B, M]`` each row's sentence as emission indices. Two gathers,
    so the ``[B, M, T]`` gold scores never pass through a ``[B, T, E]`` frame.
    """
    rows, width = key_index.shape
    columns = log_probs.shape[1]
    blank = log_probs[:, -1].index_select(0, key_index.reshape(-1)).view(rows, width)
    flat = key_index.unsqueeze(1) * columns + gold.unsqueeze(2)
    emit = log_probs.reshape(-1).index_select(0, flat.reshape(-1)).view(*flat.shape)
    return blank, emit


def alignment_components(
    blank: torch.Tensor,
    emit: torch.Tensor,
    span_end: torch.Tensor,
    ends: torch.Tensor,
    key_mask: torch.Tensor,
    position_mask: torch.Tensor,
    corrupted: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """The per-component losses of #216 §7, read at each row's typed cover.

    The typed cover -- the spans the typist's keys were cut into -- is one of
    the gold's legal alignments, and the components are read on it:

    - ``segmentation``: the emit-or-blank decision at every key, a binary
      cross-entropy whose target is "emit" exactly at the cover's span ends
      (*ends* ``[B, T]``), averaged over real keys (*key_mask*);
    - ``clean_spans`` and ``typo_spans``: the character term at each span's
      end, ``-log p(c_j | emit)``, averaged over the spans augmentation left
      alone and over the ones it corrupted (*corrupted* ``[B, M]``);
    - ``typo_share``: the corrupted spans' share of the positions, so a
      ``typo_spans`` of zero is told apart from a batch without typos.

    *span_end* ``[B, M]`` is each position's end key. Diagnostics: nothing
    here is differentiated.
    """
    # log(1 - p(blank)), kept finite where the blank holds all the mass.
    emitting = torch.log(-torch.expm1(blank.clamp(max=-1e-7)))
    decision = torch.where(ends, emitting, blank)
    real_keys = key_mask.to(blank.dtype)
    segmentation = -(decision * real_keys).sum() / real_keys.sum()
    gold = emit.gather(2, span_end.unsqueeze(2)).squeeze(2)
    character = -(gold - emitting.gather(1, span_end))
    clean = (position_mask & ~corrupted).to(blank.dtype)
    typo = (position_mask & corrupted).to(blank.dtype)
    positions = position_mask.to(blank.dtype).sum()
    return {
        "segmentation_loss": segmentation,
        "clean_span_loss": (character * clean).sum() / clean.sum().clamp(min=1.0),
        "typo_span_loss": (character * typo).sum() / typo.sum().clamp(min=1.0),
        "typo_share": typo.sum() / positions,
    }
