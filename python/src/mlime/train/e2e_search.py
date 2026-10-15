"""The end-to-end model's beam search over typed lanes.

A lane is one reading of one input: the encoder's vectors at its typed
positions and the characters each position admits. The evaluation searches
every lattice path of every record as a lane; the ranking loss (#116) searches
every segmentation of every training example the same way, for the negatives
it ranks the gold sentence against. Both read the same scores: each step's
log-probability over the whole emittable set, restricted afterwards.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from mlime.train.charlm_vocab import BOS
from mlime.train.e2e_model import E2EModel


@dataclass(frozen=True)
class Lane:
    """One lattice path ready to decode: whose it is, what the encoder said, what it admits.

    ``vectors [n, enc]`` is the encoder's hidden state at the path's typed
    positions and ``candidates`` one ``[E]`` bool row per position, the
    emission indices that position's span admits.
    """

    owner: int
    vectors: torch.Tensor
    candidates: torch.Tensor

    @property
    def length(self) -> int:
        """The number of typed positions."""
        return int(self.vectors.shape[0])


def decode_lanes(
    model: E2EModel, lanes: Sequence[Lane], beam_width: int
) -> list[list[tuple[list[int], float]]]:
    """Beam-search lanes together; each lane's hypotheses and scores.

    Every lane owns a block of ``beam_width`` rows from the first step, the
    unused ones at a score of minus infinity, so the row layout never changes
    and the encoder keys and values are expanded once. Each step's scores are
    log-probabilities over the full emittable row, and the candidate
    restriction masks the characters the position does not admit afterwards,
    so a path's score is comparable with another segmentation's. Rows still at minus
    infinity when the path ends are slots the lattice never filled and are
    dropped, so a lane admitting fewer than ``beam_width`` sentences returns
    fewer.

    Lanes may differ in length. A shorter lane's span vectors are padded and
    masked out of the cross-attention, and once it has ended each of its rows
    continues with a single free step, so its beam is carried unchanged to the
    last position and cut back to its own length on the way out. Lanes of one
    length -- what the evaluation's buckets hold -- take no mask and no
    padding, so their arithmetic is the equal-length search's exactly.
    """
    decoder = model.decoder
    device = decoder.emit_ids.device
    count, width = len(lanes), beam_width
    rows = count * width
    lengths = [lane.length for lane in lanes]
    length = max(lengths)
    vectors = nn.utils.rnn.pad_sequence([lane.vectors for lane in lanes], batch_first=True).to(
        device
    )
    masks = nn.utils.rnn.pad_sequence([lane.candidates for lane in lanes], batch_first=True).to(
        device
    )
    emittable = masks.shape[-1]
    everything = torch.arange(emittable, device=device)
    lane_of_row = torch.arange(count, device=device).repeat_interleave(width)
    encoder_kv = [
        tuple(part.repeat_interleave(width, dim=0) for part in cross.kv(vectors))
        for cross in decoder.gated_layers()
    ]
    ragged = min(lengths) != length
    lane_lengths = torch.tensor(lengths, device=device)
    encoder_mask = (
        (torch.arange(length, device=device) < lane_lengths.unsqueeze(1))[lane_of_row]
        if ragged
        else None
    )
    # The step an ended row takes: emission 0 for free, everything else barred.
    carried = torch.full((emittable,), float("-inf"), device=device)
    carried[0] = 0.0
    layers, heads = decoder.reader.config.layers, decoder.reader.config.heads
    head_dim = decoder.reader.config.hidden // heads
    keys = torch.zeros(rows, layers, heads, 0, head_dim, device=device)
    values = keys.clone()
    scores = torch.full((count, width), float("-inf"), device=device)
    scores[:, 0] = 0.0
    tokens = torch.full((rows,), BOS, dtype=torch.long, device=device)
    block = torch.arange(count, device=device).unsqueeze(1) * width
    parents: list[torch.Tensor] = []
    choices: list[torch.Tensor] = []
    for position in range(length):
        logits, (keys, values) = decoder.step(
            tokens,
            vectors[lane_of_row, position],
            (keys, values),
            encoder_kv,  # type: ignore[arg-type]
            encoder_mask,
            candidates=everything,
        )
        admitted = masks[lane_of_row, position]
        log_probs = logits.float().log_softmax(dim=-1).masked_fill(~admitted, float("-inf"))
        if ragged:
            ended = (lane_lengths <= position)[lane_of_row]
            log_probs = torch.where(ended.unsqueeze(1), carried, log_probs)
        total = (scores.reshape(rows, 1) + log_probs).reshape(count, width * emittable)
        top = total.topk(width, dim=1)
        parent = top.indices // emittable
        choice = top.indices % emittable
        source = (block + parent).reshape(rows)
        keys, values = keys[source], values[source]
        tokens = decoder.emit_ids[choice.reshape(rows)]
        scores = top.values
        parents.append(parent)
        choices.append(choice)

    # Plain lists: the walk back reads one element at a time, which a tensor
    # charges a dispatch for.
    parent_steps: list[list[list[int]]] = torch.stack(parents).tolist()
    choice_steps: list[list[list[int]]] = torch.stack(choices).tolist()
    final: list[list[float]] = scores.tolist()
    decoded: list[list[tuple[list[int], float]]] = []
    for lane in range(count):
        hypotheses = []
        for beam, score in enumerate(final[lane]):
            if score == float("-inf"):
                continue
            sequence = []
            row = beam
            for position in range(length - 1, -1, -1):
                sequence.append(choice_steps[position][lane][row])
                row = parent_steps[position][lane][row]
            sequence.reverse()
            hypotheses.append((sequence[: lengths[lane]], score))
        decoded.append(hypotheses)
    return decoded
