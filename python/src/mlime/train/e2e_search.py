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
from mlime.train.e2e_model import Decoder, E2EModel


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


def bucket(size: int) -> int:
    """*size* rounded up to the next of four steps an octave: 8, 10, 12, 14, 16, 20, ...

    A captured search graph serves one shape; padding every search to these
    sizes bounds the graphs a run records by a few an octave, at a quarter
    of padding at most.
    """
    if size <= 8:
        return 8
    quantum = 1 << (size.bit_length() - 3)
    return -(-size // quantum) * quantum


@dataclass(frozen=True)
class SearchInputs:
    """What a search reads on the device, lanes padded to one length.

    ``vectors [L, T, enc]`` and ``masks [L, T, E]`` are every lane's span
    vectors and admitted characters, ``lane_lengths [L]`` each lane's own
    length. ``ragged`` is whether the lanes differ in length, decided on the
    host: an equal-length search takes no mask, so its arithmetic is the
    unpadded search's exactly.
    """

    vectors: torch.Tensor
    masks: torch.Tensor
    lane_lengths: torch.Tensor
    ragged: bool
    width: int


def search_inputs(
    lanes: Sequence[Lane], beam_width: int, device: torch.device, *, bucketed: bool
) -> SearchInputs:
    """Pad *lanes* into the tensors a search reads; with *bucketed*, to :func:`bucket` sizes.

    The extra lanes a bucket adds admit nothing, so every row of theirs stays
    at minus infinity, and the extra positions are the ragged search's ended
    steps.
    """
    lengths = [lane.length for lane in lanes]
    length = max(lengths)
    if bucketed:
        length = bucket(length)
        filler = lanes[0]
        padding = bucket(len(lanes)) - len(lanes)
        lanes = [
            *lanes,
            *[
                Lane(-1, filler.vectors[:1], torch.zeros_like(filler.candidates[:1]))
                for _ in range(padding)
            ],
        ]
        lengths += [1] * padding
    vectors = nn.utils.rnn.pad_sequence([lane.vectors for lane in lanes], batch_first=True)
    masks = nn.utils.rnn.pad_sequence([lane.candidates for lane in lanes], batch_first=True)
    vectors = nn.functional.pad(vectors, (0, 0, 0, length - vectors.shape[1]))
    masks = nn.functional.pad(masks, (0, 0, 0, length - masks.shape[1]))
    return SearchInputs(
        vectors=vectors.to(device),
        masks=masks.to(device),
        lane_lengths=torch.tensor(lengths).to(device),
        ragged=min(lengths) != length,
        width=beam_width,
    )


def run_search(
    decoder: Decoder, inputs: SearchInputs
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The whole search on the device: ``parents``, ``choices [T, L, W]`` and the final scores.

    Every lane owns a block of ``beam_width`` rows from the first step, the
    unused ones at a score of minus infinity, so the row layout never changes
    and a lane's beams read its encoder keys and values as one block, computed
    once per lane. Nothing here reads back to the host, so the whole search
    can be captured as one CUDA graph (:class:`SearchGraphs`).
    """
    vectors, masks = inputs.vectors, inputs.masks
    device = vectors.device
    count, length, emittable = masks.shape
    width = inputs.width
    rows = count * width
    everything = torch.arange(emittable, device=device)
    lane_of_row = torch.arange(count, device=device).repeat_interleave(width)
    # Contiguous once here, or every step's attention copies them into shape.
    encoder_kv = [
        (keys.contiguous(), values.contiguous())
        for keys, values in (cross.kv(vectors) for cross in decoder.gated_layers())
    ]
    lane_lengths = inputs.lane_lengths
    encoder_mask = (
        torch.arange(length, device=device) < lane_lengths.unsqueeze(1) if inputs.ragged else None
    )
    # The step an ended row takes: emission 0 for free, everything else barred.
    carried = torch.full((emittable,), float("-inf"), device=device)
    carried[:1].fill_(0.0)
    layers, heads = decoder.reader.config.layers, decoder.reader.config.heads
    head_dim = decoder.reader.config.hidden // heads
    # The cache holds the whole path from the start, in the dtype the steps
    # compute in: under autocast a float32 cache would hold every key at
    # twice the bytes for the same values.
    cached = (
        torch.get_autocast_dtype(device.type)
        if torch.is_autocast_enabled(device.type)
        else decoder.reader.embed.weight.dtype
    )
    keys = torch.zeros(layers, rows, heads, length, head_dim, dtype=cached, device=device)
    values = torch.zeros_like(keys)
    scores = torch.full((count, width), float("-inf"), device=device)
    scores[:, :1].fill_(0.0)
    tokens = torch.full((rows,), BOS, dtype=torch.long, device=device)
    block = torch.arange(count, device=device).unsqueeze(1) * width
    positions = torch.arange(length, device=device)
    parents: list[torch.Tensor] = []
    choices: list[torch.Tensor] = []
    # Rows, length and span count change with every search; marked here, a
    # compiled advance's first graph is the dynamic one.
    for tensor in (lane_of_row, tokens, scores, block, lane_lengths, vectors, masks):
        torch._dynamo.maybe_mark_dynamic(tensor, 0)
    for tensor in (vectors, masks):
        torch._dynamo.maybe_mark_dynamic(tensor, 1)
    for tensor in (keys, values):
        torch._dynamo.maybe_mark_dynamic(tensor, (1, 3))
    for pair in encoder_kv:
        for part in pair:
            torch._dynamo.maybe_mark_dynamic(part, (0, 2))
    for position in range(length):
        tokens, scores, (keys, values), parent, choice = decoder.advance(
            positions[position],
            tokens,
            scores,
            (keys, values),
            vectors,
            masks,
            lane_of_row,
            encoder_kv,
            encoder_mask,
            lane_lengths,
            carried,
            everything,
            block,
        )
        parents.append(parent)
        choices.append(choice)
    return torch.stack(parents), torch.stack(choices), scores


class SearchGraphs:
    """Beam searches captured as CUDA graphs, one a bucketed shape, replayed.

    A search position is some four hundred kernels, and issuing them one by
    one from the host is what bounds the ranking step on a fast card (#137):
    replayed from a graph, a whole search is one launch. Searches are padded
    to :func:`bucket` shapes, so a run captures a few graphs an octave of lane
    count and length. Every graph's inputs are views of one set of flat
    buffers, grown when a larger shape arrives (which drops the graphs that
    baked the old addresses), and every graph's working memory comes from one
    shared pool: searches never overlap, so neither costs more than the
    largest search alone.
    """

    def __init__(self) -> None:
        self.pool = torch.cuda.graph_pool_handle()
        self.flat: dict[str, torch.Tensor] = {}
        self.captured: dict[
            tuple[object, ...],
            tuple[torch.cuda.CUDAGraph, SearchInputs, tuple[torch.Tensor, ...]],
        ] = {}

    def _view(self, name: str, like: torch.Tensor) -> torch.Tensor:
        """A view shaped like *like* into the flat buffer *name*, grown when too small."""
        flat = self.flat.get(name)
        if flat is None or flat.numel() < like.numel() or flat.dtype != like.dtype:
            self.flat[name] = torch.empty(
                max(like.numel(), 0 if flat is None else flat.numel()),
                dtype=like.dtype,
                device=like.device,
            )
            self.captured.clear()
            flat = self.flat[name]
        return flat[: like.numel()].view(like.shape)

    def run(
        self, decoder: Decoder, inputs: SearchInputs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """:func:`run_search` over *inputs*, from this shape's captured graph."""
        autocast = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else None
        key = (inputs.masks.shape, inputs.vectors.dtype, inputs.ragged, inputs.width, autocast)
        given = (inputs.vectors, inputs.masks, inputs.lane_lengths)
        captured = self.captured.get(key)
        if captured is None:
            vectors, masks, lane_lengths = (
                self._view(name, tensor)
                for name, tensor in zip(("vectors", "masks", "lane_lengths"), given, strict=True)
            )
            static = SearchInputs(
                vectors, masks, lane_lengths, ragged=inputs.ragged, width=inputs.width
            )
        else:
            static = captured[1]
        for view, tensor in zip(
            (static.vectors, static.masks, static.lane_lengths), given, strict=True
        ):
            view.copy_(tensor)
        if captured is None:
            captured = self.captured[key] = self._capture(decoder, static, autocast)
        graph, _, outputs = captured
        graph.replay()
        parents, choices, scores = outputs
        return parents, choices, scores

    def _capture(
        self, decoder: Decoder, static: SearchInputs, autocast: torch.dtype | None
    ) -> tuple[torch.cuda.CUDAGraph, SearchInputs, tuple[torch.Tensor, ...]]:
        """Warm the search up on a side stream, then capture it over *static*.

        Autocast's weight-cast cache would hand the graph casts made outside
        it, so the casts are taken inside, uncached, like every other kernel.
        """

        def search() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if autocast is None:
                return run_search(decoder, static)
            with torch.autocast("cuda", dtype=autocast, cache_enabled=False):
                return run_search(decoder, static)

        side = torch.cuda.Stream()  # type: ignore[no-untyped-call]  # torch ships it untyped
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            search()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            outputs = search()
        return graph, static, outputs


def decode_lanes(
    model: E2EModel,
    lanes: Sequence[Lane],
    beam_width: int,
    *,
    graphs: SearchGraphs | None = None,
) -> list[list[tuple[list[int], float]]]:
    """Beam-search lanes together; each lane's hypotheses and scores.

    Each step's scores are log-probabilities over the full emittable row, and
    the candidate restriction masks the characters the position does not
    admit afterwards, so a path's score is comparable with another
    segmentation's. Rows still at minus infinity when the path ends are slots
    the lattice never filled and are dropped, so a lane admitting fewer than
    ``beam_width`` sentences returns fewer.

    Lanes may differ in length. A shorter lane's span vectors are padded and
    masked out of the cross-attention, and once it has ended each of its rows
    continues with a single free step, so its beam is carried unchanged to the
    last position and cut back to its own length on the way out. Lanes of one
    length -- what the evaluation's buckets hold -- take no mask and no
    padding, so their arithmetic is the equal-length search's exactly.

    With *graphs* the search is padded to bucketed shapes and replayed from
    the graph captured for its shape.
    """
    decoder = model.decoder
    inputs = search_inputs(lanes, beam_width, decoder.emit_ids.device, bucketed=graphs is not None)
    if graphs is None:
        parents, choices, scores = run_search(decoder, inputs)
    else:
        parents, choices, scores = graphs.run(decoder, inputs)
    return walk_back(parents, choices, scores, lanes)


def walk_back(
    parents: torch.Tensor,
    choices: torch.Tensor,
    scores: torch.Tensor,
    lanes: Sequence[Lane],
) -> list[list[tuple[list[int], float]]]:
    """Each lane's hypotheses, read back from a search's parents and choices.

    Lanes past ``lanes`` -- a bucket's padding -- are not read; each
    sequence is cut back to its own lane's length.
    """
    # Plain lists: the walk back reads one element at a time, which a tensor
    # charges a dispatch for.
    parent_steps: list[list[list[int]]] = parents.tolist()
    choice_steps: list[list[list[int]]] = choices.tolist()
    final: list[list[float]] = scores.tolist()
    length = len(parent_steps)
    decoded: list[list[tuple[list[int], float]]] = []
    for lane, own in enumerate(lane.length for lane in lanes):
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
            hypotheses.append((sequence[:own], score))
        decoded.append(hypotheses)
    return decoded
