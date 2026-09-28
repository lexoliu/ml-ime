"""The character language models the decoder's transition can run.

Two architectures behind one interface. The recurrent one carries a fixed
vector per beam; the attention one carries the keys and values of everything
it has read, which is where its strength on abbreviated input comes from and
what makes its state grow with the sentence. The decoder does not care which
it is: a model reads the prelude (``<bos> context <sep>``) once through its
*prefill* graph, whose outputs are the state every beam starts from, and then
advances a batch of beams by one character through its *step* graph.

The state is split in two so the beams do not each carry the context. The
*prefix* tensors are what the prelude produced and are shared by every beam
of a record (the transformer's key/value cache over the context); the *state*
tensors are per beam (the LSTM's hidden and cell, the transformer's cache over
the sentence so far). A step's batch is a rectangle: the rows are laid out
``[workers, width]``, every worker owning exactly ``width`` consecutive rows
under one prefix row, so the prefix is fed once per worker and broadcast over
the width axis rather than packed per row. Every tensor is batch-first, so the
Rust side stacks beams by rows without knowing what the rows hold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import nn

from mlime.train.charlm_vocab import PAD

Arch = Literal["lstm", "transformer"]

#: Characters of context a sequence keeps, the end nearest the text. Matches the
#: context tower's window (64 tokens with its two sentinels).
DEFAULT_CONTEXT_CHARS = 62


@dataclass(frozen=True)
class CharLmConfig:
    """The model's shape."""

    arch: Arch = "lstm"
    embedding: int = 384
    hidden: int = 1024
    layers: int = 2
    #: Attention heads; the transformer only.
    heads: int = 8
    #: Width of the feed-forward block; the transformer only.
    feedforward: int = 2048
    #: Positions the transformer has embeddings for: the prelude plus the sentence.
    max_positions: int = 512
    dropout: float = 0.1
    context_chars: int = DEFAULT_CONTEXT_CHARS

    def __post_init__(self) -> None:
        for name in ("embedding", "hidden", "layers", "heads", "feedforward", "max_positions"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.arch == "transformer" and self.hidden % self.heads:
            raise ValueError(f"hidden {self.hidden} does not split over {self.heads} heads")


class CharLm(nn.Module):
    """What both architectures share: the tied embedding and the interface.

    ``forward`` scores a whole batch of sequences for training. ``prefill``
    reads a prelude and returns the features at its last position, the prefix
    tensors and the per-beam state tensors; ``step`` takes one token per row
    with the prefix and state and returns the features and the new state. The
    features go through :meth:`logits` to become scores over the alphabet.
    """

    def __init__(self, vocab_size: int, config: CharLmConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(vocab_size, config.embedding, padding_idx=PAD)
        self.project = nn.Linear(config.hidden, config.embedding)
        self.dropout = nn.Dropout(config.dropout)

    @property
    def prefix_names(self) -> tuple[str, ...]:
        """Names of the shared tensors ``prefill`` produces, in order."""
        raise NotImplementedError

    @property
    def state_names(self) -> tuple[str, ...]:
        """Names of the per-beam tensors, in order."""
        raise NotImplementedError

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        """Scores over the alphabet from features, through the tied embedding."""
        scores: torch.Tensor = self.project(self.dropout(features)) @ self.embed.weight.T
        return scores

    def features(self, tokens: torch.Tensor) -> torch.Tensor:
        """Hidden features for the token after each position of ``tokens`` (``[B, T, H]``)."""
        raise NotImplementedError

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Logits for the token after each position of ``tokens`` (``[B, T, V]``)."""
        return self.logits(self.features(tokens))

    def prefill(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Read ``tokens [B, T]``: features at the last position, prefix tensors, state tensors."""
        raise NotImplementedError

    def step(
        self,
        token: torch.Tensor,
        prefix: tuple[torch.Tensor, ...],
        state: tuple[torch.Tensor, ...],
        prefix_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Advance ``token [workers * width]`` on ``state``: features and the new state.

        The batch is rectangular: ``workers`` blocks of ``width`` consecutive
        rows each, one block per record under its own prefix row. A prefix
        tensor is ``[workers, ...]`` -- one row per block, padded on the left
        up to the widest prelude, with ``prefix_mask [workers, T]`` marking
        the real positions (``None`` reads as all true) -- and it broadcasts
        over the ``width`` rows under it. A model without prefix tensors takes
        a flat batch and ignores the mask.
        """
        raise NotImplementedError


class LstmCharLm(CharLm):
    """Embedding, LSTM stack, projection back to the embedding, tied output."""

    def __init__(self, vocab_size: int, config: CharLmConfig):
        super().__init__(vocab_size, config)
        self.lstm = nn.LSTM(
            config.embedding,
            config.hidden,
            num_layers=config.layers,
            batch_first=True,
            dropout=config.dropout if config.layers > 1 else 0.0,
        )

    @property
    def prefix_names(self) -> tuple[str, ...]:
        return ()

    @property
    def state_names(self) -> tuple[str, ...]:
        return ("hidden", "cell")

    def features(self, tokens: torch.Tensor) -> torch.Tensor:
        features: torch.Tensor
        features, _ = self.lstm(self.dropout(self.embed(tokens)))
        return features

    def prefill(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        features, (hidden, cell) = self.lstm(self.embed(tokens))
        # The LSTM's states are [layers, B, hidden]; the interface is batch-first.
        return features[:, -1], (), (hidden.transpose(0, 1), cell.transpose(0, 1))

    def step(
        self,
        token: torch.Tensor,
        prefix: tuple[torch.Tensor, ...],
        state: tuple[torch.Tensor, ...],
        prefix_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        hidden, cell = state
        features, (hidden, cell) = self.lstm(
            self.embed(token).unsqueeze(1),
            (hidden.transpose(0, 1).contiguous(), cell.transpose(0, 1).contiguous()),
        )
        return features.squeeze(1), (hidden.transpose(0, 1), cell.transpose(0, 1))


class Block(nn.Module):
    """One pre-norm transformer block with a key/value cache on its attention."""

    def __init__(self, config: CharLmConfig):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.hidden // config.heads
        self.norm1 = nn.LayerNorm(config.hidden)
        self.qkv = nn.Linear(config.hidden, 3 * config.hidden)
        self.out = nn.Linear(config.hidden, config.hidden)
        self.norm2 = nn.LayerNorm(config.hidden)
        self.ff = nn.Sequential(
            nn.Linear(config.hidden, config.feedforward),
            nn.GELU(),
            nn.Linear(config.feedforward, config.hidden),
        )
        self.dropout = nn.Dropout(config.dropout)

    def split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Queries, keys and values as ``[B, heads, T, head_dim]``."""
        batch, length, _ = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        shape = (batch, length, self.heads, self.head_dim)
        return (
            q.view(shape).transpose(1, 2),
            k.view(shape).transpose(1, 2),
            v.view(shape).transpose(1, 2),
        )

    def merge(self, x: torch.Tensor, attended: torch.Tensor) -> torch.Tensor:
        """The block's output from ``attended [B, heads, T, head_dim]``."""
        batch, _, length, _ = attended.shape
        x = x + self.dropout(self.out(attended.transpose(1, 2).reshape(batch, length, -1)))
        output: torch.Tensor = x + self.dropout(self.ff(self.norm2(x)))
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Causal self-attention over a whole sequence, for training."""
        q, k, v = self.split(x)
        attended = nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout.p if self.training else 0.0
        )
        return self.merge(x, attended)

    def attend(
        self,
        x: torch.Tensor,
        prefix_keys: torch.Tensor,
        prefix_values: torch.Tensor,
        prefix_mask: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One new position ``x [workers * width, 1, H]`` against the two caches and itself.

        ``prefix_keys``/``prefix_values`` hold each worker's context's
        positions -- one row per ``width``-row block of the batch, the real
        positions right-aligned under ``prefix_mask`` -- and
        ``keys``/``values`` the sentence's; scoring them separately and
        concatenating only the scores keeps the prefix cache out of the
        step's copies. The queries fold the batch back into its
        ``[workers, width]`` rectangle so the prefix row broadcasts over the
        rows under it: ``MatMul`` does that on its own where an ``Expand``
        would materialise the prefix per row. Masked prefix positions take
        ``-inf`` before the softmax, which leaves them weightless however the
        padding was filled. Written with plain matrix products so the step
        graph exports as a handful of ONNX operators. Returns the output and
        the new key/value.
        """
        q, k, v = self.split(x)
        keys = torch.cat([keys, k], dim=2)
        values = torch.cat([values, v], dim=2)
        workers = prefix_keys.shape[0]
        queries = q.view(workers, -1, self.heads, 1, self.head_dim)
        prefix_scores = (queries @ prefix_keys.unsqueeze(1).transpose(-1, -2)).masked_fill(
            prefix_mask.logical_not().view(workers, 1, 1, 1, -1),
            # A typed fill value: the python float would trace as fp32 and the
            # legacy exporter would reconcile by casting the block to float.
            torch.tensor(float("-inf"), dtype=q.dtype),
        )
        scores = torch.cat(
            [
                prefix_scores,
                (q @ keys.transpose(-1, -2)).view(workers, -1, self.heads, 1, keys.shape[2]),
            ],
            dim=-1,
        ) / torch.tensor(
            math.sqrt(self.head_dim),
            # A typed divisor: the python float would trace as fp32 and the
            # legacy exporter would reconcile by casting the block to float.
            dtype=q.dtype,
        )
        probs = torch.softmax(scores, dim=-1)
        split = prefix_keys.shape[2]
        attended = probs[..., :split] @ prefix_values.unsqueeze(1) + (
            probs[..., split:] @ values.view(workers, -1, self.heads, keys.shape[2], self.head_dim)
        )
        return self.merge(x, attended.view(q.shape)), k, v


class TransformerCharLm(CharLm):
    """Embedding plus learned positions, pre-norm blocks, tied output.

    The cache is ``[batch, layers, heads, T, head_dim]`` for keys and for
    values -- over the prelude the batch axis is the worker (the prefix,
    shared by the ``width`` rows under it) and over the sentence so far it is
    the row (per beam). A step's position is the sum of the two lengths, read
    off the tensors so the graph needs no counter.
    """

    def __init__(self, vocab_size: int, config: CharLmConfig):
        super().__init__(vocab_size, config)
        self.input = nn.Linear(config.embedding, config.hidden)
        self.positions = nn.Embedding(config.max_positions, config.hidden)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = nn.LayerNorm(config.hidden)

    def layers(self) -> list[Block]:
        """The blocks in order, typed (``ModuleList`` yields bare modules)."""
        return [cast(Block, block) for block in self.blocks]

    @property
    def prefix_names(self) -> tuple[str, ...]:
        return ("prefix_keys", "prefix_values")

    @property
    def state_names(self) -> tuple[str, ...]:
        return ("keys", "values")

    def embed_at(self, tokens: torch.Tensor, first: int | torch.Tensor) -> torch.Tensor:
        """Token embeddings plus positions counted from *first* (``[B, T, H]``).

        *first* is a scalar when every row starts at the same position, or one
        entry per row ``[B]`` when preludes of different lengths were padded
        to a common width.
        """
        span = torch.arange(tokens.shape[1], device=tokens.device)
        positions = span + first if isinstance(first, int) else span + first.unsqueeze(-1)
        embedded: torch.Tensor = self.dropout(
            self.input(self.embed(tokens)) + self.positions(positions)
        )
        return embedded

    def features(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed_at(tokens, 0)
        for block in self.layers():
            x = block(x)
        normed: torch.Tensor = self.norm(x)
        return normed

    def prefill(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        x = self.embed_at(tokens, 0)
        keys, values = [], []
        for block in self.layers():
            q, k, v = block.split(x)
            # The same arithmetic as ``scaled_dot_product_attention`` spelled
            # out: the fused operator's internals trace scalar constants as
            # fp32, which spoils an fp16 export.
            scores = (q @ k.transpose(-1, -2)) / torch.tensor(
                math.sqrt(block.head_dim), dtype=q.dtype
            )
            length = q.shape[2]
            causal = torch.triu(
                torch.ones(length, length, dtype=torch.bool, device=q.device), diagonal=1
            )
            scores = scores.masked_fill(causal, torch.tensor(float("-inf"), dtype=q.dtype))
            attended = torch.softmax(scores, dim=-1) @ v
            x = block.merge(x, attended)
            keys.append(k)
            values.append(v)
        prefix = (torch.stack(keys, dim=1), torch.stack(values, dim=1))
        batch = tokens.shape[0]
        empty = torch.zeros(
            (
                batch,
                self.config.layers,
                self.config.heads,
                0,
                self.config.hidden // self.config.heads,
            ),
            device=tokens.device,
            dtype=x.dtype,
        )
        last: torch.Tensor = self.norm(x)[:, -1]
        return last, prefix, (empty, empty.clone())

    def step(
        self,
        token: torch.Tensor,
        prefix: tuple[torch.Tensor, ...],
        state: tuple[torch.Tensor, ...],
        prefix_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        prefix_keys, prefix_values = prefix
        keys, values = state
        if prefix_mask is None:
            prefix_mask = torch.ones(
                (prefix_keys.shape[0], prefix_keys.shape[3]),
                dtype=torch.bool,
                device=token.device,
            )
        # A row's next position is its worker's prelude's real length plus
        # the sentence's so far, so the mask's count is what ``embed_at``
        # counts from and padded prelude positions hold no position id either.
        # The mask counts per worker; every row under it counts from the same
        # position.
        workers = prefix_mask.shape[0]
        first = (prefix_mask.sum(-1) + keys.shape[3]).unsqueeze(-1)
        first = first.expand(workers, token.shape[0] // workers).reshape(-1)
        x = self.embed_at(token.unsqueeze(1), first)
        new_keys, new_values = [], []
        for layer, block in enumerate(self.layers()):
            x, k, v = block.attend(
                x,
                prefix_keys[:, layer],
                prefix_values[:, layer],
                prefix_mask,
                keys[:, layer],
                values[:, layer],
            )
            new_keys.append(k)
            new_values.append(v)
        keys = torch.cat([keys, torch.stack(new_keys, dim=1)], dim=3)
        values = torch.cat([values, torch.stack(new_values, dim=1)], dim=3)
        features: torch.Tensor = self.norm(x).squeeze(1)
        return features, (keys, values)


def build(vocab_size: int, config: CharLmConfig) -> CharLm:
    """The model *config* describes."""
    if config.arch == "lstm":
        return LstmCharLm(vocab_size, config)
    return TransformerCharLm(vocab_size, config)


class Restricted(nn.Module):
    """Log probabilities from features, over the whole alphabet or a kept part of it.

    With *keep*, the next-character distribution is taken over those ids only
    -- the probability of each conditioned on the next character being one of
    them -- and every other id is scored at ``UNREACHABLE``. The decoder only
    ever proposes characters the lattice can emit, so restricting the alphabet
    to them loses nothing and cuts the output projection to their share of it.
    """

    #: The log probability written for an id outside the kept alphabet: far
    #: below any real score, finite so sums never turn into NaN.
    UNREACHABLE = -1.0e9

    def __init__(self, model: CharLm, keep: torch.Tensor | None):
        super().__init__()
        self.model = model
        self.keep: torch.Tensor | None
        if keep is None:
            self.keep = None
        else:
            self.register_buffer("keep", keep)
            self.register_buffer("kept_weight", model.embed.weight.detach()[keep])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.keep is None:
            return torch.log_softmax(self.model.logits(features).float(), dim=-1)
        scores = self.model.project(features) @ self.kept_weight.T
        kept = torch.log_softmax(scores.float(), dim=-1)
        batch = features.shape[0]
        full = torch.full((batch, self.model.embed.num_embeddings), self.UNREACHABLE)
        index = self.keep.unsqueeze(0).expand(batch, -1)
        return full.scatter(1, index, kept)


class ResidentStepModule(nn.Module):
    """The step graph with the beam state resident beside the session.

    Between runs the state tensors live wherever the session's allocator
    puts them -- the device for a GPU backend, the heap for the CPU -- and
    the host passes only which row each new beam continues from:

    ``token [rows]`` is each beam's next character, ``source_row [rows]``
    its index into the state buffers, and ``candidates [rows, K]`` the
    alphabet ids the produced state will be scored on at its next position
    (``<eos>`` included where the path can end there, a throwaway id as
    padding). ``keys``/``values`` (the transformer) or ``hidden``/``cell``
    (the LSTM) are the resident state tensors the batch gathers, laid out
    ``[rows, layers, heads, time, head_dim]``; the prefix buffers take one
    slot per worker, ``prefix_row [workers]`` naming which slot feeds each
    ``width``-row block, and ``prefix_mask`` ``[slots, T]`` marks the real
    positions. The output ``candidate_log_probs [rows, K]`` is the
    log-softmax gathered at each row's candidates -- the only tensor that
    crosses the bus -- and ``next_*`` are the rows the caller binds over
    fresh buffers -- read through them, the old buffers release when their
    last beam moves on, so nothing ever writes into a row a live state
    still reads.
    """

    def __init__(self, model: CharLm, keep: torch.Tensor | None = None):
        super().__init__()
        self.model = model
        self.restricted = Restricted(model, keep)

    def forward(
        self,
        token: torch.Tensor,
        source_row: torch.Tensor,
        candidates: torch.Tensor,
        *tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        split = len(self.model.prefix_names)
        prefix_buffers, rest = tensors[:split], tensors[split:]
        if split:
            prefix_row, prefix_mask = rest[0], rest[1]
            state_buffers = rest[2:]
            prefix = tuple(buffer.index_select(0, prefix_row) for buffer in prefix_buffers)
            mask = prefix_mask.index_select(0, prefix_row)
        else:
            prefix, mask, state_buffers = (), None, rest
        state = tuple(buffer.index_select(0, source_row) for buffer in state_buffers)
        features, state = self.model.step(token, prefix, state, mask)
        return (self.restricted(features).gather(1, candidates), *state)


class PrefillModule(nn.Module):
    """The prefill graph, emitted in the resident layout.

    ``tokens [1, T], candidates [1, K] -> (candidate_log_probs, *prefix,
    prefix_mask, *state)``. A prefix row lands directly in its resident
    slot, whose buffers are shaped to the widest prelude -- ``<bos>`` +
    ``context_chars`` + ``<sep>`` -- so each prefix tensor is left-padded
    to that width on its time axis and the mask row the step graph reads
    is emitted with it; the step binds the graph's outputs over the
    slot's rows without a copy. ``candidates`` is the first position's
    candidate ids and ``candidate_log_probs [1, K]`` the log-softmax
    gathered at them. A model without prefix tensors (the LSTM) emits
    ``(candidate_log_probs, *state)``.
    """

    def __init__(self, model: CharLm, keep: torch.Tensor | None = None):
        super().__init__()
        self.model = model
        self.restricted = Restricted(model, keep)

    def forward(self, tokens: torch.Tensor, candidates: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features, prefix, state = self.model.prefill(tokens)
        gathered = self.restricted(features).gather(1, candidates)
        if not prefix:
            return (gathered, *state)
        width = self.model.config.context_chars + 2
        pad = width - tokens.shape[1]
        padded = tuple(
            torch.cat(
                [
                    tensor.new_zeros(
                        tensor.shape[0],
                        tensor.shape[1],
                        tensor.shape[2],
                        pad,
                        tensor.shape[4],
                    ),
                    tensor,
                ],
                dim=3,
            )
            for tensor in prefix
        )
        mask = torch.arange(width, device=tokens.device).unsqueeze(0) >= pad
        return (gathered, *padded, mask, *state)
