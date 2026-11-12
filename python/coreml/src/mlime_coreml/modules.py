"""The three function families as plain-torch modules, in the form Core ML converts.

Each module is the key-level model's arithmetic rebuilt from the bundle's weights, with the
choices the Neural Engine needs made once:

* attention is spelled out (matrix product, additive mask, softmax, matrix product), never the
  fused op, which does not compile for the Neural Engine;
* every mask is an additive :data:`~mlime_coreml.spec.NEG`, so a fully masked row stays finite,
  and every query row keeps at least one unmasked key;
* every shape is static, the batch is 1, and tables that depend on the shape (the causal mask,
  the head's concatenated weights) are buffers that bake into the graph;
* ids are int32 and masks are floats.

The parameter names are the bundle's own, so a family's tensors load with ``strict=True``.
Nothing here imports mlime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from mlime_coreml import spec
from mlime_coreml.bundle import Bundle, DecoderDims, KeyFrontDims, Manifest, TowerDims
from mlime_coreml.spec import Family


def split_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    """``[1, L, heads * d]`` to ``[1, heads, L, d]``."""
    return x.view(1, x.shape[1], heads, -1).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """``[1, heads, L, d]`` to ``[1, L, heads * d]``."""
    return x.transpose(1, 2).reshape(1, x.shape[2], -1)


def attend(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    """Attention spelled out: scaled scores, the additive *bias*, softmax, weighted values.

    The queries are scaled before the product, not the scores after it: the same value, and the
    product's float16 range is the smaller.
    """
    scores = (q * q.shape[-1] ** -0.5) @ k.transpose(-1, -2)
    if bias is not None:
        scores = scores + bias
    return cast(torch.Tensor, scores.softmax(dim=-1) @ v)


def padding_bias(mask: torch.Tensor) -> torch.Tensor:
    """``[1, L]`` mask (1 real, 0 padding) to the additive ``[1, 1, 1, L]`` bias."""
    return ((1.0 - mask) * spec.NEG)[:, None, None, :]


# --- BERT towers (the context tower and the fill tower) ---------------------------------------


class Embeddings(nn.Module):
    """BERT's embeddings: word, position and token type 0, then LayerNorm."""

    def __init__(self, dims: TowerDims):
        super().__init__()
        self.word_embeddings = nn.Embedding(dims.vocabulary, dims.hidden)
        self.position_embeddings = nn.Embedding(dims.positions, dims.hidden)
        self.token_type_embeddings = nn.Embedding(2, dims.hidden)
        self.LayerNorm = nn.LayerNorm(dims.hidden, eps=dims.layer_norm_eps)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """``[1, L, H]`` word vectors, summed with the positions ``0..L-1`` and type 0."""
        length = inputs_embeds.shape[1]
        summed = (
            inputs_embeds
            + self.position_embeddings.weight[:length]
            + self.token_type_embeddings.weight[0]
        )
        return cast(torch.Tensor, self.LayerNorm(summed))


class SelfAttention(nn.Module):
    """BERT's query, key and value projections."""

    def __init__(self, hidden: int):
        super().__init__()
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)


class AttentionOutput(nn.Module):
    """BERT's attention output: a projection and the residual LayerNorm."""

    def __init__(self, hidden: int, eps: float):
        super().__init__()
        self.dense = nn.Linear(hidden, hidden)
        self.LayerNorm = nn.LayerNorm(hidden, eps=eps)


class Attention(nn.Module):
    """``self`` and ``output`` of one BERT layer, named as the bundle's tensors are."""

    def __init__(self, dims: TowerDims):
        super().__init__()
        self.self = SelfAttention(dims.hidden)
        self.output = AttentionOutput(dims.hidden, dims.layer_norm_eps)


class Intermediate(nn.Module):
    """The feed-forward's first projection."""

    def __init__(self, dims: TowerDims):
        super().__init__()
        self.dense = nn.Linear(dims.hidden, dims.intermediate)


class Output(nn.Module):
    """The feed-forward's second projection and the residual LayerNorm."""

    def __init__(self, dims: TowerDims):
        super().__init__()
        self.dense = nn.Linear(dims.intermediate, dims.hidden)
        self.LayerNorm = nn.LayerNorm(dims.hidden, eps=dims.layer_norm_eps)


class BertLayer(nn.Module):
    """One post-norm BERT layer under an additive attention bias."""

    def __init__(self, dims: TowerDims):
        super().__init__()
        self.heads = dims.heads
        self.attention = Attention(dims)
        self.intermediate = Intermediate(dims)
        self.output = Output(dims)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """``[1, L, H]`` and the ``[1, 1, 1, L]`` bias to ``[1, L, H]``."""
        own = self.attention.self
        attended = merge_heads(
            attend(
                split_heads(own.query(x), self.heads),
                split_heads(own.key(x), self.heads),
                split_heads(own.value(x), self.heads),
                bias,
            )
        )
        x = self.attention.output.LayerNorm(self.attention.output.dense(attended) + x)
        lifted = nn.functional.gelu(self.intermediate.dense(x))
        return cast(torch.Tensor, self.output.LayerNorm(self.output.dense(lifted) + x))


class Encoder(nn.Module):
    """A BERT tower's layers."""

    def __init__(self, dims: TowerDims):
        super().__init__()
        self.layer = nn.ModuleList(BertLayer(dims) for _ in range(dims.layers))


class ContextAttention(nn.Module):
    """``nn.MultiheadAttention``'s parameters, read as a plain attention into same-width rows."""

    def __init__(self, hidden: int):
        super().__init__()
        self.in_proj_weight = nn.Parameter(torch.zeros(3 * hidden, hidden))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * hidden))
        self.out_proj = nn.Linear(hidden, hidden)


class GatedContext(nn.Module):
    """A fill layer's gated attention into the context rows, switched by whether any is real."""

    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(hidden, eps=spec.DEFAULT_LAYER_NORM_EPS)
        self.attention = ContextAttention(hidden)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(
        self, x: torch.Tensor, context: torch.Tensor, bias: torch.Tensor, present: torch.Tensor
    ) -> torch.Tensor:
        """*x* ``[1, W, H]`` plus the gated term read from *context* ``[1, C, H]``.

        *present* ``[1, 1, 1]`` is 1 when the context has a real token and 0 when none, which
        multiplies the term to exactly zero (a context of none attends uniformly, finitely).
        """
        wq, wk, wv = self.attention.in_proj_weight.chunk(3)
        bq, bk, bv = self.attention.in_proj_bias.chunk(3)
        attended = merge_heads(
            attend(
                split_heads(nn.functional.linear(self.norm(x), wq, bq), self.heads),
                split_heads(nn.functional.linear(context, wk, bk), self.heads),
                split_heads(nn.functional.linear(context, wv, bv), self.heads),
                bias,
            )
        )
        return cast(torch.Tensor, x + self.gate * present * self.attention.out_proj(attended))


class BertTower(nn.Module):
    """Embeddings and layers: the context tower, or the fill tower (hooked at its top layers)."""

    def __init__(self, dims: TowerDims):
        super().__init__()
        self.embeddings = Embeddings(dims)
        self.encoder = Encoder(dims)


@dataclass(frozen=True)
class ContextRead:
    """What the fill tower's top layers read of the context: blocks, rows and masks."""

    gates: nn.ModuleList
    context: torch.Tensor
    bias: torch.Tensor
    present: torch.Tensor


def run_tower(
    tower: BertTower,
    inputs_embeds: torch.Tensor,
    bias: torch.Tensor,
    read: ContextRead | None = None,
) -> torch.Tensor:
    """*tower* over ``[1, L, H]`` vectors; with *read*, each top layer is followed by a gate."""
    x: torch.Tensor = tower.embeddings(inputs_embeds)
    layers = list(tower.encoder.layer)
    first_gated = len(layers) - (0 if read is None else len(read.gates))
    for depth, layer in enumerate(layers):
        x = layer(x, bias)
        if read is not None and depth >= first_gated:
            x = read.gates[depth - first_gated](x, read.context, read.bias, read.present)
    return x


# --- the context family -----------------------------------------------------------------------


class ContextEncoder(nn.Module):
    """``ctx_L``: the context tower over ``[1, L]`` token ids, once per context text."""

    def __init__(self, dims: TowerDims):
        super().__init__()
        self.context = BertTower(dims)

    def forward(self, context_ids: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
        """``[1, L]`` ids (int32) and mask (1 real) to the ``[1, L, H]`` context rows.

        A padded position's row is whatever the tower computed there; the fill tower's
        cross-attention masks it with the same mask.
        """
        words = self.context.embeddings.word_embeddings(context_ids)
        return run_tower(self.context, words, padding_bias(context_mask))


# --- the keys family --------------------------------------------------------------------------


class KeyBlock(nn.Module):
    """One causal pre-norm block of the key front."""

    def __init__(self, width: int, heads: int, feedforward: int):
        super().__init__()
        self.heads = heads
        self.attention_norm = nn.LayerNorm(width, eps=spec.DEFAULT_LAYER_NORM_EPS)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.feedforward_norm = nn.LayerNorm(width, eps=spec.DEFAULT_LAYER_NORM_EPS)
        self.up = nn.Linear(width, feedforward)
        self.down = nn.Linear(feedforward, width)

    def forward(self, x: torch.Tensor, causal: torch.Tensor) -> torch.Tensor:
        """``[1, W, H]`` under the baked causal bias ``[W, W]``."""
        q, k, v = self.qkv(self.attention_norm(x)).chunk(3, dim=-1)
        attended = merge_heads(
            attend(
                split_heads(q, self.heads),
                split_heads(k, self.heads),
                split_heads(v, self.heads),
                causal,
            )
        )
        x = x + self.out(attended)
        return cast(
            torch.Tensor, x + self.down(nn.functional.gelu(self.up(self.feedforward_norm(x))))
        )


class KeyFront(nn.Module):
    """Letters (with their QWERTY coordinates folded in) plus positions, then causal blocks."""

    def __init__(self, dims: KeyFrontDims, width: int, hidden: int, letters: int):
        super().__init__()
        self.letter_table = nn.Parameter(torch.zeros(letters, hidden))
        self.positions = nn.Embedding(dims.positions, hidden)
        self.blocks = nn.ModuleList(
            KeyBlock(hidden, dims.heads, dims.feedforward) for _ in range(dims.layers)
        )
        causal = torch.triu(torch.full((width, width), spec.NEG), diagonal=1)
        self.register_buffer("causal", causal, persistent=False)

    def forward(self, letters: torch.Tensor) -> torch.Tensor:
        """``[1, W]`` letter ids to ``[1, W, H]``."""
        x = (
            nn.functional.embedding(letters, self.letter_table)
            + self.positions.weight[: letters.shape[1]]
        )
        for block in self.blocks:
            x = block(x, self.causal)
        return x


class Transform(nn.Module):
    """The MLM head's transform: a projection, GELU and LayerNorm."""

    def __init__(self, hidden: int, eps: float):
        super().__init__()
        self.dense = nn.Linear(hidden, hidden)
        self.LayerNorm = nn.LayerNorm(hidden, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[1, W, H]`` states to the transformed ``[1, W, H]``."""
        return cast(torch.Tensor, self.LayerNorm(nn.functional.gelu(self.dense(x))))


class Head(nn.Module):
    """The restricted MLM head's transform and its output rows."""

    def __init__(self, hidden: int, emitted: int, eps: float):
        super().__init__()
        self.transform = Transform(hidden, eps)
        self.decoder = nn.Linear(hidden, emitted)


class KeyValue(nn.Module):
    """One decoder block's key and value projections of the key states."""

    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.key = nn.Linear(width, hidden)
        self.value = nn.Linear(width, hidden)


class KeyValueStack(nn.Module):
    """The decoder's cross-attention projections, as far as the keys function reads them."""

    def __init__(self, layers: int, width: int, hidden: int):
        super().__init__()
        self.cross_attention = nn.ModuleList(KeyValue(width, hidden) for _ in range(layers))


class KeysGraph(nn.Module):
    """``keys_W``: one keystroke row to the CTC logits and the decoder's cross keys and values."""

    logit_weight: torch.Tensor
    logit_bias: torch.Tensor
    key_weight: torch.Tensor
    key_bias: torch.Tensor
    value_weight: torch.Tensor
    value_bias: torch.Tensor

    def __init__(self, manifest: Manifest, width: int):
        super().__init__()
        dims = manifest.dims
        hidden = dims.fill.hidden
        self.fill = BertTower(dims.fill)
        self.cross_attention = nn.ModuleList(
            GatedContext(hidden, dims.fill.heads) for _ in range(dims.gated)
        )
        self.key_front = KeyFront(dims.key_front, width, hidden, manifest.letter_pad + 1)
        self.head = Head(hidden, dims.emitted, dims.fill.layer_norm_eps)
        self.blank = nn.Linear(hidden, 1)
        self.decoder = KeyValueStack(dims.decoder.layers, hidden, dims.decoder.hidden)
        self.letter_pad = manifest.letter_pad
        self.layer_count = dims.decoder.layers
        self.decoder_heads = dims.decoder.heads
        self.slots = spec.N_MAX

    def bake(self) -> None:
        """Derive the graph's constants: the fused logits head and the stacked projections."""
        projections = [cast(KeyValue, layer) for layer in self.decoder.cross_attention]
        derived = {
            "logit_weight": torch.cat([self.head.decoder.weight, self.blank.weight]),
            "logit_bias": torch.cat([self.head.decoder.bias, self.blank.bias]),
            "key_weight": torch.cat([layer.key.weight for layer in projections]),
            "key_bias": torch.cat([layer.key.bias for layer in projections]),
            "value_weight": torch.cat([layer.value.weight for layer in projections]),
            "value_bias": torch.cat([layer.value.bias for layer in projections]),
        }
        for name, value in derived.items():
            self.register_buffer(name, value.detach().clone(), persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        key_letters: torch.Tensor,
        attention_mask: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The keys function.

        ``input_ids`` is ``[CLS] [MASK]*n [SEP] [PAD]*``, ``key_letters`` the key's letter at
        each ``[MASK]`` and the pad id elsewhere, ``attention_mask`` 1 through ``[SEP]``;
        ``context`` and ``context_mask`` are the context function's output and mask. Returns the
        CTC logits ``[1, W, E + 1]`` (blank last; row ``t`` is key ``t``'s at ``1 <= t <= n``) and
        the cross keys and values ``[layers, heads, N_MAX, head_dim]``, key ``j`` at slot ``j``
        and zeros after the real keys.
        """
        at_key = (key_letters != self.letter_pad).to(attention_mask.dtype).unsqueeze(-1)
        words = self.fill.embeddings.word_embeddings(input_ids)
        inputs = words + self.key_front(key_letters) * at_key
        present = context_mask.amax(dim=-1, keepdim=True).unsqueeze(-1)
        read = ContextRead(self.cross_attention, context, padding_bias(context_mask), present)
        states = run_tower(self.fill, inputs, padding_bias(attention_mask), read)
        transformed = self.head.transform(states)
        logits = nn.functional.linear(transformed, self.logit_weight, self.logit_bias)
        real = states[:, 1 : states.shape[1] - 1]
        keep = at_key[:, 1 : states.shape[1] - 1]
        keys = self._slots(nn.functional.linear(real, self.key_weight, self.key_bias) * keep)
        values = self._slots(nn.functional.linear(real, self.value_weight, self.value_bias) * keep)
        return logits, keys, values

    def _slots(self, projected: torch.Tensor) -> torch.Tensor:
        """``[1, K, layers * hidden]`` to ``[layers, heads, N_MAX, head_dim]``, zero-padded."""
        count = projected.shape[1]
        shaped = projected.view(count, self.layer_count * self.decoder_heads, -1).transpose(0, 1)
        padded = nn.functional.pad(shaped, (0, 0, 0, self.slots - count))
        return padded.view(self.layer_count, self.decoder_heads, self.slots, -1)


# --- the tree family --------------------------------------------------------------------------


class ReaderBlock(nn.Module):
    """One pre-norm block of the char LM reader."""

    def __init__(self, dims: DecoderDims):
        super().__init__()
        self.heads = dims.heads
        self.norm1 = nn.LayerNorm(dims.hidden, eps=dims.layer_norm_eps)
        self.qkv = nn.Linear(dims.hidden, 3 * dims.hidden)
        self.out = nn.Linear(dims.hidden, dims.hidden)
        self.norm2 = nn.LayerNorm(dims.hidden, eps=dims.layer_norm_eps)
        self.ff = nn.Sequential(
            nn.Linear(dims.hidden, dims.feedforward),
            nn.GELU(),
            nn.Linear(dims.feedforward, dims.hidden),
        )

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """Self-attention under the tree's additive bias ``[1, 1, S, S]``, then the feed-forward."""
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        attended = merge_heads(
            attend(
                split_heads(q, self.heads),
                split_heads(k, self.heads),
                split_heads(v, self.heads),
                bias,
            )
        )
        x = x + self.out(attended)
        return x + cast(torch.Tensor, self.ff(self.norm2(x)))


class Reader(nn.Module):
    """The char LM reader: embeddings, positions, blocks, the final norm and the tied head."""

    def __init__(self, dims: DecoderDims):
        super().__init__()
        self.embed = nn.Embedding(dims.vocabulary, dims.embedding)
        self.input = nn.Linear(dims.embedding, dims.hidden)
        self.positions = nn.Embedding(dims.positions, dims.hidden)
        self.blocks = nn.ModuleList(ReaderBlock(dims) for _ in range(dims.layers))
        self.norm = nn.LayerNorm(dims.hidden, eps=dims.layer_norm_eps)
        self.project = nn.Linear(dims.hidden, dims.embedding)


class TreeCross(nn.Module):
    """One decoder block's gated cross-attention into the keys function's cross keys and values."""

    def __init__(self, dims: DecoderDims):
        super().__init__()
        self.heads = dims.heads
        self.norm = nn.LayerNorm(dims.hidden, eps=spec.DEFAULT_LAYER_NORM_EPS)
        self.query = nn.Linear(dims.hidden, dims.hidden)
        self.out = nn.Linear(dims.hidden, dims.hidden)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(
        self, x: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        """Nodes *x* ``[1, S, hidden]`` attend into ``[heads, N, d]`` keys and values."""
        q = split_heads(self.query(self.norm(x)), self.heads)
        attended = merge_heads(attend(q, keys.unsqueeze(0), values.unsqueeze(0), bias))
        return cast(torch.Tensor, x + self.gate * self.out(attended))


class TreeDecoder(nn.Module):
    """The reader with its cross-attentions and the emission map, named as the bundle's are."""

    emit_ids: torch.Tensor

    def __init__(self, dims: DecoderDims, emitted: int):
        super().__init__()
        self.reader = Reader(dims)
        self.cross_attention = nn.ModuleList(TreeCross(dims) for _ in range(dims.layers))
        self.register_buffer("emit_ids", torch.zeros(emitted, dtype=torch.long))


class DecoderTree(nn.Module):
    """``tree_S``: every node's next-character logits over a packed, ancestor-closed prefix set."""

    emitted: torch.Tensor

    def __init__(self, manifest: Manifest):
        super().__init__()
        self.decoder = TreeDecoder(manifest.dims.decoder, manifest.dims.emitted)

    def bake(self) -> None:
        """Derive the emittable characters' embedding rows, the tied head's weight."""
        emit = self.decoder.emit_ids
        rows = self.decoder.reader.embed.weight.index_select(0, emit)
        self.register_buffer("emitted", rows.detach().clone(), persistent=False)

    def forward(
        self,
        reader_ids: torch.Tensor,
        positions: torch.Tensor,
        tree_bias: torch.Tensor,
        cross_keys: torch.Tensor,
        cross_values: torch.Tensor,
        cross_bias: torch.Tensor,
    ) -> torch.Tensor:
        """The tree function.

        ``reader_ids`` and ``positions`` are ``[1, S]`` int32 (a node's last token as a reader id,
        its depth), ``tree_bias`` the additive ``[1, 1, S, S]`` (0 for a node's ancestors and
        itself), ``cross_keys`` and ``cross_values`` ``[layers, heads, N_MAX, head_dim]`` and
        ``cross_bias`` ``[1, 1, 1, N_MAX]``. Returns the raw logits ``[1, S, E]``.
        """
        reader = self.decoder.reader
        x = reader.input(reader.embed(reader_ids)) + reader.positions(positions)
        for layer, block in enumerate(reader.blocks):
            x = block(x, tree_bias)
            x = self.decoder.cross_attention[layer](
                x, cross_keys[layer], cross_values[layer], cross_bias
            )
        return cast(torch.Tensor, reader.project(reader.norm(x)) @ self.emitted.T)


# --- building from a bundle -------------------------------------------------------------------


def build_module(bundle: Bundle, family: Family, length: int) -> nn.Module:
    """The *family* module over *bundle*'s weights, for sequences of *length*, in eval mode.

    For ``keys`` *length* is the row width (any width the key front has positions for, a bucket
    or an exact length), for ``context`` the token count and for ``tree`` unused: the node count
    is the input's. The tensors the family reads load with ``strict=True``, so a weight the
    module lacks or a module parameter the bundle lacks is an error.
    """
    manifest = bundle.manifest
    module: nn.Module
    if family == "context":
        module = ContextEncoder(manifest.dims.context)
    elif family == "keys":
        if length > manifest.dims.key_front.positions:
            raise ValueError(
                f"width {length} exceeds the key front's {manifest.dims.key_front.positions} "
                "positions"
            )
        module = KeysGraph(manifest, length)
    else:
        module = DecoderTree(manifest)
    module.load_state_dict(bundle.family_weights(family), strict=True)
    if isinstance(module, KeysGraph | DecoderTree):
        module.bake()
    return module.eval().requires_grad_(False)
