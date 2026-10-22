"""Wiring for `mlime train e2e`: assemble the end-to-end model, run route A's loop.

The model is built twice in its life, from different files: a run's first
segment builds it from the two initial checkpoints (route A's towers and the
reader's transformer), and every later segment -- and the evaluation -- builds
it from the e2e checkpoint that first segment wrote. Both go through
:func:`from_checkpoint` in spirit: the same ``e2e`` record carries the whole
config either way, and a resume under a different one is refused by the loop
itself.
"""

from __future__ import annotations

import functools
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from transformers import BertConfig

from mlime.data.shards import shard_paths
from mlime.logging import log
from mlime.train.charlm_model import CharLmConfig
from mlime.train.charlm_vocab import CharVocab
from mlime.train.e2e_model import E2EConfig, E2EModel, RankingConfig, emittable_ids
from mlime.train.e2e_search import SearchGraphs
from mlime.train.key_model import KeyE2EModel, key_evaluate, letter_token_ids
from mlime.train.keys import FillTowerTraining, KeyExample, KeyFrontConfig
from mlime.train.lexicon import Lexicon
from mlime.train.loop import Distributed, MetricLog, TrainingConfig, evaluate, train
from mlime.train.model import RouteAConfig
from mlime.train.qwen_tower import (
    QwenContextConfig,
    QwenContextTower,
    TowerGraphs,
    carried_cross_attention,
    load_pretrained,
)
from mlime.train.ranking import RankingE2EModel
from mlime.train.run import (
    CorpusLane,
    LaneRanking,
    RunPaths,
    RunResult,
    Slices,
    Vocabularies,
    corpus_digest,
    git_commit,
    held_out_examples,
    paused_run,
    step_losses,
)
from mlime.train.samples import DEFAULT_CONTEXT_TOKENS, Augmentation


def _record(state: dict[str, Any], key: str, path: Path | str) -> dict[str, Any]:
    """The named config record in a checkpoint, or a refusal it is not that model."""
    record = state.get(key)
    if not isinstance(record, dict):
        raise ValueError(f"{path} holds no {key} record; it is not a checkpoint of this model")
    return dict(record)


def _weights(state: dict[str, Any], path: Path | str) -> dict[str, torch.Tensor]:
    weights = state.get("model")
    if not isinstance(weights, dict):
        raise ValueError(f"{path} holds no model weights")
    return weights


def model_class(config: E2EConfig) -> type[E2EModel]:
    """The model a config trains: the key-level one, the ranking one, or the plain one."""
    if config.keys is not None:
        return KeyE2EModel
    return E2EModel if config.ranking is None else RankingE2EModel


def initialise(
    encoder_state: dict[str, Any],
    reader_state: dict[str, Any],
    lexicon: Lexicon,
    aux_weight: float,
    init_encoder: Path | str = "encoder checkpoint",
    init_decoder: Path | str = "decoder checkpoint",
    bert_config: BertConfig | None = None,
    ranking: RankingConfig | None = None,
    whole_alphabet: bool = False,
) -> E2EModel:
    """Build the model from a route A checkpoint and a transformer char-lm one.

    The encoder's weights must cover every parameter the run did not add --
    the towers, the span table, the context gates and the head; anything left
    over in either file is a wrong checkpoint, and a character the reader never
    saw is an error rather than an UNK. The reader's alphabet and shape come
    out of its own file, so a checkpoint made from a different character table
    is caught by :func:`emittable_ids`, not by a shape mismatch somewhere deep.
    """
    route = RouteAConfig(**_record(encoder_state, "route_a", init_encoder))
    decoder = CharLmConfig(**_record(reader_state, "config", init_decoder))
    vocab = CharVocab(chars=tuple(reader_state["vocab"]))
    emit_ids = emittable_ids(lexicon, vocab)
    config = E2EConfig(
        base_model=route.base_model,
        cross_attention_layers=route.cross_attention_layers,
        label_smoothing=route.label_smoothing,
        cross_attention_dropout=route.cross_attention_dropout,
        decoder=decoder,
        aux_weight=aux_weight,
        ranking=ranking,
        whole_alphabet=whole_alphabet,
    )
    model = model_class(config).compose(
        bert_config or BertConfig.from_pretrained(route.base_model),
        lexicon,
        config,
        len(vocab),
        emit_ids,
    )
    model.load_resumed(_weights(encoder_state, init_encoder), init_encoder)
    # The loaded buffers replace the lexicon's own tables: a character table
    # that drifted since route A trained would otherwise run against a stale
    # homophone set without a word.
    if not torch.equal(model.candidate_mask, lexicon.candidate_mask):
        raise ValueError(f"{init_encoder}'s candidate mask disagrees with the lexicon's")
    if not torch.equal(model.emittable_token_ids, lexicon.token_ids):
        raise ValueError(f"{init_encoder}'s emittable inventory disagrees with the lexicon's")
    model.decoder.reader.load_state_dict(_weights(reader_state, init_decoder), strict=True)
    log.info(
        "e2e model initialised",
        encoder_step=encoder_state.get("step"),
        reader_step=reader_state.get("step"),
        aux_weight=aux_weight,
    )
    return model


def from_checkpoint(
    state: dict[str, Any],
    lexicon: Lexicon,
    config: E2EConfig | None = None,
    bert_config: BertConfig | None = None,
) -> E2EModel:
    """Build the model an e2e checkpoint describes, without its weights.

    *config* replaces the record's, deliberately: the resume check in the
    loop compares the built model's config against the checkpoint's record, so
    a segment resumed under a different objective is refused with the field
    name rather than silently switching mid-schedule, and a run started from
    a finished checkpoint (``--init-e2e``) trains the objective it asks for.
    """
    if config is None:
        config = E2EConfig.from_record(_record(state, "e2e", "checkpoint"))
    weights = _weights(state, "checkpoint")
    emit_ids = weights.get("decoder.emit_ids")
    if not isinstance(emit_ids, torch.Tensor):
        raise ValueError("the checkpoint's model record holds no decoder.emit_ids buffer")
    if emit_ids.numel() != lexicon.size:
        raise ValueError(
            f"the checkpoint maps {emit_ids.numel()} emittable characters and this "
            f"lexicon holds {lexicon.size}; they were built from different tables"
        )
    embed = weights.get("decoder.reader.embed.weight")
    if not isinstance(embed, torch.Tensor):
        raise ValueError("the checkpoint's model record holds no decoder.reader.embed.weight")
    return model_class(config).compose(
        bert_config or BertConfig.from_pretrained(config.base_model),
        lexicon,
        config,
        int(embed.shape[0]),
        emit_ids.long(),
    )


def load_model(
    path: Path, lexicon: Lexicon, device: torch.device, bert_config: BertConfig | None = None
) -> tuple[E2EModel, int]:
    """The e2e checkpoint at *path*: the model it describes, and the step it left at.

    Its callers read lattices -- the beam search, the emission, the export --
    which a key-level model does not: its search is the constrained one over
    keys (#224), so its checkpoint is refused here rather than decoded as the
    span model it carries inside.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    recorded = E2EConfig.from_record(_record(state, "e2e", path))
    if recorded.keys is not None:
        raise ValueError(
            f"{path} is a key-level model; the lattice search does not decode it (#224)"
        )
    model = from_checkpoint(state, lexicon, bert_config=bert_config)
    model.load_resumed(_weights(state, path), path)
    model.to(device).eval()
    return model, int(state["step"])


#: What a :class:`~mlime.train.model.GatedCrossAttention` block stores, all of
#: which a swap to Qwen's tower reads.
GATED_BLOCK_WEIGHTS = (
    "norm.weight",
    "norm.bias",
    "attention.in_proj_weight",
    "attention.in_proj_bias",
    "attention.out_proj.weight",
    "attention.out_proj.bias",
    "gate",
)


def swap_in_qwen(model: E2EModel, weights: dict[str, torch.Tensor], source: Path | str) -> None:
    """Load a MacBERT-context run's weights into *model*, whose context tower is Qwen's.

    Everything but the context path loads as it was trained. MacBERT's tower
    (``context.*``) is dropped whole, Qwen's comes from its release, and each
    gated block keeps its norm, query, output projection, gate and biases
    (:func:`~mlime.train.qwen_tower.carried_cross_attention`). What is left at
    its initialisation is exactly what the model names as new on its context
    path -- the LayerNorm on the context rows and the blocks' key and value
    weights -- and anything else missing or left over is refused.
    """
    tower = model.context
    if not isinstance(tower, QwenContextTower):
        raise TypeError(f"the model's context tower is a {type(tower).__name__}, not Qwen's")
    hidden = int(model.fill.config.hidden_size)
    blocks = len(model.gated_layers())
    carried: dict[str, torch.Tensor] = {}
    for index in range(blocks):
        prefix = f"cross_attention.{index}."
        carried.update(
            {
                prefix + name: value
                for name, value in carried_cross_attention(weights, prefix, hidden).items()
            }
        )
    packed = {
        f"cross_attention.{index}.{name}" for index in range(blocks) for name in GATED_BLOCK_WEIGHTS
    }
    unread = sorted(
        name for name in weights if name.startswith("cross_attention.") and name not in packed
    )
    if unread:
        raise ValueError(f"{source}'s gated blocks hold weights the swap does not read: {unread}")
    kept = {
        name: value
        for name, value in weights.items()
        if not name.startswith(("context.", "cross_attention."))
    }
    result = model.load_state_dict({**kept, **carried}, strict=False)
    fresh = model.new_context_parameters()
    missing = sorted(
        name
        for name in result.missing_keys
        if name not in fresh and not name.startswith("context.model.")
    )
    if missing or result.unexpected_keys:
        raise ValueError(
            f"{source} does not fit the model with Qwen's tower swapped in: "
            f"missing {missing[:4]}, unexpected {result.unexpected_keys[:4]}"
        )
    load_pretrained(tower)
    log.info(
        "context tower swapped to qwen",
        dropped=sum(name.startswith("context.") for name in weights),
        carried_blocks=blocks,
        fresh=sorted(fresh),
    )


def build_from(
    source: Path,
    char_table: Path,
    *,
    with_weights: bool,
    aux_weight: float,
    ranking: RankingConfig | None,
    whole_alphabet: bool,
    context: QwenContextConfig | None,
    context_warm_up: bool,
    keys: KeyFrontConfig | None = None,
    key_warm_up: bool = False,
    train_fill_tower: FillTowerTraining | None = None,
) -> tuple[E2EModel, Vocabularies]:
    """The model the e2e checkpoint at *source* describes, under this stage's losses.

    *with_weights* -- a stage starting from a finished run -- carries the
    checkpoint's weights over; a resume takes them from the loop instead. A
    stage that names a Qwen *context* tower over a run that read its context
    with MacBERT swaps the tower in (:func:`swap_in_qwen`); any other change
    of tower is refused, and a resume under a different one is refused by the
    loop. A stage that names a key front over a span model's run warms the
    front up over it (:meth:`~mlime.train.key_model.KeyE2EModel.load_parent`),
    reading the context with the parent's own tower; a key run keeps its
    front. The checkpoint is gigabytes, and it lives only in this frame:
    nothing holds it once the model is built.
    """
    state = torch.load(source, map_location="cpu", weights_only=False)
    recorded = E2EConfig.from_record(_record(state, "e2e", source))
    wanted = replace(
        recorded,
        aux_weight=aux_weight,
        ranking=ranking,
        whole_alphabet=whole_alphabet,
        context=context,
        context_warm_up=context_warm_up,
        keys=keys,
        key_warm_up=key_warm_up,
        train_fill_tower=train_fill_tower,
    )
    vocabularies = Vocabularies.load(char_table, recorded.base_model, context)
    model = from_checkpoint(state, vocabularies.lexicon, wanted)
    if recorded.keys is not None and recorded.keys != keys:
        raise ValueError(f"{source} has the key front {recorded.keys}; a stage keeps it as it is")
    if with_weights:
        if recorded.keys is None and keys is not None:
            if recorded.context != context:
                raise ValueError(
                    f"{source} reads its context with {recorded.context}; a key front warms "
                    "up over its parent's own tower"
                )
            if not isinstance(model, KeyE2EModel):
                raise TypeError(f"a key front config built a {type(model).__name__}")
            model.load_parent(
                _weights(state, source), source, letter_token_ids(vocabularies.vocabulary)
            )
        elif recorded.context == context:
            model.load_resumed(_weights(state, source), source)
        elif recorded.context is None:
            swap_in_qwen(model, _weights(state, source), source)
        else:
            raise ValueError(
                f"{source} reads its context with {recorded.context}; a stage swaps only "
                "MacBERT's tower for Qwen's"
            )
        log.info("e2e model initialised from a finished run", step=state.get("step"))
    return model, vocabularies


def build_fresh(
    init_encoder: Path,
    init_decoder: Path,
    char_table: Path,
    *,
    aux_weight: float,
    ranking: RankingConfig | None,
    whole_alphabet: bool,
) -> tuple[E2EModel, Vocabularies]:
    """A new e2e model composed from route A's and the reader's checkpoints.

    The two files are most of a gigabyte each, and they live only in this
    frame, as :func:`build_from`'s checkpoint does: ``initialise`` copies
    their weights into the model, and nothing holds either once it is built.
    """
    encoder_state = torch.load(init_encoder, map_location="cpu", weights_only=False)
    base_model = RouteAConfig(**_record(encoder_state, "route_a", init_encoder)).base_model
    vocabularies = Vocabularies.load(char_table, base_model, None)
    reader_state = torch.load(init_decoder, map_location="cpu", weights_only=False)
    model = initialise(
        encoder_state,
        reader_state,
        vocabularies.lexicon,
        aux_weight,
        init_encoder,
        init_decoder,
        ranking=ranking,
        whole_alphabet=whole_alphabet,
    )
    return model, vocabularies


def compile_modules(model: E2EModel, mode: str) -> None:
    """torch.compile the encoder pass and the decoder stack, in place.

    The callables are compiled rather than the modules: an ``OptimizedModule``
    nested in the model leaks an ``_orig_mod`` prefix into ``state_dict`` and
    ``named_parameters``, which would poison the checkpoint format and the
    parameter-group split. Rebinding the methods keeps the module tree -- and
    every checkpoint written or resumed -- untouched. ``encode`` is one compiled
    unit (the towers' real call path loops its layers itself) and the decoder's
    ``forward`` another. ``dynamic=True`` because width and span count change
    with every batch. Applied before ``train`` wraps the model in DDP, so the
    compiled calls run inside the wrapped forward, where dynamo cuts each
    graph at DDP's gradient buckets: each bucket's allreduce then starts while
    the rest of backward still runs.

    The decoder's ``advance`` -- one search position: the step and the beam's
    bookkeeping after it -- is the third: the ranking loss's search runs it
    once per position over every reading's beam, about a thousand small ops
    a position, and eager dispatch of those took longer than the card took
    to run them. It has no backward, so the bucket cut buys nothing and costs
    a compiled call per bucket -- nine a position, whose launches took four
    times the device time they bought on a T4 -- and it is compiled whole.

    Every graph's wrapper -- the code that sizes the buffers and launches the
    kernels -- is generated as C++ rather than Python. A search position is
    some four hundred launches, and from Python the host took longer to issue
    them than the card took to run them: on two T4s the ranking step went from
    19 to 23.4 examples a second per rank and the cards from 70% to 93% busy.
    The C++ links against ``libcuda.so``, which the host must provide.

    A context tower that runs apart from the fill pass (``separate``: Qwen's,
    frozen or replayed from its own graphs) stays outside the compiled region,
    and only the fill pass that reads its memory is compiled.

    A key-level model runs the fill pass twice a step -- over the spans for
    its teacher and over the keys -- and reads the context once for both, so
    its units are the context tower, the fill pass and the key front, each
    one graph whatever fills the fill pass's positions.
    """
    options = {**torch._inductor.list_mode_options(mode), "cpp_wrapper": True}
    if isinstance(model, KeyE2EModel):
        if not model.context.separate:
            model.encode_context = torch.compile(  # type: ignore[method-assign]
                model.encode_context, dynamic=True, options=options
            )
        model.fill_pass = torch.compile(  # type: ignore[method-assign]
            model.fill_pass, dynamic=True, options=options
        )
        model.key_front.forward = torch.compile(  # type: ignore[method-assign]
            model.key_front.forward, dynamic=True, options=options
        )
        model.decoder.forward = torch.compile(  # type: ignore[method-assign]
            model.decoder.forward, dynamic=True, options=options
        )
        log.info("key front, towers and decoder compiled", mode=mode)
        return
    if model.context.separate:
        model.encode_fill = torch.compile(  # type: ignore[method-assign]
            model.encode_fill, dynamic=True, options=options
        )
    else:
        model.encode = torch.compile(model.encode, dynamic=True, options=options)  # type: ignore[method-assign]
    model.decoder.forward = torch.compile(  # type: ignore[method-assign]
        model.decoder.forward, dynamic=True, options=options
    )
    advance = torch.compile(model.decoder.advance, dynamic=True, options=options)

    @functools.wraps(model.decoder.advance)
    def whole_advance(*args: Any, **kwargs: Any) -> Any:
        # Read when dynamo compiles a frame, which is inside this call.
        with torch._dynamo.config.patch(optimize_ddp=False):
            return advance(*args, **kwargs)

    model.decoder.advance = whole_advance  # type: ignore[method-assign]
    log.info("towers and decoder compiled", mode=mode)


def e2e(
    paths: RunPaths,
    slices: Slices,
    training: TrainingConfig,
    init_encoder: Path | None = None,
    init_decoder: Path | None = None,
    aux_weight: float = 0.3,
    augmentation: Augmentation | None = None,
    context_dropout: float = 0.3,
    max_context_tokens: int = DEFAULT_CONTEXT_TOKENS,
    resume: Path | None = None,
    activation_checkpointing: bool = False,
    run_config: Path | None = None,
    init_e2e: Path | None = None,
    ranking: RankingConfig | None = None,
    segmenter: Path | None = None,
    whole_alphabet: bool = False,
    search_graphs: bool = False,
    context: QwenContextConfig | None = None,
    context_warm_up: bool = False,
    tower_graphs: bool = False,
    keys: KeyFrontConfig | None = None,
    key_warm_up: bool = False,
    train_fill_tower: FillTowerTraining | None = None,
) -> RunResult:
    """Train the end-to-end model over *slices.train*; score held-out both ways.

    The data, the loop, the optimiser shape and the held-out measurement are
    route A's own: the same shards, typed the same way, against the same
    restricted loss. What differs is the model under them and where its weights
    start. A run starts one of three ways: from route A's and the reader's
    checkpoints (*init_encoder* and *init_decoder*), from a finished e2e
    checkpoint with a fresh optimiser and schedule (*init_e2e*), or as the
    next segment of the run that wrote *resume*; anything else is an error.
    *ranking* adds the sequence-level loss of #116, which reads every
    example's other segmentations through the ``ime-segment`` binary at
    *segmenter*. *whole_alphabet* normalises the decoder's per-position term
    over the whole emittable set (#121). *search_graphs* replays the
    ranking loss's search from captured CUDA graphs (#137). *run_config* is
    the kernel's own run-config file, copied into each interval trio
    verbatim; the loop refuses ``checkpoint_minutes`` without it.

    *context* names the Qwen tower that reads the context (#222): a stage
    from a MacBERT run swaps it in, and every segment after names it again.
    *context_warm_up* trains only the context path (S0). *tower_graphs*
    replays a frozen Qwen tower's forward from captured CUDA graphs.

    *keys* reads keystrokes through a key front with a per-key CTC head
    (#223), whose examples are typed as keys with their legal alignments;
    *key_warm_up* trains only the front and the head, distilling the parent's
    encoder states (S1). *train_fill_tower* trains the fill tower beside
    them at the pretrained rate -- and with ``cross_attention`` its gated
    context blocks -- distilling towards frozen copies of the parent's
    (#241). A finished key run is scored by the S1 gate's measure --
    the parent's decoder fed the front's states -- beside the parent's own on
    the same examples.
    """
    world = Distributed.from_environment()
    starts = (
        init_encoder is not None or init_decoder is not None,
        init_e2e is not None,
        resume is not None,
    )
    if sum(starts) != 1:
        raise ValueError(
            "a run starts from --init-encoder with --init-decoder, from --init-e2e, "
            "or resumes with --resume: exactly one of them"
        )
    if (ranking is None) != (segmenter is None):
        raise ValueError("the ranking loss and the --segmenter binary come together")
    augmented = augmentation or Augmentation()
    if ranking is not None and augmented.typos:
        raise ValueError(
            "the ranking loss segments clean keystrokes; the typo noise model is not "
            "wired into ime-segment"
        )
    if context_warm_up and augmented.typos:
        raise ValueError(
            "the context warm-up trains the context path alone; the typo inputs it would "
            "corrupt belong to the frozen letter encoder"
        )
    if tower_graphs and (context is None or context.lora is not None):
        raise ValueError("--tower-graphs replays a frozen Qwen tower; name one without LoRA")
    if tower_graphs and not torch.cuda.is_available():
        raise ValueError("--tower-graphs replays CUDA graphs and this run has no CUDA device")
    if keys is not None and (init_encoder is not None or init_decoder is not None):
        raise ValueError(
            "a key front warms up over a finished run; start the stage with --init-e2e"
        )
    if train_fill_tower is not None and not key_warm_up:
        raise ValueError(
            "--train-fill-tower trains the tower in the key warm-up; add --key-warm-up"
        )
    source = init_e2e if init_e2e is not None else resume
    if source is not None:
        model, vocabularies = build_from(
            source,
            paths.char_table,
            with_weights=init_e2e is not None,
            aux_weight=aux_weight,
            ranking=ranking,
            whole_alphabet=whole_alphabet,
            context=context,
            context_warm_up=context_warm_up,
            keys=keys,
            key_warm_up=key_warm_up,
            train_fill_tower=train_fill_tower,
        )
    else:
        if init_encoder is None or init_decoder is None:
            raise ValueError("a new run needs both --init-encoder and --init-decoder")
        if context is not None or context_warm_up:
            raise ValueError(
                "a Qwen tower is swapped into a finished run, and warmed up there; "
                "start the stage with --init-e2e"
            )
        model, vocabularies = build_fresh(
            init_encoder,
            init_decoder,
            paths.char_table,
            aux_weight=aux_weight,
            ranking=ranking,
            whole_alphabet=whole_alphabet,
        )
    if activation_checkpointing:
        model.enable_encoder_checkpointing()
        log.info("encoder activation checkpointing on", trade="memory for one extra forward")
    if training.compile:
        compile_modules(model, training.compile_mode)
    if search_graphs:
        if not isinstance(model, RankingE2EModel):
            raise ValueError("--search-graphs captures the ranking loss's search; add --ranking")
        if not torch.cuda.is_available():
            raise ValueError("--search-graphs replays CUDA graphs and this run has no CUDA device")
        model.search_graphs = SearchGraphs()
    if tower_graphs:
        tower = model.context
        if not isinstance(tower, QwenContextTower):
            raise TypeError(f"--tower-graphs replays Qwen's tower, not a {type(tower).__name__}")
        tower.graphs = TowerGraphs()
    lexicon, tokenizer = vocabularies.lexicon, vocabularies.tokenizer

    # One lane per rank this process stands for: lane j of a process takes
    # the shard deal of world rank `rank * virtual + j`, and every lane's
    # collator starts from the same seed -- exactly the objects a real world
    # of `world_size * virtual` ranks would have built.
    virtual = training.virtual_ranks
    total_world = world.world_size * virtual
    lane_ranking = (
        LaneRanking(ranking, segmenter) if ranking is not None and segmenter is not None else None
    )
    lanes = [
        CorpusLane(
            paths=paths,
            base_model=model.config.base_model,
            shards=slices.train,
            augmentation=augmented,
            seed=training.seed,
            rank=world.rank * virtual + index,
            world_size=total_world,
            context_dropout=context_dropout,
            max_context_tokens=max_context_tokens,
            ranking=lane_ranking,
            context=context,
            keys=keys,
        )
        for index in range(virtual)
    ]
    # Frozen while the noise model is off, so the fill tower stays exactly the
    # pre-letter one and the ONNX export's span_ids-only graph still describes it.
    model.letter_encoder.requires_grad_(augmented.typos)
    if context_warm_up:
        model.freeze_for_context_warm_up()
        log.info(
            "context warm-up: only the context path trains",
            trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    if key_warm_up:
        if not isinstance(model, KeyE2EModel):
            raise TypeError(
                f"the key warm-up needs a key-level model, not a {type(model).__name__}"
            )
        model.freeze_for_key_warm_up()
        log.info(
            "key warm-up: the key front and the CTC head train",
            beside=list(model.trained_parts()),
            trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        )

    paths.out.mkdir(parents=True, exist_ok=True)
    with MetricLog(paths.out / "metrics.jsonl") as provenance:
        if world.is_main:
            provenance.write(
                event="provenance",
                commit=git_commit(),
                corpus=corpus_digest(shard_paths(paths.samples, "*")),
                labels=corpus_digest(shard_paths(paths.labels, "*")),
                label_source="g2pw",
                reading_arbitration=vocabularies.arbitration.digest,
                train_shards=list(slices.train),
                held_out_shards=list(slices.held_out),
                augmentation=asdict(augmentation or Augmentation()),
                e2e=asdict(model.config),
                init_encoder=None if init_encoder is None else str(init_encoder),
                init_decoder=None if init_decoder is None else str(init_decoder),
                init_e2e=None if init_e2e is None else str(init_e2e),
                emittable_characters=lexicon.size,
                typed_spans=lexicon.spans,
                resume=str(resume) if resume is not None else None,
            )

    segment = train(model, lanes, training, paths.out, world, resume, run_config=run_config)
    losses = step_losses(segment.metrics)
    if not segment.finished:
        return paused_run(segment, losses, model.gates(), world)
    device = world.device
    parents: dict[str, float] = {}
    if isinstance(model, KeyE2EModel):
        if keys is None:
            raise TypeError("a key-level model was built without its key front config")
        key_held_out = held_out_examples(
            paths, vocabularies.key_builder(augmented, training.seed + 1, keys), slices
        )
        held: list[KeyExample] = []
        for example in key_held_out:
            if not isinstance(example, KeyExample):
                raise TypeError(f"the key builder built a {type(example).__name__}")
            held.append(example)
        with_context, parent_with = key_evaluate(
            model,
            held,
            vocabularies.key_collator(0.0, max_context_tokens),
            device,
            training.token_budget,
        )
        without_context, parent_without = key_evaluate(
            model,
            held,
            vocabularies.key_collator(1.0, max_context_tokens),
            device,
            training.token_budget,
        )
        parents = {
            "parent_accuracy_with_context": parent_with.rate,
            "parent_accuracy_without_context": parent_without.rate,
        }
        evaluated = len(held)
    else:
        evaluation = held_out_examples(
            paths, vocabularies.builder(augmented, training.seed + 1), slices
        )
        candidates = vocabularies.candidate_space(augmented.typos)
        reading = vocabularies.context_tokenizer
        with_context = evaluate(
            model, evaluation, tokenizer, reading, device, training.token_budget, True, candidates
        )
        without_context = evaluate(
            model, evaluation, tokenizer, reading, device, training.token_budget, False, candidates
        )
        evaluated = len(evaluation)
    result = RunResult(
        metrics=segment.metrics,
        first_loss=losses[0],
        last_loss=losses[-1],
        steps=len(losses),
        step=segment.step,
        finished=True,
        with_context=with_context,
        without_context=without_context,
        gates=model.gates(),
    )
    if world.is_main:
        with MetricLog(segment.metrics) as summary:
            summary.write(
                event="summary",
                finished=True,
                step=result.step,
                first_loss=result.first_loss,
                last_loss=result.last_loss,
                logged_steps=result.steps,
                held_out_examples=evaluated,
                accuracy_with_context=with_context.rate,
                accuracy_without_context=without_context.rate,
                **parents,
                scored_characters=with_context.scored,
                gates=result.gates,
                build_counts=segment.build_counts,
            )
    log.info(
        "e2e run finished",
        first_loss=round(result.first_loss, 4),
        last_loss=round(result.last_loss, 4),
        accuracy_with_context=round(with_context.rate, 4),
        accuracy_without_context=round(without_context.rate, 4),
        **{name: round(rate, 4) for name, rate in parents.items()},
        held_out=evaluated,
        gates=[round(gate, 5) for gate in result.gates],
    )
    return result
