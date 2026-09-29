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

from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from transformers import BertConfig

from mlime.data.shards import shard_paths
from mlime.logging import log
from mlime.train.charlm_model import CharLmConfig
from mlime.train.charlm_vocab import CharVocab
from mlime.train.e2e_model import E2EConfig, E2EModel, emittable_ids
from mlime.train.lexicon import Lexicon
from mlime.train.loop import Distributed, MetricLog, TrainingConfig, evaluate, train
from mlime.train.model import RouteAConfig
from mlime.train.run import (
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


def initialise(
    encoder_state: dict[str, Any],
    reader_state: dict[str, Any],
    lexicon: Lexicon,
    aux_weight: float,
    init_encoder: Path | str = "encoder checkpoint",
    init_decoder: Path | str = "decoder checkpoint",
    bert_config: BertConfig | None = None,
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
    )
    model = E2EModel.compose(
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
    aux_weight: float | None = None,
    bert_config: BertConfig | None = None,
) -> E2EModel:
    """Build the model an e2e checkpoint describes, without its weights.

    *aux_weight* overrides the record's, deliberately: the resume check in the
    loop compares the built model's config against the checkpoint's record, so
    a segment resumed under a different auxiliary weight is refused with the
    field name rather than silently switching objective mid-schedule.
    """
    config = E2EConfig.from_record(_record(state, "e2e", "checkpoint"))
    if aux_weight is not None:
        config = replace(config, aux_weight=aux_weight)
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
    return E2EModel.compose(
        bert_config or BertConfig.from_pretrained(config.base_model),
        lexicon,
        config,
        int(embed.shape[0]),
        emit_ids.long(),
    )


def load_model(
    path: Path, lexicon: Lexicon, device: torch.device, bert_config: BertConfig | None = None
) -> tuple[E2EModel, int]:
    """The e2e checkpoint at *path*: the model it describes, and the step it left at."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = from_checkpoint(state, lexicon, bert_config=bert_config)
    model.load_resumed(_weights(state, path), path)
    model.to(device).eval()
    return model, int(state["step"])


def compile_modules(model: E2EModel, mode: str) -> None:
    """torch.compile the encoder pass and the decoder stack, in place.

    The callables are compiled rather than the modules: an ``OptimizedModule``
    nested in the model leaks an ``_orig_mod`` prefix into ``state_dict`` and
    ``named_parameters``, which would poison the checkpoint format and the
    parameter-group split. Rebinding the methods keeps the module tree -- and
    every checkpoint written or resumed -- untouched. ``encode`` is one compiled
    unit (the towers' real call path loops its layers itself) and the decoder's
    ``forward`` another. ``dynamic=True`` because width and span count change
    with every batch. Applied before ``train`` wraps the model in DDP -- the
    compiled calls sit *inside* the wrapped forward, so no
    ``optimize_ddp`` knob is needed.
    """
    model.encode = torch.compile(model.encode, dynamic=True, mode=mode)  # type: ignore[method-assign]
    model.decoder.forward = torch.compile(  # type: ignore[method-assign]
        model.decoder.forward, dynamic=True, mode=mode
    )
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
) -> RunResult:
    """Train the end-to-end model over *slices.train*; score held-out both ways.

    The data, the loop, the optimiser shape and the held-out measurement are
    route A's own: the same shards, typed the same way, against the same
    restricted loss. What differs is the model under them and where its weights
    start. With *resume* this continues the run that wrote the checkpoint --
    ``--init-*`` belongs only to the first segment and passing both is an error.
    *run_config* is the kernel's own run-config file, copied into each
    interval trio verbatim; the loop refuses ``checkpoint_minutes`` without it.
    """
    world = Distributed.from_environment()
    if resume is None:
        if init_encoder is None or init_decoder is None:
            raise ValueError(
                "a new run needs --init-encoder and --init-decoder; a resumed run needs --resume"
            )
        encoder_state = torch.load(init_encoder, map_location="cpu", weights_only=False)
        base_model = RouteAConfig(**_record(encoder_state, "route_a", init_encoder)).base_model
        vocabularies = Vocabularies.load(paths.char_table, base_model)
        reader_state = torch.load(init_decoder, map_location="cpu", weights_only=False)
        model = initialise(
            encoder_state,
            reader_state,
            vocabularies.lexicon,
            aux_weight,
            init_encoder,
            init_decoder,
        )
    else:
        if init_encoder is not None or init_decoder is not None:
            raise ValueError(
                "--resume continues the run a checkpoint describes; "
                "--init-encoder and --init-decoder start a new one, and are refused beside it"
            )
        resumed_state = torch.load(resume, map_location="cpu", weights_only=False)
        base_model = E2EConfig.from_record(_record(resumed_state, "e2e", resume)).base_model
        vocabularies = Vocabularies.load(paths.char_table, base_model)
        model = from_checkpoint(resumed_state, vocabularies.lexicon, aux_weight)
    if activation_checkpointing:
        model.enable_encoder_checkpointing()
        log.info("encoder activation checkpointing on", trade="memory for one extra forward")
    if training.compile:
        compile_modules(model, training.compile_mode)
    lexicon, tokenizer = vocabularies.lexicon, vocabularies.tokenizer

    # One (stream, collator) lane per rank this process stands for: lane j of
    # a process takes the shard deal of world rank `rank * virtual + j`, and
    # every lane's collator starts from the same seed -- exactly the objects a
    # real world of `world_size * virtual` ranks would have built.
    virtual = training.virtual_ranks
    total_world = world.world_size * virtual
    augmented = augmentation or Augmentation()
    builders = [vocabularies.builder(augmented, training.seed) for _ in range(virtual)]
    lanes = [
        (
            vocabularies.stream(
                paths,
                slices.train,
                builders[index],
                world.rank * virtual + index,
                total_world,
            ),
            vocabularies.collator(
                context_dropout, max_context_tokens, training.seed, typos=augmented.typos
            ),
        )
        for index in range(virtual)
    ]
    # Frozen while the noise model is off, so the fill tower stays exactly the
    # pre-letter one and the ONNX export's span_ids-only graph still describes it.
    model.letter_encoder.requires_grad_(augmented.typos)

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
                emittable_characters=lexicon.size,
                typed_spans=lexicon.spans,
                resume=str(resume) if resume is not None else None,
            )

    segment = train(model, lanes, training, paths.out, world, resume, run_config=run_config)
    losses = step_losses(segment.metrics)
    if not segment.finished:
        build_counts: Counter[str] = Counter()
        for builder in builders:
            build_counts.update(builder.counts.as_dict())
        return paused_run(segment, losses, model.gates(), dict(build_counts), world)
    evaluation = held_out_examples(
        paths, vocabularies.builder(augmented, training.seed + 1), slices
    )
    device = world.device
    candidates = vocabularies.candidate_space(augmented.typos)
    with_context = evaluate(
        model, evaluation, tokenizer, device, training.token_budget, True, candidates
    )
    without_context = evaluate(
        model, evaluation, tokenizer, device, training.token_budget, False, candidates
    )
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
                held_out_examples=len(evaluation),
                accuracy_with_context=with_context.rate,
                accuracy_without_context=without_context.rate,
                scored_characters=with_context.scored,
                gates=result.gates,
                build_counts=builder.counts.as_dict(),
            )
    log.info(
        "e2e run finished",
        first_loss=round(result.first_loss, 4),
        last_loss=round(result.last_loss, 4),
        accuracy_with_context=round(with_context.rate, 4),
        accuracy_without_context=round(without_context.rate, 4),
        held_out=len(evaluation),
        gates=[round(gate, 5) for gate in result.gates],
    )
    return result
