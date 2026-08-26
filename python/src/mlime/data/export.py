"""Export what the Rust side reads: an evaluation set, and the text to train on.

The two exports are one module because they are two halves of one split. Every
sentence drawn into the evaluation set must be absent from the training text, or
the baseline is scored on sentences it memorised, so the exclusion list and the
evaluation draw have to agree on what a sentence *is* -- the exact target string,
after the corpus normaliser has already run.

Three constraints shape what is eligible. The line's ``pinyin`` field is what a
user *types*, so it is the toneless syllables run together with no separator --
the harness re-segments it, which is half of what is being evaluated. Only
sentences both annotators agreed on can appear, because a disputed reading would
score a correct conversion as wrong. And the target must be entirely Han: a
comma or a digit inside it has no keystrokes behind it, so it would make the
syllable count disagree with the character count. Those sentences stay in the
training data, where the mismatch is harmless, and are simply not evaluated on.

Sampling is stratified by source and seeded, so the number quoted in one report
is the number another run reproduces.

``ngram-corpus`` is the other half: every prepared target, one per line, minus
the held-out sentences. It reads the same JSON Lines shape it writes, so the
evaluation set -- or the wider pool the evaluation set was drawn from -- is
handed straight back as the exclusion list. An exclusion that matches nothing is
an error rather than a warning: the held-out sentences came out of these very
shards, so a miss means the wrong directory or a normaliser that has moved under
the file, and both of those silently put the evaluation sentences back into
training.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from pathlib import Path

import polars as pl
from pydantic import BaseModel, ValidationError

from mlime.logging import log

from .shards import shard_paths
from .text import han_characters, toneless


class EvalItem(BaseModel):
    """One line of the evaluation file: what was typed, what it should become."""

    pinyin: str
    text: str
    context: str | None


class HeldOutLine(BaseModel):
    """The one field an exclusion file has to carry, whatever else it holds.

    Both the evaluation set and the pool it was drawn from are JSON Lines with a
    ``text`` field, so either can be handed to ``--exclude`` unchanged.
    """

    text: str


def eligible(annotated: pl.DataFrame) -> pl.DataFrame:
    """Rows both annotators agreed on whose target is Han characters only."""
    return annotated.filter(
        pl.col("agree_all") & (pl.col("text").str.len_chars() == pl.col("characters").list.len())
    )


def to_item(row: Mapping[str, object]) -> EvalItem:
    """Build one evaluation line, checking the syllables really do cover the target."""
    text = str(row["text"])
    syllables = row["g2pw"]
    if not isinstance(syllables, Sequence) or isinstance(syllables, str):
        raise TypeError(f"expected a list of syllables for {text!r}, got {syllables!r}")
    characters = han_characters(text)
    if len(syllables) != len(characters) or len(characters) != len(text):
        raise ValueError(
            f"{text!r} has {len(text)} characters, {len(characters)} of them Han, "
            f"against {len(syllables)} syllables"
        )
    context = row["context"]
    return EvalItem(
        pinyin="".join(toneless(str(syllable)) for syllable in syllables),
        text=text,
        context=None if context is None else str(context),
    )


def allocate(available: Mapping[str, int], size: int) -> dict[str, int]:
    """Split *size* as evenly as possible across sources, spilling any shortfall.

    A source that cannot fill its share does not shrink the export; its unused
    quota is offered to the sources that can, so a small dialogue slice never
    silently caps the whole evaluation set.
    """
    total = sum(available.values())
    if total < size:
        raise ValueError(f"only {total} eligible samples across {sorted(available)}, need {size}")
    quotas = dict.fromkeys(available, 0)
    remaining = size
    open_sources = {name for name, count in available.items() if count > 0}
    while remaining and open_sources:
        share = max(1, remaining // len(open_sources))
        for name in sorted(open_sources):
            if not remaining:
                break
            take = min(share, available[name] - quotas[name], remaining)
            quotas[name] += take
            remaining -= take
            if quotas[name] == available[name]:
                open_sources.discard(name)
    return quotas


def sample_rows(rows: pl.DataFrame, size: int, seed: int) -> pl.DataFrame:
    """Deterministically draw *size* rows, stratified across the ``source`` column."""
    by_source = {
        str(name[0]): frame
        for name, frame in rows.sort("id").group_by("source", maintain_order=True)
    }
    quotas = allocate({name: frame.height for name, frame in by_source.items()}, size)
    rng = random.Random(seed)
    drawn = []
    for name in sorted(by_source):
        frame = by_source[name]
        indices = sorted(rng.sample(range(frame.height), quotas[name]))
        drawn.append(frame[indices])
    return pl.concat(drawn)


def export_eval_set(annotated: pl.DataFrame, out_path: Path, size: int, seed: int = 0) -> int:
    """Write the sampled evaluation set to *out_path* as JSON Lines. Returns the count."""
    rows = eligible(annotated)
    log.info("eligible for evaluation", annotated=annotated.height, eligible=rows.height)
    selected = sample_rows(rows, size, seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in selected.iter_rows(named=True):
            handle.write(to_item(row).model_dump_json())
            handle.write("\n")
    log.info(
        "evaluation set written",
        path=str(out_path),
        items=selected.height,
        by_source=dict(selected["source"].value_counts().iter_rows()),
    )
    return selected.height


def read_exclusions(paths: Sequence[Path]) -> frozenset[str]:
    """The target sentences held out of training, from the ``text`` field of each line."""
    held_out: set[str] = set()
    for path in paths:
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise ValueError(f"{path} holds no exclusions; drop the flag rather than passing it")
        for number, line in enumerate(lines, start=1):
            try:
                held_out.add(HeldOutLine.model_validate_json(line).text)
            except ValidationError as error:
                raise ValueError(f"{path}:{number} has no usable `text` field: {line!r}") from error
    log.info("exclusions loaded", files=len(paths), sentences=len(held_out))
    return frozenset(held_out)


def export_ngram_corpus(samples_dir: Path, out_path: Path, held_out: frozenset[str]) -> int:
    """Write every prepared target except *held_out* to *out_path*, one per line.

    Shards are read one at a time: the corpus this feeds is millions of lines and
    the whole point of the parquet shards is that no stage has to hold them all.
    """
    paths = shard_paths(samples_dir, "*")
    if not paths:
        raise FileNotFoundError(f"no sample shards under {samples_dir}")
    unmatched = set(held_out)
    written = 0
    excluded = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for path in paths:
            for value in pl.read_parquet(path, columns=["text"])["text"].to_list():
                if not isinstance(value, str):
                    raise TypeError(f"{path} holds a non-string target: {value!r}")
                if value in held_out:
                    unmatched.discard(value)
                    excluded += 1
                    continue
                handle.write(value)
                handle.write("\n")
                written += 1
    if unmatched:
        raise ValueError(
            f"{len(unmatched)} of {len(held_out)} held-out sentences are not in {samples_dir}, "
            f"so nothing was held out for them: {sorted(unmatched)[:3]}"
        )
    log.info(
        "n-gram corpus written",
        path=str(out_path),
        lines=written,
        excluded=excluded,
        shards=len(paths),
    )
    return written
