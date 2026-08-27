"""Per-character reading labels for the training corpus.

Milestone 3's v1 labels are g2pW alone. The dual annotation measured it at 98.97%
character agreement against the LLM, and its one systematic error (和 read han4)
still lands on a valid key sequence, so the noise is affordable for training --
while the *evaluation* labels stay dual-annotated and are never produced here.

Labels are keyed by sample id and sharded one file per sample shard, under the
same file name. That is what lets the sample builder stream a 5M-row corpus with
one shard of each in memory rather than joining two 500MB tables, and it makes a
partially labelled corpus a well-defined thing: the shards that exist are usable
and the ones that do not are simply absent.

Two producers write this schema. On a GPU box it is g2pW itself, through
:func:`generate`, with the ONNX session forced onto CUDA. Alternatively the Rust
``ime-cli g2p g2pw`` command emits the same readings as
``{"text", "syllables", "refusal"}`` JSON Lines, keyed by text rather than id;
:func:`labels_from_jsonl` adapts that stream back onto the sample ids by
position, checking the texts line up.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from mlime.data.corpus import Sample
from mlime.data.g2p import Outcome, Reading
from mlime.data.shards import shard_paths
from mlime.data.text import toneless
from mlime.logging import log

#: One row per sample: the readings g2pW produced, or why it produced none.
LABEL_SCHEMA = pl.Schema(
    {
        "id": pl.String(),
        "syllables": pl.List(pl.String()),
        "refusal": pl.String(),
    }
)

#: Execution providers the label job demands, in priority order.
CUDA_PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")

#: Initials after which pinyin orthography drops the umlaut, because no other
#: vowel can follow them: ``ju`` is unambiguously ``jü``, and ``ju`` is what the
#: keyboard produces. ``l`` and ``n`` are absent on purpose -- ``lu`` and ``lv``
#: are different syllables and both are typeable.
UMLAUT_FREE_INITIALS = ("j", "q", "x", "y")


def keyboard_form(syllable: str) -> str:
    """The letters a pinyin keyboard produces for the reading *syllable*.

    ``mlime.data.text.toneless`` drops the tone and spells ``ü`` as ``v``, which
    is right for ``lv`` and ``nv`` and wrong for ``jv``/``qv``/``xv``/``yv``:
    those are spelled ``ju``/``qu``/``xu``/``yu`` on a keyboard and in the
    syllable inventory. g2pW happens to emit the keyboard form already, so this
    fold is a no-op on its output -- it is here because the inventory lookup
    downstream is a hard failure, and an annotator that spells the umlaut the
    other way should not turn into a wall of dropped samples.
    """
    spelling = toneless(syllable)
    if len(spelling) > 1 and spelling[0] in UMLAUT_FREE_INITIALS and spelling[1] == "v":
        return f"{spelling[0]}u{spelling[2:]}"
    return spelling


@dataclass
class LabelCounts:
    """How a labelling run went."""

    labelled: int = 0
    refused: int = 0
    seconds: float = 0.0

    @property
    def sentences_per_second(self) -> float:
        """Throughput over the whole run, refusals included."""
        total = self.labelled + self.refused
        return total / self.seconds if self.seconds else 0.0


@contextmanager
def forced_providers(providers: Sequence[str]) -> Iterator[list[Any]]:
    """Build every ONNX session inside the block on *providers*, and hand them back.

    ``g2pw`` constructs its ``InferenceSession`` positionally inside
    ``G2PWConverter.__init__`` with no way to pass providers, and onnxruntime has
    required an explicit ``providers`` argument since 1.9 -- so on the GPU wheel
    the converter cannot be constructed at all without this. Wrapping the
    constructor for the duration of the call is the smallest intervention that
    does not fork the package.

    The sessions are yielded so the caller can assert which provider actually
    took the graph. A silent fall back to CPU would not fail; it would just make
    the measured throughput a lie.
    """
    import onnxruntime

    created: list[Any] = []
    original = onnxruntime.InferenceSession

    def build(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("providers", list(providers))
        session = original(*args, **kwargs)
        created.append(session)
        return session

    onnxruntime.InferenceSession = build
    try:
        yield created
    finally:
        onnxruntime.InferenceSession = original


def load_cuda_annotator(model_dir: Path, batch_size: int) -> Any:
    """A g2pW annotator whose ONNX session is running on CUDA.

    Raises if the session came up on anything else, because the whole reason to
    run this on a T4 is the throughput.
    """
    from mlime.data.g2pw_annotator import G2pwAnnotator

    with forced_providers(CUDA_PROVIDERS) as sessions:
        annotator = G2pwAnnotator(model_dir, batch_size=batch_size)
    if not sessions:
        raise RuntimeError("g2pw built no ONNX session; the upstream API has changed")
    active = sessions[0].get_providers()
    if CUDA_PROVIDERS[0] not in active:
        raise RuntimeError(
            f"the g2pw session came up on {active}; onnxruntime-gpu is not installed "
            "or no CUDA device is visible"
        )
    log.info("g2pw on cuda", providers=active, batch_size=batch_size)
    return annotator


def _row(sample_id: str, outcome: Outcome) -> dict[str, object]:
    """One ``LABEL_SCHEMA`` row from an annotator outcome."""
    if isinstance(outcome, Reading):
        return {"id": sample_id, "syllables": list(outcome.syllables), "refusal": None}
    return {"id": sample_id, "syllables": None, "refusal": outcome.reason}


def write_shard(rows: list[dict[str, object]], path: Path) -> None:
    """Write one label shard."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=LABEL_SCHEMA).write_parquet(path)


def select_shards(
    samples_dir: Path, names: Sequence[str] | None = None, limit: int | None = None
) -> list[Path]:
    """The sample shards a job should cover, named or counted.

    Naming them is what a stratified slice needs: the shards sort by source, so
    "the first five" is five shards of dialogue and says nothing about news.
    """
    available = shard_paths(samples_dir, "*")
    if not available:
        raise FileNotFoundError(f"no sample shards under {samples_dir}")
    if names:
        wanted = {name if name.endswith(".parquet") else f"{name}.parquet" for name in names}
        chosen = [path for path in available if path.name in wanted]
        missing = wanted - {path.name for path in chosen}
        if missing:
            raise FileNotFoundError(f"no such shards under {samples_dir}: {sorted(missing)}")
        return chosen
    return available[:limit] if limit is not None else available


async def generate(
    shards: Sequence[Path],
    out_dir: Path,
    annotator: Any,
    sentences_per_batch: int = 512,
    metrics: Path | None = None,
) -> LabelCounts:
    """Label each of *shards*, writing one label shard per sample shard.

    A shard whose output already exists is skipped, so a kernel that ran out of
    time resumes where it stopped instead of redoing the corpus.
    """
    paths = list(shards)
    if not paths:
        raise ValueError("no shards to label")
    counts = LabelCounts()
    for path in paths:
        target = out_dir / path.name
        if target.exists():
            log.info("label shard present, skipping", shard=path.name)
            continue
        started = time.monotonic()
        samples = [Sample.from_row(row) for row in pl.read_parquet(path).iter_rows(named=True)]
        rows: list[dict[str, object]] = []
        for start in range(0, len(samples), sentences_per_batch):
            batch = samples[start : start + sentences_per_batch]
            outcomes = await annotator.annotate([sample.text for sample in batch])
            for sample, outcome in zip(batch, outcomes, strict=True):
                rows.append(_row(sample.id, outcome))
                if isinstance(outcome, Reading):
                    counts.labelled += 1
                else:
                    counts.refused += 1
        write_shard(rows, target)
        elapsed = time.monotonic() - started
        counts.seconds += elapsed
        log.info(
            "label shard written",
            shard=path.name,
            sentences=len(samples),
            seconds=round(elapsed, 1),
            sentences_per_second=round(len(samples) / elapsed, 1),
            labelled=counts.labelled,
            refused=counts.refused,
        )
        if metrics is not None:
            _append_metric(
                metrics,
                {
                    "shard": path.name,
                    "sentences": len(samples),
                    "seconds": round(elapsed, 3),
                    "sentences_per_second": round(len(samples) / elapsed, 2),
                    "labelled": counts.labelled,
                    "refused": counts.refused,
                },
            )
    log.info(
        "labelling finished",
        labelled=counts.labelled,
        refused=counts.refused,
        seconds=round(counts.seconds, 1),
        sentences_per_second=round(counts.sentences_per_second, 1),
    )
    return counts


def _append_metric(path: Path, record: dict[str, object]) -> None:
    """Append one JSON Lines metric record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def labels_from_jsonl(shard: Path, readings: Path, out_dir: Path) -> LabelCounts:
    """Adapt the Rust ``g2p g2pw`` output for one sample shard into a label shard.

    ``ime-cli g2p g2pw --input <jsonl> --out <jsonl>`` reads ``{"text": ...}``
    lines and writes ``{"text", "syllables", "refusal"}`` in the same order,
    carrying no sample id. So the join is positional and every line's text is
    checked against the shard's -- a mismatch means the two files came from
    different runs, which would otherwise attach one sentence's readings to
    another's id.
    """
    samples = [Sample.from_row(row) for row in pl.read_parquet(shard).iter_rows(named=True)]
    counts = LabelCounts()
    rows: list[dict[str, object]] = []
    with readings.open(encoding="utf-8") as handle:
        lines = (line for line in handle if line.strip())
        for sample, line in zip(samples, lines, strict=True):
            record = json.loads(line)
            if record["text"] != sample.text:
                raise ValueError(
                    f"{readings} is not aligned with {shard}: sample {sample.id} is "
                    f"{sample.text!r} but the readings line is {record['text']!r}"
                )
            syllables = record.get("syllables")
            rows.append({"id": sample.id, "syllables": syllables, "refusal": record.get("refusal")})
            if syllables is None:
                counts.refused += 1
            else:
                counts.labelled += 1
    write_shard(rows, out_dir / shard.name)
    log.info(
        "label shard adapted",
        shard=shard.name,
        labelled=counts.labelled,
        refused=counts.refused,
    )
    return counts
