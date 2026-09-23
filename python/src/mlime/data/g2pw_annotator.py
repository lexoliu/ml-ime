"""The g2pW annotator: a BERT polyphone disambiguator behind an ONNX session.

``g2pw`` is the only untyped, thread-blocking dependency in the annotation path,
so it is confined to this module: the ONNX session runs on a worker thread and
the rest of the pipeline sees nothing but :class:`~mlime.data.g2p.Outcome`.

The model is trained on traditional Chinese and ships its own
simplified-to-traditional table, which is why ``enable_non_tradional_chinese`` is
on -- the corpus is normalised to simplified.

Its dataloader workers are platform-dependent. They deadlock the interpreter at
shutdown on macOS, where the annotator runs over pools of a few thousand
sentences and their absence costs nothing; on Linux -- which is where the corpus
is actually labelled, on a GPU -- the deadlock does not arise, and the batch
preparation they parallelise (tokenising every sentence and building the query
tensors) is a real share of the wall clock next to a CUDA session, so they are
on there.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from mlime.logging import log

from .g2p import Annotator, Outcome, Reading, Refusal
from .text import HAN

#: Where the converter caches its ~200MB ONNX model when nothing else is asked for.
DEFAULT_MODEL_DIR = Path.home() / ".cache" / "mlime" / "G2PWModel"


def default_workers() -> int:
    """How many DataLoader workers g2pW's batch preparation gets on this machine.

    Zero anywhere but Linux: the workers are spawned processes that deadlock the
    interpreter at shutdown on macOS, and the platforms this runs on are macOS
    for development and Linux for the labelling job, so "Linux or nothing" is the
    whole rule.

    On Linux the split is half the cores to the loader and half to the ONNX
    session, which is threaded itself and is the stage that must not starve. The
    T4 machines this labels on have four vCPUs, so that is two workers there,
    and it scales with whatever the box turns out to have.
    """
    if not sys.platform.startswith("linux"):
        return 0
    return max(1, (os.cpu_count() or 1) // 2)


class G2pwAnnotator(Annotator):
    """Per-character pinyin from g2pW, aligned to a sentence's Han characters."""

    def __init__(
        self,
        model_dir: Path = DEFAULT_MODEL_DIR,
        batch_size: int = 32,
        num_workers: int | None = None,
    ):
        from g2pw import G2PWConverter

        workers = default_workers() if num_workers is None else num_workers
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        log.info("loading g2pw", model_dir=str(model_dir), num_workers=workers)
        self._converter = G2PWConverter(
            model_dir=str(model_dir),
            style="pinyin",
            enable_non_tradional_chinese=True,
            num_workers=workers,
            batch_size=batch_size,
            turnoff_tqdm=True,
        )
        # `G2PWConverter.__init__` stores `num_workers if num_workers else
        # self.config.num_workers`, so a 0 passed above is falsy and is silently
        # replaced by the packaged config's 2 -- which on macOS spawns the
        # DataLoader workers that deadlock at shutdown. Assigning afterwards is
        # the only way to mean a number, zero included, without editing the
        # upstream package, so the resolved count is always written here.
        self._converter.num_workers = workers
        self.num_workers = workers

    @property
    def name(self) -> str:
        """Column name this annotator's readings are stored under."""
        return "g2pw"

    async def annotate(self, texts: Sequence[str]) -> list[Outcome]:
        """Run the batch on a worker thread so the event loop stays free."""
        predictions = await asyncio.to_thread(self._converter, list(texts))
        return [self._outcome(text, row) for text, row in zip(texts, predictions, strict=True)]

    def _outcome(self, text: str, predictions: Sequence[str | None]) -> Outcome:
        """Keep the Han positions, refusing the sentence if any of them came back empty."""
        if len(predictions) != len(text):
            return Refusal(f"g2pw returned {len(predictions)} readings for {len(text)} characters")
        syllables = []
        for character, syllable in zip(text, predictions, strict=True):
            if not HAN.match(character):
                continue
            if syllable is None:
                return Refusal(f"g2pw has no reading for {character!r}")
            syllables.append(syllable)
        if not syllables:
            return Refusal("no Han characters to read")
        return Reading(tuple(syllables))
