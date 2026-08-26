"""The g2pW annotator: a BERT polyphone disambiguator behind an ONNX session.

``g2pw`` is the only untyped, thread-blocking dependency in the annotation path,
so it is confined to this module: the ONNX session runs on a worker thread and
the rest of the pipeline sees nothing but :class:`~mlime.data.g2p.Outcome`.

The model is trained on traditional Chinese and ships its own
simplified-to-traditional table, which is why ``enable_non_tradional_chinese`` is
on -- the corpus is normalised to simplified. Its dataloader workers are disabled
because they deadlock the interpreter at shutdown on macOS; the ONNX session is
already internally threaded, so they bought nothing.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

from mlime.logging import log

from .g2p import Annotator, Outcome, Reading, Refusal
from .text import HAN

#: Where the converter caches its ~200MB ONNX model when nothing else is asked for.
DEFAULT_MODEL_DIR = Path.home() / ".cache" / "mlime" / "G2PWModel"


class G2pwAnnotator(Annotator):
    """Per-character pinyin from g2pW, aligned to a sentence's Han characters."""

    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR, batch_size: int = 32):
        from g2pw import G2PWConverter

        model_dir.parent.mkdir(parents=True, exist_ok=True)
        log.info("loading g2pw", model_dir=str(model_dir))
        self._converter = G2PWConverter(
            model_dir=str(model_dir),
            style="pinyin",
            enable_non_tradional_chinese=True,
            num_workers=0,
            batch_size=batch_size,
            turnoff_tqdm=True,
        )
        # `G2PWConverter.__init__` stores `num_workers if num_workers else
        # self.config.num_workers`, so the 0 above is falsy and is silently
        # replaced by the packaged config's 2. That spawns DataLoader worker
        # processes, which is both pointless here -- `annotate` already runs the
        # batch on a worker thread -- and fragile, because the spawned workers
        # break on macOS. Setting the attribute after construction is the only
        # way to mean zero without editing the upstream package.
        self._converter.num_workers = 0

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
