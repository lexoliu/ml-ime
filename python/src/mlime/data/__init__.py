"""Corpus and lexicon data preparation.

Each stage reads the previous stage's output and writes its own, all under one
root that the caller chooses. Nothing here knows an absolute path: the same
commands run against a checkout's ``data/`` and against ``/kaggle/working``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataLayout:
    """Where each pipeline stage keeps its parquet shards under one data root."""

    root: Path

    @property
    def documents(self) -> Path:
        """Normalised documents, straight off the network."""
        return self.root / "documents"

    @property
    def samples(self) -> Path:
        """Filtered target sentences with their preceding context."""
        return self.root / "samples"

    @property
    def annotations(self) -> Path:
        """Dual g2p annotations, plus the hard and refused subsets."""
        return self.root / "annotations"

    @property
    def lexicon(self) -> Path:
        """Word-pinyin pairs from external dictionaries (Sogou scel, etc.)."""
        return self.root / "lexicon"

    @property
    def scel_cache(self) -> Path:
        """Downloaded .scel files, cached to avoid repeat downloads."""
        return self.root / ".scel-cache"
