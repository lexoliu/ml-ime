"""Find repository-relative files without baking a developer's absolute path in.

The same pipeline runs from a checkout and from a Kaggle or Colab kernel, where
the repository may sit anywhere and some of these files simply do not exist. So
paths are discovered by walking up from the working directory to the repository
root, and their absence is reported by the caller with a message naming the
option to pass instead.
"""

from __future__ import annotations

from pathlib import Path

#: Presence of this entry marks the repository root; the search stops there so a
#: stray file in a home directory can never be picked up instead.
ROOT_MARKER = ".git"


def find_upwards(relative: str | Path, start: Path | None = None) -> Path | None:
    """Return the first existing ``<ancestor>/<relative>`` at or above *start*."""
    here = (start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        candidate = directory / relative
        if candidate.exists():
            return candidate
        if (directory / ROOT_MARKER).exists():
            return None
    return None
