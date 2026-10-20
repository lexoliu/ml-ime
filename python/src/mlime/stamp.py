"""Copies of a committed kernel, configured by rewriting its ``NAME = value`` lines.

A kernel is configured the one way the chain has always configured it: a copy
whose top-level fields are rewritten as literals. :func:`stamp` is that
rewrite, shared by the Colab stager -- which stamps each leg's segment, commit
and session into the copy it runs -- and by :func:`stamp_e2e_v2`, which writes
a v2 run's kernel directory from ``kaggle/e2e``.

Standard library only: the stager imports this from the leg's checkout before
it has installed anything.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: v1's training recipe, field by field, as ``kaggle/e2e/kernel.py`` commits
#: it. v2 trains exactly this -- only the stream under it differs -- so a v2
#: kernel is stamped only from a template that still holds every one of them.
#: ``LEARNING_RATES = None`` is ``mlime train e2e``'s defaults, 3e-5 for the
#: pretrained weights and 1e-4 for the new ones.
V1_RECIPE: dict[str, Any] = {
    "TOKEN_BUDGET": 8192,
    "ACCUMULATE": 4,
    "WORLD": 2,
    "LEARNING_RATES": None,
    "AUX_WEIGHT": 0.3,
    "EPOCHS": 1,
    "COMPILE": True,
    "COMPILE_MODE": "default",
    "INIT_FROM": None,
    "RANKING": None,
    "WHOLE_ALPHABET": False,
    "TYPOS": False,
    "ACTIVATION_CHECKPOINTING": False,
}

#: What a v2 leg mounts, each a Hugging Face dataset repo the stager downloads:
#: the assets, the four shard repos, and the two initialisation checkpoints.
#: Nothing of v1's chain -- its census, its resume datasets, its segments --
#: and nothing for the code, which a leg fetches from GitHub at its commit.
E2E_V2_SOURCES = (
    "lexoliu/mlime-route-a-assets",
    "lexoliu/mlime-run3-v1-samples",
    "lexoliu/mlime-run3-v1-labels",
    "lexoliu/mlime-run3-rest-samples",
    "lexoliu/mlime-run3-rest-labels",
    "lexoliu/mlime-e2e-init",
)

#: Every v2 run is named inside this namespace. A run's name is where its
#: trios live in the resume bucket and what its Kaggle segments are called, so
#: a v2 run can never resume or publish under v1's ``mlime-e2e`` or a stage's
#: name.
E2E_V2_RUN = re.compile(r"mlime-e2e-v2-[a-z0-9]+(-[a-z0-9]+)*")


def literal(source: str, name: str) -> Any:
    """The value of *source*'s top-level ``NAME = <literal>`` assignment."""
    for node in ast.parse(source).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            return ast.literal_eval(node.value)
    raise KeyError(f"the kernel holds no `{name} =` assignment")


def _verbatim(line: str) -> Callable[[re.Match[str]], str]:
    """A replacement that writes *line* as it is: a literal's backslashes are not escapes."""
    return lambda _match: line


def stamp(source: str, **fields: Any) -> str:
    """Rewrite a ``NAME = value`` kernel line per field as the value's literal.

    A field the kernel has no line for is refused: a stamp that lands nowhere
    would leave the kernel's default in force without a word. So is a stamp
    that does not read back as its value -- a field whose default spans lines,
    rewritten on its first one only.
    """
    for key, value in fields.items():
        source, count = re.subn(
            rf"^{key} = .*$",
            _verbatim(f"{key} = {value!r}"),
            source,
            count=1,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise KeyError(f"the kernel holds no `{key} =` line to stamp")
    for key, value in fields.items():
        if literal(source, key) != value:
            raise ValueError(f"`{key}` does not read back as the {value!r} stamped into it")
    return source


@dataclass(frozen=True)
class E2EV2Run:
    """One v2 run's kernel: its name and what its legs measure and train on.

    *profile_steps* and *trace_steps* are the loop's timing window after the
    warm-up (``PROFILE_STEPS``, ``TRACE_STEPS``). *train_shards* narrows the
    epoch to those shards (``TRAIN_SHARDS``): a check that runs a short epoch
    to ``max_steps`` and through the evaluations on a cheap card.
    """

    run: str
    profile_steps: int = 30
    trace_steps: int = 3
    train_shards: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if E2E_V2_RUN.fullmatch(self.run) is None:
            raise ValueError(
                f"{self.run!r} is not a v2 run name: {E2E_V2_RUN.pattern}, such as "
                "mlime-e2e-v2-a100"
            )
        if not 0 <= self.trace_steps <= self.profile_steps:
            raise ValueError(
                f"the trace covers steps of the profile window: 0 <= {self.trace_steps} "
                f"<= {self.profile_steps}"
            )
        if self.train_shards is not None and not self.train_shards:
            raise ValueError("a narrowed epoch names at least one shard")

    @property
    def slug(self) -> str:
        """The kernel's slug: the run's first segment, the one every Colab leg carries."""
        return f"lexoliu/{self.run}-s0"

    def fields(self) -> dict[str, Any]:
        """The kernel fields this run stamps."""
        return {
            "PROFILE_STEPS": self.profile_steps,
            "TRACE_STEPS": self.trace_steps,
            "TRAIN_SHARDS": self.train_shards,
        }


def stamp_e2e_v2(template: Path, out: Path, run: E2EV2Run) -> dict[str, Any]:
    """Write *run*'s kernel directory from the e2e kernel in *template*; return its recipe.

    The copy differs from the template only in the run's own fields, and its
    metadata names the run's slug and :data:`E2E_V2_SOURCES` alone. A template
    whose recipe is no longer v1's is refused by field, since v2 is defined as
    v1's recipe on a different stream.
    """
    source = (template / "kernel.py").read_text(encoding="utf-8")
    recipe = {name: literal(source, name) for name in V1_RECIPE}
    drifted = sorted(name for name, value in recipe.items() if value != V1_RECIPE[name])
    if drifted:
        raise ValueError(f"{template / 'kernel.py'} is no longer v1's recipe in {drifted}")
    metadata = json.loads((template / "kernel-metadata.json").read_text(encoding="utf-8"))
    metadata.update(
        id=run.slug,
        title=run.slug.split("/")[1],
        dataset_sources=list(E2E_V2_SOURCES),
        kernel_sources=[],
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "kernel.py").write_text(stamp(source, **run.fields()), encoding="utf-8")
    (out / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return recipe
