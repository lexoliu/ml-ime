"""Copies of a committed kernel, configured by rewriting its ``NAME = value`` lines.

A kernel is configured the one way the chain has always configured it: a copy
whose top-level fields are rewritten as literals. :func:`stamp` is that
rewrite, shared by the Colab stager -- which stamps each leg's segment, commit
and session into the copy it runs -- and by :func:`stamp_e2e_v2`, which writes
a v2 run's kernel directory from ``kaggle/e2e``, or a v2 stage's: a run that
starts from a finished v2 run with its own ranking, shards and schedule.

Standard library only: the stager imports this from the leg's checkout before
it has installed anything. ``RankingConfig`` is named for the type checker
alone.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlime.train.e2e_model import RankingConfig

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
    "CONTEXT_TOWER": None,
    "CONTEXT_WARM_UP": False,
    "KEYS": None,
    "KEY_WARM_UP": False,
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

#: Every v2 run is named inside this namespace: ``mlime-e2e-v2-*`` for the span
#: model's runs, ``mlime-e2e-keys-*`` for the key-level model's (#223). A run's
#: name is where its trios live in the resume bucket and what its Kaggle
#: segments are called, so a v2 run can never resume or publish under v1's
#: ``mlime-e2e`` or a stage's name, and a key-level run never under a span
#: model's.
E2E_V2_RUN = re.compile(r"mlime-e2e-(v2|keys)-[a-z0-9]+(-[a-z0-9]+)*")

#: The namespace of the key-level model's runs.
E2E_KEYS_RUN = "mlime-e2e-keys-"

#: What a stamped ``CONTEXT_TOWER`` holds: the kernel's ``--context-*`` options.
CONTEXT_TOWER_FIELDS = frozenset({"tower", "revision", "lora_rank", "lora_alpha"})

#: What a stamped ``KEYS`` holds: the kernel's key front options (#223).
KEYS_FIELDS = frozenset({"layers", "distill_weight", "correct_syllables"})


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


def _v2_name(run: str) -> None:
    """Refuse *run* unless it is named inside the v2 namespace."""
    if E2E_V2_RUN.fullmatch(run) is None:
        raise ValueError(
            f"{run!r} is not a v2 run name: {E2E_V2_RUN.pattern}, such as mlime-e2e-v2-a100 "
            "or mlime-e2e-keys-s1"
        )


@dataclass(frozen=True)
class E2EV2Stage:
    """What a v2 stage changes of v1's recipe: where it starts and how it trains.

    A stage starts from the finished v2 run *init_from* (``INIT_FROM``) -- its
    weights, with a fresh optimiser and schedule -- and with *ranking* adds the
    sequence-level ranking loss (``RANKING``), which only a stage can: the
    ranking loss is trained on top of a finished run, never from route A and
    the reader. *learning_rates* (``LEARNING_RATES``) is the (pretrained, new)
    pair, None for ``mlime train e2e``'s defaults; *token_budget* and
    *accumulate* default to v1's. *context_tower* (``CONTEXT_TOWER``) reads
    the context with a Qwen model, swapped in over *init_from*'s MacBERT tower
    (#222): its ``tower`` and ``revision``, and a ``lora_rank`` and
    ``lora_alpha`` where 0 keeps it frozen. *context_warm_up*
    (``CONTEXT_WARM_UP``) trains only the context path, which is S0.
    *keys* (``KEYS``) reads keystrokes through a key front with a per-key CTC
    head (#223): its ``layers``, ``distill_weight`` and ``correct_syllables``.
    *key_warm_up* (``KEY_WARM_UP``) trains only the front and the head,
    distilling *init_from*'s encoder states, which is S1.
    """

    init_from: str
    ranking: RankingConfig | None = None
    learning_rates: tuple[float, float] | None = None
    token_budget: int = V1_RECIPE["TOKEN_BUDGET"]
    accumulate: int = V1_RECIPE["ACCUMULATE"]
    context_tower: dict[str, Any] | None = None
    context_warm_up: bool = False
    keys: dict[str, Any] | None = None
    key_warm_up: bool = False

    def __post_init__(self) -> None:
        _v2_name(self.init_from)
        if self.context_tower is not None and set(self.context_tower) != CONTEXT_TOWER_FIELDS:
            raise ValueError(
                f"a context tower names {sorted(CONTEXT_TOWER_FIELDS)}, "
                f"got {sorted(self.context_tower)}"
            )
        if self.keys is not None:
            if set(self.keys) != KEYS_FIELDS:
                raise ValueError(
                    f"a key front names {sorted(KEYS_FIELDS)}, got {sorted(self.keys)}"
                )
            if self.ranking is not None or self.context_warm_up:
                raise ValueError(
                    "a key-level stage ranks no lattice readings and warms up only its key front"
                )
            if not self.key_warm_up:
                raise ValueError("a key front enters through its warm-up (S1); stamp KEY_WARM_UP")
        elif self.key_warm_up:
            raise ValueError("the key warm-up trains a key front; name one with keys")

    def fields(self) -> dict[str, Any]:
        """The recipe fields this stage stamps over v1's."""
        return {
            "INIT_FROM": self.init_from,
            "RANKING": None if self.ranking is None else asdict(self.ranking),
            "LEARNING_RATES": self.learning_rates,
            "TOKEN_BUDGET": self.token_budget,
            "ACCUMULATE": self.accumulate,
            "CONTEXT_TOWER": self.context_tower,
            "CONTEXT_WARM_UP": self.context_warm_up,
            "KEYS": self.keys,
            "KEY_WARM_UP": self.key_warm_up,
        }


@dataclass(frozen=True)
class E2EV2Run:
    """One v2 run's kernel: its name and what its legs measure and train on.

    *profile_steps* and *trace_steps* are the loop's timing window after the
    warm-up (``PROFILE_STEPS``, ``TRACE_STEPS``), opening at the run's step
    *profile_start* at the earliest (``PROFILE_START``). *train_shards* narrows the
    epoch to those shards (``TRAIN_SHARDS``): a check that runs a short epoch
    to ``max_steps`` and through the evaluations on a cheap card, or a stage's
    pick of the corpus. *stage* makes the run a stage; without one it trains
    v1's recipe exactly. *search_graphs* replays a ranking stage's search from
    captured CUDA graphs (``SEARCH_GRAPHS``): how the run computes, not what it
    trains, so it is no part of the recipe. *tower_graphs* replays a frozen
    Qwen context tower's forward from captured CUDA graphs (``TOWER_GRAPHS``),
    the same kind of choice.
    """

    run: str
    profile_steps: int = 30
    profile_start: int = 0
    trace_steps: int = 3
    train_shards: tuple[str, ...] | None = None
    stage: E2EV2Stage | None = None
    search_graphs: bool = False
    tower_graphs: bool = False

    def __post_init__(self) -> None:
        _v2_name(self.run)
        if self.stage is not None and self.stage.init_from == self.run:
            raise ValueError(
                f"{self.run} cannot start from itself: a stage publishes under its own name"
            )
        keyed = self.stage is not None and self.stage.keys is not None
        if keyed != self.run.startswith(E2E_KEYS_RUN):
            raise ValueError(
                f"{self.run}: a key-level run is named {E2E_KEYS_RUN}*, and only a key-level run is"
            )
        if self.search_graphs and (self.stage is None or self.stage.ranking is None):
            raise ValueError("only a ranking stage has a search to replay from CUDA graphs")
        tower = None if self.stage is None else self.stage.context_tower
        if self.tower_graphs and (tower is None or tower["lora_rank"]):
            raise ValueError("only a frozen Qwen context tower has a forward to replay")
        if self.profile_start < 0:
            raise ValueError(f"the window opens at a step, not {self.profile_start}")
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
        """The kernel fields this run stamps beside its recipe."""
        return {
            "PROFILE_STEPS": self.profile_steps,
            "PROFILE_START": self.profile_start,
            "TRACE_STEPS": self.trace_steps,
            "TRAIN_SHARDS": self.train_shards,
            "SEARCH_GRAPHS": self.search_graphs,
            "TOWER_GRAPHS": self.tower_graphs,
        }

    def recipe(self) -> dict[str, Any]:
        """The recipe this run trains: v1's, with a stage's fields over it."""
        return {**V1_RECIPE, **({} if self.stage is None else self.stage.fields())}


def stamp_e2e_v2(template: Path, out: Path, run: E2EV2Run) -> dict[str, Any]:
    """Write *run*'s kernel directory from the e2e kernel in *template*; return its recipe.

    The copy differs from the template only in the run's own fields and a
    stage's, and its metadata names the run's slug and :data:`E2E_V2_SOURCES`
    alone. A template whose recipe is no longer v1's is refused by field,
    since v2 is defined as v1's recipe on a different stream and a stage as
    v1's recipe with its own fields over it. The recipe returned is the one
    read back from the copy.
    """
    source = (template / "kernel.py").read_text(encoding="utf-8")
    committed = {name: literal(source, name) for name in V1_RECIPE}
    drifted = sorted(name for name, value in committed.items() if value != V1_RECIPE[name])
    if drifted:
        raise ValueError(f"{template / 'kernel.py'} is no longer v1's recipe in {drifted}")
    stage = {} if run.stage is None else run.stage.fields()
    kernel = stamp(source, **run.fields(), **stage)
    metadata = json.loads((template / "kernel-metadata.json").read_text(encoding="utf-8"))
    metadata.update(
        id=run.slug,
        title=run.slug.split("/")[1],
        dataset_sources=list(E2E_V2_SOURCES),
        kernel_sources=[],
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "kernel.py").write_text(kernel, encoding="utf-8")
    (out / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return {name: literal(kernel, name) for name in V1_RECIPE}
