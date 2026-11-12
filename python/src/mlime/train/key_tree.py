"""The decoder step's packing policy: which prefixes one tree call carries (#266).

The key model's decoder runs on Core ML as a fixed-shape *tree recompute*: one
call computes the next-character row of every node of a packed, ancestor-closed
set of prefixes, a node attending to its ancestors and itself
(:meth:`~mlime.train.key_model.KeyE2EModel.tree_log_probs`). Which prefixes
share a call, in which node order and in which bucket is the **contract**
between the Python reference and the Rust session: a row's fp16 value may
depend at the last bit on its bucket and its neighbours, so both sides must
make exactly the same calls. This module is that policy as pure code, with no
torch in it, and :func:`write_tree_fixture` pins it in a model-free fixture the
Rust side reproduces.

**The policy** (:func:`plan`, version :data:`POLICY_VERSION`):

1. The requested prefixes are deduplicated and sorted by ``(depth, ids)``.
2. A call is filled greedily in that order: a prefix is added with every
   ancestor it does not yet share, root included, while the call's node count
   stays within the largest bucket. The one that does not fit starts the next
   call, which holds that prefix's ancestors and itself.
3. A call's nodes are in ``(depth, ids)`` order, so every parent precedes its
   children and the root comes first.
4. The call runs in the smallest bucket that holds its nodes; the bucket's
   remaining rows are padding, last. A padding node sits at depth 0 and sees
   only itself, so it changes no real row.

A prefix is a tuple of emission indices; ``()`` is the root, whose next
character is the sentence's first.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: A prefix: the emission indices of the characters emitted so far.
Prefix = tuple[int, ...]

#: The tree function family's node buckets.
TREE_BUCKETS: tuple[int, ...] = (32, 64, 128, 256)

#: Bumped whenever :func:`plan` may answer differently for the same requests.
POLICY_VERSION = 1

#: Where the fixture lives, relative to the repository root.
TREE_FIXTURE_RELATIVE = Path("crates/ime-neural/tests/fixtures/key-tree.json")


@dataclass(frozen=True)
class TreeCall:
    """One tree call: its nodes, the bucket that runs them, and which node answers which request.

    ``nodes`` is ancestor-closed and in ``(depth, ids)`` order; ``parents[i]``
    is the index of node ``i``'s parent, or ``-1`` for the root. ``answers``
    pairs each request this call answers (an index into :func:`plan`'s
    requests) with the node whose row is its answer. The ``bucket - len(nodes)``
    rows after the nodes are padding.
    """

    bucket: int
    nodes: tuple[Prefix, ...]
    parents: tuple[int, ...]
    answers: tuple[tuple[int, int], ...]

    @property
    def padding(self) -> int:
        """Rows of the bucket that carry no node."""
        return self.bucket - len(self.nodes)

    @property
    def depths(self) -> tuple[int, ...]:
        """Each node's depth, its prefix length: the decoder's position there."""
        return tuple(len(node) for node in self.nodes)

    def visible(self, row: int) -> tuple[int, ...]:
        """The rows *row* attends to, ascending: its ancestors and itself.

        A padding row sees only itself.
        """
        if row >= len(self.nodes):
            return (row,)
        seen = [row]
        while self.parents[seen[-1]] >= 0:
            seen.append(self.parents[seen[-1]])
        return tuple(sorted(seen))


def closure_size(prefixes: Sequence[Prefix]) -> int:
    """Nodes in the ancestor closure of *prefixes*, root included."""
    return len({prefix[:length] for prefix in prefixes for length in range(len(prefix) + 1)})


def plan(requests: Sequence[Prefix], buckets: Sequence[int] = TREE_BUCKETS) -> list[TreeCall]:
    """The tree calls that answer *requests*, by the policy above.

    A prefix whose own ancestor chain exceeds the largest bucket cannot be
    answered and raises. No request yields no call.
    """
    ladder = sorted(buckets)
    if not ladder:
        raise ValueError("a plan needs at least one bucket")
    largest = ladder[-1]
    wanted = sorted(set(requests), key=lambda prefix: (len(prefix), prefix))
    for prefix in wanted:
        if len(prefix) + 1 > largest:
            raise ValueError(
                f"a prefix of {len(prefix)} characters needs {len(prefix) + 1} nodes; "
                f"the largest bucket holds {largest}"
            )
    filled: list[tuple[set[Prefix], list[Prefix]]] = []  # (nodes, requested prefixes) per call
    nodes: set[Prefix] = set()
    asked: list[Prefix] = []
    for prefix in wanted:
        missing = {prefix[:length] for length in range(len(prefix) + 1)} - nodes
        if len(nodes) + len(missing) > largest:
            filled.append((nodes, asked))
            nodes, asked = set(), []
            missing = {prefix[:length] for length in range(len(prefix) + 1)}
        nodes |= missing
        asked.append(prefix)
    if asked:
        filled.append((nodes, asked))
    wanted_at: dict[Prefix, list[int]] = {}
    for index, prefix in enumerate(requests):
        wanted_at.setdefault(prefix, []).append(index)
    calls = []
    for members, asked in filled:
        ordered = sorted(members, key=lambda prefix: (len(prefix), prefix))
        row = {prefix: index for index, prefix in enumerate(ordered)}
        calls.append(
            TreeCall(
                bucket=next(size for size in ladder if size >= len(ordered)),
                nodes=tuple(ordered),
                parents=tuple(row[prefix[:-1]] if prefix else -1 for prefix in ordered),
                answers=tuple(
                    (request, row[prefix]) for prefix in asked for request in wanted_at[prefix]
                ),
            )
        )
    return calls


# --- the model-free fixture ------------------------------------------------------

#: The fixture's own account of what it pins.
DEFINITIONS: dict[str, str] = {
    "prefix": "A prefix is the list of emission indices emitted so far; [] is the root.",
    "plan": (
        "Dedupe the requests and sort them by (depth, ids), depth the length and ids compared "
        "as lists. Fill a call in that order: add a prefix with every ancestor not yet in the "
        "call (root included) while the call holds at most the largest bucket's nodes; the "
        "prefix that does not fit starts the next call with its own ancestors and itself. "
        "A call's nodes are sorted by (depth, ids) and run in the smallest bucket that holds "
        "them; the rest of the bucket is padding."
    ),
    "calls": (
        "'tokens' are the call's nodes in row order, each as its prefix's last emission index "
        "(-1 for the root), and 'parents' each node's parent row (-1 for the root), which "
        "together give every node's prefix; 'answers' pairs a request index with the row that "
        "answers it, in node order and then request order. A node sees its ancestors and itself; "
        "a padding row only itself."
    ),
}

#: The scripted cases: a name and the requested prefixes.
_SCRIPTED: tuple[tuple[str, tuple[Prefix, ...]], ...] = (
    ("none", ()),
    ("root", ((),)),
    ("single", ((4,),)),
    ("chain", ((), (3,), (3, 1), (3, 1, 2))),
    ("unsorted-and-repeated", ((2, 1), (0,), (2, 1), (), (2,), (0,))),
    ("siblings", tuple((index,) for index in range(7, -1, -1))),
    ("beam-of-eight", tuple((1, 2, index) for index in range(8))),
)

#: Fixed seeds of the generated cases, each with how many prefixes it asks for, up to
#: what depth, over how many emission indices.
_GENERATED: tuple[tuple[str, int, int, int, int], ...] = (
    ("random-small", 11, 6, 5, 9),
    ("random-wide", 12, 20, 12, 40),
    ("random-deep", 13, 8, 60, 7000),
)


def _chains(count: int, depth: int, width: int) -> tuple[Prefix, ...]:
    """*count* disjoint chains of *depth* characters, the deepest prefix of each."""
    return tuple(
        tuple((chain * 7 + step) % width for step in range(depth)) for chain in range(count)
    )


def _random_requests(seed: int, count: int, depth: int, width: int) -> tuple[Prefix, ...]:
    """*count* prefixes from the seed; each grows from an earlier one half the time."""
    rng = random.Random(seed)
    made: list[Prefix] = []
    for _ in range(count):
        if made and rng.random() < 0.5:
            base = rng.choice(made)
            extra = rng.randint(1, 3)
            prefix = (*base, *(rng.randrange(width) for _ in range(extra)))[:depth]
        else:
            prefix = tuple(rng.randrange(width) for _ in range(rng.randint(0, depth)))
        made.append(prefix)
    return tuple(made)


def fixture_cases() -> list[tuple[str, tuple[Prefix, ...]]]:
    """Every case of the fixture: the scripted ones, the seeded ones and the bucket edges.

    The edges are chosen by closure size: exactly a bucket's nodes and one
    more, for every bucket (the one above the largest splits into two calls),
    plus sets that need several calls.
    """
    cases = list(_SCRIPTED)
    for name, seed, count, depth, width in _GENERATED:
        cases.append((name, _random_requests(seed, count, depth, width)))
    for bucket in TREE_BUCKETS:
        # One chain of `bucket - 1` characters has exactly `bucket` nodes with its root.
        edge = _chains(1, bucket - 1, 1000)
        cases.append((f"fills-{bucket}", edge))
        cases.append((f"above-{bucket}", (*edge, (999,))))
    cases.append(("eight-deep-chains", _chains(8, 45, 7322)))
    cases.append(("four-long-chains", _chains(4, 100, 7322)))
    return cases


def build_tree_fixture() -> dict[str, Any]:
    """The fixture's document: the policy's account and, per case, requests and calls."""
    records = []
    for name, requests in fixture_cases():
        records.append(
            {
                "name": name,
                "requests": [list(prefix) for prefix in requests],
                "calls": [
                    {
                        "bucket": call.bucket,
                        "tokens": [node[-1] if node else -1 for node in call.nodes],
                        "parents": list(call.parents),
                        "answers": [list(pair) for pair in call.answers],
                    }
                    for call in plan(requests)
                ],
            }
        )
    return {
        "definitions": DEFINITIONS,
        "policy_version": POLICY_VERSION,
        "buckets": list(TREE_BUCKETS),
        "records": records,
    }


def write_tree_fixture(out: Path) -> int:
    """Write the shared tree-policy fixture to *out*; the number of calls it pins.

    One line per record keeps the file diffable, and the bytes depend on
    nothing but this module.
    """
    document = build_tree_fixture()
    lines = ["{"]
    for name in ("definitions", "policy_version", "buckets"):
        lines.append(f"{json.dumps(name)}: {json.dumps(document[name], ensure_ascii=False)},")
    lines.append('"records": [')
    lines.append(
        ",\n".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            for record in document["records"]
        )
    )
    lines += ["]", "}"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sum(len(record["calls"]) for record in document["records"])
