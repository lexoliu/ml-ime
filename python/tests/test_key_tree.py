"""The decoder step's packing policy and its model-free fixture (#266)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from mlime.train.key_tree import (
    POLICY_VERSION,
    TREE_BUCKETS,
    TREE_FIXTURE_RELATIVE,
    Prefix,
    TreeCall,
    closure_size,
    fixture_cases,
    plan,
    write_tree_fixture,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / TREE_FIXTURE_RELATIVE


def check_call(call: TreeCall) -> None:
    """The invariants every call holds, whatever it was asked."""
    nodes = call.nodes
    assert len(nodes) <= call.bucket and call.bucket in TREE_BUCKETS
    assert call.bucket == next(size for size in TREE_BUCKETS if size >= len(nodes))
    assert list(nodes) == sorted(nodes, key=lambda prefix: (len(prefix), prefix))
    assert len(set(nodes)) == len(nodes)
    assert nodes[0] == () and call.parents[0] == -1
    for row, node in enumerate(nodes):
        if row:
            assert nodes[call.parents[row]] == node[:-1]
            assert call.parents[row] < row
    assert set(nodes) == {node[:length] for node in nodes for length in range(len(node) + 1)}
    for row in range(call.bucket):
        seen = call.visible(row)
        assert row in seen
        if row < len(nodes):
            assert {nodes[i] for i in seen} == {
                nodes[row][:length] for length in range(len(nodes[row]) + 1)
            }
        else:
            assert seen == (row,)


def test_no_request_makes_no_call() -> None:
    assert plan([]) == []


def test_a_request_is_answered_by_its_own_row() -> None:
    requests: list[Prefix] = [(2, 1), (0,), (2, 1), (), (2,), (0,)]
    [call] = plan(requests)
    check_call(call)
    assert call.bucket == 32 and call.padding == 32 - len(call.nodes)
    assert call.nodes == ((), (0,), (2,), (2, 1))
    assert call.parents == (-1, 0, 0, 2)
    assert call.depths == (0, 1, 1, 2)
    answered = {request: call.nodes[row] for request, row in call.answers}
    assert answered == dict(enumerate(requests))
    assert call.visible(3) == (0, 2, 3)


def test_the_bucket_is_the_smallest_that_holds_the_closure() -> None:
    for chain in (31, 32, 63, 64, 127, 128, 255):
        prefix = tuple(range(chain))
        [call] = plan([prefix])
        assert len(call.nodes) == chain + 1
        assert call.bucket == next(size for size in TREE_BUCKETS if size >= chain + 1), chain
        check_call(call)


def test_a_set_past_the_largest_bucket_splits_in_depth_order() -> None:
    chains = [tuple(chain * 1000 + step for step in range(120)) for chain in range(8)]
    calls = plan(chains)
    assert [len(call.nodes) for call in calls] == [241, 241, 241, 241]
    assert all(call.bucket == 256 for call in calls)
    for call in calls:
        check_call(call)
    # Every request is answered exactly once, by the call that was filled with it.
    answered = sorted(request for call in calls for request, _ in call.answers)
    assert answered == list(range(8))


def test_a_prefix_that_does_not_fit_starts_the_next_call() -> None:
    deep = [tuple(range(chain * 100, chain * 100 + 130)) for chain in range(2)]
    shallow = (9999,)
    calls = plan([shallow, *deep])
    # Sorted by depth: the shallow prefix, then the two chains; the second chain does not fit.
    assert [len(call.nodes) for call in calls] == [1 + 1 + 130, 131]
    assert [sorted(r for r, _ in call.answers) for call in calls] == [[0, 1], [2]]


def test_a_prefix_deeper_than_the_largest_bucket_is_refused() -> None:
    with pytest.raises(ValueError, match="needs 257 nodes"):
        plan([tuple(range(256))])
    plan([tuple(range(255))])
    with pytest.raises(ValueError, match="at least one bucket"):
        plan([()], buckets=())


def test_the_plan_does_not_depend_on_the_order_or_repeats_of_the_requests() -> None:
    rng = random.Random(5)
    requests = [tuple(rng.randrange(9) for _ in range(rng.randint(0, 7))) for _ in range(40)]
    shuffled = requests[:]
    rng.shuffle(shuffled)
    first = plan(requests)
    second = plan(shuffled)
    assert [(c.bucket, c.nodes, c.parents) for c in first] == [
        (c.bucket, c.nodes, c.parents) for c in second
    ]
    for calls, asked in ((first, requests), (second, shuffled)):
        for call in calls:
            check_call(call)
            for request, row in call.answers:
                assert call.nodes[row] == asked[request]
        assert sorted(r for call in calls for r, _ in call.answers) == list(range(len(asked)))


def test_a_different_ladder_changes_only_the_buckets() -> None:
    requests: list[Prefix] = [(1, 2), (1, 3), (4,)]
    [small] = plan(requests, buckets=(8, 16))
    [big] = plan(requests, buckets=(32,))
    assert small.nodes == big.nodes and small.parents == big.parents
    assert (small.bucket, big.bucket) == (8, 32)


def test_closure_size_counts_the_root_and_shares_ancestors() -> None:
    assert closure_size([]) == 0
    assert closure_size([()]) == 1
    assert closure_size([(1, 2), (1, 3)]) == 4


# --- the fixture -----------------------------------------------------------------


def test_the_fixture_is_current(tmp_path: Path) -> None:
    """The committed file is byte for byte what this module writes."""
    regenerated = tmp_path / "key-tree.json"
    calls = write_tree_fixture(regenerated)
    assert calls > 0
    assert regenerated.read_bytes() == FIXTURE.read_bytes(), (
        "stale: run `uv run --project python mlime gen-key-tree-fixture`"
    )


def test_the_fixture_records_the_policy_it_pins() -> None:
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert document["policy_version"] == POLICY_VERSION
    assert document["buckets"] == list(TREE_BUCKETS)
    assert [record["name"] for record in document["records"]] == [n for n, _ in fixture_cases()]
    names = {record["name"] for record in document["records"]}
    for bucket in TREE_BUCKETS:
        assert {f"fills-{bucket}", f"above-{bucket}"} <= names
    by_name = {record["name"]: record for record in document["records"]}
    assert by_name["none"]["calls"] == []
    assert [call["bucket"] for call in by_name["above-32"]["calls"]] == [64]
    # The chain's 256 nodes and the root-sibling's 2 do not share a call: 257 > 256.
    assert [call["bucket"] for call in by_name["above-256"]["calls"]] == [32, 256]
    assert [len(call["tokens"]) for call in by_name["four-long-chains"]["calls"]] == [201, 201]
    for record in document["records"]:
        requests = [tuple(request) for request in record["requests"]]
        for entry in record["calls"]:
            # A consumer rebuilds each prefix from its token and its parent's prefix.
            nodes: list[Prefix] = []
            for token, parent in zip(entry["tokens"], entry["parents"], strict=True):
                nodes.append(() if parent < 0 else (*nodes[parent], token))
            call = TreeCall(
                bucket=entry["bucket"],
                nodes=tuple(nodes),
                parents=tuple(entry["parents"]),
                answers=tuple(tuple(pair) for pair in entry["answers"]),
            )
            check_call(call)
            for request, row in call.answers:
                assert call.nodes[row] == requests[request]
        assert [e["answers"] for e in record["calls"]] == [
            [list(p) for p in c.answers] for c in plan(requests)
        ]
    assert FIXTURE.stat().st_size < 1_000_000
