//! The packing policy, held to `mlime.train.key_tree.plan` by its model-free
//! fixture.
//!
//! `tests/fixtures/key-tree.json` is what `mlime`'s `write_tree_fixture`
//! writes: for each case, the requested prefixes and the calls the Python
//! policy makes for them. A call's node is given by its last token (`-1` for
//! the root) and its parent row (`-1` for the root), and its answers pair a
//! request index with the row that answers it.

use ime_neural::keys::{POLICY_VERSION, PlanError, TREE_BUCKETS, plan};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    policy_version: u32,
    buckets: Vec<usize>,
    records: Vec<Record>,
}

#[derive(Deserialize)]
struct Record {
    name: String,
    requests: Vec<Vec<usize>>,
    calls: Vec<Call>,
}

#[derive(Deserialize, Debug, PartialEq, Eq)]
struct Call {
    bucket: usize,
    tokens: Vec<i64>,
    parents: Vec<i64>,
    answers: Vec<[usize; 2]>,
}

fn signed(value: Option<usize>) -> i64 {
    value.map_or(-1, |value| i64::try_from(value).expect("a row fits i64"))
}

#[test]
fn the_rust_policy_makes_the_python_calls() {
    let fixture: Fixture =
        serde_json::from_str(include_str!("fixtures/key-tree.json")).expect("the fixture parses");
    assert_eq!(fixture.policy_version, POLICY_VERSION);
    assert_eq!(fixture.buckets, TREE_BUCKETS);
    assert!(!fixture.records.is_empty());
    for record in &fixture.records {
        let calls: Vec<Call> = plan(&record.requests)
            .expect("every fixture case fits the largest bucket")
            .iter()
            .map(|call| {
                for (row, node) in call.nodes().iter().enumerate() {
                    let parent = call.parents()[row];
                    assert_eq!(
                        parent.map(|parent| call.nodes()[parent]),
                        node.split_last().map(|(_, parent)| parent),
                        "{}: a node's parent row holds its prefix's parent",
                        record.name
                    );
                }
                assert_eq!(call.padding(), call.bucket() - call.nodes().len());
                Call {
                    bucket: call.bucket(),
                    tokens: call
                        .nodes()
                        .iter()
                        .map(|node| signed(node.last().copied()))
                        .collect(),
                    parents: call
                        .parents()
                        .iter()
                        .map(|&parent| signed(parent))
                        .collect(),
                    answers: call
                        .answers()
                        .iter()
                        .map(|&(request, row)| [request, row])
                        .collect(),
                }
            })
            .collect();
        assert_eq!(calls, record.calls, "{}", record.name);
    }
}

#[test]
fn a_prefix_deeper_than_the_largest_bucket_is_refused() {
    let largest = TREE_BUCKETS[TREE_BUCKETS.len() - 1];
    let fits: Vec<usize> = vec![0; largest - 1];
    assert_eq!(plan(&[&fits]).map(|calls| calls.len()), Ok(1));
    let deep: Vec<usize> = vec![0; largest];
    assert_eq!(
        plan(&[&deep]),
        Err(PlanError {
            depth: largest,
            largest
        })
    );
}
