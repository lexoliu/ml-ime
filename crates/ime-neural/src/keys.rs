//! The key-level model's decoder contract: how prefixes are packed into tree
//! calls, and what a keystroke asks the model for.
//!
//! The decoder runs as a fixed-shape *tree recompute*: one call computes the
//! next-character row of every node of a packed, ancestor-closed set of
//! prefixes, a node attending to its ancestors and itself. Which prefixes
//! share a call, in which node order and in which bucket is the contract
//! between the Python reference (`mlime.train.key_tree`) and the Rust session,
//! because a row's fp16 value may depend at the last bit on its bucket and its
//! neighbours. [`plan`] is that policy, version [`POLICY_VERSION`], held to the
//! Python side by `tests/fixtures/key-tree.json`:
//!
//! 1. The requested prefixes are deduplicated and sorted by `(depth, ids)`.
//! 2. A call is filled greedily in that order: a prefix is added with every
//!    ancestor it does not yet share, root included, while the call's node
//!    count stays within the largest bucket. The one that does not fit starts
//!    the next call, which holds that prefix's ancestors and itself.
//! 3. A call's nodes are in `(depth, ids)` order, so every parent precedes its
//!    children and the root comes first.
//! 4. The call runs in the smallest bucket that holds its nodes; the bucket's
//!    remaining rows are padding, last.
//!
//! [`KeyModel`] is what one keystroke of the key engine needs from a model:
//! the CTC log-probs of the typed keys under the context, and the rows of a
//! planned tree call.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use ime_pinyin::{EmitId, Keys};
use thiserror::Error;

/// The tree function family's node buckets, ascending.
pub const TREE_BUCKETS: [usize; 4] = [32, 64, 128, 256];

/// Bumped whenever [`plan`] may answer differently for the same requests.
pub const POLICY_VERSION: u32 = 1;

/// Why requests cannot be planned.
#[derive(Debug, Error, PartialEq, Eq)]
#[error(
    "a prefix of {depth} characters needs {} nodes; the largest bucket holds {largest}",
    depth + 1
)]
pub struct PlanError {
    /// The characters of the prefix that does not fit.
    pub depth: usize,
    /// The largest bucket's nodes.
    pub largest: usize,
}

/// One tree call: its nodes, the bucket that runs them, and which node answers
/// which request.
///
/// A node is a prefix of the token type `T` (an emission id in the session),
/// borrowed from the requests it was planned from; the empty prefix is the
/// root, whose next character is the sentence's first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreeCall<'r, T> {
    bucket: usize,
    nodes: Vec<&'r [T]>,
    parents: Vec<Option<usize>>,
    answers: Vec<(usize, usize)>,
}

impl<'r, T> TreeCall<'r, T> {
    /// The bucket the call runs in: the smallest of [`TREE_BUCKETS`] that
    /// holds its nodes.
    #[must_use]
    pub const fn bucket(&self) -> usize {
        self.bucket
    }

    /// The call's nodes in row order: ancestor-closed, sorted by
    /// `(depth, ids)`, the root first.
    #[must_use]
    pub fn nodes(&self) -> &[&'r [T]] {
        &self.nodes
    }

    /// Each node's parent row, `None` for the root.
    #[must_use]
    pub fn parents(&self) -> &[Option<usize>] {
        &self.parents
    }

    /// `(request, row)`: each request this call answers, as an index into the
    /// planned requests, with the node whose row answers it; in node order,
    /// then request order.
    #[must_use]
    pub fn answers(&self) -> &[(usize, usize)] {
        &self.answers
    }

    /// The bucket's rows after the nodes, which carry no node.
    #[must_use]
    pub const fn padding(&self) -> usize {
        self.bucket - self.nodes.len()
    }
}

/// Every prefix of *prefix*, the root and *prefix* included, shortest first.
fn ancestors<T>(prefix: &[T]) -> impl Iterator<Item = &[T]> {
    (0..=prefix.len()).map(move |length| &prefix[..length])
}

/// `(depth, ids)` order: the policy's sort key.
fn by_depth<T: Ord>(a: &[T], b: &[T]) -> std::cmp::Ordering {
    a.len().cmp(&b.len()).then_with(|| a.cmp(b))
}

/// One call being filled: its nodes, and the requested prefixes it answers
/// in fill order.
struct Fill<'r, T> {
    nodes: HashSet<&'r [T]>,
    asked: Vec<&'r [T]>,
}

impl<T> Default for Fill<'_, T> {
    fn default() -> Self {
        Self {
            nodes: HashSet::new(),
            asked: Vec::new(),
        }
    }
}

/// The tree calls that answer *requests*, by the policy of version
/// [`POLICY_VERSION`]. No request yields no call.
///
/// # Errors
///
/// If a prefix's own ancestor chain is longer than the largest bucket.
pub fn plan<T, P>(requests: &[P]) -> Result<Vec<TreeCall<'_, T>>, PlanError>
where
    T: Copy + Ord + Hash,
    P: AsRef<[T]>,
{
    let largest = TREE_BUCKETS[TREE_BUCKETS.len() - 1];
    let mut wanted: Vec<&[T]> = requests.iter().map(AsRef::as_ref).collect();
    wanted.sort_by(|a, b| by_depth(a, b));
    wanted.dedup();
    if let Some(deepest) = wanted.last()
        && deepest.len() + 1 > largest
    {
        return Err(PlanError {
            depth: deepest.len(),
            largest,
        });
    }
    // The requests' indices per prefix, in request order.
    let mut wanted_at: HashMap<&[T], Vec<usize>> = HashMap::new();
    for (index, request) in requests.iter().enumerate() {
        wanted_at.entry(request.as_ref()).or_default().push(index);
    }
    let mut filled: Vec<Fill<'_, T>> = Vec::new();
    let mut fill = Fill::default();
    for prefix in wanted {
        let missing = ancestors(prefix)
            .filter(|ancestor| !fill.nodes.contains(ancestor))
            .count();
        if fill.nodes.len() + missing > largest {
            filled.push(std::mem::take(&mut fill));
        }
        fill.nodes.extend(ancestors(prefix));
        fill.asked.push(prefix);
    }
    if !fill.asked.is_empty() {
        filled.push(fill);
    }
    Ok(filled
        .into_iter()
        .map(|Fill { nodes, asked }| {
            let mut ordered: Vec<&[T]> = nodes.into_iter().collect();
            ordered.sort_by(|a, b| by_depth(a, b));
            let row: HashMap<&[T], usize> = ordered
                .iter()
                .enumerate()
                .map(|(index, &prefix)| (prefix, index))
                .collect();
            // The smallest bucket that holds the nodes; the fill kept them
            // within the largest.
            let bucket = TREE_BUCKETS[TREE_BUCKETS.partition_point(|&size| size < ordered.len())];
            let parents = ordered
                .iter()
                .map(|prefix| prefix.split_last().map(|(_, parent)| row[parent]))
                .collect();
            let answers = asked
                .iter()
                .flat_map(|prefix| {
                    wanted_at[prefix]
                        .iter()
                        .map(|&request| (request, row[prefix]))
                })
                .collect();
            TreeCall {
                bucket,
                nodes: ordered,
                parents,
                answers,
            }
        })
        .collect())
}

/// What one keystroke of the key engine asks of the key-level model.
///
/// A keystroke is one [`KeyModel::encode`] over all the typed keys, then any
/// number of [`KeyModel::tree`] calls over that encoding, each one call that
/// [`plan`] made. Nothing of the decoder survives a keystroke: the encoding
/// and every call's logits are dropped with it.
pub trait KeyModel {
    /// Why a call failed.
    type Error: std::error::Error + Send + Sync + 'static;
    /// One keystroke's encoding of the keys: what the tree calls read.
    type Encoding;
    /// One tree call's output: every node's next-character logits.
    type Logits;

    /// Encode *keys* under *context*: write the head's CTC log-probs into
    /// *`log_probs`* (cleared first), `keys.len()` rows of the emittable
    /// characters and the blank, the blank last, and return what the tree
    /// calls read.
    ///
    /// # Errors
    ///
    /// Whatever the model reports, more keys than it takes included.
    fn encode(
        &mut self,
        keys: &Keys,
        context: Option<&str>,
        log_probs: &mut Vec<f32>,
    ) -> Result<Self::Encoding, Self::Error>;

    /// Run *call* over *encoding*: the logits of every node of the call.
    ///
    /// # Errors
    ///
    /// Whatever the model reports.
    fn tree(
        &mut self,
        encoding: &Self::Encoding,
        call: &TreeCall<'_, EmitId>,
    ) -> Result<Self::Logits, Self::Error>;

    /// Write the log-softmax of *node*'s row of *logits* into *out*, one
    /// entry per emittable character: the host's f32 log-softmax, run only
    /// for the rows the search reads.
    fn log_softmax(&self, logits: &Self::Logits, node: usize, out: &mut [f32]);
}
