//! The constrained beam search over keys, a port of `mlime.train.key_search`.
//!
//! The key model's head gives, at every key `t` of a record with `n` keys, a
//! log-softmax over the `E` emittable characters and a blank (the last
//! column). An output is a cover of the keys by contiguous spans with one
//! character per span, emitted at the span's last key and blank elsewhere; it
//! is legal when each span matches its character under the legality automaton
//! ([`KeyLattice`]), possibly at a correction cost `w`. A path with cut points
//! `0 = b_0 < ... < b_m = n` and characters `c_1..c_m` scores
//!
//! ```text
//! sum of blank log-probs over the non-end keys + sum_j emit_{b_j - 1}(c_j)
//!   + beta * sum_j T(c_j | c_1..c_{j-1}) - gamma * sum_j w(span_j, c_j)
//! ```
//!
//! with `T` the [`KeyTransition`] the caller supplies. Written with the
//! blanks telescoped, a path scores the record's constant plus the sum of its
//! characters' *gains* (`emit - blank` at the key, plus the weighted
//! transition and correction), so two hypotheses at the same key compare
//! without blank bookkeeping.
//!
//! The search walks key offsets `1..=n`. A hypothesis is a character prefix
//! (a [`PrefixId`] into the search's [`PrefixTrie`]); it holds one state per
//! open cut point, the combination (log-sum-exp, or the max under
//! [`Combine::Viterbi`]) over the covers of the prefix that end there. A state
//! whose open span cannot grow into any arc any more is dropped. A state is
//! ranked by its gains so far plus an admissible optimistic bound on the gain
//! of its next character (the transition's share bounded by 0, which is why a
//! [`Weight`] is never negative), and the beam keeps the best
//! [`KeySearchOptions::beam`] prefixes with all their live states.
//!
//! The transition is queried once per distinct prefix, once per key offset for
//! every beam prefix about to extend that has no row yet. A row is kept while
//! the prefix or one of its ancestors is in the beam.
//!
//! **Word arcs.** With the word arcs of a lexicon ([`KeyWordArc`]) and a
//! prior `pi > 0` the search proposes whole entries. At each offset `b` the
//! best [`KeySearchOptions::arc_width`] arcs ending there (by their best
//! chain's gains plus `pi`, the optimistic gain) are kept; every beam prefix
//! with a live cut at an arc's start lands it whole: the prefix extended by
//! the entry gets a state at cut `b` holding the chain's gains, the
//! transition's log-probs of the entry's characters in turn (the rows of the
//! inner prefixes are asked in rounds, round `k` the prefixes `P + c_1..c_k`),
//! the chain's correction costs, its cut points and the prior. Several chains
//! of one arc combine as covers. A cover is raised by `pi` for the largest
//! set of non-overlapping arcs it contains under [`Combine::Viterbi`], and its
//! mass multiplied by the sum over its sets `S` of non-overlapping arcs of
//! `(e^pi - 1)^|S|` under [`Combine::Sum`]: a landed arc adds the additional
//! mass `mass * (e^pi - 1)` to the state the character-by-character path
//! reached for the landed prefix, or, when that path did not reach it, creates
//! the state with the whole mass `mass * e^pi`. A cut with a kept arc ending
//! after the current key is not dead, and the optimistic bound of an open span
//! there also takes the arc's optimistic gain, with the transition's share
//! bounded by 0, so a prefix about to land an idiom stays in the beam. At
//! `pi = 0`, or with no arcs, the search is the search without arcs.
//!
//! The arithmetic is f64 in the order the Python search performs it, with
//! numpy's `logaddexp` and Python's compensated `sum`, so a search over the
//! same inputs reproduces the Python search's hypotheses and its sequence of
//! transition calls; `tests/key_search.rs` holds the two together on a shared
//! fixture. Every ordering decision is a stable sort.

use std::cmp::Ordering;
use std::num::NonZeroUsize;

use hashbrown::{HashMap, HashSet};
use ime_pinyin::{EmitId, KeyLattice};
use thiserror::Error;

use crate::key_arcs::KeyWordArc;

/// Why a log-prob table is not one.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum KeyLogProbsError {
    /// The table has no column for the blank.
    #[error("a row needs at least the blank column")]
    NoBlank,
    /// The data is not `keys` rows of `width` columns.
    #[error("{len} log-probs are not {keys} keys of {width} columns")]
    Shape {
        /// The key count the table was declared with.
        keys: usize,
        /// The columns per key.
        width: usize,
        /// The data's length.
        len: usize,
    },
}

/// The head's log-probs over one record's keys: `keys` rows of `width = E + 1`
/// columns, row-major, the blank last.
#[derive(Copy, Clone, Debug)]
pub struct KeyLogProbs<'a> {
    keys: usize,
    width: usize,
    data: &'a [f32],
}

impl<'a> KeyLogProbs<'a> {
    /// The table of `keys` rows over *data*.
    ///
    /// # Errors
    ///
    /// If *width* is zero or *data* is not exactly `keys * width` long.
    pub fn new(keys: usize, width: usize, data: &'a [f32]) -> Result<Self, KeyLogProbsError> {
        if width == 0 {
            return Err(KeyLogProbsError::NoBlank);
        }
        if keys.checked_mul(width) != Some(data.len()) {
            return Err(KeyLogProbsError::Shape {
                keys,
                width,
                len: data.len(),
            });
        }
        Ok(Self { keys, width, data })
    }

    /// How many keys the table covers.
    #[must_use]
    pub const fn keys(&self) -> usize {
        self.keys
    }

    /// The columns per key: the emittable characters and the blank.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// How many characters are emittable: every column but the blank.
    #[must_use]
    pub const fn emittable(&self) -> usize {
        self.width - 1
    }

    fn row(&self, key: usize) -> &'a [f32] {
        &self.data[key * self.width..(key + 1) * self.width]
    }
}

/// How a prefix's covers combine into its state.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Combine {
    /// Log-sum-exp over the covers: the CTC sum.
    Sum,
    /// The best cover alone.
    Viterbi,
}

/// A search weight: finite and not negative, because the optimistic bound
/// prices the transition at most 0 and needs `beta >= 0`.
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct Weight(f64);

/// Why a number is not a [`Weight`].
#[derive(Debug, Error, PartialEq)]
#[error("a weight is finite and not negative, got {0}")]
pub struct WeightError(f64);

impl Weight {
    /// The weight that switches its term off.
    pub const ZERO: Self = Self(0.0);

    /// The weight's value.
    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for Weight {
    type Error = WeightError;

    fn try_from(value: f64) -> Result<Self, WeightError> {
        if value.is_finite() && value >= 0.0 {
            Ok(Self(value))
        } else {
            Err(WeightError(value))
        }
    }
}

/// What a search is asked to do.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct KeySearchOptions {
    /// The transition's weight.
    pub beta: Weight,
    /// The correction cost's weight.
    pub gamma: Weight,
    /// The prefixes kept after each key; `None` prunes nothing.
    pub beam: Option<NonZeroUsize>,
    /// The outputs returned; `None` takes the beam, or every legal output
    /// without one.
    pub top_k: Option<NonZeroUsize>,
    /// How a prefix's covers combine.
    pub combine: Combine,
    /// The prior `pi` each contained word arc adds; 0 ignores the arcs.
    pub prior: Weight,
    /// The arcs landing per key offset: the best by optimistic gain.
    pub arc_width: NonZeroUsize,
}

/// The arcs landing per offset the product's search uses: its shape, as the
/// beam's eight.
pub const DEFAULT_ARC_WIDTH: NonZeroUsize = NonZeroUsize::new(8).expect("8 is not zero");

/// A prefix of a search's outputs: an index into its [`PrefixTrie`].
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct PrefixId(u32);

impl PrefixId {
    /// The empty prefix.
    pub const ROOT: Self = Self(0);

    /// The prefix's index in the arena.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Copy, Clone, Debug)]
struct TrieNode {
    /// The parent and the last emission id; `None` for the root.
    link: Option<(PrefixId, EmitId)>,
    depth: usize,
}

/// Every distinct prefix a search has seen, as an arena of
/// `(parent, last emission id, depth)` with the root at [`PrefixId::ROOT`].
#[derive(Debug)]
pub struct PrefixTrie {
    nodes: Vec<TrieNode>,
    children: HashMap<(PrefixId, EmitId), PrefixId>,
}

impl PrefixTrie {
    fn new() -> Self {
        Self {
            nodes: vec![TrieNode {
                link: None,
                depth: 0,
            }],
            children: HashMap::new(),
        }
    }

    /// *parent* extended by *emit*; the same prefix always has the same id.
    fn child(&mut self, parent: PrefixId, emit: EmitId) -> PrefixId {
        if let Some(&known) = self.children.get(&(parent, emit)) {
            return known;
        }
        let id = PrefixId(u32::try_from(self.nodes.len()).expect("a search's prefixes fit u32"));
        let depth = self.nodes[parent.index()].depth + 1;
        self.nodes.push(TrieNode {
            link: Some((parent, emit)),
            depth,
        });
        self.children.insert((parent, emit), id);
        id
    }

    fn link(&self, id: PrefixId) -> Option<(PrefixId, EmitId)> {
        self.nodes[id.index()].link
    }

    /// *id*'s prefix without its last id; `None` for the root.
    #[must_use]
    pub fn parent(&self, id: PrefixId) -> Option<PrefixId> {
        self.nodes[id.index()].link.map(|(parent, _)| parent)
    }

    /// *id*'s last emission id; `None` for the root.
    #[must_use]
    pub fn last(&self, id: PrefixId) -> Option<EmitId> {
        self.nodes[id.index()].link.map(|(_, emit)| emit)
    }

    /// How many ids *id*'s prefix holds.
    #[must_use]
    pub fn depth(&self, id: PrefixId) -> usize {
        self.nodes[id.index()].depth
    }

    /// *id*'s emission ids, first to last, into *out* (cleared first).
    pub fn ids_into(&self, id: PrefixId, out: &mut Vec<EmitId>) {
        out.clear();
        let mut at = id;
        while let Some((parent, emit)) = self.nodes[at.index()].link {
            out.push(emit);
            at = parent;
        }
        out.reverse();
    }
}

/// The search's transition: next-character log-probs for a batch of prefixes.
pub trait KeyTransition {
    /// Why a call failed.
    type Error;

    /// Fill *out* with `log T(. | prefix)` over the emittable set for each of
    /// *queries*, in order: `out.len() == queries.len() * E`, and row `i`
    /// is `out[i * E..(i + 1) * E]`. Called once per key offset with the
    /// prefixes that have no row yet, never with none.
    ///
    /// # Errors
    ///
    /// Whatever the transition's model reports.
    fn rows(
        &mut self,
        trie: &PrefixTrie,
        queries: &[PrefixId],
        out: &mut [f32],
    ) -> Result<(), Self::Error>;
}

/// One output of the search, with its score's terms.
#[derive(Clone, PartialEq, Debug)]
pub struct KeyHypothesis {
    /// The output's characters, as emission ids.
    pub ids: Vec<EmitId>,
    /// The best cover's cut points `b_1..b_m`; `b_m` is the key count.
    pub cuts: Vec<usize>,
    /// The combination over the kept covers of the path score.
    pub score: f64,
    /// The best cover's CTC term: its blanks and its characters' emissions.
    pub ctc: f64,
    /// The summed transition log-prob of the characters.
    pub transition: f64,
    /// The best cover's summed correction cost.
    pub correction: f64,
    /// The arcs in the best cover's chosen set: its score carries the prior
    /// for each.
    pub arcs: usize,
    /// The best cover's running score, one entry per character: `steps[j]`
    /// is the path score of the first `j + 1` characters over the keys
    /// before their cut `cuts[j]`, blanks included, weighted as `score` is. A
    /// landed arc's prior is in the step of its last character. The last is
    /// the best cover's whole path score.
    pub steps: Vec<f64>,
}

/// What a search cost.
#[derive(Copy, Clone, Default, PartialEq, Eq, Debug)]
pub struct KeySearchStats {
    /// How many times the transition was called.
    pub calls: usize,
    /// How many prefix rows those calls asked for, the landings' included.
    pub rows: usize,
    /// Arcs enumerated over the record's lattice, before the per-offset width.
    pub arcs_offered: usize,
    /// Kept arcs that at least one beam prefix landed.
    pub arcs_landed: usize,
    /// The rows asked for landings alone, on top of the beam's own.
    pub landing_rows: usize,
}

/// A cover of the keys so far, as a persistent list in an arena.
#[derive(Copy, Clone, Debug)]
struct CoverId(u32);

#[derive(Default)]
struct Covers {
    /// `(cut, the running score through it, the cover before it)`.
    nodes: Vec<(usize, f64, Option<CoverId>)>,
}

impl Covers {
    fn push(&mut self, before: Option<CoverId>, cut: usize, step: f64) -> CoverId {
        let id = CoverId(u32::try_from(self.nodes.len()).expect("a search's covers fit u32"));
        self.nodes.push((cut, step, before));
        id
    }

    /// The cut points of the cover ending at *last* and its running score
    /// through each, ascending.
    fn walk(&self, last: Option<CoverId>) -> (Vec<usize>, Vec<f64>) {
        let (mut cuts, mut steps) = (Vec::new(), Vec::new());
        let mut at = last;
        while let Some(id) = at {
            let (cut, step, before) = self.nodes[id.0 as usize];
            cuts.push(cut);
            steps.push(step);
            at = before;
        }
        cuts.reverse();
        steps.reverse();
        (cuts, steps)
    }
}

/// A prefix's state at one cut point: the covers of the prefix ending there.
#[derive(Copy, Clone, Debug)]
struct CutState {
    cut: usize,
    /// The combined gains, without the transition, over the covers.
    mass: f64,
    /// The best cover's gains, without the transition.
    best: f64,
    /// The best cover's `sum (emit - blank)`.
    ctc: f64,
    /// The best cover's summed correction cost.
    correction: f64,
    cover: Option<CoverId>,
    /// The arcs in the best cover's chosen set.
    arcs: usize,
}

/// A hypothesis: a prefix, its transition sum and its live cuts, ascending.
struct PrefixState {
    id: PrefixId,
    transition: f64,
    cuts: Vec<CutState>,
}

/// One character a beam prefix can emit at the current key, its covers combined.
#[derive(Copy, Clone, Debug)]
struct Extension {
    emit: EmitId,
    mass: f64,
    best: f64,
    ctc: f64,
    correction: f64,
    /// The best cover the character lands from.
    cover: Option<CoverId>,
    /// The arcs in that cover's chosen set.
    arcs: usize,
    transition: f64,
    score: f64,
}

/// Every extension of one beam prefix at the current key, by character.
struct Group {
    parent: PrefixId,
    extensions: Vec<Extension>,
}

/// The arcs ending at one offset that start at `start`: `lo..hi` of the arc arrays.
#[derive(Copy, Clone, Debug)]
struct Span {
    start: usize,
    lo: usize,
    hi: usize,
}

/// One record's arcs, gains and bound table: Python's `_Record` without its beam.
struct Tables {
    keys: usize,
    blank_total: f64,
    /// `blank_through[t]`: the blank log-probs of keys `..t` summed, for the
    /// running scores only.
    blank_through: Vec<f64>,
    emits: Vec<EmitId>,
    /// Each arc's end offset, in the arcs' order.
    ends: Vec<usize>,
    cost: Vec<f64>,
    ctc_gain: Vec<f64>,
    gain: Vec<f64>,
    /// `by_end[e]`: the arcs ending at offset `e`, by ascending start.
    by_end: Vec<Vec<Span>>,
    /// The furthest end of any arc from each offset; 0 where none starts.
    max_end: Vec<usize>,
    /// `bound[c * (n + 1) + t]`: the best gain, the transition's share bounded
    /// by 0, of an arc from `c` ending after key `t`.
    bound: Vec<f64>,
}

impl Tables {
    fn new(lattice: &KeyLattice, log_probs: &KeyLogProbs<'_>, gamma: f64) -> Self {
        let n = lattice.keys();
        let blank: Vec<f64> = (0..n)
            .map(|key| f64::from(log_probs.row(key)[log_probs.emittable()]))
            .collect();
        let mut tables = Self {
            keys: n,
            blank_total: numpy_sum(&blank),
            blank_through: std::iter::once(0.0)
                .chain(blank.iter().scan(0.0, |total, &value| {
                    *total += value;
                    Some(*total)
                }))
                .collect(),
            emits: Vec::with_capacity(lattice.arc_count()),
            ends: Vec::with_capacity(lattice.arc_count()),
            cost: Vec::with_capacity(lattice.arc_count()),
            ctc_gain: Vec::with_capacity(lattice.arc_count()),
            gain: Vec::with_capacity(lattice.arc_count()),
            by_end: vec![Vec::new(); n + 1],
            max_end: vec![0; n + 1],
            bound: vec![f64::NEG_INFINITY; (n + 1) * (n + 1)],
        };
        for edge in lattice.edges() {
            let (start, end) = (edge.start(), edge.end());
            let row = log_probs.row(end - 1);
            let lo = tables.emits.len();
            for (&emit, &cost) in edge.ids().iter().zip(edge.costs()) {
                let ctc_gain = f64::from(row[emit.index()]) - blank[end - 1];
                tables.emits.push(emit);
                tables.ends.push(end);
                tables.cost.push(cost);
                tables.ctc_gain.push(ctc_gain);
                tables.gain.push(ctc_gain - gamma * cost);
            }
            let hi = tables.emits.len();
            let top = tables.gain[lo..hi]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            tables.by_end[end].push(Span { start, lo, hi });
            tables.max_end[start] = tables.max_end[start].max(end);
            for slot in &mut tables.bound[start * (n + 1)..start * (n + 1) + end] {
                *slot = slot.max(top);
            }
        }
        tables
    }

    fn bound(&self, cut: usize, key: usize) -> f64 {
        self.bound[cut * (self.keys + 1) + key]
    }
}

/// numpy's `np.logaddexp` on two doubles.
#[expect(
    clippy::float_cmp,
    reason = "numpy's logaddexp separates a tie, and the infinities of one sign, by exact equality"
)]
fn logaddexp(x: f64, y: f64) -> f64 {
    if x == y {
        return x + std::f64::consts::LN_2;
    }
    let difference = x - y;
    if difference > 0.0 {
        x + (-difference).exp().ln_1p()
    } else if difference <= 0.0 {
        y + difference.exp().ln_1p()
    } else {
        difference
    }
}

/// Python's `sum` of floats: Neumaier's compensated summation from 0.
fn compensated_sum(values: impl Iterator<Item = f64>) -> f64 {
    let (mut total, mut compensation) = (0.0_f64, 0.0_f64);
    for value in values {
        let next = total + value;
        if total.abs() >= value.abs() {
            compensation += (total - next) + value;
        } else {
            compensation += (value - next) + total;
        }
        total = next;
    }
    if compensation != 0.0 && compensation.is_finite() {
        total += compensation;
    }
    total
}

/// numpy's pairwise summation of a contiguous slice (blocks of 128, eight
/// accumulators within a block).
fn pairwise(values: &[f64]) -> f64 {
    const BLOCK: usize = 128;
    let n = values.len();
    if n < 8 {
        values.iter().fold(-0.0, |total, &value| total + value)
    } else if n <= BLOCK {
        let mut lanes = [0.0; 8];
        lanes.copy_from_slice(&values[..8]);
        let body = n - n % 8;
        for chunk in values[8..body].as_chunks::<8>().0 {
            for (lane, &value) in lanes.iter_mut().zip(chunk) {
                *lane += value;
            }
        }
        let mut total = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
            + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
        for &value in &values[body..] {
            total += value;
        }
        total
    } else {
        let mut half = n / 2;
        half -= half % 8;
        pairwise(&values[..half]) + pairwise(&values[half..])
    }
}

/// `ndarray.sum()` of a non-empty contiguous slice: the first element, then
/// the pairwise sum of the rest.
fn numpy_sum(values: &[f64]) -> f64 {
    match values.split_first() {
        Some((&first, rest)) => first + pairwise(rest),
        None => 0.0,
    }
}

/// The best of *values*, or under [`Combine::Sum`] their log-sum-exp.
fn combine(values: &[f64], mode: Combine) -> f64 {
    let top = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if mode == Combine::Viterbi || values.len() == 1 || top == f64::NEG_INFINITY {
        return top;
    }
    top + compensated_sum(values.iter().map(|value| (value - top).exp())).ln()
}

/// Descending order of *a* against *b*, as a stable sort by the negated key.
fn descending(a: f64, b: f64) -> Ordering {
    (-a).partial_cmp(&(-b)).unwrap_or(Ordering::Equal)
}

/// The extensions no old prefix took, as `(group, position)` in group order:
/// all of them, or the best *limit* by score. Every new prefix has the one cut
/// `end`, so the bound they are ranked by is common and the best `limit` by
/// score are all that can make the beam. Exact ties at the boundary go to the
/// earlier extension.
fn fresh_extensions(
    groups: &[Group],
    taken: &[Vec<bool>],
    limit: Option<NonZeroUsize>,
) -> Vec<(usize, usize)> {
    let mut fresh: Vec<(usize, usize)> = groups
        .iter()
        .enumerate()
        .flat_map(|(index, group)| (0..group.extensions.len()).map(move |at| (index, at)))
        .filter(|&(index, at)| !taken[index][at])
        .collect();
    if let Some(limit) = limit.filter(|limit| fresh.len() > limit.get()) {
        let score = |&(index, at): &(usize, usize)| groups[index].extensions[at].score;
        let mut order: Vec<usize> = (0..fresh.len()).collect();
        order.sort_by(|&a, &b| descending(score(&fresh[a]), score(&fresh[b])));
        order.truncate(limit.get());
        order.sort_unstable();
        fresh = order.into_iter().map(|index| fresh[index]).collect();
    }
    fresh
}

/// A word arc priced against its record: what a prefix landing it gains,
/// whichever prefix it is.
struct Landing {
    entry: Box<[EmitId]>,
    start: usize,
    /// The chains' summed gains combined as covers.
    mass: f64,
    /// The best chain's gains alone.
    best: f64,
    /// The best chain's `sum (emit - blank)` and summed correction cost.
    ctc: f64,
    correction: f64,
    /// The best chain's gain at each of its arcs, and the cut each ends at.
    gains: Box<[f64]>,
    cuts: Box<[usize]>,
    /// The best chain's gains plus the prior: what the bound prices the arc at.
    optimistic: f64,
}

/// One beam prefix landing one kept arc at the current offset.
struct Planned {
    /// Index into the offset's landings.
    landing: usize,
    /// The prefix `P`, then `P + c_1`, ..., then `P + entry`.
    path: Box<[PrefixId]>,
    /// The beam prefix's transition sum.
    transition: f64,
    /// The beam prefix's state at the arc's start cut.
    state: CutState,
}

/// Every arc landing on one prefix at one offset, combined.
struct Arrival {
    id: PrefixId,
    transition: f64,
    /// The additional mass the arcs give; what stands in for the state when the
    /// character-by-character path did reach the prefix.
    extra: f64,
    /// The best landing's state with its own whole mass and the prior: what
    /// stands in when that path did not reach the prefix. Landings that overlap
    /// are alternatives of one cover, so their whole masses are not summed and
    /// the state is a lower bound of the covers' mass.
    cut: CutState,
}

/// One search over one record.
struct Search {
    tables: Tables,
    trie: PrefixTrie,
    covers: Covers,
    /// The beam, in its order: insertion order, or rank order after a prune.
    beam: Vec<PrefixState>,
    rows: HashMap<PrefixId, Box<[f32]>>,
    beta: f64,
    combine: Combine,
    beam_width: Option<NonZeroUsize>,
    top_k: Option<NonZeroUsize>,
    prior: f64,
    /// The extra mass a landed arc adds in the sum: `log(e^pi - 1)`.
    bonus: f64,
    /// The kept arcs by their end offset, best first.
    landing: Vec<Vec<Landing>>,
    /// The plan of the current offset: which beam prefix lands which arc.
    plan: Vec<Planned>,
}

impl Search {
    fn new(
        lattice: &KeyLattice,
        log_probs: &KeyLogProbs<'_>,
        words: &[KeyWordArc],
        options: &KeySearchOptions,
        stats: &mut KeySearchStats,
    ) -> Self {
        let root = PrefixState {
            id: PrefixId::ROOT,
            transition: 0.0,
            cuts: vec![CutState {
                cut: 0,
                mass: 0.0,
                best: 0.0,
                ctc: 0.0,
                correction: 0.0,
                cover: None,
                arcs: 0,
            }],
        };
        let prior = options.prior.get();
        let mut search = Self {
            tables: Tables::new(lattice, log_probs, options.gamma.get()),
            trie: PrefixTrie::new(),
            covers: Covers::default(),
            beam: vec![root],
            rows: HashMap::new(),
            beta: options.beta.get(),
            combine: options.combine,
            beam_width: options.beam,
            top_k: options.top_k.or(options.beam),
            prior,
            bonus: if prior > 0.0 {
                prior + (-(-prior).exp()).ln_1p()
            } else {
                f64::NEG_INFINITY
            },
            landing: Vec::new(),
            plan: Vec::new(),
        };
        if prior > 0.0 && !words.is_empty() {
            search.keep(words, options.arc_width.get());
            stats.arcs_offered += words.len();
        }
        search
    }

    /// Price the arcs, keep the best *width* ending at each offset, and open
    /// their cuts: a kept arc keeps its start cut alive until it can land and
    /// raises the bound of the open span there by its optimistic gain.
    fn keep(&mut self, words: &[KeyWordArc], width: usize) {
        let tables = &mut self.tables;
        let n = tables.keys;
        let mut by_end: Vec<Vec<Landing>> = (0..=n).map(|_| Vec::new()).collect();
        for word in words {
            let sums: Vec<f64> = word
                .chains()
                .map(|chain| {
                    let gains: Vec<f64> = chain.iter().map(|&arc| tables.gain[arc]).collect();
                    numpy_sum(&gains)
                })
                .collect();
            let top = sums
                .iter()
                .enumerate()
                .fold(0, |top, (at, &sum)| if sum > sums[top] { at } else { top });
            let chain = word
                .chains()
                .nth(top)
                .expect("an arc has at least one chain");
            let gain_of = |values: &[f64]| -> f64 {
                numpy_sum(&chain.iter().map(|&arc| values[arc]).collect::<Vec<f64>>())
            };
            by_end[word.end()].push(Landing {
                entry: word.entry().into(),
                start: word.start(),
                mass: combine(&sums, self.combine),
                best: sums[top],
                ctc: gain_of(&tables.ctc_gain),
                correction: gain_of(&tables.cost),
                gains: chain.iter().map(|&arc| tables.gain[arc]).collect(),
                cuts: chain.iter().map(|&arc| tables.ends[arc]).collect(),
                optimistic: sums[top] + self.prior,
            });
        }
        for (end, group) in by_end.iter_mut().enumerate() {
            group.sort_by(|a, b| {
                descending(a.optimistic, b.optimistic)
                    .then_with(|| a.start.cmp(&b.start))
                    .then_with(|| a.entry.cmp(&b.entry))
            });
            group.truncate(width);
            for arc in group.iter() {
                let row = arc.start * (n + 1);
                for slot in &mut tables.bound[row..row + end] {
                    *slot = slot.max(arc.optimistic);
                }
                tables.max_end[arc.start] = tables.max_end[arc.start].max(end);
            }
        }
        self.landing = by_end;
    }

    /// Plan offset *end*'s landings: each kept arc lands from every beam
    /// prefix with its start cut live.
    fn prepare(&mut self, end: usize, stats: &mut KeySearchStats) {
        self.plan.clear();
        let Some(landings) = self.landing.get(end) else {
            return;
        };
        for (index, arc) in landings.iter().enumerate() {
            let before = self.plan.len();
            for prefix in &self.beam {
                let Some(&source) = prefix.cuts.iter().find(|cut| cut.cut == arc.start) else {
                    continue;
                };
                let mut path = Vec::with_capacity(arc.entry.len() + 1);
                path.push(prefix.id);
                for &emit in &arc.entry {
                    let last = path[path.len() - 1];
                    path.push(self.trie.child(last, emit));
                }
                self.plan.push(Planned {
                    landing: index,
                    path: path.into(),
                    transition: prefix.transition,
                    state: source,
                });
            }
            stats.arcs_landed += usize::from(self.plan.len() > before);
        }
    }

    /// The rounds of transition rows the current plan needs: its longest entry.
    fn depth(&self) -> usize {
        self.plan
            .iter()
            .map(|planned| planned.path.len() - 1)
            .max()
            .unwrap_or(0)
    }

    /// The prefixes whose rows round *depth* asks for, and how many only
    /// landings ask. Round 0 is the beam's own ([`Self::needs`]) and each
    /// landing's source prefix; round `k` the inner prefixes `P + c_1..c_k` of
    /// the entries longer than `k`.
    fn requests(&self, end: usize, depth: usize) -> (Vec<PrefixId>, usize) {
        let mut asked = if depth == 0 {
            self.needs(end)
        } else {
            Vec::new()
        };
        let beams = asked.len();
        let mut seen: HashSet<PrefixId> = asked.iter().copied().collect();
        for planned in &self.plan {
            if planned.path.len() - 1 > depth {
                let inner = planned.path[depth];
                if !self.rows.contains_key(&inner) && seen.insert(inner) {
                    asked.push(inner);
                }
            }
        }
        let landings = asked.len() - beams;
        (asked, landings)
    }

    /// The planned landings combined per landed prefix, with their transition
    /// sums, in the order each prefix was first landed.
    fn arrivals(&mut self, end: usize) -> Vec<Arrival> {
        let Self {
            plan,
            landing,
            rows,
            tables,
            covers,
            combine: mode,
            beta,
            prior,
            bonus,
            ..
        } = self;
        // `T(entry | prefix)` summed character by character, then added to the prefix's own.
        let transition_of = |planned: &Planned| -> f64 {
            let entry = &landing[end][planned.landing].entry;
            let mut total = planned.transition;
            for (step, emit) in entry.iter().enumerate() {
                total += f64::from(rows[&planned.path[step]][emit.index()]);
            }
            total
        };
        // Per landed prefix: its first landing's transition, the combined extra
        // mass, and the landing (a plan index and its best) whose state stands for it.
        let mut pending: Vec<(PrefixId, f64, f64, usize, f64)> = Vec::new();
        let mut index: HashMap<PrefixId, usize> = HashMap::new();
        for (at, planned) in plan.iter().enumerate() {
            let arc = &landing[end][planned.landing];
            let id = planned.path[planned.path.len() - 1];
            let base = planned.state.mass + arc.mass;
            let whole = base + *prior;
            let extra = if *mode == Combine::Viterbi {
                whole
            } else {
                base + *bonus
            };
            let best = planned.state.best + arc.best + *prior;
            match index.get(&id) {
                None => {
                    index.insert(id, pending.len());
                    pending.push((id, transition_of(planned), extra, at, best));
                }
                Some(&held) => {
                    let (_, _, combined, winner, winning) = &mut pending[held];
                    *combined = combine(&[*combined, extra], *mode);
                    if best > *winning {
                        (*winner, *winning) = (at, best);
                    }
                }
            }
        }
        pending
            .into_iter()
            .map(|(id, transition, extra, at, best)| {
                let planned = &plan[at];
                let arc = &landing[end][planned.landing];
                let state = planned.state;
                // The running score of the cover through each of the arc's characters.
                let mut cover = state.cover;
                let mut running = state.best;
                let mut total = planned.transition;
                for step in 0..arc.entry.len() {
                    total += f64::from(rows[&planned.path[step]][arc.entry[step].index()]);
                    running += arc.gains[step];
                    let gained = if step + 1 == arc.entry.len() {
                        best
                    } else {
                        running
                    };
                    let score = tables.blank_through[arc.cuts[step]] + gained + *beta * total;
                    cover = Some(covers.push(cover, arc.cuts[step], score));
                }
                Arrival {
                    id,
                    transition,
                    extra,
                    cut: CutState {
                        cut: end,
                        mass: state.mass + arc.mass + *prior,
                        best,
                        ctc: state.ctc + arc.ctc,
                        correction: state.correction + arc.correction,
                        cover,
                        arcs: state.arcs + 1,
                    },
                }
            })
            .collect()
    }

    /// The state at a landed prefix's cut: what the character-by-character
    /// path reached, plus the arcs. Reached, the arcs add their additional
    /// mass to it; not reached, they stand for it.
    fn merged(&self, reached: Option<CutState>, arrival: &Arrival) -> CutState {
        let Some(reached) = reached else {
            return arrival.cut;
        };
        let winner = if arrival.cut.best > reached.best {
            arrival.cut
        } else {
            reached
        };
        CutState {
            mass: combine(&[reached.mass, arrival.extra], self.combine),
            ..winner
        }
    }

    /// The beam prefixes that emit at *end* and have no transition row yet.
    fn needs(&self, end: usize) -> Vec<PrefixId> {
        let arcs = &self.tables.by_end[end];
        if arcs.is_empty() {
            return Vec::new();
        }
        self.beam
            .iter()
            .filter(|prefix| {
                !self.rows.contains_key(&prefix.id)
                    && prefix
                        .cuts
                        .iter()
                        .any(|state| arcs.iter().any(|span| span.start == state.cut))
            })
            .map(|prefix| prefix.id)
            .collect()
    }

    /// Every prefix's emissions at key `end - 1`, combined per (prefix, character).
    fn extensions(&self, end: usize) -> Vec<Group> {
        let tables = &self.tables;
        let arcs = &tables.by_end[end];
        let mut groups = Vec::new();
        for prefix in &self.beam {
            let sources: Vec<(&CutState, &Span)> = prefix
                .cuts
                .iter()
                .filter_map(|state| {
                    arcs.iter()
                        .find(|span| span.start == state.cut)
                        .map(|span| (state, span))
                })
                .collect();
            if sources.is_empty() {
                continue;
            }
            let mut found: Vec<Extension> = Vec::new();
            for &(state, span) in &sources {
                for arc in span.lo..span.hi {
                    found.push(Extension {
                        emit: tables.emits[arc],
                        mass: state.mass + tables.gain[arc],
                        best: state.best + tables.gain[arc],
                        ctc: state.ctc + tables.ctc_gain[arc],
                        correction: state.correction + tables.cost[arc],
                        cover: state.cover,
                        arcs: state.arcs,
                        transition: 0.0,
                        score: 0.0,
                    });
                }
            }
            let mut extensions = if sources.len() == 1 {
                found
            } else {
                // By character, the best cover first within each.
                found.sort_by(|a, b| a.emit.cmp(&b.emit).then_with(|| descending(a.best, b.best)));
                let mut merged: Vec<Extension> = Vec::with_capacity(found.len());
                for each in found {
                    match merged.last_mut() {
                        Some(head) if head.emit == each.emit => {
                            head.mass = match self.combine {
                                Combine::Sum => logaddexp(head.mass, each.mass),
                                Combine::Viterbi => head.mass.max(each.mass),
                            };
                        }
                        _ => merged.push(each),
                    }
                }
                merged
            };
            let row = self
                .rows
                .get(&prefix.id)
                .expect("a prefix that extends has its transition row");
            for extension in &mut extensions {
                extension.transition = prefix.transition + f64::from(row[extension.emit.index()]);
                extension.score = extension.mass + self.beta * extension.transition;
            }
            groups.push(Group {
                parent: prefix.id,
                extensions,
            });
        }
        groups
    }

    /// The cut state *extension* lands in at *end*.
    fn landed(&mut self, extension: &Extension, end: usize) -> CutState {
        let step =
            self.tables.blank_through[end] + extension.best + self.beta * extension.transition;
        CutState {
            cut: end,
            mass: extension.mass,
            best: extension.best,
            ctc: extension.ctc,
            correction: extension.correction,
            cover: Some(self.covers.push(extension.cover, end, step)),
            arcs: extension.arcs,
        }
    }

    /// Consume key `end - 1`: emit, drop dead cuts, merge, prune to the beam.
    /// At the last key, the finished outputs.
    fn step(&mut self, end: usize) -> Option<Vec<KeyHypothesis>> {
        let groups = self.extensions(end);
        let by_parent: HashMap<PrefixId, usize> = groups
            .iter()
            .enumerate()
            .map(|(index, group)| (group.parent, index))
            .collect();
        let mut taken: Vec<Vec<bool>> = groups
            .iter()
            .map(|group| vec![false; group.extensions.len()])
            .collect();
        let mut survivors = self.carry_over(end, &groups, &by_parent, &mut taken);
        let final_key = end == self.tables.keys;
        let limit = if final_key {
            self.top_k
        } else {
            self.beam_width
        };
        for (index, at) in fresh_extensions(&groups, &taken, limit) {
            let group = &groups[index];
            let extension = group.extensions[at];
            let id = self.trie.child(group.parent, extension.emit);
            let state = self.landed(&extension, end);
            survivors.push(PrefixState {
                id,
                transition: extension.transition,
                cuts: vec![state],
            });
        }
        self.land_arcs(&mut survivors, end);
        if final_key {
            return Some(self.finish(&survivors));
        }
        self.beam = self.prune(survivors, end);
        self.evict();
        None
    }

    /// Add the planned arc landings to *survivors*: a prefix no one reached
    /// is created with the arcs' state, one that was reached takes the arcs'
    /// additional mass at its cut *end*.
    fn land_arcs(&mut self, survivors: &mut Vec<PrefixState>, end: usize) {
        let mut index: HashMap<PrefixId, usize> = survivors
            .iter()
            .enumerate()
            .map(|(at, prefix)| (prefix.id, at))
            .collect();
        for arrival in self.arrivals(end) {
            let Some(&at) = index.get(&arrival.id) else {
                index.insert(arrival.id, survivors.len());
                survivors.push(PrefixState {
                    id: arrival.id,
                    transition: arrival.transition,
                    cuts: vec![arrival.cut],
                });
                continue;
            };
            let cuts = &mut survivors[at].cuts;
            match cuts.iter().position(|state| state.cut == end) {
                Some(slot) => cuts[slot] = self.merged(Some(cuts[slot]), &arrival),
                None => cuts.push(arrival.cut),
            }
        }
    }

    /// The old beam's prefixes at *end*: the cuts whose span cannot grow past
    /// this key dropped, and the cut an extension of the parent lands on here
    /// taken (and marked in *taken*). A prefix left with no cut is gone.
    fn carry_over(
        &mut self,
        end: usize,
        groups: &[Group],
        by_parent: &HashMap<PrefixId, usize>,
        taken: &mut [Vec<bool>],
    ) -> Vec<PrefixState> {
        let mut survivors: Vec<PrefixState> = Vec::with_capacity(self.beam.len());
        for old in std::mem::take(&mut self.beam) {
            let mut cuts: Vec<CutState> = old
                .cuts
                .into_iter()
                .filter(|state| self.tables.max_end[state.cut] > end)
                .collect();
            if let Some((parent, last)) = self.trie.link(old.id)
                && let Some(&index) = by_parent.get(&parent)
                && let Ok(at) = groups[index]
                    .extensions
                    .binary_search_by_key(&last, |extension| extension.emit)
            {
                taken[index][at] = true;
                cuts.push(self.landed(&groups[index].extensions[at], end));
            }
            if !cuts.is_empty() {
                survivors.push(PrefixState {
                    id: old.id,
                    transition: old.transition,
                    cuts,
                });
            }
        }
        survivors
    }

    /// *survivors* cut to the beam by rank: each prefix is ranked by the
    /// combination of its states' gains, the transition and the optimistic
    /// bound of the next character, and the beam keeps the best in rank order.
    fn prune(&self, survivors: Vec<PrefixState>, end: usize) -> Vec<PrefixState> {
        let Some(beam) = self.beam_width.filter(|beam| survivors.len() > beam.get()) else {
            return survivors;
        };
        let ranks: Vec<f64> = survivors
            .iter()
            .map(|prefix| {
                let values: Vec<f64> = prefix
                    .cuts
                    .iter()
                    .map(|state| {
                        state.mass
                            + self.beta * prefix.transition
                            + self.tables.bound(state.cut, end)
                    })
                    .collect();
                combine(&values, self.combine)
            })
            .collect();
        let mut order: Vec<usize> = (0..survivors.len()).collect();
        order.sort_by(|&a, &b| descending(ranks[a], ranks[b]));
        order.truncate(beam.get());
        let mut slots: Vec<Option<PrefixState>> = survivors.into_iter().map(Some).collect();
        order
            .into_iter()
            .filter_map(|index| slots[index].take())
            .collect()
    }

    /// The complete outputs at the last key, best first.
    fn finish(&self, outputs: &[PrefixState]) -> Vec<KeyHypothesis> {
        let n = self.tables.keys;
        let mut results: Vec<KeyHypothesis> = outputs
            .iter()
            .map(|prefix| {
                let state = prefix
                    .cuts
                    .iter()
                    .find(|state| state.cut == n)
                    .expect("every output at the last key has the cut at the key count");
                let mut ids = Vec::new();
                self.trie.ids_into(prefix.id, &mut ids);
                let (cuts, steps) = self.covers.walk(state.cover);
                KeyHypothesis {
                    ids,
                    cuts,
                    steps,
                    score: state.mass + self.beta * prefix.transition + self.tables.blank_total,
                    ctc: state.ctc + self.tables.blank_total,
                    transition: prefix.transition,
                    correction: state.correction,
                    arcs: state.arcs,
                }
            })
            .collect();
        results.sort_by(|a, b| descending(a.score, b.score));
        if let Some(top_k) = self.top_k {
            results.truncate(top_k.get());
        }
        results
    }

    /// Drop the rows of prefixes no beam prefix can extend into again: a row
    /// stays while its prefix, or one of its ancestors, is in the beam.
    fn evict(&mut self) {
        let members: HashSet<PrefixId> = self.beam.iter().map(|prefix| prefix.id).collect();
        let trie = &self.trie;
        self.rows.retain(|&id, _| {
            let mut at = Some(id);
            while let Some(prefix) = at {
                if members.contains(&prefix) {
                    return true;
                }
                at = trie.parent(prefix);
            }
            false
        });
    }
}

/// Whether `log_probs` and `lattice` agree; a violation is logged, and panics in a debug build.
fn inputs_agree(lattice: &KeyLattice, log_probs: &KeyLogProbs<'_>) -> bool {
    let emittable = log_probs.emittable();
    let agree = log_probs.keys() == lattice.keys()
        && lattice
            .edges()
            .all(|edge| edge.ids().iter().all(|emit| emit.index() < emittable));
    if !agree {
        tracing::error!(
            lattice_keys = lattice.keys(),
            log_prob_keys = log_probs.keys(),
            emittable,
            "the log-probs do not fit the lattice: the search returns nothing"
        );
        debug_assert!(
            agree,
            "the log-probs do not fit the lattice: {} keys against {}, {emittable} emittable",
            lattice.keys(),
            log_probs.keys(),
        );
    }
    agree
}

/// The best legal outputs over *lattice*'s keys, best first.
///
/// `log_probs` holds the head's log-softmax at the record's keys, the blank
/// last. *words* are the record's word arcs
/// ([`KeyArcLexicon::arcs`](crate::KeyArcLexicon::arcs)) over *lattice*, which
/// the search proposes whole when the prior is above 0 and ignores otherwise.
/// *transition* is called once per round of rows at each key offset: the
/// beam's prefixes about to extend that have no row yet, with the inner
/// prefixes of the landing arcs in later rounds; *stats* counts the calls and
/// what the arcs cost. A lattice with no complete cover has no outputs.
///
/// # Errors
///
/// The first error the transition returns.
pub fn key_search<T: KeyTransition>(
    lattice: &KeyLattice,
    log_probs: &KeyLogProbs<'_>,
    words: &[KeyWordArc],
    transition: &mut T,
    options: &KeySearchOptions,
    stats: &mut KeySearchStats,
) -> Result<Vec<KeyHypothesis>, T::Error> {
    if lattice.is_empty() || !inputs_agree(lattice, log_probs) {
        return Ok(Vec::new());
    }
    let width = log_probs.emittable();
    let mut search = Search::new(lattice, log_probs, words, options, stats);
    for end in 1..=lattice.keys() {
        if search.beam.is_empty() {
            break;
        }
        search.prepare(end, stats);
        for depth in 0..search.depth().max(1) {
            let (queries, landings) = search.requests(end, depth);
            if queries.is_empty() {
                continue;
            }
            let mut rows = vec![0.0_f32; queries.len() * width];
            transition.rows(&search.trie, &queries, &mut rows)?;
            stats.calls += 1;
            stats.rows += queries.len();
            stats.landing_rows += landings;
            for (&id, row) in queries.iter().zip(rows.chunks_exact(width)) {
                search.rows.insert(id, row.into());
            }
        }
        if let Some(results) = search.step(end) {
            return Ok(results);
        }
    }
    Ok(Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_weight_is_finite_and_not_negative() {
        assert_eq!(Weight::try_from(0.0), Ok(Weight::ZERO));
        assert_eq!(Weight::try_from(1.5).map(Weight::get), Ok(1.5));
        for bad in [-0.5, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(Weight::try_from(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn a_table_is_keys_rows_of_a_blank_and_characters() {
        let data = [0.0_f32; 6];
        let table = KeyLogProbs::new(2, 3, &data).expect("two keys of three columns");
        assert_eq!((table.keys(), table.width(), table.emittable()), (2, 3, 2));
        assert_eq!(
            KeyLogProbs::new(2, 3, &data[..5]).err(),
            Some(KeyLogProbsError::Shape {
                keys: 2,
                width: 3,
                len: 5
            })
        );
        assert_eq!(
            KeyLogProbs::new(0, 0, &[]).err(),
            Some(KeyLogProbsError::NoBlank)
        );
    }

    #[test]
    fn logaddexp_matches_numpy_on_ties_and_infinities() {
        assert_eq!(logaddexp(-1.0, -1.0), -1.0 + std::f64::consts::LN_2);
        assert_eq!(
            logaddexp(f64::NEG_INFINITY, f64::NEG_INFINITY),
            f64::NEG_INFINITY
        );
        assert_eq!(logaddexp(f64::NEG_INFINITY, -3.0), -3.0);
        assert_eq!(logaddexp(-3.0, f64::NEG_INFINITY), -3.0);
        let both = logaddexp(-2.0, -5.0);
        assert_eq!(both, -2.0 + (-3.0_f64).exp().ln_1p());
    }

    #[test]
    fn the_compensated_sum_keeps_what_a_plain_one_loses() {
        let values = [1.0, 1e-16, 1e-16, 1e-16, 1e-16];
        let plain: f64 = values.iter().sum();
        assert_eq!(plain, 1.0);
        assert!(compensated_sum(values.iter().copied()) > 1.0);
    }

    #[test]
    fn the_pairwise_sum_is_the_exact_sum_on_exact_inputs() {
        for n in [1, 2, 7, 8, 9, 31, 128, 129, 300] {
            let values: Vec<f64> = (0..n).map(|i| -f64::from(i % 17) / 65536.0).collect();
            let exact: f64 = values.iter().sum();
            assert_eq!(numpy_sum(&values), exact, "{n}");
        }
    }

    #[test]
    fn the_trie_numbers_each_prefix_once() {
        let mut trie = PrefixTrie::new();
        let legality = ime_pinyin::Legality::over(&['啊', '安']).expect("emittable");
        let lattice = legality.lattice(
            &ime_pinyin::Keys::new("a").expect("keys"),
            ime_pinyin::LegalityOptions::OFFLINE,
        );
        let emits: Vec<EmitId> = lattice
            .edges()
            .flat_map(|edge| edge.ids().to_vec())
            .collect();
        let (first, second) = (emits[0], emits[1]);
        let a = trie.child(PrefixId::ROOT, first);
        let ab = trie.child(a, second);
        assert_eq!(trie.child(PrefixId::ROOT, first), a);
        assert_eq!(trie.child(a, second), ab);
        assert_ne!(trie.child(PrefixId::ROOT, second), a);
        assert_eq!(
            (trie.parent(ab), trie.last(ab), trie.depth(ab)),
            (Some(a), Some(second), 2)
        );
        assert_eq!(trie.parent(PrefixId::ROOT), None);
        let mut ids = Vec::new();
        trie.ids_into(ab, &mut ids);
        assert_eq!(ids, [first, second]);
    }
}
