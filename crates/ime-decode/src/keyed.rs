//! The constrained beam search over keystrokes (#216 §3 and §5).
//!
//! The end-to-end network reads the raw keys and, at every key, scores blank
//! or one character of the emittable set ([`KeyEmission`]). A sentence is the
//! sequence of non-blank emissions, and it is kept only when it is a legal
//! reading of the keys: the search walks the legality automaton
//! ([`KeyLattice`]) left to right, and at key `t` a hypothesis whose last
//! emission closed at offset `p` may
//!
//! - emit blank, when the open span `p..=t` is still a prefix of a kept arc
//!   ([`Step::blank`](ime_pinyin::Step::blank)), or
//! - emit one character the span admits, closing it at `t + 1`.
//!
//! There is no reading list and no pruning but the beam's: every legal cover
//! is reachable.
//!
//! **Score.** A hypothesis is `(prefix, pending)`: the characters emitted so
//! far and the offset its last emission closed at. Its score is
//!
//! ```text
//! log Σ_alignments exp(Σ_keys log p_t(blank | char) − γ · correction cost)
//!   + β · log P_transition(prefix)
//! ```
//!
//! The first term sums over every alignment of the prefix that ends at
//! `pending`, as the constrained-CTC loss sums them; two ways of reaching the
//! same `(prefix, pending)` hold disjoint alignment sets and merge by
//! log-sum-exp. The transition term does not depend on where a character
//! lands, so it is computed once per distinct prefix: each prefix is one node
//! of a prefix tree carrying one transition state, however many alignments
//! and pending offsets share it. At the last key only emitting hypotheses
//! survive, every hypothesis of one sentence has merged into one, and the
//! transition's end-of-sentence term is added.
//!
//! **Word arcs.** A curated word ([`WordArcs`]) is a legal multi-character
//! emission: offered from offset `s` to `e` when some cover of keys `s..e`
//! spells it character by character on the lattice's own edges, so an arc
//! never introduces an illegal output. The lineage it starts is held outside
//! the beam until it lands at key `e − 1`, scored by the same CTC and
//! transition terms over all its covers, plus the word prior `λ`. An arc and
//! the character-by-character path to the same `(prefix, pending)` share
//! alignments, so they meet by maximum, not by sum.
//!
//! **Per-key cost.** With `B` the beam width and `C` the
//! [`Legality::admitted_bound`](ime_pinyin::Legality::admitted_bound), one key
//! costs at most:
//!
//! - `B` lattice steps (two table reads each) and one emission gather of at
//!   most `C` entries per distinct pending offset (at most
//!   `min(B, MAX_CORRECTED_LEN)`), shared by every hypothesis there;
//! - `B · (1 + C)` scored candidates: one blank and at most `C` characters a
//!   hypothesis, each one addition of emission, correction and transition;
//! - one [`KeyTransition::score`] gather a distinct prefix (at most `B`), over
//!   at most `B · C` characters in all;
//! - one `O(B · (1 + C))` selection of the `B` survivors;
//! - one [`KeyTransition::advance`] over at most `B` new prefixes.
//!
//! Word arcs add, at the keys where a hypothesis can start one, one transition
//! row per arc character per spawning hypothesis. [`SearchStats`] records the
//! largest per-key counts, and the tests hold them to this bound.

mod words;

use std::num::NonZeroUsize;

use hashbrown::HashMap;
use ime_pinyin::{EmitId, KeyLattice, MAX_CORRECTED_LEN};
use thiserror::Error;

pub use words::{WordArcs, WordStats};

/// The per-key output of the CTC head: blank and every emittable character.
///
/// Scores are log-probabilities normalised over the whole emittable set plus
/// blank, the way training normalises them; the search masks without
/// renormalising (#216 §3, #115's lesson).
pub trait KeyEmission {
    /// How many keys the emission scores.
    fn keys(&self) -> usize;

    /// The blank's log-probability at *key*.
    fn blank(&self, key: usize) -> f32;

    /// `out[i]` = the log-probability of `ids[i]` at *key*.
    ///
    /// *ids* is one lattice step's admitted set, sorted; `out` holds exactly
    /// `ids.len()` slots.
    fn gather(&self, key: usize, ids: &[EmitId], out: &mut [f32]);
}

impl<E: KeyEmission + ?Sized> KeyEmission for &E {
    fn keys(&self) -> usize {
        (**self).keys()
    }

    fn blank(&self, key: usize) -> f32 {
        (**self).blank(key)
    }

    fn gather(&self, key: usize, ids: &[EmitId], out: &mut [f32]) {
        (**self).gather(key, ids, out);
    }
}

/// How well a character follows the prefix before it: the jointly trained
/// decoder of #216 §5 (a), or any stand-in.
///
/// The state stands for one prefix, context and keys included: a model that
/// reads them binds them when it is built for one search. A prefix's
/// next-character distribution does not depend on where the next character
/// lands, so the search asks for it once per distinct prefix.
pub trait KeyTransition {
    /// What the search keeps per prefix: a decoder's cache row, an n-gram's
    /// history.
    type State: Clone;

    /// The state of the empty prefix.
    fn start(&self) -> Self::State;

    /// `out[i]` = `log P(ids[i] | prefix)`, for the prefix *state* stands for.
    ///
    /// *ids* is sorted; `out` holds exactly `ids.len()` slots.
    fn score(&self, state: &Self::State, ids: &[EmitId], out: &mut [f32]);

    /// The log-probability that the sentence ends after the prefix.
    fn finish(&self, state: &Self::State) -> f32;

    /// The state after each `(state, character)` step, in order: one batch a
    /// key, so a model whose step is a matrix product runs it once.
    fn advance(&self, steps: &[(&Self::State, EmitId)]) -> Vec<Self::State>;
}

impl<T: KeyTransition + ?Sized> KeyTransition for &T {
    type State = T::State;

    fn start(&self) -> Self::State {
        (**self).start()
    }

    fn score(&self, state: &Self::State, ids: &[EmitId], out: &mut [f32]) {
        (**self).score(state, ids, out);
    }

    fn finish(&self, state: &Self::State) -> f32 {
        (**self).finish(state)
    }

    fn advance(&self, steps: &[(&Self::State, EmitId)]) -> Vec<Self::State> {
        (**self).advance(steps)
    }
}

/// Why emission rows were rejected.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum RowsError {
    /// The buffer holds no row.
    #[error("the emission holds no key")]
    Empty,
    /// The buffer is not a whole number of rows.
    #[error("{len} scores are not whole rows of {stride} (the emittable set plus blank)")]
    Ragged {
        /// The buffer's length.
        len: usize,
        /// One row's length.
        stride: usize,
    },
}

/// The CTC head's output as a dense matrix: one row a key, each row the
/// emittable set in emission order followed by blank.
#[derive(Clone, Debug)]
pub struct EmissionRows {
    stride: usize,
    rows: Box<[f32]>,
}

impl EmissionRows {
    /// Rows over an emittable set of *emittable* characters.
    ///
    /// # Errors
    ///
    /// If *rows* is empty or not a whole number of `emittable + 1` rows.
    pub fn new(emittable: usize, rows: impl Into<Box<[f32]>>) -> Result<Self, RowsError> {
        let rows = rows.into();
        let stride = emittable + 1;
        if rows.is_empty() {
            return Err(RowsError::Empty);
        }
        if rows.len() % stride != 0 {
            return Err(RowsError::Ragged {
                len: rows.len(),
                stride,
            });
        }
        Ok(Self { stride, rows })
    }

    /// How many characters a row scores, blank excluded.
    #[must_use]
    pub const fn emittable(&self) -> usize {
        self.stride - 1
    }

    fn row(&self, key: usize) -> &[f32] {
        &self.rows[key * self.stride..(key + 1) * self.stride]
    }
}

impl KeyEmission for EmissionRows {
    fn keys(&self) -> usize {
        self.rows.len() / self.stride
    }

    fn blank(&self, key: usize) -> f32 {
        self.row(key)[self.stride - 1]
    }

    fn gather(&self, key: usize, ids: &[EmitId], out: &mut [f32]) {
        let row = self.row(key);
        for (slot, id) in out.iter_mut().zip(ids) {
            *slot = row[id.index()];
        }
    }
}

/// Why a search shape was rejected.
#[derive(Debug, Error, PartialEq, Eq)]
#[error("top {top_k} cannot come out of a beam {width} wide")]
pub struct ShapeError {
    width: usize,
    top_k: usize,
}

/// How wide the beam is and how many sentences it returns: `top_k ≤ width`.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Shape {
    width: NonZeroUsize,
    top_k: NonZeroUsize,
}

impl Shape {
    /// The product's shape: eight beams, the panel's top 8.
    pub const PRODUCT: Self = Self {
        width: NonZeroUsize::MIN.saturating_add(7),
        top_k: NonZeroUsize::MIN.saturating_add(7),
    };

    /// A beam *width* wide returning its best `top_k`.
    ///
    /// # Errors
    ///
    /// If `top_k > width`: the search returns only sentences it carried.
    pub const fn new(width: NonZeroUsize, top_k: NonZeroUsize) -> Result<Self, ShapeError> {
        if top_k.get() > width.get() {
            return Err(ShapeError {
                width: width.get(),
                top_k: top_k.get(),
            });
        }
        Ok(Self { width, top_k })
    }

    /// How many hypotheses survive each key.
    #[must_use]
    pub const fn width(self) -> usize {
        self.width.get()
    }

    /// How many sentences the search returns.
    #[must_use]
    pub const fn top_k(self) -> usize {
        self.top_k.get()
    }
}

/// The path score's weights, chosen on dev (#216 §5).
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Weights {
    /// `β`, on the transition's log-probability.
    pub transition: f32,
    /// `γ`, on a correction arc's `−log P(span | syllable)`.
    pub correction: f32,
    /// `λ`, added once for each word arc a sentence lands.
    pub word: f32,
}

impl Weights {
    /// The constrained-CTC loss's own weighting: transition and corrections
    /// at weight one, no word prior.
    pub const UNIT: Self = Self {
        transition: 1.0,
        correction: 1.0,
        word: 0.0,
    };
}

/// One sentence the search returns: a legal reading of the keys.
///
/// Only the search builds one, from a path through the lattice, so holding a
/// `Sentence` is holding a legal output.
#[derive(Clone, Debug)]
pub struct Sentence {
    chars: Box<[EmitId]>,
    ends: Box<[usize]>,
    prefix_scores: Box<[f32]>,
    score: f32,
}

impl Sentence {
    /// The characters, as emission indices.
    #[must_use]
    pub fn chars(&self) -> &[EmitId] {
        &self.chars
    }

    /// Where each character's span ends in the sentence's best alignment:
    /// `ends[j]` keys are consumed through character `j`. The last is the
    /// key count.
    #[must_use]
    pub fn ends(&self) -> &[usize] {
        &self.ends
    }

    /// The hypothesis's score when character `j` was emitted, along the best
    /// alignment; the characters of one word arc share the arc's score.
    #[must_use]
    pub fn prefix_scores(&self) -> &[f32] {
        &self.prefix_scores
    }

    /// The whole score, the end-of-sentence term included. Larger is better.
    #[must_use]
    pub const fn score(&self) -> f32 {
        self.score
    }
}

/// What one search cost, against the bound in the module docs.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SearchStats {
    /// Keys searched.
    pub keys: usize,
    /// The most candidates (blank and character steps) scored at one key.
    pub max_candidates: usize,
    /// The most transition rows advanced at one key for character steps.
    pub max_advanced: usize,
    /// The most [`KeyTransition::score`] gathers at one key.
    pub max_transition_gathers: usize,
    /// Transition rows advanced over the whole search, word arcs included.
    pub advanced: usize,
    /// Word arcs the lattice offered.
    pub arcs_offered: usize,
    /// Word-arc lineages started (an arc times a hypothesis that could start it).
    pub arcs_started: usize,
}

/// What a search returns.
#[derive(Clone, Debug)]
pub struct Searched {
    /// The best sentences, best first; at most the shape's `top_k`.
    pub sentences: Vec<Sentence>,
    /// What the search cost.
    pub stats: SearchStats,
}

/// No node: the root's parent, the empty cut chain.
const NONE: u32 = u32::MAX;

/// `log(exp(a) + exp(b))`, exact at `−∞`.
fn log_add(a: f32, b: f32) -> f32 {
    let (high, low) = if a >= b { (a, b) } else { (b, a) };
    if low == f32::NEG_INFINITY {
        return high;
    }
    high + (low - high).exp().ln_1p()
}

fn to_u32(value: usize) -> u32 {
    u32::try_from(value).expect("a search's indices fit u32")
}

/// One distinct prefix.
struct Node<S> {
    parent: u32,
    ch: Option<EmitId>,
    /// `log P_transition(prefix)`, unweighted.
    trans: f32,
    /// The transition state, held while a hypothesis or an arc stands here.
    state: Option<S>,
    /// The last key at which the node was live.
    live: usize,
}

/// The prefix tree: one node a distinct prefix.
struct Prefixes<S> {
    nodes: Vec<Node<S>>,
    children: HashMap<(u32, EmitId), u32>,
    /// Nodes holding a state, so dropping dead ones is linear in them.
    stateful: Vec<u32>,
}

impl<S: Clone> Prefixes<S> {
    fn new(start: S) -> Self {
        Self {
            nodes: vec![Node {
                parent: NONE,
                ch: None,
                trans: 0.0,
                state: Some(start),
                live: 0,
            }],
            children: HashMap::new(),
            stateful: vec![0],
        }
    }

    fn state(&self, node: u32) -> &S {
        self.nodes[node as usize]
            .state
            .as_ref()
            .expect("a live prefix holds its state")
    }

    /// The node for *parent* followed by *ch*, scored *step* when new.
    fn child(&mut self, parent: u32, ch: EmitId, step: f32) -> u32 {
        let next = to_u32(self.nodes.len());
        let id = *self.children.entry((parent, ch)).or_insert(next);
        if id == next {
            let trans = self.nodes[parent as usize].trans + step;
            self.nodes.push(Node {
                parent,
                ch: Some(ch),
                trans,
                state: None,
                live: 0,
            });
        }
        id
    }

    /// Advance every node of *nodes* that holds no state, in one batch;
    /// returns how many rows were advanced.
    fn fill<T: KeyTransition<State = S>>(&mut self, transition: &T, nodes: &[u32]) -> usize {
        let mut missing: Vec<u32> = nodes
            .iter()
            .copied()
            .filter(|&node| self.nodes[node as usize].state.is_none())
            .collect();
        missing.sort_unstable();
        missing.dedup();
        if missing.is_empty() {
            return 0;
        }
        let steps: Vec<(&S, EmitId)> = missing
            .iter()
            .map(|&node| {
                let entry = &self.nodes[node as usize];
                (
                    self.state(entry.parent),
                    entry.ch.expect("only the root has no character"),
                )
            })
            .collect();
        let states = transition.advance(&steps);
        assert_eq!(
            states.len(),
            missing.len(),
            "advance returns one state a step"
        );
        for (&node, state) in missing.iter().zip(states) {
            self.nodes[node as usize].state = Some(state);
            self.stateful.push(node);
        }
        missing.len()
    }

    /// Drop the state of every node not marked live at *key*.
    fn release(&mut self, key: usize) {
        let nodes = &mut self.nodes;
        self.stateful.retain(|&node| {
            let entry = &mut nodes[node as usize];
            if entry.live == key {
                true
            } else {
                entry.state = None;
                false
            }
        });
    }

    fn chars(&self, mut node: u32) -> Vec<EmitId> {
        let mut chars = Vec::new();
        while let Some(ch) = self.nodes[node as usize].ch {
            chars.push(ch);
            node = self.nodes[node as usize].parent;
        }
        chars.reverse();
        chars
    }
}

/// One emission of a hypothesis's best alignment.
#[derive(Copy, Clone)]
struct Cut {
    parent: u32,
    end: u32,
    score: f32,
}

/// A hypothesis: a prefix and the offset its last emission closed at.
#[derive(Copy, Clone)]
struct Hyp {
    prefix: u32,
    pending: u32,
    /// The log-sum over its alignments of emission and correction terms,
    /// word priors included.
    align: f32,
    /// The best alignment's last cut.
    cut: u32,
}

/// A word-arc lineage between its start and the key it lands at.
struct InFlight {
    /// The node before the word's last character.
    parent: u32,
    last: EmitId,
    /// The node of the whole word.
    node: u32,
    /// The key it lands at: its last span's last key.
    lands: usize,
    /// The spawning hypothesis's alignment term plus the arc's.
    align: f32,
    /// The arc's last cut.
    cut: u32,
    /// The word's length, the cuts it pushed.
    len: usize,
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum Kind {
    /// The hypothesis reads the key as blank.
    Blank,
    /// The hypothesis emits the character.
    Emit(EmitId),
    /// A word arc lands with the character as its last.
    Arc(EmitId),
}

impl Kind {
    /// The character a non-blank step emits.
    fn ch(self) -> EmitId {
        match self {
            Self::Emit(ch) | Self::Arc(ch) => ch,
            Self::Blank => unreachable!("a blank step emits nothing"),
        }
    }
}

/// One way forward at one key, before selection.
#[derive(Copy, Clone)]
struct Cand {
    total: f32,
    align: f32,
    /// `log P(ch | parent)`, for emissions and arcs.
    step: f32,
    /// The prefix node it extends (blank: the node it stays at).
    parent: u32,
    /// The hypothesis (blank, emission) or in-flight arc it comes from.
    source: u32,
    kind: Kind,
}

/// One word arc as the search uses it.
struct Offered {
    start: usize,
    end: usize,
    chars: Box<[EmitId]>,
    /// The arc's alignment term over keys `start..end`, prior included.
    align: f32,
    /// The best cover's span ends.
    ends: Box<[u32]>,
}

/// The alignment term of *chars* over keys `start..end`: the log-sum over
/// every cover of the span by the characters, on *lattice*'s edges, of the
/// emission and correction terms, and the best cover's ends.
fn arc_term<E: KeyEmission>(
    lattice: &KeyLattice,
    emission: &E,
    blanks: &[f32],
    weights: Weights,
    start: usize,
    end: usize,
    chars: &[EmitId],
) -> Option<(f32, Box<[u32]>)> {
    let width = end - start;
    // `sum[o]`/`best[o]`: covers of keys `start..start + o` by the characters
    // placed so far; `back[j][o]`: the best cover's previous offset.
    let mut sum = vec![f32::NEG_INFINITY; width + 1];
    let mut best = vec![f32::NEG_INFINITY; width + 1];
    sum[0] = 0.0;
    best[0] = 0.0;
    let mut back = vec![vec![usize::MAX; width + 1]; chars.len()];
    let mut out = [0.0f32];
    for (j, &ch) in chars.iter().enumerate() {
        let mut next_sum = vec![f32::NEG_INFINITY; width + 1];
        let mut next_best = vec![f32::NEG_INFINITY; width + 1];
        for from in 0..width {
            if sum[from] == f32::NEG_INFINITY {
                continue;
            }
            let mut held = 0.0f32;
            for to in from + 1..=width.min(from + MAX_CORRECTED_LEN) {
                let Some(edge) = lattice.edge(start + from, start + to) else {
                    held += blanks[start + to - 1];
                    continue;
                };
                if let Some(cost) = edge.cost_of(ch) {
                    emission.gather(start + to - 1, &[ch], &mut out);
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "a correction cost is a few nats"
                    )]
                    let term = held + out[0] - weights.correction * cost as f32;
                    next_sum[to] = log_add(next_sum[to], sum[from] + term);
                    if best[from] + term > next_best[to] {
                        next_best[to] = best[from] + term;
                        back[j][to] = from;
                    }
                }
                held += blanks[start + to - 1];
            }
        }
        sum = next_sum;
        best = next_best;
    }
    if sum[width] == f32::NEG_INFINITY {
        return None;
    }
    let mut ends = vec![0u32; chars.len()];
    let mut at = width;
    for j in (0..chars.len()).rev() {
        ends[j] = to_u32(start + at);
        at = back[j][at];
    }
    Some((sum[width] + weights.word, ends.into()))
}

/// The constrained beam search over *lattice*'s keys.
///
/// *emission* scores the same keys; *transition* is bound to the same keys
/// and context; *words*, when given, offers its entries as arcs.
///
/// # Panics
///
/// If *emission* scores a different number of keys than *lattice* holds, or
/// the transition breaks its contract.
#[must_use]
#[expect(
    clippy::too_many_lines,
    reason = "one key's expansion, selection and advance read as one sequence"
)]
pub fn search<E: KeyEmission, T: KeyTransition>(
    lattice: &KeyLattice,
    emission: &E,
    transition: &T,
    words: Option<&WordArcs>,
    shape: Shape,
    weights: Weights,
) -> Searched {
    let keys = lattice.keys();
    assert_eq!(
        emission.keys(),
        keys,
        "the emission scores {} keys, the lattice holds {keys}",
        emission.keys()
    );
    let beta = weights.transition;
    let mut stats = SearchStats {
        keys,
        ..SearchStats::default()
    };
    let blanks: Vec<f32> = (0..keys).map(|key| emission.blank(key)).collect();

    // The word arcs, by start offset.
    let mut offered: Vec<Offered> = Vec::new();
    if let Some(words) = words {
        for (start, end, chars) in words.offered(lattice) {
            stats.arcs_offered += 1;
            if let Some((align, ends)) =
                arc_term(lattice, emission, &blanks, weights, start, end, chars)
            {
                offered.push(Offered {
                    start,
                    end,
                    chars: chars.into(),
                    align,
                    ends,
                });
            }
        }
    }
    let mut arcs_from: Vec<Vec<usize>> = vec![Vec::new(); keys];
    for (index, arc) in offered.iter().enumerate() {
        arcs_from[arc.start].push(index);
    }

    let mut prefixes = Prefixes::new(transition.start());
    let mut cuts: Vec<Cut> = Vec::new();
    let mut hyps = vec![Hyp {
        prefix: 0,
        pending: 0,
        align: 0.0,
        cut: NONE,
    }];
    let mut inflight: Vec<InFlight> = Vec::new();
    let mut cands: Vec<Cand> = Vec::new();
    let mut gathered: Vec<(u32, Vec<f32>)> = Vec::new();
    let mut merged: Vec<(EmitId, f32, u32, f32)> = Vec::new();
    let mut trans_out: Vec<f32> = Vec::new();
    let mut ids: Vec<EmitId> = Vec::new();
    // Where each prefix's emission candidates sit in `cands` at this key.
    let mut groups: HashMap<u32, (usize, usize)> = HashMap::new();

    for key in 0..keys {
        let last = key + 1 == keys;
        hyps.sort_unstable_by_key(|hyp| (hyp.prefix, hyp.pending));

        // Word arcs start from the hypotheses that stand at this offset.
        if !arcs_from[key].is_empty() {
            spawn(
                &arcs_from[key],
                &offered,
                &hyps,
                key,
                transition,
                &mut prefixes,
                &mut cuts,
                &mut inflight,
                &mut stats,
            );
        }

        cands.clear();
        gathered.clear();
        groups.clear();
        let mut gathers = 0usize;
        let mut group_start = 0;
        while group_start < hyps.len() {
            let prefix = hyps[group_start].prefix;
            let group_end = group_start
                + hyps[group_start..]
                    .iter()
                    .take_while(|hyp| hyp.prefix == prefix)
                    .count();
            merged.clear();
            for (offset, hyp) in hyps[group_start..group_end].iter().enumerate() {
                let source = to_u32(group_start + offset);
                let step = lattice.step(hyp.pending as usize, key);
                if step.blank() && !last {
                    cands.push(Cand {
                        total: hyp.align
                            + blanks[key]
                            + beta * prefixes.nodes[prefix as usize].trans,
                        align: hyp.align + blanks[key],
                        step: 0.0,
                        parent: prefix,
                        source,
                        kind: Kind::Blank,
                    });
                }
                if step.ids().is_empty() {
                    continue;
                }
                // One gather per distinct pending offset, shared by every
                // hypothesis that stands there.
                let row =
                    if let Some(index) = gathered.iter().position(|(at, _)| *at == hyp.pending) {
                        index
                    } else {
                        let mut row = vec![0.0f32; step.ids().len()];
                        emission.gather(key, step.ids(), &mut row);
                        gathered.push((hyp.pending, row));
                        gathered.len() - 1
                    };
                let row = &gathered[row].1;
                for ((&id, &cost), &logp) in step.ids().iter().zip(step.costs()).zip(row) {
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "a correction cost is a few nats"
                    )]
                    let align = hyp.align + logp - weights.correction * cost as f32;
                    merged.push((id, align, source, align));
                }
            }
            if !merged.is_empty() {
                // Several pending offsets of one prefix may emit the same
                // character here: their alignments are disjoint, so they sum.
                if group_end - group_start > 1 {
                    merged.sort_unstable_by_key(|entry| entry.0);
                    let mut write = 0;
                    for read in 0..merged.len() {
                        if write > 0 && merged[write - 1].0 == merged[read].0 {
                            let (_, align, source, _) = merged[read];
                            let kept = &mut merged[write - 1];
                            kept.1 = log_add(kept.1, align);
                            if align > kept.3 {
                                kept.2 = source;
                                kept.3 = align;
                            }
                        } else {
                            merged[write] = merged[read];
                            write += 1;
                        }
                    }
                    merged.truncate(write);
                }
                ids.clear();
                ids.extend(merged.iter().map(|entry| entry.0));
                trans_out.clear();
                trans_out.resize(ids.len(), 0.0);
                transition.score(prefixes.state(prefix), &ids, &mut trans_out);
                gathers += 1;
                let base = prefixes.nodes[prefix as usize].trans;
                let first = cands.len();
                for (&(ch, align, source, _), &step) in merged.iter().zip(&trans_out) {
                    cands.push(Cand {
                        total: align + beta * (base + step),
                        align,
                        step,
                        parent: prefix,
                        source,
                        kind: Kind::Emit(ch),
                    });
                }
                groups.insert(prefix, (first, cands.len()));
            }
            group_start = group_end;
        }

        // Word arcs that land here join their character-by-character twins.
        land(&inflight, key, beta, &prefixes, &groups, &mut cands);

        stats.max_candidates = stats.max_candidates.max(cands.len());
        stats.max_transition_gathers = stats.max_transition_gathers.max(gathers);

        // The `width` best, in score order, ties by candidate order.
        let width = shape.width().min(cands.len());
        let mut order: Vec<usize> = (0..cands.len()).collect();
        let better =
            |a: &usize, b: &usize| cands[*b].total.total_cmp(&cands[*a].total).then(a.cmp(b));
        if width < order.len() {
            order.select_nth_unstable_by(width - 1, better);
            order.truncate(width);
        }
        order.sort_unstable_by(better);

        let mut next: Vec<Hyp> = Vec::with_capacity(order.len());
        let mut fresh: Vec<u32> = Vec::new();
        for &index in &order {
            let cand = cands[index];
            match cand.kind {
                Kind::Blank => {
                    let hyp = hyps[cand.source as usize];
                    next.push(Hyp {
                        align: cand.align,
                        ..hyp
                    });
                }
                Kind::Emit(ch) => {
                    let node = prefixes.child(cand.parent, ch, cand.step);
                    cuts.push(Cut {
                        parent: hyps[cand.source as usize].cut,
                        end: to_u32(key + 1),
                        score: cand.total,
                    });
                    next.push(Hyp {
                        prefix: node,
                        pending: to_u32(key + 1),
                        align: cand.align,
                        cut: to_u32(cuts.len() - 1),
                    });
                    fresh.push(node);
                }
                Kind::Arc(_) => {
                    let arc = &inflight[cand.source as usize];
                    let mut cut = arc.cut;
                    for _ in 0..arc.len {
                        cuts[cut as usize].score = cand.total;
                        cut = cuts[cut as usize].parent;
                    }
                    next.push(Hyp {
                        prefix: arc.node,
                        pending: to_u32(key + 1),
                        align: cand.align,
                        cut: arc.cut,
                    });
                }
            }
        }
        let advanced = prefixes.fill(transition, &fresh);
        stats.max_advanced = stats.max_advanced.max(advanced);
        stats.advanced += advanced;

        inflight.retain(|arc| arc.lands > key);
        for hyp in &next {
            prefixes.nodes[hyp.prefix as usize].live = key + 1;
        }
        for arc in &inflight {
            prefixes.nodes[arc.node as usize].live = key + 1;
        }
        prefixes.release(key + 1);
        hyps = next;
    }

    let mut sentences: Vec<Sentence> = hyps
        .iter()
        .map(|hyp| {
            debug_assert_eq!(hyp.pending as usize, keys, "only closed hypotheses finish");
            let node = &prefixes.nodes[hyp.prefix as usize];
            let score =
                hyp.align + beta * (node.trans + transition.finish(prefixes.state(hyp.prefix)));
            let chars = prefixes.chars(hyp.prefix);
            let mut ends = Vec::with_capacity(chars.len());
            let mut prefix_scores = Vec::with_capacity(chars.len());
            let mut cut = hyp.cut;
            while cut != NONE {
                let entry = cuts[cut as usize];
                ends.push(entry.end as usize);
                prefix_scores.push(entry.score);
                cut = entry.parent;
            }
            ends.reverse();
            prefix_scores.reverse();
            assert_eq!(ends.len(), chars.len(), "one cut a character");
            Sentence {
                chars: chars.into(),
                ends: ends.into(),
                prefix_scores: prefix_scores.into(),
                score,
            }
        })
        .collect();
    sentences.sort_by(|a, b| b.score.total_cmp(&a.score));
    sentences.truncate(shape.top_k());
    Searched { sentences, stats }
}

/// Start the arcs *starting* here from every hypothesis whose last emission
/// closed at *key*: score and advance the transition through each word, one
/// batch a character position, and hold the lineage until it lands.
#[expect(
    clippy::too_many_arguments,
    reason = "one spawn reads the arcs, the beam and the search's arenas"
)]
fn spawn<T: KeyTransition>(
    starting: &[usize],
    offered: &[Offered],
    hyps: &[Hyp],
    key: usize,
    transition: &T,
    prefixes: &mut Prefixes<T::State>,
    cuts: &mut Vec<Cut>,
    inflight: &mut Vec<InFlight>,
    stats: &mut SearchStats,
) {
    // `(arc, hypothesis, the node the lineage stands on)`.
    let mut lineages: Vec<(usize, usize, u32)> = Vec::new();
    for (index, hyp) in hyps.iter().enumerate() {
        if hyp.pending as usize == key {
            lineages.extend(starting.iter().map(|&arc| (arc, index, hyp.prefix)));
        }
    }
    if lineages.is_empty() {
        return;
    }
    stats.arcs_started += lineages.len();
    let longest = lineages
        .iter()
        .map(|&(arc, _, _)| offered[arc].chars.len())
        .max()
        .unwrap_or(0);
    let mut out = [0.0f32];
    let mut nodes = Vec::with_capacity(lineages.len());
    for position in 0..longest {
        nodes.clear();
        for lineage in &mut lineages {
            let Some(&ch) = offered[lineage.0].chars.get(position) else {
                continue;
            };
            transition.score(prefixes.state(lineage.2), &[ch], &mut out);
            lineage.2 = prefixes.child(lineage.2, ch, out[0]);
            nodes.push(lineage.2);
        }
        stats.advanced += prefixes.fill(transition, &nodes);
    }
    for &(arc, hyp, node) in &lineages {
        let word = &offered[arc];
        let mut cut = hyps[hyp].cut;
        for &end in &word.ends {
            cuts.push(Cut {
                parent: cut,
                end,
                score: 0.0,
            });
            cut = to_u32(cuts.len() - 1);
        }
        inflight.push(InFlight {
            parent: prefixes.nodes[node as usize].parent,
            last: *word.chars.last().expect("a word has characters"),
            node,
            lands: word.end - 1,
            align: hyps[hyp].align + word.align,
            cut,
            len: word.chars.len(),
        });
    }
}

/// Add the arcs landing at *key* to *cands*: arcs on one word node sum,
/// and an arc meets its character-by-character twin by maximum.
fn land<S: Clone>(
    inflight: &[InFlight],
    key: usize,
    beta: f32,
    prefixes: &Prefixes<S>,
    groups: &HashMap<u32, (usize, usize)>,
    cands: &mut Vec<Cand>,
) {
    let mut landing: Vec<usize> = (0..inflight.len())
        .filter(|&index| inflight[index].lands == key)
        .collect();
    if landing.is_empty() {
        return;
    }
    landing.sort_by(|&a, &b| {
        inflight[a]
            .node
            .cmp(&inflight[b].node)
            .then(inflight[b].align.total_cmp(&inflight[a].align))
    });
    let mut index = 0;
    while index < landing.len() {
        let first = landing[index];
        let node = inflight[first].node;
        let mut align = inflight[first].align;
        index += 1;
        while index < landing.len() && inflight[landing[index]].node == node {
            align = log_add(align, inflight[landing[index]].align);
            index += 1;
        }
        let arc = &inflight[first];
        let entry = &prefixes.nodes[node as usize];
        let step = entry.trans - prefixes.nodes[arc.parent as usize].trans;
        let total = align + beta * entry.trans;
        let twin = groups.get(&arc.parent).and_then(|&(from, to)| {
            cands[from..to]
                .binary_search_by_key(&arc.last, |cand| cand.kind.ch())
                .ok()
                .map(|offset| from + offset)
        });
        let cand = Cand {
            total,
            align,
            step,
            parent: arc.parent,
            source: to_u32(first),
            kind: Kind::Arc(arc.last),
        };
        match twin {
            Some(at) if cands[at].align >= align => {}
            Some(at) => cands[at] = cand,
            None => cands.push(cand),
        }
    }
}

#[cfg(test)]
mod tests;
