//! The key engine: the key-level model through the key search, as a
//! session's [`Engine`].
//!
//! A keystroke builds the legality lattice of the pending letters — a typed
//! `'` a forced cut, the last syllable possibly unfinished unless a `'`
//! closed it — runs the model's keys function once over all the letters, and
//! searches. The search's transition is a row cache over that keystroke's
//! encoding: before the search asks anything, a speculative pass computes the
//! rows of the prefixes the previous search over these letters' prefix
//! queried; a query whose row is still missing is a miss, and the misses of
//! one search call are planned together into tree calls. Every node of every
//! call is cached, the first value written staying, so a row is the same
//! wherever the search reads it.
//!
//! The model is bidirectional over the keys, so no model state survives a
//! keystroke: the encoding and the rows are dropped with it. What survives is
//! the queried prefixes, one set per pending length, so a backspace
//! speculates with what the shorter input asked before.

use crate::{Engine, Pending, Placed, Sentence};
use ime_decode::{
    KeyHypothesis, KeyLogProbs, KeyLogProbsError, KeySearchOptions, KeySearchStats, KeyTransition,
    PrefixId, PrefixTrie, key_search,
};
use ime_neural::keys::{KeyModel, PlanError, TREE_BUCKETS, plan};
use ime_pinyin::{EmitId, Legality, LegalityOptions};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

/// A prefix of a sentence: the emission ids of its characters so far.
type Prefix = Box<[EmitId]>;

/// What a key engine searches under.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct KeyEngineOptions {
    /// The legality rules: [`LegalityOptions::LIVE`] for typing, whose
    /// unfinished last syllable a closing `'` turns off.
    pub legality: LegalityOptions,
    /// The search's weights, beam and outputs.
    pub search: KeySearchOptions,
}

/// Why a key engine's keystroke failed.
#[derive(Debug, Error)]
pub enum KeyEngineError<E: std::error::Error + 'static> {
    /// The model failed a call.
    #[error(transparent)]
    Model(E),
    /// A prefix did not fit the largest tree call.
    #[error(transparent)]
    Plan(#[from] PlanError),
    /// The model's log-probs do not fit the emittable set.
    #[error(transparent)]
    LogProbs(#[from] KeyLogProbsError),
}

/// What one keystroke's transition did: what the bucket set and the
/// speculation are judged by.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KeystrokeStats {
    /// The keys the keystroke searched.
    pub keys: usize,
    /// Tree calls made, the speculative pass's and the misses', by bucket in
    /// [`TREE_BUCKETS`] order.
    pub calls: [u64; TREE_BUCKETS.len()],
    /// Rows the speculative pass cached: the closure of the queries it
    /// replayed.
    pub speculative_rows: usize,
    /// Tree calls the misses made.
    pub miss_calls: usize,
    /// Distinct prefixes the search asked.
    pub queried: usize,
    /// Nodes of their ancestor closure, root included.
    pub closure: usize,
    /// Rows handed to the search.
    pub answered: usize,
    /// The search's own count of its transition calls and rows.
    pub search: KeySearchStats,
}

/// The queried prefixes that survive keystrokes: per letter count, what the
/// latest search over that prefix of the current letters asked.
#[derive(Default)]
struct Asked {
    /// The letters the entries are prefixes of.
    letters: String,
    /// `(letter count, the distinct prefixes queried in ask order)`, by
    /// ascending letter count.
    entries: Vec<(usize, Vec<Prefix>)>,
}

impl Asked {
    /// Keep only the entries that are prefixes of *letters* too; the queries
    /// of the longest one, the ones to speculate with.
    fn align(&mut self, letters: &str) -> &[Prefix] {
        let common = self
            .letters
            .bytes()
            .zip(letters.bytes())
            .take_while(|(a, b)| a == b)
            .count();
        while self
            .entries
            .last()
            .is_some_and(|&(count, _)| count > common)
        {
            self.entries.pop();
        }
        self.entries.last().map_or(&[], |(_, queried)| queried)
    }

    /// What the search over *letters* queried, replacing an earlier search
    /// over the same letters.
    fn record(&mut self, letters: &str, queried: Vec<Prefix>) {
        if self
            .entries
            .last()
            .is_some_and(|&(count, _)| count == letters.len())
        {
            self.entries.pop();
        }
        self.entries.push((letters.len(), queried));
        letters.clone_into(&mut self.letters);
    }

    fn clear(&mut self) {
        self.letters.clear();
        self.entries.clear();
    }
}

/// One keystroke's transition: the model's rows over its encoding, cached
/// by prefix.
struct RowCache<'k, M: KeyModel> {
    model: &'k mut M,
    encoding: &'k M::Encoding,
    /// The emittable characters: a row's width.
    emittable: usize,
    /// Every call's logits, in call order.
    outputs: Vec<M::Logits>,
    /// prefix -> (its call, its row there): the first call that computed it.
    rows: HashMap<Prefix, (usize, usize)>,
    /// The distinct prefixes the search asked, in ask order.
    queried: Vec<Prefix>,
    seen: HashSet<Prefix>,
    stats: &'k mut KeystrokeStats,
}

impl<M: KeyModel> RowCache<'_, M> {
    /// Plan *requests* into tree calls, run them and cache every node the
    /// cache does not hold yet; the number of calls made.
    fn run<P: AsRef<[EmitId]>>(
        &mut self,
        requests: &[P],
    ) -> Result<usize, KeyEngineError<M::Error>> {
        let calls = plan(requests)?;
        for call in &calls {
            let logits = self
                .model
                .tree(self.encoding, call)
                .map_err(KeyEngineError::Model)?;
            let at = self.outputs.len();
            self.outputs.push(logits);
            let bucket = TREE_BUCKETS
                .iter()
                .position(|&size| size == call.bucket())
                .expect("a call runs in one of the buckets");
            self.stats.calls[bucket] += 1;
            for (row, &node) in call.nodes().iter().enumerate() {
                if !self.rows.contains_key(node) {
                    self.rows.insert(node.into(), (at, row));
                }
            }
        }
        Ok(calls.len())
    }
}

impl<M: KeyModel> KeyTransition for RowCache<'_, M> {
    type Error = KeyEngineError<M::Error>;

    fn rows(
        &mut self,
        trie: &PrefixTrie,
        queries: &[PrefixId],
        out: &mut [f32],
    ) -> Result<(), Self::Error> {
        let mut ids = Vec::new();
        let prefixes: Vec<Prefix> = queries
            .iter()
            .map(|&query| {
                trie.ids_into(query, &mut ids);
                ids.as_slice().into()
            })
            .collect();
        let missing: Vec<&[EmitId]> = prefixes
            .iter()
            .map(AsRef::as_ref)
            .filter(|prefix| !self.rows.contains_key(*prefix))
            .collect();
        if !missing.is_empty() {
            self.stats.miss_calls += self.run(&missing)?;
        }
        for (prefix, row) in prefixes
            .into_iter()
            .zip(out.chunks_exact_mut(self.emittable))
        {
            let (call, node) = self.rows[&prefix];
            self.model.log_softmax(&self.outputs[call], node, row);
            if self.seen.insert(prefix.clone()) {
                self.queried.push(prefix);
            }
        }
        self.stats.answered += queries.len();
        Ok(())
    }
}

/// Nodes in the ancestor closure of *prefixes*, root included.
fn closure_size(prefixes: &[Prefix]) -> usize {
    prefixes
        .iter()
        .flat_map(|prefix| (0..=prefix.len()).map(move |length| &prefix[..length]))
        .collect::<HashSet<&[EmitId]>>()
        .len()
}

/// The key-level model *M* through the key search, as a session's
/// [`Engine`].
pub struct KeyEngine<'a, M> {
    model: M,
    legality: &'a Legality,
    options: KeyEngineOptions,
    /// The queried prefixes per pending length.
    asked: Asked,
    /// The keys function's log-probs, reused across keystrokes.
    log_probs: Vec<f32>,
    /// What the last keystroke's transition did.
    last: KeystrokeStats,
}

impl<'a, M: KeyModel> KeyEngine<'a, M> {
    /// An engine over *model*, whose emission order is *legality*'s.
    #[must_use]
    pub fn new(model: M, legality: &'a Legality, options: KeyEngineOptions) -> Self {
        Self {
            model,
            legality,
            options,
            asked: Asked::default(),
            log_probs: Vec::new(),
            last: KeystrokeStats::default(),
        }
    }

    /// The model the engine runs.
    #[must_use]
    pub const fn model(&self) -> &M {
        &self.model
    }

    /// What the last keystroke's transition did.
    #[must_use]
    pub const fn last_keystroke(&self) -> &KeystrokeStats {
        &self.last
    }

    /// The distinct prefixes the last search queried, in ask order: what the
    /// next keystroke over these letters speculates with.
    #[must_use]
    pub fn queried(&self) -> &[Prefix] {
        self.asked
            .entries
            .last()
            .map_or(&[], |(_, queried)| queried)
    }

    /// A hypothesis as a sentence, its characters by the emittable set.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the session's candidates carry f32 scores"
    )]
    fn sentence(&self, hypothesis: &KeyHypothesis) -> Sentence {
        let emittable = self.legality.emittable();
        Sentence {
            chars: hypothesis
                .ids
                .iter()
                .zip(&hypothesis.cuts)
                .zip(&hypothesis.steps)
                .map(|((id, &end), &step)| Placed {
                    ch: emittable[id.index()],
                    end,
                    score: step as f32,
                })
                .collect(),
            score: hypothesis.score as f32,
        }
    }
}

impl<M: KeyModel> Engine for KeyEngine<'_, M> {
    type Error = KeyEngineError<M::Error>;

    fn sentences(
        &mut self,
        pending: &Pending,
        context: Option<&str>,
    ) -> Result<Vec<Sentence>, Self::Error> {
        let letters = pending.letters();
        let options = LegalityOptions {
            unfinished_tail: self.options.legality.unfinished_tail && !pending.closed(),
            ..self.options.legality
        };
        let mut lattice = self.legality.lattice(letters, options);
        if !pending.cuts().is_empty() {
            lattice = lattice.with_forced_cuts(pending.cuts());
        }
        let speculate = self.asked.align(letters.as_str());
        let mut stats = KeystrokeStats {
            keys: letters.len(),
            ..KeystrokeStats::default()
        };
        if lattice.is_empty() {
            // No cover: the search would ask nothing, so nothing is run.
            self.asked.record(letters.as_str(), Vec::new());
            self.last = stats;
            return Ok(Vec::new());
        }
        let encoding = self
            .model
            .encode(letters, context, &mut self.log_probs)
            .map_err(KeyEngineError::Model)?;
        let emittable = self.legality.emittable().len();
        let log_probs = KeyLogProbs::new(letters.len(), emittable + 1, &self.log_probs)?;
        let mut rows = RowCache {
            model: &mut self.model,
            encoding: &encoding,
            emittable,
            outputs: Vec::new(),
            rows: HashMap::new(),
            queried: Vec::new(),
            seen: HashSet::new(),
            stats: &mut stats,
        };
        if !speculate.is_empty() {
            rows.run(speculate)?;
        }
        rows.stats.speculative_rows = rows.rows.len();
        let mut search = KeySearchStats::default();
        let found = key_search(
            &lattice,
            &log_probs,
            &[],
            &mut rows,
            &self.options.search,
            &mut search,
        )?;
        let queried = rows.queried;
        stats.search = search;
        stats.queried = queried.len();
        stats.closure = closure_size(&queried);
        self.asked.record(letters.as_str(), queried);
        self.last = stats;
        Ok(found
            .iter()
            .map(|hypothesis| self.sentence(hypothesis))
            .collect())
    }

    fn clear(&mut self) {
        self.asked.clear();
    }
}
