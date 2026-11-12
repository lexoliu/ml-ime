//! The fused engine: route A's towers for the emissions, whatever transition
//! the run carries — the kn-trigram, the character LM, the two fused at their
//! weights, or none — the decode `fused-eval` runs, as a session's
//! [`Engine`].
//!
//! A refresh segments the pending input into its k-best readings (each
//! `'`-separated piece alone, only the last one possibly half-typed), builds
//! the candidate sets and the word arcs over them, scores them with the
//! towers, and runs the beam; a hypothesis becomes a sentence whose
//! characters end where its reading's syllables end.
//!
//! `incremental` is the session's reuse switch: the context tower's output
//! depends only on the context, so a keystroke that changes none of it never
//! re-encodes it. The fill tower's output is not cached: it sees the whole
//! reading, and earlier positions' scores do move when a syllable joins —
//! the issue's gate is that the incremental answer equals the full decode's,
//! so the fill graph re-runs on each keystroke as one batched forward.

use crate::{Engine, Pending, Placed, Sentence, nanos};
use ime_decode::{
    BeamOptions, Breakdown, Candidates, DecodeError, Emittable, LatticeRecord, Record, Scored,
    Transition, Weighted, WordLexicon, decode_many_stats, lattice_paths,
};
use ime_neural::{EncodedContext, NeuralError, RouteA};
use ime_pinyin::{
    CharId, Lexicon, SegmentError, SegmentLattice, SegmentOptions, Segmentation, SyllableTable,
};
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::time::Instant;
use thiserror::Error;

/// What a fused engine decodes under, apart from the transition.
#[derive(Clone)]
pub struct FusedConfig<'a> {
    /// The syllable inventory the keystrokes segment against.
    pub table: &'a SyllableTable,
    /// How the keystrokes may segment. The live default admits a trailing
    /// half-typed syllable; an offline replay can turn it off to mirror an
    /// evaluation exactly.
    pub segment: SegmentOptions,
    /// The beam's width and arity — the product's shape is width 16, top 8
    /// — and the word arcs' prior and slots.
    pub beam: BeamOptions,
    /// The curated lexicon whose entries the beam proposes whole where the
    /// keystrokes spell them; `None` decodes character by character only.
    pub words: Option<&'a WordLexicon>,
    /// The emittable set the lattice's candidates are restricted to.
    pub emittable: &'a Emittable,
    /// The character lexicon every table is indexed through: the emittable
    /// set, the candidates and the sentences name characters through it.
    pub lexicon: &'a Lexicon,
    /// Neural weight — `Weighted`'s scale on every emission score.
    pub weight: f32,
    /// What a candidate the model has no output row for scores.
    pub floor: f32,
    /// Whether the context tower feeds the fill decoder's gate.
    pub with_context: bool,
    /// Whether the context tower's output is kept between keystrokes. Off is
    /// the baseline every keystroke re-runs; on is the measured reuse.
    pub incremental: bool,
}

/// Cumulative counters — where a fused engine's emission time goes.
///
/// `Cell` fields rather than a snapshot method because the engine is borrowed
/// shared through the decode: the counters read back without a mutable
/// borrow.
#[derive(Debug, Default)]
pub struct EngineStats {
    /// Context-tower forwards run.
    pub context_calls: Cell<u64>,
    /// Context-tower calls the per-context cache answered.
    pub context_cached: Cell<u64>,
    /// Nanos inside the context tower.
    pub context_ns: Cell<u64>,
    /// Nanos inside the fill tower.
    pub fill_ns: Cell<u64>,
    /// Fill-tower forwards run — the measured session's own counter, so a
    /// replay's interleaved baseline cannot double it.
    pub fill_calls: Cell<u64>,
    /// Tokens the fill graph encoded, `rows * width` summed over calls.
    pub fill_tokens: Cell<u64>,
}

/// Where a fused engine's refreshes spend their time, cumulative.
#[derive(Clone, Debug, Default)]
pub struct FusedStages {
    /// Refreshes that produced no reading: a half-typed syllable under a
    /// strict segmentation is a normal, cheap miss, not an error.
    pub undecodable: u64,
    /// Nanos spent segmenting and resolving the candidate masks.
    pub segment_ns: u64,
    /// Nanos inside the towers' emission.
    pub emission_ns: u64,
    /// Nanos inside the beam decode.
    pub decode_ns: u64,
    /// The beam's host-side breakdown — expansion, selection, and the
    /// finish pass, cumulative beside `decode_ns`.
    pub breakdown: Breakdown,
}

/// How a keystroke changed the readings, across a fused engine's refreshes.
///
/// Bookkeeping for the incremental step's evidence: a keystroke that only
/// extends the last syllable leaves the ranked paths' earlier spans alone,
/// and the histogram says where the first change lands when it does not.
#[derive(Clone, Debug, Default)]
pub struct Invalidation {
    /// Refreshes that had a previous lattice to compare against.
    pub compared: u64,
    /// Of them, how many kept every ranked path's spans identical.
    pub stable: u64,
    /// First position where a ranked path's spans diverged, counted.
    pub first_divergence: BTreeMap<usize, u64>,
}

/// Why a fused refresh failed.
#[derive(Debug, Error)]
pub enum FusedError {
    /// The keystrokes could not be read under the options.
    #[error(transparent)]
    Segment(#[from] SegmentError),
    /// The candidate masks could not be built.
    #[error(transparent)]
    Decode(#[from] DecodeError),
    /// The towers failed their forward.
    #[error(transparent)]
    Neural(#[from] NeuralError),
}

/// Cumulative counters — where a transition's calls go. The beam's `advance`
/// is the LM's step graph or the n-gram's lookups; `start` is the LM's
/// context prefill where one is fused.
#[derive(Debug, Default)]
pub struct TransitionStats {
    /// `start` calls.
    pub start_calls: Cell<u64>,
    /// Nanos inside `start`.
    pub start_ns: Cell<u64>,
    /// `advance` calls — one per beam position per keystroke.
    pub advance_calls: Cell<u64>,
    /// Nanos inside `advance`.
    pub advance_ns: Cell<u64>,
    /// Keystrokes presented through `advance` — the steps' total length.
    pub advanced_states: Cell<u64>,
    /// `score` calls — one per (beam, character) pair the beam expands.
    pub score_calls: Cell<u64>,
    /// Nanos inside `score`.
    pub score_ns: Cell<u64>,
}

/// A transition measured: delegates every call to the inner model and keeps
/// the nanos. `HISTORY` and `State` are the inner's own, so a timed model
/// decodes exactly as the model alone does.
#[derive(Debug, Default)]
pub struct Timed<T> {
    /// The model doing the work.
    inner: T,
    /// Where its calls spent their time.
    stats: TransitionStats,
}

impl<T> Timed<T> {
    /// Wrap *transition*.
    #[must_use]
    pub fn new(transition: T) -> Self {
        Self {
            inner: transition,
            stats: TransitionStats::default(),
        }
    }

    /// Where the wrapped model's calls have gone, cumulative.
    #[must_use]
    pub const fn stats(&self) -> &TransitionStats {
        &self.stats
    }
}

impl<T: Transition> Transition for Timed<T> {
    const HISTORY: usize = T::HISTORY;

    type State = T::State;

    fn start(&self, context: Option<&str>, asked: &ime_decode::Asked<'_>) -> T::State {
        let tick = Instant::now();
        let state = self.inner.start(context, asked);
        self.stats.start_calls.set(self.stats.start_calls.get() + 1);
        self.stats
            .start_ns
            .set(self.stats.start_ns.get() + nanos(tick));
        state
    }

    fn score(&self, state: &T::State, candidate: CharId) -> f32 {
        let tick = Instant::now();
        let score = self.inner.score(state, candidate);
        self.stats.score_calls.set(self.stats.score_calls.get() + 1);
        self.stats
            .score_ns
            .set(self.stats.score_ns.get() + nanos(tick));
        score
    }

    fn score_many(
        &self,
        state: &T::State,
        weight: f32,
        allowed: &[ime_pinyin::CharId],
        out: &mut [f32],
    ) {
        let tick = Instant::now();
        self.inner.score_many(state, weight, allowed, out);
        self.stats
            .score_calls
            .set(self.stats.score_calls.get() + u64::try_from(allowed.len()).unwrap_or(u64::MAX));
        self.stats
            .score_ns
            .set(self.stats.score_ns.get() + nanos(tick));
    }

    fn finish(&self, state: &T::State) -> f32 {
        self.inner.finish(state)
    }

    fn advance(&self, steps: &[(&T::State, CharId, ime_decode::Asked<'_>)]) -> Vec<T::State> {
        let tick = Instant::now();
        let states = self.inner.advance(steps);
        self.stats
            .advance_calls
            .set(self.stats.advance_calls.get() + 1);
        self.stats
            .advanced_states
            .set(self.stats.advanced_states.get() + steps.len() as u64);
        self.stats
            .advance_ns
            .set(self.stats.advance_ns.get() + nanos(tick));
        states
    }

    fn compact(&self, state: &T::State) -> T::State {
        self.inner.compact(state)
    }
}

/// The fused engine: the towers' emissions over *T*, the decode `fused-eval`
/// runs, as a session's [`Engine`].
pub struct Fused<'a, T> {
    /// The towers the export directory loaded into.
    towers: &'a RouteA,
    /// The transition the beam advances against.
    transition: T,
    /// The engine's knobs.
    config: FusedConfig<'a>,
    /// The last context the tower encoded, by its text.
    context_cache: RefCell<Option<(String, EncodedContext)>>,
    /// Where the emissions' time goes, cumulative.
    stats: EngineStats,
    /// Where the refreshes' time goes, cumulative.
    stages: FusedStages,
    /// How the keystrokes have changed the lattice, cumulative.
    invalidation: Invalidation,
    /// The previous refresh's per-path span boundaries, kept to measure how
    /// much of the lattice a keystroke changed.
    last_spans: Option<Vec<Vec<(usize, usize)>>>,
}

impl<'a, T> Fused<'a, T> {
    /// An engine over the loaded models. A missing model is an error at the
    /// loader that produced it, never here.
    #[must_use]
    pub fn new(towers: &'a RouteA, transition: T, config: FusedConfig<'a>) -> Self {
        Self {
            towers,
            transition,
            config,
            context_cache: RefCell::new(None),
            stats: EngineStats::default(),
            stages: FusedStages::default(),
            invalidation: Invalidation::default(),
            last_spans: None,
        }
    }

    /// The transition the engine decodes against — the way to its stats.
    #[must_use]
    pub const fn transition(&self) -> &T {
        &self.transition
    }

    /// Where the emissions have spent their time, cumulative.
    #[must_use]
    pub const fn stats(&self) -> &EngineStats {
        &self.stats
    }

    /// Where the refreshes have spent their time, cumulative.
    #[must_use]
    pub const fn stages(&self) -> &FusedStages {
        &self.stages
    }

    /// How the keystrokes have changed the lattice, cumulative.
    #[must_use]
    pub const fn invalidation(&self) -> &Invalidation {
        &self.invalidation
    }

    /// The pending input's readings, in k-best order; `Ok(None)` is a normal
    /// miss — a half-typed syllable under a strict segmentation — not an
    /// error.
    fn segment(&self, pending: &Pending) -> Result<Option<Vec<Segmentation>>, SegmentError> {
        // Each piece segments alone; a `'` is the user ending a syllable by
        // hand, so only the last piece may be a half-typed one.
        let count = pending.cuts().len() + 1;
        let mut paths: Vec<Segmentation> = Vec::new();
        for (index, piece) in pending.pieces().enumerate() {
            let options = SegmentOptions {
                allow_incomplete_tail: self.config.segment.allow_incomplete_tail
                    && index + 1 == count,
                ..self.config.segment.clone()
            };
            let lattice = match SegmentLattice::build(piece, self.config.table, &options) {
                Ok(lattice) => lattice,
                Err(SegmentError::NoSegmentation { .. }) => return Ok(None),
                Err(error) => return Err(error),
            };
            let kbest = lattice.k_best(&options);
            if kbest.is_empty() {
                return Ok(None);
            }
            if paths.is_empty() {
                paths = kbest;
                continue;
            }
            let mut merged = Vec::with_capacity(paths.len().saturating_mul(kbest.len()));
            for left in &paths {
                for right in &kbest {
                    merged.push(Segmentation::concat(&[left, right]));
                }
            }
            merged.sort_by(|a, b| a.cost().total_cmp(&b.cost()));
            merged.truncate(self.config.segment.max_paths);
            paths = merged;
        }
        Ok(Some(paths))
    }

    /// How much of the previous lattice survived this keystroke: the first
    /// position any ranked path's spans diverge, over the paths both hold.
    fn observe(&mut self, paths: &[Segmentation]) {
        let spans: Vec<Vec<(usize, usize)>> = paths
            .iter()
            .map(|path| {
                path.segments()
                    .iter()
                    .map(|segment| (segment.start(), segment.end()))
                    .collect()
            })
            .collect();
        if let Some(previous) = &self.last_spans {
            self.invalidation.compared += 1;
            let mut divergence = None;
            for (old, new) in previous.iter().zip(&spans) {
                let position = old
                    .iter()
                    .zip(new.iter())
                    .position(|(a, b)| a != b)
                    .unwrap_or_else(|| old.len().min(new.len()));
                if position < old.len().max(new.len())
                    && divergence.is_none_or(|first| position < first)
                {
                    divergence = Some(position);
                }
            }
            match divergence {
                Some(position) => {
                    *self
                        .invalidation
                        .first_divergence
                        .entry(position)
                        .or_insert(0) += 1;
                }
                None => self.invalidation.stable += 1,
            }
        }
        self.last_spans = Some(spans);
    }

    /// The context tower's output for *text*, from the cache when the session
    /// reuses it.
    fn encoded_context(&self, text: &str) -> Result<EncodedContext, NeuralError> {
        if self.config.incremental {
            let cache = self.context_cache.borrow();
            if let Some((cached, encoded)) = cache.as_ref()
                && cached == text
            {
                self.stats
                    .context_cached
                    .set(self.stats.context_cached.get() + 1);
                return Ok(encoded.clone());
            }
            drop(cache);
        }
        let tick = Instant::now();
        let encoded = self.towers.context(text)?;
        self.stats
            .context_ns
            .set(self.stats.context_ns.get() + nanos(tick));
        self.stats
            .context_calls
            .set(self.stats.context_calls.get() + 1);
        if self.config.incremental {
            *self.context_cache.borrow_mut() = Some((text.to_owned(), encoded.clone()));
        }
        Ok(encoded)
    }

    /// The towers' answer for *lattice*: the context tower once when the
    /// session has a context, the fill tower over every reading.
    fn scores(&self, lattice: &LatticeRecord) -> Result<Vec<Vec<Vec<f32>>>, NeuralError> {
        let context = lattice
            .context
            .as_deref()
            .filter(|_| self.config.with_context);
        let encoded = context.map(|text| self.encoded_context(text)).transpose()?;
        let tick = Instant::now();
        let scores = self.towers.emission_with(lattice, None, encoded.as_ref())?;
        self.stats
            .fill_ns
            .set(self.stats.fill_ns.get() + nanos(tick));
        self.stats.fill_calls.set(self.stats.fill_calls.get() + 1);
        let width = lattice
            .paths
            .iter()
            .fold(0usize, |w, path| w.max(path.spans.len() + 2));
        let tokens = u64::try_from(lattice.paths.len().saturating_mul(width)).unwrap_or(u64::MAX);
        self.stats
            .fill_tokens
            .set(self.stats.fill_tokens.get() + tokens);
        Ok(scores)
    }
}

impl<T: Transition> Engine for Fused<'_, T> {
    type Error = FusedError;

    /// Segment, score with the towers weighted as `fused-eval` fuses them,
    /// and decode; every hypothesis the beam keeps is a sentence.
    fn sentences(
        &mut self,
        pending: &Pending,
        context: Option<&str>,
    ) -> Result<Vec<Sentence>, FusedError> {
        let tick = Instant::now();
        let Some(paths) = self.segment(pending)? else {
            self.stages.undecodable += 1;
            self.last_spans = None;
            return Ok(Vec::new());
        };
        self.observe(&paths);
        let pinyin = pending.letters().as_str();
        let lexicon = self.config.lexicon;
        let candidates = Candidates::build(&paths, lexicon)?;
        let candidates = match self.config.words {
            Some(words) => candidates.with_arcs(
                &paths,
                pinyin,
                words,
                lexicon,
                self.config.table,
                &self.config.segment,
            ),
            None => candidates,
        };
        self.stages.segment_ns += nanos(tick);

        let tick = Instant::now();
        let lattice = LatticeRecord {
            record: 0,
            pinyin: pinyin.to_owned(),
            context: context.map(str::to_owned),
            paths: lattice_paths(pinyin, &paths, &candidates, self.config.emittable, lexicon),
        };
        let scores = self.scores(&lattice)?;
        let emission = Weighted {
            inner: Scored::attach(
                0,
                &candidates,
                self.config.emittable,
                scores,
                self.config.floor,
            )
            .map_err(NeuralError::from)?,
            weight: self.config.weight,
        };
        self.stages.emission_ns += nanos(tick);

        let tick = Instant::now();
        let mut answers = decode_many_stats(
            &[Record {
                candidates: &candidates,
                emission,
                context,
            }],
            &self.transition,
            &self.config.beam,
            &mut self.stages.breakdown,
        )?;
        self.stages.decode_ns += nanos(tick);
        let hypotheses = answers.pop().expect("one record decodes to one answer");
        Ok(hypotheses
            .iter()
            .map(|hypothesis| {
                let segments = paths[hypothesis.path()].segments();
                Sentence {
                    chars: hypothesis
                        .chars()
                        .iter()
                        .zip(segments)
                        .zip(hypothesis.steps())
                        .map(|((&id, segment), &score)| Placed {
                            ch: lexicon.character(id),
                            end: segment.end(),
                            score,
                        })
                        .collect(),
                    score: hypothesis.score(),
                }
            })
            .collect())
    }

    fn clear(&mut self) {
        self.last_spans = None;
    }
}
