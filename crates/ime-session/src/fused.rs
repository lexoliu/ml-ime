//! The fused engine: route A's towers for the emissions, whatever transition
//! the run carries — the kn-trigram, the character LM, the two fused at their
//! weights, or none — the decode `fused-eval` runs, as a session's
//! [`Engine`].
//!
//! `incremental` is the session's reuse switch: the context tower's output
//! depends only on the context, so a keystroke that changes none of it never
//! re-encodes it. The fill tower's output is not cached: it sees the whole
//! reading, and earlier positions' scores do move when a syllable joins —
//! the issue's gate is that the incremental answer equals the full decode's,
//! so the fill graph re-runs on each keystroke as one batched forward.

use crate::{Engine, Reading, nanos};
use ime_decode::{Emittable, LatticeRecord, Scored, Transition, Weighted, lattice_paths};
use ime_neural::{EncodedContext, NeuralError, RouteA};
use ime_pinyin::{CharId, Lexicon};
use std::cell::{Cell, RefCell};
use std::time::Instant;

/// The towers' knobs a fused engine runs under, apart from the transition.
#[derive(Clone)]
pub struct FusedConfig<'a> {
    /// The emittable set the lattice's candidates are restricted to.
    pub emittable: &'a Emittable,
    /// The lexicon the emittable set and the candidates name characters
    /// through.
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
    type Emission<'b>
        = Weighted<Scored<'b>>
    where
        Self: 'b;
    type Trans = T;
    type Error = NeuralError;

    fn transition(&self) -> &T {
        &self.transition
    }

    /// The towers' table for *reading*, weighted as `fused-eval` fuses it.
    fn emission<'b>(&'b self, reading: &Reading<'b>) -> Result<Weighted<Scored<'b>>, NeuralError> {
        let lattice = LatticeRecord {
            record: 0,
            pinyin: reading.pinyin.to_owned(),
            context: reading.context.map(str::to_owned),
            paths: lattice_paths(
                reading.pinyin,
                reading.segmentations,
                reading.candidates,
                self.config.emittable,
                self.config.lexicon,
            ),
        };
        let scores = self.scores(&lattice)?;
        Ok(Weighted {
            inner: Scored::attach(
                0,
                reading.candidates,
                self.config.emittable,
                scores,
                self.config.floor,
            )?,
            weight: self.config.weight,
        })
    }
}
