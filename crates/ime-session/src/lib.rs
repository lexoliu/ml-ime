//! A typing session: keystrokes in, candidates out.
//!
//! The input method drives this API — the macOS one through a C ABI — and the
//! latency between [`Session::key`] and the updated [`Session::candidates`] is
//! the product's cost. Where scores come from is the [`Engine`] trait rather
//! than a concrete model, so the end-to-end network replaces today's engine —
//! [`Fused`]: route A's towers for the emissions, the kn-trigram and the
//! character LM fused for the transitions, the same decode `fused-eval` runs —
//! without the session noticing.
//!
//! A session is single-threaded by contract: it is called from the input
//! method's keystroke handler, one event at a time. The models it borrows are
//! the shared, thread-per-caller kind the engine carries.

mod cache;
mod fused;

use ime_decode::{
    BeamOptions, Breakdown, Candidates, DecodeError, Emission, Hypothesis, Record, Transition,
    decode_many_stats,
};
use ime_pinyin::{
    Lexicon, SegmentError, SegmentLattice, SegmentOptions, Segmentation, SyllableTable,
};
use std::collections::BTreeMap;
use std::time::Instant;
use thiserror::Error;

pub use cache::{CacheReport, Cached};
pub use fused::{EngineStats, Fused, FusedConfig, Timed, TransitionStats};

/// What produces the scores for one keystroke's decode.
///
/// The session asks two things of it: the scored lattice for the pending
/// keystrokes ([`Engine::emission`]) and the transition model the beam
/// advances against ([`Engine::transition`]). The fused engine is today's
/// answer; the end-to-end network implements the same trait directly on the
/// keystrokes.
pub trait Engine {
    /// The scored lattice one decode runs over.
    type Emission<'a>: Emission
    where
        Self: 'a;
    /// The transition model the beam advances against.
    type Trans: Transition;
    /// The engine's own failure type.
    type Error: std::error::Error + Send + Sync + 'static;

    /// The transition the decode runs against.
    fn transition(&self) -> &Self::Trans;

    /// Score *reading*'s candidates under its context.
    ///
    /// # Errors
    ///
    /// Engine-defined: a failed forward, a missing model.
    fn emission<'a>(&'a self, reading: &Reading<'a>) -> Result<Self::Emission<'a>, Self::Error>;
}

/// What one keystroke asks the engine to score.
///
/// `pinyin` is the pending input with `'` boundaries folded out — the letters
/// the segmentations span — and `segmentations` are its readings in the
/// lattice's k-best order, `candidates` their per-position masks. The engine
/// decides how to score them: the fused engine renders them as a
/// [`LatticeRecord`] for the towers.
///
/// [`LatticeRecord`]: ime_decode::LatticeRecord
pub struct Reading<'a> {
    /// The pending keystrokes, `'` boundaries removed.
    pub pinyin: &'a str,
    /// The text before the caret, when the session has one.
    pub context: Option<&'a str>,
    /// The readings of `pinyin`, in k-best order.
    pub segmentations: &'a [Segmentation],
    /// The candidate sets, aligned with `segmentations`.
    pub candidates: &'a Candidates,
}

/// The tables and options a session decodes under.
#[derive(Clone)]
pub struct SessionOptions<'a> {
    /// The character lexicon every table is indexed through.
    pub lexicon: &'a Lexicon,
    /// The syllable inventory the keystrokes segment against.
    pub table: &'a SyllableTable,
    /// How the keystrokes may segment. The live default admits a trailing
    /// half-typed syllable; an offline replay can turn it off to mirror an
    /// evaluation exactly.
    pub segment: SegmentOptions,
    /// The beam's width and arity — the product's shape is width 16, top 8.
    pub beam: BeamOptions,
    /// How many whole-input sentences lead the candidate list; the rest of
    /// the list is prefix candidates.
    pub sentences: usize,
}

/// The candidate list's cap — the panel's top 8.
const LIST: usize = 8;

/// One item of the candidate list.
#[derive(Clone, Debug)]
pub struct Candidate {
    /// The candidate's text.
    pub text: String,
    /// How many of the pending keystrokes — letters and `'`s — the candidate's
    /// reading consumes. A whole-input sentence consumes the pending input;
    /// a prefix candidate consumes the first of its syllables, and
    /// [`Session::select`] leaves the rest pending.
    pub consumes: usize,
    /// The score the beam gave it — a prefix candidate's is its hypothesis's
    /// score up to the last position it covers.
    pub score: f32,
}

/// What [`Session::select`] committed.
#[derive(Clone, Debug)]
pub struct Commit {
    /// The committed text.
    pub text: String,
    /// How many pending keystrokes it consumed.
    pub consumed: usize,
}

/// Where a session's refreshes spend their time, cumulative.
#[derive(Clone, Debug, Default)]
pub struct Stages {
    /// Refreshes run — keystrokes, context changes, selects.
    pub refreshes: u64,
    /// Refreshes that produced no reading: a half-typed syllable under a
    /// strict segmentation is a normal, cheap miss, not an error.
    pub undecodable: u64,
    /// Nanos spent segmenting and resolving the candidate masks.
    pub segment_ns: u64,
    /// Nanos inside the engine's emission.
    pub emission_ns: u64,
    /// Nanos inside the beam decode.
    pub decode_ns: u64,
    /// Nanos assembling the candidate list — the sentence hypotheses plus
    /// the walk over them that picks the prefix candidates.
    pub candidates_ns: u64,
    /// The beam's host-side breakdown — expansion, selection, and the
    /// finish pass, cumulative beside `decode_ns`.
    pub breakdown: Breakdown,
}

/// How a keystroke changed the readings, across a session's refreshes.
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

/// The pending input's readings, as [`Session::segment`] returns them.
struct Pieces {
    /// The letters the pieces join into — the `'`s folded out.
    pinyin: String,
    /// How many pending keystrokes the joined paths consume.
    consumed: usize,
    /// The readings of `pinyin`, in k-best order.
    paths: Vec<Segmentation>,
}

/// Why a session call failed.
#[derive(Debug, Error)]
pub enum SessionError<E: std::error::Error> {
    /// A keystroke outside the session's alphabet.
    #[error("{0:?} is not a letter or an apostrophe")]
    Key(char),
    /// The keystrokes could not be read under the options.
    #[error(transparent)]
    Segment(#[from] SegmentError),
    /// The candidate masks could not be built.
    #[error(transparent)]
    Decode(#[from] DecodeError),
    /// The engine failed its forward.
    #[error(transparent)]
    Engine(E),
    /// [`Session::select`] named a candidate the list does not hold.
    #[error("no candidate {0}")]
    Select(usize),
}

/// A typing session over an [`Engine`]: the pending keystrokes, the context
/// they were typed into, and the latest candidates.
///
/// `Session` owns no model — the engine and the tables in [`SessionOptions`]
/// are borrowed, so the loaded models are shared by value-free references and
/// a session is cheap to drop and remake.
pub struct Session<'a, E: Engine> {
    engine: E,
    options: SessionOptions<'a>,
    /// The text before the caret, which committed text joins.
    context: String,
    /// The pending keystrokes — letters and `'` boundaries — not yet consumed
    /// by a `select`.
    typed: String,
    candidates: Vec<Candidate>,
    stages: Stages,
    invalidation: Invalidation,
    /// The previous refresh's per-path span boundaries, kept to measure how
    /// much of the lattice a keystroke changed.
    last_spans: Option<Vec<Vec<(usize, usize)>>>,
}

impl<'a, E: Engine> Session<'a, E> {
    /// A session over *engine*, decoding under *options*.
    ///
    /// A missing model is an error at the engine's own construction —
    /// `Session::new` holds what is already loaded.
    #[must_use]
    pub fn new(engine: E, options: SessionOptions<'a>) -> Self {
        Self {
            engine,
            options,
            context: String::new(),
            typed: String::new(),
            candidates: Vec::new(),
            stages: Stages::default(),
            invalidation: Invalidation::default(),
            last_spans: None,
        }
    }

    /// The text before the caret, which committed text joins. Replacing it
    /// re-decodes the pending keystrokes against it.
    ///
    /// # Errors
    ///
    /// As [`Session::key`]'s.
    pub fn set_context(&mut self, text: &str) -> Result<(), SessionError<E::Error>> {
        text.clone_into(&mut self.context);
        self.refresh()
    }

    /// One letter, or a `'` forcing a syllable boundary: `xi'an` reads as the
    /// two pieces `xi` and `an`, each segmented alone.
    ///
    /// A keystroke that leaves the input unreadable — a half-typed syllable
    /// under a strict segmentation — is not an error: the candidates are just
    /// empty until the input reads again.
    ///
    /// # Errors
    ///
    /// If *ch* is not `a-z` or `'`, the lattice cannot be built, or the engine
    /// fails.
    pub fn key(&mut self, ch: char) -> Result<(), SessionError<E::Error>> {
        if !(ch.is_ascii_lowercase() || ch == '\'') {
            return Err(SessionError::Key(ch));
        }
        self.typed.push(ch);
        self.refresh()
    }

    /// Take back one keystroke.
    ///
    /// # Errors
    ///
    /// As [`Session::key`]'s.
    pub fn backspace(&mut self) -> Result<(), SessionError<E::Error>> {
        self.typed.pop();
        self.refresh()
    }

    /// The current top candidates, in the beam's order.
    #[must_use]
    pub fn candidates(&self) -> &[Candidate] {
        &self.candidates
    }

    /// The keystrokes pending consumption — the count `key` has taken minus
    /// what `select` committed, `'` boundaries included.
    #[must_use]
    pub fn pending(&self) -> usize {
        self.typed.len()
    }

    /// The engine the session decodes through — the way to its stats.
    #[must_use]
    pub const fn engine(&self) -> &E {
        &self.engine
    }

    /// Commit the *index*-th candidate: its text joins the context and the
    /// keystrokes it consumed leave the pending input, the rest stay.
    ///
    /// # Errors
    ///
    /// If the list has no *index*-th candidate, or the re-decode of the
    /// remaining keystrokes fails.
    pub fn select(&mut self, index: usize) -> Result<Commit, SessionError<E::Error>> {
        let candidate = self
            .candidates
            .get(index)
            .ok_or(SessionError::Select(index))?;
        let commit = Commit {
            text: candidate.text.clone(),
            consumed: candidate.consumes,
        };
        self.typed.replace_range(..commit.consumed, "");
        self.context.push_str(&commit.text);
        self.refresh()?;
        Ok(commit)
    }

    /// Drop the pending keystrokes, the context, and the candidates — a new
    /// input line. The accumulated stage counters survive for the harness.
    pub fn reset(&mut self) {
        self.context.clear();
        self.typed.clear();
        self.candidates.clear();
        self.last_spans = None;
    }

    /// Where this session's refreshes have spent their time, cumulative.
    #[must_use]
    pub const fn stages(&self) -> &Stages {
        &self.stages
    }

    /// How the keystrokes have changed the lattice, cumulative.
    #[must_use]
    pub const fn invalidation(&self) -> &Invalidation {
        &self.invalidation
    }

    /// The letter runs of `typed`, as byte ranges split at `'`.
    fn pieces(&self) -> Vec<(usize, usize)> {
        let mut pieces = Vec::new();
        let mut start = None;
        for (index, ch) in self.typed.char_indices() {
            if ch == '\'' {
                if let Some(began) = start.take() {
                    pieces.push((began, index));
                }
            } else if start.is_none() {
                start = Some(index);
            }
        }
        if let Some(began) = start {
            pieces.push((began, self.typed.len()));
        }
        pieces
    }

    /// The session's context as the engines see it: empty is none at all.
    fn context_text(&self) -> Option<&str> {
        (!self.context.is_empty()).then_some(self.context.as_str())
    }

    /// The pending input's readings: the `pinyin` the pieces join into, how
    /// many keystrokes the joined paths consume, and the paths themselves.
    /// `Ok(None)` is a normal miss — a half-typed syllable under a strict
    /// segmentation — not an error.
    fn segment(&self, pieces: &[(usize, usize)]) -> Result<Option<Pieces>, SessionError<E::Error>> {
        // Each piece segments alone; a `'` is the user ending a syllable by
        // hand, so only the last piece may be a half-typed one.
        let mut pinyin = String::with_capacity(self.typed.len());
        let mut paths: Vec<Segmentation> = Vec::new();
        let mut consumed_end = 0usize;
        for (index, (start, end)) in pieces.iter().enumerate() {
            let options = SegmentOptions {
                allow_incomplete_tail: self.options.segment.allow_incomplete_tail
                    && index + 1 == pieces.len(),
                ..self.options.segment.clone()
            };
            let lattice = match SegmentLattice::build(
                &self.typed[*start..*end],
                self.options.table,
                &options,
            ) {
                Ok(lattice) => lattice,
                Err(SegmentError::NoSegmentation { .. }) => return Ok(None),
                Err(error) => return Err(error.into()),
            };
            let kbest = lattice.k_best(&options);
            if kbest.is_empty() {
                return Ok(None);
            }
            pinyin.push_str(&self.typed[*start..*end]);
            consumed_end = *end;
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
            merged.truncate(self.options.segment.max_paths);
            paths = merged;
        }
        // Every path covers every piece, so the consumed keystrokes are the
        // last piece's end plus any `'`s right after it.
        let mut consumed = consumed_end;
        while self.typed.as_bytes().get(consumed) == Some(&b'\'') {
            consumed += 1;
        }
        Ok(Some(Pieces {
            pinyin,
            consumed,
            paths,
        }))
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

    /// Re-segment the pending input and re-decode it.
    ///
    /// Every refresh runs the whole pipeline — segmentation, the engine's
    /// emission, the beam — which is the baseline the incremental levers are
    /// measured against. What the refresh keeps between calls lives inside
    /// the engine (the context tower's output, the LM's resident pool), so
    /// the comparison is the engine's `incremental` switch, not two session
    /// implementations.
    fn refresh(&mut self) -> Result<(), SessionError<E::Error>> {
        self.stages.refreshes += 1;
        self.candidates.clear();
        let pieces = self.pieces();
        if pieces.is_empty() {
            self.last_spans = None;
            return Ok(());
        }
        let tick = Instant::now();
        let Some(pieces) = self.segment(&pieces)? else {
            self.stages.undecodable += 1;
            self.last_spans = None;
            return Ok(());
        };
        self.observe(&pieces.paths);
        let candidates = Candidates::build(&pieces.paths, self.options.lexicon)?;
        self.stages.segment_ns += nanos(tick);

        // The context is an owned copy: `emission`'s lifetime parameter
        // unifies the engine's borrow with `reading`'s, so a
        // `self.context_text()` borrow inside it would hold all of `self`
        // against the breakdown's `&mut` below.
        let context = self.context_text().map(str::to_owned);
        let reading = Reading {
            pinyin: &pieces.pinyin,
            context: context.as_deref(),
            segmentations: &pieces.paths,
            candidates: &candidates,
        };
        let tick = Instant::now();
        let emission = self
            .engine
            .emission(&reading)
            .map_err(SessionError::Engine)?;
        let emission_ns = nanos(tick);

        let record = Record {
            candidates: &candidates,
            emission,
            context: context.as_deref(),
        };
        let tick = Instant::now();
        let mut answers = decode_many_stats(
            &[record],
            self.engine.transition(),
            &self.options.beam,
            &mut self.stages.breakdown,
        )?;
        // The emission holds `self.engine` borrowed through the decode; the
        // counters fold in once the borrow ends.
        let decode_ns = nanos(tick);
        self.stages.emission_ns += emission_ns;
        self.stages.decode_ns += decode_ns;
        let hypotheses = answers.pop().expect("one record decodes to one answer");
        let tick = Instant::now();
        self.candidates = assemble(&hypotheses, &pieces, &self.typed, &self.options);
        self.stages.candidates_ns += nanos(tick);
        Ok(())
    }
}

/// The candidate list: the `sentences` best whole-input hypotheses, then for
/// each proper prefix of the top hypothesis's reading — longest first — the
/// best surviving text that has not already appeared as the start of a
/// listed sentence.
fn assemble(
    hypotheses: &[Hypothesis],
    pieces: &Pieces,
    typed: &str,
    options: &SessionOptions<'_>,
) -> Vec<Candidate> {
    let lexicon = options.lexicon;
    let mut list: Vec<Candidate> = hypotheses
        .iter()
        .take(options.sentences)
        .map(|hypothesis| Candidate {
            text: hypothesis.text(lexicon),
            consumes: pieces.consumed,
            score: hypothesis.score(),
        })
        .collect();
    let Some(best) = hypotheses.first() else {
        return list;
    };
    let reading = &pieces.paths[best.path()];
    for k in (1..reading.len()).rev() {
        // The best text covering the first `k` positions, among the finished
        // and surviving hypotheses, by its score up to position `k`.
        let mut chosen: Option<(String, f32)> = None;
        for hypothesis in hypotheses {
            let path = &pieces.paths[hypothesis.path()];
            if path.segments().len() < k || path.segments()[..k] != reading.segments()[..k] {
                continue;
            }
            let text: String = hypothesis.chars()[..k]
                .iter()
                .map(|id| lexicon.character(*id))
                .collect();
            // Not already in the list: a listed sentence's *full* text can
            // never collide with a shorter prefix, so this covers dedup
            // against the sentences and the prefixes listed so far.
            if list.iter().any(|candidate| candidate.text == text) {
                continue;
            }
            let score = hypothesis.steps()[k - 1];
            if chosen.as_ref().is_none_or(|(_, best)| score > *best) {
                chosen = Some((text, score));
            }
        }
        if let Some((text, score)) = chosen {
            let letters = reading.segments()[k - 1].end();
            list.push(Candidate {
                text,
                consumes: consumed_keystrokes(typed, letters),
                score,
            });
            if list.len() == LIST {
                break;
            }
        }
    }
    list
}

/// How many keystrokes hold *letters* letters: `'` boundaries inside them
/// are keystrokes too, and a boundary right after the last letter goes with
/// the syllable it closed.
fn consumed_keystrokes(typed: &str, letters: usize) -> usize {
    let mut seen = 0;
    let mut end = 0;
    for (index, byte) in typed.bytes().enumerate() {
        if byte != b'\'' {
            seen += 1;
            if seen == letters {
                end = index + 1;
                break;
            }
        }
    }
    while typed.as_bytes().get(end) == Some(&b'\'') {
        end += 1;
    }
    end
}

/// An [`Instant::elapsed`] in nanos, saturating the counter at a duration a
/// keystroke cannot reach.
fn nanos(tick: Instant) -> u64 {
    u64::try_from(tick.elapsed().as_nanos()).unwrap_or(u64::MAX)
}
