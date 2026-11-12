//! A typing session: keystrokes in, candidates out.
//!
//! The input method drives this API — the macOS one through a C ABI — and the
//! latency between [`Session::key`] and the updated [`Session::candidates`] is
//! the product's cost. What reads the keystrokes is an [`Engine`]: it takes
//! the pending input and the context and returns ranked sentences, each with
//! the letters consumed through each of its characters, and the session
//! assembles the candidate list from them — the whole-input sentences, then
//! the prefix candidates that end where a sentence's characters end. Two
//! engines implement it: [`Fused`], route A's towers for the emissions and
//! the kn-trigram and the character LM fused for the transitions, the same
//! decode `fused-eval` runs; and [`KeyEngine`], the key-level model through
//! the key search.
//!
//! A session is single-threaded by contract: it is called from the input
//! method's keystroke handler, one event at a time. The models it borrows are
//! the shared, thread-per-caller kind the engine carries.

mod cache;
mod fused;
mod keys;

use ime_pinyin::{Keys, KeysError};
use std::time::Instant;
use thiserror::Error;

pub use cache::{CacheReport, Cached};
pub use fused::{
    EngineStats, Fused, FusedConfig, FusedError, FusedStages, Invalidation, Timed, TransitionStats,
};
pub use keys::{KeyEngine, KeyEngineError, KeyEngineOptions, KeystrokeStats};

/// What reads the pending keystrokes into sentences.
///
/// The session calls [`Engine::sentences`] on every refresh — a keystroke, a
/// context change, a select — with the whole pending input, so an engine
/// needs no notion of an edit: what it keeps between calls is its own
/// business, and [`Engine::clear`] says when nothing typed so far relates to
/// what comes next.
pub trait Engine {
    /// The engine's own failure type.
    type Error: std::error::Error + Send + Sync + 'static;

    /// The sentences *pending* reads as under *context*, best first.
    ///
    /// No sentence is a normal answer — a half-typed syllable the engine
    /// cannot read yet — not an error.
    ///
    /// # Errors
    ///
    /// Engine-defined: a failed forward, a malformed table.
    fn sentences(
        &mut self,
        pending: &Pending,
        context: Option<&str>,
    ) -> Result<Vec<Sentence>, Self::Error>;

    /// The pending input emptied or a new line began: drop whatever the
    /// engine compares the next input against.
    fn clear(&mut self);
}

/// The pending keystrokes as an engine reads them: the letters with every
/// `'` folded out, and where the `'`s cut them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pending {
    letters: Keys,
    /// Letter offsets strictly inside the letters where a `'` was typed,
    /// ascending, each once.
    cuts: Vec<usize>,
    /// Whether a `'` follows the last letter: the user closed the last
    /// syllable by hand.
    closed: bool,
}

impl Pending {
    /// The pending input *typed*, letters and `'` boundaries; `None` when it
    /// holds no letter.
    ///
    /// # Errors
    ///
    /// If *typed* holds anything but `a-z` and `'`.
    pub fn parse(typed: &str) -> Result<Option<Self>, KeysError> {
        let letters: String = typed.chars().filter(|&ch| ch != '\'').collect();
        let letters = match Keys::new(&letters) {
            Ok(letters) => letters,
            Err(KeysError::Empty) => return Ok(None),
            Err(error) => return Err(error),
        };
        let mut cuts: Vec<usize> = Vec::new();
        let (mut seen, mut closed) = (0, false);
        for ch in typed.chars() {
            if ch == '\'' {
                closed = seen > 0;
                continue;
            }
            if closed && cuts.last() != Some(&seen) {
                cuts.push(seen);
            }
            closed = false;
            seen += 1;
        }
        Ok(Some(Self {
            letters,
            cuts,
            closed,
        }))
    }

    /// The letters, `'`s folded out.
    #[must_use]
    pub const fn letters(&self) -> &Keys {
        &self.letters
    }

    /// The letter offsets strictly inside [`Pending::letters`] where a `'`
    /// was typed: a syllable boundary the user forced.
    #[must_use]
    pub fn cuts(&self) -> &[usize] {
        &self.cuts
    }

    /// Whether a `'` follows the last letter, closing the last syllable.
    #[must_use]
    pub const fn closed(&self) -> bool {
        self.closed
    }

    /// The letter runs between the cuts, in order.
    pub fn pieces(&self) -> impl Iterator<Item = &str> {
        let letters = self.letters.as_str();
        let starts = std::iter::once(0).chain(self.cuts.iter().copied());
        let ends = self
            .cuts
            .iter()
            .copied()
            .chain(std::iter::once(letters.len()));
        starts
            .zip(ends)
            .map(move |(start, end)| &letters[start..end])
    }
}

/// One character of a [`Sentence`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placed {
    /// The character.
    pub ch: char,
    /// How many of the pending letters the sentence consumes through this
    /// character, `'`s not counted.
    pub end: usize,
    /// The sentence's running score through this character, comparable
    /// between sentences whose characters so far end at the same letters.
    pub score: f32,
}

/// One reading of the pending input, as an engine ranks it.
#[derive(Clone, Debug, PartialEq)]
pub struct Sentence {
    /// The characters, left to right; the last one's `end` is every letter.
    pub chars: Vec<Placed>,
    /// The whole sentence's score. Larger is better.
    pub score: f32,
}

impl Sentence {
    /// The sentence's first *count* characters as text.
    fn text(&self, count: usize) -> String {
        self.chars[..count].iter().map(|placed| placed.ch).collect()
    }
}

/// What a session lists.
#[derive(Clone, Debug)]
pub struct SessionOptions {
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
    /// a prefix candidate consumes the letters through its last character,
    /// and [`Session::select`] leaves the rest pending.
    pub consumes: usize,
    /// The score the engine gave it — a prefix candidate's is its sentence's
    /// running score through its last character.
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

/// Where a session's refreshes spend their time, cumulative. What happens
/// inside the engine is the engine's own account.
#[derive(Clone, Debug, Default)]
pub struct Stages {
    /// Refreshes run — keystrokes, context changes, selects.
    pub refreshes: u64,
    /// Nanos inside the engine.
    pub engine_ns: u64,
    /// Nanos assembling the candidate list from the engine's sentences.
    pub candidates_ns: u64,
}

/// Why a session call failed.
#[derive(Debug, Error)]
pub enum SessionError<E: std::error::Error> {
    /// A keystroke outside the session's alphabet.
    #[error("{0:?} is not a letter or an apostrophe")]
    Key(char),
    /// The pending input is not letters and apostrophes.
    #[error(transparent)]
    Pending(#[from] KeysError),
    /// The engine failed.
    #[error(transparent)]
    Engine(E),
    /// [`Session::select`] named a candidate the list does not hold.
    #[error("no candidate {0}")]
    Select(usize),
}

/// A typing session over an [`Engine`]: the pending keystrokes, the context
/// they were typed into, and the latest candidates.
///
/// `Session` owns no model — whatever the engine loaded is the engine's — so
/// a session is cheap to drop and remake.
pub struct Session<E: Engine> {
    engine: E,
    options: SessionOptions,
    /// The text before the caret, which committed text joins.
    context: String,
    /// The pending keystrokes — letters and `'` boundaries — not yet consumed
    /// by a `select`.
    typed: String,
    /// The engine's latest answer.
    sentences: Vec<Sentence>,
    candidates: Vec<Candidate>,
    stages: Stages,
}

impl<E: Engine> Session<E> {
    /// A session over *engine*, listing under *options*.
    ///
    /// A missing model is an error at the engine's own construction —
    /// `Session::new` holds what is already loaded.
    #[must_use]
    pub const fn new(engine: E, options: SessionOptions) -> Self {
        Self {
            engine,
            options,
            context: String::new(),
            typed: String::new(),
            sentences: Vec::new(),
            candidates: Vec::new(),
            stages: Stages {
                refreshes: 0,
                engine_ns: 0,
                candidates_ns: 0,
            },
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
    /// two pieces `xi` and `an`.
    ///
    /// A keystroke that leaves the input unreadable — a half-typed syllable
    /// the engine cannot read yet — is not an error: the candidates are just
    /// empty until the input reads again.
    ///
    /// # Errors
    ///
    /// If *ch* is not `a-z` or `'`, or the engine fails.
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

    /// The current top candidates: the whole-input sentences, then the
    /// prefix candidates.
    #[must_use]
    pub fn candidates(&self) -> &[Candidate] {
        &self.candidates
    }

    /// The engine's latest sentences, best first, the candidates were
    /// assembled from.
    #[must_use]
    pub fn sentences(&self) -> &[Sentence] {
        &self.sentences
    }

    /// The keystrokes pending consumption — the count `key` has taken minus
    /// what `select` committed, `'` boundaries included.
    #[must_use]
    pub const fn pending(&self) -> usize {
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
        self.sentences.clear();
        self.candidates.clear();
        self.engine.clear();
    }

    /// Where this session's refreshes have spent their time, cumulative.
    #[must_use]
    pub const fn stages(&self) -> &Stages {
        &self.stages
    }

    /// Ask the engine for the pending input's sentences and list them.
    fn refresh(&mut self) -> Result<(), SessionError<E::Error>> {
        self.stages.refreshes += 1;
        self.sentences.clear();
        self.candidates.clear();
        let Some(pending) = Pending::parse(&self.typed)? else {
            self.engine.clear();
            return Ok(());
        };
        let context = (!self.context.is_empty()).then_some(self.context.as_str());
        let tick = Instant::now();
        self.sentences = self
            .engine
            .sentences(&pending, context)
            .map_err(SessionError::Engine)?;
        self.stages.engine_ns += nanos(tick);
        let tick = Instant::now();
        self.candidates = assemble(&self.sentences, &self.typed, self.options.sentences);
        self.stages.candidates_ns += nanos(tick);
        Ok(())
    }
}

/// The candidate list: the `sentences` best whole-input sentences, then for
/// each proper prefix of the top sentence's reading — longest first — the
/// best text among the sentences whose characters end at the same letters,
/// by their running score there, that is not already listed.
fn assemble(sentences: &[Sentence], typed: &str, leading: usize) -> Vec<Candidate> {
    let mut list: Vec<Candidate> = sentences
        .iter()
        .take(leading)
        .map(|sentence| Candidate {
            text: sentence.text(sentence.chars.len()),
            consumes: typed.len(),
            score: sentence.score,
        })
        .collect();
    let Some(best) = sentences.first() else {
        return list;
    };
    let same_ends = |sentence: &Sentence, k: usize| {
        sentence.chars.len() >= k
            && sentence.chars[..k]
                .iter()
                .zip(&best.chars[..k])
                .all(|(a, b)| a.end == b.end)
    };
    for k in (1..best.chars.len()).rev() {
        let mut chosen: Option<(String, f32)> = None;
        for sentence in sentences.iter().filter(|sentence| same_ends(sentence, k)) {
            let text = sentence.text(k);
            // Not already in the list: a listed sentence's *full* text can
            // never collide with a shorter prefix, so this covers dedup
            // against the sentences and the prefixes listed so far.
            if list.iter().any(|candidate| candidate.text == text) {
                continue;
            }
            let score = sentence.chars[k - 1].score;
            if chosen.as_ref().is_none_or(|(_, top)| score > *top) {
                chosen = Some((text, score));
            }
        }
        if let Some((text, score)) = chosen {
            list.push(Candidate {
                text,
                consumes: consumed_keystrokes(typed, best.chars[k - 1].end),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pending_folds_apostrophes_into_cuts() {
        let pending = Pending::parse("'xi''an'").expect("valid").expect("letters");
        assert_eq!(pending.letters().as_str(), "xian");
        assert_eq!(pending.cuts(), [2]);
        assert!(pending.closed());
        assert_eq!(pending.pieces().collect::<Vec<_>>(), ["xi", "an"]);
        let open = Pending::parse("woai").expect("valid").expect("letters");
        assert!(open.cuts().is_empty() && !open.closed());
        assert_eq!(open.pieces().collect::<Vec<_>>(), ["woai"]);
        assert_eq!(Pending::parse("''"), Ok(None));
        assert!(Pending::parse("wo1").is_err());
    }
}
