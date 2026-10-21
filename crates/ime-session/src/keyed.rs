//! A typing session over the end-to-end network (#216): keys in, legal
//! sentences out, through the constrained search of
//! [`ime_decode::keyed`].
//!
//! Every change to the pending keys -- a key, a backspace, a commit, a new
//! context -- reruns the network's forward and the search over all pending
//! keys. Nothing of the previous search carries over, and that is the
//! network's shape, not a missed cache: the fill tower reads the keys
//! bidirectionally, so one more key changes every key's emission row, and the
//! decoder that scores the transition cross-attends to those rows too. The
//! legality lattice is rebuilt as well, under the live options, because the
//! last span's unfinished-syllable rule moves with the last key.

use ime_decode::keyed::{KeyEmission, KeyTransition, Sentence, Shape, Weights, WordArcs, search};
use ime_pinyin::{EmitId, Keys, Legality, LegalityOptions};
use thiserror::Error;

/// What scores one search: the network's forward over the pending keys.
///
/// One call returns both halves of the path score, because one forward
/// produces both: the CTC head's rows and the decoder bound to the same key
/// vectors and context.
pub trait KeyEngine {
    /// The per-key emission the search reads.
    type Emission<'a>: KeyEmission
    where
        Self: 'a;
    /// The transition the search advances, bound to the keys and context.
    type Transition<'a>: KeyTransition
    where
        Self: 'a;
    /// The engine's own failure type.
    type Error: std::error::Error + Send + Sync + 'static;

    /// The forward over *keys*, typed after *context*.
    ///
    /// # Errors
    ///
    /// Engine-defined: a failed forward, a missing model.
    fn forward<'a>(
        &'a self,
        keys: &Keys,
        context: &str,
    ) -> Result<(Self::Emission<'a>, Self::Transition<'a>), Self::Error>;
}

/// The tables and knobs a key session searches under.
#[derive(Clone, Copy)]
pub struct KeySessionOptions<'a> {
    /// The legality rules over the network's emittable set.
    pub legality: &'a Legality,
    /// The curated words offered as arcs, when there are any.
    pub words: Option<&'a WordArcs>,
    /// The beam: the product's is eight wide, top 8.
    pub shape: Shape,
    /// `β`, `γ` and `λ`, chosen on dev.
    pub weights: Weights,
    /// How many whole-input sentences lead the candidate list; the rest of
    /// the list is prefix candidates, up to the shape's `top_k`.
    pub sentences: usize,
}

/// One item of the candidate list.
#[derive(Clone, Debug, PartialEq)]
pub struct KeyCandidate {
    /// The candidate's text.
    pub text: String,
    /// How many pending keys committing it consumes: all of them for a
    /// whole sentence, the keys its characters' spans cover for a prefix.
    pub consumes: usize,
    /// The search's score for it; a prefix's is its sentence's score when
    /// its last character was emitted.
    pub score: f32,
}

/// What [`KeySession::commit`] committed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeyCommit {
    /// The committed text.
    pub text: String,
    /// How many pending keys it consumed.
    pub consumed: usize,
}

/// Why a key session call failed.
#[derive(Debug, Error)]
pub enum KeySessionError<E: std::error::Error> {
    /// A key outside `a`..`z`: the network reads letters only.
    #[error("{0:?} is not a key the network reads")]
    Key(char),
    /// The engine failed its forward.
    #[error(transparent)]
    Engine(E),
    /// [`KeySession::commit`] named a candidate the list does not hold.
    #[error("no candidate {0}")]
    Commit(usize),
}

/// A typing session over a [`KeyEngine`]: the pending keys, the text they
/// are typed after, and the latest candidates.
pub struct KeySession<'a, E: KeyEngine> {
    engine: E,
    options: KeySessionOptions<'a>,
    /// The text before the caret, which committed text joins.
    context: String,
    /// The pending keys, all in `a`..`z`.
    typed: String,
    candidates: Vec<KeyCandidate>,
}

impl<'a, E: KeyEngine> KeySession<'a, E> {
    /// A session over *engine* under *options*.
    #[must_use]
    pub const fn new(engine: E, options: KeySessionOptions<'a>) -> Self {
        Self {
            engine,
            options,
            context: String::new(),
            typed: String::new(),
            candidates: Vec::new(),
        }
    }

    /// Replace the text before the caret and search the pending keys again.
    ///
    /// # Errors
    ///
    /// If the engine fails.
    pub fn set_context(&mut self, text: &str) -> Result<(), KeySessionError<E::Error>> {
        text.clone_into(&mut self.context);
        self.refresh()
    }

    /// Append one key and search again.
    ///
    /// Keys that no legal sentence reads are not an error: the candidates
    /// are empty until the keys read again.
    ///
    /// # Errors
    ///
    /// If *key* is not in `a`..`z`, or the engine fails.
    pub fn key(&mut self, key: char) -> Result<(), KeySessionError<E::Error>> {
        if !key.is_ascii_lowercase() {
            return Err(KeySessionError::Key(key));
        }
        self.typed.push(key);
        self.refresh()
    }

    /// Take back the last pending key and search again.
    ///
    /// # Errors
    ///
    /// If the engine fails.
    pub fn backspace(&mut self) -> Result<(), KeySessionError<E::Error>> {
        self.typed.pop();
        self.refresh()
    }

    /// Commit the *index*-th candidate: its text joins the context, the keys
    /// it consumes leave the pending keys, and the rest are searched again.
    ///
    /// # Errors
    ///
    /// If the list has no *index*-th candidate, or the engine fails.
    pub fn commit(&mut self, index: usize) -> Result<KeyCommit, KeySessionError<E::Error>> {
        let candidate = self
            .candidates
            .get(index)
            .ok_or(KeySessionError::Commit(index))?;
        let commit = KeyCommit {
            text: candidate.text.clone(),
            consumed: candidate.consumes,
        };
        self.typed.replace_range(..commit.consumed, "");
        self.context.push_str(&commit.text);
        self.refresh()?;
        Ok(commit)
    }

    /// Drop the pending keys, the context and the candidates: a new line.
    pub fn reset(&mut self) {
        self.context.clear();
        self.typed.clear();
        self.candidates.clear();
    }

    /// The current candidates: the best sentences, then prefix candidates.
    #[must_use]
    pub fn candidates(&self) -> &[KeyCandidate] {
        &self.candidates
    }

    /// The pending keys.
    #[must_use]
    pub fn pending(&self) -> &str {
        &self.typed
    }

    /// The text before the caret.
    #[must_use]
    pub fn context(&self) -> &str {
        &self.context
    }

    /// The engine the session searches through.
    #[must_use]
    pub const fn engine(&self) -> &E {
        &self.engine
    }

    /// Search the pending keys under the live options.
    fn refresh(&mut self) -> Result<(), KeySessionError<E::Error>> {
        self.candidates.clear();
        let Ok(keys) = Keys::new(&self.typed) else {
            // `key` admits only `a`..`z`, so the one way to fail is no keys.
            return Ok(());
        };
        let options = self.options;
        let lattice = options.legality.lattice(&keys, LegalityOptions::LIVE);
        if lattice.is_empty() {
            return Ok(());
        }
        let (emission, transition) = self
            .engine
            .forward(&keys, &self.context)
            .map_err(KeySessionError::Engine)?;
        let found = search(
            &lattice,
            &emission,
            &transition,
            options.words,
            options.shape,
            options.weights,
        );
        self.candidates = assemble(&found.sentences, &options, keys.len());
        Ok(())
    }
}

/// The candidate list: the `sentences` best whole sentences, then for each
/// proper prefix of the best sentence's best alignment -- longest first --
/// the best text of that many characters over the same keys among the
/// returned sentences, unless it is already listed; at most `top_k` in all.
fn assemble(
    sentences: &[Sentence],
    options: &KeySessionOptions<'_>,
    keys: usize,
) -> Vec<KeyCandidate> {
    let emittable = options.legality.emittable();
    let text =
        |chars: &[EmitId]| -> String { chars.iter().map(|id| emittable[id.index()]).collect() };
    let cap = options.shape.top_k();
    let mut list: Vec<KeyCandidate> = sentences
        .iter()
        .take(options.sentences.min(cap))
        .map(|sentence| KeyCandidate {
            text: text(sentence.chars()),
            consumes: keys,
            score: sentence.score(),
        })
        .collect();
    let Some(best) = sentences.first() else {
        return list;
    };
    for length in (1..best.chars().len()).rev() {
        if list.len() >= cap {
            break;
        }
        let cut = best.ends()[length - 1];
        let mut chosen: Option<(String, f32)> = None;
        for sentence in sentences {
            if sentence.chars().len() <= length || sentence.ends()[length - 1] != cut {
                continue;
            }
            let prefix = text(&sentence.chars()[..length]);
            if list.iter().any(|candidate| candidate.text == prefix) {
                continue;
            }
            let score = sentence.prefix_scores()[length - 1];
            if chosen.as_ref().is_none_or(|(_, held)| score > *held) {
                chosen = Some((prefix, score));
            }
        }
        if let Some((text, score)) = chosen {
            list.push(KeyCandidate {
                text,
                consumes: cut,
                score,
            });
        }
    }
    list
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use ime_decode::keyed::EmissionRows;
    use ime_decode::mock::{self, Favoured};

    use super::*;

    /// An empty candidate list.
    const NONE: &[KeyCandidate] = &[];

    /// Flat-ish rows and a transition that favours the given sentences after
    /// the context's last character.
    struct Mock<'l> {
        legality: &'l Legality,
        favoured: Vec<Vec<EmitId>>,
    }

    impl<'l> Mock<'l> {
        fn new(legality: &'l Legality, favoured: &[&str]) -> Self {
            Self {
                legality,
                favoured: favoured
                    .iter()
                    .map(|sentence| {
                        sentence
                            .chars()
                            .map(|ch| legality.emit_id(ch).expect("emittable"))
                            .collect()
                    })
                    .collect(),
            }
        }
    }

    impl KeyEngine for Mock<'_> {
        type Emission<'a>
            = EmissionRows
        where
            Self: 'a;
        type Transition<'a>
            = Favoured
        where
            Self: 'a;
        type Error = Infallible;

        fn forward(
            &self,
            keys: &Keys,
            context: &str,
        ) -> Result<(EmissionRows, Favoured), Infallible> {
            let rows = mock::hashed_rows(keys.len(), self.legality.emittable().len(), 1, 0.5);
            let after = context
                .chars()
                .last()
                .and_then(|ch| self.legality.emit_id(ch));
            let favoured: Vec<&[EmitId]> = self.favoured.iter().map(Vec::as_slice).collect();
            Ok((rows, Favoured::new(after, &favoured, -6.0)))
        }
    }

    fn options(legality: &Legality) -> KeySessionOptions<'_> {
        KeySessionOptions {
            legality,
            words: None,
            shape: Shape::PRODUCT,
            weights: Weights::UNIT,
            sentences: 3,
        }
    }

    fn typed<E: KeyEngine>(session: &mut KeySession<'_, E>, keys: &str) {
        for key in keys.chars() {
            session.key(key).expect("the mock never fails");
        }
    }

    #[test]
    fn keys_backspace_and_commit_after_the_context() {
        let legality = Legality::over_lexicon().expect("tables");
        let mut session = KeySession::new(Mock::new(&legality, &["水饺"]), options(&legality));
        session.set_context("我们吃").expect("no keys yet");
        assert_eq!(session.candidates(), NONE);
        typed(&mut session, "shuijiap");
        assert_eq!(session.pending(), "shuijiap");
        let best = &session.candidates()[0];
        assert_eq!((best.text.as_str(), best.consumes), ("水饺", 8));

        session.backspace().expect("searches");
        assert_eq!(session.pending(), "shuijia");
        assert_eq!(
            session.candidates()[0].text,
            "水饺",
            "the live tail reads `jia` as jiao unfinished"
        );

        session.key('p').expect("searches");
        let commit = session.commit(0).expect("a candidate");
        assert_eq!(
            commit,
            KeyCommit {
                text: "水饺".to_owned(),
                consumed: 8
            }
        );
        assert_eq!(session.context(), "我们吃水饺");
        assert_eq!(session.pending(), "");
        assert_eq!(session.candidates(), NONE);
    }

    #[test]
    fn a_prefix_commit_leaves_the_rest_pending() {
        let legality = Legality::over_lexicon().expect("tables");
        let mut session = KeySession::new(Mock::new(&legality, &["水饺"]), options(&legality));
        session.set_context("我们吃").expect("no keys yet");
        typed(&mut session, "shuijiao");
        let index = session
            .candidates()
            .iter()
            .position(|candidate| candidate.text == "水")
            .expect("水 is listed as a prefix of 水饺");
        assert_eq!(session.candidates()[index].consumes, 4);
        let commit = session.commit(index).expect("a candidate");
        assert_eq!(commit.consumed, 4);
        assert_eq!(session.pending(), "jiao");
        assert_eq!(session.context(), "我们吃水");
        assert_ne!(session.candidates(), NONE);
    }

    #[test]
    fn wmzyq_never_lists_the_dumplings() {
        let legality = Legality::over_lexicon().expect("tables");
        let mut session =
            KeySession::new(Mock::new(&legality, &["我们吃水饺"]), options(&legality));
        typed(&mut session, "wmzyq");
        assert_ne!(session.candidates(), NONE);
        assert!(
            session
                .candidates()
                .iter()
                .all(|candidate| candidate.text != "我们吃水饺")
        );
    }

    #[test]
    fn only_letters_are_keys() {
        let legality = Legality::over_lexicon().expect("tables");
        let mut session = KeySession::new(Mock::new(&legality, &[]), options(&legality));
        for key in ['\'', 'A', '1'] {
            assert!(matches!(session.key(key), Err(KeySessionError::Key(k)) if k == key));
        }
        assert_eq!(session.pending(), "");
        assert!(matches!(session.commit(0), Err(KeySessionError::Commit(0))));
    }
}
