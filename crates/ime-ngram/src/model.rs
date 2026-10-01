//! The trained model: three interpolated levels and the lookups over them.

use std::path::Path;

use crate::NgramError;
use crate::mapped;
use crate::table::ProbTable;
use ime_decode::{Asked, Transition};
use ime_pinyin::{CharId, Lexicon};
use serde::{Deserialize, Serialize};

/// How many preceding characters the model conditions on.
pub const ORDER: usize = 3;

/// A position in the model's token space: the lexicon's characters, plus the two
/// boundary markers that let a line start and end somewhere.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Token(u32);

impl Token {
    /// Stands in for every position before the start of a line.
    pub const BOS: Self = Self(0);
    /// Ends a line. Unlike [`Token::BOS`] it is a prediction target, so a model
    /// that never emits it would happily run a sentence on forever.
    pub const EOS: Self = Self(1);
    /// How many tokens are not characters.
    pub const RESERVED: u32 = 2;

    /// The token standing for *id*.
    #[must_use]
    pub fn of(id: CharId) -> Self {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a Lexicon's length fits u32 by construction"
        )]
        Self(id.index() as u32 + Self::RESERVED)
    }

    /// This token's index into the model's dense per-token arrays.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// What a history resolves to before any candidate is scored against it.
///
/// The two tokens the history names and the backoff weights they select. Both
/// lookups depend on the history alone, so the decoder pays them once per beam
/// state rather than once per candidate.
#[derive(Copy, Clone, Debug)]
pub struct Context {
    before: Token,
    previous: Token,
    bigram_backoff: f32,
    trigram_backoff: Option<f32>,
}

/// A character-level interpolated Kneser-Ney trigram over hanzi.
///
/// Three levels, each backing off into the next: raw trigram counts, then
/// continuation counts over bigrams, then continuation counts over characters
/// interpolated with a uniform floor. Kneser-Ney's point is the middle two --
/// what matters about a lower-order context is how many *distinct* things it
/// followed, not how often it occurred, which is why 国 is a likely character but
/// an unlikely one to see after an arbitrary predecessor.
///
/// The tables are precomputed at training time, so a lookup is three binary
/// searches and two multiply-adds. Nothing is normalised or discounted at query
/// time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NgramModel {
    /// The lexicon this was trained against, in lexicon order. Kept so that a
    /// model file can be checked against the lexicon it is loaded with rather
    /// than silently scoring the wrong characters.
    vocabulary: Box<[char]>,
    /// `P_KN(token)`, indexed by token. Zero at [`Token::BOS`], which is never a
    /// target.
    unigram: Box<[f32]>,
    /// The bigram level's backoff weight, indexed by the preceding token. One
    /// where the context was never seen, which passes the unigram level through.
    bigram_backoff: Box<[f32]>,
    /// The bigram level's discounted term, keyed by `(previous, current)`.
    bigram: ProbTable,
    /// The trigram level's backoff weight, keyed by the two preceding tokens.
    /// Absent where the context was never seen, which passes the bigram level
    /// through.
    trigram_backoff: ProbTable,
    /// The trigram level's discounted term, keyed by all three tokens.
    trigram: ProbTable,
}

impl NgramModel {
    /// Assemble a model from its precomputed levels.
    pub(crate) fn new(
        vocabulary: Box<[char]>,
        unigram: Box<[f32]>,
        bigram_backoff: Box<[f32]>,
        bigram: ProbTable,
        trigram_backoff: ProbTable,
        trigram: ProbTable,
    ) -> Self {
        Self {
            vocabulary,
            unigram,
            bigram_backoff,
            bigram,
            trigram_backoff,
            trigram,
        }
    }

    /// How many characters the model knows.
    #[must_use]
    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary.len()
    }

    /// How many trigrams the corpus contained, as distinct types.
    #[must_use]
    pub fn trigram_types(&self) -> usize {
        self.trigram.len()
    }

    /// The lexicon's characters in vocabulary order.
    pub(crate) fn vocabulary(&self) -> &[char] {
        &self.vocabulary
    }

    /// The unigram level's probabilities, indexed by token.
    pub(crate) fn unigram(&self) -> &[f32] {
        &self.unigram
    }

    /// The bigram level's backoff weights, indexed by the preceding token.
    pub(crate) fn bigram_backoff(&self) -> &[f32] {
        &self.bigram_backoff
    }

    /// The keyed tables: bigram, trigram backoff and trigram, in the order
    /// the mapped layout stores them.
    pub(crate) fn tables(&self) -> [&ProbTable; 3] {
        [&self.bigram, &self.trigram_backoff, &self.trigram]
    }

    /// How many bigrams the corpus contained, as distinct types.
    #[must_use]
    pub fn bigram_types(&self) -> usize {
        self.bigram.len()
    }

    /// The base the token keys are packed in.
    fn base(&self) -> u64 {
        self.unigram.len() as u64
    }

    /// Pack two tokens into a key.
    fn pack2(&self, first: Token, second: Token) -> u64 {
        u64::from(first.0) * self.base() + u64::from(second.0)
    }

    /// Pack three tokens into a key.
    fn pack3(&self, first: Token, second: Token, third: Token) -> u64 {
        self.pack2(first, second) * self.base() + u64::from(third.0)
    }

    /// The lookups a history needs regardless of the candidate: the bigram
    /// level's backoff weight off the last token, and the trigram level's off
    /// the pair.
    fn context_at(&self, before: Token, previous: Token) -> Context {
        Context {
            before,
            previous,
            bigram_backoff: self.bigram_backoff[previous.index()],
            trigram_backoff: self.trigram_backoff.get(self.pack2(before, previous)),
        }
    }

    /// `P(current | context)` under the interpolated model.
    fn probability_at(&self, context: &Context, current: Token) -> f32 {
        let level1 = self.unigram[current.index()];
        let level2 = self
            .bigram
            .get(self.pack2(context.previous, current))
            .unwrap_or(0.0)
            + context.bigram_backoff * level1;
        match context.trigram_backoff {
            Some(backoff) => {
                self.trigram
                    .get(self.pack3(context.before, context.previous, current))
                    .unwrap_or(0.0)
                    + backoff * level2
            }
            None => level2,
        }
    }

    /// `P(current | previous, before)` under the interpolated model.
    ///
    /// Never zero: the unigram level interpolates with a uniform floor and every
    /// discount is strictly positive, so no token is unreachable.
    #[must_use]
    pub fn token_probability(&self, before: Token, previous: Token, current: Token) -> f32 {
        self.probability_at(&self.context_at(before, previous), current)
    }

    /// The token standing for *ch*.
    ///
    /// # Errors
    ///
    /// If *ch* is not in the model's vocabulary.
    pub fn token_of(&self, ch: char) -> Result<Token, NgramError> {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a Lexicon's length fits u32 by construction"
        )]
        self.vocabulary
            .binary_search(&ch)
            .map(|index| Token(index as u32 + Token::RESERVED))
            .map_err(|_| NgramError::UnknownCharacter { ch })
    }

    /// The two tokens preceding a position, given the characters before it.
    ///
    /// Positions before the start of the sequence are [`Token::BOS`].
    ///
    /// # Errors
    ///
    /// If *context* is longer than the model's order allows, or holds a
    /// character outside the vocabulary.
    fn context_tokens(&self, context: &[char]) -> Result<(Token, Token), NgramError> {
        if context.len() > ORDER - 1 {
            return Err(NgramError::ContextTooLong {
                len: context.len(),
                order: ORDER,
            });
        }
        let mut tokens = [Token::BOS; ORDER - 1];
        for (slot, ch) in tokens[ORDER - 1 - context.len()..].iter_mut().zip(context) {
            *slot = self.token_of(*ch)?;
        }
        Ok((tokens[0], tokens[1]))
    }

    /// `P(current | context)`, where *context* is up to two preceding characters.
    ///
    /// # Errors
    ///
    /// If *context* is too long, or any character is outside the vocabulary.
    pub fn probability(&self, context: &[char], current: char) -> Result<f32, NgramError> {
        let (before, previous) = self.context_tokens(context)?;
        Ok(self.token_probability(before, previous, self.token_of(current)?))
    }

    /// The probability that a sequence ends after *context*.
    ///
    /// # Errors
    ///
    /// If *context* is too long, or any character is outside the vocabulary.
    pub fn end_probability(&self, context: &[char]) -> Result<f32, NgramError> {
        let (before, previous) = self.context_tokens(context)?;
        Ok(self.token_probability(before, previous, Token::EOS))
    }

    /// Serialise the model.
    ///
    /// # Errors
    ///
    /// If the encoder fails, which for an in-memory buffer means the model is
    /// larger than the machine can allocate.
    pub fn to_bytes(&self) -> Result<Vec<u8>, NgramError> {
        postcard::to_stdvec(self).map_err(NgramError::Encode)
    }

    /// Load a model from disk, whichever layout the file holds.
    ///
    /// A file that opens with the mapped layout's magic is memory-mapped: its
    /// tables point into the page cache rather than the heap. Anything else
    /// is read and decoded as the postcard serialisation
    /// [`NgramModel::from_bytes`] has always loaded.
    ///
    /// # Errors
    ///
    /// If the file cannot be read, is neither layout, or fails the lexicon
    /// check [`NgramModel::from_bytes`] applies.
    pub fn open(path: &Path, lexicon: &Lexicon) -> Result<Self, NgramError> {
        if mapped::is_mapped(path) {
            return mapped::open_file(path, lexicon);
        }
        let bytes = std::fs::read(path).map_err(NgramError::Io)?;
        Self::from_bytes(&bytes, lexicon)
    }

    /// Write the model in the memory-mappable layout [`NgramModel::open`]
    /// reads. The postcard `ngram.bin` is the trainer's interchange format;
    /// this is the one a session should load, since its tables never touch
    /// the heap.
    ///
    /// # Errors
    ///
    /// If the file cannot be written.
    pub fn write_mapped(&self, path: &Path) -> Result<(), NgramError> {
        mapped::write(self, path)
    }

    /// The check `from_bytes` and `open` share: the model's vocabulary must
    /// be this lexicon, in this order, and its dense arrays must agree.
    pub(crate) fn check_lexicon(&self, lexicon: &Lexicon) -> Result<(), NgramError> {
        if self.vocabulary.len() != lexicon.len() {
            return Err(NgramError::LexiconSize {
                model: self.vocabulary.len(),
                lexicon: lexicon.len(),
            });
        }
        if self.unigram.len() != lexicon.len() + Token::RESERVED as usize
            || self.bigram_backoff.len() != self.unigram.len()
        {
            return Err(NgramError::Corrupt);
        }
        for (index, expected) in self.vocabulary.iter().enumerate() {
            let id = lexicon.id_of(*expected).ok_or(NgramError::LexiconContent {
                index,
                ch: *expected,
            })?;
            if id.index() != index {
                return Err(NgramError::LexiconContent {
                    index,
                    ch: *expected,
                });
            }
        }
        Ok(())
    }

    /// Load a model and check it against the lexicon it will be used with.
    ///
    /// The check is not a formality: a `CharId` means nothing on its own, so a
    /// model loaded against a lexicon it was not trained on would score a
    /// different character at every position and never say so.
    ///
    /// # Errors
    ///
    /// If the bytes are not a model, or the model was trained against a
    /// different lexicon.
    pub fn from_bytes(bytes: &[u8], lexicon: &Lexicon) -> Result<Self, NgramError> {
        let model: Self = postcard::from_bytes(bytes).map_err(NgramError::Decode)?;
        model.check_lexicon(lexicon)?;
        Ok(model)
    }
}

/// # Panics
///
/// If a [`CharId`] came from a different lexicon than the model was loaded
/// against. [`NgramModel::from_bytes`] rules that out for the pair it checked.
impl Transition for NgramModel {
    const HISTORY: usize = ORDER - 1;

    type State = Context;

    fn start(&self, _context: Option<&str>, _asked: &Asked<'_>) -> Context {
        self.context_at(Token::BOS, Token::BOS)
    }

    fn score(&self, state: &Context, candidate: CharId) -> f32 {
        self.probability_at(state, Token::of(candidate)).ln()
    }

    /// The whole candidate list against one context: `allowed` arrives
    /// sorted, so every candidate's bigram key shares the `previous` row and
    /// its trigram key the `(before, previous)` row — each table is walked
    /// once with a forward cursor instead of a binary search per pair. Over
    /// the mapped table that turns ~22 cold page touches per pair into one
    /// lower-bound plus adjacent entries per beam.
    fn score_many(&self, state: &Context, weight: f32, allowed: &[CharId], out: &mut [f32]) {
        let base = self.base();
        let bigram_prefix = u64::from(state.previous.0) * base;
        let mut bigram = self.bigram.row(bigram_prefix);
        let pair = state
            .trigram_backoff
            .map(|backoff| (backoff, self.pack2(state.before, state.previous) * base));
        let mut trigram = pair.map(|(_, prefix)| self.trigram.row(prefix));
        for (slot, &candidate) in out.iter_mut().zip(allowed.iter()) {
            let token = Token::of(candidate);
            let level1 = self.unigram[token.index()];
            let level2 = bigram.at(bigram_prefix + u64::from(token.0)).unwrap_or(0.0)
                + state.bigram_backoff * level1;
            let p = match (pair, &mut trigram) {
                (Some((backoff, prefix)), Some(row)) => {
                    row.at(prefix + u64::from(token.0)).unwrap_or(0.0) + backoff * level2
                }
                _ => level2,
            };
            *slot += weight * p.ln();
        }
    }

    fn finish(&self, state: &Context) -> f32 {
        self.probability_at(state, Token::EOS).ln()
    }

    fn advance(&self, steps: &[(&Context, CharId, Asked<'_>)]) -> Vec<Context> {
        steps
            .iter()
            .map(|(state, ch, _)| self.context_at(state.previous, Token::of(*ch)))
            .collect()
    }
}
