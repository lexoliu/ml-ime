//! The trained model: three interpolated levels and the lookups over them.

use crate::NgramError;
use crate::table::ProbTable;
use ime_decode::{History, Transition};
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

    /// `P(current | previous, before)` under the interpolated model.
    ///
    /// Never zero: the unigram level interpolates with a uniform floor and every
    /// discount is strictly positive, so no token is unreachable.
    #[must_use]
    pub fn token_probability(&self, before: Token, previous: Token, current: Token) -> f32 {
        let level1 = self.unigram[current.index()];
        let level2 = self
            .bigram
            .get(self.pack2(previous, current))
            .unwrap_or(0.0)
            + self.bigram_backoff[previous.index()] * level1;
        match self.trigram_backoff.get(self.pack2(before, previous)) {
            Some(backoff) => {
                self.trigram
                    .get(self.pack3(before, previous, current))
                    .unwrap_or(0.0)
                    + backoff * level2
            }
            None => level2,
        }
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
        if model.vocabulary.len() != lexicon.len() {
            return Err(NgramError::LexiconSize {
                model: model.vocabulary.len(),
                lexicon: lexicon.len(),
            });
        }
        if model.unigram.len() != lexicon.len() + Token::RESERVED as usize
            || model.bigram_backoff.len() != model.unigram.len()
        {
            return Err(NgramError::Corrupt);
        }
        for (index, expected) in model.vocabulary.iter().enumerate() {
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
        Ok(model)
    }
}

/// The token standing for a slot of decoder history: the character it holds, or
/// [`Token::BOS`] where the slot lies before the start of the sequence.
fn slot(history: History, distance: usize) -> Token {
    history.back(distance).map_or(Token::BOS, Token::of)
}

/// # Panics
///
/// If a [`CharId`] came from a different lexicon than the model was loaded
/// against. [`NgramModel::from_bytes`] rules that out for the pair it checked.
impl Transition for NgramModel {
    const HISTORY: usize = ORDER - 1;

    fn score(&self, history: History, candidate: CharId) -> f32 {
        self.token_probability(slot(history, 2), slot(history, 1), Token::of(candidate))
            .ln()
    }

    fn finish(&self, history: History) -> f32 {
        self.token_probability(slot(history, 2), slot(history, 1), Token::EOS)
            .ln()
    }
}
