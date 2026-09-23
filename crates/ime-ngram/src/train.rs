//! Counting a corpus and turning the counts into a model.

use crate::NgramError;
use crate::model::{NgramModel, ORDER, Token};
use crate::table::ProbTable;
use ime_pinyin::Lexicon;
use std::collections::HashMap;

/// A count as a float. Counts stay far below `2^53`, so this is exact.
#[expect(clippy::cast_precision_loss, reason = "counts stay far below 2^53")]
fn as_float(count: usize) -> f64 {
    count as f64
}

/// A token index unpacked from a key. Every key was packed from token indices,
/// so nothing can be lost on the way back out.
#[expect(
    clippy::cast_possible_truncation,
    reason = "keys are packed from usize token indices"
)]
fn as_index(packed: u64) -> usize {
    packed as usize
}

/// A probability narrowed to what the model stores.
#[expect(
    clippy::cast_possible_truncation,
    reason = "probabilities are stored as f32"
)]
fn as_prob(value: f64) -> f32 {
    value as f32
}

/// The absolute discount for one order, estimated from its count of counts.
///
/// The discount is what an order gives up so that the order below it has
/// something to say. A discount of zero would leave nothing over, and every
/// n-gram the corpus happens not to contain would then be not merely unlikely
/// but impossible -- a probability of exactly zero, a log score of negative
/// infinity, and a decoder that cannot rank the sentences it excludes. That is a
/// corpus too small or too repetitive to train on, so it is an error rather than
/// something to clamp away.
fn discount(order: usize, counts: impl Iterator<Item = u32>) -> Result<f64, NgramError> {
    let mut singletons = 0usize;
    let mut doubletons = 0usize;
    for count in counts {
        match count {
            1 => singletons += 1,
            2 => doubletons += 1,
            _ => {}
        }
    }
    let value = as_float(singletons) / as_float(singletons + 2 * doubletons);
    if value.is_nan() || value <= 0.0 || value > 1.0 {
        return Err(NgramError::DegenerateDiscount {
            order,
            singletons,
            doubletons,
        });
    }
    Ok(value)
}

/// Accumulates character trigrams over a corpus.
///
/// Only trigram counts are kept. Every lower order a Kneser-Ney model needs is a
/// *continuation* count -- how many distinct contexts something appeared in --
/// and those are read off the trigram types at the end, so counting them
/// separately would be counting the same thing twice.
///
/// Characters outside the lexicon -- latin, digits, punctuation, whitespace --
/// are sequence breaks rather than tokens. A model that had to spell them would
/// spend its probability mass on things an IME never emits.
pub struct Counter<'lexicon> {
    lexicon: &'lexicon Lexicon,
    trigrams: HashMap<u64, u32>,
    base: u64,
    lines: usize,
}

impl<'lexicon> Counter<'lexicon> {
    /// Start counting against *lexicon*.
    ///
    /// # Errors
    ///
    /// If the lexicon is too large for three tokens to be packed into a `u64`
    /// key.
    pub fn new(lexicon: &'lexicon Lexicon) -> Result<Self, NgramError> {
        const { assert!(ORDER == 3, "the u64 key packing assumes a trigram") };
        let base = lexicon.len() as u64 + u64::from(Token::RESERVED);
        if base.checked_pow(3).is_none() {
            return Err(NgramError::VocabularyTooLarge {
                count: lexicon.len(),
            });
        }
        Ok(Self {
            lexicon,
            trigrams: HashMap::new(),
            base,
            lines: 0,
        })
    }

    /// Count the character sequences in one line of the corpus.
    pub fn observe(&mut self, line: &str) {
        self.lines += 1;
        let mut window = [Token::BOS; ORDER - 1];
        let mut open = false;
        for ch in line.chars() {
            if let Some(id) = self.lexicon.id_of(ch) {
                let token = Token::of(id);
                self.count(window, token);
                window = [window[1], token];
                open = true;
            } else {
                if open {
                    self.count(window, Token::EOS);
                }
                window = [Token::BOS; ORDER - 1];
                open = false;
            }
        }
        if open {
            self.count(window, Token::EOS);
        }
    }

    /// How many lines have been observed.
    #[must_use]
    pub const fn lines(&self) -> usize {
        self.lines
    }

    /// How many distinct trigrams have been seen.
    #[must_use]
    pub fn trigram_types(&self) -> usize {
        self.trigrams.len()
    }

    fn count(&mut self, window: [Token; ORDER - 1], current: Token) {
        let key = self.pack3(window[0], window[1], current);
        *self.trigrams.entry(key).or_insert(0) += 1;
    }

    fn pack2(&self, first: Token, second: Token) -> u64 {
        first.index() as u64 * self.base + second.index() as u64
    }

    fn pack3(&self, first: Token, second: Token, third: Token) -> u64 {
        self.pack2(first, second) * self.base + third.index() as u64
    }

    /// Estimate the model from what has been counted.
    ///
    /// # Errors
    ///
    /// If the corpus held no character at all, or if any order's discount comes
    /// out degenerate -- see [`NgramError::DegenerateDiscount`].
    pub fn finish(self) -> Result<NgramModel, NgramError> {
        if self.trigrams.is_empty() {
            return Err(NgramError::EmptyCorpus);
        }
        let tokens = self.lexicon.len() + Token::RESERVED as usize;
        let base = self.base;

        // Trigram contexts, and the continuation counts the bigram level needs:
        // how many distinct characters preceded each bigram.
        let mut context_total: HashMap<u64, u32> = HashMap::new();
        let mut context_types: HashMap<u64, u32> = HashMap::new();
        let mut continuation2: HashMap<u64, u32> = HashMap::new();
        for (key, count) in &self.trigrams {
            let context = key / base;
            let third = key % base;
            let second = context % base;
            *context_total.entry(context).or_insert(0) += count;
            *context_types.entry(context).or_insert(0) += 1;
            *continuation2.entry(second * base + third).or_insert(0) += 1;
        }

        // The same step once more, one order down: how many distinct characters
        // preceded each character.
        let mut bigram_total = vec![0u32; tokens];
        let mut bigram_types = vec![0u32; tokens];
        let mut continuation1 = vec![0u32; tokens];
        for (key, count) in &continuation2 {
            let previous = as_index(key / base);
            let current = as_index(key % base);
            bigram_total[previous] += count;
            bigram_types[previous] += 1;
            continuation1[current] += 1;
        }
        let type_total = as_float(continuation2.len());
        let target_types = as_float(continuation1.iter().filter(|c| **c > 0).count());

        let d3 = discount(3, self.trigrams.values().copied())?;
        let d2 = discount(2, continuation2.values().copied())?;
        let d1 = discount(1, continuation1.iter().copied().filter(|c| *c > 0))?;

        // Every token but the boundary marker at the start is a prediction
        // target, so that is what the uniform floor spreads itself over.
        let uniform = 1.0 / as_float(tokens - 1);
        let escape = d1 * target_types / type_total;
        let unigram: Box<[f32]> = (0..tokens)
            .map(|token| {
                if token == Token::BOS.index() {
                    return 0.0;
                }
                let seen = f64::from(continuation1[token]);
                as_prob((seen - d1).max(0.0) / type_total + escape * uniform)
            })
            .collect();

        let bigram_backoff: Box<[f32]> = (0..tokens)
            .map(|token| {
                if bigram_total[token] == 0 {
                    return 1.0;
                }
                as_prob(d2 * f64::from(bigram_types[token]) / f64::from(bigram_total[token]))
            })
            .collect();

        let bigram = ProbTable::build(
            continuation2
                .iter()
                .map(|(key, count)| {
                    let previous = as_index(key / base);
                    let discounted =
                        (f64::from(*count) - d2).max(0.0) / f64::from(bigram_total[previous]);
                    (*key, as_prob(discounted))
                })
                .collect(),
        );

        let trigram_backoff = ProbTable::build(
            context_total
                .iter()
                .map(|(key, total)| {
                    let types = f64::from(context_types[key]);
                    (*key, as_prob(d3 * types / f64::from(*total)))
                })
                .collect(),
        );

        let trigram = ProbTable::build(
            self.trigrams
                .iter()
                .map(|(key, count)| {
                    let total = f64::from(context_total[&(key / base)]);
                    (*key, as_prob((f64::from(*count) - d3).max(0.0) / total))
                })
                .collect(),
        );

        Ok(NgramModel::new(
            self.lexicon.characters().into(),
            unigram,
            bigram_backoff,
            bigram,
            trigram_backoff,
            trigram,
        ))
    }
}
