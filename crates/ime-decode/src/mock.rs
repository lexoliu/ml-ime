//! Deterministic stand-ins for the CTC head and the transition, for tests and
//! the search benchmark: the search's correctness and cost do not depend on
//! what the scores are, only on their shape.

use ime_pinyin::EmitId;

use crate::keyed::{EmissionRows, KeyTransition};

/// A 64-bit mix (`SplitMix64`'s finaliser): a fixed function of its input.
#[must_use]
pub const fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// A value in `[0, 1)` drawn from *x*.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "24 bits of a hash, exact in f32"
)]
pub fn unit(x: u64) -> f32 {
    (mix(x) >> 40) as f32 / (1u64 << 24) as f32
}

/// A column of an emission row: a character or blank.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Column {
    /// The character with this emission index.
    Char(usize),
    /// Blank.
    Blank,
}

/// Rows over *keys* keys and *emittable* characters, each the log-softmax of
/// *logit* over its columns -- normalised over the whole set plus blank, as
/// the head is.
///
/// # Panics
///
/// Never: the buffer is whole rows by construction.
#[must_use]
pub fn rows(
    keys: usize,
    emittable: usize,
    mut logit: impl FnMut(usize, Column) -> f32,
) -> EmissionRows {
    let stride = emittable + 1;
    let mut data = vec![0.0f32; keys * stride];
    for (key, row) in data.chunks_exact_mut(stride).enumerate() {
        for (index, slot) in row.iter_mut().enumerate() {
            let column = if index == emittable {
                Column::Blank
            } else {
                Column::Char(index)
            };
            *slot = logit(key, column);
        }
        let high = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let norm = high + row.iter().map(|&v| (v - high).exp()).sum::<f32>().ln();
        for slot in row.iter_mut() {
            *slot -= norm;
        }
    }
    EmissionRows::new(emittable, data).expect("whole rows by construction")
}

/// Rows of hashed logits in `[0, spread)` under *seed*.
#[must_use]
pub fn hashed_rows(keys: usize, emittable: usize, seed: u64, spread: f32) -> EmissionRows {
    rows(keys, emittable, |key, column| {
        let index = match column {
            Column::Char(index) => index as u64,
            Column::Blank => u64::MAX,
        };
        spread * unit(seed ^ mix((key as u64) << 32 ^ index))
    })
}

/// A bigram transition of hashed scores in `(−spread, 0]`: the state is the
/// prefix's last character, so distinct prefixes share rows exactly when a
/// bigram model would.
#[derive(Copy, Clone, Debug)]
pub struct HashedBigram {
    /// What the scores are drawn under.
    pub seed: u64,
    /// How far apart the scores are.
    pub spread: f32,
}

impl HashedBigram {
    fn draw(&self, previous: Option<EmitId>, next: u64) -> f32 {
        let previous = previous.map_or(u64::MAX, |id| id.index() as u64);
        -self.spread * unit(self.seed ^ mix(previous.wrapping_mul(0x1_0000_0001) ^ next))
    }
}

impl KeyTransition for HashedBigram {
    type State = Option<EmitId>;

    fn start(&self) -> Self::State {
        None
    }

    fn score(&self, state: &Self::State, ids: &[EmitId], out: &mut [f32]) {
        for (slot, id) in out.iter_mut().zip(ids) {
            *slot = self.draw(*state, id.index() as u64);
        }
    }

    fn finish(&self, state: &Self::State) -> f32 {
        self.draw(*state, u64::MAX)
    }

    fn advance(&self, steps: &[(&Self::State, EmitId)]) -> Vec<Self::State> {
        steps.iter().map(|&(_, id)| Some(id)).collect()
    }
}

/// A transition that likes the given sentences: a character scores 0 after
/// the character it follows in one of them (the first after the context's
/// last), and *penalty* otherwise; ending scores 0 after a sentence's last
/// character and *penalty* otherwise.
#[derive(Clone, Debug)]
pub struct Favoured {
    pairs: Vec<(Option<EmitId>, EmitId)>,
    ends: Vec<EmitId>,
    first: Option<EmitId>,
    penalty: f32,
}

impl Favoured {
    /// Favour *sentences* after the context whose last character is *after*.
    #[must_use]
    pub fn new(after: Option<EmitId>, sentences: &[&[EmitId]], penalty: f32) -> Self {
        let mut pairs = Vec::new();
        let mut ends = Vec::new();
        for sentence in sentences {
            let mut previous = after;
            for &id in *sentence {
                pairs.push((previous, id));
                previous = Some(id);
            }
            ends.extend(previous);
        }
        pairs.sort_unstable();
        pairs.dedup();
        Self {
            pairs,
            ends,
            first: after,
            penalty,
        }
    }
}

impl KeyTransition for Favoured {
    type State = Option<EmitId>;

    fn start(&self) -> Self::State {
        self.first
    }

    fn score(&self, state: &Self::State, ids: &[EmitId], out: &mut [f32]) {
        for (slot, &id) in out.iter_mut().zip(ids) {
            *slot = if self.pairs.binary_search(&(*state, id)).is_ok() {
                0.0
            } else {
                self.penalty
            };
        }
    }

    fn finish(&self, state: &Self::State) -> f32 {
        if state.is_some_and(|id| self.ends.contains(&id)) {
            0.0
        } else {
            self.penalty
        }
    }

    fn advance(&self, steps: &[(&Self::State, EmitId)]) -> Vec<Self::State> {
        steps.iter().map(|&(_, id)| Some(id)).collect()
    }
}
