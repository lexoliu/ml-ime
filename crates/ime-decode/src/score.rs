//! What the decoder asks a model for.
//!
//! Two questions, kept apart because two different things answer them: how well a
//! character fits the position it would occupy ([`Emission`]), and how well it
//! follows the characters already chosen ([`Transition`]). The neural route
//! supplies the first and the n-gram the second; the n-gram baseline supplies the
//! second alone and leaves the first flat.

use ime_pinyin::CharId;

/// How many preceding characters a beam state can remember.
///
/// Two, which is what an interpolated trigram needs. Raising it widens every beam
/// state, so it is a deliberate cost rather than a free parameter.
pub const MAX_HISTORY: usize = 2;

/// The characters immediately preceding a position, oldest first.
///
/// A slot is `None` only when the position is that close to the start of the
/// sequence. Deciding what that means -- a sentence boundary token, a uniform
/// prior -- belongs to the model, not to the decoder.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct History([Option<CharId>; MAX_HISTORY]);

impl History {
    /// The history at the very start of a sequence: nothing emitted yet.
    pub const START: Self = Self([None; MAX_HISTORY]);

    /// The character *distance* positions back, or `None` if that position lies
    /// before the start of the sequence.
    ///
    /// A distance of one is the immediately preceding character.
    ///
    /// # Panics
    ///
    /// If *distance* is zero or greater than [`MAX_HISTORY`].
    #[must_use]
    pub fn back(self, distance: usize) -> Option<CharId> {
        assert!(
            (1..=MAX_HISTORY).contains(&distance),
            "history distance {distance} is outside 1..={MAX_HISTORY}"
        );
        self.0[MAX_HISTORY - distance]
    }

    /// This history with *ch* appended, dropping whatever fell off the far end.
    #[must_use]
    pub fn extended(self, ch: CharId) -> Self {
        let mut slots = self.0;
        slots.rotate_left(1);
        slots[MAX_HISTORY - 1] = Some(ch);
        Self(slots)
    }

    /// This history with everything older than the most recent *keep* characters
    /// forgotten.
    ///
    /// The decoder uses this to merge beam states that a model of a given order
    /// cannot tell apart: for a bigram, two paths agreeing on the last character
    /// score identically from here on, so only the better one need survive.
    #[must_use]
    pub fn truncated(self, keep: usize) -> Self {
        let mut slots = self.0;
        for slot in slots.iter_mut().take(MAX_HISTORY.saturating_sub(keep)) {
            *slot = None;
        }
        Self(slots)
    }
}

/// How well a character fits a position on its own.
///
/// Indexed by segmentation path as well as position: one typed string yields
/// several readings of different lengths, and a model that scores them scores all
/// of them together -- for the neural route, in a single batched forward.
pub trait Emission {
    /// Score of *candidate* at *position* of segmentation *path*, as a log
    /// probability or any other quantity where larger is better.
    fn score(&self, path: usize, position: usize, candidate: CharId) -> f32;
}

/// An emission model that prefers nothing.
///
/// Every candidate scores zero, so the decoder is driven entirely by its
/// transition model. This is the n-gram baseline, and the control the neural
/// route is measured against.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub struct Uniform;

impl Emission for Uniform {
    fn score(&self, _path: usize, _position: usize, _candidate: CharId) -> f32 {
        0.0
    }
}

/// A transition model that prefers nothing.
///
/// Every character scores zero after every history, so the decoder is driven
/// entirely by its emission model. This is the neural route stripped of the
/// n-gram -- the ablation that says how much of the fused number the emissions
/// earned on their own.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub struct NoTransition;

impl Transition for NoTransition {
    const HISTORY: usize = 1;

    fn score(&self, _history: History, _candidate: CharId) -> f32 {
        0.0
    }

    fn finish(&self, _history: History) -> f32 {
        0.0
    }
}

/// How well a character follows the ones before it.
pub trait Transition {
    /// How many preceding characters this model conditions on: one for a bigram,
    /// two for a trigram. Must lie in `1..=MAX_HISTORY`, which
    /// [`decode`](crate::decode) checks when it is instantiated.
    const HISTORY: usize;

    /// Score of *candidate* following *history*, as a log probability or any
    /// other quantity where larger is better.
    fn score(&self, history: History, candidate: CharId) -> f32;

    /// Score of the sequence ending after *history*.
    ///
    /// Without this a decoder is free to end anywhere, and a reading that trails
    /// off mid-word costs no more than one that closes.
    fn finish(&self, history: History) -> f32;
}

#[cfg(test)]
mod tests {
    use super::*;
    use ime_pinyin::{Lexicon, SyllableTable};

    fn ids() -> (CharId, CharId, CharId) {
        let table = SyllableTable::load();
        let lexicon = Lexicon::load(&table).expect("generated tables must agree");
        (
            lexicon.id_of('中').expect("中"),
            lexicon.id_of('国').expect("国"),
            lexicon.id_of('人').expect("人"),
        )
    }

    #[test]
    fn extending_shifts_the_oldest_slot_out() {
        let (a, b, c) = ids();
        let history = History::START.extended(a).extended(b).extended(c);
        assert_eq!(history.back(1), Some(c));
        assert_eq!(history.back(2), Some(b));
    }

    #[test]
    fn the_start_history_is_empty_at_every_distance() {
        assert_eq!(History::START.back(1), None);
        assert_eq!(History::START.back(2), None);
    }

    #[test]
    fn truncating_merges_states_a_lower_order_model_cannot_distinguish() {
        let (a, b, c) = ids();
        let via_a = History::START.extended(a).extended(c).truncated(1);
        let via_b = History::START.extended(b).extended(c).truncated(1);
        assert_eq!(via_a, via_b);
        assert_eq!(via_a.back(1), Some(c));
        assert_ne!(
            History::START.extended(a).extended(c).truncated(2),
            History::START.extended(b).extended(c).truncated(2)
        );
    }
}
