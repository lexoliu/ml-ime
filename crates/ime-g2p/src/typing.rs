//! What somebody would have *pressed* for a sentence, in each of three styles.
//!
//! An evaluation record's `pinyin` field is keystrokes, not readings, and people
//! do not press the same keys twice. Training already knows that: the augmentation
//! in `python/src/mlime/train/samples.py` types 55% of its examples out in full,
//! abbreviates 25% of them syllable by syllable, and types the remaining 20% full
//! at the start and abbreviated from a random point on. An evaluation set drawn
//! only in full pinyin therefore measures the model over one third of what it was
//! trained for, and says nothing about the two thirds it is weakest at.
//!
//! So the styles are defined once, here, with the same rules the trainer uses --
//! [`initial`] is the Rust twin of `mlime.train.spans.initial`, and
//! [`TypingStyle::spans`] is the twin of `mlime.train.samples.type_syllables` --
//! and an evaluation twin of a set is the same draw typed a different way.
//!
//! The draw is seeded from the record's own text rather than from a stream, for
//! the same reason [`crate::export::sample_rows`] sorts before it samples: a
//! sentence's keystrokes have to be a property of the sentence. Seeded from a
//! shared stream, the tenth record of a 1,000-record export and the tenth record
//! of a 5,525-record export would be typed differently, and the two sets could
//! not be compared. Seeded from the text, a sentence is typed the same way in
//! every set it appears in, forever.
//!
//! The one thing this does *not* reproduce is Python's generator: the trainer
//! draws from a Mersenne Twister seeded by `(seed, epoch, sample id)` and this
//! draws from `StdRng` seeded by the text, because the two are keyed on different
//! things and could not agree anyway. What has to match, and does, is the shape
//! of the result -- which syllables may be abbreviated, what an abbreviation is,
//! and where a mixed sentence is allowed to break.

use crate::error::{Error, Result};
use blake2::digest::consts::U8;
use blake2::{Blake2b, Digest as _};
use rand::rngs::StdRng;
use rand::{Rng as _, SeedableRng as _};
use std::fmt;

/// The initials spelled with two letters.
///
/// Typing `z` for `zhong` is legal and the decoder handles it, but nobody who
/// means to abbreviate `zhong` presses `z` -- they press `zh` -- so neither the
/// training augmentation nor an evaluation twin may manufacture the other thing.
const MULTI_LETTER_INITIALS: [&str; 3] = ["zh", "ch", "sh"];

/// The default rate at which an abbreviating typist drops a syllable to its
/// initial, which is the rate milestone 3 trained under.
pub const DEFAULT_ABBREVIATE_SYLLABLE: f64 = 0.7;

/// The abbreviation of `syllable`: its initial, kept whole for zh/ch/sh.
///
/// A syllable with no initial consonant -- `an`, `er`, `ou`, `ang` -- has no
/// special case: it abbreviates to its first vowel, which is exactly what a
/// person types for 安 or 而. Python's `initial` does the same by falling
/// through to `syllable[0]`, and the two must agree character for character or
/// an evaluation record would ask for keystrokes no training example ever used.
///
/// # Errors
///
/// If the syllable is empty, which means a caller folded a reading away.
pub fn initial(syllable: &str) -> Result<&str> {
    if let Some(prefix) = MULTI_LETTER_INITIALS
        .iter()
        .find(|prefix| syllable.starts_with(**prefix))
    {
        return Ok(prefix);
    }
    let first = syllable
        .chars()
        .next()
        .ok_or_else(|| Error::Invariant("the empty string is not a syllable".to_owned()))?;
    Ok(&syllable[..first.len_utf8()])
}

/// Which of the three typing styles a set is drawn in.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub enum Typing {
    /// Every syllable typed out.
    #[default]
    Full,
    /// Each syllable independently dropped to its initial, with probability
    /// [`TypingStyle::abbreviate_syllable`]. Some syllables survive whole,
    /// which is what a real abbreviation pass looks like.
    Abbreviated,
    /// Full up to a cut drawn strictly inside the sentence, abbreviated after
    /// it: somebody who starts careful and gets lazy.
    Mixed,
}

impl Typing {
    /// How the style is spelled on the command line and in a log line.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Abbreviated => "abbreviated",
            Self::Mixed => "mixed",
        }
    }
}

impl fmt::Display for Typing {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.label())
    }
}

/// A typing style and the one knob it takes: how eagerly it abbreviates.
///
/// The knob belongs to the style rather than to a call site so that an ablation
/// is a different value, not a different branch, and so that a rate outside
/// `[0, 1]` is refused where it is written rather than silently clamped where it
/// is drawn.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct TypingStyle {
    typing: Typing,
    abbreviate_syllable: f64,
}

impl Default for TypingStyle {
    fn default() -> Self {
        Self {
            typing: Typing::Full,
            abbreviate_syllable: DEFAULT_ABBREVIATE_SYLLABLE,
        }
    }
}

impl TypingStyle {
    /// A style that abbreviates at `abbreviate_syllable`.
    ///
    /// # Errors
    ///
    /// If `abbreviate_syllable` is not a probability.
    pub fn new(typing: Typing, abbreviate_syllable: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&abbreviate_syllable) {
            return Err(Error::Invariant(format!(
                "abbreviate_syllable must be a probability, got {abbreviate_syllable}"
            )));
        }
        Ok(Self {
            typing,
            abbreviate_syllable,
        })
    }

    /// Which style this is.
    #[must_use]
    pub const fn typing(self) -> Typing {
        self.typing
    }

    /// How often an abbreviating pass drops a syllable to its initial.
    #[must_use]
    pub const fn abbreviate_syllable(self) -> f64 {
        self.abbreviate_syllable
    }

    /// The spans somebody would have pressed for `syllables`, typed this way.
    ///
    /// `text` is the sentence the syllables read, and is what the draw is seeded
    /// from: the same sentence yields the same keystrokes in every set, at every
    /// size and in any draw order.
    ///
    /// A mixed sentence needs both halves to exist, so the cut falls strictly
    /// inside it; a one-syllable sentence has nowhere to put a cut and is
    /// abbreviated whole, which is what "gets lazy after the first syllable"
    /// degrades to when there is only one.
    ///
    /// # Errors
    ///
    /// If there are no syllables to type, or one of them is empty.
    pub fn spans(self, syllables: &[String], text: &str) -> Result<Vec<String>> {
        if syllables.is_empty() {
            return Err(Error::Invariant(format!(
                "{text:?} has no syllables, so there is nothing to type"
            )));
        }
        match self.typing {
            Typing::Full => Ok(syllables.to_vec()),
            Typing::Abbreviated => {
                let mut rng = seeded(text);
                syllables
                    .iter()
                    .map(|syllable| {
                        if rng.random::<f64>() < self.abbreviate_syllable {
                            initial(syllable).map(ToOwned::to_owned)
                        } else {
                            Ok(syllable.clone())
                        }
                    })
                    .collect()
            }
            Typing::Mixed => {
                let mut rng = seeded(text);
                let cut = if syllables.len() == 1 {
                    0
                } else {
                    rng.random_range(1..syllables.len())
                };
                let (typed, lazy) = syllables.split_at(cut);
                let mut spans = typed.to_vec();
                for syllable in lazy {
                    spans.push(initial(syllable)?.to_owned());
                }
                Ok(spans)
            }
        }
    }

    /// The spans run together, which is what an evaluation record carries.
    ///
    /// # Errors
    ///
    /// As [`TypingStyle::spans`].
    pub fn keystrokes(self, syllables: &[String], text: &str) -> Result<String> {
        Ok(self.spans(syllables, text)?.concat())
    }
}

/// The generator for one sentence, determined by its text and nothing else.
fn seeded(text: &str) -> StdRng {
    let mut hasher = Blake2b::<U8>::new();
    hasher.update(text.as_bytes());
    StdRng::seed_from_u64(u64::from_be_bytes(hasher.finalize().into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn syllables(spans: &[&str]) -> Vec<String> {
        spans.iter().map(|span| (*span).to_owned()).collect()
    }

    #[test]
    fn an_initial_keeps_the_two_letter_ones_whole() {
        assert_eq!(initial("zhong").expect("a syllable"), "zh");
        assert_eq!(initial("chi").expect("a syllable"), "ch");
        assert_eq!(initial("shuo").expect("a syllable"), "sh");
        assert_eq!(initial("guo").expect("a syllable"), "g");
        assert_eq!(initial("zi").expect("a syllable"), "z");
    }

    #[test]
    fn a_syllable_with_no_initial_consonant_abbreviates_to_its_vowel() {
        // Python's `initial` has no zero-initial case and falls through to
        // `syllable[0]`; this is that behaviour, and it is also what somebody
        // types for 安 and 而.
        assert_eq!(initial("an").expect("a syllable"), "a");
        assert_eq!(initial("er").expect("a syllable"), "e");
        assert_eq!(initial("ou").expect("a syllable"), "o");
        assert_eq!(initial("a").expect("a syllable"), "a");
    }

    #[test]
    fn the_empty_string_is_not_a_syllable() {
        assert!(initial("").is_err());
    }

    #[test]
    fn a_rate_outside_zero_to_one_is_refused() {
        assert!(TypingStyle::new(Typing::Abbreviated, 1.5).is_err());
        assert!(TypingStyle::new(Typing::Abbreviated, -0.1).is_err());
        assert!(TypingStyle::new(Typing::Abbreviated, 0.0).is_ok());
    }

    #[test]
    fn full_typing_is_the_syllables_unchanged_and_draws_nothing() {
        let style = TypingStyle::default();
        let readings = syllables(&["zhong", "guo", "ren", "min"]);
        assert_eq!(
            style.keystrokes(&readings, "中国人民").expect("aligned"),
            "zhongguorenmin"
        );
        // The text is only ever a seed, so full typing cannot depend on it.
        assert_eq!(
            style.spans(&readings, "中国人民").expect("aligned"),
            style.spans(&readings, "别的句子").expect("aligned")
        );
    }

    #[test]
    fn abbreviated_typing_is_the_same_for_a_text_every_time() {
        let style = TypingStyle::new(Typing::Abbreviated, DEFAULT_ABBREVIATE_SYLLABLE)
            .expect("a probability");
        let readings = syllables(&["zhong", "guo", "ren", "min"]);
        let first = style.spans(&readings, "中国人民").expect("aligned");
        let second = style.spans(&readings, "中国人民").expect("aligned");
        assert_eq!(first, second);
        // Every span is either the syllable or its initial, and nothing else.
        for (span, syllable) in first.iter().zip(&readings) {
            assert!(
                span == syllable || span == initial(syllable).expect("a syllable"),
                "{span:?} is neither {syllable:?} nor its initial"
            );
        }
        // The seed is the text, so other sentences are drawn differently.
        assert!(
            (0..16).any(|index| {
                style
                    .spans(&readings, &format!("句子{index}"))
                    .expect("aligned")
                    != first
            }),
            "sixteen sentences all typed {first:?}"
        );
    }

    #[test]
    fn abbreviating_at_certainty_drops_every_syllable_to_its_initial() {
        let style = TypingStyle::new(Typing::Abbreviated, 1.0).expect("a probability");
        let readings = syllables(&["zhong", "guo", "an", "shuo"]);
        assert_eq!(
            style.keystrokes(&readings, "中国安说").expect("aligned"),
            "zhgash"
        );
        let never = TypingStyle::new(Typing::Abbreviated, 0.0).expect("a probability");
        assert_eq!(
            never.keystrokes(&readings, "中国安说").expect("aligned"),
            "zhongguoanshuo"
        );
    }

    #[test]
    fn mixed_typing_keeps_a_full_prefix_and_abbreviates_the_suffix() {
        let style =
            TypingStyle::new(Typing::Mixed, DEFAULT_ABBREVIATE_SYLLABLE).expect("a probability");
        let readings = syllables(&["zhong", "guo", "ren", "min", "yin", "hang"]);
        for index in 0..64 {
            let text = format!("句子{index}");
            let spans = style.spans(&readings, &text).expect("aligned");
            assert_eq!(spans.len(), readings.len());
            let cut = spans
                .iter()
                .zip(&readings)
                .take_while(|(span, syllable)| span == syllable)
                .count();
            assert!(
                (1..readings.len()).contains(&cut),
                "{text}: the cut at {cut} is not strictly inside the sentence"
            );
            for (span, syllable) in spans.iter().zip(&readings).skip(cut) {
                assert_eq!(span, initial(syllable).expect("a syllable"));
            }
        }
    }

    #[test]
    fn a_one_syllable_mixed_sentence_is_its_initial() {
        let style =
            TypingStyle::new(Typing::Mixed, DEFAULT_ABBREVIATE_SYLLABLE).expect("a probability");
        assert_eq!(
            style
                .keystrokes(&syllables(&["shuo"]), "说")
                .expect("aligned"),
            "sh"
        );
        assert_eq!(
            style
                .keystrokes(&syllables(&["an"]), "安")
                .expect("aligned"),
            "a"
        );
    }

    #[test]
    fn a_sentence_with_no_syllables_has_nothing_to_type() {
        assert!(TypingStyle::default().spans(&[], "").is_err());
    }
}
