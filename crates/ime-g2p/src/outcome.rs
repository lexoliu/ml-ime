//! What an annotator produces for a sentence, and how two of them are compared.

use crate::error::{Error, Result};
use crate::text::{han_characters, toneless};
use std::future::Future;

/// One annotator's tone-numbered syllable for each Han character of a sentence.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Reading {
    /// One syllable per Han character, in order.
    pub syllables: Vec<String>,
}

impl Reading {
    /// Build a reading from anything that yields syllables.
    pub fn new<I: IntoIterator<Item = String>>(syllables: I) -> Self {
        Self {
            syllables: syllables.into_iter().collect(),
        }
    }
}

/// Why an annotator produced nothing usable for a sentence.
///
/// A refusal is recorded and counted rather than skipped, so a run that quietly
/// lost half its sentences to a broken endpoint cannot look like a clean run.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Refusal {
    /// What went wrong, in the shape `TypeName: message`.
    pub reason: String,
}

impl Refusal {
    /// Record a refusal with `reason`.
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }
}

/// What an annotator returns for one sentence.
///
/// A refusal is the error case of annotating, so it is spelled as one rather
/// than as a third state every caller has to remember to handle.
pub type Outcome = std::result::Result<Reading, Refusal>;

/// A source of per-character pinyin for a batch of sentences.
pub trait Annotator {
    /// Column name this annotator's readings are stored under.
    fn name(&self) -> &'static str;

    /// One outcome per input sentence, in order.
    fn annotate(&self, texts: &[String]) -> impl Future<Output = Vec<Outcome>> + Send;
}

/// Two annotators' readings for one sentence, position by position.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Comparison {
    characters: Vec<char>,
    first: Vec<String>,
    second: Vec<String>,
    agree: Vec<bool>,
}

impl Comparison {
    /// The Han characters the two readings are aligned against.
    #[must_use]
    pub fn characters(&self) -> &[char] {
        &self.characters
    }

    /// The first annotator's syllables, tones kept.
    #[must_use]
    pub fn first(&self) -> &[String] {
        &self.first
    }

    /// The second annotator's syllables, tones kept.
    #[must_use]
    pub fn second(&self) -> &[String] {
        &self.second
    }

    /// Per position, whether the two annotators spell the syllable the same way
    /// once the tone is dropped.
    #[must_use]
    pub fn agree(&self) -> &[bool] {
        &self.agree
    }

    /// Whether every position agrees. Sentences that do form the training set.
    #[must_use]
    pub fn unanimous(&self) -> bool {
        self.agree.iter().all(|agreed| *agreed)
    }

    /// How many positions agree.
    #[must_use]
    pub fn agreed(&self) -> usize {
        self.agree.iter().filter(|agreed| **agreed).count()
    }
}

/// Line up two readings against the Han characters of `text`.
///
/// # Errors
///
/// If the two readings and the sentence do not all have the same length, which
/// means one annotator has silently dropped or invented a position.
pub fn compare(text: &str, first: &Reading, second: &Reading) -> Result<Comparison> {
    let characters = han_characters(text);
    if characters.len() != first.syllables.len() || characters.len() != second.syllables.len() {
        return Err(Error::Misaligned {
            text: text.to_owned(),
            characters: characters.len(),
            first: first.syllables.len(),
            second: second.syllables.len(),
        });
    }
    let agree = first
        .syllables
        .iter()
        .zip(&second.syllables)
        .map(|(left, right)| toneless(left) == toneless(right))
        .collect();
    Ok(Comparison {
        characters,
        first: first.syllables.clone(),
        second: second.syllables.clone(),
        agree,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reading(syllables: &[&str]) -> Reading {
        Reading::new(syllables.iter().map(|s| (*s).to_owned()))
    }

    #[test]
    fn agreement_is_on_the_toneless_syllable() {
        let comparison = compare(
            "中国",
            &reading(&["zhong1", "guo2"]),
            &reading(&["zhong4", "guo2"]),
        )
        .expect("the lengths line up");
        assert_eq!(comparison.agree(), [true, true]);
        assert!(comparison.unanimous());
    }

    #[test]
    fn a_different_spelling_is_a_disagreement_however_the_tone_falls() {
        let comparison = compare(
            "重要",
            &reading(&["chong2", "yao4"]),
            &reading(&["zhong4", "yao4"]),
        )
        .expect("the lengths line up");
        assert_eq!(comparison.agree(), [false, true]);
        assert!(!comparison.unanimous());
        assert_eq!(comparison.agreed(), 1);
    }

    #[test]
    fn u_umlaut_folds_onto_v_before_the_two_are_compared() {
        let comparison =
            compare("绿色", &reading(&["lü4", "se4"]), &reading(&["lv4", "se4"])).expect("aligned");
        assert!(comparison.unanimous());
    }

    #[test]
    fn the_two_spellings_of_u_umlaut_after_y_are_one_reading_not_a_disagreement() {
        // g2pW writes 与 as `yu`; the prompt tells the LLM to spell every ü as
        // `v`, so it writes `yv`. Neither is wrong and both segment to the same
        // keystrokes, so the position agrees.
        let comparison =
            compare("与其", &reading(&["yv2", "qi2"]), &reading(&["yu2", "qi2"])).expect("aligned");
        assert_eq!(comparison.agree(), [true, true]);
        assert!(comparison.unanimous());
    }

    #[test]
    fn punctuation_is_not_a_position_to_compare() {
        let comparison = compare(
            "中，国",
            &reading(&["zhong1", "guo2"]),
            &reading(&["zhong1", "guo2"]),
        )
        .expect("only the Han characters count");
        assert_eq!(comparison.characters(), ['中', '国']);
    }

    #[test]
    fn a_reading_of_the_wrong_length_is_refused_rather_than_truncated() {
        let error = compare("中国", &reading(&["zhong1"]), &reading(&["zhong1", "guo2"]))
            .expect_err("the lengths disagree");
        assert!(matches!(
            error,
            Error::Misaligned {
                characters: 2,
                first: 1,
                ..
            }
        ));
    }
}
