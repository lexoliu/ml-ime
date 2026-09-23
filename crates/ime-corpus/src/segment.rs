//! What a training target actually is: one uninterrupted run of Han characters.
//!
//! The model this corpus feeds is non-autoregressive and reads one syllable per
//! output position, so every character of a target has to be a Han character the
//! pinyin lexicon can pronounce. A sentence is not that. `今天，天气不错` is
//! ninety-odd percent Han and passed every earlier gate, but the comma in the
//! middle has no syllable behind it, and half the prepared corpus turned out to
//! be exactly this shape -- unusable to the stage it was prepared for.
//!
//! So a target is not a sentence but a *typing segment*: the maximal run of Han
//! characters between two things the writer typed with some other key. That is
//! what a person converts in one go. They type `今天`, press comma, then type
//! `天气不错` against a screen that already reads `今天，`. Splitting a unit on
//! every non-Han character reproduces those two conversions, and the text before
//! the run becomes the context the second one is converted against -- which is
//! why nothing is lost by splitting: a run too short to be a target still shows
//! up in the context of every run after it.

use crate::filter::{MAX_CHARACTERS, MIN_CHARACTERS};
use ime_g2p::text::is_han;

/// One maximal run of Han characters inside a unit, and what precedes it there.
///
/// The two halves are borrowed from the same unit and are adjacent in it, so a
/// segment carries no allocation and cannot disagree with itself about where the
/// run begins.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TypingSegment<'a> {
    text: &'a str,
    prefix: &'a str,
}

impl<'a> TypingSegment<'a> {
    /// The run itself: every character Han, which is what makes it a target.
    #[must_use]
    pub fn text(&self) -> &'a str {
        self.text
    }

    /// Everything the unit holds before the run, punctuation and all.
    ///
    /// This is the tail of what the writer had already committed when they
    /// started typing the run, so it belongs on the end of the sample's context.
    #[must_use]
    pub fn prefix(&self) -> &'a str {
        self.prefix
    }
}

/// Every Han run of one unit, in the order they were typed.
///
/// Runs of *every* length are yielded, including the one-character fragments no
/// sample can be built from, because the run summary counts why each candidate
/// was dropped and cannot count what it never saw.
#[derive(Clone, Debug)]
pub struct TypingSegments<'a> {
    unit: &'a str,
    position: usize,
}

impl<'a> Iterator for TypingSegments<'a> {
    type Item = TypingSegment<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut start: Option<usize> = None;
        for (offset, character) in self.unit[self.position..].char_indices() {
            let at = self.position + offset;
            if is_han(character) {
                start.get_or_insert(at);
                continue;
            }
            if let Some(begin) = start.take() {
                self.position = at;
                return Some(TypingSegment {
                    text: &self.unit[begin..at],
                    prefix: &self.unit[..begin],
                });
            }
        }
        self.position = self.unit.len();
        let begin = start?;
        Some(TypingSegment {
            text: &self.unit[begin..],
            prefix: &self.unit[..begin],
        })
    }
}

/// Split `unit` on every non-Han character into the runs a person types in one go.
#[must_use]
pub fn typing_segments(unit: &str) -> TypingSegments<'_> {
    TypingSegments { unit, position: 0 }
}

/// Whether `text` has the shape a prepared target has: all Han, and within bounds.
///
/// The lexicon and duplicate rules are not applied here -- they need a lexicon
/// and a run's worth of history. This answers the weaker question that does not:
/// *could* this string ever be emitted as a target? A held-out sentence that
/// fails it is one the export can never find, and asking the export to find it
/// anyway would fail a run over a sentence that is correctly absent.
#[must_use]
pub fn is_typable_target(text: &str) -> bool {
    let mut length = 0_usize;
    for character in text.chars() {
        if !is_han(character) {
            return false;
        }
        length += 1;
    }
    (MIN_CHARACTERS..=MAX_CHARACTERS).contains(&length)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runs(unit: &str) -> Vec<(&str, &str)> {
        typing_segments(unit)
            .map(|segment| (segment.text(), segment.prefix()))
            .collect()
    }

    #[test]
    fn punctuation_inside_a_sentence_splits_it_into_two_typing_runs() {
        assert_eq!(
            runs("今天，天气不错"),
            vec![("今天", ""), ("天气不错", "今天，")]
        );
    }

    #[test]
    fn a_run_carries_every_earlier_character_of_its_unit_as_its_prefix() {
        assert_eq!(
            runs("使用 Python 写的程序"),
            vec![("使用", ""), ("写的程序", "使用 Python ")]
        );
    }

    #[test]
    fn a_unit_that_is_wholly_han_is_one_run_with_no_prefix() {
        assert_eq!(runs("今天天气不错"), vec![("今天天气不错", "")]);
    }

    #[test]
    fn a_unit_with_no_han_at_all_yields_nothing() {
        assert!(runs("hello, world!").is_empty());
        assert!(runs("").is_empty());
    }

    #[test]
    fn a_terminal_delimiter_ends_the_run_rather_than_joining_it() {
        assert_eq!(runs("你说呢？"), vec![("你说呢", "")]);
    }

    #[test]
    fn only_a_han_run_within_the_length_bounds_could_ever_be_a_target() {
        assert!(is_typable_target("今天天气不错"));
        assert!(!is_typable_target("今天天"));
        assert!(!is_typable_target("今天，天气不错"));
        assert!(!is_typable_target("使用Python写程序"));
        assert!(is_typable_target(&"中".repeat(MAX_CHARACTERS)));
        assert!(!is_typable_target(&"中".repeat(MAX_CHARACTERS + 1)));
        assert!(!is_typable_target(""));
    }
}
