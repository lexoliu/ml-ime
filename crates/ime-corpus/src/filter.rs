//! The gate every candidate target passes through, and the tally of why it did not.
//!
//! The rules are the Python pipeline's, unchanged, because the two write into the
//! same sample schema: a target between four and sixty-four characters, at least
//! nine tenths of its visible characters Han, every Han character one
//! `ime-pinyin` can produce readings for, and not a repeat of one already kept.
//!
//! Deduplication is per source rather than global. Two corpora may legitimately
//! contain the same sentence, and dropping the second occurrence would silently
//! skew the mix towards whichever source happened to be prepared first.

use crate::text::{DEDUPE_DIGEST_BYTES, dedupe_digest, han_ratio};
use ime_g2p::text::is_han;
use ime_pinyin::Lexicon;
use std::collections::HashSet;

/// The shortest target worth keeping, in characters.
pub const MIN_CHARACTERS: usize = 4;

/// The longest target worth keeping, in characters.
pub const MAX_CHARACTERS: usize = 64;

/// The least Han a target may be, as a fraction of its visible characters.
pub const MIN_HAN_RATIO: f64 = 0.9;

/// Why candidate targets were kept or dropped, for the run summary.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct FilterCounts {
    /// Targets that passed every rule.
    pub kept: usize,
    /// Targets shorter than [`MIN_CHARACTERS`].
    pub too_short: usize,
    /// Targets longer than [`MAX_CHARACTERS`].
    pub too_long: usize,
    /// Targets whose Han ratio fell below [`MIN_HAN_RATIO`].
    pub not_chinese_enough: usize,
    /// Targets holding a Han character the pinyin lexicon cannot read.
    pub unknown_character: usize,
    /// Targets identical to one already kept from the same source.
    pub duplicate: usize,
}

impl FilterCounts {
    /// Every candidate the filter saw.
    #[must_use]
    pub fn considered(&self) -> usize {
        self.kept
            + self.too_short
            + self.too_long
            + self.not_chinese_enough
            + self.unknown_character
            + self.duplicate
    }
}

/// The verdict on one candidate, so a caller can tally it without re-deriving it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Verdict {
    /// The target is usable.
    Kept,
    /// Shorter than [`MIN_CHARACTERS`].
    TooShort,
    /// Longer than [`MAX_CHARACTERS`].
    TooLong,
    /// Below [`MIN_HAN_RATIO`].
    NotChineseEnough,
    /// Holds a Han character the pinyin lexicon cannot read.
    UnknownCharacter,
    /// Already kept from this source.
    Duplicate,
}

/// Length, script, lexicon-coverage and duplicate gate on one source's targets.
///
/// Holding the seen-digest set here is what makes deduplication per-source, and
/// it is why one filter is built per source and carried across every shard of it
/// rather than rebuilt per shard.
pub struct SampleFilter<'a> {
    lexicon: &'a Lexicon,
    counts: FilterCounts,
    seen: HashSet<[u8; DEDUPE_DIGEST_BYTES]>,
}

impl std::fmt::Debug for SampleFilter<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SampleFilter")
            .field("counts", &self.counts)
            .field("seen", &self.seen.len())
            .finish_non_exhaustive()
    }
}

impl<'a> SampleFilter<'a> {
    /// A filter gating on `lexicon`'s character coverage.
    #[must_use]
    pub fn new(lexicon: &'a Lexicon) -> Self {
        Self {
            lexicon,
            counts: FilterCounts::default(),
            seen: HashSet::new(),
        }
    }

    /// The verdicts recorded so far.
    #[must_use]
    pub fn counts(&self) -> FilterCounts {
        self.counts
    }

    /// Whether `text` is usable as a target, recording the verdict's reason.
    pub fn accepts(&mut self, text: &str) -> bool {
        let verdict = self.judge(text);
        match verdict {
            Verdict::Kept => self.counts.kept += 1,
            Verdict::TooShort => self.counts.too_short += 1,
            Verdict::TooLong => self.counts.too_long += 1,
            Verdict::NotChineseEnough => self.counts.not_chinese_enough += 1,
            Verdict::UnknownCharacter => self.counts.unknown_character += 1,
            Verdict::Duplicate => self.counts.duplicate += 1,
        }
        verdict == Verdict::Kept
    }

    fn judge(&mut self, text: &str) -> Verdict {
        let length = text.chars().count();
        if length < MIN_CHARACTERS {
            return Verdict::TooShort;
        }
        if length > MAX_CHARACTERS {
            return Verdict::TooLong;
        }
        if han_ratio(text) < MIN_HAN_RATIO {
            return Verdict::NotChineseEnough;
        }
        if text
            .chars()
            .any(|character| is_han(character) && self.lexicon.id_of(character).is_none())
        {
            return Verdict::UnknownCharacter;
        }
        if !self.seen.insert(dedupe_digest(text)) {
            return Verdict::Duplicate;
        }
        Verdict::Kept
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ime_pinyin::SyllableTable;

    fn lexicon() -> Lexicon {
        let table = SyllableTable::load();
        Lexicon::load(&table).expect("the generated pinyin tables agree")
    }

    #[test]
    fn a_target_is_kept_once_and_counted_as_a_duplicate_thereafter() {
        let lexicon = lexicon();
        let mut filter = SampleFilter::new(&lexicon);
        assert!(filter.accepts("今天天气不错"));
        assert!(!filter.accepts("今天天气不错"));
        assert_eq!(filter.counts().kept, 1);
        assert_eq!(filter.counts().duplicate, 1);
        assert_eq!(filter.counts().considered(), 2);
    }

    #[test]
    fn the_length_bounds_are_inclusive_at_both_ends() {
        let lexicon = lexicon();
        let mut filter = SampleFilter::new(&lexicon);
        assert!(!filter.accepts("今天天"));
        assert!(filter.accepts("今天天气"));
        assert!(filter.accepts(&"中".repeat(MAX_CHARACTERS)));
        assert!(!filter.accepts(&"国".repeat(MAX_CHARACTERS + 1)));
        assert_eq!(filter.counts().too_short, 1);
        assert_eq!(filter.counts().too_long, 1);
        assert_eq!(filter.counts().kept, 2);
    }

    #[test]
    fn a_target_that_is_mostly_latin_is_not_chinese_enough() {
        let lexicon = lexicon();
        let mut filter = SampleFilter::new(&lexicon);
        assert!(!filter.accepts("hello world 你好"));
        assert_eq!(filter.counts().not_chinese_enough, 1);
        assert!(filter.accepts("这是一个很好的句子"));
    }

    #[test]
    fn a_han_character_the_lexicon_cannot_read_takes_the_whole_target_with_it() {
        let lexicon = lexicon();
        let mut filter = SampleFilter::new(&lexicon);
        let rare = '\u{2A6B2}';
        assert!(is_han(rare));
        assert!(lexicon.id_of(rare).is_none());
        assert!(!filter.accepts(&format!("这个字是{rare}啊")));
        assert_eq!(filter.counts().unknown_character, 1);
    }
}
