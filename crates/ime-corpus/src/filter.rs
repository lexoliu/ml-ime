//! The gate every candidate target passes through, and the tally of why it did not.
//!
//! The rules are what the decoder can actually be trained on: a run of between
//! four and sixty-four characters, every Han character one `ime-pinyin` can
//! produce readings for, and not a repeat of one already kept. The script rule
//! that used to sit among them is gone, because [`crate::segment`] now hands this
//! gate runs that are Han by construction -- a ratio test over them could only
//! ever return one.
//!
//! Deduplication is per source rather than global. Two corpora may legitimately
//! contain the same sentence, and dropping the second occurrence would silently
//! skew the mix towards whichever source happened to be prepared first.

use crate::text::{DEDUPE_DIGEST_BYTES, dedupe_digest};
use ime_g2p::text::is_han;
use ime_pinyin::Lexicon;
use std::collections::HashSet;

/// The shortest target worth keeping, in characters.
pub const MIN_CHARACTERS: usize = 4;

/// The longest target worth keeping, in characters.
pub const MAX_CHARACTERS: usize = 64;

/// The least Han a freely generated sentence may be, as a fraction of its
/// visible characters.
///
/// The corpus filter no longer consults it -- a typing segment is wholly Han --
/// but `ime-synth` still judges whole model-written sentences, which are not, and
/// it is the same threshold there.
pub const MIN_HAN_RATIO: f64 = 0.9;

/// Why candidate targets were kept or dropped, for the run summary.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct FilterCounts {
    /// Targets that passed every rule.
    pub kept: usize,
    /// Runs shorter than [`MIN_CHARACTERS`], which are context rather than targets.
    pub too_short_run: usize,
    /// Runs longer than [`MAX_CHARACTERS`], which are dropped whole rather than cut.
    pub too_long_run: usize,
    /// Targets holding a Han character the pinyin lexicon cannot read.
    pub unknown_character: usize,
    /// Targets identical to one already kept from the same source.
    pub duplicate: usize,
}

impl FilterCounts {
    /// Every candidate the filter saw.
    #[must_use]
    pub fn considered(&self) -> usize {
        self.kept + self.too_short_run + self.too_long_run + self.unknown_character + self.duplicate
    }
}

/// The verdict on one candidate, so a caller can tally it without re-deriving it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Verdict {
    /// The target is usable.
    Kept,
    /// Shorter than [`MIN_CHARACTERS`].
    TooShortRun,
    /// Longer than [`MAX_CHARACTERS`].
    TooLongRun,
    /// Holds a Han character the pinyin lexicon cannot read.
    UnknownCharacter,
    /// Already kept from this source.
    Duplicate,
}

/// Length, lexicon-coverage and duplicate gate on one source's targets.
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
    ///
    /// Preparation passes a [`crate::segment::TypingSegment`]'s text, which is
    /// wholly Han; `ime-synth` passes a whole generated sentence, which has been
    /// judged Chinese enough by then. Neither shape is this gate's business.
    pub fn accepts(&mut self, text: &str) -> bool {
        let verdict = self.judge(text);
        match verdict {
            Verdict::Kept => self.counts.kept += 1,
            Verdict::TooShortRun => self.counts.too_short_run += 1,
            Verdict::TooLongRun => self.counts.too_long_run += 1,
            Verdict::UnknownCharacter => self.counts.unknown_character += 1,
            Verdict::Duplicate => self.counts.duplicate += 1,
        }
        verdict == Verdict::Kept
    }

    fn judge(&mut self, text: &str) -> Verdict {
        let length = text.chars().count();
        if length < MIN_CHARACTERS {
            return Verdict::TooShortRun;
        }
        if length > MAX_CHARACTERS {
            return Verdict::TooLongRun;
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
        assert_eq!(filter.counts().too_short_run, 1);
        assert_eq!(filter.counts().too_long_run, 1);
        assert_eq!(filter.counts().kept, 2);
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
