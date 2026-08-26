//! The toneless syllable inventory and the range algebra built on top of it.

use std::fmt;

/// The generated inventory: one typing-form syllable per line, lexicographically
/// sorted. Produced by `mlime gen-pinyin-tables`; never edited by hand.
const SYLLABLES: &str = include_str!("../data/syllables.txt");

/// Longest spelling in the inventory, in bytes. Asserted against the data at load.
pub const MAX_SYLLABLE_LEN: usize = 6;

/// The three multi-letter initials. Every other initial is a single letter, and a
/// bare vowel has none, so a syllable abbreviation is one letter unless it is one
/// of these.
const MULTI_LETTER_INITIALS: [&str; 3] = ["zh", "ch", "sh"];

/// An index into a [`SyllableTable`].
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct SyllableId(u16);

impl SyllableId {
    /// This syllable's position in the table it came from.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// A contiguous, sorted run of syllables.
///
/// The inventory is stored in lexicographic order, so the syllables sharing any
/// given prefix always occupy a contiguous span. That one fact collapses every
/// match an IME needs into a single representation: an exact syllable is a run of
/// length one, an initial-only abbreviation (`z` -> `za..zuo`, `zha..zhuang`) is
/// the run under that letter, and a half-typed syllable is the run under what has
/// been typed so far. No set type, no allocation, and the ambiguity of a segment
/// is just its length.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct SyllableRange {
    start: u16,
    end: u16,
}

impl SyllableRange {
    /// Number of syllables in the run.
    #[must_use]
    pub const fn len(self) -> usize {
        (self.end - self.start) as usize
    }

    /// Whether the run matches nothing.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.start == self.end
    }

    /// The syllables in the run, in lexicographic order.
    pub fn iter(self) -> impl ExactSizeIterator<Item = SyllableId> + Clone {
        (self.start..self.end).map(SyllableId)
    }

    /// Whether *id* falls inside the run.
    #[must_use]
    pub const fn contains(self, id: SyllableId) -> bool {
        self.start <= id.0 && id.0 < self.end
    }
}

impl IntoIterator for SyllableRange {
    type Item = SyllableId;
    type IntoIter = std::iter::Map<std::ops::Range<u16>, fn(u16) -> SyllableId>;

    fn into_iter(self) -> Self::IntoIter {
        (self.start..self.end).map(SyllableId)
    }
}

/// The syllable inventory, owned rather than global so that a process can hold
/// more than one (a test fixture, a fuzzy-expanded variant) without hidden state.
pub struct SyllableTable {
    spellings: Box<[&'static str]>,
}

impl SyllableTable {
    /// Parse the embedded inventory.
    ///
    /// # Panics
    ///
    /// If the generated data violates an invariant the rest of the crate relies
    /// on -- sortedness, the `[a-z]` alphabet, the length bound, or the `u16`
    /// index bound. These are build-data errors, not runtime conditions.
    #[must_use]
    pub fn load() -> Self {
        let spellings: Box<[&'static str]> = SYLLABLES.lines().collect();
        assert!(!spellings.is_empty(), "syllable inventory is empty");
        assert!(
            u16::try_from(spellings.len()).is_ok(),
            "syllable inventory exceeds the u16 index space: {}",
            spellings.len()
        );
        for pair in spellings.windows(2) {
            assert!(
                pair[0] < pair[1],
                "syllable inventory is not strictly sorted: {:?} then {:?}",
                pair[0],
                pair[1]
            );
        }
        for &s in &spellings {
            assert!(
                s.len() <= MAX_SYLLABLE_LEN,
                "syllable {s:?} is longer than MAX_SYLLABLE_LEN ({MAX_SYLLABLE_LEN})"
            );
            assert!(
                s.bytes().all(|b| b.is_ascii_lowercase()),
                "syllable {s:?} is outside the [a-z] typing alphabet"
            );
        }
        Self { spellings }
    }

    /// Number of syllables in the inventory.
    #[must_use]
    pub fn len(&self) -> usize {
        self.spellings.len()
    }

    /// Whether the inventory is empty. Never true for a loaded table.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.spellings.is_empty()
    }

    /// The spelling of *id*.
    ///
    /// # Panics
    ///
    /// If *id* did not come from this table.
    #[must_use]
    pub fn spelling(&self, id: SyllableId) -> &'static str {
        self.spellings[id.index()]
    }

    /// Every syllable beginning with *prefix*, as a range.
    ///
    /// An empty prefix yields the whole inventory.
    #[must_use]
    pub fn prefix_range(&self, prefix: &str) -> SyllableRange {
        let start = self.spellings.partition_point(|s| *s < prefix);
        let len = self.spellings[start..].partition_point(|s| s.starts_with(prefix));
        // `start + len <= spellings.len()`, which `load` proved fits in u16.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "bounded by load()'s u16 assertion"
        )]
        SyllableRange {
            start: start as u16,
            end: (start + len) as u16,
        }
    }

    /// The syllable spelled exactly *spelling*, if the inventory has one.
    #[must_use]
    pub fn exact(&self, spelling: &str) -> Option<SyllableId> {
        let range = self.prefix_range(spelling);
        let first = SyllableId(range.start);
        (!range.is_empty() && self.spelling(first) == spelling).then_some(first)
    }

    /// The syllable spelled exactly *spelling*, as a single-element range.
    #[must_use]
    pub fn exact_range(&self, spelling: &str) -> Option<SyllableRange> {
        self.exact(spelling).map(|id| SyllableRange {
            start: id.0,
            end: id.0 + 1,
        })
    }

    /// Whether *span* is a spellable syllable abbreviation: one letter, or one of
    /// the three two-letter initials.
    #[must_use]
    pub fn is_abbreviation(span: &str) -> bool {
        span.len() == 1 && span.as_bytes()[0].is_ascii_lowercase()
            || MULTI_LETTER_INITIALS.contains(&span)
    }
}

impl fmt::Debug for SyllableTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SyllableTable")
            .field("len", &self.spellings.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inventory_loads_and_validates() {
        let table = SyllableTable::load();
        assert!(
            table.len() > 400,
            "expected a full inventory, got {}",
            table.len()
        );
    }

    #[test]
    fn exact_match_is_a_singleton_range() {
        let table = SyllableTable::load();
        let range = table.exact_range("zhong").expect("zhong is a syllable");
        assert_eq!(range.len(), 1);
        assert_eq!(table.spelling(SyllableId(range.start)), "zhong");
        assert!(table.exact("zhonk").is_none());
    }

    #[test]
    fn a_syllable_that_is_also_a_prefix_stays_distinguishable() {
        // `zhu` is both a syllable and a prefix of `zhuang`; exact and prefix
        // lookups must not collapse.
        let table = SyllableTable::load();
        assert_eq!(
            table.exact_range("zhu").expect("zhu is a syllable").len(),
            1
        );
        assert!(table.prefix_range("zhu").len() > 1);
    }

    #[test]
    fn abbreviation_range_covers_both_z_and_zh() {
        let table = SyllableTable::load();
        let z = table.prefix_range("z");
        let zhang = table.exact("zhang").expect("zhang is a syllable");
        let za = table.exact("za").expect("za is a syllable");
        assert!(z.contains(zhang), "simplified `z` must reach zh- syllables");
        assert!(z.contains(za));
        assert!(table.prefix_range("zh").len() < z.len());
    }

    #[test]
    fn umlaut_is_stored_in_typing_form() {
        let table = SyllableTable::load();
        assert!(
            table.exact("lv").is_some(),
            "the keyboard produces `lv`, not `lü`"
        );
        assert!(table.exact("nv").is_some());
    }

    #[test]
    fn abbreviation_shape_is_recognised() {
        assert!(SyllableTable::is_abbreviation("z"));
        assert!(SyllableTable::is_abbreviation("zh"));
        assert!(!SyllableTable::is_abbreviation("zg"));
        assert!(!SyllableTable::is_abbreviation("zho"));
        assert!(!SyllableTable::is_abbreviation(""));
    }

    #[test]
    fn empty_prefix_is_the_whole_inventory() {
        let table = SyllableTable::load();
        assert_eq!(table.prefix_range("").len(), table.len());
        assert!(table.prefix_range("qq").is_empty());
    }
}
