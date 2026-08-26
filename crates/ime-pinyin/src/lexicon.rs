//! The character lexicon: which characters a syllable can be written as, and the
//! per-position masks that constraint implies.

use crate::syllable::{SyllableId, SyllableRange, SyllableTable};
use thiserror::Error;

/// The generated character table: `<char>\t<py1>,<py2>,...` per line, sorted by
/// character. Produced by `mlime gen-pinyin-tables`; never edited by hand.
const CHAR_PINYIN: &str = include_str!("../data/char_pinyin.tsv");

/// An index into a [`Lexicon`].
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct CharId(u32);

impl CharId {
    /// This character's position in the lexicon it came from.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Ways the embedded character table can fail to agree with the syllable table.
#[derive(Debug, Error)]
pub enum LexiconError {
    /// A line did not have the expected `<char>\t<readings>` shape.
    #[error("line {line}: expected `<char>\\t<readings>`, got {raw:?}")]
    Malformed {
        /// One-based line number in the embedded table.
        line: usize,
        /// The offending line.
        raw: String,
    },
    /// The first field held something other than a single character.
    #[error("line {line}: expected exactly one character, got {field:?}")]
    NotOneChar {
        /// One-based line number in the embedded table.
        line: usize,
        /// The offending field.
        field: String,
    },
    /// A reading is absent from the syllable inventory, so the two generated
    /// tables were not produced by the same run.
    #[error("line {line}: reading {reading:?} is not in the syllable inventory")]
    UnknownReading {
        /// One-based line number in the embedded table.
        line: usize,
        /// The offending reading.
        reading: String,
    },
    /// The table is larger than the `u32` index space.
    #[error("character table has {count} entries, which exceeds the u32 index space")]
    TooLarge {
        /// Number of entries found.
        count: usize,
    },
    /// The table is not strictly sorted by character, which `id_of`'s binary
    /// search relies on.
    #[error("character table is not strictly sorted: {first:?} precedes {second:?}")]
    Unsorted {
        /// The earlier character.
        first: char,
        /// The character that should have sorted after it.
        second: char,
    },
}

/// Characters, their readings, and the inverse map from reading to characters.
///
/// Both directions are stored as CSR arrays rather than maps of vectors: the data
/// is built once and read constantly, and the flat form keeps a mask lookup to a
/// slice index.
pub struct Lexicon {
    chars: Box<[char]>,
    reading_offsets: Box<[u32]>,
    reading_ids: Box<[SyllableId]>,
    homophone_offsets: Box<[u32]>,
    homophone_ids: Box<[CharId]>,
}

impl Lexicon {
    /// Parse the embedded character table against *table*.
    ///
    /// # Errors
    ///
    /// If the two generated tables disagree, or the character table is malformed.
    pub fn load(table: &SyllableTable) -> Result<Self, LexiconError> {
        let mut chars = Vec::new();
        let mut reading_offsets = vec![0u32];
        let mut reading_ids = Vec::new();

        for (index, raw) in CHAR_PINYIN.lines().enumerate() {
            let line = index + 1;
            let (ch_field, readings) =
                raw.split_once('\t')
                    .ok_or_else(|| LexiconError::Malformed {
                        line,
                        raw: raw.to_owned(),
                    })?;
            let mut ch_iter = ch_field.chars();
            let (Some(ch), None) = (ch_iter.next(), ch_iter.next()) else {
                return Err(LexiconError::NotOneChar {
                    line,
                    field: ch_field.to_owned(),
                });
            };
            for reading in readings.split(',') {
                let id = table
                    .exact(reading)
                    .ok_or_else(|| LexiconError::UnknownReading {
                        line,
                        reading: reading.to_owned(),
                    })?;
                reading_ids.push(id);
            }
            chars.push(ch);
            let offset = u32::try_from(reading_ids.len()).map_err(|_| LexiconError::TooLarge {
                count: reading_ids.len(),
            })?;
            reading_offsets.push(offset);
        }

        let char_count = u32::try_from(chars.len())
            .map_err(|_| LexiconError::TooLarge { count: chars.len() })?;
        if let Some(pair) = chars.windows(2).find(|pair| pair[0] >= pair[1]) {
            return Err(LexiconError::Unsorted {
                first: pair[0],
                second: pair[1],
            });
        }

        // Invert into `syllable -> characters`, counting first so the CSR arrays
        // are allocated exactly once.
        let mut counts = vec![0u32; table.len()];
        for id in &reading_ids {
            counts[id.index()] += 1;
        }
        let mut homophone_offsets = Vec::with_capacity(table.len() + 1);
        let mut running = 0u32;
        for count in &counts {
            homophone_offsets.push(running);
            running += count;
        }
        homophone_offsets.push(running);
        let mut cursor = homophone_offsets.clone();
        let mut homophone_ids = vec![CharId(0); reading_ids.len()];
        for char_index in 0..char_count {
            let ch = CharId(char_index);
            let start = reading_offsets[ch.index()] as usize;
            let end = reading_offsets[ch.index() + 1] as usize;
            for id in &reading_ids[start..end] {
                let slot = &mut cursor[id.index()];
                homophone_ids[*slot as usize] = ch;
                *slot += 1;
            }
        }

        Ok(Self {
            chars: chars.into_boxed_slice(),
            reading_offsets: reading_offsets.into_boxed_slice(),
            reading_ids: reading_ids.into_boxed_slice(),
            homophone_offsets: homophone_offsets.into_boxed_slice(),
            homophone_ids: homophone_ids.into_boxed_slice(),
        })
    }

    /// Number of characters in the lexicon.
    #[must_use]
    pub fn len(&self) -> usize {
        self.chars.len()
    }

    /// Whether the lexicon is empty. Never true for a loaded lexicon.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.chars.is_empty()
    }

    /// The character *id* stands for.
    ///
    /// # Panics
    ///
    /// If *id* did not come from this lexicon.
    #[must_use]
    pub fn character(&self, id: CharId) -> char {
        self.chars[id.index()]
    }

    /// Look a character up by its glyph.
    #[must_use]
    pub fn id_of(&self, ch: char) -> Option<CharId> {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "load() proved the length fits u32"
        )]
        self.chars
            .binary_search(&ch)
            .ok()
            .map(|index| CharId(index as u32))
    }

    /// Every reading of *id*, in the order `pypinyin` ranks them.
    ///
    /// # Panics
    ///
    /// If *id* did not come from this lexicon.
    #[must_use]
    pub fn readings(&self, id: CharId) -> &[SyllableId] {
        let start = self.reading_offsets[id.index()] as usize;
        let end = self.reading_offsets[id.index() + 1] as usize;
        &self.reading_ids[start..end]
    }

    /// Every character that can be read as *syllable*, in lexicon order.
    ///
    /// # Panics
    ///
    /// If *syllable* did not come from the table this lexicon was loaded against.
    #[must_use]
    pub fn homophones(&self, syllable: SyllableId) -> &[CharId] {
        let start = self.homophone_offsets[syllable.index()] as usize;
        let end = self.homophone_offsets[syllable.index() + 1] as usize;
        &self.homophone_ids[start..end]
    }

    /// Collect the characters reachable from *range* into *out*, sorted and
    /// deduplicated.
    ///
    /// This is the per-position mask: the hard constraint that makes it
    /// impossible for a model to emit a character the user did not type. *out* is
    /// cleared first and is meant to be reused across keystrokes -- a full-pinyin
    /// range yields tens of characters, but an abbreviation range yields
    /// thousands, and that allocation should not recur.
    pub fn mask_into(&self, range: SyllableRange, out: &mut Vec<CharId>) {
        out.clear();
        for syllable in range {
            out.extend_from_slice(self.homophones(syllable));
        }
        out.sort_unstable();
        out.dedup();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (SyllableTable, Lexicon) {
        let table = SyllableTable::load();
        let lexicon = Lexicon::load(&table).expect("generated tables must agree");
        (table, lexicon)
    }

    #[test]
    fn readings_round_trip_through_the_inverse_map() {
        let (table, lexicon) = fixture();
        let zhong = lexicon.id_of('中').expect("中 is in the lexicon");
        let readings: Vec<_> = lexicon
            .readings(zhong)
            .iter()
            .map(|s| table.spelling(*s))
            .collect();
        assert!(readings.contains(&"zhong"), "got {readings:?}");
        for reading in lexicon.readings(zhong) {
            assert!(
                lexicon.homophones(*reading).contains(&zhong),
                "inverse map lost 中 under {:?}",
                table.spelling(*reading)
            );
        }
    }

    #[test]
    fn polyphones_carry_every_reading() {
        let (table, lexicon) = fixture();
        for (ch, expected) in [('重', ["zhong", "chong"]), ('行', ["xing", "hang"])] {
            let id = lexicon.id_of(ch).expect("polyphone is in the lexicon");
            let readings: Vec<_> = lexicon
                .readings(id)
                .iter()
                .map(|s| table.spelling(*s))
                .collect();
            for want in expected {
                assert!(
                    readings.contains(&want),
                    "{ch} lost reading {want}: {readings:?}"
                );
            }
        }
    }

    #[test]
    fn a_full_syllable_mask_is_far_smaller_than_an_abbreviation_mask() {
        let (table, lexicon) = fixture();
        let mut full = Vec::new();
        let mut abbrev = Vec::new();
        lexicon.mask_into(table.exact_range("zhong").expect("zhong"), &mut full);
        lexicon.mask_into(table.prefix_range("z"), &mut abbrev);
        assert!(!full.is_empty());
        assert!(
            abbrev.len() > full.len() * 10,
            "abbreviation should blow the candidate set up: {} vs {}",
            abbrev.len(),
            full.len()
        );
        assert!(
            full.iter().all(|id| abbrev.contains(id)),
            "`z` must subsume `zhong`"
        );
    }

    #[test]
    fn masks_are_sorted_and_deduplicated() {
        let (table, lexicon) = fixture();
        let mut mask = Vec::new();
        lexicon.mask_into(table.prefix_range("zh"), &mut mask);
        assert!(
            mask.windows(2).all(|w| w[0] < w[1]),
            "mask must be sorted and unique"
        );
    }

    #[test]
    fn mask_into_reuses_its_buffer() {
        let (table, lexicon) = fixture();
        let mut mask = Vec::new();
        lexicon.mask_into(table.prefix_range("z"), &mut mask);
        let big = mask.len();
        lexicon.mask_into(table.exact_range("ni").expect("ni"), &mut mask);
        assert!(mask.len() < big, "buffer must be cleared, not appended to");
    }
}
