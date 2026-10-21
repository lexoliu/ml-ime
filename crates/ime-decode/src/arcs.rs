//! Word arcs: whole lexicon entries proposed to the beam as units.
//!
//! The beam proposes one character at a time, and nothing it consults scores
//! an idiom as a whole against the initials it was typed with: route A scores
//! each position alone, and the trigram and the character LM price the first
//! character before the later initials pin the rest. Four initials therefore
//! rarely assemble a four-character idiom over more frequent readings. A
//! [`WordLexicon`] closes that gap by proposal, not by a new scorer: every
//! entry whose characters the keystrokes at a reading's positions can spell
//! becomes a [`WordArc`] on that reading, and the beam starts a lineage with
//! the entry's characters fixed, keeps it alive until it lands and scores it
//! with the same emission and transition models as every other lineage.
//!
//! An arc is proposed from the keystrokes, not from the segment's mask. A
//! reading that takes `m` as the syllable `m` admits only 呣 there, yet the
//! typed letter is still 马's initial, so an idiom with 马 at that position is
//! proposed on that reading too. The character is then outside the position's
//! mask, which makes it part of what the emission and the transitions are
//! asked for there ([`CandidatePath::asked`]), and the arc carries the
//! segmentation cost that reading the segment as an abbreviation costs over
//! reading it as the syllable.
//!
//! [`CandidatePath::asked`]: crate::CandidatePath::asked

use crate::emissions::Emittable;
use hashbrown::HashMap;
use ime_pinyin::{CharId, Lexicon, Segment, SegmentOptions, Segmentation, SyllableTable};
use thiserror::Error;

/// Why a word lexicon could not be read.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum WordLexiconError {
    /// A line is not `word<TAB>category`.
    #[error("line {line}: {reason}")]
    Malformed {
        /// The one-based line number.
        line: usize,
        /// What is wrong with it.
        reason: &'static str,
    },
    /// No entry of the asked categories survived.
    #[error("the lexicon holds no usable entry in the categories {categories:?}")]
    Empty {
        /// The categories that were asked for.
        categories: Vec<String>,
    },
}

/// What [`WordLexicon::parse`] kept and why it left the rest out.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WordLexiconStats {
    /// Entries kept as arcs.
    pub kept: usize,
    /// Entries of a category that was not asked for.
    pub other_category: usize,
    /// Entries holding a character the character lexicon does not hold, or
    /// one it holds with no reading.
    pub unknown: usize,
    /// Entries holding a character route A has no output row for (#214).
    pub unemittable: usize,
    /// Entries listed again under an asked category.
    pub duplicate: usize,
}

/// A curated lexicon of words the beam proposes whole.
///
/// Every entry is at least two characters long, each a character the
/// character lexicon reads and route A can emit; [`WordLexicon::parse`] is the
/// only way to build one, and it holds every entry to that.
#[derive(Clone, Debug)]
pub struct WordLexicon {
    /// The entries, each at least two characters.
    words: Vec<Box<[CharId]>>,
    /// Entry ids by the first letters of their characters' readings, one
    /// letter per character, and an empty list under every proper prefix of
    /// such a key -- so a lookup that misses can stop extending.
    index: HashMap<Box<[u8]>, Vec<u32>>,
}

impl WordLexicon {
    /// Read `word<TAB>category` lines (what `mlime lexicon knowledge`
    /// writes), keeping the entries of *categories* that the lexicon reads
    /// and route A can emit.
    ///
    /// # Errors
    ///
    /// If a line is not two tab-separated fields, a word of an asked category
    /// is shorter than two characters, or no entry survives.
    ///
    /// # Panics
    ///
    /// If the lexicon holds more than `u32::MAX` entries.
    pub fn parse(
        source: &str,
        categories: &[&str],
        lexicon: &Lexicon,
        table: &SyllableTable,
        emittable: &Emittable,
    ) -> Result<(Self, WordLexiconStats), WordLexiconError> {
        let mut stats = WordLexiconStats::default();
        let mut words: Vec<Box<[CharId]>> = Vec::new();
        let mut seen: HashMap<Box<[CharId]>, ()> = HashMap::new();
        for (index, line) in source.lines().enumerate() {
            if line.is_empty() {
                continue;
            }
            let (word, category) = line.split_once('\t').ok_or(WordLexiconError::Malformed {
                line: index + 1,
                reason: "no tab between the word and its category",
            })?;
            if category.contains('\t') {
                return Err(WordLexiconError::Malformed {
                    line: index + 1,
                    reason: "more than two fields",
                });
            }
            if !categories.contains(&category) {
                stats.other_category += 1;
                continue;
            }
            if word.chars().nth(1).is_none() {
                return Err(WordLexiconError::Malformed {
                    line: index + 1,
                    reason: "a word shorter than two characters",
                });
            }
            let ids: Option<Box<[CharId]>> = word
                .chars()
                .map(|ch| {
                    lexicon
                        .id_of(ch)
                        .filter(|&id| !lexicon.readings(id).is_empty())
                })
                .collect();
            let Some(ids) = ids else {
                stats.unknown += 1;
                continue;
            };
            if !ids.iter().all(|&id| emittable.contains(id)) {
                stats.unemittable += 1;
                continue;
            }
            if seen.insert(ids.clone(), ()).is_some() {
                stats.duplicate += 1;
                continue;
            }
            words.push(ids);
        }
        if words.is_empty() {
            return Err(WordLexiconError::Empty {
                categories: categories.iter().map(|&c| c.to_owned()).collect(),
            });
        }
        stats.kept = words.len();
        let mut index: HashMap<Box<[u8]>, Vec<u32>> = HashMap::new();
        for (id, word) in words.iter().enumerate() {
            let id = u32::try_from(id).expect("a word lexicon holds fewer than 2^32 entries");
            for key in initials(word, lexicon, table) {
                for prefix in 1..key.len() {
                    index.entry(key[..prefix].into()).or_default();
                }
                let entries = index.entry(key.into_boxed_slice()).or_default();
                if entries.last() != Some(&id) {
                    entries.push(id);
                }
            }
        }
        Ok((Self { words, index }, stats))
    }

    /// How many entries the lexicon holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.words.len()
    }

    /// Whether the lexicon is empty. Never true for a parsed one.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// Every arc *segmentation* admits: each entry the keystrokes of a run of
    /// its positions spell, one character a position, in start order.
    ///
    /// A character is spelled at a position when the segment's typed letters
    /// are one of its syllables, or an initial (a single letter, `zh`, `ch`,
    /// `sh`) one of its syllables starts with. A corrected segment spells
    /// nothing: its letters are a typo, not a reading. *masks* are the
    /// reading's per-position candidate sets, which decide which characters
    /// sit outside them.
    #[must_use]
    pub(crate) fn arcs(
        &self,
        segmentation: &Segmentation,
        masks: &[Vec<CharId>],
        pinyin: &str,
        lexicon: &Lexicon,
        table: &SyllableTable,
        options: &SegmentOptions,
    ) -> Vec<WordArc> {
        let segments = segmentation.segments();
        let mut arcs = Vec::new();
        let mut key: Vec<u8> = Vec::new();
        for start in 0..segments.len() {
            key.clear();
            for (offset, segment) in segments[start..].iter().enumerate() {
                if segment.syllables().is_none() {
                    break;
                }
                key.push(pinyin.as_bytes()[segment.start()]);
                let Some(entries) = self.index.get(key.as_slice()) else {
                    break;
                };
                if offset == 0 {
                    continue;
                }
                let run = &segments[start..=start + offset];
                for &entry in entries {
                    if let Some(arc) = spelled(
                        &self.words[entry as usize],
                        start,
                        run,
                        &masks[start..=start + offset],
                        pinyin,
                        lexicon,
                        table,
                        options,
                    ) {
                        arcs.push(arc);
                    }
                }
            }
        }
        arcs
    }
}

/// Every key *word* is indexed under: the first letter of one of each
/// character's readings, in every combination its polyphones allow.
fn initials(word: &[CharId], lexicon: &Lexicon, table: &SyllableTable) -> Vec<Vec<u8>> {
    let mut keys: Vec<Vec<u8>> = vec![Vec::with_capacity(word.len())];
    for &ch in word {
        let mut letters: Vec<u8> = lexicon
            .readings(ch)
            .iter()
            .map(|&syllable| table.spelling(syllable).as_bytes()[0])
            .collect();
        letters.sort_unstable();
        letters.dedup();
        keys = keys
            .iter()
            .flat_map(|key| {
                letters.iter().map(move |&letter| {
                    let mut next = key.clone();
                    next.push(letter);
                    next
                })
            })
            .collect();
    }
    keys
}

/// How the keystrokes at one position spell an arc's character.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Spelled {
    /// The letters are one of the character's syllables.
    Full,
    /// The letters are an initial one of its syllables starts with.
    Initial,
}

/// How *span* spells *ch*, if it does. A polyphone typed in full under one
/// reading is full, whatever its other readings start with.
fn spell(ch: CharId, span: &str, lexicon: &Lexicon, table: &SyllableTable) -> Option<Spelled> {
    let abbreviation = SyllableTable::is_abbreviation(span);
    let mut spelled = None;
    for &syllable in lexicon.readings(ch) {
        let spelling = table.spelling(syllable);
        if spelling == span {
            return Some(Spelled::Full);
        }
        if abbreviation && spelling.starts_with(span) {
            spelled = Some(Spelled::Initial);
        }
    }
    spelled
}

/// The arc *word* makes over *run*, the segments from *start* on, or `None`
/// when the keystrokes do not spell it.
#[expect(
    clippy::too_many_arguments,
    reason = "one match reads the word, the run and the tables that relate them"
)]
fn spelled(
    word: &[CharId],
    start: usize,
    run: &[Segment],
    masks: &[Vec<CharId>],
    pinyin: &str,
    lexicon: &Lexicon,
    table: &SyllableTable,
    options: &SegmentOptions,
) -> Option<WordArc> {
    if word.len() != run.len() {
        return None;
    }
    let mut abbreviated = false;
    let mut reread = 0.0f32;
    for ((&ch, segment), mask) in word.iter().zip(run).zip(masks) {
        let span = &pinyin[segment.start()..segment.end()];
        match spell(ch, span, lexicon, table)? {
            Spelled::Full => {}
            Spelled::Initial => abbreviated = true,
        }
        if mask.binary_search(&ch).is_err() {
            // The segment was read as a syllable the character does not
            // have; the arc reads it as the initial instead and pays what
            // that reading of the segment costs over the one taken.
            reread += options.ambiguity_weight
                * (log_len(table.prefix_range(span).len()) - log_len(segment.ambiguity()));
        }
    }
    Some(WordArc {
        start,
        chars: word.into(),
        abbreviated,
        reread,
    })
}

/// `ln(n)` of a segment's ambiguity.
#[expect(
    clippy::cast_precision_loss,
    reason = "an ambiguity is at most the inventory size"
)]
fn log_len(n: usize) -> f32 {
    (n as f32).ln()
}

/// One lexicon entry on one reading: the beam may emit its characters at
/// positions `start..start + len` as a unit.
///
/// At least two characters long and within its reading, which
/// [`WordLexicon`] guarantees for every arc it proposes.
#[derive(Clone, Debug, PartialEq)]
pub struct WordArc {
    start: usize,
    chars: Box<[CharId]>,
    abbreviated: bool,
    reread: f32,
}

impl WordArc {
    /// The position of the arc's first character.
    #[must_use]
    pub const fn start(&self) -> usize {
        self.start
    }

    /// The characters, left to right; never fewer than two.
    #[must_use]
    pub fn chars(&self) -> &[CharId] {
        &self.chars
    }

    /// Whether at least one character was typed as an initial: the arcs the
    /// lexicon prior applies to.
    #[must_use]
    pub const fn abbreviated(&self) -> bool {
        self.abbreviated
    }

    /// The segmentation cost of reading the arc's segments as the initials it
    /// needs, over the reading's own: positive where the reading took a
    /// segment as a syllable the arc's character does not have.
    #[must_use]
    pub const fn reread(&self) -> f32 {
        self.reread
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ime_pinyin::SegmentLattice;

    fn tables() -> (SyllableTable, Lexicon) {
        let table = SyllableTable::load();
        let lexicon = Lexicon::load(&table).expect("generated tables must agree");
        (table, lexicon)
    }

    fn every_character(lexicon: &Lexicon) -> Emittable {
        let source: String = lexicon
            .characters()
            .iter()
            .flat_map(|&ch| [ch, '\n'])
            .collect();
        Emittable::parse(&source, lexicon).expect("the lexicon's own characters")
    }

    fn offline() -> SegmentOptions {
        SegmentOptions {
            allow_incomplete_tail: false,
            ..SegmentOptions::default()
        }
    }

    fn arcs_of(
        words: &WordLexicon,
        input: &str,
        table: &SyllableTable,
        lexicon: &Lexicon,
    ) -> Vec<(Vec<String>, Vec<WordArc>)> {
        let options = offline();
        let lattice = SegmentLattice::build(input, table, &options).expect("input reads");
        lattice
            .k_best(&options)
            .iter()
            .map(|segmentation| {
                let masks: Vec<Vec<CharId>> = segmentation
                    .segments()
                    .iter()
                    .map(|segment| {
                        let mut mask = Vec::new();
                        lexicon.mask_into(segment.syllables().expect("clean"), &mut mask);
                        mask
                    })
                    .collect();
                let spans = segmentation
                    .segments()
                    .iter()
                    .map(|segment| input[segment.start()..segment.end()].to_owned())
                    .collect();
                (
                    spans,
                    words.arcs(segmentation, &masks, input, lexicon, table, &options),
                )
            })
            .collect()
    }

    fn text(lexicon: &Lexicon, arc: &WordArc) -> String {
        arc.chars()
            .iter()
            .map(|&id| lexicon.character(id))
            .collect()
    }

    #[test]
    fn only_the_asked_categories_are_kept() {
        let (table, lexicon) = tables();
        let (words, stats) = WordLexicon::parse(
            "一帆风顺\tidiom\n周杰伦\tperson\n一帆风顺\tidiom\n",
            &["idiom"],
            &lexicon,
            &table,
            &every_character(&lexicon),
        )
        .expect("the lexicon reads");
        assert_eq!(words.len(), 1);
        assert_eq!(
            stats,
            WordLexiconStats {
                kept: 1,
                other_category: 1,
                duplicate: 1,
                ..WordLexiconStats::default()
            }
        );
    }

    #[test]
    fn an_entry_route_a_cannot_emit_is_left_out() {
        let (table, lexicon) = tables();
        let narrow = Emittable::parse("一\n帆\n风\n", &lexicon).expect("characters");
        let error = WordLexicon::parse("一帆风顺\tidiom\n", &["idiom"], &lexicon, &table, &narrow)
            .expect_err("顺 is not emittable, so nothing is left");
        assert!(matches!(error, WordLexiconError::Empty { .. }));
    }

    #[test]
    fn a_line_without_a_category_is_refused() {
        let (table, lexicon) = tables();
        let error = WordLexicon::parse(
            "一帆风顺\n",
            &["idiom"],
            &lexicon,
            &table,
            &every_character(&lexicon),
        )
        .expect_err("no tab");
        assert!(matches!(error, WordLexiconError::Malformed { line: 1, .. }));
    }

    #[test]
    fn initials_propose_the_idiom_and_mark_it_abbreviated() {
        let (table, lexicon) = tables();
        let (words, _) = WordLexicon::parse(
            "一帆风顺\tidiom\n",
            &["idiom"],
            &lexicon,
            &table,
            &every_character(&lexicon),
        )
        .expect("the lexicon reads");
        let readings = arcs_of(&words, "yffs", &table, &lexicon);
        let (_, arcs) = readings
            .iter()
            .find(|(spans, _)| spans.len() == 4)
            .expect("yffs reads as four initials");
        assert_eq!(arcs.len(), 1);
        assert_eq!(text(&lexicon, &arcs[0]), "一帆风顺");
        assert_eq!(arcs[0].start(), 0);
        assert!(arcs[0].abbreviated());
        assert!(
            arcs[0].reread().abs() < f32::EPSILON,
            "every character is in its mask"
        );
    }

    #[test]
    fn full_pinyin_proposes_the_idiom_without_the_prior() {
        let (table, lexicon) = tables();
        let (words, _) = WordLexicon::parse(
            "一帆风顺\tidiom\n",
            &["idiom"],
            &lexicon,
            &table,
            &every_character(&lexicon),
        )
        .expect("the lexicon reads");
        let readings = arcs_of(&words, "woyifanfengshun", &table, &lexicon);
        let (_, arcs) = readings
            .iter()
            .find(|(spans, _)| spans.concat() == "woyifanfengshun" && spans.len() == 5)
            .expect("the five-syllable reading");
        assert_eq!(arcs.len(), 1);
        assert_eq!(arcs[0].start(), 1);
        assert!(!arcs[0].abbreviated());
    }

    #[test]
    fn a_segment_read_as_a_whole_syllable_still_spells_an_initial() {
        let (table, lexicon) = tables();
        // 马马虎虎 typed `mmhh`: the readings take `m` as the syllable m (呣)
        // as well as the initial; on a reading that took the syllable, the
        // arc still matches and pays the reread.
        let (words, _) = WordLexicon::parse(
            "马马虎虎\tidiom\n",
            &["idiom"],
            &lexicon,
            &table,
            &every_character(&lexicon),
        )
        .expect("the lexicon reads");
        let options = offline();
        let input = "mmhh";
        let lattice = SegmentLattice::build(input, &table, &options).expect("input reads");
        let mut saw_reread = false;
        for segmentation in lattice.k_best(&options) {
            if segmentation.len() != 4 {
                continue;
            }
            let masks: Vec<Vec<CharId>> = segmentation
                .segments()
                .iter()
                .map(|segment| {
                    let mut mask = Vec::new();
                    lexicon.mask_into(segment.syllables().expect("clean"), &mut mask);
                    mask
                })
                .collect();
            let arcs = words.arcs(&segmentation, &masks, input, &lexicon, &table, &options);
            assert_eq!(arcs.len(), 1, "every four-segment reading spells the idiom");
            let exact = segmentation
                .segments()
                .iter()
                .any(|segment| segment.ambiguity() == 1);
            if exact {
                assert!(arcs[0].reread() > 0.0);
                saw_reread = true;
            }
        }
        assert!(saw_reread, "some reading takes `m` as the syllable");
    }

    #[test]
    fn a_word_the_keystrokes_do_not_spell_is_not_proposed() {
        let (table, lexicon) = tables();
        let (words, _) = WordLexicon::parse(
            "一帆风顺\tidiom\n",
            &["idiom"],
            &lexicon,
            &table,
            &every_character(&lexicon),
        )
        .expect("the lexicon reads");
        for (_, arcs) in arcs_of(&words, "yffx", &table, &lexicon) {
            assert_eq!(arcs, Vec::<WordArc>::new());
        }
    }
}
