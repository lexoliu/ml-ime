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
//! An arc is proposed from the keystrokes: a position spells a character
//! when its letters are one of the character's syllables or, with
//! abbreviations allowed, an initial one of them starts with. Either way the
//! character lies in the position's mask, since the lattice reads an initial
//! through its whole prefix range, so an arc asks the models about nothing
//! the reading does not already ask.

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
    /// are one of its syllables, or, when *options* allow abbreviations, an
    /// initial (a single letter, `zh`, `ch`, `sh`) one of its syllables starts
    /// with. A corrected segment spells nothing: its letters are a typo, not a
    /// reading. Every arc character therefore lies in its position's mask: a
    /// full spelling is the segment's own syllable, and the lattice reads an
    /// initial through its whole prefix range.
    #[must_use]
    pub(crate) fn arcs(
        &self,
        segmentation: &Segmentation,
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
                        pinyin,
                        lexicon,
                        table,
                        options.allow_abbreviation,
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
/// reading is full, whatever its other readings start with; an initial
/// spells nothing unless abbreviations are allowed.
fn spell(
    ch: CharId,
    span: &str,
    lexicon: &Lexicon,
    table: &SyllableTable,
    allow_abbreviation: bool,
) -> Option<Spelled> {
    let abbreviation = allow_abbreviation && SyllableTable::is_abbreviation(span);
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
fn spelled(
    word: &[CharId],
    start: usize,
    run: &[Segment],
    pinyin: &str,
    lexicon: &Lexicon,
    table: &SyllableTable,
    allow_abbreviation: bool,
) -> Option<WordArc> {
    if word.len() != run.len() {
        return None;
    }
    let mut abbreviated = false;
    for (&ch, segment) in word.iter().zip(run) {
        let span = &pinyin[segment.start()..segment.end()];
        match spell(ch, span, lexicon, table, allow_abbreviation)? {
            Spelled::Full => {}
            Spelled::Initial => abbreviated = true,
        }
    }
    Some(WordArc {
        start,
        chars: word.into(),
        abbreviated,
    })
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
                let spans = segmentation
                    .segments()
                    .iter()
                    .map(|segment| input[segment.start()..segment.end()].to_owned())
                    .collect();
                (
                    spans,
                    words.arcs(segmentation, input, lexicon, table, &options),
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
    fn an_initial_spells_nothing_when_abbreviations_are_off() {
        let (table, lexicon) = tables();
        // `m` is the syllable m (呣) as well as 马's initial: with
        // abbreviations off the lattice reads it as the syllable alone, and
        // the arc must not read it as the initial either.
        let (words, _) = WordLexicon::parse(
            "马到成功\tidiom\n",
            &["idiom"],
            &lexicon,
            &table,
            &every_character(&lexicon),
        )
        .expect("the lexicon reads");
        let input = "mdaochenggong";
        let abbreviated: Vec<_> = arcs_of(&words, input, &table, &lexicon)
            .into_iter()
            .flat_map(|(_, arcs)| arcs)
            .collect();
        assert!(!abbreviated.is_empty() && abbreviated.iter().all(WordArc::abbreviated));
        let options = SegmentOptions {
            allow_abbreviation: false,
            ..offline()
        };
        for typed in [input, "madaochenggong"] {
            let lattice = SegmentLattice::build(typed, &table, &options).expect("input reads");
            let arcs: Vec<_> = lattice
                .k_best(&options)
                .iter()
                .flat_map(|segmentation| {
                    words.arcs(segmentation, typed, &lexicon, &table, &options)
                })
                .collect();
            if typed == input {
                assert_eq!(arcs, Vec::<WordArc>::new());
            } else {
                assert!(!arcs.is_empty() && !arcs.iter().any(WordArc::abbreviated));
            }
        }
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
