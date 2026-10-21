//! What every slice of an evaluation set shares.
//!
//! A slice is a set of positions of each record's expected text, chosen from
//! the evaluation set alone, so every system -- the fused decoder from its
//! beams, an end-to-end checkpoint or a generator from its dump -- is measured
//! on the same positions. The slices differ in which positions they choose
//! ([`crate::AmbiguitySlice`], [`crate::KnowledgeSlice`]); they share how a
//! record's keystrokes are read back onto its text, which records a dump must
//! cover, what a record's answer is, the paired test, and the listing.

use crate::EvalError;
use crate::dump::DumpRow;
use crate::record::{EvalRecord, EvalSet, Slice};
use askama::Template;
use ime_pinyin::{
    CharId, Lexicon, Segment, SegmentLattice, SegmentOptions, Segmentation, SyllableTable,
};
use rayon::iter::{
    IndexedParallelIterator as _, IntoParallelRefIterator as _, ParallelIterator as _,
};
use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap};

/// How a slice reads keystrokes: abbreviations on, no half-typed tail, the
/// default costs -- `emit-lattice`'s defaults. `max_paths` plays no part,
/// since [`SegmentLattice::cheapest_reading`] searches every edge.
fn reading_options() -> SegmentOptions {
    SegmentOptions {
        allow_incomplete_tail: false,
        ..SegmentOptions::default()
    }
}

/// A record's text read back onto its keystrokes: which span typed which
/// character.
pub(crate) struct Spelling<'a> {
    /// Their lexicon ids.
    pub(crate) ids: Vec<CharId>,
    /// The cheapest reading that spells the text, one segment per character.
    pub(crate) reading: Segmentation,
    lexicon: &'a Lexicon,
}

impl Spelling<'_> {
    /// Whether the segment that typed the character at *index* can have been
    /// typed for *character*: one of its readings lies in the segment's
    /// syllable range. The lattices are built without the noise model, so
    /// every segment has a range.
    pub(crate) fn admits(&self, index: usize, character: CharId) -> bool {
        admits(&self.reading.segments()[index], character, self.lexicon)
    }

    /// Whether the segment that typed the character at *index* admits a
    /// character other than the one the text has there.
    pub(crate) fn admits_another(&self, index: usize) -> bool {
        let wanted = self.ids[index];
        self.reading.segments()[index]
            .syllables()
            .is_some_and(|range| {
                range.into_iter().any(|syllable| {
                    self.lexicon
                        .homophones(syllable)
                        .iter()
                        .any(|&other| other != wanted)
                })
            })
    }
}

fn admits(segment: &Segment, character: CharId, lexicon: &Lexicon) -> bool {
    segment.syllables().is_some_and(|range| {
        lexicon
            .readings(character)
            .iter()
            .any(|&reading| range.contains(reading))
    })
}

/// Where a slice falls in an evaluation set: per record, the positions it
/// holds, of whatever kind the slice chooses.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct Selection<P> {
    /// Per record, in set order, the positions it holds; empty when it holds
    /// none.
    positions: Vec<Vec<P>>,
    /// Per record, whether it held candidates its keystrokes cannot spell.
    unspelled: Vec<bool>,
}

impl<P: Send> Selection<P> {
    /// Choose every record's positions, in parallel: *candidates* proposes
    /// them from the record's text, and *keep* keeps those the keystrokes
    /// typed for them leave undecided. A record without candidates costs no
    /// lattice; one whose keystrokes spell no reading of its text is counted
    /// as unspelled and holds nothing.
    ///
    /// # Errors
    ///
    /// If a record's keystrokes leave the `[a-z]` alphabet.
    pub(crate) fn select(
        set: &EvalSet,
        table: &SyllableTable,
        lexicon: &Lexicon,
        candidates: impl Fn(&EvalRecord) -> Vec<P> + Sync,
        keep: impl Fn(&Spelling<'_>, &P) -> bool + Sync,
    ) -> Result<Self, EvalError> {
        let options = reading_options();
        let chosen = set
            .records()
            .par_iter()
            .enumerate()
            .map(|(index, record)| {
                let proposed = candidates(record);
                if proposed.is_empty() {
                    return Ok((proposed, false));
                }
                Ok(match spell(index, record, table, lexicon, &options)? {
                    Some(spelling) => (
                        proposed
                            .into_iter()
                            .filter(|candidate| keep(&spelling, candidate))
                            .collect(),
                        false,
                    ),
                    None => (Vec::new(), true),
                })
            })
            .collect::<Result<Vec<_>, EvalError>>()?;
        let (positions, unspelled) = chosen.into_iter().unzip();
        Ok(Self {
            positions,
            unspelled,
        })
    }
}

impl<P> Selection<P> {
    /// The positions of the record at *index*.
    pub(crate) fn positions(&self, index: usize) -> &[P] {
        &self.positions[index]
    }

    /// The selection narrowed to *records*: every other record holds nothing.
    pub(crate) fn within(mut self, records: &BTreeSet<usize>) -> Self {
        for (index, (positions, unspelled)) in self
            .positions
            .iter_mut()
            .zip(&mut self.unspelled)
            .enumerate()
        {
            if !records.contains(&index) {
                positions.clear();
                *unspelled = false;
            }
        }
        self
    }

    /// The records of *slice* that hold positions, each with its dumped row,
    /// in set order, and how many of the slice's records could not be spelled.
    ///
    /// # Errors
    ///
    /// If *rows* name a record twice, lack a record the slice holds, or carry
    /// another text for one: the dump was not decoded from *set*.
    pub(crate) fn judged<'a>(
        &'a self,
        set: &'a EvalSet,
        slice: Slice,
        dev_share: f64,
        rows: &'a [DumpRow],
    ) -> Result<Judged<'a, P>, EvalError> {
        let mut by_record = HashMap::with_capacity(rows.len());
        for row in rows {
            if by_record.insert(row.record, row).is_some() {
                return Err(EvalError::DuplicateRow { record: row.record });
            }
        }
        let mut judged = Judged {
            records: Vec::new(),
            unspelled: 0,
        };
        for (index, record) in set.records().iter().enumerate() {
            if !slice.holds(record, dev_share) {
                continue;
            }
            if self.unspelled[index] {
                judged.unspelled += 1;
                continue;
            }
            let positions = &self.positions[index];
            if positions.is_empty() {
                continue;
            }
            let row = by_record
                .get(&index)
                .ok_or(EvalError::MissingRow { record: index })?;
            if row.text != record.text {
                return Err(EvalError::TextMismatch {
                    record: index,
                    expected: record.text.clone(),
                    found: row.text.clone(),
                });
            }
            judged.records.push(JudgedRecord {
                index,
                record,
                positions,
                row,
            });
        }
        Ok(judged)
    }
}

/// The records a measurement judges.
pub(crate) struct Judged<'a, P> {
    /// Each judged record, in set order.
    pub(crate) records: Vec<JudgedRecord<'a, P>>,
    /// Records of the slice whose keystrokes do not spell their text.
    pub(crate) unspelled: usize,
}

/// One judged record.
pub(crate) struct JudgedRecord<'a, P> {
    /// The record's index in the set.
    pub(crate) index: usize,
    /// The record.
    pub(crate) record: &'a EvalRecord,
    /// Its positions.
    pub(crate) positions: &'a [P],
    /// Its dumped beam.
    pub(crate) row: &'a DumpRow,
}

/// The cheapest reading of *record*'s text from its keystrokes, or `None`
/// when its keystrokes spell none or it holds a character the lexicon lacks.
///
/// A typo twin's record is read along its clean keystrokes: what a position
/// measures is the syllable the user meant, and the twin's slice is then
/// exactly its parent's.
fn spell<'a>(
    index: usize,
    record: &EvalRecord,
    table: &SyllableTable,
    lexicon: &'a Lexicon,
    options: &SegmentOptions,
) -> Result<Option<Spelling<'a>>, EvalError> {
    let text: Vec<char> = record.text.chars().collect();
    let Some(ids) = text
        .iter()
        .map(|&character| lexicon.id_of(character))
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(None);
    };
    let keys = record.clean.as_deref().unwrap_or(&record.pinyin);
    let lattice =
        SegmentLattice::build(keys, table, options).map_err(|source| EvalError::Unreadable {
            record: index,
            source,
        })?;
    Ok(lattice
        .cheapest_reading(
            ids.len(),
            |segment, position| admits(segment, ids[position], lexicon),
            options,
        )
        .map(|reading| Spelling {
            ids,
            reading,
            lexicon,
        }))
}

/// What a system answered for one record.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Answer {
    /// The record's index in the evaluation set.
    pub record: usize,
    /// The top hypothesis; `None` when the system answered nothing.
    pub top: Option<String>,
    /// The expected sentence's zero-based rank in the dumped beam; `None`
    /// when the beam does not hold it.
    pub rank: Option<usize>,
}

impl Answer {
    pub(crate) fn new(index: usize, record: &EvalRecord, row: &DumpRow) -> Self {
        Self {
            record: index,
            top: row
                .hypotheses
                .first()
                .map(|hypothesis| hypothesis.text.clone()),
            rank: row
                .hypotheses
                .iter()
                .position(|hypothesis| hypothesis.text == record.text),
        }
    }

    /// Whether the top hypothesis is the expected sentence.
    #[must_use]
    pub fn right(&self) -> bool {
        self.rank == Some(0)
    }

    /// The top hypothesis's characters, positionally comparable with the
    /// expected text's; empty when the system answered nothing.
    pub(crate) fn characters(&self) -> Vec<char> {
        self.top.as_deref().unwrap_or_default().chars().collect()
    }
}

/// How often two systems disagree, and in whose favour.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub struct Discordance {
    /// Right in the compared report and wrong in the baseline.
    pub gained: usize,
    /// Wrong in the compared report and right in the baseline.
    pub lost: usize,
}

impl Discordance {
    pub(crate) fn add(&mut self, ours: bool, theirs: bool) {
        match (ours, theirs) {
            (true, false) => self.gained += 1,
            (false, true) => self.lost += 1,
            _ => {}
        }
    }

    /// The paired test's statistic the notes report beside these counts,
    /// `(gained - lost) / sqrt(gained + lost)`; zero when the two never
    /// disagree.
    #[must_use]
    #[expect(clippy::cast_precision_loss, reason = "counts stay far below 2^53")]
    pub fn z(&self) -> f64 {
        let discordant = self.gained + self.lost;
        if discordant == 0 {
            return 0.0;
        }
        (self.gained as f64 - self.lost as f64) / (discordant as f64).sqrt()
    }
}

/// Whether two reports judged the same records, in the same order.
pub(crate) fn same_records<'a>(
    ours: impl ExactSizeIterator<Item = &'a Answer>,
    theirs: impl ExactSizeIterator<Item = &'a Answer>,
) -> Result<(), EvalError> {
    if ours.len() == theirs.len() && ours.zip(theirs).all(|(a, b)| a.record == b.record) {
        Ok(())
    } else {
        Err(EvalError::DifferentRecords)
    }
}

/// A slice's records one per line, tab-separated: the keystrokes, the
/// context, the wanted sentence, the top answer, the wanted sentence's rank,
/// and the spans the slice chose when it names them.
#[derive(Template)]
#[template(path = "slice_listing.txt", ext = "txt")]
pub struct Listing<'a> {
    rows: Vec<ListingRow<'a>>,
    spans: bool,
}

impl<'a> Listing<'a> {
    /// The listing of *answers*, each with the spans the slice chose in it
    /// when *spans* is given.
    ///
    /// # Panics
    ///
    /// If *set* is not the evaluation set the answers were judged on.
    pub(crate) fn new(
        set: &'a EvalSet,
        answers: impl Iterator<Item = (&'a Answer, Option<String>)>,
        spans: bool,
    ) -> Self {
        Self {
            rows: answers
                .map(|(answer, chosen)| {
                    let record = &set.records()[answer.record];
                    ListingRow {
                        record: answer.record,
                        keystrokes: field(&record.pinyin),
                        context: record.context.as_deref().map_or(Cow::Borrowed("-"), field),
                        wanted: field(&record.text),
                        top: answer.top.as_deref().map_or(Cow::Borrowed("-"), field),
                        rank: answer.rank,
                        spans: chosen.unwrap_or_default(),
                    }
                })
                .collect(),
            spans,
        }
    }
}

/// One line of a [`Listing`], its text fields escaped by [`field`].
struct ListingRow<'a> {
    record: usize,
    keystrokes: Cow<'a, str>,
    context: Cow<'a, str>,
    wanted: Cow<'a, str>,
    top: Cow<'a, str>,
    rank: Option<usize>,
    spans: String,
}

/// *text* as one tab-separated field: a backslash, tab, newline or carriage
/// return is written as its backslash escape, so a multi-line context keeps
/// its record on one line.
pub(crate) fn field(text: &str) -> Cow<'_, str> {
    if !text.contains(['\\', '\t', '\n', '\r']) {
        return Cow::Borrowed(text);
    }
    let mut escaped = String::with_capacity(text.len() + 8);
    for character in text.chars() {
        match character {
            '\\' => escaped.push_str("\\\\"),
            '\t' => escaped.push_str("\\t"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            other => escaped.push(other),
        }
    }
    Cow::Owned(escaped)
}

/// *label* padded to a report's first column, counting a CJK character as
/// the two columns a terminal gives it.
pub(crate) fn pad(label: &str) -> String {
    const COLUMN: usize = 22;
    let width: usize = label
        .chars()
        .map(|character| if character.is_ascii() { 1 } else { 2 })
        .sum();
    format!("{label}{}", " ".repeat(COLUMN.saturating_sub(width)))
}

/// A ratio, or zero where there was nothing to divide.
#[expect(clippy::cast_precision_loss, reason = "counts stay far below 2^53")]
pub(crate) fn rate(hits: usize, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    hits as f64 / total as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_listed_field_keeps_its_record_on_one_line() {
        assert!(matches!(field("她是我姐姐"), Cow::Borrowed("她是我姐姐")));
        assert_eq!(
            field("不知道，\n自求多福\t啦\\"),
            "不知道，\\n自求多福\\t啦\\\\"
        );
    }

    #[test]
    fn the_paired_statistic_is_signed_and_zero_without_disagreement() {
        let mut discordance = Discordance::default();
        assert_eq!(discordance.z(), 0.0);
        for (ours, theirs) in [(true, false), (true, false), (false, true), (true, true)] {
            discordance.add(ours, theirs);
        }
        assert_eq!(discordance, Discordance { gained: 2, lost: 1 });
        assert!((discordance.z() - 1.0 / 3.0_f64.sqrt()).abs() < 1e-12);
    }
}
