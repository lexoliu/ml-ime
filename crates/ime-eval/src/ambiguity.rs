//! The ambiguity slice: the characters that later input or the context decides.
//!
//! Some characters are typed exactly alike and differ only in what the rest of
//! the sentence says: 他 / 她 / 它 are all `ta`, and 的 / 地 / 得 are all `de`.
//! A decoder whose transitions read left to right chooses among them before
//! the evidence arrives -- `tashiwojiejie` is 她 only because of the `jiejie`
//! typed after it -- and sentence top-1 folds that capability in with
//! everything else. The slice isolates it: the positions whose expected
//! character belongs to a set and whose typed span admits another member of
//! the same set. A member the keystrokes alone rule out (得 typed `dei`) is
//! not ambiguous and stays out.
//!
//! Which span typed which character is recovered from the keystrokes and the
//! expected text alone ([`SegmentLattice::cheapest_reading`]), under lattice
//! options fixed here rather than taken from a run. The slice is therefore a
//! property of the evaluation set, and every system -- the fused decoder from
//! its beams, an end-to-end checkpoint from its dump -- is measured on the
//! same positions.

use crate::EvalError;
use crate::dump::DumpRow;
use crate::record::{EvalRecord, EvalSet, Slice};
use askama::Template;
use ime_pinyin::{CharId, Lexicon, Segment, SegmentLattice, SegmentOptions, SyllableTable};
use rayon::iter::{
    IndexedParallelIterator as _, IntoParallelRefIterator as _, ParallelIterator as _,
};
use std::collections::HashMap;

/// Characters typed alike that only the surrounding text tells apart.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Ambiguity {
    /// The members, in the order the report names them.
    pub characters: &'static [char],
}

impl Ambiguity {
    /// The set's name in the report: its members joined, `他/她/它`.
    #[must_use]
    pub fn name(&self) -> String {
        let mut name = String::new();
        for (index, character) in self.characters.iter().enumerate() {
            if index > 0 {
                name.push('/');
            }
            name.push(*character);
        }
        name
    }
}

/// The sets the slice measures. A position joins the slice for the set its
/// expected character belongs to; no character is in two sets.
pub const AMBIGUITIES: [Ambiguity; 2] = [
    Ambiguity {
        characters: &['他', '她', '它'],
    },
    Ambiguity {
        characters: &['的', '地', '得'],
    },
];

/// How the slice reads keystrokes: abbreviations on, no half-typed tail, the
/// default costs -- `emit-lattice`'s defaults. `max_paths` plays no part,
/// since [`SegmentLattice::cheapest_reading`] searches every edge.
fn reading_options() -> SegmentOptions {
    SegmentOptions {
        allow_incomplete_tail: false,
        ..SegmentOptions::default()
    }
}

/// One position of the slice.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Position {
    /// The character's index in the expected text.
    pub index: usize,
    /// Which of [`AMBIGUITIES`] the expected character belongs to.
    pub ambiguity: usize,
}

/// Where the slice falls in an evaluation set.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct AmbiguitySlice {
    /// Per record, in set order, the slice positions it holds; empty when it
    /// holds none.
    positions: Vec<Vec<Position>>,
    /// Per record, whether its text holds a set character but its keystrokes
    /// spell no reading of the text, so no position of it can be judged.
    unspelled: Vec<bool>,
}

impl AmbiguitySlice {
    /// Find the slice's positions in every record of *set*.
    ///
    /// A typo twin's record is read along its clean keystrokes: what makes a
    /// position ambiguous is the syllable the user meant, and the twin's
    /// slice is then exactly its parent's.
    ///
    /// # Errors
    ///
    /// If a set member is not in the lexicon, or a record's keystrokes leave
    /// the `[a-z]` alphabet.
    pub fn select(
        set: &EvalSet,
        table: &SyllableTable,
        lexicon: &Lexicon,
    ) -> Result<Self, EvalError> {
        let members = AMBIGUITIES
            .iter()
            .map(|ambiguity| {
                ambiguity
                    .characters
                    .iter()
                    .map(|&character| {
                        lexicon
                            .id_of(character)
                            .ok_or(EvalError::UnknownMember { character })
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let options = reading_options();
        let selected = set
            .records()
            .par_iter()
            .enumerate()
            .map(|(index, record)| select_record(index, record, table, lexicon, &members, &options))
            .collect::<Result<Vec<_>, _>>()?;
        let (positions, unspelled) = selected.into_iter().unzip();
        Ok(Self {
            positions,
            unspelled,
        })
    }

    /// The slice positions of the record at *index* of the set; empty when it
    /// holds none or could not be spelled.
    ///
    /// # Panics
    ///
    /// If *index* is not a record of the set the slice was selected from.
    #[must_use]
    pub fn positions(&self, index: usize) -> &[Position] {
        &self.positions[index]
    }

    /// Judge the dumped beams of the records of *slice* that hold slice
    /// positions.
    ///
    /// *rows* may hold more records than the slice -- an end-to-end dump
    /// holds dev and test alike -- but every record the slice needs must be
    /// there, under its own expected text.
    ///
    /// # Errors
    ///
    /// If *rows* name a record twice, lack a record the slice holds, or carry
    /// another text for one: the dump was not decoded from *set*.
    pub fn measure(
        &self,
        set: &EvalSet,
        slice: Slice,
        dev_share: f64,
        rows: &[DumpRow],
    ) -> Result<AmbiguityReport, EvalError> {
        let mut by_record = HashMap::with_capacity(rows.len());
        for row in rows {
            if by_record.insert(row.record, row).is_some() {
                return Err(EvalError::DuplicateRow { record: row.record });
            }
        }
        let mut report = AmbiguityReport::default();
        for (index, record) in set.records().iter().enumerate() {
            if !slice.holds(record, dev_share) {
                continue;
            }
            if self.unspelled[index] {
                report.unspelled += 1;
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
            report.fold(Outcome::new(index, record, positions, row));
        }
        Ok(report)
    }
}

/// One record's slice positions, and whether its text could not be spelled
/// from its keystrokes.
fn select_record(
    index: usize,
    record: &EvalRecord,
    table: &SyllableTable,
    lexicon: &Lexicon,
    members: &[Vec<CharId>],
    options: &SegmentOptions,
) -> Result<(Vec<Position>, bool), EvalError> {
    let text: Vec<char> = record.text.chars().collect();
    let candidates: Vec<Position> = text
        .iter()
        .enumerate()
        .filter_map(|(index, character)| {
            AMBIGUITIES
                .iter()
                .position(|ambiguity| ambiguity.characters.contains(character))
                .map(|ambiguity| Position { index, ambiguity })
        })
        .collect();
    if candidates.is_empty() {
        return Ok((Vec::new(), false));
    }
    let Some(ids) = text
        .iter()
        .map(|&character| lexicon.id_of(character))
        .collect::<Option<Vec<_>>>()
    else {
        return Ok((Vec::new(), true));
    };
    let keys = record.clean.as_deref().unwrap_or(&record.pinyin);
    let lattice =
        SegmentLattice::build(keys, table, options).map_err(|source| EvalError::Unreadable {
            record: index,
            source,
        })?;
    let Some(reading) = lattice.cheapest_reading(
        ids.len(),
        |segment, position| admits(segment, ids[position], lexicon),
        options,
    ) else {
        return Ok((Vec::new(), true));
    };
    let positions = candidates
        .into_iter()
        .filter(|position| {
            let segment = &reading.segments()[position.index];
            members[position.ambiguity]
                .iter()
                .any(|&other| other != ids[position.index] && admits(segment, other, lexicon))
        })
        .collect();
    Ok((positions, false))
}

/// Whether *segment* can have been typed for *character*: one of the
/// character's readings lies in the segment's syllable range. The slice's
/// lattices are built without the noise model, so every segment has a range.
fn admits(segment: &Segment, character: CharId, lexicon: &Lexicon) -> bool {
    segment.syllables().is_some_and(|range| {
        lexicon
            .readings(character)
            .iter()
            .any(|&reading| range.contains(reading))
    })
}

/// What the top hypothesis put at one slice position.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Verdict {
    /// The expected character.
    Right,
    /// Another member of the same set: the confusion the slice exists for.
    Swapped,
    /// Anything else, including nothing at all from a hypothesis too short
    /// to reach the position.
    Wrong,
}

/// One slice record's beam, judged.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Outcome {
    /// The record's index in the evaluation set.
    pub record: usize,
    /// The top hypothesis; `None` when the engine answered nothing.
    pub top: Option<String>,
    /// The expected sentence's zero-based rank in the dumped beam; `None`
    /// when the beam does not hold it.
    pub rank: Option<usize>,
    /// Each slice position with what the top hypothesis put there.
    pub positions: Vec<(Position, Verdict)>,
}

impl Outcome {
    fn new(index: usize, record: &EvalRecord, positions: &[Position], row: &DumpRow) -> Self {
        let expected: Vec<char> = record.text.chars().collect();
        let top = row
            .hypotheses
            .first()
            .map(|hypothesis| hypothesis.text.clone());
        let found: Vec<char> = top.as_deref().unwrap_or_default().chars().collect();
        let positions = positions
            .iter()
            .map(|&position| {
                let verdict = match found.get(position.index) {
                    Some(&character) if character == expected[position.index] => Verdict::Right,
                    Some(character)
                        if AMBIGUITIES[position.ambiguity]
                            .characters
                            .contains(character) =>
                    {
                        Verdict::Swapped
                    }
                    _ => Verdict::Wrong,
                };
                (position, verdict)
            })
            .collect();
        let rank = row
            .hypotheses
            .iter()
            .position(|hypothesis| hypothesis.text == record.text);
        Self {
            record: index,
            top,
            rank,
            positions,
        }
    }
}

/// One set's positions and how the top hypotheses fared on them.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub struct SetCounts {
    /// Slice positions whose expected character is in the set.
    pub positions: usize,
    /// Of those, how many the top hypothesis got right.
    pub right: usize,
    /// Of those, how many it filled with another member of the set.
    pub swapped: usize,
}

impl SetCounts {
    fn add(&mut self, verdict: Verdict) {
        self.positions += 1;
        match verdict {
            Verdict::Right => self.right += 1,
            Verdict::Swapped => self.swapped += 1,
            Verdict::Wrong => {}
        }
    }

    /// Fraction of the positions the top hypothesis got right.
    #[must_use]
    pub fn accuracy(&self) -> f64 {
        rate(self.right, self.positions)
    }
}

/// What one dump scored on the slice.
#[derive(Clone, PartialEq, Eq, Debug, Default, Template)]
#[template(path = "ambiguity.txt", ext = "txt")]
pub struct AmbiguityReport {
    /// Per set of [`AMBIGUITIES`], in order.
    sets: [SetCounts; AMBIGUITIES.len()],
    top1_hits: usize,
    /// Records of the slice whose keystrokes do not spell their text.
    unspelled: usize,
    /// Every judged record, in set order.
    outcomes: Vec<Outcome>,
}

impl AmbiguityReport {
    fn fold(&mut self, outcome: Outcome) {
        if outcome.rank == Some(0) {
            self.top1_hits += 1;
        }
        for &(position, verdict) in &outcome.positions {
            self.sets[position.ambiguity].add(verdict);
        }
        self.outcomes.push(outcome);
    }

    /// How many records hold at least one slice position.
    #[must_use]
    pub fn records(&self) -> usize {
        self.outcomes.len()
    }

    /// How many of them the top hypothesis got exactly right.
    #[must_use]
    pub const fn top1_hits(&self) -> usize {
        self.top1_hits
    }

    /// Fraction of the slice's records whose top hypothesis was exactly right.
    #[must_use]
    pub fn top1_accuracy(&self) -> f64 {
        rate(self.top1_hits, self.records())
    }

    /// Every set's counts summed.
    #[must_use]
    pub fn total(&self) -> SetCounts {
        self.sets
            .iter()
            .fold(SetCounts::default(), |sum, set| SetCounts {
                positions: sum.positions + set.positions,
                right: sum.right + set.right,
                swapped: sum.swapped + set.swapped,
            })
    }

    /// The counts of one set of [`AMBIGUITIES`].
    #[must_use]
    pub const fn set(&self, ambiguity: usize) -> SetCounts {
        self.sets[ambiguity]
    }

    /// How many records of the slice could not be spelled from their
    /// keystrokes and so were not judged.
    #[must_use]
    pub const fn unspelled(&self) -> usize {
        self.unspelled
    }

    /// Every judged record, in set order.
    #[must_use]
    pub fn outcomes(&self) -> &[Outcome] {
        &self.outcomes
    }

    /// The per-set rows of the rendered table, labels padded to its column.
    fn set_rows(&self) -> Vec<(String, SetCounts)> {
        AMBIGUITIES
            .iter()
            .zip(self.sets)
            .map(|(ambiguity, counts)| (pad(&format!("  {}", ambiguity.name())), counts))
            .collect()
    }

    /// Record by record, how *self* differs from *baseline* on the same
    /// slice: what each side alone got right, over sentence top-1 and over
    /// the positions.
    ///
    /// # Errors
    ///
    /// If the two reports did not judge the same records.
    pub fn compare(&self, baseline: &Self) -> Result<Comparison, EvalError> {
        let same = self.outcomes.len() == baseline.outcomes.len()
            && self
                .outcomes
                .iter()
                .zip(&baseline.outcomes)
                .all(|(ours, theirs)| ours.record == theirs.record);
        if !same {
            return Err(EvalError::DifferentRecords);
        }
        let mut comparison = Comparison::default();
        for (ours, theirs) in self.outcomes.iter().zip(&baseline.outcomes) {
            comparison
                .records
                .add(ours.rank == Some(0), theirs.rank == Some(0));
            for ((_, ours), (_, theirs)) in ours.positions.iter().zip(&theirs.positions) {
                comparison
                    .positions
                    .add(*ours == Verdict::Right, *theirs == Verdict::Right);
            }
        }
        Ok(comparison)
    }

    /// Every judged record by name: its keystrokes, its context, the wanted
    /// sentence, the top answer and the wanted sentence's rank.
    ///
    /// # Panics
    ///
    /// If *set* is not the evaluation set the report was measured on.
    #[must_use]
    pub fn listing<'a>(&'a self, set: &'a EvalSet) -> Listing<'a> {
        Listing {
            rows: self
                .outcomes
                .iter()
                .map(|outcome| (outcome, &set.records()[outcome.record]))
                .collect(),
        }
    }
}

/// The slice's records one per line, tab-separated.
#[derive(Template)]
#[template(path = "ambiguity_listing.txt", ext = "txt")]
pub struct Listing<'a> {
    rows: Vec<(&'a Outcome, &'a EvalRecord)>,
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
    fn add(&mut self, ours: bool, theirs: bool) {
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

/// One report against a baseline on the same slice.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default, Template)]
#[template(path = "ambiguity_comparison.txt", ext = "txt")]
pub struct Comparison {
    /// Over the slice's records, sentence top-1.
    pub records: Discordance,
    /// Over the slice's positions, the top hypothesis's character. Positions
    /// of one record are not independent, so this z overstates the evidence
    /// when a record holds several.
    pub positions: Discordance,
}

/// *label* padded to the report's first column, counting a CJK character as
/// the two columns a terminal gives it.
fn pad(label: &str) -> String {
    const COLUMN: usize = 22;
    let width: usize = label
        .chars()
        .map(|character| if character.is_ascii() { 1 } else { 2 })
        .sum();
    format!("{label}{}", " ".repeat(COLUMN.saturating_sub(width)))
}

/// A ratio, or zero where there was nothing to divide.
#[expect(clippy::cast_precision_loss, reason = "counts stay far below 2^53")]
fn rate(hits: usize, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    hits as f64 / total as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dump::{DumpedHypothesis, parse_dump};

    const PRONOUNS: &str = include_str!("../data/pronoun-cases.jsonl");

    fn tables() -> (SyllableTable, Lexicon) {
        let table = SyllableTable::load();
        let lexicon = Lexicon::load(&table).expect("the generated tables agree");
        (table, lexicon)
    }

    fn select(source: &str) -> (EvalSet, AmbiguitySlice) {
        let set = EvalSet::parse(source).expect("the fixture parses");
        let (table, lexicon) = tables();
        let slice = AmbiguitySlice::select(&set, &table, &lexicon).expect("the fixture reads");
        (set, slice)
    }

    fn at(index: usize, ambiguity: usize) -> Position {
        Position { index, ambiguity }
    }

    fn row(record: usize, text: &str, hypotheses: &[&str]) -> DumpRow {
        DumpRow {
            record,
            text: text.to_owned(),
            hypotheses: hypotheses
                .iter()
                .map(|text| DumpedHypothesis {
                    text: (*text).to_owned(),
                    score: 0.0,
                })
                .collect(),
        }
    }

    #[test]
    fn every_pronoun_case_is_in_the_slice_at_its_pronoun() {
        let (set, slice) = select(PRONOUNS);
        let expected = [0, 5, 0, 0, 0, 0, 0];
        for (record, index) in expected.into_iter().enumerate() {
            assert_eq!(
                slice.positions(record),
                [at(index, 0)],
                "{}",
                set.records()[record].text
            );
        }
    }

    #[test]
    fn a_member_the_keystrokes_already_decide_stays_out() {
        // 得 typed `dei` admits no other member; 的 typed `di` still admits
        // 地, and `de` or a bare `d` admit all three.
        let (_, slice) = select(concat!(
            "{\"pinyin\": \"deizou\", \"text\": \"得走\"}\n",
            "{\"pinyin\": \"dique\", \"text\": \"的确\"}\n",
            "{\"pinyin\": \"wodeshu\", \"text\": \"我的书\"}\n",
            "{\"pinyin\": \"wdshu\", \"text\": \"我的书\"}\n",
            "{\"pinyin\": \"paodekuai\", \"text\": \"跑得快\"}\n",
        ));
        assert_eq!(slice.positions(0), []);
        assert_eq!(slice.positions(1), [at(0, 1)]);
        assert_eq!(slice.positions(2), [at(1, 1)]);
        assert_eq!(slice.positions(3), [at(1, 1)]);
        assert_eq!(slice.positions(4), [at(1, 1)]);
    }

    #[test]
    fn a_text_the_keystrokes_cannot_spell_is_counted_not_judged() {
        let (set, slice) = select(concat!(
            "{\"pinyin\": \"nihao\", \"text\": \"他好\"}\n",
            "{\"pinyin\": \"nihao\", \"text\": \"你好\"}\n",
        ));
        assert_eq!(slice.positions(0), []);
        assert_eq!(slice.positions(1), []);
        let report = slice
            .measure(&set, Slice::All, 0.0905, &[])
            .expect("no record needs a row");
        assert_eq!(report.unspelled(), 1, "only the record holding 他");
        assert_eq!(report.records(), 0);
    }

    #[test]
    fn the_metrics_are_what_the_beams_say_they_are() {
        let (set, slice) = select(PRONOUNS);
        let rows = [
            // Right at top-1.
            row(0, "她是我姐姐", &["她是我姐姐"]),
            // The pronoun swapped, the right sentence second.
            row(
                1,
                "我不知道送他什么",
                &["我不知道送她什么", "我不知道送他什么"],
            ),
            // Wrong at the pronoun with a non-member, the right one absent.
            row(2, "它喜欢躲在床底下", &["塔喜欢躲在床底下"]),
            // Too short to reach the pronoun is wrong, not swapped.
            row(3, "她是我妹妹", &[]),
            row(4, "他是我哥哥", &["他是我哥哥"]),
            row(5, "他是我弟弟", &["她是我弟弟"]),
            row(6, "她是我姐姐", &["他是我姐姐", "她是我姐姐"]),
        ];
        let report = slice
            .measure(&set, Slice::All, 0.0905, &rows)
            .expect("every record has its row");
        assert_eq!(report.records(), 7);
        assert_eq!(report.top1_hits(), 2);
        assert_eq!(
            report.set(0),
            SetCounts {
                positions: 7,
                right: 2,
                swapped: 3,
            }
        );
        assert_eq!(report.set(1), SetCounts::default());
        assert_eq!(report.total(), report.set(0));
        let ranks: Vec<_> = report
            .outcomes()
            .iter()
            .map(|outcome| outcome.rank)
            .collect();
        assert_eq!(
            ranks,
            [Some(0), Some(1), None, None, Some(0), None, Some(1)]
        );
        assert_eq!(report.outcomes()[3].top, None);
    }

    #[test]
    fn the_dump_must_cover_the_slice_with_the_same_texts() {
        let (set, slice) = select(PRONOUNS);
        let missing = slice.measure(&set, Slice::All, 0.0905, &[row(0, "她是我姐姐", &[])]);
        assert!(
            matches!(missing, Err(EvalError::MissingRow { record: 1 })),
            "{missing:?}"
        );
        let mut rows: Vec<_> = (0..7)
            .map(|record| row(record, &set.records()[record].text, &[]))
            .collect();
        rows[4].text = "他是我姐姐".to_owned();
        let other = slice.measure(&set, Slice::All, 0.0905, &rows);
        assert!(
            matches!(other, Err(EvalError::TextMismatch { record: 4, .. })),
            "{other:?}"
        );
        rows[4].text = set.records()[4].text.clone();
        rows.push(row(4, "他是我哥哥", &[]));
        let twice = slice.measure(&set, Slice::All, 0.0905, &rows);
        assert!(
            matches!(twice, Err(EvalError::DuplicateRow { record: 4 })),
            "{twice:?}"
        );
    }

    #[test]
    fn a_slice_needs_only_its_own_records() {
        // At a dev share of one half the hash puts records 3 to 6 in dev and
        // 0 to 2 in test.
        let (set, slice) = select(PRONOUNS);
        let rows: Vec<_> = (3..7)
            .map(|record| row(record, &set.records()[record].text, &[]))
            .collect();
        let report = slice
            .measure(&set, Slice::Dev, 0.5, &rows)
            .expect("the dev records are all there");
        let judged: Vec<_> = report
            .outcomes()
            .iter()
            .map(|outcome| outcome.record)
            .collect();
        assert_eq!(judged, [3, 4, 5, 6]);
        let test = slice.measure(&set, Slice::Test, 0.5, &rows);
        assert!(
            matches!(test, Err(EvalError::MissingRow { record: 0 })),
            "{test:?}"
        );
    }

    #[test]
    fn a_comparison_counts_what_each_side_alone_got_right() {
        let (set, slice) = select(PRONOUNS);
        let texts: Vec<_> = set
            .records()
            .iter()
            .map(|record| record.text.clone())
            .collect();
        let baseline: Vec<_> = (0..7)
            .map(|record| row(record, &texts[record], &[texts[record].as_str()]))
            .collect();
        let mut ours = baseline.clone();
        ours[0] = row(0, &texts[0], &["他是我姐姐"]);
        ours[1] = row(1, &texts[1], &["我不知道送她什么"]);
        let ours = slice
            .measure(&set, Slice::All, 0.0905, &ours)
            .expect("measured");
        let baseline_rows = baseline;
        let baseline = slice
            .measure(&set, Slice::All, 0.0905, &baseline_rows)
            .expect("measured");
        let comparison = ours.compare(&baseline).expect("the same records");
        assert_eq!(comparison.records, Discordance { gained: 0, lost: 2 });
        assert_eq!(comparison.positions, Discordance { gained: 0, lost: 2 });
        assert!((comparison.records.z() + 2.0_f64.sqrt()).abs() < 1e-12);
        let reverse = baseline.compare(&ours).expect("the same records");
        assert_eq!(reverse.records, Discordance { gained: 2, lost: 0 });
        assert_eq!(Discordance::default().z(), 0.0);
        let dev = slice
            .measure(&set, Slice::Dev, 0.5, &baseline_rows)
            .expect("measured");
        assert!(matches!(
            ours.compare(&dev),
            Err(EvalError::DifferentRecords)
        ));
    }

    #[test]
    fn the_report_renders_as_a_table() {
        let (set, slice) = select(PRONOUNS);
        let rows: Vec<_> = (0..7)
            .map(|record| {
                let text = set.records()[record].text.as_str();
                row(
                    record,
                    text,
                    &[if record == 1 {
                        "我不知道送她什么"
                    } else {
                        text
                    }],
                )
            })
            .collect();
        let report = slice
            .measure(&set, Slice::All, 0.0905, &rows)
            .expect("measured");
        let rendered = report.render().expect("renders");
        for line in [
            "records               7\n",
            "sentence, top-1       0.8571  6 / 7\n",
            "character             0.8571  6 / 7, 1 to another of the set\n",
            "  他/她/它            0.8571  6 / 7, 1 to another of the set\n",
            "  的/地/得            0.0000  0 / 0, 0 to another of the set\n",
            "unspelled             0\n",
        ] {
            assert!(rendered.contains(line), "{line:?} in\n{rendered}");
        }
        let listing = report.listing(&set).render().expect("renders");
        assert!(
            listing.contains("1\twobuzhidaosongtashenme\t我男朋友生日快到了，\t我不知道送他什么\t我不知道送她什么\tabsent\n"),
            "{listing}"
        );
        assert!(
            listing.contains("0\ttashiwojiejie\t-\t她是我姐姐\t她是我姐姐\t1\n"),
            "{listing}"
        );
    }

    #[test]
    fn a_dump_parses_and_refuses_a_repeated_record() {
        let rows = parse_dump(concat!(
            "{\"record\": 1, \"text\": \"他\", \"hypotheses\": [{\"text\": \"她\", \"score\": -1.5}]}\n",
            "\n",
            "{\"record\": 0, \"text\": \"的\", \"hypotheses\": []}\n",
        ))
        .expect("the dump parses");
        assert_eq!(
            rows.iter().map(|row| row.record).collect::<Vec<_>>(),
            [0, 1]
        );
        assert_eq!(rows[1].hypotheses[0].text, "她");
        let twice = parse_dump(concat!(
            "{\"record\": 3, \"text\": \"他\", \"hypotheses\": []}\n",
            "{\"record\": 3, \"text\": \"他\", \"hypotheses\": []}\n",
        ));
        assert!(matches!(twice, Err(EvalError::DuplicateRow { record: 3 })));
        let broken =
            parse_dump("{\"record\": 0, \"text\": \"他\", \"hypotheses\": []}\nnot json\n");
        assert!(matches!(
            broken,
            Err(EvalError::MalformedDump { line: 2, .. })
        ));
    }
}
