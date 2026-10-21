//! The knowledge slice: the spans of text that name something in the world.
//!
//! Sentence top-1 folds world knowledge in with everything else a decoder
//! does. The slice isolates it: the spans of a record's expected text that
//! match an entry of a real-world lexicon -- a person, a place, an
//! organisation, an idiom, an internet meme -- each kept under its entry's
//! category, and only where the keystrokes typed for the span admit another
//! reading of the same length. A span the keystrokes already decide measures
//! nothing.
//!
//! Matching is by string, over every start and length of the text. Where two
//! matches nest, only the outer one is an entity: 北京大学 is one
//! organisation, not an organisation holding a place. Where two matches
//! overlap without nesting, both are entities, and the positions they share
//! count once.
//!
//! Like [`crate::AmbiguitySlice`], the slice is a property of the evaluation
//! set and the lexicon alone: which span typed which character comes from
//! [`ime_pinyin::SegmentLattice::cheapest_reading`] over the keystrokes and
//! the expected text, so every system is measured on the same spans.

use crate::EvalError;
use crate::dump::DumpRow;
use crate::record::{EvalRecord, EvalSet, Slice};
use crate::slice::{Answer, Discordance, Listing, Selection, pad, rate, same_records};
use askama::Template;
use ime_pinyin::{Lexicon, SyllableTable};
use std::collections::{BTreeSet, HashMap};
use std::fmt::Write as _;

/// The shortest entry a lexicon may hold. A single character is never
/// knowledge: its keystrokes are one syllable, and what goes there is the
/// language model's choice.
pub const MIN_ENTRY_CHARACTERS: usize = 2;

/// The entries a knowledge slice matches, each under its category.
///
/// The file is one `word<TAB>category` line per entry (`mlime lexicon
/// knowledge` writes it); every entry appears once, so its category is
/// settled before the slice reads it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct KnowledgeLexicon {
    /// The category names, sorted; an entry's category is an index here.
    categories: Vec<String>,
    /// Every entry and its category.
    entries: HashMap<String, usize>,
    /// The longest entry, in characters.
    longest: usize,
}

impl KnowledgeLexicon {
    /// Parse a lexicon file.
    ///
    /// # Errors
    ///
    /// If a line is not `word<TAB>category`, an entry is shorter than
    /// [`MIN_ENTRY_CHARACTERS`], an entry appears twice, or the file holds no
    /// entries.
    pub fn parse(source: &str) -> Result<Self, EvalError> {
        let mut named: Vec<(String, String)> = Vec::new();
        let mut seen = BTreeSet::new();
        for (index, line) in source.lines().enumerate() {
            let line_number = index + 1;
            let Some((word, category)) = line.split_once('\t') else {
                return Err(EvalError::MalformedEntry { line: line_number });
            };
            if category.is_empty() || category.contains('\t') {
                return Err(EvalError::MalformedEntry { line: line_number });
            }
            if word.chars().count() < MIN_ENTRY_CHARACTERS {
                return Err(EvalError::ShortEntry {
                    line: line_number,
                    word: word.to_owned(),
                });
            }
            if !seen.insert(word) {
                return Err(EvalError::DuplicateEntry {
                    line: line_number,
                    word: word.to_owned(),
                });
            }
            named.push((word.to_owned(), category.to_owned()));
        }
        if named.is_empty() {
            return Err(EvalError::EmptyLexicon);
        }
        let categories: Vec<String> = named
            .iter()
            .map(|(_, category)| category.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let longest = named
            .iter()
            .map(|(word, _)| word.chars().count())
            .max()
            .unwrap_or(MIN_ENTRY_CHARACTERS);
        let index: HashMap<&str, usize> = categories
            .iter()
            .enumerate()
            .map(|(index, name)| (name.as_str(), index))
            .collect();
        let entries = named
            .iter()
            .map(|(word, category)| (word.clone(), index[category.as_str()]))
            .collect();
        Ok(Self {
            categories,
            entries,
            longest,
        })
    }

    /// The category names, sorted; [`Entity::category`] indexes them.
    #[must_use]
    pub fn categories(&self) -> &[String] {
        &self.categories
    }

    /// How many entries the lexicon holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the lexicon holds no entries; never true for a parsed one.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Every match in *text* that no longer match contains.
    fn entities(&self, text: &[char]) -> Vec<Entity> {
        let mut found = Vec::new();
        let mut word = String::new();
        for start in 0..text.len() {
            word.clear();
            for end in start + 1..=text.len().min(start + self.longest) {
                word.push(text[end - 1]);
                if end - start < MIN_ENTRY_CHARACTERS {
                    continue;
                }
                if let Some(&category) = self.entries.get(&word) {
                    found.push(Entity {
                        start,
                        end,
                        category,
                    });
                }
            }
        }
        found
            .iter()
            .filter(|entity| {
                !found.iter().any(|other| {
                    other != *entity && other.start <= entity.start && entity.end <= other.end
                })
            })
            .copied()
            .collect()
    }
}

/// One span of a record's expected text that names an entry of the lexicon.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Entity {
    /// The first character's index in the expected text.
    pub start: usize,
    /// One past the last character's index.
    pub end: usize,
    /// The entry's category, an index into [`KnowledgeLexicon::categories`].
    pub category: usize,
}

/// Where the knowledge slice falls in an evaluation set.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct KnowledgeSlice {
    selection: Selection<Entity>,
    categories: Vec<String>,
}

impl KnowledgeSlice {
    /// Find the entities of every record of *set*.
    ///
    /// # Errors
    ///
    /// If a record's keystrokes leave the `[a-z]` alphabet.
    pub fn select(
        set: &EvalSet,
        table: &SyllableTable,
        lexicon: &Lexicon,
        knowledge: &KnowledgeLexicon,
    ) -> Result<Self, EvalError> {
        let selection = Selection::select(
            set,
            table,
            lexicon,
            |record| knowledge.entities(&record.text.chars().collect::<Vec<_>>()),
            |spelling, entity| {
                (entity.start..entity.end).any(|index| spelling.admits_another(index))
            },
        )?;
        Ok(Self {
            selection,
            categories: knowledge.categories.clone(),
        })
    }

    /// The entities of the record at *index* of the set; empty when it holds
    /// none or could not be spelled.
    ///
    /// # Panics
    ///
    /// If *index* is not a record of the set the slice was selected from.
    #[must_use]
    pub fn entities(&self, index: usize) -> &[Entity] {
        self.selection.positions(index)
    }

    /// The slice narrowed to *records*, for a system that answered only
    /// them: every other record holds no entity, so the system and its
    /// baseline are measured on the records both answered.
    #[must_use]
    pub fn within(self, records: &BTreeSet<usize>) -> Self {
        Self {
            selection: self.selection.within(records),
            categories: self.categories,
        }
    }

    /// Judge the dumped beams of the records of *slice* that hold entities.
    ///
    /// *rows* may hold more records than the slice, but every record the
    /// slice needs must be there, under its own expected text.
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
    ) -> Result<KnowledgeReport, EvalError> {
        let judged = self.selection.judged(set, slice, dev_share, rows)?;
        let mut report = KnowledgeReport {
            categories: self.categories.clone(),
            total: Counts::default(),
            by_category: vec![Counts::default(); self.categories.len()],
            unspelled: judged.unspelled,
            outcomes: Vec::with_capacity(judged.records.len()),
        };
        for record in judged.records {
            report.fold(KnowledgeOutcome::new(
                record.index,
                record.record,
                record.positions,
                record.row,
            ));
        }
        Ok(report)
    }
}

/// One character of an entity, judged.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct JudgedPosition {
    /// The character's index in the expected text.
    pub index: usize,
    /// The categories of the entities covering it, ascending and distinct.
    pub categories: Vec<usize>,
    /// Whether the top hypothesis has the expected character there.
    pub right: bool,
}

/// One slice record's beam, judged.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct KnowledgeOutcome {
    /// What the system answered.
    pub answer: Answer,
    /// Each entity, and whether the top hypothesis spells all of it.
    pub entities: Vec<(Entity, bool)>,
    /// Every character some entity covers, once, in text order.
    pub positions: Vec<JudgedPosition>,
}

impl KnowledgeOutcome {
    fn new(index: usize, record: &EvalRecord, entities: &[Entity], row: &DumpRow) -> Self {
        let expected: Vec<char> = record.text.chars().collect();
        let answer = Answer::new(index, record, row);
        let found = answer.characters();
        let right = |index: usize| found.get(index) == Some(&expected[index]);
        let mut covering: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); expected.len()];
        for entity in entities {
            for categories in &mut covering[entity.start..entity.end] {
                categories.insert(entity.category);
            }
        }
        let positions = covering
            .into_iter()
            .enumerate()
            .filter(|(_, categories)| !categories.is_empty())
            .map(|(index, categories)| JudgedPosition {
                index,
                categories: categories.into_iter().collect(),
                right: right(index),
            })
            .collect();
        let entities = entities
            .iter()
            .map(|&entity| (entity, (entity.start..entity.end).all(right)))
            .collect();
        Self {
            answer,
            entities,
            positions,
        }
    }

    /// The categories of the record's entities, ascending and distinct.
    fn categories(&self) -> BTreeSet<usize> {
        self.entities
            .iter()
            .map(|(entity, _)| entity.category)
            .collect()
    }
}

/// What one dump scored on the slice, or on one category of it.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub struct Counts {
    /// Records holding an entity.
    pub records: usize,
    /// Of those, the records whose top hypothesis is the expected sentence.
    pub records_right: usize,
    /// Entities.
    pub entities: usize,
    /// Of those, the entities the top hypothesis spells exactly.
    pub entities_right: usize,
    /// Characters inside an entity.
    pub positions: usize,
    /// Of those, the characters the top hypothesis has, positionally.
    pub positions_right: usize,
}

impl Counts {
    /// Sentence top-1 over the records.
    #[must_use]
    pub fn top1(&self) -> f64 {
        rate(self.records_right, self.records)
    }

    /// The share of entities spelled exactly.
    #[must_use]
    pub fn exact(&self) -> f64 {
        rate(self.entities_right, self.entities)
    }

    /// Character accuracy inside the entities.
    #[must_use]
    pub fn characters(&self) -> f64 {
        rate(self.positions_right, self.positions)
    }
}

/// What one dump scored on the slice.
#[derive(Clone, PartialEq, Eq, Debug, Template)]
#[template(path = "knowledge.txt", ext = "txt")]
pub struct KnowledgeReport {
    categories: Vec<String>,
    total: Counts,
    by_category: Vec<Counts>,
    /// Records of the slice whose keystrokes do not spell their text.
    unspelled: usize,
    /// Every judged record, in set order.
    outcomes: Vec<KnowledgeOutcome>,
}

impl KnowledgeReport {
    fn fold(&mut self, outcome: KnowledgeOutcome) {
        let right = outcome.answer.right();
        self.total.records += 1;
        self.total.records_right += usize::from(right);
        for category in outcome.categories() {
            self.by_category[category].records += 1;
            self.by_category[category].records_right += usize::from(right);
        }
        for &(entity, exact) in &outcome.entities {
            for counts in [&mut self.total, &mut self.by_category[entity.category]] {
                counts.entities += 1;
                counts.entities_right += usize::from(exact);
            }
        }
        for position in &outcome.positions {
            self.total.positions += 1;
            self.total.positions_right += usize::from(position.right);
            for &category in &position.categories {
                self.by_category[category].positions += 1;
                self.by_category[category].positions_right += usize::from(position.right);
            }
        }
        self.outcomes.push(outcome);
    }

    /// The whole slice's counts.
    #[must_use]
    pub const fn total(&self) -> Counts {
        self.total
    }

    /// The category names, in the order [`Self::category`] takes.
    #[must_use]
    pub fn categories(&self) -> &[String] {
        &self.categories
    }

    /// One category's counts.
    ///
    /// # Panics
    ///
    /// If *category* is not an index of [`Self::categories`].
    #[must_use]
    pub fn category(&self, category: usize) -> Counts {
        self.by_category[category]
    }

    /// How many records of the slice could not be spelled from their
    /// keystrokes and so were not judged.
    #[must_use]
    pub const fn unspelled(&self) -> usize {
        self.unspelled
    }

    /// Every judged record, in set order.
    #[must_use]
    pub fn outcomes(&self) -> &[KnowledgeOutcome] {
        &self.outcomes
    }

    /// The rendered table's rows: the whole slice, then each category, labels
    /// padded to its first column.
    fn rows(&self) -> Vec<(String, Counts)> {
        std::iter::once((pad("all"), self.total))
            .chain(
                self.categories
                    .iter()
                    .zip(&self.by_category)
                    .map(|(name, &counts)| (pad(&format!("  {name}")), counts)),
            )
            .collect()
    }

    /// Record by record, how *self* differs from *baseline* on the same
    /// slice: what each side alone got right, over sentence top-1, the
    /// entities and the characters, for the whole slice and each category.
    ///
    /// # Errors
    ///
    /// If the two reports did not judge the same records.
    pub fn compare(&self, baseline: &Self) -> Result<KnowledgeComparison, EvalError> {
        same_records(
            self.outcomes.iter().map(|outcome| &outcome.answer),
            baseline.outcomes.iter().map(|outcome| &outcome.answer),
        )?;
        let mut total = Paired::default();
        let mut by_category = vec![Paired::default(); self.categories.len()];
        for (ours, theirs) in self.outcomes.iter().zip(&baseline.outcomes) {
            let (ours_right, theirs_right) = (ours.answer.right(), theirs.answer.right());
            total.records.add(ours_right, theirs_right);
            for category in ours.categories() {
                by_category[category].records.add(ours_right, theirs_right);
            }
            for ((entity, ours), (_, theirs)) in ours.entities.iter().zip(&theirs.entities) {
                total.entities.add(*ours, *theirs);
                by_category[entity.category].entities.add(*ours, *theirs);
            }
            for (ours, theirs) in ours.positions.iter().zip(&theirs.positions) {
                total.positions.add(ours.right, theirs.right);
                for &category in &ours.categories {
                    by_category[category]
                        .positions
                        .add(ours.right, theirs.right);
                }
            }
        }
        Ok(KnowledgeComparison {
            rows: std::iter::once((pad("all"), total))
                .chain(
                    self.categories
                        .iter()
                        .zip(by_category)
                        .map(|(name, paired)| (pad(&format!("  {name}")), paired)),
                )
                .collect(),
        })
    }

    /// Every judged record by name: its keystrokes, its context, the wanted
    /// sentence, the top answer, the wanted sentence's rank, and its entities
    /// as `span/category`.
    ///
    /// # Panics
    ///
    /// If *set* is not the evaluation set the report was measured on.
    #[must_use]
    pub fn listing<'a>(&'a self, set: &'a EvalSet) -> Listing<'a> {
        Listing::new(
            set,
            self.outcomes.iter().map(|outcome| {
                let text: Vec<char> = set.records()[outcome.answer.record].text.chars().collect();
                let mut spans = String::new();
                for (index, (entity, exact)) in outcome.entities.iter().enumerate() {
                    if index > 0 {
                        spans.push(' ');
                    }
                    let span: String = text[entity.start..entity.end].iter().collect();
                    let mark = if *exact { "" } else { "!" };
                    write!(spans, "{span}/{}{mark}", self.categories[entity.category])
                        .expect("writing to a String cannot fail");
                }
                (&outcome.answer, Some(spans))
            }),
            true,
        )
    }
}

/// Two reports' disagreements over one part of the slice.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub struct Paired {
    /// Over the records, sentence top-1.
    pub records: Discordance,
    /// Over the entities, exact spelling.
    pub entities: Discordance,
    /// Over the characters inside entities. Characters of one entity are
    /// not independent, so this z overstates the evidence.
    pub positions: Discordance,
}

/// One report against a baseline on the same slice: the whole slice, then
/// each category.
#[derive(Clone, PartialEq, Debug, Template)]
#[template(path = "knowledge_comparison.txt", ext = "txt")]
pub struct KnowledgeComparison {
    rows: Vec<(String, Paired)>,
}

impl KnowledgeComparison {
    /// The whole slice's disagreements.
    #[must_use]
    pub fn total(&self) -> Paired {
        self.rows[0].1
    }

    /// One category's disagreements, in [`KnowledgeReport::categories`]'
    /// order.
    ///
    /// # Panics
    ///
    /// If *category* is not an index of the categories.
    #[must_use]
    pub fn category(&self, category: usize) -> Paired {
        self.rows[category + 1].1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dump::DumpedHypothesis;

    const LEXICON: &str = concat!(
        "北京\tplace\n",
        "北京大学\torganisation\n",
        "周杰伦\tperson\n",
        "南京市\tplace\n",
        "长江大桥\tplace\n",
        "一帆风顺\tidiom\n",
    );

    fn lexicon() -> KnowledgeLexicon {
        KnowledgeLexicon::parse(LEXICON).expect("the fixture parses")
    }

    fn select(source: &str) -> (EvalSet, KnowledgeSlice) {
        let set = EvalSet::parse(source).expect("the fixture parses");
        let table = SyllableTable::load();
        let pinyin = Lexicon::load(&table).expect("the generated tables agree");
        let slice =
            KnowledgeSlice::select(&set, &table, &pinyin, &lexicon()).expect("the fixture reads");
        (set, slice)
    }

    fn at(start: usize, end: usize, category: &str) -> Entity {
        Entity {
            start,
            end,
            category: lexicon()
                .categories()
                .iter()
                .position(|name| name == category)
                .expect("a fixture category"),
        }
    }

    fn row(record: usize, text: &str, top: Option<&str>) -> DumpRow {
        DumpRow {
            record,
            text: text.to_owned(),
            hypotheses: top
                .into_iter()
                .map(|text| DumpedHypothesis {
                    text: text.to_owned(),
                    score: 0.0,
                })
                .collect(),
        }
    }

    const SET: &str = concat!(
        "{\"pinyin\": \"woxihuanzhoujielun\", \"text\": \"我喜欢周杰伦\"}\n",
        "{\"pinyin\": \"tazaibeijingdaxue\", \"text\": \"他在北京大学\"}\n",
        "{\"pinyin\": \"nanjingshichangjiangdaqiao\", \"text\": \"南京市长江大桥\"}\n",
        "{\"pinyin\": \"jintiantianqihenhao\", \"text\": \"今天天气很好\"}\n",
        "{\"pinyin\": \"zhuniyifanfengshun\", \"text\": \"祝你一帆风顺\"}\n",
    );

    #[test]
    fn a_lexicon_parses_and_sorts_its_categories() {
        let lexicon = lexicon();
        assert_eq!(lexicon.len(), 6);
        assert_eq!(
            lexicon.categories(),
            ["idiom", "organisation", "person", "place"]
        );
    }

    #[test]
    fn a_broken_lexicon_names_its_line() {
        assert!(matches!(
            KnowledgeLexicon::parse("北京\tplace\n上海\n"),
            Err(EvalError::MalformedEntry { line: 2 })
        ));
        assert!(matches!(
            KnowledgeLexicon::parse("北京\t\n"),
            Err(EvalError::MalformedEntry { line: 1 })
        ));
        assert!(matches!(
            KnowledgeLexicon::parse("北\tplace\n"),
            Err(EvalError::ShortEntry { line: 1, .. })
        ));
        assert!(matches!(
            KnowledgeLexicon::parse("北京\tplace\n北京\tperson\n"),
            Err(EvalError::DuplicateEntry { line: 2, .. })
        ));
        assert!(matches!(
            KnowledgeLexicon::parse(""),
            Err(EvalError::EmptyLexicon)
        ));
    }

    #[test]
    fn a_nested_match_yields_to_its_outer_one_and_an_overlap_keeps_both() {
        let (_, slice) = select(SET);
        assert_eq!(slice.entities(0), [at(3, 6, "person")]);
        // 北京 sits inside 北京大学: one organisation.
        assert_eq!(slice.entities(1), [at(2, 6, "organisation")]);
        // 南京市 and 长江大桥 abut: two places.
        assert_eq!(slice.entities(2), [at(0, 3, "place"), at(3, 7, "place")]);
        assert_eq!(slice.entities(3), []);
        assert_eq!(slice.entities(4), [at(2, 6, "idiom")]);
    }

    #[test]
    fn overlapping_matches_both_count_and_share_their_positions_once() {
        let lexicon =
            KnowledgeLexicon::parse("南京市\tplace\n市长\tperson\n").expect("the fixture parses");
        let entities = lexicon.entities(&"南京市长".chars().collect::<Vec<_>>());
        assert_eq!(entities.len(), 2);
        let set = EvalSet::parse("{\"pinyin\": \"nanjingshizhang\", \"text\": \"南京市长\"}\n")
            .expect("the fixture parses");
        let table = SyllableTable::load();
        let pinyin = Lexicon::load(&table).expect("the generated tables agree");
        let slice =
            KnowledgeSlice::select(&set, &table, &pinyin, &lexicon).expect("the fixture reads");
        let report = slice
            .measure(
                &set,
                Slice::All,
                0.0905,
                &[row(0, "南京市长", Some("南京时长"))],
            )
            .expect("measured");
        assert_eq!(report.total().entities, 2);
        assert_eq!(report.total().entities_right, 0);
        assert_eq!(report.total().positions, 4, "市 counts once");
        assert_eq!(report.total().positions_right, 3);
        // 市 is in both categories' positions.
        assert_eq!(report.category(0).positions, 2);
        assert_eq!(report.category(1).positions, 3);
    }

    #[test]
    fn a_span_the_keystrokes_decide_stays_out() {
        // Under a lexicon where `zhou`, `jie` and `lun` each read one
        // character, 周杰伦 typed in full admits no other reading; adding 轮
        // to `lun` makes it undecided again.
        let set = EvalSet::parse(concat!(
            "{\"pinyin\": \"zhoujielun\", \"text\": \"周杰伦\"}\n",
            "{\"pinyin\": \"zjl\", \"text\": \"周杰伦\"}\n",
        ))
        .expect("the fixture parses");
        let table = SyllableTable::load();
        let decided = Lexicon::parse("伦\tlun\n周\tzhou\n杰\tjie\n", &table)
            .expect("the fixture lexicon parses");
        let slice =
            KnowledgeSlice::select(&set, &table, &decided, &lexicon()).expect("the fixture reads");
        assert_eq!(slice.entities(0), []);
        assert_eq!(
            slice.entities(1),
            [],
            "`z`, `j` and `l` reach nothing else either"
        );
        let undecided = Lexicon::parse("伦\tlun\n周\tzhou\n杰\tjie\n轮\tlun\n", &table)
            .expect("the fixture lexicon parses");
        let slice = KnowledgeSlice::select(&set, &table, &undecided, &lexicon())
            .expect("the fixture reads");
        assert_eq!(slice.entities(0), [at(0, 3, "person")]);
        assert_eq!(slice.entities(1), [at(0, 3, "person")]);
    }

    #[test]
    fn a_text_the_keystrokes_cannot_spell_is_counted_not_judged() {
        let (set, slice) = select(concat!(
            "{\"pinyin\": \"nihao\", \"text\": \"北京\"}\n",
            "{\"pinyin\": \"nihao\", \"text\": \"你好\"}\n",
        ));
        assert_eq!(slice.entities(0), []);
        let report = slice
            .measure(&set, Slice::All, 0.0905, &[])
            .expect("no record needs a row");
        assert_eq!(report.unspelled(), 1, "only the record naming 北京");
        assert_eq!(report.total().records, 0);
    }

    #[test]
    fn the_metrics_are_what_the_answers_say_they_are() {
        let (set, slice) = select(SET);
        let rows = [
            row(0, "我喜欢周杰伦", Some("我喜欢周杰伦")),
            // The entity right, the sentence wrong.
            row(1, "他在北京大学", Some("她在北京大学")),
            // One entity right, one wrong, the sentence wrong.
            row(2, "南京市长江大桥", Some("南京市长江大巧")),
            row(3, "今天天气很好", Some("今天天气很好")),
            // Nothing answered.
            row(4, "祝你一帆风顺", None),
        ];
        let report = slice
            .measure(&set, Slice::All, 0.0905, &rows)
            .expect("every record has its row");
        let total = report.total();
        assert_eq!((total.records, total.records_right), (4, 1));
        assert_eq!((total.entities, total.entities_right), (5, 3));
        assert_eq!((total.positions, total.positions_right), (18, 13));
        let place = report
            .categories()
            .iter()
            .position(|name| name == "place")
            .expect("the place category");
        assert_eq!(
            report.category(place),
            Counts {
                records: 1,
                records_right: 0,
                entities: 2,
                entities_right: 1,
                positions: 7,
                positions_right: 6,
            }
        );
        let rendered = report.render().expect("renders");
        for line in [
            "all                   4        0.2500  5         0.6000  18          0.7222\n",
            "  place               1        0.0000  2         0.5000  7           0.8571\n",
            "unspelled             0\n",
        ] {
            assert!(rendered.contains(line), "{line:?} in\n{rendered}");
        }
        let listing = report.listing(&set).render().expect("renders");
        assert!(
            listing.contains(
                "2\tnanjingshichangjiangdaqiao\t-\t南京市长江大桥\t南京市长江大巧\tabsent\t南京市/place 长江大桥/place!\n"
            ),
            "{listing}"
        );
        assert!(listing.starts_with("record\tkeystrokes\tcontext\twanted\ttop\trank\tspans\n"));
    }

    #[test]
    fn a_comparison_pairs_records_entities_and_categories() {
        let (set, slice) = select(SET);
        let texts: Vec<String> = set.records().iter().map(|r| r.text.clone()).collect();
        let right: Vec<_> = (0..5)
            .map(|record| row(record, &texts[record], Some(&texts[record])))
            .collect();
        let mut ours = right.clone();
        ours[0] = row(0, &texts[0], Some("我喜欢周节伦"));
        ours[1] = row(1, &texts[1], Some("她在北京大学"));
        let baseline = slice
            .measure(&set, Slice::All, 0.0905, &right)
            .expect("measured");
        let ours = slice
            .measure(&set, Slice::All, 0.0905, &ours)
            .expect("measured");
        let comparison = ours.compare(&baseline).expect("the same records");
        assert_eq!(
            comparison.total().records,
            Discordance { gained: 0, lost: 2 }
        );
        assert_eq!(
            comparison.total().entities,
            Discordance { gained: 0, lost: 1 }
        );
        assert_eq!(
            comparison.total().positions,
            Discordance { gained: 0, lost: 1 }
        );
        let person = ours
            .categories()
            .iter()
            .position(|name| name == "person")
            .expect("the person category");
        assert_eq!(
            comparison.category(person).entities,
            Discordance { gained: 0, lost: 1 }
        );
        let rendered = comparison.render().expect("renders");
        assert!(
            rendered.contains("all                   0       2       -1.41   0       1       -1.00   0       1       -1.00\n"),
            "{rendered}"
        );
    }

    #[test]
    fn a_narrowed_slice_needs_only_the_records_kept() {
        let (set, slice) = select(SET);
        let slice = slice.within(&BTreeSet::from([0, 3]));
        assert_eq!(slice.entities(1), []);
        let report = slice
            .measure(&set, Slice::All, 0.0905, &[row(0, "我喜欢周杰伦", None)])
            .expect("only record 0 holds an entity now");
        assert_eq!(report.total().records, 1);
    }
}
