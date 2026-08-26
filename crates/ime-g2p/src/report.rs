//! Summarise a dual-annotation run: the measured ceiling on every later number.
//!
//! Where g2pW and the LLM disagree, one of them is wrong, and there is no way to
//! tell which without a human. Those sentences are excluded from training, so the
//! disagreement rate is a direct loss of data -- and because the same annotation
//! built the evaluation set, it is also the accuracy no model can be shown to
//! exceed. Reporting it per frequency band matters because rare characters are
//! where disagreement concentrates, and rare characters are a small share of
//! tokens but a large share of the sentences a user notices going wrong.

use crate::annotate::{ANNOTATED, AnnotatedRow, REFUSED, RefusedRow, read_annotated};
use crate::error::Result;
use crate::shards::{read_frame, shard_paths};
use askama::Template;
use std::collections::BTreeMap;
use std::path::Path;

/// Rank boundaries between character-frequency bands, most frequent first.
pub const FREQUENCY_BREAKS: [usize; 3] = [500, 1500, 3500];

/// The band names, in the same order as the breaks that separate them.
pub const BUCKET_LABELS: [&str; 4] = ["top 500", "501-1500", "1501-3500", "rarer"];

/// How many disagreeing characters the report lists.
pub const TOP_DISAGREEMENTS: usize = 30;

/// One character position of one sentence, carrying both readings and the verdict.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Position<'a> {
    /// The Han character at this position.
    pub character: &'a str,
    /// g2pW's reading of it.
    pub g2pw: &'a str,
    /// The LLM's reading of it.
    pub llm: &'a str,
    /// Whether the two agree once the tone is dropped.
    pub agree: bool,
    /// The sentence it came from, kept as the example a disagreement is shown with.
    pub sentence: &'a str,
}

/// Explode annotated rows into one entry per character position.
#[must_use]
pub fn positions(annotated: &[AnnotatedRow]) -> Vec<Position<'_>> {
    let mut all = Vec::new();
    for row in annotated {
        for index in 0..row.characters.len() {
            let (Some(character), Some(g2pw), Some(llm), Some(agree)) = (
                row.characters.get(index),
                row.g2pw.get(index),
                row.llm.get(index),
                row.agree.get(index),
            ) else {
                continue;
            };
            all.push(Position {
                character,
                g2pw,
                llm,
                agree: *agree,
                sentence: &row.text,
            });
        }
    }
    all
}

/// The headline numbers of one annotation run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AgreementSummary {
    /// How many sentences both annotators labelled.
    pub sentences: usize,
    /// How many character positions those sentences hold.
    pub characters: usize,
    /// How many sentences both read identically end to end.
    pub sentences_agreed: usize,
    /// How many positions the two spell the same way.
    pub characters_agreed: usize,
    /// How many sentence-annotator pairs produced nothing.
    pub refusals: usize,
}

impl AgreementSummary {
    /// Share of sentences both annotators read identically end to end.
    #[must_use]
    pub fn sentence_rate(&self) -> f64 {
        ratio(self.sentences_agreed, self.sentences)
    }

    /// Share of character positions the two annotators spell the same way.
    #[must_use]
    pub fn character_rate(&self) -> f64 {
        ratio(self.characters_agreed, self.characters)
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "a rate over corpus-sized counts is reported to two decimals"
)]
fn ratio(part: usize, whole: usize) -> f64 {
    if whole == 0 {
        return 0.0;
    }
    part as f64 / whole as f64
}

/// Count sentences and character positions, agreed and total.
#[must_use]
pub fn summarise(annotated: &[AnnotatedRow], refusals: usize) -> AgreementSummary {
    let positions = positions(annotated);
    AgreementSummary {
        sentences: annotated.len(),
        characters: positions.len(),
        sentences_agreed: annotated.iter().filter(|row| row.agree_all).count(),
        characters_agreed: positions.iter().filter(|entry| entry.agree).count(),
        refusals,
    }
}

/// Agreement within one character-frequency band.
#[derive(Clone, PartialEq, Debug)]
pub struct Band {
    /// Which band, most frequent first.
    pub label: &'static str,
    /// How many distinct characters fall in it.
    pub distinct: usize,
    /// How many character positions those account for.
    pub positions: usize,
    /// The share of those positions the annotators agree on.
    pub rate: f64,
}

/// Agreement rate per character-frequency band, most frequent band first.
///
/// Characters are ranked by how many positions they occupy, ties broken by the
/// character itself so that two runs over the same data band them the same way.
#[must_use]
pub fn by_frequency(positions: &[Position<'_>]) -> Vec<Band> {
    let mut counts: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for entry in positions {
        let slot = counts.entry(entry.character).or_insert((0, 0));
        slot.0 += 1;
        slot.1 += usize::from(entry.agree);
    }
    let mut ranked: Vec<(&str, usize, usize)> = counts
        .into_iter()
        .map(|(character, (total, agreed))| (character, total, agreed))
        .collect();
    ranked.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(right.0)));

    let mut bands: Vec<Band> = BUCKET_LABELS
        .iter()
        .map(|label| Band {
            label,
            distinct: 0,
            positions: 0,
            rate: 0.0,
        })
        .collect();
    let mut agreed = vec![0_usize; BUCKET_LABELS.len()];
    for (index, (_, total, agree)) in ranked.into_iter().enumerate() {
        let band = band_of(index + 1);
        bands[band].distinct += 1;
        bands[band].positions += total;
        agreed[band] += agree;
    }
    for (band, agreed) in bands.iter_mut().zip(agreed) {
        band.rate = ratio(agreed, band.positions);
    }
    bands.retain(|band| band.positions > 0);
    bands
}

/// Which band a one-based frequency rank falls in. The breaks are right-closed.
fn band_of(rank: usize) -> usize {
    FREQUENCY_BREAKS
        .iter()
        .position(|breakpoint| rank <= *breakpoint)
        .unwrap_or(FREQUENCY_BREAKS.len())
}

/// A character the annotators fight over, with one example.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Disagreement<'a> {
    /// The character.
    pub character: &'a str,
    /// How many positions it disagreed at.
    pub count: usize,
    /// What g2pW said, the first time.
    pub g2pw: &'a str,
    /// What the LLM said, the first time.
    pub llm: &'a str,
    /// The sentence that first time came from.
    pub sentence: &'a str,
}

/// The characters the annotators fight over most, with one example each.
#[must_use]
pub fn worst_characters<'a>(positions: &[Position<'a>], limit: usize) -> Vec<Disagreement<'a>> {
    let mut found: Vec<Disagreement<'a>> = Vec::new();
    let mut seen: BTreeMap<&'a str, usize> = BTreeMap::new();
    for entry in positions.iter().filter(|entry| !entry.agree) {
        if let Some(index) = seen.get(entry.character) {
            found[*index].count += 1;
        } else {
            seen.insert(entry.character, found.len());
            found.push(Disagreement {
                character: entry.character,
                count: 1,
                g2pw: entry.g2pw,
                llm: entry.llm,
                sentence: entry.sentence,
            });
        }
    }
    found.sort_by(|left, right| {
        right
            .count
            .cmp(&left.count)
            .then_with(|| left.character.cmp(right.character))
    });
    found.truncate(limit);
    found
}

/// One row of the headline table.
pub struct Measure {
    /// The name, padded to the column width.
    pub measure: String,
    /// The value, right-aligned in its column.
    pub value: String,
}

/// One row of the frequency table, each cell already padded.
pub struct BandRow {
    /// The band name.
    pub band: String,
    /// How many distinct characters.
    pub distinct: String,
    /// How many positions.
    pub positions: String,
    /// The agreement rate.
    pub rate: String,
}

/// One row of the disagreement table, each cell already padded.
pub struct WorstRow {
    /// The character.
    pub character: String,
    /// How often it disagreed.
    pub count: String,
    /// g2pW's reading.
    pub g2pw: String,
    /// The LLM's reading.
    pub llm: String,
    /// The example sentence.
    pub sentence: String,
}

/// The three tables that make up the report, as they go to stdout.
#[derive(Template)]
#[template(path = "g2p_report.txt", ext = "txt")]
pub struct Report {
    headline: Vec<Measure>,
    band_column: String,
    distinct_column: String,
    positions_column: String,
    rate_column: String,
    bands: Vec<BandRow>,
    char_column: String,
    count_column: String,
    g2pw_column: String,
    llm_column: String,
    worst: Vec<WorstRow>,
    worst_limit: usize,
}

impl Report {
    /// Build the report from an annotation run's rows and its refusal count.
    #[must_use]
    pub fn new(annotated: &[AnnotatedRow], refusals: usize) -> Self {
        let summary = summarise(annotated, refusals);
        let positions = positions(annotated);
        let bands = by_frequency(&positions);
        let worst = worst_characters(&positions, TOP_DISAGREEMENTS);

        let measures = [
            ("Sentences annotated", thousands(summary.sentences)),
            (
                "Sentences fully agreed",
                thousands(summary.sentences_agreed),
            ),
            ("Sentence agreement", percent(summary.sentence_rate())),
            ("Character positions", thousands(summary.characters)),
            ("Character agreement", percent(summary.character_rate())),
            ("Refusals recorded", thousands(summary.refusals)),
        ];
        let measure_width = measures
            .iter()
            .map(|(name, _)| name.len())
            .max()
            .unwrap_or(0);
        let value_width = measures
            .iter()
            .map(|(_, value)| value.len())
            .max()
            .unwrap_or(0);

        let band_width = width("Band", bands.iter().map(|band| band.label.len()));
        let distinct_cells: Vec<String> =
            bands.iter().map(|band| thousands(band.distinct)).collect();
        let position_cells: Vec<String> =
            bands.iter().map(|band| thousands(band.positions)).collect();
        let rate_cells: Vec<String> = bands.iter().map(|band| percent(band.rate)).collect();
        let distinct_width = width("Distinct chars", distinct_cells.iter().map(String::len));
        let position_width = width("Positions", position_cells.iter().map(String::len));
        let rate_width = width("Agreement", rate_cells.iter().map(String::len));

        let count_cells: Vec<String> = worst.iter().map(|entry| thousands(entry.count)).collect();
        let char_width = width(
            "Char",
            worst.iter().map(|entry| display_width(entry.character)),
        );
        let count_width = width("Count", count_cells.iter().map(String::len));
        let g2pw_width = width("g2pW", worst.iter().map(|entry| entry.g2pw.len()));
        let llm_width = width("LLM", worst.iter().map(|entry| entry.llm.len()));

        Self {
            headline: measures
                .into_iter()
                .map(|(name, value)| Measure {
                    measure: format!("{name:measure_width$}"),
                    value: format!("{value:>value_width$}"),
                })
                .collect(),
            band_column: format!("{:band_width$}", "Band"),
            distinct_column: format!("{:>distinct_width$}", "Distinct chars"),
            positions_column: format!("{:>position_width$}", "Positions"),
            rate_column: format!("{:>rate_width$}", "Agreement"),
            bands: bands
                .iter()
                .zip(distinct_cells)
                .zip(position_cells)
                .zip(rate_cells)
                .map(|(((band, distinct), positions), rate)| BandRow {
                    band: format!("{:band_width$}", band.label),
                    distinct: format!("{distinct:>distinct_width$}"),
                    positions: format!("{positions:>position_width$}"),
                    rate: format!("{rate:>rate_width$}"),
                })
                .collect(),
            char_column: format!("{:char_width$}", "Char"),
            count_column: format!("{:>count_width$}", "Count"),
            g2pw_column: format!("{:g2pw_width$}", "g2pW"),
            llm_column: format!("{:llm_width$}", "LLM"),
            worst: worst
                .iter()
                .zip(count_cells)
                .map(|(entry, count)| WorstRow {
                    character: pad(entry.character, char_width),
                    count: format!("{count:>count_width$}"),
                    g2pw: format!("{:g2pw_width$}", entry.g2pw),
                    llm: format!("{:llm_width$}", entry.llm),
                    sentence: entry.sentence.to_owned(),
                })
                .collect(),
            worst_limit: TOP_DISAGREEMENTS,
        }
    }
}

fn width(header: &str, cells: impl Iterator<Item = usize>) -> usize {
    cells
        .chain(std::iter::once(header.len()))
        .max()
        .unwrap_or(0)
}

/// How many terminal columns a Han character takes, which is two, not one.
fn display_width(text: &str) -> usize {
    text.chars()
        .map(|character| if crate::text::is_han(character) { 2 } else { 1 })
        .sum()
}

fn pad(text: &str, to: usize) -> String {
    let mut padded = text.to_owned();
    for _ in display_width(text)..to {
        padded.push(' ');
    }
    padded
}

/// Group a count into thousands, the way the Python report's `{:,}` does.
fn thousands(value: usize) -> String {
    let digits = value.to_string();
    let mut grouped = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            grouped.push(',');
        }
        grouped.push(digit);
    }
    grouped
}

/// A rate as a percentage to two decimals, the way the Python report's `{:.2%}` does.
fn percent(rate: f64) -> String {
    format!("{:.2}%", rate * 100.0)
}

/// Read an annotation directory and build its report.
///
/// # Errors
///
/// If the directory holds no annotated shards, or one cannot be read.
pub fn report(directory: &Path) -> Result<Report> {
    let mut refusals = 0;
    for path in shard_paths(directory, REFUSED)? {
        refusals += read_frame(&path)?.height();
    }
    let annotated: Vec<AnnotatedRow> = read_annotated(directory, ANNOTATED)?;
    Ok(Report::new(&annotated, refusals))
}

/// The refusal rows, for callers that want to show why a run lost sentences.
///
/// # Errors
///
/// If a refusal shard cannot be read.
pub fn refusals(directory: &Path) -> Result<Vec<RefusedRow>> {
    let mut rows = Vec::new();
    for path in shard_paths(directory, REFUSED)? {
        rows.extend(<RefusedRow as crate::shards::Shardable>::from_frame(
            &read_frame(&path)?,
        )?);
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(text: &str, g2pw: &[&str], llm: &[&str]) -> AnnotatedRow {
        let agree: Vec<bool> = g2pw
            .iter()
            .zip(llm)
            .map(|(left, right)| crate::text::toneless(left) == crate::text::toneless(right))
            .collect();
        AnnotatedRow {
            id: text.to_owned(),
            source: "wiki".to_owned(),
            text: text.to_owned(),
            context: None,
            characters: crate::text::han_characters(text)
                .iter()
                .map(ToString::to_string)
                .collect(),
            g2pw: g2pw.iter().map(|s| (*s).to_owned()).collect(),
            llm: llm.iter().map(|s| (*s).to_owned()).collect(),
            agree_all: agree.iter().all(|flag| *flag),
            agree,
        }
    }

    #[test]
    fn the_summary_counts_sentences_and_positions_separately() {
        let rows = vec![
            row("中国", &["zhong1", "guo2"], &["zhong1", "guo2"]),
            row("重要", &["chong2", "yao4"], &["zhong4", "yao4"]),
        ];
        let summary = summarise(&rows, 3);
        assert_eq!(summary.sentences, 2);
        assert_eq!(summary.sentences_agreed, 1);
        assert_eq!(summary.characters, 4);
        assert_eq!(summary.characters_agreed, 3);
        assert_eq!(summary.refusals, 3);
        assert!((summary.sentence_rate() - 0.5).abs() < f64::EPSILON);
        assert!((summary.character_rate() - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn the_worst_characters_are_ranked_by_how_often_they_disagree() {
        let rows = vec![
            row("重要", &["chong2", "yao4"], &["zhong4", "yao4"]),
            row("重来", &["chong2", "lai2"], &["zhong4", "lai2"]),
            row("得到", &["de5", "dao4"], &["dei3", "dao4"]),
        ];
        let positions = positions(&rows);
        let worst = worst_characters(&positions, 10);
        assert_eq!(worst.len(), 2);
        assert_eq!(worst[0].character, "重");
        assert_eq!(worst[0].count, 2);
        assert_eq!(worst[0].g2pw, "chong2");
        assert_eq!(worst[0].llm, "zhong4");
        assert_eq!(worst[1].character, "得");
    }

    #[test]
    fn the_frequency_breaks_are_right_closed() {
        assert_eq!(band_of(1), 0);
        assert_eq!(band_of(500), 0);
        assert_eq!(band_of(501), 1);
        assert_eq!(band_of(1500), 1);
        assert_eq!(band_of(1501), 2);
        assert_eq!(band_of(3500), 2);
        assert_eq!(band_of(3501), 3);
    }

    #[test]
    fn counts_are_grouped_into_thousands() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(1_000), "1,000");
        assert_eq!(thousands(1_234_567), "1,234,567");
    }

    #[test]
    fn rates_are_percentages_to_two_decimals() {
        assert_eq!(percent(0.5), "50.00%");
        assert_eq!(percent(0.987_65), "98.77%");
    }

    #[test]
    fn the_report_renders_all_three_tables() {
        let rows = vec![
            row("中国", &["zhong1", "guo2"], &["zhong1", "guo2"]),
            row("重要", &["chong2", "yao4"], &["zhong4", "yao4"]),
        ];
        let rendered = Report::new(&rows, 1).render().expect("the report renders");
        assert!(rendered.starts_with("g2p dual annotation\n"), "{rendered}");
        assert!(rendered.contains("Sentence agreement"));
        assert!(rendered.contains("50.00%"));
        assert!(rendered.contains("Agreement by character frequency"));
        assert!(rendered.contains("top 500"));
        assert!(rendered.contains("Top 30 disagreeing characters"));
        assert!(rendered.contains("chong2"));
    }
}
