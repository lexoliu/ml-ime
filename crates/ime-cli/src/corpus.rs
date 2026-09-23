//! The corpus half of the command line: pull the sources down and prepare them.
//!
//! A thin shell over `ime-corpus`, shaped like the Python `mlime corpus` commands
//! it replaces so that a data root prepared by either is prepared the same way.
//! The two verbs sit either side of the network on purpose: `fetch` is the
//! expensive, interruptible half and `prepare` is the half worth re-running every
//! time a filter moves.

use anyhow::{Context as _, Result};
use askama::Template;
use clap::{Subcommand, ValueEnum};
use ime_corpus::source::{BILIBILI, DIALOGUE, DOUYIN, MOEGIRL, NEWS, SourceSpec, WIKI};
use ime_corpus::{DataLayout, PrepareReport};
use ime_pinyin::{Lexicon, SyllableTable};
use std::io::Write as _;
use std::path::PathBuf;

/// Which upstream a command runs against.
///
/// The specs themselves live in `ime-corpus`; this exists only so that clap can
/// parse a name, and it converts straight into one.
#[derive(Clone, Copy, PartialEq, Eq, Debug, ValueEnum)]
pub enum SourceName {
    /// Chinese Wikipedia's `20231101.zh` dump.
    Wiki,
    /// The Sina news archive behind `THUCNews`.
    News,
    /// The LCCC cleaned conversation corpus.
    Dialogue,
    /// Moe Girl Pedia's cleaned 2025-10 article dump.
    Moegirl,
    /// Douyin post captions.
    Douyin,
    /// Bilibili comments.
    Bilibili,
}

impl From<SourceName> for SourceSpec {
    fn from(name: SourceName) -> Self {
        match name {
            SourceName::Wiki => WIKI,
            SourceName::News => NEWS,
            SourceName::Dialogue => DIALOGUE,
            SourceName::Moegirl => MOEGIRL,
            SourceName::Douyin => DOUYIN,
            SourceName::Bilibili => BILIBILI,
        }
    }
}

/// Fetch and prepare the internet-authentic corpus sources.
#[derive(Debug, Subcommand)]
pub enum CorpusCommand {
    /// Download one source and write its raw documents.
    ///
    /// Only the three sources this crate fetches itself are downloadable; the
    /// three prose corpora were fetched by the Python pipeline into the same
    /// document schema, and `prepare` reads their shards unchanged.
    Fetch {
        /// Which upstream to pull.
        #[arg(long)]
        source: SourceName,
        /// Root the pipeline reads and writes.
        #[arg(long, default_value = "data")]
        data_dir: PathBuf,
        /// Stop after this many documents.
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Clean, normalise, split and filter one source's documents into samples.
    Prepare {
        /// Which upstream to prepare.
        #[arg(long)]
        source: SourceName,
        /// Root the pipeline reads and writes.
        #[arg(long, default_value = "data")]
        data_dir: PathBuf,
        /// Stop after this many samples.
        #[arg(long)]
        limit: Option<usize>,
    },
}

/// One row of the prepare summary: a filter reason and how much it took.
struct ReasonRow {
    reason: String,
    count: String,
    share: String,
}

/// The prepare summary, as it goes to stdout.
#[derive(Template)]
#[template(path = "corpus_prepare.txt", ext = "txt")]
struct PrepareTable {
    source: String,
    documents: String,
    lines: String,
    infobox_lines: String,
    considered: String,
    rows: Vec<ReasonRow>,
    written: String,
    seconds: String,
    per_second: String,
}

/// Run one `corpus` subcommand.
///
/// # Errors
///
/// Whatever the stage it dispatches to refuses on.
pub async fn run(command: CorpusCommand) -> Result<()> {
    match command {
        CorpusCommand::Fetch {
            source,
            data_dir,
            limit,
        } => {
            let spec = SourceSpec::from(source);
            let layout = DataLayout::new(data_dir);
            ime_corpus::fetch(spec, &layout, limit)
                .await
                .with_context(|| format!("could not fetch the {} source", spec.name))?;
            Ok(())
        }
        CorpusCommand::Prepare {
            source,
            data_dir,
            limit,
        } => {
            let spec = SourceSpec::from(source);
            let layout = DataLayout::new(data_dir);
            let table = SyllableTable::load();
            let lexicon = Lexicon::load(&table).context("the generated pinyin tables disagree")?;
            let report = ime_corpus::prepare(spec, &layout, &lexicon, limit)
                .with_context(|| format!("could not prepare the {} source", spec.name))?;
            print_report(&report)
        }
    }
}

fn print_report(report: &PrepareReport) -> Result<()> {
    let counts = report.counts;
    let considered = counts.considered();
    let rows = [
        ("kept", counts.kept),
        ("too_short_run", counts.too_short_run),
        ("too_long_run", counts.too_long_run),
        ("unknown_character", counts.unknown_character),
        ("duplicate", counts.duplicate),
    ];
    let table = PrepareTable {
        source: report.source.to_owned(),
        documents: report.documents.to_string(),
        lines: report.cleaning.lines.to_string(),
        infobox_lines: report.cleaning.infobox_lines.to_string(),
        considered: considered.to_string(),
        rows: rows
            .into_iter()
            .map(|(reason, count)| ReasonRow {
                reason: format!("{reason:<24}"),
                count: format!("{count:>9}"),
                share: format!("{:>6.2}%", share(count, considered)),
            })
            .collect(),
        written: report.written.to_string(),
        seconds: format!("{:.1}", elapsed_seconds(report)),
        per_second: format!("{:.0}", report.samples_per_second()),
    };
    let rendered = table.render().context("could not render the summary")?;
    write!(std::io::stdout(), "{rendered}").context("could not write to stdout")
}

#[expect(
    clippy::cast_precision_loss,
    reason = "a percentage over corpus-sized counts is printed to two decimals"
)]
fn share(count: usize, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    count as f64 * 100.0 / total as f64
}

#[expect(
    clippy::cast_precision_loss,
    reason = "an elapsed time in milliseconds is printed to one decimal"
)]
fn elapsed_seconds(report: &PrepareReport) -> f64 {
    report.milliseconds as f64 / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use ime_corpus::FilterCounts;
    use ime_corpus::clean::CleaningCounts;

    #[test]
    fn every_source_name_maps_onto_its_spec() {
        let named: Vec<&str> = [
            SourceName::Wiki,
            SourceName::News,
            SourceName::Dialogue,
            SourceName::Moegirl,
            SourceName::Douyin,
            SourceName::Bilibili,
        ]
        .into_iter()
        .map(|name| SourceSpec::from(name).name)
        .collect();
        assert_eq!(
            named,
            vec!["wiki", "news", "dialogue", "moegirl", "douyin", "bilibili"]
        );
        assert_eq!(named.len(), ime_corpus::SOURCES.len());
    }

    #[test]
    fn the_summary_reports_a_share_for_every_filter_reason() {
        let report = PrepareReport {
            source: "moegirl",
            documents: 10,
            cleaning: CleaningCounts {
                lines: 200,
                infobox_lines: 40,
            },
            counts: FilterCounts {
                kept: 50,
                too_short_run: 35,
                too_long_run: 10,
                unknown_character: 3,
                duplicate: 2,
            },
            written: 50,
            milliseconds: 500,
        };
        assert!((report.samples_per_second() - 100.0).abs() < f64::EPSILON);
        assert!((share(50, 100) - 50.0).abs() < f64::EPSILON);
        assert!((share(1, 0) - 0.0).abs() < f64::EPSILON);
        print_report(&report).expect("the summary renders");
    }
}
