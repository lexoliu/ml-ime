//! The corpus half of the command line: pull the sources down and prepare them.
//!
//! A thin shell over `ime-corpus`, shaped like the Python `mlime corpus` commands
//! it replaces so that a data root prepared by either is prepared the same way.
//! The two verbs sit either side of the network on purpose: `fetch` is the
//! expensive, interruptible half and `prepare` is the half worth re-running every
//! time a filter moves.

use anyhow::{Context as _, Result};
use askama::Template;
use clap::{Args, Subcommand, ValueEnum};
use ime_corpus::download::Hub;
use ime_corpus::ingest::{
    self, Census, Ending, GeneralPass, MinePass, Pass, Plan, Run3, RunReport, Shared,
};
use ime_corpus::leak::EvalGuard;
use ime_corpus::mine::{Category, EntryMatcher};
use ime_corpus::source::{BILIBILI, DIALOGUE, DOUYIN, MOEGIRL, NEWS, SourceSpec, WIKI};
use ime_corpus::upstream::{FineWeb2, FineWeb2Hq, MnbvcForum, Upstream, Yyets};
use ime_corpus::{DataLayout, Normalizer, PrepareReport};
use ime_pinyin::{Lexicon, SyllableTable};
use std::io::Write as _;
use std::path::{Path, PathBuf};

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

/// A streamed source of general text, every passing target of which is kept.
#[derive(Clone, Copy, PartialEq, Eq, Debug, ValueEnum)]
pub enum StreamedName {
    /// FineWeb2-HQ's Mandarin, the top decile of FineWeb-2.
    Fineweb2Hq,
    /// The MNBVC forum boards.
    MnbvcForum,
    /// The `YYeTs` fan subtitles.
    Yyets,
}

/// What every streamed run reads and how far it may go.
#[derive(Debug, Args)]
pub struct StreamArgs {
    /// Root the samples and the run state go under.
    #[arg(long)]
    data_dir: PathBuf,
    /// The most upstream files this run processes.
    #[arg(long)]
    files: usize,
    /// Take the source as only the first this many files in processing order.
    #[arg(long)]
    upstream_files: Option<usize>,
    /// Stop before a download would leave less than this many gigabytes free.
    #[arg(long, default_value_t = 60)]
    floor_gb: u64,
    /// An evaluation or held-out JSON Lines file nothing may leak into; repeat for each.
    #[arg(long = "eval", required = true)]
    evals: Vec<PathBuf>,
    /// run3's samples directory: its targets are not taken twice.
    #[arg(long)]
    run3: PathBuf,
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
    /// Stream a bounded chunk of a web-scale source into samples, resuming.
    Ingest {
        /// Which source to stream.
        #[arg(long)]
        source: StreamedName,
        /// Stop the source once it has written this many targets in all.
        #[arg(long)]
        targets: Option<usize>,
        #[command(flatten)]
        stream: StreamArgs,
    },
    /// Mine a bounded chunk of FineWeb-2 for #215's absent entries, resuming.
    Mine {
        /// #215's entry table: the knowledge lexicon with run3 occurrences, whose absent
        /// unblocked idioms and internet terms are mined for.
        #[arg(long)]
        entries: PathBuf,
        #[command(flatten)]
        stream: StreamArgs,
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
    personal: String,
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
        CorpusCommand::Ingest {
            source,
            targets,
            stream,
        } => match source {
            StreamedName::Fineweb2Hq => ingest_general::<FineWeb2Hq>(&stream, targets).await,
            StreamedName::MnbvcForum => ingest_general::<MnbvcForum>(&stream, targets).await,
            StreamedName::Yyets => ingest_general::<Yyets>(&stream, targets).await,
        },
        CorpusCommand::Mine { entries, stream } => mine(&stream, &entries).await,
    }
}

/// What every streamed run loads before its first file.
struct Loaded {
    lexicon: Lexicon,
    guard: EvalGuard,
    run3: Run3,
    plan: Plan,
}

fn load(stream: &StreamArgs, with_pairs: bool) -> Result<Loaded> {
    anyhow::ensure!(
        stream.files > 0,
        "a run has to be allowed at least one file"
    );
    let table = SyllableTable::load();
    let lexicon = Lexicon::load(&table).context("the generated pinyin tables disagree")?;
    let guard = EvalGuard::load(&stream.evals).context("could not load the evaluation records")?;
    tracing::info!(
        texts = guard.texts(),
        windows = guard.windows(),
        "evaluation guard built"
    );
    let run3 = Run3::load(&stream.run3, with_pairs).context("could not load run3's targets")?;
    Ok(Loaded {
        lexicon,
        guard,
        run3,
        plan: Plan {
            root: stream.data_dir.clone(),
            files: stream.files,
            upstream_files: stream.upstream_files,
            floor_bytes: stream.floor_gb * 1_000_000_000,
        },
    })
}

async fn ingest_general<U: Upstream>(stream: &StreamArgs, targets: Option<usize>) -> Result<()> {
    let loaded = load(stream, false)?;
    let shared = Shared::new(&loaded.lexicon, &loaded.guard, &loaded.run3)?;
    let mut pass = GeneralPass::new(&shared, targets);
    let report = stream_run::<U, _>(&loaded.plan, &mut pass).await?;
    print_ingest(U::SPEC.name, &report)
}

async fn mine(stream: &StreamArgs, entries: &Path) -> Result<()> {
    let matcher = EntryMatcher::load(entries, &Normalizer::new()?)
        .context("could not load the entries to mine for")?;
    tracing::info!(
        targets = matcher.targets().len(),
        unminable = matcher.unminable().len(),
        "mining for absent entries; the normaliser rewrites the unminable ones"
    );
    let loaded = load(stream, true)?;
    let shared = Shared::new(&loaded.lexicon, &loaded.guard, &loaded.run3)?;
    let mut pass = MinePass::new(&shared, &matcher)?;
    let report = stream_run::<FineWeb2, _>(&loaded.plan, &mut pass).await?;
    write_entry_counts(&loaded.plan.root, &matcher, pass.kept())?;
    print_ingest(FineWeb2::SPEC.name, &report)
}

async fn stream_run<U: Upstream, P: Pass>(plan: &Plan, pass: &mut P) -> Result<RunReport> {
    let hub = Hub::new()?;
    ingest::run::<U, P>(&hub, plan, pass)
        .await
        .with_context(|| format!("could not stream the {} source", U::SPEC.name))
}

/// One row of the per-entry table: every target, kept or not.
#[derive(serde::Serialize)]
struct EntryRow<'a> {
    entry: &'a str,
    category: &'static str,
    minable: bool,
    kept: usize,
}

fn category_name(category: Category) -> &'static str {
    match category {
        Category::Idiom => "idiom",
        Category::Internet => "internet",
        Category::Name => "name",
    }
}

fn write_entry_counts(root: &Path, matcher: &EntryMatcher, kept: &[usize]) -> Result<()> {
    let path = root
        .join("ingest")
        .join(FineWeb2::SPEC.name)
        .join("entries.tsv");
    let mut writer = csv::WriterBuilder::new()
        .delimiter(b'\t')
        .from_path(&path)
        .with_context(|| format!("could not write {}", path.display()))?;
    for (target, count) in matcher.targets().iter().zip(kept) {
        writer.serialize(EntryRow {
            entry: &target.entry,
            category: category_name(target.category),
            minable: true,
            kept: *count,
        })?;
    }
    for target in matcher.unminable() {
        writer.serialize(EntryRow {
            entry: &target.entry,
            category: category_name(target.category),
            minable: false,
            kept: 0,
        })?;
    }
    writer
        .flush()
        .with_context(|| format!("could not write {}", path.display()))
}

/// One census line of the ingest summary.
struct CensusRow {
    label: String,
    run: String,
    total: String,
}

/// The ingest summary, as it goes to stdout.
#[derive(Template)]
#[template(path = "corpus_ingest.txt", ext = "txt")]
struct IngestTable {
    source: String,
    ending: String,
    finished: usize,
    available: usize,
    run_files: usize,
    run_gb: String,
    run_seconds: String,
    seconds_per_gb: String,
    rows: Vec<CensusRow>,
    run_yield: String,
    total_yield: String,
}

fn census_rows(run: &Census, total: &Census) -> Vec<CensusRow> {
    let pick = |census: &Census| {
        [
            ("documents", census.documents),
            ("documents normalised", census.normalised_documents),
            ("documents quoting eval3", census.leaked_documents),
            ("sentences", census.units),
            ("personal: phone", census.personal.phone),
            ("personal: id number", census.personal.id_number),
            ("personal: email", census.personal.email),
            ("personal: address", census.personal.address),
            ("candidates", census.considered),
            ("too short", census.too_short_run),
            ("too long", census.too_long_run),
            ("unknown character", census.unknown_character),
            ("duplicate", census.duplicate),
            ("equal to an eval text", census.eval_exact),
            ("already in run3", census.in_run3),
            ("hit sentences", census.hit_sentences),
            ("unusable hits", census.unusable_hits),
            ("spun: idiom density", census.spun_idiom_density),
            ("spun: absent entries", census.spun_absent_entries),
            ("spun: novel pairs", census.spun_novel_pairs),
            ("capped", census.capped),
            ("targets written", census.written),
        ]
    };
    pick(run)
        .into_iter()
        .zip(pick(total))
        .map(|((label, run), (_, total))| CensusRow {
            label: format!("{label:<24}"),
            run: format!("{run:>12}"),
            total: format!("{total:>12}"),
        })
        .collect()
}

#[expect(
    clippy::cast_precision_loss,
    reason = "byte counts and milliseconds are printed to a few significant figures"
)]
fn print_ingest(source: &str, report: &RunReport) -> Result<()> {
    let gigabytes = report.run.downloaded_bytes as f64 / 1e9;
    let seconds = report.wall_ms as f64 / 1000.0;
    let table = IngestTable {
        source: source.to_owned(),
        ending: match report.ending {
            Ending::Exhausted => "every file finished".to_owned(),
            Ending::Chunk => "chunk finished".to_owned(),
            Ending::Budget => "target budget reached".to_owned(),
            Ending::DiskFloor { available, needed } => format!(
                "stopped at the disk floor: {:.1} GB free, the next file needs {:.1} GB",
                available as f64 / 1e9,
                needed as f64 / 1e9
            ),
        },
        finished: report.finished,
        available: report.available,
        run_files: report.run.files,
        run_gb: format!("{gigabytes:.2}"),
        run_seconds: format!("{seconds:.0}"),
        seconds_per_gb: if gigabytes > 0.0 {
            format!("{:.0}", seconds / gigabytes)
        } else {
            "-".to_owned()
        },
        rows: census_rows(&report.run, &report.total),
        run_yield: format!("{:.2}%", report.run.filter_yield() * 100.0),
        total_yield: format!("{:.2}%", report.total.filter_yield() * 100.0),
    };
    let rendered = table.render().context("could not render the summary")?;
    write!(std::io::stdout(), "{rendered}").context("could not write to stdout")
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
        personal: report.personal.total().to_string(),
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
            personal: ime_corpus::privacy::PersonalCounts::default(),
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
