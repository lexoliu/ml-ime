//! The synthesis half of the command line: generate slang usage, then read it back.
//!
//! A thin shell over `ime-synth`, with one shape worth explaining. `generate`
//! takes a *seed* root and an *output* root, and they default to being the same
//! place only because that is convenient; in practice they should differ. The
//! seeds live under the main `data/` root beside everything else, while the
//! samples this stage writes are training-only and must never be drawn into an
//! evaluation set -- so pointing `--out-dir` at a root of its own is what keeps
//! model-written sentences out of the numbers the project reports.

use anyhow::{Context as _, Result};
use askama::Template;
use clap::Subcommand;
use ime_g2p::llm::LlmSettings;
use ime_synth::llm::{DEFAULT_CONCURRENCY, DEFAULT_PER_TERM};
use ime_synth::report::Shown;
use ime_synth::summary::RunSummary;
use ime_synth::{Options, generate};
use std::io::Write as _;
use std::path::PathBuf;

/// How many samples the report prints in full.
const SHOWN_SAMPLES: usize = 10;

/// The seed the report's draw uses, so two readings of a run agree.
const SHOWN_SEED: u64 = 0;

/// Generate grounded slang usage sentences, and report on what a run produced.
#[derive(Debug, Subcommand)]
pub enum SynthCommand {
    /// Generate usage sentences for every grounded seed term.
    Generate {
        /// Root the seed lexicons are read from.
        #[arg(long, default_value = "data")]
        data_dir: PathBuf,
        /// Root the samples, provenance and summary are written to. Defaults to
        /// `--data-dir`, but a training-only batch belongs in its own root.
        #[arg(long)]
        out_dir: Option<PathBuf>,
        /// Stop after this many grounded seed terms.
        #[arg(long)]
        terms_limit: Option<usize>,
        /// How many usage examples each term is asked for.
        #[arg(long, default_value_t = DEFAULT_PER_TERM)]
        per_term: usize,
        /// Simultaneous requests in flight to the LLM.
        #[arg(long, default_value_t = DEFAULT_CONCURRENCY)]
        concurrency: usize,
    },
    /// Print what the last generation run under a root produced.
    Report {
        /// Root a run wrote its samples and summary into.
        #[arg(long, default_value = "data")]
        data_dir: PathBuf,
    },
}

/// One seed source's line of the report.
struct SeedRow {
    source: String,
    loaded: String,
    grounded: String,
    ungrounded: String,
    unusable: String,
    duplicate: String,
}

/// One validation reason and how much of the batch it took.
struct DropRow {
    reason: String,
    count: String,
    share: String,
}

/// One seed source's share of what was written.
struct WrittenRow {
    source: String,
    terms: String,
    written: String,
}

/// One refusal reason and how many terms hit it.
struct RefusalRow {
    terms: String,
    reason: String,
}

/// One sample, printed in full so a person can judge whether it reads as usage.
struct SampleRow {
    source: String,
    term: String,
    context: String,
    text: String,
}

/// The synthesis summary, as it goes to stdout.
#[derive(Template)]
#[template(path = "synth_report.txt", ext = "txt")]
struct ReportTable {
    model: String,
    source: String,
    per_term: String,
    concurrency: String,
    seeds: Vec<SeedRow>,
    terms: String,
    requested: String,
    parsed: String,
    refusals: String,
    written: String,
    per_minute: String,
    seconds: String,
    drops: Vec<DropRow>,
    written_by: Vec<WrittenRow>,
    refusal_reasons: Vec<RefusalRow>,
    shown: String,
    samples: Vec<SampleRow>,
}

/// Run one `synth` subcommand.
///
/// # Errors
///
/// Whatever the stage it dispatches to refuses on.
pub async fn run(command: SynthCommand) -> Result<()> {
    match command {
        SynthCommand::Generate {
            data_dir,
            out_dir,
            terms_limit,
            per_term,
            concurrency,
        } => {
            let out_dir = out_dir.unwrap_or_else(|| data_dir.clone());
            let settings =
                LlmSettings::load().context("the MLIME_LLM_* settings are incomplete")?;
            let summary = generate(
                &data_dir,
                &out_dir,
                &settings,
                Options {
                    terms_limit,
                    per_term,
                    concurrency,
                },
            )
            .await
            .context("the synthesis run failed")?;
            print_report(&summary, &ime_synth::report::shown(&out_dir)?)
        }
        SynthCommand::Report { data_dir } => {
            let summary = RunSummary::read(&data_dir).context("no run has written a summary")?;
            print_report(&summary, &ime_synth::report::shown(&data_dir)?)
        }
    }
}

/// Render one run's summary and a draw of its samples to stdout.
fn print_report(summary: &RunSummary, shown: &[Shown]) -> Result<()> {
    let considered = summary.drops.considered();
    let samples = ime_synth::report::draw(shown, SHOWN_SAMPLES, SHOWN_SEED);
    let table = ReportTable {
        model: summary.model.clone(),
        source: summary.source.clone(),
        per_term: summary.per_term.to_string(),
        concurrency: summary.concurrency.to_string(),
        seeds: summary
            .seeds
            .iter()
            .map(|row| SeedRow {
                source: format!("{:<14}", row.source),
                loaded: format!("{:>6}", row.loaded),
                grounded: format!("{:>8}", row.grounded),
                ungrounded: format!("{:>10}", row.skipped_ungrounded),
                unusable: format!("{:>8}", row.skipped_unusable_term),
                duplicate: format!("{:>9}", row.skipped_duplicate),
            })
            .collect(),
        terms: summary.terms.to_string(),
        requested: summary.examples_requested.to_string(),
        parsed: summary.examples_parsed.to_string(),
        refusals: summary.refusals.to_string(),
        written: summary.written.to_string(),
        per_minute: format!("{:.1}", summary.terms_per_minute()),
        seconds: format!("{:.1}", elapsed_seconds(summary)),
        drops: std::iter::once(("kept", summary.drops.kept))
            .chain(summary.drops.reasons())
            .map(|(reason, count)| DropRow {
                reason: format!("{reason:<24}"),
                count: format!("{count:>9}"),
                share: format!("{:>6.2}%", share(count, considered)),
            })
            .collect(),
        written_by: summary
            .written_by_source
            .iter()
            .map(|row| WrittenRow {
                source: format!("{:<14}", row.source),
                terms: format!("{:>6}", row.terms),
                written: format!("{:>7}", row.written),
            })
            .collect(),
        refusal_reasons: summary
            .refusal_reasons
            .iter()
            .map(|row| RefusalRow {
                terms: format!("{:>5}", row.terms),
                reason: row.reason.clone(),
            })
            .collect(),
        shown: samples.len().to_string(),
        samples: samples
            .into_iter()
            .map(|row| SampleRow {
                source: row.seed_source,
                term: row.term,
                context: row.context,
                text: row.text,
            })
            .collect(),
    };
    let rendered = table.render().context("could not render the summary")?;
    write!(std::io::stdout(), "{rendered}").context("could not write to stdout")
}

#[expect(
    clippy::cast_precision_loss,
    reason = "a percentage over batch-sized counts is printed to two decimals"
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
fn elapsed_seconds(summary: &RunSummary) -> f64 {
    summary.milliseconds as f64 / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use ime_synth::DropCounts;
    use ime_synth::SeedCounts;
    use ime_synth::summary::{RefusalCount, WrittenBySource};

    fn summary() -> RunSummary {
        RunSummary {
            source: ime_synth::SYNTHETIC.to_owned(),
            model: "gpt-5.6-luna".to_owned(),
            per_term: 5,
            concurrency: 32,
            seeds: vec![
                SeedCounts {
                    source: "sogou-premium".to_owned(),
                    loaded: 128_149,
                    grounded: 30,
                    skipped_ungrounded: 128_113,
                    skipped_unusable_term: 6,
                    skipped_duplicate: 0,
                },
                SeedCounts {
                    source: "wiki-slang".to_owned(),
                    loaded: 378,
                    grounded: 300,
                    skipped_ungrounded: 0,
                    skipped_unusable_term: 42,
                    skipped_duplicate: 36,
                },
            ],
            terms: 30,
            examples_requested: 150,
            examples_parsed: 145,
            refusals: 1,
            refusal_reasons: vec![RefusalCount {
                reason: "ValueError: got 4 examples where 5 were asked for".to_owned(),
                terms: 1,
            }],
            drops: DropCounts {
                kept: 140,
                missing_term: 3,
                duplicate: 2,
                ..DropCounts::default()
            },
            written: 140,
            written_by_source: vec![WrittenBySource {
                source: "sogou-premium".to_owned(),
                terms: 30,
                written: 140,
            }],
            milliseconds: 60_000,
        }
    }

    #[test]
    fn the_report_prints_every_seed_source_reason_and_sample() {
        let shown = vec![Shown {
            term: "爷青结".to_owned(),
            seed_source: "wiki-slang".to_owned(),
            context: "你追完了吗".to_owned(),
            text: "这季追完了我直接爷青结".to_owned(),
        }];
        print_report(&summary(), &shown).expect("the summary renders");
    }

    #[test]
    fn a_run_with_no_refusals_still_renders() {
        let clean = RunSummary {
            refusals: 0,
            refusal_reasons: Vec::new(),
            ..summary()
        };
        print_report(&clean, &[]).expect("the summary renders");
    }

    #[test]
    fn a_share_of_nothing_is_zero_rather_than_a_division_by_zero() {
        assert!((share(1, 0)).abs() < f64::EPSILON);
        assert!((share(50, 100) - 50.0).abs() < f64::EPSILON);
    }
}
