//! Command line driver for the input method engine.

mod engine;
mod g2p;

use anyhow::{Context as _, Result};
use askama::Template;
use clap::{Args, Parser, Subcommand};
use engine::Baseline;
use g2p::{ExportCommand, G2pCommand};
use ime_decode::BeamOptions;
use ime_eval::{EvalSet, evaluate};
use ime_ngram::{Counter, NgramModel};
use ime_pinyin::{Lexicon, SegmentOptions, SyllableTable};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use tracing::info;
use tracing_subscriber::EnvFilter;

/// Train, run and measure the pinyin input method.
#[derive(Debug, Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Estimate a Kneser-Ney trigram from a plain text corpus.
    TrainNgram {
        /// UTF-8 text, one document per line. Anything outside the character
        /// lexicon is a sequence break.
        #[arg(long)]
        corpus: PathBuf,
        /// Where to write the model.
        #[arg(long)]
        out: PathBuf,
    },
    /// Decode keystrokes and print the ranked candidates.
    Decode {
        /// A model written by `train-ngram`.
        #[arg(long)]
        model: PathBuf,
        /// The keystrokes, lowercase `[a-z]`.
        pinyin: String,
        #[command(flatten)]
        search: SearchArgs,
    },
    /// Dual pinyin annotation and its agreement report.
    G2p {
        #[command(subcommand)]
        command: G2pCommand,
    },
    /// Emit artefacts for the rest of the engine.
    Export {
        #[command(subcommand)]
        command: ExportCommand,
    },
    /// Run a model over an evaluation set and print the report.
    Eval {
        /// A model written by `train-ngram`.
        #[arg(long)]
        model: PathBuf,
        /// A JSON Lines evaluation set.
        #[arg(long)]
        eval_set: PathBuf,
        #[command(flatten)]
        search: SearchArgs,
    },
}

/// The search knobs, shared by every command that decodes.
#[derive(Debug, Clone, Args)]
struct SearchArgs {
    /// How many candidates to produce.
    #[arg(long, default_value = "8")]
    top_k: NonZeroUsize,
    /// How many beam states survive at each character position.
    #[arg(long, default_value = "16")]
    beam_width: NonZeroUsize,
    /// How many readings of the keystrokes to decode.
    #[arg(long, default_value = "8")]
    max_paths: NonZeroUsize,
    /// How heavily an unconventional segmentation is penalised.
    #[arg(long, default_value = "1.0")]
    segmentation_weight: f32,
    /// Accept a trailing half-typed syllable, as a live IME must. Off by
    /// default, so that an offline run cannot hide a wrong answer behind one.
    #[arg(long)]
    incomplete_tail: bool,
}

impl SearchArgs {
    fn segment(&self) -> SegmentOptions {
        SegmentOptions {
            max_paths: self.max_paths.get(),
            allow_incomplete_tail: self.incomplete_tail,
            ..SegmentOptions::default()
        }
    }

    fn beam(&self) -> BeamOptions {
        BeamOptions {
            beam_width: self.beam_width,
            top_k: self.top_k,
            segmentation_weight: self.segmentation_weight,
        }
    }
}

/// One line of `decode` output.
struct Candidate {
    text: String,
    score: f32,
}

/// The ranked candidates, as they go to stdout.
#[derive(Template)]
#[template(path = "candidates.txt", ext = "txt")]
struct CandidateList {
    candidates: Vec<Candidate>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    match Cli::parse().command {
        Command::TrainNgram { corpus, out } => train_ngram(&corpus, &out),
        Command::Decode {
            model,
            pinyin,
            search,
        } => decode(&model, &pinyin, &search),
        Command::Eval {
            model,
            eval_set,
            search,
        } => run_eval(&model, &eval_set, &search),
        Command::G2p { command } => g2p::run(command).await,
        Command::Export { command } => g2p::run_export(command),
    }
}

/// The syllable inventory and the character lexicon, which every command needs.
fn tables() -> Result<(SyllableTable, Lexicon)> {
    let table = SyllableTable::load();
    let lexicon = Lexicon::load(&table).context("the generated pinyin tables disagree")?;
    Ok((table, lexicon))
}

fn train_ngram(corpus: &Path, out: &Path) -> Result<()> {
    let (_, lexicon) = tables()?;
    let file = fs::File::open(corpus)
        .with_context(|| format!("could not open the corpus at {}", corpus.display()))?;
    let mut counter = Counter::new(&lexicon).context("the lexicon is too large to train on")?;
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line =
            line.with_context(|| format!("could not read line {} of the corpus", index + 1))?;
        counter.observe(&line);
    }
    info!(
        lines = counter.lines(),
        trigrams = counter.trigram_types(),
        "counted the corpus"
    );
    let model = counter.finish().context("could not estimate the model")?;
    let bytes = model.to_bytes().context("could not serialise the model")?;
    fs::write(out, &bytes)
        .with_context(|| format!("could not write the model to {}", out.display()))?;
    info!(
        bytes = bytes.len(),
        trigrams = model.trigram_types(),
        bigrams = model.bigram_types(),
        path = %out.display(),
        "wrote the model"
    );
    Ok(())
}

fn load_baseline(model: &Path, search: &SearchArgs) -> Result<Baseline> {
    let (table, lexicon) = tables()?;
    let bytes = fs::read(model)
        .with_context(|| format!("could not read the model at {}", model.display()))?;
    let model = NgramModel::from_bytes(&bytes, &lexicon)
        .context("the model does not match this character lexicon")?;
    info!(
        vocabulary = model.vocabulary_size(),
        trigrams = model.trigram_types(),
        "loaded the model"
    );
    Ok(Baseline::new(
        table,
        lexicon,
        model,
        search.segment(),
        search.beam(),
    ))
}

fn decode(model: &Path, pinyin: &str, search: &SearchArgs) -> Result<()> {
    let baseline = load_baseline(model, search)?;
    let hypotheses = baseline
        .candidates(pinyin, search.top_k)
        .with_context(|| format!("could not decode {pinyin:?}"))?;
    let list = CandidateList {
        candidates: hypotheses
            .iter()
            .map(|hypothesis| Candidate {
                text: hypothesis.text(baseline.lexicon()),
                score: hypothesis.score(),
            })
            .collect(),
    };
    let rendered = list.render().context("could not render the candidates")?;
    write!(std::io::stdout(), "{rendered}").context("could not write to stdout")
}

fn run_eval(model: &Path, eval_set: &Path, search: &SearchArgs) -> Result<()> {
    let source = fs::read_to_string(eval_set)
        .with_context(|| format!("could not read the eval set at {}", eval_set.display()))?;
    let set = EvalSet::parse(&source).context("the eval set is malformed")?;
    info!(
        records = set.len(),
        with_context = set.with_context(),
        "loaded the eval set; the n-gram baseline ignores context"
    );
    let baseline = load_baseline(model, search)?;
    let report = evaluate(&set, &baseline, search.top_k).context("the baseline failed a record")?;
    write!(std::io::stdout(), "{report}").context("could not write to stdout")
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory as _;

    #[test]
    fn the_command_line_is_well_formed() {
        Cli::command().debug_assert();
    }

    #[test]
    fn the_search_knobs_reach_the_options() {
        let cli = Cli::parse_from([
            "ime-cli",
            "decode",
            "--model",
            "model.bin",
            "--top-k",
            "3",
            "--beam-width",
            "32",
            "--max-paths",
            "4",
            "--segmentation-weight",
            "0.5",
            "nihao",
        ]);
        let Command::Decode { pinyin, search, .. } = cli.command else {
            panic!("expected the decode subcommand");
        };
        assert_eq!(pinyin, "nihao");
        assert_eq!(search.beam().beam_width.get(), 32);
        assert_eq!(search.beam().top_k.get(), 3);
        assert_eq!(search.segment().max_paths, 4);
        assert!(!search.segment().allow_incomplete_tail);
        assert!((search.beam().segmentation_weight - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn the_candidate_list_renders_one_ranked_line_each() {
        let list = CandidateList {
            candidates: vec![
                Candidate {
                    text: "中国".to_owned(),
                    score: -1.5,
                },
                Candidate {
                    text: "钟国".to_owned(),
                    score: -9.25,
                },
            ],
        };
        assert_eq!(
            list.render().expect("the template renders"),
            "1. 中国  -1.500\n2. 钟国  -9.250\n"
        );
    }
}
