//! Command line driver for the input method engine.

mod corpus;
mod engine;
mod g2p;
mod neural;
mod synth;

use anyhow::{Context as _, Result};
use askama::Template;
use clap::{Args, Parser, Subcommand};
use corpus::CorpusCommand;
use engine::Baseline;
use g2p::{ExportCommand, G2pCommand};
use ime_decode::BeamOptions;
use ime_eval::{EvalSet, evaluate};
use ime_lm::CharLm;
use ime_neural::RouteA;
use ime_ngram::{Counter, NgramModel};
use ime_pinyin::{Lexicon, SegmentOptions, SyllableTable};
use neural::{Models, SliceArgs, parse_weight};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use synth::SynthCommand;
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
    /// Fetch and prepare the internet-authentic corpus sources.
    Corpus {
        #[command(subcommand)]
        command: CorpusCommand,
    },
    /// Dual pinyin annotation and its agreement report.
    G2p {
        #[command(subcommand)]
        command: G2pCommand,
    },
    /// Grounded LLM synthesis of slang usage sentences, for training only.
    Synth {
        #[command(subcommand)]
        command: SynthCommand,
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
    /// Write the lattice an evaluation set decodes into, for the neural model to
    /// score.
    EmitLattice {
        /// A JSON Lines evaluation set.
        #[arg(long)]
        eval_set: PathBuf,
        /// Where the JSON Lines lattice goes.
        #[arg(long)]
        out: PathBuf,
        /// The characters the model can emit, one per line, as
        /// `mlime train emittable` writes them.
        #[arg(long)]
        emittable: PathBuf,
        #[command(flatten)]
        search: SearchArgs,
    },
    /// Decode an evaluation set with the neural emissions fused into the beam.
    FusedEval {
        /// A model written by `train-ngram`. Required unless the run drops the
        /// transition or decodes with the character model alone, and refused
        /// when it drops the transition: a model that would go unused is a
        /// mistyped command, not a request.
        #[arg(
            long,
            required_unless_present_any = ["no_transition", "lm"],
            conflicts_with = "no_transition"
        )]
        model: Option<PathBuf>,
        /// A directory holding `charlm.onnx` and `charlm.json` from `mlime export
        /// char-lm`: the recurrent character model as the transition, alone or
        /// fused with the trigram when `--model` is given too.
        #[arg(long, conflicts_with = "no_transition")]
        lm: Option<PathBuf>,
        /// Weight on the character model's scores when it is fused with the
        /// trigram; alone, it is scored at one.
        #[arg(long, default_value = "1.0", requires = "lm")]
        lm_weight: f32,
        /// The same JSON Lines evaluation set the lattice was emitted from.
        #[arg(long)]
        eval_set: PathBuf,
        /// A gzipped score file written by `mlime train emit`. Omitted, the run
        /// is the n-gram baseline over the same slice.
        #[arg(long, conflicts_with = "route_a")]
        scores: Option<PathBuf>,
        /// A directory written by `mlime export route-a`: the emissions are
        /// computed live by the towers instead of read from a score file.
        #[arg(long)]
        route_a: Option<PathBuf>,
        /// Score the records without their contexts: every record's gate is
        /// zeroed, the `context off` score file's live twin.
        #[arg(long, requires = "route_a")]
        no_context: bool,
        /// Intra-op threads for the towers' sessions on the per-record path;
        /// a lockstep run (a GPU backend or `--batch` above one) takes all of
        /// them either way.
        #[arg(long, default_value = "1", requires = "route_a")]
        route_a_threads: NonZeroUsize,
        /// Drop the n-gram and decode on the emissions alone.
        #[arg(long)]
        no_transition: bool,
        /// The same emittable set the lattice was written with.
        #[arg(long)]
        emittable: PathBuf,
        /// What a candidate the model has no output row for scores. Must match
        /// the floor the score file was written under.
        #[arg(long, default_value = "-30.0")]
        unscored: f32,
        /// A fusion weight to score at; repeatable, which is how the weight is
        /// tuned on the dev slice in one pass.
        #[arg(long, value_parser = parse_weight, default_values_t = [1.0f32])]
        weight: Vec<f32>,
        /// A directory the beam is dumped to: for every section the run
        /// evaluates, one JSON Lines file named for the section, one record's
        /// hypotheses and their scores per line. The report is unchanged.
        #[arg(long)]
        dump: Option<PathBuf>,
        /// A directory the run's per-record progress is kept in: every
        /// configuration it evaluates keeps a JSON Lines file there, so a
        /// stopped run resumes where it left off instead of decoding the
        /// slice again. Resuming under arguments a file was not written with
        /// is refused.
        #[arg(long)]
        progress: Option<PathBuf>,
        /// How many records one lockstep decode runs at once. The default of
        /// one is today's per-record rayon path; more feeds `records x paths x
        /// beams` rows to each model step, which is what a GPU backend is
        /// sized for.
        #[arg(long, default_value = "1")]
        batch: usize,
        /// Which backend the ONNX Runtime sessions run on -- the character
        /// model's and the towers' alike. Asking for a provider the binary
        /// was not compiled with is an error, never a silent CPU session.
        #[arg(long, value_enum)]
        backend: Option<neural::BackendArg>,
        /// ONNX Runtime's verbose session logging: the provider each graph
        /// node lands on, the evidence for whether a backend runs the step.
        #[arg(long)]
        ort_verbose: bool,
        #[command(flatten)]
        slice: SliceArgs,
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
    /// Guard a position when its runner-up character trails the winner by at
    /// most this many nats; the beam then keeps the best lineage per covered
    /// character alive instead of letting the winner's tails crowd it out.
    #[arg(long, default_value = "1.5")]
    diversity_gap: f32,
    /// How many distinct characters a guarded position keeps lineages for.
    #[arg(long, default_value = "3")]
    diversity_chars: usize,
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
            diversity_gap: self.diversity_gap,
            diversity_chars: self.diversity_chars,
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
    let cli = Cli::parse();
    // `ort` forwards the runtime's logging to `tracing` at TRACE, so
    // `--ort-verbose` raises the `ort` target; RUST_LOG still wins when set.
    let ort_trace = matches!(
        &cli.command,
        Command::FusedEval {
            ort_verbose: true,
            ..
        }
    );
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                EnvFilter::new(if ort_trace { "info,ort=trace" } else { "info" })
            }),
        )
        .with_writer(std::io::stderr)
        .init();

    match cli.command {
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
        Command::EmitLattice {
            eval_set,
            out,
            emittable,
            search,
        } => {
            let (table, lexicon) = tables()?;
            neural::emit_lattice(
                &eval_set,
                &out,
                &emittable,
                table,
                lexicon,
                search.segment(),
            )
        }
        Command::FusedEval {
            model,
            lm,
            lm_weight,
            eval_set,
            scores,
            route_a,
            no_context,
            route_a_threads,
            emittable,
            unscored,
            weight,
            no_transition,
            dump,
            progress,
            batch,
            backend,
            ort_verbose,
            slice,
            search,
        } => {
            if backend.is_some() && lm.is_none() && route_a.is_none() {
                anyhow::bail!("--backend applies to a neural model, and this run has none");
            }
            fused_eval(&FusedRun {
                model: model.as_deref(),
                lm: lm.as_deref(),
                lm_weight,
                eval_set: &eval_set,
                scores: scores.as_deref(),
                route_a: route_a.as_deref(),
                with_context: !no_context,
                route_a_threads,
                emittable: &emittable,
                unscored,
                weights: &weight,
                no_transition,
                dump: dump.as_deref(),
                progress: progress.as_deref(),
                batch,
                backend: backend.unwrap_or_default(),
                ort_verbose,
                slice: &slice,
                search: &search,
            })
        }
        Command::Corpus { command } => corpus::run(command).await,
        Command::G2p { command } => g2p::run(command).await,
        Command::Synth { command } => synth::run(command).await,
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
    let model = load_ngram(model, &lexicon)?;
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

/// Load a trained n-gram against the generated tables.
fn load_ngram(path: &Path, lexicon: &Lexicon) -> Result<NgramModel> {
    let bytes = fs::read(path)
        .with_context(|| format!("could not read the model at {}", path.display()))?;
    let model = NgramModel::from_bytes(&bytes, lexicon)
        .context("the model does not match this character lexicon")?;
    info!(
        vocabulary = model.vocabulary_size(),
        trigrams = model.trigram_types(),
        "loaded the model"
    );
    Ok(model)
}

/// Everything one fused-eval run is made of.
///
/// A struct rather than nine positional arguments, because every one of them is
/// a knob the report has to quote and a swapped pair of paths would produce a
/// number rather than an error.
struct FusedRun<'a> {
    /// Absent when the run was asked to drop the transition or to decode with
    /// the character model alone; the argument parser holds the flags to that
    /// relationship.
    model: Option<&'a Path>,
    /// The character model's directory, when one takes part.
    lm: Option<&'a Path>,
    lm_weight: f32,
    eval_set: &'a Path,
    scores: Option<&'a Path>,
    /// The towers' export directory, when emissions are computed live.
    route_a: Option<&'a Path>,
    /// Whether live emissions read the record's context.
    with_context: bool,
    /// Intra-op threads per towers session on the per-record path.
    route_a_threads: NonZeroUsize,
    emittable: &'a Path,
    unscored: f32,
    weights: &'a [f32],
    /// Whether the run drops the transition, emissions alone.
    no_transition: bool,
    /// Where every evaluated section's beam lands, or `None` to report only.
    dump: Option<&'a Path>,
    /// Where every configuration's per-record progress lands, or `None` to
    /// keep nothing between runs.
    progress: Option<&'a Path>,
    /// Records per lockstep batch; one is the per-record rayon path.
    batch: usize,
    /// Which backend the ONNX Runtime sessions run on.
    backend: neural::BackendArg,
    /// ONNX Runtime's verbose session logging.
    ort_verbose: bool,
    slice: &'a SliceArgs,
    search: &'a SearchArgs,
}

impl FusedRun<'_> {
    /// The shape the character model's sessions are built to. The lockstep
    /// path -- `--batch` above one, or any GPU backend -- keeps one session
    /// for the whole batch, so it gets the machine's intra-op parallelism;
    /// the per-record rayon path keeps one thread a session, the decoder
    /// itself being the parallel layer. `width` is the beam width the decode
    /// runs: a worker's survivors fill a block of the rectangle.
    fn session_shape(&self) -> Result<ime_lm::SessionShape> {
        let lockstep = self.batch > 1 || self.backend.backend() != ime_lm::Backend::Cpu;
        let intra_threads = if lockstep {
            std::thread::available_parallelism()
                .context("could not ask the platform for its thread count")?
        } else {
            NonZeroUsize::MIN
        };
        Ok(ime_lm::SessionShape {
            backend: self.backend.backend(),
            intra_threads,
            width: self.search.beam_width,
            verbose_logging: self.ort_verbose,
        })
    }

    /// The shape the towers' sessions are built to: one intra-op thread per
    /// session on the per-record path unless `--route-a-threads` asks for
    /// more, all of them on the lockstep path.
    fn neural_shape(&self) -> Result<ime_neural::SessionShape> {
        let lockstep = self.batch > 1 || self.backend.neural_backend() != ime_neural::Backend::Cpu;
        let intra_threads = if lockstep {
            std::thread::available_parallelism()
                .context("could not ask the platform for its thread count")?
        } else {
            self.route_a_threads
        };
        Ok(ime_neural::SessionShape {
            backend: self.backend.neural_backend(),
            intra_threads,
            verbose_logging: self.ort_verbose,
        })
    }
}

fn fused_eval(run: &FusedRun<'_>) -> Result<()> {
    if run.no_transition && run.scores.is_none() && run.route_a.is_none() {
        anyhow::bail!(
            "--no-transition decodes on emissions that --scores or --route-a must supply"
        );
    }
    if run.slice.select_on_dev && run.scores.is_none() && run.route_a.is_none() {
        anyhow::bail!("--select-on-dev sweeps emissions that --scores or --route-a must supply");
    }
    let (table, lexicon) = tables()?;
    let ngram = run
        .model
        .map(|path| load_ngram(path, &lexicon))
        .transpose()?;
    let lm = run
        .lm
        .map(|dir| {
            CharLm::open(dir, &lexicon, run.session_shape()?)
                .with_context(|| format!("could not open the character model in {}", dir.display()))
        })
        .transpose()?;
    let towers = run
        .route_a
        .map(|dir| {
            RouteA::open(dir, run.neural_shape()?)
                .with_context(|| format!("could not open the route A export in {}", dir.display()))
        })
        .transpose()?;
    let transition = match (ngram.as_ref(), lm.as_ref()) {
        (None, None) => Models::None,
        (Some(ngram), None) => Models::Ngram(ngram),
        (None, Some(lm)) => Models::CharLm(lm),
        (Some(ngram), Some(lm)) => Models::Both {
            ngram,
            lm,
            lm_weight: run.lm_weight,
        },
    };
    let progress = run.progress.map(|dir| neural::Progress {
        dir,
        model: run.model,
        lm: run.lm,
        stop_after: None,
    });
    let rendered = neural::fused_eval(
        run.eval_set,
        run.scores,
        run.route_a
            .zip(towers.as_ref())
            .map(|(path, towers)| (path, towers, run.with_context)),
        run.emittable,
        run.unscored,
        run.weights,
        run.slice,
        run.dump,
        progress.as_ref(),
        table,
        lexicon,
        run.search.segment(),
        &run.search.beam(),
        run.batch,
        transition,
    )?;
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
    fn the_eval_set_export_defaults_to_full_pinyin_and_takes_the_other_styles() {
        let default = Cli::parse_from(["ime-cli", "export", "eval-set"]);
        let Command::Export {
            command:
                ExportCommand::EvalSet {
                    typing,
                    abbreviate_syllable,
                    ..
                },
        } = default.command
        else {
            panic!("expected the eval-set export");
        };
        assert_eq!(typing, g2p::TypingArg::Full);
        assert!((abbreviate_syllable - 0.7).abs() < f64::EPSILON);

        let asked = Cli::parse_from([
            "ime-cli",
            "export",
            "eval-set",
            "--typing",
            "mixed",
            "--abbreviate-syllable",
            "0.5",
        ]);
        let Command::Export {
            command:
                ExportCommand::EvalSet {
                    typing,
                    abbreviate_syllable,
                    ..
                },
        } = asked.command
        else {
            panic!("expected the eval-set export");
        };
        assert_eq!(typing, g2p::TypingArg::Mixed);
        assert!((abbreviate_syllable - 0.5).abs() < f64::EPSILON);
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
