//! The data-pipeline half of the command line: dual annotation and its exports.
//!
//! Everything here is a thin shell over `ime-g2p`. The heavy loading -- an ONNX
//! session, a corpus of parquet shards -- happens inside the command bodies so
//! that `--help` stays instant, and every command takes a data root rather than
//! knowing one, because the same commands run against a checkout and against a
//! kernel's working directory.

use anyhow::{Context as _, Result};
use askama::Template;
use clap::{Args, Subcommand, ValueEnum};
use ime_g2p::annotate::{ANNOTATED, Sample, annotate, read_annotated, read_samples};
use ime_g2p::export::{export_eval_set, export_ngram_corpus, export_pool, read_exclusions};
use ime_g2p::g2pw::{
    Converter, DEFAULT_BATCH_SIZE, G2pwAnnotator, default_model_dir, default_tokenizer_path,
};
use ime_g2p::llm::{DEFAULT_CONCURRENCY, LlmAnnotator, LlmSettings};
use ime_g2p::outcome::{Annotator as _, Outcome, compare};
use ime_g2p::text::han_characters;
use ime_g2p::typing::{DEFAULT_ABBREVIATE_SYLLABLE, Typing, TypingStyle};
use ime_g2p::{DataLayout, report};
use std::io::{BufRead as _, BufReader, BufWriter, Write as _};
use std::path::{Path, PathBuf};
use tracing::info;

/// A polyphone-dense sentence: 重 chong/zhong, 还 huan/hai, 得 de/dei, 绿 lv.
const PROBE_SENTENCE: &str = "他还了钱还差一点，我得到了那件重要的绿色东西";

/// Where the g2pW model and its tokenizer live, shared by every command that
/// loads the network.
#[derive(Debug, Clone, Args)]
pub struct ModelArgs {
    /// Directory holding `g2pw.onnx` and the two character tables.
    #[arg(long)]
    g2pw_model: Option<PathBuf>,
    /// The `bert-base-chinese` `tokenizer.json`; found in the Hugging Face cache
    /// when omitted.
    #[arg(long)]
    tokenizer: Option<PathBuf>,
    /// How many query positions are handed to the network at once.
    #[arg(long, default_value_t = DEFAULT_BATCH_SIZE)]
    batch_size: usize,
}

impl ModelArgs {
    fn model_dir(&self) -> Result<PathBuf> {
        match &self.g2pw_model {
            Some(path) => Ok(path.clone()),
            None => Ok(default_model_dir()?),
        }
    }

    fn tokenizer_path(&self) -> Result<PathBuf> {
        match &self.tokenizer {
            Some(path) => Ok(path.clone()),
            None => Ok(default_tokenizer_path()?),
        }
    }

    fn converter(&self) -> Result<Converter> {
        Ok(Converter::load(
            &self.model_dir()?,
            &self.tokenizer_path()?,
            self.batch_size,
        )?)
    }

    fn annotator(&self) -> Result<G2pwAnnotator> {
        Ok(G2pwAnnotator::new(
            &self.model_dir()?,
            &self.tokenizer_path()?,
            self.batch_size,
        )?)
    }
}

/// Dual pinyin annotation and its agreement report.
#[derive(Debug, Subcommand)]
pub enum G2pCommand {
    /// Annotate one sentence with both systems and show them side by side.
    ///
    /// The LLM annotator's failure modes are all silent from a distance: a wrong
    /// base URL, an expired key, a model that ignores the JSON instruction, a
    /// proxy that drops `reasoning_effort`. Each of those turns into a wall of
    /// refusals an hour into a run, so this is the first thing to run on a new
    /// machine.
    Probe {
        /// Sentence to annotate with both systems.
        #[arg(long, default_value = PROBE_SENTENCE)]
        sentence: String,
        #[command(flatten)]
        model: ModelArgs,
    },
    /// Annotate prepared samples with both g2p systems and record where they agree.
    Annotate {
        /// Root the pipeline reads and writes.
        #[arg(long, default_value = "data")]
        data_dir: PathBuf,
        /// Stop after this many samples.
        #[arg(long)]
        limit: Option<usize>,
        /// Simultaneous requests in flight to the LLM.
        #[arg(long, default_value_t = DEFAULT_CONCURRENCY)]
        concurrency: usize,
        /// Sentences handed to the annotators at once.
        #[arg(long, default_value_t = DEFAULT_BATCH_SIZE)]
        sentences_per_batch: usize,
        #[command(flatten)]
        model: ModelArgs,
    },
    /// Print the agreement rate, its frequency breakdown, and the worst characters.
    Report {
        /// Root the pipeline reads and writes.
        #[arg(long, default_value = "data")]
        data_dir: PathBuf,
    },
    /// Run only g2pW over a JSON Lines file, one reading per Han character.
    ///
    /// This is the parity harness against the Python `G2PWConverter`: it takes
    /// the same `{"text": ...}` lines and emits the readings alone, so the two
    /// implementations can be diffed without an endpoint or an agreement rule in
    /// the way.
    G2pw {
        /// JSON Lines input; each line needs a `text` field.
        #[arg(long)]
        input: PathBuf,
        /// Where to write `{"text": ..., "syllables": [...]}` lines.
        #[arg(long)]
        out: PathBuf,
        /// Stop after this many sentences.
        #[arg(long)]
        limit: Option<usize>,
        #[command(flatten)]
        model: ModelArgs,
    },
}

/// Emit artefacts for the rest of the engine.
#[derive(Debug, Subcommand)]
pub enum ExportCommand {
    /// Draw a seeded, source-stratified evaluation set from the agreed annotations.
    EvalSet {
        /// Root the pipeline reads and writes.
        #[arg(long, default_value = "data")]
        data_dir: PathBuf,
        /// JSON Lines file to write.
        #[arg(long, default_value = "data/eval.jsonl")]
        out: PathBuf,
        /// Number of evaluation items to draw.
        #[arg(long, default_value_t = 1000)]
        size: usize,
        /// Sampling seed; the same seed reproduces the same set, whatever it is
        /// typed as.
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// How the drawn sentences are typed.
        #[arg(long, value_enum, default_value = "full")]
        typing: TypingArg,
        /// How often an abbreviating pass drops a syllable to its initial. The
        /// default is the rate the model was trained under.
        #[arg(long, default_value_t = DEFAULT_ABBREVIATE_SYLLABLE)]
        abbreviate_syllable: f64,
    },
    /// Dump prepared targets as one line each, minus the sentences held out for evaluation.
    NgramCorpus {
        /// Root the pipeline reads and writes.
        #[arg(long, default_value = "data")]
        data_dir: PathBuf,
        /// Plain text file to write.
        #[arg(long, default_value = "data/corpus.txt")]
        out: PathBuf,
        /// JSON Lines file whose `text` fields are held out of the corpus; repeatable.
        #[arg(long)]
        exclude: Vec<PathBuf>,
    },
    /// Draw a seeded, source-stratified working pool out of the prepared samples.
    Pool {
        /// Root the prepared samples live under.
        #[arg(long, default_value = "data")]
        data_dir: PathBuf,
        /// Data root to write the pool's own samples and `pool.jsonl` into.
        #[arg(long)]
        out_dir: PathBuf,
        /// Number of sentences to draw.
        #[arg(long)]
        size: usize,
        /// Sampling seed; the same seed reproduces the same pool.
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
}

/// Which keystrokes an evaluation set is drawn with.
///
/// Mirrors [`Typing`] rather than deriving `ValueEnum` on it, for the reason
/// [`crate::neural::SliceArg`] mirrors its own: the annotation crate has no
/// business depending on an argument parser.
#[derive(Copy, Clone, PartialEq, Eq, Debug, ValueEnum)]
pub enum TypingArg {
    /// Every syllable typed out, which is what the first eval sets were.
    Full,
    /// Each syllable independently dropped to its initial.
    Abbreviated,
    /// Full up to a cut inside the sentence, abbreviated after it.
    Mixed,
}

impl TypingArg {
    /// The annotation crate's own name for this style.
    const fn typing(self) -> Typing {
        match self {
            Self::Full => Typing::Full,
            Self::Abbreviated => Typing::Abbreviated,
            Self::Mixed => Typing::Mixed,
        }
    }
}

/// One line of a JSON Lines corpus file: the sentence, and nothing else needed.
#[derive(Debug, serde::Deserialize)]
struct TextLine {
    text: String,
}

/// One line of the g2pW parity output.
#[derive(Debug, serde::Serialize)]
struct G2pwLine {
    text: String,
    syllables: Option<Vec<String>>,
    refusal: Option<String>,
}

/// One position of the probe's side-by-side table.
struct ProbeRow {
    index: String,
    character: String,
    first: String,
    second: String,
    agree: &'static str,
}

/// The probe, as it goes to stdout.
#[derive(Template)]
#[template(path = "g2p_probe.txt", ext = "txt")]
struct ProbeTable {
    endpoint: String,
    model: String,
    sentence: String,
    rows: Vec<ProbeRow>,
    agreed: usize,
    positions: usize,
}

/// Run one `g2p` subcommand.
///
/// # Errors
///
/// Whatever the stage it dispatches to refuses on.
pub async fn run(command: G2pCommand) -> Result<()> {
    match command {
        G2pCommand::Probe { sentence, model } => probe(&sentence, &model).await,
        G2pCommand::Annotate {
            data_dir,
            limit,
            concurrency,
            sentences_per_batch,
            model,
        } => {
            run_annotate(
                &DataLayout::new(data_dir),
                limit,
                concurrency,
                sentences_per_batch,
                &model,
            )
            .await
        }
        G2pCommand::Report { data_dir } => print_report(&DataLayout::new(data_dir)),
        G2pCommand::G2pw {
            input,
            out,
            limit,
            model,
        } => run_g2pw(&input, &out, limit, &model),
    }
}

/// Run one `export` subcommand.
///
/// # Errors
///
/// Whatever the export it dispatches to refuses on.
pub fn run_export(command: ExportCommand) -> Result<()> {
    match command {
        ExportCommand::EvalSet {
            data_dir,
            out,
            size,
            seed,
            typing,
            abbreviate_syllable,
        } => {
            let style = TypingStyle::new(typing.typing(), abbreviate_syllable)
                .context("the abbreviation rate is not a probability")?;
            let layout = DataLayout::new(data_dir);
            let rows = read_annotated(&layout.annotations(), ANNOTATED)
                .context("could not read the annotation shards")?;
            let written = export_eval_set(&rows, &out, size, seed, style)?;
            info!(
                items = written,
                path = %out.display(),
                typing = %style.typing(),
                "evaluation set written"
            );
            Ok(())
        }
        ExportCommand::NgramCorpus {
            data_dir,
            out,
            exclude,
        } => {
            let layout = DataLayout::new(data_dir);
            let held_out = if exclude.is_empty() {
                std::collections::HashSet::new()
            } else {
                read_exclusions(&exclude)?
            };
            let (enforceable, unreachable) = split_reachable(held_out);
            if let Some(example) = unreachable.first() {
                info!(
                    sentences = unreachable.len(),
                    example, "held-out sentences that no prepared target can equal"
                );
            }
            let written = export_ngram_corpus(&layout.samples(), &out, &enforceable)?;
            info!(lines = written, path = %out.display(), "n-gram corpus written");
            Ok(())
        }
        ExportCommand::Pool {
            data_dir,
            out_dir,
            size,
            seed,
        } => {
            let layout = DataLayout::new(data_dir);
            let counts = export_pool(&layout.samples(), &out_dir, size, seed)?;
            for (source, count) in &counts {
                info!(source, count, "drawn into the pool");
            }
            Ok(())
        }
    }
}

/// Split held-out sentences into the ones a prepared target can equal and the rest.
///
/// A prepared target is one uninterrupted run of Han characters, so a held-out
/// sentence carrying a comma is one the export can never find no matter how the
/// corpus was built. Handing those to the exporter would fail the run over a
/// sentence that is correctly absent, so they are reported and set aside; the
/// rest keep the exporter's guarantee that every exclusion was actually applied.
fn split_reachable(
    held_out: std::collections::HashSet<String>,
) -> (std::collections::HashSet<String>, Vec<String>) {
    let mut enforceable = std::collections::HashSet::new();
    let mut unreachable = Vec::new();
    for text in held_out {
        if ime_corpus::is_typable_target(&text) {
            enforceable.insert(text);
        } else {
            unreachable.push(text);
        }
    }
    unreachable.sort();
    (enforceable, unreachable)
}

async fn probe(sentence: &str, model: &ModelArgs) -> Result<()> {
    let settings = LlmSettings::load()?;
    let g2pw = model.annotator()?;
    let llm = LlmAnnotator::new(&settings, 1);
    let batch = [sentence.to_owned()];
    let (first, second) = tokio::join!(g2pw.annotate(&batch), llm.annotate(&batch));
    let first = one(first, g2pw.name())?;
    let second = one(second, llm.name())?;
    let comparison = compare(sentence, &first, &second)?;

    let table = ProbeTable {
        endpoint: settings.base_url.clone(),
        model: settings.model.clone(),
        sentence: sentence.to_owned(),
        rows: han_characters(sentence)
            .iter()
            .enumerate()
            .map(|(index, character)| ProbeRow {
                index: format!("{index:>3}"),
                character: character.to_string(),
                first: format!("{:<8}", comparison.first()[index]),
                second: format!("{:<8}", comparison.second()[index]),
                agree: if comparison.agree()[index] {
                    "yes"
                } else {
                    "NO"
                },
            })
            .collect(),
        agreed: comparison.agreed(),
        positions: comparison.agree().len(),
    };
    print!("{}", table.render().context("could not render the probe")?);
    Ok(())
}

fn one(outcomes: Vec<Outcome>, name: &str) -> Result<ime_g2p::Reading> {
    let outcome = outcomes
        .into_iter()
        .next()
        .with_context(|| format!("{name} returned no outcome at all"))?;
    outcome.map_err(|refusal| anyhow::anyhow!("{name} refused: {}", refusal.reason))
}

async fn run_annotate(
    layout: &DataLayout,
    limit: Option<usize>,
    concurrency: usize,
    sentences_per_batch: usize,
    model: &ModelArgs,
) -> Result<()> {
    let settings = LlmSettings::load()?;
    let mut samples: Vec<Sample> =
        read_samples(&layout.samples()).context("could not read the sample shards")?;
    if let Some(limit) = limit {
        samples.truncate(limit);
    }
    info!(
        samples = samples.len(),
        endpoint = %settings.base_url,
        model = %settings.model,
        "annotating"
    );
    let g2pw = model.annotator()?;
    let llm = LlmAnnotator::new(&settings, concurrency);
    let counts = annotate(
        samples,
        &g2pw,
        &llm,
        &layout.annotations(),
        sentences_per_batch,
    )
    .await?;
    info!(
        agreed = counts.agreed,
        disagreed = counts.disagreed,
        refused = counts.refused,
        rate = counts.agreement_rate(),
        "annotation finished"
    );
    Ok(())
}

fn print_report(layout: &DataLayout) -> Result<()> {
    let report = report::report(&layout.annotations())?;
    print!(
        "{}",
        report.render().context("could not render the report")?
    );
    Ok(())
}

fn run_g2pw(input: &Path, out: &Path, limit: Option<usize>, model: &ModelArgs) -> Result<()> {
    let file = std::fs::File::open(input)
        .with_context(|| format!("could not open {}", input.display()))?;
    let mut texts = Vec::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line =
            line.with_context(|| format!("could not read line {} of the input", index + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let parsed: TextLine = serde_json::from_str(&line)
            .with_context(|| format!("{}:{} has no text field", input.display(), index + 1))?;
        texts.push(parsed.text);
        if limit.is_some_and(|limit| texts.len() >= limit) {
            break;
        }
    }
    info!(sentences = texts.len(), "read the parity input");

    let mut converter = model.converter()?;
    let predictions = converter.convert(&texts)?;
    let mut handle = BufWriter::new(
        std::fs::File::create(out)
            .with_context(|| format!("could not create {}", out.display()))?,
    );
    for (text, row) in texts.iter().zip(&predictions) {
        let mut syllables = Vec::new();
        let mut refusal = None;
        for (character, reading) in text.chars().zip(row) {
            if !ime_g2p::text::is_han(character) {
                continue;
            }
            match reading {
                Some(syllable) => syllables.push(syllable.clone()),
                None => refusal = Some(format!("g2pw has no reading for '{character}'")),
            }
        }
        let line = G2pwLine {
            text: text.clone(),
            syllables: if refusal.is_none() {
                Some(syllables)
            } else {
                None
            },
            refusal,
        };
        writeln!(
            handle,
            "{}",
            serde_json::to_string(&line).context("could not serialise a reading")?
        )
        .context("could not write to the output")?;
    }
    handle.flush().context("could not flush the output")?;
    info!(sentences = texts.len(), path = %out.display(), "wrote the readings");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_held_out_sentence_shaped_like_a_target_stays_enforceable() {
        let held_out: std::collections::HashSet<String> = [
            "今天天气不错",
            "白天不睡的话，累了就困了",
            "使用Python写程序",
            "太短",
        ]
        .into_iter()
        .map(ToOwned::to_owned)
        .collect();
        let (enforceable, unreachable) = split_reachable(held_out);
        assert_eq!(
            enforceable.into_iter().collect::<Vec<String>>(),
            vec!["今天天气不错".to_owned()]
        );
        assert_eq!(unreachable.len(), 3);
        // Sorted, so the report's example is the same one on every run.
        assert_eq!(unreachable[0], "使用Python写程序");
    }
}
