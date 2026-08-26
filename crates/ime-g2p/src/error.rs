//! What can go wrong in the annotation pipeline, named at the point it goes wrong.
//!
//! Nothing here is recoverable by falling back: a missing model directory, a
//! table whose shape no longer matches the network, or a sentence whose
//! characters and syllables have drifted apart all mean the run would produce
//! labels that quietly disagree with the ones the rest of the project assumes.
//! They stop the run instead.

use std::path::PathBuf;

/// The result type every fallible operation in this crate returns.
pub type Result<T> = std::result::Result<T, Error>;

/// Everything that stops an annotation run.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A file the pipeline needs is not where it was asked to look.
    #[error("{what} is missing at {}; {hint}", path.display())]
    Missing {
        /// What was being looked for, in words.
        what: &'static str,
        /// Where it was looked for.
        path: PathBuf,
        /// How to put it there.
        hint: &'static str,
    },
    /// A file was found but could not be read or parsed.
    #[error("could not read {}: {source}", path.display())]
    Read {
        /// The file that could not be read.
        path: PathBuf,
        /// Why.
        source: std::io::Error,
    },
    /// One of the vendored g2pW tables no longer has the shape the network expects.
    #[error("the g2pW tables are inconsistent: {0}")]
    Tables(String),
    /// A tokenizer file exists but the `tokenizers` crate refused it.
    #[error("could not load the BERT tokenizer at {}: {source}", path.display())]
    Tokenizer {
        /// The tokenizer file.
        path: PathBuf,
        /// Why it was refused.
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// The ONNX session failed to build or to run.
    #[error("the g2pW ONNX session failed: {0}")]
    Onnx(#[from] ort::Error),
    /// Two readings that should line up character for character do not.
    #[error(
        "cannot compare readings of different lengths: {characters} characters, {first} and {second} syllables, for {text:?}"
    )]
    Misaligned {
        /// The sentence.
        text: String,
        /// How many Han characters it has.
        characters: usize,
        /// How many syllables the first annotator produced.
        first: usize,
        /// How many the second produced.
        second: usize,
    },
    /// A parquet shard could not be written or read.
    #[error("parquet failed: {0}")]
    Parquet(#[from] polars::error::PolarsError),
    /// A JSON Lines file could not be parsed.
    #[error("{}:{line} is not a usable record: {source}", path.display())]
    JsonLine {
        /// The file.
        path: PathBuf,
        /// The one-based line number.
        line: usize,
        /// Why the line was refused.
        source: serde_json::Error,
    },
    /// The pipeline was asked for more evaluation items than exist.
    #[error("only {available} eligible samples across {sources} sources, need {wanted}")]
    NotEnough {
        /// How many rows are eligible.
        available: usize,
        /// How many sources they came from.
        sources: usize,
        /// How many were asked for.
        wanted: usize,
    },
    /// Held-out sentences were named that no shard actually holds, so nothing was held out.
    #[error(
        "{missing} of {held_out} held-out sentences are not in the samples, so nothing was held out for them: {examples:?}"
    )]
    HeldOutMissing {
        /// How many were not found.
        missing: usize,
        /// How many were asked to be held out.
        held_out: usize,
        /// The first few, to name the problem.
        examples: Vec<String>,
    },
    /// The `MLIME_LLM_*` environment is incomplete.
    #[error("{0} is not set; put it in the repository's .env or the environment")]
    Unconfigured(&'static str),
    /// The endpoint itself failed in a way retrying will not fix.
    #[error("the LLM endpoint failed: {0}")]
    Endpoint(#[from] async_openai::error::OpenAIError),
    /// An invariant the pipeline depends on was violated by its own inputs.
    #[error("{0}")]
    Invariant(String),
}
