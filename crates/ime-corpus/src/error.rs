//! What can go wrong while fetching or preparing a corpus, named where it goes wrong.
//!
//! None of these are recoverable by falling back. An upstream layout that has
//! moved, a converter that changed a sentence's length, or a download that came
//! back short all mean the shards would quietly disagree with what the rest of
//! the pipeline assumes about them, so they stop the run instead.

use std::path::PathBuf;

/// The result type every fallible operation in this crate returns.
pub type Result<T> = std::result::Result<T, Error>;

/// Everything that stops a corpus run.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A file was found but could not be read, written or parsed.
    #[error("could not read {}: {source}", path.display())]
    Read {
        /// The file that could not be read.
        path: PathBuf,
        /// Why.
        source: std::io::Error,
    },
    /// The network refused, or gave back something other than the file asked for.
    #[error("{url} failed: {message}")]
    Download {
        /// What was being fetched.
        url: String,
        /// Why it could not be fetched.
        message: String,
    },
    /// An upstream dataset no longer has the shape this crate reads.
    ///
    /// Named separately from [`Error::Invariant`] because the fix is different:
    /// an upstream that moved needs the adapter updated, not the pipeline.
    #[error("{dataset} does not have the expected layout: {detail}")]
    Layout {
        /// The Hugging Face repository identifier.
        dataset: &'static str,
        /// What was expected and what was found.
        detail: String,
    },
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
    /// A comma-separated file could not be parsed.
    #[error("could not read the CSV at {}: {source}", path.display())]
    Csv {
        /// The file.
        path: PathBuf,
        /// Why it was refused.
        source: csv::Error,
    },
    /// A parquet shard could not be written or read.
    #[error("parquet failed: {0}")]
    Parquet(#[from] polars::error::PolarsError),
    /// The traditional-to-simplified converter could not be built.
    #[error("could not load the OpenCC t2s configuration: {0}")]
    Converter(String),
    /// An invariant the pipeline depends on was violated by its own inputs.
    #[error("{0}")]
    Invariant(String),
    /// A stage of `ime-g2p` -- the shard writer, or the sample reader -- refused.
    #[error(transparent)]
    Shards(#[from] ime_g2p::Error),
}
