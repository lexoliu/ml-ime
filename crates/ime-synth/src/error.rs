//! What the synthesis stage refuses on.

use std::path::PathBuf;

/// The result type every fallible operation in this crate returns.
pub type Result<T> = std::result::Result<T, Error>;

/// Why a synthesis run could not proceed.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A seed file the run needs is not on disk.
    #[error("no {what} at {}; {hint}", path.display())]
    Missing {
        /// What was being looked for.
        what: &'static str,
        /// Where it was looked for.
        path: PathBuf,
        /// How to produce it.
        hint: &'static str,
    },
    /// A file could not be read or written.
    #[error("could not read or write {}", path.display())]
    Read {
        /// The file involved.
        path: PathBuf,
        /// What the filesystem said.
        #[source]
        source: std::io::Error,
    },
    /// The `MediaWiki` parse JSON is not the shape the parser expects.
    #[error("the wiki seed at {} is not a MediaWiki parse response: {reason}", path.display())]
    NotWikiParse {
        /// The file involved.
        path: PathBuf,
        /// What was wrong with it.
        reason: String,
    },
    /// Too much of what the endpoint returned failed validation to trust the batch.
    #[error(
        "{dropped} of {examples} generated examples ({percent:.1}%) failed validation, over the \
         {limit:.0}% the run will accept; the prompt, the endpoint or the seed list is wrong, so \
         nothing was written"
    )]
    TooManyDropped {
        /// How many parsed examples were rejected.
        dropped: usize,
        /// How many parsed examples were judged in total.
        examples: usize,
        /// The observed drop rate, as a percentage.
        percent: f64,
        /// The rate the run refuses to exceed, as a percentage.
        limit: f64,
    },
    /// No seed term survived grounding, so there is nothing to generate from.
    #[error(
        "none of the {loaded} seed terms carried an explanation, so every prompt would have been \
         ungrounded; nothing was generated"
    )]
    NoGroundedSeeds {
        /// How many seed terms were read before grounding.
        loaded: usize,
    },
    /// Something the code believes cannot happen, happened.
    #[error("{0}")]
    Invariant(String),
    /// A JSON summary could not be read or written.
    #[error("the run summary at {} is malformed", path.display())]
    Summary {
        /// The file involved.
        path: PathBuf,
        /// What serde said.
        #[source]
        source: serde_json::Error,
    },
    /// A shard could not be read or written.
    #[error(transparent)]
    Shard(#[from] ime_g2p::Error),
    /// The corpus normaliser refused a string.
    #[error(transparent)]
    Corpus(#[from] ime_corpus::Error),
    /// The pinyin lexicon would not load.
    #[error(transparent)]
    Lexicon(#[from] ime_pinyin::LexiconError),
    /// A parquet frame would not build or read back.
    #[error(transparent)]
    Polars(#[from] polars::error::PolarsError),
}
