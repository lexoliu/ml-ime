//! Grounded LLM synthesis of internet-slang usage sentences, for training only.
//!
//! An input method is judged on the words people actually type, and the words
//! people actually type include a long tail of slang that appears in no news
//! corpus and no encyclopaedia: 爷青结, 蓝瘦香菇, 不明觉厉. Two seed lexicons in
//! this project know those words -- the Sogou 网络流行新词 dictionary and the
//! zh-wikipedia slang list -- but a lexicon is a list of words, and an n-gram
//! model cannot learn anything from a word it has never seen used. This crate
//! closes that gap by asking a cheap local model for the missing sentences.
//!
//! Three rules from the decision record hold the whole thing together, and each
//! one is enforced somewhere in the code rather than in a comment:
//!
//! * **Grounded generation only.** The model's Chinese world knowledge is not
//!   trusted, so no prompt ever names a term without also carrying that term's
//!   explanation from the seed source. A term nothing explains is skipped, and
//!   [`seed`] counts the skips. This is why the Sogou dictionaries contribute
//!   almost nothing: 128k words, and an explanation for a few dozen of them.
//! * **Training only.** Every sample carries `source = "synthetic-luna"`
//!   ([`source::SYNTHETIC`]), which is what lets the evaluation draw exclude
//!   them. Model-written text in an evaluation set measures the model that wrote
//!   it, not the input method.
//! * **Per-term provenance.** [`provenance`] writes a sidecar keyed on the same
//!   content hash as the samples, so a batch that turns out to be bad -- a
//!   dictionary of poor terms, a prompt that drifted -- can be identified and
//!   dropped whole rather than argued about.
//!
//! Output is the ordinary `{id, source, text, context}` sample schema, written
//! into the ordinary `samples/` directory of a data root, so `ime-cli export
//! ngram-corpus` consumes it with no knowledge that it is synthetic. That is
//! convenient and it is also the one sharp edge: a data root holding these
//! shards must not then be run through `g2p annotate` and drawn from for an
//! evaluation set. Generate into a root of its own.

pub mod error;
pub mod generate;
pub mod llm;
pub mod provenance;
pub mod report;
pub mod seed;
pub mod source;
pub mod summary;
pub mod validate;
pub mod wiki;

pub use error::{Error, Result};
pub use generate::{MAX_DROP_RATE, Options, generate};
pub use provenance::Provenance;
pub use seed::{Seed, SeedCounts, SeedLoad};
pub use source::{SEED_SOURCES, SYNTHETIC, SeedSource};
pub use summary::RunSummary;
pub use validate::DropCounts;
pub use wiki::WikiEntry;
