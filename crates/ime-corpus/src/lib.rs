//! Fetch and prepare the internet-authentic half of the corpus.
//!
//! The first corpus this project trained on was wiki, news and cleaned chat logs:
//! three registers of edited prose, and none of them the register an input method
//! is actually used in. Slang, 梗, pinyin abbreviations and homophone play are
//! precisely where an n-gram IME fails hardest, and they appear in none of those
//! sources -- so this crate adds three that are nothing but that. Moe Girl Pedia's
//! prose is written by the internet about the internet; douyin captions are what
//! a person types under their own video; bilibili comments are what they type
//! under someone else's. All six are prepared here, because the Python pipeline
//! wrote its documents into the same schema this one reads.
//!
//! The stage is split either side of the network, exactly as the Python pipeline
//! splits it. [`fetch`] writes the upstream text as it arrives and interprets
//! nothing. [`prepare`] holds every rule -- the infobox filter, the hashtag strip,
//! the traditional-to-simplified conversion, the split into units and then into
//! [`typing_segments`], the length gate, the duplicate check -- so that changing
//! one costs a local re-run rather than another pass over 9.7 GB.
//!
//! What comes out is the same `{id, source, text, context}` sample schema the rest
//! of the engine already reads, written into the same [`DataLayout`] directories,
//! so `ime-cli export pool | eval-set | ngram-corpus` work on a run-2 data root
//! unchanged.

pub mod clean;
pub mod download;
pub mod error;
pub mod fetch;
pub mod filter;
pub mod prepare;
pub mod segment;
pub mod source;
pub mod text;

pub use error::{Error, Result};
pub use fetch::fetch;
pub use filter::{FilterCounts, SampleFilter};
pub use ime_g2p::DataLayout;
pub use prepare::{PrepareReport, prepare};
pub use segment::{TypingSegment, is_typable_target, typing_segments};
pub use source::{
    BILIBILI, DIALOGUE, DOUYIN, MOEGIRL, NEWS, RawDocument, SOURCES, SegmentUnit, SourceSpec, WIKI,
};
pub use text::Normalizer;
