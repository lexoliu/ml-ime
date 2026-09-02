//! Annotate target sentences with pinyin twice, and keep only what both agree on.
//!
//! Every accuracy number this project will ever report is bounded by the quality
//! of its pinyin labels: a training pair whose reading is wrong teaches the wrong
//! mapping, and an *evaluation* pair whose reading is wrong makes a correct model
//! look broken. Polyphones are where that bites -- 得, 朝, 还, 重 -- and they are
//! exactly the characters a single automatic labeller gets wrong.
//!
//! So two independent annotators run over every sentence: [`g2pw`], a BERT-based
//! disambiguator behind an ONNX session, and [`llm`], an OpenAI-compatible
//! endpoint prompted per sentence. Where they agree, the label is used. Where
//! they disagree, the sentence is set aside as a hard-polyphone case rather than
//! silently averaged away, and the disagreement rate published by [`report`] is
//! the measured ceiling on everything downstream.
//!
//! Comparison is on the *toneless* syllable. The input method converts what
//! someone types, and no pinyin keyboard has a tone key, so a tone disagreement
//! costs the model nothing; tones are still carried in the output columns because
//! they are what makes a disagreement interpretable.

pub mod annotate;
pub mod error;
pub mod export;
pub mod g2pw;
pub mod layout;
pub mod llm;
pub mod outcome;
pub mod report;
pub mod shards;
pub mod text;
pub mod typing;

pub use annotate::{AnnotatedRow, AnnotationCounts, RefusedRow, Sample, annotate};
pub use error::{Error, Result};
pub use layout::DataLayout;
pub use outcome::{Annotator, Comparison, Outcome, Reading, Refusal, compare};
pub use typing::{DEFAULT_ABBREVIATE_SYLLABLE, Typing, TypingStyle, initial};
