//! Pinyin input primitives: the syllable inventory, the character lexicon, and
//! the segmentation lattice that turns a keystroke sequence into candidate
//! readings.
//!
//! The crate deliberately stops short of scoring. It produces, for a typed
//! string, a small set of plausible segmentations and -- for each output
//! character position -- the set of characters that position is allowed to take.
//! Choosing among them is the language model's job ([`ime-decode`]).
//!
//! [`ime-decode`]: https://github.com/lexoliu/ml-ime

mod lexicon;
mod segment;
mod syllable;

pub use lexicon::{CharId, Lexicon, LexiconError};
pub use segment::{Segment, SegmentError, SegmentLattice, SegmentOptions, Segmentation};
pub use syllable::{MAX_SYLLABLE_LEN, SyllableId, SyllableRange, SyllableTable};
