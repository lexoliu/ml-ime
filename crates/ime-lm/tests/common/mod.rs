//! Helpers shared by the `ime-lm` integration tests: the fixture directory,
//! the fixture lexicon, the session shape the fixtures were recorded at, and
//! what `charlm_fixtures.py` recorded from the exported graphs.

use ime_lm::SessionShape;
use ime_pinyin::{Lexicon, SyllableTable};
use serde::Deserialize;
use std::fs;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

/// What `charlm_fixtures.py` recorded from the exported graphs. Which fields
/// a test binary reads depends on the test, so unused fields are fine.
#[derive(Deserialize)]
#[allow(dead_code)]
pub struct Expected {
    /// The context `start` is called with (the prelude's middle).
    pub context: String,
    /// Two beams of two tokens each, as characters.
    pub beams: Vec<String>,
    /// `log_probs` after prefill: one row of the alphabet.
    pub prefill: Vec<f32>,
    /// `log_probs` after each step: two rows of the alphabet, per step.
    pub steps: Vec<Vec<Vec<f32>>>,
}

/// `tests/fixtures/` beside this crate.
pub fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// The fixture lexicon, from the `char_pinyin.tsv` committed beside the
/// exports.
pub fn lexicon(dir: &Path) -> Lexicon {
    let table = SyllableTable::load();
    let source = fs::read_to_string(dir.join("char_pinyin.tsv"))
        .expect("char_pinyin.tsv is committed with the fixtures");
    Lexicon::parse(&source, &table).expect("the fixture table parses")
}

/// The fixture's beam width as the rectangle's, so a full batch is two live
/// rows and no dead row is ever fed.
pub fn shape() -> SessionShape {
    SessionShape {
        width: NonZeroUsize::new(2).expect("two is not zero"),
        ..SessionShape::default()
    }
}

/// The `expected.json` committed with the `arch` fixture.
pub fn expected(dir: &Path, arch: &str) -> Expected {
    serde_json::from_str(
        &fs::read_to_string(dir.join(arch).join("expected.json"))
            .expect("expected.json is committed with the fixture"),
    )
    .expect("expected.json parses")
}
