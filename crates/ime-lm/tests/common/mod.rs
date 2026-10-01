//! Helpers shared by the `ime-lm` integration tests: the fixture directory,
//! the fixture lexicon, the session shape the fixtures were recorded at, and
//! what `charlm_fixtures.py` recorded from the exported graphs.

use ime_lm::SessionShape;
use ime_pinyin::{CharId, Lexicon, SyllableTable};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

/// One request `charlm_fixtures.py` drove a `start` or `advance` row with,
/// and the gathered scores it answered.
#[derive(Deserialize)]
#[allow(dead_code)]
pub struct AskedRecord {
    /// The request's candidate characters.
    pub candidates: String,
    /// Whether `<eos>` was among the request.
    pub eos: bool,
    /// The gathered log probabilities, keyed by candidate — `"<eos>"` when
    /// the request named it.
    pub scores: HashMap<String, f32>,
}

#[allow(dead_code)]
impl AskedRecord {
    /// The request's candidates as `CharId`s.
    pub fn ids(&self, lexicon: &Lexicon) -> Vec<CharId> {
        self.candidates
            .chars()
            .map(|ch| {
                lexicon
                    .id_of(ch)
                    .expect("fixture characters are in the lexicon")
            })
            .collect()
    }
}

/// What `charlm_fixtures.py` recorded from the exported graphs. Which fields
/// a test binary reads depends on the test, so unused fields are fine.
#[derive(Deserialize)]
#[allow(dead_code)]
pub struct Expected {
    /// The context `start` is called with (the prelude's middle).
    pub context: String,
    /// Two beams of two tokens each, as characters.
    pub beams: Vec<String>,
    /// What `start` was asked and the gathered scores it answered.
    pub start: AskedRecord,
    /// Per step, each row's request and the gathered scores it answered.
    pub steps: Vec<Vec<AskedRecord>>,
    /// The prefill's row with the whole alphabet asked — the full row the
    /// gathered scores are checked against.
    pub prefill_full: Vec<f32>,
    /// Per step, each row's full-vocabulary row, the whole alphabet asked.
    pub full: Vec<Vec<Vec<f32>>>,
}

/// `tests/fixtures/` beside this crate — or wherever `CHARLM_FIXTURES`
/// points, so a shipped test binary can run on a machine whose checkout
/// path differs from the build machine's.
pub fn fixture_dir() -> PathBuf {
    std::env::var_os("CHARLM_FIXTURES").map_or_else(
        || Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures"),
        PathBuf::from,
    )
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

/// The `chars` of the `arch` fixture's manifest: the alphabet in id order —
/// `"<pad>"`, `"<bos>"`, `"<eos>"` and friends name their own columns.
#[allow(dead_code)]
pub fn alphabet(dir: &Path, arch: &str) -> Vec<String> {
    #[derive(Deserialize)]
    struct Chars {
        chars: Vec<String>,
    }
    serde_json::from_str::<Chars>(
        &fs::read_to_string(dir.join(arch).join("charlm.json"))
            .expect("the fixture manifest is committed"),
    )
    .expect("the fixture manifest parses")
    .chars
}
