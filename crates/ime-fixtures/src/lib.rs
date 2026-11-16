//! The integration-test fixtures whose graphs and weights live off the tree.
//!
//! Each set is a directory the build script assembles under `OUT_DIR`: the
//! text files committed in the set's directory (the manifests, the expected
//! scores, the character tables), beside the `.onnx` and `.weights` files
//! that `fixtures.json` pins to a Hugging Face dataset revision by SHA-256.
//! The generators that write a set are `python/scripts/charlm_fixtures.py`,
//! `python/scripts/routea_fixtures.py` and `python/scripts/keys_fixtures.py`
//! (whose compiled model is a directory of files, pinned whole under the
//! set's `fetched` entry); `python/scripts/publish_fixtures.py`
//! uploads what they wrote and rewrites the pin.

use std::path::Path;

/// `ime-lm`'s fixtures, assembled from `crates/ime-lm/tests/fixtures`: one
/// `mlime export char-lm` directory per variant (`lstm`, `transformer`, each
/// also `-fp16` and `-int8`) with its `expected.json`, and the
/// `char_pinyin.tsv` the fixture lexicon parses.
#[must_use]
pub fn charlm() -> &'static Path {
    Path::new(concat!(env!("OUT_DIR"), "/charlm"))
}

/// `ime-neural`'s fixtures, assembled from `crates/ime-neural/tests/fixtures`:
/// the route A export as `fp32/` and `int8/`, the lattice it was scored over,
/// and the `expected.json` score table.
#[must_use]
pub fn route_a() -> &'static Path {
    Path::new(concat!(env!("OUT_DIR"), "/route-a"))
}

/// `ime-session`'s key-model fixture, assembled from
/// `crates/ime-session/tests/fixtures/keys`: `model/`, the unpacked archive
/// `mlime-coreml convert` makes of a tiny random key model (`key-model.json`,
/// `tokenizer.json`, and the `context/`, `keys/` and `decoder/` directories,
/// each a compiled `.mlmodelc` and its `manifest.json`), and the
/// `expected.json` the Python Core ML search wrote over it. The files are plain
/// data, assembled on every platform; only an Apple target can run the model.
#[must_use]
pub fn keys() -> &'static Path {
    Path::new(concat!(env!("OUT_DIR"), "/keys"))
}
