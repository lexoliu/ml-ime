//! Both exported architectures run end to end: `start` is one `prefill` call
//! and `advance` stacks the surviving beams, against log probabilities the
//! Python side recorded from onnxruntime for the same prelude and tokens.
//!
//! The fixtures under `tests/fixtures/` are written by
//! `python/scripts/charlm_fixtures.py`; regenerate them with
//! `cd python && uv run python scripts/charlm_fixtures.py
//! ../crates/ime-lm/tests/fixtures`.

use ime_decode::Transition;
use ime_lm::{CharLm, LmState};
use ime_pinyin::{CharId, Lexicon, SyllableTable};
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

/// What `charlm_fixtures.py` recorded from the exported graphs.
#[derive(Deserialize)]
struct Expected {
    /// The context `start` is called with (the prelude's middle).
    context: String,
    /// Two beams of two tokens each, as characters.
    beams: Vec<String>,
    /// `log_probs` after prefill: one row of the alphabet.
    prefill: Vec<f32>,
    /// `log_probs` after each step: two rows of the alphabet, per step.
    steps: Vec<Vec<Vec<f32>>>,
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn lexicon(dir: &Path) -> Lexicon {
    let table = SyllableTable::load();
    let source = fs::read_to_string(dir.join("char_pinyin.tsv"))
        .expect("char_pinyin.tsv is committed with the fixtures");
    Lexicon::parse(&source, &table).expect("the fixture table parses")
}

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: row length differs");
    for (index, (got, want)) in got.iter().zip(want).enumerate() {
        assert!(
            (got - want).abs() <= 1e-4,
            "{what}[{index}]: got {got}, want {want}"
        );
    }
}

/// `start(Some(context))`, then two `advance` calls of two beams each, every
/// log-probability row checked against what onnxruntime produced.
fn run(arch: &str) {
    let dir = fixture_dir();
    let lexicon = lexicon(&dir);
    let model = CharLm::open(&dir.join(arch), &lexicon).expect("the fixture opens");
    let expected: Expected = serde_json::from_str(
        &fs::read_to_string(dir.join(arch).join("expected.json"))
            .expect("expected.json is committed with the fixture"),
    )
    .expect("expected.json parses");
    assert_eq!(expected.beams.len(), 2, "the fixture has two beams");

    let start = model.start(Some(&expected.context));
    assert_close(start.log_probs(), &expected.prefill, "prefill");
    // Every beam of a record starts from the same state; two clones of it are
    // the two beams the fixture recorded.
    let mut states = vec![start.clone(), start];

    let steps = expected.beams[0].chars().count();
    assert_eq!(expected.steps.len(), steps, "one expected row set per step");
    for position in 0..steps {
        let batch: Vec<(&LmState, CharId)> = states
            .iter()
            .zip(&expected.beams)
            .map(|(state, beam)| {
                let character = beam
                    .chars()
                    .nth(position)
                    .expect("every beam has two tokens");
                (
                    state,
                    lexicon
                        .id_of(character)
                        .expect("fixture characters are in the lexicon"),
                )
            })
            .collect();
        assert_eq!(
            batch.len(),
            expected.beams.len(),
            "every beam is fed a token"
        );
        states = model.advance(&batch);
        assert_eq!(
            states.len(),
            expected.beams.len(),
            "one state per beam comes back"
        );
        assert_eq!(
            expected.steps[position].len(),
            states.len(),
            "one row per beam"
        );
        for (row, state) in states.iter().enumerate() {
            assert_close(
                state.log_probs(),
                &expected.steps[position][row],
                &format!("{arch} step {position} beam {row}"),
            );
        }
    }
}

#[test]
fn lstm_export_scores_the_fixture_beams() {
    run("lstm");
}

#[test]
fn transformer_export_scores_the_fixture_beams() {
    run("transformer");
}
