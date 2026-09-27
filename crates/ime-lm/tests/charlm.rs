//! Both exported architectures run end to end: `start` is one `prefill` call
//! and `advance` stacks the surviving beams, against log probabilities the
//! Python side recorded from onnxruntime for the same prelude and tokens.
//!
//! The fixtures under `tests/fixtures/` are written by
//! `python/scripts/charlm_fixtures.py`; regenerate them with
//! `cd python && uv run python scripts/charlm_fixtures.py
//! ../crates/ime-lm/tests/fixtures`.

use ime_decode::Transition;
use ime_lm::{CharLm, LmState, SessionShape};
use ime_pinyin::{CharId, Lexicon, SyllableTable};
use serde::Deserialize;
use std::fs;
use std::num::NonZeroUsize;
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

/// The fixture's beam width as the rectangle's, so a full batch is two live
/// rows and no dead row is ever fed.
fn shape() -> SessionShape {
    SessionShape {
        width: NonZeroUsize::new(2).expect("two is not zero"),
        ..SessionShape::default()
    }
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
    let model = CharLm::open(&dir.join(arch), &lexicon, shape()).expect("the fixture opens");
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

/// Rows from different records in one `advance` keep their own prefixes: a
/// batch alternating two preludes of different lengths -- four workers of one
/// live row each, the shorter padded, a dead row under every worker -- scores
/// each row as if it had run alone.
#[test]
fn transformer_rows_from_different_records_match_solo_runs() {
    let dir = fixture_dir();
    let lexicon = lexicon(&dir);
    let model =
        CharLm::open(&dir.join("transformer"), &lexicon, shape()).expect("the fixture opens");
    let expected: Expected = serde_json::from_str(
        &fs::read_to_string(dir.join("transformer").join("expected.json"))
            .expect("expected.json is committed with the fixture"),
    )
    .expect("expected.json parses");

    // A second prelude half as long, so its row of the batch is padded.
    let short: String = expected
        .context
        .chars()
        .take(expected.context.chars().count() / 2)
        .collect();
    let token = |beam: usize| {
        let character = expected.beams[beam]
            .chars()
            .next()
            .expect("every beam has tokens");
        lexicon
            .id_of(character)
            .expect("fixture characters are in the lexicon")
    };
    // What each record's two beams score run on their own.
    let solo = |context: &str| {
        let start = model.start(Some(context));
        let states = [start.clone(), start];
        let batch: Vec<(&LmState, CharId)> = states
            .iter()
            .zip(0..2)
            .map(|(state, beam)| (state, token(beam)))
            .collect();
        model.advance(&batch)
    };
    let alone_long = solo(&expected.context);
    let alone_short = solo(&short);

    // One batch alternating the two records.
    let long = model.start(Some(&expected.context));
    let short_start = model.start(Some(&short));
    let batch: Vec<(&LmState, CharId)> = [&long, &short_start, &long, &short_start]
        .into_iter()
        .enumerate()
        .map(|(row, state)| (state, token(row / 2)))
        .collect();
    let mixed = model.advance(&batch);
    assert_eq!(mixed.len(), 4, "one state per row comes back");
    for (row, state) in mixed.iter().enumerate() {
        let solo = if row % 2 == 0 {
            &alone_long[row / 2]
        } else {
            &alone_short[row / 2]
        };
        assert_close(
            state.log_probs(),
            solo.log_probs(),
            &format!("mixed batch row {row}"),
        );
    }
}

/// A manifest whose `layout` is not `"rectangular"` is an export built for
/// another batch layout: its graphs load and step until a broadcasting
/// operator fails on the wrong shape, so `open` refuses it -- naming the
/// directory and the re-export -- whether the field holds a foreign value or
/// is absent, as every pre-rectangle manifest is.
#[test]
fn a_manifest_of_another_layout_is_refused() {
    let dir = fixture_dir();
    let lexicon = lexicon(&dir);
    let fixture: serde_json::Map<String, serde_json::Value> = serde_json::from_str(
        &fs::read_to_string(dir.join("transformer").join("charlm.json"))
            .expect("the fixture manifest is committed"),
    )
    .expect("the fixture manifest parses");

    let temp = std::env::temp_dir().join(format!("ime-lm-layout-{}", std::process::id()));
    fs::create_dir_all(&temp).expect("the temp dir is made");
    for layout in [Some("per-row"), None] {
        let mut manifest = fixture.clone();
        match layout {
            Some(layout) => {
                manifest.insert("layout".to_owned(), layout.into());
            }
            None => {
                manifest.remove("layout");
            }
        }
        fs::write(
            temp.join("charlm.json"),
            serde_json::to_string(&manifest).expect("the manifest serialises"),
        )
        .expect("the manifest is written");
        // `open` must refuse at the manifest, before the graphs -- absent in
        // the temp dir -- are ever read.
        match CharLm::open(&temp, &lexicon, shape()) {
            Err(error @ ime_lm::LmError::Layout { .. }) => {
                let message = error.to_string();
                assert!(
                    message.contains(&temp.display().to_string()),
                    "the error names the export's directory: {message}"
                );
                assert!(
                    message.contains("re-export"),
                    "the error says to re-export: {message}"
                );
            }
            Err(error) => panic!("expected the layout refusal, got {error}"),
            Ok(_) => panic!("a manifest with layout {layout:?} opened anyway"),
        }
    }
    fs::remove_dir_all(&temp).ok();
}

/// A `gpu-cuda` build asked for CUDA where the provider cannot initialise --
/// no driver, no device, no toolkit libraries -- fails `CharLm::open` with an
/// error naming the backend, rather than silently running on CPU. On a machine
/// where CUDA does initialise (the Kaggle T4) the open succeeds and there is
/// nothing to assert.
#[cfg(feature = "gpu-cuda")]
#[test]
fn cuda_backend_fails_where_cuda_cannot_initialise() {
    let dir = fixture_dir();
    let lexicon = lexicon(&dir);
    let shape = SessionShape {
        backend: ime_lm::Backend::Cuda,
        ..SessionShape::default()
    };
    match CharLm::open(&dir.join("lstm"), &lexicon, shape) {
        Err(ime_lm::LmError::Provider { backend, .. }) => {
            assert_eq!(backend, ime_lm::Backend::Cuda);
        }
        Err(error) => panic!("expected the provider failure, got {error}"),
        Ok(_) => {}
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
