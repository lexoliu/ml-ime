//! Both exported architectures run end to end: `start` is one `prefill` call
//! and `advance` stacks the surviving beams, against the gathered candidate
//! scores the Python side recorded from onnxruntime for the same prelude,
//! tokens and requests — and against the row the same step returns with the
//! whole alphabet asked.
//!
//! The fixtures under `tests/fixtures/` are written by
//! `python/scripts/charlm_fixtures.py`; regenerate them with
//! `cd python && uv run python scripts/charlm_fixtures.py
//! ../crates/ime-lm/tests/fixtures`.

mod common;

use common::{AskedRecord, alphabet, expected, fixture_dir, lexicon, shape};
use ime_decode::{Asked, Transition};
use ime_lm::{CharLm, LmState};
use ime_pinyin::Lexicon;
use std::fs;

/// *key*'s score out of *state* — `finish` for `"<eos>"`, `score` else.
fn read(model: &CharLm, lexicon: &Lexicon, state: &LmState, key: &str) -> f32 {
    if key == "<eos>" {
        model.finish(state)
    } else {
        let ch = key
            .chars()
            .next()
            .expect("a score key is one char or <eos>");
        model.score(
            state,
            lexicon
                .id_of(ch)
                .expect("fixture characters are in the lexicon"),
        )
    }
}

/// The column of a full row *key* names: the alphabet's index of the id.
fn column(alphabet: &[String], key: &str) -> usize {
    alphabet
        .iter()
        .position(|name| name == key)
        .expect("a score key names an alphabet id")
}

/// `start(Some(context))` with the fixture's first-position request, then two
/// `advance` calls of two beams each — every state's scores checked against
/// what onnxruntime gathered for the same request, and against the same
/// step's full-vocabulary row. *atol* is the fp32 fixtures' near-exact bound,
/// wider for fp16 whose half-precision arithmetic answers within a few
/// hundredths of a nat.
fn run(arch: &str, atol: f32) {
    let dir = fixture_dir();
    let lexicon = lexicon(&dir);
    let alphabet = alphabet(&dir, arch);
    let model = CharLm::open(&dir.join(arch), &lexicon, shape()).expect("the fixture opens");
    let expected = expected(&dir, arch);
    assert_eq!(expected.beams.len(), 2, "the fixture has two beams");

    let check = |state: &LmState, record: &AskedRecord, full: &[f32], what: &str| {
        assert_eq!(
            record.scores.len(),
            record.candidates.chars().count() + usize::from(record.eos),
            "{what}: the record's scores answer its request"
        );
        for (key, &want) in &record.scores {
            let got = read(&model, &lexicon, state, key);
            assert!(
                (got - want).abs() <= atol,
                "{what}: {key} got {got}, want {want}"
            );
            let column = column(&alphabet, key);
            assert!(
                (got - full[column]).abs() <= atol,
                "{what}: {key} gathered {got}, the full row's column {column} has {}",
                full[column]
            );
        }
    };

    let start_ids = expected.start.ids(&lexicon);
    let start = model.start(
        Some(&expected.context),
        &Asked {
            candidates: &start_ids,
            eos: expected.start.eos,
        },
    );
    check(&start, &expected.start, &expected.prefill_full, "prefill");

    // Every beam of a record starts from the same state; two clones of it are
    // the two beams the fixture recorded.
    let mut states = vec![start.clone(), start];
    // A step asked for every character plus <eos>: its gather answers the
    // whole row, and the narrow rows must match its columns exactly.
    let all: Vec<_> = lexicon
        .characters()
        .iter()
        .map(|&ch| lexicon.id_of(ch).expect("the lexicon indexes itself"))
        .collect();
    let wide = Asked {
        candidates: &all,
        eos: true,
    };

    let steps = expected.beams[0].chars().count();
    assert_eq!(expected.steps.len(), steps, "one expected row set per step");
    for (position, records) in expected.steps.iter().enumerate() {
        let ids: Vec<Vec<_>> = records.iter().map(|record| record.ids(&lexicon)).collect();
        let batch: Vec<(&LmState, ime_pinyin::CharId, Asked<'_>)> = states
            .iter()
            .zip(&expected.beams)
            .zip(records.iter().zip(&ids))
            .map(|((state, beam), (record, ids))| {
                let character = beam
                    .chars()
                    .nth(position)
                    .expect("every beam has two tokens");
                (
                    state,
                    lexicon
                        .id_of(character)
                        .expect("fixture characters are in the lexicon"),
                    Asked {
                        candidates: ids,
                        eos: record.eos,
                    },
                )
            })
            .collect();
        assert_eq!(
            batch.len(),
            expected.beams.len(),
            "every beam is fed a token"
        );
        // The same inputs asked for the whole alphabet: the gathered columns
        // are the full row's.
        let whole: Vec<(&LmState, ime_pinyin::CharId, Asked<'_>)> = batch
            .iter()
            .map(|(state, ch, _)| (*state, *ch, wide))
            .collect();
        let whole = model.advance(&whole);
        states = model.advance(&batch);
        assert_eq!(
            states.len(),
            expected.beams.len(),
            "one state per beam comes back"
        );
        for (row, state) in states.iter().enumerate() {
            let record = &records[row];
            let what = format!("{arch} step {position} beam {row}");
            check(state, record, &expected.full[position][row], &what);
            // The wide request's answers include every key the narrow one named.
            for (key, &want) in &record.scores {
                let got = read(&model, &lexicon, &whole[row], key);
                assert!(
                    (got - want).abs() <= atol,
                    "{what}: the whole row's {key} got {got}, want {want}"
                );
            }
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
    let expected = expected(&dir, "transformer");

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
    // Every row asks for the fixture's first step candidates, so all rows
    // answer the same keys.
    let asked_ids: Vec<_> = expected.steps[0][0].ids(&lexicon);
    let asked = Asked {
        candidates: &asked_ids,
        eos: expected.steps[0][0].eos,
    };
    // What each record's two beams score run on their own.
    let solo = |context: &str| {
        let start = model.start(Some(context), &asked);
        let states = [start.clone(), start];
        let batch: Vec<(&LmState, ime_pinyin::CharId, Asked<'_>)> = states
            .iter()
            .zip(0..2)
            .map(|(state, beam)| (state, token(beam), asked))
            .collect();
        model.advance(&batch)
    };
    let alone_long = solo(&expected.context);
    let alone_short = solo(&short);

    // One batch alternating the two records.
    let long = model.start(Some(&expected.context), &asked);
    let short_start = model.start(Some(&short), &asked);
    let batch: Vec<(&LmState, ime_pinyin::CharId, Asked<'_>)> =
        [&long, &short_start, &long, &short_start]
            .into_iter()
            .enumerate()
            .map(|(row, state)| (state, token(row / 2), asked))
            .collect();
    let mixed = model.advance(&batch);
    assert_eq!(mixed.len(), 4, "one state per row comes back");
    for (row, state) in mixed.iter().enumerate() {
        let solo = if row % 2 == 0 {
            &alone_long[row / 2]
        } else {
            &alone_short[row / 2]
        };
        for key in expected.steps[0][0].scores.keys() {
            let got = read(&model, &lexicon, state, key);
            let want = read(&model, &lexicon, solo, key);
            assert!(
                (got - want).abs() <= 1e-4,
                "mixed batch row {row}: {key} got {got}, solo {want}"
            );
        }
    }
}

/// `score` of a candidate the produced state was never asked for is a panic,
/// not a floor: the request rides with the row, and the model keeps no whole
/// row to fall back on.
#[test]
#[should_panic(expected = "the step never asked for")]
fn an_unasked_candidate_panics() {
    let dir = fixture_dir();
    let lexicon = lexicon(&dir);
    let model = CharLm::open(&dir.join("lstm"), &lexicon, shape()).expect("the fixture opens");
    let expected = expected(&dir, "lstm");
    let ids = expected.start.ids(&lexicon);
    let asked = Asked {
        candidates: &ids,
        eos: false,
    };
    let start = model.start(Some(&expected.context), &asked);
    // 谢 is not among the first position's candidates.
    let outside = lexicon.id_of('谢').expect("谢 is in the lexicon");
    let _ = model.score(&start, outside);
}

/// `finish` of a state whose request carried no `<eos>` fails fast — the
/// path cannot end there.
#[test]
#[should_panic(expected = "the step never asked for <eos>")]
fn an_unasked_eos_panics() {
    let dir = fixture_dir();
    let lexicon = lexicon(&dir);
    let model = CharLm::open(&dir.join("lstm"), &lexicon, shape()).expect("the fixture opens");
    let expected = expected(&dir, "lstm");
    let ids = expected.start.ids(&lexicon);
    let asked = Asked {
        candidates: &ids,
        eos: false,
    };
    let start = model.start(Some(&expected.context), &asked);
    let _ = model.finish(&start);
}

/// A manifest whose `layout` is not `"resident-candidates"` is an export
/// built for another batch layout: its graphs load and step until a
/// broadcasting operator fails on the wrong shape, so `open` refuses it --
/// naming the directory and the re-export -- whether the field holds a
/// foreign value or is absent, as every pre-candidates manifest is.
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
    let shape = ime_lm::SessionShape {
        backend: ime_lm::Backend::Cuda,
        ..ime_lm::SessionShape::default()
    };
    match CharLm::open(&dir.join("lstm"), &lexicon, shape) {
        Err(ime_lm::LmError::Provider { backend, .. }) => {
            assert_eq!(backend, ime_lm::Backend::Cuda);
        }
        Err(error) => panic!("expected the provider failure, got {error}"),
        Ok(_) => {}
    }
}

/// The fp16 exports open on the CPU backend -- an fp16 kernel's private
/// prepack must not trip the shared container -- and score the fixture beams
/// within 0.05 nats of what onnxruntime recorded of the same graphs.
#[test]
fn lstm_fp16_export_scores_the_fixture_beams() {
    run("lstm-fp16", 5e-2);
}

#[test]
fn lstm_export_scores_the_fixture_beams() {
    run("lstm", 1e-4);
}

#[test]
fn transformer_fp16_export_scores_the_fixture_beams() {
    run("transformer-fp16", 5e-2);
}

#[test]
fn transformer_export_scores_the_fixture_beams() {
    run("transformer", 1e-4);
}
