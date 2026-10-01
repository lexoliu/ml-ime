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

/// The int8 export's quantisation is row-independent: a row advanced alone
/// and again inside a 128-row batch of unrelated rows gets bit-for-bit the
/// same candidate scores. `quantize_dynamic`'s `DynamicQuantizeLinear` shares
/// one activation scale across the whole batch, so the legacy export moved
/// every row by up to ~0.13 nats here (issue #93).
#[test]
fn int8_scores_do_not_depend_on_batchmates() {
    let dir = fixture_dir();
    let lexicon = lexicon(&dir);
    for arch in ["transformer-int8", "lstm-int8"] {
        let model =
            CharLm::open(&dir.join(arch), &lexicon, shape()).expect("the int8 fixture opens");
        let expected = expected(&dir, arch);
        let ids = expected.steps[0][0].ids(&lexicon);
        let asked = Asked {
            candidates: &ids,
            eos: expected.steps[0][0].eos,
        };
        let alphabet: Vec<_> = "你好吗我很再见谢"
            .chars()
            .map(|ch| lexicon.id_of(ch).expect("fixture chars are in the lexicon"))
            .collect();

        // Unrelated states: a shorter context alternating with the fixture's,
        // each row advanced alone on its own first character.
        let short: String = expected
            .context
            .chars()
            .take(expected.context.chars().count() / 2)
            .collect();
        let token = |beam: usize| alphabet[beam % alphabet.len()];
        let mut fillers = Vec::new();
        for row in 0..127 {
            let context = if row % 2 == 0 {
                &expected.context
            } else {
                &short
            };
            let start = model.start(Some(context), &asked);
            fillers.push(model.advance(&[(&start, token(row), asked)]).remove(0));
        }

        // The target row: the fixture context's first beam advanced on its
        // second character, alone and in the batch.
        let start = model.start(Some(&expected.context), &asked);
        let target = alphabet[1];
        let solo = model.advance(&[(&start, target, asked)]).remove(0);
        let batch: Vec<(&LmState, ime_pinyin::CharId, Asked<'_>)> = [(&start, target, asked)]
            .into_iter()
            .chain(
                fillers
                    .iter()
                    .enumerate()
                    .map(|(row, state)| (state, token(row + 1), asked)),
            )
            .collect();
        let mixed = model.advance(&batch);
        assert_eq!(mixed.len(), 128, "one state per row comes back");

        let keys: Vec<&String> = expected.steps[0][0].scores.keys().collect();
        for key in &keys {
            assert_eq!(
                read(&model, &lexicon, &mixed[0], key).to_bits(),
                read(&model, &lexicon, &solo, key).to_bits(),
                "{arch}: {key} for the target row moves with its batchmates"
            );
        }
        // The fillers' rows are covered the same way: each repeats its own
        // advance alone and the answers must be identical.
        for (row, state) in fillers.iter().enumerate() {
            let alone = model.advance(&[(state, token(row + 1), asked)]).remove(0);
            for key in &keys {
                assert_eq!(
                    read(&model, &lexicon, &mixed[row + 1], key).to_bits(),
                    read(&model, &lexicon, &alone, key).to_bits(),
                    "{arch}: {key} for filler row {row} moves with its batchmates"
                );
            }
        }
    }
}

/// A paged state's scores must not depend on pool pressure: advancing the
/// same character history under a pool too small for the live set — every
/// later claim a reclaim or a degrade to `Pending` — must score, bit for
/// bit, what a roomy pool's fresh prefill-plus-advance of that history
/// does. The replay's 0-mismatch gate is this identity at record scale;
/// this is the per-step assert of it.
#[test]
fn reclaimed_pages_score_like_a_roomy_pool() {
    let dir = fixture_dir();
    let lexicon = lexicon(&dir);
    let arch = "transformer";

    let mut tight = shape();
    tight.cache_rows = 16;
    let mut roomy = shape();
    roomy.cache_rows = 4096;
    let tight_lm = CharLm::open(&dir.join(arch), &lexicon, tight).expect("the fixture opens");
    let roomy_lm = CharLm::open(&dir.join(arch), &lexicon, roomy).expect("the fixture opens");

    let ids: Vec<_> = lexicon
        .characters()
        .iter()
        .map(|&ch| lexicon.id_of(ch).expect("the lexicon indexes itself"))
        .collect();
    // The fixture alphabet is eight characters; the context's prelude plus
    // a step of claims already overruns the tight pool, so every later
    // claim is a reclaim or a degrade.
    let context: String = lexicon.characters().iter().collect();
    let tokens: Vec<_> = lexicon.characters()[..8]
        .iter()
        .map(|&ch| lexicon.id_of(ch).expect("the lexicon indexes itself"))
        .collect();

    let asked = Asked {
        candidates: &ids,
        eos: false,
    };
    let mut tight_beams = vec![tight_lm.start(Some(&context), &asked)];
    let mut roomy_beams = vec![roomy_lm.start(Some(&context), &asked)];
    for step in 0..6 {
        let (a, b) = (
            tokens[step % tokens.len()],
            tokens[(step + 3) % tokens.len()],
        );
        let produced_t =
            tight_lm.advance(&[(&tight_beams[0], a, asked), (&tight_beams[0], b, asked)]);
        let produced_r =
            roomy_lm.advance(&[(&roomy_beams[0], a, asked), (&roomy_beams[0], b, asked)]);
        assert_eq!(produced_t.len(), produced_r.len());
        for (beam, (tight_state, roomy_state)) in produced_t.iter().zip(&produced_r).enumerate() {
            for &ch in &ids {
                assert_eq!(
                    tight_lm.score(tight_state, ch).to_bits(),
                    roomy_lm.score(roomy_state, ch).to_bits(),
                    "step {step}, beam {beam}: pool pressure moved a score"
                );
            }
        }
        tight_beams = produced_t;
        roomy_beams = produced_r;
    }
}

/// The product shape is one session per pool: a lone `CharLm` driven
/// through a fixture sentence of `start`s and chained `advance`s must
/// never degrade a produced state to `Pending`. The replay regressed
/// exactly here — a second session sharing the pool starved the gate's
/// resolves — so the lone-session invariant gets its own assert.
#[test]
fn a_lone_session_materialises_no_pending_rows() {
    let dir = fixture_dir();
    let lexicon = lexicon(&dir);
    let arch = "transformer";
    let mut roomy = shape();
    roomy.cache_rows = 4096;
    let lm = CharLm::open(&dir.join(arch), &lexicon, roomy).expect("the fixture opens");

    let ids: Vec<_> = lexicon
        .characters()
        .iter()
        .map(|&ch| lexicon.id_of(ch).expect("the lexicon indexes itself"))
        .collect();
    let asked = Asked {
        candidates: &ids,
        eos: false,
    };
    // A fixture sentence: a context of the whole alphabet, then chained
    // advances that keep every produced state alive — the working set one
    // session's decode actually holds.
    let alphabet: Vec<char> = lexicon.characters().to_vec();
    let context: String = alphabet.iter().collect();
    let tokens: Vec<_> = alphabet[..8]
        .iter()
        .map(|&ch| lexicon.id_of(ch).expect("the lexicon indexes itself"))
        .collect();
    let mut beams = Vec::new();
    for shift in 0..8 {
        let ctx: String = (0..16)
            .map(|i| alphabet[(i + shift) % alphabet.len()])
            .collect();
        beams.push(lm.start(Some(&ctx), &asked));
    }
    assert!(!context.is_empty());
    for step in 0..12 {
        let (a, b) = (
            tokens[step % tokens.len()],
            tokens[(step + 3) % tokens.len()],
        );
        let mut next = Vec::new();
        for beam in &beams {
            let produced = lm.advance(&[(beam, a, asked), (beam, b, asked)]);
            for state in &produced {
                assert!(
                    !state.is_pending(),
                    "step {step}: a lone session's produced state went Pending"
                );
            }
            next.extend(produced);
        }
        // Keep the frontier bounded like a beam does: the newest rows of
        // every chain stay live, the rest drop dead-leaf-reclaimable.
        beams = next.split_off(next.len().saturating_sub(64));
    }
    let (mat_ns, mat_rows) = lm.bk_mat();
    assert_eq!(
        (mat_rows, mat_ns),
        (0, 0),
        "a lone session materialised pending rows"
    );
    assert!(
        lm.cache_stats().is_some_and(|stats| stats.live > 0),
        "no pool node was ever claimed — the fixture exercised nothing"
    );
}

/// A `Pending` state — one whose claims failed because every page in the
/// pool is live — feeds its chain through the scratch generation instead
/// of the paged gather, and the scratch write must land each position's
/// `[H, D]` slice where the graph's `[slots, L, H, T, D]` layout reads it.
/// Context nodes are unstealable (`ch: None` skips the sweep), so enough
/// 62-character contexts pin the floored pool shut and starts go Pending.
#[test]
fn pending_states_score_like_resident_pages() {
    let dir = fixture_dir();
    let lexicon = lexicon(&dir);
    let arch = "transformer";

    let mut starved = shape();
    starved.cache_rows = 1;
    let mut roomy = shape();
    roomy.cache_rows = 4096;
    let starved_lm = CharLm::open(&dir.join(arch), &lexicon, starved).expect("the fixture opens");
    let roomy_lm = CharLm::open(&dir.join(arch), &lexicon, roomy).expect("the fixture opens");

    let ids: Vec<_> = lexicon
        .characters()
        .iter()
        .map(|&ch| lexicon.id_of(ch).expect("the lexicon indexes itself"))
        .collect();
    let asked = Asked {
        candidates: &ids,
        eos: false,
    };
    // ~29 unstealable pages per context against a 1,024-page pool —
    // and under the fixture's 32-position ceiling — the thirty-sixth
    // context onward leaves starts with unresolved tips.
    let alphabet: Vec<char> = lexicon.characters().to_vec();
    let context = |shift: usize| -> String {
        (0..29)
            .map(|i| alphabet[(i + shift) % alphabet.len()])
            .collect()
    };
    let mut starved_states = Vec::new();
    let mut roomy_states = Vec::new();
    for shift in 0..40 {
        let ctx = context(shift);
        starved_states.push(starved_lm.start(Some(&ctx), &asked));
        roomy_states.push(roomy_lm.start(Some(&ctx), &asked));
    }
    assert!(
        starved_states.iter().any(LmState::is_pending),
        "1,200 pages of context resolved anyway — the scratch path went unexercised"
    );

    for (index, (starved_state, roomy_state)) in
        starved_states.iter().zip(&roomy_states).enumerate()
    {
        let ch = ids[index % ids.len()];
        let produced_s = starved_lm.advance(&[(starved_state, ch, asked)]);
        let produced_r = roomy_lm.advance(&[(roomy_state, ch, asked)]);
        assert_eq!(produced_s.len(), produced_r.len());
        for (beam, (s, r)) in produced_s.iter().zip(&produced_r).enumerate() {
            for &cand in &ids {
                assert_eq!(
                    starved_lm.score(s, cand).to_bits(),
                    roomy_lm.score(r, cand).to_bits(),
                    "context {index}, beam {beam}: a pending input moved a score"
                );
            }
        }
    }
}

/// The control the pending test is meaningless without: two different
/// histories advanced by the same character must score differently —
/// proof the fixture's logits read the cache bytes at all.
#[test]
fn history_changes_scores() {
    let dir = fixture_dir();
    let lexicon = lexicon(&dir);
    let arch = "transformer";
    let mut roomy = shape();
    roomy.cache_rows = 4096;
    let roomy_lm = CharLm::open(&dir.join(arch), &lexicon, roomy).expect("the fixture opens");

    let ids: Vec<_> = lexicon
        .characters()
        .iter()
        .map(|&ch| lexicon.id_of(ch).expect("the lexicon indexes itself"))
        .collect();
    let asked = Asked {
        candidates: &ids,
        eos: false,
    };
    let alphabet: Vec<char> = lexicon.characters().to_vec();
    let a_ctx: String = alphabet[..3].iter().collect();
    let b_ctx: String = alphabet[..3].iter().rev().collect();
    let a = roomy_lm.start(Some(&a_ctx), &asked);
    let b = roomy_lm.start(Some(&b_ctx), &asked);
    let ch = ids[0];
    let pa = roomy_lm.advance(&[(&a, ch, asked)]);
    let pb = roomy_lm.advance(&[(&b, ch, asked)]);
    assert!(
        ids.iter()
            .any(|&cand| roomy_lm.score(&pa[0], cand).to_bits()
                != roomy_lm.score(&pb[0], cand).to_bits()),
        "the fixture's logits ignore history — cache-content tests are vacuous"
    );
}

/// `advance` on one `CharLm` from several threads at once — the CPU
/// `sections` path's shape — must score bit for bit what the same calls
/// do serially. A scratch generation shared across threads or a reclaim
/// racing a resolve would interleave writes and flip a score, so the run
/// pushes both: `Pending` inputs against a starved pool, and claims on a
/// pool under eviction pressure.
#[test]
fn concurrent_advances_score_like_serial_ones() {
    const THREADS: usize = 4;
    let dir = fixture_dir();
    let lexicon = lexicon(&dir);
    let arch = "transformer";

    let mut tight = shape();
    tight.cache_rows = 1;
    let lm = CharLm::open(&dir.join(arch), &lexicon, tight).expect("the fixture opens");

    let ids: Vec<_> = lexicon
        .characters()
        .iter()
        .map(|&ch| lexicon.id_of(ch).expect("the lexicon indexes itself"))
        .collect();
    let asked = Asked {
        candidates: &ids,
        eos: false,
    };
    // Thirty-eight contexts of ~29 unstealable pages each overfill the
    // 1,024-page floored pool, so some starts go `Pending` — the states a
    // `step_pages` call writes through the scratch generation.
    let alphabet: Vec<char> = lexicon.characters().to_vec();
    let context = |shift: usize| -> String {
        (0..29)
            .map(|i| alphabet[(i + shift) % alphabet.len()])
            .collect()
    };
    let states: Vec<LmState> = (0..38)
        .map(|shift| lm.start(Some(&context(shift)), &asked))
        .collect();
    assert!(
        states.iter().any(LmState::is_pending),
        "the pool resolved every context — the scratch path went unexercised"
    );
    // One mixed call: `Path` and `Pending` rows in a single step, so the
    // call both reads pages and materialises scratch slots.
    let steps: Vec<(&LmState, ime_pinyin::CharId, Asked<'_>)> = states
        .iter()
        .enumerate()
        .map(|(i, state)| (state, ids[i % ids.len()], asked))
        .collect();
    let score_bits = |produced: &[LmState]| -> Vec<u32> {
        produced
            .iter()
            .flat_map(|state| {
                ids.iter()
                    .map(|&cand| lm.score(state, cand).to_bits())
                    .collect::<Vec<_>>()
            })
            .collect()
    };
    let serial = score_bits(&lm.advance(&steps));
    std::thread::scope(|scope| {
        let mut joins = Vec::with_capacity(THREADS);
        for _ in 0..THREADS {
            joins.push(scope.spawn(|| score_bits(&lm.advance(&steps))));
        }
        for (thread, join) in joins.into_iter().enumerate() {
            let parallel = join.join().expect("a concurrent advance panicked");
            assert_eq!(
                parallel, serial,
                "thread {thread}: a concurrent advance moved a score"
            );
        }
    });
}
