//! The key engine over the converted tiny model, held to the Python Core ML
//! search.
//!
//! `tests/fixtures/keys/expected.json` is what `python/scripts/keys_fixtures.py`
//! writes: for typed inputs of every style, with and without a context in
//! Chinese, Japanese or Korean, what `mlime.train.key_coreml.CoreMlKeyDecoder`
//! returned over the compiled model beside it (CPU only) and the tree calls it
//! made. The Rust search over [`CoreMlKeyModel`] has to make the same calls in
//! the same order and return the same top hypotheses, scores within 1e-4: the
//! port parity gate of the Core ML design.

#![cfg(target_vendor = "apple")]

use std::convert::Infallible;
use std::fs;
use std::num::NonZeroUsize;

use ime_decode::{Combine, DEFAULT_ARC_WIDTH, KeySearchOptions, Weight};
use ime_neural::keys::{
    ComputeUnits, CoreMlEncoding, CoreMlKeyError, CoreMlKeyModel, CoreMlLogits, KeyModel,
    POLICY_VERSION, TreeCall,
};
use ime_pinyin::{EmitId, Keys, Legality, LegalityOptions};
use ime_session::{
    Bounded, ByLength, Engine, KeyEngine, KeyEngineOptions, Pending, Placed, Sentence, Session,
    SessionOptions,
};
use serde::Deserialize;

/// How far a score may stray from the Python search's.
const SCORE_TOLERANCE: f64 = 1e-4;

/// How far a keystroke-by-keystroke score may stray from the one-shot search's
/// over the same input: a row's fp16 value depends at the last bit on the call
/// that computed it, and typing changes which call that is, so the sums of a
/// sentence's rows differ by a few fp16 ulps of the logits.
const TYPING_TOLERANCE: f64 = 1e-2;

#[derive(Deserialize)]
struct Expected {
    policy_version: u32,
    legality: LegalityRecord,
    beam: usize,
    top_k: usize,
    weights: Vec<WeightsRecord>,
    records: Vec<Record>,
}

#[derive(Deserialize)]
struct LegalityRecord {
    unfinished_tail: bool,
    correct_syllables: bool,
}

#[derive(Deserialize)]
struct WeightsRecord {
    beta: f64,
    gamma: f64,
    viterbi: bool,
}

#[derive(Deserialize)]
struct Record {
    name: String,
    keys: String,
    context: Option<String>,
    searches: Vec<Search>,
}

#[derive(Deserialize)]
struct Search {
    weights: usize,
    tree_calls: Vec<Call>,
    hypotheses: Vec<Hypothesis>,
}

#[derive(Deserialize, Debug, PartialEq, Eq)]
struct Call {
    bucket: usize,
    nodes: usize,
}

#[derive(Deserialize)]
struct Hypothesis {
    text: String,
    cuts: Vec<usize>,
    score: f64,
}

/// The Core ML model, logging every tree call it runs.
struct Recording<'m> {
    model: &'m mut CoreMlKeyModel,
    calls: Vec<Call>,
}

impl KeyModel for Recording<'_> {
    type Error = CoreMlKeyError;
    type Encoding = CoreMlEncoding;
    type Logits = CoreMlLogits;

    fn max_keys(&self) -> usize {
        self.model.max_keys()
    }

    fn encode(
        &mut self,
        keys: &Keys,
        context: Option<&str>,
        log_probs: &mut Vec<f32>,
    ) -> Result<CoreMlEncoding, CoreMlKeyError> {
        self.model.encode(keys, context, log_probs)
    }

    fn tree(
        &mut self,
        encoding: &CoreMlEncoding,
        call: &TreeCall<'_, EmitId>,
    ) -> Result<CoreMlLogits, CoreMlKeyError> {
        self.calls.push(Call {
            bucket: call.bucket(),
            nodes: call.nodes().len(),
        });
        self.model.tree(encoding, call)
    }

    fn log_softmax(&self, logits: &CoreMlLogits, node: usize, out: &mut [f32]) {
        self.model.log_softmax(logits, node, out);
    }
}

fn open() -> CoreMlKeyModel {
    CoreMlKeyModel::open(&ime_fixtures::keys().join("model"), ComputeUnits::CpuOnly)
        .expect("the fixture model opens")
}

fn expected() -> Expected {
    let path = ime_fixtures::keys().join("expected.json");
    serde_json::from_str(&fs::read_to_string(&path).expect("expected.json reads"))
        .expect("expected.json parses")
}

fn options(expected: &Expected, point: &WeightsRecord) -> KeyEngineOptions {
    KeyEngineOptions {
        legality: LegalityOptions {
            unfinished_tail: expected.legality.unfinished_tail,
            correct_syllables: expected.legality.correct_syllables,
        },
        search: KeySearchOptions {
            beta: Weight::try_from(point.beta).expect("a weight"),
            gamma: Weight::try_from(point.gamma).expect("a weight"),
            beam: NonZeroUsize::new(expected.beam),
            top_k: NonZeroUsize::new(expected.top_k),
            combine: if point.viterbi {
                Combine::Viterbi
            } else {
                Combine::Sum
            },
            prior: Weight::ZERO,
            arc_width: DEFAULT_ARC_WIDTH,
        },
    }
}

#[test]
fn the_rust_search_makes_the_python_calls_and_returns_its_hypotheses() {
    let expected = expected();
    assert_eq!(expected.policy_version, POLICY_VERSION);
    let mut model = open();
    let legality =
        Legality::over(model.characters()).expect("the model's characters have readings");
    for record in &expected.records {
        let pending = Pending::parse(&record.keys)
            .expect("the keys are letters")
            .expect("the keys are not empty");
        for search in &record.searches {
            let name = format!("{} at weights {}", record.name, search.weights);
            let point = &expected.weights[search.weights];
            let mut engine = KeyEngine::new(
                Recording {
                    model: &mut model,
                    calls: Vec::new(),
                },
                &legality,
                options(&expected, point),
            );
            let found = engine
                .sentences(&pending, record.context.as_deref())
                .expect("the search runs");
            assert_eq!(
                engine.model().calls,
                search.tree_calls,
                "{name}: the tree calls"
            );
            assert_eq!(
                found.len(),
                search.hypotheses.len(),
                "{name}: the hypotheses"
            );
            for (rank, (sentence, want)) in found.iter().zip(&search.hypotheses).enumerate() {
                let text: String = sentence.chars.iter().map(|placed| placed.ch).collect();
                let cuts: Vec<usize> = sentence.chars.iter().map(|placed| placed.end).collect();
                assert_eq!(text, want.text, "{name}: hypothesis {rank}");
                assert_eq!(cuts, want.cuts, "{name}: hypothesis {rank}'s cuts");
                let gap = (f64::from(sentence.score) - want.score).abs();
                assert!(
                    gap <= SCORE_TOLERANCE,
                    "{name}: hypothesis {rank} scores {}, Python {}",
                    sentence.score,
                    want.score
                );
            }
        }
    }
}

#[test]
fn more_keys_than_the_largest_function_holds_are_an_error_from_encode() {
    let mut model = open();
    let keys = Keys::new(&"a".repeat(127)).expect("letters");
    let mut log_probs = Vec::new();
    let Err(error) = model.encode(&keys, None, &mut log_probs) else {
        panic!("127 keys fit a function that holds 126");
    };
    assert!(
        matches!(
            error,
            CoreMlKeyError::TooManyKeys {
                keys: 127,
                largest: 126
            }
        ),
        "{error}"
    );
}

/// The engine for input past the key model's limit: one fixed sentence, so a
/// test sees which engine answered.
struct Beyond;

impl Engine for Beyond {
    type Error = Infallible;

    fn sentences(
        &mut self,
        pending: &Pending,
        _context: Option<&str>,
    ) -> Result<Vec<Sentence>, Infallible> {
        Ok(vec![Sentence {
            chars: vec![Placed {
                ch: '?',
                end: pending.letters().len(),
                score: 0.0,
            }],
            score: 0.0,
        }])
    }

    fn clear(&mut self) {}
}

#[test]
fn a_warmed_model_answers_what_a_cold_one_answers() {
    let expected = expected();
    let point = &expected.weights[0];
    let mut warmed = open();
    warmed.warm_up().expect("every function runs once");
    // A second warm-up is harmless: it leaves nothing a keystroke reads.
    warmed.warm_up().expect("the warm-up repeats");
    let legality = Legality::over(warmed.characters()).expect("the characters have readings");
    let mut warm = KeyEngine::new(warmed, &legality, options(&expected, point));
    let mut cold = KeyEngine::new(open(), &legality, options(&expected, point));
    for record in &expected.records {
        let pending = Pending::parse(&record.keys)
            .expect("the keys are letters")
            .expect("the keys are not empty");
        let after = warm
            .sentences(&pending, record.context.as_deref())
            .expect("the warmed search runs");
        let without = cold
            .sentences(&pending, record.context.as_deref())
            .expect("the cold search runs");
        assert_eq!(after.len(), without.len(), "{}", record.name);
        for (rank, (warm, cold)) in after.iter().zip(&without).enumerate() {
            let text = |sentence: &Sentence| -> String {
                sentence.chars.iter().map(|placed| placed.ch).collect()
            };
            assert_eq!(text(warm), text(cold), "{}: sentence {rank}", record.name);
            assert!(
                (f64::from(warm.score) - f64::from(cold.score)).abs() <= SCORE_TOLERANCE,
                "{}: sentence {rank} scores {} warmed, {} cold",
                record.name,
                warm.score,
                cold.score
            );
        }
    }
}

#[test]
fn the_limit_is_the_largest_keys_function() {
    let model = open();
    assert_eq!(model.max_keys(), 126);
}

#[test]
fn a_session_typed_key_by_key_ends_at_the_sentences_of_the_whole_input() {
    let expected = expected();
    let point = &expected.weights[0];
    let legality = Legality::over(open().characters()).expect("the characters have readings");
    for record in &expected.records {
        let name = &record.name;
        let mut session = Session::new(
            KeyEngine::new(open(), &legality, options(&expected, point)),
            SessionOptions { sentences: 3 },
        );
        session
            .set_context(record.context.as_deref().unwrap_or_default())
            .expect("a context sets");
        for ch in record.keys.chars() {
            session.key(ch).expect("a keystroke decodes");
        }
        let pending = Pending::parse(&record.keys)
            .expect("the keys are letters")
            .expect("the keys are not empty");
        let once = KeyEngine::new(open(), &legality, options(&expected, point))
            .sentences(&pending, record.context.as_deref())
            .expect("the search runs");
        assert!(!once.is_empty(), "{name}: the input reads");
        assert_eq!(session.sentences().len(), once.len(), "{name}");
        for (rank, (typed, whole)) in session.sentences().iter().zip(&once).enumerate() {
            let text = |sentence: &Sentence| -> String {
                sentence.chars.iter().map(|placed| placed.ch).collect()
            };
            let cuts = |sentence: &Sentence| -> Vec<usize> {
                sentence.chars.iter().map(|placed| placed.end).collect()
            };
            assert_eq!(text(typed), text(whole), "{name}: sentence {rank}");
            assert_eq!(cuts(typed), cuts(whole), "{name}: sentence {rank}'s cuts");
            let gap = f64::from(typed.score - whole.score).abs();
            assert!(
                gap <= TYPING_TOLERANCE,
                "{name}: sentence {rank} scores {}, once {}",
                typed.score,
                whole.score
            );
        }
    }
}

#[test]
fn more_keys_than_the_model_holds_go_to_the_second_engine() {
    let expected = expected();
    let legality = Legality::over(open().characters()).expect("the characters have readings");
    let key_engine = KeyEngine::new(open(), &legality, options(&expected, &expected.weights[0]));
    assert_eq!(key_engine.max_keys(), 126);
    let mut chosen = ByLength::new(key_engine, Beyond);
    let past = Pending::parse(&"a".repeat(127))
        .expect("letters")
        .expect("not empty");
    let found = chosen
        .sentences(&past, None)
        .expect("the second engine answers");
    assert_eq!(found[0].chars[0].ch, '?');
    assert_eq!(found[0].chars[0].end, 127);
}
