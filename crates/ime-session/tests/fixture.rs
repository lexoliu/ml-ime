//! The session end to end on the committed fixture towers: replay a record
//! keystroke by keystroke and its last top candidate must equal the answer
//! `decode_many` computes for the whole input — the equality `ime-cli replay`
//! asserts per record, proven on the fixture where both directions are cheap.
//!
//! The towers are the `fp32` fixture export under
//! `crates/ime-neural/tests/fixtures/`, the candidate sets are the real
//! generated tables, and the emittable restriction is the fixture's nine
//! characters.

use ime_decode::{
    BeamOptions, Candidates, Emittable, LatticeRecord, NoTransition, Record, Scored, Weighted,
    decode_many, lattice_paths,
};
use ime_neural::{RouteA, SessionShape};
use ime_pinyin::{Lexicon, SegmentLattice, SegmentOptions, SyllableTable};
use ime_session::{Fused, FusedConfig, Session, SessionOptions as Options};
use std::path::{Path, PathBuf};

/// `crates/ime-neural/tests/fixtures` — the fp32 export lives under `fp32/`.
fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../ime-neural/tests/fixtures")
}

/// The generated lexicon every table is indexed through.
fn lexicon() -> Lexicon {
    Lexicon::load(&SyllableTable::load()).expect("the committed lexicon loads")
}

/// The fixture's nine emittable characters.
fn emittable(fixtures: &Path, lexicon: &Lexicon) -> Emittable {
    let source = std::fs::read_to_string(fixtures.join("emittable.txt"))
        .expect("emittable.txt is committed");
    Emittable::parse(&source, lexicon).expect("the emittable set parses")
}

/// A session over the fixture towers, `incremental` switching the engine's
/// context reuse.
fn session<'a>(
    towers: &'a RouteA,
    table: &'a SyllableTable,
    lexicon: &'a Lexicon,
    emittable: &'a Emittable,
    incremental: bool,
) -> Session<'a, Fused<'a, NoTransition>> {
    Session::new(
        Fused::new(
            towers,
            NoTransition,
            FusedConfig {
                emittable,
                lexicon,
                weight: 1.0,
                floor: -30.0,
                with_context: true,
                incremental,
            },
        ),
        Options {
            lexicon,
            table,
            segment: SegmentOptions {
                allow_incomplete_tail: false,
                ..SegmentOptions::default()
            },
            beam: BeamOptions::default(),
            sentences: 3,
        },
    )
}

/// The answer a full decode gives the complete input — what `fused-eval`'s
/// live path computes for the same record.
fn reference_top(
    towers: &RouteA,
    pinyin: &str,
    context: Option<&str>,
    lexicon: &Lexicon,
    emittable: &Emittable,
) -> String {
    let options = SegmentOptions {
        allow_incomplete_tail: false,
        ..SegmentOptions::default()
    };
    let lattice =
        SegmentLattice::build(pinyin, &SyllableTable::load(), &options).expect("the pinyin reads");
    let segmentations = lattice.k_best(&options);
    let candidates = Candidates::build(&segmentations, lexicon).expect("the readings build");
    let record = LatticeRecord {
        record: 0,
        pinyin: pinyin.to_owned(),
        context: context.map(str::to_owned),
        paths: lattice_paths(pinyin, &segmentations, &candidates, emittable, lexicon),
    };
    let encoded = context
        .map(|text| towers.context(text))
        .transpose()
        .expect("encodes");
    let scores = towers
        .emission_with(&record, None, encoded.as_ref())
        .expect("the towers emit");
    let emission = Weighted {
        inner: Scored::attach(0, &candidates, emittable, scores, -30.0).expect("scores attach"),
        weight: 1.0,
    };
    let mut answers = decode_many(
        &[Record {
            candidates: &candidates,
            emission,
            context,
        }],
        &NoTransition,
        &BeamOptions::default(),
    )
    .expect("the record decodes");
    answers
        .pop()
        .expect("one record decodes to one answer")
        .first()
        .expect("a decode has a top candidate")
        .text(lexicon)
}

/// Replay `pinyin` one keystroke at a time on *session* and return the last
/// keystroke's top candidate.
fn replay_top<E>(session: &mut Session<'_, E>, pinyin: &str) -> Option<String>
where
    E: ime_session::Engine,
{
    for ch in pinyin.chars() {
        session.key(ch).expect("a keystroke decodes");
    }
    session.candidates().first().map(|c| c.text.clone())
}

#[test]
fn replay_equals_full_decode() {
    let dir = fixture_dir();
    let towers =
        RouteA::open(&dir.join("fp32"), SessionShape::default()).expect("the fixture towers open");
    let table = SyllableTable::load();
    let lexicon = lexicon();
    let emittable = emittable(&dir, &lexicon);
    for (pinyin, context) in [("zhongwo", Some("北京大学")), ("woaibeijing", None)] {
        let want = reference_top(&towers, pinyin, context, &lexicon, &emittable);
        for incremental in [false, true] {
            let mut session = session(&towers, &table, &lexicon, &emittable, incremental);
            if let Some(context) = context {
                session.set_context(context).expect("context sets");
            }
            let got = replay_top(&mut session, pinyin);
            assert_eq!(
                got.as_deref(),
                Some(want.as_str()),
                "incremental={incremental}: {pinyin:?} top-1 differs from the full decode"
            );
            assert_eq!(session.pending(), pinyin.len());
        }
    }
}

#[test]
fn apostrophe_is_a_boundary() {
    let dir = fixture_dir();
    let towers =
        RouteA::open(&dir.join("fp32"), SessionShape::default()).expect("the fixture towers open");
    let table = SyllableTable::load();
    let lexicon = lexicon();
    let emittable = emittable(&dir, &lexicon);
    let mut session = session(&towers, &table, &lexicon, &emittable, false);
    for ch in "zhong'wo".chars() {
        session.key(ch).expect("a keystroke decodes");
    }
    let with_boundary: Vec<String> = session
        .candidates()
        .iter()
        .map(|c| c.text.clone())
        .collect();
    session.reset();
    for ch in "zhongwo".chars() {
        session.key(ch).expect("a keystroke decodes");
    }
    let without: Vec<String> = session
        .candidates()
        .iter()
        .map(|c| c.text.clone())
        .collect();
    assert_eq!(
        with_boundary, without,
        "an explicit boundary inside a fixed segmentation changes nothing"
    );
    assert_eq!(session.pending(), "zhongwo".len());
}

#[test]
fn select_consumes_and_commits_to_context() {
    let dir = fixture_dir();
    let towers =
        RouteA::open(&dir.join("fp32"), SessionShape::default()).expect("the fixture towers open");
    let table = SyllableTable::load();
    let lexicon = lexicon();
    let emittable = emittable(&dir, &lexicon);
    let mut session = session(&towers, &table, &lexicon, &emittable, false);
    session.set_context("北京大学").expect("context sets");
    for ch in "zhongwo".chars() {
        session.key(ch).expect("a keystroke decodes");
    }
    let top = session
        .candidates()
        .first()
        .expect("a top candidate")
        .clone();
    let commit = session.select(0).expect("selecting the top commits");
    assert_eq!(commit.text, top.text);
    assert_eq!(commit.consumed, top.consumes);
    assert_eq!(session.pending(), "zhongwo".len() - commit.consumed);
    session.reset();
    assert_eq!(session.pending(), 0);
    assert!(session.candidates().is_empty());
}

#[test]
fn prefix_commit_leaves_rest_pending() {
    let dir = fixture_dir();
    let towers =
        RouteA::open(&dir.join("fp32"), SessionShape::default()).expect("the fixture towers open");
    let table = SyllableTable::load();
    let lexicon = lexicon();
    let emittable = emittable(&dir, &lexicon);
    let mut session = session(&towers, &table, &lexicon, &emittable, false);
    for ch in "woaibeijing".chars() {
        session.key(ch).expect("a keystroke decodes");
    }
    // After the whole-input sentences, a prefix candidate covers the first
    // two positions of the best reading — the "woai" syllables' keystrokes.
    let index = session
        .candidates()
        .iter()
        .position(|candidate| candidate.consumes == 4)
        .expect("a two-position prefix candidate");
    let prefix = session.candidates()[index].clone();
    assert_eq!(prefix.text.chars().count(), 2);
    let commit = session.select(index).expect("selecting a prefix commits");
    assert_eq!(commit.text, prefix.text);
    assert_eq!(commit.consumed, 4);
    assert_eq!(session.pending(), "beijing".len());
    // The committed text joined the context and only the rest decodes: the
    // new top covers exactly the pending keystrokes.
    let top = session.candidates().first().expect("a new top candidate");
    assert_eq!(top.consumes, session.pending());
}

#[test]
fn prefix_candidates_at_syllable_boundaries() {
    let dir = fixture_dir();
    let towers =
        RouteA::open(&dir.join("fp32"), SessionShape::default()).expect("the fixture towers open");
    let table = SyllableTable::load();
    let lexicon = lexicon();
    let emittable = emittable(&dir, &lexicon);
    let mut session = session(&towers, &table, &lexicon, &emittable, false);
    // "wo", "ai" full syllables followed by the abbreviated "b", "j".
    for ch in "woaibj".chars() {
        session.key(ch).expect("a keystroke decodes");
    }
    let candidates = session.candidates();
    let prefixes: Vec<_> = candidates.iter().filter(|c| c.consumes < 6).collect();
    assert!(!prefixes.is_empty(), "prefix candidates exist");
    // A prefix ends at a syllable boundary of the best reading: after "wo",
    // "woai" or "woaib" — never inside a syllable.
    for prefix in &prefixes {
        assert!(
            [2, 4, 5].contains(&prefix.consumes),
            "prefix {:?} ends mid-syllable at {}",
            prefix.text,
            prefix.consumes
        );
    }
}
