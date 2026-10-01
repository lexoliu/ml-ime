//! The state cache answers what the model would: `decode_many` over the
//! fixture LM through `Cached` returns the uncached decode's hypotheses bit
//! for bit, and re-decoding the same context is a hit. The fixture is
//! `ime-lm`'s committed transformer export and its `char_pinyin.tsv`
//! lexicon — a real graph with an eight-character alphabet — so the check
//! runs the real `advance`/`compact` path, not a mock.

use ime_decode::{BeamOptions, Candidates, Record, Uniform, decode_many};
use ime_lm::{CharLm, SessionShape};
use ime_pinyin::{Lexicon, SegmentLattice, SegmentOptions, SyllableTable};
use ime_session::Cached;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

/// `ime-lm`'s fixture directory.
fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../ime-lm/tests/fixtures")
        .canonicalize()
        .expect("the ime-lm fixtures are committed")
}

/// The fixture lexicon, parsed from its committed `char_pinyin.tsv`.
fn lexicon() -> Lexicon {
    let table = SyllableTable::load();
    let source = std::fs::read_to_string(fixture_dir().join("char_pinyin.tsv"))
        .expect("char_pinyin.tsv is committed with the fixtures");
    Lexicon::parse(&source, &table).expect("the fixture table parses")
}

/// The fixture model with a 64-row state cache.
fn model() -> CharLm {
    CharLm::open(
        &fixture_dir().join("transformer"),
        &lexicon(),
        SessionShape {
            width: NonZeroUsize::new(8).expect("eight is not zero"),
            cache_rows: 64,
            ..SessionShape::default()
        },
    )
    .expect("the fixture opens")
}

/// The candidates over *input*'s readings under the fixture lexicon. Exact
/// syllables only, and readings a position of which the eight-character
/// fixture cannot emit are pruned — what `Candidates::build` would reject
/// the full lexicon never produces.
fn candidates(input: &str, lexicon: &Lexicon, table: &SyllableTable) -> Candidates {
    let options = SegmentOptions {
        allow_abbreviation: false,
        allow_incomplete_tail: false,
        ..SegmentOptions::default()
    };
    let lattice = SegmentLattice::build(input, table, &options).expect("the input segments");
    let covered: Vec<_> = lattice
        .k_best(&options)
        .into_iter()
        .filter(|reading| {
            reading.segments().iter().all(|piece| {
                let mut mask = Vec::new();
                lexicon.mask_into(
                    piece.syllables().expect("a syllable-range segment"),
                    &mut mask,
                );
                !mask.is_empty()
            })
        })
        .collect();
    assert!(!covered.is_empty(), "{input} keeps a covered reading");
    Candidates::build(&covered, lexicon).expect("the covered readings admit candidates")
}

/// `(text, score)` pairs of a decode's top hypotheses — the identity a
/// cached decode has to reproduce bit for bit.
fn texts(hypotheses: &[ime_decode::Hypothesis], lexicon: &Lexicon) -> Vec<(String, u32)> {
    hypotheses
        .iter()
        .map(|h| (h.text(lexicon), h.score().to_bits()))
        .collect()
}

/// `nihao` decodes under the fixture model's eight characters.
const INPUT: &str = "nihao";

/// A compacted state's row moved into the pool still answers its scores
/// and advances exactly as it did in its step's generation. The fixture
/// model is row-mode — paged pooling is covered in `ime-lm`'s suite.
#[test]
fn a_compacted_row_state_answers_and_advances() {
    use ime_decode::{Asked, Transition};
    let lexicon = lexicon();
    let model = model();
    let chars: Vec<_> = lexicon
        .characters()
        .iter()
        .map(|&c| lexicon.id_of(c).expect("a lexicon character"))
        .collect();
    let asked = Asked {
        candidates: &chars,
        eos: true,
    };
    let start = Transition::start(&model, Some("我很"), &asked);
    let compacted = Transition::compact(&model, &start);
    let hao = lexicon.id_of('好').expect("好 is in the fixture");
    // The pooled copy scores identically.
    assert_eq!(
        Transition::score(&model, &compacted, hao).to_bits(),
        Transition::score(&model, &start, hao).to_bits(),
        "the pooled row answers what the produced row did"
    );
    let next = Transition::advance(&model, &[(&compacted, hao, asked)]);
    assert_eq!(next.len(), 1, "a pooled state advances");
    // And an advance from a pooled state reads its row, not garbage.
    let direct = Transition::advance(&model, &[(&start, hao, asked)]);
    assert_eq!(
        Transition::score(&model, &next[0], hao).to_bits(),
        Transition::score(&model, &direct[0], hao).to_bits(),
        "the step from the pooled row equals the step from the step's row"
    );
}

/// Capacity zero is the baseline: every request reaches the model, and
/// `decode_many` through it equals the unwrapped decode.
#[test]
fn capacity_zero_is_passthrough() {
    let lexicon = lexicon();
    let table = SyllableTable::load();
    let candidates = candidates(INPUT, &lexicon, &table);
    let model = model();
    let cached = Cached::new(&model, 0);
    let out = decode_many(
        &[Record {
            candidates: &candidates,
            emission: Uniform,
            context: Some("我很"),
        }],
        &cached,
        &BeamOptions::default(),
    )
    .expect("the capacity-0 cached decode runs");
    let stats = cached.stats();
    assert_eq!(stats.start_hits + stats.step_hits, 0, "nothing was cached");
    assert!(!out[0].is_empty(), "the decode still answers");
}

#[test]
fn cached_decode_equals_uncached_bit_for_bit() {
    let lexicon = lexicon();
    let table = SyllableTable::load();
    let candidates = candidates(INPUT, &lexicon, &table);
    let model = model();
    let context = "我很";

    let plain = decode_many(
        &[Record {
            candidates: &candidates,
            emission: Uniform,
            context: Some(context),
        }],
        &model,
        &BeamOptions::default(),
    )
    .expect("the uncached decode runs");

    let cached = Cached::new(&model, 64);
    let through = decode_many(
        &[Record {
            candidates: &candidates,
            emission: Uniform,
            context: Some(context),
        }],
        &cached,
        &BeamOptions::default(),
    )
    .expect("the cached decode runs");

    assert_eq!(
        texts(&through[0], &lexicon),
        texts(&plain[0], &lexicon),
        "the cached decode's hypotheses equal the uncached's"
    );

    // The identical decode again: every step the first one stored is a hit.
    let before = cached.stats();
    decode_many(
        &[Record {
            candidates: &candidates,
            emission: Uniform,
            context: Some(context),
        }],
        &cached,
        &BeamOptions::default(),
    )
    .expect("the second decode runs");
    let after = cached.stats();
    assert_eq!(
        after.start_misses, before.start_misses,
        "the second decode runs no prefill"
    );
    assert!(
        after.step_hits > before.step_hits,
        "the second decode hits the first's stored states"
    );
    let misses_added = after.step_misses - before.step_misses;
    assert_eq!(
        misses_added, 0,
        "a repeated decode computes nothing ({misses_added} misses)"
    );
}

#[test]
fn a_context_change_invalidates() {
    let lexicon = lexicon();
    let table = SyllableTable::load();
    let candidates = candidates(INPUT, &lexicon, &table);
    let model = model();
    let cached = Cached::new(&model, 64);

    for context in ["我很", "你好"] {
        decode_many(
            &[Record {
                candidates: &candidates,
                emission: Uniform,
                context: Some(context),
            }],
            &cached,
            &BeamOptions::default(),
        )
        .expect("the decode runs");
    }
    let between = cached.stats();
    // The second context's decode ran misses through the emptied cache.
    assert!(between.step_misses > 0, "context change refilled it");

    // Back to the first context: its entries are gone — the switch clears.
    decode_many(
        &[Record {
            candidates: &candidates,
            emission: Uniform,
            context: Some("我很"),
        }],
        &cached,
        &BeamOptions::default(),
    )
    .expect("the decode runs");
    let after = cached.stats();
    assert!(
        after.step_misses > between.step_misses,
        "returning to a dropped context misses again"
    );
    assert_eq!(
        after.start_hits, between.start_hits,
        "a dropped context's start is not replayed"
    );
}

#[test]
fn a_tiny_cache_stays_correct() {
    let lexicon = lexicon();
    let table = SyllableTable::load();
    let candidates = candidates(INPUT, &lexicon, &table);
    let model = model();

    let plain = decode_many(
        &[Record {
            candidates: &candidates,
            emission: Uniform,
            context: Some("我很"),
        }],
        &model,
        &BeamOptions::default(),
    )
    .expect("the uncached decode runs");

    // Two rows: under what the decode produces, so eviction runs.
    let cached = Cached::new(&model, 2);
    let through = decode_many(
        &[Record {
            candidates: &candidates,
            emission: Uniform,
            context: Some("我很"),
        }],
        &cached,
        &BeamOptions::default(),
    )
    .expect("the cached decode runs");
    assert_eq!(
        texts(&through[0], &lexicon),
        texts(&plain[0], &lexicon),
        "eviction does not change the answer"
    );
    assert!(
        cached.stats().evictions > 0,
        "the bound evicted under this decode"
    );
    assert!(cached.stats().entries <= 2, "the bound held");
}

#[test]
fn start_state_is_shared_across_paths() {
    let lexicon = lexicon();
    let table = SyllableTable::load();
    let one = candidates(INPUT, &lexicon, &table);
    let two = candidates("nima", &lexicon, &table);
    let model = model();
    let cached = Cached::new(&model, 64);

    decode_many(
        &[
            Record {
                candidates: &one,
                emission: Uniform,
                context: Some("我很"),
            },
            Record {
                candidates: &two,
                emission: Uniform,
                context: Some("我很"),
            },
        ],
        &cached,
        &BeamOptions::default(),
    )
    .expect("the decode runs");
    let stats = cached.stats();
    // Two records on one context run one prefill: the second start is the
    // first's stored state.
    assert_eq!(stats.start_misses, 1, "one prefill for the context");
    assert_eq!(stats.start_hits, 1, "the second path's start is a hit");
}
