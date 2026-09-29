//! The union `Candidates::build_corrections` computes, pinned against the
//! training side's `CandidateSpace.resolve` on the same fixture spans.
//!
//! `tests/fixtures/typo-candidates.json` is what `write_candidates_fixture`
//! writes: per span of the corrections fixture, the pair count of its sorted
//! `(emission id, correction prior)` list, a SHA-256 over the canonical
//! serialisation of those pairs (`id:prior` ascending, the prior `{:.9e}`,
//! `inf` for the unreachable), and the first pairs verbatim so a digest
//! mismatch still points at a real difference. The fixture's space is built
//! over the whole lexicon, so its emission index is the character table's own
//! order -- exactly what [`CharId`](ime_pinyin::CharId) numbers -- and the
//! two unions compare row for row, prior for prior.

use std::collections::BTreeMap;

use ime_decode::Candidates;
use ime_pinyin::{CorrectionTable, Lexicon, SegmentLattice, SegmentOptions, SyllableTable};
use sha2::{Digest, Sha256};

/// One fixture row: a span's pair count, the digest of its canonical
/// serialisation, and the first pairs verbatim for a diffable failure.
/// `head`'s `null` is a character the noise model cannot reach within two
/// edits -- the decoder prices it `INFINITY`, dead in the beam but still in
/// the union.
#[derive(serde::Deserialize)]
struct Row {
    n: usize,
    sha256: String,
    head: Vec<(usize, Option<f64>)>,
}

/// The whole fixture: every fixture span's row.
type Fixture = BTreeMap<String, Row>;

fn fixture() -> Fixture {
    let fixture: Fixture = serde_json::from_str(include_str!("fixtures/typo-candidates.json"))
        .expect("the fixture parses");
    assert!(!fixture.is_empty(), "the fixture lists no spans");
    fixture
}

/// A prior's digest field, byte-identical to the fixture writer's
/// `_canonical_prior`: `inf` for the unreachable price, otherwise Rust's
/// `{:.9e}`, whose bare exponent (no `+`, no padding) is exactly what the
/// Python side normalises its own `{:.9e}` to.
fn canonical_prior(cost: f64) -> String {
    if cost.is_infinite() {
        "inf".to_owned()
    } else {
        format!("{cost:.9e}")
    }
}

/// The canonical serialisation both sides hash: `id:prior` joined by `,`.
fn canonical_pairs(pairs: &[(usize, f64)]) -> String {
    pairs
        .iter()
        .map(|(id, cost)| format!("{id}:{}", canonical_prior(*cost)))
        .collect::<Vec<_>>()
        .join(",")
}

/// *got*'s digest against the fixture row; on a mismatch, panic with the
/// span, both pair counts and the first differing verbatim head pair.
fn assert_priors_agree(got: &[(usize, f64)], want: &Row, span: &str) {
    let digest = hex::encode(Sha256::digest(canonical_pairs(got)));
    if got.len() == want.n && digest == want.sha256 {
        return;
    }
    let first_diff = want
        .head
        .iter()
        .enumerate()
        .find_map(|(index, &(want_id, want_cost))| {
            let &(got_id, got_cost) = got.get(index)?;
            (got_id != want_id
                || got_cost.is_infinite() != want_cost.is_none()
                || (got_cost.is_finite()
                    && want_cost.is_some_and(|want_cost| {
                        canonical_prior(got_cost) != canonical_prior(want_cost)
                    })))
            .then_some((index, (got_id, got_cost), (want_id, want_cost)))
        });
    panic!(
        "{span:?}: the fixture row disagrees -- {} pairs vs {}, sha256 {digest} vs {}{}",
        got.len(),
        want.n,
        want.sha256,
        first_diff.map_or_else(String::new, |(index, got_pair, want_pair)| format!(
            "; first differing verbatim pair at {index}: {got_pair:?} vs {want_pair:?}"
        ))
    );
}

/// A standalone span's own readings: its exact syllable when it is one, its
/// prefix completions when it is only a prefix, nothing when the noise model
/// alone reaches it. The fixture pins this rule; a segment inside a real
/// input carries its own range instead (a tail span keeps its completions).
fn standalone_own(span: &str, table: &SyllableTable) -> Option<ime_pinyin::SyllableRange> {
    table.exact_range(span).or_else(|| {
        let range = table.prefix_range(span);
        (!range.is_empty()).then_some(range)
    })
}

#[test]
fn the_unions_agree_with_the_python_fixture() {
    let table = SyllableTable::load();
    let lexicon = Lexicon::load(&table).expect("the lexicon table parses");
    let corrections = CorrectionTable::load(&table).expect("the noise table parses");
    for (span, want) in fixture() {
        let union: Vec<(usize, f64)> = lexicon
            .corrections_priors(&span, standalone_own(&span, &table), &table, &corrections)
            .iter()
            .map(|&(id, cost)| (id.index(), cost))
            .collect();
        assert_priors_agree(&union, &want, &span);
    }
}

#[test]
fn build_corrections_carries_the_fixture_union() {
    let table = SyllableTable::load();
    let lexicon = Lexicon::load(&table).expect("the lexicon table parses");
    let corrections = CorrectionTable::load(&table).expect("the noise table parses");
    let options = SegmentOptions::default();
    // A span only reaches `build_corrections` through a segmentation, so the
    // end-to-end check applies to the fixture spans the lattice keeps whole:
    // for each, the first position's candidates are the span's union.
    let mut whole = 0;
    for (span, want) in fixture() {
        let Ok(lattice) = SegmentLattice::build_corrections(&span, &table, &options, &corrections)
        else {
            continue;
        };
        let readings = lattice.k_best(&options);
        if readings.is_empty() || readings[0].segments().len() != 1 {
            continue;
        }
        let segment = &readings[0].segments()[0];
        if segment.start() != 0 || segment.end() != span.len() {
            continue;
        }
        // The end-to-end comparison only holds where the lattice's own
        // readings are the standalone rule's: a span that surfaced through
        // corrections has none to begin with, and prices every candidate.
        if segment.syllables() != standalone_own(&span, &table) {
            continue;
        }
        whole += 1;
        let batch =
            Candidates::build_corrections(&readings[..1], &span, &table, &corrections, &lexicon)
                .expect("a one-segment reading resolves");
        let path = &batch.paths()[0];
        let union: Vec<(usize, f64)> = path.positions()[0]
            .iter()
            .enumerate()
            .map(|(slot, id)| (id.index(), path.prior(0, slot)))
            .collect();
        assert_priors_agree(&union, &want, &span);
    }
    assert!(
        whole > 100,
        "only {whole} fixture spans exercised end-to-end"
    );
}
