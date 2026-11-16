//! The key search, pinned against `mlime.train.key_search.search` on a shared fixture.
//!
//! `tests/fixtures/key-search.json` is what `mlime gen-key-search-fixture`
//! writes: records of keys over a small emittable set, each with synthetic
//! emissions and transition rows that are `SplitMix64` draws (the fixture's
//! `definitions` say exactly which), and for every setting of the grid the
//! hypotheses the Python search returned and the sequence of its transition
//! calls. This test regenerates the inputs, runs the Rust search, and holds it
//! to the fixture: ids and cuts exactly, the score terms within 1e-9, and the
//! queried prefixes call for call.
//!
//! A record with a lexicon also pins the arc lexicon's load and the word arcs
//! it enumerates over the record's lattice, and its settings run the search
//! with those arcs over a prior and an arc width: the hypotheses' arc counts
//! and what the arcs cost are held too. A record without one runs at prior 0.

use std::collections::HashMap;
use std::convert::Infallible;
use std::num::NonZeroUsize;

use ime_decode::{
    Combine, DEFAULT_ARC_WIDTH, KeyArcLexicon, KeyArcLexiconStats, KeyHypothesis, KeyLogProbs,
    KeySearchOptions, KeySearchStats, KeyTransition, KeyWordArc, PrefixId, PrefixTrie, Weight,
    key_search,
};
use ime_pinyin::{EmitId, KeyLattice, Keys, Legality, LegalityOptions};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    beta: f64,
    alphabets: HashMap<String, String>,
    lexicon: Lexicon,
    records: Vec<Record>,
}

/// The `word<TAB>category` rows the arc records read, and the categories asked for.
#[derive(Deserialize)]
struct Lexicon {
    categories: Vec<String>,
    rows: Vec<String>,
}

#[derive(Deserialize)]
struct Record {
    name: String,
    keys: String,
    alphabet: String,
    emission: String,
    transition: String,
    emission_seed: String,
    transition_seed: String,
    /// What loading the lexicon over the record's alphabet kept and skipped.
    lexicon: Option<LoadedLexicon>,
    /// The record's word arcs, sorted by start, entry and end.
    #[serde(default)]
    words: Vec<ExpectedWord>,
    settings: Vec<Setting>,
}

#[derive(Deserialize)]
struct LoadedLexicon {
    entries: usize,
    skipped_short: usize,
    skipped_outside: usize,
}

#[derive(Deserialize)]
struct ExpectedWord {
    entry: Vec<usize>,
    start: usize,
    end: usize,
    chains: Vec<Vec<usize>>,
}

#[derive(Deserialize)]
struct Setting {
    combine: String,
    beam: Option<usize>,
    gamma: f64,
    /// The arc prior; a setting without one is the search without arcs.
    #[serde(default)]
    prior: f64,
    arc_width: Option<usize>,
    hypotheses: Vec<Expected>,
    calls: Vec<Vec<Vec<usize>>>,
    counts: Option<ArcCounts>,
}

#[derive(Deserialize)]
struct ArcCounts {
    offered: usize,
    landed: usize,
    rows: usize,
}

#[derive(Deserialize)]
struct Expected {
    ids: Vec<usize>,
    cuts: Vec<usize>,
    score: f64,
    ctc: f64,
    transition: f64,
    correction: f64,
    #[serde(default)]
    arcs: usize,
}

/// `SplitMix64`'s finaliser.
const fn splitmix(value: u64) -> u64 {
    let mut value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// The fixture's `draw(seed, x)`: a multiple of 2^-16 in (-16, 0], exact in f32.
#[expect(
    clippy::cast_possible_truncation,
    reason = "a 20-bit multiple of 2^-16 is exact in f32"
)]
fn draw(seed: u64, value: u64) -> f32 {
    -(f64::from(u32::try_from(splitmix(seed ^ value) >> 44).expect("20 bits")) / 65536.0) as f32
}

fn wide(value: usize) -> u64 {
    u64::try_from(value).expect("a usize fits u64")
}

fn seed(hex: &str) -> u64 {
    u64::from_str_radix(hex, 16).expect("a seed is 16 hex digits")
}

/// The record's `[keys, E + 1]` log-probs, the blank last.
fn emissions(keys: usize, emittable: usize, seed: u64, flat: bool) -> Vec<f32> {
    let mut data = Vec::with_capacity(keys * (emittable + 1));
    for key in 0..keys {
        for column in 0..=emittable {
            let column = if flat && column < emittable {
                0
            } else {
                column
            };
            data.push(draw(seed, (wide(key) << 32) | wide(column)));
        }
    }
    data
}

/// The fixture's transition, recording every call's prefixes.
struct Hashed {
    seed: u64,
    flat: bool,
    emittable: usize,
    calls: Vec<Vec<Vec<usize>>>,
}

impl KeyTransition for Hashed {
    type Error = Infallible;

    fn rows(
        &mut self,
        trie: &PrefixTrie,
        queries: &[PrefixId],
        out: &mut [f32],
    ) -> Result<(), Infallible> {
        let mut call = Vec::with_capacity(queries.len());
        let mut ids: Vec<EmitId> = Vec::new();
        for (&query, row) in queries.iter().zip(out.chunks_exact_mut(self.emittable)) {
            trie.ids_into(query, &mut ids);
            call.push(ids.iter().map(|id| id.index()).collect());
            let hash = ids
                .iter()
                .fold(0_u64, |hash, id| splitmix(hash ^ wide(id.index())));
            for (column, slot) in row.iter_mut().enumerate() {
                *slot = if self.flat {
                    0.0
                } else {
                    draw(self.seed, hash ^ wide(column))
                };
            }
        }
        self.calls.push(call);
        Ok(())
    }
}

fn assert_close(got: f64, want: f64, what: &str) {
    assert!((got - want).abs() <= 1e-9, "{what}: {got} against {want}");
}

#[test]
fn the_rust_search_reproduces_the_python_search() {
    let fixture: Fixture =
        serde_json::from_str(include_str!("fixtures/key-search.json")).expect("the fixture parses");
    let beta = Weight::try_from(fixture.beta).expect("beta is a weight");
    let automata: HashMap<&str, Legality> = fixture
        .alphabets
        .iter()
        .map(|(name, characters)| {
            let characters: Vec<char> = characters.chars().collect();
            (
                name.as_str(),
                Legality::over(&characters).expect("the alphabet is emittable"),
            )
        })
        .collect();
    let rows: String = fixture
        .lexicon
        .rows
        .iter()
        .flat_map(|row| [row.as_str(), "\n"])
        .collect();
    let categories: Vec<&str> = fixture
        .lexicon
        .categories
        .iter()
        .map(String::as_str)
        .collect();
    let mut runs = 0;
    let mut arc_runs = 0;
    for record in &fixture.records {
        let legality = &automata[record.alphabet.as_str()];
        let emittable = legality.emittable().len();
        let keys = Keys::new(&record.keys).expect("the keys are [a-z]");
        let lattice = legality.lattice(&keys, LegalityOptions::OFFLINE);
        let words = match &record.lexicon {
            Some(loaded) => {
                let (lexicon, stats) = KeyArcLexicon::parse(&rows, &categories, legality)
                    .expect("the lexicon has entries over the alphabet");
                assert_eq!(
                    stats,
                    KeyArcLexiconStats {
                        entries: loaded.entries,
                        skipped_short: loaded.skipped_short,
                        skipped_outside: loaded.skipped_outside,
                    },
                    "{}: the lexicon's load",
                    record.name
                );
                let words = lexicon.arcs(&lattice, &keys);
                assert_words(&words, &record.words, &record.name);
                words
            }
            None => Vec::new(),
        };
        let data = emissions(
            keys.len(),
            emittable,
            seed(&record.emission_seed),
            record.emission == "flat",
        );
        let log_probs = KeyLogProbs::new(keys.len(), emittable + 1, &data).expect("a table");
        let record_inputs = Inputs {
            lattice: &lattice,
            log_probs: &log_probs,
            words: &words,
            emittable,
        };
        for setting in &record.settings {
            arc_runs += usize::from(check_setting(record, setting, beta, &record_inputs));
            runs += 1;
        }
    }
    assert!(runs > 0);
    assert!(arc_runs > 0, "the fixture holds settings with arcs");
}

/// What one record is searched over.
struct Inputs<'a> {
    lattice: &'a KeyLattice,
    log_probs: &'a KeyLogProbs<'a>,
    words: &'a [KeyWordArc],
    emittable: usize,
}

/// Run the Rust search under *setting* and hold it to the Python search's
/// result; whether the setting is one with arcs.
fn check_setting(record: &Record, setting: &Setting, beta: Weight, inputs: &Inputs<'_>) -> bool {
    let context = format!(
        "{} {} beam {:?} gamma {} prior {} arc width {:?}",
        record.name, setting.combine, setting.beam, setting.gamma, setting.prior, setting.arc_width
    );
    let options = KeySearchOptions {
        beta,
        gamma: Weight::try_from(setting.gamma).expect("gamma is a weight"),
        beam: setting.beam.and_then(NonZeroUsize::new),
        top_k: None,
        combine: match setting.combine.as_str() {
            "sum" => Combine::Sum,
            "viterbi" => Combine::Viterbi,
            other => panic!("an unknown combination {other}"),
        },
        prior: Weight::try_from(setting.prior).expect("the prior is a weight"),
        arc_width: setting.arc_width.map_or(DEFAULT_ARC_WIDTH, |width| {
            NonZeroUsize::new(width).expect("an arc width is not zero")
        }),
    };
    let mut transition = Hashed {
        seed: seed(&record.transition_seed),
        flat: record.transition == "flat",
        emittable: inputs.emittable,
        calls: Vec::new(),
    };
    let mut stats = KeySearchStats::default();
    let found = key_search(
        inputs.lattice,
        inputs.log_probs,
        inputs.words,
        &mut transition,
        &options,
        &mut stats,
    )
    .expect("the transition never fails");
    assert_hypotheses(&found, &setting.hypotheses, &context);
    for hypothesis in &found {
        assert_eq!(hypothesis.steps.len(), hypothesis.ids.len(), "{context}");
        assert_eq!(hypothesis.cuts.len(), hypothesis.ids.len(), "{context}");
        #[expect(
            clippy::cast_precision_loss,
            reason = "an output holds a handful of arcs"
        )]
        let whole = hypothesis.ctc - options.gamma.get() * hypothesis.correction
            + beta.get() * hypothesis.transition
            + options.prior.get() * hypothesis.arcs as f64;
        let last = *hypothesis.steps.last().expect("an output has characters");
        assert_close(last, whole, &format!("{context}: the last running score"));
    }
    assert_eq!(
        transition.calls, setting.calls,
        "{context}: the queried prefixes"
    );
    assert_eq!(stats.calls, setting.calls.len(), "{context}");
    assert_eq!(
        stats.rows,
        setting.calls.iter().map(Vec::len).sum::<usize>(),
        "{context}"
    );
    let cost = (stats.arcs_offered, stats.arcs_landed, stats.landing_rows);
    if let Some(counts) = &setting.counts {
        assert_eq!(
            cost,
            (counts.offered, counts.landed, counts.rows),
            "{context}: what the arcs cost"
        );
        true
    } else {
        assert_eq!(cost, (0, 0, 0), "{context}: a search without arcs");
        false
    }
}

fn assert_words(found: &[KeyWordArc], expected: &[ExpectedWord], name: &str) {
    assert_eq!(found.len(), expected.len(), "{name}: how many word arcs");
    for (got, want) in found.iter().zip(expected) {
        let entry: Vec<usize> = got.entry().iter().map(|id| id.index()).collect();
        let what = format!("{name}: arc {entry:?} {}..{}", want.start, want.end);
        assert_eq!(entry, want.entry, "{what}: entry");
        assert_eq!((got.start(), got.end()), (want.start, want.end), "{what}");
        let chains: Vec<Vec<usize>> = got.chains().map(<[usize]>::to_vec).collect();
        assert_eq!(chains, want.chains, "{what}: chains");
    }
}

fn assert_hypotheses(found: &[KeyHypothesis], expected: &[Expected], context: &str) {
    assert_eq!(
        found.len(),
        expected.len(),
        "{context}: how many hypotheses"
    );
    for (rank, (got, want)) in found.iter().zip(expected).enumerate() {
        let what = format!("{context} hypothesis {rank}");
        let ids: Vec<usize> = got.ids.iter().map(|id| id.index()).collect();
        assert_eq!(ids, want.ids, "{what}: ids");
        assert_eq!(got.cuts, want.cuts, "{what}: cuts");
        assert_eq!(got.arcs, want.arcs, "{what}: arcs");
        assert_close(got.score, want.score, &format!("{what}: score"));
        assert_close(got.ctc, want.ctc, &format!("{what}: ctc"));
        assert_close(
            got.transition,
            want.transition,
            &format!("{what}: transition"),
        );
        assert_close(
            got.correction,
            want.correction,
            &format!("{what}: correction"),
        );
    }
}
