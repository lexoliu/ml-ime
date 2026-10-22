use serde::Deserialize;

use super::*;

/// The automaton over the whole character table.
fn legality() -> Legality {
    Legality::over_lexicon().expect("the embedded tables agree")
}

fn keys(typed: &str) -> Keys {
    Keys::new(typed).expect("test keys are [a-z]")
}

fn cost(legality: &Legality, typed: &str, sentence: &str, options: LegalityOptions) -> Option<f64> {
    legality.sentence_cost(&keys(typed), sentence, options)
}

/// Offline, with corrections kept off spans that are syllables.
const STRICT: LegalityOptions = LegalityOptions {
    unfinished_tail: false,
    correct_syllables: false,
};

/// `corrections(span)`'s cost for *syllable*, which must be listed.
fn correction_cost(legality: &Legality, span: &str, syllable: &str) -> f64 {
    legality
        .correction_table()
        .corrections(span)
        .iter()
        .find(|entry| entry.syllable() == syllable)
        .unwrap_or_else(|| panic!("corrections({span:?}) lists {syllable:?}"))
        .cost()
}

#[test]
fn wmzyq_is_never_women_chi_shuijiao() {
    // 吃 needs c/ch where z was typed, 水 needs s/sh where y was, 饺 needs j
    // where q was; every key is a one-letter initial, and initials are never
    // corrected, live or offline.
    let legality = legality();
    for options in [LegalityOptions::OFFLINE, LegalityOptions::LIVE] {
        assert_eq!(cost(&legality, "wmzyq", "我们吃水饺", options), None);
    }
    assert_eq!(
        cost(&legality, "wmcsj", "我们吃水饺", LegalityOptions::OFFLINE),
        Some(0.0)
    );
}

#[test]
fn a_typo_and_a_mixed_reading_are_both_legal() {
    // shui|jiap reads jiap as jiao at the correction's cost; shui|jia|p
    // reads p as an initial at none.
    let legality = legality();
    let jiao = correction_cost(&legality, "jiap", "jiao");
    assert!(jiao > 0.0);
    assert_eq!(
        cost(&legality, "shuijiap", "水饺", LegalityOptions::OFFLINE),
        Some(jiao)
    );
    assert_eq!(
        cost(&legality, "shuijiap", "水家铺", LegalityOptions::OFFLINE),
        Some(0.0)
    );
}

#[test]
fn the_section_three_table_holds() {
    let legality = legality();
    for (typed, sentence) in [
        ("sj", "水饺"),
        ("sk", "烧烤"),
        ("sl", "沙拉"),
        ("sn", "酸奶"),
        ("zgrm", "中国人民"),
        ("xian", "西安"),
        ("xian", "先"),
        ("hm", "噷"),
        ("hm", "好们"),
    ] {
        assert_eq!(
            cost(&legality, typed, sentence, LegalityOptions::OFFLINE),
            Some(0.0),
            "{typed} -> {sentence}"
        );
    }
}

#[test]
fn a_full_syllable_reads_at_no_cost() {
    // `ni` and `hao` are no initials, the tail is off and syllables are not
    // corrected: only rule 1 reads them.
    let legality = legality();
    assert_eq!(cost(&legality, "nihao", "你好", STRICT), Some(0.0));
    assert_eq!(cost(&legality, "nihao", "你们", STRICT), None);
}

#[test]
fn an_initial_covers_every_syllable_it_begins() {
    let legality = legality();
    assert_eq!(cost(&legality, "zg", "中国", STRICT), Some(0.0));
    assert_eq!(cost(&legality, "zhg", "中国", STRICT), Some(0.0));
    assert_eq!(cost(&legality, "chf", "吃饭", STRICT), Some(0.0));
    // Three keys standing for one character are no initial: only a
    // correction reads them, at a price.
    assert!(cost(&legality, "zhg", "中", STRICT).is_some_and(|cost| cost > 0.0));
}

#[test]
fn an_initial_span_is_never_corrected() {
    // `z` and `zh` both list `a` among their corrections, yet neither reads 啊.
    let legality = legality();
    for span in ["z", "zh"] {
        correction_cost(&legality, span, "a");
        for options in [LegalityOptions::OFFLINE, LegalityOptions::LIVE] {
            assert_eq!(cost(&legality, span, "啊", options), None, "{span}");
        }
    }
}

#[test]
fn the_unfinished_tail_reads_only_last_and_only_live() {
    let legality = legality();
    assert_eq!(
        cost(&legality, "guozho", "国中", LegalityOptions::LIVE),
        Some(0.0)
    );
    assert_ne!(
        cost(&legality, "guozho", "国中", LegalityOptions::OFFLINE),
        Some(0.0)
    );
    // The same unfinished span in the middle pays its correction, if any.
    assert_ne!(
        cost(&legality, "zhoguo", "中国", LegalityOptions::LIVE),
        Some(0.0)
    );
}

#[test]
fn a_typo_off_the_inventory_reads_through_its_correction() {
    let legality = legality();
    let jia = correction_cost(&legality, "jai", "jia");
    assert_eq!(cost(&legality, "wojaijia", "我家家", STRICT), Some(jia));
}

#[test]
fn corrections_on_syllables_follow_the_option() {
    // `zao` is a syllable; read as zhao it is a fuzzy swap away.
    let legality = legality();
    let zhao = correction_cost(&legality, "zao", "zhao");
    assert_eq!(
        cost(&legality, "zao", "找", LegalityOptions::OFFLINE),
        Some(zhao)
    );
    assert_eq!(cost(&legality, "zao", "找", STRICT), None);
    assert_eq!(cost(&legality, "zao", "早", STRICT), Some(0.0));
}

#[test]
fn every_key_is_consumed_and_none_twice() {
    let legality = legality();
    assert_eq!(
        cost(&legality, "aa", "啊啊", LegalityOptions::OFFLINE),
        Some(0.0)
    );
    assert_eq!(cost(&legality, "a", "啊啊", LegalityOptions::OFFLINE), None);
    assert_eq!(cost(&legality, "ab", "啊", LegalityOptions::OFFLINE), None);
}

#[test]
fn a_polyphone_reads_under_any_of_its_readings() {
    let legality = legality();
    let zhong = legality.syllable("zhong").expect("zhong is a syllable");
    let chong = legality.syllable("chong").expect("chong is a syllable");
    let heavy = legality.emit_id('重').expect("重 is emittable");
    assert!(legality.homophones[zhong.index()].contains(&heavy));
    assert!(legality.homophones[chong.index()].contains(&heavy));
    for typed in ["zhong", "chong"] {
        assert_eq!(cost(&legality, typed, "重", STRICT), Some(0.0));
    }
}

#[test]
fn a_character_outside_the_emittable_set_is_illegal() {
    let legality = Legality::over(&['国', '中']).expect("both characters are read");
    assert_eq!(legality.emit_id('国'), Some(EmitId(0)));
    assert_eq!(cost(&legality, "zg", "中国", STRICT), Some(0.0));
    assert_eq!(cost(&legality, "zgr", "中国人", STRICT), None);
    assert!(matches!(
        Legality::over(&['中', '中']),
        Err(LegalityError::Duplicate('中'))
    ));
    assert!(matches!(
        Legality::over(&['a']),
        Err(LegalityError::Unread('a'))
    ));
}

#[test]
fn keys_are_a_non_empty_alphabet_string() {
    assert_eq!(Keys::new(""), Err(KeysError::Empty));
    assert_eq!(
        Keys::new("ni3"),
        Err(KeysError::OutsideAlphabet { offset: 2, ch: '3' })
    );
    assert_eq!(keys("nihao").len(), 5);
}

#[test]
fn the_mask_is_the_edge_table() {
    // Every step reads exactly the edge the open span names, and may blank
    // exactly when a wider kept edge leaves the same offset.
    let legality = legality();
    for typed in ["shuijiap", "zgrm", "xian", "wojaijia", "hm"] {
        for options in [LegalityOptions::OFFLINE, LegalityOptions::LIVE] {
            let lattice = legality.lattice(&keys(typed), options);
            for pending in 0..typed.len() {
                for key in pending..typed.len().min(pending + MAX_CORRECTED_LEN) {
                    let step = lattice.step(pending, key);
                    let edge = lattice.edge(pending, key + 1);
                    assert_eq!(step.ids(), edge.map_or(&[][..], |edge| edge.ids()));
                    let wider = (key + 2..=pending + MAX_CORRECTED_LEN)
                        .any(|end| lattice.edge(pending, end).is_some());
                    assert_eq!(step.blank(), wider, "{typed} {pending}..={key}");
                }
                assert!(!lattice.step(pending, typed.len() - 1).blank());
            }
        }
    }
}

#[test]
fn the_bound_covers_every_span() {
    let legality = legality();
    let fixture = fixture();
    for record in &fixture.records {
        for options in [LegalityOptions::OFFLINE, LegalityOptions::LIVE] {
            let lattice = legality.lattice(&keys(&record.keys), options);
            for edge in lattice.edges() {
                assert!(edge.ids().len() <= legality.admitted_bound());
            }
        }
    }
}

/// The cheapest way through *sentence* along the mask alone: a beam with no
/// width limit that only ever asks [`KeyLattice::step`].
fn walk(lattice: &KeyLattice, sentence: &[EmitId]) -> Option<f64> {
    // `(pending, placed) -> cost` of every live hypothesis.
    let mut live: HashMap<(usize, usize), f64> = HashMap::from([((0, 0), 0.0)]);
    for key in 0..lattice.keys() {
        let mut next: HashMap<(usize, usize), f64> = HashMap::new();
        let mut keep = |state: (usize, usize), cost: f64| {
            let slot = next.entry(state).or_insert(f64::INFINITY);
            *slot = slot.min(cost);
        };
        for (&(pending, placed), &cost) in &live {
            let step = lattice.step(pending, key);
            if step.blank() {
                keep((pending, placed), cost);
            }
            if let Some(&emit) = sentence.get(placed)
                && let Ok(index) = step.ids().binary_search(&emit)
            {
                keep((key + 1, placed + 1), cost + step.costs()[index]);
            }
        }
        live = next;
    }
    live.get(&(lattice.keys(), sentence.len())).copied()
}

#[test]
fn walking_the_mask_prices_a_sentence_as_the_lattice_does() {
    let legality = legality();
    let fixture = fixture();
    let mut cases: Vec<(String, String)> = fixture
        .records
        .iter()
        .map(|record| (record.keys.clone(), record.sentence.clone()))
        .collect();
    cases.extend(
        [
            ("shuijiap", "水饺"),
            ("wojaijia", "我家家"),
            ("zao", "找"),
            ("wmzyq", "我们吃水饺"),
        ]
        .map(|(typed, sentence)| (typed.to_owned(), sentence.to_owned())),
    );
    for (typed, sentence) in &cases {
        let ids: Vec<EmitId> = sentence
            .chars()
            .map(|ch| legality.emit_id(ch).expect("the sample is emittable"))
            .collect();
        let lattice = legality.lattice(&keys(typed), LegalityOptions::OFFLINE);
        let (walked, priced) = (walk(&lattice, &ids), lattice.cost(&ids));
        assert_eq!(walked.is_some(), priced.is_some(), "{typed} -> {sentence}");
        if let (Some(walked), Some(priced)) = (walked, priced) {
            assert!((walked - priced).abs() <= 1e-12, "{typed} -> {sentence}");
        }
    }
}

/// One `cases` row of the shared fixture.
#[derive(Deserialize)]
struct Case {
    keys: String,
    sentence: String,
    unfinished_tail: bool,
    correct_syllables: bool,
    cost: Option<f64>,
}

/// One `lattices` row: a whole lattice, summarised.
#[derive(Deserialize)]
struct LatticeRow {
    keys: String,
    unfinished_tail: bool,
    correct_syllables: bool,
    arcs: usize,
    cost_sum: f64,
    digest: String,
}

/// One sampled training record, typed in one style.
#[derive(Deserialize)]
struct Record {
    keys: String,
    sentence: String,
}

#[derive(Deserialize)]
struct Fixture {
    records: Vec<Record>,
    cases: Vec<Case>,
    lattices: Vec<LatticeRow>,
}

fn fixture() -> Fixture {
    serde_json::from_str(include_str!("../../tests/fixtures/legality.json"))
        .expect("the fixture parses")
}

/// `SplitMix64`'s finaliser: `mlime.legality._splitmix`.
const fn splitmix(value: u64) -> u64 {
    let mut value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// `mlime.legality.lattice_digest`: the wrapping sum of every arc's hash.
#[expect(
    clippy::cast_possible_truncation,
    reason = "a micro-unit cost is far inside i64"
)]
fn digest(lattice: &KeyLattice) -> String {
    let mut total = 0u64;
    for edge in lattice.edges() {
        for (emit, cost) in edge.ids().iter().zip(edge.costs()) {
            let packed =
                ((edge.start() as u64) << 40) | ((edge.end() as u64) << 24) | u64::from(emit.0);
            let micro = (cost * 1e6 + 0.5).floor() as i64;
            total = total.wrapping_add(splitmix(packed) ^ splitmix(micro.cast_unsigned()));
        }
    }
    format!("{total:016x}")
}

fn close(got: f64, want: f64) -> bool {
    (got - want).abs() <= 1e-9 * want.abs().max(1.0)
}

#[test]
fn the_automaton_agrees_with_the_python_fixture() {
    let legality = legality();
    let fixture = fixture();
    assert!(!fixture.cases.is_empty() && !fixture.lattices.is_empty());
    for case in &fixture.cases {
        let options = LegalityOptions {
            unfinished_tail: case.unfinished_tail,
            correct_syllables: case.correct_syllables,
        };
        let got = cost(&legality, &case.keys, &case.sentence, options);
        let agrees = match (got, case.cost) {
            (None, None) => true,
            (Some(got), Some(want)) => close(got, want),
            _ => false,
        };
        assert!(
            agrees,
            "{} -> {} under {options:?}: got {got:?}, want {:?}",
            case.keys, case.sentence, case.cost
        );
    }
    for row in &fixture.lattices {
        let options = LegalityOptions {
            unfinished_tail: row.unfinished_tail,
            correct_syllables: row.correct_syllables,
        };
        let lattice = legality.lattice(&keys(&row.keys), options);
        assert_eq!(
            lattice.arc_count(),
            row.arcs,
            "{} under {options:?}",
            row.keys
        );
        let sum: f64 = lattice.edges().flat_map(|edge| edge.costs().iter()).sum();
        assert!(
            close(sum, row.cost_sum),
            "{} under {options:?}: cost sum {sum}, want {}",
            row.keys,
            row.cost_sum
        );
        assert_eq!(
            digest(&lattice),
            row.digest,
            "{} under {options:?}",
            row.keys
        );
    }
}
