use std::collections::HashMap;
use std::num::NonZeroUsize;

use ime_pinyin::{EmitId, KeyLattice, Keys, Legality, LegalityOptions};

use super::*;
use crate::mock::{self, Column, Favoured, HashedBigram};

/// A shape that holds every hypothesis of the small inputs here: wide
/// enough that the beam prunes nothing and so is exact.
fn exact() -> Shape {
    let all = NonZeroUsize::new(1 << 20).expect("not zero");
    Shape::new(all, all).expect("top_k = width")
}

fn shape(width: usize) -> Shape {
    let width = NonZeroUsize::new(width).expect("not zero");
    Shape::new(width, width).expect("top_k = width")
}

fn ids(legality: &Legality, text: &str) -> Vec<EmitId> {
    text.chars()
        .map(|ch| legality.emit_id(ch).expect("emittable"))
        .collect()
}

fn text(legality: &Legality, sentence: &[EmitId]) -> String {
    sentence
        .iter()
        .map(|id| legality.emittable()[id.index()])
        .collect()
}

/// `log(exp(a) + exp(b))` in f64, for the reference.
fn log_add64(a: f64, b: f64) -> f64 {
    let (high, low) = if a >= b { (a, b) } else { (b, a) };
    if low == f64::NEG_INFINITY {
        return high;
    }
    high + (low - high).exp().ln_1p()
}

/// Every legal sentence and its score, by enumerating every cover of the
/// lattice's edges and every character on them -- the definition the beam
/// must reproduce, computed without the beam's steps, merges or pruning.
fn enumerate<E: KeyEmission, T: KeyTransition>(
    lattice: &KeyLattice,
    emission: &E,
    transition: &T,
    weights: Weights,
) -> HashMap<Vec<EmitId>, f64> {
    struct Walk<'a, E> {
        lattice: &'a KeyLattice,
        emission: &'a E,
        weights: Weights,
        aligned: HashMap<Vec<EmitId>, f64>,
    }
    impl<E: KeyEmission> Walk<'_, E> {
        fn walk(&mut self, at: usize, chars: &mut Vec<EmitId>, align: f64) {
            let keys = self.lattice.keys();
            if at == keys {
                let entry = self
                    .aligned
                    .entry(chars.clone())
                    .or_insert(f64::NEG_INFINITY);
                *entry = log_add64(*entry, align);
                return;
            }
            for end in at + 1..=keys {
                let Some(edge) = self.lattice.edge(at, end) else {
                    continue;
                };
                let blanks: f64 = (at..end - 1)
                    .map(|key| f64::from(self.emission.blank(key)))
                    .sum();
                let mut logp = vec![0.0f32; edge.ids().len()];
                self.emission.gather(end - 1, edge.ids(), &mut logp);
                for ((&id, &cost), &logp) in edge.ids().iter().zip(edge.costs()).zip(&logp) {
                    chars.push(id);
                    let term = blanks + f64::from(logp) - f64::from(self.weights.correction) * cost;
                    self.walk(end, chars, align + term);
                    chars.pop();
                }
            }
        }
    }
    let mut walk = Walk {
        lattice,
        emission,
        weights,
        aligned: HashMap::new(),
    };
    walk.walk(0, &mut Vec::new(), 0.0);
    walk.aligned
        .into_iter()
        .map(|(sentence, align)| {
            let mut state = transition.start();
            let mut trans = 0.0f64;
            let mut out = [0.0f32];
            for &id in &sentence {
                transition.score(&state, &[id], &mut out);
                trans += f64::from(out[0]);
                state = transition
                    .advance(&[(&state, id)])
                    .pop()
                    .expect("one state a step");
            }
            trans += f64::from(transition.finish(&state));
            let score = align + f64::from(weights.transition) * trans;
            (sentence, score)
        })
        .collect()
}

/// The small emittable set the exhaustive cases search over.
const SMALL: &str = "我们吃水饺家铺西安先中国人民烧烤沙拉酸奶噷哼嗯呣他她是事睡觉加交叫";

const WEIGHTS: Weights = Weights {
    transition: 0.7,
    correction: 1.3,
    word: 0.0,
};

#[test]
fn a_wide_beam_is_the_exhaustive_enumeration() {
    let legality = Legality::over(&SMALL.chars().collect::<Vec<_>>()).expect("the set reads");
    let emittable = legality.emittable().len();
    let mut cases = 0;
    for (seed, input) in [
        "shuijiao", "shuijiap", "sjsk", "xian", "zgrm", "hm", "women", "tashi", "zhongguo",
        "shuij", "wmc",
    ]
    .into_iter()
    .enumerate()
    {
        let keys = Keys::new(input).expect("keys");
        for options in [LegalityOptions::OFFLINE, LegalityOptions::LIVE] {
            let lattice = legality.lattice(&keys, options);
            let emission = mock::hashed_rows(keys.len(), emittable, seed as u64, 4.0);
            let transition = HashedBigram {
                seed: 7 + seed as u64,
                spread: 3.0,
            };
            let reference = enumerate(&lattice, &emission, &transition, WEIGHTS);
            let found = search(&lattice, &emission, &transition, None, exact(), WEIGHTS);
            assert_eq!(
                found.sentences.len(),
                reference.len(),
                "{input} {options:?}: the beam holds every legal sentence once"
            );
            for sentence in &found.sentences {
                let expected = reference
                    .get(sentence.chars())
                    .unwrap_or_else(|| panic!("{input}: the beam found an illegal sentence"));
                assert!(
                    (f64::from(sentence.score()) - expected).abs() < 1e-3,
                    "{input} {options:?} {}: beam {} against exact {expected}",
                    text(&legality, sentence.chars()),
                    sentence.score()
                );
            }
            assert!(
                found
                    .sentences
                    .windows(2)
                    .all(|pair| pair[0].score() >= pair[1].score()),
                "sentences come back best first"
            );
            cases += usize::from(!reference.is_empty());
        }
    }
    assert!(cases >= 18, "most cases have legal sentences: {cases}");
}

#[test]
fn a_narrow_beam_returns_legal_sentences_bounded_by_the_exact_scores() {
    let legality = Legality::over(&SMALL.chars().collect::<Vec<_>>()).expect("the set reads");
    let emittable = legality.emittable().len();
    for (seed, input) in ["shuijiap", "zhongguo", "women", "xian"]
        .into_iter()
        .enumerate()
    {
        let keys = Keys::new(input).expect("keys");
        let lattice = legality.lattice(&keys, LegalityOptions::OFFLINE);
        let emission = mock::hashed_rows(keys.len(), emittable, seed as u64, 4.0);
        let transition = HashedBigram {
            seed: 3,
            spread: 3.0,
        };
        let reference = enumerate(&lattice, &emission, &transition, WEIGHTS);
        let found = search(&lattice, &emission, &transition, None, shape(2), WEIGHTS);
        assert!(!found.sentences.is_empty());
        for sentence in &found.sentences {
            let exact = reference[sentence.chars()];
            // A pruned beam sums fewer alignments than exist, never more.
            assert!(f64::from(sentence.score()) <= exact + 1e-3);
            assert_cover(&lattice, sentence);
        }
    }
}

/// *sentence*'s best alignment is a cover of the keys by lattice edges.
fn assert_cover(lattice: &KeyLattice, sentence: &Sentence) {
    assert_eq!(sentence.ends().last(), Some(&lattice.keys()));
    let mut start = 0;
    for (&id, &end) in sentence.chars().iter().zip(sentence.ends()) {
        let edge = lattice
            .edge(start, end)
            .unwrap_or_else(|| panic!("no edge {start}..{end}"));
        assert!(
            edge.cost_of(id).is_some(),
            "{start}..{end} does not admit {id:?}"
        );
        start = end;
    }
}

/// Rows that put most of each key's mass on the planned column.
fn planned(legality: &Legality, keys: usize, plan: &[(usize, char)]) -> EmissionRows {
    let favoured: Vec<(usize, usize)> = plan
        .iter()
        .map(|&(key, ch)| (key, legality.emit_id(ch).expect("emittable").index()))
        .collect();
    mock::rows(keys, legality.emittable().len(), |key, column| {
        let emits_here = favoured.iter().any(|&(at, _)| at == key);
        match column {
            Column::Blank if !emits_here => 12.0,
            Column::Char(index) if favoured.contains(&(key, index)) => 12.0,
            _ => 0.0,
        }
    })
}

#[test]
fn wmzyq_never_reads_as_the_dumplings() {
    let legality = Legality::over_lexicon().expect("tables");
    let keys = Keys::new("wmzyq").expect("keys");
    let target = ids(&legality, "我们吃水饺");
    let emission = planned(
        &legality,
        5,
        &[(0, '我'), (1, '们'), (2, '吃'), (3, '水'), (4, '饺')],
    );
    let transition = Favoured::new(None, &[&target], -20.0);
    for options in [LegalityOptions::OFFLINE, LegalityOptions::LIVE] {
        let lattice = legality.lattice(&keys, options);
        let found = search(
            &lattice,
            &emission,
            &transition,
            None,
            shape(256),
            Weights::UNIT,
        );
        assert!(!found.sentences.is_empty(), "wmzyq reads some way");
        for sentence in &found.sentences {
            assert_ne!(sentence.chars(), &target[..], "{options:?}");
            assert_cover(&lattice, sentence);
        }
    }
}

#[test]
fn shuijiap_after_the_context_reaches_the_dumplings() {
    let legality = Legality::over_lexicon().expect("tables");
    let keys = Keys::new("shuijiap").expect("keys");
    let target = ids(&legality, "水饺");
    let after = legality.emit_id('吃');
    // The head puts 水 on `i` and 饺 on `p`: `jiap` read as jiao through the
    // p→o correction, against `jia|p` with `p` an initial.
    let emission = planned(&legality, 8, &[(3, '水'), (7, '饺')]);
    let transition = Favoured::new(after, &[&target], -6.0);
    for options in [LegalityOptions::OFFLINE, LegalityOptions::LIVE] {
        let lattice = legality.lattice(&keys, options);
        let found = search(
            &lattice,
            &emission,
            &transition,
            None,
            Shape::PRODUCT,
            Weights::UNIT,
        );
        let best = &found.sentences[0];
        assert_eq!(
            text(&legality, best.chars()),
            "水饺",
            "{options:?}: {:?}",
            found
                .sentences
                .iter()
                .map(|s| text(&legality, s.chars()))
                .collect::<Vec<_>>()
        );
        assert_eq!(best.ends(), &[4, 8]);
        let cost = lattice.edge(4, 8).and_then(|edge| edge.cost_of(target[1]));
        assert!(
            cost.is_some_and(|cost| cost > 0.0),
            "饺 is reached through a correction"
        );
    }
}

#[test]
fn the_per_key_cost_stays_within_its_bound() {
    let legality = Legality::over_lexicon().expect("tables");
    let emittable = legality.emittable().len();
    let bound = legality.admitted_bound();
    for (seed, input) in [
        "zuixiamianleiniaod",
        "ercylmm",
        "zhecjyzhzshb",
        "yiqianzhenkaolvguo",
        "wmzyq",
        "shuijiap",
    ]
    .into_iter()
    .enumerate()
    {
        let keys = Keys::new(input).expect("keys");
        for options in [LegalityOptions::OFFLINE, LegalityOptions::LIVE] {
            let lattice = legality.lattice(&keys, options);
            let emission = mock::hashed_rows(keys.len(), emittable, seed as u64, 6.0);
            let transition = HashedBigram {
                seed: 11,
                spread: 4.0,
            };
            let found = search(
                &lattice,
                &emission,
                &transition,
                None,
                Shape::PRODUCT,
                Weights::UNIT,
            );
            let width = Shape::PRODUCT.width();
            let stats = &found.stats;
            assert!(
                stats.max_candidates <= width * (1 + bound),
                "{input}: {stats:?}"
            );
            assert!(stats.max_advanced <= width, "{input}: {stats:?}");
            assert!(stats.max_transition_gathers <= width, "{input}: {stats:?}");
            assert!(found.sentences.len() <= Shape::PRODUCT.top_k());
            for sentence in &found.sentences {
                assert_cover(&lattice, sentence);
                assert!(
                    legality
                        .sentence_cost(&keys, &text(&legality, sentence.chars()), options)
                        .is_some()
                );
            }
        }
    }
}

#[test]
fn a_word_arc_lands_whole_at_the_character_path_score_plus_its_prior() {
    let legality = Legality::over(&"一举两得以及点大家拉的地".chars().collect::<Vec<_>>())
        .expect("the set reads");
    let keys = Keys::new("yjld").expect("keys");
    let lattice = legality.lattice(&keys, LegalityOptions::OFFLINE);
    let idiom = ids(&legality, "一举两得");
    let rival = ids(&legality, "以及两点");
    // Flat emissions, and a transition that charges every character off the
    // rival's path: the idiom trails by 4 nats a character.
    let emission = mock::hashed_rows(4, legality.emittable().len(), 5, 0.0);
    let transition = Favoured::new(None, &[&rival], -4.0);
    let (words, stats) = WordArcs::new(&legality, ["一举两得", "大家", "x", "一举两得"]);
    assert_eq!(stats.kept, 2);
    assert_eq!(stats.duplicate, 1);
    assert_eq!(stats.unemittable, 1);

    let plain = search(
        &lattice,
        &emission,
        &transition,
        None,
        shape(2),
        Weights::UNIT,
    );
    assert_eq!(plain.sentences[0].chars(), &rival[..]);
    assert!(
        plain.sentences.iter().all(|s| s.chars() != &idiom[..]),
        "a two-wide beam loses the idiom character by character"
    );

    let prior = 30.0;
    let weights = Weights {
        word: prior,
        ..Weights::UNIT
    };
    let with = search(
        &lattice,
        &emission,
        &transition,
        Some(&words),
        shape(2),
        weights,
    );
    assert_eq!(with.stats.arcs_offered, 1, "大家 is not spelled by yjld");
    let best = &with.sentences[0];
    assert_eq!(best.chars(), &idiom[..]);
    assert_cover(&lattice, best);

    // The arc sums every cover of its keys, as the character path would.
    let exact = enumerate(&lattice, &emission, &transition, Weights::UNIT)[&idiom];
    assert!(
        (f64::from(best.score()) - (exact + f64::from(prior))).abs() < 1e-3,
        "{} against {}",
        best.score(),
        exact + f64::from(prior)
    );
}

#[test]
fn a_word_the_keys_do_not_spell_is_never_offered() {
    let legality = Legality::over_lexicon().expect("tables");
    let (words, _) = WordArcs::new(&legality, ["我们吃水饺", "我们"]);
    let lattice = legality.lattice(&Keys::new("wmzyq").expect("keys"), LegalityOptions::LIVE);
    let offered = words.offered(&lattice);
    let ours = ids(&legality, "我们");
    assert!(offered.iter().all(|&(_, _, chars)| chars == &ours[..]));
    assert!(
        offered
            .iter()
            .any(|&(start, end, _)| (start, end) == (0, 2)),
        "我们 is spelled by w|m"
    );
}

#[test]
fn a_shape_cannot_return_more_than_it_carries() {
    let two = NonZeroUsize::new(2).expect("not zero");
    let three = NonZeroUsize::new(3).expect("not zero");
    assert!(Shape::new(two, three).is_err());
    assert_eq!(Shape::PRODUCT.width(), 8);
    assert_eq!(Shape::PRODUCT.top_k(), 8);
}

#[test]
fn ragged_rows_are_rejected() {
    assert_eq!(
        EmissionRows::new(3, vec![0.0; 7]).expect_err("ragged"),
        RowsError::Ragged { len: 7, stride: 4 }
    );
    assert_eq!(
        EmissionRows::new(3, Vec::new()).expect_err("empty"),
        RowsError::Empty
    );
}
