//! Reading a finished run back: its samples, their seed terms, and a sample of both.
//!
//! The point of showing rows rather than only counts is that the one failure this
//! stage cannot detect mechanically is the model answering with *definitions*
//! instead of usage. A sentence that reads 「爷青结的意思是青春结束了」 contains its
//! term, is the right length, is entirely Han, and is worthless as training data
//! for an input method. So the report prints real rows with their seed term
//! beside them, and a person decides.

use crate::error::{Error, Result};
use crate::provenance::{self, Provenance};
use crate::source::SYNTHETIC;
use ime_g2p::DataLayout;
use ime_g2p::annotate::Sample;
use ime_g2p::shards::read_shards;
use rand::SeedableRng as _;
use rand::seq::IndexedRandom as _;
use std::collections::HashMap;
use std::path::Path;

/// One sample as the report shows it: the term it was seeded from, and the turn pair.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Shown {
    /// The seed term.
    pub term: String,
    /// Which seed lexicon it came from, written `lexicon+encyclopaedia` when the
    /// two differ, as they do for every Sogou word 梗百科 explained.
    pub seed_source: String,
    /// The preceding turn, empty when there was none.
    pub context: String,
    /// The generated sentence.
    pub text: String,
}

/// Read the synthetic sample shards a run wrote under `root`.
///
/// # Errors
///
/// If no run has written any, or a shard cannot be read.
pub fn samples(root: &Path) -> Result<Vec<Sample>> {
    Ok(read_shards(&DataLayout::new(root).samples(), SYNTHETIC)?)
}

/// How a sample's origin is labelled: the lexicon that proposed the term, and
/// the source that explained it where that is a different one.
#[must_use]
pub fn label(seed_source: &str, grounding: &str) -> String {
    if seed_source == grounding {
        return seed_source.to_owned();
    }
    format!("{seed_source}+{grounding}")
}

/// Join a run's samples to their provenance rows.
///
/// # Errors
///
/// If a sample has no provenance row, which means the two outputs disagree and
/// the batch can no longer be dropped by term.
pub fn joined(samples: &[Sample], provenance: &[Provenance]) -> Result<Vec<Shown>> {
    let origins: HashMap<&str, &Provenance> = provenance
        .iter()
        .map(|row| (row.id.as_str(), row))
        .collect();
    samples
        .iter()
        .map(|sample| {
            let origin = origins.get(sample.id.as_str()).ok_or_else(|| {
                Error::Invariant(format!(
                    "sample {} has no provenance row, so its batch cannot be dropped by term",
                    sample.id
                ))
            })?;
            Ok(Shown {
                term: origin.term.clone(),
                seed_source: label(&origin.seed_source, &origin.grounding),
                context: sample.context.clone().unwrap_or_default(),
                text: sample.text.clone(),
            })
        })
        .collect()
}

/// Every written sample of a run, with the seed term it came from.
///
/// # Errors
///
/// If either output is missing, or the two disagree.
pub fn shown(root: &Path) -> Result<Vec<Shown>> {
    joined(&samples(root)?, &provenance::read(root)?)
}

/// Draw `count` rows at random, reproducibly for a given `seed`.
#[must_use]
pub fn draw<T: Clone>(rows: &[T], count: usize, seed: u64) -> Vec<T> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    rows.choose_multiple(&mut rng, count.min(rows.len()))
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::{GENGBAIKE, SOGOU_PREMIUM, WIKI_SLANG};

    fn sample(id: &str, text: &str, context: Option<&str>) -> Sample {
        Sample {
            id: id.to_owned(),
            source: SYNTHETIC.to_owned(),
            text: text.to_owned(),
            context: context.map(ToOwned::to_owned),
        }
    }

    #[test]
    fn a_sample_is_shown_beside_the_term_it_was_seeded_from() {
        let samples = vec![
            sample("a", "看完了，爷青结", Some("你追完了吗")),
            sample("b", "东大又赢了", None),
        ];
        let provenance = vec![
            Provenance::new("b".to_owned(), "东大".to_owned(), SOGOU_PREMIUM, GENGBAIKE),
            Provenance::new("a".to_owned(), "爷青结".to_owned(), WIKI_SLANG, WIKI_SLANG),
        ];
        let shown = joined(&samples, &provenance).expect("every sample has a row");
        assert_eq!(shown[0].term, "爷青结");
        assert_eq!(shown[0].seed_source, "wiki-slang");
        assert_eq!(shown[0].context, "你追完了吗");
        assert_eq!(shown[1].term, "东大");
        assert_eq!(shown[1].seed_source, "sogou-premium+gengbaike");
        assert_eq!(shown[1].context, "");
    }

    #[test]
    fn a_sample_without_provenance_is_an_invariant_failure_rather_than_a_blank_term() {
        let error = joined(&[sample("a", "看完了，爷青结", None)], &[])
            .expect_err("the sidecar does not cover it");
        assert!(
            error.to_string().contains("has no provenance row"),
            "{error}"
        );
    }

    #[test]
    fn the_draw_is_reproducible_for_a_seed_and_never_asks_for_more_than_it_has() {
        let rows: Vec<usize> = (0..100).collect();
        assert_eq!(draw(&rows, 10, 7), draw(&rows, 10, 7));
        assert_ne!(draw(&rows, 10, 7), draw(&rows, 10, 8));
        assert_eq!(draw(&rows, 200, 7).len(), 100);
    }
}
