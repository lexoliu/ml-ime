//! The sidecar that says which seed term each synthetic sample came from.
//!
//! The decision record asks for per-term provenance so that a bad generation
//! batch can be dropped wholesale. That is only possible if the link survives
//! outside this process, so every written sample gets a row here keyed on the
//! same content hash: filter the sidecar on `seed_source` or `term`, and the
//! `id`s that come back are exactly the sample rows to remove.
//!
//! It is a separate file rather than extra columns because the sample shards have
//! to stay byte-compatible with the schema every other stage reads -- `ime-cli
//! export ngram-corpus` globs the samples directory and would meet an unfamiliar
//! frame otherwise.

use crate::error::Result;
use crate::source::SeedSource;
use ime_g2p::Result as ShardResult;
use ime_g2p::shards::{Shardable, column_of_strings, read_shards, strings};
use polars::prelude::{Column, DataFrame};
use std::path::{Path, PathBuf};

/// How many rows one provenance shard holds before the next one is started.
pub const ROWS_PER_SHARD: usize = 50_000;

/// Where the sidecar lives under a data root.
#[must_use]
pub fn directory(root: &Path) -> PathBuf {
    root.join("provenance")
}

/// One synthetic sample's origin.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Provenance {
    /// The sample's content hash, as written into the sample shards.
    pub id: String,
    /// The seed term the sample was generated from.
    pub term: String,
    /// Which seed lexicon the term is attributed to.
    pub seed_source: String,
    /// The Sogou dictionary id, for the terms that came from one.
    pub dict_id: Option<i64>,
}

impl Provenance {
    /// The provenance of a sample generated from `term` out of `source`.
    #[must_use]
    pub fn new(id: String, term: String, source: SeedSource) -> Self {
        Self {
            id,
            term,
            seed_source: source.name.to_owned(),
            dict_id: source.dict_id,
        }
    }
}

impl Shardable for Provenance {
    fn frame(rows: &[Self]) -> ShardResult<DataFrame> {
        Ok(DataFrame::new(
            rows.len(),
            vec![
                strings("id", rows.iter().map(|row| row.id.clone())),
                strings("term", rows.iter().map(|row| row.term.clone())),
                strings(
                    "seed_source",
                    rows.iter().map(|row| row.seed_source.clone()),
                ),
                Column::new(
                    "dict_id".into(),
                    rows.iter().map(|row| row.dict_id).collect::<Vec<_>>(),
                ),
            ],
        )?)
    }

    fn from_frame(frame: &DataFrame) -> ShardResult<Vec<Self>> {
        let ids = column_of_strings(frame, "id")?;
        let terms = column_of_strings(frame, "term")?;
        let sources = column_of_strings(frame, "seed_source")?;
        let dict_ids: Vec<Option<i64>> = frame.column("dict_id")?.i64()?.iter().collect();
        Ok(ids
            .into_iter()
            .zip(terms)
            .zip(sources)
            .zip(dict_ids)
            .map(|(((id, term), seed_source), dict_id)| Self {
                id,
                term,
                seed_source,
                dict_id,
            })
            .collect())
    }
}

/// Read every provenance row under a data root.
///
/// # Errors
///
/// If the sidecar is missing, or one of its shards cannot be read.
pub fn read(root: &Path) -> Result<Vec<Provenance>> {
    Ok(read_shards(&directory(root), crate::source::PROVENANCE)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::{SOGOU_PREMIUM, WIKI_SLANG};

    #[test]
    fn a_row_carries_its_sources_dictionary_id_and_a_wiki_row_carries_none() {
        let sogou = Provenance::new("a".to_owned(), "东大".to_owned(), SOGOU_PREMIUM);
        assert_eq!(sogou.seed_source, "sogou-premium");
        assert_eq!(sogou.dict_id, Some(4));
        let wiki = Provenance::new("b".to_owned(), "爷青结".to_owned(), WIKI_SLANG);
        assert_eq!(wiki.dict_id, None);
    }

    #[test]
    fn the_sidecar_survives_a_round_trip_through_its_frame() {
        let rows = vec![
            Provenance::new("a".to_owned(), "东大".to_owned(), SOGOU_PREMIUM),
            Provenance::new("b".to_owned(), "爷青结".to_owned(), WIKI_SLANG),
        ];
        let frame = Provenance::frame(&rows).expect("the frame builds");
        let names: Vec<&str> = frame
            .get_column_names()
            .into_iter()
            .map(polars::prelude::PlSmallStr::as_str)
            .collect();
        assert_eq!(names, ["id", "term", "seed_source", "dict_id"]);
        assert_eq!(
            Provenance::from_frame(&frame).expect("the frame reads back"),
            rows
        );
    }
}
