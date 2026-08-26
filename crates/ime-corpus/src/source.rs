//! The three upstream corpora of the run-2 expansion, and what a fetched document is.
//!
//! The stage is split in two, and the line between them is the network. `fetch`
//! writes the upstream text as it arrives, split only where upstream already
//! split it -- one part per article, one per post, one per comment. `prepare`
//! does everything else: cleaning, normalising, sentence-splitting, filtering,
//! deduplicating. Every rule worth changing therefore lives on the side that can
//! be re-run offline, so tightening a filter costs a minute rather than another
//! pass over the network.
//!
//! The three registers are deliberate, and none of them is encyclopaedic in the
//! way `wiki` was. Moe Girl Pedia's prose is written *by* the internet about the
//! internet; douyin captions are what a person types under a video; bilibili
//! comments are what they type under someone else's. Between them they cover the
//! slang, 梗 and abbreviations an n-gram model trained on news has never seen.

use crate::clean::Cleaning;
use ime_g2p::Result as ShardResult;
use ime_g2p::shards::{
    Shardable, column_of_string_lists, column_of_strings, string_lists, strings,
};
use polars::prelude::DataFrame;

/// How many documents one raw shard holds before the next one is started.
pub const DOCUMENTS_PER_SHARD: usize = 20_000;

/// How many samples one prepared shard holds before the next one is started.
pub const SAMPLES_PER_SHARD: usize = 100_000;

/// How many preceding segments a prose sample carries as context.
const PROSE_CONTEXT_SEGMENTS: usize = 3;

/// The longest context a sample carries, in characters, counted from its end.
pub const MAX_CONTEXT_CHARACTERS: usize = 256;

/// One upstream corpus: what it is called, how it is cleaned, and where its context comes from.
///
/// `context_segments` is zero for the two social sources because a post and a
/// comment are each composed on their own -- the post above a comment is not what
/// its author was looking at while typing, and the previous comment in the file
/// is from a different video entirely. Setting it to zero is what makes their
/// `context` column null without a single branch anywhere downstream.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SourceSpec {
    /// The shard prefix and the value of the `source` column.
    pub name: &'static str,
    /// The Hugging Face repository the fetch half reads.
    pub dataset: &'static str,
    /// Which residues the prepare half strips before splitting into sentences.
    pub cleaning: Cleaning,
    /// How many preceding segments become a sample's context.
    pub context_segments: usize,
}

/// Moe Girl Pedia's cleaned 2025-10 dump: one JSON Lines record per article.
pub const MOEGIRL: SourceSpec = SourceSpec {
    name: "moegirl",
    dataset: "YCWTG/MoeGirlPedia_zh_cleaned_latest",
    cleaning: Cleaning {
        drop_infobox_lines: true,
        strip_hashtags: false,
        strip_mentions: false,
        strip_reply_prefix: false,
    },
    context_segments: PROSE_CONTEXT_SEGMENTS,
};

/// Douyin posts: the user-written `desc` caption under a video.
pub const DOUYIN: SourceSpec = SourceSpec {
    name: "douyin",
    dataset: "bendavidsteel/douyin",
    cleaning: Cleaning {
        drop_infobox_lines: false,
        strip_hashtags: true,
        strip_mentions: true,
        strip_reply_prefix: false,
    },
    context_segments: 0,
};

/// Bilibili comments: short, informal, and frequently a reply to another one.
pub const BILIBILI: SourceSpec = SourceSpec {
    name: "bilibili",
    dataset: "Midsummra/bilibilicomment",
    cleaning: Cleaning {
        drop_infobox_lines: false,
        strip_hashtags: false,
        strip_mentions: true,
        strip_reply_prefix: true,
    },
    context_segments: 0,
};

/// Every source this crate knows, in the order the CLI lists them.
pub const SOURCES: [SourceSpec; 3] = [MOEGIRL, DOUYIN, BILIBILI];

/// One upstream document, untouched apart from being split where upstream split it.
///
/// `parts` is a list rather than a string because the schema is the Python
/// pipeline's, which had to hold a dialogue's turns; none of the three sources
/// here splits a record into more than one part, but writing the same column type
/// keeps either implementation able to read the other's raw shards.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RawDocument {
    /// The upstream identifier, or the content's own hash when it has none.
    pub document_id: String,
    /// Which corpus it came from.
    pub source: String,
    /// The record's text, split only where upstream already split it.
    pub parts: Vec<String>,
}

impl Shardable for RawDocument {
    fn frame(rows: &[Self]) -> ShardResult<DataFrame> {
        Ok(DataFrame::new(
            rows.len(),
            vec![
                strings(
                    "document_id",
                    rows.iter().map(|row| row.document_id.clone()),
                ),
                strings("source", rows.iter().map(|row| row.source.clone())),
                string_lists(
                    "parts",
                    rows.len(),
                    rows.iter().map(|row| row.parts.as_slice()),
                ),
            ],
        )?)
    }

    fn from_frame(frame: &DataFrame) -> ShardResult<Vec<Self>> {
        let ids = column_of_strings(frame, "document_id")?;
        let sources = column_of_strings(frame, "source")?;
        let parts = column_of_string_lists(frame, "parts")?;
        Ok(ids
            .into_iter()
            .zip(sources)
            .zip(parts)
            .map(|((document_id, source), parts)| Self {
                document_id,
                source,
                parts,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ime_g2p::shards::{ShardWriter, read_shards};

    #[test]
    fn a_raw_document_survives_a_round_trip_through_a_shard() {
        let root = std::env::temp_dir().join(format!("ime-corpus-raw-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let mut writer =
            ShardWriter::new(&root, MOEGIRL.name, 100).expect("the directory is writable");
        writer
            .write(RawDocument {
                document_id: "初音未来".to_owned(),
                source: MOEGIRL.name.to_owned(),
                parts: vec!["初音未来\n她是虚拟歌手。".to_owned()],
            })
            .expect("the row is buffered");
        writer.finish().expect("the shard is written");

        let read: Vec<RawDocument> =
            read_shards(&root, MOEGIRL.name).expect("the shard reads back");
        assert_eq!(read.len(), 1);
        assert_eq!(read[0].document_id, "初音未来");
        assert_eq!(read[0].parts, vec!["初音未来\n她是虚拟歌手。".to_owned()]);
        std::fs::remove_dir_all(&root).expect("the fixture directory is removed");
    }

    #[test]
    fn only_the_prose_source_carries_context() {
        assert_eq!(MOEGIRL.context_segments, PROSE_CONTEXT_SEGMENTS);
        assert_eq!(DOUYIN.context_segments, 0);
        assert_eq!(BILIBILI.context_segments, 0);
    }

    #[test]
    fn every_source_has_a_distinct_name() {
        let mut names: Vec<&str> = SOURCES.iter().map(|source| source.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), SOURCES.len());
    }
}
