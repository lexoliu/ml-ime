//! The streamed upstreams: which files of a dataset are a source, in what order,
//! and how one downloaded file becomes raw documents.
//!
//! These corpora are too large to fetch whole -- FineWeb-2's Mandarin split is
//! 1.6 TB of parquet -- so [`crate::ingest`] takes them one file at a time and
//! deletes each file once it is processed. What differs between them is only
//! what this module holds: the files that make up the source, the order they
//! are taken in, and the record layout of a file.

use crate::error::{Error, Result};
use crate::source::{FINEWEB2_HQ, FINEWEB2_MINED, MNBVC_FORUM, RawDocument, SourceSpec, YYETS};
use polars::prelude::{ParquetReader, SerReader as _};
use std::io::BufRead as _;
use std::path::Path;

/// How many documents a batch hands to the parallel stage at once.
pub const DOCUMENTS_PER_BATCH: usize = 8_192;

/// One streamed upstream corpus.
pub trait Upstream: Sync {
    /// The source its documents become.
    const SPEC: SourceSpec;
    /// The directory of the dataset its files sit under.
    const PREFIX: &'static str;

    /// Whether a file under [`Self::PREFIX`] belongs to the source.
    fn selects(path: &str) -> bool;

    /// The order files are processed in.
    ///
    /// A source that is only partly processed -- a budget of targets, or a
    /// share of the files -- takes [`spread`] order, so that wherever it stops,
    /// what it has read is spread evenly over the whole dataset rather than
    /// being its first crawl dumps.
    fn order<T>(files: Vec<T>) -> Vec<T>;

    /// Read one downloaded file, handing its documents to `sink` in batches.
    ///
    /// Returns how many bytes of text the documents held.
    ///
    /// # Errors
    ///
    /// If the file does not have the source's layout, or `sink` refuses.
    fn read(path: &Path, sink: &mut dyn FnMut(Vec<RawDocument>) -> Result<Flow>) -> Result<u64>;
}

/// Whether the consumer wants more documents from the file being read.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Flow {
    /// Keep reading.
    More,
    /// Stop: the source has all it needs.
    Stop,
}

/// `items` reordered so that every prefix of the result is spread over the whole.
///
/// The order of bit-reversed indices: 0, n/2, n/4, 3n/4, ... A run that stops
/// after any number of files has drawn them evenly from the listing.
#[must_use]
pub fn spread<T>(items: Vec<T>) -> Vec<T> {
    if items.len() <= 1 {
        return items;
    }
    let bits = usize::BITS - (items.len() - 1).leading_zeros();
    let mut keyed: Vec<(usize, T)> = items
        .into_iter()
        .enumerate()
        .map(|(index, item)| (index.reverse_bits() >> (usize::BITS - bits), item))
        .collect();
    keyed.sort_by_key(|(key, _)| *key);
    keyed.into_iter().map(|(_, item)| item).collect()
}

/// The top decile of FineWeb-2's Mandarin by quality classifier, for S2's general text.
#[derive(Debug)]
pub struct FineWeb2Hq;

impl Upstream for FineWeb2Hq {
    const SPEC: SourceSpec = FINEWEB2_HQ;
    const PREFIX: &'static str = "cmn_Hani";

    fn selects(path: &str) -> bool {
        path.ends_with(".parquet")
    }

    fn order<T>(files: Vec<T>) -> Vec<T> {
        spread(files)
    }

    fn read(path: &Path, sink: &mut dyn FnMut(Vec<RawDocument>) -> Result<Flow>) -> Result<u64> {
        parquet_texts(path, Self::SPEC, sink)
    }
}

/// FineWeb-2's Mandarin, read only for the sentences that hold an absent entry.
#[derive(Debug)]
pub struct FineWeb2;

impl Upstream for FineWeb2 {
    const SPEC: SourceSpec = FINEWEB2_MINED;
    const PREFIX: &'static str = "data/cmn_Hani/train";

    fn selects(path: &str) -> bool {
        path.ends_with(".parquet")
    }

    fn order<T>(files: Vec<T>) -> Vec<T> {
        spread(files)
    }

    fn read(path: &Path, sink: &mut dyn FnMut(Vec<RawDocument>) -> Result<Flow>) -> Result<u64> {
        parquet_texts(path, Self::SPEC, sink)
    }
}

/// The MNBVC forum boards: MOP 里屋, the PKU and THU 树洞 boards, and smder.
///
/// The fifth board of the same dump, a Marxist reading forum, is long-form
/// political prose rather than the informal register the set was taken for.
#[derive(Debug)]
pub struct MnbvcForum;

/// The boards of `forum/20230253` that make up the source, by directory prefix.
const FORUM_BOARDS: [&str; 4] = ["liwu.", "pkuhole.", "thuhole.", "smder."];

/// One forum thread: its opening post and the replies under it.
#[derive(serde::Deserialize)]
struct Thread {
    #[serde(rename = "主题")]
    topic: String,
    #[serde(rename = "回复")]
    replies: Vec<Reply>,
}

#[derive(serde::Deserialize)]
struct Reply {
    #[serde(rename = "回复")]
    text: String,
}

impl Upstream for MnbvcForum {
    const SPEC: SourceSpec = MNBVC_FORUM;
    const PREFIX: &'static str = "forum/20230253";

    fn selects(path: &str) -> bool {
        let Some(board) = path
            .strip_prefix(Self::PREFIX)
            .and_then(|rest| rest.trim_start_matches('/').split('/').next())
        else {
            return false;
        };
        path.ends_with(".jsonl.gz") && FORUM_BOARDS.iter().any(|name| board.starts_with(name))
    }

    fn order<T>(files: Vec<T>) -> Vec<T> {
        files
    }

    fn read(path: &Path, sink: &mut dyn FnMut(Vec<RawDocument>) -> Result<Flow>) -> Result<u64> {
        let mut batch = Vec::with_capacity(DOCUMENTS_PER_BATCH);
        let mut bytes = 0_u64;
        let mut stopped = false;
        json_lines(path, &mut |line_number, line| {
            let thread: Thread = serde_json::from_str(line).map_err(|source| Error::JsonLine {
                path: path.to_owned(),
                line: line_number,
                source,
            })?;
            let parts: Vec<String> = std::iter::once(thread.topic)
                .chain(thread.replies.into_iter().map(|reply| reply.text))
                .filter(|part| !part.trim().is_empty())
                .collect();
            bytes += parts.iter().map(|part| part.len() as u64).sum::<u64>();
            if !parts.is_empty() {
                batch.push(RawDocument {
                    document_id: format!("{}:{line_number}", path.display()),
                    source: Self::SPEC.name.to_owned(),
                    parts,
                });
            }
            if batch.len() >= DOCUMENTS_PER_BATCH && sink(std::mem::take(&mut batch))? == Flow::Stop
            {
                stopped = true;
                return Ok(Flow::Stop);
            }
            Ok(Flow::More)
        })?;
        if !stopped && !batch.is_empty() {
            sink(batch)?;
        }
        Ok(bytes)
    }
}

/// The `YYeTs` fan subtitles: one document per subtitle file, one turn per line.
#[derive(Debug)]
pub struct Yyets;

/// One subtitle line and the file it came from.
#[derive(serde::Deserialize)]
struct SubtitleLine {
    zh_text: String,
    #[serde(rename = "扩展字段")]
    file: String,
}

impl Upstream for Yyets {
    const SPEC: SourceSpec = YYETS;
    const PREFIX: &'static str = "parallel/subtitle/yyets";

    fn selects(path: &str) -> bool {
        path.ends_with(".jsonl.gz")
    }

    fn order<T>(files: Vec<T>) -> Vec<T> {
        files
    }

    fn read(path: &Path, sink: &mut dyn FnMut(Vec<RawDocument>) -> Result<Flow>) -> Result<u64> {
        let mut batch = Vec::with_capacity(DOCUMENTS_PER_BATCH);
        let mut current: Option<(String, Vec<String>, usize)> = None;
        let mut bytes = 0_u64;
        let mut stopped = false;
        let finish = |batch: &mut Vec<RawDocument>,
                      (file, lines, first): (String, Vec<String>, usize)| {
            if !lines.is_empty() {
                batch.push(RawDocument {
                    document_id: format!("{}:{first}:{file}", path.display()),
                    source: Self::SPEC.name.to_owned(),
                    parts: lines,
                });
            }
        };
        json_lines(path, &mut |line_number, line| {
            let row: SubtitleLine =
                serde_json::from_str(line).map_err(|source| Error::JsonLine {
                    path: path.to_owned(),
                    line: line_number,
                    source,
                })?;
            if current
                .as_ref()
                .is_some_and(|(file, _, _)| *file != row.file)
            {
                if let Some(done) = current.take() {
                    finish(&mut batch, done);
                }
                if batch.len() >= DOCUMENTS_PER_BATCH
                    && sink(std::mem::take(&mut batch))? == Flow::Stop
                {
                    stopped = true;
                    return Ok(Flow::Stop);
                }
            }
            let (_, lines, _) = current.get_or_insert_with(|| (row.file, Vec::new(), line_number));
            if !row.zh_text.trim().is_empty() {
                bytes += row.zh_text.len() as u64;
                lines.push(row.zh_text);
            }
            Ok(Flow::More)
        })?;
        if !stopped {
            if let Some(done) = current.take() {
                finish(&mut batch, done);
            }
            if !batch.is_empty() {
                sink(batch)?;
            }
        }
        Ok(bytes)
    }
}

/// Every line of a gzipped JSON Lines file, numbered from one, until `each` stops.
fn json_lines(path: &Path, each: &mut dyn FnMut(usize, &str) -> Result<Flow>) -> Result<()> {
    let file = std::fs::File::open(path).map_err(|source| Error::Read {
        path: path.to_owned(),
        source,
    })?;
    let reader = std::io::BufReader::with_capacity(
        1 << 20,
        flate2::read::MultiGzDecoder::new(std::io::BufReader::with_capacity(1 << 20, file)),
    );
    for (index, line) in reader.lines().enumerate() {
        let line = line.map_err(|source| Error::Read {
            path: path.to_owned(),
            source,
        })?;
        if line.trim().is_empty() {
            continue;
        }
        if each(index + 1, &line)? == Flow::Stop {
            break;
        }
    }
    Ok(())
}

/// How many rows of a parquet file are read into memory at once.
const PARQUET_ROWS_PER_SLICE: usize = 16_384;

/// The `text` column of a parquet file, one document per row, read in slices.
///
/// A FineWeb-2 file is about four gigabytes of compressed text, several times
/// that once decoded, so it is read a slice of rows at a time with every other
/// column projected away.
fn parquet_texts(
    path: &Path,
    spec: SourceSpec,
    sink: &mut dyn FnMut(Vec<RawDocument>) -> Result<Flow>,
) -> Result<u64> {
    parquet_texts_in_slices(path, spec, PARQUET_ROWS_PER_SLICE, sink)
}

/// [`parquet_texts`] with the slice size as a parameter.
///
/// One handle serves every slice: a file read slice by slice through fresh
/// handles keeps each of them open past its slice, and a four-gigabyte file is
/// over a hundred slices.
fn parquet_texts_in_slices(
    path: &Path,
    spec: SourceSpec,
    rows_per_slice: usize,
    sink: &mut dyn FnMut(Vec<RawDocument>) -> Result<Flow>,
) -> Result<u64> {
    let mut file = std::fs::File::open(path).map_err(|source| Error::Read {
        path: path.to_owned(),
        source,
    })?;
    let mut reader = ParquetReader::new(&mut file);
    let rows = reader.num_rows()?;
    let metadata = reader.get_metadata()?.clone();
    let mut bytes = 0_u64;
    for offset in (0..rows).step_by(rows_per_slice) {
        let mut slice = ParquetReader::new(&mut file);
        slice.set_metadata(metadata.clone());
        let frame = slice
            .with_columns(Some(vec!["text".to_owned()]))
            .with_slice(Some((offset, rows_per_slice.min(rows - offset))))
            .finish()
            .map_err(|source| Error::Layout {
                dataset: spec.dataset,
                detail: format!("{} has no readable text column: {source}", path.display()),
            })?;
        let column = frame
            .column("text")
            .and_then(|column| column.str().cloned())
            .map_err(|source| Error::Layout {
                dataset: spec.dataset,
                detail: format!("{}'s text is not a string column: {source}", path.display()),
            })?;
        let batch: Vec<RawDocument> = column
            .iter()
            .enumerate()
            .filter_map(|(row, text)| {
                let text = text.filter(|text| !text.trim().is_empty())?;
                bytes += text.len() as u64;
                Some(RawDocument {
                    document_id: format!("{}:{}", path.display(), offset + row),
                    source: spec.name.to_owned(),
                    parts: vec![text.to_owned()],
                })
            })
            .collect();
        if sink(batch)? == Flow::Stop {
            break;
        }
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    #[test]
    fn spread_order_visits_both_halves_before_either_quarter() {
        let order = spread((0..8).collect());
        assert_eq!(order, vec![0, 4, 2, 6, 1, 5, 3, 7]);
        let order = spread((0..370).collect::<Vec<usize>>());
        assert_eq!(order.len(), 370);
        assert_eq!(order[..4], [0, 256, 128, 64]);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..370).collect::<Vec<_>>());
        // Every prefix of 37 draws from all ten tenths of the listing.
        let tenths: std::collections::HashSet<usize> =
            order[..37].iter().map(|index| index * 10 / 370).collect();
        assert_eq!(tenths.len(), 10);
    }

    /// How many descriptors this process holds open.
    fn open_descriptors() -> usize {
        std::fs::read_dir("/dev/fd")
            .expect("the descriptor table lists")
            .count()
    }

    #[test]
    fn a_parquet_file_reads_slice_by_slice_without_holding_descriptors() {
        let root = std::env::temp_dir().join(format!("ime-corpus-slices-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let mut writer =
            ime_g2p::shards::ShardWriter::new(&root, "texts", 10_000).expect("writable");
        for index in 0..2_000 {
            writer
                .write(ime_g2p::annotate::Sample {
                    id: index.to_string(),
                    source: "fixture".to_owned(),
                    text: format!("第{index}篇文档"),
                    context: None,
                })
                .expect("buffered");
        }
        writer.finish().expect("written");
        let path = root.join("texts-00000.parquet");
        let before = open_descriptors();
        let mut documents = 0;
        let mut slices = 0;
        let bytes = parquet_texts_in_slices(&path, FINEWEB2_HQ, 10, &mut |batch| {
            documents += batch.len();
            slices += 1;
            Ok(Flow::More)
        })
        .expect("the file reads");
        assert_eq!(slices, 200);
        assert_eq!(documents, 2_000);
        assert!(bytes > 0);
        assert!(
            open_descriptors() <= before + 2,
            "descriptors piled up over the slices"
        );
        std::fs::remove_dir_all(&root).expect("removed");
    }

    #[test]
    fn only_the_four_informal_boards_are_the_forum_source() {
        assert!(MnbvcForum::selects(
            "forum/20230253/pkuhole.20230253.2.论坛/pkuholefromarchive_0.jsonl.gz"
        ));
        assert!(MnbvcForum::selects(
            "forum/20230253/liwu.20230253.1.论坛/liwu_253874_com.jsonl.gz"
        ));
        assert!(!MnbvcForum::selects(
            "forum/20230253/marxists.20230253.5.论坛/0.jsonl.gz"
        ));
        assert!(!MnbvcForum::selects(
            "forum/20230253/smder.20230253.6.论坛/README.md"
        ));
    }

    fn gzip(path: &Path, lines: &[serde_json::Value]) {
        let file = std::fs::File::create(path).expect("writable");
        let mut encoder = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
        for line in lines {
            writeln!(encoder, "{line}").expect("written");
        }
        encoder.finish().expect("finished");
    }

    #[test]
    fn a_forum_thread_is_one_document_of_its_topic_and_replies() {
        let root = std::env::temp_dir().join(format!("ime-corpus-forum-{}", std::process::id()));
        std::fs::create_dir_all(&root).expect("writable");
        let path = root.join("board.jsonl.gz");
        gzip(
            &path,
            &[serde_json::json!({
                "ID": 0, "主题": "之前一直苦于北大没有树洞。", "来源": "北大树洞",
                "回复": [{"楼ID": "1", "回复": "史前第一条树洞留名"}, {"楼ID": "2", "回复": "  "}],
            })],
        );
        let mut documents = Vec::new();
        let bytes = MnbvcForum::read(&path, &mut |batch| {
            documents.extend(batch);
            Ok(Flow::More)
        })
        .expect("the file reads");
        assert_eq!(documents.len(), 1);
        assert_eq!(
            documents[0].parts,
            vec!["之前一直苦于北大没有树洞。", "史前第一条树洞留名"]
        );
        assert_eq!(
            bytes,
            ("之前一直苦于北大没有树洞。".len() + "史前第一条树洞留名".len()) as u64
        );
        std::fs::remove_dir_all(&root).expect("removed");
    }

    #[test]
    fn subtitle_lines_group_into_one_document_per_subtitle_file() {
        let root = std::env::temp_dir().join(format!("ime-corpus-yyets-{}", std::process::id()));
        std::fs::create_dir_all(&root).expect("writable");
        let path = root.join("yyets.jsonl.gz");
        let line = |text: &str, file: &str| serde_json::json!({"zh_text": text, "en_text": "x", "扩展字段": file});
        gzip(
            &path,
            &[
                line("联调局警报", "{\"k\": \"a.ass\"}"),
                line("", "{\"k\": \"a.ass\"}"),
                line("在公园发现被盗车辆", "{\"k\": \"a.ass\"}"),
                line("我们走吧", "{\"k\": \"b.ass\"}"),
            ],
        );
        let mut documents = Vec::new();
        Yyets::read(&path, &mut |batch| {
            documents.extend(batch);
            Ok(Flow::More)
        })
        .expect("the file reads");
        let parts: Vec<Vec<String>> = documents
            .into_iter()
            .map(|document| document.parts)
            .collect();
        assert_eq!(
            parts,
            vec![
                vec!["联调局警报".to_owned(), "在公园发现被盗车辆".to_owned()],
                vec!["我们走吧".to_owned()],
            ]
        );
        std::fs::remove_dir_all(&root).expect("removed");
    }
}
