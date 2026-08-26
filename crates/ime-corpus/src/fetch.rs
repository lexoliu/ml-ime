//! The network half: pull each upstream down and write its text as raw documents.
//!
//! No interpretation happens here beyond finding the column that holds the prose.
//! What is written is the record's own text, split only where upstream already
//! split it, so that every rule worth arguing about lives in [`crate::prepare`]
//! and can be re-run without touching the network again.
//!
//! The three sources differ in exactly one way that matters to this module: how
//! much of them has to be on disk at once. Moe Girl Pedia and the bilibili
//! comments are one file each, a few hundred megabytes, and stay where they
//! landed so a re-fetch is free. Douyin is 111 partitions and 9.7 GB of video
//! metadata around 1.13 million captions, so its partitions are taken one at a
//! time, projected down to the caption column, and deleted before the next one
//! starts -- the whole dataset never exists locally.

use crate::download::Hub;
use crate::error::{Error, Result};
use crate::source::{BILIBILI, DOCUMENTS_PER_SHARD, DOUYIN, MOEGIRL, RawDocument, SourceSpec};
use crate::text::content_id;
use ime_g2p::DataLayout;
use ime_g2p::shards::ShardWriter;
use polars::prelude::{ParquetReader, SerReader as _};
use std::path::{Path, PathBuf};
use tracing::info;

/// The column of `bendavidsteel/douyin` that holds a poster's own caption.
const DOUYIN_CAPTION: &str = "desc";

/// The column of `Midsummra/bilibilicomment` that holds a comment's body.
const BILIBILI_MESSAGE: &str = "message";

/// The single JSON Lines file `YCWTG/MoeGirlPedia_zh_cleaned_latest` ships.
const MOEGIRL_FILE: &str = "MoeGirlPedia_zh_cleaned_latest.jsonl";

/// The single CSV `Midsummra/bilibilicomment` ships.
const BILIBILI_FILE: &str = "bilibili.csv";

/// One Moe Girl Pedia article, as the cleaned dump writes it.
#[derive(Debug, serde::Deserialize)]
struct MoegirlArticle {
    title: String,
    text: String,
}

/// Where a source's downloads land under a data root.
fn downloads(layout: &DataLayout, spec: SourceSpec) -> PathBuf {
    layout.root().join("downloads").join(spec.name)
}

/// Collects raw documents into `<data-dir>/documents`, stopping at a limit.
///
/// The limit is enforced here rather than by each source's loop so that "stop
/// after N documents" means the same thing for a single JSON Lines file and for a
/// run of parquet partitions -- and so that a `--limit`ed douyin fetch stops
/// *downloading* once it has enough, rather than pulling all 9.7 GB and throwing
/// most of it away.
#[derive(Debug)]
struct DocumentSink {
    writer: ShardWriter<RawDocument>,
    source: &'static str,
    written: usize,
    limit: Option<usize>,
}

impl DocumentSink {
    fn new(layout: &DataLayout, spec: SourceSpec, limit: Option<usize>) -> Result<Self> {
        Ok(Self {
            writer: ShardWriter::new(&layout.documents(), spec.name, DOCUMENTS_PER_SHARD)?,
            source: spec.name,
            written: 0,
            limit,
        })
    }

    /// Whether the sink has all the documents it was asked for.
    fn full(&self) -> bool {
        self.limit.is_some_and(|limit| self.written >= limit)
    }

    /// Write one document, identified by `id` or by its own content hash.
    ///
    /// A record whose text is empty is skipped rather than written: it carries no
    /// prose, so it would only inflate the document count.
    fn push(&mut self, id: Option<&str>, text: &str) -> Result<()> {
        if text.trim().is_empty() || self.full() {
            return Ok(());
        }
        let document_id = id.map_or_else(|| content_id(self.source, text, None), ToOwned::to_owned);
        self.writer.write(RawDocument {
            document_id,
            source: self.source.to_owned(),
            parts: vec![text.to_owned()],
        })?;
        self.written += 1;
        Ok(())
    }

    fn finish(self) -> Result<usize> {
        let written = self.writer.finish()?;
        Ok(written)
    }
}

/// Fetch every source's raw documents, dispatching on the spec's name.
///
/// # Errors
///
/// Whatever the source's own fetch refuses on.
pub async fn fetch(spec: SourceSpec, layout: &DataLayout, limit: Option<usize>) -> Result<usize> {
    let hub = Hub::new()?;
    let written = match spec.name {
        name if name == MOEGIRL.name => fetch_moegirl(&hub, layout, limit).await,
        name if name == DOUYIN.name => fetch_douyin(&hub, layout, limit).await,
        name if name == BILIBILI.name => fetch_bilibili(&hub, layout, limit).await,
        other => Err(Error::Invariant(format!(
            "no fetcher for the {other} source"
        ))),
    }?;
    info!(source = spec.name, documents = written, "documents fetched");
    Ok(written)
}

/// One named file out of a dataset's root listing, or a clear refusal.
async fn single_file(hub: &Hub, spec: SourceSpec, name: &str, directory: &Path) -> Result<PathBuf> {
    let files = hub.files(spec.dataset, "").await?;
    let file = files
        .iter()
        .find(|file| file.path == name)
        .ok_or_else(|| Error::Layout {
            dataset: spec.dataset,
            detail: format!(
                "expected a {name} at the repository root, found {:?}",
                files.iter().map(|file| &file.path).collect::<Vec<_>>()
            ),
        })?;
    hub.download(spec.dataset, file, directory).await
}

/// Moe Girl Pedia: one JSON Lines record per article, title first.
///
/// The title joins the text as the article's first line, which is what the Python
/// pipeline does for news headlines: it is rarely a sentence in its own right and
/// usually filtered out on length, but where it survives it is the subject the
/// article's opening sentences are about, so it belongs in their context.
async fn fetch_moegirl(hub: &Hub, layout: &DataLayout, limit: Option<usize>) -> Result<usize> {
    let path = single_file(hub, MOEGIRL, MOEGIRL_FILE, &downloads(layout, MOEGIRL)).await?;
    let sink = DocumentSink::new(layout, MOEGIRL, limit)?;
    blocking(move || {
        use std::io::BufRead as _;
        let mut sink = sink;
        let file = std::fs::File::open(&path).map_err(|source| Error::Read {
            path: path.clone(),
            source,
        })?;
        for (index, line) in std::io::BufReader::new(file).lines().enumerate() {
            if sink.full() {
                break;
            }
            let line = line.map_err(|source| Error::Read {
                path: path.clone(),
                source,
            })?;
            if line.trim().is_empty() {
                continue;
            }
            let article: MoegirlArticle =
                serde_json::from_str(&line).map_err(|source| Error::JsonLine {
                    path: path.clone(),
                    line: index + 1,
                    source,
                })?;
            sink.push(
                Some(&article.title),
                &format!("{}\n{}", article.title, article.text),
            )?;
        }
        sink.finish()
    })
    .await
}

/// Douyin: 111 parquet partitions of video metadata, one caption column each.
///
/// Only `desc` is materialised, and the partition file is deleted the moment its
/// captions are in a shard. The other 179 columns are avatar URLs, music
/// identifiers and view counts; keeping them would cost 9.7 GB of disk to write
/// out a few tens of megabytes of text.
async fn fetch_douyin(hub: &Hub, layout: &DataLayout, limit: Option<usize>) -> Result<usize> {
    let directory = downloads(layout, DOUYIN);
    let partitions = hub.files(DOUYIN.dataset, "data").await?;
    if partitions.is_empty() {
        return Err(Error::Layout {
            dataset: DOUYIN.dataset,
            detail: "the data/ directory holds no partitions".to_owned(),
        });
    }
    let mut sink = DocumentSink::new(layout, DOUYIN, limit)?;
    for partition in &partitions {
        // The sink moves into the blocking worker and back out, because both the
        // parquet read and the shard write are blocking and neither may run on
        // the runtime that is waiting for the next download.
        if sink.full() {
            break;
        }
        let path = hub.download(DOUYIN.dataset, partition, &directory).await?;
        sink = blocking(move || {
            let mut sink = sink;
            let captions = column(&path, DOUYIN_CAPTION, DOUYIN)?;
            for caption in &captions {
                sink.push(None, caption)?;
            }
            std::fs::remove_file(&path).map_err(|source| Error::Read {
                path: path.clone(),
                source,
            })?;
            info!(
                partition = %path.display(),
                captions = captions.len(),
                "projected and removed"
            );
            Ok(sink)
        })
        .await?;
    }
    blocking(move || sink.finish()).await
}

/// Bilibili: one CSV of comments, whose `message` column is the comment body.
async fn fetch_bilibili(hub: &Hub, layout: &DataLayout, limit: Option<usize>) -> Result<usize> {
    let path = single_file(hub, BILIBILI, BILIBILI_FILE, &downloads(layout, BILIBILI)).await?;
    let sink = DocumentSink::new(layout, BILIBILI, limit)?;
    blocking(move || {
        let mut sink = sink;
        let mut reader = csv::ReaderBuilder::new()
            .flexible(true)
            .from_path(&path)
            .map_err(|source| Error::Csv {
                path: path.clone(),
                source,
            })?;
        let headers = reader.headers().map_err(|source| Error::Csv {
            path: path.clone(),
            source,
        })?;
        let message_at = headers
            .iter()
            .position(|name| name == BILIBILI_MESSAGE)
            .ok_or_else(|| Error::Layout {
                dataset: BILIBILI.dataset,
                detail: format!(
                    "expected a {BILIBILI_MESSAGE} column, found {:?}",
                    headers.iter().collect::<Vec<_>>()
                ),
            })?;
        for record in reader.records() {
            if sink.full() {
                break;
            }
            let record = record.map_err(|source| Error::Csv {
                path: path.clone(),
                source,
            })?;
            if let Some(message) = record.get(message_at) {
                sink.push(None, message)?;
            }
        }
        sink.finish()
    })
    .await
}

/// Read one string column out of a parquet file, projecting away everything else.
fn column(path: &Path, name: &str, spec: SourceSpec) -> Result<Vec<String>> {
    let file = std::fs::File::open(path).map_err(|source| Error::Read {
        path: path.to_owned(),
        source,
    })?;
    let frame = ParquetReader::new(file)
        .with_columns(Some(vec![name.to_owned()]))
        .finish()
        .map_err(|source| Error::Layout {
            dataset: spec.dataset,
            detail: format!("{} has no readable {name} column: {source}", path.display()),
        })?;
    let column = frame.column(name).map_err(|source| Error::Layout {
        dataset: spec.dataset,
        detail: format!("{} has no {name} column: {source}", path.display()),
    })?;
    Ok(column
        .str()
        .map_err(|source| Error::Layout {
            dataset: spec.dataset,
            detail: format!("{name} is not a string column: {source}"),
        })?
        .iter()
        .map(|value| value.unwrap_or_default().to_owned())
        .collect())
}

/// Run `work` off the async runtime, because every parser here is blocking.
async fn blocking<T: Send + 'static>(
    work: impl FnOnce() -> Result<T> + Send + 'static,
) -> Result<T> {
    tokio::task::spawn_blocking(work)
        .await
        .map_err(|error| Error::Invariant(format!("a corpus worker panicked: {error}")))?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_moegirl_record_needs_both_of_its_fields() {
        let article: MoegirlArticle =
            serde_json::from_str(r#"{"title":"初音未来","text":"她是虚拟歌手。"}"#)
                .expect("the record parses");
        assert_eq!(article.title, "初音未来");
        assert!(serde_json::from_str::<MoegirlArticle>(r#"{"title":"初音未来"}"#).is_err());
    }

    #[test]
    fn a_sink_stops_at_its_limit_and_hashes_an_unnamed_document() {
        let root = std::env::temp_dir().join(format!("ime-corpus-sink-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let layout = DataLayout::new(&root);
        let mut sink = DocumentSink::new(&layout, BILIBILI, Some(2)).expect("writable");
        sink.push(None, "第一条评论").expect("buffered");
        sink.push(None, "   ").expect("blank records are skipped");
        sink.push(None, "第二条评论").expect("buffered");
        assert!(sink.full());
        sink.push(None, "第三条评论").expect("past the limit");
        assert_eq!(sink.finish().expect("written"), 2);

        let documents: Vec<RawDocument> =
            ime_g2p::shards::read_shards(&layout.documents(), BILIBILI.name)
                .expect("the shards read back");
        assert_eq!(documents.len(), 2);
        assert_eq!(
            documents[0].document_id,
            content_id(BILIBILI.name, "第一条评论", None)
        );
        std::fs::remove_dir_all(&root).expect("the fixture directory is removed");
    }
}
