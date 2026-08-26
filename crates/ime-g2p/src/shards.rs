//! Fixed-size parquet shard writing.
//!
//! Every stage of the pipeline emits the same shape -- a long stream of small
//! rows that has to survive a kernel restart -- so they share one writer rather
//! than each growing its own buffering logic. The layout matches the Python
//! pipeline's, so either implementation can read what the other wrote.

use crate::error::Result;
use polars::prelude::{DataFrame, ParquetReader, ParquetWriter, SerReader as _};
use std::path::{Path, PathBuf};
use tracing::debug;

/// A row shape that knows how to become a parquet frame and how to come back.
pub trait Shardable: Sized {
    /// Turn a buffer of rows into the frame one shard holds.
    ///
    /// # Errors
    ///
    /// If a column cannot be built, which means the rows disagree with the schema.
    fn frame(rows: &[Self]) -> Result<DataFrame>;

    /// Read a shard's frame back into rows.
    ///
    /// # Errors
    ///
    /// If the frame is missing a column or has one of the wrong type.
    fn from_frame(frame: &DataFrame) -> Result<Vec<Self>>;
}

/// Buffer rows and flush them as `<prefix>-00000.parquet` under a directory.
///
/// [`finish`](Self::finish) is what writes the last, partial shard, so an
/// interrupted run leaves whole readable shards behind rather than one truncated
/// file -- the same contract the Python context manager has.
#[derive(Debug)]
pub struct ShardWriter<R> {
    directory: PathBuf,
    prefix: String,
    rows_per_shard: usize,
    buffer: Vec<R>,
    shards: usize,
    rows_written: usize,
}

impl<R: Shardable> ShardWriter<R> {
    /// Open a writer for `<prefix>-*.parquet` under `directory`.
    ///
    /// # Errors
    ///
    /// If the directory cannot be created.
    pub fn new(directory: &Path, prefix: &str, rows_per_shard: usize) -> Result<Self> {
        std::fs::create_dir_all(directory).map_err(|source| crate::Error::Read {
            path: directory.to_owned(),
            source,
        })?;
        Ok(Self {
            directory: directory.to_owned(),
            prefix: prefix.to_owned(),
            rows_per_shard: rows_per_shard.max(1),
            buffer: Vec::new(),
            shards: 0,
            rows_written: 0,
        })
    }

    /// Queue one row, flushing a shard once the buffer is full.
    ///
    /// # Errors
    ///
    /// If the flush it triggers fails.
    pub fn write(&mut self, row: R) -> Result<()> {
        self.buffer.push(row);
        if self.buffer.len() >= self.rows_per_shard {
            self.flush()?;
        }
        Ok(())
    }

    /// Write the buffered rows out as one shard.
    ///
    /// # Errors
    ///
    /// If the frame cannot be built or the file cannot be written.
    pub fn flush(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let path = self
            .directory
            .join(format!("{}-{:05}.parquet", self.prefix, self.shards));
        let mut frame = R::frame(&self.buffer)?;
        let file = std::fs::File::create(&path).map_err(|source| crate::Error::Read {
            path: path.clone(),
            source,
        })?;
        ParquetWriter::new(file).finish(&mut frame)?;
        self.rows_written += self.buffer.len();
        self.shards += 1;
        debug!(path = %path.display(), rows = self.buffer.len(), "shard written");
        self.buffer.clear();
        Ok(())
    }

    /// Flush the last shard and report how many rows were written in total.
    ///
    /// # Errors
    ///
    /// If the final flush fails.
    pub fn finish(mut self) -> Result<usize> {
        self.flush()?;
        Ok(self.rows_written)
    }
}

/// Every `<prefix>-*.parquet` shard in `directory`, in write order.
///
/// Empty when the stage wrote none, which is a legitimate outcome for the
/// optional outputs -- an annotation run that nothing refused writes no refusal
/// shard -- so the caller decides whether emptiness is an error.
///
/// # Errors
///
/// If the directory cannot be listed. A directory that does not exist yields no
/// shards rather than an error.
pub fn shard_paths(directory: &Path, prefix: &str) -> Result<Vec<PathBuf>> {
    if !directory.exists() {
        return Ok(Vec::new());
    }
    let entries = std::fs::read_dir(directory).map_err(|source| crate::Error::Read {
        path: directory.to_owned(),
        source,
    })?;
    let mut paths: Vec<PathBuf> = Vec::new();
    for entry in entries {
        let path = entry
            .map_err(|source| crate::Error::Read {
                path: directory.to_owned(),
                source,
            })?
            .path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let matches = name.ends_with(".parquet")
            && (prefix == "*" || name.starts_with(&format!("{prefix}-")));
        if matches {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

/// Read one parquet file into a frame.
///
/// # Errors
///
/// If the file cannot be opened or is not parquet.
pub fn read_frame(path: &Path) -> Result<DataFrame> {
    let file = std::fs::File::open(path).map_err(|source| crate::Error::Read {
        path: path.to_owned(),
        source,
    })?;
    Ok(ParquetReader::new(file).finish()?)
}

/// Read every `<prefix>-*.parquet` shard in `directory` as one list of rows.
///
/// # Errors
///
/// If no shard matches, or if one of them cannot be read.
pub fn read_shards<R: Shardable>(directory: &Path, prefix: &str) -> Result<Vec<R>> {
    let paths = shard_paths(directory, prefix)?;
    if paths.is_empty() {
        return Err(crate::Error::Missing {
            what: "annotation shards",
            path: directory.join(format!("{prefix}-*.parquet")),
            hint: "run `ime-cli g2p annotate` first",
        });
    }
    let mut rows = Vec::new();
    for path in &paths {
        rows.extend(R::from_frame(&read_frame(path)?)?);
    }
    Ok(rows)
}
