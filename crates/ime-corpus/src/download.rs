//! Resumable downloads off the Hugging Face Hub, and the file listing that drives them.
//!
//! Nothing here knows what a corpus is. It knows that a 350 MB partition over a
//! home connection will occasionally be cut off half way, and that re-downloading
//! the first 300 MB of it is the difference between a run that finishes and a run
//! that is abandoned. So every download lands in a `.part` file, every retry
//! resumes from however much of it already exists, and the `.part` is renamed to
//! its real name only once the byte count matches what the Hub advertised.
//!
//! The Hub's `datasets/<id>/tree` endpoint is what a source asks for its file
//! list, rather than any of them hard-coding a partition count: `bendavidsteel/douyin`
//! is 111 partitions today and a source that assumed so would break silently the
//! week it becomes 112.

use crate::error::{Error, Result};
use futures::StreamExt as _;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::io::AsyncWriteExt as _;
use tracing::{info, warn};

/// How many times one file is re-attempted before the run gives up on it.
const MAX_ATTEMPTS: u32 = 5;

/// How long the first retry waits; each subsequent one waits twice as long.
const FIRST_BACKOFF: Duration = Duration::from_secs(2);

/// One file in a dataset repository, as the Hub's tree endpoint describes it.
#[derive(Clone, PartialEq, Eq, Debug, serde::Deserialize)]
pub struct RepoFile {
    /// Whether the entry is a `file` or a `directory`.
    #[serde(rename = "type")]
    pub kind: String,
    /// The path within the repository, which is also the resolve URL's tail.
    pub path: String,
    /// The size the Hub reports for a plain file.
    #[serde(default)]
    pub size: u64,
    /// The size the Hub reports for a file stored in LFS, which is the real one.
    #[serde(default)]
    pub lfs: Option<LfsInfo>,
}

/// The LFS pointer's payload, of which only the size matters here.
#[derive(Clone, PartialEq, Eq, Debug, serde::Deserialize)]
pub struct LfsInfo {
    /// The size of the file the pointer stands for.
    pub size: u64,
}

impl RepoFile {
    /// The number of bytes a complete download of this file has.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.lfs.as_ref().map_or(self.size, |lfs| lfs.size)
    }
}

/// A client for one dataset repository on the Hugging Face Hub.
#[derive(Clone, Debug)]
pub struct Hub {
    client: reqwest::Client,
}

impl Hub {
    /// Build a client.
    ///
    /// # Errors
    ///
    /// If the HTTP client cannot be constructed, which means the platform has no
    /// usable TLS backend.
    pub fn new() -> Result<Self> {
        let client = reqwest::Client::builder()
            .user_agent(concat!("ime-corpus/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|error| Error::Download {
                url: "https://huggingface.co".to_owned(),
                message: error.to_string(),
            })?;
        Ok(Self { client })
    }

    /// The files under `prefix` in `dataset`, sorted by path.
    ///
    /// `prefix` is `""` for the repository root. The listing is sorted so that a
    /// `--limit`ed run takes the same partitions every time.
    ///
    /// # Errors
    ///
    /// If the endpoint refuses, or answers with something that is not a file list.
    pub async fn files(&self, dataset: &str, prefix: &str) -> Result<Vec<RepoFile>> {
        let url = format!("https://huggingface.co/api/datasets/{dataset}/tree/main/{prefix}");
        let body = self
            .client
            .get(&url)
            .send()
            .await
            .and_then(reqwest::Response::error_for_status)
            .map_err(|error| Error::Download {
                url: url.clone(),
                message: error.to_string(),
            })?
            .text()
            .await
            .map_err(|error| Error::Download {
                url: url.clone(),
                message: error.to_string(),
            })?;
        let mut files: Vec<RepoFile> =
            serde_json::from_str(&body).map_err(|error| Error::Download {
                url: url.clone(),
                message: format!("the tree listing is not a file list: {error}"),
            })?;
        files.retain(|entry| entry.kind == "file");
        files.sort_by(|left, right| left.path.cmp(&right.path));
        Ok(files)
    }

    /// Download `path` from `dataset` into `directory`, resuming and retrying.
    ///
    /// Returns the local path. A file already there at its full length is left
    /// alone, so re-running a fetch costs one `stat` rather than another download.
    ///
    /// # Errors
    ///
    /// If every attempt fails, or if the finished file is not the length the Hub
    /// advertised.
    pub async fn download(
        &self,
        dataset: &str,
        file: &RepoFile,
        directory: &Path,
    ) -> Result<PathBuf> {
        let name = file.path.rsplit('/').next().ok_or_else(|| Error::Layout {
            dataset: "hugging face",
            detail: format!("{:?} is not a usable file path", file.path),
        })?;
        tokio::fs::create_dir_all(directory)
            .await
            .map_err(|source| Error::Read {
                path: directory.to_owned(),
                source,
            })?;
        let destination = directory.join(name);
        let expected = file.bytes();
        if matches!(tokio::fs::metadata(&destination).await, Ok(meta) if meta.len() == expected) {
            info!(path = %destination.display(), bytes = expected, "already downloaded");
            return Ok(destination);
        }
        let partial = destination.with_extension("part");
        let url = format!(
            "https://huggingface.co/datasets/{dataset}/resolve/main/{}",
            file.path
        );

        let mut backoff = FIRST_BACKOFF;
        let mut last = String::new();
        for attempt in 1..=MAX_ATTEMPTS {
            match self.attempt(&url, &partial, expected).await {
                Ok(()) => {
                    tokio::fs::rename(&partial, &destination)
                        .await
                        .map_err(|source| Error::Read {
                            path: destination.clone(),
                            source,
                        })?;
                    info!(path = %destination.display(), bytes = expected, "downloaded");
                    return Ok(destination);
                }
                Err(error) => {
                    last = error.to_string();
                    warn!(url = %url, attempt, error = %last, "download attempt failed, retrying");
                    if attempt < MAX_ATTEMPTS {
                        tokio::time::sleep(backoff).await;
                        backoff *= 2;
                    }
                }
            }
        }
        Err(Error::Download {
            url,
            message: format!("gave up after {MAX_ATTEMPTS} attempts: {last}"),
        })
    }

    /// One attempt at completing `partial`, resuming from whatever it already holds.
    async fn attempt(&self, url: &str, partial: &Path, expected: u64) -> Result<()> {
        let have = match tokio::fs::metadata(partial).await {
            Ok(meta) => meta.len(),
            Err(_) => 0,
        };
        if have > expected {
            tokio::fs::remove_file(partial)
                .await
                .map_err(|source| Error::Read {
                    path: partial.to_owned(),
                    source,
                })?;
            return Err(Error::Download {
                url: url.to_owned(),
                message: format!(
                    "the partial file held {have} bytes but the file is only {expected}; \
                     it has been removed, so the next attempt starts over"
                ),
            });
        }
        if have == expected && expected > 0 {
            return Ok(());
        }

        let mut request = self.client.get(url);
        if have > 0 {
            request = request.header(reqwest::header::RANGE, format!("bytes={have}-"));
        }
        let response = request
            .send()
            .await
            .and_then(reqwest::Response::error_for_status)
            .map_err(|error| Error::Download {
                url: url.to_owned(),
                message: error.to_string(),
            })?;
        let resuming = response.status() == reqwest::StatusCode::PARTIAL_CONTENT;
        if have > 0 && !resuming {
            // The server ignored the range, so the body starts from zero again.
            tokio::fs::remove_file(partial)
                .await
                .map_err(|source| Error::Read {
                    path: partial.to_owned(),
                    source,
                })?;
        }
        let mut handle = tokio::fs::OpenOptions::new()
            .create(true)
            .append(resuming && have > 0)
            .write(true)
            .truncate(!(resuming && have > 0))
            .open(partial)
            .await
            .map_err(|source| Error::Read {
                path: partial.to_owned(),
                source,
            })?;
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|error| Error::Download {
                url: url.to_owned(),
                message: error.to_string(),
            })?;
            handle
                .write_all(&chunk)
                .await
                .map_err(|source| Error::Read {
                    path: partial.to_owned(),
                    source,
                })?;
        }
        handle.flush().await.map_err(|source| Error::Read {
            path: partial.to_owned(),
            source,
        })?;
        drop(handle);

        let written = tokio::fs::metadata(partial)
            .await
            .map_err(|source| Error::Read {
                path: partial.to_owned(),
                source,
            })?
            .len();
        if written == expected {
            return Ok(());
        }
        Err(Error::Download {
            url: url.to_owned(),
            message: format!("got {written} bytes, expected {expected}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_lfs_size_wins_over_the_pointer_size() {
        let pointer = RepoFile {
            kind: "file".to_owned(),
            path: "data/partition_7035.parquet.zstd".to_owned(),
            size: 135,
            lfs: Some(LfsInfo { size: 455_522 }),
        };
        assert_eq!(pointer.bytes(), 455_522);
        let plain = RepoFile {
            kind: "file".to_owned(),
            path: "README.md".to_owned(),
            size: 5592,
            lfs: None,
        };
        assert_eq!(plain.bytes(), 5592);
    }

    #[test]
    fn a_tree_listing_parses_into_repository_files() {
        let listing = r#"[
            {"type":"file","oid":"a","size":135,"lfs":{"oid":"b","size":455522,"pointerSize":135},
             "path":"data/partition_7035.parquet.zstd"},
            {"type":"file","oid":"c","size":5592,"path":"README.md"}
        ]"#;
        let files: Vec<RepoFile> = serde_json::from_str(listing).expect("the listing parses");
        assert_eq!(files.len(), 2);
        assert_eq!(files[0].bytes(), 455_522);
        assert_eq!(files[1].bytes(), 5592);
    }
}
