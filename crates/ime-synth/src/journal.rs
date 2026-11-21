//! Append-only prompt journal for resumable synthesis runs.
//!
//! A scale synthesis batch sends tens of thousands of prompts, and stopping a
//! multi-hour run must neither lose the replies already paid for nor change the
//! shards the finished run writes. Every completed prompt appends one JSON line
//! here as soon as its request lands; a restarted run skips every prompt id
//! already recorded (unless asked to retry refusals) and then assembles the
//! shards in deterministic seed order from the journal.

use crate::error::{Error, Result};
use crate::llm::{Example, Reply};
use crate::seed::Seed;
use ime_corpus::text::content_id;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use tracing::warn;

/// File name of the append-only synthesis journal under an output root.
pub const JOURNAL_FILE: &str = "synth-journal.jsonl";

/// Path to the journal file under `root`.
#[must_use]
pub fn path(root: &Path) -> PathBuf {
    root.join(JOURNAL_FILE)
}

/// Deterministic identifier of one prompt: its output source tag, seed mode,
/// lexicon/grounding attribution, requested count, term, foil and explanation.
#[must_use]
pub fn prompt_id(source: &str, seed: &Seed, per_term: usize) -> String {
    let header = format!(
        "{source}\x1f{}\x1f{}\x1f{}\x1f{per_term}",
        seed.mode_tag(),
        seed.source.name,
        seed.grounding.name,
    );
    let body = match &seed.foil {
        Some(foil) => format!("{foil}\x1f{}", seed.explanation),
        None => seed.explanation.clone(),
    };
    content_id(&header, &seed.term, Some(&body))
}

/// What one prompt produced.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum PromptOutcome {
    /// The model returned `examples.len() == per_term` parsed turn pairs.
    Examples {
        /// Raw turn pairs before normalisation and validation.
        examples: Vec<Example>,
    },
    /// The model explicitly abstained on an ungrounded or contrastive prompt.
    Abstained,
    /// The endpoint failed or returned an unparseable reply after retry.
    Refused {
        /// Diagnostic refusal reason.
        reason: String,
    },
}

impl From<std::result::Result<Reply, String>> for PromptOutcome {
    fn from(reply: std::result::Result<Reply, String>) -> Self {
        match reply {
            Ok(Reply::Examples(examples)) => Self::Examples { examples },
            Ok(Reply::Abstained) => Self::Abstained,
            Err(reason) => Self::Refused { reason },
        }
    }
}

/// One line in `synth-journal.jsonl`.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct JournalEntry {
    /// Deterministic prompt identifier ([`prompt_id`]).
    pub id: String,
    /// Seed term the prompt asked about, for human inspection.
    pub term: String,
    /// Outcome recorded for this prompt.
    #[serde(flatten)]
    pub outcome: PromptOutcome,
}

/// An open append-only prompt journal backed by `synth-journal.jsonl`.
#[derive(Debug)]
pub struct Journal {
    path: PathBuf,
    file: File,
    entries: HashMap<String, PromptOutcome>,
}

impl Journal {
    /// Open or create the journal under `root`, replaying every completed line.
    ///
    /// If a previous process was killed mid-write leaving an unterminated final
    /// fragment without `\n`, that partial tail is truncated back to the last
    /// newline with a warning; any complete line that fails to parse is refused.
    ///
    /// # Errors
    ///
    /// If `root` cannot be created, the journal file cannot be read or opened
    /// for append, or a complete line is malformed JSON.
    pub fn open(root: &Path) -> Result<Self> {
        std::fs::create_dir_all(root).map_err(|source| Error::Read {
            path: root.to_owned(),
            source,
        })?;
        let journal_path = path(root);
        let mut entries = HashMap::new();
        if journal_path.exists() {
            let mut bytes = std::fs::read(&journal_path).map_err(|source| Error::Read {
                path: journal_path.clone(),
                source,
            })?;
            if !bytes.is_empty() && bytes.last() != Some(&b'\n') {
                let keep = bytes
                    .iter()
                    .rposition(|&byte| byte == b'\n')
                    .map_or(0, |index| index + 1);
                warn!(
                    path = %journal_path.display(),
                    dropped_bytes = bytes.len() - keep,
                    "truncating incomplete trailing journal fragment from an interrupted write"
                );
                bytes.truncate(keep);
                std::fs::write(&journal_path, &bytes).map_err(|source| Error::Read {
                    path: journal_path.clone(),
                    source,
                })?;
            }
            let text = std::str::from_utf8(&bytes).map_err(|error| Error::Journal {
                path: journal_path.clone(),
                line: 1,
                reason: error.to_string(),
            })?;
            for (index, line) in text.lines().enumerate() {
                if line.trim().is_empty() {
                    continue;
                }
                let entry: JournalEntry =
                    serde_json::from_str(line).map_err(|error| Error::Journal {
                        path: journal_path.clone(),
                        line: index + 1,
                        reason: error.to_string(),
                    })?;
                entries.insert(entry.id, entry.outcome);
            }
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&journal_path)
            .map_err(|source| Error::Read {
                path: journal_path.clone(),
                source,
            })?;
        Ok(Self {
            path: journal_path,
            file,
            entries,
        })
    }

    /// Number of distinct prompt ids recorded so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the journal holds no recorded prompt outcomes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Look up a previously recorded outcome by prompt id.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&PromptOutcome> {
        self.entries.get(id)
    }

    /// Append one prompt's outcome to disk and flush before recording it in memory.
    ///
    /// # Errors
    ///
    /// If serialising or flushing the journal line fails.
    pub fn record(&mut self, entry: &JournalEntry) -> Result<()> {
        let mut line = serde_json::to_string(entry).map_err(|error| Error::Journal {
            path: self.path.clone(),
            line: self.entries.len() + 1,
            reason: error.to_string(),
        })?;
        line.push('\n');
        self.file
            .write_all(line.as_bytes())
            .and_then(|()| self.file.flush())
            .map_err(|source| Error::Read {
                path: self.path.clone(),
                source,
            })?;
        self.entries.insert(entry.id.clone(), entry.outcome.clone());
        Ok(())
    }
}
