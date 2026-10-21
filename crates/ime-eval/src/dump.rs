//! The beam a run decoded, as it goes to disk and comes back.
//!
//! `fused-eval --dump` writes one of these files per reported section, and the
//! end-to-end evaluation writes the same shape, so everything that reads a
//! system's answers after the fact -- the reranker, the generator, the
//! ambiguity slice -- reads every system the same way.

use crate::EvalError;
use serde::{Deserialize, Serialize};

/// One line of a dump file: the record's expected text and the beam it
/// decoded into, best first.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DumpRow {
    /// The record's zero-based line in the evaluation set, blank lines
    /// excluded -- the index [`crate::EvalSet::records`] gives it.
    pub record: usize,
    /// What the record should have produced.
    pub text: String,
    /// The hypotheses, best first; empty when the engine answered nothing.
    pub hypotheses: Vec<DumpedHypothesis>,
}

/// One hypothesis as the dump records it.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DumpedHypothesis {
    /// The sentence.
    pub text: String,
    /// The score it was ranked by; higher is better.
    pub score: f32,
}

/// Parse a dump file's JSON Lines. Blank lines are skipped.
///
/// # Errors
///
/// If a line is not a row, or two rows name the same record -- two runs
/// appended to one file.
pub fn parse_dump(source: &str) -> Result<Vec<DumpRow>, EvalError> {
    let mut rows: Vec<DumpRow> = Vec::new();
    for (index, raw) in source.lines().enumerate() {
        if raw.trim().is_empty() {
            continue;
        }
        let row = serde_json::from_str(raw).map_err(|source| EvalError::MalformedDump {
            line: index + 1,
            source,
        })?;
        rows.push(row);
    }
    rows.sort_unstable_by_key(|row| row.record);
    if let Some(pair) = rows
        .windows(2)
        .find(|pair| pair[0].record == pair[1].record)
    {
        return Err(EvalError::DuplicateRow {
            record: pair[0].record,
        });
    }
    Ok(rows)
}
