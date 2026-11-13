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
    /// The key-level search's terms of that score (#257): the CTC term of
    /// the hypothesis's best cover, its summed transition log-probability,
    /// and the best cover's correction cost. Absent from every other engine's
    /// dump.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ctc: Option<f32>,
    /// See [`DumpedHypothesis::ctc`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transition: Option<f32>,
    /// See [`DumpedHypothesis::ctc`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correction: Option<f32>,
    /// The best cover's cut points: the key offset each character ends at,
    /// the last being the key count. Present with the terms above.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuts: Option<Vec<u32>>,
    /// How many word arcs the best cover contains (#282). Present with the
    /// terms above.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arcs: Option<u32>,
}

impl DumpedHypothesis {
    /// A hypothesis with its score alone, as every engine but the key-level search dumps it.
    #[must_use]
    pub fn scored(text: String, score: f32) -> Self {
        Self {
            text,
            score,
            ctc: None,
            transition: None,
            correction: None,
            cuts: None,
            arcs: None,
        }
    }
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

#[cfg(test)]
mod tests {
    use super::{DumpedHypothesis, parse_dump};

    #[test]
    fn a_key_level_row_carries_its_terms_and_a_plain_one_writes_none() {
        let rows = parse_dump(concat!(
            "{\"record\": 2, \"text\": \"我爱北京\", \"hypotheses\": [{\"text\": \"我爱北京\", ",
            "\"score\": -3.5, \"ctc\": -2.0, \"transition\": -1.25, \"correction\": 0.0, ",
            "\"cuts\": [1, 2, 3, 6], \"arcs\": 1}]}\n",
            "{\"record\": 0, \"text\": \"他\", \"hypotheses\": [{\"text\": \"她\", \"score\": -1.5}]}\n",
            "{\"record\": 1, \"text\": \"こんにちは\", \"hypotheses\": [{\"text\": \"안녕하세요\", ",
            "\"score\": -9.0}]}\n",
        ))
        .expect("every row parses");
        assert_eq!(rows[1].hypotheses[0].ctc, None);
        assert_eq!(rows[1].hypotheses[0].cuts, None);
        assert_eq!(rows[2].hypotheses[0].cuts, Some(vec![1, 2, 3, 6]));
        assert_eq!(rows[2].hypotheses[0].arcs, Some(1));
        assert_eq!(rows[1].hypotheses[0].arcs, None);
        let terms = &rows[2].hypotheses[0];
        assert_eq!(
            (terms.ctc, terms.transition, terms.correction),
            (Some(-2.0), Some(-1.25), Some(0.0))
        );
        assert_eq!(
            rows[0].hypotheses[0],
            DumpedHypothesis::scored("她".to_owned(), -1.5)
        );
        let written = serde_json::to_string(&rows[0].hypotheses[0]).expect("serialises");
        assert_eq!(written, "{\"text\":\"她\",\"score\":-1.5}");
        assert!(
            parse_dump("{\"record\": 0, \"text\": \"他\", \"hypotheses\": [{\"text\": \"她\", \"score\": 0.0, \"cost\": 1.0}]}\n")
                .is_err()
        );
    }
}
