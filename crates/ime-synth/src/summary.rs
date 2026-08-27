//! What a generation run recorded about itself, so `report` can read it back.
//!
//! Only the samples that passed reach the shards, which means the interesting
//! half of a run -- what was skipped ungrounded, what the endpoint refused, which
//! rule dropped what -- exists nowhere on disk unless it is written down. So the
//! run leaves one JSON file beside its shards, and `synth report` reads it rather
//! than trying to reconstruct a drop breakdown from surviving rows, which is not
//! possible even in principle.

use crate::error::{Error, Result};
use crate::seed::SeedCounts;
use crate::validate::DropCounts;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// The file a run writes its summary into, under the output root.
#[must_use]
pub fn path(root: &Path) -> PathBuf {
    root.join("synth-summary.json")
}

/// How many samples one seed source contributed.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct WrittenBySource {
    /// The seed source's name.
    pub source: String,
    /// Terms of that source that reached a prompt in this run.
    pub terms: usize,
    /// Samples of that source that reached the shards.
    pub written: usize,
}

/// One refusal reason and how many terms hit it.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct RefusalCount {
    /// The reason, as the endpoint or the parser phrased it.
    pub reason: String,
    /// How many terms it accounted for.
    pub terms: usize,
}

/// Everything one generation run is worth knowing afterwards.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct RunSummary {
    /// The `source` column every written sample carries.
    pub source: String,
    /// Which model generated them.
    pub model: String,
    /// How many examples each term was asked for.
    pub per_term: usize,
    /// How many requests were allowed in flight.
    pub concurrency: usize,
    /// How each seed source fared before a single request was made.
    pub seeds: Vec<SeedCounts>,
    /// How many terms were generated from, after the limit was applied.
    pub terms: usize,
    /// How many examples those terms asked for in total.
    pub examples_requested: usize,
    /// How many examples came back parsed.
    pub examples_parsed: usize,
    /// How many terms produced nothing usable at all.
    pub refusals: usize,
    /// The refusal reasons, most frequent first.
    pub refusal_reasons: Vec<RefusalCount>,
    /// Why parsed examples were kept or dropped.
    pub drops: DropCounts,
    /// How many samples reached the shards.
    pub written: usize,
    /// The same, split by seed source.
    pub written_by_source: Vec<WrittenBySource>,
    /// How long the run took, in milliseconds.
    pub milliseconds: u128,
}

impl RunSummary {
    /// Terms generated per minute, which is what a full run is planned against.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "a throughput figure is reported to one decimal place"
    )]
    pub fn terms_per_minute(&self) -> f64 {
        if self.milliseconds == 0 {
            return 0.0;
        }
        self.terms as f64 * 60_000.0 / self.milliseconds as f64
    }

    /// Write the summary beside the shards it describes.
    ///
    /// # Errors
    ///
    /// If the file cannot be written.
    pub fn write(&self, root: &Path) -> Result<()> {
        let path = path(root);
        let rendered = serde_json::to_string_pretty(self).map_err(|source| Error::Summary {
            path: path.clone(),
            source,
        })?;
        std::fs::write(&path, rendered).map_err(|source| Error::Read { path, source })
    }

    /// Read the summary a previous run left under `root`.
    ///
    /// # Errors
    ///
    /// If no run has written one, or if it is malformed.
    pub fn read(root: &Path) -> Result<Self> {
        let path = path(root);
        if !path.exists() {
            return Err(Error::Missing {
                what: "a synthesis run summary",
                path,
                hint: "run `ime-cli synth generate` first",
            });
        }
        let source = std::fs::read_to_string(&path).map_err(|source| Error::Read {
            path: path.clone(),
            source,
        })?;
        serde_json::from_str(&source).map_err(|source| Error::Summary { path, source })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary() -> RunSummary {
        RunSummary {
            source: crate::source::SYNTHETIC.to_owned(),
            model: "gpt-5.6-luna".to_owned(),
            per_term: 5,
            concurrency: 32,
            seeds: vec![SeedCounts {
                source: "wiki-slang".to_owned(),
                loaded: 378,
                grounded: 342,
                skipped_ungrounded: 0,
                skipped_unusable_term: 36,
                skipped_duplicate: 0,
            }],
            terms: 30,
            examples_requested: 150,
            examples_parsed: 145,
            refusals: 1,
            refusal_reasons: vec![RefusalCount {
                reason: "ValueError: got 4 examples where 5 were asked for".to_owned(),
                terms: 1,
            }],
            drops: DropCounts {
                kept: 140,
                missing_term: 5,
                ..DropCounts::default()
            },
            written: 140,
            written_by_source: vec![WrittenBySource {
                source: "wiki-slang".to_owned(),
                terms: 30,
                written: 140,
            }],
            milliseconds: 60_000,
        }
    }

    #[test]
    fn a_summary_survives_a_round_trip_through_its_file() {
        let root = std::env::temp_dir().join(format!("ime-synth-summary-{}", std::process::id()));
        std::fs::create_dir_all(&root).expect("the fixture directory is writable");
        let written = summary();
        written.write(&root).expect("the summary is written");
        assert_eq!(
            RunSummary::read(&root).expect("the summary reads back"),
            written
        );
        std::fs::remove_dir_all(&root).expect("the fixture directory is removed");
    }

    #[test]
    fn throughput_is_terms_per_minute_and_zero_before_any_time_has_passed() {
        assert!((summary().terms_per_minute() - 30.0).abs() < f64::EPSILON);
        let instant = RunSummary {
            milliseconds: 0,
            ..summary()
        };
        assert!(instant.terms_per_minute().abs() < f64::EPSILON);
    }

    #[test]
    fn a_root_with_no_run_in_it_names_the_command_that_would_produce_one() {
        let error = RunSummary::read(Path::new("/nonexistent-synth-root"))
            .expect_err("there is no summary there");
        assert!(
            error.to_string().contains("ime-cli synth generate"),
            "{error}"
        );
    }
}
