//! The evaluation set on disk.

use crate::EvalError;
use blake2::digest::consts::U8;
use blake2::{Blake2b, Digest as _};
use serde::{Deserialize, Serialize};

/// One thing the engine is asked to get right.
///
/// The context field is here from the first day rather than the day the neural
/// model arrives. An eval set without it cannot be used to measure the one claim
/// the project rests on -- that conditioning on the surrounding text helps -- and
/// retrofitting it later would mean re-collecting every record.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvalRecord {
    /// The keystrokes.
    pub pinyin: String,
    /// What the engine should produce from them.
    pub text: String,
    /// What was already on screen. `null` where there was nothing, or where the
    /// record deliberately measures the engine without it.
    #[serde(default)]
    pub context: Option<String>,
}

impl EvalRecord {
    /// A stable 64-bit digest of everything the record says.
    ///
    /// Stable across processes, machines and runs, which a seeded shuffle is not
    /// once the file is reordered or appended to: a record's half of the set has
    /// to be a property of the record, or a tuning slice and a test slice
    /// silently swap places the first time the eval set grows.
    #[must_use]
    pub fn digest(&self) -> u64 {
        let mut hasher = Blake2b::<U8>::new();
        hasher.update(self.pinyin.as_bytes());
        hasher.update([0]);
        hasher.update(self.text.as_bytes());
        hasher.update([0]);
        hasher.update(self.context.as_deref().unwrap_or_default().as_bytes());
        u64::from_be_bytes(hasher.finalize().into())
    }
}

/// Which part of an evaluation set a run is scored over.
///
/// Tuning the fusion weight and reporting the number it bought have to happen on
/// different records, or the reported number is the weight fitting its own test
/// set. The two slices are named here rather than left to a caller's convention
/// so that both commands mean the same thing by "dev".
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub enum Slice {
    /// Every record.
    #[default]
    All,
    /// The records set aside for tuning.
    Dev,
    /// The records tuning never saw.
    Test,
}

impl Slice {
    /// Whether *record* belongs to this slice, given the share of the set that
    /// is development data.
    ///
    /// # Panics
    ///
    /// If `dev_share` is not a share.
    #[must_use]
    pub fn holds(self, record: &EvalRecord, dev_share: f64) -> bool {
        assert!(
            (0.0..=1.0).contains(&dev_share),
            "dev_share {dev_share} is not a share"
        );
        match self {
            Self::All => true,
            #[expect(
                clippy::cast_precision_loss,
                reason = "the comparison is a share, not a count"
            )]
            Self::Dev => (record.digest() as f64) / (u64::MAX as f64) < dev_share,
            Self::Test => !Self::Dev.holds(record, dev_share),
        }
    }
}

/// A parsed evaluation set.
///
/// The file is JSON Lines -- one record per line -- so a set can be appended to,
/// sharded and diffed, and one malformed record names its own line number.
///
/// ```text
/// {"pinyin": "zhongguorenmin", "text": "中国人民", "context": null}
/// {"pinyin": "yinhang", "text": "银行", "context": "我去中国人民"}
/// ```
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct EvalSet {
    records: Vec<EvalRecord>,
}

impl EvalSet {
    /// Parse a JSON Lines evaluation set. Blank lines are skipped.
    ///
    /// # Errors
    ///
    /// If a line is not a record, if a record has an empty `pinyin` or `text`,
    /// or if the set holds no records at all.
    pub fn parse(source: &str) -> Result<Self, EvalError> {
        let mut records = Vec::new();
        for (index, raw) in source.lines().enumerate() {
            let line = index + 1;
            if raw.trim().is_empty() {
                continue;
            }
            let record: EvalRecord = serde_json::from_str(raw)
                .map_err(|source| EvalError::Malformed { line, source })?;
            if record.pinyin.is_empty() {
                return Err(EvalError::EmptyField {
                    line,
                    field: "pinyin",
                });
            }
            if record.text.is_empty() {
                return Err(EvalError::EmptyField {
                    line,
                    field: "text",
                });
            }
            records.push(record);
        }
        if records.is_empty() {
            return Err(EvalError::Empty);
        }
        Ok(Self { records })
    }

    /// The records, in file order.
    #[must_use]
    pub fn records(&self) -> &[EvalRecord] {
        &self.records
    }

    /// How many records the set holds. Never zero.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the set is empty. Never true for a parsed set.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// How many records carry context.
    #[must_use]
    pub fn with_context(&self) -> usize {
        self.records
            .iter()
            .filter(|record| record.context.is_some())
            .count()
    }
}
