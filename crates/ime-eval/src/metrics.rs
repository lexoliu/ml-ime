//! What is measured, and the interface through which both routes are measured.

use crate::record::EvalSet;
use askama::Template;
use std::num::NonZeroUsize;

/// One thing to decode.
///
/// Context is part of the request rather than something an engine reaches for,
/// so that the n-gram baseline and the neural model are handed exactly the same
/// information and only one of them chooses to use it.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Request<'a> {
    /// The keystrokes.
    pub pinyin: &'a str,
    /// The text already on screen, if any.
    pub context: Option<&'a str>,
    /// How many hypotheses to return, best first.
    pub top_k: NonZeroUsize,
}

/// Anything that turns keystrokes into ranked sentences.
pub trait Hypothesize {
    /// What can go wrong inside the engine.
    type Error;

    /// Decode *request*, best hypothesis first, at most `request.top_k` of them.
    ///
    /// # Errors
    ///
    /// Whatever the engine reports; the harness attributes nothing to a failure
    /// and stops.
    fn hypotheses(&self, request: &Request<'_>) -> Result<Vec<String>, Self::Error>;
}

/// What an evaluation run measured.
///
/// Counts as well as rates: a rate on its own hides how many records it was
/// computed over, and an eval set that quietly shrank is exactly the kind of
/// thing that makes two runs look comparable when they are not.
#[derive(Clone, PartialEq, Eq, Debug, Template)]
#[template(path = "report.txt", ext = "txt")]
pub struct Report {
    top_k: usize,
    records: usize,
    top1_hits: usize,
    topk_hits: usize,
    characters: usize,
    character_hits: usize,
    /// Accumulated reciprocal ranks, scaled by [`RANK_SCALE`] so the report
    /// stays exactly comparable between runs. Summing floats in hash order is
    /// how two identical runs come to disagree in the last digit.
    reciprocal_ranks: u64,
    unanswered: usize,
}

/// The fixed-point scale the reciprocal ranks accumulate in.
const RANK_SCALE: u64 = 1 << 20;

impl Report {
    /// An empty report that will rank the first `top_k` hypotheses of each
    /// record.
    #[must_use]
    pub const fn new(top_k: NonZeroUsize) -> Self {
        Self {
            top_k: top_k.get(),
            records: 0,
            top1_hits: 0,
            topk_hits: 0,
            characters: 0,
            character_hits: 0,
            reciprocal_ranks: 0,
            unanswered: 0,
        }
    }

    /// Fold one decoded record into the report.
    ///
    /// Character accuracy is positional: hypothesis character *i* against
    /// expected character *i*, over the length of the expected text. Output
    /// length is the syllable count, so a hypothesis is usually the right length
    /// -- but only usually, because a wrong segmentation is a wrong length, and
    /// scoring those against the shorter of the two would reward them for it.
    pub fn observe(&mut self, expected: &str, hypotheses: &[String]) {
        self.records += 1;
        self.characters += expected.chars().count();
        let Some(top) = hypotheses.first() else {
            self.unanswered += 1;
            return;
        };
        self.character_hits += top
            .chars()
            .zip(expected.chars())
            .filter(|(found, want)| found == want)
            .count();
        if top == expected {
            self.top1_hits += 1;
        }
        if let Some(rank) = hypotheses
            .iter()
            .take(self.top_k)
            .position(|found| found == expected)
        {
            self.topk_hits += 1;
            self.reciprocal_ranks += RANK_SCALE / (rank as u64 + 1);
        }
    }

    /// How many hypotheses per record the top-k metrics look at.
    #[must_use]
    pub const fn top_k(&self) -> usize {
        self.top_k
    }

    /// How many records were scored.
    #[must_use]
    pub const fn records(&self) -> usize {
        self.records
    }

    /// How many records the engine got right on its first try.
    #[must_use]
    pub const fn top1_hits(&self) -> usize {
        self.top1_hits
    }

    /// How many records the engine got right within its top *k*.
    #[must_use]
    pub const fn topk_hits(&self) -> usize {
        self.topk_hits
    }

    /// How many characters the expected texts hold in total.
    #[must_use]
    pub const fn characters(&self) -> usize {
        self.characters
    }

    /// How many of them the top hypotheses got right, position by position.
    #[must_use]
    pub const fn character_hits(&self) -> usize {
        self.character_hits
    }

    /// How many records the engine returned nothing for.
    #[must_use]
    pub const fn unanswered(&self) -> usize {
        self.unanswered
    }

    /// Fraction of records whose top hypothesis was exactly right.
    #[must_use]
    pub fn top1_accuracy(&self) -> f64 {
        rate(self.top1_hits, self.records)
    }

    /// Fraction of records whose top *k* hypotheses contained the right answer.
    #[must_use]
    pub fn topk_accuracy(&self) -> f64 {
        rate(self.topk_hits, self.records)
    }

    /// Fraction of expected characters the top hypotheses placed correctly.
    #[must_use]
    pub fn character_accuracy(&self) -> f64 {
        rate(self.character_hits, self.characters)
    }

    /// Mean reciprocal rank of the right answer within the top *k*, counting a
    /// miss as zero.
    #[must_use]
    pub fn mean_reciprocal_rank(&self) -> f64 {
        if self.records == 0 {
            return 0.0;
        }
        #[expect(clippy::cast_precision_loss, reason = "counts stay far below 2^53")]
        let total = self.reciprocal_ranks as f64 / RANK_SCALE as f64;
        #[expect(clippy::cast_precision_loss, reason = "counts stay far below 2^53")]
        let records = self.records as f64;
        total / records
    }
}

/// A ratio, or zero where there was nothing to divide.
#[expect(clippy::cast_precision_loss, reason = "counts stay far below 2^53")]
fn rate(hits: usize, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    hits as f64 / total as f64
}

/// Run *engine* over *set* and report what it got right.
///
/// # Errors
///
/// Whatever the engine reports on the first record it cannot decode.
pub fn evaluate<H>(set: &EvalSet, engine: &H, top_k: NonZeroUsize) -> Result<Report, H::Error>
where
    H: Hypothesize,
{
    let mut report = Report::new(top_k);
    for record in set.records() {
        let hypotheses = engine.hypotheses(&Request {
            pinyin: &record.pinyin,
            context: record.context.as_deref(),
            top_k,
        })?;
        report.observe(&record.text, &hypotheses);
    }
    Ok(report)
}
