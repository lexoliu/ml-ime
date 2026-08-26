//! Running both annotators over a corpus and recording where they agree.
//!
//! The three outputs are the point of the stage. `annotated` holds every
//! sentence both annotators labelled, agreeing or not. `hard` is the subset they
//! disagreed on -- the hard-polyphone set, kept rather than averaged away.
//! `refused` holds the sentences an annotator could not label at all, counted so
//! that a run which quietly lost half its corpus to a broken endpoint cannot look
//! like a clean run.

use crate::error::{Error, Result};
use crate::outcome::{Annotator, compare};
use crate::shards::{ShardWriter, Shardable, read_shards};
use polars::prelude::{
    Column, DataFrame, IntoSeries as _, ListBooleanChunkedBuilder, ListBuilderTrait as _,
    ListStringChunkedBuilder,
};
use std::path::Path;
use tracing::info;

/// Shard prefix for sentences both annotators handled, agreeing or not.
pub const ANNOTATED: &str = "annotated";

/// Shard prefix for the subset they disagreed on -- the hard-polyphone eval set.
pub const HARD: &str = "hard";

/// Shard prefix for sentences an annotator could not label at all.
pub const REFUSED: &str = "refused";

/// How many rows one shard holds before the next one is started.
pub const ROWS_PER_SHARD: usize = 50_000;

/// One training example: what to type, and what was on screen before it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Sample {
    /// A stable identifier derived from the sample's own content.
    pub id: String,
    /// Which corpus it came from, which the evaluation draw stratifies on.
    pub source: String,
    /// The target sentence.
    pub text: String,
    /// What preceded it in the same document, if anything.
    pub context: Option<String>,
}

/// One row of the dual-annotation output.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct AnnotatedRow {
    /// The sample's identifier.
    pub id: String,
    /// Which corpus it came from.
    pub source: String,
    /// The target sentence.
    pub text: String,
    /// What preceded it, if anything.
    pub context: Option<String>,
    /// The sentence's Han characters, in order.
    pub characters: Vec<String>,
    /// g2pW's tone-numbered syllable for each of them.
    pub g2pw: Vec<String>,
    /// The LLM's tone-numbered syllable for each of them.
    pub llm: Vec<String>,
    /// Per position, whether the two agree once the tone is dropped.
    pub agree: Vec<bool>,
    /// Whether every position agrees.
    pub agree_all: bool,
}

/// One sentence an annotator would not label.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RefusedRow {
    /// The sample's identifier.
    pub id: String,
    /// The target sentence.
    pub text: String,
    /// Which annotator refused.
    pub annotator: String,
    /// Why.
    pub reason: String,
}

/// How an annotation run went, in the three ways it can go.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct AnnotationCounts {
    /// Sentences both annotators read identically end to end.
    pub agreed: usize,
    /// Sentences they read differently somewhere.
    pub disagreed: usize,
    /// Sentence-annotator pairs that produced nothing usable.
    pub refused: usize,
}

impl AnnotationCounts {
    /// Share of fully-annotated sentences both annotators read identically.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "a rate over corpus-sized counts is reported to two decimals"
    )]
    pub fn agreement_rate(&self) -> f64 {
        let total = self.agreed + self.disagreed;
        if total == 0 {
            return 0.0;
        }
        self.agreed as f64 / total as f64
    }
}

/// Run both annotators over `samples` and write the agreed, hard and refused shards.
///
/// The two annotators run concurrently on each batch, because one is a local
/// ONNX session and the other is a network round trip: running them in sequence
/// would make every batch cost the sum rather than the maximum.
///
/// # Errors
///
/// If a shard cannot be written, or if the two annotators are not the g2pW/LLM
/// pair the stored schema is named after.
pub async fn annotate<F: Annotator, S: Annotator>(
    samples: impl IntoIterator<Item = Sample>,
    first: &F,
    second: &S,
    out_dir: &Path,
    batch_size: usize,
) -> Result<AnnotationCounts> {
    if (first.name(), second.name()) != ("g2pw", "llm") {
        return Err(Error::Invariant(format!(
            "the annotated schema has a g2pw and an llm column, not {} and {}",
            first.name(),
            second.name()
        )));
    }
    let mut counts = AnnotationCounts::default();
    let mut annotated = ShardWriter::new(out_dir, ANNOTATED, ROWS_PER_SHARD)?;
    let mut hard = ShardWriter::new(out_dir, HARD, ROWS_PER_SHARD)?;
    let mut refused = ShardWriter::new(out_dir, REFUSED, ROWS_PER_SHARD)?;

    let mut batch: Vec<Sample> = Vec::with_capacity(batch_size.max(1));
    for sample in samples {
        batch.push(sample);
        if batch.len() >= batch_size.max(1) {
            run_batch(
                &batch,
                first,
                second,
                &mut counts,
                &mut annotated,
                &mut hard,
                &mut refused,
            )
            .await?;
            batch.clear();
        }
    }
    if !batch.is_empty() {
        run_batch(
            &batch,
            first,
            second,
            &mut counts,
            &mut annotated,
            &mut hard,
            &mut refused,
        )
        .await?;
    }

    annotated.finish()?;
    hard.finish()?;
    refused.finish()?;
    Ok(counts)
}

async fn run_batch<F: Annotator, S: Annotator>(
    batch: &[Sample],
    first: &F,
    second: &S,
    counts: &mut AnnotationCounts,
    annotated: &mut ShardWriter<AnnotatedRow>,
    hard: &mut ShardWriter<AnnotatedRow>,
    refused: &mut ShardWriter<RefusedRow>,
) -> Result<()> {
    let texts: Vec<String> = batch.iter().map(|sample| sample.text.clone()).collect();
    let (firsts, seconds) = tokio::join!(first.annotate(&texts), second.annotate(&texts));
    if firsts.len() != batch.len() || seconds.len() != batch.len() {
        return Err(Error::Invariant(format!(
            "an annotator returned {} and {} outcomes for {} sentences",
            firsts.len(),
            seconds.len(),
            batch.len()
        )));
    }

    for ((sample, left), right) in batch.iter().zip(firsts).zip(seconds) {
        let mut readings = Vec::with_capacity(2);
        for (name, outcome) in [(first.name(), left), (second.name(), right)] {
            match outcome {
                Ok(reading) => readings.push(reading),
                Err(refusal) => {
                    counts.refused += 1;
                    refused.write(RefusedRow {
                        id: sample.id.clone(),
                        text: sample.text.clone(),
                        annotator: name.to_owned(),
                        reason: refusal.reason,
                    })?;
                }
            }
        }
        let [left, right] = readings.as_slice() else {
            continue;
        };
        let comparison = compare(&sample.text, left, right)?;
        let row = AnnotatedRow {
            id: sample.id.clone(),
            source: sample.source.clone(),
            text: sample.text.clone(),
            context: sample.context.clone(),
            characters: comparison
                .characters()
                .iter()
                .map(ToString::to_string)
                .collect(),
            g2pw: comparison.first().to_vec(),
            llm: comparison.second().to_vec(),
            agree: comparison.agree().to_vec(),
            agree_all: comparison.unanimous(),
        };
        if row.agree_all {
            counts.agreed += 1;
            annotated.write(row)?;
        } else {
            counts.disagreed += 1;
            hard.write(row.clone())?;
            annotated.write(row)?;
        }
    }
    info!(
        agreed = counts.agreed,
        disagreed = counts.disagreed,
        refused = counts.refused,
        "annotated"
    );
    Ok(())
}

/// Read the shards written by [`annotate`].
///
/// # Errors
///
/// If no shard with that prefix exists, or one cannot be read.
pub fn read_annotated(directory: &Path, prefix: &str) -> Result<Vec<AnnotatedRow>> {
    read_shards(directory, prefix)
}

/// Read the samples the corpus stage prepared.
///
/// # Errors
///
/// If the directory holds no shards, or one cannot be read.
pub fn read_samples(directory: &Path) -> Result<Vec<Sample>> {
    read_shards(directory, "*")
}

fn strings(name: &str, values: impl Iterator<Item = String>) -> Column {
    Column::new(name.into(), values.collect::<Vec<_>>())
}

fn string_lists<'a>(name: &str, rows: usize, values: impl Iterator<Item = &'a [String]>) -> Column {
    let mut builder = ListStringChunkedBuilder::new(name.into(), rows, rows * 8);
    for row in values {
        builder.append_values_iter(row.iter().map(String::as_str));
    }
    Column::new(name.into(), builder.finish().into_series())
}

fn bool_lists<'a>(name: &str, rows: usize, values: impl Iterator<Item = &'a [bool]>) -> Column {
    let mut builder = ListBooleanChunkedBuilder::new(name.into(), rows, rows * 8);
    for row in values {
        builder.append_iter(row.iter().map(|flag| Some(*flag)));
    }
    Column::new(name.into(), builder.finish().into_series())
}

fn column_of_strings(frame: &DataFrame, name: &str) -> Result<Vec<String>> {
    Ok(frame
        .column(name)?
        .str()?
        .iter()
        .map(|value| value.unwrap_or_default().to_owned())
        .collect())
}

fn column_of_optional_strings(frame: &DataFrame, name: &str) -> Result<Vec<Option<String>>> {
    Ok(frame
        .column(name)?
        .str()?
        .iter()
        .map(|value| value.map(ToOwned::to_owned))
        .collect())
}

fn column_of_string_lists(frame: &DataFrame, name: &str) -> Result<Vec<Vec<String>>> {
    let column = frame.column(name)?;
    let lists = column.list()?;
    let mut rows = Vec::with_capacity(lists.len());
    for index in 0..lists.len() {
        let series = lists.get_as_series(index).ok_or_else(|| {
            Error::Invariant(format!(
                "a shard holds a null {name} list, which cannot happen"
            ))
        })?;
        rows.push(
            series
                .str()?
                .iter()
                .map(|item| item.unwrap_or_default().to_owned())
                .collect(),
        );
    }
    Ok(rows)
}

fn column_of_bool_lists(frame: &DataFrame, name: &str) -> Result<Vec<Vec<bool>>> {
    let column = frame.column(name)?;
    let lists = column.list()?;
    let mut rows = Vec::with_capacity(lists.len());
    for index in 0..lists.len() {
        let series = lists.get_as_series(index).ok_or_else(|| {
            Error::Invariant(format!(
                "a shard holds a null {name} list, which cannot happen"
            ))
        })?;
        rows.push(
            series
                .bool()?
                .iter()
                .map(|item| item.unwrap_or(false))
                .collect(),
        );
    }
    Ok(rows)
}

impl Shardable for Sample {
    fn frame(rows: &[Self]) -> Result<DataFrame> {
        Ok(DataFrame::new(
            rows.len(),
            vec![
                strings("id", rows.iter().map(|row| row.id.clone())),
                strings("source", rows.iter().map(|row| row.source.clone())),
                strings("text", rows.iter().map(|row| row.text.clone())),
                Column::new(
                    "context".into(),
                    rows.iter()
                        .map(|row| row.context.clone())
                        .collect::<Vec<_>>(),
                ),
            ],
        )?)
    }

    fn from_frame(frame: &DataFrame) -> Result<Vec<Self>> {
        let ids = column_of_strings(frame, "id")?;
        let sources = column_of_strings(frame, "source")?;
        let texts = column_of_strings(frame, "text")?;
        let contexts = column_of_optional_strings(frame, "context")?;
        Ok(ids
            .into_iter()
            .zip(sources)
            .zip(texts)
            .zip(contexts)
            .map(|(((id, source), text), context)| Self {
                id,
                source,
                text,
                context,
            })
            .collect())
    }
}

impl Shardable for AnnotatedRow {
    fn frame(rows: &[Self]) -> Result<DataFrame> {
        let count = rows.len();
        Ok(DataFrame::new(
            rows.len(),
            vec![
                strings("id", rows.iter().map(|row| row.id.clone())),
                strings("source", rows.iter().map(|row| row.source.clone())),
                strings("text", rows.iter().map(|row| row.text.clone())),
                Column::new(
                    "context".into(),
                    rows.iter()
                        .map(|row| row.context.clone())
                        .collect::<Vec<_>>(),
                ),
                string_lists(
                    "characters",
                    count,
                    rows.iter().map(|row| row.characters.as_slice()),
                ),
                string_lists("g2pw", count, rows.iter().map(|row| row.g2pw.as_slice())),
                string_lists("llm", count, rows.iter().map(|row| row.llm.as_slice())),
                bool_lists("agree", count, rows.iter().map(|row| row.agree.as_slice())),
                Column::new(
                    "agree_all".into(),
                    rows.iter().map(|row| row.agree_all).collect::<Vec<_>>(),
                ),
            ],
        )?)
    }

    fn from_frame(frame: &DataFrame) -> Result<Vec<Self>> {
        let ids = column_of_strings(frame, "id")?;
        let sources = column_of_strings(frame, "source")?;
        let texts = column_of_strings(frame, "text")?;
        let contexts = column_of_optional_strings(frame, "context")?;
        let characters = column_of_string_lists(frame, "characters")?;
        let g2pw = column_of_string_lists(frame, "g2pw")?;
        let llm = column_of_string_lists(frame, "llm")?;
        let agree = column_of_bool_lists(frame, "agree")?;
        let agree_all: Vec<bool> = frame
            .column("agree_all")?
            .bool()?
            .iter()
            .map(|value| value.unwrap_or(false))
            .collect();
        let mut rows = Vec::with_capacity(ids.len());
        for index in 0..ids.len() {
            rows.push(Self {
                id: ids[index].clone(),
                source: sources[index].clone(),
                text: texts[index].clone(),
                context: contexts[index].clone(),
                characters: characters[index].clone(),
                g2pw: g2pw[index].clone(),
                llm: llm[index].clone(),
                agree: agree[index].clone(),
                agree_all: agree_all[index],
            });
        }
        Ok(rows)
    }
}

impl Shardable for RefusedRow {
    fn frame(rows: &[Self]) -> Result<DataFrame> {
        Ok(DataFrame::new(
            rows.len(),
            vec![
                strings("id", rows.iter().map(|row| row.id.clone())),
                strings("text", rows.iter().map(|row| row.text.clone())),
                strings("annotator", rows.iter().map(|row| row.annotator.clone())),
                strings("reason", rows.iter().map(|row| row.reason.clone())),
            ],
        )?)
    }

    fn from_frame(frame: &DataFrame) -> Result<Vec<Self>> {
        let ids = column_of_strings(frame, "id")?;
        let texts = column_of_strings(frame, "text")?;
        let annotators = column_of_strings(frame, "annotator")?;
        let reasons = column_of_strings(frame, "reason")?;
        Ok(ids
            .into_iter()
            .zip(texts)
            .zip(annotators)
            .zip(reasons)
            .map(|(((id, text), annotator), reason)| Self {
                id,
                text,
                annotator,
                reason,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::outcome::{Outcome, Reading, Refusal};

    struct Fixed {
        name: &'static str,
        answers: Vec<Outcome>,
    }

    impl Annotator for Fixed {
        fn name(&self) -> &'static str {
            self.name
        }

        async fn annotate(&self, texts: &[String]) -> Vec<Outcome> {
            self.answers.iter().take(texts.len()).cloned().collect()
        }
    }

    fn reading(syllables: &[&str]) -> Reading {
        Reading::new(syllables.iter().map(|s| (*s).to_owned()))
    }

    fn sample(id: &str, text: &str) -> Sample {
        Sample {
            id: id.to_owned(),
            source: "wiki".to_owned(),
            text: text.to_owned(),
            context: None,
        }
    }

    #[tokio::test]
    async fn the_three_shards_split_agreed_hard_and_refused() {
        let directory = std::env::temp_dir().join(format!("ime-g2p-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        let samples = vec![
            sample("a", "中国"),
            sample("b", "重要"),
            sample("c", "你好"),
        ];
        let first = Fixed {
            name: "g2pw",
            answers: vec![
                Ok(reading(&["zhong1", "guo2"])),
                Ok(reading(&["chong2", "yao4"])),
                Err(Refusal::new("g2pw has no reading for '你'")),
            ],
        };
        let second = Fixed {
            name: "llm",
            answers: vec![
                Ok(reading(&["zhong4", "guo2"])),
                Ok(reading(&["zhong4", "yao4"])),
                Ok(reading(&["ni3", "hao3"])),
            ],
        };
        let counts = annotate(samples, &first, &second, &directory, 8)
            .await
            .expect("the run completes");
        assert_eq!(counts.agreed, 1);
        assert_eq!(counts.disagreed, 1);
        assert_eq!(counts.refused, 1);

        let annotated: Vec<AnnotatedRow> =
            read_shards(&directory, ANNOTATED).expect("the annotated shard reads back");
        assert_eq!(annotated.len(), 2);
        let agreed = annotated.iter().find(|row| row.id == "a").expect("row a");
        assert!(agreed.agree_all);
        assert_eq!(agreed.characters, ["中", "国"]);
        assert_eq!(agreed.g2pw, ["zhong1", "guo2"]);
        assert_eq!(agreed.llm, ["zhong4", "guo2"]);

        let hard: Vec<AnnotatedRow> =
            read_shards(&directory, HARD).expect("the hard shard reads back");
        assert_eq!(hard.len(), 1);
        assert_eq!(hard[0].agree, [false, true]);

        let refused: Vec<RefusedRow> =
            read_shards(&directory, REFUSED).expect("the refused shard reads back");
        assert_eq!(refused.len(), 1);
        assert_eq!(refused[0].annotator, "g2pw");
        assert_eq!(refused[0].reason, "g2pw has no reading for '你'");

        std::fs::remove_dir_all(&directory).expect("the fixture directory is removed");
    }

    #[tokio::test]
    async fn annotators_that_are_not_the_stored_pair_are_refused() {
        let error = annotate(
            Vec::new(),
            &Fixed {
                name: "llm",
                answers: Vec::new(),
            },
            &Fixed {
                name: "g2pw",
                answers: Vec::new(),
            },
            &std::env::temp_dir(),
            8,
        )
        .await
        .expect_err("the column names are part of the schema");
        assert!(matches!(error, Error::Invariant(_)));
    }

    #[test]
    fn the_agreement_rate_ignores_refusals() {
        let counts = AnnotationCounts {
            agreed: 3,
            disagreed: 1,
            refused: 10,
        };
        assert!((counts.agreement_rate() - 0.75).abs() < f64::EPSILON);
        assert!((AnnotationCounts::default().agreement_rate()).abs() < f64::EPSILON);
    }
}
