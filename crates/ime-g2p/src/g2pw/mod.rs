//! The g2pW annotator: a BERT polyphone disambiguator behind an ONNX session.
//!
//! The model is trained on traditional Chinese and ships its own
//! simplified-to-traditional table, which is why every sentence is converted
//! before it is read -- the corpus is normalised to simplified. The conversion is
//! character for character, so the readings still line up with the sentence the
//! caller handed in.
//!
//! Inference is a plain batched loop. The Python this replaces ran the same loop
//! behind a `DataLoader`, which treats `num_workers=0` as "use the default" and
//! silently spawned worker processes that deadlock at interpreter shutdown on
//! macOS; there is nothing here for that bug to happen to.

pub mod features;
pub mod tables;
pub mod tokenize;

use crate::error::{Error, Result};
use crate::outcome::{Annotator, Outcome, Reading, Refusal};
use crate::text::is_han;
use features::{Features, features};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use std::path::{Path, PathBuf};
use tables::Tables;
use tokenizers::Tokenizer;
use tokio::sync::{mpsc, oneshot};
use tracing::{info, warn};

/// How many queries are handed to the network at once, matching the batch size
/// the Python annotator asks for.
pub const DEFAULT_BATCH_SIZE: usize = 32;

/// Where the converter caches its ONNX model when nothing else is asked for.
///
/// # Errors
///
/// If the process has no home directory to cache under.
pub fn default_model_dir() -> Result<PathBuf> {
    Ok(home()?.join(".cache").join("mlime").join("G2PWModel"))
}

/// Where `transformers` leaves the `bert-base-chinese` tokenizer it downloads.
///
/// The snapshot hash is read from the cache's `refs/main` rather than hardcoded,
/// so a re-download that moves the snapshot does not silently fall back to a
/// stale vocabulary.
///
/// # Errors
///
/// If the cache, its `refs/main`, or the snapshot's `tokenizer.json` is absent.
pub fn default_tokenizer_path() -> Result<PathBuf> {
    let cache = home()?
        .join(".cache")
        .join("huggingface")
        .join("hub")
        .join("models--bert-base-chinese");
    let reference = cache.join("refs").join("main");
    let revision = std::fs::read_to_string(&reference)
        .map_err(|source| {
            if source.kind() == std::io::ErrorKind::NotFound {
                Error::Missing {
                    what: "the bert-base-chinese tokenizer cache",
                    path: reference.clone(),
                    hint: "run the Python pipeline once, or pass --tokenizer pointing at a tokenizer.json",
                }
            } else {
                Error::Read {
                    path: reference.clone(),
                    source,
                }
            }
        })?;
    let path = cache
        .join("snapshots")
        .join(revision.trim())
        .join("tokenizer.json");
    if !path.exists() {
        return Err(Error::Missing {
            what: "the bert-base-chinese tokenizer",
            path,
            hint: "pass --tokenizer pointing at a tokenizer.json",
        });
    }
    Ok(path)
}

fn home() -> Result<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from).ok_or_else(|| {
        Error::Invariant("HOME is unset, so no cache directory can be found".to_owned())
    })
}

/// g2pW's per-character readings for a batch of sentences.
///
/// One entry per character of the input sentence, `None` where g2pW has no
/// reading at all. This is the shape the Python `G2PWConverter.__call__`
/// returns, so the two can be compared position for position.
pub type Predictions = Vec<Vec<Option<String>>>;

/// The synchronous engine: the ONNX session, its tokenizer and its tables.
///
/// Kept separate from [`G2pwAnnotator`] so that it can be driven directly by the
/// parity harness, which has no event loop and wants the raw per-character
/// output rather than an [`Outcome`].
#[derive(Debug)]
pub struct Converter {
    session: Session,
    tokenizer: Tokenizer,
    tables: Tables,
    batch_size: usize,
}

impl Converter {
    /// Load the model, its tables and the BERT tokenizer.
    ///
    /// # Errors
    ///
    /// If the model directory does not hold `g2pw.onnx` and the two character
    /// tables, if the tokenizer cannot be read, or if the tables no longer match
    /// the network's label space.
    pub fn load(model_dir: &Path, tokenizer_path: &Path, batch_size: usize) -> Result<Self> {
        let model = model_dir.join("g2pw.onnx");
        if !model.exists() {
            return Err(Error::Missing {
                what: "the g2pW ONNX model",
                path: model,
                hint: "download G2PWModel-v2-onnx.zip, or point --g2pw-model at an existing copy",
            });
        }
        info!(model_dir = %model_dir.display(), tokenizer = %tokenizer_path.display(), "loading g2pw");
        let tables = Tables::load(model_dir)?;
        let tokenizer =
            Tokenizer::from_file(tokenizer_path).map_err(|source| Error::Tokenizer {
                path: tokenizer_path.to_owned(),
                source,
            })?;
        let session = open_session(&model)?;
        Ok(Self {
            session,
            tokenizer,
            tables,
            batch_size: batch_size.max(1),
        })
    }

    /// The tables this converter reads, for callers that need to explain a reading.
    #[must_use]
    pub fn tables(&self) -> &Tables {
        &self.tables
    }

    /// Read every character of every sentence.
    ///
    /// # Errors
    ///
    /// If a sentence cannot be tokenized, or if the network fails.
    pub fn convert(&mut self, sentences: &[String]) -> Result<Predictions> {
        let traditional: Vec<Vec<char>> = sentences
            .iter()
            .map(|sentence| {
                sentence
                    .chars()
                    .map(|character| self.tables.to_traditional(character))
                    .collect()
            })
            .collect();

        let mut predictions: Predictions = traditional
            .iter()
            .map(|sentence| vec![None; sentence.len()])
            .collect();
        let mut queries: Vec<(usize, usize)> = Vec::new();

        for (sentence_index, sentence) in traditional.iter().enumerate() {
            for (position, character) in sentence.iter().enumerate() {
                if self.tables.is_polyphonic(*character) {
                    queries.push((sentence_index, position));
                } else if let Some(phoneme) = self.tables.monophonic(*character) {
                    predictions[sentence_index][position] = self.tables.to_pinyin(phoneme);
                } else if let Some(phoneme) = self.tables.fallback(*character) {
                    predictions[sentence_index][position] = self.tables.to_pinyin(phoneme);
                }
            }
        }
        if queries.is_empty() {
            return Ok(predictions);
        }

        let mut batch: Vec<Features> = Vec::with_capacity(self.batch_size);
        let mut pending: Vec<(usize, usize)> = Vec::with_capacity(self.batch_size);
        for (sentence_index, position) in queries {
            batch.push(features(
                &self.tables,
                &self.tokenizer,
                &traditional[sentence_index],
                position,
            )?);
            pending.push((sentence_index, position));
            if batch.len() == self.batch_size {
                self.run_batch(&batch, &pending, &mut predictions)?;
                batch.clear();
                pending.clear();
            }
        }
        if !batch.is_empty() {
            self.run_batch(&batch, &pending, &mut predictions)?;
        }
        Ok(predictions)
    }

    /// Run one padded batch and write its readings into `predictions`.
    fn run_batch(
        &mut self,
        batch: &[Features],
        pending: &[(usize, usize)],
        predictions: &mut Predictions,
    ) -> Result<()> {
        let rows = batch.len();
        let width = batch
            .iter()
            .map(|item| item.input_ids.len())
            .max()
            .unwrap_or(0);
        let labels = self.tables.label_count();

        let mut input_ids = vec![0_i64; rows * width];
        let token_type_ids = vec![0_i64; rows * width];
        let mut attention_mask = vec![0_i64; rows * width];
        let mut phoneme_mask = vec![0.0_f32; rows * labels];
        let mut char_ids = vec![0_i64; rows];
        let mut position_ids = vec![0_i64; rows];

        for (row, item) in batch.iter().enumerate() {
            let offset = row * width;
            input_ids[offset..offset + item.input_ids.len()].copy_from_slice(&item.input_ids);
            attention_mask[offset..offset + item.input_ids.len()].fill(1);
            let mask = row * labels;
            phoneme_mask[mask..mask + labels].copy_from_slice(&item.phoneme_mask);
            char_ids[row] = item.char_id;
            position_ids[row] = item.position_id;
        }

        let shape = [rows_i64(rows), rows_i64(width)];
        let outputs = self.session.run(ort::inputs![
            "input_ids" => Tensor::from_array((shape, input_ids))?,
            "token_type_ids" => Tensor::from_array((shape, token_type_ids))?,
            "attention_mask" => Tensor::from_array((shape, attention_mask))?,
            "phoneme_mask" => Tensor::from_array(([rows_i64(rows), rows_i64(labels)], phoneme_mask))?,
            "char_ids" => Tensor::from_array(([rows_i64(rows)], char_ids))?,
            "position_ids" => Tensor::from_array(([rows_i64(rows)], position_ids))?,
        ])?;
        let (_, probabilities) = outputs["probs"].try_extract_tensor::<f32>()?;

        for (row, (sentence_index, position)) in pending.iter().enumerate() {
            let scores = &probabilities[row * labels..(row + 1) * labels];
            let best = argmax(scores);
            let label = self.tables.label(best).ok_or_else(|| {
                Error::Invariant(format!(
                    "the network chose label {best}, which the tables do not have"
                ))
            })?;
            predictions[*sentence_index][*position] = self.tables.to_pinyin(label);
        }
        Ok(())
    }
}

/// Open the ONNX session with the same options the Python `G2PWConverter` uses:
/// every graph optimization, sequential execution, two intra-op threads.
fn open_session(model: &Path) -> ort::Result<Session> {
    let mut builder = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_parallel_execution(false)?
        .with_intra_threads(2)?;
    builder.commit_from_file(model)
}

/// The first index holding the largest value, which is what `numpy.argmax` returns.
fn argmax(scores: &[f32]) -> usize {
    let mut best = 0;
    for (index, score) in scores.iter().enumerate().skip(1) {
        if *score > scores[best] {
            best = index;
        }
    }
    best
}

#[expect(
    clippy::cast_possible_wrap,
    reason = "a batch dimension cannot reach i64::MAX"
)]
fn rows_i64(value: usize) -> i64 {
    value as i64
}

/// One batch of sentences on its way to the ONNX session, and where the answer goes.
struct Request {
    texts: Vec<String>,
    reply: oneshot::Sender<Result<Predictions>>,
}

/// Per-character pinyin from g2pW, aligned to a sentence's Han characters.
///
/// The ONNX session needs `&mut` to run and blocks for as long as it runs, so it
/// lives on a thread of its own and is reached over a channel. That keeps the
/// event loop free without a lock, and it is the only place in the pipeline that
/// blocks at all.
#[derive(Debug)]
pub struct G2pwAnnotator {
    requests: Option<mpsc::UnboundedSender<Request>>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl G2pwAnnotator {
    /// Load the model and start its worker thread.
    ///
    /// # Errors
    ///
    /// Whatever [`Converter::load`] refuses on.
    pub fn new(model_dir: &Path, tokenizer_path: &Path, batch_size: usize) -> Result<Self> {
        let mut converter = Converter::load(model_dir, tokenizer_path, batch_size)?;
        let (sender, mut receiver) = mpsc::unbounded_channel::<Request>();
        let worker = std::thread::Builder::new()
            .name("g2pw".to_owned())
            .spawn(move || {
                while let Some(request) = receiver.blocking_recv() {
                    let answer = converter.convert(&request.texts);
                    if request.reply.send(answer).is_err() {
                        warn!("a g2pw batch was abandoned before its answer arrived");
                    }
                }
            })
            .map_err(|source| Error::Read {
                path: PathBuf::from("<g2pw worker thread>"),
                source,
            })?;
        Ok(Self {
            requests: Some(sender),
            worker: Some(worker),
        })
    }

    /// Read every character of every sentence, on the worker thread.
    ///
    /// # Errors
    ///
    /// Whatever [`Converter::convert`] refuses on, or if the worker has died.
    pub async fn predict(&self, texts: &[String]) -> Result<Predictions> {
        let (reply, answer) = oneshot::channel();
        let requests = self
            .requests
            .as_ref()
            .ok_or_else(|| Error::Invariant("the g2pw worker is shutting down".to_owned()))?;
        requests
            .send(Request {
                texts: texts.to_vec(),
                reply,
            })
            .map_err(|_| Error::Invariant("the g2pw worker thread has died".to_owned()))?;
        answer
            .await
            .map_err(|_| Error::Invariant("the g2pw worker dropped a batch".to_owned()))?
    }
}

impl Drop for G2pwAnnotator {
    fn drop(&mut self) {
        self.requests = None;
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl Annotator for G2pwAnnotator {
    fn name(&self) -> &'static str {
        "g2pw"
    }

    async fn annotate(&self, texts: &[String]) -> Vec<Outcome> {
        match self.predict(texts).await {
            Ok(predictions) => texts
                .iter()
                .zip(predictions)
                .map(|(text, row)| outcome(text, &row))
                .collect(),
            Err(error) => {
                let reason = format!("{error}");
                texts.iter().map(|_| Err(Refusal::new(&reason))).collect()
            }
        }
    }
}

/// Keep the Han positions, refusing the sentence if any of them came back empty.
fn outcome(text: &str, predictions: &[Option<String>]) -> Outcome {
    let length = text.chars().count();
    if predictions.len() != length {
        return Err(Refusal::new(format!(
            "g2pw returned {} readings for {length} characters",
            predictions.len()
        )));
    }
    let mut syllables = Vec::new();
    for (character, syllable) in text.chars().zip(predictions) {
        if !is_han(character) {
            continue;
        }
        let Some(syllable) = syllable else {
            return Err(Refusal::new(format!(
                "g2pw has no reading for '{character}'"
            )));
        };
        syllables.push(syllable.clone());
    }
    if syllables.is_empty() {
        return Err(Refusal::new("no Han characters to read"));
    }
    Ok(Reading::new(syllables))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_keeps_the_first_of_equal_scores_the_way_numpy_does() {
        assert_eq!(argmax(&[0.1, 0.5, 0.5, 0.2]), 1);
        assert_eq!(argmax(&[1.0]), 0);
        assert_eq!(argmax(&[0.0, 0.0, 0.0]), 0);
        assert_eq!(argmax(&[-3.0, -1.0, -2.0]), 1);
    }

    #[test]
    fn only_the_han_positions_become_syllables() {
        let reading = outcome(
            "中，国",
            &[Some("zhong1".to_owned()), None, Some("guo2".to_owned())],
        )
        .expect("the comma needs no reading");
        assert_eq!(reading.syllables, ["zhong1", "guo2"]);
    }

    #[test]
    fn a_han_position_with_no_reading_refuses_the_whole_sentence() {
        let refusal = outcome("中国", &[Some("zhong1".to_owned()), None])
            .expect_err("a missing reading is a refusal");
        assert_eq!(refusal.reason, "g2pw has no reading for '国'");
    }

    #[test]
    fn a_reading_list_of_the_wrong_length_refuses_the_sentence() {
        let refusal =
            outcome("中国", &[Some("zhong1".to_owned())]).expect_err("the lengths disagree");
        assert_eq!(refusal.reason, "g2pw returned 1 readings for 2 characters");
    }

    #[test]
    fn a_sentence_with_no_han_is_refused_rather_than_returned_empty() {
        let refusal = outcome("abc", &[None, None, None]).expect_err("nothing to read");
        assert_eq!(refusal.reason, "no Han characters to read");
    }
}
