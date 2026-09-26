//! A character language model as the decoder's transition.
//!
//! The trigram scores a character against the two before it. The model here
//! scores it against everything before it: the sentence so far and, ahead of
//! that, the context that was on screen. `mlime train char-lm` trains it and
//! `mlime export char-lm` writes the three files this crate loads:
//! `charlm.json`, a manifest that names the model's tensors and holds the
//! alphabet in id order, and the two ONNX graphs the manifest describes.
//! `prefill.onnx` reads the prelude (`<bos> context <sep>`) once and produces
//! the state every beam starts from; `charlm.onnx` advances a batch of beams
//! by one character and returns the log probabilities of the next.
//!
//! A state is two kinds of tensor. The *prefix* tensors are what the prelude
//! produced -- the transformer's key/value cache over the context -- and every
//! beam of a record shares them; the *state* tensors are per beam (the LSTM's
//! `hidden`/`cell`, the transformer's cache over the sentence so far). Every
//! tensor is batch-first: a step's batch is the rows of the surviving beams
//! stacked, and a beam's share of an output is one row, without the crate ever
//! interpreting what a row holds.
//!
//! The alphabet is the lexicon's, so every [`CharId`] the lattice can propose
//! has a row; the mapping is built once at load and a lexicon that disagrees
//! with the manifest is refused there rather than mis-scored later.
//!
//! ONNX Runtime sessions are run through `&mut self`, and the decoder runs
//! records in parallel, so each thread that decodes opens its own pair of
//! sessions from the same graphs the first time it needs one.

use ime_decode::{MAX_HISTORY, Transition};
use ime_pinyin::{CharId, Lexicon};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{Session, SessionInputValue};
use ort::value::{Tensor, TensorRef};
use serde::Deserialize;
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;
use thread_local::ThreadLocal;

/// What can go wrong opening a model.
#[derive(Debug, Error)]
pub enum LmError {
    /// The manifest or graph could not be read.
    #[error("could not read {path}")]
    Io {
        /// The file that failed.
        path: PathBuf,
        /// The underlying error.
        #[source]
        source: std::io::Error,
    },
    /// The manifest is not the JSON `mlime export char-lm` writes.
    #[error("the manifest at {path} is malformed")]
    Manifest {
        /// The file that failed.
        path: PathBuf,
        /// The underlying error.
        #[source]
        source: serde_json::Error,
    },
    /// ONNX Runtime refused the graph or a run.
    #[error("onnx runtime failed")]
    Onnx(#[from] ort::Error),
    /// A character the lexicon holds has no row in the model's alphabet.
    #[error("the model's alphabet lacks {character:?}, which the lexicon holds")]
    Alphabet {
        /// The character without a row.
        character: char,
    },
}

/// The reserved ids, as the manifest names them.
#[derive(Debug, Clone, Copy, Deserialize)]
struct Specials {
    bos: u32,
    eos: u32,
    sep: u32,
    unk: u32,
}

/// `charlm.json`, the fields the run consults; the manifest also records the
/// training step, the architecture and the restricted alphabet size, which
/// the tensor names and the log-probability rows already carry.
#[derive(Debug, Deserialize)]
struct Manifest {
    context_chars: usize,
    /// Names of the tensors `prefill` produces and every beam of a record
    /// shares: empty for the LSTM, the key/value cache over the prelude for the
    /// transformer.
    prefix: Vec<String>,
    /// Names of the per-beam tensors: `hidden`/`cell` for the LSTM, the
    /// key/value cache over the sentence so far for the transformer.
    state: Vec<String>,
    specials: Specials,
    chars: Vec<String>,
}

/// One row of a named state or prefix tensor.
///
/// Every tensor the graphs exchange is batch-first, so a beam's share is the
/// row behind the batch axis: its shape and its data, both read off the
/// graph's outputs rather than assumed, because a row's shape can change from
/// one step to the next (the transformer's time axis grows by one).
#[derive(Debug)]
struct StateTensor {
    /// The row's shape: the tensor's axes after the batch axis.
    shape: Vec<i64>,
    data: Vec<f32>,
}

/// The model's state for one beam: the shared prefix, this beam's own tensors,
/// and the log probabilities of whatever comes next.
///
/// Shared rather than owned because the decoder clones a beam whenever it keeps
/// it; cloning a beam is three reference-count bumps, and the prefix is the
/// same [`Arc`] for every beam of a record.
#[derive(Clone, Debug)]
pub struct LmState {
    /// The tensors `prefill` produced for the record's prelude.
    prefix: Arc<Vec<StateTensor>>,
    /// This beam's tensors, in the manifest's `state` order.
    tensors: Arc<Vec<StateTensor>>,
    /// One log probability per id of the alphabet.
    log_probs: Arc<Vec<f32>>,
}

impl LmState {
    /// The beam's distribution over the next character, one log probability
    /// per alphabet id.
    #[must_use]
    pub fn log_probs(&self) -> &[f32] {
        &self.log_probs
    }
}

/// A trained character language model, ready to score.
#[derive(Debug)]
pub struct CharLm {
    /// `prefill.onnx`.
    prefill_graph: PathBuf,
    /// `charlm.onnx`, the step graph.
    step_graph: PathBuf,
    prefill_sessions: ThreadLocal<RefCell<Session>>,
    step_sessions: ThreadLocal<RefCell<Session>>,
    /// The manifest's `prefix`: names of the shared tensors.
    prefix: Vec<String>,
    /// The manifest's `state`: names of the per-beam tensors, whose outputs the
    /// step graph returns as `next_<name>`.
    state: Vec<String>,
    context_chars: usize,
    specials: Specials,
    /// Alphabet id of every lexicon [`CharId`], by index.
    ids: Vec<u32>,
    /// Alphabet id of every character the context may contain.
    alphabet: HashMap<char, u32>,
}

impl CharLm {
    /// Open the model in *dir* (`charlm.json`, `prefill.onnx`, `charlm.onnx`)
    /// for *lexicon*.
    ///
    /// # Errors
    ///
    /// If the manifest cannot be read or names a field the export does not
    /// write, either graph cannot be loaded, or a character of the lexicon has
    /// no row in the model's alphabet.
    pub fn open(dir: &Path, lexicon: &Lexicon) -> Result<Self, LmError> {
        let manifest_path = dir.join("charlm.json");
        let raw = fs::read_to_string(&manifest_path).map_err(|source| LmError::Io {
            path: manifest_path.clone(),
            source,
        })?;
        let manifest: Manifest =
            serde_json::from_str(&raw).map_err(|source| LmError::Manifest {
                path: manifest_path,
                source,
            })?;
        let alphabet: HashMap<char, u32> = (0u32..)
            .zip(&manifest.chars)
            .filter_map(|(id, entry)| {
                let mut chars = entry.chars();
                match (chars.next(), chars.next()) {
                    (Some(ch), None) => Some((ch, id)),
                    _ => None,
                }
            })
            .collect();
        let ids = lexicon
            .characters()
            .iter()
            .map(|&character| {
                alphabet
                    .get(&character)
                    .copied()
                    .ok_or(LmError::Alphabet { character })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let model = Self {
            prefill_graph: dir.join("prefill.onnx"),
            step_graph: dir.join("charlm.onnx"),
            prefill_sessions: ThreadLocal::new(),
            step_sessions: ThreadLocal::new(),
            prefix: manifest.prefix,
            state: manifest.state,
            context_chars: manifest.context_chars,
            specials: manifest.specials,
            ids,
            alphabet,
        };
        model.prefill_session()?;
        model.step_session()?;
        Ok(model)
    }

    /// This thread's prefill session, opened on first use.
    fn prefill_session(&self) -> Result<&RefCell<Session>, LmError> {
        self.prefill_sessions.get_or_try(|| {
            open_session(&self.prefill_graph)
                .map(RefCell::new)
                .map_err(LmError::from)
        })
    }

    /// This thread's step session, opened on first use.
    fn step_session(&self) -> Result<&RefCell<Session>, LmError> {
        self.step_sessions.get_or_try(|| {
            open_session(&self.step_graph)
                .map(RefCell::new)
                .map_err(LmError::from)
        })
    }

    /// Read the prelude through `prefill.onnx`: the state every beam starts
    /// from, prefix tensors included.
    fn prefill(&self, context: Option<&str>) -> Result<LmState, LmError> {
        let prelude = self.prelude(context);
        let tokens: Vec<i64> = prelude.iter().copied().map(i64::from).collect();
        let length = dim(prelude.len());
        let session = self.prefill_session()?;
        let mut session = session.borrow_mut();
        let outputs = session.run(ort::inputs![
            "tokens" => Tensor::from_array(([1i64, length], tokens))?,
        ])?;
        let (_, log_probs) = outputs["log_probs"].try_extract_tensor::<f32>()?;
        // `prefill` ran a single prelude, so every tensor it produced is one
        // row; the row shape is everything after the batch axis.
        let row = |name: &String| -> Result<StateTensor, ort::Error> {
            let (shape, data) = outputs[name.as_str()].try_extract_tensor::<f32>()?;
            debug_assert_eq!(shape[0], 1);
            Ok(StateTensor {
                shape: shape[1..].to_vec(),
                data: data.to_vec(),
            })
        };
        Ok(LmState {
            prefix: Arc::new(self.prefix.iter().map(&row).collect::<Result<_, _>>()?),
            tensors: Arc::new(self.state.iter().map(&row).collect::<Result<_, _>>()?),
            log_probs: Arc::new(log_probs.to_vec()),
        })
    }

    /// Advance a batch: `tokens[i]` applied to `states[i]`, which must all
    /// share one record's prefix.
    fn step(&self, states: &[&LmState], tokens: &[u32]) -> Result<Vec<LmState>, LmError> {
        let batch = tokens.len();
        debug_assert_eq!(states.len(), batch, "one state per token");
        let prefix = &states[0].prefix;
        for state in states {
            debug_assert!(
                Arc::ptr_eq(prefix, &state.prefix),
                "every state of a step shares the record's prefix"
            );
        }
        let mut inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> =
            Vec::with_capacity(1 + self.prefix.len() + self.state.len());
        let tokens: Vec<i64> = tokens.iter().copied().map(i64::from).collect();
        inputs.push((
            Cow::Borrowed("token"),
            Tensor::from_array((vec![dim(batch)], tokens))?.into(),
        ));
        // The shared tensors go in as the one row `prefill` produced; the step
        // graph broadcasts them over the batch.
        for (name, tensor) in self.prefix.iter().zip(prefix.iter()) {
            let mut shape = Vec::with_capacity(tensor.shape.len() + 1);
            shape.push(1);
            shape.extend_from_slice(&tensor.shape);
            inputs.push((
                Cow::Owned(name.clone()),
                TensorRef::from_array_view((shape, tensor.data.as_slice()))?.into(),
            ));
        }
        // The per-beam tensors stack row by row; every row of one tensor has
        // the same shape at the same step.
        for (index, name) in self.state.iter().enumerate() {
            let row_shape = &states[0].tensors[index].shape;
            let mut data = Vec::with_capacity(batch * states[0].tensors[index].data.len());
            for state in states {
                let tensor = &state.tensors[index];
                debug_assert_eq!(
                    row_shape, &tensor.shape,
                    "every row of {name} has the same shape at one step"
                );
                data.extend_from_slice(&tensor.data);
            }
            let mut shape = Vec::with_capacity(row_shape.len() + 1);
            shape.push(dim(batch));
            shape.extend_from_slice(row_shape);
            inputs.push((
                Cow::Owned(name.clone()),
                Tensor::from_array((shape, data))?.into(),
            ));
        }
        let session = self.step_session()?;
        let mut session = session.borrow_mut();
        let outputs = session.run(inputs)?;
        let (_, log_probs) = outputs["log_probs"].try_extract_tensor::<f32>()?;
        let vocabulary = log_probs.len() / batch;
        // Each `next_*` output is split into rows by its own shape: a row's
        // shape is whatever the graph produced, which for the transformer
        // grows one position on the time axis per step.
        let next: Vec<(Vec<i64>, Vec<f32>)> = self
            .state
            .iter()
            .map(|name| {
                let (shape, data) =
                    outputs[format!("next_{name}").as_str()].try_extract_tensor::<f32>()?;
                debug_assert_eq!(shape[0], dim(batch));
                Ok((shape[1..].to_vec(), data.to_vec()))
            })
            .collect::<Result<_, ort::Error>>()?;
        Ok((0..batch)
            .map(|row| {
                let tensors = next
                    .iter()
                    .map(|(row_shape, data)| {
                        let row_len = elements(row_shape);
                        StateTensor {
                            shape: row_shape.clone(),
                            data: data[row * row_len..(row + 1) * row_len].to_vec(),
                        }
                    })
                    .collect();
                LmState {
                    prefix: Arc::clone(prefix),
                    tensors: Arc::new(tensors),
                    log_probs: Arc::new(
                        log_probs[row * vocabulary..(row + 1) * vocabulary].to_vec(),
                    ),
                }
            })
            .collect())
    }

    /// The sequence the model reads before the first character of the sentence:
    /// `<bos>`, the end of the context, `<sep>`.
    fn prelude(&self, context: Option<&str>) -> Vec<u32> {
        let mut tokens = vec![self.specials.bos];
        if let Some(context) = context {
            let chars: Vec<char> = context.chars().collect();
            let keep = chars.len().saturating_sub(self.context_chars);
            tokens.extend(
                chars[keep..]
                    .iter()
                    .map(|ch| self.alphabet.get(ch).copied().unwrap_or(self.specials.unk)),
            );
        }
        tokens.push(self.specials.sep);
        tokens
    }
}

impl Transition for CharLm {
    const HISTORY: usize = MAX_HISTORY;

    type State = LmState;

    /// # Panics
    ///
    /// If ONNX Runtime fails a run, which the session that [`CharLm::open`]
    /// verified does not do for well-formed inputs.
    fn start(&self, context: Option<&str>) -> LmState {
        self.prefill(context)
            .expect("the prefill graph runs on the prelude")
    }

    fn score(&self, state: &LmState, candidate: CharId) -> f32 {
        state.log_probs[self.ids[candidate.index()] as usize]
    }

    fn finish(&self, state: &LmState) -> f32 {
        state.log_probs[self.specials.eos as usize]
    }

    /// # Panics
    ///
    /// If ONNX Runtime fails a run; see [`CharLm::start`].
    fn advance(&self, steps: &[(&LmState, CharId)]) -> Vec<LmState> {
        if steps.is_empty() {
            return Vec::new();
        }
        let states: Vec<&LmState> = steps.iter().map(|(state, _)| *state).collect();
        let tokens: Vec<u32> = steps.iter().map(|(_, ch)| self.ids[ch.index()]).collect();
        self.step(&states, &tokens)
            .expect("the graph runs on a batch")
    }
}

/// Open a graph with one intra-op thread: the decoder is the parallel
/// layer, one record per thread, and a session that also spread its matrix
/// products over threads would oversubscribe the machine.
fn open_session(graph: &Path) -> ort::Result<Session> {
    let mut builder = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_parallel_execution(false)?
        .with_intra_threads(1)?;
    builder.commit_from_file(graph)
}

/// A tensor dimension as ONNX Runtime spells it.
#[expect(
    clippy::cast_possible_wrap,
    reason = "a dimension is a beam width, a hidden size or a sequence length, none of which approaches i64::MAX"
)]
const fn dim(n: usize) -> i64 {
    n as i64
}

/// The element count of a tensor of *shape*: the product of its axes.
#[expect(
    clippy::cast_sign_loss,
    reason = "a shape the graph produced has no negative axis"
)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "a tensor axis is at most a sequence length or beam width, far under usize::MAX even on 32-bit targets"
)]
fn elements(shape: &[i64]) -> usize {
    shape.iter().map(|&axis| axis as usize).product()
}
