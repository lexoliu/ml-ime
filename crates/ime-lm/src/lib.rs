//! A recurrent character language model as the decoder's transition.
//!
//! The trigram scores a character against the two before it. The model here
//! scores it against everything before it: the sentence so far and, ahead of
//! that, the context that was on screen, because its state is a recurrent
//! network's hidden state and the decoder carries one per beam. `mlime train
//! char-lm` trains it and `mlime export char-lm` writes the two files this crate
//! loads: `charlm.onnx`, a graph that advances a batch of states by one
//! character and returns the log probabilities of the next, and `charlm.json`,
//! the alphabet in id order with the state's shape.
//!
//! The alphabet is the lexicon's, so every [`CharId`] the lattice can propose
//! has a row; the mapping is built once at load and a lexicon that disagrees
//! with the manifest is refused there rather than mis-scored later.
//!
//! ONNX Runtime sessions are run through `&mut self`, and the decoder runs
//! records in parallel, so each thread that decodes opens its own session from
//! the same graph the first time it needs one.

use ime_decode::{MAX_HISTORY, Transition};
use ime_pinyin::{CharId, Lexicon};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use serde::Deserialize;
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

/// `charlm.json`.
#[derive(Debug, Deserialize)]
struct Manifest {
    layers: u32,
    hidden: u32,
    context_chars: usize,
    specials: Specials,
    chars: Vec<String>,
}

/// The model's state for one beam: the recurrent state after the characters so
/// far, and the log probabilities of whatever comes next.
///
/// Shared rather than owned because the decoder clones a beam whenever it keeps
/// it, and a state is a few kilobytes of floats plus a row over the alphabet.
#[derive(Clone, Debug)]
pub struct LmState(Arc<StateInner>);

#[derive(Debug)]
struct StateInner {
    /// `[layers * hidden]`, layer-major.
    hidden: Vec<f32>,
    /// `[layers * hidden]`, layer-major.
    cell: Vec<f32>,
    /// One log probability per id of the alphabet.
    log_probs: Vec<f32>,
}

/// A trained character language model, ready to score.
#[derive(Debug)]
pub struct CharLm {
    graph: PathBuf,
    sessions: ThreadLocal<RefCell<Session>>,
    layers: usize,
    hidden: usize,
    context_chars: usize,
    specials: Specials,
    /// Alphabet id of every lexicon [`CharId`], by index.
    ids: Vec<u32>,
    /// Alphabet id of every character the context may contain.
    alphabet: HashMap<char, u32>,
}

impl CharLm {
    /// Open the model in *dir* (`charlm.onnx` and `charlm.json`) for *lexicon*.
    ///
    /// # Errors
    ///
    /// If either file cannot be read, the graph cannot be loaded, or a character
    /// of the lexicon has no row in the model's alphabet.
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
        let graph = dir.join("charlm.onnx");
        let model = Self {
            graph,
            sessions: ThreadLocal::new(),
            layers: manifest.layers as usize,
            hidden: manifest.hidden as usize,
            context_chars: manifest.context_chars,
            specials: manifest.specials,
            ids,
            alphabet,
        };
        model.session()?;
        Ok(model)
    }

    /// This thread's session, opened on first use.
    fn session(&self) -> Result<&RefCell<Session>, LmError> {
        self.sessions.get_or_try(|| {
            open_session(&self.graph)
                .map(RefCell::new)
                .map_err(LmError::from)
        })
    }

    /// Advance a batch: `tokens[i]` applied to `states[i]`, or to the zero state
    /// when `states` is empty.
    fn step(&self, states: &[&LmState], tokens: &[u32]) -> Result<Vec<LmState>, LmError> {
        let batch = tokens.len();
        let tokens: Vec<i64> = tokens.iter().copied().map(i64::from).collect();
        let width = self.layers * self.hidden;
        let mut hidden = vec![0.0f32; batch * width];
        let mut cell = vec![0.0f32; batch * width];
        for (row, state) in states.iter().enumerate() {
            // The graph's states are [layers, batch, hidden]; ours are per beam.
            for layer in 0..self.layers {
                let src = layer * self.hidden..(layer + 1) * self.hidden;
                let dst =
                    (layer * batch + row) * self.hidden..(layer * batch + row + 1) * self.hidden;
                hidden[dst.clone()].copy_from_slice(&state.0.hidden[src.clone()]);
                cell[dst].copy_from_slice(&state.0.cell[src]);
            }
        }
        let shape = [dim(self.layers), dim(batch), dim(self.hidden)];
        let session = self.session()?;
        let mut session = session.borrow_mut();
        let outputs = session.run(ort::inputs![
            "token" => Tensor::from_array(([shape[1]], tokens))?,
            "hidden" => Tensor::from_array((shape, hidden))?,
            "cell" => Tensor::from_array((shape, cell))?,
        ])?;
        let (_, log_probs) = outputs["log_probs"].try_extract_tensor::<f32>()?;
        let (_, next_hidden) = outputs["next_hidden"].try_extract_tensor::<f32>()?;
        let (_, next_cell) = outputs["next_cell"].try_extract_tensor::<f32>()?;
        let vocabulary = log_probs.len() / batch;
        Ok((0..batch)
            .map(|row| {
                let mut hidden = Vec::with_capacity(width);
                let mut cell = Vec::with_capacity(width);
                for layer in 0..self.layers {
                    let src = (layer * batch + row) * self.hidden
                        ..(layer * batch + row + 1) * self.hidden;
                    hidden.extend_from_slice(&next_hidden[src.clone()]);
                    cell.extend_from_slice(&next_cell[src]);
                }
                LmState(Arc::new(StateInner {
                    hidden,
                    cell,
                    log_probs: log_probs[row * vocabulary..(row + 1) * vocabulary].to_vec(),
                }))
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
        let mut state: Option<LmState> = None;
        for token in self.prelude(context) {
            let states: Vec<&LmState> = state.iter().collect();
            state = self
                .step(&states, &[token])
                .expect("the graph runs on a single token")
                .pop();
        }
        state.expect("the prelude has at least two tokens")
    }

    fn score(&self, state: &LmState, candidate: CharId) -> f32 {
        state.0.log_probs[self.ids[candidate.index()] as usize]
    }

    fn finish(&self, state: &LmState) -> f32 {
        state.0.log_probs[self.specials.eos as usize]
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

/// Open the step graph with one intra-op thread: the decoder is the parallel
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
    reason = "a dimension is a layer count, a beam width or a hidden size, none of which approaches i64::MAX"
)]
const fn dim(n: usize) -> i64 {
    n as i64
}
