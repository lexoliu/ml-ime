//! A character language model as the decoder's transition.
//!
//! The trigram scores a character against the two before it. The model here
//! scores it against everything before it: the sentence so far and, ahead of
//! that, the context that was on screen. `mlime train char-lm` trains it and
//! `mlime export char-lm` writes the files this crate loads:
//! `charlm.json`, a manifest that names the model's tensors and holds the
//! alphabet in id order, the two ONNX graphs the manifest describes, and the
//! weights file the graphs' initializers live in. `prefill.onnx` reads the
//! prelude (`<bos> context <sep>`) once and produces the state every beam
//! starts from; `charlm.onnx` advances a batch of beams by one character and
//! returns the log probabilities of the next.
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
//! sessions from the same graphs the first time it needs one. Every session
//! is built over the graph's one shared mapping: its initializers point into
//! the mapping rather than a per-session copy, and the matrices the runtime
//! pre-packs are shared too, so the model's memory is one copy of the
//! weights plus per-thread working space no matter how many threads decode.

use ime_decode::{MAX_HISTORY, Transition};
use ime_pinyin::{CharId, Lexicon};
use memmap2::Mmap;
use ort::AsPointer;
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::builder::{GraphOptimizationLevel, PrepackedWeights};
use ort::session::{Session, SessionInputValue};
use ort::value::{DynValue, Shape, Tensor, TensorRef, TensorRefMut};
use serde::Deserialize;
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::Arc;
use thiserror::Error;
use thread_local::ThreadLocal;

/// The per-thread sessions of one graph, over one shared copy of its weights.
///
/// *initializers* are the `OrtValue`s the export's weights table described,
/// each pointing into the model's one mapping of the weights file.
/// Registering them with `with_initializer` enrolls each for the shared
/// [`PrepackedWeights`] container too, so the matrices MLAS pre-packs are
/// written once and reused instead of packed per session. `by_thread` is
/// declared first so its sessions drop before the weights they point into.
struct GraphSessions {
    /// One session per thread, opened on first use.
    by_thread: ThreadLocal<RefCell<Session>>,
    /// The graph's file.
    path: PathBuf,
    /// The prepacked weights every session of this graph shares.
    prepacked: PrepackedWeights,
    /// The graph's initializers, one `OrtValue` each shared by every session.
    initializers: Vec<(String, Arc<DynValue>)>,
}

impl GraphSessions {
    fn new(graph: PathBuf, initializers: Vec<(String, Arc<DynValue>)>) -> Self {
        Self {
            by_thread: ThreadLocal::new(),
            path: graph,
            prepacked: PrepackedWeights::new(),
            initializers,
        }
    }

    /// This thread's session, opened from the shared file on first use.
    ///
    /// Sessions may open concurrently: ONNX Runtime serializes the pre-packed
    /// weights lookups and writes itself -- `PrepackConstantInitializedTensors`
    /// holds `prepacked_weights_container_->mutex_` around them.
    fn session(&self) -> Result<&RefCell<Session>, LmError> {
        self.by_thread.get_or_try(|| {
            open_session(&self.path, &self.prepacked, &self.initializers)
                .map(RefCell::new)
                .map_err(LmError::from)
        })
    }
}

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
    /// The weights file is missing bytes the manifest's table names, or names
    /// an extent that does not fit the tensor it is for.
    #[error("the weights file {path} does not match the manifest: {reason}")]
    Weights {
        /// The file the manifest pointed at.
        path: PathBuf,
        /// How it disagrees with the table.
        reason: String,
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
    /// Where the graphs' initializers live: one shared file's table, or one
    /// table per graph when the step and prefill tensors have different names.
    weights: WeightsTable,
    specials: Specials,
    chars: Vec<String>,
}

/// The manifest's `weights`: either both graphs share one file's table, or
/// each graph has its own.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum WeightsTable {
    /// `"weights": {"file": ..., "tensors": [...]}` covering both graphs.
    Shared(WeightsFile),
    /// `"weights": {"step": ..., "prefill": ...}`, a table per graph.
    PerGraph {
        /// The step graph's (`charlm.onnx`) table.
        step: WeightsFile,
        /// The prefill graph's table.
        prefill: WeightsFile,
    },
}

/// One weights file's table: the file's name and where every shared
/// initializer sits in it, as the export wrote the `external_data` entries.
#[derive(Debug, Deserialize)]
struct WeightsFile {
    /// The file's name inside the model's directory.
    file: String,
    /// Every externalized tensor of the graph(s) the file backs.
    tensors: Vec<WeightTensor>,
}

/// One shared initializer: its name in the graph, its shape and dtype, and
/// its byte extent inside the weights file.
#[derive(Debug, Deserialize)]
struct WeightTensor {
    name: String,
    /// The element type; the export writes `float32` and anything else fails
    /// the manifest's parse.
    dtype: WeightDtype,
    shape: Vec<i64>,
    /// Byte offset into the weights file.
    offset: u64,
    /// Byte length of the tensor's raw data.
    length: u64,
}

/// The element type a shared tensor's bytes hold; anything the export
/// doesn't write is a manifest parse error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
enum WeightDtype {
    /// IEEE-754 single precision, little-endian, four bytes per element.
    #[serde(rename = "float32")]
    Float32,
    /// Signed 64-bit integer, little-endian.
    #[serde(rename = "int64")]
    Int64,
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
pub struct CharLm {
    /// `prefill.onnx`: the per-thread sessions over it.
    prefill_sessions: GraphSessions,
    /// `charlm.onnx`, the step graph: the per-thread sessions over it.
    step_sessions: GraphSessions,
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
    /// The mapping(s) the shared initializer `OrtValue`s point into.
    ///
    /// Declared after the sessions and values so field drop order unmaps the
    /// file only after everything reading it is gone -- the values' data
    /// pointers are raw, so nothing else keeps this alive.
    #[expect(
        dead_code,
        reason = "the field is held only for its Drop, which field order keeps after the sessions and values it backs"
    )]
    weights: Vec<Mmap>,
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
        let mut maps = HashMap::<String, Mmap>::new();
        let mut table_values = |table: &WeightsFile| {
            if table.tensors.is_empty() {
                // The export externalizes every initializer, so an empty table
                // describes a manifest without tensors, not an all-inline one.
                return Err(LmError::Weights {
                    path: dir.join(&table.file),
                    reason: "the table names no tensors".to_owned(),
                });
            }
            shared_values(weights_map(&mut maps, dir, &table.file)?, table)
        };
        let (prefill_initializers, step_initializers) = match &manifest.weights {
            WeightsTable::Shared(table) => {
                let values = table_values(table)?;
                (values.clone(), values)
            }
            WeightsTable::PerGraph { step, prefill } => {
                (table_values(prefill)?, table_values(step)?)
            }
        };
        let model = Self {
            prefill_sessions: GraphSessions::new(dir.join("prefill.onnx"), prefill_initializers),
            step_sessions: GraphSessions::new(dir.join("charlm.onnx"), step_initializers),
            prefix: manifest.prefix,
            state: manifest.state,
            context_chars: manifest.context_chars,
            specials: manifest.specials,
            ids,
            alphabet,
            weights: maps.into_values().collect(),
        };
        // Open one session of each graph now, so `open` reports a graph the
        // runtime rejects instead of the first decode failing.
        model.prefill_sessions.session()?;
        model.step_sessions.session()?;
        Ok(model)
    }

    /// Read the prelude through `prefill.onnx`: the state every beam starts
    /// from, prefix tensors included.
    fn prefill(&self, context: Option<&str>) -> Result<LmState, LmError> {
        let prelude = self.prelude(context);
        let tokens: Vec<i64> = prelude.iter().copied().map(i64::from).collect();
        let length = dim(prelude.len());
        let mut session = self.prefill_sessions.session()?.borrow_mut();
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
        let mut session = self.step_sessions.session()?.borrow_mut();
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

/// The session options every session of a graph shares: one intra-op thread,
/// no parallel execution, no arena, no memory-pattern reservation -- the
/// decoder is the parallel layer, one record per thread, and a session that
/// also spread its matrix products over threads or held a private buffer
/// arena would oversubscribe the machine. Memory patterns pool a session's
/// activations into one reservation, which measured both slower and ~150 MB
/// heavier here than letting each buffer come and go.
fn session_builder() -> ort::Result<ort::session::builder::SessionBuilder> {
    Ok(Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_memory_pattern(false)?
        .with_parallel_execution(false)?
        .with_intra_threads(1)?)
}

/// Open a session for a shared graph. *weights* is the one container the
/// matrices MLAS pre-packs go into, shared by every session of the graph so
/// the packing happens once rather than per session.
fn open_session(
    graph: &Path,
    weights: &PrepackedWeights,
    initializers: &[(String, Arc<DynValue>)],
) -> ort::Result<Session> {
    let mut builder = session_builder()?
        .with_config_entry("session.enable_cpu_mem_arena", "0")?
        .with_prepacked_weights(weights)?;
    for (name, value) in initializers {
        builder = builder.with_initializer(name, Arc::clone(value))?;
    }
    builder.commit_from_file(graph)
}

/// Map *dir*`/`*file* once, or hand back the mapping already made.
fn weights_map<'m>(
    maps: &'m mut HashMap<String, Mmap>,
    dir: &Path,
    file: &str,
) -> Result<&'m Mmap, LmError> {
    if !maps.contains_key(file) {
        let path = dir.join(file);
        let opened = fs::File::open(&path).map_err(|source| LmError::Io {
            path: path.clone(),
            source,
        })?;
        // Safety: the file is the export's product, opened read-only and only
        // ever read through the map while a `CharLm` is alive.
        let map = unsafe { Mmap::map(&opened) }.map_err(|source| LmError::Io {
            path: path.clone(),
            source,
        })?;
        maps.insert(file.to_owned(), map);
    }
    Ok(maps.get(file).expect("inserted above"))
}

/// The shared `OrtValue`s of one weights file, built from its manifest
/// *table* rather than the graphs' protobufs: every tensor's name, shape and
/// byte extent is what the export recorded.
///
/// The values' data pointers address *map*; the caller (`CharLm`) holds the
/// mappings in a field declared after its sessions, so the map outlives every
/// value and session built here.
fn shared_values(map: &Mmap, table: &WeightsFile) -> Result<Vec<(String, Arc<DynValue>)>, LmError> {
    let mismatched = |reason: String| LmError::Weights {
        path: PathBuf::from(&table.file),
        reason,
    };
    if table.tensors.is_empty() {
        return Err(mismatched("the table names no tensors".to_owned()));
    }
    let info = MemoryInfo::new(
        AllocationDevice::CPU,
        0,
        AllocatorType::Device,
        MemoryType::Default,
    )?;
    let mut values = Vec::with_capacity(table.tensors.len());
    for tensor in &table.tensors {
        // Only the float tensors become shared `OrtValue`s; the runtime reads
        // any other dtype out of the mapped file itself.
        if tensor.dtype != WeightDtype::Float32 {
            continue;
        }
        let start = usize::try_from(tensor.offset)
            .map_err(|_| mismatched(format!("the offset of {} overflows usize", tensor.name)))?;
        let length = usize::try_from(tensor.length)
            .map_err(|_| mismatched(format!("the length of {} overflows usize", tensor.name)))?;
        let end = start.checked_add(length);
        if end.is_none_or(|end| map.get(start..end).is_none()) {
            return Err(mismatched(format!(
                "the extent of {} runs past {} bytes",
                tensor.name,
                map.len()
            )));
        }
        let elements = tensor.shape.iter().try_fold(1usize, |count, &axis| {
            count.checked_mul(usize::try_from(axis).ok()?)
        });
        if elements.is_none_or(|count| count * size_of::<f32>() != length) {
            return Err(mismatched(format!(
                "the extent of {} does not match its shape {:?}",
                tensor.name, tensor.shape
            )));
        }
        // Safety: the extent is bounds- and shape-checked above, and `map` is
        // held by `CharLm` in a field that drops after every session and value
        // built from it. The export pads nothing between tensors, so any
        // offset is `f32`-aligned because every extent is a whole number of
        // `f32`s.
        let data = unsafe { map.as_ptr().add(start) }
            .cast_mut()
            .cast::<std::ffi::c_void>();
        let view = unsafe {
            TensorRefMut::<f32>::from_raw(
                info.clone(),
                data,
                Shape::new(tensor.shape.iter().copied()),
            )
        }?;
        let ort_ptr = NonNull::new(AsPointer::ptr(&*view).cast_mut())
            .ok_or_else(|| mismatched(format!("{} has no underlying OrtValue", tensor.name)))?;
        // `from_raw` borrowed only in the type: the `OrtValue` is ours. Hand
        // its ownership to a `DynValue` that every session shares.
        std::mem::forget(view);
        let value = unsafe { DynValue::from_ptr(ort_ptr, None) };
        values.push((tensor.name.clone(), Arc::new(value)));
    }
    Ok(values)
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
