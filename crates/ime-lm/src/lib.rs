//! A character language model as the decoder's transition.
//!
//! The trigram scores a character against the two before it. The model here
//! scores it against everything before it: the sentence so far and, ahead of
//! that, the context that was on screen. `mlime train char-lm` trains it and
//! `mlime export char-lm` writes the files this crate loads:
//! `charlm.json`, a manifest that names the model's tensors and holds the
//! alphabet in id order, the two ONNX graphs the manifest describes, and the
//! weights file the graphs' initializers live in. The manifest's `layout`
//! names the step graph's batch layout, and a manifest without the one this
//! build reads is refused at open -- a graph written for another layout
//! would load and step until a broadcasting operator failed on it.
//! `prefill.onnx` reads the prelude (`<bos> context <sep>`) once and
//! produces the state every beam starts from; `charlm.onnx` advances a
//! batch of beams by one character and
//! returns the log probabilities of the next.
//!
//! A state is two kinds of tensor. The *prefix* tensors are what the prelude
//! produced -- the transformer's key/value cache over the context -- and every
//! beam of a record shares them; the *state* tensors are per beam (the LSTM's
//! `hidden`/`cell`, the transformer's cache over the sentence so far). Every
//! tensor is batch-first, and a step's batch is the surviving beams' rows laid
//! out as a rectangle: `width` consecutive rows per worker, so the prefix
//! feeds once per worker and broadcasts over the rows under it instead of
//! being packed per row. A beam's share of an output is one row, without the
//! crate ever interpreting what a row holds.
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

use ime_decode::{BeamOptions, MAX_HISTORY, Transition};
use ime_pinyin::{CharId, Lexicon};
use memmap2::Mmap;
use ort::AsPointer;
use ort::ep::ExecutionProviderDispatch;
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::builder::{GraphOptimizationLevel, PrepackedWeights};
use ort::session::{Session, SessionInputValue};
use ort::value::{DynValue, Shape, Tensor, TensorRefMut};
use serde::Deserialize;
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::Arc;
use thiserror::Error;
use thread_local::ThreadLocal;

/// Which execution provider a model's sessions run on.
///
/// `cpu` is the decoder's default and always built in; `coreml` (Apple's ANE/
/// GPU), `webgpu` (Dawn over Metal) and `cuda` (NVIDIA's provider) exist
/// behind the `gpu-coreml`, `gpu-webgpu` and `gpu-cuda` cargo features, and
/// asking for one that was not compiled in is an error at [`CharLm::open`],
/// never a silent CPU session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// ONNX Runtime's CPU kernels; the default.
    #[default]
    Cpu,
    /// Apple's Core ML provider, behind the `gpu-coreml` feature.
    CoreMl,
    /// WebGPU through Dawn (the Metal GPU), behind the `gpu-webgpu` feature.
    WebGpu,
    /// NVIDIA's CUDA provider, behind the `gpu-cuda` feature.
    Cuda,
}

impl Backend {
    /// The providers a session of this backend registers.
    ///
    /// # Errors
    ///
    /// If the backend's provider was not compiled into this build.
    fn providers(self) -> Result<Vec<ExecutionProviderDispatch>, LmError> {
        // Every dispatch carries `error_on_failure`: a provider that cannot
        // initialise is an error at session creation, never a silent CPU
        // session.
        match self {
            Self::Cpu => Ok(Vec::new()),
            #[cfg(feature = "gpu-coreml")]
            Self::CoreMl => Ok(vec![ort::ep::CoreML::default().build().error_on_failure()]),
            #[cfg(feature = "gpu-webgpu")]
            Self::WebGpu => Ok(vec![ort::ep::WebGPU::default().build().error_on_failure()]),
            #[cfg(feature = "gpu-cuda")]
            Self::Cuda => Ok(vec![ort::ep::CUDA::default().build().error_on_failure()]),
            #[allow(
                unreachable_patterns,
                reason = "the arm is reachable only when a gpu-* feature is off; with both on, every Backend variant already has an arm"
            )]
            _ => Err(LmError::NotCompiled { backend: self }),
        }
    }
}

/// How a model's sessions are built: the provider they register, the threads
/// one step spreads over, the width a step's batch is laid out at, and
/// whether the runtime logs verbosely.
///
/// The caller states the shape the decode will run, because the decoder is
/// the parallel layer and the session must not fight it: the per-record
/// rayon path gives every session one intra-op thread, while the lockstep
/// path runs one session for the whole batch and gives it the machine's
/// parallelism.
#[derive(Debug, Clone, Copy)]
pub struct SessionShape {
    /// The execution provider the sessions register.
    pub backend: Backend,
    /// The threads one session spreads a step over; the default is one, the
    /// rayon path's shape.
    pub intra_threads: NonZeroUsize,
    /// The rows a worker owns in a step's `[workers, width]` batch -- the
    /// beam width the caller decodes with, the decoder's own default -- so
    /// a worker's surviving beams fill their block and a wider one would
    /// have to be split over extra prefix rows.
    pub width: NonZeroUsize,
    /// ONNX Runtime's verbose session logging: the provider each node lands
    /// on is the evidence for whether a backend actually runs the step.
    pub verbose_logging: bool,
}

impl Default for SessionShape {
    fn default() -> Self {
        Self {
            backend: Backend::Cpu,
            intra_threads: NonZeroUsize::MIN,
            width: BeamOptions::default().beam_width,
            verbose_logging: false,
        }
    }
}

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
    /// How the sessions are built.
    shape: SessionShape,
}

impl GraphSessions {
    fn new(
        graph: PathBuf,
        initializers: Vec<(String, Arc<DynValue>)>,
        shape: SessionShape,
    ) -> Self {
        Self {
            by_thread: ThreadLocal::new(),
            path: graph,
            prepacked: PrepackedWeights::new(),
            initializers,
            shape,
        }
    }

    /// This thread's session, opened from the shared file on first use.
    ///
    /// Sessions may open concurrently: ONNX Runtime serializes the pre-packed
    /// weights lookups and writes itself -- `PrepackConstantInitializedTensors`
    /// holds `prepacked_weights_container_->mutex_` around them.
    fn session(&self) -> Result<&RefCell<Session>, LmError> {
        self.by_thread.get_or_try(|| {
            open_session(&self.path, &self.prepacked, &self.initializers, self.shape)
                .map(RefCell::new)
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
    /// The manifest's `layout` is not the one batch layout this build reads;
    /// a graph written for another would load and run until a broadcasting
    /// operator failed on it, so the check happens at open.
    #[error(
        "the export at {path} has a step-graph batch layout of {layout:?}, not \"rectangular\": re-export it with `mlime export char-lm`"
    )]
    Layout {
        /// The export's directory.
        path: PathBuf,
        /// What the manifest's `layout` held; `None` when it lacks the field.
        layout: Option<String>,
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
    /// The backend asked for a provider this build does not carry.
    #[error("the {backend:?} backend was not compiled in (missing its gpu-* cargo feature)")]
    NotCompiled {
        /// The backend that was asked for.
        backend: Backend,
    },
    /// The backend's provider was compiled in but could not initialise --
    /// CUDA asked for on a machine without the toolkit, the driver or the
    /// device, say.
    #[error("the {backend:?} backend's provider could not initialise")]
    Provider {
        /// The backend that was asked for.
        backend: Backend,
        /// The registration error.
        #[source]
        source: ort::Error,
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

/// The only step-graph batch layout this build reads; the manifest's
/// `layout` must hold it exactly.
const STEP_LAYOUT: &str = "rectangular";

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
    /// The step graph's batch layout; absent on exports written before the
    /// field existed, which is exactly what [`CharLm::open`] refuses.
    layout: Option<String>,
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
    /// Signed 8-bit integer, the dynamic-quantized `MatMul` weights' element.
    #[serde(rename = "int8")]
    Int8,
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
    /// The backend this model's sessions run on. A GPU session is one device
    /// per thread, so callers that parallelise records must check this and
    /// serialise instead.
    pub fn backend(&self) -> Backend {
        self.step_sessions.shape.backend
    }

    /// Open the model in *dir* (`charlm.json`, `prefill.onnx`, `charlm.onnx`)
    /// for *lexicon*, its sessions built to *shape*.
    ///
    /// # Errors
    ///
    /// If the manifest cannot be read, declares a batch layout other than
    /// `rectangular` (absent on exports written before the field existed),
    /// or names a field the export does not write, either graph cannot be
    /// loaded, a character of the lexicon has
    /// no row in the model's alphabet, or *shape*'s provider was not
    /// compiled into this build.
    pub fn open(dir: &Path, lexicon: &Lexicon, shape: SessionShape) -> Result<Self, LmError> {
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
        if manifest.layout.as_deref() != Some(STEP_LAYOUT) {
            return Err(LmError::Layout {
                path: dir.to_path_buf(),
                layout: manifest.layout,
            });
        }
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
            prefill_sessions: GraphSessions::new(
                dir.join("prefill.onnx"),
                prefill_initializers,
                shape,
            ),
            step_sessions: GraphSessions::new(dir.join("charlm.onnx"), step_initializers, shape),
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

    /// Advance a batch: `tokens[i]` applied to `states[i]`, whose rows may
    /// come from different records.
    ///
    /// The batch is fed as a `[workers, width]` rectangle. Every maximal run
    /// of consecutive rows sharing one prefix is one worker's -- the decoder
    /// keeps a worker's survivors contiguous, per the contract on
    /// [`Transition::advance`] -- cut into blocks of `SessionShape::width`
    /// rows, and each block padded out to `width` with dead rows (a valid
    /// state and its token, whose outputs are dropped on the way back). The
    /// prefix tensors take one row per block, padded on the left up to the
    /// widest prelude and masked by `prefix_mask`, and the graph broadcasts
    /// each over the block's rows: the prefix is packed once per worker,
    /// never per row. A model without prefix tensors (the LSTM) has no
    /// rectangle to fill: the batch is one flat run of live rows.
    fn step(&self, states: &[&LmState], tokens: &[u32]) -> Result<Vec<LmState>, LmError> {
        let batch = tokens.len();
        debug_assert_eq!(states.len(), batch, "one state per token");
        let width = self.step_sessions.shape.width.get();
        // `(first row, live rows, padded rows)` per block of the rectangle.
        let mut blocks: Vec<(usize, usize, usize)> = Vec::new();
        if self.prefix.is_empty() {
            blocks.push((0, batch, batch));
        } else {
            let mut row = 0;
            while row < batch {
                let prefix = &states[row].prefix;
                let mut end = row + 1;
                while end < batch && Arc::ptr_eq(&states[end].prefix, prefix) {
                    end += 1;
                }
                for start in (row..end).step_by(width) {
                    blocks.push((start, (end - start).min(width), width));
                }
                row = end;
            }
        }
        let rows: usize = blocks.iter().map(|block| block.2).sum();
        // The input row feeding a block's padded row `j`: its own row while
        // `j < live`, else the block's first row stands in as a dead row.
        let fed = |(first, live, _): (usize, usize, usize), j: usize| {
            first + if j < live { j } else { 0 }
        };
        let mut inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> =
            Vec::with_capacity(2 + self.prefix.len() + self.state.len());
        let mut padded_tokens: Vec<i64> = Vec::with_capacity(rows);
        for &block in &blocks {
            for j in 0..block.2 {
                padded_tokens.push(i64::from(tokens[fed(block, j)]));
            }
        }
        inputs.push((
            Cow::Borrowed("token"),
            Tensor::from_array((vec![dim(rows)], padded_tokens))?.into(),
        ));
        if !self.prefix.is_empty() {
            self.push_worker_prefixes(&mut inputs, states, &blocks)?;
        }
        // The per-beam tensors stack row by row, dead rows included; every
        // row of one tensor has the same shape at the same step.
        for (index, name) in self.state.iter().enumerate() {
            let row_shape = &states[0].tensors[index].shape;
            let mut data = Vec::with_capacity(rows * states[0].tensors[index].data.len());
            for &block in &blocks {
                for j in 0..block.2 {
                    let tensor = &states[fed(block, j)].tensors[index];
                    debug_assert_eq!(
                        row_shape, &tensor.shape,
                        "every row of {name} has the same shape at one step"
                    );
                    data.extend_from_slice(&tensor.data);
                }
            }
            let mut shape = Vec::with_capacity(row_shape.len() + 1);
            shape.push(dim(rows));
            shape.extend_from_slice(row_shape);
            inputs.push((
                Cow::Owned(name.clone()),
                Tensor::from_array((shape, data))?.into(),
            ));
        }
        let mut session = self.step_sessions.session()?.borrow_mut();
        let outputs = session.run(inputs)?;
        let (_, log_probs) = outputs["log_probs"].try_extract_tensor::<f32>()?;
        let vocabulary = log_probs.len() / rows;
        // Each `next_*` output is split into rows by its own shape: a row's
        // shape is whatever the graph produced, which for the transformer
        // grows one position on the time axis per step.
        let next: Vec<(Vec<i64>, Vec<f32>)> = self
            .state
            .iter()
            .map(|name| {
                let (shape, data) =
                    outputs[format!("next_{name}").as_str()].try_extract_tensor::<f32>()?;
                debug_assert_eq!(shape[0], dim(rows));
                Ok((shape[1..].to_vec(), data.to_vec()))
            })
            .collect::<Result<_, ort::Error>>()?;
        // Slice each block's live rows back out in input order, dropping the
        // dead rows' outputs.
        let mut advanced = Vec::with_capacity(batch);
        let mut base = 0;
        for &(first, live, block_rows) in &blocks {
            for j in 0..live {
                let row = base + j;
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
                advanced.push(LmState {
                    prefix: Arc::clone(&states[first + j].prefix),
                    tensors: Arc::new(tensors),
                    log_probs: Arc::new(
                        log_probs[row * vocabulary..(row + 1) * vocabulary].to_vec(),
                    ),
                });
            }
            base += block_rows;
        }
        Ok(advanced)
    }

    /// Feed the `prefix_*` tensors and `prefix_mask` of a batch laid out as
    /// `[workers, width]` blocks.
    ///
    /// Each tensor stacks one row per block -- one per worker -- every row's
    /// time axis (the second-to-last) padded on the left up to the widest
    /// prelude; the mask row is `false` under the padding and `true` under
    /// real positions. A block's row is the prefix its rows share, packed
    /// once, so a batch from one record is one row as it always was.
    fn push_worker_prefixes<'a>(
        &'a self,
        inputs: &mut Vec<(Cow<'a, str>, SessionInputValue<'a>)>,
        states: &[&LmState],
        blocks: &[(usize, usize, usize)],
    ) -> Result<(), LmError> {
        let workers = blocks.len();
        let reals: Vec<usize> = blocks
            .iter()
            .map(|&(first, _, _)| prefix_width(&states[first].prefix[0]))
            .collect();
        let width = reals
            .iter()
            .copied()
            .max()
            .expect("advance never runs an empty batch");
        let mut mask = vec![false; workers * width];
        for (worker, &real) in reals.iter().enumerate() {
            mask[worker * width + (width - real)..(worker + 1) * width].fill(true);
        }
        for (index, name) in self.prefix.iter().enumerate() {
            // A prefix tensor's row is [..., time, element]; the mask pads
            // and marks the time axis, second-to-last.
            let row_shape = &states[0].prefix[index].shape;
            let time_axis = row_shape.len() - 2;
            let unit = usize::try_from(row_shape[time_axis + 1]).expect("a shape axis is positive");
            let slab = elements(&row_shape[..time_axis]);
            let mut shape = Vec::with_capacity(row_shape.len() + 1);
            shape.push(dim(workers));
            for (axis, &value) in row_shape.iter().enumerate() {
                shape.push(if axis == time_axis { dim(width) } else { value });
            }
            let row_len = elements(&shape[1..]);
            let mut data = vec![0.0; workers * row_len];
            for (worker, &(first, _, _)) in blocks.iter().enumerate() {
                let source = &states[first].prefix[index];
                debug_assert_eq!(
                    &source.shape[..time_axis],
                    &row_shape[..time_axis],
                    "every worker's {name} agrees on the static axes"
                );
                debug_assert_eq!(
                    &source.shape[time_axis + 1..],
                    &row_shape[time_axis + 1..],
                    "every worker's {name} agrees on the static axes"
                );
                let real =
                    usize::try_from(source.shape[time_axis]).expect("a shape axis is positive");
                debug_assert_eq!(
                    real, reals[worker],
                    "a worker's prefix tensors share one width"
                );
                let pad = width - real;
                let mut dst = worker * row_len + pad * unit;
                let mut src = 0;
                for _ in 0..slab {
                    data[dst..dst + real * unit]
                        .copy_from_slice(&source.data[src..src + real * unit]);
                    dst += width * unit;
                    src += real * unit;
                }
            }
            inputs.push((
                Cow::Owned(name.clone()),
                Tensor::from_array((shape, data))?.into(),
            ));
        }
        inputs.push((
            Cow::Borrowed("prefix_mask"),
            Tensor::from_array(([dim(workers), dim(width)], mask))?.into(),
        ));
        Ok(())
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

    /// Relies on the row-order guarantee of [`Transition::advance`]: a
    /// worker's survivors arrive as one contiguous run, which `step` pads
    /// to `SessionShape::width` blocks sharing the worker's prefix row.
    ///
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

/// The session options every session of a graph shares, sized by *shape*:
/// *shape*'s intra-op threads, no parallel execution, no arena, no
/// memory-pattern reservation. The decoder is the parallel layer, so a
/// rayon-path session takes one intra-op thread -- a session that also
/// spread its matrix products over threads or held a private buffer arena
/// would oversubscribe the machine -- while a lockstep run's one session
/// takes them all. Memory patterns pool a session's activations into one
/// reservation, which measured both slower and ~150 MB heavier here than
/// letting each buffer come and go. `Level2` keeps every fusion the step
/// graph benefits from and measured a hair faster than `Level3`.
fn session_builder(shape: SessionShape) -> ort::Result<ort::session::builder::SessionBuilder> {
    let mut builder = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level2)?
        .with_memory_pattern(false)?
        .with_parallel_execution(false)?
        .with_intra_threads(shape.intra_threads.get())?
        .with_config_entry("session.enable_cpu_mem_arena", "0")?;
    if shape.verbose_logging {
        builder = builder.with_log_level(ort::logging::LogLevel::Verbose)?;
    }
    Ok(builder)
}

/// Open a session for a shared graph. *weights* is the one container the
/// matrices MLAS pre-packs go into, shared by every session of the graph so
/// the packing happens once rather than per session.
fn open_session(
    graph: &Path,
    weights: &PrepackedWeights,
    initializers: &[(String, Arc<DynValue>)],
    shape: SessionShape,
) -> Result<Session, LmError> {
    let providers = shape.backend.providers()?;
    let mut builder = session_builder(shape)?;
    builder = builder
        .with_prepacked_weights(weights)
        .map_err(ort::Error::from)?;
    if !providers.is_empty() {
        builder = builder
            .with_execution_providers(providers)
            .map_err(|source| LmError::Provider {
                backend: shape.backend,
                source: ort::Error::from(source),
            })?;
    }
    for (name, value) in initializers {
        builder = builder
            .with_initializer(name, Arc::clone(value))
            .map_err(ort::Error::from)?;
    }
    Ok(builder.commit_from_file(graph)?)
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
        // Only the element types the graphs read as weights become shared
        // `OrtValue`s; the runtime reads any other dtype out of the mapped
        // file itself.
        match tensor.dtype {
            WeightDtype::Float32 => {
                shared_value::<f32>(map, &info, tensor, &mismatched, &mut values)?;
            }
            WeightDtype::Int8 => {
                shared_value::<i8>(map, &info, tensor, &mismatched, &mut values)?;
            }
            WeightDtype::Int64 => {}
        }
    }
    Ok(values)
}

/// The shared `OrtValue` of one tensor of *table*: bounds- and shape-checked
/// against *map*, then handed to every session of the graph as a
/// `TensorRefMut` whose data pointer addresses the mapping.
fn shared_value<T: ort::value::PrimitiveTensorElementType + std::fmt::Debug>(
    map: &Mmap,
    info: &MemoryInfo,
    tensor: &WeightTensor,
    mismatched: &impl Fn(String) -> LmError,
    values: &mut Vec<(String, Arc<DynValue>)>,
) -> Result<(), LmError> {
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
    if elements.is_none_or(|count| count * size_of::<T>() != length) {
        return Err(mismatched(format!(
            "the extent of {} does not match its shape {:?}",
            tensor.name, tensor.shape
        )));
    }
    // Safety: the extent is bounds- and shape-checked above, and `map` is
    // held by `CharLm` in a field that drops after every session and value
    // built from it. The export writes the weights file largest-element
    // first, so every offset is a multiple of its tensor's element size and
    // the pointer is aligned for `T`.
    let data = unsafe { map.as_ptr().add(start) }
        .cast_mut()
        .cast::<std::ffi::c_void>();
    let view = unsafe {
        TensorRefMut::<T>::from_raw(info.clone(), data, Shape::new(tensor.shape.iter().copied()))
    }?;
    let ort_ptr = NonNull::new(AsPointer::ptr(&*view).cast_mut())
        .ok_or_else(|| mismatched(format!("{} has no underlying OrtValue", tensor.name)))?;
    // `from_raw` borrowed only in the type: the `OrtValue` is ours. Hand its
    // ownership to a `DynValue` that every session shares.
    std::mem::forget(view);
    let value = unsafe { DynValue::from_ptr(ort_ptr, None) };
    values.push((tensor.name.clone(), Arc::new(value)));
    Ok(())
}

/// A tensor dimension as ONNX Runtime spells it.
#[expect(
    clippy::cast_possible_wrap,
    reason = "a dimension is a beam width, a hidden size or a sequence length, none of which approaches i64::MAX"
)]
const fn dim(n: usize) -> i64 {
    n as i64
}

/// The width of a prefix tensor's time axis: its second-to-last.
fn prefix_width(tensor: &StateTensor) -> usize {
    usize::try_from(tensor.shape[tensor.shape.len() - 2]).expect("a shape axis is positive")
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
