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
//! `hidden`/`cell`, the transformer's cache over the sentence so far). Both
//! live on the session's device: `start` binds the prefill's outputs over
//! one resident slot's rows -- the graph emits the prefix left-padded to
//! the slot's width and the mask row beside it, so the run crosses nothing
//! but the prelude's tokens up and `log_probs` down -- and a step is fed
//! `token`, `source_row` (the resident row each new beam continues from)
//! and `prefix_row` (each worker's slot) -- the
//! graph gathers the caches by index on the device and produces the new
//! rows' tensors there too, so the only tensors that ever cross the bus are
//! the indices, the tokens and `log_probs`. A step's batch stays the
//! `[workers, width]` rectangle: `width` consecutive rows per worker sharing
//! the worker's slot, so the prefix gathers once per worker instead of once
//! per row.
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

use half::f16;
use ime_decode::{BeamOptions, MAX_HISTORY, Transition};
use ime_pinyin::{CharId, Lexicon};
use memmap2::Mmap;
use ort::AsPointer;
use ort::ep::ExecutionProviderDispatch;
use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
use ort::session::builder::{GraphOptimizationLevel, PrepackedWeights};
use ort::session::{IoBinding, Session};
use ort::value::{
    DynTensor, DynValue, Outlet, PrimitiveTensorElementType, Shape, SymbolicDimensions, Tensor,
    TensorElementType, TensorRefMut, ValueType,
};
use serde::Deserialize;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Debug;
use std::fs;
use std::mem::size_of;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
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

/// Read *dir*'s `charlm.json` and refuse the manifest shapes this build
/// cannot serve: a `layout` other than `resident`, or a `rows` table that
/// does not parallel the `prefix` and `state` name lists it gives the row
/// shapes of.
fn read_manifest(dir: &Path) -> Result<Manifest, LmError> {
    let manifest_path = dir.join("charlm.json");
    let raw = fs::read_to_string(&manifest_path).map_err(|source| LmError::Io {
        path: manifest_path.clone(),
        source,
    })?;
    let manifest: Manifest = serde_json::from_str(&raw).map_err(|source| LmError::Manifest {
        path: manifest_path,
        source,
    })?;
    if manifest.layout.as_deref() != Some(STEP_LAYOUT) {
        return Err(LmError::Layout {
            path: dir.to_path_buf(),
            layout: manifest.layout,
        });
    }
    for (rows, names) in [
        (&manifest.rows.prefix, &manifest.prefix),
        (&manifest.rows.state, &manifest.state),
    ] {
        if rows.len() != names.len() {
            return Err(LmError::Rows {
                path: dir.join("charlm.json"),
                rows: rows.len(),
                names: names.len(),
            });
        }
    }
    Ok(manifest)
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

/// A session plus the [`IoBinding`] every one of its runs is rebound
/// through: creating a binding allocates an `OrtIoBinding`, so one is made
/// with the session and `clear`ed per run rather than built per run.
#[derive(Debug)]
struct BoundSession {
    /// The session.
    session: Session,
    /// Its persistent binding; a run clears it, binds the run's tensors and
    /// calls `run_binding`.
    binding: IoBinding,
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
    by_thread: ThreadLocal<RefCell<BoundSession>>,
    /// The graph's file.
    path: PathBuf,
    /// The prepacked weights every session of this graph shares, `None`
    /// when the export's dtype has kernels that cannot fill a shared
    /// container -- fp16's `LayerNormalization` prepacks into a private one
    /// instead, and a shared container present at session creation makes
    /// `PrepackConstantInitializedTensors` fail the session's open.
    prepacked: Option<PrepackedWeights>,
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
        dtype: ModelDtype,
    ) -> Self {
        Self {
            by_thread: ThreadLocal::new(),
            path: graph,
            prepacked: (dtype == ModelDtype::Float32).then(PrepackedWeights::new),
            initializers,
            shape,
        }
    }

    /// This thread's session and binding, opened from the shared file on
    /// first use.
    ///
    /// Sessions may open concurrently: ONNX Runtime serializes the pre-packed
    /// weights lookups and writes itself -- `PrepackConstantInitializedTensors`
    /// holds `prepacked_weights_container_->mutex_` around them.
    fn session(&self) -> Result<&RefCell<BoundSession>, LmError> {
        self.by_thread.get_or_try(|| {
            let session = open_session(
                &self.path,
                self.prepacked.as_ref(),
                &self.initializers,
                self.shape,
            )?;
            let binding = session.create_binding()?;
            Ok(RefCell::new(BoundSession { session, binding }))
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
        "the export at {path} has a step-graph batch layout of {layout:?}, not \"resident\": re-export it with `mlime export char-lm`"
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
    /// The manifest's `rows` does not parallel the tensor names it is a row
    /// shape for.
    #[error("the manifest at {path} lists {rows} rows for {names} tensors")]
    Rows {
        /// The manifest's file.
        path: PathBuf,
        /// How many rows the manifest declared.
        rows: usize,
        /// How many names the matching list holds.
        names: usize,
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
const STEP_LAYOUT: &str = "resident";

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
    /// The shape one resident slot's row takes in each buffer, in the name
    /// order of `prefix` and `state`: the export emits its prefix rows
    /// already left-padded to the prelude width, so these rows are exactly
    /// what the pool allocates and the bound prefill writes into.
    rows: RowShapes,
    /// The step graph's batch layout; absent on exports written before the
    /// field existed, which is exactly what [`CharLm::open`] refuses.
    layout: Option<String>,
    /// The element type the graphs' prefix and state tensors hold; absent on
    /// exports written before the field existed, all of which the `layout`
    /// pin already refuses.
    dtype: Option<ModelDtype>,
    /// Where the graphs' initializers live: one shared file's table, or one
    /// table per graph when the step and prefill tensors have different names.
    weights: WeightsTable,
    specials: Specials,
    chars: Vec<String>,
}

/// The manifest's `rows`: each resident buffer's per-slot row shape, in the
/// order the `prefix` and `state` name lists give them.
#[derive(Debug, Deserialize)]
struct RowShapes {
    /// The prefix buffers' row shapes -- for the transformer,
    /// `[layers, heads, prelude width, head_dim]` per name.
    prefix: Vec<Vec<i64>>,
    /// The generation-zero buffers' row shapes.
    state: Vec<Vec<i64>>,
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
    /// IEEE-754 half precision, little-endian, the fp16 export's element.
    #[serde(rename = "float16")]
    Float16,
    /// Signed 8-bit integer, the dynamic-quantized `MatMul` weights' element.
    #[serde(rename = "int8")]
    Int8,
    /// Signed 64-bit integer, little-endian.
    #[serde(rename = "int64")]
    Int64,
}

/// The element type the graphs' prefix and state tensors hold: the
/// manifest's `dtype`, which an fp16 export sets to halve the bytes a step
/// moves and reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
enum ModelDtype {
    /// IEEE-754 single precision.
    #[default]
    #[serde(rename = "float32")]
    Float32,
    /// IEEE-754 half precision.
    #[serde(rename = "float16")]
    Float16,
}

impl ModelDtype {
    /// The ONNX Runtime element type it maps to.
    const fn element(self) -> TensorElementType {
        match self {
            Self::Float32 => TensorElementType::Float32,
            Self::Float16 => TensorElementType::Float16,
        }
    }
}

/// The tensor's shape, or an error when *value* is not a tensor.
fn value_shape(value: &DynValue) -> Result<Shape, LmError> {
    let ValueType::Tensor { shape, .. } = value.dtype() else {
        return Err(ort::Error::new("a state or prefix value is not a tensor").into());
    };
    Ok(shape.clone())
}

/// The tensors one step produced — every `next_*` output — kept on the
/// session's device.
///
/// A generation is written once, by the step that produced it, and only read
/// afterwards, so every beam it feeds shares the one allocation: each state's
/// [`RowRef`] names its own row.
#[derive(Debug)]
struct Generation {
    /// The step's `next_*` outputs in the manifest's `state` order, each
    /// `[rows, *row shape]` on the session's device.
    tensors: Vec<DynValue>,
}

/// A state's own row: which generation's tensors hold it and its index on
/// their batch axis.
///
/// The handle is the typed form of what the step graph's `source_row` feeds
/// `Gather` — a resident row, never a `usize` into memory the caller has to
/// size — and keeping the generation's [`Arc`] is what makes the row a valid
/// read.
#[derive(Clone, Debug)]
struct RowRef {
    /// The buffers this state's row lives in.
    generation: Arc<Generation>,
    /// The row's index on the tensors' batch axis.
    index: u32,
}

/// One worker's slot in a [`CharLm`]'s resident pool: the row whose prefix
/// tensors, mask and generation-zero state belong to its record.
///
/// `start` takes a slot and uploads the record's prelude into it once; every
/// beam of the record then reads the slot by index. Dropping the handle
/// returns the slot to the pool — a cold path taken per record, never per
/// step, so the pool's lock is never on the hot path's way.
#[derive(Debug)]
struct WorkerSlot {
    /// The pool the slot is drawn from; the [`Arc`] also keeps the pool's
    /// buffers alive for as long as any of the worker's states is.
    pool: Arc<RwLock<Resident>>,
    /// The worker's index into the pool's buffers.
    index: u32,
}

impl Drop for WorkerSlot {
    fn drop(&mut self) {
        self.pool
            .write()
            .expect("the resident pool is not poisoned")
            .free
            .push(self.index);
    }
}

/// A [`CharLm`]'s resident buffers: one tensor per prefix and per state name —
/// `[slots, *row shape]` on the session's device — plus the per-slot
/// `prefix_mask`, so slot *i* of every buffer is one worker's share.
///
/// The pool is shared across every thread the model runs on: a state's prefix
/// resolves through the buffers the pool holds at the step that reads it, and
/// growth copies the live slots into fresh buffers without changing their
/// indices, so a stale handle can never point at wrong bytes.
#[derive(Debug)]
struct Resident {
    /// How many slots the current buffers hold.
    capacity: usize,
    /// Slots nobody holds, free for the next `start`.
    free: Vec<u32>,
    /// The buffers; `None` until the first `start` establishes their shapes.
    bufs: Option<WorkerBufs>,
}

/// The resident buffers themselves.
#[derive(Debug)]
struct WorkerBufs {
    /// The `prefix_*` buffers, one per manifest name: `[slots, *row]` with
    /// the row's time axis grown to the widest prelude.
    prefix: Vec<DynValue>,
    /// `prefix_mask`, `[slots, prelude width]`: `false` under the padding,
    /// `true` under the positions a prelude filled. `None` on a model without
    /// prefix tensors (the LSTM).
    mask: Option<DynValue>,
    /// Generation zero: the state rows every beam of a slot starts from.
    /// `start` writes each state tensor's row of a slot once, and a beam's
    /// first `advance` gathers it by `source_row`.
    gen0: Arc<Generation>,
}

/// Where a session's resident buffers live and how the bound step reaches
/// them, cached per thread because the allocator belongs to that thread's
/// session.
struct DeviceEnv {
    /// The session's allocator for its device — every resident buffer is
    /// allocated through it, so none ever moves.
    allocator: Allocator,
    /// The buffers' device: the GPU on a GPU backend, the heap for the CPU
    /// and Core ML paths — "resident" is host memory there, which is what
    /// lets one code path serve every backend.
    device: MemoryInfo<'static>,
    /// Host memory, where `log_probs` is bound: the one tensor that leaves
    /// the device every step.
    host: MemoryInfo<'static>,
    /// How the thread's sessions are built — the copiers take the same
    /// backend and thread shape.
    shape: SessionShape,
    /// The model's own identity sessions, one per element type a resident
    /// buffer holds, opened lazily: a pool growth copies its buffers through
    /// them instead of `ort`'s per-copy session cache, whose sessions
    /// register a device provider each and were what ate the T4's VRAM a
    /// copy at a time.
    copiers: Vec<Copier>,
}

// Safety: `DeviceEnv` lives inside a `ThreadLocal` — it is created, read and
// dropped on the one thread that owns it and is never moved across threads,
// so the `NonNull` raw pointers in `Allocator`/`MemoryInfo` (and the copier
// sessions' internals) stay thread-local.
unsafe impl Send for DeviceEnv {}

impl DeviceEnv {
    /// The copier for *ty*, opened on first use.
    fn copier(&mut self, ty: TensorElementType) -> Result<&mut Copier, LmError> {
        if !self.copiers.iter().any(|copier| copier.ty == ty) {
            self.copiers.push(Copier::open(ty, self.shape)?);
        }
        Ok(self
            .copiers
            .iter_mut()
            .find(|copier| copier.ty == ty)
            .expect("the copier was just opened"))
    }
}

/// A one-node `Identity` session — `src` -> `dst` — the model owns, so a
/// copy between resident buffers never consults `ort`'s per-copy session
/// cache at all. It is built by the model editor at the backend's provider,
/// so a CUDA buffer copies on the device without a host round-trip, and
/// reused through its persistent binding.
#[derive(Debug)]
struct Copier {
    /// The element type this session copies.
    ty: TensorElementType,
    /// The `Identity` graph's session.
    session: Session,
    /// Its binding, cleared and rebound per copy.
    binding: IoBinding,
}

impl Copier {
    /// Open the flat-tensor copy session for *ty*: `Identity` over a
    /// one-dimensional input of any length, registered with *shape*'s
    /// backend so it runs where the buffers live.
    fn open(ty: TensorElementType, shape: SessionShape) -> Result<Self, LmError> {
        let mut graph = ort::editor::Graph::new()?;
        let flat = || ValueType::Tensor {
            ty,
            shape: Shape::new([-1]),
            dimension_symbols: SymbolicDimensions::empty(1),
        };
        graph.set_inputs([Outlet::new("src", flat())])?;
        graph.set_outputs([Outlet::new("dst", flat())])?;
        graph.add_node(ort::editor::Node::new(
            "Identity",
            ort::editor::ONNX_DOMAIN,
            "copy",
            ["src"],
            ["dst"],
            [],
        )?)?;
        let mut model =
            ort::editor::Model::new([ort::editor::Opset::new(ort::editor::ONNX_DOMAIN, 17)?])?;
        model.add_graph(graph)?;
        let providers = shape.backend.providers()?;
        let mut builder = session_builder(shape)?;
        if !providers.is_empty() {
            builder = builder
                .with_execution_providers(providers)
                .map_err(|source| LmError::Provider {
                    backend: shape.backend,
                    source: ort::Error::from(source),
                })?;
        }
        let session = model.into_session(&builder)?;
        let binding = session.create_binding()?;
        Ok(Self {
            ty,
            session,
            binding,
        })
    }
}

/// Which side of the run a byte crossed on: a `start` (once per record — the
/// prefill's tokens, the resident prefix/state writes, its log-probability
/// row) or a `step` (per `advance` — tokens, indices, log-probability rows).
/// The counters are attributed at the call site, so threads' `start`s can
/// never bleed into another thread's `step` total.
#[derive(Clone, Copy, Debug)]
enum Phase {
    /// The once-per-record `start` traffic.
    Start,
    /// The per-step `advance` traffic.
    Step,
}

/// A snapshot of the model's bus counters.
#[derive(Clone, Copy, Debug, Default)]
pub struct BusTotals {
    /// Host-input bytes at `start` (prefill tokens, resident writes).
    pub start_uploaded: u64,
    /// Host-input bytes at `step` (`token`, `source_row`, `prefix_row`).
    pub step_uploaded: u64,
    /// Downloaded bytes at `start` (the prefill's log-probability row).
    pub start_downloaded: u64,
    /// Downloaded bytes at `step` (`log_probs`).
    pub step_downloaded: u64,
}

/// Byte counters of what a model's sessions move across the host↔device bus,
/// so the pressure harness can prove a step crosses only `token`,
/// `source_row`, `prefix_row` and `log_probs`.
#[derive(Debug, Default)]
pub struct Bus {
    /// Host-side bytes per phase: start uploads, step uploads.
    uploaded: [AtomicU64; 2],
    /// Host-read bytes per phase: start download, step download.
    downloaded: [AtomicU64; 2],
}

impl Bus {
    /// The counters so far.
    #[must_use]
    pub fn totals(&self) -> BusTotals {
        BusTotals {
            start_uploaded: self.uploaded[Phase::Start as usize].load(Ordering::Relaxed),
            step_uploaded: self.uploaded[Phase::Step as usize].load(Ordering::Relaxed),
            start_downloaded: self.downloaded[Phase::Start as usize].load(Ordering::Relaxed),
            step_downloaded: self.downloaded[Phase::Step as usize].load(Ordering::Relaxed),
        }
    }

    /// Count *bytes* more host-side traffic for *phase*.
    fn upload(&self, phase: Phase, bytes: usize) {
        self.uploaded[phase as usize].fetch_add(bytes as u64, Ordering::Relaxed);
    }

    /// Count *bytes* more downloaded traffic for *phase*.
    fn download(&self, phase: Phase, bytes: usize) {
        self.downloaded[phase as usize].fetch_add(bytes as u64, Ordering::Relaxed);
    }
}

/// The model's state for one beam: the worker slot whose resident buffers
/// hold the record's prefix, the beam's own resident row, and the log
/// probabilities of whatever comes next.
///
/// All three are handles — the tensors themselves never leave the session's
/// device — so cloning a beam is reference-count bumps, and dropping the
/// last beam of a record returns its slot to the pool.
#[derive(Clone, Debug)]
pub struct LmState {
    /// The record's worker slot, shared by every beam the record keeps.
    worker: Arc<WorkerSlot>,
    /// This beam's own row in the step's resident buffers.
    row: RowRef,
    /// One log probability per id of the alphabet — the only tensor the host
    /// ever holds.
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

impl Resident {
    /// The resident buffers, allocated on *env*'s device with *rows*'
    /// shapes when this is the model's first `start`.
    ///
    /// *`prelude_width`* is the mask's axis — `<bos>` + `context_chars` +
    /// `<sep>`, the width the graph already pads each prefix row to.
    fn ensure(
        &mut self,
        env: &DeviceEnv,
        dtype: ModelDtype,
        rows: &RowShapes,
        prelude_width: usize,
    ) -> Result<(), LmError> {
        if self.bufs.is_some() {
            return Ok(());
        }
        let capacity = 64;
        let resident = |row: &[i64]| -> Result<DynValue, LmError> {
            let mut shape = Vec::with_capacity(row.len() + 1);
            shape.push(dim(capacity));
            shape.extend_from_slice(row);
            Ok(DynTensor::new(&env.allocator, dtype.element(), shape)?.into_dyn())
        };
        let mut prefix = Vec::with_capacity(rows.prefix.len());
        for row in &rows.prefix {
            prefix.push(resident(row)?);
        }
        let mask = if rows.prefix.is_empty() {
            None
        } else {
            Some(
                DynTensor::new(
                    &env.allocator,
                    TensorElementType::Bool,
                    [dim(capacity), dim(prelude_width)],
                )?
                .into_dyn(),
            )
        };
        let mut gen0 = Vec::with_capacity(rows.state.len());
        for row in &rows.state {
            gen0.push(resident(row)?);
        }
        self.bufs = Some(WorkerBufs {
            prefix,
            mask,
            gen0: Arc::new(Generation { tensors: gen0 }),
        });
        self.capacity = capacity;
        self.free = (0..u32::try_from(capacity).expect("the capacity fits u32")).collect();
        Ok(())
    }

    /// A free slot, growing the pool's buffers when none is.
    fn take(&mut self, env: &mut DeviceEnv) -> Result<u32, LmError> {
        if self.free.is_empty() {
            self.grow(env)?;
        }
        self.free.pop().ok_or_else(|| {
            ort::Error::new("the resident pool grew but handed no slot".to_owned()).into()
        })
    }

    /// Double the buffers, copying every used row forward so every index
    /// keeps meaning the same worker — one flat copy per buffer through the
    /// model's own copier sessions. The free slots' stale bytes come along
    /// and are simply overwritten when each next serves a worker.
    fn grow(&mut self, env: &mut DeviceEnv) -> Result<(), LmError> {
        let capacity = self.capacity * 2;
        let used = self.capacity;
        let bufs = self.bufs.as_mut().ok_or_else(|| {
            ort::Error::new("the resident pool has no buffers to grow".to_owned())
        })?;
        let enlarge = |old: &DynValue| -> Result<DynValue, LmError> {
            let mut shape = value_shape(old)?.to_vec();
            shape[0] = dim(capacity);
            let ValueType::Tensor { ty, .. } = old.dtype() else {
                return Err(ort::Error::new("a resident buffer is not a tensor".to_owned()).into());
            };
            Ok(DynTensor::new(&env.allocator, *ty, shape)?.into_dyn())
        };
        let mut pairs: Vec<(&DynValue, DynValue)> = Vec::new();
        for old in bufs.prefix.iter().chain(&bufs.gen0.tensors) {
            pairs.push((old, enlarge(old)?));
        }
        if let Some(old) = &bufs.mask {
            pairs.push((old, enlarge(old)?));
        }
        for (src, dst) in &pairs {
            let ValueType::Tensor { ty, shape, .. } = src.dtype() else {
                return Err(ort::Error::new("a resident buffer is not a tensor".to_owned()).into());
            };
            copy_flat(env, *ty, src, dst, used * elements(&shape[1..]))?;
        }
        let dsts: Vec<DynValue> = pairs.into_iter().map(|(_, dst)| dst).collect();
        let mut dsts = dsts.into_iter();
        bufs.prefix = dsts.by_ref().take(bufs.prefix.len()).collect();
        bufs.gen0 = Arc::new(Generation {
            tensors: dsts.by_ref().take(bufs.gen0.tensors.len()).collect(),
        });
        bufs.mask = dsts.next();
        self.free.extend(
            u32::try_from(self.capacity).expect("the capacity fits u32")
                ..u32::try_from(capacity).expect("the capacity fits u32"),
        );
        self.capacity = capacity;
        Ok(())
    }
}

/// A `Tensor<T>` view over *slot*'s `[1, *row]` row inside the resident
/// buffer *buf* — see [`tensor_view`], which this is a thin wrapper on.
fn row_view<T: PrimitiveTensorElementType + Debug>(
    buf: &DynValue,
    slot: u32,
    row: &[i64],
) -> Result<Tensor<T>, LmError> {
    let mut shape = Vec::with_capacity(row.len() + 1);
    shape.push(1);
    shape.extend_from_slice(row);
    tensor_view::<T>(buf, slot as usize * elements(row), &shape)
}

/// A `Tensor<T>` view over *shape*-sized memory inside the resident buffer
/// *buf*, starting *offset* elements in, labelled with the buffer's real
/// [`MemoryInfo`].
///
/// `TensorRefMut::from_raw` cannot be used here: it calls
/// `MemoryInfo::to_owned`, which in this `ort` release unconditionally
/// rewrites the info to CPU — a device buffer labelled CPU is then written
/// with a host `memcpy` into the device pointer, which faults. The view is
/// built through `ort::api` instead, so the runtime and its bound outputs
/// see the device the buffer really sits on.
fn tensor_view<T: PrimitiveTensorElementType + Debug>(
    buf: &DynValue,
    offset: usize,
    shape: &[i64],
) -> Result<Tensor<T>, LmError> {
    let bytes = elements(shape) * size_of::<T>();
    // Safety: the view spans exactly `bytes` of `buf` at `offset`, which
    // outlives it; the callers' aliasing guarantees are stated at each call
    // site. A zero-length view gets a dangling-but-aligned pointer, since a
    // buffer of zero elements has no `data_ptr` to offset.
    let data = if bytes == 0 {
        std::ptr::dangling_mut::<std::ffi::c_void>()
    } else {
        unsafe {
            buf.data_ptr()
                .cast::<u8>()
                .add(offset * size_of::<T>())
                .cast_mut()
                .cast::<std::ffi::c_void>()
        }
    };
    let mut value = std::ptr::null_mut();
    // Safety: `data` is `bytes` of valid memory at `offset`, `shape`
    // describes it, and `value` is an out-pointer the call fills.
    unsafe {
        ort::Error::result_from_status((ort::api().CreateTensorWithDataAsOrtValue)(
            AsPointer::ptr(buf.memory_info()),
            data,
            bytes,
            shape.as_ptr(),
            shape.len(),
            T::into_tensor_element_type().into(),
            &raw mut value,
        ))?;
    }
    let value = NonNull::new(value)
        .ok_or_else(|| ort::Error::new("the view came back without a value".to_owned()))?;
    // Safety: `value` is a valid OrtValue the call just created, and it is
    // not a session output — dropping it releases the view, not the buffer.
    Ok(unsafe { Tensor::from_ptr(value, None) })
}

/// A `[1, *row]` view of *slot*'s row inside *buf*, element-typed to the
/// model's dtype — what a bound prefill output lands in.
fn resident_view(buf: &DynValue, slot: u32, dtype: ModelDtype) -> Result<DynValue, LmError> {
    let row = value_shape(buf)?[1..].to_vec();
    Ok(match dtype {
        ModelDtype::Float32 => row_view::<f32>(buf, slot, &row)?.into_dyn(),
        ModelDtype::Float16 => row_view::<f16>(buf, slot, &row)?.into_dyn(),
    })
}

/// Copy *count* elements from the start of *src* to the start of *dst* as
/// one flat run — what `grow` uses per buffer — through the element type's
/// copier session.
fn copy_flat(
    env: &mut DeviceEnv,
    ty: TensorElementType,
    src: &DynValue,
    dst: &DynValue,
    count: usize,
) -> Result<(), LmError> {
    match ty {
        TensorElementType::Float32 => copy_flat_typed::<f32>(env, src, dst, count),
        TensorElementType::Float16 => copy_flat_typed::<f16>(env, src, dst, count),
        TensorElementType::Bool => copy_flat_typed::<bool>(env, src, dst, count),
        other => {
            Err(ort::Error::new(format!("a resident buffer of {other:?} has no copier")).into())
        }
    }
}

/// The element-typed form of [`copy_flat`].
fn copy_flat_typed<T: PrimitiveTensorElementType + Debug>(
    env: &mut DeviceEnv,
    src: &DynValue,
    dst: &DynValue,
    count: usize,
) -> Result<(), LmError> {
    if count == 0 {
        return Ok(());
    }
    // Safety: the views span the first `count` elements of each buffer,
    // which `grow` owns for the copy's duration — the source rows may be
    // aliased by live states, but the copier only ever reads them.
    let src = tensor_view::<T>(src, 0, &[dim(count)])?;
    let dst = tensor_view::<T>(dst, 0, &[dim(count)])?;
    let copier = env.copier(T::into_tensor_element_type())?;
    copier.binding.clear();
    copier.binding.bind_input("src", &src)?;
    copier.binding.bind_output("dst", dst)?;
    let Copier {
        session, binding, ..
    } = copier;
    session.run_binding(binding)?;
    binding.synchronize_outputs()?;
    binding.clear();
    Ok(())
}

/// A trained character language model, ready to score.
pub struct CharLm {
    /// `prefill.onnx`: the per-thread sessions over it.
    prefill_sessions: GraphSessions,
    /// `charlm.onnx`, the step graph: the per-thread sessions over it.
    step_sessions: GraphSessions,
    /// The resident slot pool shared across every thread's sessions.
    workers: Arc<RwLock<Resident>>,
    /// The per-thread device environments: allocator, device and host memory
    /// infos.
    devices: ThreadLocal<RefCell<DeviceEnv>>,
    /// The manifest's `prefix`: names of the shared tensors.
    prefix: Vec<String>,
    /// The manifest's `state`: names of the per-beam tensors, whose outputs the
    /// step graph returns as `next_<name>`.
    state: Vec<String>,
    /// The manifest's `rows`: each resident buffer's per-slot row shape, in
    /// the name order of `prefix` and `state`.
    rows: RowShapes,
    /// The manifest's `dtype`: the element type the prefix and state tensors
    /// hold.
    dtype: ModelDtype,
    /// The host↔device byte counters the pressure harness reports.
    bus: Bus,
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

    /// The model's bus counters — what the sessions have moved across the
    /// host↔device boundary so far.
    pub fn bus(&self) -> &Bus {
        &self.bus
    }

    /// Open the model in *dir* (`charlm.json`, `prefill.onnx`, `charlm.onnx`)
    /// for *lexicon*, its sessions built to *shape*.
    ///
    /// # Errors
    ///
    /// If the manifest cannot be read, declares a batch layout other than
    /// `resident` (absent on exports written before the field existed),
    /// or names a field the export does not write, either graph cannot be
    /// loaded, a character of the lexicon has
    /// no row in the model's alphabet, or *shape*'s provider was not
    /// compiled into this build.
    pub fn open(dir: &Path, lexicon: &Lexicon, shape: SessionShape) -> Result<Self, LmError> {
        let manifest = read_manifest(dir)?;
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
        let dtype = manifest.dtype.unwrap_or_default();
        let model = Self {
            prefill_sessions: GraphSessions::new(
                dir.join("prefill.onnx"),
                prefill_initializers,
                shape,
                dtype,
            ),
            step_sessions: GraphSessions::new(
                dir.join("charlm.onnx"),
                step_initializers,
                shape,
                dtype,
            ),
            workers: Arc::new(RwLock::new(Resident {
                capacity: 0,
                free: Vec::new(),
                bufs: None,
            })),
            devices: ThreadLocal::new(),
            prefix: manifest.prefix,
            state: manifest.state,
            rows: manifest.rows,
            dtype,
            bus: Bus::default(),
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

    /// This thread's device environment: the allocator the resident buffers
    /// are allocated through, the memory infos the bound step needs, and the
    /// copier sessions a pool growth uses.
    fn device_env(&self) -> Result<&RefCell<DeviceEnv>, LmError> {
        self.devices.get_or_try(|| {
            let session = self.step_sessions.session()?;
            let device = match self.step_sessions.shape.backend {
                Backend::Cpu | Backend::CoreMl => AllocationDevice::CPU,
                Backend::WebGpu => AllocationDevice::WEBGPU_BUFFER,
                Backend::Cuda => AllocationDevice::CUDA,
            };
            let device = MemoryInfo::new(device, 0, AllocatorType::Device, MemoryType::Default)?;
            let host = MemoryInfo::new(
                AllocationDevice::CPU,
                0,
                AllocatorType::Device,
                MemoryType::Default,
            )?;
            let allocator = Allocator::new(&session.borrow().session, device.clone())?;
            Ok(RefCell::new(DeviceEnv {
                allocator,
                device,
                host,
                shape: self.step_sessions.shape,
                copiers: Vec::new(),
            }))
        })
    }

    /// Read the prelude through `prefill.onnx` into a resident slot: the
    /// state every beam starts from — a handle, since the tensors themselves
    /// stay on the session's device.
    ///
    /// The slot is taken before the run because the graph's outputs are
    /// bound over its rows: `ensure`/`take` hold the pool's write lock, the
    /// bound run then holds only the read lock, so a concurrent `take` that
    /// has to grow the pool cannot move the buffers out from under the
    /// bound row views.
    fn prefill(&self, context: Option<&str>) -> Result<LmState, LmError> {
        let prelude = self.prelude(context);
        let tokens: Vec<i64> = prelude.iter().copied().map(i64::from).collect();
        let tokens = Tensor::from_array(([1i64, dim(tokens.len())], tokens))?;
        self.bus
            .upload(Phase::Start, elements(tokens.shape()) * size_of::<i64>());
        let slot = {
            let env = self.device_env()?;
            let mut pool = self
                .workers
                .write()
                .expect("the resident pool is not poisoned");
            pool.ensure(
                &env.borrow(),
                self.dtype,
                &self.rows,
                self.context_chars + 2,
            )?;
            pool.take(&mut env.borrow_mut())?
        };
        let (log_probs, generation) = match self.prefill_into(&tokens, slot) {
            Ok(parts) => parts,
            Err(error) => {
                // The run may have half-written the slot's rows; handing it
                // back is safe — the next `start` that takes it rewrites
                // every row in full.
                self.workers
                    .write()
                    .expect("the resident pool is not poisoned")
                    .free
                    .push(slot);
                return Err(error);
            }
        };
        Ok(LmState {
            worker: Arc::new(WorkerSlot {
                pool: Arc::clone(&self.workers),
                index: slot,
            }),
            row: RowRef {
                generation,
                index: slot,
            },
            log_probs: Arc::new(log_probs),
        })
    }

    /// Run the prefill into *slot*'s rows: every prefix, mask and state
    /// output is bound over the slot's row of its resident buffer, so the
    /// only tensors to cross the bus are `tokens` up and `log_probs` down.
    fn prefill_into(
        &self,
        tokens: &Tensor<i64>,
        slot: u32,
    ) -> Result<(Vec<f32>, Arc<Generation>), LmError> {
        let mut bound = self.prefill_sessions.session()?.borrow_mut();
        let BoundSession { session, binding } = &mut *bound;
        binding.clear();
        binding.bind_input("tokens", tokens)?;
        let pool = self
            .workers
            .read()
            .expect("the resident pool is not poisoned");
        let env = self.device_env()?.borrow();
        let bufs = pool
            .bufs
            .as_ref()
            .expect("the slot's buffers were ensured before it was taken");
        for (index, name) in self.prefix.iter().enumerate() {
            binding.bind_output(
                name.as_str(),
                resident_view(&bufs.prefix[index], slot, self.dtype)?,
            )?;
        }
        if let Some(mask) = &bufs.mask {
            let width = value_shape(mask)?[1];
            binding.bind_output(
                "prefix_mask",
                row_view::<bool>(mask, slot, &[width])?.into_dyn(),
            )?;
        }
        for (index, name) in self.state.iter().enumerate() {
            binding.bind_output(
                name.as_str(),
                resident_view(&bufs.gen0.tensors[index], slot, self.dtype)?,
            )?;
        }
        binding.bind_output_to_device("log_probs", &env.host)?;
        let outputs = session.run_binding(binding)?;
        binding.synchronize_outputs()?;
        let log_probs = self.host_f32(&outputs["log_probs"], Phase::Start)?;
        Ok((log_probs, Arc::clone(&bufs.gen0)))
    }

    /// The tensor's `f32` data on the host — the bound `log_probs` output,
    /// the only thing the bus ever carries.
    fn host_f32(&self, value: &DynValue, phase: Phase) -> Result<Vec<f32>, LmError> {
        debug_assert!(
            value.memory_info().is_cpu_accessible(),
            "log_probs is bound to host memory"
        );
        let (shape, data) = value.try_extract_tensor::<f32>()?;
        self.bus
            .download(phase, shape.num_elements() * size_of::<f32>());
        Ok(data.to_vec())
    }

    /// Advance the runs of one `advance` call, one `run_binding` per
    /// generation the steps' rows live in.
    ///
    /// `decode_many`'s contract keeps every survivor of one position on the
    /// same step's outputs, so a real decode lands here as a single run; a
    /// caller that mixes states across steps (a test's interleave, say) gets
    /// one bound step per contiguous run — slower, still correct, because
    /// every run binds the buffers its own `RowRef`s name.
    fn step(&self, steps: &[(&LmState, CharId)]) -> Result<Vec<LmState>, LmError> {
        let mut advanced: Vec<Option<LmState>> = (0..steps.len()).map(|_| None).collect();
        let mut begin = 0;
        while begin < steps.len() {
            let mut end = begin + 1;
            while end < steps.len()
                && Arc::ptr_eq(&steps[end].0.row.generation, &steps[begin].0.row.generation)
            {
                end += 1;
            }
            self.step_gen(&steps[begin..end], &mut advanced[begin..end])?;
            begin = end;
        }
        Ok(advanced
            .into_iter()
            .map(|state| state.expect("every row is advanced by its run"))
            .collect())
    }

    /// One bound step over rows all living in the same generation.
    ///
    /// The batch stays a `[workers, width]` rectangle: every maximal run of
    /// consecutive rows sharing one worker slot is cut into blocks of
    /// `SessionShape::width` rows, each padded out to `width` with dead rows
    /// — a valid source row and token whose outputs are dropped on the way
    /// back. The step binds the resident buffers by name: `token` and
    /// `source_row` are the only host inputs, `prefix_row` gathers each
    /// block's prefix slot on the device, and `log_probs` is bound to host
    /// memory while the `next_*` outputs stay on the device as the next
    /// generation. A model without prefix tensors (the LSTM) has no
    /// rectangle to fill: the batch is one flat run of live rows.
    #[expect(
        clippy::too_many_lines,
        reason = "one bound step is a straight-line sequence: block the rows, feed the dead ones, build the inputs, bind, run, scatter the states"
    )]
    fn step_gen(
        &self,
        steps: &[(&LmState, CharId)],
        out: &mut [Option<LmState>],
    ) -> Result<(), LmError> {
        let generation = &steps[0].0.row.generation;
        let width = self.step_sessions.shape.width.get();
        // `(first row, live rows, padded rows)` per block of the rectangle.
        let mut blocks: Vec<(usize, usize, usize)> = Vec::new();
        if self.prefix.is_empty() {
            blocks.push((0, steps.len(), steps.len()));
        } else {
            let mut row = 0;
            while row < steps.len() {
                let worker = &steps[row].0.worker;
                let mut end = row + 1;
                while end < steps.len() && Arc::ptr_eq(&steps[end].0.worker, worker) {
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
        let mut token = Vec::with_capacity(rows);
        let mut source_row = Vec::with_capacity(rows);
        let mut prefix_row = Vec::with_capacity(blocks.len());
        let mut uploaded = 0usize;
        for &block in &blocks {
            for j in 0..block.2 {
                let row = fed(block, j);
                token.push(i64::from(self.ids[steps[row].1.index()]));
                source_row.push(i64::from(steps[row].0.row.index));
            }
            if !self.prefix.is_empty() {
                prefix_row.push(i64::from(steps[block.0].0.worker.index));
            }
        }
        uploaded += (token.len() + source_row.len() + prefix_row.len()) * size_of::<i64>();
        self.bus.upload(Phase::Step, uploaded);
        let token = Tensor::from_array((vec![dim(rows)], token))?;
        let source_row = Tensor::from_array((vec![dim(rows)], source_row))?;
        let prefix_row = if self.prefix.is_empty() {
            None
        } else {
            Some(Tensor::from_array((vec![dim(blocks.len())], prefix_row))?)
        };
        let env = self.device_env()?.borrow();
        let mut bound = self.step_sessions.session()?.borrow_mut();
        let BoundSession { session, binding } = &mut *bound;
        binding.clear();
        binding.bind_input("token", &token)?;
        binding.bind_input("source_row", &source_row)?;
        let pool = self
            .workers
            .read()
            .expect("the resident pool is not poisoned");
        {
            let bufs = pool
                .bufs
                .as_ref()
                .expect("a state exists only after a start allocated the buffers");
            if let Some(prefix_row) = &prefix_row {
                binding.bind_input("prefix_row", prefix_row)?;
                binding.bind_input(
                    "prefix_mask",
                    bufs.mask.as_ref().expect("a prefix model owns the mask"),
                )?;
                for (index, name) in self.prefix.iter().enumerate() {
                    binding.bind_input(name.as_str(), &bufs.prefix[index])?;
                }
            }
            for (index, name) in self.state.iter().enumerate() {
                binding.bind_input(name.as_str(), &generation.tensors[index])?;
            }
        }
        drop(pool);
        binding.bind_output_to_device("log_probs", &env.host)?;
        for name in &self.state {
            binding.bind_output_to_device(format!("next_{name}"), &env.device)?;
        }
        let mut outputs = session.run_binding(binding)?;
        binding.synchronize_outputs()?;
        // The next generation owns the `next_*` rows this step wrote; the
        // states it feeds point back at them by index.
        let mut tensors = Vec::with_capacity(self.state.len());
        for name in &self.state {
            tensors.push(
                outputs
                    .remove(format!("next_{name}").as_str())
                    .expect("the step graph returns every next_ output"),
            );
        }
        let next = Arc::new(Generation { tensors });
        let log_probs = self.host_f32(&outputs["log_probs"], Phase::Step)?;
        let vocabulary = log_probs.len() / rows;
        // Slice each block's live rows back out in input order, dropping the
        // dead rows' outputs.
        let mut base = 0;
        for &(first, live, block_rows) in &blocks {
            for j in 0..live {
                let row = base + j;
                out[first + j] = Some(LmState {
                    worker: Arc::clone(&steps[first + j].0.worker),
                    row: RowRef {
                        generation: Arc::clone(&next),
                        index: u32::try_from(row).expect("a step's rows fit u32"),
                    },
                    log_probs: Arc::new(
                        log_probs[row * vocabulary..(row + 1) * vocabulary].to_vec(),
                    ),
                });
            }
            base += block_rows;
        }
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
    /// worker's survivors arrive as one contiguous run, which `step` groups
    /// into `SessionShape::width` blocks sharing the worker's resident slot.
    ///
    /// # Panics
    ///
    /// If ONNX Runtime fails a run; see [`CharLm::start`].
    fn advance(&self, steps: &[(&LmState, CharId)]) -> Vec<LmState> {
        if steps.is_empty() {
            return Vec::new();
        }
        self.step(steps).expect("the graph runs on a batch")
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

/// Open a session for a shared graph. *weights*, when present, is the one
/// container the matrices MLAS pre-packs go into, shared by every session of
/// the graph so the packing happens once rather than per session.
fn open_session(
    graph: &Path,
    weights: Option<&PrepackedWeights>,
    initializers: &[(String, Arc<DynValue>)],
    shape: SessionShape,
) -> Result<Session, LmError> {
    let providers = shape.backend.providers()?;
    let mut builder = session_builder(shape)?;
    if let Some(weights) = weights {
        builder = builder
            .with_prepacked_weights(weights)
            .map_err(ort::Error::from)?;
    }
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
            WeightDtype::Float16 => {
                shared_value::<f16>(map, &info, tensor, &mismatched, &mut values)?;
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
