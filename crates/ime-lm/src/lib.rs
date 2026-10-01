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
//! batch of beams by one character and returns the log probabilities of
//! the candidates the next position admits.
//!
//! A state is two kinds of tensor. The *prefix* tensors are what the prelude
//! produced -- the transformer's key/value cache over the context -- and every
//! beam of a record shares them; the *state* tensors are per beam (the LSTM's
//! `hidden`/`cell`, the transformer's cache over the sentence so far). Both
//! live on the session's device: `start` binds the prefill's outputs over
//! one resident slot's rows -- the graph emits the prefix left-padded to
//! the slot's width and the mask row beside it, so the run crosses nothing
//! but the prelude's tokens and the first position's `candidates` up and
//! `candidate_log_probs` down -- and a step is fed `token`, `source_row`
//! (the resident row each new beam continues from), `candidates` (the
//! alphabet ids each produced row will be scored on, `<eos>` among them
//! where the path can end) and `prefix_row` (each worker's slot) -- the
//! graph gathers the caches by index on the device, gathers the
//! log-softmax at each row's candidates there too, and produces the new
//! rows' tensors there as well, so the only tensors that ever cross the
//! bus are the indices, the tokens, the candidate ids and their gathered
//! scores. A step's batch stays the
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
use hashbrown::HashMap;
use ime_decode::{Asked, BeamOptions, MAX_HISTORY, Transition};
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
use std::collections::VecDeque;
use std::fmt::Debug;
use std::fs;
use std::mem::size_of;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock, Weak};

use thiserror::Error;
use thread_local::ThreadLocal;

#[cfg(target_os = "macos")]
mod metal;
#[cfg(target_os = "macos")]
pub use metal::StepTiming;

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
    /// Hand-encoded Metal kernels for the paged step — no ONNX session —
    /// on macOS only. The prefill stays an ORT CPU session; the step itself
    /// is [`metal::MetalStep`].
    #[cfg(target_os = "macos")]
    Metal,
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
/// cannot serve: a `layout` other than `resident-pages`, or a `rows` table that
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
    /// The pages the resident pool holds — live states plus the radix
    /// cache's nodes, one page per position of a paged export — or, for a
    /// row-mode export (the LSTM), the cached rows. A paged export floors
    /// the pool at a working set since a context's own pages are claims on
    /// the same list; a row-mode export of zero leaves the cache empty.
    pub cache_rows: usize,
    /// The precision the Metal step packs its matmul weights in — ignored on
    /// every other backend.
    pub metal_weights: MetalWeights,
}

impl Default for SessionShape {
    fn default() -> Self {
        Self {
            backend: Backend::Cpu,
            intra_threads: NonZeroUsize::MIN,
            width: BeamOptions::default().beam_width,
            verbose_logging: false,
            cache_rows: 0,
            metal_weights: MetalWeights::Auto,
        }
    }
}

/// The precision the Metal step packs its matmul weights in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MetalWeights {
    /// The export's own dtype: int8 where the table is int8, fp32 otherwise.
    #[default]
    Auto,
    /// fp16 — dequantized or cast at pack; doubles the int8 slabs but drops
    /// the per-channel-scale multiply from every matmul in the kernel.
    F16,
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
        "the export at {path} has a step-graph batch layout of {layout:?}, not \"resident-pages\": re-export it with `mlime export char-lm`"
    )]
    Layout {
        /// The export's directory.
        path: PathBuf,
        /// What the manifest's `layout` held; `None` when it lacks the field.
        layout: Option<String>,
    },
    /// The manifest's state rows mark a page axis that is not the first —
    /// the only pool layout this loader reads is `[pages, layers, heads,
    /// head_dim]`, and the older order would bind a differently-shaped
    /// buffer into the step graph.
    #[error(
        "the export's state rows put the page axis at position {axis}, not 0 — re-export it with `mlime export char-lm`"
    )]
    PageAxis {
        /// The zero axis's position in a state row.
        axis: usize,
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
    /// No Metal device answered `MTLCreateSystemDefaultDevice`.
    #[cfg(target_os = "macos")]
    #[error("no Metal device is available on this machine")]
    MetalUnavailable,
    /// The Metal kernels or buffers could not be built.
    #[cfg(target_os = "macos")]
    #[error("the Metal step could not be built: {0}")]
    Metal(String),
}

/// The reserved ids, as the manifest names them.
#[derive(Debug, Clone, Copy, Deserialize)]
struct Specials {
    /// The padding id — also the filler for `candidates` rows shorter than
    /// the batch's widest request, a real column the host never reads back.
    pad: u32,
    bos: u32,
    eos: u32,
    sep: u32,
    unk: u32,
}

/// The only step-graph batch layout this build reads; the manifest's
/// `layout` must hold it exactly.
const STEP_LAYOUT: &str = "resident-pages";

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
    /// This thread's `state_*` scratch generation — the pending rows'
    /// materialisation target. It lives in the per-thread env so each
    /// `step_pages` call owns its scratch outright: one generation handed
    /// to concurrent `advance` calls was a data race the moment the CPU
    /// `sections` path ran them on one `CharLm` across rayon threads.
    /// Declared before `allocator` so the tensors free while the allocator
    /// they were made through still lives.
    scratch: Option<Scratch>,
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
    /// Whether the resident buffers are plain heap — the CPU and Core ML
    /// paths, where a copy is a `memcpy` and the copier session is skipped.
    host_resident: bool,
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

    /// Zero *count* elements of *dst* — a `memset` where the resident
    /// buffers are heap, a copy of a zeroed host staging tensor through
    /// this thread's copier where they are not. Masked tails would
    /// otherwise carry whatever bytes the allocator recycled into
    /// `probs @ values`: a pad's scores mask to 0 exactly, but
    /// `0 * NaN = NaN`.
    fn zero(&mut self, ty: TensorElementType, dst: &DynValue, count: usize) -> Result<(), LmError> {
        let element_size = match ty {
            TensorElementType::Float32 => size_of::<f32>(),
            TensorElementType::Float16 => size_of::<f16>(),
            TensorElementType::Bool => size_of::<bool>(),
            other => {
                return Err(ort::Error::new(format!(
                    "a resident buffer of {other:?} has no zeroing"
                ))
                .into());
            }
        };
        if count == 0 {
            return Ok(());
        }
        if self.host_resident {
            // Safety: `count` elements are inside `dst`, which the
            // callers bound to a whole tensor's extent.
            unsafe {
                dst.data_ptr()
                    .cast::<u8>()
                    .cast_mut()
                    .write_bytes(0, count * element_size);
            }
            return Ok(());
        }
        // A CPU `src` crosses to the device through ORT's own input
        // memcpy at the graph boundary — the copier is the session's
        // provider, so the `Identity` runs where the buffer lives.
        match ty {
            TensorElementType::Float32 => {
                let src = Tensor::from_array((vec![dim(count)], vec![0.0_f32; count]))?.into_dyn();
                device_copy::<f32>(self, &src, 0, dst, 0, count)
            }
            TensorElementType::Float16 => {
                let src =
                    Tensor::from_array((vec![dim(count)], vec![f16::ZERO; count]))?.into_dyn();
                device_copy::<f16>(self, &src, 0, dst, 0, count)
            }
            TensorElementType::Bool => {
                let src = Tensor::from_array((vec![dim(count)], vec![false; count]))?.into_dyn();
                device_copy::<bool>(self, &src, 0, dst, 0, count)
            }
            other => unreachable!("{other:?} was refused above"),
        }
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
/// prefill's tokens and candidates, the resident prefix/state writes, its
/// gathered score row) or a `step` (per `advance` — tokens, indices,
/// candidate ids, gathered score rows).
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
    /// Host-input bytes at `step` (`token`, `source_row`, `candidates`,
    /// `prefix_row`).
    pub step_uploaded: u64,
    /// Downloaded bytes at `start` (the prefill's gathered score row).
    pub start_downloaded: u64,
    /// Downloaded bytes at `step` (`candidate_log_probs`).
    pub step_downloaded: u64,
}

/// Byte counters of what a model's sessions move across the host↔device bus,
/// so the pressure harness can prove a step crosses only `token`,
/// `source_row`, `candidates`, `prefix_row` and `candidate_log_probs`.
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

/// The gathered scores a produced state answers: the alphabet ids the call
/// that made it asked for — the next position's candidates, `<eos>` among
/// them where the path can end — sorted ascending, each with its log
/// probability. A score of anything else is a bug in the caller, so the
/// lookup fails fast instead of falling back to a row the model never
/// sent.
#[derive(Clone, Debug)]
struct Scores {
    /// The request's alphabet ids, sorted — also the order the row of the
    /// `candidates` tensor listed them in. Shared between the rows of one
    /// worker's run, which all ask for the same request.
    ids: Arc<[u32]>,
    /// `ids`' gathered log probabilities, aligned.
    probs: Vec<f32>,
}

/// The model's state for one beam: the worker slot whose resident buffers
/// hold the record's prefix, the beam's own resident row, and the gathered
/// scores of whatever comes next.
///
/// All three are handles — the tensors themselves never leave the session's
/// device — so cloning a beam is reference-count bumps, and dropping the
/// last beam of a record returns its slot to the pool.
#[derive(Clone, Debug)]
pub struct LmState {
    /// The record's worker slot, shared by every beam the record keeps —
    /// the LSTM's resident gen0 row. `None` under the pages layout, where
    /// a state's prefix is its page chain and no per-worker slot exists.
    worker: Option<Arc<WorkerSlot>>,
    /// Where this beam's per-position state tensors currently live: a
    /// contiguous resident row (the LSTM), the trie of one-position pages,
    /// or a resolved chain plus the tip rows a page claim has not covered.
    row: StateRow,
    /// The character chain this state covers, shared so a produced state's
    /// compacted form resolves its position's node without a lookup walk.
    path: Arc<PathDesc>,
    /// The scores of the candidates this state was asked for — the only
    /// tensor the host ever holds.
    scores: Arc<Scores>,
    /// The cache-pool claim of a compacted row-mode state, returning its row
    /// to the pool's free list when the last clone drops. `None` on a state
    /// living in a step's own generation or held by a trie node — those
    /// free with their buffers and the node's page claim respectively.
    cache: Option<Arc<CacheSlot>>,
}

impl LmState {
    /// The emitted characters this state covers, earliest first — its
    /// history. Context positions name no character, so a fresh
    /// `start(context)` plus one `advance` per returned `CharId` rebuilds an
    /// equivalent state; probes and reproductions that need a state's own
    /// bytes — never a second live session — go through this.
    #[must_use]
    pub fn history(&self) -> Vec<CharId> {
        let mut chars = Vec::new();
        let mut desc = self.path.as_ref();
        while let Some(parent) = desc.parent.as_deref() {
            if let Some(ch) = desc.ch {
                chars.push(ch);
            }
            desc = parent;
        }
        chars.reverse();
        chars
    }

    /// Whether the state's chain has tips that never claimed a page — the
    /// `Pending` form whose next step materialises through the scratch
    /// input rather than the paged gather. Test-only inspection.
    #[doc(hidden)]
    #[must_use]
    pub fn is_pending(&self) -> bool {
        matches!(self.row, StateRow::Pending { .. })
    }
}

/// Where a state's per-beam tensors live.
#[derive(Clone, Debug)]
enum StateRow {
    /// A row in a [`Generation`]'s buffers — the LSTM's produced or pooled form.
    Buf(RowRef),
    /// The leaf of the page trie: every position of the state's prefix sits
    /// in the resident pool, and the step graph gathers them by `page_row`.
    Path(Arc<Node>),
    /// A resolved chain plus the positions no page claim covered, in
    /// production order: the first `leaf.depth` positions are pages, each
    /// `tips` row one position after — the state's own `next_*` output rows.
    Pending {
        /// The deepest resolved ancestor: positions `0..leaf.depth`.
        leaf: Arc<Node>,
        /// The unclaimed positions, shallowest first: `tips[j]` is position
        /// `leaf.depth + j`, a row of its producing step's `next_*` output.
        tips: Vec<RowRef>,
    },
}

/// A state row's logical place in the page trie: the character that extends
/// its parent chain, with the resolved node memoised on first use.
#[derive(Debug)]
struct PathDesc {
    /// The prefix chain this extends — `None` only at the root, the zero-
    /// position start every state's chain reaches.
    parent: Option<Arc<PathDesc>>,
    /// The character this desc adds — `None` at the root and on a context
    /// position, which names no emitted character.
    ch: Option<CharId>,
    /// The trie node resolved from this chain, set by [`PagePool::resolve_at`]
    /// the first time the chain is compacted through. A memo whose node's
    /// page was reclaimed is stale, so `resolve_at` checks the claim, not
    /// just the memo, and dedups through the parent's children instead —
    /// then writes the replacement back. A set-once memo is wrong here: a
    /// replacement the memo could not name would carry only its state's
    /// strong reference, leaving the sweep free to reclaim a page a live
    /// state still points at.
    node: Mutex<Option<Arc<Node>>>,
}

/// One trie position: the page holding this position's slice of every state
/// tensor, its parent, and the children indexed by the character that
/// extends them.
///
/// A node's page returns to the pool's free list when the claim inside it
/// drops — when the node's last strong reference goes away, or when the
/// pool reclaims a dead leaf to make room. A reclaimed node becomes a
/// tombstone: its [`PathDesc`] memo still points at it, but the claim is
/// `None`, so a later resolution dedups a fresh node through the parent.
#[derive(Debug)]
struct Node {
    /// The emitted character this node extends its parent by — `None` on the
    /// root and on a context position.
    ch: Option<CharId>,
    /// This node's page index, or `u32::MAX` on the root and on a tombstone.
    /// Written under `claim`'s lock and read without it: the hot walks — a
    /// `page_row` build, a memo check — never take the mutex, and a stale
    /// index only ever belongs to a node whose page the sweep already
    /// tombstoned under it.
    page: AtomicU32,
    /// This node's page in the pool — `None` on the root, which covers no
    /// positions, and inside a tombstone, whose page was reclaimed.
    claim: Mutex<Option<PageSlot>>,
    /// How many positions the chain covers; the node's own page is entry
    /// `depth - 1` of a state's `page_row`.
    depth: u32,
    /// The previous position's node, held strong so a node keeps its chain
    /// alive.
    parent: Option<Arc<Node>>,
    /// The next position's nodes by extending character, weakly so children
    /// drop with their last user.
    children: Mutex<HashMap<CharId, Weak<Node>>>,
}

/// A page index claimed from the pool: returning the index to the free
/// list when the last [`Node`] holding it drops.
#[derive(Debug)]
struct PageSlot {
    /// The page index — the second axis position of the pool buffers.
    index: u32,
    /// The pool's free list, shared so the slot hands itself back.
    free: Arc<Mutex<Vec<u32>>>,
}

impl Drop for PageSlot {
    fn drop(&mut self) {
        self.free
            .lock()
            .expect("the page pool is not poisoned")
            .push(self.index);
    }
}

/// A cached row's claim on its [`CacheGen`] slot: dropping the last clone
/// of a compacted state hands the index back, so eviction only frees a row
/// once no live beam still reads it. A page-mode node claims the same slot
/// shape for its one position.
#[derive(Debug)]
struct CacheSlot {
    /// The generation the slot indexes.
    cgen: Arc<CacheGen>,
    /// The slot this claim returns.
    index: u32,
    /// The pool's count of handable rows, released with the slot.
    budget: Arc<AtomicUsize>,
}

impl Drop for CacheSlot {
    fn drop(&mut self) {
        self.cgen
            .free
            .lock()
            .expect("the cache pool is not poisoned")
            .push(self.index);
        self.cgen.live.fetch_sub(1, Ordering::Relaxed);
        self.budget.fetch_add(1, Ordering::Relaxed);
    }
}

/// The row shapes of one [`Generation`]'s tensors, `shape[1..]` each — the
/// key of a [`CacheGen`]. The transformer's state rows carry the sequence
/// axis, so a decode's depth is a distinct shape, and the parents one step
/// advances are one shape wide.
type RowKey = Vec<Box<[i64]>>;

/// One row width's live slot buffer: `[slots, *row]` per state tensor,
/// shared as the [`Generation`] cached rows' [`RowRef`]s name so
/// same-width parents still batch as one bound `step` run. When the
/// width's slots fill, another generation is allocated beside it.
#[derive(Debug)]
struct CacheGen {
    /// The slot buffers.
    tensors: Arc<Generation>,
    /// Slot indices open for `store`.
    free: Mutex<Vec<u32>>,
    /// Slots live.
    live: AtomicUsize,
}

/// The slots a fresh [`CacheGen`] allocates — a width rarely holds more at
/// once than the beam keeps live, and a fuller width gets a second
/// generation rather than a bigger first one.
const GEN_SLOTS: usize = 64;

/// The bounded store [`CharLm::compact`] copies produced rows into.
///
/// A produced state's row sits in its step's whole-batch buffers, so
/// keeping it pins every sibling's memory. `compact` copies the row into a
/// slot of the generation for its width — one resident row per cached
/// state, the natural unit of [`SessionShape::cache_rows`]'s bound — and
/// the copy keeps batching: states cached at one decode depth share the
/// generation, so `step` still runs them as one bound call.
#[derive(Debug)]
struct CachePool {
    /// The state tensors' element type, for fresh generations.
    dtype: ModelDtype,
    /// The bound: `SessionShape::cache_rows`.
    capacity: usize,
    /// Rows still handable to `compact`: `cache_rows` minus the live slots.
    budget: Arc<AtomicUsize>,
    /// The live generations by row shape.
    gens: Mutex<HashMap<RowKey, Vec<Arc<CacheGen>>>>,
}

impl CachePool {
    /// An empty pool able to hold *rows* slots.
    fn new(dtype: ModelDtype, rows: usize) -> Self {
        Self {
            dtype,
            capacity: rows,
            budget: Arc::new(AtomicUsize::new(rows)),
            gens: Mutex::new(HashMap::new()),
        }
    }

    /// Slots currently handed out across every generation.
    fn live(&self) -> usize {
        self.gens
            .lock()
            .expect("the cache pool is not poisoned")
            .values()
            .flatten()
            .map(|cgen| cgen.live.load(Ordering::Relaxed))
            .sum()
    }

    /// Allocate *slots* buffers of *key*'s row shape on *env*'s device.
    fn alloc_gen(
        env: &DeviceEnv,
        dtype: ModelDtype,
        key: &RowKey,
        slots: u32,
    ) -> Result<CacheGen, LmError> {
        let mut tensors = Vec::with_capacity(key.len());
        for row in key {
            let mut shape = Vec::with_capacity(row.len() + 1);
            shape.push(i64::from(slots));
            shape.extend_from_slice(row);
            tensors.push(DynTensor::new(&env.allocator, dtype.element(), shape)?.into_dyn());
        }
        Ok(CacheGen {
            tensors: Arc::new(Generation { tensors }),
            free: Mutex::new((0..slots).collect()),
            live: AtomicUsize::new(0),
        })
    }

    /// Copy *state*'s row into a free slot and return its compact form; the
    /// caller falls back to the state itself when the pool is out of rows.
    ///
    /// # Panics
    ///
    /// If ONNX Runtime fails the copy, which [`CharLm::open`] verifies does
    /// not happen for a well-formed export.
    fn store(&self, env: &mut DeviceEnv, state: &LmState) -> Option<LmState> {
        if self.budget.load(Ordering::Relaxed) == 0 {
            return None;
        }
        let StateRow::Buf(state_row) = &state.row else {
            return None;
        };
        let mut key = Vec::with_capacity(state_row.generation.tensors.len());
        for tensor in &state_row.generation.tensors {
            key.push(value_shape(tensor).ok()?[1..].into());
        }
        let (cgen, slot) = {
            let mut gens = self.gens.lock().expect("the cache pool is not poisoned");
            let list = gens.entry(key.clone()).or_default();
            let mut found = list.iter().find_map(|cgen| {
                cgen.free
                    .lock()
                    .expect("the cache pool is not poisoned")
                    .pop()
                    .map(|slot| (Arc::clone(cgen), slot))
            });
            if found.is_none() {
                let budget = self.budget.load(Ordering::Relaxed);
                let slots = u32::try_from(budget.min(GEN_SLOTS)).expect("the count fits u32");
                let cgen = Arc::new(
                    Self::alloc_gen(env, self.dtype, &key, slots)
                        .expect("the cache generation allocates"),
                );
                let slot = cgen
                    .free
                    .lock()
                    .expect("the cache pool is not poisoned")
                    .pop();
                list.push(Arc::clone(&cgen));
                found = slot.map(|slot| (cgen, slot));
            }
            found?
        };
        if self
            .budget
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |b| b.checked_sub(1))
            .is_err()
        {
            cgen.free
                .lock()
                .expect("the cache pool is not poisoned")
                .push(slot);
            return None;
        }
        cgen.live.fetch_add(1, Ordering::Relaxed);
        let mut copy = || -> Result<(), LmError> {
            for (index, dst) in cgen.tensors.tensors.iter().enumerate() {
                let src = &state_row.generation.tensors[index];
                let span = elements(&key[index]);
                copy_span(
                    env,
                    tensor_element(src)?,
                    src,
                    usize::try_from(state_row.index).expect("a row index fits usize") * span,
                    dst,
                    usize::try_from(slot).expect("a slot index fits usize") * span,
                    span,
                )?;
            }
            Ok(())
        };
        if let Err(error) = copy() {
            cgen.free
                .lock()
                .expect("the cache pool is not poisoned")
                .push(slot);
            cgen.live.fetch_sub(1, Ordering::Relaxed);
            self.budget.fetch_add(1, Ordering::Relaxed);
            panic!("the cache row copy failed: {error}");
        }
        Some(LmState {
            worker: state.worker.clone(),
            path: Arc::clone(&state.path),
            row: StateRow::Buf(RowRef {
                generation: Arc::clone(&cgen.tensors),
                index: slot,
            }),
            scores: Arc::clone(&state.scores),
            cache: Some(Arc::new(CacheSlot {
                cgen,
                index: slot,
                budget: Arc::clone(&self.budget),
            })),
        })
    }
}

/// The resident page pool the paged step graph reads its history from.
///
/// A state row with a zero axis — the page axis of the export's
/// `[layers, pages, heads, head_dim]` buffers — lives one position per
/// page: the pool tensors are bound to the step graph by name and every
/// row's prefix is a `page_row` of indices the graph gathers in-graph, so
/// no KV bytes cross the host boundary per step. A produced step row
/// `[layers, heads, head_dim]` is written into its claimed page by the
/// host — `layers` contiguous slices per tensor — and a state whose claim
/// failed is [`StateRow::Pending`]: it steps through a scratch row
/// materialised from its ancestors' pages plus its tip rows, exactly the
/// bytes the page path would have read.
///
/// `SessionShape::cache_rows` is the page count. The free list serves
/// claims; when it runs dry the pool reclaims nodes in insertion order —
/// leaves still held by the trie but by no live state, detected by a
/// strong count that says the desc memo is all that keeps them.
#[derive(Debug)]
struct PagePool {
    /// The Metal backend's pool buffers: when set, `keys`/`values` live as
    /// shared `MTLBuffer`s the step's kernels gather from, and `write_page`
    /// is a plain copy. `None` on every other backend.
    #[cfg(target_os = "macos")]
    metal: Option<metal::PoolBuffers>,
    /// The state tensors' element type.
    dtype: ModelDtype,
    /// The pool tensors' shapes with the page axis bound to `capacity` —
    /// `[pages, layers, heads, head_dim]`.
    rows: Vec<Box<[i64]>>,
    /// One position's slice per tensor: `heads * head_dim` elements of one
    /// layer — a page is `layers` of these per tensor, contiguous each.
    width: usize,
    /// The head count of the state rows — `row[2]`.
    heads: usize,
    /// One head's slice per tensor: `width / heads`.
    head_dim: usize,
    /// The layer count of the pool tensors.
    layers: usize,
    /// The bound in pages: `SessionShape::cache_rows`.
    capacity: usize,
    /// The pool buffers, allocated on the first environment that runs a
    /// paged step; `None` until then. `Arc`d because an ort value clone is
    /// a deep copy — sharing the tensors themselves is the whole point.
    bufs: Mutex<Option<Arc<Vec<DynValue>>>>,
    /// Page indices open for a claim — index zero is the masked scratch
    /// page and is never handed out.
    free: Arc<Mutex<Vec<u32>>>,
    /// Every node in creation order, for the reclaim sweep.
    order: Mutex<VecDeque<Weak<Node>>>,
    /// Serialises a node acquisition — the memo-hit `Arc::clone` and the
    /// dedup `Weak::upgrade` in `resolve_at` — against the sweep's
    /// strong-count check and tombstone, so no `Path`/`Pending` can come
    /// to hold a node whose page was already stolen. Lock order: `order`
    /// → `acquire` → `children` → `claim`; `free` is leaf-level inside
    /// `PageSlot::drop` and a `desc.node` memo lock is outermost of all.
    acquire: Mutex<()>,
    /// The trie root: covers no positions, owns no pages, and lives as long
    /// as the pool so a `resolve_at` walk always bottoms out.
    root: Arc<Node>,
    /// One node's byte footprint across the state tensors, for the notes.
    node_bytes: usize,
}

/// A scratch generation the pool reuses: its sequence depth and how many
/// pending rows it covers.
#[derive(Debug)]
struct Scratch {
    /// `page_row`'s width the generation was allocated at — the gather's
    /// T axis, so a different depth cannot share it.
    depth: u32,
    /// Rows allocated — slot zero the pad, the rest pending slots.
    slots: usize,
    /// The `state_*` tensors.
    generation: Arc<Generation>,
}

impl PagePool {
    /// An empty pool of *pages* pages, with the state rows' shapes as the
    /// export recorded them — `[0, layers, heads, head_dim]` each, the zero
    /// marking the page axis. A zero anywhere else names a layout nothing
    /// exports anymore, and loading it would bind a differently-shaped
    /// buffer into the step graph.
    fn new(dtype: ModelDtype, pages: usize, state_rows: &[Vec<i64>]) -> Result<Self, LmError> {
        let mut rows = Vec::with_capacity(state_rows.len());
        let mut width = 0usize;
        let mut heads = 0usize;
        let mut layers = 0usize;
        for row in state_rows {
            let axis = row
                .iter()
                .position(|&d| d == 0)
                .expect("a paged state row marks its page axis with a zero");
            if axis != 0 {
                return Err(LmError::PageAxis { axis });
            }
            // `width` is one layer's slice — the dims after the page and
            // layer axes.
            width = elements(&row[2..]);
            heads = usize::try_from(row[2]).expect("a head count fits usize");
            layers = usize::try_from(row[1]).expect("a layer count fits usize");
            let mut shape = row.clone();
            shape[axis] = dim(pages);
            rows.push(shape.into_boxed_slice());
        }
        let node_bytes = layers
            * width
            * state_rows.len()
            * match dtype {
                ModelDtype::Float32 => size_of::<f32>(),
                ModelDtype::Float16 => size_of::<f16>(),
            };
        Ok(Self {
            #[cfg(target_os = "macos")]
            metal: None,
            dtype,
            rows,
            width,
            heads,
            head_dim: width / heads,
            layers,
            capacity: pages,
            bufs: Mutex::new(None),
            free: Arc::new(Mutex::new(
                (1..u32::try_from(pages).expect("the page count fits u32")).collect(),
            )),
            order: Mutex::new(VecDeque::new()),
            acquire: Mutex::new(()),
            root: Arc::new(Node {
                ch: None,
                page: AtomicU32::new(u32::MAX),
                claim: Mutex::new(None),
                depth: 0,
                parent: None,
                children: Mutex::new(HashMap::new()),
            }),
            node_bytes,
        })
    }

    /// The pool buffers, allocating them on *env*'s device on first use.
    /// Page zero stays zeroed: `page_row` uses it as the masked scratch, and
    /// a zeroed free region keeps the gather from reading stale bytes it
    /// would only mask anyway — on any backend, since an arena-recycled or
    /// device allocator hands uninitialised memory that a masked read can
    /// still carry into `probs @ values` as NaN.
    fn bufs(&self, env: &mut DeviceEnv) -> Result<Arc<Vec<DynValue>>, LmError> {
        let mut bufs = self.bufs.lock().expect("the page pool is not poisoned");
        if let Some(bufs) = &*bufs {
            return Ok(Arc::clone(bufs));
        }
        let mut tensors = Vec::with_capacity(self.rows.len());
        for row in &self.rows {
            let tensor =
                DynTensor::new(&env.allocator, self.dtype.element(), row.to_vec())?.into_dyn();
            env.zero(self.dtype.element(), &tensor, elements(row))?;
            tensors.push(tensor);
        }
        let tensors = Arc::new(tensors);
        *bufs = Some(Arc::clone(&tensors));
        Ok(tensors)
    }

    /// Pages currently claimed — the live plus cached node count.
    fn live(&self) -> usize {
        self.capacity
            - 1
            - self
                .free
                .lock()
                .expect("the page pool is not poisoned")
                .len()
    }

    /// A free page: pop one, or reclaim a dead leaf when the list is empty.
    fn claim(&self) -> Option<PageSlot> {
        self.free
            .lock()
            .expect("the page pool is not poisoned")
            .pop()
            .map(|index| PageSlot {
                index,
                free: Arc::clone(&self.free),
            })
            .or_else(|| self.reclaim())
    }

    /// Steal the oldest dead leaf's page. A node is dead-safe to steal when
    /// the only strong references are its desc's memo and the sweep's own
    /// upgrade — `strong_count <= 2` — meaning no state or entry holds it;
    /// stealing detaches it from its parent so a later resolve dedups a
    /// fresh node, and takes the claim so the node tombstones. The slot
    /// itself moves to the claimant: dropping it here would both return
    /// the index to `free` and hand it out — the next claim would then
    /// share one page between two live nodes. The `acquire` lock covers the
    /// whole scan so a `resolve_at` clone or upgrade cannot slip between
    /// the count check and the tombstone.
    fn reclaim(&self) -> Option<PageSlot> {
        let mut order = self.order.lock().expect("the page pool is not poisoned");
        let _acquire = self.acquire.lock().expect("the page pool is not poisoned");
        while let Some(weak) = order.pop_front() {
            let Some(node) = weak.upgrade() else {
                continue;
            };
            if Arc::strong_count(&node) > 2
                || !node
                    .children
                    .lock()
                    .expect("the page trie is not poisoned")
                    .is_empty()
            {
                continue;
            }
            let (Some(parent), Some(ch)) = (&node.parent, node.ch) else {
                continue;
            };
            {
                let mut children = parent
                    .children
                    .lock()
                    .expect("the page trie is not poisoned");
                if children.get(&ch).is_some_and(|other| other.ptr_eq(&weak)) {
                    children.remove(&ch);
                } else {
                    continue;
                }
            }
            let taken = node
                .claim
                .lock()
                .expect("the page pool is not poisoned")
                .take();
            if let Some(slot) = taken {
                // Tombstone before the slot's drop frees the index: a
                // lock-free reader that still sees it then walks past a
                // page already re-issued.
                node.page.store(u32::MAX, Ordering::Relaxed);
                return Some(slot);
            }
        }
        None
    }

    /// The node *desc* resolves to under *parent*: memo or dedup hit, else a
    /// fresh page claim written from *row*'s `next_*` position slice.
    /// `None` when the pool has no page to give — the caller keeps the
    /// state's pending tips instead of failing.
    fn resolve_at(
        &self,
        parent: &Arc<Node>,
        desc: &Arc<PathDesc>,
        row: &RowRef,
        env: &mut DeviceEnv,
    ) -> Option<Arc<Node>> {
        // The memo lock stays held through the resolve so the
        // read-dedup-write is atomic — and a replacement always lands in
        // the memo, keeping a live chain's nodes above the sweep's
        // `strong_count` threshold. Every acquisition — the memo hit and
        // the dedup upgrade — runs under `acquire`, ordered before the
        // sweep's count check and tombstone, so a live state can never be
        // handed a node whose page is being stolen.
        let mut memo = desc.node.lock().expect("the desc memo is not poisoned");
        {
            let _acquire = self.acquire.lock().expect("the page pool is not poisoned");
            if let Some(node) = memo.as_ref()
                && node.page.load(Ordering::Relaxed) != u32::MAX
            {
                return Some(Arc::clone(node));
            }
            // A character desc dedups through the parent's children; a
            // context desc (`ch: None`) names no character, so it never
            // enters those maps — two contexts share nothing.
            if let Some(ch) = desc.ch {
                let mut children = parent
                    .children
                    .lock()
                    .expect("the page trie is not poisoned");
                if let Some(node) = children.get(&ch).and_then(Weak::upgrade) {
                    if node.page.load(Ordering::Relaxed) != u32::MAX {
                        *memo = Some(Arc::clone(&node));
                        return Some(node);
                    }
                    children.remove(&ch);
                }
            }
        }
        let slot = self.claim()?;
        for (index, tensor) in row.generation.tensors.iter().enumerate() {
            if let Err(error) = self.write_page(env, tensor, row.index, slot.index, index) {
                panic!("the page write failed: {error}");
            }
        }
        let node = Arc::new(Node {
            ch: desc.ch,
            page: AtomicU32::new(slot.index),
            claim: Mutex::new(Some(slot)),
            depth: parent.depth + 1,
            parent: Some(Arc::clone(parent)),
            children: Mutex::new(HashMap::new()),
        });
        if let Some(ch) = desc.ch {
            parent
                .children
                .lock()
                .expect("the page trie is not poisoned")
                .insert(ch, Arc::downgrade(&node));
        }
        self.order
            .lock()
            .expect("the page pool is not poisoned")
            .push_back(Arc::downgrade(&node));
        *memo = Some(Arc::clone(&node));
        Some(node)
    }

    /// Write *row*'s slice of tensor `index` — one position's `[layers,
    /// heads, head_dim]` — into page *page* of the pool buffer, contiguous
    /// under the page-major layout.
    fn write_page(
        &self,
        env: &mut DeviceEnv,
        row: &DynValue,
        row_index: u32,
        page: u32,
        index: usize,
    ) -> Result<(), LmError> {
        #[cfg(target_os = "macos")]
        if let Some(pool) = &self.metal {
            let (_, src) = row.try_extract_tensor::<f32>()?;
            let width = self.layers * self.width;
            let row_index = usize::try_from(row_index).expect("a row index fits usize");
            let page = usize::try_from(page).expect("a page index fits usize");
            let dst = pool.slice_mut(index);
            dst[page * width..(page + 1) * width]
                .copy_from_slice(&src[row_index * width..(row_index + 1) * width]);
            return Ok(());
        }
        let bufs = self.bufs(env)?;
        let dst = &bufs[index];
        let ty = tensor_element(row)?;
        let row_index = usize::try_from(row_index).expect("a row index fits usize");
        let page = usize::try_from(page).expect("a page index fits usize");
        copy_span(
            env,
            ty,
            row,
            row_index * self.layers * self.width,
            dst,
            page * self.layers * self.width,
            self.layers * self.width,
        )
    }

    /// The element offset of node *node*'s *layer* slice inside a pool
    /// buffer — contiguous per page in the page-major layout.
    fn slice_at(&self, node: &Node, layer: usize) -> usize {
        let page = usize::try_from(node.page.load(Ordering::Relaxed))
            .expect("a live node's page is claimed");
        assert_ne!(
            page,
            usize::try_from(u32::MAX).unwrap_or_default(),
            "a tombstoned node reached slice_at"
        );
        (page * self.layers + layer) * self.width
    }

    /// Fill the Metal step's `page_row`/`mask` tail from *leaf*'s chain —
    /// the u32/u8 twin of `fill_page_row`, which writes the ORT i64/bool
    /// inputs.
    #[cfg(target_os = "macos")]
    fn fill_page_row_u32(leaf: &Arc<Node>, page_row: &mut [u32], mask: &mut [u8]) {
        let mut node = leaf;
        let mut position = usize::try_from(leaf.depth).unwrap_or_default();
        while let Some(parent) = &node.parent {
            position -= 1;
            let page = node.page.load(Ordering::Relaxed);
            assert_ne!(page, u32::MAX, "a tombstoned node reached fill_page_row");
            page_row[position] = page;
            mask[position] = 1;
            node = parent;
        }
    }

    /// The `materialise_pending` twin for the Metal backend, tips only:
    /// the leaf's resolved prefix stays in the pool and the kernels gather
    /// it through `page_row` like any `Path` row — only the not-yet-paged
    /// tip positions copy into the scratch slot, marked `u32::MAX` in
    /// `page_row`. The source is each tip's `next_*` generation row and
    /// the destination the scratch slot's `[slots, L, H, T, D]` f32
    /// slices — plain copies, no env.
    #[cfg(target_os = "macos")]
    fn materialise_pending_f32(
        &self,
        leaf: &Arc<Node>,
        tips: &[RowRef],
        skeys: &mut [f32],
        svals: &mut [f32],
        slot: usize,
        seq: usize,
    ) -> Result<(), LmError> {
        let row_elems = self.layers * self.heads * seq * self.head_dim;
        let base = usize::try_from(leaf.depth).expect("a depth fits usize");
        for (j, tip) in tips.iter().enumerate() {
            let row_index = usize::try_from(tip.index).expect("a row index fits usize");
            for (index, dst) in [&mut *skeys, &mut *svals].iter_mut().enumerate() {
                let (_, src) = tip.generation.tensors[index].try_extract_tensor::<f32>()?;
                for layer in 0..self.layers {
                    for head in 0..self.heads {
                        let at =
                            (row_index * self.layers + layer) * self.width + head * self.head_dim;
                        let to = slot * row_elems
                            + ((layer * self.heads + head) * seq + base + j) * self.head_dim;
                        dst[to..to + self.head_dim].copy_from_slice(&src[at..at + self.head_dim]);
                    }
                }
            }
        }
        Ok(())
    }

    /// Fill the row's `page_row`/`mask` tail from *leaf*'s chain — one walk
    /// over `depth` nodes, ids written back to front so no temporary
    /// vector holds them.
    fn fill_page_row(leaf: &Arc<Node>, page_row: &mut [i64], mask: &mut [bool]) {
        let mut node = leaf;
        let mut position = usize::try_from(leaf.depth).unwrap_or_default();
        while let Some(parent) = &node.parent {
            position -= 1;
            let page = node.page.load(Ordering::Relaxed);
            assert_ne!(page, u32::MAX, "a tombstoned node reached fill_page_row");
            page_row[position] = i64::from(page);
            mask[position] = true;
            node = parent;
        }
    }

    /// This thread's scratch generation of *slots* rows `[slots, layers,
    /// heads, T, head_dim]` — the `state_*` inputs a pending state
    /// materialises into. Cached on the per-thread env and reused across
    /// calls while the depth matches and the rows cover the pending count:
    /// the gather reads only the slots `source_row` names, so stale rows
    /// past them are invisible, and a fresh one is made only when the
    /// width or the pending front outgrows the cached one — which is also
    /// the thread's own, so no two calls ever share a generation.
    fn scratch(
        &self,
        env: &mut DeviceEnv,
        depth: u32,
        slots: usize,
    ) -> Result<Arc<Generation>, LmError> {
        if let Some(scratch) = &env.scratch
            && scratch.depth == depth
            && scratch.slots >= slots
        {
            return Ok(Arc::clone(&scratch.generation));
        }
        let slots = env
            .scratch
            .as_ref()
            .filter(|scratch| scratch.depth == depth)
            .map_or(slots, |scratch| slots.max(scratch.slots));
        let generation = Arc::new(self.scratch_gen(env, depth, slots)?);
        env.scratch = Some(Scratch {
            depth,
            slots,
            generation: Arc::clone(&generation),
        });
        Ok(generation)
    }

    /// A fresh scratch generation of *slots* rows, zeroed so a masked tail
    /// can never carry recycled bytes into the gather.
    fn scratch_gen(
        &self,
        env: &mut DeviceEnv,
        depth: u32,
        slots: usize,
    ) -> Result<Generation, LmError> {
        let mut tensors = Vec::with_capacity(self.rows.len());
        for row in &self.rows {
            let shape = vec![
                dim(slots),
                dim(self.layers),
                row[2],
                i64::from(depth),
                row[3],
            ];
            let count = elements(&shape);
            let tensor = DynTensor::new(&env.allocator, self.dtype.element(), shape)?.into_dyn();
            env.zero(self.dtype.element(), &tensor, count)?;
            tensors.push(tensor);
        }
        Ok(Generation { tensors })
    }

    /// Assemble a pending state's history into scratch row *slot* of
    /// *scratch*: each ancestor page to its `depth - 1` position, then each
    /// tip row after it, so the scratch row reads exactly what the paged
    /// gather would have.
    fn materialise_pending(
        &self,
        leaf: &Arc<Node>,
        tips: &[RowRef],
        scratch: &Generation,
        slot: u32,
        env: &mut DeviceEnv,
    ) -> Result<(), LmError> {
        let mut chain = Vec::with_capacity(usize::try_from(leaf.depth).unwrap_or_default());
        let mut node = leaf;
        while let Some(parent) = &node.parent {
            chain.push(node);
            node = parent;
        }
        chain.reverse();
        let bufs = self.bufs(env)?;
        let slot = usize::try_from(slot).expect("a scratch slot fits usize");
        let seq = usize::try_from(value_shape(&scratch.tensors[0])?[3])
            .expect("a scratch width fits usize");
        let row_elems = self.layers * self.heads * seq * self.head_dim;
        // The scratch rows are `[slots, layers, heads, T, head_dim]`: a
        // position's `[heads, head_dim]` slice is *not* contiguous (heads
        // stride by `T * head_dim`), so every position's write goes per
        // head — `(layer, head, position)` names a `head_dim` run.
        for node in &chain {
            let position = usize::try_from(node.depth).expect("a depth fits usize") - 1;
            for (index, dst) in scratch.tensors.iter().enumerate() {
                let ty = tensor_element(&bufs[index])?;
                for layer in 0..self.layers {
                    let src = self.slice_at(node, layer);
                    for head in 0..self.heads {
                        copy_span(
                            env,
                            ty,
                            &bufs[index],
                            src + head * self.head_dim,
                            dst,
                            slot * row_elems
                                + ((layer * self.heads + head) * seq + position) * self.head_dim,
                            self.head_dim,
                        )?;
                    }
                }
            }
        }
        let base = usize::try_from(leaf.depth).expect("a depth fits usize");
        for (j, tip) in tips.iter().enumerate() {
            let row_index = usize::try_from(tip.index).expect("a row index fits usize");
            for (index, dst) in scratch.tensors.iter().enumerate() {
                let src = &tip.generation.tensors[index];
                let ty = tensor_element(src)?;
                for layer in 0..self.layers {
                    for head in 0..self.heads {
                        copy_span(
                            env,
                            ty,
                            src,
                            (row_index * self.layers + layer) * self.width + head * self.head_dim,
                            dst,
                            slot * row_elems
                                + ((layer * self.heads + head) * seq + base + j) * self.head_dim,
                            self.head_dim,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }
}

/// What `compact` writes into: full resident rows for a state shape without
/// a sequence axis (the LSTM), the position trie for one with it.
#[derive(Debug)]
enum Cache {
    /// One resident row per cached state, the older granularity.
    Rows(CachePool),
    /// One page per position in a prefix trie.
    Pages(PagePool),
}

/// How full the state cache is — for the notes' memory accounting.
#[derive(Clone, Copy, Debug)]
pub struct CacheStats {
    /// The bound in entries: rows for a row-mode cache, nodes for pages.
    pub capacity: usize,
    /// Entries currently held, as `capacity - free`.
    pub live: usize,
    /// One node's bytes across every state tensor — `0` for a row cache.
    pub node_bytes: usize,
}

impl Resident {
    /// The resident buffers, allocated on *env*'s device with *rows*'
    /// shapes when this is the model's first `start`. Every allocation is
    /// zeroed: the slots' unused region is read only under masks, but a
    /// masked tail can still carry uninitialised bytes into the run as
    /// NaN where the allocator does not fresh-map it.
    ///
    /// *`prelude_width`* is the mask's axis — `<bos>` + `context_chars` +
    /// `<sep>`, the width the graph already pads each prefix row to.
    fn ensure(
        &mut self,
        env: &mut DeviceEnv,
        dtype: ModelDtype,
        rows: &RowShapes,
        prelude_width: usize,
    ) -> Result<(), LmError> {
        if self.bufs.is_some() {
            return Ok(());
        }
        let capacity = 64;
        let resident = |env: &mut DeviceEnv, row: &[i64]| -> Result<DynValue, LmError> {
            let mut shape = Vec::with_capacity(row.len() + 1);
            shape.push(dim(capacity));
            shape.extend_from_slice(row);
            let count = elements(&shape);
            let tensor = DynTensor::new(&env.allocator, dtype.element(), shape)?.into_dyn();
            env.zero(dtype.element(), &tensor, count)?;
            Ok(tensor)
        };
        let mut prefix = Vec::with_capacity(rows.prefix.len());
        for row in &rows.prefix {
            prefix.push(resident(env, row)?);
        }
        let mask = if rows.prefix.is_empty() {
            None
        } else {
            let shape = [dim(capacity), dim(prelude_width)];
            let tensor =
                DynTensor::new(&env.allocator, TensorElementType::Bool, shape.to_vec())?.into_dyn();
            env.zero(TensorElementType::Bool, &tensor, capacity * prelude_width)?;
            Some(tensor)
        };
        let mut gen0 = Vec::with_capacity(rows.state.len());
        for row in &rows.state {
            gen0.push(resident(env, row)?);
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
        let enlarge = |env: &mut DeviceEnv, old: &DynValue| -> Result<DynValue, LmError> {
            let mut shape = value_shape(old)?.to_vec();
            shape[0] = dim(capacity);
            let ValueType::Tensor { ty, .. } = old.dtype() else {
                return Err(ort::Error::new("a resident buffer is not a tensor".to_owned()).into());
            };
            let count = elements(&shape);
            let tensor = DynTensor::new(&env.allocator, *ty, shape)?.into_dyn();
            env.zero(*ty, &tensor, count)?;
            Ok(tensor)
        };
        let mut pairs: Vec<(&DynValue, DynValue)> = Vec::new();
        for old in bufs.prefix.iter().chain(&bufs.gen0.tensors) {
            pairs.push((old, enlarge(env, old)?));
        }
        if let Some(old) = &bufs.mask {
            pairs.push((old, enlarge(env, old)?));
        }
        for (src, dst) in &pairs {
            let ValueType::Tensor { ty, shape, .. } = src.dtype() else {
                return Err(ort::Error::new("a resident buffer is not a tensor".to_owned()).into());
            };
            copy_span(env, *ty, src, 0, dst, 0, used * elements(&shape[1..]))?;
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

/// Copy *count* elements between *src*'s `src_off` and *dst*'s `dst_off`
/// (both in elements) as one flat run — what `grow` uses per buffer and
/// [`CachePool::store`] per row — through the element type's copier session.
fn copy_span(
    env: &mut DeviceEnv,
    ty: TensorElementType,
    src: &DynValue,
    src_off: usize,
    dst: &DynValue,
    dst_off: usize,
    count: usize,
) -> Result<(), LmError> {
    match ty {
        TensorElementType::Float32 => {
            copy_flat_typed::<f32>(env, src, src_off, dst, dst_off, count)
        }
        TensorElementType::Float16 => {
            copy_flat_typed::<f16>(env, src, src_off, dst, dst_off, count)
        }
        TensorElementType::Bool => copy_flat_typed::<bool>(env, src, src_off, dst, dst_off, count),
        other => {
            Err(ort::Error::new(format!("a resident buffer of {other:?} has no copier")).into())
        }
    }
}

/// The element-typed form of [`copy_span`].
fn copy_flat_typed<T: PrimitiveTensorElementType + Debug>(
    env: &mut DeviceEnv,
    src: &DynValue,
    src_off: usize,
    dst: &DynValue,
    dst_off: usize,
    count: usize,
) -> Result<(), LmError> {
    if count == 0 {
        return Ok(());
    }
    if env.host_resident {
        // Both buffers are heap — a flat `memcpy` is the whole copy, and a
        // page-table materialisation calls this per position, so skipping
        // the Identity session's binding overhead is the difference between
        // the path paying and not.
        // Safety: `count` elements at the given offsets are inside both
        // buffers — the callers bound them — and no read of `src` aliases
        // `dst`: copies run between distinct buffers or disjoint spans.
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.data_ptr().cast::<u8>().add(src_off * size_of::<T>()),
                dst.data_ptr()
                    .cast::<u8>()
                    .add(dst_off * size_of::<T>())
                    .cast_mut(),
                count * size_of::<T>(),
            );
        }
        return Ok(());
    }
    device_copy::<T>(env, src, src_off, dst, dst_off, count)
}

/// The non-host half of a flat copy: flat `[count]` views over the two
/// buffers through this thread's `Identity` copier, which runs on the
/// resident device — a CPU-bound `src` crosses over ORT's own input
/// memcpy.
fn device_copy<T: PrimitiveTensorElementType + Debug>(
    env: &mut DeviceEnv,
    src: &DynValue,
    src_off: usize,
    dst: &DynValue,
    dst_off: usize,
    count: usize,
) -> Result<(), LmError> {
    // Safety: the views span `count` elements at each buffer's offset, which
    // the caller bounds — `grow` within the fresh buffer's full extent,
    // `store` within both rows — and the copier only ever reads the source.
    let src = tensor_view::<T>(src, src_off, &[dim(count)])?;
    let dst = tensor_view::<T>(dst, dst_off, &[dim(count)])?;
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

/// The tensor's element type, or an error when *value* is not a tensor.
fn tensor_element(value: &DynValue) -> Result<TensorElementType, LmError> {
    let ValueType::Tensor { ty, .. } = value.dtype() else {
        return Err(ort::Error::new("a state or prefix value is not a tensor").into());
    };
    Ok(*ty)
}

/// A trained character language model, ready to score.
pub struct CharLm {
    /// The backend the caller asked for — which may differ from
    /// `step_sessions`' shape under the Metal backend, whose step is not an
    /// ORT session at all.
    backend: Backend,
    /// `prefill.onnx`: the per-thread sessions over it.
    prefill_sessions: GraphSessions,
    /// `charlm.onnx`, the step graph: the per-thread sessions over it.
    step_sessions: GraphSessions,
    /// The resident slot pool shared across every thread's sessions.
    workers: Arc<RwLock<Resident>>,
    /// The bounded store `compact` resolves produced states into — rows for
    /// a model without a sequence axis, the position trie for one with it;
    /// `SessionShape::cache_rows` of zero leaves it `None`. Declared before
    /// `devices` so the pool's buffers free while their allocator lives.
    cache: Option<Cache>,
    /// The trie chain's root: the zero-position state every start points at.
    root_path: Arc<PathDesc>,
    /// Whether the export is the pages layout: a state row marking the page
    /// axis with a zero means every state's history lives in [`PagePool`]'s
    /// buffers and the step gathers it by `page_row`.
    paged: bool,
    /// Host-side nanos spent on the page bookkeeping — the `page_row`/
    /// `mask` build, scratch materialisation, page claims and writes — so
    /// the latency note can split the miss computation from keeping the
    /// books. The upload/download counter is `bus`'s; this one is clocks.
    bookkeeping: AtomicU64,
    /// TEMP debug: marshal vs resolve halves of the Metal step's bookkeeping.
    bk_marshal: AtomicU64,
    /// TEMP debug: post-step resolve half.
    bk_resolve: AtomicU64,
    /// TEMP debug: `materialise_pending_f32` inside marshal.
    bk_mat: AtomicU64,
    /// TEMP debug: materialise count.
    bk_mat_n: AtomicU64,
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
    /// The Metal step — the kernels, weights and buffers the paged advance
    /// runs when `backend` is `Metal`; `None` otherwise. `Mutex`d since the
    /// buffers' contents are shared-mode host memory and `step` reuses them
    /// per call.
    #[cfg(target_os = "macos")]
    metal: Option<Mutex<metal::MetalStep>>,
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
        self.backend
    }

    /// Turn per-dispatch GPU timing on the Metal backend on or off — the
    /// rows land in [`Self::take_metal_profile`]. No-op without Metal.
    ///
    /// # Panics
    /// Panics if the step's mutex is poisoned.
    #[cfg(target_os = "macos")]
    pub fn set_metal_profiling(&self, on: bool) {
        if let Some(metal) = &self.metal {
            metal
                .lock()
                .expect("the metal step is not poisoned")
                .set_profiling(on);
        }
    }

    /// Truncate the Metal step's dispatch list at *limit* ops — the
    /// measurement bisection `metal-bench`'s `--only-ops` drives. `None`
    /// runs the whole step. No-op without Metal.
    ///
    /// # Panics
    /// Panics if the step's mutex is poisoned.
    #[cfg(target_os = "macos")]
    pub fn set_metal_step_limit(&self, limit: Option<usize>) {
        if let Some(metal) = &self.metal {
            metal
                .lock()
                .expect("the metal step is not poisoned")
                .set_step_limit(limit);
        }
    }

    /// The `(op, ms, weight-bytes)` rows the last profiled `advance`
    /// collected — empty without Metal or before the first profiled call.
    ///
    /// # Panics
    /// Panics if the step's mutex is poisoned.
    #[cfg(target_os = "macos")]
    pub fn take_metal_profile(&self) -> Vec<(String, f64, u64)> {
        self.metal.as_ref().map_or_else(Vec::new, |metal| {
            metal
                .lock()
                .expect("the metal step is not poisoned")
                .take_profile()
        })
    }

    /// The Metal step's last wall split — host encode, `commit` to
    /// scheduled, scheduled to completed — or `None` without Metal.
    ///
    /// # Panics
    /// Panics if the step's mutex is poisoned.
    #[cfg(target_os = "macos")]
    pub fn metal_step_timing(&self) -> Option<StepTiming> {
        self.metal.as_ref().map(|metal| {
            metal
                .lock()
                .expect("the metal step is not poisoned")
                .timing()
        })
    }

    /// The model's bus counters — what the sessions have moved across the
    /// host↔device boundary so far.
    pub fn bus(&self) -> &Bus {
        &self.bus
    }

    /// Nanos the host spent on the page bookkeeping since open — the
    /// `page_row`/`mask` builds, scratch materialisation and the resolve
    /// pass. Subtract it from `advance`'s wall time for the misses' own
    /// compute.
    pub fn bookkeeping_ns(&self) -> u64 {
        self.bookkeeping.load(Ordering::Relaxed)
    }

    /// TEMP debug halves.
    pub fn bk_marshal_ns(&self) -> u64 {
        self.bk_marshal.load(Ordering::Relaxed)
    }

    /// TEMP debug halves.
    pub fn bk_resolve_ns(&self) -> u64 {
        self.bk_resolve.load(Ordering::Relaxed)
    }

    /// TEMP debug halves.
    pub fn bk_mat(&self) -> (u64, u64) {
        (
            self.bk_mat.load(Ordering::Relaxed),
            self.bk_mat_n.load(Ordering::Relaxed),
        )
    }

    /// How full the state cache is, for the notes' memory accounting —
    /// `None` for a row-mode export with a zero `cache_rows`.
    pub fn cache_stats(&self) -> Option<CacheStats> {
        self.cache.as_ref().map(|cache| match cache {
            Cache::Rows(pool) => CacheStats {
                capacity: pool.capacity,
                live: pool.live(),
                node_bytes: 0,
            },
            Cache::Pages(pool) => CacheStats {
                capacity: pool.capacity,
                live: pool.live(),
                node_bytes: pool.node_bytes,
            },
        })
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
    ///
    /// # Panics
    ///
    /// If the manifest's state rows name a shape the resident pool cannot
    /// hold, or a shared lock inside the resident cache is poisoned.
    #[expect(
        clippy::too_many_lines,
        reason = "session construction is one ordered pass over the manifest's tensors"
    )]
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
        let root_path = Arc::new(PathDesc {
            parent: None,
            ch: None,
            node: Mutex::new(None),
        });
        let paged = manifest.rows.state.iter().any(|row| row.contains(&0));
        let backend = shape.backend;
        // The Metal backend's step is kernels, not an ORT session: the
        // prefill and any validation sessions run on the CPU shape instead,
        // while `backend` keeps what the caller asked for.
        #[cfg(target_os = "macos")]
        let ort_shape = if backend == Backend::Metal {
            SessionShape {
                backend: Backend::Cpu,
                ..shape
            }
        } else {
            shape
        };
        #[cfg(not(target_os = "macos"))]
        let ort_shape = shape;
        // The Metal step builds its kernels and packs the weights file into
        // buffers now — it needs the mapped bytes the sessions read the
        // same tensors from, before `maps` moves into the model.
        #[cfg(target_os = "macos")]
        let metal = if backend == Backend::Metal {
            if !paged {
                return Err(LmError::Metal(
                    "the Metal step reads only the resident-pages layout".to_owned(),
                ));
            }
            if dtype != ModelDtype::Float32 {
                return Err(LmError::Metal(
                    "the Metal step reads only float32 state tensors".to_owned(),
                ));
            }
            let table = match &manifest.weights {
                WeightsTable::Shared(table) => table,
                WeightsTable::PerGraph { step, .. } => step,
            };
            let map = weights_map(&mut maps, dir, &table.file)?;
            Some(Mutex::new(metal::MetalStep::new(
                &manifest.rows.state,
                shape.cache_rows.max(1024),
                table,
                map,
                shape.metal_weights,
            )?))
        } else {
            None
        };
        let mut model = Self {
            backend,
            prefill_sessions: GraphSessions::new(
                dir.join("prefill.onnx"),
                prefill_initializers,
                ort_shape,
                dtype,
            ),
            step_sessions: GraphSessions::new(
                dir.join("charlm.onnx"),
                step_initializers,
                ort_shape,
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
            cache: None,
            root_path,
            paged,
            bookkeeping: AtomicU64::new(0),
            bk_marshal: AtomicU64::new(0),
            bk_resolve: AtomicU64::new(0),
            bk_mat: AtomicU64::new(0),
            bk_mat_n: AtomicU64::new(0),
            context_chars: manifest.context_chars,
            specials: manifest.specials,
            ids,
            alphabet,
            #[cfg(target_os = "macos")]
            metal,
            weights: maps.into_values().collect(),
        };
        // Open one session of each graph now, so `open` reports a graph the
        // runtime rejects instead of the first decode failing. The Metal
        // backend has no step session to validate — its kernels are already
        // compiled or `open` failed above.
        model.prefill_sessions.session()?;
        #[cfg(not(target_os = "macos"))]
        model.step_sessions.session()?;
        #[cfg(target_os = "macos")]
        if model.backend != Backend::Metal {
            model.step_sessions.session()?;
        }
        // A paged export always owns its page pool: it is the layout every
        // state's history lives in, not only the cache — `cache_rows` of
        // zero floors at a working set, since a context's own pages are
        // claimed from the same list. A row-mode export (the LSTM) gets the
        // whole-row store only when the caller sizes it.
        if model.paged {
            let mut pool = PagePool::new(dtype, shape.cache_rows.max(1024), &model.rows.state)?;
            #[cfg(target_os = "macos")]
            if let Some(metal) = &model.metal {
                pool.metal = Some(
                    metal
                        .lock()
                        .expect("the metal step is not poisoned")
                        .pool_buffers(),
                );
            }
            model.cache = Some(Cache::Pages(pool));
        } else if shape.cache_rows > 0 {
            model.cache = Some(Cache::Rows(CachePool::new(dtype, shape.cache_rows)));
        }
        Ok(model)
    }

    /// This thread's device environment: the allocator the resident buffers
    /// are allocated through, the memory infos the bound step needs, and the
    /// copier sessions a pool growth uses.
    fn device_env(&self) -> Result<&RefCell<DeviceEnv>, LmError> {
        self.devices.get_or_try(|| {
            // The Metal backend has no step session: the prefill session's
            // allocator stands in for the CPU environment its resolve pass
            // needs.
            #[cfg(target_os = "macos")]
            let sessions = if self.backend == Backend::Metal {
                &self.prefill_sessions
            } else {
                &self.step_sessions
            };
            #[cfg(not(target_os = "macos"))]
            let sessions = &self.step_sessions;
            let session = sessions.session()?;
            let backend = self.step_sessions.shape.backend;
            let device = match backend {
                Backend::Cpu | Backend::CoreMl => AllocationDevice::CPU,
                Backend::WebGpu => AllocationDevice::WEBGPU_BUFFER,
                Backend::Cuda => AllocationDevice::CUDA,
                #[cfg(target_os = "macos")]
                Backend::Metal => AllocationDevice::CPU,
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
                host_resident: matches!(backend, Backend::Cpu | Backend::CoreMl),
                copiers: Vec::new(),
                scratch: None,
            }))
        })
    }

    /// The alphabet ids *asked* names — the request's candidates mapped
    /// through the lexicon's rows, `<eos>` appended where the path can end —
    /// sorted, which is both the order a `candidates` tensor row lists them
    /// in and the order the stored [`Scores`] are keyed by. `Arc`d so the
    /// one request a worker's rows share becomes one allocation.
    fn asked_ids(&self, asked: &Asked<'_>) -> Arc<[u32]> {
        let mut ids: Vec<u32> = asked
            .candidates
            .iter()
            .map(|ch| self.ids[ch.index()])
            .collect();
        if asked.eos {
            ids.push(self.specials.eos);
        }
        ids.sort_unstable();
        assert!(
            !ids.is_empty(),
            "a request with no candidates and no <eos> asks for a state that answers nothing"
        );
        ids.into()
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
    fn prefill(&self, context: Option<&str>, asked: &Asked<'_>) -> Result<LmState, LmError> {
        if self.paged {
            return self.prefill_pages(context, asked);
        }
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
                &mut env.borrow_mut(),
                self.dtype,
                &self.rows,
                self.context_chars + 2,
            )?;
            pool.take(&mut env.borrow_mut())?
        };
        let (scores, generation) = match self.prefill_into(&tokens, slot, asked) {
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
            worker: Some(Arc::new(WorkerSlot {
                pool: Arc::clone(&self.workers),
                index: slot,
            })),
            row: StateRow::Buf(RowRef {
                generation,
                index: slot,
            }),
            path: Arc::clone(&self.root_path),
            scores: Arc::new(scores),
            cache: None,
        })
    }

    /// Run the paged prefill: `candidate_log_probs` to the host and each
    /// `prefix_*` output — one position per row — to the device, then every
    /// position's slice written into its own claimed page. The pages form
    /// the context's chain: the record's root every beam shares.
    #[expect(
        clippy::too_many_lines,
        reason = "one prefill is a straight-line sequence: bind and run the graph, then chain the prelude's pages or tips position by position"
    )]
    fn prefill_pages(&self, context: Option<&str>, asked: &Asked<'_>) -> Result<LmState, LmError> {
        let Some(Cache::Pages(pool)) = &self.cache else {
            unreachable!("a paged export always owns its page pool");
        };
        let prelude = self.prelude(context);
        let tokens: Vec<i64> = prelude.iter().copied().map(i64::from).collect();
        let tokens = Tensor::from_array(([1i64, dim(tokens.len())], tokens))?;
        self.bus
            .upload(Phase::Start, elements(tokens.shape()) * size_of::<i64>());
        let ids = self.asked_ids(asked);
        let candidates = Tensor::from_array((
            [1i64, dim(ids.len())],
            ids.iter().map(|&id| i64::from(id)).collect::<Vec<_>>(),
        ))?;
        self.bus.upload(Phase::Start, ids.len() * size_of::<i64>());
        let env_cell = self.device_env()?;
        let (host, device) = {
            let env = env_cell.borrow();
            (env.host.clone(), env.device.clone())
        };
        let (prefix_outs, probs) = {
            let mut bound = self.prefill_sessions.session()?.borrow_mut();
            let BoundSession { session, binding } = &mut *bound;
            binding.clear();
            binding.bind_input("tokens", &tokens)?;
            binding.bind_input("candidates", &candidates)?;
            for name in &self.prefix {
                binding.bind_output_to_device(name.as_str(), &device)?;
            }
            binding.bind_output_to_device("candidate_log_probs", &host)?;
            let mut outputs = session.run_binding(binding)?;
            binding.synchronize_outputs()?;
            let probs = self.host_f32(&outputs["candidate_log_probs"], Phase::Start)?;
            let mut prefix_outs = Vec::with_capacity(self.prefix.len());
            for name in &self.prefix {
                prefix_outs.push(
                    outputs
                        .remove(name.as_str())
                        .expect("the prefill graph returns every prefix output"),
                );
            }
            (prefix_outs, probs)
        };
        // Write the prelude's pages: one claimed page per position, the
        // prefix output's row `position` landing at the chain's next page.
        // Positions a claim cannot cover ride as pending tips, the same
        // fallthrough `advance` uses — `start` under a saturated pool
        // degrades rather than fails.
        let tick = std::time::Instant::now();
        let prefix_outs = Arc::new(Generation {
            tensors: prefix_outs,
        });
        let mut env = env_cell.borrow_mut();
        let mut leaf = Arc::clone(&pool.root);
        let mut desc = Arc::clone(&self.root_path);
        let mut tips = Vec::new();
        let positions = usize::try_from(value_shape(&prefix_outs.tensors[0])?[0])
            .expect("a prelude length fits usize");
        for position in 0..positions {
            let Some(slot) = pool.claim() else {
                tips.push(RowRef {
                    generation: Arc::clone(&prefix_outs),
                    index: u32::try_from(position).expect("a position fits u32"),
                });
                desc = Arc::new(PathDesc {
                    parent: Some(desc),
                    ch: None,
                    node: Mutex::new(None),
                });
                continue;
            };
            for (index, tensor) in prefix_outs.tensors.iter().enumerate() {
                pool.write_page(
                    &mut env,
                    tensor,
                    u32::try_from(position).expect("a position fits u32"),
                    slot.index,
                    index,
                )?;
            }
            let child = Arc::new(Node {
                ch: None,
                page: AtomicU32::new(slot.index),
                claim: Mutex::new(Some(slot)),
                depth: leaf.depth + 1,
                parent: Some(Arc::clone(&leaf)),
                children: Mutex::new(HashMap::new()),
            });
            pool.order
                .lock()
                .expect("the page pool is not poisoned")
                .push_back(Arc::downgrade(&child));
            let child_desc = Arc::new(PathDesc {
                parent: Some(desc),
                ch: None,
                node: Mutex::new(None),
            });
            *child_desc
                .node
                .lock()
                .expect("the desc memo is not poisoned") = Some(Arc::clone(&child));
            leaf = child;
            desc = child_desc;
        }
        drop(env);
        self.bookkeeping.fetch_add(
            u64::try_from(tick.elapsed().as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        Ok(LmState {
            worker: None,
            row: if tips.is_empty() {
                StateRow::Path(leaf)
            } else {
                StateRow::Pending { leaf, tips }
            },
            path: desc,
            scores: Arc::new(Scores { ids, probs }),
            cache: None,
        })
    }

    /// Run the prefill into *slot*'s rows: every prefix, mask and state
    /// output is bound over the slot's row of its resident buffer, so the
    /// only tensors to cross the bus are `tokens` and `candidates` up and
    /// `candidate_log_probs` down.
    fn prefill_into(
        &self,
        tokens: &Tensor<i64>,
        slot: u32,
        asked: &Asked<'_>,
    ) -> Result<(Scores, Arc<Generation>), LmError> {
        let ids = self.asked_ids(asked);
        let candidates = Tensor::from_array((
            [1i64, dim(ids.len())],
            ids.iter().map(|&id| i64::from(id)).collect::<Vec<_>>(),
        ))?;
        self.bus.upload(Phase::Start, ids.len() * size_of::<i64>());
        let mut bound = self.prefill_sessions.session()?.borrow_mut();
        let BoundSession { session, binding } = &mut *bound;
        binding.clear();
        binding.bind_input("tokens", tokens)?;
        binding.bind_input("candidates", &candidates)?;
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
        binding.bind_output_to_device("candidate_log_probs", &env.host)?;
        let outputs = session.run_binding(binding)?;
        binding.synchronize_outputs()?;
        let probs = self.host_f32(&outputs["candidate_log_probs"], Phase::Start)?;
        Ok((Scores { ids, probs }, Arc::clone(&bufs.gen0)))
    }

    /// The tensor's `f32` data on the host — the bound `candidate_log_probs`
    /// output, the only thing the bus ever carries.
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
    fn step(&self, steps: &[(&LmState, CharId, Asked<'_>)]) -> Result<Vec<LmState>, LmError> {
        if self.paged {
            #[cfg(target_os = "macos")]
            if self.backend == Backend::Metal {
                return self.step_pages_metal(steps);
            }
            return self.step_pages(steps);
        }
        let rows: Vec<RowRef> = steps
            .iter()
            .map(|(state, _, _)| match &state.row {
                StateRow::Buf(row) => row.clone(),
                StateRow::Path(..) | StateRow::Pending { .. } => {
                    unreachable!("a paged state exists only under the pages layout")
                }
            })
            .collect();
        let mut advanced: Vec<Option<LmState>> = (0..steps.len()).map(|_| None).collect();
        let mut begin = 0;
        while begin < steps.len() {
            let mut end = begin + 1;
            while end < steps.len() && Arc::ptr_eq(&rows[end].generation, &rows[begin].generation) {
                end += 1;
            }
            self.step_gen(
                &steps[begin..end],
                &rows[begin..end],
                &mut advanced[begin..end],
            )?;
            begin = end;
        }
        Ok(advanced
            .into_iter()
            .map(|state| state.expect("every row is advanced by its run"))
            .collect())
    }

    /// One bound step over rows all living in the same generation — the
    /// row-mode path, which serves the LSTM's flat batch: `token`,
    /// `source_row` and `candidates` are the host inputs, the state buffers
    /// the batch gathers by `source_row`, `candidate_log_probs` is bound to
    /// host memory while the `next_*` outputs stay on the device as the next
    /// generation.
    fn step_gen(
        &self,
        steps: &[(&LmState, CharId, Asked<'_>)],
        sources: &[RowRef],
        out: &mut [Option<LmState>],
    ) -> Result<(), LmError> {
        let generation = &sources[0].generation;
        let rows = steps.len();
        let mut token = Vec::with_capacity(rows);
        let mut source_row = Vec::with_capacity(rows);
        // The request of each row — as the sorted alphabet ids the row's
        // `candidate_log_probs` columns answer.
        let mut asked: Vec<Arc<[u32]>> = Vec::with_capacity(rows);
        for (state, ch, request) in steps {
            token.push(i64::from(self.ids[ch.index()]));
            source_row.push(i64::from(match &state.row {
                StateRow::Buf(row) => row.index,
                StateRow::Path(..) | StateRow::Pending { .. } => {
                    unreachable!("a paged state exists only under the pages layout")
                }
            }));
            asked.push(self.asked_ids(request));
        }
        // The widest request sets K; shorter rows pad with `<pad>`, whose
        // gathered column is real but never read.
        let count = asked.iter().map(|ids| ids.len()).max().unwrap_or(0);
        let mut candidates = vec![i64::from(self.specials.pad); rows * count];
        for (row, ids) in asked.iter().enumerate() {
            for (column, &id) in ids.iter().enumerate() {
                candidates[row * count + column] = i64::from(id);
            }
        }
        let uploaded = (token.len() + source_row.len() + candidates.len()) * size_of::<i64>();
        self.bus.upload(Phase::Step, uploaded);
        let token = Tensor::from_array((vec![dim(rows)], token))?;
        let source_row = Tensor::from_array((vec![dim(rows)], source_row))?;
        let candidates = Tensor::from_array(([dim(rows), dim(count)], candidates))?;
        let mut bound = self.step_sessions.session()?.borrow_mut();
        let BoundSession { session, binding } = &mut *bound;
        binding.clear();
        binding.bind_input("token", &token)?;
        binding.bind_input("source_row", &source_row)?;
        binding.bind_input("candidates", &candidates)?;
        for (index, name) in self.state.iter().enumerate() {
            binding.bind_input(name.as_str(), &generation.tensors[index])?;
        }
        {
            let env = self.device_env()?.borrow();
            binding.bind_output_to_device("candidate_log_probs", &env.host)?;
            for name in &self.state {
                binding.bind_output_to_device(format!("next_{name}"), &env.device)?;
            }
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
        let gathered = self.host_f32(&outputs["candidate_log_probs"], Phase::Step)?;
        for (row, (state, ch, _)) in steps.iter().enumerate() {
            let ids = Arc::clone(&asked[row]);
            out[row] = Some(LmState {
                worker: state.worker.clone(),
                row: StateRow::Buf(RowRef {
                    generation: Arc::clone(&next),
                    index: u32::try_from(row).expect("a step's rows fit u32"),
                }),
                path: Arc::new(PathDesc {
                    parent: Some(Arc::clone(&state.path)),
                    ch: Some(*ch),
                    node: Mutex::new(None),
                }),
                scores: Arc::new(Scores {
                    probs: gathered[row * count..row * count + ids.len()].to_vec(),
                    ids,
                }),
                cache: None,
            });
        }
        Ok(())
    }

    /// One bound step over a pages-layout batch.
    ///
    /// Every state carries its history as the trie leaf its `page_row`
    /// names plus, on a pending state, the positions no page claim has
    /// covered. `token`, `page_row`, `mask`, `candidates` and `source_row`
    /// are the host inputs; the pool's `keys`/`values` buffers and the
    /// `state_*` scratch rows (slot zero a pad, then one per pending state)
    /// are bound by name, so a step moves indices and scores — no row of
    /// KV crosses the host boundary. A produced state resolves to its leaf
    /// at once: a page claim plus `layers` slice copies per tensor, and a
    /// state the pool cannot page keeps its `next_*` row as the next
    /// pending tip.
    #[expect(
        clippy::too_many_lines,
        reason = "one bound step is a straight-line sequence: materialise the pending rows, build the index inputs, bind, run, scatter and resolve the states"
    )]
    fn step_pages(&self, steps: &[(&LmState, CharId, Asked<'_>)]) -> Result<Vec<LmState>, LmError> {
        let Some(Cache::Pages(pool)) = &self.cache else {
            unreachable!("a paged export always owns its page pool");
        };
        let env_cell = self.device_env()?;
        let mut env = env_cell.borrow_mut();
        let bufs = pool.bufs(&mut env)?;
        let rows = steps.len();
        // History depth per row; the widest sets `page_row`'s T.
        let depths: Vec<usize> = steps
            .iter()
            .map(|(state, _, _)| match &state.row {
                StateRow::Path(leaf) => usize::try_from(leaf.depth).expect("a depth fits usize"),
                StateRow::Pending { leaf, tips } => {
                    usize::try_from(leaf.depth).expect("a depth fits usize") + tips.len()
                }
                StateRow::Buf(..) => {
                    unreachable!("a row state exists only under the row layout")
                }
            })
            .collect();
        let width = depths.iter().copied().max().unwrap_or(0);
        // The pending rows' scratch generation: slot zero pads the paged
        // rows' `source_row` clamp, one slot per pending state after it.
        let pending = depths
            .iter()
            .enumerate()
            .filter(|(row, _)| matches!(steps[*row].0.row, StateRow::Pending { .. }))
            .map(|(row, _)| row);
        let mut scratch = None;
        let mut scratch_slot = 1u32;
        let mut page_row = vec![0i64; rows * width];
        let mut mask = vec![false; rows * width];
        let mut source_row = vec![-1i64; rows];
        let mut token = Vec::with_capacity(rows);
        let mut asked: Vec<Arc<[u32]>> = Vec::with_capacity(rows);
        // A worker's rows ask for the same request — the decode's steps
        // come in per-worker runs — so the sorted id list behind every
        // `Scores` is computed once per run and shared.
        let mut last: Option<(*const CharId, usize, bool)> = None;
        let mut last_ids: Option<Arc<[u32]>> = None;
        let tick = std::time::Instant::now();
        for (row, (state, ch, request)) in steps.iter().enumerate() {
            token.push(i64::from(self.ids[ch.index()]));
            let key = (
                request.candidates.as_ptr(),
                request.candidates.len(),
                request.eos,
            );
            let ids = match (&last, &last_ids) {
                (Some(seen), Some(ids)) if *seen == key => Arc::clone(ids),
                _ => {
                    let ids = self.asked_ids(request);
                    last = Some(key);
                    last_ids = Some(Arc::clone(&ids));
                    ids
                }
            };
            asked.push(ids);
            match &state.row {
                StateRow::Path(leaf) => {
                    PagePool::fill_page_row(
                        leaf,
                        &mut page_row[row * width..row * width + depths[row]],
                        &mut mask[row * width..row * width + depths[row]],
                    );
                }
                StateRow::Pending { leaf, tips } => {
                    if scratch.is_none() {
                        scratch = Some(pool.scratch(
                            &mut env,
                            u32::try_from(width).expect("a width fits u32"),
                            pending.clone().count() + 1,
                        )?);
                    }
                    let slot = scratch_slot;
                    scratch_slot += 1;
                    source_row[row] = i64::from(slot);
                    pool.materialise_pending(
                        leaf,
                        tips,
                        scratch.as_ref().expect("allocated above"),
                        slot,
                        &mut env,
                    )?;
                    for position in 0..depths[row] {
                        mask[row * width + position] = true;
                    }
                }
                StateRow::Buf(..) => unreachable!("a row state exists only under the row layout"),
            }
        }
        drop(env);
        self.bookkeeping.fetch_add(
            u64::try_from(tick.elapsed().as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        // The widest request sets K; shorter rows pad with `<pad>`, whose
        // gathered column is real but never read.
        let count = asked.iter().map(|ids| ids.len()).max().unwrap_or(0);
        let mut candidates = vec![i64::from(self.specials.pad); rows * count];
        for (row, ids) in asked.iter().enumerate() {
            for (column, &id) in ids.iter().enumerate() {
                candidates[row * count + column] = i64::from(id);
            }
        }
        let uploaded = (token.len() + page_row.len() + source_row.len() + candidates.len())
            * size_of::<i64>()
            + mask.len();
        self.bus.upload(Phase::Step, uploaded);
        let token = Tensor::from_array((vec![dim(rows)], token))?;
        let page_row = Tensor::from_array(([dim(rows), dim(width)], page_row))?;
        let mask = Tensor::from_array(([dim(rows), dim(width)], mask))?;
        let candidates = Tensor::from_array(([dim(rows), dim(count)], candidates))?;
        let source_row = Tensor::from_array((vec![dim(rows)], source_row))?;
        let scratch = match scratch {
            Some(scratch) => scratch,
            // No pending rows: one slot stands in, its bytes unread —
            // a `source_row` of -1 only names it through `clamp(0)`.
            None => pool.scratch(
                &mut env_cell.borrow_mut(),
                u32::try_from(width).expect("fits"),
                1,
            )?,
        };
        let mut bound = self.step_sessions.session()?.borrow_mut();
        let BoundSession { session, binding } = &mut *bound;
        binding.clear();
        binding.bind_input("token", &token)?;
        binding.bind_input("page_row", &page_row)?;
        binding.bind_input("mask", &mask)?;
        binding.bind_input("candidates", &candidates)?;
        for (index, name) in self.state.iter().enumerate() {
            binding.bind_input(name.as_str(), &bufs[index])?;
        }
        binding.bind_input("state_keys", &scratch.tensors[0])?;
        binding.bind_input("state_values", &scratch.tensors[1])?;
        binding.bind_input("source_row", &source_row)?;
        let (host, device) = {
            let env = env_cell.borrow();
            (env.host.clone(), env.device.clone())
        };
        binding.bind_output_to_device("candidate_log_probs", &host)?;
        for name in &self.state {
            binding.bind_output_to_device(format!("next_{name}"), &device)?;
        }
        let mut outputs = session.run_binding(binding)?;
        binding.synchronize_outputs()?;
        let mut tensors = Vec::with_capacity(self.state.len());
        for name in &self.state {
            tensors.push(
                outputs
                    .remove(format!("next_{name}").as_str())
                    .expect("the step graph returns every next_ output"),
            );
        }
        let next = Arc::new(Generation { tensors });
        let gathered = self.host_f32(&outputs["candidate_log_probs"], Phase::Step)?;
        // Resolve each produced state at once: the step's `next_*` row is
        // the one position's KV the new node claims a page for.
        let tick = std::time::Instant::now();
        let mut env = env_cell.borrow_mut();
        let mut advanced = Vec::with_capacity(rows);
        for (row, (state, ch, _)) in steps.iter().enumerate() {
            let tip = RowRef {
                generation: Arc::clone(&next),
                index: u32::try_from(row).expect("a step's rows fit u32"),
            };
            let desc = Arc::new(PathDesc {
                parent: Some(Arc::clone(&state.path)),
                ch: Some(*ch),
                node: Mutex::new(None),
            });
            let (leaf, mut tips) = Self::resolve_state(pool, state, &mut env);
            let row_state = if tips.is_empty() {
                match pool.resolve_at(&leaf, &desc, &tip, &mut env) {
                    Some(node) => StateRow::Path(node),
                    None => StateRow::Pending {
                        leaf,
                        tips: vec![tip],
                    },
                }
            } else {
                tips.push(tip);
                StateRow::Pending { leaf, tips }
            };
            let ids = Arc::clone(&asked[row]);
            advanced.push(LmState {
                worker: None,
                row: row_state,
                path: desc,
                scores: Arc::new(Scores {
                    probs: gathered[row * count..row * count + ids.len()].to_vec(),
                    ids,
                }),
                cache: None,
            });
        }
        self.bookkeeping.fetch_add(
            u64::try_from(tick.elapsed().as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        Ok(advanced)
    }

    /// The `step_pages` twin for the Metal backend: identical marshalling
    /// and resolve pass, but the step itself is `MetalStep`'s one command
    /// buffer instead of a `run_binding` — u32/u8/i32 inputs, the scratch
    /// slots live in shared `MTLBuffer`s, and the `next_*` outputs come back
    /// as f32 slices turned into the next generation's tensors.
    #[cfg(target_os = "macos")]
    #[expect(
        clippy::too_many_lines,
        reason = "same straight-line sequence as step_pages: materialise the pending rows, build the index inputs, run, scatter and resolve the states"
    )]
    fn step_pages_metal(
        &self,
        steps: &[(&LmState, CharId, Asked<'_>)],
    ) -> Result<Vec<LmState>, LmError> {
        let Some(Cache::Pages(pool)) = &self.cache else {
            unreachable!("a paged export always owns its page pool");
        };
        let mut metal = self
            .metal
            .as_ref()
            .expect("a metal backend owns its step")
            .lock()
            .expect("the metal step is not poisoned");
        let rows = steps.len();
        let depths: Vec<usize> = steps
            .iter()
            .map(|(state, _, _)| match &state.row {
                StateRow::Path(leaf) => usize::try_from(leaf.depth).expect("a depth fits usize"),
                StateRow::Pending { leaf, tips } => {
                    usize::try_from(leaf.depth).expect("a depth fits usize") + tips.len()
                }
                StateRow::Buf(..) => {
                    unreachable!("a row state exists only under the row layout")
                }
            })
            .collect();
        let width = depths.iter().copied().max().unwrap_or(0);
        let mut depth_in = Vec::with_capacity(rows);
        let mut token = Vec::with_capacity(rows);
        let mut asked: Vec<Arc<[u32]>> = Vec::with_capacity(rows);
        let mut last: Option<(*const CharId, usize, bool)> = None;
        let mut last_ids: Option<Arc<[u32]>> = None;
        for (row, (_, ch, request)) in steps.iter().enumerate() {
            token.push(self.ids[ch.index()]);
            depth_in.push(u32::try_from(depths[row]).expect("a depth fits u32"));
            let key = (
                request.candidates.as_ptr(),
                request.candidates.len(),
                request.eos,
            );
            let ids = match (&last, &last_ids) {
                (Some(seen), Some(ids)) if *seen == key => Arc::clone(ids),
                _ => {
                    let ids = self.asked_ids(request);
                    last = Some(key);
                    last_ids = Some(Arc::clone(&ids));
                    ids
                }
            };
            asked.push(ids);
        }
        let count = asked.iter().map(|ids| ids.len()).max().unwrap_or(0);
        let mut cand = vec![self.specials.pad; rows * count];
        for (row, ids) in asked.iter().enumerate() {
            for (column, &id) in ids.iter().enumerate() {
                cand[row * count + column] = id;
            }
        }
        let pending_count = steps
            .iter()
            .filter(|(state, _, _)| matches!(state.row, StateRow::Pending { .. }))
            .count();
        let slots = pending_count + 1;
        // Grow the shared buffers before the scratch slices are handed out
        // — `scratch_mut`'s pointers must see the sized allocation.
        metal.ensure(rows, width, count, slots);
        let tick = std::time::Instant::now();
        let mut page_row = vec![0u32; rows * width];
        let mut mask = vec![0u8; rows * width];
        let mut source_row = vec![-1i32; rows];
        let mut scratch_slot = 1usize;
        for (row, (state, _, _)) in steps.iter().enumerate() {
            match &state.row {
                StateRow::Path(leaf) => {
                    PagePool::fill_page_row_u32(
                        leaf,
                        &mut page_row[row * width..row * width + depths[row]],
                        &mut mask[row * width..row * width + depths[row]],
                    );
                }
                StateRow::Pending { leaf, tips } => {
                    let slot = scratch_slot;
                    scratch_slot += 1;
                    source_row[row] = i32::try_from(slot).expect("a slot fits i32");
                    let (skeys, svals) = metal.scratch_mut();
                    let t0 = std::time::Instant::now();
                    pool.materialise_pending_f32(leaf, tips, skeys, svals, slot, width)?;
                    self.bk_mat.fetch_add(
                        u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX),
                        Ordering::Relaxed,
                    );
                    self.bk_mat_n.fetch_add(1, Ordering::Relaxed);
                    // The resolved prefix stays paged — `page_row` names it
                    // exactly like a `Path` row; each pending position gets
                    // the sentinel so the gather reads the scratch slot.
                    let base = usize::try_from(leaf.depth).expect("a depth fits usize");
                    PagePool::fill_page_row_u32(
                        leaf,
                        &mut page_row[row * width..row * width + base],
                        &mut mask[row * width..row * width + base],
                    );
                    for position in base..depths[row] {
                        page_row[row * width + position] = u32::MAX;
                        mask[row * width + position] = 1;
                    }
                }
                StateRow::Buf(..) => unreachable!("a row state exists only under the row layout"),
            }
        }
        let bk = u64::try_from(tick.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.bookkeeping.fetch_add(bk, Ordering::Relaxed);
        self.bk_marshal.fetch_add(bk, Ordering::Relaxed);
        self.bus.upload(
            Phase::Step,
            (token.len() + page_row.len() + source_row.len() + cand.len() + depth_in.len())
                * size_of::<u32>()
                + mask.len(),
        );
        metal.step(
            &token,
            &page_row,
            &mask,
            &source_row,
            &depth_in,
            &cand,
            count,
            width,
            slots,
        );
        let gathered = metal.logp(rows, count).to_vec();
        self.bus
            .download(Phase::Step, gathered.len() * size_of::<f32>());
        let next = Arc::new(Generation {
            tensors: vec![
                Tensor::from_array((
                    [dim(rows), dim(pool.layers), dim(pool.width)],
                    metal.next_k(rows).to_vec(),
                ))?
                .into_dyn(),
                Tensor::from_array((
                    [dim(rows), dim(pool.layers), dim(pool.width)],
                    metal.next_v(rows).to_vec(),
                ))?
                .into_dyn(),
            ],
        });
        drop(metal);
        // Resolve each produced state at once — identical to step_pages.
        let tick = std::time::Instant::now();
        let env_cell = self.device_env()?;
        let mut env = env_cell.borrow_mut();
        let mut advanced = Vec::with_capacity(rows);
        for (row, (state, ch, _)) in steps.iter().enumerate() {
            let tip = RowRef {
                generation: Arc::clone(&next),
                index: u32::try_from(row).expect("a step's rows fit u32"),
            };
            let desc = Arc::new(PathDesc {
                parent: Some(Arc::clone(&state.path)),
                ch: Some(*ch),
                node: Mutex::new(None),
            });
            let (leaf, mut tips) = Self::resolve_state(pool, state, &mut env);
            let row_state = if tips.is_empty() {
                match pool.resolve_at(&leaf, &desc, &tip, &mut env) {
                    Some(node) => StateRow::Path(node),
                    None => StateRow::Pending {
                        leaf,
                        tips: vec![tip],
                    },
                }
            } else {
                tips.push(tip);
                StateRow::Pending { leaf, tips }
            };
            let ids = Arc::clone(&asked[row]);
            advanced.push(LmState {
                worker: None,
                row: row_state,
                path: desc,
                scores: Arc::new(Scores {
                    probs: gathered[row * count..row * count + ids.len()].to_vec(),
                    ids,
                }),
                cache: None,
            });
        }
        let bk = u64::try_from(tick.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.bookkeeping.fetch_add(bk, Ordering::Relaxed);
        self.bk_resolve.fetch_add(bk, Ordering::Relaxed);
        Ok(advanced)
    }

    /// Resolve a paged state's chain: the deepest live node plus the rows
    /// still pending pages after it, in production order. `tips` is empty
    /// when every position resolved — the state is a plain path.
    fn resolve_state(
        pool: &PagePool,
        state: &LmState,
        env: &mut DeviceEnv,
    ) -> (Arc<Node>, Vec<RowRef>) {
        let StateRow::Pending { leaf, tips } = &state.row else {
            let StateRow::Path(leaf) = &state.row else {
                unreachable!("a row state exists only under the row layout");
            };
            return (Arc::clone(leaf), Vec::new());
        };
        // The descs the pending tips resolve, shallowest first: the state
        // path's last `tips.len()` ancestors.
        let mut descs = Vec::with_capacity(tips.len());
        let mut desc = &state.path;
        for _ in 0..tips.len() {
            descs.push(desc);
            desc = desc.parent.as_ref().expect("a pending tip has its desc");
        }
        descs.reverse();
        let mut node = Arc::clone(leaf);
        let mut left = Vec::new();
        for (j, (desc, tip)) in descs.into_iter().zip(tips.iter()).enumerate() {
            if let Some(resolved) = pool.resolve_at(&node, desc, tip, env) {
                node = resolved;
            } else {
                left.extend_from_slice(&tips[j..]);
                break;
            }
        }
        (node, left)
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

    /// The Metal step's last buffers — every input and output — for the
    /// debug harness that bisects a wrong call against a reference. `None`
    /// on every other backend or before the first `advance`.
    #[doc(hidden)]
    #[cfg(target_os = "macos")]
    pub fn debug_metal_dump(&self) -> Option<Vec<(String, Vec<u8>)>> {
        self.metal
            .as_ref()
            .map(|metal| metal.lock().expect("the metal step is not poisoned").dump())
    }
}

impl Transition for CharLm {
    const HISTORY: usize = MAX_HISTORY;

    type State = LmState;

    /// # Panics
    ///
    /// If ONNX Runtime fails a run, which the session that [`CharLm::open`]
    /// verified does not do for well-formed inputs.
    fn start(&self, context: Option<&str>, asked: &Asked<'_>) -> LmState {
        self.prefill(context, asked)
            .expect("the prefill graph runs on the prelude")
    }

    /// # Panics
    ///
    /// If the state was never asked to score *candidate* — the request that
    /// produced it did not name it, and the model keeps no row to fall back
    /// on.
    fn score(&self, state: &LmState, candidate: CharId) -> f32 {
        let id = self.ids[candidate.index()];
        state.scores.probs[state
            .scores
            .ids
            .binary_search(&id)
            .unwrap_or_else(|_| {
                panic!(
                    "the step never asked for {candidate:?} — a score the search did not request is a bug, not a fallback"
                )
            })]
    }

    /// # Panics
    ///
    /// If the state was never asked for `<eos>` — its request said the path
    /// cannot end there.
    fn finish(&self, state: &LmState) -> f32 {
        state.scores.probs[state
            .scores
            .ids
            .binary_search(&self.specials.eos)
            .unwrap_or_else(|_| {
                panic!(
                    "the step never asked for <eos> — a finish on a path that cannot end is a bug, not a fallback"
                )
            })]
    }

    /// Relies on the row-order guarantee of [`Transition::advance`]: a
    /// worker's survivors arrive as one contiguous run, which `step` groups
    /// into `SessionShape::width` blocks sharing the worker's resident slot.
    ///
    /// # Panics
    ///
    /// If ONNX Runtime fails a run; see [`CharLm::start`].
    fn advance(&self, steps: &[(&LmState, CharId, Asked<'_>)]) -> Vec<LmState> {
        if steps.is_empty() {
            return Vec::new();
        }
        self.step(steps).expect("the graph runs on a batch")
    }

    /// Move *state*'s rows into the cache: a row-mode store gets its own
    /// row; a page-mode store resolves the state's chain to its trie leaf —
    /// each position's page written once — and the returned state names the
    /// leaf instead of pinning the whole step generation it was produced in.
    /// A pool out of room — `SessionShape::cache_rows` is the bound — keeps
    /// the state as it is rather than failing.
    ///
    /// # Panics
    ///
    /// If ONNX Runtime fails the copy; see [`CharLm::start`].
    fn compact(&self, state: &LmState) -> LmState {
        let Some(cache) = &self.cache else {
            return state.clone();
        };
        match cache {
            Cache::Rows(pool) => {
                if state.cache.is_some() {
                    return state.clone();
                }
                let mut env = self
                    .device_env()
                    .expect("the device env opens")
                    .borrow_mut();
                pool.store(&mut env, state).unwrap_or_else(|| state.clone())
            }
            Cache::Pages(pool) => {
                let StateRow::Pending { .. } = &state.row else {
                    return state.clone();
                };
                let mut env = self
                    .device_env()
                    .expect("the device env opens")
                    .borrow_mut();
                let (leaf, tips) = Self::resolve_state(pool, state, &mut env);
                if tips.is_empty() {
                    LmState {
                        worker: None,
                        row: StateRow::Path(leaf),
                        path: Arc::clone(&state.path),
                        scores: Arc::clone(&state.scores),
                        cache: None,
                    }
                } else {
                    state.clone()
                }
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use ime_pinyin::{Lexicon, SyllableTable};

    /// The fixture lexicon's first character id — the reclaim test needs a
    /// real `CharId` to queue a node under, not a fabricated one.
    fn a_char() -> CharId {
        let source = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/char_pinyin.tsv"
        ))
        .expect("the fixture character table is committed");
        let lexicon = Lexicon::parse(&source, &SyllableTable::load())
            .expect("the fixture character table parses");
        let character = lexicon.characters()[0];
        lexicon
            .id_of(character)
            .expect("the lexicon indexes its own table")
    }

    /// A page the sweep steals must reach exactly one claimant: `reclaim`
    /// used to return the stolen index while the victim's `PageSlot` also
    /// dropped it back into `free`, so the next claim issued the same page
    /// to a second live node and their position writes corrupted both
    /// chains.
    #[test]
    fn reclaim_hands_a_stolen_page_to_one_claimant() {
        // `Float32` with a two-layer, 3-by-4 state row — the tensors are
        // never allocated on this path, so the dtype is a formality.
        let pool = PagePool::new(ModelDtype::Float32, 4, &[vec![0, 2, 3, 4]])
            .expect("a page-major shape loads");
        let ch = a_char();
        // The victim claims a page the way a produced node does, then
        // stops growing — a leaf a single live state still names, which is
        // what the sweep's `strong_count <= 2` names stealable: the walk's
        // own upgrade takes the count to two here. (Dropping the node
        // would just free its slot through the `PageSlot` drop — the
        // sweep only ever sees live leaves.)
        let stolen = pool.claim().expect("a first claim");
        let index = stolen.index;
        let victim = Arc::new(Node {
            ch: Some(ch),
            page: AtomicU32::new(index),
            claim: Mutex::new(Some(stolen)),
            depth: 1,
            parent: Some(Arc::clone(&pool.root)),
            children: Mutex::new(HashMap::new()),
        });
        pool.root
            .children
            .lock()
            .expect("the page trie is not poisoned")
            .insert(ch, Arc::downgrade(&victim));
        pool.order
            .lock()
            .expect("the page pool is not poisoned")
            .push_back(Arc::downgrade(&victim));
        // Drain the rest of the free list so the next claim must sweep.
        let held: Vec<PageSlot> = (0..2)
            .map(|_| pool.claim().expect("two free pages remain"))
            .collect();
        let slot = pool
            .claim()
            .expect("the sweep reclaims the dead leaf's page");
        assert_eq!(slot.index, index, "the reclaimed page is re-issued");
        assert!(
            !pool
                .free
                .lock()
                .expect("the page pool is not poisoned")
                .contains(&index),
            "a page a node owns must not also sit in the free list"
        );
        drop(slot);
        let count = pool
            .free
            .lock()
            .expect("the page pool is not poisoned")
            .iter()
            .filter(|&&i| i == index)
            .count();
        assert_eq!(count, 1, "a page's drop frees its index once");
        drop(held);
    }

    /// A node a live state still names must stay above the sweep's
    /// `strong_count` threshold, which is why a desc's memo has to be
    /// replaceable: when `resolve_at` re-resolves a desc whose memo'd node
    /// was stolen, the replacement has to land in the memo, or its only
    /// strong reference is the state's row — one owner, inside the sweep's
    /// reach, and a later claim tombstones a page the live chain still
    /// walks.
    #[test]
    fn a_live_chain_node_stays_above_the_sweeps_threshold() {
        // Three pages: one for the live leaf, one for the dead leaf, and
        // page zero the pool never issues.
        let pool = PagePool::new(ModelDtype::Float32, 3, &[vec![0, 2, 3, 4]])
            .expect("a page-major shape loads");
        let ch = a_char();
        // The live leaf: claimed, queued, and named twice the way a
        // produced state names it — the row handle a `LmState` keeps, and
        // its `PathDesc`'s memo.
        let slot = pool.claim().expect("a first claim");
        let live_index = slot.index;
        let leaf = Arc::new(Node {
            ch: Some(ch),
            page: AtomicU32::new(slot.index),
            claim: Mutex::new(Some(slot)),
            depth: 1,
            parent: Some(Arc::clone(&pool.root)),
            children: Mutex::new(HashMap::new()),
        });
        pool.root
            .children
            .lock()
            .expect("the page trie is not poisoned")
            .insert(ch, Arc::downgrade(&leaf));
        pool.order
            .lock()
            .expect("the page pool is not poisoned")
            .push_back(Arc::downgrade(&leaf));
        let row_handle = Arc::clone(&leaf);
        let desc = Arc::new(PathDesc {
            parent: Some(Arc::new(PathDesc {
                parent: None,
                ch: None,
                node: Mutex::new(None),
            })),
            ch: Some(ch),
            node: Mutex::new(Some(Arc::clone(&leaf))),
        });
        // The dead leaf: claimed and queued under a second parent, then
        // unreferenced — a legitimate steal the sweep should take instead.
        let other_parent = Arc::new(Node {
            ch: Some(ch),
            page: AtomicU32::new(u32::MAX),
            claim: Mutex::new(None),
            depth: 1,
            parent: None,
            children: Mutex::new(HashMap::new()),
        });
        let dead = pool.claim().expect("a second claim");
        let dead_index = dead.index;
        let dead_leaf = Arc::new(Node {
            ch: Some(ch),
            page: AtomicU32::new(dead.index),
            claim: Mutex::new(Some(dead)),
            depth: 2,
            parent: Some(Arc::clone(&other_parent)),
            children: Mutex::new(HashMap::new()),
        });
        other_parent
            .children
            .lock()
            .expect("the page trie is not poisoned")
            .insert(ch, Arc::downgrade(&dead_leaf));
        pool.order
            .lock()
            .expect("the page pool is not poisoned")
            .push_back(Arc::downgrade(&dead_leaf));
        drop(dead_leaf);
        let stolen = pool.claim().expect("the sweep steals the dead leaf");
        assert_eq!(stolen.index, dead_index);
        assert_eq!(
            leaf.page.load(Ordering::Relaxed),
            live_index,
            "a page a live chain names is never reclaimed"
        );
        let _ = (row_handle, desc);
    }
}
