//! The key-level model through native Core ML: `ondevice-coreml` runs the
//! converted functions, and [`CoreMlKeyModel`] is the [`KeyModel`] the key
//! engine searches over.
//!
//! This is a port of `mlime.train.key_coreml` (`CoreMlKeyModel`, `key_row`,
//! `tree_row`, the directory manifests, `log_softmax`), function for function,
//! so that the Rust search and the Python reference make the same calls over
//! the same compiled model. `mlime-coreml convert` writes the directory it
//! opens: `key-model.json` and `tokenizer.json` beside one directory per
//! function family, `context/`, `keys/` and `decoder/`, each a
//! `<name>.mlmodelc` holding the family's functions and the `manifest.json`
//! that names their tensors. The manifests are read and checked, never
//! assumed: a function whose inputs are not the ones this module feeds, or a
//! family converted from another bundle, is refused at [`CoreMlKeyModel::open`].
//!
//! A keystroke is one keys function over all the typed keys, in the smallest
//! width that holds them, and any number of tree functions over its cross keys
//! and values, which stay in Core ML storage between the two. The context
//! encoder runs once per context text; the last text's encoding is kept, so
//! the keystrokes of one context share it. A tree call's logits are float16;
//! the float32 log-softmax runs on the host, over the rows the search reads.
//!
//! Opening compiles every function for the compute units asked for. A cache
//! miss recompiles for the Neural Engine for tens of seconds, and the cache is
//! keyed by process name, so the model is opened off the typing path.

use std::cell::OnceCell;
use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

use ime_pinyin::{EmitId, Keys};
pub use ondevice_coreml::ComputeUnits;
use ondevice_coreml::{Function, Inputs, Model, Tensor, f16};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use thiserror::Error;
use tokenizers::{Tokenizer, TruncationParams};
use tracing::{debug, info};

use super::{KeyModel, POLICY_VERSION, TREE_BUCKETS, TreeCall};

/// The bundle's manifest, beside the model directories.
const MANIFEST: &str = "key-model.json";

/// The tokenizer the context encoder reads.
const TOKENIZER: &str = "tokenizer.json";

/// The text the context function is warmed with; any text the tokenizer
/// reads does.
const WARM_CONTEXT: &str = "a";

/// Each model directory's manifest.
const DIRECTORY_MANIFEST: &str = "manifest.json";

/// The sentinels the keys function's row is framed by: `[CLS]` and `[SEP]`.
const KEY_SENTINELS: usize = 2;

/// Why a Core ML key model could not be opened or run.
#[derive(Debug, Error)]
pub enum CoreMlKeyError {
    /// A file could not be read.
    #[error("could not read {path}")]
    Io {
        /// The file that failed.
        path: PathBuf,
        /// The underlying error.
        #[source]
        source: io::Error,
    },
    /// A manifest is not the JSON `mlime-coreml convert` writes.
    #[error("{path} is malformed")]
    Json {
        /// The manifest.
        path: PathBuf,
        /// The parse failure.
        #[source]
        source: serde_json::Error,
    },
    /// A manifest parsed but does not describe this model.
    #[error("{path}: {reason}")]
    Manifest {
        /// The manifest.
        path: PathBuf,
        /// What is wrong with it.
        reason: String,
    },
    /// The tokenizer could not be read or could not encode a context.
    #[error("the tokenizer at {path} failed")]
    Tokenizer {
        /// The tokenizer file.
        path: PathBuf,
        /// The tokenizer's error.
        #[source]
        source: tokenizers::Error,
    },
    /// Core ML failed to load a function or to run one.
    #[error(transparent)]
    CoreMl(#[from] ondevice_coreml::Error),
    /// More keys than the largest keys function holds.
    #[error("{keys} keys do not fit the largest keys function, which holds {largest}")]
    TooManyKeys {
        /// The keys typed.
        keys: usize,
        /// What the largest function holds.
        largest: usize,
    },
    /// A tree call asks for a bucket the model has no function for.
    #[error("the model has no tree function of {0} nodes")]
    TreeBucket(usize),
    /// The context encodes to more tokens than the context function takes.
    #[error("the context encodes to {tokens} tokens; the context function takes {capacity}")]
    ContextTooLong {
        /// The tokens the context encoded to.
        tokens: usize,
        /// The context function's length.
        capacity: usize,
    },
    /// A function produced a tensor of another shape than its manifest declares.
    #[error("{function} produced {name} of shape {got:?}, expected {expected:?}")]
    OutputShape {
        /// The function.
        function: String,
        /// The output.
        name: &'static str,
        /// The shape it has.
        got: Vec<usize>,
        /// The shape the manifest declares.
        expected: Vec<usize>,
    },
    /// A value fed as an int32 does not fit one.
    #[error("{0} does not fit an int32")]
    Int32(&'static str),
}

// --- the manifests ------------------------------------------------------------------------

/// The function families, by the name the manifests give them.
#[derive(Copy, Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Family {
    Context,
    Keys,
    Tree,
}

impl Family {
    /// The directory the family converts into, and the model it holds.
    const fn directory(self) -> &'static str {
        match self {
            Self::Context => "context",
            Self::Keys => "keys",
            Self::Tree => "decoder",
        }
    }

    /// The inputs a function of the family is fed, with their element types.
    const fn inputs(self) -> &'static [(&'static str, Dtype)] {
        match self {
            Self::Context => &[
                ("context_ids", Dtype::Int32),
                ("context_mask", Dtype::Float16),
            ],
            Self::Keys => &[
                ("input_ids", Dtype::Int32),
                ("key_letters", Dtype::Int32),
                ("attention_mask", Dtype::Float16),
                ("context", Dtype::Float16),
                ("context_mask", Dtype::Float16),
            ],
            Self::Tree => &[
                ("reader_ids", Dtype::Int32),
                ("positions", Dtype::Int32),
                ("tree_bias", Dtype::Float16),
                ("cross_keys", Dtype::Float16),
                ("cross_values", Dtype::Float16),
                ("cross_bias", Dtype::Float16),
            ],
        }
    }

    /// The outputs a function of the family produces, with their element types.
    const fn outputs(self) -> &'static [(&'static str, Dtype)] {
        match self {
            Self::Context => &[("context", Dtype::Float16)],
            Self::Keys => &[
                ("ctc_logits", Dtype::Float16),
                ("cross_keys", Dtype::Float16),
                ("cross_values", Dtype::Float16),
            ],
            Self::Tree => &[("logits", Dtype::Float16)],
        }
    }
}

/// A tensor's element type in a directory manifest.
#[derive(Copy, Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Dtype {
    Int32,
    Float16,
}

/// One tensor of a directory manifest. Its shape is not read here: Core ML
/// holds every function to its static shapes.
#[derive(Deserialize)]
struct TensorSpec {
    dtype: Dtype,
}

/// What of a model directory's `manifest.json` a loader reads; other fields
/// are ignored.
#[derive(Deserialize)]
struct DirectoryManifest {
    family: Family,
    step: u64,
    policy_version: u32,
    functions: BTreeMap<String, usize>,
    inputs: BTreeMap<String, TensorSpec>,
    outputs: BTreeMap<String, TensorSpec>,
    n_max: usize,
    mask: f32,
}

/// One function of a family in `key-model.json`.
#[derive(Deserialize)]
struct FunctionSpec {
    name: String,
    length: usize,
    capacity: usize,
}

/// One family in `key-model.json`: its functions, smallest first.
#[derive(Deserialize)]
struct FamilySpec {
    name: Family,
    functions: Vec<FunctionSpec>,
}

/// The tokenizer's special ids the key row is built from.
#[derive(Deserialize)]
struct Specials {
    pad: i32,
    cls: i32,
    sep: i32,
    mask: i32,
}

/// The fill tower's dimension the context rows are as wide as.
#[derive(Deserialize)]
struct Tower {
    hidden: usize,
}

/// The dimensions this module reads.
#[derive(Deserialize)]
struct Dims {
    fill: Tower,
    emitted: usize,
}

/// What of `key-model.json` a runner reads; other fields are ignored.
#[derive(Deserialize)]
struct KeyManifest {
    step: u64,
    policy_version: u32,
    correct_syllables: bool,
    n_max: usize,
    neg: f32,
    context_tokens: usize,
    specials: Specials,
    letter_pad: i32,
    characters: Vec<char>,
    reader_ids: Vec<i32>,
    reader_bos: i32,
    dims: Dims,
    families: Vec<FamilySpec>,
}

fn read(path: &Path) -> Result<Vec<u8>, CoreMlKeyError> {
    std::fs::read(path).map_err(|source| CoreMlKeyError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, CoreMlKeyError> {
    serde_json::from_slice(&read(path)?).map_err(|source| CoreMlKeyError::Json {
        path: path.to_path_buf(),
        source,
    })
}

fn refuse(path: &Path, reason: impl Into<String>) -> CoreMlKeyError {
    CoreMlKeyError::Manifest {
        path: path.to_path_buf(),
        reason: reason.into(),
    }
}

impl KeyManifest {
    /// The family *family*'s functions, which `key-model.json` must list.
    fn family(&self, path: &Path, family: Family) -> Result<&[FunctionSpec], CoreMlKeyError> {
        self.families
            .iter()
            .find(|held| held.name == family)
            .map(|held| held.functions.as_slice())
            .ok_or_else(|| refuse(path, format!("lists no {family:?} family")))
    }

    /// Refuse a manifest this code does not speak: another packing policy,
    /// buckets other than [`TREE_BUCKETS`], or tables that disagree.
    fn check(&self, path: &Path) -> Result<(), CoreMlKeyError> {
        if self.policy_version != POLICY_VERSION {
            return Err(refuse(
                path,
                format!(
                    "was written for packing policy {}; this code implements {POLICY_VERSION}",
                    self.policy_version
                ),
            ));
        }
        if self.reader_ids.len() != self.characters.len()
            || self.dims.emitted != self.characters.len()
        {
            return Err(refuse(
                path,
                format!(
                    "{} reader ids and a head of {} for {} emittable characters",
                    self.reader_ids.len(),
                    self.dims.emitted,
                    self.characters.len()
                ),
            ));
        }
        let context = self.family(path, Family::Context)?;
        if context.len() != 1
            || context[0].length != self.context_tokens
            || context[0].name != format!("ctx_{}", self.context_tokens)
            || self.context_tokens.checked_sub(KEY_SENTINELS) != Some(context[0].capacity)
        {
            return Err(refuse(
                path,
                "the context family is not one function of the context length",
            ));
        }
        let keys = self.family(path, Family::Keys)?;
        let widths_ascend = keys.windows(2).all(|pair| pair[0].length < pair[1].length);
        if keys.is_empty()
            || !widths_ascend
            || keys.iter().any(|function| {
                function.name != format!("keys_{}", function.length)
                    || function.length < KEY_SENTINELS
                    || function.capacity != function.length - KEY_SENTINELS
            })
        {
            return Err(refuse(
                path,
                "the keys family is not widths ascending, each holding two keys fewer",
            ));
        }
        let tree = self.family(path, Family::Tree)?;
        let buckets: Vec<usize> = tree.iter().map(|function| function.length).collect();
        if buckets != TREE_BUCKETS
            || tree.iter().any(|function| {
                function.name != format!("tree_{}", function.length)
                    || function.capacity != function.length
            })
        {
            return Err(refuse(
                path,
                format!("the tree family is {buckets:?}; this code's buckets are {TREE_BUCKETS:?}"),
            ));
        }
        Ok(())
    }
}

/// Check a directory manifest against the bundle's: the family, the
/// functions and what they were converted from, and the tensors this module
/// feeds and reads.
fn check_directory(
    path: &Path,
    held: &DirectoryManifest,
    family: Family,
    manifest: &KeyManifest,
    functions: &[FunctionSpec],
) -> Result<(), CoreMlKeyError> {
    let expected: BTreeMap<&str, usize> = functions
        .iter()
        .map(|function| (function.name.as_str(), function.length))
        .collect();
    let declared: BTreeMap<&str, usize> = held
        .functions
        .iter()
        .map(|(name, &length)| (name.as_str(), length))
        .collect();
    if held.family != family
        || declared != expected
        || held.step != manifest.step
        || held.policy_version != manifest.policy_version
        || held.n_max != manifest.n_max
        || held.mask.total_cmp(&manifest.neg).is_ne()
    {
        return Err(refuse(
            path,
            "was not converted from this bundle: its functions, step, policy or masks differ \
             from key-model.json",
        ));
    }
    for (what, declared, expected) in [
        ("inputs", &held.inputs, family.inputs()),
        ("outputs", &held.outputs, family.outputs()),
    ] {
        let found: Vec<(&str, Dtype)> = declared
            .iter()
            .map(|(name, spec)| (name.as_str(), spec.dtype))
            .collect();
        let mut wanted = expected.to_vec();
        wanted.sort_unstable_by_key(|&(name, _)| name);
        if found != wanted {
            return Err(refuse(
                path,
                format!("its {what} are {found:?}; this code feeds and reads {wanted:?}"),
            ));
        }
    }
    Ok(())
}

// --- the functions ------------------------------------------------------------------------

/// Load every function of *family* at *units*, in the order *functions* lists
/// them, from the family's directory under *directory*.
fn load_family(
    directory: &Path,
    family: Family,
    manifest: &KeyManifest,
    units: ComputeUnits,
) -> Result<Vec<(usize, Function)>, CoreMlKeyError> {
    let name = family.directory();
    let folder = directory.join(name);
    let path = folder.join(DIRECTORY_MANIFEST);
    let functions = manifest.family(&directory.join(MANIFEST), family)?;
    check_directory(&path, &read_json(&path)?, family, manifest, functions)?;
    let model = Model::open(folder.join(format!("{name}.mlmodelc")))?;
    functions
        .iter()
        .map(|spec| -> Result<_, CoreMlKeyError> {
            let started = Instant::now();
            let function = model.function(Some(&spec.name), units)?;
            info!(
                function = spec.name.as_str(),
                seconds = started.elapsed().as_secs_f64(),
                "loaded a Core ML function"
            );
            Ok((spec.capacity, function))
        })
        .collect()
}

/// What the keys function reads of a context: the context function's output
/// rows and the mask over them. A keystroke without a context reads zeros
/// under an all-zero mask, which the function's in-graph switch reads as none.
struct Context {
    rows: Tensor<f16>,
    mask: Tensor<f16>,
}

/// The context encoder: the tokenizer, the function, and the last text's
/// encoding.
struct ContextEncoder {
    function: Function,
    tokenizer: Tokenizer,
    /// The context function's length, sentinels included.
    tokens: usize,
    pad: i32,
    /// No context: zeros.
    none: Context,
    /// The latest context text and its encoding.
    last: Option<(Box<str>, Context)>,
}

impl ContextEncoder {
    /// The encoding of *text*, run unless it is the one just encoded.
    fn encode(&mut self, text: Option<&str>) -> Result<&Context, CoreMlKeyError> {
        let Some(text) = text else {
            return Ok(&self.none);
        };
        let pair = match self.last.take() {
            Some((held, context)) if &*held == text => (held, context),
            _ => (text.into(), self.run(text)?),
        };
        Ok(&self.last.insert(pair).1)
    }

    /// The context function over the last `tokens - 2` characters of *text*,
    /// which are the ones beside the cursor, tokenized and padded.
    fn run(&mut self, text: &str) -> Result<Context, CoreMlKeyError> {
        let tail: String = text
            .chars()
            .rev()
            .take(self.tokens - KEY_SENTINELS)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        let encoding = self
            .tokenizer
            .encode(tail.as_str(), true)
            .map_err(|source| CoreMlKeyError::Tokenizer {
                path: PathBuf::from(TOKENIZER),
                source,
            })?;
        let real = encoding.get_ids().len();
        if real > self.tokens {
            return Err(CoreMlKeyError::ContextTooLong {
                tokens: real,
                capacity: self.tokens,
            });
        }
        let mut ids = vec![self.pad; self.tokens];
        for (slot, &id) in ids.iter_mut().zip(encoding.get_ids()) {
            *slot = i32::try_from(id).map_err(|_| CoreMlKeyError::Int32("a token id"))?;
        }
        let mut mask = vec![f16::ZERO; self.tokens];
        for (slot, &bit) in mask.iter_mut().zip(encoding.get_attention_mask()) {
            *slot = if bit == 0 { f16::ZERO } else { f16::ONE };
        }
        let mask = Tensor::from_slice(&[1, self.tokens], &mask)?;
        let mut inputs = Inputs::new();
        inputs.insert("context_ids", &Tensor::from_slice(&[1, self.tokens], &ids)?)?;
        inputs.insert("context_mask", &mask)?;
        let rows = self.function.predict(&inputs)?.tensor::<f16>("context")?;
        debug!(tokens = real, "encoded a context");
        Ok(Context { rows, mask })
    }
}

/// One keys function: the keys it holds, its width, and the function.
struct KeysFunction {
    capacity: usize,
    function: Function,
}

/// One tree function and the nodes it holds.
struct TreeFunction {
    bucket: usize,
    function: Function,
}

/// One keystroke's keys function: what the tree calls read.
///
/// The cross keys and values are the function's own outputs, left in Core ML
/// storage, and `bias` masks their unused slots.
pub struct CoreMlEncoding {
    keys: Tensor<f16>,
    values: Tensor<f16>,
    bias: Tensor<f16>,
}

/// The tree function's node inputs: per node its reader id and depth, and the
/// additive bias of the nodes over each other.
struct TreeRow {
    reader_ids: Vec<i32>,
    positions: Vec<i32>,
    bias: Vec<f16>,
}

/// One tree call's logits: float16 `[1, bucket, emittable]`, in Core ML
/// storage until a row is read, and then on the host.
pub struct CoreMlLogits {
    tensor: Tensor<f16>,
    host: OnceCell<Vec<f16>>,
}

/// The converted key model: three function families, each function loaded.
pub struct CoreMlKeyModel {
    step: u64,
    correct_syllables: bool,
    characters: Vec<char>,
    specials: Specials,
    letter_pad: i32,
    reader_ids: Vec<i32>,
    reader_bos: i32,
    neg: f32,
    n_max: usize,
    context: ContextEncoder,
    keys: Vec<KeysFunction>,
    tree: Vec<TreeFunction>,
}

impl CoreMlKeyModel {
    /// Open the unpacked archive in *directory* and load every function for
    /// *units*.
    ///
    /// # Errors
    ///
    /// If a manifest is missing, malformed or describes another model than the
    /// bundle's, the tokenizer cannot be read, or Core ML cannot load a
    /// function.
    pub fn open(directory: &Path, units: ComputeUnits) -> Result<Self, CoreMlKeyError> {
        let started = Instant::now();
        let path = directory.join(MANIFEST);
        let manifest: KeyManifest = read_json(&path)?;
        manifest.check(&path)?;

        let tokenizer_path = directory.join(TOKENIZER);
        let tokenizer_error = |source| CoreMlKeyError::Tokenizer {
            path: tokenizer_path.clone(),
            source,
        };
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(tokenizer_error)?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: manifest.context_tokens,
                ..TruncationParams::default()
            }))
            .map_err(tokenizer_error)?;

        let [(_, context)] =
            <[_; 1]>::try_from(load_family(directory, Family::Context, &manifest, units)?)
                .map_err(|_| refuse(&path, "the context family is not one function"))?;
        let tokens = manifest.context_tokens;
        let pad = manifest.specials.pad;
        let hidden = manifest.dims.fill.hidden;
        let none = Context {
            rows: Tensor::from_slice(&[1, tokens, hidden], &vec![f16::ZERO; tokens * hidden])?,
            mask: Tensor::from_slice(&[1, tokens], &vec![f16::ZERO; tokens])?,
        };
        let keys = load_family(directory, Family::Keys, &manifest, units)?
            .into_iter()
            .map(|(capacity, function)| KeysFunction { capacity, function })
            .collect();
        let tree = TREE_BUCKETS
            .into_iter()
            .zip(load_family(directory, Family::Tree, &manifest, units)?)
            .map(|(bucket, (_, function))| TreeFunction { bucket, function })
            .collect();
        info!(
            step = manifest.step,
            seconds = started.elapsed().as_secs_f64(),
            "opened the Core ML key model"
        );
        Ok(Self {
            step: manifest.step,
            correct_syllables: manifest.correct_syllables,
            characters: manifest.characters,
            specials: manifest.specials,
            letter_pad: manifest.letter_pad,
            reader_ids: manifest.reader_ids,
            reader_bos: manifest.reader_bos,
            neg: manifest.neg,
            n_max: manifest.n_max,
            context: ContextEncoder {
                function: context,
                tokenizer,
                tokens,
                pad,
                none,
                last: None,
            },
            keys,
            tree,
        })
    }

    /// The training step the bundle was written at.
    #[must_use]
    pub const fn step(&self) -> u64 {
        self.step
    }

    /// Whether the model was trained to read a span that is itself a syllable
    /// as a correction: the [`ime_pinyin::LegalityOptions::correct_syllables`]
    /// it is searched under.
    #[must_use]
    pub const fn correct_syllables(&self) -> bool {
        self.correct_syllables
    }

    /// The emittable characters in emission order: the rows of the head and
    /// of the decoder. A [`ime_pinyin::Legality`] over the same characters
    /// names the same rows.
    #[must_use]
    pub fn characters(&self) -> &[char] {
        &self.characters
    }

    /// The keys function's row inputs: `[CLS] [MASK]*n [SEP] [PAD]*` at
    /// *width*, the letters of the keys, and the mask over the real columns.
    fn key_row(&self, keys: &Keys, width: usize) -> (Vec<i32>, Vec<i32>, Vec<f16>) {
        let count = keys.len();
        let specials = &self.specials;
        let mut ids = vec![specials.pad; width];
        ids[0] = specials.cls;
        ids[1..=count].fill(specials.mask);
        ids[count + 1] = specials.sep;
        let mut letters = vec![self.letter_pad; width];
        for (slot, key) in letters[1..=count].iter_mut().zip(keys.as_str().bytes()) {
            *slot = i32::from(key - b'a');
        }
        let mut attention = vec![f16::ZERO; width];
        attention[..count + KEY_SENTINELS].fill(f16::ONE);
        (ids, letters, attention)
    }

    /// The tree function's node inputs for *call*: the reader ids, depths and
    /// additive bias.
    ///
    /// A node's token is its prefix's last character as the reader's id,
    /// `<bos>` for the root and for padding; a node sees its ancestors and
    /// itself, a padding row only itself.
    fn tree_row(&self, call: &TreeCall<'_, EmitId>) -> Result<TreeRow, CoreMlKeyError> {
        let size = call.bucket();
        let mut reader_ids = vec![self.reader_bos; size];
        let mut positions = vec![0_i32; size];
        let mut visible = vec![false; size * size];
        for row in 0..size {
            visible[row * size + row] = true;
        }
        for (row, node) in call.nodes().iter().enumerate() {
            if let Some(last) = node.last() {
                reader_ids[row] = self.reader_ids[last.index()];
            }
            positions[row] =
                i32::try_from(node.len()).map_err(|_| CoreMlKeyError::Int32("a prefix's depth"))?;
            if let Some(parent) = call.parents()[row] {
                // A parent's row precedes its child's, so it is in `before`.
                let (before, rest) = visible.split_at_mut(row * size);
                for (slot, &seen) in rest[..size]
                    .iter_mut()
                    .zip(&before[parent * size..(parent + 1) * size])
                {
                    *slot |= seen;
                }
            }
        }
        let hidden = f16::from_f32(self.neg);
        let bias = visible
            .iter()
            .map(|&seen| if seen { f16::ZERO } else { hidden })
            .collect();
        Ok(TreeRow {
            reader_ids,
            positions,
            bias,
        })
    }
}

/// The row-wise log-softmax of the float16 *row* into *out*, in float32: the
/// maximum subtracted, then the log of the sum of exponentials.
fn log_softmax_into(row: &[f16], out: &mut [f32]) {
    for (slot, value) in out.iter_mut().zip(row) {
        *slot = value.to_f32();
    }
    let top = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for slot in out.iter_mut() {
        *slot -= top;
    }
    let norm = out.iter().copied().map(f32::exp).sum::<f32>().ln();
    for slot in out.iter_mut() {
        *slot -= norm;
    }
}

/// Refuse an output that is not the shape its manifest declares.
fn check_shape(
    function: &str,
    name: &'static str,
    tensor: &Tensor<f16>,
    expected: &[usize],
) -> Result<(), CoreMlKeyError> {
    if tensor.shape() == expected {
        Ok(())
    } else {
        Err(CoreMlKeyError::OutputShape {
            function: function.to_owned(),
            name,
            got: tensor.shape().to_vec(),
            expected: expected.to_vec(),
        })
    }
}

impl KeyModel for CoreMlKeyModel {
    type Error = CoreMlKeyError;
    type Encoding = CoreMlEncoding;
    type Logits = CoreMlLogits;

    fn max_keys(&self) -> usize {
        self.keys.last().map_or(0, |held| held.capacity)
    }

    fn encode(
        &mut self,
        keys: &Keys,
        context: Option<&str>,
        log_probs: &mut Vec<f32>,
    ) -> Result<CoreMlEncoding, CoreMlKeyError> {
        let count = keys.len();
        let largest = self.max_keys();
        let position = self
            .keys
            .iter()
            .position(|held| held.capacity >= count)
            .ok_or(CoreMlKeyError::TooManyKeys {
                keys: count,
                largest,
            })?;
        let width = self.keys[position].capacity + KEY_SENTINELS;
        let (ids, letters, attention) = self.key_row(keys, width);
        let context = self.context.encode(context)?;
        let mut inputs = Inputs::new();
        inputs.insert("input_ids", &Tensor::from_slice(&[1, width], &ids)?)?;
        inputs.insert("key_letters", &Tensor::from_slice(&[1, width], &letters)?)?;
        inputs.insert(
            "attention_mask",
            &Tensor::from_slice(&[1, width], &attention)?,
        )?;
        inputs.insert("context", &context.rows)?;
        inputs.insert("context_mask", &context.mask)?;
        let name = format!("keys_{width}");
        let out = self.keys[position].function.predict(&inputs)?;

        // The head's rows over the real keys, the blank last; the host's
        // float32 log-softmax.
        let columns = self.characters.len() + 1;
        let ctc = out.tensor::<f16>("ctc_logits")?;
        check_shape(&name, "ctc_logits", &ctc, &[1, width, columns])?;
        let data = ctc.read().into_data();
        log_probs.clear();
        log_probs.resize(count * columns, 0.0);
        for (key, row) in log_probs.chunks_exact_mut(columns).enumerate() {
            let at = (key + 1) * columns;
            log_softmax_into(&data[at..at + columns], row);
        }

        let mut bias = vec![f16::from_f32(self.neg); self.n_max];
        bias[..count].fill(f16::ZERO);
        debug!(keys = count, width, "ran the keys function");
        Ok(CoreMlEncoding {
            keys: out.tensor("cross_keys")?,
            values: out.tensor("cross_values")?,
            bias: Tensor::from_slice(&[1, 1, 1, self.n_max], &bias)?,
        })
    }

    fn tree(
        &mut self,
        encoding: &CoreMlEncoding,
        call: &TreeCall<'_, EmitId>,
    ) -> Result<CoreMlLogits, CoreMlKeyError> {
        let row = self.tree_row(call)?;
        self.run_tree(encoding, call.bucket(), &row, call.nodes().len())
    }

    fn log_softmax(&self, logits: &CoreMlLogits, node: usize, out: &mut [f32]) {
        let width = out.len();
        let host = logits.host.get_or_init(|| logits.tensor.read().into_data());
        log_softmax_into(&host[node * width..(node + 1) * width], out);
    }
}

impl CoreMlKeyModel {
    /// Run the tree function of *size* nodes over *encoding* and *row*;
    /// *nodes* is how many of the rows carry a node, for the log.
    fn run_tree(
        &mut self,
        encoding: &CoreMlEncoding,
        size: usize,
        row: &TreeRow,
        nodes: usize,
    ) -> Result<CoreMlLogits, CoreMlKeyError> {
        let TreeRow {
            reader_ids,
            positions,
            bias,
        } = row;
        let held = self
            .tree
            .iter_mut()
            .find(|held| held.bucket == size)
            .ok_or(CoreMlKeyError::TreeBucket(size))?;
        let mut inputs = Inputs::new();
        inputs.insert("reader_ids", &Tensor::from_slice(&[1, size], reader_ids)?)?;
        inputs.insert("positions", &Tensor::from_slice(&[1, size], positions)?)?;
        inputs.insert("tree_bias", &Tensor::from_slice(&[1, 1, size, size], bias)?)?;
        inputs.insert("cross_keys", &encoding.keys)?;
        inputs.insert("cross_values", &encoding.values)?;
        inputs.insert("cross_bias", &encoding.bias)?;
        let tensor = held.function.predict(&inputs)?.tensor::<f16>("logits")?;
        check_shape(
            &format!("tree_{size}"),
            "logits",
            &tensor,
            &[1, size, self.characters.len()],
        )?;
        debug!(bucket = size, nodes, "ran a tree function");
        Ok(CoreMlLogits {
            tensor,
            host: OnceCell::new(),
        })
    }

    /// Run every loaded function once, so the first keystroke pays none of
    /// the first call's cost: Core ML's lazy allocation and the Neural
    /// Engine's program load happen on the first prediction, not at
    /// [`CoreMlKeyModel::open`].
    ///
    /// The inputs are valid and carry nothing of a user's: a short context
    /// text, keys filling each keys function, and tree rows that are all
    /// padding. Each function's seconds go to `tracing`. No state a keystroke
    /// reads is left behind: the context encoder's last text is untouched.
    /// Call it off the typing path, once after `open`.
    ///
    /// # Errors
    ///
    /// If a function fails to run.
    ///
    /// # Panics
    ///
    /// If a keys function holds no key, which no converted model does.
    pub fn warm_up(&mut self) -> Result<(), CoreMlKeyError> {
        let started = Instant::now();
        let tick = Instant::now();
        self.context.run(WARM_CONTEXT)?;
        info!(
            function = "context",
            seconds = tick.elapsed().as_secs_f64(),
            "warmed a function"
        );
        let mut log_probs = Vec::new();
        let mut encoding = None;
        for position in 0..self.keys.len() {
            let capacity = self.keys[position].capacity;
            let keys = Keys::new(&"a".repeat(capacity))
                .expect("a function holds at least one key, and `a` is a key");
            let tick = Instant::now();
            // The smallest function that holds `capacity` keys is this one:
            // the capacities ascend.
            let encoded = self.encode(&keys, None, &mut log_probs)?;
            info!(
                function = format!("keys_{}", capacity + KEY_SENTINELS),
                seconds = tick.elapsed().as_secs_f64(),
                "warmed a function"
            );
            encoding.get_or_insert(encoded);
        }
        let Some(encoding) = encoding else {
            return Ok(());
        };
        let mut row = vec![0.0; self.characters.len()];
        let buckets: Vec<usize> = self.tree.iter().map(|held| held.bucket).collect();
        for size in buckets {
            let tick = Instant::now();
            let padding = self.padding_row(size);
            let logits = self.run_tree(&encoding, size, &padding, 0)?;
            self.log_softmax(&logits, 0, &mut row);
            info!(
                function = format!("tree_{size}"),
                seconds = tick.elapsed().as_secs_f64(),
                "warmed a function"
            );
        }
        info!(
            seconds = started.elapsed().as_secs_f64(),
            "warmed the Core ML key model"
        );
        Ok(())
    }

    /// A tree call's inputs with no node: every row is padding, which sees
    /// only itself.
    fn padding_row(&self, size: usize) -> TreeRow {
        let hidden = f16::from_f32(self.neg);
        let mut bias = vec![hidden; size * size];
        for row in 0..size {
            bias[row * size + row] = f16::ZERO;
        }
        TreeRow {
            reader_ids: vec![self.reader_bos; size],
            positions: vec![0; size],
            bias,
        }
    }
}
