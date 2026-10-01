//! Route A's context encoder and fill decoder over ONNX Runtime.
//!
//! `mlime export route-a` writes the directory this crate opens:
//! `route-a.json`, a manifest that names the graphs' tensors and holds both
//! vocabularies, the typed spans and the emission alphabet; `context.onnx`,
//! the encoder over the text already on screen; `fill.onnx`, the
//! non-autoregressive decoder over the lattice's readings; the two graphs'
//! shared weights files; and `tokenizer.json`, the encoder's input
//! vocabulary. The manifest's `layout` pins the way the towers are wired, and
//! a manifest without the one this build reads is refused at open -- a graph
//! written for another layout would load and run until an input the build
//! does not feed failed it.
//!
//! The graphs split because the encoder's output depends on the context
//! alone: the decoder scores every keystroke, so the encoder running once
//! per record rather than once per path is most of the input method's
//! latency budget. A record typed with nothing before it sets `has_context`
//! to zero, and the cross attention's gate zeroes the tower's contribution
//! exactly, so the context tower is skipped and the fill graph is fed a
//! zeros row.
//!
//! [`RouteA::emission`] answers one [`LatticeRecord`] with what one line of
//! `mlime train emit`'s score file holds before the writer's rounding: for
//! every reading, every position, a raw log probability per asked candidate,
//! in the lattice's own order. [`RouteA::scored`] binds the same table to
//! the lattice it answers as a [`Scored`], the decoder's [`Emission`].
//!
//! Sessions run through `&mut self`, and the decoder scores records in
//! parallel, so each thread opens its own pair of sessions from the same
//! graphs the first time it needs one. Every session is built over the
//! graphs' one shared mapping of the weights: its initializers point into the
//! mapping rather than a per-session copy, and the matrices the runtime
//! pre-packs are shared too, so the model's memory is one copy of the weights
//! plus per-thread working space no matter how many threads score.

use ime_decode::{Candidates, EmissionError, Emittable, LatticeRecord, Scored};
use ime_pinyin::CorrectionTable;
use memmap2::Mmap;
use ort::AsPointer;
use ort::ep::ExecutionProviderDispatch;
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::Session;
use ort::session::builder::{GraphOptimizationLevel, PrepackedWeights};
use ort::value::{DynValue, Outlet, PrimitiveTensorElementType, Shape, Tensor, TensorRefMut};
use serde::Deserialize;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs;
use std::mem::size_of;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::Arc;
use thiserror::Error;
use thread_local::ThreadLocal;
use tokenizers::{Tokenizer, TruncationParams};
use tracing::info;

/// Which execution provider a model's sessions run on.
///
/// `cpu` is the decoder's default and always built in; `coreml` (Apple's ANE/
/// GPU), `webgpu` (Dawn over Metal) and `cuda` (NVIDIA's provider) exist
/// behind the `gpu-coreml`, `gpu-webgpu` and `gpu-cuda` cargo features, and
/// asking for one that was not compiled in is an error at [`RouteA::open`],
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
    fn providers(self) -> Result<Vec<ExecutionProviderDispatch>, NeuralError> {
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
            _ => Err(NeuralError::NotCompiled { backend: self }),
        }
    }
}

/// How a model's sessions are built: the provider they register, the threads
/// one forward spreads over, and whether the runtime logs verbosely.
///
/// The caller states the shape the decode will run, because the decoder is
/// the parallel layer and the session must not fight it: the per-record
/// rayon path gives every session one intra-op thread. The fill graph's
/// batch is the record's readings -- the path count, which the lattice
/// bounds at eight.
#[derive(Debug, Clone, Copy)]
pub struct SessionShape {
    /// The execution provider the sessions register.
    pub backend: Backend,
    /// The threads one session spreads a forward over; the default is one,
    /// the rayon path's shape.
    pub intra_threads: NonZeroUsize,
    /// ONNX Runtime's verbose session logging: the provider each node lands
    /// on is the evidence for whether a backend actually runs the step.
    pub verbose_logging: bool,
}

impl Default for SessionShape {
    fn default() -> Self {
        Self {
            backend: Backend::Cpu,
            intra_threads: NonZeroUsize::MIN,
            verbose_logging: false,
        }
    }
}

/// The context tower's output for one context, kept for reuse.
///
/// It encodes only the context's text, so a session whose context is
/// unchanged across keystrokes computes it once: [`RouteA::context`] makes
/// one, [`RouteA::emission_with`] consumes it.
#[derive(Clone, Debug)]
pub struct EncodedContext {
    /// The tower's last hidden state, `[tokens, hidden]` flattened.
    hidden: Vec<f32>,
    /// The attention mask the fill decoder reads alongside.
    mask: Vec<i64>,
}

/// Route A's towers, loaded from an `mlime export route-a` directory.
///
/// One `RouteA` serves every decoding thread at once: the graphs' weights
/// are one shared mapping, and each thread that scores opens its own
/// sessions over them on first use.
#[derive(Debug)]
pub struct RouteA {
    /// The context encoder's sessions, one per thread.
    context_sessions: GraphSessions,
    /// The fill decoder's sessions, one per thread.
    fill_sessions: GraphSessions,
    /// The context tower's input vocabulary, truncating to `context_tokens`.
    tokenizer: Tokenizer,
    /// The widest context the encoder reads, sentinels included.
    context_tokens: usize,
    /// The towers' hidden size, which shapes the context tensor.
    hidden: usize,
    /// `<pad>`/`<cls>`/`<sep>`/`<mask>`, as the manifest records them.
    specials: Specials,
    /// Each typed span's id: `model.span_embeddings`' row and a `span_ids`
    /// value, in the manifest's order.
    spans: HashMap<String, usize>,
    /// The reserved ``<unk>`` row's id: the row a typoed span embeds through,
    /// since a span no syllable's prefix covers has no row of its own.
    unknown_span: usize,
    /// Each character's index into the emission axis: `log_probs`' last axis
    /// and a `candidate_mask` column, in the manifest's order.
    emissions: HashMap<char, usize>,
    /// `candidate_mask`, flattened row-major over `[spans, emissions]`: which
    /// spans may emit which characters.
    mask: Vec<bool>,
    /// The mapping(s) the shared initializer `OrtValue`s point into.
    ///
    /// Declared after the sessions so field drop order unmaps the file only
    /// after everything reading it is gone -- the values' data pointers are
    /// raw, so nothing else keeps this alive.
    #[expect(
        dead_code,
        reason = "the field is held only for its Drop, which field order keeps after the sessions and values it backs"
    )]
    weights: Vec<Mmap>,
}

impl RouteA {
    /// Open the export *dir* holds, refusing anything that does not describe
    /// the towers this build drives.
    ///
    /// # Errors
    ///
    /// If the directory is missing files, the manifest is malformed or names
    /// another layout, a weights file disagrees with the manifest, the
    /// tokenizer cannot be read, or ONNX Runtime refuses a graph.
    ///
    /// # Panics
    ///
    /// If a weights map the open walk just inserted is absent again — a bug,
    /// not a state the caller can recover.
    pub fn open(dir: &Path, shape: SessionShape) -> Result<Self, NeuralError> {
        let manifest = read_manifest(dir)?;
        if manifest.context_tokens < 3 {
            return Err(NeuralError::ManifestShape {
                path: dir.to_path_buf(),
                reason: format!(
                    "a context needs room for two sentinels and a character, got {} tokens",
                    manifest.context_tokens
                ),
            });
        }
        let tokenizer_path = dir.join("tokenizer.json");
        let mut tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|source| NeuralError::Tokenizer {
                path: tokenizer_path,
                source,
            })?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: manifest.context_tokens,
                ..TruncationParams::default()
            }))
            .map_err(|source| NeuralError::Tokenizer {
                path: dir.join("tokenizer.json"),
                source,
            })?;
        let (emissions, spans, unknown_span) = vocabularies(&manifest, dir)?;

        let mut maps = HashMap::new();
        let context_table = &manifest.weights.context;
        let fill_table = &manifest.weights.fill;
        let context_initializers = shared_values(
            weights_map(&mut maps, dir, &context_table.file)?,
            context_table,
        )?;
        let fill_initializers =
            shared_values(weights_map(&mut maps, dir, &fill_table.file)?, fill_table)?;
        // The mask is registered as an initializer with the fill sessions, but
        // the table it answers -- the lattice's candidates -- needs the bytes
        // themselves: reading them out of the same bytes keeps the file's
        // bits authoritative.
        let mask = read_mask(
            maps.get(&fill_table.file)
                .expect("the fill weights map was opened above"),
            fill_table,
            &manifest.candidate_mask,
            manifest.spans.len(),
            manifest.characters.len(),
            dir,
        )?;

        // Sharing pre-packed matrices is what the fp32 export supports; a
        // quantized graph's prepacks are not enrolled in the shared container,
        // so int8 sessions each pack their own.
        let shareable = manifest.quantize.is_none();
        let context_sessions = GraphSessions::new(
            dir.join(&manifest.graphs.context),
            context_initializers,
            shape,
            shareable,
        );
        let fill_sessions = GraphSessions::new(
            dir.join(&manifest.graphs.fill),
            fill_initializers,
            shape,
            shareable,
        );
        let model = Self {
            context_sessions,
            fill_sessions,
            tokenizer,
            context_tokens: manifest.context_tokens,
            hidden: manifest.hidden,
            specials: manifest.specials,
            spans,
            unknown_span,
            emissions,
            mask,
            weights: maps.into_values().collect(),
        };
        // Fail a graph that does not declare the inputs this build feeds at
        // open rather than at the first emission.
        model.context_sessions.check(CONTEXT_INPUTS)?;
        model.fill_sessions.check(FILL_INPUTS)?;
        info!(
            step = manifest.step,
            base_model = %manifest.base_model,
            dtype = %manifest.dtype,
            quantize = ?manifest.quantize,
            spans = manifest.spans.len(),
            emissions = manifest.characters.len(),
            hidden = manifest.hidden,
            "route A model opened"
        );
        Ok(model)
    }

    /// Score one record's lattice: the context tower once, the fill decoder
    /// once over every reading, and each position's raw log probability per
    /// asked candidate -- one [`ScoreRecord`]'s `paths`. Rounding to the
    /// score file's four decimals is the writer's business: the product path
    /// keeps the full-precision values.
    ///
    /// *`with_context`* switches the towers between the `context on` and
    /// `context off` score files: off, or a record with no context, runs the
    /// fill graph with the gate zeroed and the context tower skipped.
    ///
    /// *corrections*, when set, is the shared typo noise model: an
    /// off-inventory span embeds through the `<unk>` row, and a position's
    /// admission check is the union over its corrections' mask rows rather
    /// than the span's own -- the same widening the training side's
    /// `CandidateSpace.resolve` applies.
    ///
    /// [`ScoreRecord`]: ime_decode::ScoreRecord
    ///
    /// # Errors
    ///
    /// If a span or a candidate the lattice asks about is not in the
    /// export's vocabulary, the candidate mask refuses it, or ONNX Runtime
    /// fails a forward.
    pub fn emission(
        &self,
        record: &LatticeRecord,
        corrections: Option<&CorrectionTable>,
        with_context: bool,
    ) -> Result<Vec<Vec<Vec<f32>>>, NeuralError> {
        let context = match record.context.as_deref().filter(|_| with_context) {
            Some(text) => Some(self.context(text)?),
            None => None,
        };
        self.emission_with(record, corrections, context.as_ref())
    }

    /// The context tower's output for *text*: its hidden states and the
    /// mask that attends them. The output depends only on the context, so a
    /// caller whose context survives the next keystroke encodes it once --
    /// the session keeps it across `key()` calls and hands it back through
    /// [`RouteA::emission_with`].
    ///
    /// # Errors
    ///
    /// If the tokenizer fails on *text* or ONNX Runtime fails the forward.
    pub fn context(&self, text: &str) -> Result<EncodedContext, NeuralError> {
        let (ids, mask) = self.encode_context(text)?;
        Ok(EncodedContext {
            hidden: self.context_hidden(&ids, &mask)?,
            mask,
        })
    }

    /// The fill decoder over every reading of *record*, against an
    /// [`EncodedContext`] already computed: `None` runs the gate zeroed, the
    /// `context off` score file's live twin. This is [`RouteA::emission`]
    /// with the context tower's step taken outside. *corrections* is the
    /// same typo noise model [`RouteA::emission`] takes.
    ///
    /// # Errors
    ///
    /// As [`RouteA::emission`]'s, minus the tokenizer's.
    pub fn emission_with(
        &self,
        record: &LatticeRecord,
        corrections: Option<&CorrectionTable>,
        context: Option<&EncodedContext>,
    ) -> Result<Vec<Vec<Vec<f32>>>, NeuralError> {
        if record.paths.is_empty() {
            return Err(NeuralError::Malformed {
                record: record.record,
                reason: "a lattice record with no paths".to_owned(),
            });
        }
        let FillInputs {
            width,
            input_ids,
            attention_mask,
            span_ids,
            span_letters,
            span_positions,
            asked,
        } = self.fill_inputs(record, corrections)?;
        let rows = record.paths.len();
        // The context tower once per record; a gated-off context is a zeros
        // row, bitwise equivalent to encoding the sentinels and multiplying
        // the gate by zero.
        let gated_hidden = vec![0.0; self.hidden];
        let gated_mask = [0i64];
        let (hidden_state, context_mask) = if let Some(context) = context {
            (context.hidden.as_slice(), context.mask.as_slice())
        } else {
            (gated_hidden.as_slice(), gated_mask.as_slice())
        };
        let context_tokens = context_mask.len();

        let mut context_rows = vec![0.0f32; rows * context_tokens * self.hidden];
        let mut context_masks = vec![0i64; rows * context_tokens];
        for row in 0..rows {
            context_rows
                [row * context_tokens * self.hidden..(row + 1) * context_tokens * self.hidden]
                .copy_from_slice(hidden_state);
            context_masks[row * context_tokens..(row + 1) * context_tokens]
                .copy_from_slice(context_mask);
        }
        let has = vec![if context.is_some() { 1.0f32 } else { 0.0f32 }; rows];

        let session = self.fill_sessions.session()?;
        let mut session = session.borrow_mut();
        let outputs = session.run(ort::inputs![
            "input_ids" => Tensor::from_array((vec![dim(rows), dim(width)], input_ids))?,
            "attention_mask" => Tensor::from_array((vec![dim(rows), dim(width)], attention_mask))?,
            "span_ids" => Tensor::from_array((vec![dim(rows), dim(width)], span_ids))?,
            "span_letters" => Tensor::from_array((
                vec![dim(rows), dim(width), dim(MAX_SPAN_LETTERS)],
                span_letters
            ))?,
            "span_positions" =>
                Tensor::from_array((vec![dim(rows), dim(width)], span_positions))?,
            "context" => Tensor::from_array((
                vec![dim(rows), dim(context_tokens), dim(self.hidden)],
                context_rows
            ))?,
            "context_mask" =>
                Tensor::from_array((vec![dim(rows), dim(context_tokens)], context_masks))?,
            "has_context" => Tensor::from_array((vec![dim(rows)], has))?,
        ])?;
        let (shape, log_probs) = outputs["log_probs"].try_extract_tensor::<f32>()?;
        let found: Vec<i64> = shape.iter().copied().collect();
        let expected = [dim(rows), dim(width), dim(self.emissions.len())];
        if found != expected {
            return Err(NeuralError::Shape {
                tensor: "log_probs",
                expected: expected.to_vec(),
                found,
            });
        }

        Ok(asked_scores(&asked, log_probs, width, self.emissions.len()))
    }

    /// The fill graph's inputs for *record*: the `[CLS, MASK x n, SEP]`
    /// batch of its readings, the span layout the fill embeddings answer,
    /// and every asked candidate resolved to its emission index with the
    /// candidate mask checked -- a candidate the model was not trained to
    /// admit is an error, never a floor score.
    fn fill_inputs(
        &self,
        record: &LatticeRecord,
        corrections: Option<&CorrectionTable>,
    ) -> Result<FillInputs, NeuralError> {
        let width = record
            .paths
            .iter()
            .fold(0usize, |width, path| width.max(path.spans.len() + 2));
        let rows = record.paths.len();
        let mut span_ids = vec![0i64; rows * width];
        let mut span_letters = vec![LETTER_PAD; rows * width * MAX_SPAN_LETTERS];
        let mut asked: Vec<Vec<Vec<usize>>> = Vec::with_capacity(rows);
        let mut input_ids = vec![i64::from(self.specials.pad); rows * width];
        let mut attention_mask = vec![0i64; rows * width];
        let mut span_positions = vec![false; rows * width];
        for (row, path) in record.paths.iter().enumerate() {
            if path.spans.len() != path.candidates.len() {
                return Err(NeuralError::Malformed {
                    record: record.record,
                    reason: format!(
                        "reading {row} has {} spans and {} candidate lists",
                        path.spans.len(),
                        path.candidates.len()
                    ),
                });
            }
            let base = row * width;
            input_ids[base] = i64::from(self.specials.cls);
            input_ids[base + path.spans.len() + 1] = i64::from(self.specials.sep);
            attention_mask[base..base + path.spans.len() + 2].fill(1);
            let mut positions = Vec::with_capacity(path.spans.len());
            for (position, (span, candidates)) in
                path.spans.iter().zip(&path.candidates).enumerate()
            {
                let (span_id, admitted) = self.span_layout(record.record, span, corrections)?;
                input_ids[base + position + 1] = i64::from(self.specials.mask);
                span_ids[base + position + 1] = dim(span_id);
                span_letters[(base + position + 1) * MAX_SPAN_LETTERS
                    ..(base + position + 2) * MAX_SPAN_LETTERS]
                    .copy_from_slice(&letter_ids(record.record, span)?);
                span_positions[base + position + 1] = true;
                let mut emissions = Vec::with_capacity(candidates.chars().count());
                for character in candidates.chars() {
                    let emission =
                        *self
                            .emissions
                            .get(&character)
                            .ok_or(NeuralError::Vocabulary {
                                record: record.record,
                                character,
                            })?;
                    let admitted_here = admitted
                        .iter()
                        .any(|row| self.mask[row * self.emissions.len() + emission]);
                    if !admitted_here {
                        return Err(NeuralError::Unadmitted {
                            record: record.record,
                            span: span.clone(),
                            character,
                        });
                    }
                    emissions.push(emission);
                }
                positions.push(emissions);
            }
            asked.push(positions);
        }
        Ok(FillInputs {
            width,
            input_ids,
            attention_mask,
            span_ids,
            span_letters,
            span_positions,
            asked,
        })
    }

    /// The same table bound to the lattice it answers: the decoder's
    /// [`Emission`] over this record.
    ///
    /// [`Emission`]: ime_decode::Emission
    ///
    /// # Errors
    ///
    /// Forwards [`RouteA::emission`]'s failures, plus
    /// [`Scored::attach`]'s when the computed table and the lattice disagree
    /// -- which they cannot unless a caller built the `LatticeRecord` by a
    /// different route than the decoder took.
    pub fn scored<'a>(
        &self,
        record: &LatticeRecord,
        candidates: &'a Candidates,
        emittable: &Emittable,
        floor: f32,
        with_context: bool,
    ) -> Result<Scored<'a>, NeuralError> {
        let scores = self.emission(record, None, with_context)?;
        Ok(Scored::attach(
            record.record,
            candidates,
            emittable,
            scores,
            floor,
        )?)
    }

    /// The context encoder's token ids and mask for *text*: the last
    /// `context_tokens - 2` characters, which are the ones beside the cursor,
    /// tokenized and truncated to the tower's width.
    fn encode_context(&self, text: &str) -> Result<(Vec<i64>, Vec<i64>), NeuralError> {
        let tail: String = text
            .chars()
            .rev()
            .take(self.context_tokens - 2)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        let encoding = self
            .tokenizer
            .encode(tail.as_str(), true)
            .map_err(|source| NeuralError::Tokenizer {
                path: PathBuf::from("tokenizer.json"),
                source,
            })?;
        let ids = encoding.get_ids().iter().map(|&id| i64::from(id)).collect();
        let mask = encoding
            .get_attention_mask()
            .iter()
            .map(|&bit| i64::from(bit))
            .collect();
        Ok((ids, mask))
    }

    /// The span-table row *span* embeds through and the mask rows a
    /// candidate at its position must be admitted by.
    ///
    /// Without the noise model the rows are the span's own row alone, today's
    /// check. With it, the span's `corrections` name the rows: an
    /// off-inventory span embeds through the reserved ``<unk>`` row while
    /// every correction's row contributes to the union -- the mask a typoed
    /// span asks about is the mask of every syllable it could have meant.
    fn span_layout(
        &self,
        record: usize,
        span: &str,
        corrections: Option<&CorrectionTable>,
    ) -> Result<(usize, Vec<usize>), NeuralError> {
        if let Some(noise) = corrections {
            let id = self.spans.get(span).copied().unwrap_or(self.unknown_span);
            let rows = noise
                .corrections(span)
                .iter()
                .map(|entry| {
                    self.spans
                        .get(entry.syllable())
                        .copied()
                        .ok_or(NeuralError::Span {
                            record,
                            span: entry.syllable().to_owned(),
                        })
                })
                .collect::<Result<Vec<usize>, NeuralError>>()?;
            return Ok((id, rows));
        }
        let id = *self.spans.get(span).ok_or(NeuralError::Span {
            record,
            span: span.to_owned(),
        })?;
        Ok((id, vec![id]))
    }

    /// The context tower's last hidden state over the encoded context:
    /// `[1, tokens, hidden]`, flattened.
    fn context_hidden(&self, ids: &[i64], mask: &[i64]) -> Result<Vec<f32>, NeuralError> {
        let session = self.context_sessions.session()?;
        let mut session = session.borrow_mut();
        let outputs = session.run(ort::inputs![
            "context_ids" =>
                Tensor::from_array((vec![1, dim(ids.len())], ids.to_vec()))?,
            "context_mask" =>
                Tensor::from_array((vec![1, dim(mask.len())], mask.to_vec()))?,
        ])?;
        let (shape, hidden) = outputs["context"].try_extract_tensor::<f32>()?;
        let expected = [1i64, dim(ids.len()), dim(self.hidden)];
        if *shape != Shape::new(expected) {
            return Err(NeuralError::Shape {
                tensor: "context",
                expected: expected.to_vec(),
                found: shape.iter().copied().collect(),
            });
        }
        Ok(hidden.to_vec())
    }
}

/// The per-position candidate scores out of the flat `[rows, width, E]`
/// log-prob table — the slot after `position` is the MASK cell the graph
/// scores, so the offset is `(row * width + position + 1) * E`.
fn asked_scores(
    asked: &[Vec<Vec<usize>>],
    log_probs: &[f32],
    width: usize,
    emissions_len: usize,
) -> Vec<Vec<Vec<f32>>> {
    let mut scores = Vec::with_capacity(asked.len());
    for (row, positions) in asked.iter().enumerate() {
        let mut positions_scores = Vec::with_capacity(positions.len());
        for (position, emissions) in positions.iter().enumerate() {
            let base = (row * width + position + 1) * emissions_len;
            positions_scores.push(
                emissions
                    .iter()
                    .map(|&emission| log_probs[base + emission])
                    .collect(),
            );
        }
        scores.push(positions_scores);
    }
    scores
}

/// Read *dir*'s `route-a.json` and refuse the manifest shapes this build
/// cannot serve: a `layout` other than `towers` most of all, which a graph
/// written for another wiring would carry.
fn read_manifest(dir: &Path) -> Result<Manifest, NeuralError> {
    let manifest_path = dir.join("route-a.json");
    let raw = fs::read_to_string(&manifest_path).map_err(|source| NeuralError::Io {
        path: manifest_path.clone(),
        source,
    })?;
    let manifest: Manifest =
        serde_json::from_str(&raw).map_err(|source| NeuralError::Manifest {
            path: manifest_path,
            source,
        })?;
    if manifest.layout != LAYOUT {
        return Err(NeuralError::Layout {
            path: dir.to_path_buf(),
            layout: manifest.layout,
        });
    }
    Ok(manifest)
}

/// The per-thread sessions of one graph, over one shared copy of its weights.
///
/// *initializers* are the `OrtValue`s the export's weights table described,
/// each pointing into the model's one mapping of the weights file.
/// Registering them with `with_initializer` enrolls each for the shared
/// [`PrepackedWeights`] container too, so the matrices MLAS pre-packs are
/// written once and reused instead of packed per session. `by_thread` is
/// declared first so its sessions drop before the weights they point into.
/// The tensors `emission` hands the fill graph and the resolved candidates
/// they answer: `asked[path][position]` is the emission indices of that
/// position's candidates, in the order the score file lists them.
#[derive(Debug)]
struct FillInputs {
    /// The widest reading's width, sentinels included.
    width: usize,
    /// `[CLS, MASK x n, SEP]` per reading, padded.
    input_ids: Vec<i64>,
    /// 1 over each reading's real tokens, 0 over padding.
    attention_mask: Vec<i64>,
    /// The span vocabulary id at each span position, 0 elsewhere.
    span_ids: Vec<i64>,
    /// The keys pressed per span position, `a`-`z` as 0-25 and 26 at padding,
    /// the collator's encoding.
    span_letters: Vec<i64>,
    /// True at each span position, false at the sentinels and padding.
    span_positions: Vec<bool>,
    /// `asked[path][position]` holds the emission indices to read back.
    asked: Vec<Vec<Vec<usize>>>,
}

#[derive(Debug)]
struct GraphSessions {
    /// One session per thread, opened on first use.
    by_thread: ThreadLocal<RefCell<Session>>,
    /// The graph's file.
    path: PathBuf,
    /// The prepacked weights every session of this graph shares; `None` for
    /// a quantized export, whose prepacks do not join a shared container.
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
        shareable: bool,
    ) -> Self {
        Self {
            by_thread: ThreadLocal::new(),
            path: graph,
            prepacked: shareable.then(PrepackedWeights::new),
            initializers,
            shape,
        }
    }

    /// This thread's session, opened from the shared file on first use.
    ///
    /// Sessions may open concurrently: ONNX Runtime serializes the pre-packed
    /// weights lookups and writes itself -- `PrepackConstantInitializedTensors`
    /// holds `prepacked_weights_container_->mutex_` around them.
    fn session(&self) -> Result<&RefCell<Session>, NeuralError> {
        self.by_thread.get_or_try(|| {
            Ok(RefCell::new(open_session(
                &self.path,
                self.prepacked.as_ref(),
                &self.initializers,
                self.shape,
            )?))
        })
    }

    /// Open a session now and refuse the graph when its inputs are not the
    /// ones *expected* -- a graph from another export would load and fail at
    /// the first run otherwise.
    fn check(&self, expected: &[&str]) -> Result<(), NeuralError> {
        let session = self.session()?.borrow();
        let declared: Vec<&str> = session.inputs().iter().map(Outlet::name).collect();
        let missing: Vec<&&str> = expected
            .iter()
            .filter(|name| !declared.contains(&**name))
            .collect();
        if !missing.is_empty() {
            return Err(NeuralError::Inputs {
                graph: self.path.clone(),
                missing: missing.iter().map(|name| (*name).to_string()).collect(),
            });
        }
        Ok(())
    }
}

/// What can go wrong opening or running a model.
#[derive(Debug, Error)]
pub enum NeuralError {
    /// A file could not be read.
    #[error("could not read {path}")]
    Io {
        /// The file that failed.
        path: PathBuf,
        /// The underlying error.
        #[source]
        source: std::io::Error,
    },
    /// The manifest is not the JSON `mlime export route-a` writes.
    #[error("the manifest at {path} is malformed")]
    Manifest {
        /// The file that failed.
        path: PathBuf,
        /// The underlying error.
        #[source]
        source: serde_json::Error,
    },
    /// The manifest parses but holds a value this build cannot serve.
    #[error("the export at {path} is malformed: {reason}")]
    ManifestShape {
        /// The export's directory.
        path: PathBuf,
        /// How it disagrees.
        reason: String,
    },
    /// The manifest's `layout` is not the one this build reads; a graph
    /// written for another wiring would load and run until a missing input
    /// failed it, so the check happens at open.
    #[error(
        "the export at {path} has a layout of {layout:?}, not \"towers\": re-export it with `mlime export route-a`"
    )]
    Layout {
        /// The export's directory.
        path: PathBuf,
        /// What the manifest's `layout` held.
        layout: String,
    },
    /// ONNX Runtime refused a graph or a run.
    #[error("onnx runtime failed")]
    Onnx(#[from] ort::Error),
    /// The export's `tokenizer.json` could not be read or could not encode a
    /// context.
    #[error("the tokenizer at {path} failed")]
    Tokenizer {
        /// The tokenizer file, or its name when the failure is at encode time.
        path: PathBuf,
        /// The tokenizer's error.
        #[source]
        source: tokenizers::Error,
    },
    /// A graph does not declare an input this build feeds.
    #[error("the graph at {graph} does not take inputs {missing:?}")]
    Inputs {
        /// The graph's file.
        graph: PathBuf,
        /// The input names it lacks.
        missing: Vec<String>,
    },
    /// A graph produced a tensor shaped other than the session's inputs
    /// promise.
    #[error("{tensor} came back shaped {found:?}; expected {expected:?}")]
    Shape {
        /// The output's name.
        tensor: &'static str,
        /// The shape the build expected.
        expected: Vec<i64>,
        /// What the graph produced.
        found: Vec<i64>,
    },
    /// A lattice record's spans and candidates do not parallel each other.
    #[error("the lattice of record {record} is malformed: {reason}")]
    Malformed {
        /// The lattice record's index.
        record: usize,
        /// How it disagrees.
        reason: String,
    },
    /// The lattice names a typed span the model was not built to read.
    #[error("record {record} asks about span {span:?}, which is not in the model's span table")]
    Span {
        /// The lattice record's index.
        record: usize,
        /// The span without an id.
        span: String,
    },
    /// The lattice asks about a character the model has no output row for.
    #[error(
        "record {record} asks about {character:?}, which is not in the model's emission vocabulary"
    )]
    Vocabulary {
        /// The lattice record's index.
        record: usize,
        /// The character without a row.
        character: char,
    },
    /// The candidate mask refuses a candidate the lattice asked about -- the
    /// same refusal `mlime train emit` raises, since a score file could never
    /// have been written over such a lattice.
    #[error(
        "record {record} asks {character:?} of span {span:?}, which the candidate mask refuses"
    )]
    Unadmitted {
        /// The lattice record's index.
        record: usize,
        /// The span whose mask row refused.
        span: String,
        /// The candidate refused.
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
    /// The backend's provider was compiled in but could not initialise.
    #[error("the {backend:?} backend's provider could not initialise")]
    Provider {
        /// The backend that was asked for.
        backend: Backend,
        /// The registration error.
        #[source]
        source: ort::Error,
    },
    /// The computed table and the lattice disagree about its shape; only a
    /// `LatticeRecord` built by another route than the decoder's can produce
    /// it.
    #[error("the emissions do not fit the lattice")]
    Scores(#[from] EmissionError),
}

/// The only towers wiring this build reads; the manifest's `layout` must
/// hold it exactly.
const LAYOUT: &str = "towers";

/// The character and span tables as name -> id maps, with the reserved
/// ``<unk>`` span row's id.
type Vocabularies = (HashMap<char, usize>, HashMap<String, usize>, usize);

/// The manifest's character and span tables as name -> id maps, with the
/// reserved ``<unk>`` span row's id resolved.
fn vocabularies(manifest: &Manifest, dir: &Path) -> Result<Vocabularies, NeuralError> {
    let mut emissions = HashMap::with_capacity(manifest.characters.len());
    for (index, character) in manifest.characters.iter().enumerate() {
        let mut chars = character.chars();
        let (Some(character), None) = (chars.next(), chars.next()) else {
            return Err(NeuralError::ManifestShape {
                path: dir.to_path_buf(),
                reason: format!("{character:?} is not one character"),
            });
        };
        emissions.insert(character, index);
    }
    let spans: HashMap<String, usize> = manifest
        .spans
        .iter()
        .enumerate()
        .map(|(id, span)| (span.clone(), id))
        .collect();
    let unknown_span = *spans
        .get("<unk>")
        .ok_or_else(|| NeuralError::ManifestShape {
            path: dir.to_path_buf(),
            reason: "the span table has no reserved <unk> row".to_owned(),
        })?;
    Ok((emissions, spans, unknown_span))
}

/// A span's `span_letters` row: `a`-`z` as 0-25 and `LETTER_PAD` past its
/// end -- the collator's encoding.
///
/// # Errors
///
/// If the span is longer than `MAX_SPAN_LETTERS` or holds a byte outside
/// `a`-`z`.
fn letter_ids(record: usize, span: &str) -> Result<[i64; MAX_SPAN_LETTERS], NeuralError> {
    if span.len() > MAX_SPAN_LETTERS {
        return Err(NeuralError::Malformed {
            record,
            reason: format!(
                "span {span:?} is {} letters; the letter encoding reads at most {MAX_SPAN_LETTERS}",
                span.len()
            ),
        });
    }
    let mut letters = [LETTER_PAD; MAX_SPAN_LETTERS];
    for (slot, byte) in letters.iter_mut().zip(span.bytes()) {
        if !byte.is_ascii_lowercase() {
            return Err(NeuralError::Malformed {
                record,
                reason: format!("span {span:?} holds byte {byte:#04x}, which is not a letter"),
            });
        }
        *slot = i64::from(byte - b'a');
    }
    Ok(letters)
}

/// The context graph's input names.
const CONTEXT_INPUTS: &[&str] = &["context_ids", "context_mask"];

/// The fill graph's input names.
const FILL_INPUTS: &[&str] = &[
    "input_ids",
    "attention_mask",
    "span_ids",
    "span_letters",
    "span_positions",
    "context",
    "context_mask",
    "has_context",
];

/// The letter id of a padding slot in `span_letters`: `a`..`z` are 0..25, so
/// 26 marks a column no letter occupies -- the collator's encoding.
const LETTER_PAD: i64 = 26;

/// The most letters a span's `span_letters` encoding reads; the collator's
/// `MAX_SPAN_LETTERS`.
const MAX_SPAN_LETTERS: usize = 12;

/// `route-a.json`, the fields the run consults; the manifest also records the
/// training step, the base model and the element types, which the tensor
/// tables and the graphs themselves already carry.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    /// The training step the checkpoint was at; provenance only.
    step: u64,
    /// The base model the towers were trained from; provenance only.
    base_model: String,
    /// The towers' wiring; absent or wrong on exports meant for another
    /// layout, which [`RouteA::open`] refuses.
    layout: String,
    /// The weights' element type before quantization; provenance only.
    dtype: String,
    /// The towers' hidden size, which shapes the broadcast context tensor.
    hidden: usize,
    /// The widest context the encoder reads, sentinels included.
    context_tokens: usize,
    /// The sentinel token ids the fill inputs are built from.
    specials: Specials,
    /// The typed spans in `span_ids` order.
    spans: Vec<String>,
    /// The emission alphabet in `log_probs` axis order.
    characters: Vec<String>,
    /// What the export quantized to, if anything.
    quantize: Option<String>,
    /// The two graphs' files.
    graphs: Graphs,
    /// The name of the `candidate_mask` tensor in the fill weights table.
    candidate_mask: String,
    /// Each graph's weights table.
    weights: GraphWeights,
}

/// The sentinel ids, as the manifest names them.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct Specials {
    /// `<pad>`.
    pad: u32,
    /// `<cls>`, the first token of every fill row.
    cls: u32,
    /// `<sep>`, the last.
    sep: u32,
    /// `<mask>`, the token a span position reads.
    mask: u32,
}

/// The manifest's `graphs`: the two towers' ONNX files.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Graphs {
    /// The context encoder's.
    context: String,
    /// The fill decoder's.
    fill: String,
}

/// The manifest's `weights`: one file per tower.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphWeights {
    /// The context graph's table.
    context: WeightsFile,
    /// The fill graph's table.
    fill: WeightsFile,
}

/// One weights file's table: the file's name and where every shared
/// initializer sits in it, as the export wrote the `external_data` entries.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WeightsFile {
    /// The file's name inside the export's directory.
    file: String,
    /// Every externalized tensor of the graph the file backs.
    tensors: Vec<WeightTensor>,
}

/// One shared initializer: its name in the graph, its shape and dtype, and
/// its byte extent inside the weights file.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WeightTensor {
    /// The initializer's name.
    name: String,
    /// The element type; the export writes `float32`, `int8` and `bool`, and
    /// anything else fails the manifest's parse.
    dtype: WeightDtype,
    /// The tensor's shape.
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
    /// Signed 32-bit integer, little-endian.
    #[serde(rename = "int32")]
    Int32,
    /// Signed 64-bit integer, little-endian.
    #[serde(rename = "int64")]
    Int64,
    /// ONNX's boolean, one byte per element: `candidate_mask`'s dtype.
    #[serde(rename = "bool")]
    Bool,
}

/// Map *dir*`/`*file* once, or hand back the mapping already made.
fn weights_map<'m>(
    maps: &'m mut HashMap<String, Mmap>,
    dir: &Path,
    file: &str,
) -> Result<&'m Mmap, NeuralError> {
    if !maps.contains_key(file) {
        let path = dir.join(file);
        let opened = fs::File::open(&path).map_err(|source| NeuralError::Io {
            path: path.clone(),
            source,
        })?;
        // Safety: the file is the export's product, opened read-only and only
        // ever read through the map while a `RouteA` is alive.
        let map = unsafe { Mmap::map(&opened) }.map_err(|source| NeuralError::Io {
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
/// The values' data pointers address *map*; the caller (`RouteA`) holds the
/// mappings in a field declared after its sessions, so the map outlives every
/// value and session built here.
fn shared_values(
    map: &Mmap,
    table: &WeightsFile,
) -> Result<Vec<(String, Arc<DynValue>)>, NeuralError> {
    let mismatched = |reason: String| NeuralError::Weights {
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
            WeightDtype::Bool => {
                shared_value::<bool>(map, &info, tensor, &mismatched, &mut values)?;
            }
            WeightDtype::Int32 | WeightDtype::Int64 => {}
        }
    }
    Ok(values)
}

/// The shared `OrtValue` of one tensor of *table*: bounds- and shape-checked
/// against *map*, then handed to every session of the graph as a
/// `TensorRefMut` whose data pointer addresses the mapping.
fn shared_value<T: PrimitiveTensorElementType + std::fmt::Debug>(
    map: &Mmap,
    info: &MemoryInfo,
    tensor: &WeightTensor,
    mismatched: &impl Fn(String) -> NeuralError,
    values: &mut Vec<(String, Arc<DynValue>)>,
) -> Result<(), NeuralError> {
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
    let count = tensor.shape.iter().try_fold(1usize, |count, &axis| {
        count.checked_mul(usize::try_from(axis).ok()?)
    });
    if count.is_none_or(|count| count * size_of::<T>() != length) {
        return Err(mismatched(format!(
            "the extent of {} does not match its shape {:?}",
            tensor.name, tensor.shape
        )));
    }
    // Safety: the extent is bounds- and shape-checked above, and `map` is
    // held by `RouteA` in a field that drops after every session and value
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

/// The `candidate_mask` bytes out of the fill weights file: *name* must be
/// the one boolean tensor shaped `[spans, emissions]`.
fn read_mask(
    map: &[u8],
    table: &WeightsFile,
    name: &str,
    spans: usize,
    emissions: usize,
    dir: &Path,
) -> Result<Vec<bool>, NeuralError> {
    let mismatched = |reason: String| NeuralError::Weights {
        path: dir.join(&table.file),
        reason,
    };
    let tensor = table
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| mismatched(format!("no tensor named {name}")))?;
    if tensor.dtype != WeightDtype::Bool {
        return Err(mismatched(format!("{name} is not a bool tensor")));
    }
    if tensor.shape != [dim(spans), dim(emissions)] {
        return Err(mismatched(format!(
            "{name} is shaped {:?}, not [{spans}, {emissions}]",
            tensor.shape
        )));
    }
    let start = usize::try_from(tensor.offset)
        .map_err(|_| mismatched(format!("the offset of {name} overflows usize")))?;
    let length = usize::try_from(tensor.length)
        .map_err(|_| mismatched(format!("the length of {name} overflows usize")))?;
    let bytes = map.get(start..start + length).ok_or_else(|| {
        mismatched(format!(
            "the extent of {name} runs past {} bytes",
            map.len()
        ))
    })?;
    Ok(bytes.iter().map(|&byte| byte != 0).collect())
}

/// The session options every session of a graph shares, sized by *shape*:
/// *shape*'s intra-op threads, no parallel execution, no arena, no
/// memory-pattern reservation. The decoder is the parallel layer, so a
/// rayon-path session takes one intra-op thread -- a session that also
/// spread its matrix products over threads or held a private buffer arena
/// would oversubscribe the machine. Memory patterns pool a session's
/// activations into one reservation, which measured both slower and ~150 MB
/// heavier on `ime-lm` than letting each buffer come and go. `Level2` keeps
/// every fusion the fill graph benefits from.
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
) -> Result<Session, NeuralError> {
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
            .map_err(|source| NeuralError::Provider {
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

/// A tensor dimension as ONNX Runtime spells it.
#[expect(
    clippy::cast_possible_wrap,
    reason = "a dimension is a path count, a hidden size or a sequence length, none of which approaches i64::MAX"
)]
const fn dim(n: usize) -> i64 {
    n as i64
}
