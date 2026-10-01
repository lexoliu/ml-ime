//! The bridge between the neural model and the decoder.
//!
//! Two commands, and between them two files. `emit-lattice` writes what the
//! model is being asked: for every record of an evaluation set, every reading of
//! its keystrokes, and for every position of every reading, the letters typed
//! there and the characters that position admits. The Python side answers with a
//! log probability per candidate, in the same order. `fused-eval` reads both
//! back, decodes with the emissions fused into the same beam Viterbi the
//! baseline uses, and reports what came out. Asked to with `--dump`, it also
//! writes the beam itself: one JSON Lines file per reported section, one
//! record's ranked hypotheses and their scores per line.
//!
//! Both commands segment through [`engine::read`] with the same
//! [`SegmentOptions`], because the score file is positional: it identifies a
//! candidate by where it sat in the lattice and by nothing else. Two different
//! search settings produce two different lattices, and the shape checks in
//! [`Scored::attach`] are what turn that from a wrong answer into a refusal.
//!
//! The ablation is not a separate program. An emission weight of zero is the
//! n-gram baseline, a weight above zero with the trigram is the fused system,
//! and the same weight with [`NoTransition`] is the emissions alone -- one
//! decode path, three configurations, so a difference between the numbers is a
//! difference between the models.

use crate::engine::read;
use anyhow::{Context as _, Result, anyhow, bail};
use askama::Template;
use blake2::{Blake2b, Digest as _, digest::consts::U8};
use clap::{Args, ValueEnum};
use flate2::read::MultiGzDecoder;
use ime_decode::{
    BeamOptions, Both, Candidates, Emission, Emittable, Hypothesis, LatticeRecord, NoTransition,
    Record, ScoreRecord, Scored, Transition, Uniform, Weighted, decode_many, lattice_paths,
};
use ime_eval::{EvalRecord, EvalSet, Observation, Report, Slice};
use ime_lm::CharLm;
use ime_neural::RouteA;
use ime_ngram::NgramModel;
use ime_pinyin::{CorrectionTable, Lexicon, SegmentOptions, Segmentation, SyllableTable};
use rayon::iter::{
    IndexedParallelIterator as _, IntoParallelRefIterator as _, ParallelIterator as _,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufRead as _, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::mpsc::{self, Sender};
use std::thread::{self, JoinHandle};
use tracing::info;

/// Which part of the evaluation set a command runs over.
///
/// Mirrors [`Slice`] rather than deriving `ValueEnum` on it, because the
/// evaluation crate has no business depending on an argument parser.
#[derive(Copy, Clone, PartialEq, Eq, Debug, ValueEnum)]
pub enum SliceArg {
    /// Every record.
    All,
    /// The records the fusion weight is tuned on.
    Dev,
    /// The records tuning never saw.
    Test,
}

impl SliceArg {
    /// The evaluation crate's own name for this slice.
    pub(crate) const fn slice(self) -> Slice {
        match self {
            Self::All => Slice::All,
            Self::Dev => Slice::Dev,
            Self::Test => Slice::Test,
        }
    }

    /// How the slice is spelled in the report.
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Dev => "dev",
            Self::Test => "test",
        }
    }
}

/// How the evaluation set is cut into a tuning half and a reporting half.
#[derive(Debug, Clone, Args)]
pub struct SliceArgs {
    /// Which records to score.
    #[arg(long, value_enum, default_value = "all")]
    pub slice: SliceArg,
    /// What share of the set the dev slice holds. Membership is decided by a
    /// hash of the record itself, so it survives the file being reordered.
    #[arg(long, default_value = "0.0905")]
    pub dev_share: f64,
    /// Sweep every `--weight` on the dev slice, then report the test slice at
    /// the weight whose sentence top-1 was highest; ties go to the weight given
    /// first. Replaces `--slice`, and meaningless without emissions to fuse.
    #[arg(long, conflicts_with = "slice")]
    pub select_on_dev: bool,
}

/// A `--backend` answer, translated to the crate's own [`Backend`].
///
/// Mirrors [`ime_lm::Backend`] rather than deriving `ValueEnum` on it, because
/// the model crate has no business depending on an argument parser.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default, ValueEnum)]
pub enum BackendArg {
    /// ONNX Runtime's CPU kernels.
    #[default]
    Cpu,
    /// Apple's Core ML provider, if the build carries `gpu-coreml`.
    Coreml,
    /// WebGPU through Dawn (the Metal GPU), if the build carries `gpu-webgpu`.
    Webgpu,
    /// NVIDIA's CUDA provider, if the build carries `gpu-cuda`.
    Cuda,
    /// Hand-encoded Metal kernels for the LM step — macOS only. The towers
    /// stay on their own backend; only the character LM switches.
    #[cfg(target_os = "macos")]
    Metal,
}

impl BackendArg {
    /// The model crate's name for this backend.
    pub const fn backend(self) -> ime_lm::Backend {
        match self {
            Self::Cpu => ime_lm::Backend::Cpu,
            Self::Coreml => ime_lm::Backend::CoreMl,
            Self::Webgpu => ime_lm::Backend::WebGpu,
            Self::Cuda => ime_lm::Backend::Cuda,
            #[cfg(target_os = "macos")]
            Self::Metal => ime_lm::Backend::Metal,
        }
    }

    /// The towers crate's name for this backend — the towers run on ONNX
    /// Runtime everywhere, so `--backend metal` leaves them on the CPU.
    pub const fn neural_backend(self) -> ime_neural::Backend {
        match self {
            Self::Cpu => ime_neural::Backend::Cpu,
            Self::Coreml => ime_neural::Backend::CoreMl,
            Self::Webgpu => ime_neural::Backend::WebGpu,
            Self::Cuda => ime_neural::Backend::Cuda,
            #[cfg(target_os = "macos")]
            Self::Metal => ime_neural::Backend::Cpu,
        }
    }
}

/// A `--metal-weights` answer, translated to [`ime_lm::MetalWeights`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default, ValueEnum)]
pub enum MetalWeightsArg {
    /// The export's own dtype.
    #[default]
    Auto,
    /// fp16 packed weights — dequantized or cast at load.
    F16,
}

impl MetalWeightsArg {
    /// The model crate's name for this precision.
    pub const fn weights(self) -> ime_lm::MetalWeights {
        match self {
            Self::Auto => ime_lm::MetalWeights::Auto,
            Self::F16 => ime_lm::MetalWeights::F16,
        }
    }
}

/// Supplies the emission model for one record's lattice.
///
/// A generic rather than a trait object: the beam calls
/// [`Emission::score`] once per candidate per beam state, which is the hottest
/// line in the program, and the three configurations differ only in what that
/// call compiles to.
trait Emissions {
    /// The emission model, borrowing the record's candidate sets.
    type Model<'a>: Emission
    where
        Self: 'a;

    /// The model for *record*, over *segmentations* and *candidates*.
    ///
    /// # Errors
    ///
    /// If nothing scored this record, or the scores do not describe this lattice.
    fn model<'a>(
        &'a self,
        record: usize,
        eval: &EvalRecord,
        segmentations: &[Segmentation],
        candidates: &'a Candidates,
    ) -> Result<Self::Model<'a>>;
}

/// No emissions at all: the n-gram baseline.
struct NoEmissions;

impl Emissions for NoEmissions {
    type Model<'a> = Uniform;

    fn model<'a>(
        &'a self,
        _record: usize,
        _eval: &EvalRecord,
        _segmentations: &[Segmentation],
        _candidates: &'a Candidates,
    ) -> Result<Uniform> {
        Ok(Uniform)
    }
}

/// The model's log probabilities, computed live by the exported towers.
///
/// A record's table depends only on the record, so it is computed once and
/// shared through `cache` across every weight the sweep decodes at and the
/// dev/test passes alike -- the towers run once per record per run, not once
/// per section.
struct LiveEmissions<'a> {
    /// The towers the export directory loaded into.
    towers: &'a RouteA,
    /// The typo noise model the run decodes under, when it does.
    corrections: Option<&'a CorrectionTable>,
    /// The emittable set the lattice's candidates are restricted to.
    emittable: &'a Emittable,
    /// The lexicon `restrict`'s ids name characters through.
    lexicon: &'a Lexicon,
    /// Every table this run has already computed, by record index.
    cache: &'a Mutex<HashMap<usize, Vec<Vec<Vec<f32>>>>>,
    weight: f32,
    floor: f32,
    /// Whether the context tower feeds the gate: off runs every record with
    /// the gate zeroed, the `context off` score file's counterpart.
    with_context: bool,
}

impl Emissions for LiveEmissions<'_> {
    type Model<'a>
        = Weighted<Scored<'a>>
    where
        Self: 'a;

    fn model<'a>(
        &'a self,
        record: usize,
        eval: &EvalRecord,
        segmentations: &[Segmentation],
        candidates: &'a Candidates,
    ) -> Result<Weighted<Scored<'a>>> {
        let hit = self
            .cache
            .lock()
            .expect("the emissions cache is not poisoned")
            .get(&record)
            .cloned();
        let scores = if let Some(scores) = hit {
            scores
        } else {
            let lattice = LatticeRecord {
                record,
                pinyin: eval.pinyin.clone(),
                context: eval.context.clone(),
                paths: lattice_paths(
                    &eval.pinyin,
                    segmentations,
                    candidates,
                    self.emittable,
                    self.lexicon,
                ),
            };
            let computed = self
                .towers
                .emission(&lattice, self.corrections, self.with_context)
                .with_context(|| format!("could not score record {record} with the towers"))?;
            self.cache
                .lock()
                .expect("the emissions cache is not poisoned")
                .entry(record)
                .or_insert(computed)
                .clone()
        };
        Ok(Weighted {
            inner: Scored::attach(record, candidates, self.emittable, scores, self.floor)
                .with_context(|| format!("the towers' answer does not fit record {record}"))?,
            weight: self.weight,
        })
    }
}

/// Where the emissions come from: the score file, or the towers it was
/// written from. Both answer the same table, so the decoder sees one type.
enum Sources<'a> {
    /// `NeuralEmissions`: the score file read once up front.
    File(NeuralEmissions<'a>),
    /// `LiveEmissions`: the towers, run per record.
    Live(LiveEmissions<'a>),
}

impl Emissions for Sources<'_> {
    type Model<'a>
        = Weighted<Scored<'a>>
    where
        Self: 'a;

    fn model<'a>(
        &'a self,
        record: usize,
        eval: &EvalRecord,
        segmentations: &[Segmentation],
        candidates: &'a Candidates,
    ) -> Result<Weighted<Scored<'a>>> {
        match self {
            Self::File(emissions) => emissions.model(record, eval, segmentations, candidates),
            Self::Live(emissions) => emissions.model(record, eval, segmentations, candidates),
        }
    }
}

/// The model's log probabilities, read out of a score file.
struct NeuralEmissions<'a> {
    scores: &'a HashMap<usize, Vec<Vec<Vec<f32>>>>,
    emittable: &'a Emittable,
    weight: f32,
    floor: f32,
}

impl Emissions for NeuralEmissions<'_> {
    type Model<'a>
        = Weighted<Scored<'a>>
    where
        Self: 'a;

    fn model<'a>(
        &'a self,
        record: usize,
        _eval: &EvalRecord,
        _segmentations: &[Segmentation],
        candidates: &'a Candidates,
    ) -> Result<Weighted<Scored<'a>>> {
        let scores = self
            .scores
            .get(&record)
            .with_context(|| format!("the score file has no record {record}"))?;
        Ok(Weighted {
            inner: Scored::attach(
                record,
                candidates,
                self.emittable,
                scores.clone(),
                self.floor,
            )
            .context("the score file does not describe this lattice")?,
            weight: self.weight,
        })
    }
}

/// One configuration's line in the report.
struct Section {
    emission: &'static str,
    weight: f32,
    /// The weight this section's candidates paid their correction prior at.
    typo_weight: f32,
    /// Whether the run corrected keystrokes; the report shows the prior's
    /// weight only where it can act.
    corrections: bool,
    transition: &'static str,
    slice: &'static str,
    /// Whether the weight won a dev sweep rather than being given on the
    /// command line.
    selected: bool,
    report: Report,
    /// The beam of every record in the slice, sorted by record. Filled only
    /// when a dump directory was given; the report ignores it.
    rows: Vec<DumpRow>,
}

/// One line of a dump file: the record's expected text and the beam it
/// decoded into, best first.
#[derive(Serialize)]
struct DumpRow {
    record: usize,
    text: String,
    hypotheses: Vec<DumpedHypothesis>,
}

/// One hypothesis as the dump records it. Deserializable because the
/// progress file stores the same shape and a resumed run builds its dump out
/// of it again.
#[derive(Debug, Serialize, Deserialize)]
struct DumpedHypothesis {
    text: String,
    score: f32,
}

/// Every configuration that was run, as it goes to stdout.
#[derive(Template)]
#[template(path = "fused_eval.txt", ext = "txt")]
struct Ablation {
    sections: Vec<Section>,
}

/// The tables and search settings both commands read the lattice with.
struct Reader {
    table: SyllableTable,
    lexicon: Lexicon,
    segment: SegmentOptions,
    /// The typo noise model, when the run corrects keystrokes: absent, the
    /// lattice and the candidate masks are exactly the clean path's.
    corrections: Option<CorrectionTable>,
}

impl Reader {
    /// Resolve one record's keystrokes, or `None` when they admit no reading.
    ///
    /// A record whose keystrokes cannot be segmented at all is not an error:
    /// corrupted input is expected to leave the lattice sometimes, and a record
    /// the engine could not even read is the extreme case of one it got wrong,
    /// so callers score it as unanswered. Every other failure still raises.
    fn read(&self, record: &EvalRecord) -> Result<Option<(Vec<Segmentation>, Candidates)>> {
        match read(
            &record.pinyin,
            &self.table,
            &self.segment,
            self.corrections.as_ref(),
            &self.lexicon,
        ) {
            Ok(resolved) => Ok(Some(resolved)),
            Err(crate::engine::BaselineError::Segment {
                source: ime_pinyin::SegmentError::NoSegmentation { .. },
                ..
            }) => Ok(None),
            Err(error) => Err(error).with_context(|| format!("could not read {:?}", record.pinyin)),
        }
    }
}

/// Write the lattice an evaluation set decodes into, for the model to score.
///
/// # Errors
///
/// If the evaluation set cannot be read, a record cannot be segmented, or the
/// output cannot be written.
pub fn emit_lattice(
    eval_set: &Path,
    out: &Path,
    emittable: &Path,
    table: SyllableTable,
    lexicon: Lexicon,
    segment: SegmentOptions,
    corrections: Option<CorrectionTable>,
) -> Result<()> {
    let set = load_set(eval_set)?;
    let emittable = load_emittable(emittable, &lexicon)?;
    let reader = Reader {
        table,
        lexicon,
        segment,
        corrections,
    };
    let file =
        fs::File::create(out).with_context(|| format!("could not create {}", out.display()))?;
    let mut sink = BufWriter::new(file);
    let mut positions = 0usize;
    let mut slots = 0usize;
    for (index, record) in set.records().iter().enumerate() {
        let (segmentations, candidates) = reader
            .read(record)?
            .with_context(|| format!("{:?} admits no reading", record.pinyin))?;
        let paths = lattice_paths(
            &record.pinyin,
            &segmentations,
            &candidates,
            &emittable,
            &reader.lexicon,
        );
        positions += paths.iter().map(|path| path.spans.len()).sum::<usize>();
        slots += paths
            .iter()
            .flat_map(|path| &path.candidates)
            .map(|set| set.chars().count())
            .sum::<usize>();
        let line = LatticeRecord {
            record: index,
            pinyin: record.pinyin.clone(),
            context: record.context.clone(),
            paths,
        };
        serde_json::to_writer(&mut sink, &line).context("could not serialise a lattice record")?;
        sink.write_all(b"\n")
            .context("could not write a lattice record")?;
    }
    sink.flush().context("could not flush the lattice")?;
    info!(
        records = set.len(),
        positions,
        candidates = slots,
        path = %out.display(),
        "wrote the lattice"
    );
    Ok(())
}

/// Which transition models a run decodes with.
///
/// The n-gram, the character language model, both at once, or neither -- the
/// last being the emissions alone, the ablation that says what the fill tower
/// earned on its own.
#[derive(Copy, Clone)]
pub enum Models<'a> {
    /// No transition at all.
    None,
    /// The Kneser-Ney trigram.
    Ngram(&'a NgramModel),
    /// The recurrent character model.
    CharLm(&'a CharLm),
    /// The trigram at weight one and the character model at `lm_weight`.
    Both {
        /// The trigram.
        ngram: &'a NgramModel,
        /// The character model.
        lm: &'a CharLm,
        /// Weight on the character model's scores.
        lm_weight: f32,
    },
}

/// Resumable per-record progress for `fused-eval` (`--progress`).
///
/// Set, every `measure` call of the run keeps a JSON Lines file in *dir* named
/// for the part of the configuration that says which decode it is -- the
/// inputs, the slice, the emission weight, the transition. The file's first
/// line is the whole configuration key and every later line is one record's
/// outcome, so a stopped run is continued by re-running the same command with
/// the same directory, and resuming under arguments the file was not written
/// with is refused rather than restarted.
pub struct Progress<'a> {
    /// The directory the per-configuration files are kept in.
    pub dir: &'a Path,
    /// The path the n-gram model was loaded from, when it was.
    pub model: Option<&'a Path>,
    /// The directory the character model was loaded from, when it was.
    pub lm: Option<&'a Path>,
    /// Stop every section after this many newly written records -- the test
    /// stand-in for a kill, since the command line never sets it.
    #[doc(hidden)]
    pub stop_after: Option<usize>,
}

/// An emission source before a weight is applied to it: the data a
/// `Sources` variant is built from inside `measure_at`.
enum SourceInput<'a> {
    /// A loaded score file.
    File(&'a HashMap<usize, Vec<Vec<Vec<f32>>>>),
    /// The opened towers.
    Live(&'a RouteA),
}

/// Everything a run needs to score one transition model over its sections.
struct Run<'a> {
    set: &'a EvalSet,
    /// Where the set was read from; part of the progress key.
    eval_set: &'a Path,
    reader: &'a Reader,
    scores: Option<&'a HashMap<usize, Vec<Vec<Vec<f32>>>>>,
    /// Where the score file was read from; part of the progress key.
    scores_path: Option<&'a Path>,
    /// The towers a live run emits through; exclusive with *scores*.
    route_a: Option<&'a RouteA>,
    /// Where the export directory lives; part of the progress key.
    route_a_path: Option<&'a Path>,
    /// Whether live emissions read the record's context. Score files come as
    /// written; the flag does not reach them.
    with_context: bool,
    emittable: &'a Emittable,
    /// Where the emittable set was read from; part of the progress key.
    emittable_path: &'a Path,
    floor: f32,
    weights: &'a [f32],
    /// The correction-prior weights the sweep tries; each is one entry in the
    /// `weight x typo-weight` product a select-on-dev run decodes.
    typo_weights: &'a [f32],
    slice: &'a SliceArgs,
    beam: &'a BeamOptions,
    /// How many records one `decode_many` call decodes in lockstep. One keeps
    /// the per-record rayon path; more runs the records' steps as one batch.
    batch: usize,
    dump: bool,
    progress: Option<&'a Progress<'a>>,
}

impl<'a> Run<'a> {
    /// The progress file this run's decode would keep, when one is asked for.
    fn request(
        &self,
        label: &'static str,
        lm_weight: f32,
        weight: f32,
        typo_weight: f32,
    ) -> Option<ProgressRequest<'_>> {
        self.progress.map(|progress| ProgressRequest {
            dir: progress.dir,
            stop_after: progress.stop_after,
            eval_set: self.eval_set,
            scores: self.scores_path,
            route_a: self.route_a_path,
            with_context: self.route_a.map(|_| self.with_context),
            emittable: self.emittable_path,
            model: progress.model,
            lm: progress.lm,
            lm_weight,
            floor: self.floor,
            weight,
            typo_weight,
            transition: label,
        })
    }

    /// What emits for this run's sections, and the label the report gives it:
    /// the score file, the towers live, or nothing at all.
    ///
    /// # Errors
    ///
    /// If both the score file and the towers were given: the argument parser
    /// refuses the pair, but a caller that is not the command line could still
    /// hand both in.
    fn source(&self) -> Result<Option<(SourceInput<'a>, &'static str)>> {
        match (self.scores, self.route_a) {
            (Some(_), Some(_)) => {
                bail!("--scores and --route-a are two answers to the same question")
            }
            (Some(scores), None) => Ok(Some((SourceInput::File(scores), "neural"))),
            (None, Some(towers)) => Ok(Some((
                SourceInput::Live(towers),
                if self.with_context {
                    "route-a"
                } else {
                    "route-a-no-context"
                },
            ))),
            (None, None) => Ok(None),
        }
    }

    /// The sections one transition model produces: without a score file the
    /// transition alone over the requested slice; with one, every weight over
    /// the slice, or the dev sweep and the test section at its winner.
    ///
    /// *lockstep* runs the slice serially through `decode_many` chunks even
    /// when `batch` is one -- a GPU-backed model cannot share its device with
    /// the rayon's per-record parallelism the CPU path uses.
    fn sections<T: Transition + Sync>(
        &self,
        transition: &T,
        label: &'static str,
        lm_weight: f32,
        lockstep: bool,
    ) -> Result<Vec<Section>> {
        let Some((input, emission_label)) = self.source()? else {
            return self.transition_only(transition, label, lm_weight, lockstep);
        };
        let emission_cache = Mutex::new(HashMap::new());
        let measure_at = |which: SliceArg, weight: f32, typo_weight: f32| -> Result<Section> {
            let beam = BeamOptions {
                typo_weight,
                ..(*self.beam).clone()
            };
            let emissions = match input {
                SourceInput::File(scores) => Sources::File(NeuralEmissions {
                    scores,
                    emittable: self.emittable,
                    weight,
                    floor: self.floor,
                }),
                SourceInput::Live(towers) => Sources::Live(LiveEmissions {
                    towers,
                    corrections: self.reader.corrections.as_ref(),
                    emittable: self.emittable,
                    lexicon: &self.reader.lexicon,
                    cache: &emission_cache,
                    weight,
                    floor: self.floor,
                    with_context: self.with_context,
                }),
            };
            let request = self.request(label, lm_weight, weight, typo_weight);
            let (report, rows) = measure(
                self.set,
                which,
                self.slice.dev_share,
                self.reader,
                &emissions,
                transition,
                &beam,
                self.batch,
                lockstep,
                self.dump,
                request.as_ref(),
            )?;
            Ok(Section {
                emission: emission_label,
                weight,
                typo_weight,
                corrections: self.reader.corrections.is_some(),
                transition: label,
                slice: which.label(),
                selected: false,
                report,
                rows,
            })
        };
        // Without the noise model every prior is zero and the sweep is one
        // entry wide, however many `--typo-weight`s the command line lists.
        let typo_weights = if self.reader.corrections.is_some() {
            self.typo_weights
        } else {
            &self.typo_weights[..self.typo_weights.len().min(1)]
        };
        let mut sections = Vec::new();
        if self.slice.select_on_dev {
            for &weight in self.weights {
                for &typo_weight in typo_weights {
                    sections.push(measure_at(SliceArg::Dev, weight, typo_weight)?);
                }
            }
            let mut winner = 0;
            for (index, section) in sections.iter().enumerate().skip(1) {
                if section.report.top1_hits() > sections[winner].report.top1_hits() {
                    winner = index;
                }
            }
            let mut test = measure_at(
                SliceArg::Test,
                sections[winner].weight,
                sections[winner].typo_weight,
            )?;
            test.selected = true;
            sections.push(test);
        } else {
            for &weight in self.weights {
                for &typo_weight in typo_weights {
                    sections.push(measure_at(self.slice.slice, weight, typo_weight)?);
                }
            }
        }
        Ok(sections)
    }

    /// The one section a run without emissions produces: the transition alone
    /// over the requested slice, at the sweep's first prior weight -- the
    /// `--weight` sweep has no emission term to scale here.
    fn transition_only<T: Transition + Sync>(
        &self,
        transition: &T,
        label: &'static str,
        lm_weight: f32,
        lockstep: bool,
    ) -> Result<Vec<Section>> {
        let typo_weight = self.typo_weights.first().copied().unwrap_or(1.0);
        let beam = BeamOptions {
            typo_weight,
            ..(*self.beam).clone()
        };
        let request = self.request(label, lm_weight, 0.0, typo_weight);
        let (report, rows) = measure(
            self.set,
            self.slice.slice,
            self.slice.dev_share,
            self.reader,
            &NoEmissions,
            transition,
            &beam,
            self.batch,
            lockstep,
            self.dump,
            request.as_ref(),
        )?;
        Ok(vec![Section {
            emission: "none",
            weight: 0.0,
            typo_weight,
            corrections: self.reader.corrections.is_some(),
            transition: label,
            slice: self.slice.slice.label(),
            selected: false,
            report,
            rows,
        }])
    }
}

/// Score an evaluation set with the n-gram, with the neural emissions, or with
/// both fused.
///
/// Without a score file the run is the n-gram baseline; without an n-gram it is
/// the emissions alone; with both, every weight in *weights* paired with every
/// correction-prior weight in *`typo_weights`* is one fused configuration. With
/// `--select-on-dev` the weights are swept on the dev
/// slice instead and the report closes with the test slice at whichever weight
/// scored the highest sentence top-1 there. With *dump* every evaluated
/// section's beam is written there too, one JSON Lines file per section.
///
/// *batch* is how many records one `decode_many` call runs in lockstep: one
/// keeps the per-record rayon path, more feeds `records x paths x beams` rows
/// to each model step, which is the shape a GPU backend is sized for.
///
/// # Errors
///
/// If any input cannot be read, a
/// record cannot be decoded, a score file does not describe the lattice the
/// records segment into, the dump cannot be written, or a progress file was
/// written under a different configuration.
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a distinct axis of the ablation the command exists to run"
)]
pub fn fused_eval(
    eval_set: &Path,
    scores_path: Option<&Path>,
    route_a: Option<(&Path, &RouteA, bool)>,
    emittable_path: &Path,
    floor: f32,
    weights: &[f32],
    typo_weights: &[f32],
    slice: &SliceArgs,
    dump: Option<&Path>,
    progress: Option<&Progress<'_>>,
    table: SyllableTable,
    lexicon: Lexicon,
    segment: SegmentOptions,
    corrections: Option<CorrectionTable>,
    beam: &BeamOptions,
    batch: usize,
    transition: Models<'_>,
) -> Result<String> {
    let set = load_set(eval_set)?;
    let emittable = load_emittable(emittable_path, &lexicon)?;
    let reader = Reader {
        table,
        lexicon,
        segment,
        corrections,
    };
    if let Some(dir) = dump {
        fs::create_dir_all(dir)
            .with_context(|| format!("could not create the dump directory {}", dir.display()))?;
    }
    let scores = scores_path.map(load_scores).transpose()?;
    let run = Run {
        set: &set,
        eval_set,
        reader: &reader,
        scores: scores.as_ref(),
        scores_path,
        route_a: route_a.map(|(_, towers, _)| towers),
        route_a_path: route_a.map(|(path, _, _)| path),
        with_context: route_a.is_none_or(|(_, _, with_context)| with_context),
        emittable: &emittable,
        emittable_path,
        floor,
        weights,
        typo_weights,
        slice,
        beam,
        batch,
        dump: dump.is_some(),
        progress,
    };
    let sections = match transition {
        Models::None => run.sections(&NoTransition, "none", 1.0, false)?,
        Models::Ngram(ngram) => run.sections(ngram, "kn-trigram", 1.0, false)?,
        Models::CharLm(lm) => {
            run.sections(lm, "char-lm", 1.0, lm.backend() != ime_lm::Backend::Cpu)?
        }
        Models::Both {
            ngram,
            lm,
            lm_weight,
        } => run.sections(
            &Both {
                first: ngram,
                first_weight: 1.0,
                second: lm,
                second_weight: lm_weight,
            },
            "kn-trigram+char-lm",
            lm_weight,
            lm.backend() != ime_lm::Backend::Cpu,
        )?,
    };
    if let Some(dir) = dump {
        for section in &sections {
            write_dump(dir, section)?;
        }
    }
    Ablation { sections }
        .render()
        .context("could not render the ablation")
}

/// Decode every record of the chosen slice and fold it into one report.
///
/// Records are independent, so the slice decodes in parallel and the per-record
/// observations merge; every metric is a ratio of summed counters, so the
/// report is identical to a sequential pass. When *dump* is set, each record's
/// beam rides along beside its observation and the rows come back sorted by
/// record; unset, nothing is collected beyond the report.
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a distinct input to the decode the function exists to fold"
)]
fn measure<E, T>(
    set: &EvalSet,
    slice: SliceArg,
    dev_share: f64,
    reader: &Reader,
    emissions: &E,
    transition: &T,
    beam: &BeamOptions,
    batch: usize,
    lockstep: bool,
    dump: bool,
    progress: Option<&ProgressRequest<'_>>,
) -> Result<(Report, Vec<DumpRow>)>
where
    E: Emissions + Sync,
    T: Transition + Sync,
{
    if let Some(progress) = progress {
        return measure_resumable(
            set, slice, dev_share, reader, emissions, transition, beam, batch, lockstep, dump,
            progress,
        );
    }
    if batch > 1 || lockstep {
        return measure_batched(
            set,
            slice,
            dev_share,
            reader,
            emissions,
            transition,
            beam,
            batch.max(1),
            dump,
        );
    }
    let (report, mut rows) = set
        .records()
        .par_iter()
        .enumerate()
        .filter(|(_, record)| slice.slice().holds(record, dev_share))
        .map(|(index, record)| -> Result<(Report, Vec<DumpRow>)> {
            let (hypotheses, texts) =
                decode_record(index, record, reader, emissions, transition, beam)?
                    .unwrap_or_default();
            let mut report = Report::new(beam.top_k);
            report.observe(&record.text, &texts);
            let row = dump.then(|| DumpRow {
                record: index,
                text: record.text.clone(),
                hypotheses: hypotheses
                    .iter()
                    .zip(texts)
                    .map(|(hypothesis, text)| DumpedHypothesis {
                        text,
                        score: hypothesis.score(),
                    })
                    .collect(),
            });
            Ok((report, row.into_iter().collect::<Vec<_>>()))
        })
        .try_reduce(
            || (Report::new(beam.top_k), Vec::new()),
            |(report, mut rows), (other, more)| {
                rows.extend(more);
                Ok((report.merge(&other), rows))
            },
        )?;
    rows.sort_unstable_by_key(|row| row.record);
    Ok((report, rows))
}

/// Decode one readable record down to its hypotheses and their strings.
///
/// `None` when the keystrokes admit no reading -- the caller then scores the
/// record as unanswered rather than skipping it.
///
/// # Errors
///
/// If a record's scores do not describe its lattice or it cannot be decoded.
fn decode_record<E, T>(
    index: usize,
    record: &EvalRecord,
    reader: &Reader,
    emissions: &E,
    transition: &T,
    beam: &BeamOptions,
) -> Result<Option<(Vec<Hypothesis>, Vec<String>)>>
where
    E: Emissions,
    T: Transition,
{
    let Some((segmentations, candidates)) = reader.read(record)? else {
        return Ok(None);
    };
    let emission = emissions.model(index, record, &segmentations, &candidates)?;
    let hypotheses = decode_many(
        &[Record {
            candidates: &candidates,
            emission,
            context: record.context.as_deref(),
        }],
        transition,
        beam,
    )
    .map(|mut results| results.pop().expect("one record decodes to one result"))
    .with_context(|| format!("could not decode record {index}"))?;
    let texts: Vec<String> = hypotheses
        .iter()
        .map(|hypothesis| hypothesis.text(&reader.lexicon))
        .collect();
    Ok(Some((hypotheses, texts)))
}

/// Decode one batch of records in lockstep.
///
/// Every record's lattice and emission model is built before the decode
/// starts, because `decode_many` borrows all of them at once.
///
/// # Errors
///
/// If a record other than an unreadable one cannot be read, its scores do not
/// describe its lattice, or the batch cannot be decoded. A record whose
/// keystrokes admit no reading decodes to no hypotheses -- it is scored as
/// unanswered, not skipped.
fn decode_chunk<E, T>(
    chunk: &[(usize, &EvalRecord)],
    reader: &Reader,
    emissions: &E,
    transition: &T,
    beam: &BeamOptions,
) -> Result<Vec<Vec<Hypothesis>>>
where
    E: Emissions,
    T: Transition,
{
    let mut lattices = Vec::with_capacity(chunk.len());
    for (position, &(index, record)) in chunk.iter().enumerate() {
        if let Some((segmentations, candidates)) = reader.read(record)? {
            lattices.push((position, index, record, segmentations, candidates));
        }
    }
    let mut requests = Vec::with_capacity(lattices.len());
    for (_, index, record, segmentations, candidates) in &lattices {
        requests.push(Record {
            candidates,
            emission: emissions.model(*index, record, segmentations, candidates)?,
            context: record.context.as_deref(),
        });
    }
    let mut decoded = decode_many(&requests, transition, beam)
        .with_context(|| {
            format!(
                "could not decode records {}..={}",
                chunk[0].0,
                chunk[chunk.len() - 1].0
            )
        })?
        .into_iter();
    let mut answers: Vec<Vec<Hypothesis>> = (0..chunk.len()).map(|_| Vec::new()).collect();
    for &(position, ..) in &lattices {
        answers[position] = decoded.next().expect("one request decodes to one result");
    }
    Ok(answers)
}

/// Decode the slice's records in lockstep batches of *batch* records each.
///
/// Batched because the transition's step is one `advance` call covering the
/// whole batch's rows, so chunks run serially: the batch is where the records
/// run together, not a thread pool.
///
/// # Errors
///
/// If a record cannot be read, its scores do not describe its lattice, or a
/// batch cannot be decoded.
#[expect(
    clippy::too_many_arguments,
    reason = "it takes exactly what `measure` does minus what progress needs"
)]
fn measure_batched<E, T>(
    set: &EvalSet,
    slice: SliceArg,
    dev_share: f64,
    reader: &Reader,
    emissions: &E,
    transition: &T,
    beam: &BeamOptions,
    batch: usize,
    dump: bool,
) -> Result<(Report, Vec<DumpRow>)>
where
    E: Emissions,
    T: Transition,
{
    let selected: Vec<(usize, &EvalRecord)> = set
        .records()
        .iter()
        .enumerate()
        .filter(|(_, record)| slice.slice().holds(record, dev_share))
        .collect();
    let mut report = Report::new(beam.top_k);
    let mut rows = Vec::new();
    let mut done = 0usize;
    for chunk in selected.chunks(batch) {
        let results = decode_chunk(chunk, reader, emissions, transition, beam)?;
        for (&(index, record), hypotheses) in chunk.iter().zip(results) {
            let texts: Vec<String> = hypotheses
                .iter()
                .map(|hypothesis| hypothesis.text(&reader.lexicon))
                .collect();
            report.observe(&record.text, &texts);
            if dump {
                rows.push(DumpRow {
                    record: index,
                    text: record.text.clone(),
                    hypotheses: hypotheses
                        .iter()
                        .zip(texts)
                        .map(|(hypothesis, text)| DumpedHypothesis {
                            text,
                            score: hypothesis.score(),
                        })
                        .collect(),
                });
            }
        }
        done += chunk.len();
        info!(done, total = selected.len(), "decoded a lockstep batch");
    }
    Ok((report, rows))
}

/// Everything a `measure` call needs to keep progress: where the files live
/// and every input its configuration key pins down.
struct ProgressRequest<'a> {
    /// The directory the per-configuration files are kept in.
    dir: &'a Path,
    /// The test stand-in for a kill: stop after this many new records.
    stop_after: Option<usize>,
    eval_set: &'a Path,
    scores: Option<&'a Path>,
    /// The export directory live emissions run from, when they do.
    route_a: Option<&'a Path>,
    /// Whether live emissions read the context; `None` with a score file.
    with_context: Option<bool>,
    emittable: &'a Path,
    model: Option<&'a Path>,
    lm: Option<&'a Path>,
    lm_weight: f32,
    floor: f32,
    /// The emission weight this call decodes at.
    weight: f32,
    /// The correction-prior weight this call decodes at.
    typo_weight: f32,
    /// The transition's label in the report.
    transition: &'static str,
}

/// The part of a configuration that names a progress file: which records are
/// decoded, over which inputs, at which emission weight, through which
/// transition.
///
/// Everything else a run can change is checked against the file's header
/// instead of being part of the name, so that resuming under different search
/// settings is refused rather than restarted under another name -- a restart
/// is exactly what `--progress` exists to avoid.
#[derive(Debug, Serialize, Deserialize)]
struct Name {
    eval_set: PathBuf,
    scores: Option<PathBuf>,
    /// The export directory a live-emission run decodes through.
    #[serde(default)]
    route_a: Option<PathBuf>,
    /// Whether live emissions read the context; absent from a score-file
    /// run's key.
    #[serde(default)]
    with_context: Option<bool>,
    emittable: PathBuf,
    model: Option<PathBuf>,
    lm: Option<PathBuf>,
    slice: String,
    weight: f32,
    /// The correction-prior weight the decode ran at; part of the name so a
    /// sweep's files stay distinct.
    typo_weight: f32,
    transition: String,
}

/// The whole configuration a `measure` call ran under, serialised as the
/// progress file's first line. Resuming requires an exact match; a mismatch
/// names the fields that differ.
#[derive(Debug, Serialize, Deserialize)]
struct Key {
    /// Which decode this is: the part the file is named for.
    #[serde(flatten)]
    name: Name,
    /// The character model's weight within the fused transition.
    lm_weight: f32,
    /// What a candidate the model has no row for scores.
    floor: f32,
    /// How many readings of the keystrokes are decoded.
    max_paths: usize,
    /// Whether a lone initial may stand for a syllable.
    allow_abbreviation: bool,
    /// Whether the lattice corrected keystrokes against the typo noise model.
    corrections: bool,
    /// Whether a trailing half-typed syllable is accepted.
    incomplete_tail: bool,
    /// The flat cost of one more character position.
    segment_cost: f32,
    /// The weight on a segment's log ambiguity.
    ambiguity_weight: f32,
    /// The penalty on unconventional segmentations.
    segmentation_weight: f32,
    /// How many beam states survive each position.
    beam_width: usize,
    /// How many hypotheses each record is ranked on.
    top_k: usize,
    /// The share of the set the dev slice holds.
    dev_share: f64,
}

/// One record's outcome: one line of the progress file.
#[derive(Debug, Serialize, Deserialize)]
struct ProgressLine {
    /// The record's index in the evaluation set.
    record: usize,
    /// What the record contributed to the report.
    #[serde(flatten)]
    observation: Observation,
    /// The beam it decoded into, kept so that a resumed run's `--dump` is the
    /// file an uninterrupted run's would have been.
    hypotheses: Vec<DumpedHypothesis>,
}

/// One configuration's progress file.
///
/// The file's first line is the key it was written under; every later line is
/// one record's outcome, appended by a single writer thread the parallel
/// decode hands lines to over a channel, flushed as it lands. A kill at any
/// moment loses at most the record in flight.
struct Journal {
    /// Where the file lives, so the report can be folded back out of it.
    path: PathBuf,
    /// Records the file already held when it was opened.
    done: HashSet<usize>,
    /// The channel to the writer thread.
    sender: Option<Sender<ProgressLine>>,
    /// The writer thread, joined on close.
    writer: Option<JoinHandle<Result<()>>>,
}

impl Journal {
    /// Open the file for *key* in *dir*: read the records it already holds,
    /// refuse a key it was not written under, and start the writer that
    /// appends what this run decodes.
    ///
    /// # Errors
    ///
    /// If the directory cannot be created, the file cannot be read or
    /// written, its header was written under another configuration, or a
    /// complete line of it is not a record's outcome.
    fn open(dir: &Path, key: &Key) -> Result<Self> {
        fs::create_dir_all(dir).with_context(|| {
            format!("could not create the progress directory {}", dir.display())
        })?;
        let path = dir.join(format!("{}.jsonl", digest(&key.name)?));
        let mut done = HashSet::new();
        match fs::read(&path) {
            Ok(bytes) => {
                // A line is committed only once its newline has landed; a
                // trailing fragment is the record a kill cut off mid-write and
                // is dropped so the next append does not glue onto it.
                let complete = bytes
                    .iter()
                    .rposition(|&byte| byte == b'\n')
                    .map_or(0, |end| end + 1);
                if complete == 0 {
                    // No complete line means the run that created the file was
                    // stopped before its header landed; nothing was committed,
                    // so starting the file over restarts nothing.
                    write_header(&path, key)?;
                } else {
                    let text = std::str::from_utf8(&bytes[..complete])
                        .with_context(|| format!("{} is not UTF-8", path.display()))?;
                    check_header(text, key, &path)?;
                    for line in read_outcomes(text, &path)? {
                        done.insert(line.record);
                    }
                    if complete < bytes.len() {
                        fs::File::options()
                            .write(true)
                            .open(&path)
                            .with_context(|| format!("could not truncate {}", path.display()))?
                            .set_len(u64::try_from(complete).expect("a file length fits in u64"))
                            .with_context(|| format!("could not truncate {}", path.display()))?;
                    }
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                write_header(&path, key)?;
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("could not read the progress file {}", path.display())
                });
            }
        }
        info!(
            path = %path.display(),
            done = done.len(),
            "opened the progress file"
        );
        let (sender, receiver) = mpsc::channel::<ProgressLine>();
        let writer_path = path.clone();
        let writer = thread::spawn(move || -> Result<()> {
            let mut sink = fs::File::options()
                .append(true)
                .open(&writer_path)
                .with_context(|| {
                    format!("could not open {} for appending", writer_path.display())
                })?;
            while let Ok(line) = receiver.recv() {
                serde_json::to_writer(&mut sink, &line).with_context(|| {
                    format!("could not serialise a record of {}", writer_path.display())
                })?;
                sink.write_all(b"\n").with_context(|| {
                    format!("could not write a record of {}", writer_path.display())
                })?;
                sink.flush()
                    .with_context(|| format!("could not flush {}", writer_path.display()))?;
            }
            Ok(())
        });
        Ok(Self {
            path,
            done,
            sender: Some(sender),
            writer: Some(writer),
        })
    }

    /// Hand one finished record's outcome to the writer.
    ///
    /// # Errors
    ///
    /// If the writer thread is gone.
    fn send(&self, line: ProgressLine) -> Result<()> {
        self.sender
            .as_ref()
            .context("the progress file is already closed")?
            .send(line)
            .map_err(|_| anyhow!("the progress writer for {} stopped", self.path.display()))
    }

    /// Close the file: the channel ends, the writer flushes its last line and
    /// is joined.
    ///
    /// # Errors
    ///
    /// Whatever the writer last failed on.
    fn finish(&mut self) -> Result<()> {
        drop(self.sender.take());
        if let Some(writer) = self.writer.take() {
            writer.join().map_err(|_| {
                anyhow!("the progress writer for {} panicked", self.path.display())
            })??;
        }
        Ok(())
    }

    /// Every record line the file holds, oldest first in file order -- which a
    /// parallel decode does not promise.
    ///
    /// # Errors
    ///
    /// If the file cannot be read or a line is not a record's outcome.
    fn lines(&self) -> Result<Vec<ProgressLine>> {
        let bytes = fs::read(&self.path)
            .with_context(|| format!("could not re-read {}", self.path.display()))?;
        let complete = bytes
            .iter()
            .rposition(|&byte| byte == b'\n')
            .map_or(0, |end| end + 1);
        let text = std::str::from_utf8(&bytes[..complete])
            .with_context(|| format!("{} is not UTF-8", self.path.display()))?;
        read_outcomes(text, &self.path)
    }
}

/// The file one configuration's progress is kept in: `<hash>.jsonl`, the hash
/// of the name's serialised form.
fn digest(name: &Name) -> Result<String> {
    let serialised =
        serde_json::to_vec(name).context("could not serialise the progress file's name")?;
    Ok(const_hex::encode(Blake2b::<U8>::digest(&serialised)))
}

/// Start a fresh progress file under *key*: the file's first line is the key
/// itself.
fn write_header(path: &Path, key: &Key) -> Result<()> {
    let mut file =
        fs::File::create(path).with_context(|| format!("could not create {}", path.display()))?;
    serde_json::to_writer(&mut file, key)
        .with_context(|| format!("could not write the header of {}", path.display()))?;
    file.write_all(b"\n")
        .with_context(|| format!("could not write the header of {}", path.display()))?;
    file.flush()
        .with_context(|| format!("could not flush {}", path.display()))
}

/// Refuse *path* when its first line was written under another configuration.
fn check_header(text: &str, key: &Key, path: &Path) -> Result<()> {
    let header = text
        .lines()
        .next()
        .context("the progress file holds no configuration header")?;
    let stored: Key = serde_json::from_str(header).with_context(|| {
        format!(
            "the first line of {} is not a configuration key",
            path.display()
        )
    })?;
    let differing = differing(&stored, key)?;
    if !differing.is_empty() {
        bail!(
            "the progress file {} was written under a different configuration: {}",
            path.display(),
            differing.join(", ")
        );
    }
    Ok(())
}

/// The fields on which a stored key and this run's disagree, each rendered as
/// `name (stored, wanted)` for the refusal.
fn differing(stored: &Key, key: &Key) -> Result<Vec<String>> {
    let stored = serde_json::to_value(stored).context("could not re-serialise the stored key")?;
    let key = serde_json::to_value(key).context("could not serialise the key")?;
    let (serde_json::Value::Object(stored), serde_json::Value::Object(key)) = (stored, key) else {
        bail!("a configuration key is not an object");
    };
    let render = |value: Option<&serde_json::Value>| {
        value.map_or_else(|| "unset".to_owned(), ToString::to_string)
    };
    let mut differing: Vec<String> = key
        .keys()
        .filter(|field| stored.get(*field) != key.get(*field))
        .map(|field| {
            format!(
                "{field} ({} != {})",
                render(stored.get(field)),
                render(key.get(field))
            )
        })
        .collect();
    differing.sort();
    Ok(differing)
}

/// The record lines of a progress file's *text*: every complete line but the
/// header.
fn read_outcomes(text: &str, path: &Path) -> Result<Vec<ProgressLine>> {
    if text.is_empty() {
        bail!("{} holds no configuration header", path.display());
    }
    let mut lines = Vec::new();
    let mut seen = HashSet::new();
    for (index, raw) in text.lines().enumerate().skip(1) {
        let line: ProgressLine = serde_json::from_str(raw).with_context(|| {
            format!(
                "line {} of {} is not a record's outcome",
                index + 1,
                path.display()
            )
        })?;
        if !seen.insert(line.record) {
            bail!("{} lists record {} twice", path.display(), line.record);
        }
        lines.push(line);
    }
    Ok(lines)
}

/// Decode the slice's pending records, appending each outcome to the
/// configuration's progress file as it lands, then fold the report out of the
/// file's lines so it is identical whether or not the run was interrupted.
#[expect(
    clippy::too_many_arguments,
    reason = "it takes exactly what `measure` does plus what the progress file needs"
)]
fn measure_resumable<E, T>(
    set: &EvalSet,
    slice: SliceArg,
    dev_share: f64,
    reader: &Reader,
    emissions: &E,
    transition: &T,
    beam: &BeamOptions,
    batch: usize,
    lockstep: bool,
    dump: bool,
    progress: &ProgressRequest<'_>,
) -> Result<(Report, Vec<DumpRow>)>
where
    E: Emissions + Sync,
    T: Transition + Sync,
{
    let key = Key {
        name: Name {
            eval_set: progress.eval_set.to_owned(),
            scores: progress.scores.map(Path::to_owned),
            route_a: progress.route_a.map(Path::to_owned),
            with_context: progress.with_context,
            emittable: progress.emittable.to_owned(),
            model: progress.model.map(Path::to_owned),
            lm: progress.lm.map(Path::to_owned),
            slice: slice.label().to_owned(),
            weight: progress.weight,
            typo_weight: progress.typo_weight,
            transition: progress.transition.to_owned(),
        },
        lm_weight: progress.lm_weight,
        floor: progress.floor,
        max_paths: reader.segment.max_paths,
        allow_abbreviation: reader.segment.allow_abbreviation,
        corrections: reader.corrections.is_some(),
        incomplete_tail: reader.segment.allow_incomplete_tail,
        segment_cost: reader.segment.segment_cost,
        ambiguity_weight: reader.segment.ambiguity_weight,
        segmentation_weight: beam.segmentation_weight,
        beam_width: beam.beam_width.get(),
        top_k: beam.top_k.get(),
        dev_share,
    };
    let mut journal = Journal::open(progress.dir, &key)?;
    let mut pending: Vec<(usize, &EvalRecord)> = set
        .records()
        .par_iter()
        .enumerate()
        .filter(|(_, record)| slice.slice().holds(record, dev_share))
        .filter(|(index, _)| !journal.done.contains(index))
        .collect();
    pending.sort_unstable_by_key(|(index, _)| *index);
    if let Some(stop_after) = progress.stop_after {
        pending.truncate(stop_after);
    }
    if batch > 1 || lockstep {
        for chunk in pending.chunks(batch.max(1)) {
            let results = decode_chunk(chunk, reader, emissions, transition, beam)?;
            for (&(index, record), hypotheses) in chunk.iter().zip(results) {
                let texts: Vec<String> = hypotheses
                    .iter()
                    .map(|hypothesis| hypothesis.text(&reader.lexicon))
                    .collect();
                journal.send(ProgressLine {
                    record: index,
                    observation: Observation::new(&record.text, &texts, beam.top_k.get()),
                    hypotheses: hypotheses
                        .iter()
                        .zip(texts)
                        .map(|(hypothesis, text)| DumpedHypothesis {
                            text,
                            score: hypothesis.score(),
                        })
                        .collect(),
                })?;
            }
        }
        journal.finish()?;
        return fold_journal(&journal, set, beam, dump);
    }
    pending
        .par_iter()
        .map(|&(index, record)| -> Result<()> {
            let (hypotheses, texts) =
                decode_record(index, record, reader, emissions, transition, beam)?
                    .unwrap_or_default();
            journal.send(ProgressLine {
                record: index,
                observation: Observation::new(&record.text, &texts, beam.top_k.get()),
                hypotheses: hypotheses
                    .iter()
                    .zip(texts)
                    .map(|(hypothesis, text)| DumpedHypothesis {
                        text,
                        score: hypothesis.score(),
                    })
                    .collect(),
            })
        })
        .collect::<Result<Vec<()>>>()?;
    journal.finish()?;
    fold_journal(&journal, set, beam, dump)
}

/// Fold a closed progress file into the section's report and dump rows --
/// identical whether the run was interrupted or not.
///
/// # Errors
///
/// If the file cannot be read back or a line names a record the set does not
/// hold.
fn fold_journal(
    journal: &Journal,
    set: &EvalSet,
    beam: &BeamOptions,
    dump: bool,
) -> Result<(Report, Vec<DumpRow>)> {
    let mut report = Report::new(beam.top_k);
    let mut rows = Vec::new();
    for line in journal.lines()? {
        let record = set.records().get(line.record).with_context(|| {
            format!(
                "{} names record {}, but the eval set holds {}",
                journal.path.display(),
                line.record,
                set.len()
            )
        })?;
        report.fold(&line.observation);
        if dump {
            rows.push(DumpRow {
                record: line.record,
                text: record.text.clone(),
                hypotheses: line.hypotheses,
            });
        }
    }
    rows.sort_unstable_by_key(|row| row.record);
    Ok((report, rows))
}

/// Write one section's beam to its file in the dump directory: a JSON Lines
/// file named for the section, one record per line in file order.
///
/// # Errors
///
/// If the file cannot be created or written.
fn write_dump(dir: &Path, section: &Section) -> Result<()> {
    let typo_weight = if section.corrections {
        format!("-t{:.3}", section.typo_weight)
    } else {
        String::new()
    };
    let path = dir.join(format!(
        "{}-w{:.3}{}-{}-{}.jsonl",
        section.emission, section.weight, typo_weight, section.transition, section.slice
    ));
    let file =
        fs::File::create(&path).with_context(|| format!("could not create {}", path.display()))?;
    let mut sink = BufWriter::new(file);
    for row in &section.rows {
        serde_json::to_writer(&mut sink, row)
            .with_context(|| format!("could not serialise a row of {}", path.display()))?;
        sink.write_all(b"\n")
            .with_context(|| format!("could not write a row of {}", path.display()))?;
    }
    sink.flush()
        .with_context(|| format!("could not flush {}", path.display()))?;
    info!(rows = section.rows.len(), path = %path.display(), "wrote the hypotheses");
    Ok(())
}

/// Read an evaluation set off disk.
pub(crate) fn load_set(path: &Path) -> Result<EvalSet> {
    let source = fs::read_to_string(path)
        .with_context(|| format!("could not read the eval set at {}", path.display()))?;
    EvalSet::parse(&source).context("the eval set is malformed")
}

/// Read the characters the model can score.
pub(crate) fn load_emittable(path: &Path, lexicon: &Lexicon) -> Result<Emittable> {
    let source = fs::read_to_string(path)
        .with_context(|| format!("could not read the emittable set at {}", path.display()))?;
    let emittable = Emittable::parse(&source, lexicon)
        .context("the emittable set names characters this lexicon does not hold")?;
    info!(characters = emittable.len(), path = %path.display(), "loaded the emittable set");
    Ok(emittable)
}

/// Lines of the score file read ahead of the parse at a time: large enough
/// that the parallel parse dominates the sequential decompress, small enough
/// that a chunk is a rounding error next to the map it folds into.
const SCORE_CHUNK: usize = 1 << 16;

/// Read a gzipped score file into the records it answers.
///
/// Gzipped because it is not small: twenty-one million log probabilities is a
/// couple of hundred megabytes as text and a fifth of that compressed, and the
/// file has to come off a Kaggle kernel before anything can be measured. The
/// parse, not the decompress, is the expensive half, so each chunk of lines is
/// parsed in parallel and folded into the map in file order -- the resident
/// text is one chunk rather than the file, and the duplicate-record check
/// still names the same line it always did.
fn load_scores(path: &Path) -> Result<HashMap<usize, Vec<Vec<Vec<f32>>>>> {
    let file = fs::File::open(path)
        .with_context(|| format!("could not read the scores at {}", path.display()))?;
    let mut lines = BufReader::new(MultiGzDecoder::new(file))
        .lines()
        .enumerate();
    let mut scores = HashMap::new();
    loop {
        let chunk: Vec<(usize, String)> = lines
            .by_ref()
            .take(SCORE_CHUNK)
            .map(|(index, line)| {
                line.map(|line| (index, line))
                    .with_context(|| format!("could not read line {} of the scores", index + 1))
            })
            .collect::<Result<_>>()?;
        if chunk.is_empty() {
            break;
        }
        let parsed: Vec<(usize, ScoreRecord)> = chunk
            .par_iter()
            .filter(|(_, line)| !line.trim().is_empty())
            .map(|(index, line)| {
                serde_json::from_str::<ScoreRecord>(line)
                    .with_context(|| format!("line {} of the scores is not a record", index + 1))
                    .map(|record| (*index + 1, record))
            })
            .collect::<Result<_>>()?;
        for (line, record) in parsed {
            if scores.insert(record.record, record.paths).is_some() {
                bail!(
                    "the score file answers record {} twice, at line {}",
                    record.record,
                    line
                );
            }
        }
    }
    if scores.is_empty() {
        bail!("{} holds no scores", path.display());
    }
    info!(records = scores.len(), path = %path.display(), "loaded the emissions");
    Ok(scores)
}

/// Parse the fusion weights a run sweeps over.
///
/// # Errors
///
/// If a weight is not a number, or is negative -- a negative weight would ask
/// the decoder to prefer the characters the model ruled out.
pub fn parse_weight(raw: &str) -> Result<f32, String> {
    let weight: f32 = raw
        .parse()
        .map_err(|_| format!("{raw:?} is not a fusion weight"))?;
    if weight < 0.0 || !weight.is_finite() {
        return Err(format!(
            "a fusion weight must be finite and non-negative, got {weight}"
        ));
    }
    Ok(weight)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A fresh temporary directory per test, removed on drop.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let dir = std::env::temp_dir().join(format!(
                "ml-ime-fused-eval-{name}-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&dir).expect("the scratch directory is creatable");
            Self(dir)
        }

        fn path(&self, file: &str) -> PathBuf {
            self.0.join(file)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// A synthetic fused-eval fixture: an eval set, an emittable set and a
    /// score file answering them with zeros. Zero scores carry no signal,
    /// which is what makes the decode deterministic and a dev sweep break its
    /// ties to the first weight.
    struct Fixture {
        scratch: Scratch,
        eval_set: PathBuf,
        emittable: PathBuf,
        scores: PathBuf,
        progress_dir: PathBuf,
        /// The `char_pinyin.tsv` the run's lexicon is parsed from; empty for
        /// the generated lexicon.
        lexicon: String,
    }

    impl Fixture {
        /// Build a fixture over *records* (`(pinyin, expected)` pairs), with
        /// *emittable* the characters the pretend model can score and
        /// *lexicon* the character table, or empty for the generated one.
        fn new(records: &[(&str, &str)], emittable: &str, lexicon: &str) -> Self {
            let scratch = Scratch::new("fixture");
            let eval_set = scratch.path("eval.jsonl");
            let mut source = String::new();
            for &(pinyin, text) in records {
                let record = EvalRecord {
                    pinyin: pinyin.to_owned(),
                    text: text.to_owned(),
                    context: None,
                    clean: None,
                };
                source.push_str(&serde_json::to_string(&record).expect("a record serialises"));
                source.push('\n');
            }
            fs::write(&eval_set, source).expect("the eval set writes");
            let emittable_path = scratch.path("emittable.txt");
            fs::write(&emittable_path, emittable).expect("the emittable set writes");
            let fixture = Self {
                progress_dir: scratch.path("progress"),
                scores: scratch.path("scores.jsonl.gz"),
                scratch,
                eval_set,
                emittable: emittable_path,
                lexicon: lexicon.to_owned(),
            };
            fixture.write_scores();
            fixture
        }

        /// The lexicon this fixture decodes with.
        fn lexicon(&self) -> Lexicon {
            let table = SyllableTable::load();
            if self.lexicon.is_empty() {
                Lexicon::load(&table).expect("the generated lexicon loads")
            } else {
                Lexicon::parse(&self.lexicon, &table).expect("the fixture lexicon parses")
            }
        }

        /// Write the score file that answers this fixture's lattice with zeros.
        fn write_scores(&self) {
            let lattice = self.scratch.path("lattice.jsonl");
            emit_lattice(
                &self.eval_set,
                &lattice,
                &self.emittable,
                SyllableTable::load(),
                self.lexicon(),
                segment(),
                None,
            )
            .expect("the lattice emits");
            let source = fs::read_to_string(&lattice).expect("the lattice reads");
            let file = fs::File::create(&self.scores).expect("the score file is creatable");
            let mut sink = flate2::write::GzEncoder::new(file, flate2::Compression::default());
            for raw in source.lines() {
                let lattice: LatticeRecord =
                    serde_json::from_str(raw).expect("a lattice record parses");
                let paths = lattice
                    .paths
                    .iter()
                    .map(|path| {
                        path.candidates
                            .iter()
                            .map(|candidates| vec![0.0; candidates.chars().count()])
                            .collect()
                    })
                    .collect();
                let record = ScoreRecord {
                    record: lattice.record,
                    paths,
                };
                serde_json::to_writer(&mut sink, &record).expect("a score record serialises");
                sink.write_all(b"\n").expect("a score record writes");
            }
            sink.finish().expect("the score file flushes");
        }

        /// One `fused-eval` run over the fixture.
        ///
        /// # Errors
        ///
        /// Whatever the run reports.
        fn run(
            &self,
            weights: &[f32],
            dev_share: f64,
            select_on_dev: bool,
            transition: Models<'_>,
            progress: Option<&Progress<'_>>,
        ) -> Result<String> {
            fused_eval(
                &self.eval_set,
                Some(&self.scores),
                None,
                &self.emittable,
                -30.0,
                weights,
                &[1.0],
                &SliceArgs {
                    slice: SliceArg::All,
                    dev_share,
                    select_on_dev,
                },
                None,
                progress,
                SyllableTable::load(),
                self.lexicon(),
                segment(),
                None,
                &BeamOptions {
                    beam_width: NonZeroUsize::new(16).expect("16 is not zero"),
                    ..BeamOptions::default()
                },
                1,
                transition,
            )
        }

        /// A `Progress` into this fixture's directory.
        fn progress<'a>(&'a self, stop_after: Option<usize>, lm: Option<&'a Path>) -> Progress<'a> {
            Progress {
                dir: &self.progress_dir,
                model: None,
                lm,
                stop_after,
            }
        }

        /// How many records fall in the dev slice at *share*, with the total.
        fn dev_records(&self, share: f64) -> (usize, usize) {
            let set =
                EvalSet::parse(&fs::read_to_string(&self.eval_set).expect("the eval set reads"))
                    .expect("the eval set parses");
            let dev = set
                .records()
                .iter()
                .filter(|record| Slice::Dev.holds(record, share))
                .count();
            (dev, set.len())
        }

        /// The files a *slice* pass keeps its progress in, with their bytes.
        fn pass_files(&self, slice: &str) -> Vec<(PathBuf, Vec<u8>)> {
            let mut files = fs::read_dir(&self.progress_dir)
                .expect("the progress directory reads")
                .map(|entry| entry.expect("an entry reads").path())
                .filter(|path| {
                    fs::read_to_string(path)
                        .expect("a progress file reads")
                        .lines()
                        .next()
                        .and_then(|header| serde_json::from_str::<serde_json::Value>(header).ok())
                        .is_some_and(|key| key["slice"].as_str() == Some(slice))
                })
                .map(|path| {
                    let bytes = fs::read(&path).expect("a progress file reads");
                    (path, bytes)
                })
                .collect::<Vec<_>>();
            files.sort();
            files
        }

        /// The progress files, `<hash>.jsonl`, as `(header, record lines)`.
        fn progress_files(&self) -> Vec<(serde_json::Value, Vec<String>)> {
            let mut files = fs::read_dir(&self.progress_dir)
                .expect("the progress directory reads")
                .map(|entry| entry.expect("an entry reads").path())
                .collect::<Vec<_>>();
            files.sort();
            files
                .iter()
                .map(|path| {
                    let source = fs::read_to_string(path).expect("a progress file reads");
                    let mut lines = source.lines();
                    let header: serde_json::Value =
                        serde_json::from_str(lines.next().expect("a progress file has a header"))
                            .expect("the header parses");
                    let records = lines.map(str::to_owned).collect();
                    (header, records)
                })
                .collect()
        }
    }

    /// The record lines a progress file's bytes carry (header excluded).
    fn record_lines(bytes: &[u8]) -> Vec<String> {
        let text = String::from_utf8(bytes.to_vec()).expect("a progress file is UTF-8");
        let mut lines = text.lines().skip(1).map(str::to_owned).collect::<Vec<_>>();
        lines.sort();
        lines
    }

    /// The segment options the runs and their lattice share: exact readings
    /// only, so the fixture's eight syllables need no abbreviation or partial
    /// tail to stay decodable.
    fn segment() -> SegmentOptions {
        SegmentOptions {
            allow_abbreviation: false,
            allow_incomplete_tail: false,
            ..SegmentOptions::default()
        }
    }

    #[test]
    fn a_resumed_run_reports_what_an_uninterrupted_one_did() {
        let fixture = Fixture::new(
            &[
                ("nihao", "你好"),
                ("zhongguo", "中国"),
                ("renmin", "人民"),
                ("yinhang", "银行"),
                ("beijing", "北京"),
                ("tianqi", "天气"),
                ("women", "我们"),
                ("zaijian", "再见"),
                ("henhao", "很好"),
                ("xiexie", "谢谢"),
            ],
            "你\n好\n中\n国\n人\n民\n银\n行\n北\n京\n天\n气\n我\n们\n再\n见\n很\n谢",
            "",
        );
        let baseline = fixture
            .run(&[1.0], 0.5, false, Models::None, None)
            .expect("the uninterrupted run decodes");
        fixture
            .run(
                &[1.0],
                0.5,
                false,
                Models::None,
                Some(&fixture.progress(Some(3), None)),
            )
            .expect("the stopped run decodes");
        let resumed = fixture
            .run(
                &[1.0],
                0.5,
                false,
                Models::None,
                Some(&fixture.progress(None, None)),
            )
            .expect("the resumed run decodes");
        assert_eq!(resumed, baseline);
        let files = fixture.progress_files();
        assert_eq!(files.len(), 1, "one configuration, one file");
        assert_eq!(
            files[0].1.len(),
            10,
            "every record landed exactly once: no line was written twice on resume"
        );
        // Resuming a finished run decodes nothing and reports the same.
        let again = fixture
            .run(
                &[1.0],
                0.5,
                false,
                Models::None,
                Some(&fixture.progress(None, None)),
            )
            .expect("the finished run decodes");
        assert_eq!(again, baseline);
        assert_eq!(fixture.progress_files()[0].1.len(), 10);
    }

    #[test]
    fn a_resumed_run_under_changed_arguments_is_refused() {
        let fixture = Fixture::new(
            &[("nihao", "你好"), ("zhongguo", "中国"), ("renmin", "人民")],
            "你\n好\n中\n国\n人\n民",
            "",
        );
        fixture
            .run(
                &[1.0],
                0.5,
                false,
                Models::None,
                Some(&fixture.progress(None, None)),
            )
            .expect("the first run decodes");
        let refused = fixture
            .run(
                &[1.0],
                0.75,
                false,
                Models::None,
                Some(&fixture.progress(None, None)),
            )
            .expect_err("a changed dev share must be refused");
        let message = format!("{refused:#}");
        assert!(
            message.contains("dev_share"),
            "the refusal names the field that changed: {message}"
        );
        // A different weight is a different configuration -- its own file,
        // not an error.
        fixture
            .run(
                &[0.5],
                0.5,
                false,
                Models::None,
                Some(&fixture.progress(None, None)),
            )
            .expect("another weight is another file, not a refusal");
        assert_eq!(fixture.progress_files().len(), 2);
    }

    #[test]
    fn select_on_dev_resumes_each_pass_on_its_own_file() {
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../ime-lm/tests/fixtures");
        let lexicon =
            fs::read_to_string(fixtures.join("char_pinyin.tsv")).expect("the table reads");
        // The fixture lexicon covers eight syllables, and a record fails to
        // read if any segmentation of its keystrokes needs one outside them --
        // so the records stick to `ni`, `zai` and `wo`, the three whose every
        // concatenation segments in exactly one way (the other five split into
        // uncovered pieces: `m`/`a`, `he`/`n`, `ha`/`o`, `ji`/`an`, `xi`/`e`).
        let fixture = Fixture::new(
            &[
                ("niwo", "你我"),
                ("woni", "我你"),
                ("nizai", "你再"),
                ("zaini", "再你"),
                ("wozai", "我再"),
                ("zaiwo", "再我"),
                ("nini", "你你"),
                ("wowo", "我我"),
                ("zaizai", "再再"),
                ("niwozai", "你我再"),
                ("wozaini", "我再你"),
                ("zainiwo", "再你我"),
            ],
            "你\n再\n我",
            &lexicon,
        );
        // Both halves of the set must be populated for the sweep to mean
        // anything.
        let (dev, total) = fixture.dev_records(0.4);
        assert!(dev > 0 && dev < total, "the fixture must split");
        let table = SyllableTable::load();
        let lm_lexicon = Lexicon::parse(&lexicon, &table).expect("the fixture lexicon parses");
        let lm = CharLm::open(
            &fixtures.join("transformer"),
            &lm_lexicon,
            ime_lm::SessionShape::default(),
        )
        .expect("the transformer fixture opens");
        let lm_dir = fixtures.join("transformer");
        let weights = [0.5, 1.0];
        let baseline = fixture
            .run(&weights, 0.4, true, Models::CharLm(&lm), None)
            .expect("the uninterrupted run decodes");
        fixture
            .run(
                &weights,
                0.4,
                true,
                Models::CharLm(&lm),
                Some(&fixture.progress(Some(1), Some(&lm_dir))),
            )
            .expect("the stopped run decodes");
        let resumed = fixture
            .run(
                &weights,
                0.4,
                true,
                Models::CharLm(&lm),
                Some(&fixture.progress(None, Some(&lm_dir))),
            )
            .expect("the resumed run decodes");
        assert_eq!(resumed, baseline);
        // Two dev weights and the test pass at the winner are three files.
        let files = fixture.progress_files();
        assert_eq!(files.len(), 3, "dev sweep plus test pass");
        for (header, lines) in &files {
            let expected = match header["slice"].as_str() {
                Some("dev") => dev,
                Some("test") => total - dev,
                other => panic!("a pass writes to a named slice, not {other:?}"),
            };
            assert_eq!(lines.len(), expected, "every record of the slice landed");
        }
        // Each file resumes on its own: deleting the test file re-decodes the
        // test pass only, and the dev files are left untouched.
        let dev_files = fixture.pass_files("dev");
        let test_files = fixture.pass_files("test");
        assert_eq!(dev_files.len(), 2);
        let [(_, test_bytes)] = test_files.as_slice() else {
            panic!("one test file, not {}", test_files.len());
        };
        let test_lines = record_lines(test_bytes);
        fs::remove_file(&test_files[0].0).expect("the test file is removable");
        let again = fixture
            .run(
                &weights,
                0.4,
                true,
                Models::CharLm(&lm),
                Some(&fixture.progress(None, Some(&lm_dir))),
            )
            .expect("the run rebuilds only the test pass");
        assert_eq!(again, baseline);
        assert_eq!(
            fixture.pass_files("dev"),
            dev_files,
            "the dev files were resumed, not rewritten"
        );
        let rebuilt = fixture.pass_files("test");
        assert_eq!(rebuilt.len(), 1, "the test file was rebuilt");
        assert_eq!(record_lines(&rebuilt[0].1), test_lines);
    }

    #[test]
    fn a_negative_fusion_weight_is_refused() {
        assert!(parse_weight("-0.5").is_err());
        assert!(parse_weight("nan").is_err());
        assert!((parse_weight("0.75").expect("0.75 parses") - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn the_slices_partition_the_set() {
        let record = EvalRecord {
            pinyin: "nihao".to_owned(),
            text: "你好".to_owned(),
            context: None,
            clean: None,
        };
        assert!(SliceArg::All.slice().holds(&record, 0.5));
        assert_ne!(
            SliceArg::Dev.slice().holds(&record, 0.5),
            SliceArg::Test.slice().holds(&record, 0.5)
        );
    }
}
