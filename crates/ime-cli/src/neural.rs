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
use anyhow::{Context as _, Result, bail};
use askama::Template;
use clap::{Args, ValueEnum};
use flate2::read::MultiGzDecoder;
use ime_decode::{
    BeamOptions, Both, Candidates, Emission, Emittable, LatticePath, LatticeRecord, NoTransition,
    ScoreRecord, Scored, Transition, Uniform, Weighted, decode,
};
use ime_eval::{EvalRecord, EvalSet, Report, Slice};
use ime_lm::CharLm;
use ime_ngram::NgramModel;
use ime_pinyin::{CharId, Lexicon, SegmentOptions, SyllableTable};
use rayon::iter::{
    IndexedParallelIterator as _, IntoParallelRefIterator as _, ParallelIterator as _,
};
use serde::Serialize;
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead as _, BufReader, BufWriter, Write};
use std::path::Path;
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
    const fn slice(self) -> Slice {
        match self {
            Self::All => Slice::All,
            Self::Dev => Slice::Dev,
            Self::Test => Slice::Test,
        }
    }

    /// How the slice is spelled in the report.
    const fn label(self) -> &'static str {
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
    /// first. Replaces `--slice`, and meaningless without a score file to fuse.
    #[arg(long, requires = "scores", conflicts_with = "slice")]
    pub select_on_dev: bool,
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

    /// The model for *record*, over *candidates*.
    ///
    /// # Errors
    ///
    /// If nothing scored this record, or the scores do not describe this lattice.
    fn model<'a>(&'a self, record: usize, candidates: &'a Candidates) -> Result<Self::Model<'a>>;
}

/// No emissions at all: the n-gram baseline.
struct NoEmissions;

impl Emissions for NoEmissions {
    type Model<'a> = Uniform;

    fn model<'a>(&'a self, _record: usize, _candidates: &'a Candidates) -> Result<Uniform> {
        Ok(Uniform)
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

/// One hypothesis as the dump records it.
#[derive(Serialize)]
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
}

impl Reader {
    /// Resolve one record's keystrokes.
    fn read(&self, record: &EvalRecord) -> Result<(Vec<ime_pinyin::Segmentation>, Candidates)> {
        read(&record.pinyin, &self.table, &self.segment, &self.lexicon)
            .with_context(|| format!("could not read {:?}", record.pinyin))
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
) -> Result<()> {
    let set = load_set(eval_set)?;
    let emittable = load_emittable(emittable, &lexicon)?;
    let reader = Reader {
        table,
        lexicon,
        segment,
    };
    let file =
        fs::File::create(out).with_context(|| format!("could not create {}", out.display()))?;
    let mut sink = BufWriter::new(file);
    let mut positions = 0usize;
    let mut slots = 0usize;
    for (index, record) in set.records().iter().enumerate() {
        let (segmentations, candidates) = reader.read(record)?;
        let mut paths = Vec::with_capacity(candidates.len());
        for (segmentation, reading) in segmentations.iter().zip(candidates.paths()) {
            let spans: Vec<String> = segmentation
                .segments()
                .iter()
                .map(|segment| record.pinyin[segment.start()..segment.end()].to_owned())
                .collect();
            let admitted: Vec<String> = reading
                .positions()
                .iter()
                .map(|allowed| {
                    emittable
                        .restrict(allowed)
                        .iter()
                        .map(|id| reader.lexicon.character(*id))
                        .collect()
                })
                .collect();
            positions += admitted.len();
            slots += admitted
                .iter()
                .map(|set| set.chars().count())
                .sum::<usize>();
            paths.push(LatticePath {
                spans,
                candidates: admitted,
            });
        }
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

/// A transition model shared rather than owned.
///
/// [`Both`] pairs two transitions by value, but the fused run borrows its
/// models from the caller -- a [`CharLm`] is not `Clone`, its ONNX sessions
/// living one per decoding thread -- so the pair refers to them. Every method
/// of the trait borrows the model, so a shared reference forwards the whole
/// of it.
struct Shared<'a, T: ?Sized> {
    /// The model.
    model: &'a T,
}

impl<T: Transition + ?Sized> Transition for Shared<'_, T> {
    const HISTORY: usize = T::HISTORY;

    type State = T::State;

    fn start(&self, context: Option<&str>) -> Self::State {
        self.model.start(context)
    }

    fn score(&self, state: &Self::State, candidate: CharId) -> f32 {
        self.model.score(state, candidate)
    }

    fn finish(&self, state: &Self::State) -> f32 {
        self.model.finish(state)
    }

    fn advance(&self, steps: &[(&Self::State, CharId)]) -> Vec<Self::State> {
        self.model.advance(steps)
    }
}

/// Everything a run needs to score one transition model over its sections.
struct Run<'a> {
    set: &'a EvalSet,
    reader: &'a Reader,
    scores: Option<&'a HashMap<usize, Vec<Vec<Vec<f32>>>>>,
    emittable: &'a Emittable,
    floor: f32,
    weights: &'a [f32],
    slice: &'a SliceArgs,
    beam: &'a BeamOptions,
    dump: bool,
}

impl Run<'_> {
    /// The sections one transition model produces: without a score file the
    /// transition alone over the requested slice; with one, every weight over
    /// the slice, or the dev sweep and the test section at its winner.
    fn sections<T: Transition + Sync>(
        &self,
        transition: &T,
        label: &'static str,
    ) -> Result<Vec<Section>> {
        let Some(scores) = self.scores else {
            let (report, rows) = measure(
                self.set,
                self.slice.slice,
                self.slice.dev_share,
                self.reader,
                &NoEmissions,
                transition,
                self.beam,
                self.dump,
            )?;
            return Ok(vec![Section {
                emission: "none",
                weight: 0.0,
                transition: label,
                slice: self.slice.slice.label(),
                selected: false,
                report,
                rows,
            }]);
        };
        let measure_at = |which: SliceArg, weight: f32| -> Result<Section> {
            let emissions = NeuralEmissions {
                scores,
                emittable: self.emittable,
                weight,
                floor: self.floor,
            };
            let (report, rows) = measure(
                self.set,
                which,
                self.slice.dev_share,
                self.reader,
                &emissions,
                transition,
                self.beam,
                self.dump,
            )?;
            Ok(Section {
                emission: "neural",
                weight,
                transition: label,
                slice: which.label(),
                selected: false,
                report,
                rows,
            })
        };
        let mut sections = Vec::new();
        if self.slice.select_on_dev {
            for &weight in self.weights {
                sections.push(measure_at(SliceArg::Dev, weight)?);
            }
            let mut winner = 0;
            for (index, section) in sections.iter().enumerate().skip(1) {
                if section.report.top1_hits() > sections[winner].report.top1_hits() {
                    winner = index;
                }
            }
            let mut test = measure_at(SliceArg::Test, sections[winner].weight)?;
            test.selected = true;
            sections.push(test);
        } else {
            for &weight in self.weights {
                sections.push(measure_at(self.slice.slice, weight)?);
            }
        }
        Ok(sections)
    }
}

/// Score an evaluation set with the n-gram, with the neural emissions, or with
/// both fused.
///
/// Without a score file the run is the n-gram baseline; without an n-gram it is
/// the emissions alone; with both, every weight in *weights* is one fused
/// configuration. With `--select-on-dev` the weights are swept on the dev
/// slice instead and the report closes with the test slice at whichever weight
/// scored the highest sentence top-1 there. With *dump* every evaluated
/// section's beam is written there too, one JSON Lines file per section.
///
/// # Errors
///
/// If any input cannot be read, a
/// record cannot be decoded, a score file does not describe the lattice the
/// records segment into, or the dump cannot be written.
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a distinct axis of the ablation the command exists to run"
)]
pub fn fused_eval(
    eval_set: &Path,
    scores: Option<&Path>,
    emittable: &Path,
    floor: f32,
    weights: &[f32],
    slice: &SliceArgs,
    dump: Option<&Path>,
    table: SyllableTable,
    lexicon: Lexicon,
    segment: SegmentOptions,
    beam: &BeamOptions,
    transition: Models<'_>,
) -> Result<String> {
    let set = load_set(eval_set)?;
    let emittable = load_emittable(emittable, &lexicon)?;
    let reader = Reader {
        table,
        lexicon,
        segment,
    };
    if let Some(dir) = dump {
        fs::create_dir_all(dir)
            .with_context(|| format!("could not create the dump directory {}", dir.display()))?;
    }
    let scores = scores.map(load_scores).transpose()?;
    let run = Run {
        set: &set,
        reader: &reader,
        scores: scores.as_ref(),
        emittable: &emittable,
        floor,
        weights,
        slice,
        beam,
        dump: dump.is_some(),
    };
    let sections = match transition {
        Models::None => run.sections(&NoTransition, "none")?,
        Models::Ngram(ngram) => run.sections(ngram, "kn-trigram")?,
        Models::CharLm(lm) => run.sections(lm, "char-lm")?,
        Models::Both {
            ngram,
            lm,
            lm_weight,
        } => run.sections(
            &Both {
                first: Shared { model: ngram },
                first_weight: 1.0,
                second: Shared { model: lm },
                second_weight: lm_weight,
            },
            "kn-trigram+char-lm",
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
    dump: bool,
) -> Result<(Report, Vec<DumpRow>)>
where
    E: Emissions + Sync,
    T: Transition + Sync,
{
    let (report, mut rows) = set
        .records()
        .par_iter()
        .enumerate()
        .filter(|(_, record)| slice.slice().holds(record, dev_share))
        .map(|(index, record)| -> Result<(Report, Vec<DumpRow>)> {
            let (_, candidates) = reader.read(record)?;
            let emission = emissions.model(index, &candidates)?;
            let hypotheses = decode(
                &candidates,
                &emission,
                transition,
                record.context.as_deref(),
                beam,
            )
            .with_context(|| format!("could not decode record {index}"))?;
            let texts: Vec<String> = hypotheses
                .iter()
                .map(|hypothesis| hypothesis.text(&reader.lexicon))
                .collect();
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

/// Write one section's beam to its file in the dump directory: a JSON Lines
/// file named for the section, one record per line in file order.
///
/// # Errors
///
/// If the file cannot be created or written.
fn write_dump(dir: &Path, section: &Section) -> Result<()> {
    let path = dir.join(format!(
        "{}-w{:.3}-{}-{}.jsonl",
        section.emission, section.weight, section.transition, section.slice
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
fn load_set(path: &Path) -> Result<EvalSet> {
    let source = fs::read_to_string(path)
        .with_context(|| format!("could not read the eval set at {}", path.display()))?;
    EvalSet::parse(&source).context("the eval set is malformed")
}

/// Read the characters the model can score.
fn load_emittable(path: &Path, lexicon: &Lexicon) -> Result<Emittable> {
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
        };
        assert!(SliceArg::All.slice().holds(&record, 0.5));
        assert_ne!(
            SliceArg::Dev.slice().holds(&record, 0.5),
            SliceArg::Test.slice().holds(&record, 0.5)
        );
    }
}
