//! The bridge between the neural model and the decoder.
//!
//! Two commands, and between them two files. `emit-lattice` writes what the
//! model is being asked: for every record of an evaluation set, every reading of
//! its keystrokes, and for every position of every reading, the letters typed
//! there and the characters that position admits. The Python side answers with a
//! log probability per candidate, in the same order. `fused-eval` reads both
//! back, decodes with the emissions fused into the same beam Viterbi the
//! baseline uses, and reports what came out.
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
    BeamOptions, Candidates, Emission, Emittable, LatticePath, LatticeRecord, NoTransition,
    ScoreRecord, Scored, Transition, Uniform, Weighted, decode,
};
use ime_eval::{EvalRecord, EvalSet, Report, Slice};
use ime_ngram::NgramModel;
use ime_pinyin::{Lexicon, SegmentOptions, SyllableTable};
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
    report: Report,
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

/// Score an evaluation set with the n-gram, with the neural emissions, or with
/// both fused.
///
/// Without a score file the run is the n-gram baseline; without an n-gram it is
/// the emissions alone; with both, every weight in *weights* is one fused
/// configuration.
///
/// # Errors
///
/// If neither a score file nor an n-gram was given, any input cannot be read, a
/// record cannot be decoded, or a score file does not describe the lattice the
/// records segment into.
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
    table: SyllableTable,
    lexicon: Lexicon,
    segment: SegmentOptions,
    beam: &BeamOptions,
    ngram: Option<&NgramModel>,
) -> Result<String> {
    let set = load_set(eval_set)?;
    let emittable = load_emittable(emittable, &lexicon)?;
    let reader = Reader {
        table,
        lexicon,
        segment,
    };
    let mut sections = Vec::new();
    match (scores, ngram) {
        (None, None) => {
            bail!("a run with neither emissions nor a transition model scores nothing")
        }
        (None, Some(ngram)) => sections.push(Section {
            emission: "none",
            weight: 0.0,
            transition: "kn-trigram",
            slice: slice.slice.label(),
            report: measure(&set, slice, &reader, &NoEmissions, ngram, beam)?,
        }),
        (Some(path), ngram) => {
            let scores = load_scores(path)?;
            for &weight in weights {
                let emissions = NeuralEmissions {
                    scores: &scores,
                    emittable: &emittable,
                    weight,
                    floor,
                };
                let (transition, report) = match ngram {
                    Some(ngram) => (
                        "kn-trigram",
                        measure(&set, slice, &reader, &emissions, ngram, beam)?,
                    ),
                    None => (
                        "none",
                        measure(&set, slice, &reader, &emissions, &NoTransition, beam)?,
                    ),
                };
                sections.push(Section {
                    emission: "neural",
                    weight,
                    transition,
                    slice: slice.slice.label(),
                    report,
                });
            }
        }
    }
    Ablation { sections }
        .render()
        .context("could not render the ablation")
}

/// Decode every record of the chosen slice and fold it into one report.
fn measure<E, T>(
    set: &EvalSet,
    slice: &SliceArgs,
    reader: &Reader,
    emissions: &E,
    transition: &T,
    beam: &BeamOptions,
) -> Result<Report>
where
    E: Emissions,
    T: Transition,
{
    let mut report = Report::new(beam.top_k);
    for (index, record) in set.records().iter().enumerate() {
        if !slice.slice.slice().holds(record, slice.dev_share) {
            continue;
        }
        let (_, candidates) = reader.read(record)?;
        let emission = emissions.model(index, &candidates)?;
        let hypotheses = decode(&candidates, &emission, transition, beam)
            .with_context(|| format!("could not decode record {index}"))?;
        let texts: Vec<String> = hypotheses
            .iter()
            .map(|hypothesis| hypothesis.text(&reader.lexicon))
            .collect();
        report.observe(&record.text, &texts);
    }
    Ok(report)
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

/// Read a gzipped score file into the records it answers.
///
/// Gzipped because it is not small: twenty-one million log probabilities is a
/// couple of hundred megabytes as text and a fifth of that compressed, and the
/// file has to come off a Kaggle kernel before anything can be measured.
fn load_scores(path: &Path) -> Result<HashMap<usize, Vec<Vec<Vec<f32>>>>> {
    let file = fs::File::open(path)
        .with_context(|| format!("could not read the scores at {}", path.display()))?;
    let mut scores = HashMap::new();
    for (index, line) in BufReader::new(MultiGzDecoder::new(file))
        .lines()
        .enumerate()
    {
        let line =
            line.with_context(|| format!("could not read line {} of the scores", index + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let record: ScoreRecord = serde_json::from_str(&line)
            .with_context(|| format!("line {} of the scores is not a record", index + 1))?;
        if scores.insert(record.record, record.paths).is_some() {
            bail!(
                "the score file answers record {} twice, at line {}",
                record.record,
                index + 1
            );
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
