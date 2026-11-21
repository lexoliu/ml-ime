//! `ime-cli replay-keys`: an evaluation slice through the key engine, one
//! keystroke at a time, measuring what each keystroke costs.
//!
//! The harness is `replay`'s: a fresh [`Session`] per record, each `key` call
//! timed, the slice cut by the same hash, the latency summarised by the same
//! code, so the two engines' reports over the same records compare field for
//! field. The engine is [`KeyEngine`] over the converted Core ML model. The
//! model is opened and warmed before the first timed keystroke; both spans are
//! in the report.
//!
//! The keystrokes are the record's letters, as the Python evaluation types
//! them: a `'` of the eval set is not pressed. A record with more letters than
//! the model's largest keys function holds is not replayed and is counted
//! apart.
//!
//! `--reference` names the dump `mlime eval keys --coreml` wrote for the same
//! eval set, slice, context setting and weights, and the report then says how
//! many records' last keystroke ranked the dump's top 1 first, and which did
//! not.

use crate::neural::{SliceArgs, load_set};
use crate::replay::{
    LatencyReport, peak_rss, phys_footprint, select_records, start_record, timed_key,
};
use anyhow::{Context, Result, bail, ensure};
use clap::{Args, ValueEnum};
use flate2::read::MultiGzDecoder;
use ime_decode::{Combine, DEFAULT_ARC_WIDTH, KeySearchOptions, Weight};
use ime_neural::keys::{ComputeUnits, CoreMlKeyModel, KeyModel, TREE_BUCKETS};
use ime_pinyin::{Legality, LegalityOptions};
use ime_session::{KeyEngine, KeyEngineOptions, KeystrokeStats, Session, SessionOptions};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing::info;

/// Which hardware Core ML runs the functions on.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, ValueEnum)]
enum UnitsArg {
    /// The Neural Engine, with the CPU for what it does not run.
    #[default]
    CpuAndNeuralEngine,
    /// The CPU alone.
    CpuOnly,
    /// The GPU, with the CPU for what it does not run.
    CpuAndGpu,
    /// Whatever Core ML picks.
    All,
}

impl UnitsArg {
    const fn units(self) -> ComputeUnits {
        match self {
            Self::CpuAndNeuralEngine => ComputeUnits::CpuAndNeuralEngine,
            Self::CpuOnly => ComputeUnits::CpuOnly,
            Self::CpuAndGpu => ComputeUnits::CpuAndGpu,
            Self::All => ComputeUnits::All,
        }
    }
}

/// How a prefix's covers combine, spelled as `mlime` writes it in a
/// selection.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, ValueEnum, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
enum CombineArg {
    /// Log-sum-exp over the covers.
    #[default]
    Sum,
    /// The best cover alone.
    Viterbi,
}

impl CombineArg {
    const fn combine(self) -> Combine {
        match self {
            Self::Sum => Combine::Sum,
            Self::Viterbi => Combine::Viterbi,
        }
    }
}

/// A search weight from the command line.
fn parse_weight(raw: &str) -> Result<Weight, String> {
    let value: f64 = raw
        .parse()
        .map_err(|_| format!("{raw:?} is not a number"))?;
    Weight::try_from(value).map_err(|error| error.to_string())
}

/// The `replay-keys` command's arguments.
#[derive(Debug, Clone, Args)]
pub(crate) struct ReplayKeysArgs {
    /// The unpacked Core ML model: the directory `mlime-coreml convert`
    /// leaves, `key-model.json` beside the function directories.
    #[arg(long)]
    model: PathBuf,
    /// The compute units the functions are compiled for and run on.
    #[arg(long, value_enum, default_value_t)]
    units: UnitsArg,
    /// The JSON Lines evaluation set to replay.
    #[arg(long)]
    eval_set: PathBuf,
    /// Score the records without their contexts.
    #[arg(long)]
    no_context: bool,
    /// Cap the records replayed.
    #[arg(long)]
    records: Option<usize>,
    /// Where the JSON report is written.
    #[arg(long)]
    out: PathBuf,
    /// The transition's weight.
    #[arg(long, value_parser = parse_weight, default_value = "1.0")]
    beta: Weight,
    /// The correction cost's weight.
    #[arg(long, value_parser = parse_weight, default_value = "1.0")]
    gamma: Weight,
    /// How a prefix's covers combine.
    #[arg(long, value_enum, default_value_t)]
    combine: CombineArg,
    /// The `selection.json` a `mlime eval keys --select-on-dev` run wrote:
    /// the weights it chose replace `--beta`, `--gamma` and `--combine`.
    #[arg(long, conflicts_with_all = ["beta", "gamma", "combine"])]
    selection: Option<PathBuf>,
    /// Prefixes the search keeps after each key.
    #[arg(long, default_value = "8")]
    beam: NonZeroUsize,
    /// Hypotheses the search returns.
    #[arg(long, default_value = "8")]
    top_k: NonZeroUsize,
    /// Read the last syllable as possibly unfinished, as the live input
    /// method does. Off by default, as the Python evaluation decodes.
    #[arg(long)]
    unfinished_tail: bool,
    /// Let a span that is itself a syllable be read as a correction. The
    /// model's manifest says which it was trained for; a value here
    /// overrides it.
    #[arg(long)]
    correct_syllables: Option<bool>,
    /// A dump `mlime eval keys --coreml` wrote (plain or gzipped JSON Lines)
    /// for this eval set, slice, context setting and weights: the gate the
    /// last keystrokes' top 1 are held to.
    #[arg(long)]
    reference: Option<PathBuf>,
    #[command(flatten)]
    slice: SliceArgs,
}

/// The `chosen` weights of a `selection.json`.
#[derive(Deserialize)]
struct Selection {
    chosen: Chosen,
}

/// One point of the weight grid, as `mlime` writes it.
#[derive(Deserialize)]
struct Chosen {
    beta: f64,
    gamma: f64,
    combination: CombineArg,
    #[serde(default)]
    prior: f64,
}

/// The search weights a run decodes with.
#[derive(Copy, Clone, Debug, PartialEq)]
struct Weights {
    beta: Weight,
    gamma: Weight,
    combine: CombineArg,
}

impl Weights {
    /// The weights the flags name, or the selection's choice.
    fn resolve(args: &ReplayKeysArgs) -> Result<Self> {
        let Some(path) = &args.selection else {
            return Ok(Self {
                beta: args.beta,
                gamma: args.gamma,
                combine: args.combine,
            });
        };
        let source = std::fs::read_to_string(path)
            .with_context(|| format!("could not read the selection at {}", path.display()))?;
        let Selection { chosen } = serde_json::from_str(&source)
            .with_context(|| format!("{} is not a selection", path.display()))?;
        ensure!(
            chosen.prior == 0.0,
            "the selection chose arc prior {}; the key engine has no word arcs",
            chosen.prior
        );
        Ok(Self {
            beta: Weight::try_from(chosen.beta).context("the selection's beta")?,
            gamma: Weight::try_from(chosen.gamma).context("the selection's gamma")?,
            combine: chosen.combination,
        })
    }
}

/// A reference dump's row: the record and its ranked hypotheses' texts.
#[derive(Deserialize)]
struct DumpRow {
    record: usize,
    hypotheses: Vec<DumpHypothesis>,
}

/// A hypothesis of a reference dump; its terms are not read.
#[derive(Deserialize)]
struct DumpHypothesis {
    text: String,
}

/// The top 1 of every record a dump holds, by eval-set index; `None` where
/// the search returned nothing.
fn read_dump(path: &Path) -> Result<HashMap<usize, Option<String>>> {
    let file = File::open(path)
        .with_context(|| format!("could not open the reference dump {}", path.display()))?;
    let source: Box<dyn Read> = if path.extension().is_some_and(|ext| ext == "gz") {
        Box::new(MultiGzDecoder::new(file))
    } else {
        Box::new(file)
    };
    let mut tops = HashMap::new();
    for (number, line) in BufReader::new(source).lines().enumerate() {
        let line = line.with_context(|| format!("could not read {}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let row: DumpRow = serde_json::from_str(&line)
            .with_context(|| format!("{} line {} is not a dump row", path.display(), number + 1))?;
        let top = row.hypotheses.into_iter().next().map(|first| first.text);
        ensure!(
            tops.insert(row.record, top).is_none(),
            "{} holds record {} twice",
            path.display(),
            row.record
        );
    }
    Ok(tops)
}

/// One keystroke's cost and where it went, in the JSON report.
#[derive(Serialize)]
struct Keystroke {
    /// The record's index in the eval set.
    record: usize,
    /// How many keystrokes are pending after this one.
    typed: usize,
    /// The whole `key` call.
    key_ns: u64,
    /// Of it: the keys function (and the context encoder, when the context
    /// changed).
    keys_call_ns: u64,
    /// Tree calls the keystroke made.
    tree_calls: u64,
    /// Of them, the speculative pass's.
    speculative_calls: usize,
    /// Of them, the misses'.
    miss_calls: usize,
    /// Nanos in the speculative pass's tree calls.
    speculative_ns: u64,
    /// Nanos in the misses' tree calls.
    miss_ns: u64,
    /// Nanos in the search, the misses' tree calls included.
    search_ns: u64,
    /// The search without the model's tree calls: `search_ns - miss_ns`.
    host_search_ns: u64,
    /// Distinct prefixes the search asked.
    queried: usize,
    /// Of them, the ones a speculative call had already computed.
    speculated_queried: usize,
}

impl Keystroke {
    fn new(record: usize, typed: usize, key_ns: u64, stats: &KeystrokeStats) -> Self {
        Self {
            record,
            typed,
            key_ns,
            keys_call_ns: stats.encode_ns,
            tree_calls: stats.calls.iter().sum(),
            speculative_calls: stats.speculative_calls,
            miss_calls: stats.miss_calls,
            speculative_ns: stats.speculative_ns,
            miss_ns: stats.miss_ns,
            search_ns: stats.search_ns,
            host_search_ns: stats.search_ns.saturating_sub(stats.miss_ns),
            queried: stats.queried,
            speculated_queried: stats.speculated_queried,
        }
    }
}

/// The keystrokes' stage totals.
#[derive(Default, Serialize)]
struct Totals {
    keystrokes: usize,
    keys_call_ns: u64,
    /// Tree calls by bucket, in `buckets` order.
    tree_calls_by_bucket: [u64; TREE_BUCKETS.len()],
    tree_calls: u64,
    speculative_calls: u64,
    miss_calls: u64,
    speculative_ns: u64,
    miss_ns: u64,
    search_ns: u64,
    host_search_ns: u64,
    queried_rows: u64,
    /// Of them, those a speculative call had already computed.
    speculated_rows: u64,
    /// `speculated_rows / queried_rows`; `0` when nothing was queried.
    speculated_share: f64,
}

impl Totals {
    fn add(&mut self, sample: &Keystroke, stats: &KeystrokeStats) {
        self.keystrokes += 1;
        self.keys_call_ns += sample.keys_call_ns;
        for (total, calls) in self.tree_calls_by_bucket.iter_mut().zip(stats.calls) {
            *total += calls;
        }
        self.tree_calls += sample.tree_calls;
        self.speculative_calls += u64::try_from(sample.speculative_calls).unwrap_or(u64::MAX);
        self.miss_calls += u64::try_from(sample.miss_calls).unwrap_or(u64::MAX);
        self.speculative_ns += sample.speculative_ns;
        self.miss_ns += sample.miss_ns;
        self.search_ns += sample.search_ns;
        self.host_search_ns += sample.host_search_ns;
        self.queried_rows += u64::try_from(sample.queried).unwrap_or(u64::MAX);
        self.speculated_rows += u64::try_from(sample.speculated_queried).unwrap_or(u64::MAX);
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "row counts never approach 2^53; the f64 mantissa holds them"
    )]
    fn finish(&mut self) {
        self.speculated_share = if self.queried_rows == 0 {
            0.0
        } else {
            self.speculated_rows as f64 / self.queried_rows as f64
        };
    }
}

/// A record whose last keystroke's top 1 is not the reference dump's.
#[derive(Debug, Serialize)]
struct Mismatch {
    /// The record's index in the eval set.
    record: usize,
    /// Its keys.
    keys: String,
    /// The dump's top 1.
    expected: Option<String>,
    /// The replay's.
    got: Option<String>,
}

/// The equality gate against a reference dump.
#[derive(Debug, Serialize)]
struct Gate {
    /// The dump held to.
    reference: String,
    /// Records replayed and found in the dump.
    compared: usize,
    /// Of them, those whose top 1 equals the dump's.
    equal: usize,
    /// The ones whose top 1 differs.
    mismatches: Vec<Mismatch>,
    /// Replayed records the dump has no row for.
    not_in_dump: Vec<usize>,
}

impl Gate {
    /// A gate against the dump at *reference*, nothing compared yet.
    fn new(reference: &Path) -> Self {
        Self {
            reference: reference.display().to_string(),
            compared: 0,
            equal: 0,
            mismatches: Vec::new(),
            not_in_dump: Vec::new(),
        }
    }
}

impl Gate {
    /// Hold record *index*, typed as *keys* and ending at *got*, to the dump's
    /// row for it, *expected*, which is `None` where the dump has no row.
    fn check(
        &mut self,
        index: usize,
        keys: String,
        expected: Option<&Option<String>>,
        got: Option<String>,
    ) {
        let Some(expected) = expected else {
            self.not_in_dump.push(index);
            return;
        };
        self.compared += 1;
        if *expected == got {
            self.equal += 1;
        } else {
            self.mismatches.push(Mismatch {
                record: index,
                keys,
                expected: expected.clone(),
                got,
            });
        }
    }
}

impl ConfigEcho {
    /// The run's shape.
    fn new(
        args: &ReplayKeysArgs,
        weights: Weights,
        correct_syllables: bool,
        step: u64,
        max_keys: usize,
    ) -> Self {
        Self {
            eval_set: args.eval_set.display().to_string(),
            slice: args.slice.slice.label().to_owned(),
            dev_share: args.slice.dev_share,
            records_cap: args.records,
            model: args.model.display().to_string(),
            step,
            units: format!("{:?}", args.units.units()),
            with_context: !args.no_context,
            beta: weights.beta.get(),
            gamma: weights.gamma.get(),
            combine: weights.combine,
            beam: args.beam.get(),
            top_k: args.top_k.get(),
            unfinished_tail: args.unfinished_tail,
            correct_syllables,
            max_keys,
        }
    }
}

impl ReplayKeysArgs {
    /// The key engine's options: the legality and search the flags name.
    fn engine_options(
        &self,
        weights: Weights,
        correct_syllables: bool,
        score_eos: bool,
    ) -> KeyEngineOptions {
        KeyEngineOptions {
            legality: LegalityOptions {
                unfinished_tail: self.unfinished_tail,
                correct_syllables,
            },
            search: KeySearchOptions {
                beta: weights.beta,
                gamma: weights.gamma,
                beam: Some(self.beam),
                top_k: Some(self.top_k),
                combine: weights.combine.combine(),
                prior: Weight::ZERO,
                arc_width: DEFAULT_ARC_WIDTH,
                score_eos,
            },
        }
    }
}

/// The run's shape, echoed into the report.
#[derive(Serialize)]
struct ConfigEcho {
    eval_set: String,
    slice: String,
    dev_share: f64,
    records_cap: Option<usize>,
    model: String,
    /// The training step the bundle was written at.
    step: u64,
    units: String,
    with_context: bool,
    beta: f64,
    gamma: f64,
    combine: CombineArg,
    beam: usize,
    top_k: usize,
    unfinished_tail: bool,
    correct_syllables: bool,
    /// The most keys one keystroke may hold.
    max_keys: usize,
}

/// The JSON report `replay-keys` writes.
#[derive(Serialize)]
struct Report {
    config: ConfigEcho,
    /// Records replayed.
    records: usize,
    /// Records with more keys than the model holds: not replayed.
    records_too_long: usize,
    /// Seconds `CoreMlKeyModel::open` took.
    open_seconds: f64,
    /// Seconds the warm-up took.
    warm_seconds: f64,
    /// Latency summaries, overall and per typed-length band: the fused
    /// replay's own fields.
    latency: LatencyReport,
    /// Where the time went, cumulative.
    stages: Totals,
    /// The equality gate; `None` without `--reference`.
    gate: Option<Gate>,
    /// The process's peak resident set, bytes.
    peak_rss_bytes: i64,
    /// Its physical footprint, bytes.
    phys_footprint_bytes: i64,
    /// Every keystroke's timings, in order.
    samples: Vec<Keystroke>,
}

/// The seconds since *tick*.
fn seconds(tick: Instant) -> f64 {
    tick.elapsed().as_secs_f64()
}

/// Replay the slice and build the report.
fn replay(args: &ReplayKeysArgs) -> Result<Report> {
    ensure!(
        !args.slice.select_on_dev,
        "--select-on-dev sweeps weights; the key search's weights are --beta, --gamma and \
         --combine, or --selection"
    );
    let weights = Weights::resolve(args)?;
    let set = load_set(&args.eval_set)?;
    let reference = args.reference.as_deref().map(read_dump).transpose()?;

    let tick = Instant::now();
    let mut model = CoreMlKeyModel::open(&args.model, args.units.units()).with_context(|| {
        format!(
            "could not open the Core ML model at {}",
            args.model.display()
        )
    })?;
    let open_seconds = seconds(tick);
    let tick = Instant::now();
    model.warm_up().context("could not warm the model up")?;
    let warm_seconds = seconds(tick);

    let legality =
        Legality::over(model.characters()).context("the model's characters have no readings")?;
    let correct_syllables = args.correct_syllables.unwrap_or(model.correct_syllables());
    let score_eos = model.score_eos();
    let max_keys = model.max_keys();
    let step = model.step();
    let mut session = Session::new(
        KeyEngine::new(
            model,
            &legality,
            args.engine_options(weights, correct_syllables, score_eos),
        ),
        SessionOptions { sentences: 3 },
    );

    let records = select_records(&set, &args.slice, args.records);
    info!(
        records = records.len(),
        slice = args.slice.slice.label(),
        open_seconds,
        warm_seconds,
        "replaying"
    );
    let mut samples = Vec::new();
    let mut stages = Totals::default();
    let mut gate = args.reference.as_deref().map(Gate::new);
    let mut replayed = 0;
    let mut too_long = 0;
    for &(index, record) in &records {
        let keys: String = record
            .pinyin
            .chars()
            .filter(char::is_ascii_lowercase)
            .collect();
        ensure!(
            !keys.is_empty(),
            "record {index} ({:?}) has no letter to press",
            record.pinyin
        );
        if keys.len() > max_keys {
            too_long += 1;
            continue;
        }
        replayed += 1;
        let context = if args.no_context {
            None
        } else {
            record.context.as_deref()
        };
        start_record(&mut session, index, context)?;
        for ch in keys.chars() {
            let key_ns = timed_key(&mut session, index, ch)?;
            let stats = session.engine().last_keystroke();
            let sample = Keystroke::new(index, session.pending(), key_ns, stats);
            stages.add(&sample, stats);
            samples.push(sample);
        }
        if let (Some(gate), Some(tops)) = (gate.as_mut(), reference.as_ref()) {
            let got = session.candidates().first().map(|top| top.text.clone());
            gate.check(index, keys, tops.get(&index), got);
        }
    }
    stages.finish();
    Ok(Report {
        config: ConfigEcho::new(args, weights, correct_syllables, step, max_keys),
        records: replayed,
        records_too_long: too_long,
        open_seconds,
        warm_seconds,
        latency: LatencyReport::of(samples.iter().map(|sample| (sample.typed, sample.key_ns))),
        stages,
        gate,
        peak_rss_bytes: peak_rss(),
        phys_footprint_bytes: phys_footprint(),
        samples,
    })
}

/// The `replay-keys` command.
///
/// # Errors
///
/// If the model, the eval set or the reference dump do not load, a keystroke
/// fails, or the report cannot be written.
pub(crate) fn run(args: &ReplayKeysArgs) -> Result<()> {
    let report = replay(args)?;
    info!(
        records = report.records,
        records_too_long = report.records_too_long,
        count = report.latency.all.count,
        median_ms = report.latency.all.median_ms,
        p95_ms = report.latency.all.p95_ms,
        max_ms = report.latency.all.max_ms,
        tree_calls = report.stages.tree_calls,
        speculated_share = report.stages.speculated_share,
        peak_rss_mb = report.peak_rss_bytes / 1_000_000,
        phys_footprint_mb = report.phys_footprint_bytes / 1_000_000,
        "replay finished"
    );
    if let Some(gate) = &report.gate {
        info!(
            compared = gate.compared,
            equal = gate.equal,
            mismatches = gate.mismatches.len(),
            not_in_dump = gate.not_in_dump.len(),
            "gate against the reference dump"
        );
    }
    let file = File::create(&args.out)
        .with_context(|| format!("could not create {}", args.out.display()))?;
    serde_json::to_writer_pretty(std::io::BufWriter::new(file), &report)
        .context("could not write the replay report")?;
    if report
        .gate
        .as_ref()
        .is_some_and(|gate| !gate.mismatches.is_empty())
    {
        bail!(
            "{} records' top 1 differ from the reference dump; see the report's gate",
            report.gate.as_ref().map_or(0, |gate| gate.mismatches.len())
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use std::io::Write;

    /// The command's arguments as the binary parses them.
    #[derive(Parser)]
    struct Wrapper {
        #[command(flatten)]
        args: ReplayKeysArgs,
    }

    #[derive(Deserialize)]
    struct Expected {
        records: Vec<ExpectedRecord>,
    }

    #[derive(Deserialize)]
    struct ExpectedRecord {
        keys: String,
        context: Option<String>,
        searches: Vec<ExpectedSearch>,
    }

    #[derive(Deserialize)]
    struct ExpectedSearch {
        weights: usize,
        hypotheses: Vec<DumpHypothesis>,
    }

    /// A scratch directory under the system's, removed when dropped.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!("{name}-{}", std::process::id()));
            std::fs::create_dir_all(&path).expect("the scratch directory is created");
            Self(path)
        }

        fn file(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).ok();
        }
    }

    /// An eval set of the fixture's own keys, in every style, with Chinese,
    /// Japanese and Korean contexts and one with none, and one record past
    /// the model's keys; and the dump the Python search's top 1s make of it
    /// (`expected.json`, the weights it was written at).
    struct Case {
        set: PathBuf,
        dump: Vec<(usize, String)>,
        replayable: usize,
    }

    fn case(scratch: &Scratch) -> Case {
        let expected: Expected = serde_json::from_str(
            &std::fs::read_to_string(ime_fixtures::keys().join("expected.json"))
                .expect("expected.json reads"),
        )
        .expect("expected.json parses");
        let mut lines = Vec::new();
        let mut dump = Vec::new();
        for record in &expected.records {
            let Some(search) = record.searches.iter().find(|search| search.weights == 0) else {
                continue;
            };
            dump.push((lines.len(), search.hypotheses[0].text.clone()));
            lines.push(serde_json::json!({
                "pinyin": record.keys,
                "text": search.hypotheses[0].text,
                "context": record.context,
            }));
        }
        let replayable = lines.len();
        assert!(replayable >= 5, "the fixture covers every style");
        lines.push(serde_json::json!({
            "pinyin": "a".repeat(127),
            "text": "啊",
            "context": "今日は東京に行きます",
        }));
        let set = scratch.file("eval.jsonl");
        std::fs::write(
            &set,
            lines
                .iter()
                .map(|line| line.to_string() + "\n")
                .collect::<String>(),
        )
        .expect("the eval set writes");
        Case {
            set,
            dump,
            replayable,
        }
    }

    fn write_dump(path: &Path, rows: &[(usize, String)]) {
        let text: String = rows
            .iter()
            .map(|(record, top)| {
                serde_json::json!({
                    "record": record,
                    "text": top,
                    "hypotheses": [{"text": top, "score": -1.0}],
                })
                .to_string()
                    + "\n"
            })
            .collect();
        if path.extension().is_some_and(|ext| ext == "gz") {
            let mut sink = flate2::write::GzEncoder::new(
                File::create(path).expect("the dump file is created"),
                flate2::Compression::fast(),
            );
            sink.write_all(text.as_bytes()).expect("the dump writes");
            sink.finish().expect("the dump finishes");
        } else {
            std::fs::write(path, text).expect("the dump writes");
        }
    }

    fn parse(scratch: &Scratch, set: &Path, extra: &[&str]) -> ReplayKeysArgs {
        let model = ime_fixtures::keys().join("model");
        let mut argv = vec![
            "replay-keys",
            "--model",
            model.to_str().expect("a UTF-8 path"),
            "--units",
            "cpu-only",
            "--eval-set",
            set.to_str().expect("a UTF-8 path"),
            "--out",
        ];
        let out = scratch.file("report.json");
        argv.push(out.to_str().expect("a UTF-8 path"));
        argv.extend_from_slice(extra);
        Wrapper::try_parse_from(argv)
            .expect("the arguments parse")
            .args
    }

    #[test]
    fn a_slice_replays_end_to_end_and_the_counts_add_up() {
        let scratch = Scratch::new("replay-keys-counts");
        let case = case(&scratch);
        let args = parse(&scratch, &case.set, &[]);
        run(&args).expect("the replay runs");
        let report: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&args.out).expect("the report reads"))
                .expect("the report is JSON");
        assert_eq!(report["records"], case.replayable);
        assert_eq!(report["records_too_long"], 1);
        assert!(report["gate"].is_null());
        let samples = report["samples"].as_array().expect("samples");
        let typed: usize = case
            .dump
            .iter()
            .map(|(index, _)| {
                let count = samples
                    .iter()
                    .filter(|sample| sample["record"] == *index)
                    .count();
                assert!(count > 0, "record {index} was typed");
                count
            })
            .sum();
        assert_eq!(typed, samples.len(), "no keystroke of a skipped record");
        assert_eq!(report["latency"]["all"]["count"], samples.len());
        assert_eq!(report["stages"]["keystrokes"], samples.len());
        let calls: u64 = samples
            .iter()
            .map(|sample| sample["tree_calls"].as_u64().expect("calls"))
            .sum();
        assert_eq!(report["stages"]["tree_calls"], calls);
        let split: u64 = samples
            .iter()
            .map(|sample| {
                sample["speculative_calls"].as_u64().expect("calls")
                    + sample["miss_calls"].as_u64().expect("calls")
            })
            .sum();
        assert_eq!(split, calls, "every call is speculative or a miss");
        let bucketed: u64 = report["stages"]["tree_calls_by_bucket"]
            .as_array()
            .expect("buckets")
            .iter()
            .map(|calls| calls.as_u64().expect("calls"))
            .sum();
        assert_eq!(bucketed, calls);
        let queried: u64 = samples
            .iter()
            .map(|sample| sample["queried"].as_u64().expect("rows"))
            .sum();
        assert_eq!(report["stages"]["queried_rows"], queried);
        assert!(
            report["stages"]["speculated_rows"].as_u64().expect("rows") <= queried,
            "the speculation cannot have cached more rows than were queried"
        );
        assert!(report["open_seconds"].as_f64().expect("seconds") > 0.0);
        assert!(report["warm_seconds"].as_f64().expect("seconds") > 0.0);
    }

    #[test]
    fn the_gate_against_the_python_top_ones_has_no_mismatch() {
        let scratch = Scratch::new("replay-keys-gate");
        let case = case(&scratch);
        let dump = scratch.file("dev-beta1-gamma1-sum.jsonl");
        write_dump(&dump, &case.dump);
        let args = parse(
            &scratch,
            &case.set,
            &["--reference", dump.to_str().expect("a UTF-8 path")],
        );
        run(&args).expect("a replay equal to the dump passes");
        let report: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&args.out).expect("the report reads"))
                .expect("the report is JSON");
        assert_eq!(report["gate"]["compared"], case.replayable);
        assert_eq!(report["gate"]["equal"], case.replayable);
        assert_eq!(report["gate"]["mismatches"], serde_json::json!([]));
        assert_eq!(report["gate"]["not_in_dump"], serde_json::json!([]));
    }

    #[test]
    fn a_gzipped_dump_that_differs_lists_the_record_with_both_texts() {
        let scratch = Scratch::new("replay-keys-mismatch");
        let case = case(&scratch);
        let mut rows = case.dump.clone();
        let wanted = rows[1].1.clone();
        rows[1].1 = "不是".to_owned();
        rows.pop();
        let dump = scratch.file("dev-beta1-gamma1-sum.jsonl.gz");
        write_dump(&dump, &rows);
        let args = parse(
            &scratch,
            &case.set,
            &["--reference", dump.to_str().expect("a UTF-8 path")],
        );
        let error = run(&args).expect_err("a differing top 1 fails the gate");
        assert!(error.to_string().contains("1 records"), "{error}");
        let report: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&args.out).expect("the report reads"))
                .expect("the report is JSON");
        let gate = &report["gate"];
        assert_eq!(gate["compared"], case.replayable - 1);
        assert_eq!(gate["equal"], case.replayable - 2);
        assert_eq!(gate["mismatches"][0]["record"], 1);
        assert_eq!(gate["mismatches"][0]["expected"], "不是");
        assert_eq!(gate["mismatches"][0]["got"], wanted);
        assert_eq!(
            gate["not_in_dump"],
            serde_json::json!([case.replayable - 1])
        );
    }

    #[test]
    fn a_selection_replaces_the_flag_weights() {
        let scratch = Scratch::new("replay-keys-selection");
        let selection = scratch.file("selection.json");
        std::fs::write(
            &selection,
            r#"{"grid": [], "chosen": {"beta": 0.7, "gamma": 1.3, "combination": "viterbi", "prior": 0.0}}"#,
        )
        .expect("the selection writes");
        let set = scratch.file("eval.jsonl");
        let args = parse(
            &scratch,
            &set,
            &["--selection", selection.to_str().expect("a UTF-8 path")],
        );
        let weights = Weights::resolve(&args).expect("the selection reads");
        assert_eq!(weights.beta.get(), 0.7);
        assert_eq!(weights.gamma.get(), 1.3);
        assert_eq!(weights.combine, CombineArg::Viterbi);
        assert!(
            Wrapper::try_parse_from([
                "replay-keys",
                "--model",
                "m",
                "--eval-set",
                "e",
                "--out",
                "o",
                "--selection",
                "s",
                "--beta",
                "2",
            ])
            .is_err(),
            "a selection and a flag weight name the weights twice"
        );
    }
}
