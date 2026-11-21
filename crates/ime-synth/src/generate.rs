//! The run: seeds in, journaled requests out, validated samples on disk.
//!
//! Three properties of the ordering here are deliberate.
//!
//! Requests go out through `buffer_unordered`, so the endpoint sees the asked-for
//! concurrency, and each reply is flushed to `synth-journal.jsonl` the moment it
//! arrives so an interrupted run loses no completed prompt. Once every pending
//! seed has been journaled, replies are read back from the journal in exact seed
//! order before anything is judged: two runs — one uninterrupted and one stopped
//! halfway and resumed — present the same sequence to [`Validator`] and write
//! byte-identical sample and provenance shards.
//!
//! Nothing is written to `samples/` or `provenance/` until every seed has a
//! journal entry and the drop rate over parsed examples has been checked against
//! [`MAX_DROP_RATE`]. Explicit `ABSTAIN` replies on ungrounded or contrastive
//! seeds are tallied separately from both refusals and drops.

use crate::error::{Error, Result};
use crate::journal::{self, Journal, JournalEntry, PromptOutcome};
use crate::llm::{Effort, Reply, Synthesizer};
use crate::provenance::{self, Provenance};
use crate::seed::{self, Seed, SeedOptions};
use crate::source::{CONTRASTIVE, PROVENANCE, SEED_SOURCES, SYNTHETIC, SeedSource};
use crate::summary::{RefusalCount, RunSummary, WrittenBySource};
use crate::validate::{DEFAULT_MAX_PREFIX_REPEATS, Validator};
use futures::StreamExt as _;
use ime_corpus::Normalizer;
use ime_corpus::leak::EvalGuard;
use ime_corpus::source::SAMPLES_PER_SHARD;
use ime_corpus::text::content_id;
use ime_g2p::DataLayout;
use ime_g2p::annotate::Sample;
use ime_g2p::llm::LlmSettings;
use ime_g2p::shards::{ShardWriter, shard_paths};
use ime_pinyin::{Lexicon, SyllableTable};
use std::collections::HashMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::time::Instant;
use tracing::{info, warn};

/// The share of generated examples that may fail validation before a run is
/// treated as broken rather than merely lossy.
pub const MAX_DROP_RATE: f64 = 0.30;

/// What one generation run was asked for.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Options {
    /// Stop after this many seed prompts.
    pub terms_limit: Option<usize>,
    /// How many usage examples each prompt asks for.
    pub per_term: usize,
    /// How many requests are allowed in flight at once.
    pub concurrency: usize,
    /// Reasoning effort sent to the chat endpoint.
    pub effort: Effort,
    /// Value written into `Sample.source` and used as the sample shard prefix.
    pub source: String,
    /// Extra seed classes (ungrounded lexicon entries and contrastive pairs).
    pub seeds: SeedOptions,
    /// Evaluation JSONL files protected by [`EvalGuard`] against exact and 13-Han leaks.
    pub eval_guard: Vec<PathBuf>,
    /// Maximum occurrences of one non-seed 4-Han opening prefix (`0` disables).
    pub max_prefix_repeats: usize,
    /// Re-request prompts whose previous journal entry was `Refused`.
    pub retry_refused: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            terms_limit: None,
            per_term: crate::llm::DEFAULT_PER_TERM,
            concurrency: crate::llm::DEFAULT_CONCURRENCY,
            effort: Effort::High,
            source: SYNTHETIC.to_owned(),
            seeds: SeedOptions::default(),
            eval_guard: Vec::new(),
            max_prefix_repeats: DEFAULT_MAX_PREFIX_REPEATS,
            retry_refused: false,
        }
    }
}

/// Answers one seed prompt with parsed examples, an abstention, or a refusal reason.
pub trait PromptRunner: Sync {
    /// Model name recorded in [`RunSummary`].
    fn model(&self) -> &str;

    /// Number of examples requested per prompt.
    fn per_term(&self) -> usize;

    /// Complete one seed prompt.
    fn complete_seed<'a>(
        &'a self,
        seed: &'a Seed,
    ) -> Pin<Box<dyn Future<Output = std::result::Result<Reply, String>> + Send + 'a>>;
}

impl PromptRunner for Synthesizer {
    fn model(&self) -> &str {
        Self::model(self)
    }

    fn per_term(&self) -> usize {
        Self::per_term(self)
    }

    fn complete_seed<'a>(
        &'a self,
        seed: &'a Seed,
    ) -> Pin<Box<dyn Future<Output = std::result::Result<Reply, String>> + Send + 'a>> {
        Box::pin(self.reply(seed))
    }
}

/// One accepted example, with the seed it came from, before it is written.
struct Generated {
    sample: Sample,
    provenance: Provenance,
    source: SeedSource,
}

/// Delete the shards a previous run of the same stage wrote.
///
/// # Errors
///
/// If the directory cannot be listed, or a stale shard cannot be removed.
pub fn remove_stale(directory: &Path, prefix: &str) -> Result<usize> {
    let paths = shard_paths(directory, prefix)?;
    for path in &paths {
        std::fs::remove_file(path).map_err(|source| Error::Read {
            path: path.clone(),
            source,
        })?;
    }
    Ok(paths.len())
}

/// Generate, validate and write one batch of synthetic training samples over HTTP.
///
/// # Errors
///
/// If a seed source is missing, if no seed survives loading, if the journal cannot
/// be opened or written, if more than [`MAX_DROP_RATE`] of parsed examples fail
/// validation, or if shards cannot be written.
pub async fn generate(
    seed_root: &Path,
    out_root: &Path,
    settings: &LlmSettings,
    options: Options,
) -> Result<RunSummary> {
    let synthesizer = Synthesizer::new(settings, options.per_term).with_effort(options.effort);
    generate_with(seed_root, out_root, &synthesizer, options).await
}

/// Generate, journal, validate and write one batch using `runner`.
///
/// Completed prompt outcomes are appended to `synth-journal.jsonl` under `out_root`
/// as each request finishes; prompts already recorded there are skipped on restart.
///
/// # Errors
///
/// If seeds or `eval_guard` files cannot be loaded, if no seed is usable, if
/// journal I/O fails, if validation drops exceed [`MAX_DROP_RATE`], or if shard
/// writing fails.
pub async fn generate_with<R: PromptRunner>(
    seed_root: &Path,
    out_root: &Path,
    runner: &R,
    options: Options,
) -> Result<RunSummary> {
    let started = Instant::now();
    let normalizer = Normalizer::new()?;
    let lexicon = Lexicon::load(&SyllableTable::load())?;
    let eval_guard = if options.eval_guard.is_empty() {
        None
    } else {
        Some(EvalGuard::load(&options.eval_guard)?)
    };
    let load = seed::load_with(seed_root, &normalizer, &lexicon, &options.seeds)?;
    if load.seeds.is_empty() {
        let loaded = load.counts.iter().map(|count| count.loaded).sum();
        return Err(Error::NoGroundedSeeds { loaded });
    }
    let mut seeds = load.seeds;
    if let Some(limit) = options.terms_limit {
        seeds.truncate(limit);
    }

    let mut journal = Journal::open(out_root)?;
    let ids: Vec<String> = seeds
        .iter()
        .map(|seed| journal::prompt_id(&options.source, seed, runner.per_term()))
        .collect();
    let pending: Vec<usize> = ids
        .iter()
        .enumerate()
        .filter_map(|(index, id)| match journal.get(id) {
            None => Some(index),
            Some(PromptOutcome::Refused { .. }) if options.retry_refused => Some(index),
            Some(_) => None,
        })
        .collect();

    info!(
        terms = seeds.len(),
        resumed = seeds.len() - pending.len(),
        pending = pending.len(),
        per_term = runner.per_term(),
        concurrency = options.concurrency,
        model = runner.model(),
        source = %options.source,
        "generating usage sentences"
    );

    request_into_journal(
        runner,
        &ids,
        &seeds,
        &pending,
        options.concurrency,
        &mut journal,
    )
    .await?;

    let validator = Validator::with_guards(
        &normalizer,
        &lexicon,
        eval_guard.as_ref(),
        options.max_prefix_repeats,
    );
    let batch = accept(&journal, &ids, &seeds, &options.source, validator)?;

    if batch.drops.drop_rate() > MAX_DROP_RATE {
        return Err(Error::TooManyDropped {
            dropped: batch.drops.dropped(),
            examples: batch.drops.considered(),
            percent: batch.drops.drop_rate() * 100.0,
            limit: MAX_DROP_RATE * 100.0,
        });
    }

    let written_by_source = write(out_root, &options.source, batch.generated)?;
    let summary = build_summary(
        options.source,
        runner,
        options.concurrency,
        load.counts,
        &seeds,
        batch.parsed,
        batch.abstained,
        batch.refusals,
        batch.drops,
        &batch.terms_by_source,
        &written_by_source,
        started.elapsed().as_millis(),
    );
    summary.write(out_root)?;
    info!(
        terms = summary.terms,
        parsed = summary.examples_parsed,
        abstained = summary.abstained,
        written = summary.written,
        refusals = summary.refusals,
        dropped = summary.drops.dropped(),
        per_minute = summary.terms_per_minute(),
        "synthetic samples written"
    );
    Ok(summary)
}

#[expect(
    clippy::too_many_arguments,
    reason = "assembles the serialized RunSummary fields from one completed batch"
)]
fn build_summary<R: PromptRunner>(
    source: String,
    runner: &R,
    concurrency: usize,
    seed_counts: Vec<seed::SeedCounts>,
    seeds: &[Seed],
    examples_parsed: usize,
    abstained: usize,
    refusal_reasons: Vec<RefusalCount>,
    drops: crate::validate::DropCounts,
    terms_by_source: &HashMap<&'static str, usize>,
    written_by_source: &HashMap<&'static str, usize>,
    milliseconds: u128,
) -> RunSummary {
    let written = written_by_source.values().sum();
    let mut reported_sources: Vec<SeedSource> = SEED_SOURCES.to_vec();
    if seeds.iter().any(|seed| seed.source == CONTRASTIVE) {
        reported_sources.push(CONTRASTIVE);
    }
    let refusals = refusal_reasons.iter().map(|count| count.terms).sum();
    RunSummary {
        source,
        model: runner.model().to_owned(),
        per_term: runner.per_term(),
        concurrency,
        seeds: seed_counts,
        terms: seeds.len(),
        examples_requested: seeds.len() * runner.per_term(),
        examples_parsed,
        abstained,
        refusals,
        refusal_reasons,
        drops,
        written,
        written_by_source: reported_sources
            .into_iter()
            .map(|seed_source| WrittenBySource {
                source: seed_source.name.to_owned(),
                terms: terms_by_source
                    .get(seed_source.name)
                    .copied()
                    .unwrap_or_default(),
                written: written_by_source
                    .get(seed_source.name)
                    .copied()
                    .unwrap_or_default(),
            })
            .collect(),
        milliseconds,
    }
}

async fn complete_one<'a, R: PromptRunner>(
    runner: &'a R,
    id: &'a str,
    seed: &'a Seed,
) -> (&'a str, &'a str, std::result::Result<Reply, String>) {
    (id, seed.term.as_str(), runner.complete_seed(seed).await)
}

async fn request_into_journal<R: PromptRunner>(
    runner: &R,
    ids: &[String],
    seeds: &[Seed],
    pending: &[usize],
    concurrency: usize,
    journal: &mut Journal,
) -> Result<()> {
    let mut stream = futures::stream::iter(
        pending
            .iter()
            .copied()
            .map(|index| complete_one(runner, &ids[index], &seeds[index])),
    )
    .buffer_unordered(concurrency.max(1));

    while let Some((id, term, reply)) = stream.next().await {
        let entry = JournalEntry {
            id: id.to_owned(),
            term: term.to_owned(),
            outcome: PromptOutcome::from(reply),
        };
        journal.record(&entry)?;
    }
    Ok(())
}

/// What the validation pass made of one run's journaled replies.
struct Batch {
    generated: Vec<Generated>,
    refusals: Vec<RefusalCount>,
    drops: crate::validate::DropCounts,
    parsed: usize,
    abstained: usize,
    terms_by_source: HashMap<&'static str, usize>,
}

fn accept(
    journal: &Journal,
    ids: &[String],
    seeds: &[Seed],
    source_tag: &str,
    mut validator: Validator<'_>,
) -> Result<Batch> {
    let mut generated = Vec::new();
    let mut refused: HashMap<String, usize> = HashMap::new();
    let mut parsed = 0_usize;
    let mut abstained = 0_usize;
    let mut terms_by_source: HashMap<&'static str, usize> = HashMap::new();

    for (seed, id) in seeds.iter().zip(ids) {
        *terms_by_source.entry(seed.source.name).or_default() += 1;
        let outcome = journal.get(id).ok_or_else(|| {
            Error::Invariant(format!(
                "seed {} ({id}) has no entry in the journal after completion",
                seed.term
            ))
        })?;
        let examples = match outcome {
            PromptOutcome::Examples { examples } => examples,
            PromptOutcome::Abstained => {
                abstained += 1;
                continue;
            }
            PromptOutcome::Refused { reason } => {
                warn!(term = seed.term, reason, "term refused");
                *refused.entry(reason.clone()).or_default() += 1;
                continue;
            }
        };
        parsed += examples.len();
        for example in examples {
            let Some(accepted) = validator.judge_seed(seed, example)? else {
                continue;
            };
            let sample_id = content_id(source_tag, &accepted.text, accepted.context.as_deref());
            generated.push(Generated {
                sample: Sample {
                    id: sample_id.clone(),
                    source: source_tag.to_owned(),
                    text: accepted.text,
                    context: accepted.context,
                },
                provenance: Provenance::new(
                    sample_id,
                    seed.term.clone(),
                    seed.source,
                    seed.grounding,
                ),
                source: seed.source,
            });
        }
    }

    let mut refusals: Vec<RefusalCount> = refused
        .into_iter()
        .map(|(reason, terms)| RefusalCount { reason, terms })
        .collect();
    refusals.sort_by(|left, right| {
        right
            .terms
            .cmp(&left.terms)
            .then_with(|| left.reason.cmp(&right.reason))
    });
    Ok(Batch {
        generated,
        refusals,
        drops: validator.counts(),
        parsed,
        abstained,
        terms_by_source,
    })
}

fn write(
    out_root: &Path,
    source_tag: &str,
    generated: Vec<Generated>,
) -> Result<HashMap<&'static str, usize>> {
    let samples_dir = DataLayout::new(out_root).samples();
    let provenance_dir = provenance::directory(out_root);
    let stale =
        remove_stale(&samples_dir, source_tag)? + remove_stale(&provenance_dir, PROVENANCE)?;
    if stale > 0 {
        info!(shards = stale, "removed the previous run's shards");
    }
    let mut samples = ShardWriter::new(&samples_dir, source_tag, SAMPLES_PER_SHARD)?;
    let mut sidecar = ShardWriter::new(&provenance_dir, PROVENANCE, provenance::ROWS_PER_SHARD)?;
    let mut by_source: HashMap<&'static str, usize> = HashMap::new();
    for row in generated {
        *by_source.entry(row.source.name).or_default() += 1;
        samples.write(row.sample)?;
        sidecar.write(row.provenance)?;
    }
    let written = samples.finish()?;
    let sidecar_rows = sidecar.finish()?;
    if written != sidecar_rows {
        return Err(Error::Invariant(format!(
            "{written} samples were written against {sidecar_rows} provenance rows"
        )));
    }
    Ok(by_source)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::Example;
    use std::fs::OpenOptions;
    use std::io::Write as _;
    use std::sync::Mutex;

    struct ScriptedRunner {
        per_term: usize,
        stop_after: Option<usize>,
        calls: Mutex<Vec<String>>,
    }

    impl ScriptedRunner {
        fn new(per_term: usize, stop_after: Option<usize>) -> Self {
            Self {
                per_term,
                stop_after,
                calls: Mutex::new(Vec::new()),
            }
        }

        fn called_terms(&self) -> Vec<String> {
            self.calls.lock().expect("unpoisoned").clone()
        }
    }

    impl PromptRunner for ScriptedRunner {
        fn model(&self) -> &'static str {
            "swe-2-test"
        }

        fn per_term(&self) -> usize {
            self.per_term
        }

        fn complete_seed<'a>(
            &'a self,
            seed: &'a Seed,
        ) -> Pin<Box<dyn Future<Output = std::result::Result<Reply, String>> + Send + 'a>> {
            Box::pin(async move {
                let index = {
                    let mut calls = self.calls.lock().expect("unpoisoned");
                    calls.push(seed.term.clone());
                    calls.len()
                };
                assert!(
                    self.stop_after.is_none_or(|limit| index <= limit),
                    "interrupted test runner stopped after {:?}",
                    self.stop_after
                );
                if seed.term == "冷僻词" {
                    return Ok(Reply::Abstained);
                }
                let examples = (0..self.per_term)
                    .map(|item| Example {
                        context: format!("今天大家都在聊第{}集剧情", item + 1),
                        text: format!("看到第{}幕真的让人觉得{}", item + 1, seed.term),
                    })
                    .collect();
                Ok(Reply::Examples(examples))
            })
        }
    }

    fn write_contrastive_fixture(path: &Path) {
        let rows = [
            serde_json::json!({"term": "公式", "foil": "公事", "explanation": "数学式子"}),
            serde_json::json!({"term": "冷僻词", "foil": "冷辟词", "explanation": "测试弃权"}),
            serde_json::json!({"term": "权利", "foil": "权力", "explanation": "公民权益"}),
            serde_json::json!({"term": "制定", "foil": "制订", "explanation": "正式定下"}),
        ];
        let mut content = String::new();
        for row in rows {
            content.push_str(&row.to_string());
            content.push('\n');
        }
        std::fs::write(path, content).expect("fixture written");
    }

    fn shard_bytes(dir: &Path) -> Vec<(String, Vec<u8>)> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .expect("dir readable")
            .map(|entry| entry.expect("entry").path())
            .collect();
        files.sort();
        files
            .into_iter()
            .map(|file| {
                let name = file
                    .file_name()
                    .expect("name")
                    .to_string_lossy()
                    .into_owned();
                let bytes = std::fs::read(&file).expect("bytes");
                (name, bytes)
            })
            .collect()
    }

    #[tokio::test]
    async fn resuming_from_an_interrupted_journal_writes_byte_identical_shards() {
        let root =
            std::env::temp_dir().join(format!("ime-synth-resume-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("writable");
        let seeds_file = root.join("contrastive.jsonl");
        write_contrastive_fixture(&seeds_file);

        let options = Options {
            per_term: 2,
            concurrency: 1,
            source: "synthetic-swe2".to_owned(),
            seeds: SeedOptions {
                allow_ungrounded: false,
                contrastive_seeds: Some(seeds_file),
            },
            max_prefix_repeats: 0,
            ..Options::default()
        };

        // 1. Uninterrupted reference run.
        let whole_dir = root.join("whole");
        let whole_runner = ScriptedRunner::new(2, None);
        let whole_summary = generate_with(&root, &whole_dir, &whole_runner, options.clone())
            .await
            .expect("whole run succeeds");
        assert_eq!(whole_summary.abstained, 1);
        assert_eq!(whole_summary.written, 6);
        assert_eq!(whole_runner.called_terms().len(), 4);

        // 2. Interrupted run: panics on the 3rd seed after journaling the first 2.
        let resumed_dir = root.join("resumed");
        let first_half = ScriptedRunner::new(2, Some(2));
        let interrupted = tokio::spawn({
            let root = root.clone();
            let resumed_dir = resumed_dir.clone();
            let options = options.clone();
            async move { generate_with(&root, &resumed_dir, &first_half, options).await }
        })
        .await;
        assert!(
            interrupted.is_err(),
            "simulated crash after two journaled seeds"
        );
        let journal = Journal::open(&resumed_dir).expect("journal opens cleanly");
        assert_eq!(journal.len(), 2);

        // Append an incomplete trailing fragment (no newline) to simulate a mid-write crash.
        let mut raw = OpenOptions::new()
            .append(true)
            .open(journal::path(&resumed_dir))
            .expect("journal exists");
        raw.write_all(b"{\"id\":\"truncated")
            .expect("fragment written");
        drop(raw);

        // 3. Resume on the same out_root: only the remaining 2 seeds are requested!
        let second_half = ScriptedRunner::new(2, None);
        let resumed_summary = generate_with(&root, &resumed_dir, &second_half, options)
            .await
            .expect("resumed run finishes");
        assert_eq!(second_half.called_terms(), vec!["权利", "制定"]);
        assert_eq!(resumed_summary.abstained, whole_summary.abstained);
        assert_eq!(resumed_summary.written, whole_summary.written);
        assert_eq!(resumed_summary.drops, whole_summary.drops);

        // 4. Sample shards and provenance shards are byte-for-byte identical!
        assert_eq!(
            shard_bytes(&DataLayout::new(&resumed_dir).samples()),
            shard_bytes(&DataLayout::new(&whole_dir).samples()),
        );
        assert_eq!(
            shard_bytes(&provenance::directory(&resumed_dir)),
            shard_bytes(&provenance::directory(&whole_dir)),
        );
        assert_eq!(
            crate::report::shown(&resumed_dir).expect("report reads custom source"),
            crate::report::shown(&whole_dir).expect("report reads custom source"),
        );

        std::fs::remove_dir_all(&root).expect("cleaned up");
    }
}
