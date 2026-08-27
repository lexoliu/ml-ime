//! The run: seeds in, grounded requests out, validated samples on disk.
//!
//! Three properties of the ordering here are deliberate.
//!
//! Requests go out through `buffer_unordered`, so the endpoint sees exactly the
//! asked-for concurrency, but the replies are put back into seed order before
//! anything is judged. Otherwise the duplicate rule -- and therefore which of two
//! identical sentences survives -- would depend on network timing, and two runs
//! of the same seed list would disagree.
//!
//! Nothing is written until every reply is in and the drop rate has been checked.
//! A prompt that has stopped working, an endpoint that has started refusing, a
//! seed list full of terms that cannot appear in Chinese sentences: all three
//! show up as a collapse in the validation pass rate, and all three would
//! otherwise land as a quietly thinner corpus that nobody notices. Above
//! [`MAX_DROP_RATE`] the run fails with the numbers instead.
//!
//! And the previous run's shards are deleted first. `ShardWriter` numbers from
//! zero every time, so a smaller second run would otherwise leave the tail of the
//! first behind -- mixed into the same directory, carrying the same source name,
//! and no longer described by the provenance sidecar.

use crate::error::{Error, Result};
use crate::llm::{Example, Synthesizer};
use crate::provenance::{self, Provenance};
use crate::seed::{self, Seed};
use crate::source::{PROVENANCE, SEED_SOURCES, SYNTHETIC, SeedSource};
use crate::summary::{RefusalCount, RunSummary, WrittenBySource};
use crate::validate::Validator;
use futures::StreamExt as _;
use ime_corpus::Normalizer;
use ime_corpus::source::SAMPLES_PER_SHARD;
use ime_corpus::text::content_id;
use ime_g2p::DataLayout;
use ime_g2p::annotate::Sample;
use ime_g2p::llm::LlmSettings;
use ime_g2p::shards::{ShardWriter, shard_paths};
use ime_pinyin::{Lexicon, SyllableTable};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;
use tracing::{info, warn};

/// The share of generated examples that may fail validation before a run is
/// treated as broken rather than merely lossy.
pub const MAX_DROP_RATE: f64 = 0.30;

/// What one generation run was asked for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Options {
    /// Stop after this many grounded seed terms.
    pub terms_limit: Option<usize>,
    /// How many usage examples each term is asked for.
    pub per_term: usize,
    /// How many requests are allowed in flight at once.
    pub concurrency: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            terms_limit: None,
            per_term: crate::llm::DEFAULT_PER_TERM,
            concurrency: crate::llm::DEFAULT_CONCURRENCY,
        }
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

/// Generate, validate and write one batch of synthetic training samples.
///
/// Seeds are read from `seed_root`; shards, the provenance sidecar and the run
/// summary are written under `out_root`.
///
/// # Errors
///
/// If a seed source is missing, if no seed can be grounded, if more than
/// [`MAX_DROP_RATE`] of the generated examples fail validation, or if a shard
/// cannot be written.
pub async fn generate(
    seed_root: &Path,
    out_root: &Path,
    settings: &LlmSettings,
    options: Options,
) -> Result<RunSummary> {
    let started = Instant::now();
    let normalizer = Normalizer::new()?;
    let lexicon = Lexicon::load(&SyllableTable::load())?;
    let load = seed::load(seed_root, &normalizer, &lexicon)?;
    if load.seeds.is_empty() {
        let loaded = load.counts.iter().map(|count| count.loaded).sum();
        return Err(Error::NoGroundedSeeds { loaded });
    }
    let mut seeds = load.seeds;
    if let Some(limit) = options.terms_limit {
        seeds.truncate(limit);
    }

    let synthesizer = Synthesizer::new(settings, options.per_term);
    info!(
        terms = seeds.len(),
        per_term = synthesizer.per_term(),
        concurrency = options.concurrency,
        model = synthesizer.model(),
        "generating grounded usage sentences"
    );
    let replies = request(&synthesizer, &seeds, options.concurrency).await;
    let batch = accept(replies, &seeds, &normalizer, &lexicon)?;

    if batch.drops.drop_rate() > MAX_DROP_RATE {
        return Err(Error::TooManyDropped {
            dropped: batch.drops.dropped(),
            examples: batch.drops.considered(),
            percent: batch.drops.drop_rate() * 100.0,
            limit: MAX_DROP_RATE * 100.0,
        });
    }

    let written_by_source = write(out_root, batch.generated)?;
    let written = written_by_source.values().sum();
    let summary = RunSummary {
        source: SYNTHETIC.to_owned(),
        model: synthesizer.model().to_owned(),
        per_term: synthesizer.per_term(),
        concurrency: options.concurrency,
        seeds: load.counts,
        terms: seeds.len(),
        examples_requested: seeds.len() * synthesizer.per_term(),
        examples_parsed: batch.parsed,
        refusals: batch.refusals.iter().map(|count| count.terms).sum(),
        refusal_reasons: batch.refusals,
        drops: batch.drops,
        written,
        written_by_source: SEED_SOURCES
            .iter()
            .map(|source| WrittenBySource {
                source: source.name.to_owned(),
                terms: batch
                    .terms_by_source
                    .get(source.name)
                    .copied()
                    .unwrap_or_default(),
                written: written_by_source
                    .get(source.name)
                    .copied()
                    .unwrap_or_default(),
            })
            .collect(),
        milliseconds: started.elapsed().as_millis(),
    };
    summary.write(out_root)?;
    info!(
        terms = summary.terms,
        parsed = summary.examples_parsed,
        written = summary.written,
        refusals = summary.refusals,
        dropped = summary.drops.dropped(),
        per_minute = summary.terms_per_minute(),
        "synthetic samples written"
    );
    Ok(summary)
}

/// One reply per seed, put back into seed order once they have all landed.
///
/// Ordering here is what makes a run reproducible: the duplicate rule keeps the
/// first of two identical sentences, and without this sort "first" would mean
/// "whichever request the network happened to answer soonest".
async fn request(
    synthesizer: &Synthesizer,
    seeds: &[Seed],
    concurrency: usize,
) -> Vec<std::result::Result<Vec<Example>, String>> {
    let mut replies: Vec<(usize, std::result::Result<Vec<Example>, String>)> =
        futures::stream::iter(
            seeds
                .iter()
                .enumerate()
                .map(|(index, seed)| async move { (index, synthesizer.examples(seed).await) }),
        )
        .buffer_unordered(concurrency.max(1))
        .collect()
        .await;
    replies.sort_by_key(|(index, _)| *index);
    replies.into_iter().map(|(_, reply)| reply).collect()
}

/// What the validation pass made of one run's replies.
struct Batch {
    generated: Vec<Generated>,
    refusals: Vec<RefusalCount>,
    drops: crate::validate::DropCounts,
    parsed: usize,
    terms_by_source: HashMap<&'static str, usize>,
}

/// Validate every reply against the seed it answers, tallying as it goes.
///
/// # Errors
///
/// If the normaliser's length invariant breaks, or a reply has no seed.
fn accept(
    replies: Vec<std::result::Result<Vec<Example>, String>>,
    seeds: &[Seed],
    normalizer: &Normalizer,
    lexicon: &Lexicon,
) -> Result<Batch> {
    if replies.len() != seeds.len() {
        return Err(Error::Invariant(format!(
            "{} replies for {} seeds, which cannot happen",
            replies.len(),
            seeds.len()
        )));
    }
    let mut validator = Validator::new(normalizer, lexicon);
    let mut generated = Vec::new();
    let mut refused: HashMap<String, usize> = HashMap::new();
    let mut parsed = 0_usize;
    let mut terms_by_source: HashMap<&'static str, usize> = HashMap::new();
    for (seed, reply) in seeds.iter().zip(replies) {
        *terms_by_source.entry(seed.source.name).or_default() += 1;
        let examples = match reply {
            Ok(examples) => examples,
            Err(reason) => {
                warn!(term = seed.term, reason, "term refused");
                *refused.entry(reason).or_default() += 1;
                continue;
            }
        };
        parsed += examples.len();
        for example in &examples {
            let Some(accepted) = validator.judge(&seed.term, example)? else {
                continue;
            };
            let id = content_id(SYNTHETIC, &accepted.text, accepted.context.as_deref());
            generated.push(Generated {
                sample: Sample {
                    id: id.clone(),
                    source: SYNTHETIC.to_owned(),
                    text: accepted.text,
                    context: accepted.context,
                },
                provenance: Provenance::new(id, seed.term.clone(), seed.source, seed.grounding),
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
        terms_by_source,
    })
}

/// Write the batch and its sidecar, replacing whatever a previous run left.
///
/// # Errors
///
/// If a stale shard cannot be removed, if a shard cannot be written, or if the
/// two outputs end up with different row counts.
fn write(out_root: &Path, generated: Vec<Generated>) -> Result<HashMap<&'static str, usize>> {
    let samples_dir = DataLayout::new(out_root).samples();
    let provenance_dir = provenance::directory(out_root);
    let stale = remove_stale(&samples_dir, SYNTHETIC)? + remove_stale(&provenance_dir, PROVENANCE)?;
    if stale > 0 {
        info!(shards = stale, "removed the previous run's shards");
    }
    let mut samples = ShardWriter::new(&samples_dir, SYNTHETIC, SAMPLES_PER_SHARD)?;
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
