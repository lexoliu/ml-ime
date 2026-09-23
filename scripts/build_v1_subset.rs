#!/usr/bin/env rust-script
//! Draw the v1 training subset out of the run3 typing-segment corpus.
//!
//! Run with `rust-script scripts/build_v1_subset.rs --data-dir data/run3 --out-dir
//! data/run3-v1 --exclude data/run1_pool/pool.jsonl --exclude data/run3_pool/pool.jsonl`.
//!
//! This is a one-off: v1 needs one balanced 10M-segment draw with a fixed
//! mixture, and a mixture that is chosen once by hand does not belong in
//! `ime-cli` next to the stratified draws that are part of the pipeline. It
//! stays a script so the numbers in the data card are reproducible from a file
//! rather than from a shell history.
//!
//! Two properties matter and are enforced rather than assumed. Nothing in either
//! held-out pool may survive into the subset, so both pools are matched on the
//! exact target text and the hits are counted per source; the run3 eval pool was
//! drawn from these very shards, so *zero* hits there is a bug and fails the run,
//! while the run1 pool was cut under the old sentence segmentation and is
//! expected to miss almost everywhere. And the draw is stratified within each
//! source: every input shard contributes in proportion to its size, sampled with
//! a seed derived from the shard's own name, so one shard's rows are the same
//! rows on any machine and the output is a 1:1 image of the input shard layout.
//!
//! ```cargo
//! [dependencies]
//! polars = { version = "0.55.2", default-features = false, features = ["parquet"] }
//! rand = "0.9.2"
//! rayon = "1.12.0"
//! serde_json = "1.0.151"
//! ```

use polars::prelude::{
    DataFrame, IdxCa, IdxSize, ParquetReader, ParquetWriter, PlSmallStr, SerReader,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// The v1 mixture: how many segments each source contributes, `None` meaning all
/// of it. Chosen to put the weight on conversational text -- what an IME is
/// actually typed into -- while keeping enough encyclopedic and news prose to
/// cover the formal vocabulary, and taking the two short-video sources whole
/// because they are small and are the only in-domain slang the corpus has.
const MIXTURE: [(&str, Option<usize>); 6] = [
    ("dialogue", Some(3_000_000)),
    ("moegirl", Some(2_500_000)),
    ("news", Some(2_000_000)),
    ("wiki", Some(1_500_000)),
    ("douyin", None),
    ("bilibili", None),
];

/// Base seed for the draw. Mixed with each shard's name, never used raw.
const SEED: u64 = 11;

/// One shard's share of the work: where it came from, and how many rows of it to keep.
struct Draw {
    path: PathBuf,
    keep: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = Arguments::parse()?;
    let excluded = Exclusions::read(&arguments.excludes)?;
    println!(
        "excluding {} texts from {} pool(s)",
        excluded.total(),
        arguments.excludes.len()
    );

    let out_samples = arguments.out_dir.join("samples");
    std::fs::create_dir_all(&out_samples)?;
    let report = Mutex::new(BTreeMap::<String, SourceReport>::new());

    for (source, quota) in MIXTURE {
        let shards = shards_of(&arguments.data_dir.join("samples"), source)?;
        if shards.is_empty() {
            return Err(format!("no {source} shards under {}", arguments.data_dir.display()).into());
        }
        let sizes = shards
            .iter()
            .map(|path| rows_in(path))
            .collect::<Result<Vec<usize>, Box<dyn Error>>>()?;
        let available: usize = sizes.iter().sum();
        let wanted = quota.unwrap_or(available).min(available);
        let keeps = allocate(&sizes, wanted);
        let draws: Vec<Draw> = shards
            .into_iter()
            .zip(keeps)
            .map(|(path, keep)| Draw { path, keep })
            .collect();

        let counted: Vec<Shard> = draws
            .par_iter()
            .map(|draw| write_draw(draw, &excluded, &out_samples))
            .collect::<Result<Vec<Shard>, String>>()?;
        let drawn: usize = counted.iter().map(|shard| shard.written).sum();
        let hits: BTreeMap<String, usize> =
            counted
                .iter()
                .fold(BTreeMap::new(), |mut totals, shard| {
                    for (pool, count) in &shard.hits {
                        *totals.entry(pool.clone()).or_default() += count;
                    }
                    totals
                });
        println!(
            "{source}: {drawn} drawn of {available} available ({} shards), exclusion hits {hits:?}",
            draws.len()
        );
        report.lock().map_err(|_| "the report lock was poisoned")?.insert(
            source.to_owned(),
            SourceReport {
                available,
                drawn,
                hits,
            },
        );
    }

    let report = report.into_inner().map_err(|_| "the report lock was poisoned")?;
    let total: usize = report.values().map(|entry| entry.drawn).sum();
    let pool_hits: usize = report
        .values()
        .flat_map(|entry| entry.hits.values())
        .sum();
    println!();
    println!("{:<10} {:>12} {:>12} {:>8}", "source", "available", "drawn", "share");
    for (source, entry) in &report {
        let share = if total == 0 {
            0.0
        } else {
            100.0 * entry.drawn as f64 / total as f64
        };
        println!(
            "{source:<10} {:>12} {:>12} {share:>7.2}%",
            entry.available, entry.drawn
        );
    }
    println!("subset: {total} segments, {pool_hits} held-out texts dropped");
    if pool_hits == 0 {
        return Err("no held-out text matched anything; the pools and the corpus disagree".into());
    }
    Ok(())
}

/// What one source contributed.
struct SourceReport {
    available: usize,
    drawn: usize,
    hits: BTreeMap<String, usize>,
}

/// What one shard contributed.
struct Shard {
    written: usize,
    hits: BTreeMap<String, usize>,
}

/// The command line: where the corpus is, where the subset goes, what to hold out.
struct Arguments {
    data_dir: PathBuf,
    out_dir: PathBuf,
    excludes: Vec<PathBuf>,
}

impl Arguments {
    /// Parse `--data-dir`, `--out-dir` and repeatable `--exclude`.
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut data_dir = None;
        let mut out_dir = None;
        let mut excludes = Vec::new();
        let mut arguments = std::env::args().skip(1);
        while let Some(flag) = arguments.next() {
            let value = arguments
                .next()
                .ok_or_else(|| format!("{flag} takes a value"))?;
            match flag.as_str() {
                "--data-dir" => data_dir = Some(PathBuf::from(value)),
                "--out-dir" => out_dir = Some(PathBuf::from(value)),
                "--exclude" => excludes.push(PathBuf::from(value)),
                other => return Err(format!("unknown flag {other}").into()),
            }
        }
        Ok(Self {
            data_dir: data_dir.ok_or("--data-dir is required")?,
            out_dir: out_dir.ok_or("--out-dir is required")?,
            excludes,
        })
    }
}

/// The held-out texts, kept one set per pool so a pool that matches nothing is visible.
struct Exclusions {
    pools: Vec<(String, HashSet<String>)>,
}

impl Exclusions {
    /// Read every pool's `text` field.
    fn read(paths: &[PathBuf]) -> Result<Self, Box<dyn Error>> {
        let mut pools = Vec::new();
        for path in paths {
            let name = path
                .parent()
                .and_then(|parent| parent.file_name())
                .and_then(|name| name.to_str())
                .unwrap_or("pool")
                .to_owned();
            let contents = std::fs::read_to_string(path)?;
            let mut texts = HashSet::new();
            for line in contents.lines().filter(|line| !line.trim().is_empty()) {
                let value: serde_json::Value = serde_json::from_str(line)?;
                let text = value
                    .get("text")
                    .and_then(|text| text.as_str())
                    .ok_or_else(|| format!("{} has a line with no text field", path.display()))?;
                texts.insert(text.to_owned());
            }
            if texts.is_empty() {
                return Err(format!("{} holds no texts", path.display()).into());
            }
            pools.push((name, texts));
        }
        Ok(Self { pools })
    }

    /// How many texts are held out in total.
    fn total(&self) -> usize {
        self.pools.iter().map(|(_, texts)| texts.len()).sum()
    }

    /// The pool holding `text`, if any.
    fn hit(&self, text: &str) -> Option<&str> {
        self.pools
            .iter()
            .find(|(_, texts)| texts.contains(text))
            .map(|(name, _)| name.as_str())
    }
}

/// Every `<source>-*.parquet` shard under `directory`, in name order.
fn shards_of(directory: &Path, source: &str) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(directory)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<PathBuf>, std::io::Error>>()?
        .into_iter()
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    name.starts_with(&format!("{source}-")) && name.ends_with(".parquet")
                })
        })
        .collect();
    paths.sort();
    Ok(paths)
}

/// How many rows a shard holds, from its parquet metadata alone.
fn rows_in(path: &Path) -> Result<usize, Box<dyn Error>> {
    let file = File::open(path)?;
    Ok(ParquetReader::new(file).num_rows()?)
}

/// Split `wanted` across shards in proportion to their sizes, exactly.
///
/// Floor first, then hand the remainder to the shards with the largest fractional
/// part, so the quotas sum to `wanted` rather than to `wanted` minus rounding.
fn allocate(sizes: &[usize], wanted: usize) -> Vec<usize> {
    let total: usize = sizes.iter().sum();
    if total == 0 || wanted == 0 {
        return vec![0; sizes.len()];
    }
    let mut keeps: Vec<usize> = sizes
        .iter()
        .map(|size| size * wanted / total)
        .collect();
    let mut remainders: Vec<(usize, usize)> = sizes
        .iter()
        .enumerate()
        .map(|(index, size)| (index, size * wanted % total))
        .collect();
    remainders.sort_by(|left, right| right.1.cmp(&left.1).then(left.0.cmp(&right.0)));
    let mut short = wanted - keeps.iter().sum::<usize>();
    for (index, _) in remainders {
        if short == 0 {
            break;
        }
        if keeps[index] < sizes[index] {
            keeps[index] += 1;
            short -= 1;
        }
    }
    keeps
}

/// Draw one shard's rows and write them under the same file name.
fn write_draw(draw: &Draw, excluded: &Exclusions, out_dir: &Path) -> Result<Shard, String> {
    let name = draw
        .path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("{} has no file name", draw.path.display()))?
        .to_owned();
    let frame = read_frame(&draw.path).map_err(|error| format!("{name}: {error}"))?;
    let texts = frame
        .column("text")
        .and_then(polars::prelude::Column::str)
        .map_err(|error| format!("{name}: no text column: {error}"))?;

    let mut eligible: Vec<IdxSize> = Vec::with_capacity(texts.len());
    let mut hits: BTreeMap<String, usize> = BTreeMap::new();
    for (index, text) in texts.iter().enumerate() {
        let text = text.ok_or_else(|| format!("{name} row {index} has a null text"))?;
        match excluded.hit(text) {
            Some(pool) => *hits.entry(pool.to_owned()).or_default() += 1,
            None => eligible.push(
                IdxSize::try_from(index).map_err(|_| format!("{name} is longer than an index"))?,
            ),
        }
    }

    let keep = draw.keep.min(eligible.len());
    let mut rng = StdRng::seed_from_u64(SEED ^ fnv1a(&name));
    let mut chosen: Vec<IdxSize> = rand::seq::index::sample(&mut rng, eligible.len(), keep)
        .into_iter()
        .map(|position| eligible[position])
        .collect();
    chosen.sort_unstable();

    let indices = IdxCa::from_vec(PlSmallStr::from_static("idx"), chosen);
    let mut drawn = frame
        .take(&indices)
        .map_err(|error| format!("{name}: {error}"))?;
    let file = File::create(out_dir.join(&name)).map_err(|error| format!("{name}: {error}"))?;
    ParquetWriter::new(file)
        .finish(&mut drawn)
        .map_err(|error| format!("{name}: {error}"))?;
    Ok(Shard {
        written: keep,
        hits,
    })
}

/// Read one shard into a frame.
fn read_frame(path: &Path) -> Result<DataFrame, Box<dyn Error>> {
    let file = File::open(path)?;
    Ok(ParquetReader::new(file).finish()?)
}

/// A stable 64-bit hash of a shard name, so the per-shard seed does not depend on
/// the standard library's randomised hasher.
fn fnv1a(name: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in name.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}
