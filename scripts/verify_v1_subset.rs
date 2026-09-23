#!/usr/bin/env rust-script
//! Check a drawn training subset against the mixture it was supposed to be.
//!
//! Run with `rust-script scripts/verify_v1_subset.rs --data-dir data/run3-v1
//! --exclude data/run1_pool/pool.jsonl --exclude data/run3_pool/pool.jsonl`.
//!
//! `build_v1_subset.rs` enforces the mixture and the exclusions while it writes.
//! This reads the result back and checks the same two properties from the output
//! alone, because the interesting failure is a subset drawn *before* a pool was
//! cut: the builder would have reported zero hits for a pool it never saw, and
//! the only evidence left on disk is the held-out text sitting in the subset.
//! So every target is matched against every pool -- a spot check on five texts
//! cannot distinguish "excluded" from "never overlapped" -- and any survivor is
//! printed with the shard it is in and fails the run.
//!
//! ```cargo
//! [dependencies]
//! polars = { version = "0.55.2", default-features = false, features = ["parquet"] }
//! rayon = "1.12.0"
//! serde_json = "1.0.151"
//! ```

use polars::prelude::{DataFrame, ParquetReader, SerReader};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::fs::File;
use std::path::{Path, PathBuf};

/// How many surviving held-out texts are printed before the list is truncated.
const SHOWN_HITS: usize = 10;

/// One shard's contribution to the verdict.
struct Shard {
    source: String,
    rows: usize,
    /// The held-out texts that survived into this shard, as (pool, text).
    survivors: Vec<(String, String)>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = Arguments::parse()?;
    let excluded = Exclusions::read(&arguments.excludes)?;
    let samples = arguments.data_dir.join("samples");
    let shards = shards_of(&samples)?;
    if shards.is_empty() {
        return Err(format!("no shards under {}", samples.display()).into());
    }
    println!(
        "verifying {} shards under {} against {} held-out texts from {} pool(s)",
        shards.len(),
        samples.display(),
        excluded.total(),
        excluded.pools.len()
    );

    let counted: Vec<Shard> = shards
        .par_iter()
        .map(|path| verify_shard(path, &excluded))
        .collect::<Result<Vec<Shard>, String>>()?;

    let mut rows: BTreeMap<String, usize> = BTreeMap::new();
    let mut files: BTreeMap<String, usize> = BTreeMap::new();
    for shard in &counted {
        *rows.entry(shard.source.clone()).or_default() += shard.rows;
        *files.entry(shard.source.clone()).or_default() += 1;
    }
    let total: usize = rows.values().sum();

    println!();
    println!(
        "{:<10} {:>7} {:>14} {:>8}",
        "source", "shards", "segments", "share"
    );
    for (source, count) in &rows {
        let share = if total == 0 {
            0.0
        } else {
            100.0 * *count as f64 / total as f64
        };
        println!(
            "{source:<10} {:>7} {count:>14} {share:>7.2}%",
            files[source]
        );
    }
    println!("{:<10} {:>7} {total:>14} {:>7.2}%", "total", shards.len(), 100.0);

    let survivors: Vec<&(String, String)> = counted
        .iter()
        .flat_map(|shard| shard.survivors.iter())
        .collect();
    println!();
    if survivors.is_empty() {
        println!("no held-out text survived into the subset");
        return Ok(());
    }
    for (pool, text) in survivors.iter().take(SHOWN_HITS) {
        println!("held out in {pool} but present in the subset: {text}");
    }
    Err(format!(
        "{} held-out text(s) survived into the subset",
        survivors.len()
    )
    .into())
}

/// Count one shard's rows and collect any held-out text it still carries.
fn verify_shard(path: &Path, excluded: &Exclusions) -> Result<Shard, String> {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("{} has no file name", path.display()))?;
    let source = name
        .split_once('-')
        .map(|(source, _)| source.to_owned())
        .ok_or_else(|| format!("{name} is not named <source>-<index>.parquet"))?;
    let frame = read_frame(path).map_err(|error| format!("{name}: {error}"))?;
    let texts = frame
        .column("text")
        .and_then(polars::prelude::Column::str)
        .map_err(|error| format!("{name}: no text column: {error}"))?;

    let mut survivors = Vec::new();
    for (index, text) in texts.iter().enumerate() {
        let text = text.ok_or_else(|| format!("{name} row {index} has a null text"))?;
        if let Some(pool) = excluded.hit(text) {
            survivors.push((pool.to_owned(), text.to_owned()));
        }
    }
    Ok(Shard {
        source,
        rows: frame.height(),
        survivors,
    })
}

/// Read one parquet shard whole.
fn read_frame(path: &Path) -> Result<DataFrame, Box<dyn Error>> {
    Ok(ParquetReader::new(File::open(path)?).finish()?)
}

/// Every `*.parquet` shard under `directory`, in name order.
fn shards_of(directory: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(directory)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<PathBuf>, std::io::Error>>()?
        .into_iter()
        .filter(|path| {
            path.extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension == "parquet")
        })
        .collect();
    paths.sort();
    Ok(paths)
}

/// The command line: which subset to read, and which pools must be absent from it.
struct Arguments {
    data_dir: PathBuf,
    excludes: Vec<PathBuf>,
}

impl Arguments {
    /// Parse `--data-dir` and repeatable `--exclude`.
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut data_dir = None;
        let mut excludes = Vec::new();
        let mut arguments = std::env::args().skip(1);
        while let Some(flag) = arguments.next() {
            let value = arguments
                .next()
                .ok_or_else(|| format!("{flag} takes a value"))?;
            match flag.as_str() {
                "--data-dir" => data_dir = Some(PathBuf::from(value)),
                "--exclude" => excludes.push(PathBuf::from(value)),
                other => return Err(format!("unknown flag {other}").into()),
            }
        }
        let excludes = if excludes.is_empty() {
            return Err("at least one --exclude is required".into());
        } else {
            excludes
        };
        Ok(Self {
            data_dir: data_dir.ok_or("--data-dir is required")?,
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
