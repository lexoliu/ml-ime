#!/usr/bin/env rust-script
//! Cut the rows of run3 that the v1 subset did not draw, for v2 to label.
//!
//! Run with `rust-script scripts/build_v2_rest.rs --data-dir data/run3 --v1-dir
//! data/run3-v1 --out-dir data/run3-rest --exclude data/run1_pool/pool.jsonl
//! --exclude data/run3_pool/pool.jsonl`.
//!
//! v2 trains on all of run3, and ten million of its rows already carry v1's
//! labels. Labelling costs a T4 about two hours per million rows, so those rows
//! are not labelled twice: this script writes the *complement* of the v1 draw,
//! shard for shard, and v2 stages the v1 shards and the rest shards side by side.
//!
//! The v1 draw was a 1:1 image of run3's shard layout -- `dialogue-00007.parquet`
//! in v1 holds rows of `dialogue-00007.parquet` in run3 and of nothing else --
//! so the complement is exact and needs no global index: a row is in the rest
//! when its id is not in the v1 shard of the same name. That alignment is checked
//! rather than trusted, because it is the whole basis of the split: every v1 id
//! must be found in its same-named run3 shard, and a v1 shard with no run3
//! partner fails the run. The held-out pools are matched on the exact target text
//! as the v1 draw matched them, and the run3 pool must hit something.
//!
//! The rest shards are named `<source>-rest-<index>.parquet` so that they can sit
//! in one directory with the v1 shards, whose names they would otherwise repeat.
//!
//! ```cargo
//! [dependencies]
//! polars = { version = "0.55.2", default-features = false, features = ["parquet"] }
//! rayon = "1.12.0"
//! serde_json = "1.0.151"
//! ```

use polars::prelude::{BooleanChunked, DataFrame, ParquetReader, ParquetWriter, SerReader};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::fs::File;
use std::path::{Path, PathBuf};

/// The sources run3 holds, in the order the report lists them.
const SOURCES: [&str; 6] = ["dialogue", "moegirl", "news", "wiki", "douyin", "bilibili"];

/// What one shard contributed.
struct Shard {
    available: usize,
    in_v1: usize,
    written: usize,
    hits: BTreeMap<String, usize>,
}

/// What one source contributed.
#[derive(Default)]
struct SourceReport {
    available: usize,
    in_v1: usize,
    written: usize,
    hits: BTreeMap<String, usize>,
}

impl SourceReport {
    fn add(&mut self, shard: &Shard) {
        self.available += shard.available;
        self.in_v1 += shard.in_v1;
        self.written += shard.written;
        for (pool, count) in &shard.hits {
            *self.hits.entry(pool.clone()).or_default() += count;
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = Arguments::parse()?;
    let excluded = Exclusions::read(&arguments.excludes)?;
    println!(
        "excluding {} texts from {} pool(s)",
        excluded.total(),
        arguments.excludes.len()
    );

    let run3_samples = arguments.data_dir.join("samples");
    let v1_samples = arguments.v1_dir.join("samples");
    let out_samples = arguments.out_dir.join("samples");
    std::fs::create_dir_all(&out_samples)?;

    let mut report = BTreeMap::<String, SourceReport>::new();
    for source in SOURCES {
        let shards = shards_of(&run3_samples, source)?;
        if shards.is_empty() {
            return Err(format!("no {source} shards under {}", run3_samples.display()).into());
        }
        for v1_shard in shards_of(&v1_samples, source)? {
            let name = file_name(&v1_shard)?;
            if !run3_samples.join(&name).is_file() {
                return Err(format!("v1 shard {name} has no run3 shard of that name").into());
            }
        }
        let counted: Vec<Shard> = shards
            .par_iter()
            .map(|path| write_rest(path, &v1_samples, &excluded, &out_samples))
            .collect::<Result<Vec<Shard>, String>>()?;
        let entry = report.entry(source.to_owned()).or_default();
        for shard in &counted {
            entry.add(shard);
        }
        println!(
            "{source}: {} rest of {} available ({} in v1, {} shards), exclusion hits {:?}",
            entry.written,
            entry.available,
            entry.in_v1,
            counted.len(),
            entry.hits
        );
    }

    let total: usize = report.values().map(|entry| entry.written).sum();
    let pool_hits: usize = report.values().flat_map(|entry| entry.hits.values()).sum();
    println!();
    println!(
        "{:<10} {:>12} {:>12} {:>12} {:>8}",
        "source", "available", "in v1", "rest", "share"
    );
    for (source, entry) in &report {
        let share = if total == 0 {
            0.0
        } else {
            100.0 * entry.written as f64 / total as f64
        };
        println!(
            "{source:<10} {:>12} {:>12} {:>12} {share:>7.2}%",
            entry.available, entry.in_v1, entry.written
        );
    }
    println!("rest: {total} segments, {pool_hits} held-out texts dropped");
    if pool_hits == 0 {
        return Err("no held-out text matched anything; the pools and the corpus disagree".into());
    }
    Ok(())
}

/// The command line: the run3 corpus, the v1 subset, where the rest goes, what to hold out.
struct Arguments {
    data_dir: PathBuf,
    v1_dir: PathBuf,
    out_dir: PathBuf,
    excludes: Vec<PathBuf>,
}

impl Arguments {
    /// Parse `--data-dir`, `--v1-dir`, `--out-dir` and repeatable `--exclude`.
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut data_dir = None;
        let mut v1_dir = None;
        let mut out_dir = None;
        let mut excludes = Vec::new();
        let mut arguments = std::env::args().skip(1);
        while let Some(flag) = arguments.next() {
            let value = arguments
                .next()
                .ok_or_else(|| format!("{flag} takes a value"))?;
            match flag.as_str() {
                "--data-dir" => data_dir = Some(PathBuf::from(value)),
                "--v1-dir" => v1_dir = Some(PathBuf::from(value)),
                "--out-dir" => out_dir = Some(PathBuf::from(value)),
                "--exclude" => excludes.push(PathBuf::from(value)),
                other => return Err(format!("unknown flag {other}").into()),
            }
        }
        Ok(Self {
            data_dir: data_dir.ok_or("--data-dir is required")?,
            v1_dir: v1_dir.ok_or("--v1-dir is required")?,
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

/// The file name of a shard, as text.
fn file_name(path: &Path) -> Result<String, String> {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(str::to_owned)
        .ok_or_else(|| format!("{} has no file name", path.display()))
}

/// The ids one shard holds.
fn ids_in(path: &Path) -> Result<HashSet<String>, String> {
    let frame = read_frame(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let ids = frame
        .column("id")
        .and_then(polars::prelude::Column::str)
        .map_err(|error| format!("{}: no id column: {error}", path.display()))?;
    ids.iter()
        .enumerate()
        .map(|(index, id)| {
            id.map(str::to_owned)
                .ok_or_else(|| format!("{} row {index} has a null id", path.display()))
        })
        .collect()
}

/// Write the rows of one run3 shard that v1 did not draw and no pool holds.
fn write_rest(
    path: &Path,
    v1_samples: &Path,
    excluded: &Exclusions,
    out_dir: &Path,
) -> Result<Shard, String> {
    let name = file_name(path)?;
    let v1_path = v1_samples.join(&name);
    let mut drawn = if v1_path.is_file() {
        ids_in(&v1_path)?
    } else {
        HashSet::new()
    };
    let in_v1 = drawn.len();

    let frame = read_frame(path).map_err(|error| format!("{name}: {error}"))?;
    let ids = frame
        .column("id")
        .and_then(polars::prelude::Column::str)
        .map_err(|error| format!("{name}: no id column: {error}"))?;
    let texts = frame
        .column("text")
        .and_then(polars::prelude::Column::str)
        .map_err(|error| format!("{name}: no text column: {error}"))?;

    let mut keep: Vec<bool> = Vec::with_capacity(frame.height());
    let mut hits: BTreeMap<String, usize> = BTreeMap::new();
    for (index, (id, text)) in ids.iter().zip(texts.iter()).enumerate() {
        let id = id.ok_or_else(|| format!("{name} row {index} has a null id"))?;
        let text = text.ok_or_else(|| format!("{name} row {index} has a null text"))?;
        if drawn.remove(id) {
            keep.push(false);
        } else if let Some(pool) = excluded.hit(text) {
            *hits.entry(pool.to_owned()).or_default() += 1;
            keep.push(false);
        } else {
            keep.push(true);
        }
    }
    if !drawn.is_empty() {
        return Err(format!(
            "{name}: {} v1 ids are not in the run3 shard of that name; the v1 draw is not a 1:1 image",
            drawn.len()
        ));
    }

    let written = keep.iter().filter(|kept| **kept).count();
    let mask: BooleanChunked = keep.iter().copied().collect();
    let mut rest = frame
        .filter(&mask)
        .map_err(|error| format!("{name}: {error}"))?;
    let rest_name = rest_name(&name)?;
    let file =
        File::create(out_dir.join(&rest_name)).map_err(|error| format!("{rest_name}: {error}"))?;
    ParquetWriter::new(file)
        .finish(&mut rest)
        .map_err(|error| format!("{rest_name}: {error}"))?;
    Ok(Shard {
        available: frame.height(),
        in_v1,
        written,
        hits,
    })
}

/// `<source>-<index>.parquet` becomes `<source>-rest-<index>.parquet`.
fn rest_name(name: &str) -> Result<String, String> {
    let (source, index) = name
        .strip_suffix(".parquet")
        .and_then(|stem| stem.rsplit_once('-'))
        .ok_or_else(|| format!("{name} is not a <source>-<index>.parquet shard"))?;
    Ok(format!("{source}-rest-{index}.parquet"))
}

/// Read one shard into a frame.
fn read_frame(path: &Path) -> Result<DataFrame, Box<dyn Error>> {
    let file = File::open(path)?;
    Ok(ParquetReader::new(file).finish()?)
}
