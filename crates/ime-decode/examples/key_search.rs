//! The keyed search's cost at the product's shape, with mock models.
//!
//! Replays each record's keys one keystroke at a time, as a live session
//! does: every keystroke rebuilds the legality lattice under the live
//! options and reruns the search over all keys typed so far, at eight beams
//! and the top 8. The models are hashed stand-ins, so the times are the
//! search's own -- lattice, masks, merges, selection -- with the network's
//! forward excluded. A second pass searches each whole record under the
//! offline options.
//!
//! ```text
//! key_search [--limit N] EVAL.jsonl...
//! ```
//!
//! Each file is a JSONL evaluation set with a `pinyin` field. The report is
//! one JSON object a file on stdout.

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;

use ime_decode::keyed::{EmissionRows, KeyEmission, Shape, Weights, search};
use ime_decode::mock::{self, HashedBigram};
use ime_pinyin::{EmitId, Keys, Legality, LegalityOptions};
use serde::{Deserialize, Serialize};

/// The rows the stand-in head draws from, shared by every record.
const ROWS: usize = 64;

/// One record of an evaluation set; the rest of the line is not read.
#[derive(Deserialize)]
struct Record {
    pinyin: String,
}

/// The stand-in head over one record's first `keys` keys: key `k` reads the
/// shared table's row `(offset + k) % ROWS`.
struct Head<'a> {
    table: &'a EmissionRows,
    offset: usize,
    keys: usize,
}

impl KeyEmission for Head<'_> {
    fn keys(&self) -> usize {
        self.keys
    }

    fn blank(&self, key: usize) -> f32 {
        self.table.blank((self.offset + key) % ROWS)
    }

    fn gather(&self, key: usize, ids: &[EmitId], out: &mut [f32]) {
        self.table.gather((self.offset + key) % ROWS, ids, out);
    }
}

/// Order statistics of a sample, in microseconds.
#[derive(Serialize)]
struct Summary {
    count: usize,
    median_us: f64,
    p95_us: f64,
    max_us: f64,
    mean_us: f64,
}

#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "nanosecond samples and their ranks"
)]
fn summary(mut nanos: Vec<u64>) -> Summary {
    nanos.sort_unstable();
    let at = |q: f64| nanos[((nanos.len() - 1) as f64 * q).round() as usize] as f64 / 1e3;
    Summary {
        count: nanos.len(),
        median_us: at(0.5),
        p95_us: at(0.95),
        max_us: at(1.0),
        mean_us: nanos.iter().sum::<u64>() as f64 / nanos.len() as f64 / 1e3,
    }
}

/// One file's report.
#[derive(Serialize)]
struct Report {
    file: String,
    records: usize,
    skipped: usize,
    mean_keys: f64,
    beam_width: usize,
    top_k: usize,
    admitted_bound: usize,
    /// Lattice and search, per keystroke, over all keys typed so far.
    keystroke: Summary,
    /// The lattice alone, per keystroke.
    keystroke_lattice: Summary,
    /// The search alone per keystroke, divided by the keys it searched.
    per_key: Summary,
    /// The whole record under the offline options, lattice and search.
    offline_record: Summary,
    /// The most candidates scored at one key, over every search.
    max_candidates: usize,
}

fn nanos(tick: Instant) -> u64 {
    u64::try_from(tick.elapsed().as_nanos()).expect("a search ends within u64 nanoseconds")
}

#[expect(
    clippy::cast_precision_loss,
    reason = "a mean over a few thousand records"
)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut limit = usize::MAX;
    let mut files = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--limit" {
            limit = args.next().ok_or("--limit takes a count")?.parse()?;
        } else {
            files.push(arg);
        }
    }
    if files.is_empty() {
        return Err("usage: key_search [--limit N] EVAL.jsonl...".into());
    }

    let legality = Legality::over_lexicon()?;
    let emittable = legality.emittable().len();
    let table = mock::hashed_rows(ROWS, emittable, 0x5eed, 6.0);
    let transition = HashedBigram {
        seed: 0x7a11,
        spread: 4.0,
    };
    let shape = Shape::PRODUCT;
    let weights = Weights::UNIT;

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for file in files {
        let mut keystroke = Vec::new();
        let mut keystroke_lattice = Vec::new();
        let mut per_key = Vec::new();
        let mut offline = Vec::new();
        let (mut records, mut skipped, mut keys_total, mut max_candidates) = (0, 0, 0, 0);
        for line in BufReader::new(File::open(&file)?).lines() {
            if records == limit {
                break;
            }
            let record: Record = serde_json::from_str(&line?)?;
            let Ok(keys) = Keys::new(&record.pinyin) else {
                skipped += 1;
                continue;
            };
            let offset = records % ROWS;
            records += 1;
            keys_total += keys.len();
            for typed in 1..=keys.len() {
                let prefix = Keys::new(&keys.as_str()[..typed])?;
                let head = Head {
                    table: &table,
                    offset,
                    keys: typed,
                };
                let tick = Instant::now();
                let lattice = legality.lattice(&prefix, LegalityOptions::LIVE);
                let built = nanos(tick);
                let searched = Instant::now();
                let found = search(&lattice, &head, &transition, None, shape, weights);
                let search_ns = nanos(searched);
                keystroke.push(nanos(tick));
                keystroke_lattice.push(built);
                per_key.push(search_ns / typed as u64);
                max_candidates = max_candidates.max(found.stats.max_candidates);
            }
            let head = Head {
                table: &table,
                offset,
                keys: keys.len(),
            };
            let tick = Instant::now();
            let lattice = legality.lattice(&keys, LegalityOptions::OFFLINE);
            let found = search(&lattice, &head, &transition, None, shape, weights);
            offline.push(nanos(tick));
            max_candidates = max_candidates.max(found.stats.max_candidates);
        }
        if records == 0 {
            return Err(format!("{file} holds no readable record").into());
        }
        let report = Report {
            file,
            records,
            skipped,
            mean_keys: keys_total as f64 / records as f64,
            beam_width: shape.width(),
            top_k: shape.top_k(),
            admitted_bound: legality.admitted_bound(),
            keystroke: summary(keystroke),
            keystroke_lattice: summary(keystroke_lattice),
            per_key: summary(per_key),
            offline_record: summary(offline),
            max_candidates,
        };
        serde_json::to_writer(&mut out, &report)?;
        out.write_all(b"\n")?;
    }
    Ok(())
}
