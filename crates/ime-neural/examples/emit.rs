//! Emit a score file through the exported towers.
//!
//! `ime-cli emit-lattice` asks what the model must answer; this answers it:
//! every lattice record's per-position candidate log probabilities, written
//! as `ScoreRecord` lines, plus each record's wall-clock emission time for
//! the cost table. The file is the `mlime train emit` score file's Rust
//! twin, and the identity check diffs the two numerically.
//!
//! ```text
//! emit <export-dir> <lattice.jsonl> <out.jsonl> [--no-context] [--raw]
//!      [--threads N] [--records indices.txt]
//! ```
//!
//! `--raw` skips the score file's four-decimal rounding, for checks that
//! compare the towers' output against the file's values at sub-rounding
//! tolerance.

use ime_decode::{LatticeRecord, ScoreRecord};
use ime_neural::{NeuralError, RouteA, SessionShape};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead as _, BufReader, BufWriter, Write};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::time::Instant;
use thiserror::Error;

/// Why an emission run failed.
#[derive(Debug, Error)]
enum EmitError {
    /// A file could not be opened, read or written.
    #[error("could not {what} {path}: {source}")]
    Io {
        /// What the file was being opened or written for.
        what: &'static str,
        /// The file in question.
        path: PathBuf,
        /// The platform's answer.
        source: std::io::Error,
    },
    /// A lattice line is not a `LatticeRecord`.
    #[error("could not parse a lattice record in {path}: {source}")]
    Lattice {
        /// The lattice file.
        path: PathBuf,
        /// `serde_json`'s complaint.
        source: serde_json::Error,
    },
    /// A score record could not be serialized into the output file.
    #[error("could not serialize a score record into {path}: {source}")]
    Serialize {
        /// The output file.
        path: PathBuf,
        /// `serde_json`'s complaint.
        source: serde_json::Error,
    },
    /// The towers failed.
    #[error(transparent)]
    Neural(#[from] NeuralError),
}

/// What the command line asked for.
struct Args {
    /// The export directory to open.
    export: PathBuf,
    /// The lattice to answer.
    lattice: PathBuf,
    /// Where the score lines land.
    out: PathBuf,
    /// Whether the context tower feeds the gate.
    with_context: bool,
    /// Write the raw log probabilities rather than the file's four decimals.
    raw: bool,
    /// Intra-op threads per session.
    threads: NonZeroUsize,
    /// Emit only these record indices, when given.
    only: Option<HashSet<usize>>,
}

fn parse_args() -> Result<Args, EmitError> {
    let mut positional = Vec::new();
    let mut with_context = true;
    let mut raw = false;
    let mut threads = NonZeroUsize::MIN;
    let mut only = None;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--no-context" => with_context = false,
            "--raw" => raw = true,
            "--threads" => {
                threads = args
                    .next()
                    .and_then(|value| value.parse().ok())
                    .expect("--threads takes a non-zero count");
            }
            "--records" => {
                let path = PathBuf::from(args.next().expect("--records takes a file"));
                let file = File::open(&path).map_err(|source| EmitError::Io {
                    what: "open",
                    path: path.clone(),
                    source,
                })?;
                only = Some(
                    BufReader::new(file)
                        .lines()
                        .map(|line| {
                            line.expect("an index line reads")
                                .parse()
                                .expect("an index line is a record number")
                        })
                        .collect(),
                );
            }
            flag if flag.starts_with("--") => {
                tracing::error!(flag, "unknown flag");
                std::process::exit(2);
            }
            _ => positional.push(flag),
        }
    }
    let [export, lattice, out] = <[String; 3]>::try_from(positional).unwrap_or_else(|_| {
        tracing::error!("usage: emit <export-dir> <lattice.jsonl> <out.jsonl>");
        std::process::exit(2);
    });
    Ok(Args {
        export: PathBuf::from(export),
        lattice: PathBuf::from(lattice),
        out: PathBuf::from(out),
        with_context,
        raw,
        threads,
        only,
    })
}

fn open(path: &PathBuf, write: bool) -> Result<File, EmitError> {
    let opened = if write {
        File::create(path)
    } else {
        File::open(path)
    };
    opened.map_err(|source| EmitError::Io {
        what: if write { "create" } else { "open" },
        path: path.clone(),
        source,
    })
}

fn sink(path: &PathBuf) -> Result<BufWriter<File>, EmitError> {
    Ok(BufWriter::new(open(path, true)?))
}

fn emit(args: &Args) -> Result<(), EmitError> {
    let towers = RouteA::open(
        &args.export,
        SessionShape {
            backend: ime_neural::Backend::Cpu,
            intra_threads: args.threads,
            verbose_logging: false,
        },
    )?;
    let mut scores = sink(&args.out)?;
    let times_path = args.out.with_extension("times.tsv");
    let mut times = sink(&times_path)?;
    for line in BufReader::new(open(&args.lattice, false)?).lines() {
        let line = line.map_err(|source| EmitError::Io {
            what: "read",
            path: args.lattice.clone(),
            source,
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let record: LatticeRecord =
            serde_json::from_str(&line).map_err(|source| EmitError::Lattice {
                path: args.lattice.clone(),
                source,
            })?;
        if let Some(only) = &args.only
            && !only.contains(&record.record)
        {
            continue;
        }
        let start = Instant::now();
        let paths = towers.emission(&record, None, args.with_context)?;
        // The score file carries four decimals, as `mlime train emit` writes
        // it; `emission` itself returns the raw log probabilities.
        let paths: Vec<Vec<Vec<f32>>> = if args.raw {
            paths
        } else {
            paths
                .into_iter()
                .map(|positions| {
                    positions
                        .into_iter()
                        .map(|candidates| candidates.into_iter().map(rounded).collect())
                        .collect()
                })
                .collect()
        };
        writeln!(
            times,
            "{}\t{}",
            record.record,
            start.elapsed().as_secs_f64() * 1e3
        )
        .map_err(|source| EmitError::Io {
            what: "write",
            path: times_path.clone(),
            source,
        })?;
        serde_json::to_writer(
            &mut scores,
            &ScoreRecord {
                record: record.record,
                paths,
            },
        )
        .map_err(|source| EmitError::Serialize {
            path: args.out.clone(),
            source,
        })?;
        scores.write_all(b"\n").map_err(|source| EmitError::Io {
            what: "write",
            path: args.out.clone(),
            source,
        })?;
    }
    scores.flush().map_err(|source| EmitError::Io {
        what: "flush",
        path: args.out.clone(),
        source,
    })?;
    times.flush().map_err(|source| EmitError::Io {
        what: "flush",
        path: times_path,
        source,
    })
}

/// A value rounded to the score file's four decimals: `round(value, 4)` as
/// `mlime train emit` writes it, which formats *value* correctly rounded to
/// four decimal places, ties to even -- exactly what `{:.4}` prints and the
/// parse reads back.
fn rounded(value: f32) -> f32 {
    format!("{:.4}", f64::from(value))
        .parse()
        .expect("a four-decimal float parses back")
}

fn main() -> Result<(), EmitError> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();
    emit(&parse_args()?)
}
