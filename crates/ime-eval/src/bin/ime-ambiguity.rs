//! The ambiguity slice of dumped beams, rescored without decoding anything.
//!
//! `fused-eval --dump` and `mlime eval e2e --dump` both write each record's
//! ranked hypotheses as JSON Lines. This binary reads one or more of those
//! files against the evaluation set they were decoded from and reports the
//! slice of [`ime_eval::AmbiguitySlice`]: character accuracy at the positions
//! the surrounding text decides, and sentence top-1 over the records that
//! hold them. Given several dumps, each after the first is also compared with
//! the first, record by record. `fused-eval` reports the same slice from its
//! own beams through the same code, and the end-to-end evaluation calls this
//! binary on its dump, so every system is measured identically.

use std::fmt::Write as _;
use std::io::{self, Write as _};
use std::path::PathBuf;
use std::process::ExitCode;

use askama::Template as _;
use clap::Parser;
use ime_eval::cli::{chain, read};
use ime_eval::{AmbiguityReport, AmbiguitySlice, EvalSet, Slice, parse_dump};
use ime_pinyin::{Lexicon, SyllableTable};

/// Report the ambiguity slice of dumped beams.
#[derive(Debug, Parser)]
#[command(version, about, long_about = None)]
struct Args {
    /// The JSON Lines evaluation set the dumps were decoded from.
    #[arg(long)]
    eval_set: PathBuf,
    /// A dump to score. Repeat it to compare: every later dump is compared
    /// with the first, record by record.
    #[arg(long, required = true)]
    dump: Vec<PathBuf>,
    /// Which records to score: all, dev or test.
    #[arg(long)]
    slice: Slice,
    /// What share of the set the dev slice holds; membership is the record
    /// hash every command uses.
    #[arg(long, default_value = "0.0905")]
    dev_share: f64,
    /// List every slice record: its keystrokes, context, wanted sentence, top
    /// answer and the wanted sentence's rank in the dump.
    #[arg(long)]
    by_record: bool,
}

fn run(args: &Args) -> Result<String, String> {
    let set = EvalSet::parse(&read(&args.eval_set, "eval set")?)
        .map_err(|error| format!("{}: {}", args.eval_set.display(), chain(&error)))?;
    let table = SyllableTable::load();
    let lexicon = Lexicon::load(&table).map_err(|error| chain(&error))?;
    let slice = AmbiguitySlice::select(&set, &table, &lexicon).map_err(|error| chain(&error))?;
    let mut out = String::new();
    let mut baseline: Option<AmbiguityReport> = None;
    for path in &args.dump {
        let rows = parse_dump(&read(path, "dump")?)
            .map_err(|error| format!("{}: {}", path.display(), chain(&error)))?;
        let report = slice
            .measure(&set, args.slice, args.dev_share, &rows)
            .map_err(|error| format!("{}: {}", path.display(), chain(&error)))?;
        let render = |error: askama::Error| format!("could not render the report: {error}");
        writeln!(out, "dump {}", path.display()).map_err(|error| error.to_string())?;
        out.push_str(&report.render().map_err(render)?);
        if args.by_record {
            out.push('\n');
            out.push_str(&report.listing(&set).render().map_err(render)?);
        }
        if let Some(first) = &baseline {
            let comparison = report
                .compare(first)
                .map_err(|error| format!("{}: {}", path.display(), chain(&error)))?;
            writeln!(out, "\nagainst {}", args.dump[0].display())
                .map_err(|error| error.to_string())?;
            out.push_str(&comparison.render().map_err(render)?);
        } else {
            baseline = Some(report);
        }
        out.push('\n');
    }
    Ok(out)
}

fn main() -> ExitCode {
    let args = Args::parse();
    match run(&args).and_then(|out| {
        io::stdout()
            .write_all(out.as_bytes())
            .map_err(|error| format!("could not write stdout: {error}"))
    }) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("ime-ambiguity: {message}");
            ExitCode::FAILURE
        }
    }
}
