//! The knowledge slice of dumped answers, rescored without decoding anything.
//!
//! Every system's answers reach this binary as a dump file: `fused-eval
//! --dump` and `mlime eval e2e --dump` write one directly, and `mlime eval
//! dump-answers` / `dump-gui` / `dump-rime` turn a generator's answers, a GUI
//! engine's journal and RIME's report into one. This binary reads them
//! against the evaluation set and the knowledge lexicon and reports the slice
//! of [`ime_eval::KnowledgeSlice`]: sentence top-1 over the records naming an
//! entity, the entities spelled exactly, and the characters inside them, for
//! the whole slice and per category. Given several dumps, each after the
//! first is also compared with the first, record by record.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::io::{self, Write as _};
use std::path::PathBuf;
use std::process::ExitCode;

use askama::Template as _;
use clap::Parser;
use ime_eval::cli::{chain, read};
use ime_eval::{EvalSet, KnowledgeLexicon, KnowledgeReport, KnowledgeSlice, Slice, parse_dump};
use ime_pinyin::{Lexicon, SyllableTable};

/// Report the knowledge slice of dumped answers.
#[derive(Debug, Parser)]
#[command(version, about, long_about = None)]
struct Args {
    /// The JSON Lines evaluation set the dumps were decoded from.
    #[arg(long)]
    eval_set: PathBuf,
    /// The `word<TAB>category` lexicon `mlime lexicon knowledge` writes.
    #[arg(long)]
    lexicon: PathBuf,
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
    /// Score only the records this dump holds: the subset a system answered,
    /// so every dump is measured on the records it covers.
    #[arg(long)]
    within: Option<PathBuf>,
    /// List every slice record: its keystrokes, context, wanted sentence, top
    /// answer, the wanted sentence's rank in the dump and its entities, an
    /// entity the top answer misspells marked `!`.
    #[arg(long)]
    by_record: bool,
}

fn run(args: &Args) -> Result<String, String> {
    let set = EvalSet::parse(&read(&args.eval_set, "eval set")?)
        .map_err(|error| format!("{}: {}", args.eval_set.display(), chain(&error)))?;
    let knowledge = KnowledgeLexicon::parse(&read(&args.lexicon, "lexicon")?)
        .map_err(|error| format!("{}: {}", args.lexicon.display(), chain(&error)))?;
    let table = SyllableTable::load();
    let lexicon = Lexicon::load(&table).map_err(|error| chain(&error))?;
    let mut slice = KnowledgeSlice::select(&set, &table, &lexicon, &knowledge)
        .map_err(|error| chain(&error))?;
    if let Some(path) = &args.within {
        let records: BTreeSet<usize> = parse_dump(&read(path, "dump")?)
            .map_err(|error| format!("{}: {}", path.display(), chain(&error)))?
            .into_iter()
            .map(|row| row.record)
            .collect();
        slice = slice.within(&records);
    }
    let render = |error: askama::Error| format!("could not render the report: {error}");
    let mut out = String::new();
    let mut baseline: Option<KnowledgeReport> = None;
    for path in &args.dump {
        let rows = parse_dump(&read(path, "dump")?)
            .map_err(|error| format!("{}: {}", path.display(), chain(&error)))?;
        let report = slice
            .measure(&set, args.slice, args.dev_share, &rows)
            .map_err(|error| format!("{}: {}", path.display(), chain(&error)))?;
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
            eprintln!("ime-knowledge: {message}");
            ExitCode::FAILURE
        }
    }
}
