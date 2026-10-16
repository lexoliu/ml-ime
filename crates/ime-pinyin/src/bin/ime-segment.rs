//! The k-best segmentations of keystrokes, one request per line.
//!
//! Training's sequence-level loss (#116) ranks a sentence against the other
//! readings of its keystrokes, and those readings have to be the ones the
//! evaluation lattice offers. This binary is that segmenter, served over a
//! pipe: each stdin line is one keystroke string, and each stdout line is the
//! JSON array of its readings, cheapest first, every reading an array of the
//! typed spans. Two readings that cut the keys into the same spans -- `m` as
//! the syllable and `m` as an initial -- are one reading here, because the
//! model sees spans and admits a span's every character. Output is flushed
//! per line so a caller can hold one request in flight at a time.
//!
//! The options are `emit-lattice`'s: `--max-paths` readings (8 there) and no
//! half-typed tail. Keystrokes that admit no reading are an error, not an
//! empty answer: training only sends what a sentence's own syllables spell.

use std::io::{self, BufRead, BufWriter, Write};
use std::process::ExitCode;

use ime_pinyin::{SegmentLattice, SegmentOptions, SyllableTable};

fn options() -> Result<SegmentOptions, String> {
    let mut args = std::env::args().skip(1);
    let max_paths = match (args.next().as_deref(), args.next(), args.next()) {
        (Some("--max-paths"), Some(value), None) => value
            .parse::<usize>()
            .ok()
            .filter(|&paths| paths > 0)
            .ok_or_else(|| format!("--max-paths takes a positive count, got {value:?}"))?,
        _ => return Err("usage: ime-segment --max-paths <N>".to_owned()),
    };
    Ok(SegmentOptions {
        max_paths,
        allow_incomplete_tail: false,
        ..SegmentOptions::default()
    })
}

fn serve(options: &SegmentOptions) -> Result<(), String> {
    let table = SyllableTable::load();
    let mut out = BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let keys = line.map_err(|error| format!("could not read stdin: {error}"))?;
        let lattice = SegmentLattice::build(&keys, &table, options)
            .map_err(|error| format!("{keys:?} admits no reading: {error}"))?;
        let mut readings: Vec<Vec<&str>> = Vec::new();
        for reading in lattice.k_best(options) {
            let spans: Vec<&str> = reading
                .segments()
                .iter()
                .map(|segment| &keys[segment.start()..segment.end()])
                .collect();
            if !readings.contains(&spans) {
                readings.push(spans);
            }
        }
        serde_json::to_writer(&mut out, &readings)
            .map_err(|error| format!("could not write a reading: {error}"))?;
        out.write_all(b"\n")
            .and_then(|()| out.flush())
            .map_err(|error| format!("could not write stdout: {error}"))?;
    }
    Ok(())
}

fn main() -> ExitCode {
    match options().and_then(|options| serve(&options)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("ime-segment: {message}");
            ExitCode::FAILURE
        }
    }
}
