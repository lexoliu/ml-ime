//! The per-keystroke cost of decoding under the typo noise model.
//!
//! A session segments and resolves candidates once per keystroke, so the
//! price the feature charges is the difference between `build`/`Candidates::
//! build` and `build_corrections`/`Candidates::build_corrections` on the same
//! growing prefixes. The example replays a handful of typings -- clean ones
//! and the typo twins the feature exists for -- prefix by prefix, and reports
//! the median cost per keystroke in each mode, split into the lattice's share
//! and candidate resolution's.
//!
//! Run from the repository root:
//!
//! ```text
//! cargo run --release -p ime-decode --example keystroke-cost -- [iterations]
//! ```

use ime_decode::Candidates;
use ime_pinyin::{CorrectionTable, Lexicon, SegmentLattice, SegmentOptions, SyllableTable};
use std::hint::black_box;
use std::time::{Duration, Instant};

/// The typings to replay: full-sentence clean inputs, sentence inputs with a
/// typo in them, and the short prefixes a session starts every sentence with.
const RECORDS: &[&str] = &[
    "zhongguorenmingongheguochengli",
    "womenxuexizhongwenhengaoxing",
    "jintiantianqihenhao",
    "wozaijia",
    "nihao",
    "wojaijia",
    "niqao",
    "zhomgguorenmin",
    "wo",
    "zgrm",
];

/// Median of a set of durations.
fn median(times: &mut [Duration]) -> Duration {
    times.sort_unstable();
    times[times.len() / 2]
}

/// One keystroke of one record under one mode: segment the prefix, take its
/// k-best readings, resolve the per-position candidate sets. Returns the
/// lattice's share and the resolution's share of the elapsed time.
fn measure(
    pinyin: &str,
    table: &SyllableTable,
    corrections: Option<&CorrectionTable>,
    options: &SegmentOptions,
    lexicon: &Lexicon,
) -> (Duration, Duration) {
    let lattice_start = Instant::now();
    let readings = if let Some(model) = corrections {
        SegmentLattice::build_corrections(pinyin, table, options, model)
            .expect("input reads")
            .k_best(options)
    } else {
        SegmentLattice::build(pinyin, table, options)
            .expect("input reads")
            .k_best(options)
    };
    let lattice_time = lattice_start.elapsed();
    let candidates_start = Instant::now();
    if let Some(model) = corrections {
        black_box(
            Candidates::build_corrections(&readings, pinyin, table, model, lexicon)
                .expect("masks are non-empty"),
        );
    } else {
        black_box(Candidates::build(&readings, lexicon).expect("masks are non-empty"));
    }
    (lattice_time, candidates_start.elapsed())
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();
    let iterations: usize = std::env::args()
        .nth(1)
        .map_or(200, |arg| arg.parse().expect("iterations is a count"));
    let table = SyllableTable::load();
    let lexicon = Lexicon::load(&table).expect("generated tables must agree");
    let model = CorrectionTable::load(&table).expect("the noise model loads");
    let options = SegmentOptions {
        allow_incomplete_tail: true,
        ..SegmentOptions::default()
    };

    // The measurement sums every prefix's median cost across records, so the
    // reported row is "one keystroke's share" of either mode, split by stage.
    let mut clean_lattice = Duration::ZERO;
    let mut clean_candidates = Duration::ZERO;
    let mut noisy_lattice = Duration::ZERO;
    let mut noisy_candidates = Duration::ZERO;
    let mut keystrokes = 0usize;
    for record in RECORDS {
        for end in 1..=record.len() {
            if !record.is_char_boundary(end) {
                continue;
            }
            let prefix = &record[..end];
            keystrokes += 1;
            let mut clean_l = Vec::with_capacity(iterations);
            let mut clean_c = Vec::with_capacity(iterations);
            let mut noisy_l = Vec::with_capacity(iterations);
            let mut noisy_c = Vec::with_capacity(iterations);
            for _ in 0..iterations {
                let (l, c) = measure(prefix, &table, None, &options, &lexicon);
                clean_l.push(l);
                clean_c.push(c);
                let (l, c) = measure(prefix, &table, Some(&model), &options, &lexicon);
                noisy_l.push(l);
                noisy_c.push(c);
            }
            clean_lattice += median(&mut clean_l);
            clean_candidates += median(&mut clean_c);
            noisy_lattice += median(&mut noisy_l);
            noisy_candidates += median(&mut noisy_c);
        }
    }
    #[expect(clippy::cast_precision_loss, reason = "keystrokes is a small count")]
    let keystrokes_f64 = keystrokes as f64;
    tracing::info!(
        keystrokes,
        iterations,
        clean_lattice_micros = clean_lattice.as_secs_f64() * 1e6 / keystrokes_f64,
        clean_candidates_micros = clean_candidates.as_secs_f64() * 1e6 / keystrokes_f64,
        noisy_lattice_micros = noisy_lattice.as_secs_f64() * 1e6 / keystrokes_f64,
        noisy_candidates_micros = noisy_candidates.as_secs_f64() * 1e6 / keystrokes_f64,
        "median per-keystroke cost, split by stage"
    );
}
