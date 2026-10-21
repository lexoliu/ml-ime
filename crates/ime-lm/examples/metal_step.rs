//! Times the Metal char-LM step per call at one row count — the A/B harness
//! for kernel changes to `MetalStep`:
//!
//! ```sh
//! cargo build --release -p ime-lm --example metal_step
//! target/release/examples/metal_step <resident-pages export> --rows 8 \
//!     [--warmup 50] [--iters 300] [--out samples.tsv]
//! ```
//!
//! One `start` over a fixed context, compacted to a paged leaf as the
//! decoder does, then `--rows` beams of it advanced together, each fed its
//! own token and asked the same `CANDIDATES` ids. Every call advances the
//! same parents and drops what it produced, so each call runs the identical
//! step at a constant history width. `--warmup` calls run untimed (the first
//! grows the step's buffers), then `--iters` timed calls each record:
//!
//! - `wall_ms` — host time around `advance`: marshal, the Metal step, the
//!   logp gather, the produced states — the decoder's per-step cost;
//! - `gpu_ms` — the step's `commit` to command-buffer completion;
//! - `encode_ms` / `submit_ms` — host encode, `commit` to scheduled.
//!
//! `--out` gets one tab-separated line per timed call under a header; the
//! run's median and p95 go to the log.

use std::path::PathBuf;

/// How many candidates a row's request carries — the median homophone count
/// over the syllable inventory, `pressure`'s stand-in for one lattice
/// position.
const CANDIDATES: usize = 94;

/// The prelude every call's beams descend from: fixed, so every build steps
/// over the same history.
const CONTEXT: &str =
    "现在我们在测试输入法的每一步延迟，看看在真实的上下文长度下模型一步需要多长时间才能给出候选";

/// The command line.
struct Args {
    dir: PathBuf,
    rows: usize,
    warmup: usize,
    iters: usize,
    out: Option<PathBuf>,
}

fn args() -> Args {
    const USAGE: &str = "usage: metal_step <resident-pages export> --rows N [--warmup N] [--iters N] [--out samples.tsv]";
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().expect(USAGE));
    let mut rows = None;
    let mut warmup = 50usize;
    let mut iters = 300usize;
    let mut out = None;
    let count = |args: &mut std::iter::Skip<std::env::Args>, flag: &str| -> usize {
        args.next()
            .and_then(|value| value.parse().ok())
            .unwrap_or_else(|| panic!("{flag} takes a count — {USAGE}"))
    };
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--rows" => rows = Some(count(&mut args, "--rows")),
            "--warmup" => warmup = count(&mut args, "--warmup"),
            "--iters" => iters = count(&mut args, "--iters"),
            "--out" => out = Some(PathBuf::from(args.next().expect(USAGE))),
            other => panic!("unknown argument {other:?} — {USAGE}"),
        }
    }
    let rows = rows.expect(USAGE);
    assert!(rows > 0 && iters > 0, "--rows and --iters are at least one");
    Args {
        dir,
        rows,
        warmup,
        iters,
        out,
    }
}

/// The nearest-rank *p*-quantile of an ascending slice.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "sample counts are far under 2^53 and the rank is in 1..=len"
)]
fn quantile(sorted: &[f64], p: f64) -> f64 {
    let rank = (p * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted[rank - 1]
}

#[cfg(target_os = "macos")]
fn main() {
    use ime_decode::{Asked, Transition};
    use ime_lm::{Backend, CharLm, MetalWeights, SessionShape};
    use ime_pinyin::{CharId, Lexicon, SyllableTable};
    use std::fmt::Write as _;
    use std::num::NonZeroUsize;
    use std::time::Instant;

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let args = args();
    let lexicon = Lexicon::load(&SyllableTable::load()).expect("the lexicon loads");
    let ids: Vec<CharId> = lexicon
        .characters()
        .iter()
        .map(|&character| {
            lexicon
                .id_of(character)
                .expect("the lexicon indexes itself")
        })
        .collect();
    let candidates = &ids[..CANDIDATES];
    let asked = Asked {
        candidates,
        eos: false,
    };
    let environment = ort::init()
        .build()
        .expect("the process builds its only environment");
    let shape = SessionShape {
        backend: Backend::Metal,
        metal_weights: MetalWeights::Auto,
        width: NonZeroUsize::new(8).expect("eight is not zero"),
        cache_rows: 4096,
        ..SessionShape::default()
    };
    let model = CharLm::open(&environment, &args.dir, &lexicon, shape)
        .expect("the export opens on the Metal backend");
    let parent = model.compact(&model.start(Some(CONTEXT), &asked));
    let steps: Vec<_> = (0..args.rows)
        .map(|row| (&parent, ids[CANDIDATES + row], asked))
        .collect();

    for _ in 0..args.warmup {
        drop(model.advance(&steps));
    }
    let mut wall = Vec::with_capacity(args.iters);
    let mut gpu = Vec::with_capacity(args.iters);
    let mut table = String::from("wall_ms\tgpu_ms\tencode_ms\tsubmit_ms\n");
    for _ in 0..args.iters {
        let started = Instant::now();
        let produced = model.advance(&steps);
        let wall_ms = started.elapsed().as_secs_f64() * 1e3;
        drop(produced);
        let timing = model
            .metal_step_timing()
            .expect("the Metal backend reports its step timing");
        wall.push(wall_ms);
        gpu.push(timing.gpu_ms());
        writeln!(
            table,
            "{wall_ms:.6}\t{:.6}\t{:.6}\t{:.6}",
            timing.gpu_ms(),
            timing.encode_ms(),
            timing.submit_ms()
        )
        .expect("a String takes every write");
    }
    if let Some(out) = &args.out {
        std::fs::write(out, table).expect("the samples file is writable");
    }
    wall.sort_by(f64::total_cmp);
    gpu.sort_by(f64::total_cmp);
    tracing::info!(
        rows = args.rows,
        iters = args.iters,
        wall_median_ms = quantile(&wall, 0.5),
        wall_p95_ms = quantile(&wall, 0.95),
        gpu_median_ms = quantile(&gpu, 0.5),
        gpu_p95_ms = quantile(&gpu, 0.95),
        "metal step timed"
    );
}

#[cfg(not(target_os = "macos"))]
fn main() {
    let _ = (args, quantile, CANDIDATES, CONTEXT);
    panic!("the Metal step exists only on macOS");
}
