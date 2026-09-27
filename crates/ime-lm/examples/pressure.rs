//! Drives a `CharLm` the way `fused-eval` does -- records through `start` plus
//! `STEPS` `advance` calls over a beam of `BEAM` -- to make per-step latency
//! and per-session memory measurable:
//!
//! ```sh
//! uv run --project python python/scripts/charlm_sized.py target/sized-model
//! cargo run --release -p ime-lm --example pressure -- target/sized-model
//! cargo run --release -p ime-lm --example pressure --features gpu-webgpu -- \
//!     target/sized-model --batch 64 --backend webgpu
//! /usr/bin/time -l target/release/examples/pressure target/sized-model
//! ```
//!
//! With the default `--batch 1`, each of `THREADS` threads decodes `RECORDS`
//! records one at a time -- the decoder's own shape. `--batch N` runs the same
//! records serially in lockstep chunks of N, so one `advance` call covers
//! `N x BEAM` rows drawn from N different records -- the shape `fused-eval
//! --batch` feeds a GPU backend. A GPU backend takes the serial path at any
//! batch, its session holding the one device.

use ime_decode::Transition;
use ime_lm::{Backend, CharLm, LmState, SessionShape};
use ime_pinyin::{CharId, Lexicon, SyllableTable};
use rand::Rng;
use rand::seq::IndexedRandom;
use std::fs;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::thread;
use std::time::Instant;

const THREADS: usize = 8;
const RECORDS: usize = 100;
const BEAM: usize = 8;
const STEPS: usize = 20;

/// A record's prelude plus the random characters its `STEPS` steps feed.
struct Work {
    context: String,
    tokens: Vec<CharId>,
}

/// The command line: the export directory, the lockstep batch width, and the
/// shape the model's sessions are built to -- the rayon path's one intra-op
/// thread at batch one, the machine's parallelism for the one session a
/// lockstep run keeps, on whichever `--backend` named. `--ort-verbose` turns
/// on ONNX Runtime's verbose logging, the per-node provider placement that
/// proves which backend the step ran on.
fn args() -> (PathBuf, usize, SessionShape) {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().expect(
        "usage: pressure <export dir written by charlm_sized.py> [--batch N] [--backend cpu|coreml|webgpu|cuda] [--ort-verbose]",
    ));
    let mut batch = 1usize;
    let mut backend = Backend::Cpu;
    let mut verbose_logging = false;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--batch" => {
                batch = args
                    .next()
                    .expect("--batch takes a record count")
                    .parse()
                    .expect("--batch takes a record count");
            }
            "--backend" => {
                backend = match args.next().expect("--backend takes a name").as_str() {
                    "cpu" => Backend::Cpu,
                    "coreml" => Backend::CoreMl,
                    "webgpu" => Backend::WebGpu,
                    "cuda" => Backend::Cuda,
                    other => panic!("unknown --backend {other:?}"),
                };
            }
            "--ort-verbose" => verbose_logging = true,
            other => panic!("unknown argument {other:?}"),
        }
    }
    assert!(batch > 0, "--batch is at least one");
    let intra_threads = if batch > 1 || !matches!(backend, Backend::Cpu) {
        std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN)
    } else {
        NonZeroUsize::MIN
    };
    (
        dir,
        batch,
        SessionShape {
            backend,
            intra_threads,
            verbose_logging,
        },
    )
}

/// One record's workload: a random-width context and the `STEPS * BEAM`
/// characters its steps feed.
fn work(rng: &mut impl Rng, lexicon: &Lexicon, ids: &[CharId]) -> Work {
    let width = rng.random_range(40..=62);
    let context: String = (0..width)
        .map(|_| {
            *lexicon
                .characters()
                .choose(rng)
                .expect("the lexicon is not empty")
        })
        .collect();
    let tokens = (0..STEPS * BEAM)
        .map(|_| *ids.choose(rng).expect("the lexicon is not empty"))
        .collect();
    Work { context, tokens }
}

/// A `Work` decoded to a running state: `start` over its context, `BEAM` deep.
fn states(model: &CharLm, work: &Work) -> Vec<LmState> {
    vec![model.start(Some(&work.context)); BEAM]
}

#[expect(
    clippy::cast_precision_loss,
    reason = "record and step counts are far under 2^53"
)]
fn main() {
    tracing_subscriber::fmt::init();
    let (dir, batch, shape) = args();
    let table = SyllableTable::load();
    let source = fs::read_to_string(dir.join("char_pinyin.tsv"))
        .expect("char_pinyin.tsv is written beside the export");
    let lexicon = Lexicon::parse(&source, &table).expect("the character table parses");
    let ids: Vec<CharId> = lexicon
        .characters()
        .iter()
        .map(|&character| {
            lexicon
                .id_of(character)
                .expect("the lexicon indexes itself")
        })
        .collect();
    let model = CharLm::open(&dir, &lexicon, shape).expect("the export opens");
    let total = THREADS * RECORDS;
    tracing::info!(
        threads = THREADS,
        records = total,
        batch,
        backend = ?shape.backend,
        intra_threads = shape.intra_threads.get(),
        "pressure run starting"
    );
    let started = Instant::now();
    let mut step_calls = 0usize;
    if batch == 1 && matches!(shape.backend, Backend::Cpu) {
        // The decoder's own shape: one record per thread, `BEAM` rows a step.
        // A GPU backend skips it -- its session holds the device alone.
        thread::scope(|scope| {
            for _ in 0..THREADS {
                let model = &model;
                let lexicon = &lexicon;
                let ids = &ids;
                scope.spawn(move || {
                    let mut rng = rand::rng();
                    for _ in 0..RECORDS {
                        let work = work(&mut rng, lexicon, ids);
                        let mut states = states(model, &work);
                        for step in 0..STEPS {
                            let rows: Vec<(&LmState, CharId)> = states
                                .iter()
                                .enumerate()
                                .map(|(beam, state)| (state, work.tokens[step * BEAM + beam]))
                                .collect();
                            states = model.advance(&rows);
                        }
                    }
                });
            }
        });
        step_calls = THREADS * RECORDS * STEPS;
    } else {
        // Lockstep: `batch` records' beams advance as one `batch x BEAM`
        // call; batch one on a GPU is this loop with chunks of one.
        let mut rng = rand::rng();
        let mut works = (0..total).map(|_| work(&mut rng, &lexicon, &ids));
        loop {
            let chunk: Vec<Work> = works.by_ref().take(batch).collect();
            if chunk.is_empty() {
                break;
            }
            let mut beams: Vec<Vec<LmState>> =
                chunk.iter().map(|work| states(&model, work)).collect();
            for step in 0..STEPS {
                let mut rows: Vec<(&LmState, CharId)> = Vec::with_capacity(chunk.len() * BEAM);
                for (record, states) in beams.iter().enumerate() {
                    for (beam, state) in states.iter().enumerate() {
                        rows.push((state, chunk[record].tokens[step * BEAM + beam]));
                    }
                }
                let mut advanced = model.advance(&rows).into_iter();
                beams = beams
                    .iter()
                    .map(|states| {
                        advanced
                            .by_ref()
                            .take(states.len())
                            .collect::<Vec<LmState>>()
                    })
                    .collect();
            }
            step_calls += STEPS;
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    tracing::info!(
        records = total,
        seconds = elapsed,
        records_per_second = total as f64 / elapsed,
        step_calls,
        ms_per_step = elapsed * 1e3 / step_calls as f64,
        "pressure run done"
    );
}
