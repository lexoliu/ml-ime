//! Drives a `CharLm` the way `fused-eval` does -- one record per thread -- to
//! make per-session memory use measurable:
//!
//! ```sh
//! uv run --project python python/scripts/charlm_sized.py target/sized-model
//! cargo run --release -p ime-lm --example pressure -- target/sized-model
//! /usr/bin/time -l target/release/examples/pressure target/sized-model
//! ```
//!
//! Each of eight threads decodes `RECORDS` records: a `start` over a random
//! context, then `STEPS` `advance` calls over a beam of `BEAM` hypotheses fed
//! random characters.

use ime_decode::Transition;
use ime_lm::{CharLm, LmState};
use ime_pinyin::{CharId, Lexicon, SyllableTable};
use rand::Rng;
use rand::seq::IndexedRandom;
use std::fs;
use std::path::PathBuf;
use std::thread;
use std::time::Instant;

const THREADS: usize = 8;
const RECORDS: usize = 100;
const BEAM: usize = 8;
const STEPS: usize = 20;

#[expect(
    clippy::cast_precision_loss,
    reason = "a record count is far under 2^53"
)]
fn main() {
    tracing_subscriber::fmt::init();
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("usage: pressure <export dir written by charlm_sized.py>"),
    );
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
    let model = CharLm::open(&dir, &lexicon).expect("the export opens");
    tracing::info!(
        threads = THREADS,
        records = RECORDS,
        "pressure run starting"
    );
    let started = Instant::now();
    thread::scope(|scope| {
        for _ in 0..THREADS {
            let model = &model;
            let lexicon = &lexicon;
            let ids = &ids;
            scope.spawn(move || {
                let mut rng = rand::rng();
                for _ in 0..RECORDS {
                    let width = rng.random_range(40..=62);
                    let context: String = (0..width)
                        .map(|_| {
                            *lexicon
                                .characters()
                                .choose(&mut rng)
                                .expect("the lexicon is not empty")
                        })
                        .collect();
                    let mut states = vec![model.start(Some(&context)); BEAM];
                    for _ in 0..STEPS {
                        let batch: Vec<(&LmState, CharId)> = states
                            .iter()
                            .map(|state| {
                                (
                                    state,
                                    *ids.choose(&mut rng).expect("the lexicon is not empty"),
                                )
                            })
                            .collect();
                        states = model.advance(&batch);
                    }
                }
            });
        }
    });
    let elapsed = started.elapsed().as_secs_f64();
    let records = THREADS * RECORDS;
    tracing::info!(
        records,
        seconds = elapsed,
        records_per_second = records as f64 / elapsed,
        "pressure run done"
    );
}
