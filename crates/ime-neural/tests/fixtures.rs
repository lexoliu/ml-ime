//! The exported towers run end to end: `RouteA::emission` answers a real
//! lattice with the score table `routea_fixtures.py` recorded from
//! onnxruntime over the exported graphs.
//!
//! The fixtures under `tests/fixtures/` are written by
//! `python/scripts/routea_fixtures.py`; regenerate them with
//! `uv run --project python python/scripts/routea_fixtures.py
//! crates/ime-neural/tests/fixtures` from the repository root.

use ime_decode::{Candidates, Emission, Emittable, LatticeRecord};
use ime_neural::{RouteA, SessionShape};
use ime_pinyin::{Lexicon, SegmentLattice, SegmentOptions, SyllableTable};
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

/// `tests/fixtures/` beside this crate.
fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// The full lexicon, the one `emit-lattice` and the decoder build candidate
/// sets against.
fn lexicon() -> Lexicon {
    Lexicon::load(&SyllableTable::load()).expect("the committed lexicon loads")
}

/// The emittable set the lattice was written with.
fn emittable(dir: &Path, lexicon: &Lexicon) -> Emittable {
    let source = fs::read_to_string(dir.join("emittable.txt"))
        .expect("emittable.txt is committed with the fixtures");
    Emittable::parse(&source, lexicon).expect("the emittable set parses")
}

/// The lattice the fixture was scored over, as `ime-cli emit-lattice` wrote
/// it.
fn lattice(dir: &Path) -> Vec<LatticeRecord> {
    fs::read_to_string(dir.join("lattice.jsonl"))
        .expect("lattice.jsonl is committed with the fixtures")
        .lines()
        .map(|line| serde_json::from_str(line).expect("a lattice record parses"))
        .collect()
}

/// What `routea_fixtures.py` recorded from the exported graphs.
#[derive(Deserialize)]
struct Expected {
    /// Which export the table came from: `fp32` or `int8`.
    variant: String,
    /// Whether the context gate was open.
    with_context: bool,
    /// The lattice record's index.
    record: usize,
    /// The score table: `paths[path][position][candidate]`.
    paths: Vec<Vec<Vec<f32>>>,
}

/// The `expected.json` committed beside the exports.
fn expected(dir: &Path) -> Vec<Expected> {
    serde_json::from_str(
        &fs::read_to_string(dir.join("expected.json"))
            .expect("expected.json is committed with the fixture"),
    )
    .expect("expected.json parses")
}

/// The lattice `emit-lattice` would write for a record, rebuilt the way the
/// decoder builds it: `SegmentLattice::k_best`, `Candidates::build`, and the
/// emittable restriction -- so the fixture's lattice and the Rust one are
/// proven the same shape before a score is ever compared.
fn rebuild(record: &LatticeRecord, lexicon: &Lexicon, emittable: &Emittable) -> Candidates {
    let options = SegmentOptions {
        allow_incomplete_tail: false,
        ..SegmentOptions::default()
    };
    let lattice = SegmentLattice::build(&record.pinyin, &SyllableTable::load(), &options)
        .expect("the fixture pinyin reads");
    let segmentations = lattice.k_best(&options);
    let candidates = Candidates::build(&segmentations, lexicon).expect("the readings build");
    assert_eq!(
        candidates.len(),
        record.paths.len(),
        "record {}: the rebuilt lattice has a different path count",
        record.record
    );
    for (path, (segmentation, reading)) in segmentations.iter().zip(candidates.paths()).enumerate()
    {
        let written = &record.paths[path];
        let spans: Vec<&str> = segmentation
            .segments()
            .iter()
            .map(|segment| &record.pinyin[segment.start()..segment.end()])
            .collect();
        assert_eq!(
            spans, written.spans,
            "record {path}: the rebuilt reading's spans differ"
        );
        for (position, allowed) in reading.positions().iter().enumerate() {
            let asked: String = emittable
                .restrict(allowed)
                .iter()
                .map(|id| lexicon.character(*id))
                .collect::<String>();
            assert_eq!(
                asked, written.candidates[position],
                "record {}, path {path}, position {position}: candidates differ",
                record.record
            );
        }
    }
    candidates
}

/// A score table against the recorded one, cell by cell.
fn assert_table(got: &[Vec<Vec<f32>>], want: &[Vec<Vec<f32>>], what: &str, atol: f32) {
    assert_eq!(got.len(), want.len(), "{what}: path count differs");
    for (path, (got_path, want_path)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            got_path.len(),
            want_path.len(),
            "{what}[{path}]: position count differs"
        );
        for (position, (got_row, want_row)) in got_path.iter().zip(want_path).enumerate() {
            assert_eq!(
                got_row.len(),
                want_row.len(),
                "{what}[{path}][{position}]: candidate count differs"
            );
            for (slot, (got, want)) in got_row.iter().zip(want_row).enumerate() {
                assert!(
                    (got - want).abs() <= atol,
                    "{what}[{path}][{position}][{slot}]: got {got}, want {want}"
                );
            }
        }
    }
}

/// The fp32 tolerance `test_routea_export.py` holds an export to `PyTorch` at
/// (`atol=1e-4`): the table was recorded by onnxruntime on one platform and
/// this leg runs it on another, and ORT's kernels are not bit-stable across
/// them -- the drift here is a couple of f32 ulps.
const FP32_ATOL: f32 = 1e-4;

/// The int8-versus-fp32 bound `test_int8_export_tracks_fp32` holds the
/// Python export to (`atol=0.05`).
const INT8_ATOL: f32 = 0.05;

/// Both exports score the committed lattice the way onnxruntime recorded it:
/// fp32 within `FP32_ATOL`, int8 against the *fp32* table within `INT8_ATOL` --
/// the kernels' answers differ between platforms, so a recorded int8 table
/// would be bound to the platform it was recorded on.
#[test]
fn the_exported_towers_reproduce_the_recorded_tables() {
    let dir = fixture_dir();
    let lexicon = lexicon();
    let emittable = emittable(&dir, &lexicon);
    let records = lattice(&dir);
    let expected = expected(&dir);
    for record in &records {
        rebuild(record, &lexicon, &emittable);
    }

    for variant in ["fp32", "int8"] {
        let atol = if variant == "fp32" {
            FP32_ATOL
        } else {
            INT8_ATOL
        };
        let model =
            RouteA::open(&dir.join(variant), SessionShape::default()).expect("the fixture opens");
        for with_context in [true, false] {
            for record in &records {
                let want = expected
                    .iter()
                    .find(|entry| {
                        entry.variant == "fp32"
                            && entry.with_context == with_context
                            && entry.record == record.record
                    })
                    .expect("every lattice record is recorded");
                let got = model
                    .emission(record, with_context)
                    .expect("the fixture scores");
                assert_table(
                    &got,
                    &want.paths,
                    &format!("{variant} ctx={with_context} record {}", record.record),
                    atol,
                );
            }
        }
    }
}

/// `scored` binds the same table to the lattice it came from, so the
/// decoder's `Emission::score` reads the recorded candidates (within
/// `FP32_ATOL`'s cross-platform kernel drift) and the floor stands everywhere
/// else.
#[test]
fn scored_attaches_the_table_to_the_lattice() {
    let dir = fixture_dir();
    let lexicon = lexicon();
    let emittable = emittable(&dir, &lexicon);
    let records = lattice(&dir);
    let expected = expected(&dir);
    let model =
        RouteA::open(&dir.join("fp32"), SessionShape::default()).expect("the fixture opens");
    let floor = -30.0f32;

    for record in &records {
        let candidates = rebuild(record, &lexicon, &emittable);
        let want = expected
            .iter()
            .find(|entry| {
                entry.variant == "fp32" && entry.with_context && entry.record == record.record
            })
            .expect("every lattice record is recorded");
        let scored = model
            .scored(record, &candidates, &emittable, floor, true)
            .expect("the table attaches");
        for (path, reading) in candidates.paths().iter().enumerate() {
            for (position, allowed) in reading.positions().iter().enumerate() {
                let asked = emittable.restrict(allowed);
                assert_eq!(
                    asked.len(),
                    want.paths[path][position].len(),
                    "the written candidate row has the asked candidates"
                );
                for (slot, &candidate) in asked.iter().enumerate() {
                    let got = scored.score(path, position, candidate);
                    let want = want.paths[path][position][slot];
                    assert!(
                        (got - want).abs() <= FP32_ATOL,
                        "record {}, path {path}, position {position}: got {got}, want {want}",
                        record.record
                    );
                }
                // A character the position does not admit scores at the floor.
                let absent = lexicon
                    .characters()
                    .iter()
                    .find_map(|&character| {
                        lexicon.id_of(character).filter(|id| !allowed.contains(id))
                    })
                    .expect("the fixture lexicon exceeds one position's set");
                assert_eq!(
                    scored.score(path, position, absent).to_bits(),
                    floor.to_bits()
                );
            }
        }
    }
}
