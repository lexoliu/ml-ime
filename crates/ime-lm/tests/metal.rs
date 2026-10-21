//! The Metal step against the CPU reference at every row-tile boundary:
//! rows {1, 31, 32, 33, 64, 65, 100}. The 32x64 tiled GEMM's second row
//! tile read past its staged A chunk and wrote into the next tile's rows,
//! so every size past one 32-row tile returned garbage — this asserts each
//! size stays within the accumulated-quantisation band the f16 and naive
//! kernels both show (≤0.25 nats; the gate is 0.5, the bug landed 1e1–4e32).
//!
//! Every size runs under both weight packings — the export's own dtype and
//! fp16 — and the candidates include rare characters a restricted export
//! drops, which both backends must score as unreachable.
//!
//! Runs against a real `resident-pages` export — int8 or fp32, restricted
//! or over the whole alphabet — that `CHARLM_LM_DIR` must name. The test is
//! `#[ignore]`d so `cargo test` does not silently pass where the export is
//! absent.

#![cfg(target_os = "macos")]

use ime_decode::{Asked, Transition};
use ime_lm::{Backend, CharLm, LmState, MetalWeights, SessionShape};
use ime_pinyin::{CharId, Lexicon, SyllableTable};
use std::path::PathBuf;

/// The |gpu − cpu| logp bound per row count — see the module doc.
const GATE: f64 = 0.5;

/// The sweep's row counts: one tile, its last row, both sides of the
/// boundary, and several multi-tile shapes.
const SWEEP_ROWS: [usize; 7] = [1, 31, 32, 33, 64, 65, 100];

/// The step tuples `advance` takes for `states` — state, token, request.
fn divergence_steps<'s, 'a>(
    states: &'s [LmState],
    ids: &[CharId],
    asked: Asked<'a>,
) -> Vec<(&'s LmState, CharId, Asked<'a>)> {
    states
        .iter()
        .enumerate()
        .map(|(i, st)| (st, ids[i % ids.len()], asked))
        .collect()
}

/// Worst |gpu − cpu| logp over both `start` answers and two `advance` calls
/// at `rows` rows — the bench's comparison, minus the timing.
fn divergence(cpu: &CharLm, metal: &CharLm, ids: &[CharId], asked: Asked<'_>, rows: usize) -> f64 {
    let context = "现在我们在测试输入法的每一步延迟";
    let mut sc = vec![cpu.start(Some(context), &asked)];
    let mut sg = vec![metal.start(Some(context), &asked)];
    let mut max_diff = 0f64;
    let mut diff_pair = |a: &LmState, b: &LmState| {
        for &candidate in asked.candidates {
            max_diff = max_diff.max(f64::from(
                (cpu.score(a, candidate) - metal.score(b, candidate)).abs(),
            ));
        }
        max_diff = max_diff.max(f64::from((cpu.finish(a) - metal.finish(b)).abs()));
    };
    diff_pair(&sc[0], &sg[0]);
    for _call in 0..2 {
        while sc.len() < rows {
            sc.push(sc[0].clone());
            sg.push(sg[0].clone());
        }
        sc = cpu.advance(&divergence_steps(&sc, ids, asked));
        sg = metal.advance(&divergence_steps(&sg, ids, asked));
        for (a, b) in sc.iter().zip(&sg) {
            diff_pair(a, b);
        }
    }
    max_diff
}

/// The sweep's inputs: the export directory `CHARLM_LM_DIR` names, the
/// lexicon, 100 common ids, and the candidates — those ids plus rare
/// characters a restricted export drops.
struct Sweep {
    dir: PathBuf,
    lexicon: Lexicon,
    ids: Vec<CharId>,
    candidates: Vec<CharId>,
}

impl Sweep {
    fn load() -> Self {
        let dir = PathBuf::from(
            std::env::var_os("CHARLM_LM_DIR")
                .expect("needs a resident-pages export: set CHARLM_LM_DIR"),
        );
        let lexicon = Lexicon::load(&SyllableTable::load()).expect("the lexicon loads");
        let chars: Vec<char> = "的一了是我不在人们有来他这上着个地到大里说就去子得也和那要下看天时过出小么起你都把好还多没为又可家学只以主会样年想生同老中十从自面前头道它后然走很像见两用她国动进成回什边作对开而己些现山民候经发工向事命给长水几义三声于高手知理眼志点心战二问清身方空写等入死由平员客关先义军文世些边光即今从黑联步至".chars().collect();
        let ids: Vec<CharId> = chars
            .iter()
            .filter_map(|&c| lexicon.id_of(c))
            .take(100)
            .collect();
        assert_eq!(ids.len(), 100, "the sweep alphabet supplies 100 ids");
        // Characters outside any input method's emittable set: a restricted
        // export drops them, so their candidates exercise the unkept column.
        let rare: Vec<CharId> = "龘靐齉麤爩鱻"
            .chars()
            .map(|c| {
                lexicon
                    .id_of(c)
                    .expect("the lexicon reads every rare character")
            })
            .collect();
        let candidates: Vec<CharId> = ids.iter().chain(&rare).copied().collect();
        Self {
            dir,
            lexicon,
            ids,
            candidates,
        }
    }

    fn asked(&self) -> Asked<'_> {
        Asked {
            candidates: &self.candidates,
            eos: true,
        }
    }

    fn open(&self, environment: &ort::environment::Environment, shape: SessionShape) -> CharLm {
        CharLm::open(environment, &self.dir, &self.lexicon, shape).expect("the backend opens")
    }
}

#[test]
#[ignore = "needs a resident-pages export: set CHARLM_LM_DIR"]
fn metal_step_matches_cpu_at_every_row_tile() {
    let sweep = Sweep::load();
    let environment = ort::init()
        .build()
        .expect("the test builds its environment");
    let cpu = sweep.open(&environment, SessionShape::default());
    let asked = sweep.asked();

    for metal_weights in [MetalWeights::Auto, MetalWeights::F16] {
        let metal = sweep.open(
            &environment,
            SessionShape {
                backend: Backend::Metal,
                metal_weights,
                ..SessionShape::default()
            },
        );
        for &rows in &SWEEP_ROWS {
            let diff = divergence(&cpu, &metal, &sweep.ids, asked, rows);
            assert!(
                diff < GATE,
                "{metal_weights:?} rows={rows}: metal logp diverges {diff:.3e} past the {GATE} gate"
            );
        }
    }
}

/// Every score `start` and two `advance` calls at `rows` rows answer, as
/// bits — the sequence `divergence` drives, on one backend.
fn score_bits(lm: &CharLm, ids: &[CharId], asked: Asked<'_>, rows: usize) -> Vec<u32> {
    let context = "现在我们在测试输入法的每一步延迟";
    let mut states = vec![lm.start(Some(context), &asked)];
    let mut bits = Vec::new();
    let mut record = |state: &LmState| {
        bits.extend(
            asked
                .candidates
                .iter()
                .map(|&candidate| lm.score(state, candidate).to_bits()),
        );
        bits.push(lm.finish(state).to_bits());
    };
    record(&states[0]);
    for _call in 0..2 {
        while states.len() < rows {
            states.push(states[0].clone());
        }
        states = lm.advance(&divergence_steps(&states, ids, asked));
        for state in &states {
            record(state);
        }
    }
    bits
}

/// The Metal step is a pure function of its inputs: the same calls answer
/// the same bits on a second pass and on a second model — no reduction in
/// the step may depend on the order the GPU schedules its threadgroups.
#[test]
#[ignore = "needs a resident-pages export: set CHARLM_LM_DIR"]
fn metal_scores_repeat_bit_identical() {
    let sweep = Sweep::load();
    let environment = ort::init()
        .build()
        .expect("the test builds its environment");
    let asked = sweep.asked();
    for metal_weights in [MetalWeights::Auto, MetalWeights::F16] {
        let shape = SessionShape {
            backend: Backend::Metal,
            metal_weights,
            ..SessionShape::default()
        };
        let first = sweep.open(&environment, shape);
        let second = sweep.open(&environment, shape);
        for &rows in &SWEEP_ROWS {
            let reference = score_bits(&first, &sweep.ids, asked, rows);
            for pass in 0..4 {
                let lm = if pass % 2 == 0 { &first } else { &second };
                let again = score_bits(lm, &sweep.ids, asked, rows);
                let differing = reference.iter().zip(&again).filter(|(a, b)| a != b).count();
                assert_eq!(
                    differing,
                    0,
                    "{metal_weights:?} rows={rows} pass {pass}: {differing} of {} scores differ",
                    reference.len()
                );
            }
        }
    }
}
