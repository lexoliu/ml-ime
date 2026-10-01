//! The Metal step against the CPU reference at every row-tile boundary:
//! rows {1, 31, 32, 33, 64, 65, 100}. The 32x64 tiled GEMM's second row
//! tile read past its staged A chunk and wrote into the next tile's rows,
//! so every size past one 32-row tile returned garbage — this asserts each
//! size stays within the accumulated-quantisation band the f16 and naive
//! kernels both show (≤0.25 nats; the gate is 0.5, the bug landed 1e1–4e32).
//!
//! Runs against the real `resident-pages` int8 export the Metal kernels are
//! compiled for — `CHARLM_LM_DIR` must name it. The test is `#[ignore]`d so
//! `cargo test` does not silently pass where the export is absent.

#![cfg(target_os = "macos")]

use ime_decode::{Asked, Transition};
use ime_lm::{Backend, CharLm, LmState, SessionShape};
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

#[test]
#[ignore = "needs a resident-pages int8 export: set CHARLM_LM_DIR"]
fn metal_step_matches_cpu_at_every_row_tile() {
    let dir = PathBuf::from(
        std::env::var_os("CHARLM_LM_DIR")
            .expect("needs a resident-pages int8 export: set CHARLM_LM_DIR"),
    );
    let lexicon = Lexicon::load(&SyllableTable::load()).expect("the lexicon loads");
    let shape = |backend| SessionShape {
        backend,
        ..SessionShape::default()
    };
    let cpu = CharLm::open(&dir, &lexicon, shape(Backend::Cpu)).expect("the cpu backend opens");
    let metal =
        CharLm::open(&dir, &lexicon, shape(Backend::Metal)).expect("the metal backend opens");

    let chars: Vec<char> = "的一了是我不在人们有来他这上着个地到大里说就去子得也和那要下看天时过出小么起你都把好还多没为又可家学只以主会样年想生同老中十从自面前头道它后然走很像见两用她国动进成回什边作对开而己些现山民候经发工向事命给长水几义三声于高手知理眼志点心战二问清身方空写等入死由平员客关先义军文世些边光即今从黑联步至".chars().collect();
    let ids: Vec<CharId> = chars
        .iter()
        .filter_map(|&c| lexicon.id_of(c))
        .take(100)
        .collect();
    assert_eq!(ids.len(), 100, "the sweep alphabet supplies 100 ids");
    let asked = Asked {
        candidates: &ids[..ids.len().min(200)],
        eos: true,
    };

    for &rows in &SWEEP_ROWS {
        let diff = divergence(&cpu, &metal, &ids, asked, rows);
        assert!(
            diff < GATE,
            "rows={rows}: metal logp diverges {diff:.3e} past the {GATE} gate"
        );
    }
}
