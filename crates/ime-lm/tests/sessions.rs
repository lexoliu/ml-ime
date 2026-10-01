//! `ort`'s per-copy identity sessions -- one `+NEW Session` per
//! `copy_into`, ~22 MB of CUDA EP state apiece -- were what drained the
//! T4's 15 GB inside the first `start`. The resident path opens no session
//! a run: on the host-resident backends even a pool growth is a `memcpy`,
//! so the two graph sessions are the whole count, whether 70 `start`s grow
//! the pool or two hundred `start`/`advance` rounds follow.
//!
//! This test is its own binary, so a single global subscriber counts
//! `+NEW Session` events process-wide with no thread filter: `fmt`'s
//! thread-name formatting calls `std::thread::current()`, which aborts on
//! macOS when ONNX Runtime's teardown logs after a thread's locals are
//! already gone.

mod common;

use common::{expected, fixture_dir, lexicon, shape};
use ime_decode::{Asked, Transition};
use ime_lm::{CharLm, LmState};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::field::{Field, Visit};
use tracing::{Event, Subscriber};
use tracing_subscriber::Registry;
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};

/// Counts the `+NEW Session` events the `ort::lifetime` target emits. An
/// atomic behind an `Arc`, and nothing thread-local read -- `on_event`
/// runs on whichever thread emitted the event, including teardown threads.
#[derive(Clone, Default)]
struct SessionCount(Arc<AtomicUsize>);

/// The `message` field of an event, formatted.
#[derive(Default)]
struct Message(Option<String>);

impl Visit for Message {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        if field.name() == "message" {
            self.0 = Some(format!("{value:?}"));
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.0 = Some(value.to_owned());
        }
    }
}

impl<S> Layer<S> for SessionCount
where
    S: Subscriber,
{
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        if event.metadata().target() != "ort::lifetime" {
            return;
        }
        let mut message = Message::default();
        event.record(&mut message);
        // "+NEW SessionBuilder" shares the prefix -- the ` @` keeps the
        // count to committed sessions alone.
        if message
            .0
            .is_some_and(|text| text.contains("+NEW Session @"))
        {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[test]
fn starts_and_steps_do_not_open_new_sessions() {
    let sessions = SessionCount::default();
    tracing::subscriber::set_global_default(Registry::default().with(sessions.clone()))
        .expect("the test binary installs the only global default");
    tracing::callsite::rebuild_interest_cache();

    let dir = fixture_dir();
    let lexicon = lexicon(&dir);
    let model =
        CharLm::open(&dir.join("transformer"), &lexicon, shape()).expect("the fixture opens");
    let character = lexicon
        .id_of(lexicon.characters()[0])
        .expect("the lexicon indexes itself");
    let context = expected(&dir, "transformer").context;
    // Any request serves — the count is about sessions, not scores.
    let asked = Asked {
        candidates: std::slice::from_ref(&character),
        eos: true,
    };

    // Past the pool's starting capacity of 64, so `grow` runs — on the
    // host-resident backend its copies are `memcpy`s, no copier session.
    let mut states: Vec<LmState> = (0..70)
        .map(|_| model.start(Some(&context), &asked))
        .collect();
    drop(states.drain(..));
    for _ in 0..200 {
        let state = model.start(Some(&context), &asked);
        let next = model.advance(&[(&state, character, asked)]);
        drop(next);
        drop(state);
    }
    drop(model);

    assert_eq!(
        sessions.0.load(Ordering::Relaxed),
        2,
        "prefill + step are the whole count on a host-resident backend"
    );
}
