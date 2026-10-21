//! The CPU backend's sessions run on the CPU provider alone.
//!
//! `ort` presets ONNX Runtime's device-selection policy on every session
//! builder, and a session that appends no provider is then placed by ONNX
//! Runtime's device discovery: on Apple silicon it attached Core ML to the
//! CPU backend's prefill and step graphs and split them into a hundred-odd
//! partitions, each compiled to a model file in the user temp directory.
//! The model is opened with verbose logging, and every node-placement line
//! ONNX Runtime writes must put all of a graph's nodes on the CPU.
//!
//! This test is its own binary, so its global subscriber sees every
//! session's placement lines.

mod common;

use common::{environment, fixture_dir, lexicon, shape};
use ime_lm::{CharLm, SessionShape};
use std::fmt;
use std::sync::{Arc, Mutex};
use tracing::field::{Field, Visit};
use tracing::{Event, Subscriber};
use tracing_subscriber::Registry;
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};

/// The node-placement lines ONNX Runtime logged, in order.
#[derive(Clone, Default)]
struct Placements(Arc<Mutex<Vec<String>>>);

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

impl<S> Layer<S> for Placements
where
    S: Subscriber,
{
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        if event.metadata().target() != "ort::logging" {
            return;
        }
        let mut message = Message::default();
        event.record(&mut message);
        if let Some(text) = message.0.filter(|text| text.contains("placed on [")) {
            self.0
                .lock()
                .expect("no recorder panics while holding the lock")
                .push(text.trim().to_owned());
        }
    }
}

#[test]
fn cpu_sessions_place_every_node_on_the_cpu() {
    let placements = Placements::default();
    tracing::subscriber::set_global_default(Registry::default().with(placements.clone()))
        .expect("the test binary installs the only global default");
    tracing::callsite::rebuild_interest_cache();

    let dir = fixture_dir();
    let lexicon = lexicon(&dir);
    let model = CharLm::open(
        environment(),
        &dir.join("transformer"),
        &lexicon,
        SessionShape {
            verbose_logging: true,
            ..shape()
        },
    )
    .expect("the fixture opens");
    drop(model);

    let placements = placements
        .0
        .lock()
        .expect("no recorder panics while holding the lock");
    // A graph split across providers logs one `Node(s) placed on [..]` line
    // per provider; one on a single provider logs one `All nodes` line.
    assert_eq!(
        placements.len(),
        2,
        "one placement line each for the prefill and the step: {placements:#?}"
    );
    for line in placements.iter() {
        assert!(
            line.starts_with("All nodes placed on [CPUExecutionProvider]"),
            "a CPU session placed nodes off the CPU: {placements:#?}"
        );
    }
}
