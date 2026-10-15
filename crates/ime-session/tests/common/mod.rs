//! Helpers shared by the `ime-session` integration tests.

use ort::environment::Environment;
use std::sync::OnceLock;

/// The test binary's ONNX Runtime environment. ONNX Runtime allows one
/// environment per process and the harness runs every test of a binary in
/// one process, so the tests share it.
pub fn environment() -> &'static Environment {
    static ENVIRONMENT: OnceLock<Environment> = OnceLock::new();
    ENVIRONMENT.get_or_init(|| {
        ort::init()
            .build()
            .expect("the test binary builds its only environment")
    })
}
