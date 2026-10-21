//! What the rescoring binaries share: reading their inputs and reporting a
//! failure on one line.

use std::error::Error;
use std::fs;
use std::path::Path;

/// The file at *path*, or a message naming it as the *what* that could not be
/// read.
///
/// # Errors
///
/// If the file cannot be read as UTF-8.
pub fn read(path: &Path, what: &str) -> Result<String, String> {
    fs::read_to_string(path)
        .map_err(|error| format!("could not read the {what} at {}: {error}", path.display()))
}

/// *error* and every cause behind it, on one line.
#[must_use]
pub fn chain(error: &dyn Error) -> String {
    let mut message = error.to_string();
    let mut cause = error.source();
    while let Some(inner) = cause {
        message.push_str(": ");
        message.push_str(&inner.to_string());
        cause = inner.source();
    }
    message
}
