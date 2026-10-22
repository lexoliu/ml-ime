//! Assembles each fixture set under `OUT_DIR/<set>`: the files committed in
//! the set's directory, and beside them the graphs and weights `fixtures.json`
//! pins to a Hugging Face dataset revision, each checked against its SHA-256.
//!
//! A download is kept under `OUT_DIR/blobs/<sha256>` only once it verifies,
//! so a rebuild fetches only what the manifest changed.

use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;
use ureq::Agent;
use ureq::tls::{RootCerts, TlsConfig, TlsProvider};

/// `fixtures.json`: where the pinned files live and what each one hashes to.
#[derive(Deserialize)]
struct Manifest {
    /// The public Hugging Face dataset repository.
    repository: String,
    /// The repository commit every file is fetched at.
    revision: String,
    /// Per set, its committed directory and its pinned files.
    sets: BTreeMap<String, Set>,
}

/// One fixture set.
#[derive(Deserialize)]
struct Set {
    /// The set's committed directory, relative to the workspace root.
    committed: PathBuf,
    /// Each pinned file's path within the set, to its SHA-256.
    files: BTreeMap<String, String>,
}

/// The extensions of the files that leave the tree: a committed directory's
/// files with these are the generator's local output, never copied, so a
/// graph the manifest does not pin is missing rather than silently local.
/// `publish_fixtures.py` uploads the same ones.
const FETCHED: [&str; 2] = ["onnx", "weights"];

/// One request's bound; the largest fixture is a few hundred kilobytes.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

fn main() {
    let crate_dir = PathBuf::from(var("CARGO_MANIFEST_DIR"));
    let workspace = crate_dir.join("../..");
    let out = PathBuf::from(var("OUT_DIR"));
    println!("cargo::rerun-if-changed=fixtures.json");
    let manifest: Manifest = serde_json::from_str(
        &fs::read_to_string(crate_dir.join("fixtures.json")).expect("fixtures.json reads"),
    )
    .expect("fixtures.json parses");

    let blobs = out.join("blobs");
    fs::create_dir_all(&blobs).expect("the blob cache is created");
    // One download per missing blob: files with the same bytes (a graph's
    // two weights files often are) share it.
    let mut missing = BTreeMap::new();
    for (name, set) in &manifest.sets {
        for (file, sha256) in &set.files {
            if !blobs.join(sha256).exists() {
                missing.entry(sha256.as_str()).or_insert_with(|| {
                    format!(
                        "https://huggingface.co/datasets/{}/resolve/{}/{name}/{file}",
                        manifest.repository, manifest.revision
                    )
                });
            }
        }
    }
    let agent = agent();
    // Every missing blob at once: a few dozen small files, whose time is the
    // round trip, so the bound is the manifest's blob count.
    thread::scope(|scope| {
        for (sha256, url) in &missing {
            let blob = blobs.join(sha256);
            let agent = &agent;
            scope.spawn(move || fetch(agent, url, sha256, &blob));
        }
    });

    for (name, set) in &manifest.sets {
        let committed = workspace.join(&set.committed);
        println!("cargo::rerun-if-changed={}", committed.display());
        let assembled = out.join(name);
        if assembled.exists() {
            fs::remove_dir_all(&assembled).expect("the previous assembly is removed");
        }
        copy_committed(&committed, &assembled);
        for (file, sha256) in &set.files {
            let path = assembled.join(file);
            fs::create_dir_all(path.parent().expect("a fixture file has a directory"))
                .expect("the fixture directory is created");
            fs::copy(blobs.join(sha256), &path).expect("a verified blob is copied into its set");
        }
    }
}

/// The environment variable cargo sets for a build script.
fn var(key: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| panic!("cargo sets {key} for a build script"))
}

/// The client: HTTPS only, the platform's verifier, the proxy the
/// environment names.
fn agent() -> Agent {
    Agent::new_with_config(
        Agent::config_builder()
            .proxy(ureq::Proxy::try_from_env())
            .https_only(true)
            .tls_config(
                TlsConfig::builder()
                    .provider(TlsProvider::NativeTls)
                    .root_certs(RootCerts::PlatformVerifier)
                    .build(),
            )
            .timeout_global(Some(REQUEST_TIMEOUT))
            .build(),
    )
}

/// Download *url*, check it hashes to *sha256*, and keep it as *blob*.
fn fetch(agent: &Agent, url: &str, sha256: &str, blob: &Path) {
    let bytes = agent
        .get(url)
        .call()
        .and_then(|mut response| response.body_mut().read_to_vec())
        .unwrap_or_else(|error| panic!("fetching {url}: {error}"));
    let got = const_hex::encode(Sha256::digest(&bytes));
    assert_eq!(
        got, sha256,
        "{url} hashes to {got}, but fixtures.json pins {sha256}"
    );
    // Written whole and renamed, so an interrupted build never leaves a
    // truncated blob that a later one would trust.
    let partial = blob.with_extension("part");
    fs::write(&partial, &bytes).expect("the download is written");
    fs::rename(&partial, blob).expect("the verified download is kept");
}

/// Copy every file under *from* to the same path under *to*, but the
/// extensions in [`FETCHED`].
fn copy_committed(from: &Path, to: &Path) {
    fs::create_dir_all(to).expect("the set directory is created");
    for entry in fs::read_dir(from).expect("the committed fixture directory reads") {
        let path = entry.expect("a directory entry reads").path();
        let destination = to.join(path.file_name().expect("an entry has a name"));
        if path.is_dir() {
            copy_committed(&path, &destination);
        } else if !path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| FETCHED.contains(&extension))
        {
            fs::copy(&path, &destination).expect("a committed fixture file is copied");
        }
    }
}
