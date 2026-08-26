//! The two annotators, tested against the things that actually break them.
//!
//! The LLM annotator is checked against a canned endpoint on a loopback socket
//! rather than a mock client, because the behaviour under test -- one retry, then
//! a refusal, and *exactly* two requests -- is about what goes over the wire.
//!
//! The g2pW tables are checked against the real files in the model directory,
//! since a vendored table that has drifted from the model's is precisely the
//! failure this crate cannot detect any other way. The 606MB network itself is
//! not needed for that, but the two 200KB character tables are, so the test says
//! loudly what it skipped when the directory is absent rather than passing
//! silently.

use askama::Template;
use ime_g2p::g2pw::default_model_dir;
use ime_g2p::g2pw::tables::{LABELS, Tables};
use ime_g2p::llm::{LlmAnnotator, LlmSettings};
use ime_g2p::outcome::Annotator as _;
use std::io::{Read as _, Write as _};
use std::net::TcpListener;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// One canned chat completion, framed as HTTP/1.1 with the CRLFs the protocol wants.
#[derive(Template)]
#[template(path = "canned_http_response.txt", ext = "txt")]
struct CannedResponse {
    length: usize,
    body: String,
}

fn chat_completion(content: &str) -> String {
    let body = serde_json::json!({
        "id": "canned",
        "object": "chat.completion",
        "created": 0,
        "model": "canned",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
    })
    .to_string();
    CannedResponse {
        length: body.len(),
        body,
    }
    .render()
    .expect("the canned response renders")
}

/// A loopback endpoint that answers each request with the next canned reply.
struct FakeEndpoint {
    base_url: String,
    requests: Arc<AtomicUsize>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl FakeEndpoint {
    fn serving(replies: Vec<String>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a loopback port is free");
        let address = listener.local_addr().expect("the socket has an address");
        let requests = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&requests);
        let worker = std::thread::spawn(move || {
            for (reply, stream) in replies.iter().zip(listener.incoming()) {
                let Ok(mut stream) = stream else { break };
                drain_request(&mut stream);
                counter.fetch_add(1, Ordering::SeqCst);
                let _ = stream.write_all(chat_completion(reply).as_bytes());
                let _ = stream.flush();
            }
        });
        Self {
            base_url: format!("http://{address}/v1"),
            requests,
            worker: Some(worker),
        }
    }

    fn settings(&self) -> LlmSettings {
        LlmSettings {
            base_url: self.base_url.clone(),
            api_key: "canned".to_owned(),
            model: "canned".to_owned(),
        }
    }

    fn served(&self) -> usize {
        self.requests.load(Ordering::SeqCst)
    }
}

impl Drop for FakeEndpoint {
    fn drop(&mut self) {
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Read one request off the socket, headers and declared body, so the client is
/// not answered before it has finished speaking.
fn drain_request(stream: &mut std::net::TcpStream) {
    let mut raw = Vec::new();
    let mut byte = [0_u8; 1];
    let mut length = None;
    loop {
        match stream.read(&mut byte) {
            Ok(0) | Err(_) => return,
            Ok(_) => raw.push(byte[0]),
        }
        if length.is_none() && raw.ends_with(b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&raw).to_lowercase();
            length = Some(
                head.lines()
                    .find_map(|line| line.strip_prefix("content-length:"))
                    .and_then(|value| value.trim().parse::<usize>().ok())
                    .unwrap_or(0),
            );
            raw.clear();
        }
        if let Some(wanted) = length
            && raw.len() >= wanted
        {
            return;
        }
    }
}

#[tokio::test]
async fn a_malformed_reply_is_retried_once_and_then_recorded_as_a_refusal() {
    let endpoint = FakeEndpoint::serving(vec![
        "I am afraid I cannot".to_owned(),
        "still not JSON".to_owned(),
    ]);
    let annotator = LlmAnnotator::new(&endpoint.settings(), 4);
    let outcomes = annotator.annotate(&["中国".to_owned()]).await;

    let refusal = outcomes
        .into_iter()
        .next()
        .expect("one sentence, one outcome")
        .expect_err("two malformed replies are a refusal");
    assert!(
        refusal
            .reason
            .starts_with("ValueError: reply is not the expected JSON object"),
        "{}",
        refusal.reason
    );
    assert_eq!(
        endpoint.served(),
        2,
        "one attempt plus exactly one retry, never more"
    );
}

#[tokio::test]
async fn a_retry_that_parses_is_kept_rather_than_refused() {
    let endpoint = FakeEndpoint::serving(vec![
        "{\"readings\": [\"zhong1\"]}".to_owned(),
        "{\"readings\": [\"zhong1\", \"guo2\"]}".to_owned(),
    ]);
    let annotator = LlmAnnotator::new(&endpoint.settings(), 4);
    let outcomes = annotator.annotate(&["中国".to_owned()]).await;

    let reading = outcomes
        .into_iter()
        .next()
        .expect("one sentence, one outcome")
        .expect("the second reply is well formed");
    assert_eq!(reading.syllables, ["zhong1", "guo2"]);
    assert_eq!(endpoint.served(), 2, "the first reply was the wrong length");
}

#[tokio::test]
async fn a_sentence_with_no_han_never_reaches_the_endpoint() {
    let endpoint = FakeEndpoint::serving(Vec::new());
    let annotator = LlmAnnotator::new(&endpoint.settings(), 4);
    let outcomes = annotator.annotate(&["hello, world".to_owned()]).await;

    let refusal = outcomes
        .into_iter()
        .next()
        .expect("one sentence, one outcome")
        .expect_err("there is nothing to read");
    assert_eq!(refusal.reason, "no Han characters to read");
    assert_eq!(endpoint.served(), 0);
}

#[test]
fn the_canned_response_keeps_the_crlfs_http_requires() {
    let framed = chat_completion("hi");
    assert!(framed.starts_with("HTTP/1.1 200 OK\r\n"), "{framed:?}");
    assert!(framed.contains("\r\n\r\n{"), "{framed:?}");
}

/// Load the model directory's character tables, or say loudly why the test is
/// not really running.
fn tables() -> Option<Tables> {
    let Ok(directory) = default_model_dir() else {
        eprintln!("SKIPPED: no home directory, so the g2pW model cache cannot be located");
        return None;
    };
    if !directory.join("POLYPHONIC_CHARS.txt").exists() {
        eprintln!(
            "SKIPPED: {} holds no g2pW character tables, so the table assertions did NOT run",
            directory.display()
        );
        return None;
    }
    Some(Tables::load(&directory).expect("the character tables load"))
}

#[test]
fn a_monophonic_character_is_read_without_asking_the_network() {
    let Some(tables) = tables() else { return };
    assert!(!tables.is_polyphonic('七'));
    let bopomofo = tables.monophonic('七').expect("七 has one reading");
    assert_eq!(bopomofo, "ㄑㄧ1");
    assert_eq!(tables.to_pinyin(bopomofo).as_deref(), Some("qi1"));
    assert_eq!(tables.monophonic('的'), None);
}

#[test]
fn a_polyphonic_character_masks_exactly_the_readings_it_is_allowed() {
    let Some(tables) = tables() else { return };
    assert_eq!(tables.label_count(), LABELS);
    assert!(tables.is_polyphonic('的'));

    let allowed = tables.phonemes_of('的').expect("的 is polyphonic");
    let spelled: Vec<&str> = allowed
        .iter()
        .map(|index| tables.label(*index).expect("every index names a label"))
        .collect();
    assert_eq!(spelled, ["ㄉㄜ5", "ㄉㄧ4", "ㄉㄧ2"]);

    // The mask the network is handed is one float per label, hot exactly where
    // the character is allowed to land -- so a masked-off label can never win.
    let mut mask = vec![0.0_f32; tables.label_count()];
    for index in allowed {
        mask[*index] = 1.0;
    }
    assert_eq!(mask.iter().filter(|value| **value > 0.0).count(), 3);
    for (index, value) in mask.iter().enumerate() {
        assert_eq!(
            *value > 0.0,
            allowed.contains(&index),
            "label {index} is masked wrongly"
        );
    }

    // The character index the network conditions on is a position in the sorted
    // polyphonic vocabulary, so it has to be stable and it has to exist.
    let id = tables.char_id('的').expect("的 has a vocabulary index");
    assert!(id < 10_000);
    assert_eq!(tables.char_id('a'), None);
}

#[test]
fn the_simplified_corpus_is_converted_to_the_traditional_the_network_was_trained_on() {
    let Some(tables) = tables() else { return };
    assert_eq!(tables.to_traditional('万'), '萬');
    assert_eq!(tables.to_traditional('与'), '與');
    assert_eq!(tables.to_traditional('中'), '中');
}
