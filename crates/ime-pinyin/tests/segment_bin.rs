//! `ime-segment` over its pipe: one reading list per line, flushed per line,
//! readings that type the same spans collapsed, and a refusal for keys that
//! read no way at all.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

fn segment() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_ime-segment"));
    command
        .args(["--max-paths", "8"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    command
}

#[test]
fn each_line_is_answered_before_the_next_is_sent() {
    let mut child = segment().spawn().expect("ime-segment starts");
    let mut stdin = child.stdin.take().expect("stdin is piped");
    let mut stdout = BufReader::new(child.stdout.take().expect("stdout is piped"));
    let mut ask = |keys: &str| -> Vec<Vec<String>> {
        writeln!(stdin, "{keys}").expect("a request is written");
        stdin.flush().expect("the request is flushed");
        let mut line = String::new();
        stdout.read_line(&mut line).expect("an answer comes back");
        serde_json::from_str(&line).expect("the answer is a JSON reading list")
    };

    let nihao = ask("nihao");
    assert_eq!(nihao[0], ["ni", "hao"]);
    for (index, reading) in nihao.iter().enumerate() {
        assert_eq!(reading.concat(), "nihao");
        assert!(!nihao[..index].contains(reading), "{reading:?} repeats");
    }
    // `m` is a syllable and an initial: one reading here, not two.
    assert_eq!(ask("zgrm"), [["z", "g", "r", "m"]]);

    drop(stdin);
    assert!(child.wait().expect("ime-segment exits").success());
}

#[test]
fn keys_that_read_no_way_fail_loudly() {
    let mut child = segment().spawn().expect("ime-segment starts");
    child
        .stdin
        .take()
        .expect("stdin is piped")
        .write_all(b"nihao1\n")
        .expect("a request is written");
    let output = child.wait_with_output().expect("ime-segment exits");
    assert!(!output.status.success());
    assert_eq!(output.stdout, b"");
    assert!(String::from_utf8_lossy(&output.stderr).contains("admits no reading"));
}
