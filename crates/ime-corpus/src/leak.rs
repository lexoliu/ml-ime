//! The leakage guard: nothing an evaluation record holds reaches training.
//!
//! Web-scale corpora contain the pages the evaluation sentences were drawn
//! from, and a model that has trained on a record scores it from memory. Two
//! checks keep that out, the ones the #220 survey measured:
//!
//! - **exact**: a target equal to the text of any evaluation or held-out record
//!   is dropped. These are mostly short common phrases (idioms, film titles),
//!   the collisions run3's held-out filter already handled;
//! - **near-duplicate**: a document that shares any [`WINDOW`]-character run of
//!   Han with any record's context followed by its text is dropped whole. Han
//!   only, so punctuation, spacing and Latin differences do not hide a copy; and
//!   the whole document, because a page that holds a record's context holds the
//!   neighbouring sentences the record was cut from as well.
//!
//! A record shorter than [`WINDOW`] Han characters has no window, and only the
//! exact check covers it. Both checks compare normalised text: the evaluation
//! sets were written by the same normaliser this crate runs.

use crate::error::{Error, Result};
use foldhash::fast::RandomState;
use ime_g2p::text::is_han;
use std::collections::HashSet;
use std::io::BufRead as _;
use std::path::{Path, PathBuf};

/// How many consecutive Han characters a shared run needs to count as a copy.
pub const WINDOW: usize = 13;

/// One evaluation or held-out line: its target, and the context it was typed in.
#[derive(serde::Deserialize)]
struct Record {
    text: String,
    #[serde(default)]
    context: Option<String>,
}

/// The texts and the Han windows of every protected record.
pub struct EvalGuard {
    texts: HashSet<String, RandomState>,
    windows: HashSet<[char; WINDOW], RandomState>,
}

impl std::fmt::Debug for EvalGuard {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("EvalGuard")
            .field("texts", &self.texts.len())
            .field("windows", &self.windows.len())
            .finish()
    }
}

impl EvalGuard {
    /// A guard over `(text, context)` records.
    #[must_use]
    pub fn new<'a>(records: impl IntoIterator<Item = (&'a str, Option<&'a str>)>) -> Self {
        let mut texts = HashSet::with_hasher(RandomState::default());
        let mut windows = HashSet::with_hasher(RandomState::default());
        for (text, context) in records {
            texts.insert(text.to_owned());
            let han: Vec<char> = context
                .unwrap_or_default()
                .chars()
                .chain(text.chars())
                .filter(|character| is_han(*character))
                .collect();
            for window in han.windows(WINDOW) {
                let mut key = ['\0'; WINDOW];
                key.copy_from_slice(window);
                windows.insert(key);
            }
        }
        Self { texts, windows }
    }

    /// A guard over every record of every JSON Lines file in `paths`.
    ///
    /// Each line needs a `text` and may carry a `context`; the evaluation sets
    /// carry both, the held-out pools only the text.
    ///
    /// # Errors
    ///
    /// If a file cannot be read, a line is not such a record, or a file holds no
    /// record at all -- a guard that silently protects nothing is the failure
    /// this module exists to prevent.
    pub fn load(paths: &[PathBuf]) -> Result<Self> {
        let mut records: Vec<Record> = Vec::new();
        for path in paths {
            let before = records.len();
            records.extend(read_records(path)?);
            if records.len() == before {
                return Err(Error::Invariant(format!(
                    "{} holds no evaluation record",
                    path.display()
                )));
            }
        }
        Ok(Self::new(records.iter().map(|record| {
            (record.text.as_str(), record.context.as_deref())
        })))
    }

    /// How many distinct protected texts the guard holds.
    #[must_use]
    pub fn texts(&self) -> usize {
        self.texts.len()
    }

    /// How many distinct Han windows the guard holds.
    #[must_use]
    pub fn windows(&self) -> usize {
        self.windows.len()
    }

    /// Whether `target` is the text of a protected record.
    #[must_use]
    pub fn holds_text(&self, target: &str) -> bool {
        self.texts.contains(target)
    }

    /// Whether the units of one document, read in order, share a window with a record.
    ///
    /// The units' Han characters are read as one run, so a copy that straddles a
    /// sentence boundary is found as surely as one inside a sentence.
    #[must_use]
    pub fn shares_window<'a>(&self, units: impl IntoIterator<Item = &'a str>) -> bool {
        let mut ring = ['\0'; WINDOW];
        let mut seen = 0_usize;
        for character in units
            .into_iter()
            .flat_map(str::chars)
            .filter(|character| is_han(*character))
        {
            ring[seen % WINDOW] = character;
            seen += 1;
            if seen >= WINDOW {
                let start = seen % WINDOW;
                let mut key = ['\0'; WINDOW];
                key[..WINDOW - start].copy_from_slice(&ring[start..]);
                key[WINDOW - start..].copy_from_slice(&ring[..start]);
                if self.windows.contains(&key) {
                    return true;
                }
            }
        }
        false
    }
}

fn read_records(path: &Path) -> Result<Vec<Record>> {
    let file = std::fs::File::open(path).map_err(|source| Error::Read {
        path: path.to_owned(),
        source,
    })?;
    let mut records = Vec::new();
    for (index, line) in std::io::BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|source| Error::Read {
            path: path.to_owned(),
            source,
        })?;
        if line.trim().is_empty() {
            continue;
        }
        records.push(
            serde_json::from_str(&line).map_err(|source| Error::JsonLine {
                path: path.to_owned(),
                line: index + 1,
                source,
            })?,
        );
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONTEXT: &str = "今天下午的雨下得特别大啊，这下我们只好躲在书店里等雨停了，";
    const TEXT: &str = "窗外桂花真香";

    fn guard() -> EvalGuard {
        EvalGuard::new([(TEXT, Some(CONTEXT)), ("一张旧车票", None)])
    }

    #[test]
    fn a_target_equal_to_a_record_text_is_caught_and_a_neighbour_is_not() {
        let guard = guard();
        assert!(guard.holds_text("窗外桂花真香"));
        assert!(guard.holds_text("一张旧车票"));
        assert!(!guard.holds_text("二次元真的美"));
    }

    #[test]
    fn a_document_quoting_a_record_across_its_context_and_text_is_caught() {
        let guard = guard();
        // The record's last context clause and its text, re-punctuated and split
        // across two sentences of a page: the Han run is the same.
        let page = ["只好躲在书店里等雨停了！", "窗外桂花真香。别骗自己了。"];
        assert!(guard.shares_window(page));
    }

    #[test]
    fn a_document_sharing_twelve_characters_is_not_a_copy() {
        let guard = guard();
        let han: Vec<char> = CONTEXT.chars().filter(|c| is_han(*c)).collect();
        let twelve: String = han[..WINDOW - 1].iter().collect();
        // Twelve of the record's characters with other Han on both sides.
        assert!(!guard.shares_window([format!("前面{twelve}后来").as_str()]));
        let thirteen: String = han[..WINDOW].iter().collect();
        assert!(guard.shares_window([thirteen.as_str()]));
    }

    #[test]
    fn a_short_record_has_no_window_and_is_left_to_the_exact_check() {
        let guard = EvalGuard::new([("一张旧车票", None)]);
        assert_eq!(guard.windows(), 0);
        assert!(!guard.shares_window(["我找到了一张旧车票给你看看吧好不好呀朋友们"]));
    }

    #[test]
    fn records_load_from_eval_and_pool_files_and_an_empty_file_is_refused() {
        let root = std::env::temp_dir().join(format!("ime-corpus-leak-{}", std::process::id()));
        std::fs::create_dir_all(&root).expect("writable");
        let eval = root.join("eval.jsonl");
        std::fs::write(
            &eval,
            format!(
                "{}\n{}\n",
                serde_json::json!({"pinyin": "x", "text": TEXT, "context": CONTEXT}),
                serde_json::json!({"pinyin": "y", "text": "一张旧车票", "context": null}),
            ),
        )
        .expect("written");
        let pool = root.join("pool.jsonl");
        std::fs::write(
            &pool,
            format!("{}\n", serde_json::json!({"text": "刚出炉的面包"})),
        )
        .expect("written");
        let guard = EvalGuard::load(&[eval, pool]).expect("the records load");
        assert_eq!(guard.texts(), 3);
        assert!(guard.holds_text("刚出炉的面包"));
        assert!(guard.windows() > 0);

        let empty = root.join("empty.jsonl");
        std::fs::write(&empty, "\n").expect("written");
        assert!(EvalGuard::load(&[empty]).is_err());
        std::fs::remove_dir_all(&root).expect("the fixture directory is removed");
    }
}
