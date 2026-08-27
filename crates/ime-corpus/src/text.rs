//! Chinese text primitives: normalisation, sentence splitting, and content hashing.
//!
//! These mirror `mlime.data.text` and `mlime.data.corpus.content_id` exactly,
//! because the shards written here are read by the same stages that read the
//! Python pipeline's: a sentence split one character differently, or an
//! identifier hashed over a different byte string, would make one run's samples
//! silently incomparable with another's.
//!
//! Everything here answers one question -- what would a person actually have
//! typed to produce this string? That is why normalisation converts traditional
//! characters to simplified (wiki-style text stores whatever variant its editors
//! used, mixed within a single article) and folds full-width Latin letters and
//! digits while leaving full-width *punctuation* alone: `，`, `！`, `？` and `；`
//! live in the same Unicode block as the letters but are what a Chinese keyboard
//! emits, and the sentence splitter needs them.

use crate::error::{Error, Result};
use blake2::digest::consts::{U12, U16};
use blake2::{Blake2b, Digest as _};
use ferrous_opencc::OpenCC;
use ferrous_opencc::config::BuiltinConfig;
use ime_g2p::text::is_han;
use unicode_normalization::UnicodeNormalization as _;

/// Characters that terminate a sentence, full-width and ASCII alike.
pub const SENTENCE_DELIMITERS: &str = "。！？；!?;";

/// Punctuation that occupies a full-width cell, so no space is ever typed beside it.
const CJK_PUNCTUATION: &str = "。，、；：！？“”‘’（）《》〈〉【】「」『』—…·";

/// The opening halves of the bracket pairs that inline templates leave empty.
const OPENING_BRACKETS: [char; 7] = ['（', '(', '【', '[', '「', '『', '《'];

/// The closing halves, in the same order, so a pair is matched by index.
const CLOSING_BRACKETS: [char; 7] = ['）', ')', '】', ']', '」', '』', '》'];

/// The identifier hash: twelve bytes of `blake2b`.
///
/// Twelve, because the Python pipeline's `hashlib.blake2b(digest_size=12)` wrote
/// the identifiers that are already in `data/`, and an identifier is only useful
/// if the same sample hashes to it on both sides.
type ContentDigest = Blake2b<U12>;

/// The duplicate check's hash: sixteen bytes of `blake2b` over the target alone.
type DedupeDigest = Blake2b<U16>;

/// How many bytes of `blake2b` output the duplicate check keeps.
pub const DEDUPE_DIGEST_BYTES: usize = 16;

/// Whether `character` terminates a sentence.
#[must_use]
pub fn is_sentence_delimiter(character: char) -> bool {
    SENTENCE_DELIMITERS.contains(character)
}

/// Whether `character` occupies a full-width cell, so no space is typed beside it.
fn is_wide(character: char) -> bool {
    is_han(character) || CJK_PUNCTUATION.contains(character)
}

/// Fold a full-width Latin letter, digit or space onto its ASCII form.
///
/// Everything else is left alone, punctuation above all: a Chinese keyboard emits
/// `，` and `。` directly, so folding them would be undoing what the writer typed.
fn fold_fullwidth(character: char) -> char {
    match character {
        '\u{3000}' => ' ',
        '\u{FF10}'..='\u{FF19}' | '\u{FF21}'..='\u{FF3A}' | '\u{FF41}'..='\u{FF5A}' => {
            char::from_u32(character as u32 - 0xFEE0).unwrap_or(character)
        }
        other => other,
    }
}

/// Canonicalises raw corpus text into the form a simplified-Chinese IME types.
///
/// Holds the `OpenCC` converter, which is expensive to build and cheap to reuse, so
/// build one per run and share it; it is [`Sync`], so rayon workers can share one.
pub struct Normalizer {
    to_simplified: OpenCC,
}

impl std::fmt::Debug for Normalizer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Normalizer").finish_non_exhaustive()
    }
}

impl Normalizer {
    /// Build a normaliser around `OpenCC`'s `t2s` conversion chain.
    ///
    /// # Errors
    ///
    /// If the bundled `t2s` configuration cannot be loaded.
    pub fn new() -> Result<Self> {
        let to_simplified = OpenCC::from_config(BuiltinConfig::T2s)
            .map_err(|error| Error::Converter(error.to_string()))?;
        Ok(Self { to_simplified })
    }

    /// Normalise `raw`, preserving newlines as sentence boundaries.
    ///
    /// # Errors
    ///
    /// If the traditional-to-simplified conversion changes the character count,
    /// which means a phrase rule rewrote the text rather than transliterating it
    /// and per-character alignment with the source is lost.
    pub fn normalize(&self, raw: &str) -> Result<String> {
        let folded = fold(raw);
        let converted = self.to_simplified.convert(&folded);
        if converted.chars().count() != folded.chars().count() {
            return Err(Error::Invariant(format!(
                "the t2s conversion changed the character count, so per-character \
                 alignment is lost: {folded:?} -> {converted:?}"
            )));
        }
        Ok(converted.trim().to_owned())
    }
}

/// Everything the normaliser does before the script conversion.
///
/// Kept separate so the length invariant has an un-converted string of the same
/// shape to compare the conversion's output against.
fn fold(raw: &str) -> String {
    let folded: String = raw.nfc().map(fold_fullwidth).collect();
    let collapsed = collapse_horizontal_space(&folded);
    let joined = drop_space_between_wide(&collapsed);
    drop_empty_brackets(&joined)
}

/// Collapse runs of horizontal whitespace to one space and runs of newlines to one.
fn collapse_horizontal_space(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut pending_space = false;
    let mut pending_newline = false;
    for character in text.chars() {
        if character == '\n' {
            pending_space = false;
            pending_newline = true;
            continue;
        }
        if character.is_whitespace() {
            pending_space = true;
            continue;
        }
        if pending_newline {
            out.push('\n');
        } else if pending_space {
            out.push(' ');
        }
        pending_space = false;
        pending_newline = false;
        out.push(character);
    }
    if pending_newline {
        out.push('\n');
    } else if pending_space {
        out.push(' ');
    }
    out
}

/// Drop the spaces between two full-width characters.
///
/// Word-segmented upstreams put a space between every token, and the same spaces
/// appear before their punctuation. Nobody types a space between two full-width
/// characters, so those go -- while `使用 Python` keeps its space, because one
/// side is not full-width.
fn drop_space_between_wide(text: &str) -> String {
    let characters: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut index = 0;
    while index < characters.len() {
        if characters[index] == ' ' {
            let mut end = index;
            while end < characters.len() && characters[end] == ' ' {
                end += 1;
            }
            let before = index.checked_sub(1).map(|at| characters[at]);
            let after = characters.get(end).copied();
            if before.is_some_and(is_wide) && after.is_some_and(is_wide) {
                index = end;
                continue;
            }
            for _ in index..end {
                out.push(' ');
            }
            index = end;
            continue;
        }
        out.push(characters[index]);
        index += 1;
    }
    out
}

/// Drop bracket pairs left empty once inline templates are stripped.
///
/// As in `文学（），在狭义上`: the language annotation is gone but the parentheses
/// stay, and they would drag the Han ratio of an otherwise clean sentence down.
fn drop_empty_brackets(text: &str) -> String {
    let characters: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut index = 0;
    while index < characters.len() {
        if let Some(pair) = OPENING_BRACKETS
            .iter()
            .position(|opening| *opening == characters[index])
        {
            let mut end = index + 1;
            while end < characters.len() && characters[end].is_whitespace() {
                end += 1;
            }
            if characters.get(end) == Some(&CLOSING_BRACKETS[pair]) {
                index = end + 1;
                continue;
            }
        }
        out.push(characters[index]);
        index += 1;
    }
    out
}

/// The sentences of `text`, each keeping its own terminal delimiter.
///
/// Newlines separate sentences and are dropped; the delimiters are kept because a
/// sentence is a unit of *context*, not a target -- a run of them reassembles into
/// readable context, and [`crate::segment`] cuts the targets out of each one.
#[must_use]
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    for line in text.split('\n') {
        let mut current = String::new();
        for character in line.chars() {
            current.push(character);
            if is_sentence_delimiter(character) {
                push_sentence(&mut sentences, &current);
                current.clear();
            }
        }
        push_sentence(&mut sentences, &current);
    }
    sentences
}

fn push_sentence(sentences: &mut Vec<String>, candidate: &str) {
    let trimmed = candidate.trim();
    if !trimmed.is_empty() {
        sentences.push(trimmed.to_owned());
    }
}

/// Fraction of `text`'s non-whitespace characters that are Han. Empty text scores 0.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "a ratio over one sentence's characters is compared against a threshold"
)]
pub fn han_ratio(text: &str) -> f64 {
    let visible = text.chars().filter(|character| !character.is_whitespace());
    let mut total = 0_usize;
    let mut han = 0_usize;
    for character in visible {
        total += 1;
        if is_han(character) {
            han += 1;
        }
    }
    if total == 0 {
        return 0.0;
    }
    han as f64 / total as f64
}

/// A stable identifier derived from the sample's own content.
///
/// The hashed string is `source`, `text` and `context` joined by NUL, which is
/// what the Python pipeline hashes, so the same sample keeps the same identifier
/// whichever implementation wrote it.
#[must_use]
pub fn content_id(source: &str, text: &str, context: Option<&str>) -> String {
    let mut hasher = ContentDigest::new();
    hasher.update(source.as_bytes());
    hasher.update(b"\0");
    hasher.update(text.as_bytes());
    hasher.update(b"\0");
    hasher.update(context.unwrap_or("").as_bytes());
    const_hex::encode(hasher.finalize())
}

/// The digest the duplicate check keys on: sixteen bytes over the target alone.
#[must_use]
pub fn dedupe_digest(text: &str) -> [u8; DEDUPE_DIGEST_BYTES] {
    let mut hasher = DedupeDigest::new();
    hasher.update(text.as_bytes());
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normalizer() -> Normalizer {
        Normalizer::new().expect("the bundled t2s configuration loads")
    }

    #[test]
    fn traditional_text_comes_out_simplified() {
        let normalizer = normalizer();
        assert_eq!(
            normalizer
                .normalize("這個角色的髮色是黑的")
                .expect("converts"),
            "这个角色的发色是黑的"
        );
    }

    #[test]
    fn a_mixed_script_line_comes_out_wholly_simplified() {
        let normalizer = normalizer();
        assert_eq!(
            normalizer
                .normalize("萌娘百科歡迎您參與完善本條目")
                .expect("converts"),
            "萌娘百科欢迎您参与完善本条目"
        );
    }

    #[test]
    fn fullwidth_letters_and_digits_fold_but_punctuation_does_not() {
        let normalizer = normalizer();
        assert_eq!(
            normalizer
                .normalize("使用Ｐｙｔｈｏｎ３，很好！")
                .expect("converts"),
            "使用Python3，很好！"
        );
    }

    #[test]
    fn spaces_between_two_wide_characters_go_and_others_stay() {
        assert_eq!(
            drop_space_between_wide("自贡 哪里 有 好吃 的 鱼"),
            "自贡哪里有好吃的鱼"
        );
        assert_eq!(
            drop_space_between_wide("使用 Python 写的"),
            "使用 Python 写的"
        );
        assert_eq!(drop_space_between_wide("中国 ， 很大"), "中国，很大");
    }

    #[test]
    fn brackets_left_empty_by_a_stripped_template_are_dropped() {
        assert_eq!(drop_empty_brackets("文学（），在狭义上"), "文学，在狭义上");
        assert_eq!(drop_empty_brackets("文学（艺术）"), "文学（艺术）");
        assert_eq!(drop_empty_brackets("test[ ]case"), "testcase");
    }

    #[test]
    fn runs_of_space_collapse_and_runs_of_newline_become_one_boundary() {
        assert_eq!(collapse_horizontal_space("a  \t b\n\n\nc"), "a b\nc");
        assert_eq!(collapse_horizontal_space("  a  "), " a ");
    }

    #[test]
    fn sentences_split_on_the_terminal_punctuation_and_on_newlines() {
        assert_eq!(
            split_sentences("今天天气不错。你说呢？真的！\n新的一行；结束"),
            vec!["今天天气不错。", "你说呢？", "真的！", "新的一行；", "结束"]
        );
    }

    #[test]
    fn ascii_terminators_split_the_same_way_the_fullwidth_ones_do() {
        assert_eq!(
            split_sentences("好的!真的?是;的"),
            vec!["好的!", "真的?", "是;", "的"]
        );
    }

    #[test]
    fn the_han_ratio_ignores_whitespace_and_counts_only_han() {
        assert!((han_ratio("中国 人") - 1.0).abs() < f64::EPSILON);
        assert!((han_ratio("中国人abc") - 0.5).abs() < f64::EPSILON);
        assert!((han_ratio("   ") - 0.0).abs() < f64::EPSILON);
        assert!((han_ratio("中国，") - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn the_content_identifier_is_the_python_pipelines_blake2b_over_the_same_bytes() {
        assert_eq!(content_id("moegirl", "中国", None).len(), 24);
        assert_ne!(
            content_id("moegirl", "中国", None),
            content_id("douyin", "中国", None)
        );
        assert_ne!(
            content_id("moegirl", "中国", None),
            content_id("moegirl", "中国", Some("前文"))
        );
        // hashlib.blake2b("moegirl\x00中国\x00".encode(), digest_size=12).hexdigest()
        assert_eq!(
            content_id("moegirl", "中国", None),
            "7bf00cb09043893252a4358b"
        );
        // hashlib.blake2b("douyin\x00哈哈\x00前文".encode(), digest_size=12).hexdigest()
        assert_eq!(
            content_id("douyin", "哈哈", Some("前文")),
            "3951de975a819f223e501e6b"
        );
    }
}
