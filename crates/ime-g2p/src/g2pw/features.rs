//! One network input per polyphonic character position.
//!
//! A port of `g2pw.dataset.TextDataset.__getitem__` and its two truncations. The
//! order of operations is load-bearing and is the order the Python uses: window
//! the *raw* sentence around the query, lowercase the window, tokenize, truncate
//! again if the tokens overflow the network's positions, and only then read the
//! query character out of the lowered window.

use super::tables::Tables;
use super::tokenize::{Tokens, tokenize_and_map};
use crate::error::{Error, Result};
use tokenizers::Tokenizer;

/// How many characters of context the network sees around a query, from
/// `config.py`'s `window_size`.
pub const WINDOW: usize = 32;

/// The longest token sequence the network has positions for, `[CLS]` and `[SEP]`
/// included.
pub const MAX_LEN: usize = 512;

/// One position of one sentence, as the network is asked about it.
#[derive(Clone, PartialEq, Debug)]
pub struct Features {
    /// `[CLS]`, the window's tokens, `[SEP]`.
    pub input_ids: Vec<i64>,
    /// One float per label: 1 where the query character may take that reading.
    pub phoneme_mask: Vec<f32>,
    /// The query character's index in the sorted polyphonic vocabulary.
    pub char_id: i64,
    /// Which token of `input_ids` the query character is, `[CLS]` counted.
    pub position_id: i64,
}

/// Build the network input for `query` -- a character index into `text`.
///
/// # Errors
///
/// If lowercasing changes the character count (which would silently move every
/// later index), if the tokenizer fails, or if the query character turns out not
/// to be one the network disambiguates.
pub fn features(
    tables: &Tables,
    tokenizer: &Tokenizer,
    text: &[char],
    query: usize,
) -> Result<Features> {
    let (window, query) = window(text, query);
    let window = lowercase(&window)?;

    let Tokens {
        tokens,
        char_to_token,
        token_to_char,
    } = tokenize_and_map(tokenizer, &window).map_err(|source| Error::Tokenizer {
        path: std::path::PathBuf::from("<in memory>"),
        source,
    })?;
    let Aligned {
        text: window,
        query,
        tokens,
        char_to_token,
        ..
    } = truncate(
        MAX_LEN,
        Aligned {
            text: window,
            query,
            tokens,
            char_to_token,
            token_to_char,
        },
    );

    let unknown = tokenizer
        .token_to_id("[UNK]")
        .ok_or_else(|| Error::Tables("the BERT vocabulary has no [UNK]".to_owned()))?;
    let mut input_ids = Vec::with_capacity(tokens.len() + 2);
    input_ids.push(i64::from(id_of(tokenizer, "[CLS]", unknown)));
    for token in &tokens {
        input_ids.push(i64::from(id_of(tokenizer, token, unknown)));
    }
    input_ids.push(i64::from(id_of(tokenizer, "[SEP]", unknown)));

    let query_char = *window.get(query).ok_or_else(|| {
        Error::Invariant(format!(
            "the query moved outside the window of {:?}",
            window.iter().collect::<String>()
        ))
    })?;
    let phonemes = tables.phonemes_of(query_char).ok_or_else(|| {
        Error::Invariant(format!(
            "{query_char:?} is not a character the network disambiguates"
        ))
    })?;
    let mut phoneme_mask = vec![0.0_f32; tables.label_count()];
    for index in phonemes {
        if let Some(slot) = phoneme_mask.get_mut(*index) {
            *slot = 1.0;
        }
    }
    let char_id = tables.char_id(query_char).ok_or_else(|| {
        Error::Invariant(format!(
            "{query_char:?} has no index in the polyphonic vocabulary"
        ))
    })?;
    let position = char_to_token.get(query).copied().flatten().ok_or_else(|| {
        Error::Invariant(format!(
            "no token covers {query_char:?} in {:?}",
            window.iter().collect::<String>()
        ))
    })?;

    Ok(Features {
        input_ids,
        phoneme_mask,
        char_id: as_i64(char_id),
        position_id: as_i64(position + 1),
    })
}

fn id_of(tokenizer: &Tokenizer, token: &str, unknown: u32) -> u32 {
    tokenizer.token_to_id(token).unwrap_or(unknown)
}

/// `usize` indices that came out of a sentence at most 512 tokens long always fit.
#[expect(
    clippy::cast_possible_wrap,
    reason = "an index into a windowed sentence cannot reach i64::MAX"
)]
fn as_i64(value: usize) -> i64 {
    value as i64
}

/// The `window_size` characters around `query`, and where the query landed in them.
fn window(text: &[char], query: usize) -> (Vec<char>, usize) {
    let half = WINDOW / 2;
    let start = query.saturating_sub(half);
    let end = (query + half).min(text.len());
    (text[start..end].to_vec(), query - start)
}

/// Lowercase a window, refusing to continue if that moves any later index.
fn lowercase(window: &[char]) -> Result<Vec<char>> {
    let source: String = window.iter().collect();
    let lowered: Vec<char> = source.to_lowercase().chars().collect();
    if lowered.len() != window.len() {
        return Err(Error::Invariant(format!(
            "lowercasing {source:?} changed its character count, so the query position is lost"
        )));
    }
    Ok(lowered)
}

/// A sentence, a query position in it, and its token alignment -- the five things
/// the two truncations move together.
struct Aligned {
    text: Vec<char>,
    query: usize,
    tokens: Vec<String>,
    char_to_token: Vec<Option<usize>>,
    token_to_char: Vec<(usize, usize)>,
}

/// The second truncation: keep the query's token roughly centred in whatever the
/// network has positions for.
///
/// With a 32-character window this never fires, but it is what the Python does
/// and the two implementations have to agree for every input, not just the ones
/// that are convenient.
fn truncate(max_len: usize, aligned: Aligned) -> Aligned {
    let keep = max_len - 2;
    if aligned.tokens.len() <= keep {
        return aligned;
    }
    let Some(Some(position)) = aligned.char_to_token.get(aligned.query).copied() else {
        return aligned;
    };

    let half = isize::try_from(keep / 2).unwrap_or(0);
    let mut token_start = isize::try_from(position).unwrap_or(isize::MAX) - half;
    let mut token_end = token_start + isize::try_from(keep).unwrap_or(0);
    let overshoot_front = -token_start;
    let overshoot_back = token_end - isize::try_from(aligned.tokens.len()).unwrap_or(isize::MAX);
    if overshoot_front > 0 {
        token_start += overshoot_front;
        token_end += overshoot_front;
    } else if overshoot_back > 0 {
        token_start -= overshoot_back;
        token_end -= overshoot_back;
    }
    let token_start = usize::try_from(token_start).unwrap_or(0);
    let token_end = usize::try_from(token_end)
        .unwrap_or(0)
        .min(aligned.tokens.len());

    let start = aligned.token_to_char[token_start].0;
    let end = aligned.token_to_char[token_end - 1].1;

    Aligned {
        text: aligned.text[start..end].to_vec(),
        query: aligned.query - start,
        tokens: aligned.tokens[token_start..token_end].to_vec(),
        char_to_token: aligned.char_to_token[start..end]
            .iter()
            .map(|token| token.map(|index| index - token_start))
            .collect(),
        token_to_char: aligned.token_to_char[token_start..token_end]
            .iter()
            .map(|(from, to)| (from - start, to - start))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chars(text: &str) -> Vec<char> {
        text.chars().collect()
    }

    #[test]
    fn the_window_centres_the_query_and_clamps_at_both_ends() {
        let text = chars(&"中".repeat(100));
        let (windowed, query) = window(&text, 50);
        assert_eq!(windowed.len(), 32);
        assert_eq!(query, 16);

        let (windowed, query) = window(&text, 3);
        assert_eq!(windowed.len(), 19);
        assert_eq!(query, 3);

        let (windowed, query) = window(&text, 98);
        assert_eq!(windowed.len(), 18);
        assert_eq!(query, 16);
    }

    #[test]
    fn a_short_sentence_is_its_own_window() {
        let text = chars("他还了钱");
        let (windowed, query) = window(&text, 1);
        assert_eq!(windowed, text);
        assert_eq!(query, 1);
    }

    #[test]
    fn lowercasing_han_is_the_identity() {
        assert_eq!(
            lowercase(&chars("中Ab国")).expect("stable"),
            chars("中ab国")
        );
    }

    #[test]
    fn the_token_truncation_is_a_no_op_below_the_limit() {
        let tokens = vec!["中".to_owned(), "国".to_owned()];
        let kept = truncate(
            MAX_LEN,
            Aligned {
                text: chars("中国"),
                query: 1,
                tokens: tokens.clone(),
                char_to_token: vec![Some(0), Some(1)],
                token_to_char: vec![(0, 1), (1, 2)],
            },
        );
        assert_eq!(kept.text, chars("中国"));
        assert_eq!(kept.query, 1);
        assert_eq!(kept.tokens, tokens);
    }
}
