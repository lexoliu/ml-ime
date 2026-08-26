//! The character-to-token alignment g2pW's dataset builds before it can point the
//! network at one position of a sentence.
//!
//! This is a direct port of `g2pw.utils.wordize_and_map` and
//! `g2pw.utils.tokenize_and_map`. It exists because the network is told *which
//! token* to disambiguate, and `WordPiece` does not map one token to one character:
//! a run of Latin letters or digits becomes one word and then several `##`
//! subtokens, and each of those covers a different span of the original string.
//! Get the span arithmetic wrong by one and the network reads the wrong position
//! of a perfectly correct sentence.
//!
//! Everything here indexes by Unicode character, never by byte, because the
//! Python it mirrors does.

use tokenizers::Tokenizer;

/// A sentence split the way g2pW splits it: runs of ASCII alphanumerics stay
/// together, every other character is its own word, and spaces belong to nothing.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Words {
    /// The words, in order.
    pub words: Vec<String>,
    /// For each character position, the word it belongs to. `None` for spaces.
    pub char_to_word: Vec<Option<usize>>,
    /// For each word, the half-open character span it covers.
    pub word_to_char: Vec<(usize, usize)>,
}

/// Split `text` into g2pW's notion of words.
#[must_use]
pub fn wordize(text: &[char]) -> Words {
    let mut words = Vec::new();
    let mut char_to_word: Vec<Option<usize>> = Vec::new();
    let mut word_to_char = Vec::new();

    let mut cursor = 0;
    while cursor < text.len() {
        if text[cursor] == ' ' {
            let start = cursor;
            while cursor < text.len() && text[cursor] == ' ' {
                cursor += 1;
            }
            char_to_word.extend(std::iter::repeat_n(None, cursor - start));
            continue;
        }
        let start = cursor;
        if text[cursor].is_ascii_alphanumeric() {
            while cursor < text.len() && text[cursor].is_ascii_alphanumeric() {
                cursor += 1;
            }
        } else {
            cursor += 1;
        }
        word_to_char.push((start, cursor));
        char_to_word.extend(std::iter::repeat_n(Some(words.len()), cursor - start));
        words.push(text[start..cursor].iter().collect());
    }

    Words {
        words,
        char_to_word,
        word_to_char,
    }
}

/// A sentence's `WordPiece` tokens, and the two maps that line them up with its
/// characters.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Tokens {
    /// The tokens, without `[CLS]`/`[SEP]`.
    pub tokens: Vec<String>,
    /// For each character position, the token that covers it. `None` for spaces
    /// no token claimed.
    pub char_to_token: Vec<Option<usize>>,
    /// For each token, the half-open character span it covers.
    pub token_to_char: Vec<(usize, usize)>,
}

/// Tokenize `text` word by word and align the result back onto its characters.
///
/// Each word is tokenized on its own, exactly as the Python does, so that a word
/// boundary can never be crossed by a subtoken -- which is what makes the span
/// arithmetic below sound.
///
/// # Errors
///
/// If the tokenizer itself fails on a word.
pub fn tokenize_and_map(tokenizer: &Tokenizer, text: &[char]) -> Result<Tokens, tokenizers::Error> {
    let words = wordize(text);
    let mut tokens: Vec<String> = Vec::new();
    let mut token_to_char: Vec<(usize, usize)> = Vec::new();

    for (word, (word_start, word_end)) in words.words.iter().zip(&words.word_to_char) {
        let encoding = tokenizer.encode(word.as_str(), false)?;
        let pieces = encoding.get_tokens();
        if pieces.is_empty() || pieces == ["[UNK]"] {
            token_to_char.push((*word_start, *word_end));
            tokens.push("[UNK]".to_owned());
            continue;
        }
        let mut start = *word_start;
        for piece in pieces {
            let length = piece.strip_prefix("##").unwrap_or(piece).chars().count();
            token_to_char.push((start, start + length));
            start += length;
            tokens.push(piece.clone());
        }
    }

    let mut char_to_token = words.char_to_word;
    for (index, (start, end)) in token_to_char.iter().enumerate() {
        for position in *start..*end {
            if let Some(slot) = char_to_token.get_mut(position) {
                *slot = Some(index);
            }
        }
    }

    Ok(Tokens {
        tokens,
        char_to_token,
        token_to_char,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chars(text: &str) -> Vec<char> {
        text.chars().collect()
    }

    #[test]
    fn han_characters_are_one_word_each() {
        let words = wordize(&chars("中国人"));
        assert_eq!(words.words, ["中", "国", "人"]);
        assert_eq!(words.word_to_char, [(0, 1), (1, 2), (2, 3)]);
        assert_eq!(words.char_to_word, [Some(0), Some(1), Some(2)]);
    }

    #[test]
    fn a_run_of_latin_or_digits_is_one_word() {
        let words = wordize(&chars("用python3写"));
        assert_eq!(words.words, ["用", "python3", "写"]);
        assert_eq!(words.word_to_char, [(0, 1), (1, 8), (8, 9)]);
    }

    #[test]
    fn spaces_belong_to_no_word() {
        let words = wordize(&chars("a  中"));
        assert_eq!(words.words, ["a", "中"]);
        assert_eq!(words.char_to_word, [Some(0), None, None, Some(1)]);
        // The spans are positions in the original string, spaces included, so
        // the second word starts at 3 rather than at 1.
        assert_eq!(words.word_to_char, [(0, 1), (3, 4)]);
    }

    #[test]
    fn punctuation_is_its_own_word() {
        let words = wordize(&chars("好，好"));
        assert_eq!(words.words, ["好", "，", "好"]);
    }

    #[test]
    fn the_word_spans_ignore_the_space_run_the_way_the_python_does() {
        // `wordize_and_map` measures a word's start from the length of the
        // character map, which the space run has already extended -- so the
        // spans are positions in the original string, not in a despaced one.
        let words = wordize(&chars("中 国"));
        assert_eq!(words.word_to_char, [(0, 1), (2, 3)]);
    }
}
