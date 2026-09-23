//! The lookup tables g2pW disambiguates against.
//!
//! Three of them ship with the `g2pw` Python package rather than with the model
//! archive, so they are vendored under `crates/ime-g2p/data/` (from g2pw 0.1.1)
//! and compiled in: they are what turns a network output index back into a
//! syllable, and a run that silently used a different revision of them would
//! produce labels that no longer match the ones already on disk.
//!
//! The other two -- `POLYPHONIC_CHARS.txt` and `MONOPHONIC_CHARS.txt` -- live
//! beside `g2pw.onnx` in the model directory, because they are versioned with the
//! network's label space.

use crate::error::{Error, Result};
use std::collections::HashMap;
use std::path::Path;

/// Bopomofo syllable to its toneless pinyin spelling, from the g2pw package.
const BOPOMOFO_TO_PINYIN: &str = include_str!("../../data/bopomofo_to_pinyin_wo_tune_dict.json");

/// Every character's readings, used for characters the network was never trained on.
const CHAR_BOPOMOFO: &str = include_str!("../../data/char_bopomofo_dict.json");

/// Simplified to traditional, restricted to what `bert-base-chinese` can tokenize.
const SIMPLIFIED_TO_TRADITIONAL: &str = include_str!("../../data/bert-base-chinese_s2t_dict.txt");

/// How many labels the network's `phoneme_mask` and `probs` tensors are wide.
///
/// Asserted at load rather than trusted: the label list is derived from a text
/// file in the model directory, and if that file and the network ever drift
/// apart every prediction would be quietly off by however many labels moved.
pub const LABELS: usize = 1305;

/// Everything the converter looks a character up in.
#[derive(Debug)]
pub struct Tables {
    labels: Vec<String>,
    char_to_phonemes: HashMap<char, Vec<usize>>,
    char_ids: HashMap<char, usize>,
    monophonic: HashMap<char, String>,
    char_bopomofo: HashMap<char, Vec<String>>,
    bopomofo_to_pinyin: HashMap<String, String>,
    simplified_to_traditional: HashMap<char, char>,
}

impl Tables {
    /// Load the two model-directory tables and combine them with the vendored ones.
    ///
    /// # Errors
    ///
    /// If either table is absent from `model_dir`, unreadable, or has a shape the
    /// network cannot have been trained against.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let polyphonic = read_pairs(&model_dir.join("POLYPHONIC_CHARS.txt"))?;
        let monophonic_pairs = read_pairs(&model_dir.join("MONOPHONIC_CHARS.txt"))?;

        let mut labels: Vec<String> = polyphonic
            .iter()
            .map(|(_, phoneme)| phoneme.clone())
            .collect();
        labels.sort_unstable();
        labels.dedup();
        if labels.len() != LABELS {
            return Err(Error::Tables(format!(
                "POLYPHONIC_CHARS.txt yields {} distinct phonemes, but the network has {LABELS} labels",
                labels.len()
            )));
        }
        let label_ids: HashMap<&str, usize> = labels
            .iter()
            .enumerate()
            .map(|(index, label)| (label.as_str(), index))
            .collect();

        let mut char_to_phonemes: HashMap<char, Vec<usize>> = HashMap::new();
        for (character, phoneme) in &polyphonic {
            let id = *label_ids
                .get(phoneme.as_str())
                .ok_or_else(|| Error::Tables(format!("{phoneme:?} vanished from the label set")))?;
            char_to_phonemes.entry(*character).or_default().push(id);
        }

        // The network conditions on a character index, and the indices are
        // positions in the *sorted* polyphonic vocabulary, so the sort is part
        // of the model contract rather than tidiness.
        let mut chars: Vec<char> = char_to_phonemes.keys().copied().collect();
        chars.sort_unstable();
        let char_ids = chars
            .into_iter()
            .enumerate()
            .map(|(index, character)| (character, index))
            .collect();

        let monophonic = monophonic_pairs.into_iter().collect();

        let bopomofo_to_pinyin: HashMap<String, String> = serde_json::from_str(BOPOMOFO_TO_PINYIN)
            .map_err(|error| {
                Error::Tables(format!("the vendored bopomofo table is broken: {error}"))
            })?;
        let char_bopomofo: HashMap<char, Vec<String>> = parse_char_bopomofo()?;
        let simplified_to_traditional = parse_s2t()?;

        Ok(Self {
            labels,
            char_to_phonemes,
            char_ids,
            monophonic,
            char_bopomofo,
            bopomofo_to_pinyin,
            simplified_to_traditional,
        })
    }

    /// The label at `index`, as the network numbers them.
    #[must_use]
    pub fn label(&self, index: usize) -> Option<&str> {
        self.labels.get(index).map(String::as_str)
    }

    /// How many labels the network chooses between.
    #[must_use]
    pub fn label_count(&self) -> usize {
        self.labels.len()
    }

    /// Whether `character` is one the network disambiguates.
    #[must_use]
    pub fn is_polyphonic(&self, character: char) -> bool {
        self.char_to_phonemes.contains_key(&character)
    }

    /// The label indices `character` is allowed to take, which become its mask.
    #[must_use]
    pub fn phonemes_of(&self, character: char) -> Option<&[usize]> {
        self.char_to_phonemes.get(&character).map(Vec::as_slice)
    }

    /// `character`'s index in the sorted polyphonic vocabulary, which the network
    /// conditions on.
    #[must_use]
    pub fn char_id(&self, character: char) -> Option<usize> {
        self.char_ids.get(&character).copied()
    }

    /// The single reading of a character the network never had to disambiguate.
    #[must_use]
    pub fn monophonic(&self, character: char) -> Option<&str> {
        self.monophonic.get(&character).map(String::as_str)
    }

    /// The fallback reading list for a character in neither table, most common first.
    #[must_use]
    pub fn fallback(&self, character: char) -> Option<&str> {
        self.char_bopomofo
            .get(&character)
            .and_then(|readings| readings.first())
            .map(String::as_str)
    }

    /// The traditional character `character` is written as in the network's training data.
    #[must_use]
    pub fn to_traditional(&self, character: char) -> char {
        self.simplified_to_traditional
            .get(&character)
            .copied()
            .unwrap_or(character)
    }

    /// Spell a tone-numbered bopomofo syllable as tone-numbered pinyin.
    ///
    /// Returns `None` for a syllable the table has no pinyin for, which the
    /// caller turns into a refusal rather than a guess.
    #[must_use]
    pub fn to_pinyin(&self, bopomofo: &str) -> Option<String> {
        let mut characters = bopomofo.chars();
        let tone = characters.next_back()?;
        if !matches!(tone, '1'..='5') {
            tracing::warn!(bopomofo, "the bopomofo syllable has no tone digit");
            return None;
        }
        let component = characters.as_str();
        let Some(pinyin) = self.bopomofo_to_pinyin.get(component) else {
            tracing::warn!(bopomofo, "the bopomofo syllable has no pinyin spelling");
            return None;
        };
        Some(format!("{pinyin}{tone}"))
    }
}

/// Read a two-column tab-separated table the way the Python package does:
/// strip the whole file, split on newlines, split each line on the first tab.
fn read_pairs(path: &Path) -> Result<Vec<(char, String)>> {
    let raw = std::fs::read_to_string(path).map_err(|source| {
        if source.kind() == std::io::ErrorKind::NotFound {
            Error::Missing {
                what: "a g2pW character table",
                path: path.to_owned(),
                hint: "point --g2pw-model at the directory holding g2pw.onnx",
            }
        } else {
            Error::Read {
                path: path.to_owned(),
                source,
            }
        }
    })?;
    let mut pairs = Vec::new();
    for line in raw.trim().lines() {
        let (character, phoneme) = line.split_once('\t').ok_or_else(|| {
            Error::Tables(format!(
                "{} holds an untabbed line {line:?}",
                path.display()
            ))
        })?;
        let mut characters = character.chars();
        let (Some(single), None) = (characters.next(), characters.next()) else {
            return Err(Error::Tables(format!(
                "{} keys a reading on {character:?}, which is not one character",
                path.display()
            )));
        };
        pairs.push((single, phoneme.to_owned()));
    }
    Ok(pairs)
}

fn parse_char_bopomofo() -> Result<HashMap<char, Vec<String>>> {
    let raw: HashMap<String, Vec<String>> =
        serde_json::from_str(CHAR_BOPOMOFO).map_err(|error| {
            Error::Tables(format!("the vendored character table is broken: {error}"))
        })?;
    raw.into_iter()
        .map(|(key, readings)| {
            let mut characters = key.chars();
            match (characters.next(), characters.next()) {
                (Some(single), None) => Ok((single, readings)),
                _ => Err(Error::Tables(format!(
                    "the vendored character table keys a reading on {key:?}"
                ))),
            }
        })
        .collect()
}

fn parse_s2t() -> Result<HashMap<char, char>> {
    let mut table = HashMap::new();
    for line in SIMPLIFIED_TO_TRADITIONAL.trim().lines() {
        let (simplified, traditional) = line
            .split_once('\t')
            .ok_or_else(|| Error::Tables(format!("the vendored s2t table holds {line:?}")))?;
        let mut left = simplified.chars();
        let mut right = traditional.chars();
        match (left.next(), left.next(), right.next(), right.next()) {
            (Some(from), None, Some(to), None) => {
                table.insert(from, to);
            }
            _ => {
                return Err(Error::Tables(format!(
                    "the vendored s2t table maps {simplified:?} to {traditional:?}, which is not a character pair"
                )));
            }
        }
    }
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_vendored_tables_parse() {
        let s2t = parse_s2t().expect("the s2t table parses");
        assert_eq!(s2t.get(&'万'), Some(&'萬'));
        assert_eq!(s2t.get(&'与'), Some(&'與'));

        let chars = parse_char_bopomofo().expect("the character table parses");
        assert_eq!(
            chars.get(&'〇').map(Vec::as_slice),
            Some(
                [
                    "ㄌㄧㄥ2".to_owned(),
                    "ㄩㄢ2".to_owned(),
                    "ㄒㄧㄥ1".to_owned()
                ]
                .as_slice()
            )
        );

        let pinyin: HashMap<String, String> =
            serde_json::from_str(BOPOMOFO_TO_PINYIN).expect("the bopomofo table parses");
        assert_eq!(pinyin.get("ㄌㄧㄥ").map(String::as_str), Some("ling"));
    }
}
