//! Chinese text primitives shared by the annotators, the report and the exports.
//!
//! These mirror `mlime.data.text` exactly, because the two implementations write
//! into the same parquet shards and the same evaluation file: a Han test that
//! disagreed by one character, or a `toneless` that folded `ü` differently, would
//! make one side's agreement flags meaningless to the other.

use unicode_script::{Script, UnicodeScript as _};

/// Whether `character` is Han, in the Unicode-script sense rather than a
/// hand-drawn range.
///
/// This is the counterpart of Python's `\p{Han}`, which is also the Script
/// property rather than Script Extensions -- so `，` and `、`, which are
/// `Script=Common`, are *not* Han on either side.
#[must_use]
pub fn is_han(character: char) -> bool {
    character.script() == Script::Han
}

/// The Han characters of `text`, in order, with duplicates kept.
#[must_use]
pub fn han_characters(text: &str) -> Vec<char> {
    text.chars()
        .filter(|character| is_han(*character))
        .collect()
}

/// Strip the tone digit off `syllable` and spell it the way a keyboard does.
///
/// The input method's masks are keyed on what the user types, and no pinyin
/// keyboard has a `ü` key or a tone key, so `lü4`-style output from either
/// annotator has to collapse onto `lv` before the two can be compared.
///
/// This deviates from the Python `toneless`, deliberately. Python folds `ü` and
/// `u:` onto `v` and stops there, which leaves `yu`/`yv` -- and `ju`/`jv`,
/// `qu`/`qv`, `xu`/`xv` -- as two spellings of one syllable. They are not two
/// readings: after `j`, `q`, `x` and `y` there is no plain `u` to be confused
/// with, so orthography writes the `ü` as `u`, and `ime-pinyin`'s syllable table
/// agrees (`yu`, `ju`, `que`, `xue` are syllables; `yv`, `jv`, `qve`, `xve` are
/// not). g2pW spells those `yu`; the prompt tells the LLM to write every `ü` as
/// `v`, so it spells them `yv`. Measured on the 4,000-sentence run, 45 of the
/// 701 "hard" rows are that artefact and nothing else. Folding it here removes
/// the artefact without hiding a real disagreement: after `l` and `n` the fold
/// does *not* apply, so `lu` and `lv` stay the distinct readings of 绿 that they
/// are.
#[must_use]
pub fn toneless(syllable: &str) -> String {
    let lowered = syllable.trim().to_lowercase();
    let stripped = lowered
        .strip_suffix(|character: char| character.is_ascii_digit())
        .unwrap_or(&lowered);
    let folded = stripped.replace('ü', "v").replace("u:", "v");
    if folded.starts_with(['j', 'q', 'x', 'y']) {
        return folded.replace('v', "u");
    }
    folded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn han_is_the_script_property_and_not_the_punctuation_that_shares_its_block() {
        assert!(is_han('中'));
        assert!(is_han('〇'));
        assert!(!is_han('，'));
        assert!(!is_han('。'));
        assert!(!is_han('a'));
        assert!(!is_han('1'));
        assert!(!is_han('ヶ'));
    }

    #[test]
    fn han_characters_keeps_order_and_duplicates() {
        assert_eq!(han_characters("中，中a国"), vec!['中', '中', '国']);
        assert!(han_characters("abc, 123").is_empty());
    }

    #[test]
    fn toneless_drops_the_tone_and_types_u_umlaut_as_v() {
        assert_eq!(toneless("zhong1"), "zhong");
        assert_eq!(toneless("de5"), "de");
        assert_eq!(toneless("lü4"), "lv");
        assert_eq!(toneless("lu:4"), "lv");
        assert_eq!(toneless(" DE5 "), "de");
        assert_eq!(toneless("nve4"), "nve");
    }

    #[test]
    fn toneless_leaves_a_syllable_that_never_carried_a_tone_alone() {
        assert_eq!(toneless("de"), "de");
    }

    #[test]
    fn after_j_q_x_and_y_the_umlaut_is_written_u_so_yv_and_yu_are_one_syllable() {
        assert_eq!(toneless("yv2"), toneless("yu2"));
        assert_eq!(toneless("yv2"), "yu");
        assert_eq!(toneless("jve2"), toneless("jue2"));
        assert_eq!(toneless("qv4"), toneless("qu4"));
        assert_eq!(toneless("xve2"), toneless("xue2"));
        assert_eq!(toneless("yüan2"), "yuan");
    }

    #[test]
    fn after_l_and_n_the_fold_does_not_apply_because_lu_and_lv_are_different_readings() {
        assert_ne!(toneless("lv4"), toneless("lu4"));
        assert_ne!(toneless("nv3"), toneless("nu3"));
        assert_eq!(toneless("lü4"), toneless("lv4"));
        assert_eq!(toneless("lve4"), "lve");
    }
}
