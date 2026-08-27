//! Source-specific residue removal, run before the text is split into sentences.
//!
//! Every rule here exists because a measured sample of one upstream carried
//! something no person ever typed as prose. A flattened infobox line
//! (`发色=黑`) is a template's parameter list, not a sentence; a `#梗` is a
//! platform affordance the poster clicked rather than composed; an `@昵称` is a
//! handle whose characters belong to somebody's screen name rather than to the
//! sentence around it. Left in, each would teach the model to predict text that
//! nobody types into an input method.
//!
//! They are line- and token-level on purpose. The alternative -- letting the Han
//! ratio filter downstream catch them -- throws away the whole sentence a
//! hashtag was appended to, which is exactly the internet-register prose the run
//! exists to collect.

/// Which residues one source carries, and so which strips run over its text.
///
/// A source names its own rules rather than the pipeline guessing: `#` is a
/// hashtag on douyin and a heading marker nowhere else in this corpus, and
/// `key=value` is an infobox on a wiki and an equation anywhere else.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "this is a checklist of independent rules, and each one really is on or off"
)]
pub struct Cleaning {
    /// Drop whole lines that are a flattened infobox row: `<key>=<value>`.
    pub drop_infobox_lines: bool,
    /// Drop `#hashtag` tokens, which run to the next space or `#`.
    pub strip_hashtags: bool,
    /// Drop `@mention` handles.
    pub strip_mentions: bool,
    /// Drop a comment's `回复 @昵称 :` reply header.
    pub strip_reply_prefix: bool,
}

impl Cleaning {
    /// No residue removal at all, for a source that carries none.
    pub const NONE: Self = Self {
        drop_infobox_lines: false,
        strip_hashtags: false,
        strip_mentions: false,
        strip_reply_prefix: false,
    };
}

/// How many lines each line-level rule removed, for the run summary.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct CleaningCounts {
    /// Lines the whole rule saw.
    pub lines: usize,
    /// Lines dropped for being a flattened infobox row.
    pub infobox_lines: usize,
}

impl CleaningCounts {
    /// Fold another document's counts into these.
    pub fn merge(&mut self, other: Self) {
        self.lines += other.lines;
        self.infobox_lines += other.infobox_lines;
    }
}

/// Whether `line` is a flattened infobox row rather than prose.
///
/// The shape is a run of non-whitespace, then `=`: `发色=黑`,
/// `萌点=短发、裸足`. Prose that happens to contain `=` keeps its space or its
/// punctuation before the sign, so `x = 1` and `结论是 a=b` both survive.
#[must_use]
pub fn is_infobox_line(line: &str) -> bool {
    let trimmed = line.trim();
    let Some(key_end) = trimmed.find('=') else {
        return false;
    };
    key_end > 0 && !trimmed[..key_end].chars().any(char::is_whitespace)
}

/// Apply `cleaning` to `raw`, returning the surviving text and what was removed.
#[must_use]
pub fn clean(raw: &str, cleaning: Cleaning) -> (String, CleaningCounts) {
    let mut counts = CleaningCounts::default();
    let mut kept: Vec<String> = Vec::new();
    for line in raw.split('\n') {
        counts.lines += 1;
        if cleaning.drop_infobox_lines && is_infobox_line(line) {
            counts.infobox_lines += 1;
            continue;
        }
        let mut line = line.to_owned();
        if cleaning.strip_reply_prefix {
            line = strip_reply_prefix(&line);
        }
        if cleaning.strip_hashtags {
            line = strip_tagged(&line, '#');
        }
        if cleaning.strip_mentions {
            line = strip_tagged(&line, '@');
        }
        kept.push(line);
    }
    (kept.join("\n"), counts)
}

/// Drop the `回复 @昵称 :` header a threaded comment carries.
///
/// Stripping only the handle would leave `回复 :` in the middle of the target --
/// interface chrome that reads as prose to every filter downstream.
fn strip_reply_prefix(line: &str) -> String {
    let trimmed = line.trim_start();
    let Some(rest) = trimmed.strip_prefix("回复") else {
        return line.to_owned();
    };
    let rest = rest.trim_start();
    if !rest.starts_with('@') {
        return line.to_owned();
    }
    let Some(colon) = rest.find([':', '：']) else {
        return line.to_owned();
    };
    let body = &rest[colon..];
    let mut characters = body.chars();
    characters.next();
    characters.as_str().trim_start().to_owned()
}

/// Drop every `marker`-prefixed token, where a token runs to the next space or marker.
///
/// Chinese hashtags have no word boundary to end on, so the platform ends them at
/// whitespace or at the next `#`; mentions behave the same way. The marker goes
/// with the token, and one space stands in for the whole of it -- including the
/// whitespace that terminated it -- so that the words either side neither fuse
/// nor end up separated by a gap the writer never typed.
fn strip_tagged(line: &str, marker: char) -> String {
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(at) = rest.find(marker) {
        out.push_str(&rest[..at]);
        let after = &rest[at + marker.len_utf8()..];
        let end = after
            .find(|character: char| character.is_whitespace() || character == marker)
            .unwrap_or(after.len());
        if end == 0 {
            // A bare marker with nothing attached is punctuation, not a tag.
            out.push(marker);
            rest = after;
            continue;
        }
        if !out.ends_with(char::is_whitespace) {
            out.push(' ');
        }
        rest = after[end..].trim_start_matches(char::is_whitespace);
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const MOEGIRL: Cleaning = Cleaning {
        drop_infobox_lines: true,
        strip_hashtags: false,
        strip_mentions: false,
        strip_reply_prefix: false,
    };

    const DOUYIN: Cleaning = Cleaning {
        drop_infobox_lines: false,
        strip_hashtags: true,
        strip_mentions: true,
        strip_reply_prefix: false,
    };

    const BILIBILI: Cleaning = Cleaning {
        drop_infobox_lines: false,
        strip_hashtags: false,
        strip_mentions: true,
        strip_reply_prefix: true,
    };

    #[test]
    fn a_flattened_infobox_row_is_recognised_and_prose_with_an_equals_sign_is_not() {
        assert!(is_infobox_line("发色=黑"));
        assert!(is_infobox_line("萌点=短发、裸足、傲娇"));
        assert!(is_infobox_line("  本名=初音未来  "));
        assert!(!is_infobox_line("她的发色是黑色的。"));
        assert!(!is_infobox_line("设 x = 1，则结果为二。"));
        assert!(!is_infobox_line("=开头没有键"));
        assert!(!is_infobox_line(""));
    }

    #[test]
    fn the_infobox_filter_removes_whole_lines_and_counts_them() {
        let raw = "初音未来是虚拟歌手。\n发色=青\n身高=158cm\n她很受欢迎。";
        let (kept, counts) = clean(raw, MOEGIRL);
        assert_eq!(kept, "初音未来是虚拟歌手。\n她很受欢迎。");
        assert_eq!(counts.lines, 4);
        assert_eq!(counts.infobox_lines, 2);
    }

    #[test]
    fn douyin_hashtags_end_at_whitespace_or_the_next_hash() {
        let (kept, _) = clean(
            "这个女人可不好惹！  #GQ说电影  #智取威虎山电影解说 ",
            DOUYIN,
        );
        assert_eq!(kept.trim(), "这个女人可不好惹！");
        let (fused, _) = clean("好看#热门#推荐#搞笑", DOUYIN);
        assert_eq!(fused.trim(), "好看");
    }

    #[test]
    fn a_mention_goes_but_the_sentence_around_it_stays() {
        let (kept, _) = clean("谢谢@小明 的分享，太好了", DOUYIN);
        assert_eq!(kept, "谢谢 的分享，太好了");
    }

    #[test]
    fn a_bare_marker_with_nothing_attached_is_left_as_punctuation() {
        let (kept, _) = clean("价格是 # 号", DOUYIN);
        assert_eq!(kept, "价格是 # 号");
    }

    #[test]
    fn a_bilibili_reply_header_goes_whole_rather_than_leaving_its_chrome() {
        let (kept, _) = clean("回复 @孙子团-小鸭 :哦哦好的谢谢", BILIBILI);
        assert_eq!(kept, "哦哦好的谢谢");
        let (fullwidth, _) = clean("回复 @某人 ：说得对", BILIBILI);
        assert_eq!(fullwidth, "说得对");
    }

    #[test]
    fn a_comment_that_only_looks_like_a_reply_keeps_its_text() {
        let (kept, _) = clean("回复得很快啊", BILIBILI);
        assert_eq!(kept, "回复得很快啊");
    }

    #[test]
    fn cleaning_nothing_leaves_the_text_exactly_as_it_was() {
        let raw = "发色=黑\n#标签 @谁 回复 @人 :话";
        let (kept, counts) = clean(raw, Cleaning::default());
        assert_eq!(kept, raw);
        assert_eq!(counts.infobox_lines, 0);
        assert_eq!(counts.lines, 2);
    }
}
