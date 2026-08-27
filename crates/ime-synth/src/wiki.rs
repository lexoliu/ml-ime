//! Turn the zh-wikipedia slang list's wikitext into `{term, explanation}` pairs.
//!
//! This is the only seed source that carries an explanation, and the grounding
//! rule makes the explanation the whole point: a prompt without one would be
//! asking the model to use a term out of its own memory, which is exactly the
//! failure the decision record forbids. So the parser is strict about the shape
//! it accepts -- a list line whose first thing is a bold term followed
//! immediately by a colon -- and silently ignores every other line rather than
//! guessing.
//!
//! What comes out has to be usable *verbatim* inside a prompt, so the markup is
//! removed rather than escaped: reference tags and the citation templates inside
//! them, wiki links reduced to their display text, bold and italic markers, and
//! HTML comments. Nesting is why this is a scanner and not a pattern match --
//! a `<ref>` routinely contains a `{{cite web}}` which contains a `[[link]]`.

/// One entry of the slang list: the term, and the prose that explains it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct WikiEntry {
    /// The bolded term at the head of the list item.
    pub term: String,
    /// Its explanation, with every trace of wiki markup removed.
    pub explanation: String,
}

/// The bold marker wikipedia writes a term in.
const BOLD: &str = "'''";

/// The italic marker, which the explanations also use.
const ITALIC: &str = "''";

/// Every `*`-prefixed `'''term'''：explanation` line of `wikitext`, in order.
///
/// A term written as `我伙惊/我伙呆` is two entries sharing one explanation,
/// because both spellings are typed and both are grounded by the same prose.
#[must_use]
pub fn entries(wikitext: &str) -> Vec<WikiEntry> {
    let mut found = Vec::new();
    for line in wikitext.lines() {
        let Some((term, explanation)) = entry(line) else {
            continue;
        };
        if explanation.is_empty() {
            continue;
        }
        for variant in term.split(['/', '／']) {
            let variant = variant.trim();
            if !variant.is_empty() {
                found.push(WikiEntry {
                    term: variant.to_owned(),
                    explanation: explanation.clone(),
                });
            }
        }
    }
    found
}

/// Split one list line into its bold term and its cleaned explanation.
fn entry(line: &str) -> Option<(String, String)> {
    let rest = line.trim_start().strip_prefix('*')?.trim_start_matches('*');
    let rest = rest.trim_start().strip_prefix(BOLD)?;
    let end = rest.find(BOLD)?;
    let term = strip(&rest[..end]);
    let tail = rest[end + BOLD.len()..].trim_start();
    let explanation = tail.strip_prefix('：').or_else(|| tail.strip_prefix(':'))?;
    Some((term, strip(explanation)))
}

/// Remove every piece of markup an explanation can carry.
#[must_use]
pub fn strip(wikitext: &str) -> String {
    let without_comments = remove_tags(wikitext, "<!--", "-->");
    let without_refs = remove_refs(&without_comments);
    let without_templates = remove_nested(&without_refs, "{{", "}}");
    let linked = resolve_links(&without_templates);
    collapse(&unbold(&linked))
}

/// Drop `<ref>…</ref>` blocks and self-closing `<ref … />` tags alike.
fn remove_refs(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("<ref") {
        out.push_str(&rest[..start]);
        let after = &rest[start..];
        let Some(open_end) = after.find('>') else {
            return out;
        };
        if after[..open_end].ends_with('/') {
            rest = &after[open_end + 1..];
            continue;
        }
        let Some(close) = after.find("</ref>") else {
            return out;
        };
        rest = &after[close + "</ref>".len()..];
    }
    out.push_str(rest);
    out
}

/// Drop every `open`…`close` span, honouring nesting.
fn remove_nested(text: &str, open: &str, close: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut depth = 0_usize;
    let mut index = 0;
    while index < text.len() {
        if text[index..].starts_with(open) {
            depth += 1;
            index += open.len();
            continue;
        }
        if text[index..].starts_with(close) {
            depth = depth.saturating_sub(1);
            index += close.len();
            continue;
        }
        let Some(character) = text[index..].chars().next() else {
            break;
        };
        if depth == 0 {
            out.push(character);
        }
        index += character.len_utf8();
    }
    out
}

/// Drop every `open`…`close` span without honouring nesting, which comments do not have.
fn remove_tags(text: &str, open: &str, close: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find(open) {
        out.push_str(&rest[..start]);
        let Some(end) = rest[start..].find(close) else {
            return out;
        };
        rest = &rest[start + end + close.len()..];
    }
    out.push_str(rest);
    out
}

/// Reduce `[[target|display]]` to `display` and `[[target]]` to `target`.
fn resolve_links(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("[[") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        let Some(end) = after.find("]]") else {
            out.push_str(after);
            return out;
        };
        let inside = &after[..end];
        let display = inside.rsplit('|').next().unwrap_or(inside);
        out.push_str(display);
        rest = &after[end + 2..];
    }
    out.push_str(rest);
    out
}

/// Remove bold and italic markers, which the explanations use to highlight
/// which character of the term each clause contributes.
fn unbold(text: &str) -> String {
    text.replace(BOLD, "").replace(ITALIC, "")
}

/// Collapse runs of whitespace and trim, so a prompt never carries stray layout.
fn collapse(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut spaced = false;
    for character in text.chars() {
        if character.is_whitespace() {
            spaced = !out.is_empty();
            continue;
        }
        if spaced {
            out.push(' ');
            spaced = false;
        }
        out.push(character);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("../tests/wikitext_fixture.txt");

    #[test]
    fn a_plain_entry_yields_its_term_and_its_explanation() {
        let parsed = entries("*'''滴滴'''：拟声词，属于一种较为温和的打招呼方式。");
        assert_eq!(
            parsed,
            vec![WikiEntry {
                term: "滴滴".to_owned(),
                explanation: "拟声词，属于一种较为温和的打招呼方式。".to_owned(),
            }]
        );
    }

    #[test]
    fn a_reference_and_the_citation_inside_it_leave_nothing_behind() {
        let parsed = entries(
            "*'''YYDS'''：“永远的神”的汉语拼音缩写。<ref name=\"YYDS\">{{cite web|url=http://a|title=[[b|c]]}}</ref>",
        );
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].explanation, "“永远的神”的汉语拼音缩写。");
    }

    #[test]
    fn a_self_closing_reference_is_dropped_without_swallowing_the_rest() {
        let parsed = entries("*'''滴滴'''：温和的打招呼。<ref name=\"ttshow\"/>还可以叠用。");
        assert_eq!(parsed[0].explanation, "温和的打招呼。还可以叠用。");
    }

    #[test]
    fn the_bold_highlights_inside_an_explanation_are_removed_but_their_text_is_kept() {
        let parsed =
            entries("*'''爷青结'''：'''爷'''的'''青'''春'''结'''束了。感慨自己的青春不再。");
        assert_eq!(parsed[0].term, "爷青结");
        assert_eq!(
            parsed[0].explanation,
            "爷的青春结束了。感慨自己的青春不再。"
        );
    }

    #[test]
    fn a_slashed_term_becomes_one_entry_per_spelling_sharing_the_explanation() {
        let parsed = entries("*'''我伙惊/我伙呆'''：我和我的小伙伴们都惊呆了。");
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].term, "我伙惊");
        assert_eq!(parsed[1].term, "我伙呆");
        assert_eq!(parsed[0].explanation, parsed[1].explanation);
    }

    #[test]
    fn a_wiki_link_keeps_only_what_a_reader_would_see() {
        assert_eq!(
            strip("源自[[食神 (電影)|食神]]和[[領悟]]。"),
            "源自食神和領悟。"
        );
    }

    #[test]
    fn a_line_that_is_not_a_bold_term_followed_by_a_colon_is_not_an_entry() {
        assert!(entries("=== 谐音 ===").is_empty());
        assert!(entries("很多网络用语的使用相当普及。").is_empty());
        assert!(entries("*'''没有冒号''' 后面直接是解释").is_empty());
        assert!(entries("*'''空解释'''：").is_empty());
    }

    #[test]
    fn a_leading_space_before_the_bold_term_is_tolerated_the_way_the_list_writes_it() {
        let parsed = entries("* '''你生我梦'''：你的生活我的梦。用来表达强烈的羡慕之情。");
        assert_eq!(parsed[0].term, "你生我梦");
    }

    #[test]
    fn the_fixture_slice_parses_into_exactly_its_entries() {
        let parsed = entries(FIXTURE);
        let terms: Vec<&str> = parsed.iter().map(|entry| entry.term.as_str()).collect();
        assert_eq!(
            terms,
            [
                "Pick",
                "NMSL",
                "滴滴",
                "不明觉厉",
                "我伙惊",
                "我伙呆",
                "蓝瘦香菇"
            ]
        );
        assert!(
            parsed
                .iter()
                .all(|entry| !entry.explanation.contains(['[', ']', '{', '}', '<'])),
            "{parsed:?}"
        );
        assert_eq!(
            parsed[3].explanation,
            "虽然不明白是什么，但是感觉好厉害啊。对于某技术高超者发表的见解表示赞叹。"
        );
    }

    #[test]
    fn an_html_comment_does_not_reach_the_prompt() {
        assert_eq!(strip("解释<!-- 编辑注记 -->继续。"), "解释继续。");
    }

    #[test]
    fn runs_of_whitespace_collapse_to_one_space_and_the_ends_are_trimmed() {
        assert_eq!(collapse("  a\t\tb \n c  "), "a b c");
    }
}
