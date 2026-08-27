//! Turn a 梗百科 crawl into `{title, explanation}` pairs.
//!
//! The Sogou dictionaries know tens of thousands of internet words and explain
//! none of them, and the grounding rule turns that into a hard ceiling: a term
//! nothing explains is never generated from. 梗百科 is the missing half --
//! 2.3k 梗 entries, each one written to answer the single question "what does
//! this mean" -- so it grounds the Sogou words it shares a title with and seeds
//! the ones it does not.
//!
//! The crawl kept the rendered article rather than its markup, so what has to be
//! removed here is page furniture rather than wikitext: the 目录 block the site
//! prints above the article, the 编辑本段 ("edit this section") link that ends
//! every heading, and the bare view count on the last line. Three article shapes
//! survive that, and all three are seen in the data:
//!
//! * a 目录 block followed by 编辑本段 headings (1008 of 2293 entries),
//! * no 目录, one short heading line, then the prose (1238),
//! * no heading at all, prose from the first line (47).
//!
//! What comes out is one paragraph of prose per entry, because it is going
//! verbatim into a prompt beside 「释义：」. Where the article names its own
//! sections, the section that answers "X是什么梗" is the explanation and the
//! trivia sections are dropped; where it does not, the whole article is, capped
//! at [`MAX_EXPLANATION`] characters and cut at a sentence boundary so the model
//! is never handed half a clause.

use crate::error::{Error, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use tracing::debug;

/// The suffix the site appends to every section heading.
const EDIT_SUFFIX: &str = "编辑本段";

/// The line the 目录 block opens with.
const CONTENTS: &str = "目录";

/// What a heading has to say for its section to be the term's explanation.
const MEANING_MARKERS: [&str; 3] = ["什么梗", "什么意思", "是什么"];

/// The longest a heading line can be before it is prose rather than a title.
const MAX_HEADING: usize = 30;

/// How much explanation one prompt carries, in characters.
///
/// Long enough for the definition and where it came from, short enough that the
/// instructions after it still weigh something.
pub const MAX_EXPLANATION: usize = 600;

/// The punctuation an explanation may be cut after.
const SENTENCE_ENDS: [char; 5] = ['。', '！', '？', '…', '；'];

/// One 梗百科 entry: its title, and the prose that explains it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Entry {
    /// The article title, which is the term itself.
    pub title: String,
    /// The explanation, cleaned of page furniture and capped.
    pub explanation: String,
}

/// One line of the crawl, of which this crate reads two fields.
#[derive(Debug, Deserialize)]
struct Row {
    /// The article title.
    title: String,
    /// The rendered article body.
    content: String,
}

/// Where the 梗百科 crawl lives under a seed root.
#[must_use]
pub fn path(root: &Path) -> PathBuf {
    root.join("gengbaike").join("gengbaike.jsonl")
}

/// Read every entry of a 梗百科 crawl, cleaned and ready to ground a prompt.
///
/// Entries whose article cleans down to nothing are dropped, since an empty
/// explanation grounds nothing.
///
/// # Errors
///
/// If the file cannot be read, or one of its lines is not an entry.
pub fn read(file: &Path) -> Result<Vec<Entry>> {
    let source = std::fs::read_to_string(file).map_err(|source| Error::Read {
        path: file.to_path_buf(),
        source,
    })?;
    let mut entries = Vec::new();
    for (index, line) in source.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: Row = serde_json::from_str(line).map_err(|error| Error::NotGengbaikeEntry {
            path: file.to_path_buf(),
            line: index + 1,
            reason: error.to_string(),
        })?;
        let explanation = explanation(&row.title, &row.content);
        if explanation.is_empty() {
            debug!(title = row.title, "gengbaike entry explains nothing");
            continue;
        }
        entries.push(Entry {
            title: row.title,
            explanation,
        });
    }
    Ok(entries)
}

/// The explanation `content` gives for `title`, as one capped paragraph.
///
/// Empty when the article carries no prose at all.
#[must_use]
pub fn explanation(title: &str, content: &str) -> String {
    let sections = sections(title, content);
    let meaning = sections
        .iter()
        .find(|section| {
            MEANING_MARKERS
                .iter()
                .any(|marker| section.heading.contains(marker))
        })
        .filter(|section| !section.body.is_empty());
    let paragraphs = match meaning {
        Some(section) => section.body.clone(),
        None => sections
            .iter()
            .flat_map(|section| section.body.clone())
            .collect(),
    };
    cap(&paragraphs.join(" "))
}

/// One section of an article: what it is called, and the prose under it.
#[derive(Clone, PartialEq, Eq, Debug)]
struct Section {
    /// The heading, with the 编辑本段 link removed. Empty for the prose that
    /// stands above the first heading, and for articles that have no headings.
    heading: String,
    /// The paragraphs under it, in order.
    body: Vec<String>,
}

/// Split one article into its sections, dropping the page furniture around them.
fn sections(title: &str, content: &str) -> Vec<Section> {
    let lines = body_lines(content);
    let mut sections = Vec::new();
    let mut current = Section {
        heading: String::new(),
        body: Vec::new(),
    };
    for (index, line) in lines.iter().enumerate() {
        match heading_of(index, line, title, lines.len()) {
            Some(heading) => {
                let next = Section {
                    heading: heading.to_owned(),
                    body: Vec::new(),
                };
                sections.push(std::mem::replace(&mut current, next));
            }
            None => current.body.push((*line).to_owned()),
        }
    }
    sections.push(current);
    sections
}

/// The heading `line` is, if it is one.
///
/// Two shapes count. A line ending in the 编辑本段 link is one of the article's
/// own section headings. Failing that, a first line that names the term and asks
/// a question -- 「老哥稳是什么意思」 -- is the title line the headingless shape
/// opens with; it is only read as a heading when prose follows it, so a one-line
/// article stays an explanation rather than becoming an empty section.
fn heading_of<'a>(index: usize, line: &'a str, title: &str, lines: usize) -> Option<&'a str> {
    if let Some(heading) = line.strip_suffix(EDIT_SUFFIX) {
        return Some(heading.trim_end());
    }
    (index == 0
        && lines > 1
        && line.chars().count() <= MAX_HEADING
        && line.contains(title)
        && MEANING_MARKERS.iter().any(|marker| line.contains(marker)))
    .then_some(line)
}

/// The article's paragraphs: no blank lines, no 目录 block, no view count.
fn body_lines(content: &str) -> Vec<&str> {
    let lines: Vec<&str> = content
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    let mut start = 0;
    if lines.first() == Some(&CONTENTS) {
        start = 1;
        while lines.get(start).is_some_and(|line| is_contents_item(line)) {
            start += 1;
        }
    }
    let mut end = lines.len();
    while end > start && is_view_count(lines[end - 1]) {
        end -= 1;
    }
    lines[start..end].to_vec()
}

/// Whether a line is one of the 目录 block's numbered entries.
///
/// They are written 「1人人是什么梗」, and the heading they point at ends with the
/// 编辑本段 link, which is what stops this eating the article itself.
fn is_contents_item(line: &str) -> bool {
    line.starts_with(|character: char| character.is_ascii_digit()) && !line.ends_with(EDIT_SUFFIX)
}

/// Whether a line is the bare view count the crawl kept from the page footer.
fn is_view_count(line: &str) -> bool {
    !line.is_empty() && line.chars().all(|character| character.is_ascii_digit())
}

/// Cut `explanation` to [`MAX_EXPLANATION`] characters at a sentence boundary.
fn cap(explanation: &str) -> String {
    let mut kept = String::new();
    let mut sentence_end = None;
    for character in explanation.chars().take(MAX_EXPLANATION) {
        kept.push(character);
        if SENTENCE_ENDS.contains(&character) {
            sentence_end = Some(kept.len());
        }
    }
    if explanation.chars().count() <= MAX_EXPLANATION {
        return kept;
    }
    match sentence_end {
        Some(end) => kept[..end].to_owned(),
        None => kept,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three real entries, one of each article shape the crawl produced.
    fn fixture() -> Vec<Entry> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/gengbaike_fixture.jsonl");
        read(&path).expect("the fixture is a 梗百科 crawl")
    }

    fn entry(title: &str) -> Entry {
        fixture()
            .into_iter()
            .find(|entry| entry.title == title)
            .unwrap_or_else(|| panic!("{title} is in the fixture"))
    }

    #[test]
    fn an_article_with_a_contents_block_explains_itself_from_its_meaning_section() {
        let entry = entry("人人");
        assert!(
            entry.explanation.starts_with("人人，其实就是代表韩文中的"),
            "{}",
            entry.explanation
        );
        assert!(entry.explanation.ends_with("但着实不可取，有失公正。"));
        assert!(!entry.explanation.contains(CONTENTS));
        assert!(!entry.explanation.contains(EDIT_SUFFIX));
        assert!(
            !entry.explanation.contains("斗鱼主播"),
            "trivia section kept"
        );
        assert!(!entry.explanation.contains('\n'));
    }

    #[test]
    fn an_article_whose_first_line_is_its_title_drops_that_line_and_keeps_the_prose() {
        let entry = entry("老哥稳");
        assert!(
            entry.explanation.starts_with("老哥稳，夸赞对方很牛很厉害"),
            "{}",
            entry.explanation
        );
        assert!(!entry.explanation.contains("老哥稳是什么意思"));
        assert!(entry.explanation.contains("该词最早出现于戒赌吧和足彩吧"));
    }

    #[test]
    fn an_article_that_is_prose_from_its_first_line_keeps_all_of_it() {
        let entry = entry("精苏");
        assert_eq!(
            entry.explanation,
            "愿意指江苏米粉，后被尝用来表示苏联的狂热粉丝。 \
             例：小明特别喜欢苏联的武器，自称是个精苏。 \
             后来一发不可收拾，又出现了精日、精德、精清、精明等等。"
        );
    }

    #[test]
    fn no_entry_keeps_the_footers_view_count() {
        for entry in fixture() {
            assert!(
                !entry
                    .explanation
                    .ends_with(|character: char| character.is_ascii_digit()),
                "{entry:?}"
            );
        }
    }

    #[test]
    fn a_long_explanation_is_cut_after_a_sentence_rather_than_mid_clause() {
        let sentence = "这个梗真的很火。";
        let long: String = std::iter::repeat_n(sentence, 100).collect();
        let capped = cap(&long);
        assert!(capped.chars().count() <= MAX_EXPLANATION);
        assert!(capped.ends_with('。'));
        assert!(capped.chars().count() > MAX_EXPLANATION - sentence.chars().count());
    }

    #[test]
    fn an_explanation_with_no_sentence_end_is_cut_at_the_limit() {
        let long: String = std::iter::repeat_n('梗', MAX_EXPLANATION + 10).collect();
        assert_eq!(cap(&long).chars().count(), MAX_EXPLANATION);
    }

    #[test]
    fn a_line_that_is_not_an_entry_names_the_line_it_is_on() {
        let path =
            std::env::temp_dir().join(format!("ime-synth-gengbaike-{}.jsonl", std::process::id()));
        std::fs::write(&path, "{\"title\": \"人人\"}\n").expect("the fixture is writable");
        let error = read(&path).expect_err("the line has no content field");
        std::fs::remove_file(&path).expect("the fixture is removed");
        assert!(error.to_string().contains("line 1"), "{error}");
    }
}
