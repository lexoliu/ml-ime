//! Load the seed terms, and refuse the ones that cannot be grounded or typed.
//!
//! Two gates stand between a lexicon entry and a prompt, and both of them exist
//! to stop the run wasting tokens on something that could never become a usable
//! training sample.
//!
//! The first is the grounding rule from the decision record: a prompt must embed
//! the term's own explanation, so a term without one is skipped rather than
//! generated from the model's own memory. Only the wikipedia list carries
//! explanations, so a Sogou word is grounded exactly when the same word appears
//! there -- and the overwhelming majority do not.
//!
//! The second is coverage. A generated sentence has to survive the same filter
//! the rest of the corpus passes: nine tenths Han, every character readable by
//! `ime-pinyin`. A term like `NMSL` or `Pick` cannot appear in such a sentence at
//! all, so seeding on it guarantees five wasted requests and five validation
//! drops. Those terms are skipped up front, under their own counter, so the drop
//! rate that guards the run measures the *model* rather than the seed list.

use crate::error::{Error, Result};
use crate::source::{SEED_SOURCES, SeedSource};
use crate::wiki;
use ime_corpus::Normalizer;
use ime_g2p::shards::{column_of_strings, read_frame, shard_paths};
use ime_g2p::text::is_han;
use ime_pinyin::Lexicon;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::info;

/// One term to generate from, and the explanation that grounds it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Seed {
    /// The term itself, normalised the way the corpus normalises text.
    pub term: String,
    /// The prose that tells the model what the term means.
    pub explanation: String,
    /// Which seed lexicon it is attributed to.
    pub source: SeedSource,
}

/// How one seed source fared, in the numbers the report prints.
#[derive(Clone, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub struct SeedCounts {
    /// Which source these counts are for.
    pub source: String,
    /// Terms read from it.
    pub loaded: usize,
    /// Terms that reached a prompt.
    pub grounded: usize,
    /// Terms skipped because no explanation could be found for them.
    pub skipped_ungrounded: usize,
    /// Terms skipped because they are not Han the pinyin lexicon can read.
    pub skipped_unusable_term: usize,
    /// Terms skipped because a higher-priority source already claimed them.
    pub skipped_duplicate: usize,
}

/// Every seed the run will generate from, and why the rest were left out.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SeedLoad {
    /// The grounded, typable seeds, in priority order.
    pub seeds: Vec<Seed>,
    /// One entry per seed source, in priority order.
    pub counts: Vec<SeedCounts>,
}

/// Where the Sogou lexicon shards live under a seed root.
#[must_use]
pub fn lexicon_dir(root: &Path) -> PathBuf {
    root.join("lexicon")
}

/// Where the wikipedia slang list's `MediaWiki` parse response lives under a seed root.
#[must_use]
pub fn wiki_path(root: &Path) -> PathBuf {
    root.join("slang").join("wangluo-yongyu.json")
}

/// The one field of a `MediaWiki` `action=parse` response this crate reads.
#[derive(Debug, Deserialize)]
struct ParseResponse {
    parse: ParseBody,
}

/// The parse body, which carries the article's source.
#[derive(Debug, Deserialize)]
struct ParseBody {
    wikitext: String,
}

/// Read the wikipedia slang list and parse it into its entries.
///
/// # Errors
///
/// If the file is absent, unreadable, or not a `MediaWiki` parse response.
pub fn read_wiki_entries(root: &Path) -> Result<Vec<wiki::WikiEntry>> {
    let path = wiki_path(root);
    if !path.exists() {
        return Err(Error::Missing {
            what: "the wikipedia slang list",
            path,
            hint: "fetch 中国大陆网络用语列表 through the MediaWiki parse API first",
        });
    }
    let source = std::fs::read_to_string(&path).map_err(|source| Error::Read {
        path: path.clone(),
        source,
    })?;
    let response: ParseResponse =
        serde_json::from_str(&source).map_err(|error| Error::NotWikiParse {
            path: path.clone(),
            reason: error.to_string(),
        })?;
    Ok(wiki::entries(&response.parse.wikitext))
}

/// Read one Sogou dictionary's words, in the rank order they were written.
///
/// # Errors
///
/// If the dictionary has no shards, or one of them cannot be read.
pub fn read_lexicon_words(root: &Path, source: SeedSource) -> Result<Vec<String>> {
    let Some(slug) = source.slug else {
        return Err(Error::Invariant(format!(
            "{} is not a lexicon source and has no shards to read",
            source.name
        )));
    };
    let directory = lexicon_dir(root);
    let paths = shard_paths(&directory, slug)?;
    if paths.is_empty() {
        return Err(Error::Missing {
            what: "Sogou lexicon shards",
            path: directory.join(format!("{slug}-*.parquet")),
            hint: "run `uv run mlime lexicon fetch` under python/ first",
        });
    }
    let mut words = Vec::new();
    for path in &paths {
        words.extend(column_of_strings(&read_frame(path)?, "word")?);
    }
    Ok(words)
}

/// Whether a term can appear inside a sentence the corpus filter would keep.
#[must_use]
pub fn is_typable(term: &str, lexicon: &Lexicon) -> bool {
    !term.is_empty()
        && term
            .chars()
            .all(|character| is_han(character) && lexicon.id_of(character).is_some())
}

/// Load every seed source, ground each term, and attribute it to one source.
///
/// The sources are walked in [`SEED_SOURCES`] order, so a term carried by more
/// than one is attributed to the highest-priority source that has it and
/// generated from exactly once.
///
/// # Errors
///
/// If a seed file is missing or malformed, or if the normaliser refuses a term.
pub fn load(root: &Path, normalizer: &Normalizer, lexicon: &Lexicon) -> Result<SeedLoad> {
    let entries = read_wiki_entries(root)?;
    let mut explanations: HashMap<String, String> = HashMap::with_capacity(entries.len());
    let mut wiki_terms: Vec<String> = Vec::with_capacity(entries.len());
    for entry in &entries {
        let term = normalizer.normalize(&entry.term)?;
        if explanations
            .insert(term.clone(), normalizer.normalize(&entry.explanation)?)
            .is_none()
        {
            wiki_terms.push(term);
        }
    }

    let mut seeds = Vec::new();
    let mut counts = Vec::with_capacity(SEED_SOURCES.len());
    let mut claimed: HashMap<String, &'static str> = HashMap::new();
    for source in SEED_SOURCES {
        let terms = match source.slug {
            Some(_) => read_lexicon_words(root, source)?
                .iter()
                .map(|word| normalizer.normalize(word).map_err(Error::from))
                .collect::<Result<Vec<String>>>()?,
            None => wiki_terms.clone(),
        };
        let mut count = SeedCounts {
            source: source.name.to_owned(),
            loaded: terms.len(),
            ..SeedCounts::default()
        };
        for term in terms {
            if claimed.contains_key(&term) {
                count.skipped_duplicate += 1;
                continue;
            }
            let Some(explanation) = explanations.get(&term) else {
                count.skipped_ungrounded += 1;
                continue;
            };
            if !is_typable(&term, lexicon) {
                count.skipped_unusable_term += 1;
                continue;
            }
            claimed.insert(term.clone(), source.name);
            count.grounded += 1;
            seeds.push(Seed {
                term,
                explanation: explanation.clone(),
                source,
            });
        }
        info!(
            source = source.name,
            loaded = count.loaded,
            grounded = count.grounded,
            ungrounded = count.skipped_ungrounded,
            unusable = count.skipped_unusable_term,
            duplicate = count.skipped_duplicate,
            "seed source loaded"
        );
        counts.push(count);
    }
    Ok(SeedLoad { seeds, counts })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ime_pinyin::SyllableTable;

    fn lexicon() -> Lexicon {
        Lexicon::load(&SyllableTable::load()).expect("the generated pinyin tables agree")
    }

    #[test]
    fn a_term_of_readable_han_is_typable_and_anything_else_is_not() {
        let lexicon = lexicon();
        assert!(is_typable("爷青结", &lexicon));
        assert!(!is_typable("NMSL", &lexicon));
        assert!(!is_typable("我伙惊/我伙呆", &lexicon));
        assert!(!is_typable("", &lexicon));
        assert!(!is_typable("\u{2A6B2}", &lexicon));
    }
}
