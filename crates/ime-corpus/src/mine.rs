//! Mining web text for the lexicon entries run3 never shows the model.
//!
//! #215 found 24,643 idiom and internet entries of the knowledge lexicon that
//! occur nowhere in run3 (blocked entries aside). Web text holds most of them,
//! but #220's survey found that most web hits on a rare idiom are not anyone
//! using it: a spinner swaps common words for rare idioms to disguise a copied
//! article (判若鸿沟的反差), and SEO pages string idioms together as filler. So a
//! hit is kept only when it passes two gates:
//!
//! - **the sentence** ([`EntryMatcher::hits`]): the target is at least three
//!   characters longer than the entry, lexicon entries cover at most half of it,
//!   and a function character sits outside the entry -- the survey's heuristic;
//! - **the document** ([`DocumentProfile::spun`]): a page is stuffed when
//!   idioms are dense in it, and spun when it holds several absent entries at
//!   once or when its character pairs are ones no run3 sentence ever writes.
//!
//! Each entry keeps at most [`PER_ENTRY_CAP`] sentences over the whole pass.
//! What passes is still only a candidate: `mlime corpus judge` drops word-boundary
//! false matches and asks two models whether the sentence and its page read as
//! natural human writing before any of it is trained on.

use crate::error::{Error, Result};
use crate::text::Normalizer;
use aho_corasick::{AhoCorasick, MatchKind};
use ferrous_opencc::OpenCC;
use ferrous_opencc::config::BuiltinConfig;
use foldhash::fast::RandomState;
use ime_g2p::text::is_han;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// The most sentences one entry keeps over the whole mining pass.
pub const PER_ENTRY_CAP: usize = 50;

/// Characters whose presence outside the entry marks a sentence as ordinary prose.
const FUNCTION: &str =
    "的了是在不我你他她它这那有就也都而和与把被让给着过地得之其以为所很还又会能要说没吗呢吧啊";

/// The most idiom occurrences a document may hold per hundred Han characters.
pub const MAX_IDIOMS_PER_HUNDRED: f64 = 1.5;

/// The fewest idiom occurrences that can make a document stuffed.
///
/// A one-line reply that uses one idiom is dense by arithmetic, not by stuffing.
pub const MIN_STUFFED_IDIOMS: usize = 4;

/// The most distinct absent entries one document may hold before it reads as spun.
pub const MAX_ABSENT_PER_DOCUMENT: usize = 2;

/// The run3 counts at or under which a character pair is rare, for the profile.
pub const RARE_PAIR_COUNTS: [u32; 3] = [0, 3, 30];

/// The largest share of a document's Han character pairs run3 never writes.
pub const MAX_NOVEL_PAIR_SHARE: f64 = 0.03;

/// What kind of lexicon entry a word is, as `knowledge.tsv` files it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Category {
    /// A 成语 or set phrase.
    Idiom,
    /// An internet meme or slang term.
    Internet,
    /// A person, place or organisation.
    Name,
}

/// One knowledge-lexicon word.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Word {
    /// The word itself.
    pub text: String,
    /// What kind of entry it is.
    pub category: Category,
}

/// One entry the pass is mining for.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Target {
    /// The entry itself.
    pub entry: String,
    /// Idiom or internet.
    pub category: Category,
}

/// The knowledge lexicon, the targets among it, and a matcher over both.
pub struct EntryMatcher {
    words: Vec<String>,
    categories: Vec<Category>,
    /// For each lexicon word, its index among the targets when it is one.
    target_of: Vec<Option<u32>>,
    targets: Vec<Target>,
    all: AhoCorasick,
    /// Every target in simplified and traditional form, for the raw-text prefilter.
    raw: AhoCorasick,
    /// Targets the normaliser rewrites, which no normalised sentence can hold.
    unminable: Vec<Target>,
}

impl std::fmt::Debug for EntryMatcher {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("EntryMatcher")
            .field("words", &self.words.len())
            .field("targets", &self.targets.len())
            .finish_non_exhaustive()
    }
}

/// One target entry found in one target sentence, and whether the sentence passes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct EntryHit {
    /// Index of the entry among the targets.
    pub target: u32,
    /// Whether the sentence passes the sentence gate for this entry.
    pub usable: bool,
}

impl EntryMatcher {
    /// Read #215's entry table and build the matchers.
    ///
    /// The table has a header, then one row per knowledge-lexicon entry: entry,
    /// category, run3 occurrences, sentences, two exposure figures and the
    /// blocked flag. Every row is a lexicon word; a target is an idiom or
    /// internet entry with no occurrence in run3 that is not blocked.
    ///
    /// A target the normaliser rewrites (不乾不净 becomes 不干不净) can never
    /// match normalised text in its listed form, so it is set aside as
    /// [`Self::unminable`] rather than mined for.
    ///
    /// # Errors
    ///
    /// If the table cannot be read or has another shape, or if there are no targets.
    pub fn load(entries: &Path, normalizer: &Normalizer) -> Result<Self> {
        let table = std::fs::read_to_string(entries).map_err(|source| Error::Read {
            path: entries.to_owned(),
            source,
        })?;
        let mut lines = table.lines();
        let header = lines.next().unwrap_or_default();
        if !header.starts_with("entry\tcategory\toccurrences") || !header.ends_with("\tblocked") {
            return Err(Error::Invariant(format!(
                "{} is not #215's entry table: its header is {header:?}",
                entries.display()
            )));
        }
        let mut lexicon = Vec::new();
        let mut targets = Vec::new();
        for line in lines.filter(|line| !line.is_empty()) {
            let fields: Vec<&str> = line.split('\t').collect();
            let [entry, category, occurrences, _, _, _, blocked] = fields[..] else {
                return Err(Error::Invariant(format!(
                    "{} has a row of {} fields: {line:?}",
                    entries.display(),
                    fields.len()
                )));
            };
            let category = parse_category(category);
            let occurrences: u32 = occurrences.parse().map_err(|error| {
                Error::Invariant(format!(
                    "{} has a bad occurrence count in {line:?}: {error}",
                    entries.display()
                ))
            })?;
            if category != Category::Name && occurrences == 0 && blocked == "0" {
                targets.push(Target {
                    entry: entry.to_owned(),
                    category,
                });
            }
            lexicon.push(Word {
                text: entry.to_owned(),
                category,
            });
        }
        let mut unminable = Vec::new();
        let mut minable = Vec::with_capacity(targets.len());
        for target in targets {
            if normalizer.normalize(&target.entry)? == target.entry {
                minable.push(target);
            } else {
                unminable.push(target);
            }
        }
        let mut matcher = Self::new(lexicon, minable)?;
        matcher.unminable = unminable;
        Ok(matcher)
    }

    /// Build the matchers over `lexicon`, mining for `targets`.
    ///
    /// # Errors
    ///
    /// If a target is not in the lexicon, if there are no targets, or if the
    /// matchers cannot be built.
    pub fn new(lexicon: Vec<Word>, targets: Vec<Target>) -> Result<Self> {
        if targets.is_empty() {
            return Err(Error::Invariant("there is no entry to mine for".to_owned()));
        }
        let position: HashMap<&str, u32> = targets
            .iter()
            .enumerate()
            .map(|(index, target)| (target.entry.as_str(), index_u32(index)))
            .collect();
        let target_of: Vec<Option<u32>> = lexicon
            .iter()
            .map(|word| position.get(word.text.as_str()).copied())
            .collect();
        let found = target_of.iter().flatten().count();
        if found != targets.len() {
            return Err(Error::Invariant(format!(
                "{} of {} target entries are not in the lexicon",
                targets.len() - found,
                targets.len()
            )));
        }
        let categories: Vec<Category> = lexicon.iter().map(|word| word.category).collect();
        let words: Vec<String> = lexicon.into_iter().map(|word| word.text).collect();
        let all = AhoCorasick::builder()
            .match_kind(MatchKind::Standard)
            .build(&words)
            .map_err(|error| Error::Invariant(format!("the lexicon matcher: {error}")))?;
        let mut forms: Vec<String> = Vec::new();
        for config in [BuiltinConfig::S2t, BuiltinConfig::S2tw] {
            let converter =
                OpenCC::from_config(config).map_err(|error| Error::Converter(error.to_string()))?;
            forms.extend(
                targets
                    .iter()
                    .map(|target| converter.convert(&target.entry)),
            );
        }
        forms.extend(targets.iter().map(|target| target.entry.clone()));
        forms.sort_unstable();
        forms.dedup();
        let raw = AhoCorasick::builder()
            .match_kind(MatchKind::LeftmostFirst)
            .build(&forms)
            .map_err(|error| Error::Invariant(format!("the prefilter matcher: {error}")))?;
        Ok(Self {
            words,
            categories,
            target_of,
            targets,
            all,
            raw,
            unminable: Vec::new(),
        })
    }

    /// The entries being mined for, in index order.
    #[must_use]
    pub fn targets(&self) -> &[Target] {
        &self.targets
    }

    /// The entries set aside because the normaliser rewrites them.
    #[must_use]
    pub fn unminable(&self) -> &[Target] {
        &self.unminable
    }

    /// Whether raw, unnormalised text could hold a target at all.
    ///
    /// Normalising a document (the traditional-to-simplified conversion above
    /// all) is most of the work of a pass, and almost no document holds an
    /// absent entry. Every target is matched in its simplified and two
    /// traditional spellings, so only a document written in some third variant
    /// is passed over.
    #[must_use]
    pub fn may_hold_target(&self, raw: &str) -> bool {
        self.raw.is_match(raw)
    }

    /// The target entries `sentence` holds, each with its sentence-gate verdict.
    ///
    /// `sentence` is one typing segment, wholly Han. One verdict per entry: an
    /// entry that occurs twice is usable when either occurrence is.
    #[must_use]
    pub fn hits(&self, sentence: &str) -> Vec<EntryHit> {
        let matches: Vec<(usize, usize, usize)> = self
            .all
            .find_overlapping_iter(sentence)
            .map(|found| (found.pattern().as_usize(), found.start(), found.end()))
            .collect();
        if !matches
            .iter()
            .any(|(word, _, _)| self.target_of[*word].is_some())
        {
            return Vec::new();
        }
        let mut covered = vec![false; sentence.len()];
        for &(_, start, end) in &matches {
            covered[start..end].iter_mut().for_each(|byte| *byte = true);
        }
        let dense = covered.iter().filter(|byte| **byte).count() * 2 > sentence.len();
        let length = sentence.chars().count();
        let mut hits: Vec<EntryHit> = matches
            .iter()
            .filter_map(|&(word, start, end)| {
                let target = self.target_of[word]?;
                let entry_length = sentence[start..end].chars().count();
                let outside = sentence[..start].chars().chain(sentence[end..].chars());
                let usable = length >= entry_length + 3
                    && !dense
                    && outside
                        .into_iter()
                        .any(|character| FUNCTION.contains(character));
                Some(EntryHit { target, usable })
            })
            .collect();
        hits.sort_unstable_by(|left, right| {
            left.target
                .cmp(&right.target)
                .then(right.usable.cmp(&left.usable))
        });
        hits.dedup_by_key(|hit| hit.target);
        hits
    }

    /// The figures the document gate reads, over a document's units.
    #[must_use]
    pub fn profile<'a>(
        &self,
        units: impl IntoIterator<Item = &'a str> + Clone,
        pairs: &CharacterPairs,
    ) -> DocumentProfile {
        let mut han = 0_usize;
        let mut idioms = 0_usize;
        let mut absent: HashSet<u32, RandomState> = HashSet::with_hasher(RandomState::default());
        let mut pair_count = 0_usize;
        let mut rare = [0_usize; RARE_PAIR_COUNTS.len()];
        for unit in units {
            for found in self.all.find_overlapping_iter(unit) {
                let word = found.pattern().as_usize();
                if self.categories[word] == Category::Idiom {
                    idioms += 1;
                }
                if let Some(target) = self.target_of[word] {
                    absent.insert(target);
                }
            }
            let mut previous: Option<char> = None;
            for character in unit.chars() {
                if !is_han(character) {
                    previous = None;
                    continue;
                }
                han += 1;
                if let Some(before) = previous {
                    pair_count += 1;
                    let count = pairs.count(before, character);
                    for (rare, ceiling) in rare.iter_mut().zip(RARE_PAIR_COUNTS) {
                        if count <= ceiling {
                            *rare += 1;
                        }
                    }
                }
                previous = Some(character);
            }
        }
        DocumentProfile {
            han,
            idioms,
            absent: absent.len(),
            pairs: pair_count,
            rare_pairs: rare,
        }
    }
}

fn parse_category(category: &str) -> Category {
    match category {
        "idiom" => Category::Idiom,
        "internet" => Category::Internet,
        _ => Category::Name,
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the lexicon has about a hundred thousand entries"
)]
fn index_u32(index: usize) -> u32 {
    index as u32
}

/// What the document gate reads off one document.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, serde::Serialize)]
pub struct DocumentProfile {
    /// Han characters in the document.
    pub han: usize,
    /// Idiom occurrences, overlapping matches counted.
    pub idioms: usize,
    /// Distinct absent entries the document holds.
    pub absent: usize,
    /// Adjacent Han character pairs.
    pub pairs: usize,
    /// Of those, the pairs run3's targets hold at most [`RARE_PAIR_COUNTS`] times.
    pub rare_pairs: [usize; RARE_PAIR_COUNTS.len()],
}

impl DocumentProfile {
    /// Idiom occurrences per hundred Han characters.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "a density compared against a threshold"
    )]
    pub fn idioms_per_hundred(&self) -> f64 {
        if self.han == 0 {
            return 0.0;
        }
        self.idioms as f64 * 100.0 / self.han as f64
    }

    /// The share of the document's character pairs run3 holds at most
    /// `RARE_PAIR_COUNTS[level]` times.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "a share compared against a threshold"
    )]
    pub fn rare_pair_share(&self, level: usize) -> f64 {
        if self.pairs == 0 {
            return 0.0;
        }
        self.rare_pairs[level] as f64 / self.pairs as f64
    }

    /// The share of the document's character pairs run3 never writes.
    #[must_use]
    pub fn novel_pair_share(&self) -> f64 {
        self.rare_pair_share(0)
    }

    /// Why the document reads as stuffed or spun, if it does.
    #[must_use]
    pub fn spun(&self) -> Option<SpunReason> {
        if self.idioms >= MIN_STUFFED_IDIOMS && self.idioms_per_hundred() > MAX_IDIOMS_PER_HUNDRED {
            Some(SpunReason::IdiomDensity)
        } else if self.absent > MAX_ABSENT_PER_DOCUMENT {
            Some(SpunReason::AbsentEntries)
        } else if self.novel_pair_share() > MAX_NOVEL_PAIR_SHARE {
            Some(SpunReason::NovelPairs)
        } else {
            None
        }
    }
}

/// Why a document's hits were refused.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SpunReason {
    /// Idioms are denser than any prose writes them: keyword stuffing.
    IdiomDensity,
    /// More absent entries than one honest page uses: a spinner's substitutions.
    AbsentEntries,
    /// Character pairs nobody writes: a spinner's synonym swaps.
    NovelPairs,
}

/// How often each adjacent pair of Han characters occurs in run3's targets.
pub struct CharacterPairs {
    counts: HashMap<u64, u32, RandomState>,
}

impl std::fmt::Debug for CharacterPairs {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CharacterPairs")
            .field("pairs", &self.counts.len())
            .finish()
    }
}

impl CharacterPairs {
    /// The pair counts of every text in `texts`.
    #[must_use]
    pub fn new<'a>(texts: impl IntoIterator<Item = &'a str>) -> Self {
        let mut counts: HashMap<u64, u32, RandomState> =
            HashMap::with_hasher(RandomState::default());
        for text in texts {
            for pair in pairs_of(text) {
                *counts.entry(pair).or_default() += 1;
            }
        }
        Self { counts }
    }

    /// Add another set's counts into this one.
    pub fn merge(&mut self, other: Self) {
        for (pair, count) in other.counts {
            let total = self.counts.entry(pair).or_default();
            *total = total.saturating_add(count);
        }
    }

    /// How many distinct pairs the set holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.counts.len()
    }

    /// Whether the set holds no pair.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }

    fn count(&self, before: char, after: char) -> u32 {
        self.counts
            .get(&pair_key(before, after))
            .copied()
            .unwrap_or(0)
    }
}

fn pair_key(before: char, after: char) -> u64 {
    (u64::from(u32::from(before)) << 32) | u64::from(u32::from(after))
}

fn pairs_of(text: &str) -> impl Iterator<Item = u64> + '_ {
    let mut previous: Option<char> = None;
    text.chars().filter_map(move |character| {
        let pair = previous
            .filter(|_| is_han(character))
            .map(|before| pair_key(before, character));
        previous = is_han(character).then_some(character);
        pair
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(text: &str, category: Category) -> Word {
        Word {
            text: text.to_owned(),
            category,
        }
    }

    fn matcher() -> EntryMatcher {
        let lexicon = vec![
            word("一丁不识", Category::Idiom),
            word("判若鸿沟", Category::Idiom),
            word("一模一样", Category::Idiom),
            word("画蛇添足", Category::Idiom),
            word("撕伞", Category::Internet),
            word("北京", Category::Name),
        ];
        let targets = vec![
            Target {
                entry: "一丁不识".to_owned(),
                category: Category::Idiom,
            },
            Target {
                entry: "判若鸿沟".to_owned(),
                category: Category::Idiom,
            },
            Target {
                entry: "撕伞".to_owned(),
                category: Category::Internet,
            },
        ];
        EntryMatcher::new(lexicon, targets).expect("the fixture lexicon builds")
    }

    #[test]
    fn an_entry_the_normaliser_rewrites_is_set_aside_rather_than_mined() {
        let root = std::env::temp_dir().join(format!("ime-corpus-entries-{}", std::process::id()));
        std::fs::create_dir_all(&root).expect("writable");
        let entries = root.join("entries.tsv");
        std::fs::write(
            &entries,
            "entry\tcategory\toccurrences\tsentences\tall_initials_per_epoch\tany_initial_per_epoch\tblocked\n\
             一丁不识\tidiom\t0\t0\t0.000\t0.000\t0\n\
             不乾不净\tidiom\t0\t0\t0.000\t0.000\t0\n\
             北京\tplace\t0\t0\t0.000\t0.000\t0\n\
             撕伞\tinternet\t3\t3\t0.000\t0.000\t0\n",
        )
        .expect("written");
        let normalizer = Normalizer::new().expect("the t2s configuration loads");
        let matcher = EntryMatcher::load(&entries, &normalizer).expect("the tables load");
        let mined: Vec<&str> = matcher
            .targets()
            .iter()
            .map(|target| target.entry.as_str())
            .collect();
        assert_eq!(mined, vec!["一丁不识"]);
        let aside: Vec<&str> = matcher
            .unminable()
            .iter()
            .map(|target| target.entry.as_str())
            .collect();
        assert_eq!(aside, vec!["不乾不净"]);
        std::fs::remove_dir_all(&root).expect("removed");
    }

    #[test]
    fn a_hit_in_ordinary_prose_is_usable() {
        let hits = matcher().hits("他小时候家里穷得一丁不识");
        assert_eq!(
            hits,
            vec![EntryHit {
                target: 0,
                usable: true
            }]
        );
    }

    #[test]
    fn a_sentence_that_is_little_more_than_the_entry_is_not_usable() {
        assert_eq!(
            matcher().hits("一丁不识啊"),
            vec![EntryHit {
                target: 0,
                usable: false
            }]
        );
    }

    #[test]
    fn a_sentence_made_of_idioms_is_not_usable() {
        // Lexicon entries cover more than half of it.
        let hits = matcher().hits("一丁不识画蛇添足一模一样的");
        assert_eq!(
            hits,
            vec![EntryHit {
                target: 0,
                usable: false
            }]
        );
    }

    #[test]
    fn a_sentence_with_no_function_character_outside_the_entry_is_not_usable() {
        assert_eq!(
            matcher().hits("判若鸿沟北京上海广州深圳"),
            vec![EntryHit {
                target: 1,
                usable: false
            }]
        );
    }

    #[test]
    fn a_sentence_with_no_target_has_no_hit() {
        assert_eq!(matcher().hits("这两个人长得一模一样的"), Vec::new());
    }

    #[test]
    fn the_prefilter_finds_a_target_in_traditional_script() {
        let matcher = matcher();
        assert!(matcher.may_hold_target("他小時候一丁不識。"));
        assert!(matcher.may_hold_target("一丁不识"));
        assert!(!matcher.may_hold_target("今天天气不错"));
    }

    #[test]
    fn a_page_stuffed_with_idioms_or_holding_several_absent_entries_is_spun() {
        let matcher = matcher();
        let pairs =
            CharacterPairs::new(["一丁不识画蛇添足一模一样判若鸿沟撕伞这是一篇文章写的很好"]);
        let stuffed = matcher.profile(["画蛇添足一模一样画蛇添足一丁不识"], &pairs);
        assert_eq!(stuffed.spun(), Some(SpunReason::IdiomDensity));
        let body = "这是一篇文章写的很好".repeat(60);
        let page = format!("{body}一丁不识{body}");
        // Every pair of the page is one run3 writes, so only the gates under test speak.
        let honest = matcher.profile([page.as_str()], &CharacterPairs::new([page.as_str()]));
        assert_eq!(honest.absent, 1);
        assert_eq!(honest.spun(), None);
        let spun = DocumentProfile {
            han: 10_000,
            idioms: 3,
            absent: MAX_ABSENT_PER_DOCUMENT + 1,
            pairs: 9_000,
            rare_pairs: [0; RARE_PAIR_COUNTS.len()],
        };
        assert_eq!(spun.spun(), Some(SpunReason::AbsentEntries));
    }

    #[test]
    fn a_page_of_pairs_nobody_writes_is_spun() {
        let matcher = matcher();
        let pairs = CharacterPairs::new(["今天天气不错"]);
        assert_eq!(pairs.len(), 5);
        let swapped = matcher.profile(["今天邦度不错"], &pairs);
        assert_eq!(swapped.pairs, 5);
        assert_eq!(swapped.rare_pairs[0], 3);
        assert_eq!(swapped.spun(), Some(SpunReason::NovelPairs));
        assert_eq!(matcher.profile(["今天天气不错"], &pairs).spun(), None);
    }

    #[test]
    fn pairs_do_not_span_punctuation() {
        let pairs = CharacterPairs::new(["今天，天气"]);
        assert_eq!(pairs.len(), 2);
    }
}
