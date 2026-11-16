//! Word arcs over a keystroke lattice, a port of `mlime.train.key_arcs`.
//!
//! A *word arc* is a lexicon entry `c_1..c_L` (`L >= 2`) together with a chain
//! of the lattice's own arcs `(a -> x_1, c_1), (x_1 -> x_2, c_2), ...,
//! (x_{L-1} -> b, c_L)` that emits it character by character, at least one of
//! whose spans is typed as an initial ([`is_initial`] of the typed span). The
//! lattice is the only source of chains, so an arc can never make an illegal
//! output; its correction costs are whatever its lattice arcs cost. One
//! `(entry, a, b)` can have several chains, different cut points through the
//! same keys: they are covers of the same sentence, and the search combines
//! them as it combines covers.
//!
//! A chain names its lattice arcs by their index in the order
//! [`KeyLattice::edges`] lists them: by start, then end, then emission id.
//! [`crate::key_search`] numbers its arcs the same way.
//!
//! The lexicon is `word<TAB>category` rows, read for the categories asked
//! for. An entry shorter than two characters, or holding a character outside
//! the emittable set, is skipped, and the load counts both. Entries are held
//! in a character trie over emission ids; [`KeyArcLexicon::arcs`] walks the
//! trie jointly with the lattice's arcs grouped by start, so a record is
//! enumerated in the work its own lattice and the lexicon's matching prefixes
//! cost, never by scanning the lexicon.

use std::collections::BTreeMap;

use hashbrown::HashMap;
use ime_pinyin::{EmitId, KeyLattice, Keys, Legality, is_initial};

use crate::arcs::WordLexiconError;

/// An entry shorter than this is a single character: no word, no arc.
const MIN_ARC_CHARACTERS: usize = 2;

/// What [`KeyArcLexicon::parse`] kept and what it skipped.
#[derive(Copy, Clone, Default, PartialEq, Eq, Debug)]
pub struct KeyArcLexiconStats {
    /// Distinct entries kept.
    pub entries: usize,
    /// Rows of an asked category shorter than two characters.
    pub skipped_short: usize,
    /// Rows of an asked category holding a character outside the emittable set.
    pub skipped_outside: usize,
}

/// One lexicon entry over one stretch of a record's keys, with every chain
/// that emits it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct KeyWordArc {
    entry: Box<[EmitId]>,
    start: usize,
    end: usize,
    /// The chains, row-major: `chain_count * entry.len()` lattice arc indices.
    chains: Box<[usize]>,
}

impl KeyWordArc {
    /// The entry's characters, as emission ids.
    #[must_use]
    pub fn entry(&self) -> &[EmitId] {
        &self.entry
    }

    /// The cut the arc starts at.
    #[must_use]
    pub const fn start(&self) -> usize {
        self.start
    }

    /// The cut the arc ends at.
    #[must_use]
    pub const fn end(&self) -> usize {
        self.end
    }

    /// Each chain's lattice arc indices, in the entry's order; at least one chain.
    pub fn chains(&self) -> impl ExactSizeIterator<Item = &[usize]> {
        self.chains.chunks_exact(self.entry.len())
    }
}

#[derive(Default, Debug)]
struct Node {
    children: BTreeMap<EmitId, usize>,
    /// Whether an entry ends here.
    word: bool,
}

const ROOT: usize = 0;

/// A lexicon of entries over emission ids, ready to be walked against lattices.
#[derive(Debug)]
pub struct KeyArcLexicon {
    nodes: Vec<Node>,
    entries: usize,
}

impl KeyArcLexicon {
    /// The entries of *source*'s *categories* over *legality*'s emittable set.
    ///
    /// *source* is `word<TAB>category` rows. Entries shorter than two
    /// characters or holding a character outside the emittable set are skipped
    /// and counted.
    ///
    /// # Errors
    ///
    /// If a row is not `word<TAB>category`, or no entry survives.
    pub fn parse(
        source: &str,
        categories: &[&str],
        legality: &Legality,
    ) -> Result<(Self, KeyArcLexiconStats), WordLexiconError> {
        let mut lexicon = Self {
            nodes: vec![Node::default()],
            entries: 0,
        };
        let mut stats = KeyArcLexiconStats::default();
        let mut ids: Vec<EmitId> = Vec::new();
        for (index, line) in source.lines().enumerate() {
            let (word, category) = line
                .split_once('\t')
                .filter(|(word, category)| !word.is_empty() && !category.is_empty())
                .ok_or(WordLexiconError::Malformed {
                    line: index + 1,
                    reason: "not a `word<TAB>category` row",
                })?;
            if !categories.contains(&category) {
                continue;
            }
            if word.chars().nth(MIN_ARC_CHARACTERS - 1).is_none() {
                stats.skipped_short += 1;
                continue;
            }
            ids.clear();
            ids.extend(word.chars().map_while(|ch| legality.emit_id(ch)));
            if ids.len() != word.chars().count() {
                stats.skipped_outside += 1;
                continue;
            }
            lexicon.insert(&ids);
        }
        if lexicon.entries == 0 {
            return Err(WordLexiconError::Empty {
                categories: categories.iter().map(|&c| c.to_owned()).collect(),
            });
        }
        stats.entries = lexicon.entries;
        Ok((lexicon, stats))
    }

    fn insert(&mut self, entry: &[EmitId]) {
        let mut node = ROOT;
        for &emit in entry {
            node = if let Some(&child) = self.nodes[node].children.get(&emit) {
                child
            } else {
                self.nodes.push(Node::default());
                let child = self.nodes.len() - 1;
                self.nodes[node].children.insert(emit, child);
                child
            };
        }
        if !self.nodes[node].word {
            self.nodes[node].word = true;
            self.entries += 1;
        }
    }

    /// How many distinct entries the lexicon holds.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.entries
    }

    /// Always false: a lexicon holds at least one entry.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries == 0
    }

    /// Every word arc of *lattice*, over the *keys* it was built from.
    ///
    /// An arc is offered when some chain of the lattice's arcs from `a` to `b`
    /// emits a lexicon entry with at least one initial-typed span. The arcs
    /// come by start cut first, then entry, then end; a keystroke string that
    /// is not the lattice's is a contract violation, logged in a release build,
    /// and has no arcs.
    #[must_use]
    pub fn arcs(&self, lattice: &KeyLattice, keys: &Keys) -> Vec<KeyWordArc> {
        let agree = keys.len() == lattice.keys();
        if !agree {
            tracing::error!(
                keys = keys.len(),
                lattice_keys = lattice.keys(),
                "the keys are not the lattice's: no word arcs"
            );
            debug_assert!(
                agree,
                "{} keys against a lattice of {}",
                keys.len(),
                lattice.keys()
            );
            return Vec::new();
        }
        let mut leaving: Vec<Vec<Leaving<'_>>> = vec![Vec::new(); lattice.keys() + 1];
        let mut base = 0;
        for edge in lattice.edges() {
            leaving[edge.start()].push(Leaving {
                end: edge.end(),
                ids: edge.ids(),
                base,
            });
            base += edge.ids().len();
        }
        let walk = Walk {
            lexicon: self,
            keys: keys.as_str(),
            leaving: &leaving,
        };
        let mut found = Found::default();
        for (cut, from) in leaving.iter().enumerate() {
            if !from.is_empty() {
                walk.visit(&mut found, ROOT, cut, cut, false);
            }
        }
        let mut arcs = found.arcs;
        arcs.sort_by(|a, b| (a.start, &a.entry, a.end).cmp(&(b.start, &b.entry, b.end)));
        arcs
    }
}

/// The lattice arcs of one edge leaving a cut: its end, its sorted characters
/// and the index of its first arc.
#[derive(Copy, Clone)]
struct Leaving<'lattice> {
    end: usize,
    ids: &'lattice [EmitId],
    base: usize,
}

/// The arcs found so far, and the chain being walked.
#[derive(Default)]
struct Found {
    arcs: Vec<KeyWordArc>,
    /// `(start, entry, end)` to its place in `arcs`.
    index: HashMap<(usize, Box<[EmitId]>, usize), usize>,
    entry: Vec<EmitId>,
    chain: Vec<usize>,
}

impl Found {
    /// The chain being walked emits the entry being walked, from *start* to *end*.
    fn record(&mut self, start: usize, end: usize) {
        let key = (start, self.entry.clone().into_boxed_slice(), end);
        if let Some(&at) = self.index.get(&key) {
            let mut chains = self.arcs[at].chains.to_vec();
            chains.extend_from_slice(&self.chain);
            self.arcs[at].chains = chains.into();
        } else {
            self.index.insert(key.clone(), self.arcs.len());
            self.arcs.push(KeyWordArc {
                entry: key.1,
                start,
                end,
                chains: self.chain.as_slice().into(),
            });
        }
    }
}

struct Walk<'a> {
    lexicon: &'a KeyArcLexicon,
    keys: &'a str,
    leaving: &'a [Vec<Leaving<'a>>],
}

impl Walk<'_> {
    /// Walk the trie from *node* against the arcs leaving *cut*; *typed* is
    /// whether the chain so far has an initial-typed span.
    fn visit(&self, found: &mut Found, node: usize, cut: usize, origin: usize, typed: bool) {
        for (&emit, &child) in &self.lexicon.nodes[node].children {
            for edge in &self.leaving[cut] {
                let Ok(at) = edge.ids.binary_search(&emit) else {
                    continue;
                };
                found.entry.push(emit);
                found.chain.push(edge.base + at);
                let reaches = typed || is_initial(&self.keys[cut..edge.end]);
                let child_node = &self.lexicon.nodes[child];
                if child_node.word && reaches {
                    found.record(origin, edge.end);
                }
                if !child_node.children.is_empty() {
                    self.visit(found, child, edge.end, origin, reaches);
                }
                found.chain.pop();
                found.entry.pop();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ime_pinyin::LegalityOptions;

    fn legality(alphabet: &str) -> Legality {
        Legality::over(&alphabet.chars().collect::<Vec<_>>()).expect("the alphabet is emittable")
    }

    fn lattice(legality: &Legality, keys: &str) -> (KeyLattice, Keys) {
        let keys = Keys::new(keys).expect("keys");
        (legality.lattice(&keys, LegalityOptions::OFFLINE), keys)
    }

    #[test]
    fn a_load_counts_what_it_skips_in_every_script() {
        let legality = legality("是时十石");
        let source = "是时\tidiom\n时十\tidiom\n是时\tidiom\n十\tidiom\n\
                      日本語\tidiom\nありがとう\tidiom\n안녕하세요\tidiom\n是石\tname\n";
        let (lexicon, stats) =
            KeyArcLexicon::parse(source, &["idiom"], &legality).expect("entries survive");
        assert_eq!(
            stats,
            KeyArcLexiconStats {
                entries: 2,
                skipped_short: 1,
                skipped_outside: 3,
            }
        );
        assert_eq!((lexicon.len(), lexicon.is_empty()), (2, false));
    }

    #[test]
    fn a_load_refuses_a_malformed_row_and_an_empty_lexicon() {
        let legality = legality("是时十石");
        assert_eq!(
            KeyArcLexicon::parse("是时\tidiom\n时十\n", &["idiom"], &legality).err(),
            Some(WordLexiconError::Malformed {
                line: 2,
                reason: "not a `word<TAB>category` row",
            })
        );
        assert_eq!(
            KeyArcLexicon::parse("十\tidiom\n", &["idiom"], &legality).err(),
            Some(WordLexiconError::Empty {
                categories: vec!["idiom".to_owned()],
            })
        );
    }

    #[test]
    fn an_entry_typed_in_full_syllables_is_no_arc() {
        let legality = legality("是时十");
        let (lexicon, _) =
            KeyArcLexicon::parse("是时\tidiom\n", &["idiom"], &legality).expect("entries");
        let (full, keys) = lattice(&legality, "shishi");
        assert!(
            lexicon
                .arcs(&full, &keys)
                .iter()
                .all(|arc| arc.chains().all(|chain| chain.len() == 2))
        );
        let (typed, keys) = lattice(&legality, "ss");
        let arcs = lexicon.arcs(&typed, &keys);
        assert_eq!(arcs.len(), 1);
        assert_eq!(
            (arcs[0].start(), arcs[0].end(), arcs[0].chains().len()),
            (0, 2, 1)
        );
        assert_eq!(arcs[0].entry().len(), 2);
    }
}
