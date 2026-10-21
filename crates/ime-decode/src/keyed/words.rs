//! Curated words as legal multi-character emissions (#212, #216 §5).

use ime_pinyin::{EmitId, KeyLattice, Legality, MAX_CORRECTED_LEN};

/// What [`WordArcs::new`] kept and why it left the rest out.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WordStats {
    /// Words kept as arcs.
    pub kept: usize,
    /// Words shorter than two characters: a one-character word is an
    /// ordinary emission.
    pub short: usize,
    /// Words holding a character outside the emittable set.
    pub unemittable: usize,
    /// Words listed more than once.
    pub duplicate: usize,
}

/// One node of the word trie.
#[derive(Clone, Debug, Default)]
struct TrieNode {
    /// `(character, child)`, sorted by character.
    children: Vec<(EmitId, u32)>,
    /// The word that ends here, as its characters.
    word: Option<Box<[EmitId]>>,
}

/// A trie of curated words over the emittable set.
///
/// [`WordArcs::offered`] walks it along a [`KeyLattice`]'s edges, so a word is
/// offered over keys `s..e` only when some cover of those keys spells it one
/// character a span under the legality rules -- an arc can never make an
/// illegal sentence.
#[derive(Clone, Debug)]
pub struct WordArcs {
    nodes: Vec<TrieNode>,
}

impl WordArcs {
    /// The trie over *words*, each a string of emittable characters.
    ///
    /// # Panics
    ///
    /// If the trie outgrows `u32` node indices.
    #[must_use]
    pub fn new<'w>(
        legality: &Legality,
        words: impl IntoIterator<Item = &'w str>,
    ) -> (Self, WordStats) {
        let mut stats = WordStats::default();
        let mut nodes = vec![TrieNode::default()];
        for word in words {
            let Some(chars) = word
                .chars()
                .map(|ch| legality.emit_id(ch))
                .collect::<Option<Box<[EmitId]>>>()
            else {
                stats.unemittable += 1;
                continue;
            };
            if chars.len() < 2 {
                stats.short += 1;
                continue;
            }
            let mut node = 0usize;
            for &ch in &chars {
                let next = u32::try_from(nodes.len()).expect("a word trie fits u32 nodes");
                let children = &mut nodes[node].children;
                node = match children.binary_search_by_key(&ch, |entry| entry.0) {
                    Ok(found) => children[found].1 as usize,
                    Err(at) => {
                        children.insert(at, (ch, next));
                        nodes.push(TrieNode::default());
                        next as usize
                    }
                };
            }
            if nodes[node].word.is_some() {
                stats.duplicate += 1;
            } else {
                nodes[node].word = Some(chars);
                stats.kept += 1;
            }
        }
        (Self { nodes }, stats)
    }

    /// Every word the keys spell, as `(start, end, characters)`: once per
    /// key span, however many covers spell it there; in start order.
    pub(crate) fn offered(&self, lattice: &KeyLattice) -> Vec<(usize, usize, &[EmitId])> {
        let keys = lattice.keys();
        let mut found: Vec<(usize, usize, u32)> = Vec::new();
        let mut stack: Vec<(u32, usize)> = Vec::new();
        for start in 0..keys {
            stack.push((0, start));
            while let Some((node, at)) = stack.pop() {
                let children = &self.nodes[node as usize].children;
                for end in at + 1..=keys.min(at + MAX_CORRECTED_LEN) {
                    let Some(edge) = lattice.edge(at, end) else {
                        continue;
                    };
                    intersect(edge.ids(), children, |child| {
                        if self.nodes[child as usize].word.is_some() {
                            found.push((start, end, child));
                        }
                        if !self.nodes[child as usize].children.is_empty() && end < keys {
                            stack.push((child, end));
                        }
                    });
                }
            }
        }
        found.sort_unstable();
        found.dedup();
        found
            .into_iter()
            .map(|(start, end, node)| {
                let word = self.nodes[node as usize]
                    .word
                    .as_deref()
                    .expect("only word nodes are found");
                (start, end, word)
            })
            .collect()
    }
}

/// Call *visit* with the child of every character in both *ids* and
/// *children*, both sorted: a probe of the longer from each of the shorter.
fn intersect(ids: &[EmitId], children: &[(EmitId, u32)], mut visit: impl FnMut(u32)) {
    if ids.len() <= children.len() {
        for id in ids {
            if let Ok(found) = children.binary_search_by_key(id, |entry| entry.0) {
                visit(children[found].1);
            }
        }
    } else {
        for &(ch, child) in children {
            if ids.binary_search(&ch).is_ok() {
                visit(child);
            }
        }
    }
}
