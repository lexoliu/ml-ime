//! The legality automaton over keys: which sentences are a reading of a
//! keystroke string (#216 §3).
//!
//! A sentence `c1..cm` is legal for keys `k1..kn` when the keys cut into `m`
//! contiguous spans, span `j` matching `cj` under one of four rules, each
//! against some reading `r` of the character:
//!
//! 1. **Full syllable** -- the span is `r`; cost 0.
//! 2. **Initial** -- the span is one letter, or `zh`/`ch`/`sh`, and a prefix of
//!    `r`; cost 0.
//! 3. **Unfinished tail** -- the span is the last one, a prefix of `r`, and
//!    [`LegalityOptions::unfinished_tail`] is on; cost 0.
//! 4. **Correction** -- the span has two keys or more, is not `zh`/`ch`/`sh`,
//!    and `(r, w)` is among [`CorrectionTable::corrections`]; cost `w`. A span
//!    that is itself a syllable is corrected only under
//!    [`LegalityOptions::correct_syllables`].
//!
//! A span's cost for a character is its cheapest rule over the character's
//! readings; a sentence's cost is its cheapest cover. A span is at most
//! [`MAX_CORRECTED_LEN`] keys wide.
//!
//! [`Legality`] holds the rules over an emittable character set, built once.
//! [`Legality::lattice`] instantiates them over one keystroke string as a
//! [`KeyLattice`]: the automaton whose states are key offsets and whose arcs
//! are `(character, cost)` pairs, pruned to complete covers. The lattice is
//! the search's mask and the evaluation's verifier. `mlime.legality`
//! implements the same rules for the training loss; the fixture at
//! `tests/fixtures/legality.json` holds the two to the same answers.

use std::collections::HashMap;

use thiserror::Error;

use crate::corrections::{CORRECTIONS_K, CorrectionTable, MAX_CORRECTED_LEN, NoiseError};
use crate::lexicon::{Lexicon, LexiconError};
use crate::syllable::SyllableTable;

/// The three initials spelled with two letters.
const MULTI_LETTER_INITIALS: [&str; 3] = ["zh", "ch", "sh"];

/// The slot of a width with no kept arc.
const NO_EDGE: u32 = u32::MAX;

/// A keystroke string: non-empty, every key in `a`..`z`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Keys(Box<str>);

/// Why a string is not a keystroke string.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum KeysError {
    /// The string has no keys.
    #[error("the keys are empty")]
    Empty,
    /// A character left the `[a-z]` alphabet.
    #[error("byte {offset} is {ch:?}, outside the [a-z] keys")]
    OutsideAlphabet {
        /// Byte offset of the offending character.
        offset: usize,
        /// The offending character.
        ch: char,
    },
}

impl Keys {
    /// *keys* as a keystroke string.
    ///
    /// # Errors
    ///
    /// If *keys* is empty or holds anything outside `[a-z]`.
    pub fn new(keys: &str) -> Result<Self, KeysError> {
        if keys.is_empty() {
            return Err(KeysError::Empty);
        }
        if let Some((offset, ch)) = keys.char_indices().find(|(_, c)| !c.is_ascii_lowercase()) {
            return Err(KeysError::OutsideAlphabet { offset, ch });
        }
        Ok(Self(keys.into()))
    }

    /// The keys, as typed.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// How many keys there are; never zero.
    #[must_use]
    #[expect(clippy::len_without_is_empty, reason = "a Keys is never empty")]
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

/// What the automaton admits beyond the rules that always hold.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct LegalityOptions {
    /// Whether the last span may be an unfinished syllable: on for live
    /// input, off in offline evaluation.
    pub unfinished_tail: bool,
    /// Whether corrections also apply to a span that is itself a syllable
    /// (`zao` meant as zhao); chosen on the eval3-typo dev slice.
    pub correct_syllables: bool,
}

impl LegalityOptions {
    /// Offline evaluation and training: complete input, corrections everywhere.
    pub const OFFLINE: Self = Self {
        unfinished_tail: false,
        correct_syllables: true,
    };

    /// A live session: the last syllable may still be being typed.
    pub const LIVE: Self = Self {
        unfinished_tail: true,
        correct_syllables: true,
    };
}

/// An index into the emittable set: the output head's character row.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct EmitId(u32);

impl EmitId {
    /// This character's row in the emittable set.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Why the automaton could not be built.
#[derive(Debug, Error)]
pub enum LegalityError {
    /// The character table disagrees with the syllable inventory.
    #[error(transparent)]
    Lexicon(#[from] LexiconError),
    /// The typo table is malformed.
    #[error(transparent)]
    Noise(#[from] NoiseError),
    /// An emittable character has no readings.
    #[error("{0:?} is emittable but not in the character table")]
    Unread(char),
    /// An emittable character is listed twice.
    #[error("{0:?} appears twice in the emittable set")]
    Duplicate(char),
    /// The emittable set is larger than the `u32` index space.
    #[error("the emittable set holds {0} characters, past the u32 index space")]
    TooLarge(usize),
}

/// The four rules over an emittable character set, built once.
pub struct Legality {
    syllables: SyllableTable,
    corrections: CorrectionTable,
    emittable: Box<[char]>,
    ids: HashMap<char, EmitId>,
    /// `homophones[syllable]`: the emittable characters reading it, sorted.
    homophones: Box<[Box<[EmitId]>]>,
    /// Every prefix of every reading, the reading included: the emittable
    /// characters with a reading that begins with it, sorted. What the
    /// initial and tail rules admit.
    prefixed: HashMap<Box<str>, Box<[EmitId]>>,
    /// The most characters one span can admit; see [`Legality::admitted_bound`].
    bound: usize,
}

impl Legality {
    /// The automaton over the whole character table, in its own order.
    ///
    /// # Errors
    ///
    /// If the embedded tables disagree or are malformed.
    pub fn over_lexicon() -> Result<Self, LegalityError> {
        let syllables = SyllableTable::load();
        let lexicon = Lexicon::load(&syllables)?;
        let emittable = lexicon.characters().to_vec();
        Self::build(syllables, &lexicon, &emittable)
    }

    /// The automaton over *emittable*, whose order is the emission order.
    ///
    /// # Errors
    ///
    /// If the embedded tables disagree or are malformed, or a character of
    /// *emittable* is repeated or has no readings.
    pub fn over(emittable: &[char]) -> Result<Self, LegalityError> {
        let syllables = SyllableTable::load();
        let lexicon = Lexicon::load(&syllables)?;
        Self::build(syllables, &lexicon, emittable)
    }

    fn build(
        syllables: SyllableTable,
        lexicon: &Lexicon,
        emittable: &[char],
    ) -> Result<Self, LegalityError> {
        let corrections = CorrectionTable::load(&syllables)?;
        let mut ids = HashMap::with_capacity(emittable.len());
        let mut homophones: Vec<Vec<EmitId>> = vec![Vec::new(); syllables.len()];
        let mut prefixed: HashMap<Box<str>, Vec<EmitId>> = HashMap::new();
        for (index, &ch) in emittable.iter().enumerate() {
            let emit =
                EmitId(u32::try_from(index).map_err(|_| LegalityError::TooLarge(emittable.len()))?);
            if ids.insert(ch, emit).is_some() {
                return Err(LegalityError::Duplicate(ch));
            }
            let id = lexicon.id_of(ch).ok_or(LegalityError::Unread(ch))?;
            for &reading in lexicon.readings(id) {
                homophones[reading.index()].push(emit);
                let spelling = syllables.spelling(reading);
                for length in 1..=spelling.len() {
                    prefixed
                        .entry(spelling[..length].into())
                        .or_default()
                        .push(emit);
                }
            }
        }
        let sorted = |mut group: Vec<EmitId>| {
            group.sort_unstable();
            group.dedup();
            group.into_boxed_slice()
        };
        let homophones: Box<[Box<[EmitId]>]> = homophones.into_iter().map(sorted).collect();
        let prefixed: HashMap<Box<str>, Box<[EmitId]>> = prefixed
            .into_iter()
            .map(|(prefix, group)| (prefix, sorted(group)))
            .collect();
        let widest_prefix = prefixed
            .values()
            .map(|group| group.len())
            .max()
            .unwrap_or(0);
        let widest_syllable = homophones
            .iter()
            .map(|group| group.len())
            .max()
            .unwrap_or(0);
        Ok(Self {
            syllables,
            corrections,
            emittable: emittable.into(),
            ids,
            homophones,
            prefixed,
            bound: widest_prefix + CORRECTIONS_K * widest_syllable,
        })
    }

    /// The emittable set, in emission order.
    #[must_use]
    pub fn emittable(&self) -> &[char] {
        &self.emittable
    }

    /// *ch*'s emission index, when it is emittable.
    #[must_use]
    pub fn emit_id(&self, ch: char) -> Option<EmitId> {
        self.ids.get(&ch).copied()
    }

    /// The most characters a single span can admit: the widest prefix set
    /// (an initial or a tail) plus `K` corrections each as wide as the widest
    /// homophone set. This is `C` in the beam mask's per-step cost.
    #[must_use]
    pub const fn admitted_bound(&self) -> usize {
        self.bound
    }

    /// Whether rule 4 applies to *span* under *options*.
    fn corrects(&self, span: &str, options: LegalityOptions) -> bool {
        span.len() >= 2
            && !MULTI_LETTER_INITIALS.contains(&span)
            && (options.correct_syllables || self.syllables.exact(span).is_none())
            && self.corrections.reaches(span)
    }

    /// Every character *span* admits, at its cheapest rule, into *out*:
    /// sorted by emission index, one entry per character. *last* says the
    /// span ends at the last key, where the tail rule may apply.
    fn span_matches(
        &self,
        span: &str,
        last: bool,
        options: LegalityOptions,
        out: &mut Vec<(EmitId, f64)>,
    ) {
        out.clear();
        let own: Option<&[EmitId]> = if is_initial(span) || (last && options.unfinished_tail) {
            self.prefixed.get(span).map(|group| &group[..])
        } else {
            self.syllables
                .exact(span)
                .map(|syllable| &self.homophones[syllable.index()][..])
        };
        if let Some(own) = own {
            out.extend(own.iter().map(|&emit| (emit, 0.0)));
        }
        if self.corrects(span, options) {
            for entry in self.corrections.corrections(span).iter() {
                if let Some(syllable) = self.syllables.exact(entry.syllable()) {
                    let cost = entry.cost();
                    out.extend(
                        self.homophones[syllable.index()]
                            .iter()
                            .map(|&emit| (emit, cost)),
                    );
                }
            }
            out.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
            out.dedup_by_key(|entry| entry.0);
        }
    }

    /// The automaton over *keys*: every legal arc, pruned to complete covers.
    #[must_use]
    pub fn lattice(&self, keys: &Keys, options: LegalityOptions) -> KeyLattice {
        let input = keys.as_str();
        let width = input.len();
        let mut found: Vec<FoundEdge> = Vec::new();
        let mut matches = Vec::new();
        for start in 0..width {
            for end in (start + 1)..=width.min(start + MAX_CORRECTED_LEN) {
                self.span_matches(&input[start..end], end == width, options, &mut matches);
                if !matches.is_empty() {
                    found.push(FoundEdge {
                        start,
                        end,
                        arcs: std::mem::take(&mut matches),
                    });
                }
            }
        }
        // `found` is in (start, end) order, so one forward and one backward
        // sweep settle reachability from the start and to the end.
        let mut reachable = vec![false; width + 1];
        reachable[0] = true;
        for edge in &found {
            if reachable[edge.start] {
                reachable[edge.end] = true;
            }
        }
        let mut productive = vec![false; width + 1];
        productive[width] = true;
        for edge in found.iter().rev() {
            if productive[edge.end] {
                productive[edge.start] = true;
            }
        }
        KeyLattice::from_edges(
            width,
            found
                .into_iter()
                .filter(|edge| reachable[edge.start] && productive[edge.end]),
        )
    }

    /// *sentence*'s cost over *keys*: its cheapest legal cover, `None` when
    /// it is illegal -- a character outside the emittable set included.
    #[must_use]
    pub fn sentence_cost(
        &self,
        keys: &Keys,
        sentence: &str,
        options: LegalityOptions,
    ) -> Option<f64> {
        let ids: Option<Vec<EmitId>> = sentence.chars().map(|ch| self.emit_id(ch)).collect();
        self.lattice(keys, options).cost(&ids?)
    }

    /// The syllable spelled *spelling*, for tests that name readings.
    #[cfg(test)]
    fn syllable(&self, spelling: &str) -> Option<crate::syllable::SyllableId> {
        self.syllables.exact(spelling)
    }

    /// The correction table the automaton reads.
    #[cfg(test)]
    const fn correction_table(&self) -> &CorrectionTable {
        &self.corrections
    }
}

/// Whether *span* has the initial's shape: one letter, or zh/ch/sh.
fn is_initial(span: &str) -> bool {
    span.len() == 1 || MULTI_LETTER_INITIALS.contains(&span)
}

/// A span with admitted characters, before pruning to complete covers.
struct FoundEdge {
    start: usize,
    end: usize,
    /// Sorted by emission index, one entry per character.
    arcs: Vec<(EmitId, f64)>,
}

/// One kept edge's place in the flat arc arrays.
#[derive(Copy, Clone, Debug)]
struct EdgeEntry {
    start: u32,
    end: u32,
    offset: u32,
    len: u32,
}

/// Every arc from one key offset to another: the characters the span between
/// them admits, sorted by emission index, with their costs.
#[derive(Copy, Clone, Debug)]
pub struct Edge<'lattice> {
    start: usize,
    end: usize,
    ids: &'lattice [EmitId],
    costs: &'lattice [f64],
}

impl<'lattice> Edge<'lattice> {
    /// The offset of the span's first key.
    #[must_use]
    pub const fn start(&self) -> usize {
        self.start
    }

    /// One past the span's last key: the key the character is emitted at is
    /// `end - 1`.
    #[must_use]
    pub const fn end(&self) -> usize {
        self.end
    }

    /// The admitted characters, sorted.
    #[must_use]
    pub const fn ids(&self) -> &'lattice [EmitId] {
        self.ids
    }

    /// Each admitted character's cost, aligned with [`Edge::ids`].
    #[must_use]
    pub const fn costs(&self) -> &'lattice [f64] {
        self.costs
    }

    /// The cost of *emit* on this edge, when the span admits it.
    #[must_use]
    pub fn cost_of(&self, emit: EmitId) -> Option<f64> {
        self.ids
            .binary_search(&emit)
            .ok()
            .map(|index| self.costs[index])
    }
}

/// What one hypothesis may do at one key: see [`KeyLattice::step`].
#[derive(Copy, Clone, Debug)]
pub struct Step<'lattice> {
    blank: bool,
    ids: &'lattice [EmitId],
    costs: &'lattice [f64],
}

impl<'lattice> Step<'lattice> {
    /// Whether the key may emit blank: the open span stays a prefix of some
    /// arc that lies on a complete cover.
    #[must_use]
    pub const fn blank(&self) -> bool {
        self.blank
    }

    /// The characters the key may emit, closing the open span; sorted.
    #[must_use]
    pub const fn ids(&self) -> &'lattice [EmitId] {
        self.ids
    }

    /// Each emittable character's correction cost, aligned with [`Step::ids`].
    #[must_use]
    pub const fn costs(&self) -> &'lattice [f64] {
        self.costs
    }

    /// Whether the hypothesis has no legal continuation at this key.
    #[must_use]
    pub const fn is_dead(&self) -> bool {
        !self.blank && self.ids.is_empty()
    }
}

/// The automaton instantiated over one keystroke string.
///
/// States are key offsets `0..=n`; an edge from `s` to `e` carries every
/// `(character, cost)` arc whose span is keys `s..e`. Every kept edge lies on
/// a complete cover, so a search that follows [`KeyLattice::step`] never
/// strands a hypothesis.
pub struct KeyLattice {
    keys: usize,
    /// `slots[start * MAX_CORRECTED_LEN + width - 1]`: the edge from `start`
    /// over `width` keys, as an index into `edges`, or [`NO_EDGE`].
    slots: Box<[u32]>,
    /// `open[start * MAX_CORRECTED_LEN + width - 1]`: whether some edge from
    /// `start` is wider than `width`.
    open: Box<[bool]>,
    edges: Box<[EdgeEntry]>,
    ids: Box<[EmitId]>,
    costs: Box<[f64]>,
}

impl KeyLattice {
    fn from_edges(keys: usize, kept: impl Iterator<Item = FoundEdge>) -> Self {
        let mut slots = vec![NO_EDGE; keys * MAX_CORRECTED_LEN];
        let mut open = vec![false; keys * MAX_CORRECTED_LEN];
        let mut edges = Vec::new();
        let mut ids = Vec::new();
        let mut costs = Vec::new();
        for FoundEdge { start, end, arcs } in kept {
            let width = end - start;
            let slot = start * MAX_CORRECTED_LEN + width - 1;
            slots[slot] = u32::try_from(edges.len()).expect("a record's edges fit u32");
            for narrower in &mut open[start * MAX_CORRECTED_LEN..slot] {
                *narrower = true;
            }
            #[expect(
                clippy::cast_possible_truncation,
                reason = "offsets and arc counts of one record fit u32"
            )]
            let entry = EdgeEntry {
                start: start as u32,
                end: end as u32,
                offset: ids.len() as u32,
                len: arcs.len() as u32,
            };
            edges.push(entry);
            for (emit, cost) in arcs {
                ids.push(emit);
                costs.push(cost);
            }
        }
        Self {
            keys,
            slots: slots.into(),
            open: open.into(),
            edges: edges.into(),
            ids: ids.into(),
            costs: costs.into(),
        }
    }

    /// How many keys the lattice covers.
    #[must_use]
    pub const fn keys(&self) -> usize {
        self.keys
    }

    /// Whether some cover of the keys exists at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// How many `(start, end, character)` arcs the lattice holds.
    #[must_use]
    pub fn arc_count(&self) -> usize {
        self.ids.len()
    }

    fn edge_at(&self, index: usize) -> Edge<'_> {
        let entry = self.edges[index];
        let range = entry.offset as usize..(entry.offset + entry.len) as usize;
        Edge {
            start: entry.start as usize,
            end: entry.end as usize,
            ids: &self.ids[range.clone()],
            costs: &self.costs[range],
        }
    }

    /// Every kept edge, in `(start, end)` order.
    pub fn edges(&self) -> impl Iterator<Item = Edge<'_>> {
        (0..self.edges.len()).map(|index| self.edge_at(index))
    }

    /// The edge over keys `start..end`, when it is kept.
    #[must_use]
    pub fn edge(&self, start: usize, end: usize) -> Option<Edge<'_>> {
        let width = end.checked_sub(start)?;
        if start >= self.keys || width == 0 || width > MAX_CORRECTED_LEN {
            return None;
        }
        let slot = self.slots[start * MAX_CORRECTED_LEN + width - 1];
        (slot != NO_EDGE).then(|| self.edge_at(slot as usize))
    }

    /// The mask for a hypothesis whose last emission closed at offset
    /// *pending*, reading key *key*: the open span is keys `pending..=key`.
    ///
    /// The cost is two table reads and two borrowed slices, whatever the key
    /// count: masking a beam of `B` hypotheses at one key is `O(B · (1 + C))`
    /// with `C` at most [`Legality::admitted_bound`] -- one blank flag and at
    /// most `C` admitted characters each, gathered from the key's emission
    /// row that all hypotheses share.
    ///
    /// # Panics
    ///
    /// If `key < pending`, or the open span is wider than
    /// [`MAX_CORRECTED_LEN`] -- a hypothesis that ignored an earlier step's
    /// [`Step::blank`].
    #[must_use]
    pub fn step(&self, pending: usize, key: usize) -> Step<'_> {
        let width = key + 1 - pending;
        assert!(
            (1..=MAX_CORRECTED_LEN).contains(&width) && key < self.keys,
            "the open span {pending}..={key} is not a span of the {} keys",
            self.keys
        );
        let slot = pending * MAX_CORRECTED_LEN + width - 1;
        let edge = self.slots[slot];
        let (ids, costs) = if edge == NO_EDGE {
            (&[][..], &[][..])
        } else {
            let entry = self.edges[edge as usize];
            let range = entry.offset as usize..(entry.offset + entry.len) as usize;
            (&self.ids[range.clone()], &self.costs[range])
        };
        Step {
            blank: self.open[slot],
            ids,
            costs,
        }
    }

    /// *sentence*'s cheapest legal cover's cost, `None` when it is illegal.
    ///
    /// `O(m · n · MAX_CORRECTED_LEN · log C)` for `m` characters over `n` keys.
    #[must_use]
    pub fn cost(&self, sentence: &[EmitId]) -> Option<f64> {
        let (width, length) = (self.keys, sentence.len());
        if length == 0 || length > width {
            return None;
        }
        // `best[offset]`: the cheapest cover of keys `..offset` by the
        // characters placed so far.
        let mut best = vec![f64::INFINITY; width + 1];
        best[0] = 0.0;
        for &emit in sentence {
            let mut next = vec![f64::INFINITY; width + 1];
            for (start, &before) in best.iter().enumerate().take(width) {
                if before.is_infinite() {
                    continue;
                }
                for span in 1..=MAX_CORRECTED_LEN.min(width - start) {
                    let end = start + span;
                    if let Some(cost) = self.edge(start, end).and_then(|edge| edge.cost_of(emit)) {
                        next[end] = next[end].min(before + cost);
                    }
                }
            }
            best = next;
        }
        best[width].is_finite().then_some(best[width])
    }
}

#[cfg(test)]
mod tests;
