//! Turning a keystroke sequence into candidate readings.
//!
//! A typed string does not determine how many characters it will become. `xian`
//! is one syllable or two; `zgrm` is four abbreviated ones. So segmentation is a
//! lattice, not a function, and what this module produces is the *k* cheapest
//! paths through it -- which downstream is exactly the batch of candidate lengths
//! that go into a single encoder forward.

use std::sync::Arc;

use crate::corrections::{CORRECTIONS_MAX_EDITS, Correction, CorrectionTable};
use crate::syllable::{MAX_SYLLABLE_LEN, SyllableRange, SyllableTable};
use thiserror::Error;

/// The longest typed span a corrected segment can cover: a full syllable can
/// only drift two edits' worth of letters from what was meant.
const MAX_CORRECTED_LEN: usize = MAX_SYLLABLE_LEN + CORRECTIONS_MAX_EDITS;

/// What a typed span can be read as. The inventory case keeps the syllable
/// range; a span the inventory does not reach -- a typo -- instead carries the
/// noise model's corrections with their costs.
#[derive(Clone, PartialEq, Debug)]
pub enum Readings {
    /// The span is a syllable or a prefix of one: the table range covering
    /// every syllable the keystrokes could complete to or abbreviate.
    Syllables(SyllableRange),
    /// The span reaches the inventory only through the noise model: its
    /// `K`-capped corrections, cheapest first. The list is shared -- every
    /// segment spelling the same span points at the same entries.
    Corrections(Arc<[Correction]>),
}

/// Add the `start..end` segment to the bucket unless its readings are empty
/// or an identical edge already exists.
fn push(edges: &mut Vec<Segment>, start: u32, end: u32, readings: Readings) {
    let segment = Segment {
        start,
        end,
        readings,
    };
    let empty = matches!(&segment.readings, Readings::Syllables(range) if range.is_empty());
    if !empty && !edges.contains(&segment) {
        edges.push(segment);
    }
}

/// One output character position: the span of input it consumes, and every
/// syllable that span can be read as.
///
/// The three cases an IME distinguishes -- a complete syllable, an initial-only
/// abbreviation, a syllable still being typed -- differ only in how wide
/// [`Readings::Syllables`] is, so they need no tag. A fourth case exists only
/// when the lattice is built against a [`CorrectionTable`]: the span is a
/// typo, and its readings are the syllables the noise model says were meant.
#[derive(Clone, PartialEq, Debug)]
pub struct Segment {
    start: u32,
    end: u32,
    readings: Readings,
}

impl Segment {
    /// Byte offset where this segment starts in the typed string.
    #[must_use]
    pub const fn start(&self) -> usize {
        self.start as usize
    }

    /// Byte offset one past this segment's last keystroke.
    #[must_use]
    pub const fn end(&self) -> usize {
        self.end as usize
    }

    /// The readings this segment admits.
    #[must_use]
    pub const fn readings(&self) -> &Readings {
        &self.readings
    }

    /// The syllable range this segment admits, when the span is on the
    /// inventory. `None` for a corrected segment -- its readings are syllables
    /// the noise model picked, not a prefix range.
    #[must_use]
    pub const fn syllables(&self) -> Option<SyllableRange> {
        match &self.readings {
            Readings::Syllables(range) => Some(*range),
            Readings::Corrections(_) => None,
        }
    }

    /// Whether this segment exists only because the noise model reaches it --
    /// i.e. the typed span is no syllable or prefix.
    #[must_use]
    pub const fn corrected(&self) -> bool {
        matches!(self.readings, Readings::Corrections(_))
    }

    /// The corrections this segment reads as, when it is a corrected one.
    #[must_use]
    pub const fn corrections(&self) -> Option<&Arc<[Correction]>> {
        match &self.readings {
            Readings::Syllables(_) => None,
            Readings::Corrections(entries) => Some(entries),
        }
    }

    /// How many readings this segment admits. One means the user typed the
    /// syllable out in full.
    #[must_use]
    pub fn ambiguity(&self) -> usize {
        match &self.readings {
            Readings::Syllables(range) => range.len(),
            Readings::Corrections(entries) => entries.len(),
        }
    }

    /// The prior a corrected span costs: the cheapest correction's `-log P`.
    fn prior(&self) -> f32 {
        match &self.readings {
            Readings::Syllables(_) => 0.0,
            Readings::Corrections(entries) =>
            {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "a -log of a two-edit probability stays small"
                )]
                entries.first().map_or(0.0, |entry| entry.cost() as f32)
            }
        }
    }

    /// This segment's contribution to a path's cost.
    #[must_use]
    fn cost(&self, options: &SegmentOptions) -> f32 {
        #[expect(
            clippy::cast_precision_loss,
            reason = "ambiguity is at most the inventory size"
        )]
        let ambiguity = self.ambiguity() as f32;
        options.segment_cost + options.ambiguity_weight * ambiguity.ln() + self.prior()
    }
}

/// Knobs on how permissive segmentation is, and on what it prefers.
///
/// The defaults describe a live IME: abbreviations on, a half-typed trailing
/// syllable accepted. An offline evaluation over complete pinyin should turn the
/// trailing allowance off so that a wrong answer cannot hide behind it.
#[derive(Clone, Debug)]
pub struct SegmentOptions {
    /// How many segmentations [`SegmentLattice::k_best`] returns.
    pub max_paths: usize,
    /// Whether a lone initial may stand for a whole syllable (`zgrm`).
    pub allow_abbreviation: bool,
    /// Whether the final segment may be an unfinished syllable (`zhonggu`).
    pub allow_incomplete_tail: bool,
    /// Flat cost of using one more character position. Higher values prefer
    /// readings that consume the input in fewer, longer syllables.
    pub segment_cost: f32,
    /// Weight on a segment's log ambiguity. Higher values prefer fully typed
    /// syllables over abbreviations.
    pub ambiguity_weight: f32,
}

impl Default for SegmentOptions {
    fn default() -> Self {
        Self {
            max_paths: 8,
            allow_abbreviation: true,
            allow_incomplete_tail: true,
            segment_cost: 1.0,
            ambiguity_weight: 1.0,
        }
    }
}

/// Why a typed string could not be segmented.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SegmentError {
    /// The input had no keystrokes.
    #[error("input is empty")]
    Empty,
    /// The input left the `[a-z]` keyboard alphabet.
    #[error("byte {offset} is {ch:?}, outside the [a-z] input alphabet")]
    InvalidCharacter {
        /// Byte offset of the offending character.
        offset: usize,
        /// The offending character.
        ch: char,
    },
    /// No path spans the input under the given options.
    #[error("{input:?} has no reading under the current options")]
    NoSegmentation {
        /// The input that could not be read.
        input: String,
    },
    /// The input was longer than the `u32` offsets used inside a segment.
    #[error("input is {len} bytes, which exceeds the u32 offset space")]
    TooLong {
        /// Length of the offending input.
        len: usize,
    },
}

/// One complete reading of the input: a character position per segment.
#[derive(Clone, PartialEq, Debug)]
pub struct Segmentation {
    segments: Vec<Segment>,
    cost: f32,
}

impl Segmentation {
    /// The character positions, left to right.
    #[must_use]
    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    /// How many characters this reading produces.
    #[must_use]
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// Whether this reading produces no characters. Never true for a reading
    /// returned by [`SegmentLattice::k_best`].
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Path cost. Lower is a more conventional reading of the keystrokes.
    #[must_use]
    pub const fn cost(&self) -> f32 {
        self.cost
    }

    /// The reading that walks *parts* in order: how the session cuts the input
    /// at `'` boundaries, each piece segmented alone and the pieces joined.
    /// The segments are re-offset against the pieces' letters read as one
    /// string, so the result reads exactly like a single lattice's reading.
    ///
    /// Cost is the parts' sum, which ranks two joined readings the way their
    /// pieces' own costs rank them.
    #[must_use]
    pub fn concat(parts: &[&Segmentation]) -> Self {
        let mut segments = Vec::with_capacity(parts.iter().map(|part| part.len()).sum());
        let mut cost = 0.0f32;
        let mut base = 0u32;
        for part in parts {
            let mut end = 0u32;
            for segment in part.segments() {
                end = segment.end;
                segments.push(Segment {
                    start: segment.start + base,
                    end: segment.end + base,
                    readings: segment.readings().clone(),
                });
            }
            cost += part.cost;
            base += end;
        }
        Self { segments, cost }
    }
}

/// Every way the typed string can be cut into character positions.
#[derive(Debug)]
pub struct SegmentLattice<'input> {
    input: &'input str,
    /// `edges[i]` are the segments starting at byte `i`. Segments that cannot lie
    /// on any complete path are pruned at build time.
    edges: Vec<Vec<Segment>>,
}

/// A partial path during the k-best sweep.
#[derive(Clone, Copy, Debug)]
struct Trace {
    /// Summed segment costs -- a corrected segment adds its prior through
    /// [`Segment::cost`], so paths rank by summed prior the way #101
    /// specifies, and a clean path's cost is exactly today's.
    cost: f32,
    /// `(start position, index into `edges[start]`, rank within `best[start]`)`.
    /// Absent only for the empty path at position zero.
    back: Option<(usize, usize, usize)>,
}

impl<'input> SegmentLattice<'input> {
    /// Build the lattice for *input*, without corrections.
    ///
    /// # Errors
    ///
    /// If the input is empty, leaves the `[a-z]` alphabet, is longer than the
    /// `u32` offset space, or admits no reading under *options*.
    pub fn build(
        input: &'input str,
        table: &SyllableTable,
        options: &SegmentOptions,
    ) -> Result<Self, SegmentError> {
        Self::build_inner(input, table, options, None)
    }

    /// Build the lattice for *input* against the typo noise model.
    ///
    /// Inventory spans segment exactly as [`SegmentLattice::build`] does. A
    /// typed stretch that is no syllable and no prefix of one may still become
    /// a segment when the noise model reaches it -- its candidates are the
    /// span's corrections and it carries the cheapest one's cost as a prior.
    ///
    /// # Errors
    ///
    /// The same conditions [`SegmentLattice::build`] rejects.
    pub fn build_corrections(
        input: &'input str,
        table: &SyllableTable,
        options: &SegmentOptions,
        corrections: &CorrectionTable,
    ) -> Result<Self, SegmentError> {
        Self::build_inner(input, table, options, Some(corrections))
    }

    /// Build *input*'s edge table, with corrected edges where the inventory
    /// cannot reach and *corrections* can.
    fn build_inner(
        input: &'input str,
        table: &SyllableTable,
        options: &SegmentOptions,
        corrections: Option<&CorrectionTable>,
    ) -> Result<Self, SegmentError> {
        if input.is_empty() {
            return Err(SegmentError::Empty);
        }
        if u32::try_from(input.len()).is_err() {
            return Err(SegmentError::TooLong { len: input.len() });
        }
        if let Some((offset, ch)) = input.char_indices().find(|(_, c)| !c.is_ascii_lowercase()) {
            return Err(SegmentError::InvalidCharacter { offset, ch });
        }

        let len = input.len();
        let max_span = if corrections.is_some() {
            MAX_CORRECTED_LEN
        } else {
            MAX_SYLLABLE_LEN
        };
        let mut edges: Vec<Vec<Segment>> = vec![Vec::new(); len];
        for start in 0..len {
            for end in (start + 1)..=len.min(start + max_span) {
                let span = &input[start..end];
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "len fits u32, checked above"
                )]
                let (s, e) = (start as u32, end as u32);
                if !table.prefix_range(span).is_empty() {
                    // A span that is an initial is read through its whole
                    // prefix range, which holds the syllable it may also
                    // spell (`m` is 呣 as well as every m- syllable). The
                    // exact edge would be a second edge over the same keys
                    // whose only difference is a smaller ambiguity cost, so
                    // it would win the path on price alone and spend k-best
                    // slots on duplicate span tuples.
                    if options.allow_abbreviation && SyllableTable::is_abbreviation(span) {
                        push(
                            &mut edges[start],
                            s,
                            e,
                            Readings::Syllables(table.prefix_range(span)),
                        );
                    } else if let Some(exact) = table.exact_range(span) {
                        push(&mut edges[start], s, e, Readings::Syllables(exact));
                    }
                    if options.allow_incomplete_tail && end == len {
                        push(
                            &mut edges[start],
                            s,
                            e,
                            Readings::Syllables(table.prefix_range(span)),
                        );
                    }
                } else if let Some(noise) = corrections {
                    let entries = noise.corrections(span);
                    if !entries.is_empty() {
                        push(&mut edges[start], s, e, Readings::Corrections(entries));
                    }
                }
            }
        }

        let lattice = Self { input, edges };
        let pruned = lattice.pruned_to_complete_paths()?;
        Ok(pruned)
    }

    /// Drop every edge that lies on no path from the start to the end.
    ///
    /// Without this the k-best sweep spends its slots extending prefixes that can
    /// never be completed, and `max_paths` stops meaning what it says.
    fn pruned_to_complete_paths(self) -> Result<Self, SegmentError> {
        let len = self.input.len();
        let mut reachable = vec![false; len + 1];
        reachable[0] = true;
        for start in 0..len {
            if !reachable[start] {
                continue;
            }
            for segment in &self.edges[start] {
                reachable[segment.end()] = true;
            }
        }
        let mut productive = vec![false; len + 1];
        productive[len] = true;
        for start in (0..len).rev() {
            productive[start] = self.edges[start].iter().any(|s| productive[s.end()]);
        }
        if !(reachable[len] && productive[0]) {
            return Err(SegmentError::NoSegmentation {
                input: self.input.to_owned(),
            });
        }
        let Self { input, mut edges } = self;
        for (start, bucket) in edges.iter_mut().enumerate() {
            if reachable[start] {
                bucket.retain(|s| productive[s.end()]);
            } else {
                bucket.clear();
            }
        }
        Ok(Self { input, edges })
    }

    /// The typed string this lattice was built from.
    #[must_use]
    pub const fn input(&self) -> &'input str {
        self.input
    }

    /// The `options.max_paths` cheapest readings, cheapest first.
    ///
    /// Never empty: [`SegmentLattice::build`] rejects an input with no reading.
    #[must_use]
    pub fn k_best(&self, options: &SegmentOptions) -> Vec<Segmentation> {
        let len = self.input.len();
        let k = options.max_paths.max(1);
        let mut best: Vec<Vec<Trace>> = vec![Vec::new(); len + 1];
        let mut pool: Vec<Vec<Trace>> = vec![Vec::new(); len + 1];
        best[0].push(Trace {
            cost: 0.0,
            back: None,
        });

        for position in 0..=len {
            if position > 0 {
                let mut traces = std::mem::take(&mut pool[position]);
                traces.sort_by(|a, b| a.cost.total_cmp(&b.cost));
                traces.truncate(k);
                best[position] = traces;
            }
            if best[position].is_empty() || position == len {
                continue;
            }
            for (edge, segment) in self.edges[position].iter().enumerate() {
                let step = segment.cost(options);
                for (rank, trace) in best[position].iter().enumerate() {
                    pool[segment.end()].push(Trace {
                        cost: trace.cost + step,
                        back: Some((position, edge, rank)),
                    });
                }
            }
        }

        best[len]
            .iter()
            .map(|trace| self.reconstruct(&best, *trace))
            .collect()
    }

    /// The cheapest reading exactly *positions* characters long whose every
    /// segment passes *admits*, given the segment and its position; `None`
    /// when no reading does.
    ///
    /// [`SegmentLattice::k_best`] ranks the readings of the keystrokes alone.
    /// This instead asks which reading a known sentence was typed along: a
    /// caller that tests each segment against the sentence's character at
    /// that position learns which span typed which character. The search
    /// covers every edge of the lattice, not only the `max_paths` cheapest
    /// readings, so a sentence typed along an unconventional reading is still
    /// found. Ties go to the reading found first, which is deterministic.
    #[must_use]
    pub fn cheapest_reading(
        &self,
        positions: usize,
        admits: impl Fn(&Segment, usize) -> bool,
        options: &SegmentOptions,
    ) -> Option<Segmentation> {
        let len = self.input.len();
        let width = positions + 1;
        // `best[offset * width + count]`: the cheapest way to cover the first
        // `offset` bytes with `count` admitted segments, and the edge that
        // ended it, as `(cost, previous offset, edge index)`.
        let mut best: Vec<Option<(f32, usize, usize)>> = vec![None; (len + 1) * width];
        best[0] = Some((0.0, 0, 0));
        for offset in 0..len {
            for count in 0..positions {
                let Some((cost, _, _)) = best[offset * width + count] else {
                    continue;
                };
                for (edge, segment) in self.edges[offset].iter().enumerate() {
                    if !admits(segment, count) {
                        continue;
                    }
                    let total = cost + segment.cost(options);
                    let slot = &mut best[segment.end() * width + count + 1];
                    if slot.is_none_or(|(known, _, _)| total < known) {
                        *slot = Some((total, offset, edge));
                    }
                }
            }
        }
        let (cost, _, _) = best[len * width + positions]?;
        let mut segments = Vec::with_capacity(positions);
        let (mut offset, mut count) = (len, positions);
        while count > 0 {
            let (_, previous, edge) = best[offset * width + count]?;
            segments.push(self.edges[previous][edge].clone());
            offset = previous;
            count -= 1;
        }
        segments.reverse();
        Some(Segmentation { segments, cost })
    }

    /// Walk a finished trace back to the start, producing the reading it encodes.
    fn reconstruct(&self, best: &[Vec<Trace>], trace: Trace) -> Segmentation {
        let cost = trace.cost;
        let mut segments = Vec::new();
        let mut current = trace;
        while let Some((position, edge, rank)) = current.back {
            segments.push(self.edges[position][edge].clone());
            current = best[position][rank];
        }
        segments.reverse();
        Segmentation { segments, cost }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spellings(table: &SyllableTable, seg: &Segmentation) -> Vec<String> {
        seg.segments()
            .iter()
            .map(|s| match s.readings() {
                Readings::Corrections(entries) => format!("{}?", entries.len()),
                Readings::Syllables(range) if range.len() == 1 => table
                    .spelling(range.iter().next().expect("singleton"))
                    .to_owned(),
                Readings::Syllables(range) => format!("{}*", range.len()),
            })
            .collect()
    }

    fn offline() -> SegmentOptions {
        SegmentOptions {
            allow_incomplete_tail: false,
            ..SegmentOptions::default()
        }
    }

    #[test]
    fn full_pinyin_beats_the_abbreviated_reading_of_the_same_keys() {
        let table = SyllableTable::load();
        let options = offline();
        let lattice = SegmentLattice::build("nihao", &table, &options).expect("nihao reads");
        let paths = lattice.k_best(&options);
        assert_eq!(spellings(&table, &paths[0]), ["ni", "hao"]);
    }

    #[test]
    fn a_pure_abbreviation_reads_once() {
        // 中国人民 typed as initials. The trailing `m` is an initial and also the
        // syllable 呒, and one segment carries both: its readings are the whole
        // m- range, so 呒 and 民 compete inside one position instead of as two
        // readings that differ only in price.
        let table = SyllableTable::load();
        let options = offline();
        let lattice = SegmentLattice::build("zgrm", &table, &options).expect("zgrm reads");
        let paths = lattice.k_best(&options);
        assert_eq!(paths.len(), 1, "got {paths:?}");
        let m = &paths[0].segments()[3];
        let range = m.syllables().expect("`m` is on the inventory");
        assert_eq!(range, table.prefix_range("m"));
        assert!(range.contains(table.exact("m").expect("`m` is a syllable")));
        assert!(m.ambiguity() > 1, "the m- initials stay readable");
    }

    #[test]
    fn an_initial_that_spells_a_syllable_has_one_edge() {
        // `a e o m n` are each a syllable and an initial. Two edges over the same
        // key would read the same characters at two prices, and the cheaper
        // exact one would win every path it is on.
        let table = SyllableTable::load();
        let options = offline();
        for letter in ["a", "e", "o", "m", "n"] {
            assert!(table.exact(letter).is_some(), "{letter} is a syllable");
            let lattice = SegmentLattice::build(letter, &table, &options).expect("a letter reads");
            assert_eq!(
                lattice.edges[0],
                [Segment {
                    start: 0,
                    end: 1,
                    readings: Readings::Syllables(table.prefix_range(letter)),
                }],
                "{letter}"
            );
        }
    }

    #[test]
    fn readings_never_repeat_a_span_tuple() {
        // Every reading of the batch cuts the keys differently: a repeat would
        // spend one of the `max_paths` slots on spans already in the batch.
        let table = SyllableTable::load();
        let options = offline();
        for input in ["nmhm", "wmdzgrmao", "zhenme", "aeonm"] {
            let paths = SegmentLattice::build(input, &table, &options)
                .expect("the input reads")
                .k_best(&options);
            let mut tuples: Vec<Vec<(usize, usize)>> = paths
                .iter()
                .map(|path| {
                    path.segments()
                        .iter()
                        .map(|s| (s.start(), s.end()))
                        .collect()
                })
                .collect();
            let readings = tuples.len();
            tuples.sort();
            tuples.dedup();
            assert_eq!(tuples.len(), readings, "{input}: {paths:?}");
        }
    }

    #[test]
    fn without_abbreviations_a_lone_letter_is_its_syllable() {
        let table = SyllableTable::load();
        let strict = SegmentOptions {
            allow_abbreviation: false,
            allow_incomplete_tail: false,
            ..SegmentOptions::default()
        };
        let lattice = SegmentLattice::build("e", &table, &strict).expect("`e` is a syllable");
        assert_eq!(
            lattice.edges[0],
            [Segment {
                start: 0,
                end: 1,
                readings: Readings::Syllables(table.exact_range("e").expect("`e` is a syllable")),
            }]
        );
    }

    #[test]
    fn abbreviations_make_segmentation_total() {
        // With initials enabled every letter is a segment, so no keystroke
        // sequence over the alphabet can fail to read. Callers that need failure
        // to mean something must turn abbreviations off.
        let table = SyllableTable::load();
        let options = offline();
        for input in ["zzz", "qqq", "xyzw", "bpmf"] {
            let lattice = SegmentLattice::build(input, &table, &options)
                .unwrap_or_else(|e| panic!("{input} should read: {e}"));
            assert!(!lattice.k_best(&options).is_empty(), "{input} segments");
        }
    }

    #[test]
    fn two_letter_initials_stay_together() {
        // `zhg` is 中国 typed as zh + g, not z + h + g.
        let table = SyllableTable::load();
        let options = offline();
        let lattice = SegmentLattice::build("zhg", &table, &options).expect("zhg reads");
        let best = &lattice.k_best(&options)[0];
        assert_eq!(best.len(), 2, "got {:?}", spellings(&table, best));
    }

    #[test]
    fn genuinely_ambiguous_input_keeps_both_lengths() {
        // 西安 (xi + an) and 咸 (xian) are the same keystrokes; both must survive
        // into the batch the encoder scores.
        let table = SyllableTable::load();
        let options = offline();
        let lattice = SegmentLattice::build("xian", &table, &options).expect("xian reads");
        let readings: Vec<_> = lattice
            .k_best(&options)
            .iter()
            .map(|p| spellings(&table, p))
            .collect();
        assert!(
            readings.contains(&vec!["xian".to_owned()]),
            "got {readings:?}"
        );
        assert!(
            readings.contains(&vec!["xi".to_owned(), "an".to_owned()]),
            "got {readings:?}"
        );
    }

    #[test]
    fn k_best_is_ordered_and_bounded() {
        let table = SyllableTable::load();
        let options = SegmentOptions {
            max_paths: 3,
            ..offline()
        };
        let lattice = SegmentLattice::build("nihaoshijie", &table, &options).expect("reads");
        let paths = lattice.k_best(&options);
        assert!(paths.len() <= 3);
        assert!(paths.windows(2).all(|w| w[0].cost() <= w[1].cost()));
    }

    #[test]
    fn an_unfinished_syllable_reads_only_when_allowed() {
        // Abbreviations have to be off for this to be observable at all -- see
        // `abbreviations_make_segmentation_total`. `zhongg` is 中国 with the
        // second syllable one keystroke in.
        let table = SyllableTable::load();
        let strict = SegmentOptions {
            allow_abbreviation: false,
            allow_incomplete_tail: false,
            ..SegmentOptions::default()
        };
        let typing = SegmentOptions {
            allow_incomplete_tail: true,
            ..strict.clone()
        };
        let lattice = SegmentLattice::build("zhongg", &table, &typing).expect("mid-word reads");
        let best = &lattice.k_best(&typing)[0];
        assert_eq!(best.len(), 2);
        assert_eq!(
            best.segments()[1].ambiguity(),
            table.prefix_range("g").len()
        );
        assert_eq!(
            SegmentLattice::build("zhongg", &table, &strict).expect_err("zhongg must not read"),
            SegmentError::NoSegmentation {
                input: "zhongg".to_owned()
            }
        );
    }

    #[test]
    fn input_outside_the_keyboard_alphabet_is_rejected() {
        let table = SyllableTable::load();
        let options = SegmentOptions::default();
        assert_eq!(
            SegmentLattice::build("", &table, &options).expect_err("an empty input must not read"),
            SegmentError::Empty
        );
        assert_eq!(
            SegmentLattice::build("ni3", &table, &options).expect_err("ni3 must not read"),
            SegmentError::InvalidCharacter { offset: 2, ch: '3' }
        );
        assert_eq!(
            SegmentLattice::build("ni好", &table, &options).expect_err("ni好 must not read"),
            SegmentError::InvalidCharacter {
                offset: 2,
                ch: '好'
            }
        );
    }

    #[test]
    fn segments_tile_the_input_exactly() {
        let table = SyllableTable::load();
        let options = offline();
        let lattice =
            SegmentLattice::build("womenzaijianzhongwen", &table, &options).expect("reads");
        for path in lattice.k_best(&options) {
            let mut cursor = 0;
            for segment in path.segments() {
                assert_eq!(segment.start(), cursor, "gap or overlap in {path:?}");
                cursor = segment.end();
            }
            assert_eq!(cursor, lattice.input().len(), "path did not span the input");
        }
    }

    #[test]
    fn a_typo_segments_only_under_corrections() {
        // `wo jai jia` typed with `jai` for `jia`: with abbreviations off no
        // inventory reading tiles the input, so without the noise model it
        // cannot be read at all.
        let table = SyllableTable::load();
        let corrections = CorrectionTable::load(&table).expect("the noise table parses");
        let strict = SegmentOptions {
            allow_abbreviation: false,
            allow_incomplete_tail: false,
            ..SegmentOptions::default()
        };
        assert_eq!(
            SegmentLattice::build("wojaijia", &table, &strict)
                .expect_err("a typo must not segment without corrections"),
            SegmentError::NoSegmentation {
                input: "wojaijia".to_owned()
            }
        );
        let lattice = SegmentLattice::build_corrections("wojaijia", &table, &strict, &corrections)
            .expect("the typo reads through corrections");
        let best = &lattice.k_best(&strict)[0];
        let corrected: Vec<_> = best.segments().iter().filter(|s| s.corrected()).collect();
        assert_eq!(corrected.len(), 1, "exactly one segment is a typo");
        let segment = corrected[0];
        assert_eq!(
            &lattice.input()[segment.start()..segment.end()],
            "jai",
            "the corrected segment covers the typoed span"
        );
        assert_eq!(
            segment
                .corrections()
                .expect("a corrected segment holds its entries")[0]
                .syllable(),
            "jia",
            "the cheapest correction of `jai` must be `jia`"
        );
    }

    #[test]
    fn corrections_never_reorder_clean_segmentations() {
        // The invariance the decoder depends on: with the noise model on, a
        // clean path's cost is byte-identical to today's -- a corrected
        // segment only ever adds a positive prior -- so the clean subset of
        // the batch is a subsequence of the plain lattice's list, in order.
        let table = SyllableTable::load();
        let corrections = CorrectionTable::load(&table).expect("the noise table parses");
        let options = SegmentOptions {
            max_paths: 8,
            ..offline()
        };
        for input in [
            "nihao",
            "xian",
            "zgrm",
            "womenzaijianzhongwen",
            "zhongguorenm",
        ] {
            let plain = SegmentLattice::build(input, &table, &options).expect("reads");
            let noisy = SegmentLattice::build_corrections(input, &table, &options, &corrections)
                .expect("reads");
            let survivors: Vec<_> = noisy
                .k_best(&options)
                .into_iter()
                .filter(|path| path.segments().iter().all(|s| !s.corrected()))
                .collect();
            let mut expected = plain.k_best(&options).into_iter();
            for survivor in &survivors {
                assert_eq!(
                    Some(survivor),
                    expected.find(|e| e == survivor).as_ref(),
                    "{input:?}: clean paths must keep today's order"
                );
            }
        }
    }

    #[test]
    fn the_correction_prior_prices_typoed_paths() {
        // `jai` is `jia` one transposition in -- or `j` + `ai` already. The
        // batch must offer both: the summed prior puts the honest path first
        // and the corrected one close enough for the decoder to pick it.
        let table = SyllableTable::load();
        let corrections = CorrectionTable::load(&table).expect("the noise table parses");
        let options = offline();
        let lattice = SegmentLattice::build_corrections("jai", &table, &options, &corrections)
            .expect("jai reads");
        let paths = lattice.k_best(&options);
        assert!(
            paths[0].segments().iter().all(|s| !s.corrected()),
            "the clean j+ai reading still ranks ahead"
        );
        let corrected = paths
            .iter()
            .find(|p| p.segments().iter().any(Segment::corrected))
            .expect("the corrected reading is in the batch");
        let span_cost = corrections
            .corrections("jai")
            .first()
            .expect("jai has corrections")
            .cost();
        assert!(
            f64::from(corrected.cost()) > span_cost,
            "the path cost includes the correction prior"
        );
    }

    /// The span each character of *text* was typed with, along the cheapest
    /// reading of *input* that spells it; `None` when none does.
    fn typed_along(input: &str, text: &str) -> Option<Vec<String>> {
        let table = SyllableTable::load();
        let lexicon = crate::Lexicon::load(&table).expect("the generated tables agree");
        let options = offline();
        let lattice = SegmentLattice::build(input, &table, &options).expect("the input reads");
        let ids: Vec<_> = text
            .chars()
            .map(|ch| lexicon.id_of(ch).expect("the test text is in the lexicon"))
            .collect();
        let reading = lattice.cheapest_reading(
            ids.len(),
            |segment, position| {
                segment.syllables().is_some_and(|range| {
                    lexicon
                        .readings(ids[position])
                        .iter()
                        .any(|&reading| range.contains(reading))
                })
            },
            &options,
        )?;
        Some(
            reading
                .segments()
                .iter()
                .map(|segment| input[segment.start()..segment.end()].to_owned())
                .collect(),
        )
    }

    #[test]
    fn the_sentence_picks_the_reading_it_was_typed_along() {
        // `xian` reads best as one syllable, but 西安 can only have been typed
        // as two, and the reading follows the sentence rather than the cost.
        assert_eq!(typed_along("xian", "先").expect("先 spells"), ["xian"]);
        assert_eq!(
            typed_along("xian", "西安").expect("西安 spells"),
            ["xi", "an"]
        );
        assert_eq!(
            typed_along("tashiwojiejie", "她是我姐姐").expect("the full pinyin spells"),
            ["ta", "shi", "wo", "jie", "jie"]
        );
        assert_eq!(
            typed_along("zhg", "中国").expect("the initials spell"),
            ["zh", "g"]
        );
    }

    #[test]
    fn a_sentence_the_keystrokes_cannot_spell_has_no_reading() {
        assert_eq!(typed_along("nihao", "你们"), None);
        assert_eq!(typed_along("nihao", "你"), None);
    }
}
