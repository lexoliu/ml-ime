//! Turning a keystroke sequence into candidate readings.
//!
//! A typed string does not determine how many characters it will become. `xian`
//! is one syllable or two; `zgrm` is four abbreviated ones. So segmentation is a
//! lattice, not a function, and what this module produces is the *k* cheapest
//! paths through it -- which downstream is exactly the batch of candidate lengths
//! that go into a single encoder forward.

use crate::syllable::{MAX_SYLLABLE_LEN, SyllableRange, SyllableTable};
use thiserror::Error;

/// One output character position: the span of input it consumes, and every
/// syllable that span can be read as.
///
/// The three cases an IME distinguishes -- a complete syllable, an initial-only
/// abbreviation, a syllable still being typed -- differ only in how wide
/// [`Segment::syllables`] is, so they need no tag.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Segment {
    start: u32,
    end: u32,
    syllables: SyllableRange,
}

impl Segment {
    /// Byte offset where this segment starts in the typed string.
    #[must_use]
    pub const fn start(self) -> usize {
        self.start as usize
    }

    /// Byte offset one past this segment's last keystroke.
    #[must_use]
    pub const fn end(self) -> usize {
        self.end as usize
    }

    /// The readings this segment admits.
    #[must_use]
    pub const fn syllables(self) -> SyllableRange {
        self.syllables
    }

    /// How many readings this segment admits. One means the user typed the
    /// syllable out in full.
    #[must_use]
    pub const fn ambiguity(self) -> usize {
        self.syllables.len()
    }

    /// This segment's contribution to a path's cost.
    #[must_use]
    fn cost(self, options: &SegmentOptions) -> f32 {
        #[expect(
            clippy::cast_precision_loss,
            reason = "ambiguity is at most the inventory size"
        )]
        let ambiguity = self.ambiguity() as f32;
        options.segment_cost + options.ambiguity_weight * ambiguity.ln()
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
#[derive(Clone, Debug)]
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
    cost: f32,
    /// `(start position, index into `edges[start]`, rank within `best[start]`)`.
    /// Absent only for the empty path at position zero.
    back: Option<(usize, usize, usize)>,
}

impl<'input> SegmentLattice<'input> {
    /// Build the lattice for *input*.
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
        let mut edges: Vec<Vec<Segment>> = vec![Vec::new(); len];
        for start in 0..len {
            for end in (start + 1)..=len.min(start + MAX_SYLLABLE_LEN) {
                let span = &input[start..end];
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "len fits u32, checked above"
                )]
                let (s, e) = (start as u32, end as u32);
                let mut push = |syllables: SyllableRange| {
                    let segment = Segment {
                        start: s,
                        end: e,
                        syllables,
                    };
                    if !syllables.is_empty() && !edges[start].contains(&segment) {
                        edges[start].push(segment);
                    }
                };
                if let Some(exact) = table.exact_range(span) {
                    push(exact);
                }
                if options.allow_abbreviation && SyllableTable::is_abbreviation(span) {
                    push(table.prefix_range(span));
                }
                if options.allow_incomplete_tail && end == len {
                    push(table.prefix_range(span));
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

    /// Walk a finished trace back to the start, producing the reading it encodes.
    fn reconstruct(&self, best: &[Vec<Trace>], trace: Trace) -> Segmentation {
        let cost = trace.cost;
        let mut segments = Vec::new();
        let mut current = trace;
        while let Some((position, edge, rank)) = current.back {
            segments.push(self.edges[position][edge]);
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
            .map(|s| {
                if s.ambiguity() == 1 {
                    table
                        .spelling(s.syllables().iter().next().expect("singleton"))
                        .to_owned()
                } else {
                    format!("{}*", s.ambiguity())
                }
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
    fn a_pure_abbreviation_still_reads() {
        // 中国人民 typed as initials. Every path is four characters long, but the
        // cheapest one reads the trailing `m` as the syllable 呒 rather than as an
        // initial, because a single letter can be both. Both readings belong in
        // the batch; only the language model can tell them apart.
        let table = SyllableTable::load();
        let options = offline();
        let lattice = SegmentLattice::build("zgrm", &table, &options).expect("zgrm reads");
        let paths = lattice.k_best(&options);
        assert!(
            paths.iter().all(|p| p.len() == 4),
            "one character per typed initial"
        );
        assert!(
            paths
                .iter()
                .any(|p| p.segments().iter().all(|s| s.ambiguity() > 1)),
            "the all-initials reading must survive into the batch"
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
            assert!(!lattice.k_best(&options).is_empty());
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
}
