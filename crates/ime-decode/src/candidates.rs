//! The decoder's input: what each output position is allowed to be.

use crate::DecodeError;
use ime_pinyin::{CharId, CorrectionTable, Lexicon, Segmentation, SyllableTable};

/// One reading of the typed string, resolved down to characters.
///
/// The segmentation said how many output positions there are and which syllables
/// each may take; this is the same thing with [`Lexicon::mask_into`] applied, so
/// the decoder never touches the pinyin layer.
#[derive(Clone, Debug)]
pub struct CandidatePath {
    positions: Vec<Vec<CharId>>,
    /// Each candidate's correction prior, aligned with `positions`: the
    /// cheapest correction's `-log P(span | syllable)` the character is
    /// reachable through, 0 for the span's own readings. `None` for a path
    /// resolved without the noise model -- every candidate costs 0 there.
    priors: Option<Vec<Vec<f64>>>,
    cost: f32,
}

impl CandidatePath {
    /// The candidate characters at each output position, left to right. Every
    /// entry is sorted, deduplicated and non-empty.
    #[must_use]
    pub fn positions(&self) -> &[Vec<CharId>] {
        &self.positions
    }

    /// The correction prior of `positions()[position][slot]`: the cheapest
    /// correction's `-log P(span | syllable)` that admits the character, 0 for
    /// the span's own readings and 0 when the path was resolved without the
    /// noise model.
    #[must_use]
    pub fn prior(&self, position: usize, slot: usize) -> f64 {
        self.priors
            .as_ref()
            .map_or(0.0, |priors| priors[position][slot])
    }

    /// How many characters this reading produces.
    #[must_use]
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Whether this reading produces no characters. Never true for a path built
    /// by [`Candidates::build`].
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// The segmentation cost this reading carried over. Lower is a more
    /// conventional way to cut the keystrokes up.
    #[must_use]
    pub const fn cost(&self) -> f32 {
        self.cost
    }
}

/// A span's memo slot in one [`Candidates::build_corrections`] call: its typed
/// letters plus how many syllables its segment's own readings cover (the exact
/// range is a one-syllable set, the prefix range whatever the incomplete tail
/// reaches, a corrected segment none).
type SpanKey<'a> = (&'a str, Option<usize>);

/// What a span resolves to once per call: its union of candidate characters
/// and the correction prior each carries.
type ResolvedSpan = (Vec<CharId>, Vec<f64>);

/// Every reading of one typed string, ready to decode.
///
/// The paths differ in length, which is why they are a batch and not a branch:
/// `xian` is 西安 or 咸, and both go through the model together.
#[derive(Clone, Debug)]
pub struct Candidates {
    paths: Vec<CandidatePath>,
}

impl Candidates {
    /// Resolve *segmentations* into per-position candidate sets.
    ///
    /// # Errors
    ///
    /// If there are no segmentations, if one of them is empty, or if a position
    /// admits no character at all -- which means the two generated tables
    /// disagree, since every syllable in the inventory has at least one
    /// homophone.
    pub fn build(segmentations: &[Segmentation], lexicon: &Lexicon) -> Result<Self, DecodeError> {
        if segmentations.is_empty() {
            return Err(DecodeError::NoSegmentations);
        }
        let mut paths = Vec::with_capacity(segmentations.len());
        for (path, segmentation) in segmentations.iter().enumerate() {
            if segmentation.is_empty() {
                return Err(DecodeError::EmptySegmentation { path });
            }
            let mut positions = Vec::with_capacity(segmentation.len());
            for (position, segment) in segmentation.segments().iter().enumerate() {
                let mut mask = Vec::new();
                if let Some(range) = segment.syllables() {
                    lexicon.mask_into(range, &mut mask);
                }
                if mask.is_empty() {
                    return Err(DecodeError::EmptyCandidateSet { path, position });
                }
                positions.push(mask);
            }
            paths.push(CandidatePath {
                positions,
                priors: None,
                cost: segmentation.cost(),
            });
        }
        Ok(Self { paths })
    }

    /// Resolve *segmentations* into per-position candidate sets under the typo
    /// noise model.
    ///
    /// Every position's set is the union of the homophones over its span's
    /// `corrections` -- the same widening the training side's
    /// `CandidateSpace.resolve` applies, so a corrected span admits every
    /// syllable the noise model could have meant, and an honest span keeps its
    /// own readings through the cost-0 self entry. Each candidate also carries
    /// its correction prior (see [`CandidatePath::prior`]), so a character
    /// reachable only through a correction pays that correction's cost while
    /// the segment's own readings stay free. *pinyin* is the typed input the
    /// segmentations were built on; the spans are sliced back out of it.
    ///
    /// # Errors
    ///
    /// The same conditions [`Candidates::build`] rejects, plus a position whose
    /// corrections admit no character -- the empty-union case a typo no
    /// syllable reaches within two edits produces.
    pub fn build_corrections(
        segmentations: &[Segmentation],
        pinyin: &str,
        syllables: &SyllableTable,
        corrections: &CorrectionTable,
        lexicon: &Lexicon,
    ) -> Result<Self, DecodeError> {
        if segmentations.is_empty() {
            return Err(DecodeError::NoSegmentations);
        }
        // One keystroke asks about the same spans over and over -- every
        // reading shares most of them -- so the union a span resolves to is
        // computed once per call, not once per position. The prior the same
        // span carries still depends on its segment's own readings: the exact
        // syllable mid-word, the whole prefix range at the incomplete tail,
        // none on a corrected segment -- so the memo key carries that too.
        let mut unions: std::collections::HashMap<SpanKey, ResolvedSpan> =
            std::collections::HashMap::new();
        let mut paths = Vec::with_capacity(segmentations.len());
        for (path, segmentation) in segmentations.iter().enumerate() {
            if segmentation.is_empty() {
                return Err(DecodeError::EmptySegmentation { path });
            }
            let mut positions = Vec::with_capacity(segmentation.len());
            let mut priors = Vec::with_capacity(segmentation.len());
            for (position, segment) in segmentation.segments().iter().enumerate() {
                let span = &pinyin[segment.start()..segment.end()];
                let own = segment.syllables();
                let resolved = unions
                    .entry((span, own.map(ime_pinyin::SyllableRange::len)))
                    .or_insert_with(|| {
                        let priors = lexicon.corrections_priors(span, own, syllables, corrections);
                        (
                            priors.iter().map(|&(id, _)| id).collect(),
                            priors.iter().map(|&(_, cost)| cost).collect(),
                        )
                    });
                if resolved.0.is_empty() {
                    return Err(DecodeError::EmptyCandidateSet { path, position });
                }
                positions.push(resolved.0.clone());
                priors.push(resolved.1.clone());
            }
            paths.push(CandidatePath {
                positions,
                priors: Some(priors),
                cost: segmentation.cost(),
            });
        }
        Ok(Self { paths })
    }

    /// The readings, in the order [`SegmentLattice::k_best`] ranked them.
    ///
    /// [`SegmentLattice::k_best`]: ime_pinyin::SegmentLattice::k_best
    #[must_use]
    pub fn paths(&self) -> &[CandidatePath] {
        &self.paths
    }

    /// How many readings are in the batch. Never zero.
    #[must_use]
    pub fn len(&self) -> usize {
        self.paths.len()
    }

    /// Whether the batch is empty. Never true for a batch built by
    /// [`Candidates::build`].
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }
}
