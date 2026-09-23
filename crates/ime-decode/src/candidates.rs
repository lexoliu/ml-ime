//! The decoder's input: what each output position is allowed to be.

use crate::DecodeError;
use ime_pinyin::{CharId, Lexicon, Segmentation};

/// One reading of the typed string, resolved down to characters.
///
/// The segmentation said how many output positions there are and which syllables
/// each may take; this is the same thing with [`Lexicon::mask_into`] applied, so
/// the decoder never touches the pinyin layer.
#[derive(Clone, Debug)]
pub struct CandidatePath {
    positions: Vec<Vec<CharId>>,
    cost: f32,
}

impl CandidatePath {
    /// The candidate characters at each output position, left to right. Every
    /// entry is sorted, deduplicated and non-empty.
    #[must_use]
    pub fn positions(&self) -> &[Vec<CharId>] {
        &self.positions
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
                lexicon.mask_into(segment.syllables(), &mut mask);
                if mask.is_empty() {
                    return Err(DecodeError::EmptyCandidateSet { path, position });
                }
                positions.push(mask);
            }
            paths.push(CandidatePath {
                positions,
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
