//! The n-gram baseline as a complete input method.

use ime_decode::{BeamOptions, Candidates, Hypothesis, Uniform, decode};
use ime_eval::{Hypothesize, Request};
use ime_ngram::NgramModel;
use ime_pinyin::{Lexicon, SegmentLattice, SegmentOptions, SyllableTable};
use std::num::NonZeroUsize;
use thiserror::Error;

/// Why a typed string could not be turned into candidates.
#[derive(Debug, Error)]
pub enum BaselineError {
    /// The keystrokes could not be segmented.
    #[error("could not segment {input:?}")]
    Segment {
        /// The offending input.
        input: String,
        /// What segmentation objected to.
        #[source]
        source: ime_pinyin::SegmentError,
    },
    /// The candidate lattice could not be decoded.
    #[error("could not decode {input:?}")]
    Decode {
        /// The offending input.
        input: String,
        /// What decoding objected to.
        #[source]
        source: ime_decode::DecodeError,
    },
}

/// Segmentation lattice, homophone masks and an n-gram, wired together.
///
/// This is the conventional architecture in full: a lattice over the keystrokes,
/// a hard per-position mask, and a language model that ranks what the mask
/// allows. The neural route replaces exactly one part of it -- [`Uniform`] --
/// which is what makes the comparison a comparison.
pub struct Baseline {
    table: SyllableTable,
    lexicon: Lexicon,
    model: NgramModel,
    segment: SegmentOptions,
    beam: BeamOptions,
}

impl Baseline {
    /// Assemble a baseline from a loaded model.
    #[must_use]
    pub fn new(
        table: SyllableTable,
        lexicon: Lexicon,
        model: NgramModel,
        segment: SegmentOptions,
        beam: BeamOptions,
    ) -> Self {
        Self {
            table,
            lexicon,
            model,
            segment,
            beam,
        }
    }

    /// The lexicon candidates are drawn from.
    #[must_use]
    pub const fn lexicon(&self) -> &Lexicon {
        &self.lexicon
    }

    /// Decode *pinyin* into at most `top_k` ranked hypotheses.
    ///
    /// # Errors
    ///
    /// If the keystrokes have no reading, or the lattice cannot be decoded.
    pub fn candidates(
        &self,
        pinyin: &str,
        top_k: NonZeroUsize,
    ) -> Result<Vec<Hypothesis>, BaselineError> {
        let lattice =
            SegmentLattice::build(pinyin, &self.table, &self.segment).map_err(|source| {
                BaselineError::Segment {
                    input: pinyin.to_owned(),
                    source,
                }
            })?;
        let batch =
            Candidates::build(&lattice.k_best(&self.segment), &self.lexicon).map_err(|source| {
                BaselineError::Decode {
                    input: pinyin.to_owned(),
                    source,
                }
            })?;
        let options = BeamOptions {
            top_k,
            ..self.beam.clone()
        };
        decode(&batch, &Uniform, &self.model, &options).map_err(|source| BaselineError::Decode {
            input: pinyin.to_owned(),
            source,
        })
    }
}

/// The baseline is deliberately blind to context: it is the control, and the
/// number worth having is how much the neural route gains by not being.
impl Hypothesize for Baseline {
    type Error = BaselineError;

    fn hypotheses(&self, request: &Request<'_>) -> Result<Vec<String>, Self::Error> {
        Ok(self
            .candidates(request.pinyin, request.top_k)?
            .iter()
            .map(|hypothesis| hypothesis.text(&self.lexicon))
            .collect())
    }
}
