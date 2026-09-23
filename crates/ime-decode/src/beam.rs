//! Beam Viterbi over the candidate lattice.

use crate::candidates::{CandidatePath, Candidates};
use crate::score::{Emission, History, MAX_HISTORY, Transition};
use ime_pinyin::{CharId, Lexicon};
use std::collections::{HashMap, HashSet, hash_map::Entry};
use std::num::NonZeroUsize;

/// How wide the search is and how it trades segmentation against language model.
#[derive(Clone, Debug)]
pub struct BeamOptions {
    /// How many beam states survive at each position. A trigram state is a pair
    /// of characters, so the beam has to be wide enough to hold several distinct
    /// second-to-last characters before the trigram term can do anything.
    pub beam_width: NonZeroUsize,
    /// How many hypotheses [`decode`] returns.
    pub top_k: NonZeroUsize,
    /// Weight on a reading's segmentation cost when it is folded into the
    /// sequence score. Zero lets the language model choose the reading unaided.
    pub segmentation_weight: f32,
}

impl Default for BeamOptions {
    fn default() -> Self {
        Self {
            beam_width: NonZeroUsize::new(16).expect("16 is not zero"),
            top_k: NonZeroUsize::new(8).expect("8 is not zero"),
            segmentation_weight: 1.0,
        }
    }
}

/// One decoded sentence.
#[derive(Clone, Debug)]
pub struct Hypothesis {
    chars: Vec<CharId>,
    score: f32,
    path: usize,
}

impl Hypothesis {
    /// The characters, left to right.
    #[must_use]
    pub fn chars(&self) -> &[CharId] {
        &self.chars
    }

    /// Total score: emissions, transitions, the end-of-sequence term, and the
    /// weighted segmentation cost. Larger is better.
    #[must_use]
    pub const fn score(&self) -> f32 {
        self.score
    }

    /// Which reading of the keystrokes this hypothesis came from, as an index
    /// into [`Candidates::paths`].
    #[must_use]
    pub const fn path(&self) -> usize {
        self.path
    }

    /// The sentence as text.
    ///
    /// # Panics
    ///
    /// If *lexicon* is not the one the hypothesis was decoded against.
    #[must_use]
    pub fn text(&self, lexicon: &Lexicon) -> String {
        self.chars.iter().map(|id| lexicon.character(*id)).collect()
    }
}

/// A live search state: the last few characters, the score of the best way to
/// reach them, and where that way came from.
#[derive(Copy, Clone, Debug)]
struct Beam {
    history: History,
    score: f32,
    ch: CharId,
    /// Index into the previous position's surviving beams. Meaningless at
    /// position zero.
    parent: usize,
}

/// Decode every reading in *candidates* and merge the results.
///
/// Hypotheses from different readings compete on one scale -- sequence score
/// minus the weighted segmentation cost -- and identical sentences reached by
/// different readings collapse to the best-scoring one.
///
/// # Errors
///
/// If *candidates* is empty, which [`Candidates::build`] already rules out.
pub fn decode<E, T>(
    candidates: &Candidates,
    emission: &E,
    transition: &T,
    options: &BeamOptions,
) -> Result<Vec<Hypothesis>, crate::DecodeError>
where
    E: Emission,
    T: Transition,
{
    const {
        assert!(
            T::HISTORY >= 1,
            "a transition model must condition on at least one character"
        );
        assert!(
            T::HISTORY <= MAX_HISTORY,
            "a transition model cannot condition on more than MAX_HISTORY characters"
        );
    }
    if candidates.is_empty() {
        return Err(crate::DecodeError::NoSegmentations);
    }

    let mut merged: Vec<Hypothesis> = Vec::new();
    for (path, reading) in candidates.paths().iter().enumerate() {
        let penalty = options.segmentation_weight * reading.cost();
        for mut hypothesis in decode_path(path, reading, emission, transition, options) {
            hypothesis.score -= penalty;
            merged.push(hypothesis);
        }
    }
    merged.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));

    let mut seen: HashSet<&[CharId]> = HashSet::new();
    let mut keep = Vec::with_capacity(options.top_k.get());
    for (index, hypothesis) in merged.iter().enumerate() {
        if keep.len() == options.top_k.get() {
            break;
        }
        if seen.insert(hypothesis.chars.as_slice()) {
            keep.push(index);
        }
    }
    Ok(keep
        .into_iter()
        .map(|index| merged[index].clone())
        .collect())
}

/// Beam Viterbi over a single reading.
fn decode_path<E, T>(
    path: usize,
    reading: &CandidatePath,
    emission: &E,
    transition: &T,
    options: &BeamOptions,
) -> Vec<Hypothesis>
where
    E: Emission,
    T: Transition,
{
    let width = options.beam_width.get();
    let mut beams: Vec<Vec<Beam>> = Vec::with_capacity(reading.len());
    let mut index: HashMap<History, usize> = HashMap::new();

    for (position, allowed) in reading.positions().iter().enumerate() {
        let mut next: Vec<Beam> = Vec::with_capacity(allowed.len());
        index.clear();
        if position == 0 {
            for &ch in allowed {
                let score = emission.score(path, 0, ch) + transition.score(History::START, ch);
                relax(
                    &mut next,
                    &mut index,
                    Beam {
                        history: History::START.extended(ch).truncated(T::HISTORY),
                        score,
                        ch,
                        parent: 0,
                    },
                );
            }
        } else {
            for (parent, beam) in beams[position - 1].iter().enumerate() {
                for &ch in allowed {
                    let score = beam.score
                        + emission.score(path, position, ch)
                        + transition.score(beam.history, ch);
                    relax(
                        &mut next,
                        &mut index,
                        Beam {
                            history: beam.history.extended(ch).truncated(T::HISTORY),
                            score,
                            ch,
                            parent,
                        },
                    );
                }
            }
        }
        next.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
        next.truncate(width);
        beams.push(next);
    }

    let Some(last) = beams.last() else {
        return Vec::new();
    };
    let mut finished: Vec<(usize, f32)> = last
        .iter()
        .enumerate()
        .map(|(slot, beam)| (slot, beam.score + transition.finish(beam.history)))
        .collect();
    finished.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    finished
        .into_iter()
        .map(|(slot, score)| Hypothesis {
            chars: reconstruct(&beams, slot),
            score,
            path,
        })
        .collect()
}

/// Keep only the best way of reaching each beam state.
fn relax(next: &mut Vec<Beam>, index: &mut HashMap<History, usize>, beam: Beam) {
    match index.entry(beam.history) {
        Entry::Occupied(slot) => {
            let incumbent = &mut next[*slot.get()];
            if beam.score > incumbent.score {
                *incumbent = beam;
            }
        }
        Entry::Vacant(slot) => {
            slot.insert(next.len());
            next.push(beam);
        }
    }
}

/// Follow the backpointers from a finished beam to the start of the sequence.
fn reconstruct(beams: &[Vec<Beam>], slot: usize) -> Vec<CharId> {
    let mut chars = Vec::with_capacity(beams.len());
    let mut current = slot;
    for level in beams.iter().rev() {
        let beam = level[current];
        chars.push(beam.ch);
        current = beam.parent;
    }
    chars.reverse();
    chars
}
