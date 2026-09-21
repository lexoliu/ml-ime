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

/// A way of reaching a beam state, before the model has been advanced into it:
/// the last few characters, the score of the best way there, and where it came
/// from.
#[derive(Copy, Clone, Debug)]
struct Candidate {
    history: History,
    score: f32,
    ch: CharId,
    /// Index into the previous position's surviving beams. Meaningless at
    /// position zero.
    parent: usize,
}

/// A surviving beam: a [`Candidate`] together with the transition model's state
/// after its last character.
#[derive(Clone, Debug)]
struct Beam<S> {
    candidate: Candidate,
    state: S,
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
    context: Option<&str>,
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
        for mut hypothesis in decode_path(path, reading, emission, transition, context, options) {
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
///
/// Each position scores every candidate against every surviving beam's state,
/// keeps the best way of reaching each distinct history, truncates to the beam
/// width, and only then advances the model into the survivors -- one batched
/// call, so a model whose step is expensive pays for the beams it keeps and not
/// for the candidates it discards.
fn decode_path<E, T>(
    path: usize,
    reading: &CandidatePath,
    emission: &E,
    transition: &T,
    context: Option<&str>,
    options: &BeamOptions,
) -> Vec<Hypothesis>
where
    E: Emission,
    T: Transition,
{
    let width = options.beam_width.get();
    let start = transition.start(context);
    let mut beams: Vec<Vec<Beam<T::State>>> = Vec::with_capacity(reading.len());
    let mut index: HashMap<History, usize> = HashMap::new();

    for (position, allowed) in reading.positions().iter().enumerate() {
        let mut next: Vec<Candidate> = Vec::with_capacity(allowed.len());
        index.clear();
        if position == 0 {
            for &ch in allowed {
                let score = emission.score(path, 0, ch) + transition.score(&start, ch);
                relax(
                    &mut next,
                    &mut index,
                    Candidate {
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
                    let score = beam.candidate.score
                        + emission.score(path, position, ch)
                        + transition.score(&beam.state, ch);
                    relax(
                        &mut next,
                        &mut index,
                        Candidate {
                            history: beam.candidate.history.extended(ch).truncated(T::HISTORY),
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
        let steps: Vec<(&T::State, CharId)> = next
            .iter()
            .map(|candidate| {
                let state = if position == 0 {
                    &start
                } else {
                    &beams[position - 1][candidate.parent].state
                };
                (state, candidate.ch)
            })
            .collect();
        let states = transition.advance(&steps);
        assert_eq!(
            states.len(),
            steps.len(),
            "a transition model must return one state per step"
        );
        beams.push(
            next.into_iter()
                .zip(states)
                .map(|(candidate, state)| Beam { candidate, state })
                .collect(),
        );
    }

    let Some(last) = beams.last() else {
        return Vec::new();
    };
    let mut finished: Vec<(usize, f32)> = last
        .iter()
        .enumerate()
        .map(|(slot, beam)| (slot, beam.candidate.score + transition.finish(&beam.state)))
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
fn relax(next: &mut Vec<Candidate>, index: &mut HashMap<History, usize>, candidate: Candidate) {
    match index.entry(candidate.history) {
        Entry::Occupied(slot) => {
            let incumbent = &mut next[*slot.get()];
            if candidate.score > incumbent.score {
                *incumbent = candidate;
            }
        }
        Entry::Vacant(slot) => {
            slot.insert(next.len());
            next.push(candidate);
        }
    }
}

/// Follow the backpointers from a finished beam to the start of the sequence.
fn reconstruct<S>(beams: &[Vec<Beam<S>>], slot: usize) -> Vec<CharId> {
    let mut chars = Vec::with_capacity(beams.len());
    let mut current = slot;
    for level in beams.iter().rev() {
        let candidate = level[current].candidate;
        chars.push(candidate.ch);
        current = candidate.parent;
    }
    chars.reverse();
    chars
}
