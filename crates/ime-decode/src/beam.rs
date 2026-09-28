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

/// One record decoded in lockstep with a batch's other records.
///
/// Everything [`decode_many`] needs of one typed string that is not shared
/// with the batch's other records: its own lattice, its own emission model
/// over that lattice's paths, and its own context.
pub struct Record<'a, E> {
    /// The readings of the record's keystrokes.
    pub candidates: &'a Candidates,
    /// The emission scores, indexed by `candidates`' own paths.
    pub emission: E,
    /// The text on screen before the keystrokes, for transition models that
    /// read context.
    pub context: Option<&'a str>,
}

/// One reading of one record, mid-search.
///
/// A worker's current position is how many beam levels it has filled, so every
/// live worker of the batch stands at the same position index in the same
/// round, and one [`Transition::advance`] call covers all of them.
struct Worker<'a, S> {
    /// Which record of the batch the reading belongs to.
    record: usize,
    /// The reading's index in the record's [`Candidates::paths`].
    path: usize,
    /// The reading being decoded.
    reading: &'a CandidatePath,
    /// The state position zero advances from: `start` over the record's
    /// context.
    start: S,
    /// The candidates surviving at each finished position — the backpointer
    /// chain `reconstruct` walks. Keeping them without their states releases
    /// a level's states the moment the next level lands, so a dead beam's
    /// resident row is freed then rather than at `finish`.
    history: Vec<Vec<Candidate>>,
    /// The beams surviving at the current position; pushing a level's
    /// `advance` replaces it wholesale.
    latest: Vec<Beam<S>>,
}

impl<S> Worker<'_, S> {
    /// Whether the worker still has a position to decode.
    fn pending(&self) -> bool {
        self.history.len() < self.reading.len()
    }

    /// Score this worker's candidates at its current position down to the
    /// `width` survivors the model will be advanced into.
    ///
    /// Each position scores every candidate against every surviving beam's
    /// state and keeps the best way of reaching each distinct history; the
    /// caller gathers the survivors' `(state, ch)` steps into the batch's one
    /// `advance` call, so a model whose step is expensive pays for the beams
    /// the batch keeps and not for the candidates it discards.
    fn survivors<E, T>(
        &self,
        emission: &E,
        transition: &T,
        width: usize,
        index: &mut HashMap<History, usize>,
    ) -> Vec<Candidate>
    where
        E: Emission,
        T: Transition<State = S>,
    {
        let position = self.history.len();
        let allowed = &self.reading.positions()[position];
        let mut next: Vec<Candidate> = Vec::with_capacity(allowed.len());
        index.clear();
        if position == 0 {
            for &ch in allowed {
                let score = emission.score(self.path, 0, ch) + transition.score(&self.start, ch);
                relax(
                    &mut next,
                    index,
                    Candidate {
                        history: History::START.extended(ch).truncated(T::HISTORY),
                        score,
                        ch,
                        parent: 0,
                    },
                );
            }
        } else {
            for (parent, beam) in self.latest.iter().enumerate() {
                for &ch in allowed {
                    let score = beam.candidate.score
                        + emission.score(self.path, position, ch)
                        + transition.score(&beam.state, ch);
                    relax(
                        &mut next,
                        index,
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
        next
    }

    /// Follow the backpointers from a finished beam to the start of the
    /// sequence.
    fn reconstruct(&self, slot: usize) -> Vec<CharId> {
        let mut chars = Vec::with_capacity(self.history.len());
        let mut current = slot;
        for level in self.history.iter().rev() {
            let candidate = level[current];
            chars.push(candidate.ch);
            current = candidate.parent;
        }
        chars.reverse();
        chars
    }
}

/// Decode every reading of every record in *records* in lockstep and merge
/// each record's results.
///
/// Each record's readings each become one worker, and the per-position loop
/// runs over all of them at once: at every position every live worker's
/// surviving candidates are scored, gathered into one [`Transition::advance`]
/// call over `records x paths x beams` rows, and scattered back. A worker
/// ends when its reading runs out; a record is done when its last worker is.
/// The call a model pays per step is the batch's, so a reader on a GPU prices
/// its weights read once per step rather than once per record.
///
/// Within one record, hypotheses from different readings compete on one scale
/// -- sequence score minus the weighted segmentation cost -- and identical
/// sentences reached by different readings collapse to the best-scoring one.
/// The answer has one entry per record, in the records' order.
///
/// # Errors
///
/// If a record's lattice is empty, which [`Candidates::build`] already rules
/// out.
///
/// # Panics
///
/// If the transition model does not return one state per step, which is a bug
/// in the model, not an error in the input.
#[expect(
    clippy::panic_in_result_fn,
    reason = "the only panic is the contract on Transition::advance, a bug in the model rather than bad input"
)]
pub fn decode_many<E, T>(
    records: &[Record<'_, E>],
    transition: &T,
    options: &BeamOptions,
) -> Result<Vec<Vec<Hypothesis>>, crate::DecodeError>
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
    for record in records {
        if record.candidates.is_empty() {
            return Err(crate::DecodeError::NoSegmentations);
        }
    }
    let width = options.beam_width.get();
    let mut workers: Vec<Worker<'_, T::State>> = Vec::new();
    for (record, request) in records.iter().enumerate() {
        for (path, reading) in request.candidates.paths().iter().enumerate() {
            workers.push(Worker {
                record,
                path,
                reading,
                start: transition.start(request.context),
                history: Vec::with_capacity(reading.len()),
                latest: Vec::new(),
            });
        }
    }
    let mut index: HashMap<History, usize> = HashMap::new();
    while workers.iter().any(Worker::pending) {
        // Score: each live worker relaxes its candidates to the best way of
        // reaching each history and truncates to the beam width, the same
        // shape a single-record search has; then every survivor's step goes
        // on the batch's one list, each worker's as a contiguous run in
        // worker order -- the guarantee `Transition::advance` is built on.
        let mut steps: Vec<(&T::State, CharId)> = Vec::new();
        let mut chosen: Vec<Vec<Candidate>> = Vec::with_capacity(workers.len());
        for worker in workers.iter().filter(|worker| worker.pending()) {
            let next = worker.survivors(
                &records[worker.record].emission,
                transition,
                width,
                &mut index,
            );
            for candidate in &next {
                let state = if worker.history.is_empty() {
                    &worker.start
                } else {
                    &worker.latest[candidate.parent].state
                };
                steps.push((state, candidate.ch));
            }
            chosen.push(next);
        }
        let states = transition.advance(&steps);
        assert_eq!(
            states.len(),
            steps.len(),
            "a transition model must return one state per step"
        );
        let mut states = states.into_iter();
        for (worker, next) in workers
            .iter_mut()
            .filter(|worker| worker.pending())
            .zip(chosen)
        {
            let beams: Vec<Beam<T::State>> = next
                .into_iter()
                .zip(&mut states)
                .map(|(candidate, state)| Beam { candidate, state })
                .collect();
            worker
                .history
                .push(beams.iter().map(|beam| beam.candidate).collect());
            worker.latest = beams;
        }
        debug_assert!(states.next().is_none(), "every advanced state lands");
    }

    Ok(finish(&workers, records.len(), transition, options))
}

/// Fold finished workers into one ranked hypothesis list per record.
fn finish<T: Transition>(
    workers: &[Worker<'_, T::State>],
    records: usize,
    transition: &T,
    options: &BeamOptions,
) -> Vec<Vec<Hypothesis>> {
    let mut merged: Vec<Vec<Hypothesis>> = (0..records).map(|_| Vec::new()).collect();
    for worker in workers {
        if worker.latest.is_empty() {
            continue;
        }
        let last = &worker.latest;
        let penalty = options.segmentation_weight * worker.reading.cost();
        let mut finished: Vec<(usize, f32)> = last
            .iter()
            .enumerate()
            .map(|(slot, beam)| (slot, beam.candidate.score + transition.finish(&beam.state)))
            .collect();
        finished.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        merged[worker.record].extend(finished.into_iter().map(|(slot, score)| Hypothesis {
            chars: worker.reconstruct(slot),
            score: score - penalty,
            path: worker.path,
        }));
    }
    merged
        .into_iter()
        .map(|mut hypotheses| {
            hypotheses.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
            let mut seen: HashSet<&[CharId]> = HashSet::new();
            let mut keep = Vec::with_capacity(options.top_k.get());
            for (index, hypothesis) in hypotheses.iter().enumerate() {
                if keep.len() == options.top_k.get() {
                    break;
                }
                if seen.insert(hypothesis.chars.as_slice()) {
                    keep.push(index);
                }
            }
            keep.into_iter()
                .map(|index| hypotheses[index].clone())
                .collect()
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
