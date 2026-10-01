//! Beam Viterbi over the candidate lattice.

use crate::candidates::{CandidatePath, Candidates};
use crate::score::{Asked, Emission, History, MAX_HISTORY, Transition};
use hashbrown::hash_map::Entry;
use hashbrown::{HashMap, HashSet};
use ime_pinyin::{CharId, Lexicon};
use std::num::NonZeroUsize;
use std::time::Instant;

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
    /// How close a position's runner-up character must trail the winner, in
    /// nats of cumulative score, for the position to be guarded: the beam then
    /// reserves a slot per covered character so the runner-up's best lineage
    /// stays alive through the remaining positions instead of being crowded
    /// out by the winner's tails. The reserved slots come out of
    /// `beam_width`, never on top of it. A non-positive value disables
    /// guarding.
    pub diversity_gap: f32,
    /// How many distinct characters a guarded position keeps lineages for.
    pub diversity_chars: usize,
    /// Weight on each candidate's correction prior, the `-log P(span |
    /// syllable)` a character reachable only through a typo correction pays.
    /// Zero prices a widened character like a span's own reading; one charges
    /// the noise model's own rate.
    pub typo_weight: f32,
}

impl Default for BeamOptions {
    fn default() -> Self {
        Self {
            beam_width: NonZeroUsize::new(16).expect("16 is not zero"),
            top_k: NonZeroUsize::new(8).expect("8 is not zero"),
            segmentation_weight: 1.0,
            diversity_gap: 1.5,
            diversity_chars: 3,
            typo_weight: 1.0,
        }
    }
}

/// Where `decode_many`'s host time goes, accumulated across calls — the
/// `advance` calls themselves are timed by the transition's own counters;
/// this is the CPU work around them.
#[derive(Clone, Debug, Default)]
pub struct Breakdown {
    /// Nanos inside candidate expansion: the emission and transition score
    /// calls, `relax`, and the history-merge index.
    pub expand_ns: u64,
    /// Nanos inside survivor selection: the guard passes, the partition,
    /// and the head sort.
    pub select_ns: u64,
    /// Nanos finishing: `Transition::finish`, reconstruction, `keep_diverse`.
    pub finish_ns: u64,
    /// (beam, character) pairs scored during expansion — the candidate
    /// count the scores paid.
    pub expanded: u64,
}

/// One decoded sentence.
#[derive(Clone, Debug)]
pub struct Hypothesis {
    chars: Vec<CharId>,
    score: f32,
    /// The running score at each position: `steps[k - 1]` is the score of the
    /// hypothesis's first `k` characters — emissions and transitions up to that
    /// position, before the end-of-sequence term and the segmentation penalty.
    steps: Vec<f32>,
    path: usize,
}

impl Hypothesis {
    /// The characters, left to right.
    #[must_use]
    pub fn chars(&self) -> &[CharId] {
        &self.chars
    }

    /// The running score, one entry per position, same order as [`chars`].
    ///
    /// [`chars`]: Self::chars
    #[must_use]
    pub fn steps(&self) -> &[f32] {
        &self.steps
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

/// How many positions one worker can carry guarded readings for. Guards are
/// declared earliest first, so a worker whose sentence is ambiguous at many
/// positions keeps the runner-ups where the reader's priors are weakest.
const MAX_GUARDED: usize = 8;

/// The characters a candidate's lineage took at the positions the worker has
/// guarded, in guard order.
///
/// Marks join the history merge key: two candidates whose truncated histories
/// agree still merge only when their marks agree, so a runner-up's lineage at
/// a guarded position cannot be merged away under a short-memory transition
/// -- under a unigram history, `她x` and `他x` would otherwise collapse to the
/// better-scoring one and the loser would die at the very next position.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
struct Marks {
    chars: [Option<CharId>; MAX_GUARDED],
    len: u8,
}

impl Marks {
    /// The marks of a survivor just emitted at a newly guarded position:
    /// every survivor's own character is its lineage there.
    fn pushed(self, ch: CharId) -> Self {
        let mut marks = self;
        marks.chars[usize::from(marks.len)] = Some(ch);
        marks.len += 1;
        marks
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
    /// What the lineage read at each guarded position.
    marks: Marks,
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
    /// The positions so far whose runner-up characters stay guarded, each
    /// listed as its covered characters in score order. Entry `i` matches
    /// `marks[i]` on every candidate produced since it was declared.
    guarded: Vec<Vec<CharId>>,
}

impl<S> Worker<'_, S> {
    /// Whether the worker still has a position to decode.
    fn pending(&self) -> bool {
        self.history.len() < self.reading.len()
    }

    /// Score this worker's candidates at its current position down to the
    /// `beam_width` survivors the model will be advanced into.
    ///
    /// Each position scores every candidate against every surviving beam's
    /// state and keeps the best way of reaching each distinct history; the
    /// caller gathers the survivors' `(state, ch)` steps into the batch's one
    /// `advance` call, so a model whose step is expensive pays for the beams
    /// the batch keeps and not for the candidates it discards.
    ///
    /// Survivor selection reserves before it fills: the best candidate of
    /// each covered character at every guarded position keeps its slot, and
    /// the position itself is guarded when its runner-up character trails
    /// the winner by no more than `diversity_gap` -- that is where a
    /// score-ordered beam silently collapses the sentence onto one reading.
    /// Reserved slots come out of `beam_width`; when more guards would fit
    /// than the beam has room for, the earliest positions keep theirs.
    fn survivors<E, T>(
        &mut self,
        emission: &E,
        transition: &T,
        options: &BeamOptions,
        index: &mut HashMap<(History, Marks), usize>,
        breakdown: &mut Breakdown,
    ) -> Vec<Candidate>
    where
        E: Emission,
        T: Transition<State = S>,
    {
        let position = self.history.len();
        let allowed = &self.reading.positions()[position];
        let typo_weight = options.typo_weight;
        let mut next: Vec<Candidate> = Vec::with_capacity(allowed.len() * self.latest.len().max(1));
        index.clear();
        let tick = Instant::now();
        // The emission term reads no beam state: score each admitted
        // character once. The transition term is the whole candidate list in
        // one `score_many` call per beam — a model with sorted-row lookups
        // merges over `allowed` instead of probing per pair.
        let emitted: Vec<f32> = allowed
            .iter()
            .map(|&ch| emission.score(self.path, position, ch))
            .collect();
        let mut scored = vec![0.0f32; allowed.len()];
        if position == 0 {
            transition.score_many(&self.start, 1.0, allowed, &mut scored);
            for (slot, ((&ch, &emit), &trans)) in
                allowed.iter().zip(&emitted).zip(&scored).enumerate()
            {
                relax(
                    &mut next,
                    index,
                    Candidate {
                        history: History::START.extended(ch).truncated(T::HISTORY),
                        score: emit + trans - priced(typo_weight, self.reading.prior(0, slot)),
                        ch,
                        parent: 0,
                        marks: Marks::default(),
                    },
                );
            }
        } else {
            for (parent, beam) in self.latest.iter().enumerate() {
                scored.fill(0.0);
                transition.score_many(&beam.state, 1.0, allowed, &mut scored);
                for (slot, ((&ch, &emit), &trans)) in
                    allowed.iter().zip(&emitted).zip(&scored).enumerate()
                {
                    relax(
                        &mut next,
                        index,
                        Candidate {
                            history: beam.candidate.history.extended(ch).truncated(T::HISTORY),
                            score: beam.candidate.score + emit + trans
                                - priced(typo_weight, self.reading.prior(position, slot)),
                            ch,
                            parent,
                            marks: beam.candidate.marks,
                        },
                    );
                }
            }
        }
        breakdown.expand_ns += u64::try_from(tick.elapsed().as_nanos()).unwrap_or(u64::MAX);
        breakdown.expanded +=
            u64::try_from(allowed.len() * if position == 0 { 1 } else { self.latest.len() })
                .unwrap_or(u64::MAX);
        let tick = Instant::now();
        let (keep, guarded_here) = self.reserve(&next, options);
        let mut chosen = Vec::with_capacity(keep.len());
        for index in keep {
            let mut candidate = next[index];
            if guarded_here {
                candidate.marks = candidate.marks.pushed(candidate.ch);
            }
            chosen.push(candidate);
        }
        breakdown.select_ns += u64::try_from(tick.elapsed().as_nanos()).unwrap_or(u64::MAX);
        chosen
    }

    /// Which entries of *next* survive: each guarded position's covered
    /// characters first, then selection order (score, then relax order) to
    /// the beam width. The order's few taken entries reach the head through
    /// a partition, not a full sort of every relaxed pair. Returns the
    /// chosen indices and whether this position declared its own guard, so
    /// the caller can mark every survivor's lineage.
    #[expect(
        clippy::too_many_lines,
        reason = "one survivor selection is a straight-line sequence: guard picks, a declared guard, then the fill"
    )]
    fn reserve(&mut self, next: &[Candidate], options: &BeamOptions) -> (Vec<usize>, bool) {
        let width = options.beam_width.get();
        let gap = options.diversity_gap;
        let covered = options.diversity_chars;
        // The selection order: score, then relax order, so every position's
        // sorted prefix is uniquely defined. The guard picks and the fill
        // below read exactly what a full sort in it would list; the tail
        // never pays for one — a partition moves the few the beam takes to
        // the head, and only the head is sorted.
        let before = |a: usize, b: usize| {
            next[b]
                .score
                .total_cmp(&next[a].score)
                .then(a.cmp(&b))
                .is_lt()
        };
        // The best index per guard slot's character — but only when this
        // worker already carries guards; an unguarded worker never reads
        // `first_of`, so the pass is skipped then rather than built and
        // ignored. "Best" is the first in selection order, so the guard
        // picks below match what a scan over the sorted `next` found.
        let first_of: HashMap<(usize, CharId), usize> = if self.guarded.is_empty() {
            HashMap::new()
        } else {
            let mut first_of = HashMap::new();
            for (index, candidate) in next.iter().enumerate() {
                for guard in 0..usize::from(candidate.marks.len) {
                    if let Some(ch) = candidate.marks.chars[guard] {
                        match first_of.entry((guard, ch)) {
                            Entry::Vacant(slot) => {
                                slot.insert(index);
                            }
                            Entry::Occupied(mut slot) => {
                                if before(index, *slot.get()) {
                                    slot.insert(index);
                                }
                            }
                        }
                    }
                }
            }
            first_of
        };
        let mut keep: Vec<usize> = Vec::new();
        let mut kept: Vec<bool> = vec![false; next.len()];
        // Existing guards: the best lineage of each covered character keeps
        // its slot, however far it has slipped in score order.
        for (guard, chars) in self.guarded.iter().enumerate() {
            for &ch in chars {
                if let Some(&index) = first_of.get(&(guard, ch))
                    && !kept[index]
                {
                    kept[index] = true;
                    keep.push(index);
                }
            }
        }
        // This position's own guard: declared when a runner-up character is
        // close enough that the position is a coin toss the score lost. The
        // best index per character is built only when the declaration is
        // still possible — most positions never declare one.
        let mut guarded_here = false;
        if !next.is_empty()
            && gap > 0.0
            && covered > 1
            && self.guarded.len() < MAX_GUARDED
            && keep.len() + covered <= width
        {
            let mut first_char: HashMap<CharId, usize> = HashMap::new();
            for (index, candidate) in next.iter().enumerate() {
                match first_char.entry(candidate.ch) {
                    Entry::Vacant(slot) => {
                        slot.insert(index);
                    }
                    Entry::Occupied(mut slot) => {
                        if before(index, *slot.get()) {
                            slot.insert(index);
                        }
                    }
                }
            }
            // The best index per character in selection order — what the
            // first `covered` distinct characters of the sorted `next` are.
            let mut chars: Vec<usize> = first_char.values().copied().collect();
            chars
                .sort_unstable_by(|&a, &b| next[b].score.total_cmp(&next[a].score).then(a.cmp(&b)));
            let winner = next[chars[0]];
            let runner_up = chars
                .iter()
                .map(|&index| next[index])
                .find(|candidate| candidate.ch != winner.ch);
            if let Some(runner_up) = runner_up
                && winner.score - runner_up.score <= gap
            {
                self.guarded.push(
                    chars
                        .iter()
                        .take(covered)
                        .map(|&index| next[index].ch)
                        .collect(),
                );
                for &index in chars.iter().take(covered) {
                    if !kept[index] {
                        kept[index] = true;
                        keep.push(index);
                    }
                }
                guarded_here = true;
            }
        }
        for &index in keep.get(width.min(keep.len())..).unwrap_or(&[]) {
            kept[index] = false;
        }
        keep.truncate(width);
        if keep.len() < width && !next.is_empty() {
            // Fill in selection order: the kept entries inside the head make
            // the prefix the fill draws on `keep.len() + width` deep.
            let head = (keep.len() + width).min(next.len());
            let mut order: Vec<usize> = (0..next.len()).collect();
            order.select_nth_unstable_by(head - 1, |&a, &b| {
                next[b].score.total_cmp(&next[a].score).then(a.cmp(&b))
            });
            order[..head]
                .sort_unstable_by(|&a, &b| next[b].score.total_cmp(&next[a].score).then(a.cmp(&b)));
            for index in order[..head].iter().copied() {
                if keep.len() == width {
                    break;
                }
                if !kept[index] {
                    kept[index] = true;
                    keep.push(index);
                }
            }
        }
        (keep, guarded_here)
    }

    /// Follow the backpointers from a finished beam to the start of the
    /// sequence, collecting each position's running score alongside its
    /// character.
    fn reconstruct(&self, slot: usize) -> (Vec<CharId>, Vec<f32>) {
        let mut chars = Vec::with_capacity(self.history.len());
        let mut steps = Vec::with_capacity(self.history.len());
        let mut current = slot;
        for level in self.history.iter().rev() {
            let candidate = level[current];
            chars.push(candidate.ch);
            steps.push(candidate.score);
            current = candidate.parent;
        }
        chars.reverse();
        steps.reverse();
        (chars, steps)
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
pub fn decode_many<E, T>(
    records: &[Record<'_, E>],
    transition: &T,
    options: &BeamOptions,
) -> Result<Vec<Vec<Hypothesis>>, crate::DecodeError>
where
    E: Emission,
    T: Transition,
{
    decode_many_stats(records, transition, options, &mut Breakdown::default())
}

/// [`decode_many`], accumulating its host-side breakdown into *breakdown*
/// — expansion, selection, and the finish pass, so the decode's cost split
/// survives beyond the one call. `Transition::advance`'s own time is not
/// here; the transition's `Timed` wrapper carries it.
///
/// # Errors
///
/// As [`decode_many`].
///
/// # Panics
///
/// As [`decode_many`].
pub fn decode_many_stats<E, T>(
    records: &[Record<'_, E>],
    transition: &T,
    options: &BeamOptions,
    breakdown: &mut Breakdown,
) -> Result<Vec<Vec<Hypothesis>>, crate::DecodeError>
where
    E: Emission,
    T: Transition,
{
    run(records, transition, options, breakdown)
}

/// The [`decode_many_stats`] body: the lockstep expansion loop.
///
/// [`decode_many_stats`]: crate::decode_many_stats
#[expect(
    clippy::panic_in_result_fn,
    reason = "as decode_many — the panic is the contract on Transition::advance"
)]
fn run<E, T>(
    records: &[Record<'_, E>],
    transition: &T,
    options: &BeamOptions,
    breakdown: &mut Breakdown,
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
    let mut workers: Vec<Worker<'_, T::State>> = Vec::new();
    for (record, request) in records.iter().enumerate() {
        for (path, reading) in request.candidates.paths().iter().enumerate() {
            workers.push(Worker {
                record,
                path,
                reading,
                // A reading always has a first position, and its start state
                // is only ever scored there -- never finished, so no <eos>.
                start: transition.start(
                    request.context,
                    &Asked {
                        candidates: &reading.positions()[0],
                        eos: false,
                    },
                ),
                history: Vec::with_capacity(reading.len()),
                latest: Vec::new(),
                guarded: Vec::new(),
            });
        }
    }
    let mut index: HashMap<(History, Marks), usize> = HashMap::new();
    while workers.iter().any(Worker::pending) {
        // Score: each live worker relaxes its candidates to the best way of
        // reaching each history and truncates to the beam width, the same
        // shape a single-record search has; then every survivor's step goes
        // on the batch's one list, each worker's as a contiguous run in
        // worker order -- the guarantee `Transition::advance` is built on.
        let mut steps: Vec<(&T::State, CharId, Asked<'_>)> =
            Vec::with_capacity(workers.len() * options.beam_width.get());
        let mut chosen: Vec<Vec<Candidate>> = Vec::with_capacity(workers.len());
        for worker in workers.iter_mut().filter(|worker| worker.pending()) {
            let next = worker.survivors(
                &records[worker.record].emission,
                transition,
                options,
                &mut index,
                breakdown,
            );
            // The states this step produces stand one position ahead: they
            // will be scored on that position's candidates, or finished when
            // the reading ends there -- the worker's last step asks only for
            // <eos>.
            let after = worker.history.len() + 1;
            let positions = worker.reading.positions().get(after);
            let asked = Asked {
                candidates: positions.map_or(&[][..], Vec::as_slice),
                eos: positions.is_none(),
            };
            for candidate in &next {
                let state = if worker.history.is_empty() {
                    &worker.start
                } else {
                    &worker.latest[candidate.parent].state
                };
                steps.push((state, candidate.ch, asked));
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

    let tick = Instant::now();
    let out = finish(&workers, records.len(), transition, options);
    breakdown.finish_ns += u64::try_from(tick.elapsed().as_nanos()).unwrap_or(u64::MAX);
    Ok(out)
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
        merged[worker.record].extend(finished.into_iter().map(|(slot, score)| {
            let (chars, steps) = worker.reconstruct(slot);
            Hypothesis {
                chars,
                score: score - penalty,
                steps,
                path: worker.path,
            }
        }));
    }
    merged
        .into_iter()
        .map(|mut hypotheses| keep_diverse(&mut hypotheses, options.top_k.get()))
        .collect()
}

/// Pick a record's `top_k` hypotheses out of its finished, score-sorted beams.
///
/// A plain top-*k* collapses onto one reading of an ambiguous character: when
/// the candidates at one position score close together, tail variants of the
/// winner fill the list and the runner-up at another position is unreachable.
/// The list therefore covers before it fills: the best surviving hypothesis
/// that differs from the winner at exactly one position is kept for every such
/// position, so each uncertain character's runner-up stays reachable. Only the
/// best variant per position is guaranteed a slot — every other slot keeps
/// score order — and the winner itself is always first, so top-1 is the
/// unchanged Viterbi result.
fn keep_diverse(hypotheses: &mut [Hypothesis], top_k: usize) -> Vec<Hypothesis> {
    hypotheses.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
    let mut seen: HashSet<&[CharId]> = HashSet::default();
    let mut unique: Vec<usize> = Vec::new();
    for (index, hypothesis) in hypotheses.iter().enumerate() {
        if seen.insert(hypothesis.chars.as_slice()) {
            unique.push(index);
        }
    }
    let Some(&first) = unique.first() else {
        return Vec::new();
    };
    // The best hypothesis covering each position the winner is uncertain at:
    // a hypothesis that differs from it there and nowhere else. `unique` is
    // score-sorted, so the first met for a position is that position's best.
    // When more positions diverge than the list has winner slots, the earliest
    // positions win: the first characters of a sentence are where the reader's
    // priors are weakest.
    let winner = &hypotheses[first].chars;
    let mut covered: HashSet<usize> = HashSet::default();
    let mut winners: Vec<(usize, usize)> = Vec::new();
    for &index in unique.iter().skip(1) {
        let chars = &hypotheses[index].chars;
        if chars.len() != winner.len() {
            continue;
        }
        let mut divergent = None;
        for (position, (&candidate, &chosen)) in chars.iter().zip(winner).enumerate() {
            if candidate == chosen {
                continue;
            }
            if divergent.is_some() {
                // Differs at a second position too: not the reading a single
                // uncertain character produces.
                divergent = None;
                break;
            }
            divergent = Some(position);
        }
        if let Some(position) = divergent
            && covered.insert(position)
        {
            winners.push((position, index));
        }
    }
    winners.sort_unstable_by_key(|&(position, _)| position);
    winners.truncate(top_k - 1);

    let mut keep: Vec<usize> = Vec::with_capacity(top_k);
    let mut kept: HashSet<usize> = HashSet::default();
    keep.push(first);
    kept.insert(first);
    for &(_, index) in &winners {
        keep.push(index);
        kept.insert(index);
    }
    for &index in &unique {
        if keep.len() == top_k {
            break;
        }
        if kept.insert(index) {
            keep.push(index);
        }
    }
    keep.sort_unstable_by(|a, b| hypotheses[*b].score.total_cmp(&hypotheses[*a].score));
    keep.into_iter()
        .map(|index| hypotheses[index].clone())
        .collect()
}

/// A candidate prior's contribution at `typo_weight`. An unreachable
/// character's `INFINITY` stays dead whenever the weight is on and weighs
/// nothing when it is off -- `0 * inf` would price the character `NaN` and
/// poison the beam's ordering.
#[expect(
    clippy::cast_possible_truncation,
    reason = "a -log of a two-edit probability stays small"
)]
fn priced(typo_weight: f32, prior: f64) -> f32 {
    if prior.is_infinite() {
        if typo_weight > 0.0 {
            return f32::INFINITY;
        }
        return 0.0;
    }
    typo_weight * prior as f32
}

/// Keep only the best way of reaching each beam state. Candidates whose
/// marks disagree are different lineages at a guarded position and never
/// merge, so a guarded runner-up survives a history its winner's lineage
/// also reaches.
fn relax(
    next: &mut Vec<Candidate>,
    index: &mut HashMap<(History, Marks), usize>,
    candidate: Candidate,
) {
    match index.entry((candidate.history, candidate.marks)) {
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
