//! What the neural model tells the decoder, and the two files that carry it.
//!
//! The model runs under Python and the search runs here, so the two halves meet
//! as files rather than as a function call. A [`LatticeRecord`] goes out: for one
//! evaluation record, every reading of the keystrokes, and for every position of
//! every reading, the letters typed there and the characters that position
//! admits. A [`ScoreRecord`] comes back: one log probability per candidate, in
//! exactly the order the lattice listed them.
//!
//! That order is the entire contract. Nothing in the score file names a
//! character, which is what keeps it to a size worth moving but means a score
//! file can only ever be read against the lattice it answers. [`Scored::attach`]
//! therefore checks every length before it will score anything: a lattice
//! regenerated with different search options has different path counts, and
//! silently scoring 你 with 泥's log probability is exactly the kind of failure
//! that shows up as a plausible-looking accuracy number.
//!
//! The lattice is cut down to what the model can actually answer. The character
//! lexicon holds 41,923 characters and a MacBERT-based model has an output row
//! for 7,322 of them, so listing every homophone of every syllable spends four
//! fifths of the file on characters no model will ever score. The [`Emittable`]
//! set is therefore part of the contract: it decides which candidates the
//! lattice asks about, the model answers all of them, and every other candidate
//! scores at a fixed floor.
//!
//! The floor is finite on purpose. An infinity would make an emission weight of
//! zero produce a NaN instead of the transition-only baseline, and reproducing
//! the baseline through the fused code path is the only thing that proves the
//! fused code path is not doing something else.

use crate::candidates::Candidates;
use crate::score::Emission;
use ime_pinyin::{CharId, Lexicon};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Why a score file could not be attached to a lattice.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum EmissionError {
    /// The score file answered a different number of readings than the lattice
    /// asked about.
    #[error("record {record}: scored {scored} readings, the lattice has {expected}")]
    PathCount {
        /// Which evaluation record.
        record: usize,
        /// Readings the score file carried.
        scored: usize,
        /// Readings the lattice holds.
        expected: usize,
    },
    /// A reading was scored at a different number of positions than it has.
    #[error(
        "record {record}, reading {path}: scored {scored} positions, the reading has {expected}"
    )]
    PositionCount {
        /// Which evaluation record.
        record: usize,
        /// Which reading of the keystrokes.
        path: usize,
        /// Positions the score file carried.
        scored: usize,
        /// Positions the reading holds.
        expected: usize,
    },
    /// A position was scored over a different number of candidates than the
    /// model was asked about there.
    #[error(
        "record {record}, reading {path}, position {position}: scored {scored} candidates, the model was asked about {expected}"
    )]
    CandidateCount {
        /// Which evaluation record.
        record: usize,
        /// Which reading of the keystrokes.
        path: usize,
        /// Which position within the reading.
        position: usize,
        /// Candidates the score file carried.
        scored: usize,
        /// Candidates the lattice asked about.
        expected: usize,
    },
    /// The emittable set named a character the lexicon does not hold.
    #[error("the model claims to emit {character:?}, which is not in the character lexicon")]
    UnknownCharacter {
        /// The offending character.
        character: char,
    },
    /// The emittable set was empty, which would leave the model with nothing to
    /// say anywhere.
    #[error("the emittable set is empty")]
    NoEmittableCharacters,
}

/// The characters a model has an output row for.
///
/// Its own type rather than a set built at each call site, because it is part of
/// the file contract: the same set decides what the lattice asks about and what
/// the fused decode expects to have been answered, and if the two ever differ
/// the score file lines up against the wrong characters.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Emittable {
    members: Vec<CharId>,
}

impl Emittable {
    /// Read an emittable set: one character per line, as `mlime train emittable`
    /// writes it.
    ///
    /// # Errors
    ///
    /// If a line is not a single character the lexicon holds, or the set is
    /// empty.
    pub fn parse(source: &str, lexicon: &Lexicon) -> Result<Self, EmissionError> {
        let mut members = Vec::new();
        for line in source.lines() {
            let mut characters = line.trim().chars();
            let Some(character) = characters.next() else {
                continue;
            };
            if characters.next().is_some() {
                return Err(EmissionError::UnknownCharacter { character });
            }
            let id = lexicon
                .id_of(character)
                .ok_or(EmissionError::UnknownCharacter { character })?;
            members.push(id);
        }
        if members.is_empty() {
            return Err(EmissionError::NoEmittableCharacters);
        }
        members.sort_unstable();
        members.dedup();
        Ok(Self { members })
    }

    /// How many characters the model can score.
    #[must_use]
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Whether the set is empty. Never true for a parsed set.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// Whether the model has a row for *id*.
    #[must_use]
    pub fn contains(&self, id: CharId) -> bool {
        self.members.binary_search(&id).is_ok()
    }

    /// The members of *allowed* the model can score, in the order they were
    /// given.
    ///
    /// This function defines the score file's layout, which is why the command
    /// that writes the lattice and the command that reads the scores back both
    /// call it instead of filtering for themselves.
    #[must_use]
    pub fn restrict(&self, allowed: &[CharId]) -> Vec<CharId> {
        allowed
            .iter()
            .copied()
            .filter(|id| self.contains(*id))
            .collect()
    }
}

/// One reading of a typed string, as the model is asked about it.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LatticePath {
    /// The letters typed at each position, one entry per position. A position's
    /// span is a prefix of some syllable, which is what makes it an entry of the
    /// model's own typed-span table.
    pub spans: Vec<String>,
    /// The characters the model is asked about at each position: one string per
    /// position, one `char` per candidate, in the order the score file answers
    /// them.
    ///
    /// Concatenated rather than listed because every candidate is a single
    /// character and a JSON array of one-character strings spends four bytes of
    /// punctuation on every three of content. Empty where a position admits
    /// nothing the model can emit, which happens and is not an error: the decode
    /// falls back to the transition model there.
    pub candidates: Vec<String>,
}

/// Everything the model needs to score one evaluation record.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LatticeRecord {
    /// The record's zero-based line number in the evaluation set, which is what
    /// pairs a score line with the record it answers.
    pub record: usize,
    /// The keystrokes.
    pub pinyin: String,
    /// What was already on screen, if anything.
    #[serde(default)]
    pub context: Option<String>,
    /// The readings, in the order [`Candidates::paths`] holds them.
    pub paths: Vec<LatticePath>,
}

/// The model's answer for one evaluation record.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScoreRecord {
    /// The record this answers, matching [`LatticeRecord::record`].
    pub record: usize,
    /// Log probabilities indexed `[path][position][candidate]`, aligned with the
    /// lattice record of the same number.
    pub paths: Vec<Vec<Vec<f32>>>,
}

/// A model's log probabilities, bound to the candidate sets they were computed
/// over.
///
/// Borrowing the [`Candidates`] rather than copying the character ids out is
/// what makes the pairing unforgeable: a `Scored` cannot outlive the lattice it
/// was checked against, so it cannot be handed to a decode of anything else.
#[derive(Clone, Debug)]
pub struct Scored<'a> {
    asked: Vec<Vec<Vec<CharId>>>,
    scores: Vec<Vec<Vec<f32>>>,
    floor: f32,
    candidates: &'a Candidates,
}

impl<'a> Scored<'a> {
    /// Bind *scores* to *candidates*, checking that they answer the questions the
    /// lattice asked.
    ///
    /// # Errors
    ///
    /// If the two disagree about how many readings the record has, how long a
    /// reading is, or how many candidates the model was asked about at a
    /// position.
    pub fn attach(
        record: usize,
        candidates: &'a Candidates,
        emittable: &Emittable,
        scores: Vec<Vec<Vec<f32>>>,
        floor: f32,
    ) -> Result<Self, EmissionError> {
        if scores.len() != candidates.len() {
            return Err(EmissionError::PathCount {
                record,
                scored: scores.len(),
                expected: candidates.len(),
            });
        }
        let mut asked = Vec::with_capacity(candidates.len());
        for (path, (scored, reading)) in scores.iter().zip(candidates.paths()).enumerate() {
            if scored.len() != reading.len() {
                return Err(EmissionError::PositionCount {
                    record,
                    path,
                    scored: scored.len(),
                    expected: reading.len(),
                });
            }
            let mut positions = Vec::with_capacity(reading.len());
            for (position, (values, allowed)) in scored.iter().zip(reading.positions()).enumerate()
            {
                let restricted = emittable.restrict(allowed);
                if values.len() != restricted.len() {
                    return Err(EmissionError::CandidateCount {
                        record,
                        path,
                        position,
                        scored: values.len(),
                        expected: restricted.len(),
                    });
                }
                positions.push(restricted);
            }
            asked.push(positions);
        }
        Ok(Self {
            asked,
            scores,
            floor,
            candidates,
        })
    }

    /// The lattice these scores answer.
    #[must_use]
    pub const fn candidates(&self) -> &Candidates {
        self.candidates
    }
}

impl Emission for Scored<'_> {
    fn score(&self, path: usize, position: usize, candidate: CharId) -> f32 {
        match self.asked[path][position].binary_search(&candidate) {
            Ok(slot) => self.scores[path][position][slot],
            Err(_) => self.floor,
        }
    }
}

/// An emission model scaled by a fusion weight.
///
/// The neural log probabilities and the n-gram's are not on the same scale --
/// one is a distribution over a homophone set, the other over the whole
/// vocabulary -- so the fused score is `weight * emission + transition` and the
/// weight is tuned. At zero the wrapper is the transition model alone, which is
/// how the baseline is reproduced through the same code path as the ablation.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Weighted<E> {
    /// The emission model being scaled.
    pub inner: E,
    /// How much a log probability from *inner* counts against a transition score.
    pub weight: f32,
}

impl<E: Emission> Emission for Weighted<E> {
    fn score(&self, path: usize, position: usize, candidate: CharId) -> f32 {
        self.weight * self.inner.score(path, position, candidate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::Uniform;
    use ime_pinyin::{SegmentLattice, SegmentOptions, SyllableTable};

    const FLOOR: f32 = -30.0;

    fn fixture(input: &str) -> (Lexicon, Candidates) {
        let table = SyllableTable::load();
        let lexicon = Lexicon::load(&table).expect("generated tables must agree");
        let options = SegmentOptions {
            allow_incomplete_tail: false,
            ..SegmentOptions::default()
        };
        let lattice = SegmentLattice::build(input, &table, &options).expect("input reads");
        let batch =
            Candidates::build(&lattice.k_best(&options), &lexicon).expect("masks are non-empty");
        (lexicon, batch)
    }

    /// An emittable set holding the first two candidates of the first position
    /// and nothing else, so the restriction is visible rather than incidental.
    fn narrow(lexicon: &Lexicon, batch: &Candidates) -> Emittable {
        let mut source = String::new();
        for id in batch.paths()[0].positions()[0].iter().take(2) {
            source.push(lexicon.character(*id));
            source.push('\n');
        }
        Emittable::parse(&source, lexicon).expect("the characters come from the lexicon")
    }

    fn asked(batch: &Candidates, emittable: &Emittable, value: f32) -> Vec<Vec<Vec<f32>>> {
        batch
            .paths()
            .iter()
            .map(|reading| {
                reading
                    .positions()
                    .iter()
                    .map(|allowed| vec![value; emittable.restrict(allowed).len()])
                    .collect()
            })
            .collect()
    }

    #[test]
    fn scores_are_read_off_in_candidate_order() {
        let (lexicon, batch) = fixture("nihao");
        let emittable = narrow(&lexicon, &batch);
        let mut scores = asked(&batch, &emittable, -9.0);
        scores[0][0][1] = -0.5;
        let second = emittable.restrict(&batch.paths()[0].positions()[0])[1];
        let emission = Scored::attach(0, &batch, &emittable, scores, FLOOR).expect("shapes agree");
        assert!((emission.score(0, 0, second) - -0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn a_candidate_the_model_cannot_emit_takes_the_floor() {
        let (lexicon, batch) = fixture("nihao");
        let emittable = narrow(&lexicon, &batch);
        let scores = asked(&batch, &emittable, -1.0);
        let emission = Scored::attach(0, &batch, &emittable, scores, FLOOR).expect("shapes agree");
        let outside = batch.paths()[0].positions()[0]
            .iter()
            .copied()
            .find(|id| !emittable.contains(*id))
            .expect("the narrow set leaves candidates out");
        assert!((emission.score(0, 0, outside) - FLOOR).abs() < f32::EPSILON);
    }

    #[test]
    fn a_score_file_for_a_different_lattice_is_refused() {
        let (lexicon, batch) = fixture("nihao");
        let emittable = narrow(&lexicon, &batch);
        let mut scores = asked(&batch, &emittable, 0.0);
        scores.pop();
        let error =
            Scored::attach(7, &batch, &emittable, scores, FLOOR).expect_err("path counts differ");
        assert!(matches!(error, EmissionError::PathCount { record: 7, .. }));
    }

    #[test]
    fn a_position_scored_over_the_wrong_candidate_set_is_refused() {
        let (lexicon, batch) = fixture("nihao");
        let emittable = narrow(&lexicon, &batch);
        let mut scores = asked(&batch, &emittable, 0.0);
        scores[0][0].push(0.0);
        let error = Scored::attach(0, &batch, &emittable, scores, FLOOR)
            .expect_err("candidate counts differ");
        assert!(matches!(
            error,
            EmissionError::CandidateCount {
                record: 0,
                path: 0,
                position: 0,
                ..
            }
        ));
    }

    #[test]
    fn a_weight_of_zero_is_the_uniform_emission() {
        let (lexicon, batch) = fixture("nihao");
        let emittable = narrow(&lexicon, &batch);
        let scores = asked(&batch, &emittable, -3.5);
        let emission = Scored::attach(0, &batch, &emittable, scores, FLOOR).expect("shapes agree");
        let weighted = Weighted {
            inner: emission,
            weight: 0.0,
        };
        let first = batch.paths()[0].positions()[0][0];
        assert!((weighted.score(0, 0, first) - Uniform.score(0, 0, first)).abs() < f32::EPSILON);
    }

    #[test]
    fn an_emittable_character_outside_the_lexicon_is_refused() {
        let (lexicon, _) = fixture("nihao");
        let error = Emittable::parse("A\n", &lexicon).expect_err("A is not a Han character");
        assert_eq!(error, EmissionError::UnknownCharacter { character: 'A' });
    }

    #[test]
    fn a_lattice_record_round_trips_through_json() {
        let record = LatticeRecord {
            record: 3,
            pinyin: "nihao".to_owned(),
            context: Some("今天".to_owned()),
            paths: vec![LatticePath {
                spans: vec!["ni".to_owned(), "hao".to_owned()],
                candidates: vec!["你泥".to_owned(), "好号".to_owned()],
            }],
        };
        let line = serde_json::to_string(&record).expect("a record serialises");
        assert_eq!(
            serde_json::from_str::<LatticeRecord>(&line).expect("and parses back"),
            record
        );
    }
}
