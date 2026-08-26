//! An n-gram character model: the baseline every neural result is measured
//! against, and the transition term in the decoder's Viterbi pass.
//!
//! It is both at once on purpose. Milestone 3 asks whether a context-conditioned
//! neural model beats a conventional IME, and the honest way to ask is to run
//! both routes through the same search over the same masks, changing only where
//! the scores come from. So this crate implements
//! [`Transition`](ime_decode::Transition) and nothing else: paired with
//! [`Uniform`](ime_decode::Uniform) it *is* the conventional IME, and paired with
//! a neural emission model it is the term that repairs what a
//! non-autoregressive decoder cannot say.

mod model;
mod table;
mod train;

pub use model::{NgramModel, ORDER, Token};
pub use train::Counter;

use thiserror::Error;

/// Why a model could not be trained, loaded or queried.
#[derive(Debug, Error)]
pub enum NgramError {
    /// The corpus held no character the lexicon knows.
    #[error("the corpus held no character in the lexicon")]
    EmptyCorpus,
    /// An order's absolute discount came out at zero or worse, which would leave
    /// unseen n-grams of that order no probability at all.
    #[error(
        "the {order}-gram discount is degenerate ({singletons} n-grams seen once, \
         {doubletons} seen twice): the corpus is too small or too repetitive to \
         estimate a backoff weight from"
    )]
    DegenerateDiscount {
        /// Which order failed: 1, 2 or 3.
        order: usize,
        /// How many n-grams of that order occurred exactly once.
        singletons: usize,
        /// How many occurred exactly twice.
        doubletons: usize,
    },
    /// The lexicon is too large for three tokens to share a `u64` key.
    #[error("a lexicon of {count} characters exceeds what the model's key packing supports")]
    VocabularyTooLarge {
        /// Size of the offending lexicon.
        count: usize,
    },
    /// A character was asked about that the model was never trained to spell.
    #[error("{ch:?} is not in the model's vocabulary")]
    UnknownCharacter {
        /// The offending character.
        ch: char,
    },
    /// More context was supplied than the model's order can use.
    #[error("a context of {len} characters exceeds the model's order of {order}")]
    ContextTooLong {
        /// How much context was supplied.
        len: usize,
        /// The model's order.
        order: usize,
    },
    /// The model was trained against a lexicon of a different size.
    #[error("the model was trained against {model} characters, but this lexicon holds {lexicon}")]
    LexiconSize {
        /// Vocabulary size recorded in the model.
        model: usize,
        /// Size of the lexicon it was loaded with.
        lexicon: usize,
    },
    /// The model was trained against a lexicon that ordered its characters
    /// differently, so every `CharId` would mean the wrong thing.
    #[error("the model expects {ch:?} at index {index}, which this lexicon does not")]
    LexiconContent {
        /// Where the two lexicons diverge.
        index: usize,
        /// The character the model expected there.
        ch: char,
    },
    /// The file decoded, but its tables do not agree with each other.
    #[error("the model file is internally inconsistent")]
    Corrupt,
    /// The model could not be serialised.
    #[error("could not serialise the model")]
    Encode(#[source] postcard::Error),
    /// The bytes are not a model.
    #[error("could not deserialise the model")]
    Decode(#[source] postcard::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use ime_decode::{BeamOptions, Candidates, Transition, Uniform, decode};
    use ime_pinyin::{Lexicon, SegmentLattice, SegmentOptions, SyllableTable};

    /// A handful of sentences; small enough to reason about, varied enough that
    /// every order has both singletons and repeats to estimate a discount from.
    const CORPUS: &str = include_str!("../data/tiny-corpus.txt");

    fn fixture() -> (SyllableTable, Lexicon) {
        let table = SyllableTable::load();
        let lexicon = Lexicon::load(&table).expect("generated tables must agree");
        (table, lexicon)
    }

    fn trained(lexicon: &Lexicon) -> NgramModel {
        let mut counter = Counter::new(lexicon).expect("the lexicon fits the key packing");
        for line in CORPUS.lines() {
            counter.observe(line);
        }
        counter
            .finish()
            .expect("the corpus is rich enough to train on")
    }

    fn chars(text: &str) -> Vec<char> {
        text.chars().collect()
    }

    #[test]
    fn a_seen_trigram_beats_an_unseen_one() {
        let (_, lexicon) = fixture();
        let model = trained(&lexicon);
        let seen = model
            .probability(&chars("中国"), '人')
            .expect("中国人 is in the corpus");
        let unseen = model
            .probability(&chars("中国"), '雨')
            .expect("雨 is in the corpus but not after 中国");
        assert!(seen > unseen, "{seen} should beat {unseen}");
    }

    #[test]
    fn a_seen_bigram_beats_an_unseen_one_after_an_unseen_context() {
        // 银 never follows 好, so the trigram level has nothing to say and the
        // ranking has to come from the bigram level underneath it.
        let (_, lexicon) = fixture();
        let model = trained(&lexicon);
        let seen = model
            .probability(&chars("好银"), '行')
            .expect("银行 is in the corpus");
        let unseen = model
            .probability(&chars("好银"), '天')
            .expect("银天 is not");
        assert!(seen > unseen, "{seen} should beat {unseen}");
    }

    #[test]
    fn nothing_is_impossible() {
        let (_, lexicon) = fixture();
        let model = trained(&lexicon);
        for (context, current) in [("", '龘'), ("中国", '龘'), ("龘龘", '龘')] {
            let p = model
                .probability(&chars(context), current)
                .expect("every lexicon character is scorable");
            assert!(
                p > 0.0 && p.is_finite(),
                "P({current:?}|{context:?}) = {p} leaves the decoder no ranking"
            );
        }
    }

    #[test]
    fn every_distribution_sums_to_one() {
        let (_, lexicon) = fixture();
        let model = trained(&lexicon);
        for context in ["", "中", "中国", "龘", "龘龘"] {
            let context = chars(context);
            let mut total = f64::from(
                model
                    .end_probability(&context)
                    .expect("the end is always a target"),
            );
            for ch in lexicon.characters() {
                total += f64::from(
                    model
                        .probability(&context, *ch)
                        .expect("every lexicon character is a target"),
                );
            }
            assert!(
                (total - 1.0).abs() < 1e-3,
                "P(.|{context:?}) sums to {total}, not 1"
            );
        }
    }

    #[test]
    fn a_context_longer_than_the_order_is_rejected() {
        let (_, lexicon) = fixture();
        let model = trained(&lexicon);
        assert!(matches!(
            model.probability(&chars("中国人"), '民'),
            Err(NgramError::ContextTooLong { len: 3, order: 3 })
        ));
    }

    #[test]
    fn a_character_outside_the_lexicon_is_rejected() {
        let (_, lexicon) = fixture();
        let model = trained(&lexicon);
        assert!(matches!(
            model.probability(&[], 'a'),
            Err(NgramError::UnknownCharacter { ch: 'a' })
        ));
    }

    #[test]
    fn punctuation_breaks_the_sequence_rather_than_joining_it() {
        // `你好，我` never makes 好我 a bigram, because the comma is a break.
        let (_, lexicon) = fixture();
        let model = trained(&lexicon);
        let across = model.probability(&chars("你好"), '我').expect("scorable");
        let within = model.probability(&chars("你好"), '世').expect("scorable");
        assert!(
            within > across,
            "the comma should have cut 好我 apart: {within} vs {across}"
        );
    }

    #[test]
    fn a_corpus_with_no_hanzi_is_rejected() {
        let (_, lexicon) = fixture();
        let mut counter = Counter::new(&lexicon).expect("the lexicon fits");
        counter.observe("hello, world!");
        assert!(matches!(counter.finish(), Err(NgramError::EmptyCorpus)));
    }

    #[test]
    fn a_corpus_too_repetitive_to_discount_is_rejected() {
        let (_, lexicon) = fixture();
        let mut counter = Counter::new(&lexicon).expect("the lexicon fits");
        for _ in 0..3 {
            counter.observe("中国中国");
        }
        assert!(matches!(
            counter.finish(),
            Err(NgramError::DegenerateDiscount { .. })
        ));
    }

    #[test]
    fn a_model_survives_a_round_trip() {
        let (_, lexicon) = fixture();
        let model = trained(&lexicon);
        let bytes = model.to_bytes().expect("the model serialises");
        let reloaded = NgramModel::from_bytes(&bytes, &lexicon).expect("the model reloads");
        assert_eq!(reloaded.vocabulary_size(), model.vocabulary_size());
        assert_eq!(reloaded.trigram_types(), model.trigram_types());
        for (context, current) in [("", '中'), ("中", '国'), ("中国", '人')] {
            assert_eq!(
                reloaded.probability(&chars(context), current).ok(),
                model.probability(&chars(context), current).ok()
            );
        }
    }

    #[test]
    fn garbage_is_not_a_model() {
        let (_, lexicon) = fixture();
        assert!(matches!(
            NgramModel::from_bytes(&[0xff, 0x00, 0x13], &lexicon),
            Err(NgramError::Decode(_) | NgramError::LexiconSize { .. })
        ));
    }

    #[test]
    fn the_baseline_decodes_a_phrase_it_was_trained_on() {
        let (table, lexicon) = fixture();
        let model = trained(&lexicon);
        let options = SegmentOptions {
            allow_incomplete_tail: false,
            ..SegmentOptions::default()
        };
        let lattice =
            SegmentLattice::build("zhongguorenmin", &table, &options).expect("the input reads");
        let batch =
            Candidates::build(&lattice.k_best(&options), &lexicon).expect("masks are non-empty");
        let best =
            decode(&batch, &Uniform, &model, &BeamOptions::default()).expect("the batch decodes");
        assert_eq!(best[0].text(&lexicon), "中国人民");
    }

    #[test]
    fn the_baseline_decodes_an_abbreviation_it_was_trained_on() {
        let (table, lexicon) = fixture();
        let model = trained(&lexicon);
        let options = SegmentOptions {
            allow_incomplete_tail: false,
            max_paths: 16,
            ..SegmentOptions::default()
        };
        let lattice = SegmentLattice::build("zgrm", &table, &options).expect("the input reads");
        let batch =
            Candidates::build(&lattice.k_best(&options), &lexicon).expect("masks are non-empty");
        let best = decode(
            &batch,
            &Uniform,
            &model,
            &BeamOptions {
                beam_width: std::num::NonZeroUsize::new(64).expect("64 is not zero"),
                ..BeamOptions::default()
            },
        )
        .expect("the batch decodes");
        let texts: Vec<String> = best.iter().map(|h| h.text(&lexicon)).collect();
        assert!(
            texts.contains(&"中国人民".to_owned()),
            "expected 中国人民 among {texts:?}"
        );
    }

    #[test]
    fn the_transition_model_conditions_on_two_characters() {
        assert_eq!(<NgramModel as Transition>::HISTORY, ORDER - 1);
    }
}
