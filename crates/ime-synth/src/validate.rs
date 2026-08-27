//! The gate a generated example has to pass to become a training sample.
//!
//! Synthetic text gets the *same* filter as fetched text -- `ime-corpus`'s
//! [`SampleFilter`], unchanged -- because a sample that a downstream stage would
//! choke on is worse than no sample at all, and because holding synthetic text to
//! a looser bar than real text is how a corpus rots. Two checks are added on top,
//! and both are specific to synthesis:
//!
//! * the sentence must contain its seed term verbatim, which is the only
//!   mechanical evidence that the grounded prompt was followed at all; and
//! * the preceding turn has to be typable too, because it is conditioning text
//!   the context model will be trained on, even though it is not itself a target
//!   and so has no minimum length and no duplicate rule.
//!
//! Everything is normalised with the corpus normaliser first. The model answers
//! in whichever script it feels like, wikipedia's terms arrive in a mix of both,
//! and a traditional 爺青結 against a simplified 爷青结 would fail the verbatim
//! check for a reason that has nothing to do with the model's behaviour.

use crate::error::Result;
use crate::llm::Example;
use ime_corpus::text::han_ratio;
use ime_corpus::{Normalizer, SampleFilter};
use ime_g2p::text::is_han;
use ime_pinyin::Lexicon;
use serde::{Deserialize, Serialize};

/// One example that passed, normalised and ready to be written.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Accepted {
    /// The target sentence.
    pub text: String,
    /// The preceding turn, `None` when the model returned none.
    pub context: Option<String>,
}

/// Why generated examples were kept or dropped, for the run summary.
///
/// The five filter reasons are `ime-corpus`'s own, carried across so that a
/// synthetic batch's drop profile can be read beside a fetched source's.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub struct DropCounts {
    /// Examples that passed every rule.
    pub kept: usize,
    /// Sentences that do not contain their seed term.
    pub missing_term: usize,
    /// Preceding turns that are not typable Chinese.
    pub bad_context: usize,
    /// Sentences shorter than the corpus minimum.
    pub too_short: usize,
    /// Sentences longer than the corpus maximum.
    pub too_long: usize,
    /// Sentences below the corpus Han ratio.
    pub not_chinese_enough: usize,
    /// Sentences holding a character the pinyin lexicon cannot read.
    pub unknown_character: usize,
    /// Sentences identical to one already kept in this run.
    pub duplicate: usize,
}

impl DropCounts {
    /// Every example judged.
    #[must_use]
    pub fn considered(&self) -> usize {
        self.kept + self.dropped()
    }

    /// Every example rejected.
    #[must_use]
    pub fn dropped(&self) -> usize {
        self.missing_term
            + self.bad_context
            + self.too_short
            + self.too_long
            + self.not_chinese_enough
            + self.unknown_character
            + self.duplicate
    }

    /// The share of judged examples that were rejected, as a fraction.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "a rate over batch-sized counts is compared against a percentage"
    )]
    pub fn drop_rate(&self) -> f64 {
        let considered = self.considered();
        if considered == 0 {
            return 0.0;
        }
        self.dropped() as f64 / considered as f64
    }

    /// The reasons and their counts, in the order the report prints them.
    #[must_use]
    pub fn reasons(&self) -> [(&'static str, usize); 7] {
        [
            ("missing term", self.missing_term),
            ("bad context", self.bad_context),
            ("too short", self.too_short),
            ("too long", self.too_long),
            ("not Chinese enough", self.not_chinese_enough),
            ("unknown character", self.unknown_character),
            ("duplicate", self.duplicate),
        ]
    }
}

/// Whether a preceding turn is Chinese the pinyin lexicon can read throughout.
#[must_use]
pub fn is_typable_context(context: &str, lexicon: &Lexicon) -> bool {
    context.chars().count() <= ime_corpus::filter::MAX_CHARACTERS
        && han_ratio(context) >= ime_corpus::filter::MIN_HAN_RATIO
        && !context
            .chars()
            .any(|character| is_han(character) && lexicon.id_of(character).is_none())
}

/// Normalises and judges generated examples, tallying why each was dropped.
///
/// One validator is built per run rather than per term, so the duplicate rule
/// spans the whole batch: a term the model has a single favourite sentence for
/// should contribute that sentence once.
pub struct Validator<'a> {
    normalizer: &'a Normalizer,
    lexicon: &'a Lexicon,
    filter: SampleFilter<'a>,
    missing_term: usize,
    bad_context: usize,
}

impl<'a> Validator<'a> {
    /// A validator gating on `lexicon`'s coverage, normalising with `normalizer`.
    #[must_use]
    pub fn new(normalizer: &'a Normalizer, lexicon: &'a Lexicon) -> Self {
        Self {
            normalizer,
            lexicon,
            filter: SampleFilter::new(lexicon),
            missing_term: 0,
            bad_context: 0,
        }
    }

    /// Judge one example against its seed term, returning it only if it passed.
    ///
    /// # Errors
    ///
    /// If the normaliser refuses the text, which means its length invariant
    /// broke rather than that the example was bad.
    pub fn judge(&mut self, term: &str, example: &Example) -> Result<Option<Accepted>> {
        let text = self.normalizer.normalize(&example.text)?;
        if !text.contains(term) {
            self.missing_term += 1;
            return Ok(None);
        }
        let raw_context = self.normalizer.normalize(&example.context)?;
        let context = (!raw_context.is_empty()).then_some(raw_context);
        if context
            .as_deref()
            .is_some_and(|context| !is_typable_context(context, self.lexicon))
        {
            self.bad_context += 1;
            return Ok(None);
        }
        if !self.filter.accepts(&text) {
            return Ok(None);
        }
        Ok(Some(Accepted { text, context }))
    }

    /// The verdicts recorded so far.
    #[must_use]
    pub fn counts(&self) -> DropCounts {
        let filter = self.filter.counts();
        DropCounts {
            kept: filter.kept,
            missing_term: self.missing_term,
            bad_context: self.bad_context,
            too_short: filter.too_short,
            too_long: filter.too_long,
            not_chinese_enough: filter.not_chinese_enough,
            unknown_character: filter.unknown_character,
            duplicate: filter.duplicate,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ime_pinyin::SyllableTable;

    fn lexicon() -> Lexicon {
        Lexicon::load(&SyllableTable::load()).expect("the generated pinyin tables agree")
    }

    fn normalizer() -> Normalizer {
        Normalizer::new().expect("the bundled t2s configuration loads")
    }

    fn example(context: &str, text: &str) -> Example {
        Example {
            context: context.to_owned(),
            text: text.to_owned(),
        }
    }

    #[test]
    fn a_usable_example_comes_back_normalised_with_its_context() {
        let lexicon = lexicon();
        let normalizer = normalizer();
        let mut validator = Validator::new(&normalizer, &lexicon);
        let accepted = validator
            .judge(
                "爷青结",
                &example("你看完结局了吗", "这季追完了我直接爷青结"),
            )
            .expect("the normaliser holds")
            .expect("the example passes");
        assert_eq!(accepted.text, "这季追完了我直接爷青结");
        assert_eq!(accepted.context.as_deref(), Some("你看完结局了吗"));
        assert_eq!(validator.counts().kept, 1);
    }

    #[test]
    fn an_empty_context_becomes_none_rather_than_an_empty_string() {
        let lexicon = lexicon();
        let normalizer = normalizer();
        let mut validator = Validator::new(&normalizer, &lexicon);
        let accepted = validator
            .judge("爷青结", &example("", "这季追完了我直接爷青结"))
            .expect("the normaliser holds")
            .expect("the example passes");
        assert_eq!(accepted.context, None);
    }

    #[test]
    fn a_traditional_reply_is_simplified_before_the_term_is_looked_for() {
        let lexicon = lexicon();
        let normalizer = normalizer();
        let mut validator = Validator::new(&normalizer, &lexicon);
        let accepted = validator
            .judge("爷青结", &example("", "這季追完了我直接爺青結"))
            .expect("the normaliser holds")
            .expect("the example passes once it is simplified");
        assert_eq!(accepted.text, "这季追完了我直接爷青结");
        assert_eq!(validator.counts().missing_term, 0);
    }

    #[test]
    fn a_sentence_without_its_term_is_dropped_under_its_own_reason() {
        let lexicon = lexicon();
        let normalizer = normalizer();
        let mut validator = Validator::new(&normalizer, &lexicon);
        assert!(
            validator
                .judge("爷青结", &example("", "这部番真的完结了"))
                .expect("the normaliser holds")
                .is_none()
        );
        assert_eq!(validator.counts().missing_term, 1);
        assert_eq!(validator.counts().kept, 0);
    }

    #[test]
    fn the_corpus_length_script_and_duplicate_rules_all_still_apply() {
        let lexicon = lexicon();
        let normalizer = normalizer();
        let mut validator = Validator::new(&normalizer, &lexicon);
        assert!(
            validator
                .judge("爷青", &example("", "爷青"))
                .expect("the normaliser holds")
                .is_none()
        );
        assert!(
            validator
                .judge(
                    "爷青结",
                    &example("", &format!("{}爷青结", "很".repeat(80)))
                )
                .expect("the normaliser holds")
                .is_none()
        );
        assert!(
            validator
                .judge("爷青结", &example("", "爷青结 finally over guys"))
                .expect("the normaliser holds")
                .is_none()
        );
        assert!(
            validator
                .judge("爷青结", &example("", "这季追完了我直接爷青结"))
                .expect("the normaliser holds")
                .is_some()
        );
        assert!(
            validator
                .judge("爷青结", &example("啊", "这季追完了我直接爷青结"))
                .expect("the normaliser holds")
                .is_none()
        );
        let counts = validator.counts();
        assert_eq!(counts.too_short, 1);
        assert_eq!(counts.too_long, 1);
        assert_eq!(counts.not_chinese_enough, 1);
        assert_eq!(counts.duplicate, 1);
        assert_eq!(counts.kept, 1);
        assert_eq!(counts.considered(), 5);
    }

    #[test]
    fn a_context_that_is_not_typable_chinese_takes_the_example_with_it() {
        let lexicon = lexicon();
        let normalizer = normalizer();
        let mut validator = Validator::new(&normalizer, &lexicon);
        assert!(
            validator
                .judge(
                    "爷青结",
                    &example("lol what happened", "这季追完了我直接爷青结")
                )
                .expect("the normaliser holds")
                .is_none()
        );
        assert!(
            validator
                .judge(
                    "爷青结",
                    &example(&"前".repeat(100), "这季追完了我直接爷青结")
                )
                .expect("the normaliser holds")
                .is_none()
        );
        assert_eq!(validator.counts().bad_context, 2);
    }

    #[test]
    fn the_drop_rate_is_the_share_of_judged_examples_that_were_rejected() {
        let counts = DropCounts {
            kept: 7,
            missing_term: 2,
            duplicate: 1,
            ..DropCounts::default()
        };
        assert_eq!(counts.considered(), 10);
        assert_eq!(counts.dropped(), 3);
        assert!((counts.drop_rate() - 0.3).abs() < f64::EPSILON);
        assert!(DropCounts::default().drop_rate().abs() < f64::EPSILON);
    }
}
