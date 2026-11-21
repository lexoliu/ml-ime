//! The gate a generated example has to pass to become a training sample.
//!
//! Synthetic text gets the *same* length, coverage and duplicate rules as fetched
//! text -- `ime-corpus`'s [`SampleFilter`] -- because a sample that a downstream
//! stage would choke on is worse than no sample at all, and because holding
//! synthetic text to a looser bar than real text is how a corpus rots. Three
//! checks are added on top, and all three are specific to synthesis:
//!
//! * the sentence must contain its seed term verbatim, which is the only
//!   mechanical evidence that the grounded prompt was followed at all;
//! * it must be Chinese throughout, which the corpus filter no longer checks
//!   because it is handed runs of Han characters and this is handed whatever the
//!   model wrote; and
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
use crate::seed::Seed;
use ime_corpus::filter::MIN_HAN_RATIO;
use ime_corpus::leak::EvalGuard;
use ime_corpus::text::han_ratio;
use ime_corpus::{Normalizer, SampleFilter};
use ime_g2p::text::is_han;
use ime_pinyin::Lexicon;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Leading Han characters tracked per sentence to cap repetitive opening templates.
pub const PREFIX_HAN: usize = 4;

/// Default maximum times one non-seed 4-Han opening prefix may appear in a batch.
pub const DEFAULT_MAX_PREFIX_REPEATS: usize = 8;

/// Meta-definitional phrases that turn a sentence into a gloss rather than usage.
const GLOSS_MARKERS: [&str; 8] = [
    "意思是",
    "指的是",
    "用来表示",
    "泛指",
    "特指",
    "网络用语",
    "流行语",
    "这个梗",
];

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
    /// Sentences that do not contain their seed term (or contain a contrastive foil).
    pub missing_term: usize,
    /// Preceding turns that are not typable Chinese, or missing/leaking on contrastive prompts.
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
    /// Examples overlapping an evaluation record exactly or over a 13-Han window.
    #[serde(default)]
    pub eval_leak: usize,
    /// Definitional meta-prose or over-repeated opening template prefixes.
    #[serde(default)]
    pub boilerplate: usize,
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
            + self.eval_leak
            + self.boilerplate
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
    pub fn reasons(&self) -> [(&'static str, usize); 9] {
        [
            ("missing term", self.missing_term),
            ("bad context", self.bad_context),
            ("too short", self.too_short),
            ("too long", self.too_long),
            ("not Chinese enough", self.not_chinese_enough),
            ("unknown character", self.unknown_character),
            ("duplicate", self.duplicate),
            ("eval leak", self.eval_leak),
            ("boilerplate", self.boilerplate),
        ]
    }
}

/// Whether a preceding turn is Chinese the pinyin lexicon can read throughout.
#[must_use]
pub fn is_typable_context(context: &str, lexicon: &Lexicon) -> bool {
    context.chars().count() <= ime_corpus::filter::MAX_CHARACTERS
        && han_ratio(context) >= MIN_HAN_RATIO
        && !context
            .chars()
            .any(|character| is_han(character) && lexicon.id_of(character).is_none())
}

fn leading_han_prefix(text: &str) -> Option<[char; PREFIX_HAN]> {
    let mut prefix = ['\0'; PREFIX_HAN];
    let mut count = 0;
    for character in text.chars().filter(|character| is_han(*character)) {
        prefix[count] = character;
        count += 1;
        if count == PREFIX_HAN {
            return Some(prefix);
        }
    }
    None
}

fn has_gloss_boilerplate(text: &str, term: &str) -> bool {
    let stripped = text.replace(term, "");
    GLOSS_MARKERS.iter().any(|marker| stripped.contains(marker))
}

/// Normalises and judges generated examples, tallying why each was dropped.
///
/// One validator is built per run rather than per term, so the duplicate rule
/// and the opening-prefix repetition cap span the whole batch.
pub struct Validator<'a> {
    normalizer: &'a Normalizer,
    lexicon: &'a Lexicon,
    eval_guard: Option<&'a EvalGuard>,
    max_prefix_repeats: usize,
    prefixes: HashMap<[char; PREFIX_HAN], usize>,
    filter: SampleFilter<'a>,
    missing_term: usize,
    bad_context: usize,
    not_chinese_enough: usize,
    eval_leak: usize,
    boilerplate: usize,
}

impl<'a> Validator<'a> {
    /// A validator gating on `lexicon`'s coverage, normalising with `normalizer`.
    #[must_use]
    pub fn new(normalizer: &'a Normalizer, lexicon: &'a Lexicon) -> Self {
        Self::with_guards(normalizer, lexicon, None, DEFAULT_MAX_PREFIX_REPEATS)
    }

    /// A validator also enforcing an [`EvalGuard`] and a template-prefix cap (`0` disables).
    #[must_use]
    pub fn with_guards(
        normalizer: &'a Normalizer,
        lexicon: &'a Lexicon,
        eval_guard: Option<&'a EvalGuard>,
        max_prefix_repeats: usize,
    ) -> Self {
        Self {
            normalizer,
            lexicon,
            eval_guard,
            max_prefix_repeats,
            prefixes: HashMap::new(),
            filter: SampleFilter::new(lexicon),
            missing_term: 0,
            bad_context: 0,
            not_chinese_enough: 0,
            eval_leak: 0,
            boilerplate: 0,
        }
    }

    /// Judge one example against `term` with no contrastive foil.
    ///
    /// # Errors
    ///
    /// If the normaliser's character-count invariant fails.
    pub fn judge(&mut self, term: &str, example: &Example) -> Result<Option<Accepted>> {
        self.judge_with_foil(term, None, example)
    }

    /// Judge one example against its full [`Seed`] (including contrastive foil rules).
    ///
    /// # Errors
    ///
    /// If the normaliser's character-count invariant fails.
    pub fn judge_seed(&mut self, seed: &Seed, example: &Example) -> Result<Option<Accepted>> {
        self.judge_with_foil(&seed.term, seed.foil.as_deref(), example)
    }

    fn judge_with_foil(
        &mut self,
        term: &str,
        foil: Option<&str>,
        example: &Example,
    ) -> Result<Option<Accepted>> {
        let text = self.normalizer.normalize(&example.text)?;
        if !text.contains(term) || foil.is_some_and(|competitor| text.contains(competitor)) {
            self.missing_term += 1;
            return Ok(None);
        }
        let raw_context = self.normalizer.normalize(&example.context)?;
        let context = (!raw_context.is_empty()).then_some(raw_context);
        if context
            .as_deref()
            .is_some_and(|ctx| !is_typable_context(ctx, self.lexicon))
        {
            self.bad_context += 1;
            return Ok(None);
        }
        if let Some(competitor) = foil {
            let Some(ctx) = context.as_deref() else {
                self.bad_context += 1;
                return Ok(None);
            };
            if ctx.contains(term) || ctx.contains(competitor) {
                self.bad_context += 1;
                return Ok(None);
            }
        }
        if han_ratio(&text) < MIN_HAN_RATIO {
            self.not_chinese_enough += 1;
            return Ok(None);
        }
        if has_gloss_boilerplate(&text, term)
            || context
                .as_deref()
                .is_some_and(|ctx| has_gloss_boilerplate(ctx, term))
        {
            self.boilerplate += 1;
            return Ok(None);
        }
        if let Some(guard) = self.eval_guard
            && (guard.holds_text(&text)
                || guard.shares_window(
                    context
                        .iter()
                        .map(String::as_str)
                        .chain(std::iter::once(text.as_str())),
                ))
        {
            self.eval_leak += 1;
            return Ok(None);
        }
        let prefix = (!text.starts_with(term))
            .then(|| leading_han_prefix(&text))
            .flatten();
        if self.max_prefix_repeats > 0
            && let Some(key) = prefix
            && self.prefixes.get(&key).copied().unwrap_or(0) >= self.max_prefix_repeats
        {
            self.boilerplate += 1;
            return Ok(None);
        }
        if !self.filter.accepts(&text) {
            return Ok(None);
        }
        if let Some(key) = prefix {
            *self.prefixes.entry(key).or_default() += 1;
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
            too_short: filter.too_short_run,
            too_long: filter.too_long_run,
            not_chinese_enough: self.not_chinese_enough,
            unknown_character: filter.unknown_character,
            duplicate: filter.duplicate,
            eval_leak: self.eval_leak,
            boilerplate: self.boilerplate,
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

    #[test]
    fn eval3_exact_and_thirteen_han_window_leaks_are_rejected() {
        let lexicon = lexicon();
        let normalizer = normalizer();
        let guard = EvalGuard::new([
            (
                "这季追完了我直接爷青结",
                Some("今天下午我们一起把最后两集看完了，"),
            ),
            ("一张旧车票", None),
        ]);
        let mut validator = Validator::with_guards(&normalizer, &lexicon, Some(&guard), 8);

        // Exact target match against eval3.
        assert!(
            validator
                .judge("爷青结", &example("随便聊两句", "这季追完了我直接爷青结"))
                .expect("normalises")
                .is_none()
        );
        // 13-character Han overlap across context + text against eval3's context + text.
        assert!(
            validator
                .judge(
                    "爷青结",
                    &example("我们一起把最后两集看完了", "这季看完真的爷青结啊朋友们")
                )
                .expect("normalises")
                .is_none()
        );
        // Distinct usage sentence with <13 shared Han characters passes.
        assert!(
            validator
                .judge(
                    "爷青结",
                    &example("漫画终于更新到大结局了", "看到最后一页真的爷青结")
                )
                .expect("normalises")
                .is_some()
        );
        assert_eq!(validator.counts().eval_leak, 2);
        assert_eq!(validator.counts().kept, 1);
    }

    #[test]
    fn gloss_prose_and_overused_opening_prefixes_are_rejected_as_boilerplate() {
        let lexicon = lexicon();
        let normalizer = normalizer();
        let mut validator = Validator::with_guards(&normalizer, &lexicon, None, 2);

        assert!(
            validator
                .judge("爷青结", &example("", "爷青结的意思是青春结束了"))
                .expect("normalises")
                .is_none()
        );
        assert!(
            validator
                .judge("爷青结", &example("", "不得不说看完结局真的爷青结"))
                .expect("normalises")
                .is_some()
        );
        assert!(
            validator
                .judge("爷青结", &example("", "不得不说听到片尾曲就爷青结"))
                .expect("normalises")
                .is_some()
        );
        assert!(
            validator
                .judge("爷青结", &example("", "不得不说这次联动直接爷青结"))
                .expect("normalises")
                .is_none()
        );
        assert_eq!(validator.counts().boilerplate, 2);
        assert_eq!(validator.counts().kept, 2);
    }

    #[test]
    fn contrastive_seeds_require_disambiguating_context_without_leaking_either_word() {
        let lexicon = lexicon();
        let normalizer = normalizer();
        let mut validator = Validator::new(&normalizer, &lexicon);
        let seed = Seed::contrastive("公式".to_owned(), "公事".to_owned(), String::new());

        // Missing context -> rejected.
        assert!(
            validator
                .judge_seed(&seed, &example("", "这道物理题套第二个公式就能算出来"))
                .expect("normalises")
                .is_none()
        );
        // Context mentions the target word verbatim -> rejected.
        assert!(
            validator
                .judge_seed(
                    &seed,
                    &example("你把那个公式背下来了吗", "这道题套第二个公式就能算出来")
                )
                .expect("normalises")
                .is_none()
        );
        // Target contains the homophone foil -> rejected.
        assert!(
            validator
                .judge_seed(
                    &seed,
                    &example("今天数学卷子最后一道大题好难", "我忙完公事再来看这个公式")
                )
                .expect("normalises")
                .is_none()
        );
        // Clean disambiguating context + target -> kept.
        assert!(
            validator
                .judge_seed(
                    &seed,
                    &example(
                        "今天数学卷子最后一道大题好难",
                        "这道题套第二个公式就能算出来"
                    )
                )
                .expect("normalises")
                .is_some()
        );
        assert_eq!(validator.counts().bad_context, 2);
        assert_eq!(validator.counts().missing_term, 1);
        assert_eq!(validator.counts().kept, 1);
    }
}
