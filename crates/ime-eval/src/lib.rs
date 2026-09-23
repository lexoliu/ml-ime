//! The evaluation harness: sentence and character accuracy over full pinyin,
//! abbreviations, fuzzy readings and typos, with and without context.
//!
//! The harness knows nothing about how a sentence was produced. Both routes
//! reach it through [`Hypothesize`], which is handed the context whether or not
//! the engine behind it has any use for one -- the n-gram baseline ignores it,
//! the neural model conditions on it, and the difference between the two numbers
//! is the entire question milestone 3 asks.

mod metrics;
mod record;

pub use metrics::{Hypothesize, Report, Request, evaluate};
pub use record::{EvalRecord, EvalSet, Slice};

use thiserror::Error;

/// Why an evaluation set could not be read.
#[derive(Debug, Error)]
pub enum EvalError {
    /// A line was not a JSON record of the expected shape.
    #[error("line {line} is not an evaluation record")]
    Malformed {
        /// One-based line number in the file.
        line: usize,
        /// What the JSON parser objected to.
        #[source]
        source: serde_json::Error,
    },
    /// A record left a required field empty.
    #[error("line {line}: the {field} field is empty")]
    EmptyField {
        /// One-based line number in the file.
        line: usize,
        /// Which field.
        field: &'static str,
    },
    /// The file held no records.
    #[error("the evaluation set holds no records")]
    Empty,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::convert::Infallible;
    use std::num::NonZeroUsize;

    const SET: &str = include_str!("../data/tiny-eval.jsonl");

    fn k(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).expect("test constants are not zero")
    }

    /// An engine with its answers written down in advance, so the metrics can be
    /// checked against numbers worked out by hand.
    struct Canned {
        answers: HashMap<&'static str, Vec<String>>,
    }

    impl Canned {
        fn new() -> Self {
            let answers = [
                ("zhongguo", vec!["中国", "钟国", "中过"]),
                ("renmin", vec!["人们", "人民"]),
                ("yinhang", vec!["因航", "阴行", "银行"]),
                ("beijing", vec!["背景", "被警"]),
                ("tianqi", vec!["天气"]),
            ]
            .into_iter()
            .map(|(pinyin, texts)| {
                (
                    pinyin,
                    texts.into_iter().map(str::to_owned).collect::<Vec<_>>(),
                )
            })
            .collect();
            Self { answers }
        }
    }

    impl Hypothesize for Canned {
        type Error = Infallible;

        fn hypotheses(&self, request: &Request<'_>) -> Result<Vec<String>, Self::Error> {
            Ok(self
                .answers
                .get(request.pinyin)
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .take(request.top_k.get())
                .collect())
        }
    }

    #[test]
    fn a_set_parses_with_and_without_context() {
        let set = EvalSet::parse(SET).expect("the fixture parses");
        assert_eq!(set.len(), 5);
        assert_eq!(set.with_context(), 2);
        assert_eq!(set.records()[0].text, "中国");
        assert_eq!(set.records()[1].context, None);
        assert_eq!(set.records()[4].context.as_deref(), Some("今天的"));
    }

    #[test]
    fn a_malformed_line_names_itself() {
        let broken = "{\"pinyin\": \"a\", \"text\": \"中\"}\nnot json\n";
        assert!(matches!(
            EvalSet::parse(broken),
            Err(EvalError::Malformed { line: 2, .. })
        ));
    }

    #[test]
    fn an_unknown_field_is_rejected_rather_than_ignored() {
        let extra = "{\"pinyin\": \"a\", \"text\": \"中\", \"weight\": 2}";
        assert!(matches!(
            EvalSet::parse(extra),
            Err(EvalError::Malformed { line: 1, .. })
        ));
    }

    #[test]
    fn an_empty_field_is_rejected() {
        let blank = "{\"pinyin\": \"\", \"text\": \"中\"}";
        assert!(matches!(
            EvalSet::parse(blank),
            Err(EvalError::EmptyField {
                line: 1,
                field: "pinyin"
            })
        ));
    }

    #[test]
    fn an_empty_set_is_rejected() {
        assert!(matches!(EvalSet::parse("\n\n  \n"), Err(EvalError::Empty)));
    }

    #[test]
    fn the_metrics_are_what_the_answers_say_they_are() {
        let set = EvalSet::parse(SET).expect("the fixture parses");
        let report = evaluate(&set, &Canned::new(), k(3)).expect("the canned engine cannot fail");

        assert_eq!(report.records(), 5);
        // 中国 and 天气 are right first time.
        assert_eq!(report.top1_hits(), 2);
        // Those two, plus 人民 second and 银行 third; 北京 never comes at all.
        assert_eq!(report.topk_hits(), 4);
        assert_eq!(report.unanswered(), 0);
        // 中国 2, 人们 1, 因航 0, 背景 0, 天气 2, over five two-character texts.
        assert_eq!(report.characters(), 10);
        assert_eq!(report.character_hits(), 5);

        assert!((report.top1_accuracy() - 0.4).abs() < 1e-12);
        assert!((report.topk_accuracy() - 0.8).abs() < 1e-12);
        assert!((report.character_accuracy() - 0.5).abs() < 1e-12);
        let expected = (1.0 + 0.5 + 1.0 / 3.0 + 0.0 + 1.0) / 5.0;
        assert!(
            (report.mean_reciprocal_rank() - expected).abs() < 1e-5,
            "got {}",
            report.mean_reciprocal_rank()
        );
    }

    #[test]
    fn a_narrower_top_k_loses_the_answers_that_needed_the_room() {
        let set = EvalSet::parse(SET).expect("the fixture parses");
        let report = evaluate(&set, &Canned::new(), k(1)).expect("the canned engine cannot fail");
        assert_eq!(report.top1_hits(), 2);
        assert_eq!(report.topk_hits(), 2);
        assert!((report.mean_reciprocal_rank() - 0.4).abs() < 1e-9);
    }

    #[test]
    fn a_silent_engine_is_counted_rather_than_credited() {
        let mut report = Report::new(k(3));
        report.observe("中国", &[]);
        report.observe("人民", &["人民".to_owned()]);
        assert_eq!(report.records(), 2);
        assert_eq!(report.unanswered(), 1);
        assert_eq!(report.characters(), 4);
        assert_eq!(report.character_hits(), 2);
        assert!((report.top1_accuracy() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn a_wrong_length_hypothesis_is_scored_over_the_expected_length() {
        // A hypothesis from a wrong segmentation is short; the missing positions
        // are misses, not free passes.
        let mut report = Report::new(k(3));
        report.observe("西安人", &["咸人".to_owned()]);
        assert_eq!(report.characters(), 3);
        assert_eq!(report.character_hits(), 0);
    }

    #[test]
    fn the_report_renders_as_a_table() {
        let set = EvalSet::parse(SET).expect("the fixture parses");
        let report = evaluate(&set, &Canned::new(), k(3)).expect("the canned engine cannot fail");
        let rendered = report.to_string();
        assert!(
            rendered.contains("sentence, top-1       0.4000  2 / 5"),
            "{rendered}"
        );
        assert!(
            rendered.contains("sentence, top-3       0.8000  4 / 5"),
            "{rendered}"
        );
        assert!(
            rendered.contains("character             0.5000  5 / 10"),
            "{rendered}"
        );
        assert!(rendered.contains("MRR@3"), "{rendered}");
        assert!(
            rendered.ends_with('\n'),
            "the table must end a line: {rendered:?}"
        );
    }

    #[test]
    fn the_context_reaches_the_engine() {
        struct Recorder;
        impl Hypothesize for Recorder {
            type Error = Infallible;
            fn hypotheses(&self, request: &Request<'_>) -> Result<Vec<String>, Self::Error> {
                Ok(vec![request.context.unwrap_or("none").to_owned()])
            }
        }
        let set = EvalSet::parse(SET).expect("the fixture parses");
        let mut seen = Vec::new();
        for record in set.records() {
            let hypotheses = Recorder
                .hypotheses(&Request {
                    pinyin: &record.pinyin,
                    context: record.context.as_deref(),
                    top_k: k(1),
                })
                .expect("the recorder cannot fail");
            seen.push(hypotheses[0].clone());
        }
        assert_eq!(seen, ["none", "none", "我要去", "none", "今天的"]);
    }
}
