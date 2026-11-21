//! One grounded request per seed term, over an OpenAI-compatible API.
//!
//! The prompt is the whole safety property of this stage. The decision record's
//! grounding rule says the model must never be asked to use a term out of its own
//! memory, so the term's explanation is embedded verbatim and the instruction is
//! explicit that the explanation wins where the two disagree -- Luna's Chinese
//! world knowledge is exactly what cannot be trusted here.
//!
//! Everything else is the same contract `ime-g2p` established for its annotator,
//! for the same reasons. `reasoning_effort` is pinned to `high`, because that is
//! the setting the endpoint was measured on. The reply must be a JSON array of
//! exactly the requested length, so a short reply is a refusal rather than a
//! quietly thinner batch. A reply that will not parse is retried once and then
//! recorded, never repaired: a patched-up example is indistinguishable from a
//! real one downstream, and these samples are going into training data.

use crate::error::{Error, Result};
use crate::seed::Seed;
use crate::source::UNGROUNDED;
use askama::Template;
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_openai::types::chat::{
    ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs, ReasoningEffort,
};
use ime_corpus::filter::{MAX_CHARACTERS, MIN_CHARACTERS, MIN_HAN_RATIO};
use serde::{Deserialize, Serialize};
use tracing::debug;

/// How many usage examples one term is asked for when nothing else is asked.
pub const DEFAULT_PER_TERM: usize = 5;

/// How many requests are allowed in flight at once when nothing else is asked.
pub const DEFAULT_CONCURRENCY: usize = 32;

/// Reasoning effort requested from an OpenAI-compatible chat endpoint.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Effort {
    /// Omit `reasoning_effort` from the request payload (for endpoints that reject it).
    None,
    /// Low reasoning budget.
    Low,
    /// Medium reasoning budget.
    Medium,
    /// High reasoning budget (the default Luna was measured under).
    #[default]
    High,
}

impl Effort {
    fn into_openai(self) -> Option<ReasoningEffort> {
        match self {
            Self::None => None,
            Self::Low => Some(ReasoningEffort::Low),
            Self::Medium => Some(ReasoningEffort::Medium),
            Self::High => Some(ReasoningEffort::High),
        }
    }
}

/// The grounded synthesis prompt, kept in a file so it can be diffed and reviewed.
#[derive(Template)]
#[template(path = "synth_prompt.txt", ext = "txt")]
pub struct Prompt<'a> {
    /// The term to use, normalised the way the corpus normalises text.
    pub term: &'a str,
    /// The explanation that grounds it.
    pub explanation: &'a str,
    /// How many examples the reply must carry.
    pub count: usize,
    /// The shortest sentence the corpus filter keeps.
    pub minimum: usize,
    /// The longest sentence the corpus filter keeps.
    pub maximum: usize,
    /// How many Han characters the filter's script ratio buys per punctuation mark.
    pub han_per_mark: usize,
}

impl<'a> Prompt<'a> {
    /// The grounded prompt for one seed, asking for `count` examples.
    #[must_use]
    pub fn new(seed: &'a Seed, count: usize) -> Self {
        Self {
            term: &seed.term,
            explanation: &seed.explanation,
            count,
            minimum: MIN_CHARACTERS,
            maximum: MAX_CHARACTERS,
            han_per_mark: han_per_mark(),
        }
    }
}

/// Prompt for an ungrounded lexicon entry with an explicit `ABSTAIN` escape.
#[derive(Template)]
#[template(path = "ungrounded_prompt.txt", ext = "txt")]
pub struct UngroundedPrompt<'a> {
    /// The lexicon term to use or abstain on.
    pub term: &'a str,
    /// How many examples a non-abstaining reply must carry.
    pub count: usize,
    /// The shortest sentence the corpus filter keeps.
    pub minimum: usize,
    /// The longest sentence the corpus filter keeps.
    pub maximum: usize,
    /// How many Han characters the filter's script ratio buys per punctuation mark.
    pub han_per_mark: usize,
}

/// Prompt for contrastive context disambiguation between `term` and homophone `foil`.
#[derive(Template)]
#[template(path = "contrastive_prompt.txt", ext = "txt")]
pub struct ContrastivePrompt<'a> {
    /// Target word that must appear in `text`.
    pub term: &'a str,
    /// Homophone competitor that `context` must rule out.
    pub foil: &'a str,
    /// Optional disambiguation hint.
    pub explanation: &'a str,
    /// How many examples a non-abstaining reply must carry.
    pub count: usize,
    /// The shortest sentence the corpus filter keeps.
    pub minimum: usize,
    /// The longest sentence the corpus filter keeps.
    pub maximum: usize,
    /// How many Han characters the filter's script ratio buys per punctuation mark.
    pub han_per_mark: usize,
}

/// Render whichever prompt template `seed` calls for.
///
/// # Errors
///
/// If the compiled template fails to render.
pub fn render_seed_prompt(seed: &Seed, count: usize) -> Result<String> {
    let rendered = if let Some(foil) = &seed.foil {
        ContrastivePrompt {
            term: &seed.term,
            foil,
            explanation: &seed.explanation,
            count,
            minimum: MIN_CHARACTERS,
            maximum: MAX_CHARACTERS,
            han_per_mark: han_per_mark(),
        }
        .render()
    } else if seed.grounding == UNGROUNDED {
        UngroundedPrompt {
            term: &seed.term,
            count,
            minimum: MIN_CHARACTERS,
            maximum: MAX_CHARACTERS,
            han_per_mark: han_per_mark(),
        }
        .render()
    } else {
        Prompt::new(seed, count).render()
    };
    rendered.map_err(|error| {
        Error::Invariant(format!("the synthesis prompt would not render: {error}"))
    })
}

/// How many Han characters a sentence must carry for each non-Han character.
///
/// The corpus filter states its script rule as a ratio, which is not something a
/// person -- or a model -- writes to. Turned around, it is a budget: at nine
/// tenths Han, every comma has to be paid for with nine characters. Saying it
/// that way in the prompt is what stops the model returning six-character
/// sentences with a comma in them, every one of which the filter would drop.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the ratio is a fixed fraction below one, so its budget is a small positive integer"
)]
pub fn han_per_mark() -> usize {
    (MIN_HAN_RATIO / (1.0 - MIN_HAN_RATIO)).ceil() as usize
}

/// One generated usage example, exactly as the reply carries it.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct Example {
    /// The turn before it, empty when the speaker opened the exchange.
    pub context: String,
    /// The sentence the term appears in.
    pub text: String,
}

/// Parsed model response: either the requested examples or an explicit `ABSTAIN`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Reply {
    /// Exactly `per_term` parsed turn pairs.
    Examples(Vec<Example>),
    /// Model opted out because it is unsure of the term or pair.
    Abstained,
}

/// Why one reply could not be turned into examples.
#[derive(Debug, thiserror::Error)]
enum ReplyError {
    #[error("reply is not the expected JSON array: {0:?}")]
    NotJson(String),
    #[error("model answered ABSTAIN on a grounded term whose explanation was given")]
    UnexpectedAbstain,
    #[error("got {got} examples where {expected} were asked for")]
    WrongLength {
        /// How many the reply carried.
        got: usize,
        /// How many were asked for.
        expected: usize,
    },
    #[error("the endpoint returned a message with no content")]
    NoContent,
}

/// Strip a ```` ``` ````-fenced block, if the whole reply is one.
fn unfence(reply: &str) -> Option<&str> {
    let rest = reply.strip_prefix("```")?;
    let newline = rest.find('\n')?;
    if !rest[..newline].chars().all(|c| c.is_ascii_lowercase()) {
        return None;
    }
    rest[newline + 1..]
        .strip_suffix("```")
        .and_then(|body| body.strip_suffix('\n'))
}

/// Parse a reply into `expected` examples or `Reply::Abstained` when `allow_abstain` is set.
///
/// # Errors
///
/// If the reply is neither `ABSTAIN` (when allowed) nor a JSON array of `expected` examples.
pub fn parse_reply(
    content: &str,
    expected: usize,
    allow_abstain: bool,
) -> std::result::Result<Reply, String> {
    parse_reply_inner(content, expected, allow_abstain).map_err(|error| error.to_string())
}

fn parse_reply_inner(
    content: &str,
    expected: usize,
    allow_abstain: bool,
) -> std::result::Result<Reply, ReplyError> {
    let trimmed = content.trim();
    let body = unfence(trimmed).unwrap_or(trimmed).trim();
    if body == "ABSTAIN" {
        if allow_abstain {
            return Ok(Reply::Abstained);
        }
        return Err(ReplyError::UnexpectedAbstain);
    }
    let examples: Vec<Example> = serde_json::from_str(body)
        .map_err(|_| ReplyError::NotJson(body.chars().take(200).collect()))?;
    if examples.len() != expected {
        return Err(ReplyError::WrongLength {
            got: examples.len(),
            expected,
        });
    }
    Ok(Reply::Examples(examples))
}

/// Parse a reply into `expected` examples, disallowing `ABSTAIN`.
#[cfg(test)]
fn parse_examples(content: &str, expected: usize) -> std::result::Result<Vec<Example>, ReplyError> {
    match parse_reply_inner(content, expected, false)? {
        Reply::Examples(examples) => Ok(examples),
        Reply::Abstained => Err(ReplyError::UnexpectedAbstain),
    }
}

/// Generates usage examples for one seed term at a time.
#[derive(Debug)]
pub struct Synthesizer {
    client: Client<OpenAIConfig>,
    model: String,
    per_term: usize,
    retries: usize,
    effort: Effort,
}

impl Synthesizer {
    /// Build a client from settings read out of the `MLIME_LLM_*` environment.
    #[must_use]
    pub fn new(settings: &ime_g2p::llm::LlmSettings, per_term: usize) -> Self {
        let config = OpenAIConfig::new()
            .with_api_base(settings.base_url.clone())
            .with_api_key(settings.api_key.clone());
        Self {
            client: Client::with_config(config),
            model: settings.model.clone(),
            per_term: per_term.max(1),
            retries: 1,
            effort: Effort::High,
        }
    }

    /// Override the reasoning effort sent with each completion request.
    #[must_use]
    pub fn with_effort(mut self, effort: Effort) -> Self {
        self.effort = effort;
        self
    }

    /// Which model the requests name, for the run summary.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    /// How many examples each request asks for.
    #[must_use]
    pub fn per_term(&self) -> usize {
        self.per_term
    }

    /// Render the synthesis prompt for `seed`.
    ///
    /// # Errors
    ///
    /// If the compiled template refuses the seed.
    pub fn prompt(&self, seed: &Seed) -> Result<String> {
        render_seed_prompt(seed, self.per_term)
    }

    /// Generate one seed's reply (`Examples` or `Abstained`), retrying once on parse/API failure.
    ///
    /// # Errors
    ///
    /// Diagnostic refusal reason if all attempts fail.
    pub async fn reply(&self, seed: &Seed) -> std::result::Result<Reply, String> {
        let mut last = String::new();
        let allow_abstain = seed.allows_abstain();
        for attempt in 0..=self.retries {
            match self.complete(seed).await {
                Ok(content) => match parse_reply_inner(&content, self.per_term, allow_abstain) {
                    Ok(reply) => return Ok(reply),
                    Err(error) => last = format!("ValueError: {error}"),
                },
                Err(error) => last = format!("OpenAIError: {error}"),
            }
            debug!(term = seed.term, attempt, reason = %last, "synthesis retry");
        }
        Err(last)
    }

    /// Generate one grounded term's examples, retrying once before recording a refusal.
    ///
    /// # Errors
    ///
    /// The refusal reason, or if the model abstained where examples were required.
    pub async fn examples(&self, seed: &Seed) -> std::result::Result<Vec<Example>, String> {
        match self.reply(seed).await? {
            Reply::Examples(examples) => Ok(examples),
            Reply::Abstained => Err("ValueError: unexpected ABSTAIN".to_owned()),
        }
    }

    /// One chat completion, at the configured reasoning effort.
    async fn complete(
        &self,
        seed: &Seed,
    ) -> std::result::Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let message = ChatCompletionRequestUserMessageArgs::default()
            .content(self.prompt(seed)?)
            .build()?;
        let mut builder = CreateChatCompletionRequestArgs::default();
        builder
            .model(self.model.clone())
            .messages(vec![message.into()]);
        if let Some(effort) = self.effort.into_openai() {
            builder.reasoning_effort(effort);
        }
        let request = builder.build()?;
        let response = self.client.chat().create(request).await?;
        let content = response
            .choices
            .into_iter()
            .next()
            .and_then(|choice| choice.message.content);
        content.ok_or_else(|| {
            Box::new(ReplyError::NoContent) as Box<dyn std::error::Error + Send + Sync>
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::WIKI_SLANG;

    fn seed() -> Seed {
        Seed::grounded(
            "爷青结".to_owned(),
            "爷的青春结束了。感慨自己的青春不再。".to_owned(),
            WIKI_SLANG,
            WIKI_SLANG,
        )
    }

    #[test]
    fn the_prompt_renders_byte_for_byte_against_its_fixture() {
        let rendered = Prompt::new(&seed(), DEFAULT_PER_TERM)
            .render()
            .expect("the template renders");
        assert_eq!(rendered, include_str!("../tests/synth_prompt_expected.txt"));
    }

    #[test]
    fn the_prompt_carries_the_explanation_verbatim_so_the_generation_is_grounded() {
        let seed = seed();
        let rendered = Prompt::new(&seed, DEFAULT_PER_TERM)
            .render()
            .expect("the template renders");
        assert!(rendered.contains(&seed.explanation), "{rendered}");
        assert!(rendered.contains(&seed.term));
    }

    #[test]
    fn a_well_formed_reply_becomes_examples() {
        let examples = parse_examples(
            r#"[{"context": "你看完结局了吗", "text": "看完了，爷青结"}]"#,
            1,
        )
        .expect("the reply is well formed");
        assert_eq!(examples[0].context, "你看完结局了吗");
        assert_eq!(examples[0].text, "看完了，爷青结");
    }

    #[test]
    fn a_fenced_reply_is_unwrapped_before_it_is_parsed() {
        let examples = parse_examples(
            "```json\n[{\"context\": \"\", \"text\": \"爷青结\"}]\n```",
            1,
        )
        .expect("the fence is stripped");
        assert_eq!(examples[0].context, "");
    }

    #[test]
    fn a_reply_that_is_not_json_is_refused_with_its_first_characters() {
        let error = parse_examples("I cannot help with that", 1).expect_err("not JSON");
        assert_eq!(
            error.to_string(),
            "reply is not the expected JSON array: \"I cannot help with that\""
        );
    }

    #[test]
    fn a_short_reply_is_a_refusal_rather_than_a_thinner_batch() {
        let error = parse_examples(r#"[{"context": "", "text": "爷青结"}]"#, 5)
            .expect_err("one example where five were asked for");
        assert_eq!(error.to_string(), "got 1 examples where 5 were asked for");
    }

    #[test]
    fn an_unfenced_reply_is_left_alone() {
        assert_eq!(unfence("[]"), None);
        assert_eq!(unfence("```\n[]\n```"), Some("[]"));
        assert_eq!(unfence("```JSON\n[]\n```"), None);
    }

    #[test]
    fn abstain_is_accepted_on_ungrounded_and_contrastive_seeds_and_refused_on_grounded_ones() {
        assert_eq!(
            parse_reply("  ABSTAIN \n", 5, true).expect("ungrounded seeds may abstain"),
            Reply::Abstained
        );
        assert!(parse_reply("ABSTAIN", 5, false).is_err());

        let ungrounded = Seed::ungrounded("尊嘟假嘟".to_owned(), WIKI_SLANG);
        let rendered_ungrounded = render_seed_prompt(&ungrounded, 3).expect("renders");
        assert!(rendered_ungrounded.contains("ABSTAIN"));
        assert!(rendered_ungrounded.contains("尊嘟假嘟"));

        let contrastive = Seed::contrastive(
            "公式".to_owned(),
            "公事".to_owned(),
            "数学表达式与公务的区别".to_owned(),
        );
        let rendered_contrastive = render_seed_prompt(&contrastive, 3).expect("renders");
        assert!(rendered_contrastive.contains("公式"));
        assert!(rendered_contrastive.contains("公事"));
        assert!(rendered_contrastive.contains("数学表达式与公务的区别"));
    }
}
