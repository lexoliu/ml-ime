//! The LLM annotator: one prompted request per sentence, over an OpenAI-compatible API.
//!
//! This is the second, independent opinion that makes the agreement filter worth
//! anything, so it must not share g2pW's failure modes. Three things make that
//! hold:
//!
//! * `reasoning_effort` is pinned to `high`. Measured on a polyphone probe, the
//!   same model scores 5/6 at its default effort and 10/10 at high, and the items
//!   that flip are precisely the hard ones (得 dé/děi, 朝 cháo/zhāo) -- the ones
//!   this whole stage exists to catch.
//! * The prompt hands the model an *enumerated* character list and demands an
//!   array of the same length, so a reply can be checked against the sentence
//!   instead of being aligned by guesswork.
//! * A reply that will not parse, or comes back the wrong length, is retried once
//!   and then recorded as a refusal. It is never patched up, because a repaired
//!   label is indistinguishable from a correct one downstream.

use crate::error::{Error, Result};
use crate::outcome::{Annotator, Outcome, Reading, Refusal};
use crate::text::han_characters;
use askama::Template;
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_openai::types::chat::{
    ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs, ReasoningEffort,
};
use serde::Deserialize;
use tokio::sync::Semaphore;
use tracing::debug;

/// How many requests are allowed in flight at once when nothing else is asked for.
pub const DEFAULT_CONCURRENCY: usize = 32;

/// Connection details for the OpenAI-compatible annotation endpoint.
///
/// The API key is never logged and never rendered by [`Debug`], because the
/// settings travel through error paths that do get logged.
#[derive(Clone)]
pub struct LlmSettings {
    /// Root of the OpenAI-compatible API, ending in `/v1`.
    pub base_url: String,
    /// Bearer token for the endpoint.
    pub api_key: String,
    /// Model id to annotate with.
    pub model: String,
}

impl std::fmt::Debug for LlmSettings {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LlmSettings")
            .field("base_url", &self.base_url)
            .field("api_key", &"<redacted>")
            .field("model", &self.model)
            .finish()
    }
}

impl LlmSettings {
    /// Read the settings from `MLIME_LLM_*`, loading a repository `.env` first.
    ///
    /// A missing setting is refused here rather than surfacing as a 401 from the
    /// endpoint an hour into a run.
    ///
    /// # Errors
    ///
    /// If any of the three variables is unset or empty.
    pub fn load() -> Result<Self> {
        let _ = dotenvy::dotenv();
        Ok(Self {
            base_url: variable("MLIME_LLM_BASE_URL")?,
            api_key: variable("MLIME_LLM_API_KEY")?,
            model: variable("MLIME_LLM_MODEL")?,
        })
    }
}

fn variable(name: &'static str) -> Result<String> {
    match std::env::var(name) {
        Ok(value) if !value.trim().is_empty() => Ok(value),
        _ => Err(Error::Unconfigured(name)),
    }
}

/// The annotation prompt, kept in a file so it can be diffed and reviewed.
#[derive(Template)]
#[template(path = "g2p_prompt.txt", ext = "txt")]
struct Prompt<'a> {
    sentence: &'a str,
    characters: &'a [char],
}

/// The JSON body the prompt asks for.
#[derive(Debug, Deserialize)]
struct LlmReadings {
    readings: Vec<String>,
}

/// Why one reply could not be turned into a reading.
///
/// The variants carry the same wording as the Python `ValueError`s they replace,
/// so a refusal recorded by either implementation reads the same in the shards.
#[derive(Debug, thiserror::Error)]
enum ReplyError {
    #[error("reply is not the expected JSON object: {0:?}")]
    NotJson(String),
    #[error("got {got} readings for {expected} characters: {readings:?}")]
    WrongLength {
        got: usize,
        expected: usize,
        readings: Vec<String>,
    },
    #[error("{syllable:?} is not a tone-numbered syllable (for {character:?})")]
    NotASyllable { syllable: String, character: char },
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

/// A tone-numbered syllable in typing spelling: letters only, then one tone digit.
fn is_syllable(candidate: &str) -> bool {
    let Some(tone) = candidate.chars().next_back() else {
        return false;
    };
    if !matches!(tone, '1'..='5') {
        return false;
    }
    let letters = &candidate[..candidate.len() - 1];
    !letters.is_empty() && letters.chars().all(|c| c.is_ascii_lowercase())
}

/// Parse a reply into one syllable per expected character, refusing any deviation.
fn parse_readings(content: &str, expected: &[char]) -> std::result::Result<Reading, ReplyError> {
    let trimmed = content.trim();
    let body = unfence(trimmed).unwrap_or(trimmed);
    let payload: LlmReadings = serde_json::from_str(body)
        .map_err(|_| ReplyError::NotJson(body.chars().take(200).collect()))?;
    if payload.readings.len() != expected.len() {
        return Err(ReplyError::WrongLength {
            got: payload.readings.len(),
            expected: expected.len(),
            readings: payload.readings,
        });
    }
    for (character, syllable) in expected.iter().zip(&payload.readings) {
        if !is_syllable(syllable) {
            return Err(ReplyError::NotASyllable {
                syllable: syllable.clone(),
                character: *character,
            });
        }
    }
    Ok(Reading::new(payload.readings))
}

/// Annotates sentences through an OpenAI-compatible chat completion endpoint.
#[derive(Debug)]
pub struct LlmAnnotator {
    client: Client<OpenAIConfig>,
    model: String,
    retries: usize,
    effort: ReasoningEffort,
    gate: Semaphore,
}

impl LlmAnnotator {
    /// Build a client from settings read out of the `MLIME_LLM_*` environment.
    #[must_use]
    pub fn new(settings: &LlmSettings, concurrency: usize) -> Self {
        let config = OpenAIConfig::new()
            .with_api_base(settings.base_url.clone())
            .with_api_key(settings.api_key.clone());
        Self {
            client: Client::with_config(config),
            model: settings.model.clone(),
            retries: 1,
            effort: ReasoningEffort::High,
            gate: Semaphore::new(concurrency.max(1)),
        }
    }

    /// Render the annotation prompt for `text`.
    ///
    /// # Errors
    ///
    /// If the compiled template refuses the sentence, which it does not.
    pub fn prompt(&self, text: &str) -> Result<String> {
        let characters = han_characters(text);
        Prompt {
            sentence: text,
            characters: &characters,
        }
        .render()
        .map_err(|error| {
            Error::Invariant(format!("the annotation prompt would not render: {error}"))
        })
    }

    /// Annotate one sentence, retrying once before giving up on it.
    async fn one(&self, text: &str) -> Outcome {
        let characters = han_characters(text);
        if characters.is_empty() {
            return Err(Refusal::new("no Han characters to read"));
        }
        let mut last = String::new();
        for attempt in 0..=self.retries {
            {
                let Ok(_permit) = self.gate.acquire().await else {
                    return Err(Refusal::new("the concurrency gate was closed mid-run"));
                };
                match self.complete(text).await {
                    Ok(content) => match parse_readings(&content, &characters) {
                        Ok(reading) => return Ok(reading),
                        Err(error) => last = format!("ValueError: {error}"),
                    },
                    Err(error) => last = format!("OpenAIError: {error}"),
                }
            }
            debug!(text, attempt, reason = %last, "llm annotation retry");
        }
        Err(Refusal::new(last))
    }

    /// One chat completion, at the pinned reasoning effort.
    async fn complete(
        &self,
        text: &str,
    ) -> std::result::Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let message = ChatCompletionRequestUserMessageArgs::default()
            .content(self.prompt(text)?)
            .build()?;
        let request = CreateChatCompletionRequestArgs::default()
            .model(self.model.clone())
            .messages(vec![message.into()])
            .reasoning_effort(self.effort.clone())
            .build()?;
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

impl Annotator for LlmAnnotator {
    fn name(&self) -> &'static str {
        "llm"
    }

    async fn annotate(&self, texts: &[String]) -> Vec<Outcome> {
        futures::future::join_all(texts.iter().map(|text| self.one(text))).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_prompt_matches_the_python_jinja_rendering_byte_for_byte() {
        let sentence = "他还了钱，我得到了绿色东西";
        let characters = han_characters(sentence);
        let rendered = Prompt {
            sentence,
            characters: &characters,
        }
        .render()
        .expect("the template renders");
        assert_eq!(rendered, include_str!("../tests/g2p_prompt_expected.txt"));
    }

    #[test]
    fn a_well_formed_reply_becomes_a_reading() {
        let reading = parse_readings(r#"{"readings": ["zhong1", "guo2"]}"#, &['中', '国'])
            .expect("the reply is well formed");
        assert_eq!(reading.syllables, ["zhong1", "guo2"]);
    }

    #[test]
    fn a_fenced_reply_is_unwrapped_before_it_is_parsed() {
        let reading = parse_readings("```json\n{\"readings\": [\"de5\"]}\n```", &['的'])
            .expect("the fence is stripped");
        assert_eq!(reading.syllables, ["de5"]);
    }

    #[test]
    fn a_reply_that_is_not_json_is_refused_with_its_first_characters() {
        let error = parse_readings("I cannot do that", &['中']).expect_err("not JSON");
        assert_eq!(
            error.to_string(),
            "reply is not the expected JSON object: \"I cannot do that\""
        );
    }

    #[test]
    fn a_reply_of_the_wrong_length_names_both_counts() {
        let error = parse_readings(r#"{"readings": ["zhong1"]}"#, &['中', '国'])
            .expect_err("one reading for two characters");
        assert_eq!(
            error.to_string(),
            "got 1 readings for 2 characters: [\"zhong1\"]"
        );
    }

    #[test]
    fn a_reading_that_is_not_a_tone_numbered_syllable_is_refused() {
        let error =
            parse_readings(r#"{"readings": ["zhōng"]}"#, &['中']).expect_err("no tone digit");
        assert_eq!(
            error.to_string(),
            "\"zhōng\" is not a tone-numbered syllable (for '中')"
        );
        assert!(parse_readings(r#"{"readings": ["zhong6"]}"#, &['中']).is_err());
        assert!(parse_readings(r#"{"readings": ["1"]}"#, &['中']).is_err());
        assert!(parse_readings(r#"{"readings": [""]}"#, &['中']).is_err());
    }

    #[test]
    fn extra_fields_in_the_reply_are_ignored_the_way_pydantic_ignores_them() {
        let reading = parse_readings(r#"{"readings": ["de5"], "note": "hi"}"#, &['的'])
            .expect("extra keys do not matter");
        assert_eq!(reading.syllables, ["de5"]);
    }

    #[test]
    fn an_unfenced_reply_is_left_alone() {
        assert_eq!(unfence("{\"readings\": []}"), None);
        assert_eq!(unfence("```\n{}\n```"), Some("{}"));
        assert_eq!(unfence("```JSON\n{}\n```"), None);
    }

    #[test]
    fn the_settings_never_print_the_key() {
        let settings = LlmSettings {
            base_url: "http://localhost:8317/v1".to_owned(),
            api_key: "sk-secret".to_owned(),
            model: "gpt-5.6-luna".to_owned(),
        };
        let rendered = format!("{settings:?}");
        assert!(!rendered.contains("sk-secret"), "{rendered}");
        assert!(rendered.contains("<redacted>"));
    }
}
