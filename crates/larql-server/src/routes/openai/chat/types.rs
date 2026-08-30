//! Request/response wire types for `POST /v1/chat/completions`.
//!
//! Serde shapes mirror the OpenAI Chat Completions API exactly so
//! existing SDKs deserialize responses without adapters.

use serde::{Deserialize, Serialize};

use crate::routes::openai::util::StopSpec;

#[derive(Deserialize, Clone, Debug)]
pub struct ChatMessage {
    pub role: String,
    /// Free-text content. Optional because assistant messages that
    /// emitted tool_calls send `content: null` per OpenAI's wire shape.
    #[serde(default)]
    pub content: Option<String>,
    /// Echoed back on `role: "assistant"` messages in multi-turn
    /// conversations so the model can see its own prior tool dispatch.
    #[serde(default)]
    pub tool_calls: Option<serde_json::Value>,
    /// Set on `role: "tool"` messages — the call id this result
    /// corresponds to.
    #[serde(default)]
    pub tool_call_id: Option<String>,
    /// Optional `function.name` echoed on tool messages by some clients.
    /// Treated as informational; we already get the name from the
    /// matching `tool_calls[i].function.name` when available.
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Deserialize)]
pub struct ChatCompletionsRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// Newer name for `max_tokens` used by current OpenAI SDKs
    /// (`max_tokens` is deprecated upstream). When both are set,
    /// `max_completion_tokens` wins.
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Nucleus (top-p) filter applied after temperature scaling. Only
    /// honoured when `temperature > 0`; for greedy decoding it's a no-op.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Streaming via SSE — emits one `chat.completion.chunk` per token,
    /// terminated by `data: [DONE]\n\n`.
    #[serde(default)]
    pub stream: Option<bool>,
    /// Number of completions per prompt — only n=1 supported.
    #[serde(default)]
    pub n: Option<usize>,
    /// Stop strings — first match halts generation.
    #[serde(default)]
    pub stop: Option<StopSpec>,
    /// Top-k log-probs — request accepted, response field always null.
    #[serde(default)]
    pub logprobs: Option<bool>,
    /// Newer log-probs field used by recent SDKs — same handling as `logprobs`.
    #[serde(default)]
    pub top_logprobs: Option<usize>,
    /// Tool definitions — slice 4 (N0.6 constrained decoding); 400 if non-empty.
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    /// Tool choice — same as `tools` (slice 4).
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    /// Response format (`{type: "json_object" | "json_schema", ...}`) —
    /// slice 4. Returns 400 for any non-text response_format.
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,
    /// Seed for reproducible sampling. Same seed + same temperature +
    /// same prompt produces the same tokens. No-op for greedy mode
    /// (greedy is already deterministic on argmax).
    #[serde(default)]
    pub seed: Option<u64>,
    /// End-user id — logged via tracing if set.
    #[serde(default)]
    pub user: Option<String>,
    /// Frequency / presence penalties — accepted for shape compat;
    /// the sampler does not yet apply repetition penalties (F19).
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct ChatChoiceMessage {
    pub role: &'static str,
    /// Always present, but `null` when the assistant emitted tool_calls
    /// rather than free text. Serialised as `content: null` in that case
    /// (OpenAI's contract).
    pub content: Option<String>,
    /// One or more tool calls produced by constrained decoding when
    /// `tools` was on the request. Omitted entirely for plain text
    /// completions so non-tools responses stay shape-clean.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// OpenAI's tool-call shape on the response side: `id`, `type`,
/// `function: {name, arguments}`. `arguments` is JSON-stringified.
#[derive(Debug, Serialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: ToolCallFunction,
}

#[derive(Debug, Serialize)]
pub struct ToolCallFunction {
    pub name: String,
    /// JSON-encoded string, not a nested object — preserves the wire
    /// shape SDKs expect.
    pub arguments: String,
}

#[derive(Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatChoiceMessage,
    pub finish_reason: &'static str,
    /// Populated when the request set `logprobs: true`. `None`
    /// (serialised as `null`) otherwise — the OpenAI default.
    pub logprobs: Option<ChatLogprobs>,
}

/// `choices[i].logprobs` payload for chat completions. Mirrors
/// OpenAI's `{content: [{token, logprob, bytes, top_logprobs}]}`.
#[derive(Serialize)]
pub struct ChatLogprobs {
    pub content: Vec<TokenLogprob>,
}

/// One per-token entry in a logprobs payload (chat or completions —
/// the chat shape is identical for the inner item).
///
/// `top_logprobs` is an empty array until the inference layer exposes
/// per-step top-K alternatives (follow-up). Until then we still emit
/// the picked-token entry so client parsers don't break on the field.
#[derive(Serialize)]
pub struct TokenLogprob {
    pub token: String,
    pub logprob: f64,
    pub bytes: Vec<u8>,
    pub top_logprobs: Vec<TokenLogprob>,
}

#[derive(Serialize)]
pub struct ChatUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Serialize)]
pub struct ChatCompletionsResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: ChatUsage,
}
