//! Tool / `response_format` resolution for chat completions —
//! synthesising the constrained-decoding schema, parsing constrained
//! output back into `tool_calls`, and building the vocab mask.

use crate::error::ServerError;
use crate::routes::openai::prompt::ASSISTANT_ROLE;
use crate::routes::openai::schema::{self, ObjectSchema, Schema};
use crate::routes::openai::util::new_id_suffix;

use super::types::{ChatChoiceMessage, ChatCompletionsRequest, ToolCall, ToolCallFunction};

// ── chat-only request validation helper ─────────────────────────────────────

pub(super) fn is_empty_json_array(v: &serde_json::Value) -> bool {
    v.as_array().map(|a| a.is_empty()).unwrap_or(false)
}

/// Resolve `tools` + `tool_choice` into a synthesised `Schema`.
///
/// Returns `Ok(None)` when no tools are bound (or `tool_choice="none"`)
/// so the caller falls through to `response_format` /unconstrained.
/// Returns `Ok(Some(schema))` with the discriminated-union shape over
/// each function (one branch per tool); the chat handler then post-
/// parses the JSON output into `tool_calls`.
pub(super) fn resolve_tools(req: &ChatCompletionsRequest) -> Result<Option<Schema>, ServerError> {
    use crate::routes::openai::schema::{resolve_tool_choice, synth_tools_schema};

    let tools_present = req
        .tools
        .as_ref()
        .is_some_and(|v| !v.is_null() && !is_empty_json_array(v));

    let tool_names: Vec<String> = req
        .tools
        .as_ref()
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|t| {
                    t.get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str())
                        .map(|s| s.to_string())
                })
                .collect()
        })
        .unwrap_or_default();

    let mode = resolve_tool_choice(tools_present, req.tool_choice.as_ref(), &tool_names)
        .map_err(ServerError::BadRequest)?;

    if !tools_present || matches!(mode, schema::ToolMode::None) {
        return Ok(None);
    }

    let tools = req
        .tools
        .as_ref()
        .expect("tools_present checked above")
        .clone();
    let result = synth_tools_schema(&tools, &mode).map_err(ServerError::BadRequest)?;
    Ok(result.map(|(schema, _names)| schema))
}

/// Parse a constrained-decoder output back into a `ChatChoiceMessage`
/// with `tool_calls` populated.
///
/// Constrained decoding guarantees a well-formed JSON object as the
/// model's full emission, so the only legit input variability is
/// surrounding whitespace. Earlier versions of this function tried to
/// be clever with `find('{')` + `rfind('}')` substring slicing — but
/// that mis-handles model-output drift (trailing junk, multiple JSON
/// objects, markdown-wrapped output) by silently picking the wrong
/// slice and surfacing the failure as a 500 internal error. The
/// straight-line `serde_json::from_str` here gives a clean diagnostic
/// (`invalid JSON: …`) at the call site, which then surfaces as a
/// 400 invalid_request_error so the client can see the failure mode
/// and either retry, reduce tool complexity, or fall back.
pub(in crate::routes::openai) fn build_tool_call_message(
    text: &str,
) -> Result<ChatChoiceMessage, String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err("tool output was empty".to_string());
    }
    let parsed: serde_json::Value =
        serde_json::from_str(trimmed).map_err(|e| format!("invalid JSON: {e}"))?;
    if !parsed.is_object() {
        return Err(format!(
            "tool output must be a JSON object, got {} value",
            json_value_kind(&parsed)
        ));
    }
    let name = parsed
        .get("name")
        .and_then(|n| n.as_str())
        .ok_or_else(|| "tool output missing `name`".to_string())?
        .to_string();
    let arguments_value = parsed
        .get("arguments")
        .ok_or_else(|| "tool output missing `arguments`".to_string())?;
    // OpenAI sends arguments as a JSON-stringified object — reserialise
    // to canonical compact form so SDKs `json.loads` cleanly.
    let arguments = serde_json::to_string(arguments_value)
        .map_err(|e| format!("failed to serialise arguments: {e}"))?;
    Ok(ChatChoiceMessage {
        role: ASSISTANT_ROLE,
        content: None,
        tool_calls: Some(vec![ToolCall {
            id: format!("call_{}", new_id_suffix()),
            kind: "function",
            function: ToolCallFunction { name, arguments },
        }]),
    })
}

fn json_value_kind(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

/// Map an OpenAI `response_format` field to the `Schema` the FSM
/// should enforce. `None` (or `{type: "text"}`) means "no constrained
/// decoding" — fall through to the sampling path.
///
/// `json_object` compiles to `Schema::Object(any)`. `json_schema`
/// reaches into `json_schema.schema` and runs the JSON Schema parser
/// with `strict: true` when the `strict` field is set (matching
/// OpenAI's structured-outputs contract).
pub(in crate::routes::openai) fn schema_for_response_format(
    rf: Option<&serde_json::Value>,
) -> Result<Option<Schema>, ServerError> {
    let Some(rf) = rf else {
        return Ok(None);
    };
    let kind = rf.get("type").and_then(|t| t.as_str()).unwrap_or("text");
    match kind {
        "text" => Ok(None),
        "json_object" => Ok(Some(Schema::object(ObjectSchema::any()))),
        "json_schema" => {
            let js = rf.get("json_schema").ok_or_else(|| {
                ServerError::BadRequest(
                    "response_format.type=json_schema requires a json_schema field".into(),
                )
            })?;
            let schema_value = js.get("schema").ok_or_else(|| {
                ServerError::BadRequest("response_format.json_schema.schema is required".into())
            })?;
            // OpenAI's `strict: true` flips the additionalProperties default
            // to false. Default is `false` here so non-strict callers can
            // still send extra keys.
            let strict = js.get("strict").and_then(|v| v.as_bool()).unwrap_or(false);
            let opts = schema::ParseOptions { strict };
            let parsed = schema::parse_schema_with(schema_value, opts)
                .map_err(|e| ServerError::BadRequest(format!("invalid json_schema: {e}")))?;
            Ok(Some(parsed))
        }
        other => Err(ServerError::BadRequest(format!(
            "response_format.type {other:?} is not supported (expected \
             \"text\" | \"json_object\" | \"json_schema\")"
        ))),
    }
}

/// Resolve common end-of-turn token ids for the loaded model. The
/// constrained-mask uses these to gate EOS — the model can't truncate
/// while the FSM is mid-structure, but once the FSM is complete the
/// EOS tokens become legal again.
///
/// Looks up a small set of well-known special markers
/// (`<end_of_turn>`, `<|im_end|>`, `<eos>`, `</s>`, etc.) via
/// `tokenizer.token_to_id` and ignores any that aren't present in the
/// vocab.
fn resolve_eos_token_ids(
    tokenizer: &larql_inference::tokenizers::Tokenizer,
) -> std::collections::HashSet<u32> {
    let mut ids = std::collections::HashSet::new();
    for tok in [
        "<end_of_turn>",
        "<|end_of_turn|>",
        "<|im_end|>",
        "<|eot_id|>",
        "<|eom_id|>",
        "<|endoftext|>",
        "<|end_of_text|>",
        "<eos>",
        "</s>",
    ] {
        if let Some(id) = tokenizer.token_to_id(tok) {
            ids.insert(id);
        }
    }
    ids
}

/// Build the masked-vocab callback the constrained generator expects.
/// Wraps the tokenizer in `Arc` (the schema mask caches surface forms
/// per id), seeds a fresh FSM from `schema`, and includes the model's
/// EOS marker ids so structured output can terminate cleanly once the
/// FSM hits `is_complete()`.
pub(in crate::routes::openai) fn build_constrained_mask(
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    schema: Schema,
) -> impl FnMut(&[u32], &mut Vec<f32>) {
    let eos_ids = resolve_eos_token_ids(tokenizer);
    let tk: std::sync::Arc<larql_inference::tokenizers::Tokenizer> =
        std::sync::Arc::new(tokenizer.clone());
    schema::build_mask(tk, schema::Fsm::new(schema), String::new(), eos_ids)
}
