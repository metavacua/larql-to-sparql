//! Unit tests for the chat completions module.

use super::handler::build_chat_logprobs;
use super::stream::build_chat_tool_calls_chunk;
use super::tools::build_tool_call_message;
use super::{ChatCompletionsRequest, ToolCall, ToolCallFunction};

// Multi-turn template rendering is tested in
// `larql_inference::prompt::render_messages_tests` (Gemma, ChatML,
// Llama, Mistral, Plain). This handler only marshals JSON to the
// inference helper, so our tests focus on the request-validation
// surface and shape decisions specific to the OpenAI wire.

#[test]
fn deserialize_chat_request_min() {
    let json = serde_json::json!({
        "messages": [{"role": "user", "content": "hi"}]
    });
    let req: ChatCompletionsRequest = serde_json::from_value(json).unwrap();
    assert_eq!(req.messages.len(), 1);
    assert_eq!(req.messages[0].role, "user");
}

#[test]
fn deserialize_chat_request_full() {
    let json = serde_json::json!({
        "model": "gemma-3-4b",
        "messages": [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "What is 2+2?"}
        ],
        "max_tokens": 50,
        "temperature": 0.0,
        "top_p": 0.9,
        "n": 1,
        "stream": false,
        "stop": ["\n\n"],
        "seed": 42
    });
    let req: ChatCompletionsRequest = serde_json::from_value(json).unwrap();
    assert_eq!(req.messages.len(), 2);
    assert_eq!(req.max_tokens, Some(50));
    assert_eq!(req.temperature, Some(0.0));
}

#[test]
fn build_chat_tool_calls_chunk_shapes_delta_correctly() {
    let calls = vec![ToolCall {
        id: "call_xyz".into(),
        kind: "function",
        function: ToolCallFunction {
            name: "calc".into(),
            arguments: "{\"a\":1,\"b\":2}".into(),
        },
    }];
    let chunk = build_chat_tool_calls_chunk("chatcmpl-x", "gemma", &calls);
    let v: serde_json::Value = serde_json::from_str(&chunk).unwrap();
    assert_eq!(v["object"], "chat.completion.chunk");
    assert_eq!(v["choices"][0]["delta"]["tool_calls"][0]["index"], 0);
    assert_eq!(v["choices"][0]["delta"]["tool_calls"][0]["id"], "call_xyz");
    assert_eq!(
        v["choices"][0]["delta"]["tool_calls"][0]["type"],
        "function"
    );
    assert_eq!(
        v["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
        "calc"
    );
    // arguments is JSON-stringified.
    assert_eq!(
        v["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
        "{\"a\":1,\"b\":2}"
    );
    assert!(v["choices"][0]["finish_reason"].is_null());
}

#[test]
fn build_chat_logprobs_emits_one_entry_per_token() {
    let toks = vec![("Paris".to_string(), 1.0), (".".to_string(), 1.0)];
    let lp = build_chat_logprobs(&toks);
    assert_eq!(lp.content.len(), 2);
    assert_eq!(lp.content[0].token, "Paris");
    assert_eq!(lp.content[0].bytes, b"Paris".to_vec());
    assert!(lp.content[0].top_logprobs.is_empty());
    // prob=1.0 → logprob=0.0 (placeholder until inference exposes
    // real per-token softmax probs).
    assert!((lp.content[0].logprob - 0.0).abs() < 1e-6);
}

#[test]
fn deserialize_chat_message_with_tool_call_replay() {
    // Multi-turn shape OpenAI clients send back: assistant tool-call
    // + tool result + (next) assistant turn the model would emit.
    let json = serde_json::json!({
        "messages": [
            {"role": "user", "content": "Weather?"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "get_weather", "arguments": "{\"city\":\"London\"}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "23C"}
        ]
    });
    let req: ChatCompletionsRequest = serde_json::from_value(json).unwrap();
    assert_eq!(req.messages.len(), 3);
    assert!(req.messages[1].content.is_none());
    assert!(req.messages[1].tool_calls.is_some());
    assert_eq!(req.messages[2].role, "tool");
    assert_eq!(req.messages[2].tool_call_id.as_deref(), Some("call_1"));
    assert_eq!(req.messages[2].content.as_deref(), Some("23C"));
}

// ─────────────────────────────────────────────────────────────────
// REV5 — build_tool_call_message: replace fragile find/rfind slicer
// with serde_json::from_str on the trimmed text. Failure surfaces
// as ServerError::BadRequest (400 + invalid_request_error) at the
// entry handler, not Internal (500).
// ─────────────────────────────────────────────────────────────────

#[test]
fn build_tool_call_happy_path() {
    let text = r#"{"name":"get_weather","arguments":{"city":"Paris"}}"#;
    let msg = build_tool_call_message(text).unwrap();
    let calls = msg.tool_calls.as_ref().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "get_weather");
    assert_eq!(calls[0].function.arguments, r#"{"city":"Paris"}"#);
    assert!(msg.content.is_none());
}

#[test]
fn build_tool_call_tolerates_surrounding_whitespace() {
    let text = "  \n\t {\"name\":\"f\",\"arguments\":{}} \n";
    let msg = build_tool_call_message(text).unwrap();
    assert_eq!(msg.tool_calls.unwrap()[0].function.name, "f");
}

#[test]
fn build_tool_call_handles_nested_braces_in_arguments() {
    // Pre-REV5 the rfind('}') would have walked back past the
    // outer closing brace correctly here (it's still the LAST
    // '}'), so this case actually worked before. We keep the test
    // to lock that the cleaner serde_json approach also handles
    // nested braces — the property the original code was trying
    // to preserve.
    let text = r#"{"name":"f","arguments":{"x":"{}","y":[{"z":1}]}}"#;
    let msg = build_tool_call_message(text).unwrap();
    let args: serde_json::Value =
        serde_json::from_str(&msg.tool_calls.unwrap()[0].function.arguments).unwrap();
    assert_eq!(args["x"], "{}");
    assert_eq!(args["y"][0]["z"], 1);
}

#[test]
fn build_tool_call_rejects_trailing_junk_with_clean_error() {
    // Pre-REV5 this would have produced an invalid slice (the
    // rfind('}') matched the trailing brace inside "extra}") and
    // serfailed → 500 Internal. Post-REV5 the parse fails at the
    // first non-JSON character, surfacing a clean
    // `invalid JSON: trailing characters …` diagnostic which the
    // entry handler maps to 400.
    let text = r#"{"name":"f","arguments":{}} extra}"#;
    let err = build_tool_call_message(text).unwrap_err();
    assert!(
        err.starts_with("invalid JSON:"),
        "want diagnostic prefix; got {err:?}"
    );
}

#[test]
fn build_tool_call_rejects_empty_input() {
    assert_eq!(
        build_tool_call_message("   ").unwrap_err(),
        "tool output was empty"
    );
}

#[test]
fn build_tool_call_rejects_non_object_top_level() {
    let err = build_tool_call_message(r#"["not","an","object"]"#).unwrap_err();
    assert!(
        err.starts_with("tool output must be a JSON object"),
        "got {err:?}"
    );
    assert!(
        err.contains("array"),
        "kind should be reported; got {err:?}"
    );
}

#[test]
fn build_tool_call_rejects_missing_name() {
    let err = build_tool_call_message(r#"{"arguments":{}}"#).unwrap_err();
    assert_eq!(err, "tool output missing `name`");
}

#[test]
fn build_tool_call_rejects_missing_arguments() {
    let err = build_tool_call_message(r#"{"name":"f"}"#).unwrap_err();
    assert_eq!(err, "tool output missing `arguments`");
}

#[test]
fn build_tool_call_rejects_invalid_json() {
    let err = build_tool_call_message("not json at all").unwrap_err();
    assert!(err.starts_with("invalid JSON:"));
}

// ── stop-string trim helpers ────────────────────────────────────────
//
// The end-to-end stop path is nondeterministic on the Q4K synthetic
// fixture (its CPU path buffers tokens without callbacks and can emit
// empty text), so the byte-accounting helpers are pinned directly.

fn scored(tokens: &[&str]) -> Vec<(String, f64)> {
    tokens.iter().map(|t| (t.to_string(), 1.0)).collect()
}

#[test]
fn trim_tokens_to_text_keeps_tokens_covering_the_text() {
    let tokens = scored(&["Par", "is", " is", " nice"]);
    let out = super::handler::trim_tokens_to_text(&tokens, "Paris");
    assert_eq!(out, scored(&["Par", "is"]));
}

#[test]
fn trim_tokens_to_text_empty_text_keeps_nothing() {
    let tokens = scored(&["a", "b"]);
    assert!(super::handler::trim_tokens_to_text(&tokens, "").is_empty());
}

#[test]
fn trim_tokens_to_text_text_longer_than_tokens_keeps_all() {
    let tokens = scored(&["ab", "cd"]);
    let out = super::handler::trim_tokens_to_text(&tokens, "abcdefgh");
    assert_eq!(out.len(), 2);
}

#[test]
fn trim_tokens_to_text_partial_token_overlap_keeps_covering_token() {
    // The trim boundary can land mid-token; the covering token is kept
    // (a good approximation, per the handler's doc comment).
    let tokens = scored(&["abc", "def"]);
    let out = super::handler::trim_tokens_to_text(&tokens, "abcd");
    assert_eq!(out, scored(&["abc", "def"]));
}

#[test]
fn count_tokens_covering_counts_leading_tokens() {
    let tokens = scored(&["Par", "is", " is", " nice"]);
    assert_eq!(super::v3::count_tokens_covering(&tokens, 5), 2);
}

#[test]
fn count_tokens_covering_zero_len_counts_none() {
    let tokens = scored(&["a"]);
    assert_eq!(super::v3::count_tokens_covering(&tokens, 0), 0);
}

#[test]
fn count_tokens_covering_len_past_tokens_counts_all() {
    let tokens = scored(&["ab", "cd"]);
    assert_eq!(super::v3::count_tokens_covering(&tokens, 100), 2);
}
