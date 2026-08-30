//! Convert Responses-API `input` items into the internal chat-message
//! shape the shared prompt renderer understands.
//!
//! The Responses wire allows four item kinds we serve: plain `message`
//! turns, `function_call` (an assistant tool dispatch being replayed),
//! `function_call_output` (the tool's result), and `reasoning` items
//! (ignored — larql emits none, but clients replaying a prior
//! conversation may echo them back). Anything else — images, audio,
//! hosted-tool items — is rejected with a clear message the handler
//! turns into a 400.

use crate::routes::openai::chat::ChatMessage;
use crate::routes::openai::prompt::{ASSISTANT_ROLE, SYSTEM_ROLE, TOOL_ROLE, USER_ROLE};

use super::types::{ContentPart, InputItem, ItemContent, ResponseInput};

/// Item `type` discriminators on the Responses wire.
const ITEM_MESSAGE: &str = "message";
const ITEM_FUNCTION_CALL: &str = "function_call";
const ITEM_FUNCTION_CALL_OUTPUT: &str = "function_call_output";
const ITEM_REASONING: &str = "reasoning";

/// Content-part `type` values that carry plain text.
const TEXT_PART_KINDS: [&str; 3] = ["input_text", "output_text", "text"];

/// The Responses API's `developer` role — rendered as a system turn.
const DEVELOPER_ROLE: &str = "developer";

/// Flatten `input` (bare string or item list) to chat messages.
pub(super) fn input_to_messages(input: &ResponseInput) -> Result<Vec<ChatMessage>, String> {
    match input {
        ResponseInput::Text(text) => Ok(vec![text_message(USER_ROLE, text.clone())]),
        ResponseInput::Items(items) => items_to_messages(items),
    }
}

/// Convert one item list. Errors name the offending index so clients
/// can find the bad element in a long conversation replay.
pub(super) fn items_to_messages(items: &[InputItem]) -> Result<Vec<ChatMessage>, String> {
    let mut out = Vec::with_capacity(items.len());
    for (i, item) in items.iter().enumerate() {
        let kind = item.kind.as_deref().unwrap_or(ITEM_MESSAGE);
        match kind {
            ITEM_MESSAGE => out.push(message_item(i, item)?),
            ITEM_FUNCTION_CALL => out.push(function_call_item(i, item)?),
            ITEM_FUNCTION_CALL_OUTPUT => out.push(function_call_output_item(i, item)?),
            ITEM_REASONING => {} // replayed reasoning summaries carry no prompt text
            other => {
                return Err(format!(
                    "input[{i}].type {other:?} is not supported (expected \
                     \"message\" | \"function_call\" | \"function_call_output\")"
                ))
            }
        }
    }
    Ok(out)
}

fn message_item(i: usize, item: &InputItem) -> Result<ChatMessage, String> {
    let role = item
        .role
        .as_deref()
        .ok_or_else(|| format!("input[{i}] message requires a role"))?;
    let role = match role {
        // `developer` replaced `system` in the Responses API; both are
        // rendered through the templates' system slot.
        DEVELOPER_ROLE => SYSTEM_ROLE,
        USER_ROLE | ASSISTANT_ROLE | SYSTEM_ROLE => role,
        other => {
            return Err(format!(
                "input[{i}].role {other:?} is not supported (expected \
                 \"user\" | \"assistant\" | \"system\" | \"developer\")"
            ))
        }
    };
    let content = item
        .content
        .as_ref()
        .ok_or_else(|| format!("input[{i}] message requires content"))?;
    Ok(text_message(role, content_text(i, content)?))
}

fn function_call_item(i: usize, item: &InputItem) -> Result<ChatMessage, String> {
    let name = item
        .name
        .as_deref()
        .ok_or_else(|| format!("input[{i}] function_call requires name"))?;
    let arguments = item.arguments.as_deref().unwrap_or("{}");
    let call_id = item.call_id.as_deref().unwrap_or("");
    // Re-shape into the chat `tool_calls` echo so the shared renderer's
    // tool-flattening applies unchanged.
    let tool_calls = serde_json::json!([{
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }]);
    Ok(ChatMessage {
        role: ASSISTANT_ROLE.to_string(),
        content: None,
        tool_calls: Some(tool_calls),
        tool_call_id: None,
        name: None,
    })
}

fn function_call_output_item(i: usize, item: &InputItem) -> Result<ChatMessage, String> {
    let call_id = item
        .call_id
        .as_deref()
        .ok_or_else(|| format!("input[{i}] function_call_output requires call_id"))?;
    let output = item
        .output
        .as_ref()
        .ok_or_else(|| format!("input[{i}] function_call_output requires output"))?;
    Ok(ChatMessage {
        role: TOOL_ROLE.to_string(),
        content: Some(content_text(i, output)?),
        tool_calls: None,
        tool_call_id: Some(call_id.to_string()),
        name: None,
    })
}

/// Flatten message content to plain text, rejecting non-text parts.
fn content_text(i: usize, content: &ItemContent) -> Result<String, String> {
    match content {
        ItemContent::Text(s) => Ok(s.clone()),
        ItemContent::Parts(parts) => {
            let mut out = String::new();
            for part in parts {
                out.push_str(&text_part(i, part)?);
            }
            Ok(out)
        }
    }
}

fn text_part(i: usize, part: &ContentPart) -> Result<String, String> {
    if !TEXT_PART_KINDS.contains(&part.kind.as_str()) {
        return Err(format!(
            "input[{i}] content part type {:?} is not supported (text-only server)",
            part.kind
        ));
    }
    Ok(part.text.clone().unwrap_or_default())
}

fn text_message(role: &str, content: String) -> ChatMessage {
    ChatMessage {
        role: role.to_string(),
        content: Some(content),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }
}
