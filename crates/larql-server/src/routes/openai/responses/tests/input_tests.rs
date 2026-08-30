use crate::routes::openai::responses::input::{input_to_messages, items_to_messages};
use crate::routes::openai::responses::types::{InputItem, ResponseInput};

fn items(v: serde_json::Value) -> Vec<InputItem> {
    serde_json::from_value(v).unwrap()
}

#[test]
fn bare_string_becomes_one_user_turn() {
    let messages = input_to_messages(&ResponseInput::Text("hello".into())).unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].role, "user");
    assert_eq!(messages[0].content.as_deref(), Some("hello"));
}

#[test]
fn message_items_with_string_and_part_content() {
    let messages = items_to_messages(&items(serde_json::json!([
        {"role": "user", "content": "plain"},
        {"type": "message", "role": "assistant",
         "content": [{"type": "output_text", "text": "a"}, {"type": "text", "text": "b"}]},
    ])))
    .unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].content.as_deref(), Some("plain"));
    assert_eq!(messages[1].role, "assistant");
    assert_eq!(messages[1].content.as_deref(), Some("ab"));
}

#[test]
fn developer_role_maps_to_system() {
    let messages = items_to_messages(&items(serde_json::json!([
        {"role": "developer", "content": "be brief"}
    ])))
    .unwrap();
    assert_eq!(messages[0].role, "system");
}

#[test]
fn function_call_item_becomes_assistant_tool_call_echo() {
    let messages = items_to_messages(&items(serde_json::json!([
        {"type": "function_call", "call_id": "call_9", "name": "get_weather",
         "arguments": "{\"city\":\"Paris\"}"}
    ])))
    .unwrap();
    assert_eq!(messages[0].role, "assistant");
    assert!(messages[0].content.is_none());
    let calls = messages[0].tool_calls.as_ref().unwrap();
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    assert_eq!(calls[0]["id"], "call_9");
}

#[test]
fn function_call_output_becomes_tool_turn() {
    let messages = items_to_messages(&items(serde_json::json!([
        {"type": "function_call_output", "call_id": "call_9", "output": "23C"}
    ])))
    .unwrap();
    assert_eq!(messages[0].role, "tool");
    assert_eq!(messages[0].tool_call_id.as_deref(), Some("call_9"));
    assert_eq!(messages[0].content.as_deref(), Some("23C"));
}

#[test]
fn reasoning_items_are_skipped() {
    let messages = items_to_messages(&items(serde_json::json!([
        {"type": "reasoning", "content": [{"type": "text", "text": "…"}]},
        {"role": "user", "content": "hi"}
    ])))
    .unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].role, "user");
}

#[test]
fn unknown_item_type_is_rejected_with_index() {
    let err = items_to_messages(&items(serde_json::json!([
        {"role": "user", "content": "hi"},
        {"type": "image_generation_call"}
    ])))
    .unwrap_err();
    assert!(err.contains("input[1]"), "{err}");
    assert!(err.contains("image_generation_call"), "{err}");
}

#[test]
fn message_without_role_is_rejected() {
    let err = items_to_messages(&items(serde_json::json!([
        {"type": "message", "content": "hi"}
    ])))
    .unwrap_err();
    assert!(err.contains("requires a role"), "{err}");
}

#[test]
fn message_without_content_is_rejected() {
    let err = items_to_messages(&items(serde_json::json!([
        {"role": "user"}
    ])))
    .unwrap_err();
    assert!(err.contains("requires content"), "{err}");
}

#[test]
fn unsupported_role_is_rejected() {
    let err = items_to_messages(&items(serde_json::json!([
        {"role": "moderator", "content": "hi"}
    ])))
    .unwrap_err();
    assert!(err.contains("moderator"), "{err}");
}

#[test]
fn non_text_content_part_is_rejected() {
    let err = items_to_messages(&items(serde_json::json!([
        {"role": "user", "content": [{"type": "input_image", "text": null}]}
    ])))
    .unwrap_err();
    assert!(err.contains("input_image"), "{err}");
    assert!(err.contains("text-only"), "{err}");
}

#[test]
fn function_call_without_name_is_rejected() {
    let err = items_to_messages(&items(serde_json::json!([
        {"type": "function_call", "call_id": "call_1"}
    ])))
    .unwrap_err();
    assert!(err.contains("requires name"), "{err}");
}

#[test]
fn function_call_output_requires_call_id_and_output() {
    let err = items_to_messages(&items(serde_json::json!([
        {"type": "function_call_output", "output": "x"}
    ])))
    .unwrap_err();
    assert!(err.contains("requires call_id"), "{err}");

    let err = items_to_messages(&items(serde_json::json!([
        {"type": "function_call_output", "call_id": "call_1"}
    ])))
    .unwrap_err();
    assert!(err.contains("requires output"), "{err}");
}
