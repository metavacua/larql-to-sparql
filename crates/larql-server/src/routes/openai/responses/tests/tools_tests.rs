use crate::routes::openai::responses::tools::{
    chat_shaped_tool_choice, chat_shaped_tools, response_format_from_text,
};

#[test]
fn flattened_tool_gains_function_wrapper() {
    let out = chat_shaped_tools(&serde_json::json!([
        {"type": "function", "name": "calc", "description": "adds",
         "parameters": {"type": "object"}, "strict": true}
    ]))
    .unwrap();
    assert_eq!(out[0]["type"], "function");
    assert_eq!(out[0]["function"]["name"], "calc");
    assert_eq!(out[0]["function"]["description"], "adds");
    assert_eq!(out[0]["function"]["parameters"]["type"], "object");
}

#[test]
fn chat_shaped_tool_passes_through() {
    let chat_tool = serde_json::json!([
        {"type": "function", "function": {"name": "calc", "parameters": {}}}
    ]);
    let out = chat_shaped_tools(&chat_tool).unwrap();
    assert_eq!(out, chat_tool);
}

#[test]
fn non_function_tool_is_rejected() {
    let err = chat_shaped_tools(&serde_json::json!([
        {"type": "web_search"}
    ]))
    .unwrap_err();
    assert!(err.contains("web_search"), "{err}");
}

#[test]
fn tool_without_name_is_rejected() {
    let err = chat_shaped_tools(&serde_json::json!([{"type": "function"}])).unwrap_err();
    assert!(err.contains("requires a name"), "{err}");
}

#[test]
fn non_array_tools_is_rejected() {
    let err = chat_shaped_tools(&serde_json::json!({"type": "function"})).unwrap_err();
    assert!(err.contains("must be an array"), "{err}");
}

#[test]
fn string_tool_choice_passes_through() {
    for mode in ["auto", "none", "required"] {
        let v = serde_json::json!(mode);
        assert_eq!(chat_shaped_tool_choice(&v), v);
    }
}

#[test]
fn flat_forced_tool_choice_gains_function_wrapper() {
    let out = chat_shaped_tool_choice(&serde_json::json!({"type": "function", "name": "calc"}));
    assert_eq!(out["function"]["name"], "calc");
}

#[test]
fn nested_forced_tool_choice_passes_through() {
    let nested = serde_json::json!({"type": "function", "function": {"name": "calc"}});
    assert_eq!(chat_shaped_tool_choice(&nested), nested);
}

#[test]
fn missing_or_text_format_is_unconstrained() {
    assert!(response_format_from_text(None).unwrap().is_none());
    assert!(response_format_from_text(Some(&serde_json::json!({})))
        .unwrap()
        .is_none());
    assert!(
        response_format_from_text(Some(&serde_json::json!({"format": {"type": "text"}})))
            .unwrap()
            .is_none()
    );
}

#[test]
fn json_object_format_maps_through() {
    let out = response_format_from_text(Some(&serde_json::json!({
        "format": {"type": "json_object"}
    })))
    .unwrap()
    .unwrap();
    assert_eq!(out["type"], "json_object");
}

#[test]
fn json_schema_format_nests_schema_and_strict() {
    let out = response_format_from_text(Some(&serde_json::json!({
        "format": {"type": "json_schema", "name": "answer",
                   "schema": {"type": "object"}, "strict": true}
    })))
    .unwrap()
    .unwrap();
    assert_eq!(out["type"], "json_schema");
    assert_eq!(out["json_schema"]["schema"]["type"], "object");
    assert_eq!(out["json_schema"]["strict"], true);
}

#[test]
fn json_schema_without_schema_is_rejected() {
    let err = response_format_from_text(Some(&serde_json::json!({
        "format": {"type": "json_schema"}
    })))
    .unwrap_err();
    assert!(err.contains("requires a schema"), "{err}");
}

#[test]
fn unknown_format_type_is_rejected() {
    let err = response_format_from_text(Some(&serde_json::json!({
        "format": {"type": "grammar"}
    })))
    .unwrap_err();
    assert!(err.contains("grammar"), "{err}");
}
