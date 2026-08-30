//! Adapters between the Responses-API tool wire shapes and the chat
//! shapes the existing constrained-decoding machinery
//! (`routes::openai::schema`) consumes.
//!
//! Responses flattened a level relative to chat:
//!
//! ```text
//! chat:      {type: "function", function: {name, description, parameters}}
//! responses: {type: "function", name, description, parameters, strict}
//! ```
//!
//! and likewise for `tool_choice` (`{type:"function", name}` vs
//! `{type:"function", function:{name}}`). These converters normalise to
//! the chat shape so one schema synthesiser serves both endpoints.

const FUNCTION_TOOL_TYPE: &str = "function";

/// Convert a Responses `tools` array to the chat shape.
///
/// Tools already in the nested chat shape pass through untouched so
/// clients migrating between the two APIs aren't punished for it.
pub(super) fn chat_shaped_tools(tools: &serde_json::Value) -> Result<serde_json::Value, String> {
    let arr = tools
        .as_array()
        .ok_or_else(|| "tools must be an array".to_string())?;
    let mut out = Vec::with_capacity(arr.len());
    for (i, tool) in arr.iter().enumerate() {
        let kind = tool.get("type").and_then(|t| t.as_str()).unwrap_or("");
        if kind != FUNCTION_TOOL_TYPE {
            return Err(format!(
                "tools[{i}].type {kind:?} is not supported (only \"function\" tools)"
            ));
        }
        if tool.get("function").is_some() {
            // Already chat-shaped.
            out.push(tool.clone());
            continue;
        }
        let name = tool
            .get("name")
            .and_then(|n| n.as_str())
            .ok_or_else(|| format!("tools[{i}] requires a name"))?;
        let mut function = serde_json::Map::new();
        function.insert("name".into(), serde_json::Value::String(name.to_string()));
        if let Some(desc) = tool.get("description") {
            function.insert("description".into(), desc.clone());
        }
        if let Some(params) = tool.get("parameters") {
            function.insert("parameters".into(), params.clone());
        }
        out.push(serde_json::json!({
            "type": FUNCTION_TOOL_TYPE,
            "function": serde_json::Value::Object(function),
        }));
    }
    Ok(serde_json::Value::Array(out))
}

/// Convert a Responses `tool_choice` to the chat shape. String modes
/// (`"auto" | "none" | "required"`) pass through; a forced
/// `{type:"function", name}` gains the nested `function` wrapper.
pub(super) fn chat_shaped_tool_choice(choice: &serde_json::Value) -> serde_json::Value {
    if let Some(name) = choice
        .as_object()
        .filter(|o| o.get("function").is_none())
        .and_then(|o| o.get("name"))
        .and_then(|n| n.as_str())
    {
        return serde_json::json!({
            "type": FUNCTION_TOOL_TYPE,
            "function": {"name": name},
        });
    }
    choice.clone()
}

/// Map the Responses `text.format` field to chat's `response_format`
/// shape, ready for `chat::schema_for_response_format`.
///
/// `{format: {type:"json_schema", name, schema, strict}}` becomes
/// `{type:"json_schema", json_schema: {schema, strict}}`;
/// `json_object` / `text` pass through as `{type}`. `None` (or a
/// missing/`text` format) means unconstrained.
pub(super) fn response_format_from_text(
    text: Option<&serde_json::Value>,
) -> Result<Option<serde_json::Value>, String> {
    let Some(format) = text.and_then(|t| t.get("format")) else {
        return Ok(None);
    };
    let kind = format
        .get("type")
        .and_then(|t| t.as_str())
        .unwrap_or("text");
    match kind {
        "text" => Ok(None),
        "json_object" => Ok(Some(serde_json::json!({"type": "json_object"}))),
        "json_schema" => {
            let schema = format
                .get("schema")
                .ok_or_else(|| "text.format.type=json_schema requires a schema".to_string())?;
            let strict = format.get("strict").cloned().unwrap_or(false.into());
            Ok(Some(serde_json::json!({
                "type": "json_schema",
                "json_schema": {"schema": schema, "strict": strict},
            })))
        }
        other => Err(format!(
            "text.format.type {other:?} is not supported (expected \
             \"text\" | \"json_object\" | \"json_schema\")"
        )),
    }
}
