//! Synthesise a [`Schema`] from OpenAI's `tools` + `tool_choice`
//! request shape.
//!
//! Output schema shape — a discriminated union per tool:
//!
//! ```json
//! {"oneOf": [
//!   {"type": "object", "properties": {
//!     "name": {"const": "tool_a"},
//!     "arguments": <args_schema_a>
//!   }, "required": ["name", "arguments"], "additionalProperties": false},
//!   {"type": "object", "properties": {
//!     "name": {"const": "tool_b"},
//!     "arguments": <args_schema_b>
//!   }, "required": ["name", "arguments"], "additionalProperties": false}
//! ]}
//! ```
//!
//! After generation, the server parses the produced JSON and fills out
//! the OpenAI `tool_calls` response shape (one `{id, type: "function",
//! function: {name, arguments}}` entry per object in the output).

use std::collections::BTreeMap;

use serde_json::Value;

use super::ast::{ObjectSchema, Schema};
use super::parser::{parse_schema_with, ParseOptions};

/// Resolved tool-choice mode.
///
/// - `None` — request has no `tools` (or `tool_choice == "none"`); skip
///   constrained decoding.
/// - `Any` — model must emit a call to *some* listed tool (`tool_choice
///   == "auto"` or `"required"`).
/// - `Specific(name)` — model must emit a call to this exact tool.
#[derive(Debug, Clone)]
pub enum ToolMode {
    None,
    Any,
    Specific(String),
}

/// Parse `tool_choice` against the listed tools. Returns the resolved
/// mode, or an error if the choice references an unknown tool.
pub fn resolve_tool_choice(
    tools_present: bool,
    tool_choice: Option<&Value>,
    tool_names: &[String],
) -> Result<ToolMode, String> {
    if !tools_present {
        // Even if tool_choice is set, no tools means nothing to call.
        return Ok(ToolMode::None);
    }
    match tool_choice {
        None => Ok(ToolMode::Any), // OpenAI default when tools are present
        Some(Value::String(s)) => match s.as_str() {
            "none" => Ok(ToolMode::None),
            "auto" | "required" => Ok(ToolMode::Any),
            other => Err(format!(
                "tool_choice string must be \"none\" | \"auto\" | \"required\" (got {other:?})"
            )),
        },
        Some(v) if v.is_object() => {
            let kind = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
            if kind != "function" {
                return Err(format!(
                    "tool_choice.type must be \"function\" (got {kind:?})"
                ));
            }
            let name = v
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .ok_or_else(|| {
                    "tool_choice.function.name is required when tool_choice.type=function"
                        .to_string()
                })?;
            if !tool_names.iter().any(|t| t == name) {
                return Err(format!(
                    "tool_choice.function.name {name:?} is not in tools list"
                ));
            }
            Ok(ToolMode::Specific(name.to_string()))
        }
        Some(other) => Err(format!(
            "tool_choice must be a string or {{type, function}} object (got {other:?})"
        )),
    }
}

/// Build a `Schema` from the `tools` array. Each tool is expected to be
/// `{type: "function", function: {name, parameters}}`. Returns the
/// extracted tool names alongside the schema so the handler can use
/// them when shaping the `tool_calls` response.
///
/// `mode` filters which branches end up in the schema:
/// - `ToolMode::Any` → all tools.
/// - `ToolMode::Specific(name)` → only that tool.
/// - `ToolMode::None` → returns `None` (caller should not constrain).
pub fn synth_tools_schema(
    tools: &Value,
    mode: &ToolMode,
) -> Result<Option<(Schema, Vec<String>)>, String> {
    let arr = tools
        .as_array()
        .ok_or_else(|| "tools must be an array".to_string())?;
    if arr.is_empty() || matches!(mode, ToolMode::None) {
        return Ok(None);
    }
    let mut branches = Vec::new();
    let mut names = Vec::new();
    for (i, t) in arr.iter().enumerate() {
        let kind = t.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if kind != "function" {
            return Err(format!(
                "tools[{i}].type must be \"function\" (got {kind:?})"
            ));
        }
        let func = t
            .get("function")
            .ok_or_else(|| format!("tools[{i}].function is required"))?;
        let name = func
            .get("name")
            .and_then(|n| n.as_str())
            .ok_or_else(|| format!("tools[{i}].function.name is required"))?
            .to_string();
        // Filter by tool_choice if the caller pinned a specific function.
        if let ToolMode::Specific(target) = mode {
            if &name != target {
                continue;
            }
        }
        // `parameters` is the JSON Schema for arguments. Missing or `{}`
        // means "no constraints" (Schema::Any). We always parse with
        // `strict: true` for tool args — OpenAI's structured-outputs for
        // tools is strict by default and the runtime guarantees match
        // accordingly.
        let args_schema = match func.get("parameters") {
            Some(p) => parse_schema_with(p, ParseOptions { strict: true })
                .map_err(|e| format!("tools[{i}].function.parameters: {e}"))?,
            None => Schema::Any,
        };
        branches.push(make_tool_branch(&name, args_schema));
        names.push(name);
    }
    if branches.is_empty() {
        return Err("no tool matched the requested tool_choice".into());
    }
    let schema = if branches.len() == 1 {
        branches.into_iter().next().unwrap()
    } else {
        Schema::OneOf(branches)
    };
    Ok(Some((schema, names)))
}

/// `{type: "object", properties: {name: const "<name>", arguments:
/// <args_schema>}, required: ["name", "arguments"],
/// additionalProperties: false}` — one branch of the per-tool union.
fn make_tool_branch(name: &str, args_schema: Schema) -> Schema {
    let mut props: BTreeMap<String, Schema> = BTreeMap::new();
    props.insert("name".into(), Schema::Const(serde_json::json!(name)));
    props.insert("arguments".into(), args_schema);
    Schema::Object(ObjectSchema {
        properties: props,
        required: vec!["name".into(), "arguments".into()],
        additional: None,
    })
}

#[cfg(test)]
mod tests {
    use super::super::fsm::{Fsm, StepResult};
    use super::*;

    fn tool(name: &str, params: serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "type": "function",
            "function": {"name": name, "parameters": params}
        })
    }

    #[test]
    fn resolve_none_when_no_tools() {
        let mode = resolve_tool_choice(false, None, &[]).unwrap();
        assert!(matches!(mode, ToolMode::None));
    }

    #[test]
    fn resolve_any_default_when_tools_present() {
        let mode = resolve_tool_choice(true, None, &["a".into()]).unwrap();
        assert!(matches!(mode, ToolMode::Any));
    }

    #[test]
    fn resolve_string_modes() {
        for (s, expected_any) in [("auto", true), ("required", true)] {
            let m = resolve_tool_choice(true, Some(&serde_json::json!(s)), &["a".into()]).unwrap();
            assert_eq!(matches!(m, ToolMode::Any), expected_any);
        }
        let m = resolve_tool_choice(true, Some(&serde_json::json!("none")), &["a".into()]).unwrap();
        assert!(matches!(m, ToolMode::None));
    }

    #[test]
    fn resolve_specific_function() {
        let choice = serde_json::json!({"type": "function", "function": {"name": "calc"}});
        let mode =
            resolve_tool_choice(true, Some(&choice), &["calc".into(), "search".into()]).unwrap();
        assert!(matches!(mode, ToolMode::Specific(ref n) if n == "calc"));
    }

    #[test]
    fn resolve_specific_unknown_errors() {
        let choice = serde_json::json!({"type": "function", "function": {"name": "missing"}});
        let err = resolve_tool_choice(true, Some(&choice), &["calc".into()]).unwrap_err();
        assert!(err.contains("not in tools list"), "{err}");
    }

    #[test]
    fn synth_one_tool_drops_oneof_wrapper() {
        let tools = serde_json::json!([tool("calc", serde_json::json!({"type": "object"}))]);
        let (schema, names) = synth_tools_schema(&tools, &ToolMode::Any).unwrap().unwrap();
        assert_eq!(names, vec!["calc".to_string()]);
        // Single tool → single branch (no OneOf wrapper needed).
        assert!(matches!(schema, Schema::Object(_)));
    }

    #[test]
    fn synth_two_tools_oneof_wraps() {
        let tools = serde_json::json!([
            tool("calc", serde_json::json!({"type": "object"})),
            tool("search", serde_json::json!({"type": "object"})),
        ]);
        let (schema, names) = synth_tools_schema(&tools, &ToolMode::Any).unwrap().unwrap();
        assert_eq!(names.len(), 2);
        assert!(matches!(schema, Schema::OneOf(_)));
    }

    #[test]
    fn synth_specific_filters_branches() {
        let tools = serde_json::json!([
            tool("calc", serde_json::json!({"type": "object"})),
            tool("search", serde_json::json!({"type": "object"})),
        ]);
        let (schema, names) = synth_tools_schema(&tools, &ToolMode::Specific("calc".into()))
            .unwrap()
            .unwrap();
        assert_eq!(names, vec!["calc".to_string()]);
        assert!(matches!(schema, Schema::Object(_)));
    }

    #[test]
    fn fsm_enforces_tool_call_shape() {
        // Two tools with distinct argument shapes — the FSM must
        // commit to the right branch as soon as `name` disambiguates.
        let tools = serde_json::json!([
            tool(
                "set_temp",
                serde_json::json!({
                    "type": "object",
                    "properties": {"degrees": {"type": "integer"}},
                    "required": ["degrees"]
                })
            ),
            tool(
                "send_message",
                serde_json::json!({
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"]
                })
            ),
        ]);
        let (schema, _) = synth_tools_schema(&tools, &ToolMode::Any).unwrap().unwrap();
        // set_temp call — integer arg.
        let mut fsm = Fsm::new(schema.clone());
        assert_eq!(
            fsm.step_str(r#"{"name":"set_temp","arguments":{"degrees":21}}"#),
            StepResult::Ok
        );
        assert!(fsm.is_complete());
        // send_message call — string arg.
        let mut fsm2 = Fsm::new(schema.clone());
        assert_eq!(
            fsm2.step_str(r#"{"name":"send_message","arguments":{"text":"hi"}}"#),
            StepResult::Ok
        );
        assert!(fsm2.is_complete());
        // Crossing the streams: send_message with degrees should reject.
        let mut fsm3 = Fsm::new(schema);
        let r = fsm3.step_str(r#"{"name":"send_message","arguments":{"degrees":21}}"#);
        // Either step_str rejected, or it completed but without is_complete
        // matching the strict-required signal.
        assert!(r == StepResult::Reject || !fsm3.is_complete());
    }

    #[test]
    fn synth_none_mode_returns_no_schema() {
        let tools = serde_json::json!([tool("x", serde_json::json!({}))]);
        let result = synth_tools_schema(&tools, &ToolMode::None).unwrap();
        assert!(result.is_none());
    }
}
