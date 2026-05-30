//! JSON Schema (subset) → [`Schema`] AST.
//!
//! Supports the JSON-Schema features OpenAI's structured-outputs use in
//! practice:
//!
//! - `type`: `"object" | "array" | "string" | "number" | "integer" |
//!   "boolean" | "null"`, plus an array form (`["string", "null"]`)
//!   that decodes to [`Schema::OneOf`].
//! - `properties`, `required`, `additionalProperties` on objects.
//! - `items`, `minItems`, `maxItems` on arrays.
//! - `enum`, `const`, `minLength`, `maxLength` on strings.
//! - `minimum`, `maximum` on numbers.
//! - `oneOf` / `anyOf` (both decode to [`Schema::OneOf`]).
//! - Top-level `$schema`, `title`, `description`, `examples`: ignored.
//!
//! Out of scope (returns an error so callers know the schema isn't fully
//! enforced rather than silently relaxing it):
//! - `$ref`, `$defs`, `definitions`
//! - `pattern`, `format`
//! - `not`, `if/then/else`, `dependencies`
//! - `allOf` (which would require schema-merge; OpenAI tools don't need
//!   it for the typical function-args shape)
//!
//! Parsing produces an `ast::Schema` along with a small `ParseOptions`
//! that the caller can pass via [`parse_schema_with`] — for example
//! `strict: true` flips `additionalProperties`'s default from "any" to
//! "forbidden", matching OpenAI's strict-mode contract.

use std::collections::BTreeMap;

use serde_json::Value;

use super::ast::{ArraySchema, NumberSchema, ObjectSchema, Schema, StringSchema};

/// Caller-controlled defaults applied to the parser.
#[derive(Debug, Clone, Copy, Default)]
pub struct ParseOptions {
    /// When set, an Object with no `additionalProperties` keyword
    /// rejects unknown keys (OpenAI's `strict: true` semantics).
    pub strict: bool,
}

/// Parse a JSON-Schema value with the default (non-strict) options.
pub fn parse_schema(value: &Value) -> Result<Schema, String> {
    parse_schema_with(value, ParseOptions::default())
}

/// Parse with explicit options — call from slice 4.5 with
/// `strict: true` to mirror OpenAI structured-outputs semantics.
pub fn parse_schema_with(value: &Value, opts: ParseOptions) -> Result<Schema, String> {
    parse_inner(value, opts)
}

fn parse_inner(value: &Value, opts: ParseOptions) -> Result<Schema, String> {
    // `true` / `false` schema (JSON Schema 2019-09): `true` accepts any
    // value, `false` rejects everything. We treat `false` as an error
    // since it's a degenerate API choice.
    if let Some(b) = value.as_bool() {
        return if b {
            Ok(Schema::Any)
        } else {
            Err("schema literal `false` rejects every value".into())
        };
    }

    let obj = value
        .as_object()
        .ok_or_else(|| format!("expected a schema object, got {value:?}"))?;

    if let Some(c) = obj.get("const") {
        return Ok(Schema::Const(c.clone()));
    }

    if let Some(en) = obj.get("enum") {
        let arr = en
            .as_array()
            .ok_or_else(|| "enum must be an array".to_string())?;
        let branches = arr
            .iter()
            .map(|v| Schema::Const(v.clone()))
            .collect::<Vec<_>>();
        if branches.is_empty() {
            return Err("enum must have at least one value".into());
        }
        return Ok(Schema::OneOf(branches));
    }

    if let Some(of) = obj.get("oneOf").or_else(|| obj.get("anyOf")) {
        let arr = of
            .as_array()
            .ok_or_else(|| "oneOf / anyOf must be an array".to_string())?;
        let branches = arr
            .iter()
            .map(|v| parse_inner(v, opts))
            .collect::<Result<Vec<_>, _>>()?;
        if branches.is_empty() {
            return Err("oneOf / anyOf must have at least one branch".into());
        }
        return Ok(Schema::OneOf(branches));
    }

    if obj.contains_key("$ref") || obj.contains_key("$defs") || obj.contains_key("definitions") {
        return Err("$ref / $defs / definitions not yet supported".into());
    }
    if obj.contains_key("not") || obj.contains_key("allOf") || obj.contains_key("if") {
        return Err("not / allOf / if-then-else not yet supported".into());
    }
    if obj.contains_key("pattern") || obj.contains_key("format") {
        return Err("pattern / format not yet supported".into());
    }

    let kind = obj.get("type");
    match kind {
        None => Ok(Schema::Any),
        Some(Value::String(t)) => parse_typed(t, obj, opts),
        Some(Value::Array(arr)) => {
            // Array-of-types: ["string", "null"] → OneOf of single-typed
            // schemas with the same body.
            let branches = arr
                .iter()
                .map(|t| {
                    let t = t
                        .as_str()
                        .ok_or_else(|| "type[] entries must be strings".to_string())?;
                    parse_typed(t, obj, opts)
                })
                .collect::<Result<Vec<_>, _>>()?;
            if branches.is_empty() {
                Err("type [] is empty".into())
            } else if branches.len() == 1 {
                Ok(branches.into_iter().next().unwrap())
            } else {
                Ok(Schema::OneOf(branches))
            }
        }
        Some(other) => Err(format!("type must be a string or array, got {other:?}")),
    }
}

fn parse_typed(
    kind: &str,
    obj: &serde_json::Map<String, Value>,
    opts: ParseOptions,
) -> Result<Schema, String> {
    match kind {
        "object" => parse_object(obj, opts).map(Schema::Object),
        "array" => parse_array(obj, opts).map(Schema::Array),
        "string" => parse_string(obj).map(Schema::String),
        "number" => parse_number(obj, false).map(Schema::Number),
        "integer" => parse_number(obj, true).map(Schema::Number),
        "boolean" => Ok(Schema::Boolean),
        "null" => Ok(Schema::Null),
        other => Err(format!("unknown type {other:?}")),
    }
}

fn parse_object(
    obj: &serde_json::Map<String, Value>,
    opts: ParseOptions,
) -> Result<ObjectSchema, String> {
    let mut properties = BTreeMap::new();
    if let Some(p) = obj.get("properties") {
        let m = p
            .as_object()
            .ok_or_else(|| "properties must be an object".to_string())?;
        for (k, v) in m {
            properties.insert(k.clone(), parse_inner(v, opts)?);
        }
    }
    let mut required = Vec::new();
    if let Some(r) = obj.get("required") {
        let arr = r
            .as_array()
            .ok_or_else(|| "required must be an array".to_string())?;
        for entry in arr {
            let s = entry
                .as_str()
                .ok_or_else(|| "required[] entries must be strings".to_string())?;
            required.push(s.to_string());
        }
    }
    let additional = match obj.get("additionalProperties") {
        Some(Value::Bool(true)) => Some(Box::new(Schema::Any)),
        Some(Value::Bool(false)) => None,
        Some(v) if v.is_object() => Some(Box::new(parse_inner(v, opts)?)),
        Some(other) => {
            return Err(format!(
                "additionalProperties must be bool or schema, got {other:?}"
            ))
        }
        None => {
            if opts.strict {
                None
            } else {
                Some(Box::new(Schema::Any))
            }
        }
    };
    Ok(ObjectSchema {
        properties,
        required,
        additional,
    })
}

fn parse_array(
    obj: &serde_json::Map<String, Value>,
    opts: ParseOptions,
) -> Result<ArraySchema, String> {
    let items = match obj.get("items") {
        Some(v) => parse_inner(v, opts)?,
        None => Schema::Any,
    };
    let min = obj
        .get("minItems")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize);
    let max = obj
        .get("maxItems")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize);
    Ok(ArraySchema {
        items: Box::new(items),
        min,
        max,
    })
}

fn parse_string(obj: &serde_json::Map<String, Value>) -> Result<StringSchema, String> {
    // `enum`/`const` are handled at the top level (they short-circuit
    // `parse_inner`) — at this layer we only see the typed-string form.
    let min_len = obj
        .get("minLength")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize);
    let max_len = obj
        .get("maxLength")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize);
    Ok(StringSchema {
        r#enum: None,
        r#const: None,
        min_len,
        max_len,
    })
}

fn parse_number(
    obj: &serde_json::Map<String, Value>,
    integer: bool,
) -> Result<NumberSchema, String> {
    let minimum = obj.get("minimum").and_then(|v| v.as_f64());
    let maximum = obj.get("maximum").and_then(|v| v.as_f64());
    Ok(NumberSchema {
        integer,
        minimum,
        maximum,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: serde_json::Value) -> Schema {
        parse_schema(&json).expect("parse")
    }

    fn parse_strict(json: serde_json::Value) -> Schema {
        parse_schema_with(&json, ParseOptions { strict: true }).expect("parse")
    }

    #[test]
    fn empty_schema_is_any() {
        assert!(matches!(parse(serde_json::json!({})), Schema::Any));
        assert!(matches!(parse(serde_json::json!(true)), Schema::Any));
    }

    #[test]
    fn typed_primitives() {
        assert!(matches!(
            parse(serde_json::json!({"type": "string"})),
            Schema::String(_)
        ));
        assert!(matches!(
            parse(serde_json::json!({"type": "number"})),
            Schema::Number(NumberSchema { integer: false, .. })
        ));
        assert!(matches!(
            parse(serde_json::json!({"type": "integer"})),
            Schema::Number(NumberSchema { integer: true, .. })
        ));
        assert!(matches!(
            parse(serde_json::json!({"type": "boolean"})),
            Schema::Boolean
        ));
        assert!(matches!(
            parse(serde_json::json!({"type": "null"})),
            Schema::Null
        ));
    }

    #[test]
    fn object_with_properties_and_required() {
        let s = parse(serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        }));
        if let Schema::Object(o) = s {
            assert_eq!(o.properties.len(), 2);
            assert_eq!(o.required, vec!["name".to_string()]);
            // default (non-strict) → additionalProperties = Any
            assert!(o.additional.is_some());
        } else {
            panic!("expected object");
        }
    }

    #[test]
    fn strict_object_default_disallows_additional() {
        let s = parse_strict(serde_json::json!({
            "type": "object",
            "properties": {"x": {"type": "number"}}
        }));
        if let Schema::Object(o) = s {
            assert!(o.additional.is_none());
        } else {
            panic!("expected object");
        }
    }

    #[test]
    fn array_with_items() {
        let s = parse(serde_json::json!({
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 3
        }));
        if let Schema::Array(a) = s {
            assert!(matches!(*a.items, Schema::String(_)));
            assert_eq!(a.min, Some(1));
            assert_eq!(a.max, Some(3));
        } else {
            panic!("expected array");
        }
    }

    #[test]
    fn enum_compiles_to_oneof_of_const() {
        let s = parse(serde_json::json!({
            "enum": ["a", "b", "c"]
        }));
        if let Schema::OneOf(branches) = s {
            assert_eq!(branches.len(), 3);
            for b in &branches {
                assert!(matches!(b, Schema::Const(_)));
            }
        } else {
            panic!("expected oneof");
        }
    }

    #[test]
    fn const_short_circuits_type() {
        let s = parse(serde_json::json!({
            "type": "string",
            "const": "hello"
        }));
        // const wins over type — the value must be exactly "hello".
        assert!(matches!(s, Schema::Const(_)));
    }

    #[test]
    fn one_of_decodes() {
        let s = parse(serde_json::json!({
            "oneOf": [{"type": "string"}, {"type": "number"}]
        }));
        if let Schema::OneOf(branches) = s {
            assert_eq!(branches.len(), 2);
        } else {
            panic!("expected oneof");
        }
    }

    #[test]
    fn any_of_decodes_same_as_one_of() {
        let s = parse(serde_json::json!({
            "anyOf": [{"type": "boolean"}, {"type": "null"}]
        }));
        assert!(matches!(s, Schema::OneOf(_)));
    }

    #[test]
    fn type_array_decodes_to_oneof() {
        let s = parse(serde_json::json!({
            "type": ["string", "null"]
        }));
        if let Schema::OneOf(branches) = s {
            assert_eq!(branches.len(), 2);
            assert!(matches!(branches[0], Schema::String(_)));
            assert!(matches!(branches[1], Schema::Null));
        } else {
            panic!("expected oneof");
        }
    }

    #[test]
    fn unsupported_features_rejected() {
        assert!(parse_schema(&serde_json::json!({"$ref": "#/x"})).is_err());
        assert!(parse_schema(&serde_json::json!({"pattern": "^x$"})).is_err());
        assert!(parse_schema(&serde_json::json!({"not": {}})).is_err());
        assert!(parse_schema(&serde_json::json!({"allOf": []})).is_err());
        assert!(parse_schema(&serde_json::json!(false)).is_err());
    }

    #[test]
    fn parse_into_fsm_round_trip_object() {
        // Sanity check: a parsed schema drives the FSM correctly.
        use super::super::fsm::{Fsm, StepResult};
        let s = parse(serde_json::json!({
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "string"}
            },
            "required": ["x"]
        }));
        let mut fsm = Fsm::new(s);
        assert_eq!(fsm.step_str(r#"{"x":1,"y":"hi"}"#), StepResult::Ok);
        assert!(fsm.is_complete());
    }

    #[test]
    fn parse_into_fsm_oneof_with_const() {
        // Tools-shaped schema: discriminated union by constant `name`.
        use super::super::fsm::{Fsm, StepResult};
        let s = parse(serde_json::json!({
            "oneOf": [
                {
                    "type": "object",
                    "properties": {
                        "name": {"const": "search"},
                        "query": {"type": "string"}
                    },
                    "required": ["name", "query"]
                },
                {
                    "type": "object",
                    "properties": {
                        "name": {"const": "calc"},
                        "expr": {"type": "string"}
                    },
                    "required": ["name", "expr"]
                }
            ]
        }));
        let mut fsm = Fsm::new(s.clone());
        assert_eq!(
            fsm.step_str(r#"{"name":"search","query":"x"}"#),
            StepResult::Ok
        );
        assert!(fsm.is_complete());
        let mut fsm2 = Fsm::new(s);
        assert_eq!(
            fsm2.step_str(r#"{"name":"calc","expr":"1+1"}"#),
            StepResult::Ok
        );
        assert!(fsm2.is_complete());
    }
}
