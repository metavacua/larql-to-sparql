//! Schema AST — the typed grammar the FSM walks.
//!
//! Subset chosen to match what OpenAI's structured-outputs and tool
//! schemas use in practice:
//!
//! - `type`: `"object" | "array" | "string" | "number" | "integer" |
//!   "boolean" | "null"` (and the `Schema::Any` catch-all for missing
//!   `type`)
//! - `properties`, `required`, `additionalProperties` on objects
//! - `items`, `minItems`, `maxItems` on arrays
//! - `enum`, `const`, `minLength`, `maxLength` on strings
//! - `minimum`, `maximum`, integer-vs-number distinction on numbers
//! - `oneOf` / `anyOf` (treated identically — first matching branch
//!   wins; OpenAI tool definitions effectively need anyOf semantics
//!   because tool names disambiguate at the const-name field)
//! - `const` at the top level (any JSON literal)
//!
//! Out of scope (for now): `$ref` / `$defs`, `pattern`, `format`,
//! `not`, `if/then/else`, `dependencies`, `allOf`. These can be added
//! incrementally; the FSM design accommodates them as new `Schema`
//! variants without rewriting the existing branches.

use std::collections::BTreeMap;

/// A single schema node. Sized via `Box`-ed children so the recursive
/// variants (Object, Array, OneOf) don't blow up the enum's stack size.
#[derive(Debug, Clone)]
pub enum Schema {
    /// Any structurally-valid JSON value.
    Any,
    /// Any of the listed branches; commit when only one remains viable.
    /// `oneOf` and `anyOf` both decode to this — formal `oneOf` requires
    /// exactly-one match, but for token-level decoding both behave the
    /// same: the FSM commits to whichever branch the model's output lines
    /// up with.
    OneOf(Vec<Schema>),
    /// JSON object (`{...}`).
    Object(ObjectSchema),
    /// JSON array (`[...]`).
    Array(ArraySchema),
    /// JSON string (`"..."`).
    String(StringSchema),
    /// JSON number; with `integer` set, decimal point is rejected.
    Number(NumberSchema),
    /// Literal `true` / `false`.
    Boolean,
    /// Literal `null`.
    Null,
    /// Required exact value — any JSON literal. The FSM serialises this
    /// canonically and matches char-by-char.
    Const(serde_json::Value),
}

impl Schema {
    pub fn object(spec: ObjectSchema) -> Schema {
        Schema::Object(spec)
    }
    pub fn array(items: Schema) -> Schema {
        Schema::Array(ArraySchema {
            items: Box::new(items),
            min: None,
            max: None,
        })
    }
    pub fn string() -> Schema {
        Schema::String(StringSchema::default())
    }
    pub fn number() -> Schema {
        Schema::Number(NumberSchema::default())
    }
    pub fn integer() -> Schema {
        Schema::Number(NumberSchema {
            integer: true,
            ..Default::default()
        })
    }
}

/// Object-typed schema. Property iteration order is `BTreeMap`'s key
/// order, which is stable across runs — the FSM doesn't need a
/// specific order, but determinism makes mask caches reusable.
#[derive(Debug, Clone, Default)]
pub struct ObjectSchema {
    pub properties: BTreeMap<String, Schema>,
    pub required: Vec<String>,
    /// Schema applied to keys not in `properties`. `None` means
    /// `additionalProperties: false` — the FSM rejects unknown keys.
    /// `Some(Schema::Any)` means free-form (the OpenAI default when
    /// the request doesn't pass `additionalProperties`).
    pub additional: Option<Box<Schema>>,
}

impl ObjectSchema {
    /// `{}` — accept any object with any keys / any values. Equivalent
    /// to `{"type": "object"}` with no further constraints.
    pub fn any() -> Self {
        Self {
            properties: BTreeMap::new(),
            required: Vec::new(),
            additional: Some(Box::new(Schema::Any)),
        }
    }

    /// `{"type": "object", "additionalProperties": false}` — empty
    /// strict object.
    pub fn empty_strict() -> Self {
        Self {
            properties: BTreeMap::new(),
            required: Vec::new(),
            additional: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ArraySchema {
    pub items: Box<Schema>,
    pub min: Option<usize>,
    pub max: Option<usize>,
}

#[derive(Debug, Clone, Default)]
pub struct StringSchema {
    /// Restrict to one of these literal strings.
    pub r#enum: Option<Vec<String>>,
    /// Required exact value (overrides `enum` if both set).
    pub r#const: Option<String>,
    pub min_len: Option<usize>,
    pub max_len: Option<usize>,
}

#[derive(Debug, Clone, Default)]
pub struct NumberSchema {
    pub integer: bool,
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
}
