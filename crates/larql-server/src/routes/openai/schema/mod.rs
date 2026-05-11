//! Schema-typed JSON constrained decoding.
//!
//! The pipeline:
//!
//! 1. **AST** ([`ast`]) — typed Schema enum (`Object`, `Array`, `String`,
//!    `Number`, `OneOf`, `Const`, etc.) the FSM walks.
//! 2. **FSM** ([`fsm`]) — character-level state machine that consumes
//!    JSON and rejects anything that diverges from the schema.
//! 3. **Mask** ([`mask`]) — adapter that wraps the FSM into the
//!    `FnMut(&[u32], &mut Vec<f32>)` signature
//!    `larql_inference::generate_constrained` expects.
//!
//! Slices 4.4 and 4.6 add JSON-Schema parsing and tool-call schema
//! synthesis on top of this AST.

pub mod ast;
pub mod fsm;
pub mod mask;
pub mod parser;
pub mod tools;

pub use ast::{ArraySchema, NumberSchema, ObjectSchema, Schema, StringSchema};
pub use fsm::{Fsm, StepResult};
pub use mask::build_mask;
pub use parser::{parse_schema, parse_schema_with, ParseOptions};
pub use tools::{resolve_tool_choice, synth_tools_schema, ToolMode};
