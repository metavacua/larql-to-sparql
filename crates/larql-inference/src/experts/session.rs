//! `ExpertSession` — high-level glue between an [`ExpertRegistry`] and a
//! generation loop.
//!
//! Three responsibilities, kept independent so each can be tested in
//! isolation and composed however the caller likes:
//!
//!   1. [`ExpertSession::system_prompt`] — build a model-agnostic system prompt
//!      that enumerates available ops + their argument keys.
//!   2. [`ExpertSession::build_prompt`] — wrap a user prompt with the system
//!      prompt + a [`ChatTemplate`], producing a string ready for the
//!      tokenizer.
//!   3. [`ExpertSession::dispatch`] — parse [`OpCall`] JSON out of free-form
//!      model output and dispatch through the registry.
//!
//! The session does *not* own the generation loop — pass it model output and
//! it returns a [`DispatchOutcome`]. This keeps the session usable from any
//! decode path (CPU, Metal, remote, mock) without coupling.

use serde_json::Value;

use crate::experts::caller::ExpertResult;
use crate::experts::parser::{parse_op_call, OpCall};
use crate::experts::registry::ExpertRegistry;
use crate::prompt::ChatTemplate;

/// Result of a successful expert dispatch.
#[derive(Debug, Clone)]
pub struct DispatchOutcome {
    /// The op-call extracted from model output.
    pub call: OpCall,
    /// The expert's response.
    pub result: ExpertResult,
}

/// Reasons a dispatch attempt produced no [`DispatchOutcome`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchSkip {
    /// Model output contained no parseable `{"op":"...","args":{...}}` block.
    NoOpCall,
    /// An op-call was extracted but no loaded expert advertises that op.
    UnknownOp(String),
    /// The expert was found but declined the call (bad args, runtime error).
    ExpertDeclined { op: String, args: Value },
}

/// High-level session orchestrating prompt construction + dispatch over an
/// [`ExpertRegistry`].
pub struct ExpertSession {
    registry: ExpertRegistry,
}

impl ExpertSession {
    /// Wrap a registry. The session takes ownership; use
    /// [`Self::registry_mut`] for low-level access.
    pub fn new(registry: ExpertRegistry) -> Self {
        Self { registry }
    }

    pub fn registry(&self) -> &ExpertRegistry {
        &self.registry
    }

    pub fn registry_mut(&mut self) -> &mut ExpertRegistry {
        &mut self.registry
    }

    pub fn into_registry(self) -> ExpertRegistry {
        self.registry
    }

    /// Build a model-agnostic system prompt enumerating every advertised op.
    ///
    /// The format is deterministic (ops sorted alphabetically) so identical
    /// registries produce byte-identical prompts — that matters for prompt
    /// caching and reproducible benchmarking.
    pub fn system_prompt(&self) -> String {
        let mut ops: Vec<&str> = self.registry.ops();
        ops.sort_unstable();

        let mut out = String::new();
        out.push_str("You are a tool-using assistant. When the user's request \
                      can be solved by exactly one of the ops below, respond \
                      with a single JSON object and nothing else:\n");
        out.push_str("  {\"op\":\"<op_name>\",\"args\":{...}}\n\n");
        out.push_str("Available ops:\n");
        for op in ops {
            out.push_str("  - ");
            out.push_str(op);
            out.push('\n');
        }
        out.push_str("\nRules:\n");
        out.push_str("  - Emit the JSON object only. No prose, no code fences, no commentary.\n");
        out.push_str("  - Use exact op names from the list above.\n");
        out.push_str("  - All argument values must be JSON literals (numbers, strings, arrays, objects).\n");
        out
    }

    /// Build a complete prompt: `<system>\n\n<user>`, then wrapped by `template`.
    pub fn build_prompt(&self, user_prompt: &str, template: ChatTemplate) -> String {
        let combined = format!("{}\n\n{user_prompt}", self.system_prompt());
        template.wrap(&combined)
    }

    /// Parse + dispatch a single op-call from `model_output`.
    ///
    /// Returns `Ok(outcome)` when an op-call was extracted and the registry
    /// returned a result. Returns `Err(reason)` for the three skip paths so
    /// callers can decide whether to retry, log, or fall back.
    pub fn dispatch(&mut self, model_output: &str) -> Result<DispatchOutcome, DispatchSkip> {
        let call = parse_op_call(model_output).ok_or(DispatchSkip::NoOpCall)?;

        if !self.registry.ops().iter().any(|o| **o == call.op) {
            return Err(DispatchSkip::UnknownOp(call.op));
        }

        match self.registry.call(&call.op, &call.args) {
            Some(result) => Ok(DispatchOutcome { call, result }),
            None => Err(DispatchSkip::ExpertDeclined {
                op: call.op,
                args: call.args,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn wasm_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../larql-experts/target/wasm32-wasip1/release")
    }

    fn registry_or_skip() -> Option<ExpertRegistry> {
        let dir = wasm_dir();
        if !dir.exists() {
            eprintln!("skip: wasm dir missing at {} — run `cargo build --target wasm32-wasip1 --release` in larql-experts", dir.display());
            return None;
        }
        ExpertRegistry::load_dir(&dir).ok()
    }

    #[test]
    fn system_prompt_is_deterministic() {
        let Some(reg) = registry_or_skip() else { return };
        let session = ExpertSession::new(reg);
        let a = session.system_prompt();
        let b = session.system_prompt();
        assert_eq!(a, b, "system prompt must be deterministic");
    }

    #[test]
    fn system_prompt_lists_known_ops() {
        let Some(reg) = registry_or_skip() else { return };
        let session = ExpertSession::new(reg);
        let p = session.system_prompt();
        // Sample a handful of ops we know exist across the workspace.
        assert!(p.contains("gcd"), "system prompt missing 'gcd':\n{p}");
        assert!(p.contains("is_prime"), "system prompt missing 'is_prime':\n{p}");
        assert!(p.contains("base64_encode"), "system prompt missing 'base64_encode':\n{p}");
    }

    #[test]
    fn system_prompt_ops_are_sorted() {
        let Some(reg) = registry_or_skip() else { return };
        let session = ExpertSession::new(reg);
        let p = session.system_prompt();

        // Pull the lines between "Available ops:" and the following blank
        // line (the Rules section also uses bulleted lines, so a naive prefix
        // strip would conflate ops with rules).
        let ops: Vec<&str> = p
            .lines()
            .skip_while(|l| !l.starts_with("Available ops:"))
            .skip(1)
            .take_while(|l| !l.is_empty())
            .filter_map(|l| l.strip_prefix("  - "))
            .collect();
        assert!(!ops.is_empty(), "expected ops list to be non-empty");

        let mut sorted = ops.clone();
        sorted.sort_unstable();
        assert_eq!(ops, sorted, "ops in system prompt must be sorted");
    }

    #[test]
    fn build_prompt_wraps_via_template() {
        let Some(reg) = registry_or_skip() else { return };
        let session = ExpertSession::new(reg);
        let wrapped = session.build_prompt("What is 2+2?", ChatTemplate::Gemma);
        assert!(wrapped.starts_with("<start_of_turn>user\n"));
        assert!(wrapped.contains("What is 2+2?"));
        assert!(wrapped.contains("Available ops:"));
        assert!(wrapped.ends_with("<start_of_turn>model\n"));
    }

    #[test]
    fn build_prompt_plain_template_passes_through_unwrapped() {
        let Some(reg) = registry_or_skip() else { return };
        let session = ExpertSession::new(reg);
        let wrapped = session.build_prompt("hi", ChatTemplate::Plain);
        // No template tags injected.
        assert!(!wrapped.contains("<start_of_turn>"));
        assert!(!wrapped.contains("[INST]"));
        // System prompt + user is present.
        assert!(wrapped.contains("Available ops:"));
        assert!(wrapped.ends_with("hi"));
    }

    #[test]
    fn dispatch_happy_path_returns_outcome() {
        let Some(reg) = registry_or_skip() else { return };
        let mut session = ExpertSession::new(reg);
        let out = session
            .dispatch(r#"{"op":"gcd","args":{"a":144,"b":60}}"#)
            .expect("dispatch");
        assert_eq!(out.call.op, "gcd");
        assert_eq!(out.result.value, serde_json::json!(12));
        assert_eq!(out.result.expert_id, "arithmetic");
    }

    #[test]
    fn dispatch_with_preamble_still_finds_call() {
        let Some(reg) = registry_or_skip() else { return };
        let mut session = ExpertSession::new(reg);
        let raw = "Sure, here is the call:\n{\"op\":\"is_prime\",\"args\":{\"n\":97}}\n";
        let out = session.dispatch(raw).expect("dispatch");
        assert_eq!(out.call.op, "is_prime");
        assert_eq!(out.result.value, serde_json::json!(true));
    }

    #[test]
    fn dispatch_no_op_call_returns_no_op_call_skip() {
        let Some(reg) = registry_or_skip() else { return };
        let mut session = ExpertSession::new(reg);
        let err = session.dispatch("just a free-text answer").unwrap_err();
        assert_eq!(err, DispatchSkip::NoOpCall);
    }

    #[test]
    fn dispatch_unknown_op_returns_unknown_op_skip() {
        let Some(reg) = registry_or_skip() else { return };
        let mut session = ExpertSession::new(reg);
        let err = session
            .dispatch(r#"{"op":"definitely_not_a_real_op","args":{}}"#)
            .unwrap_err();
        assert_eq!(err, DispatchSkip::UnknownOp("definitely_not_a_real_op".into()));
    }

    #[test]
    fn dispatch_expert_declined_returns_expert_declined_skip() {
        // arithmetic.gcd requires {a, b} — pass garbage to provoke a decline.
        let Some(reg) = registry_or_skip() else { return };
        let mut session = ExpertSession::new(reg);
        let err = session
            .dispatch(r#"{"op":"gcd","args":{"unrelated":42}}"#)
            .unwrap_err();
        assert!(matches!(err, DispatchSkip::ExpertDeclined { ref op, .. } if op == "gcd"));
    }
}
