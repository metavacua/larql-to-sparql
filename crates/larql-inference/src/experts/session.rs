//! `ExpertSession` — high-level glue between a dispatcher (typically
//! [`ExpertRegistry`]) and a generation loop.
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
//!
//! ## Dispatcher trait
//!
//! [`ExpertSession`] is generic over [`Dispatcher`] (default [`ExpertRegistry`])
//! so unit tests can substitute a mock without instantiating the WASM runtime.
//! The trait surface is intentionally minimal — `ops()` + `call()` — to keep
//! the contract narrow and easy to satisfy.

use serde_json::Value;

use crate::experts::caller::{ExpertResult, OpSpec};
use crate::experts::parser::{parse_op_call, OpCall};
use crate::experts::registry::ExpertRegistry;
use crate::prompt::ChatTemplate;

/// Minimal contract a session needs from its op-dispatch backend.
///
/// Implemented for [`ExpertRegistry`] so production code uses real WASM
/// experts; tests can implement it on a struct with hard-coded responses
/// to avoid loading WASM.
pub trait Dispatcher {
    /// Every (op, args-schema) pair this dispatcher can handle. Used to
    /// render prompts that tell the model the exact argument keys per op.
    fn op_specs(&self) -> Vec<OpSpec>;

    /// Invoke `op` with `args`. Returns `None` if the op is unknown to the
    /// dispatcher OR the underlying expert declined the call.
    fn call(&mut self, op: &str, args: &Value) -> Option<ExpertResult>;
}

impl Dispatcher for ExpertRegistry {
    fn op_specs(&self) -> Vec<OpSpec> {
        ExpertRegistry::op_specs(self).into_iter().cloned().collect()
    }

    fn call(&mut self, op: &str, args: &Value) -> Option<ExpertResult> {
        ExpertRegistry::call(self, op, args)
    }
}

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

/// High-level session orchestrating prompt construction + dispatch over a
/// [`Dispatcher`] (defaults to [`ExpertRegistry`]).
pub struct ExpertSession<D = ExpertRegistry>
where
    D: Dispatcher,
{
    registry: D,
}

impl<D: Dispatcher> ExpertSession<D> {
    /// Wrap a dispatcher. The session takes ownership; use
    /// [`Self::registry_mut`] for low-level access.
    pub fn new(registry: D) -> Self {
        Self { registry }
    }

    pub fn registry(&self) -> &D {
        &self.registry
    }

    pub fn registry_mut(&mut self) -> &mut D {
        &mut self.registry
    }

    pub fn into_registry(self) -> D {
        self.registry
    }

    /// Build a model-agnostic system prompt enumerating every advertised op
    /// along with its argument keys, e.g. `gcd(a, b)`.
    ///
    /// The format is deterministic (ops sorted alphabetically) so identical
    /// registries produce byte-identical prompts — that matters for prompt
    /// caching and reproducible benchmarking.
    pub fn system_prompt(&self) -> String {
        let mut specs = self.registry.op_specs();
        specs.sort_by(|a, b| a.name.cmp(&b.name));

        let mut out = String::new();
        out.push_str("You are a tool-using assistant. When the user's request \
                      can be solved by exactly one of the ops below, respond \
                      with a single JSON object and nothing else:\n");
        out.push_str("  {\"op\":\"<op_name>\",\"args\":{...}}\n\n");
        out.push_str("Available ops:\n");
        for spec in &specs {
            out.push_str("  - ");
            out.push_str(&spec.name);
            out.push('(');
            for (i, arg) in spec.args.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(arg);
            }
            out.push_str(")\n");
        }
        out.push_str("\nRules:\n");
        out.push_str("  - Emit the JSON object only. No prose, no code fences, no commentary.\n");
        out.push_str("  - Use exact op names from the list above.\n");
        out.push_str("  - The keys inside `args` MUST match the parameter names in parentheses.\n");
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

        let known = self
            .registry
            .op_specs()
            .iter()
            .any(|s| s.name == call.op);
        if !known {
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

/// Mock-backed tests — exercise [`ExpertSession`] without WASM. These run
/// unconditionally on every `cargo test` so the session contract is covered
/// even on machines that haven't built the larql-experts WASM modules.
#[cfg(test)]
mod mock_tests {
    use super::*;

    /// Hand-rolled [`Dispatcher`] that records calls and returns canned
    /// responses keyed by op name. Lives in tests to avoid leaking a mock
    /// trait implementation into the public API.
    struct MockDispatcher {
        ops: Vec<OpSpec>,
        /// op → response. `None` means the dispatcher will refuse the call
        /// (modelling an expert that declined).
        responses: std::collections::HashMap<String, Option<ExpertResult>>,
        calls: std::cell::RefCell<Vec<(String, Value)>>,
    }

    impl MockDispatcher {
        /// Construct from `[(op_name, [arg_keys])]` pairs.
        fn new(ops: &[(&str, &[&str])]) -> Self {
            Self {
                ops: ops
                    .iter()
                    .map(|(name, args)| OpSpec {
                        name: (*name).to_string(),
                        args: args.iter().map(|a| (*a).to_string()).collect(),
                    })
                    .collect(),
                responses: std::collections::HashMap::new(),
                calls: std::cell::RefCell::new(Vec::new()),
            }
        }

        fn with_response(mut self, op: &str, value: Value) -> Self {
            self.responses.insert(
                op.to_string(),
                Some(ExpertResult {
                    value,
                    confidence: 1.0,
                    latency_ns: 0,
                    expert_id: "mock".into(),
                    op: op.to_string(),
                }),
            );
            self
        }

        fn with_decline(mut self, op: &str) -> Self {
            self.responses.insert(op.to_string(), None);
            self
        }

        fn calls(&self) -> Vec<(String, Value)> {
            self.calls.borrow().clone()
        }
    }

    impl Dispatcher for MockDispatcher {
        fn op_specs(&self) -> Vec<OpSpec> {
            self.ops.clone()
        }

        fn call(&mut self, op: &str, args: &Value) -> Option<ExpertResult> {
            self.calls.borrow_mut().push((op.to_string(), args.clone()));
            self.responses.get(op).cloned().flatten()
        }
    }

    #[test]
    fn system_prompt_is_deterministic_with_mock() {
        let mock = MockDispatcher::new(&["b_op", "a_op"]);
        let session = ExpertSession::new(mock);
        let a = session.system_prompt();
        let b = session.system_prompt();
        assert_eq!(a, b);
    }

    #[test]
    fn system_prompt_lists_provided_ops_sorted() {
        // Mock returns ops out-of-order — system_prompt must sort them.
        let mock = MockDispatcher::new(&["zzz", "aaa", "mmm"]);
        let session = ExpertSession::new(mock);
        let p = session.system_prompt();
        let aaa = p.find("aaa").expect("aaa missing");
        let mmm = p.find("mmm").expect("mmm missing");
        let zzz = p.find("zzz").expect("zzz missing");
        assert!(aaa < mmm && mmm < zzz, "ops should appear in alphabetical order");
    }

    #[test]
    fn system_prompt_handles_empty_op_list() {
        let mock = MockDispatcher::new(&[]);
        let session = ExpertSession::new(mock);
        let p = session.system_prompt();
        assert!(p.contains("Available ops:"), "header missing:\n{p}");
        // No bulleted op lines between header and Rules.
        let between: Vec<&str> = p
            .lines()
            .skip_while(|l| !l.starts_with("Available ops:"))
            .skip(1)
            .take_while(|l| !l.is_empty())
            .collect();
        assert!(between.is_empty(), "expected no ops, got: {between:?}");
    }

    #[test]
    fn build_prompt_with_gemma_template_includes_system_and_user() {
        let mock = MockDispatcher::new(&["foo"]);
        let session = ExpertSession::new(mock);
        let wrapped = session.build_prompt("hello", ChatTemplate::Gemma);
        assert!(wrapped.starts_with("<start_of_turn>user\n"));
        assert!(wrapped.contains("Available ops:"));
        assert!(wrapped.contains("foo"));
        assert!(wrapped.contains("hello"));
        assert!(wrapped.ends_with("<start_of_turn>model\n"));
    }

    #[test]
    fn build_prompt_with_each_template_variant_round_trips() {
        let mock = MockDispatcher::new(&["x"]);
        let session = ExpertSession::new(mock);
        for tpl in [
            ChatTemplate::Gemma,
            ChatTemplate::Mistral,
            ChatTemplate::Llama,
            ChatTemplate::ChatML,
            ChatTemplate::Plain,
        ] {
            let wrapped = session.build_prompt("Q?", tpl);
            assert!(wrapped.contains("Q?"), "template {} dropped user prompt", tpl.name());
            assert!(wrapped.contains("x"), "template {} dropped op list", tpl.name());
        }
    }

    #[test]
    fn dispatch_happy_path_with_mock() {
        let mock = MockDispatcher::new(&["gcd"]).with_response("gcd", serde_json::json!(12));
        let mut session = ExpertSession::new(mock);
        let out = session
            .dispatch(r#"{"op":"gcd","args":{"a":144,"b":60}}"#)
            .expect("dispatch");
        assert_eq!(out.call.op, "gcd");
        assert_eq!(out.result.value, serde_json::json!(12));
        assert_eq!(out.result.expert_id, "mock");
    }

    #[test]
    fn dispatch_no_op_call_with_mock() {
        let mock = MockDispatcher::new(&["gcd"]);
        let mut session = ExpertSession::new(mock);
        let err = session.dispatch("plain text, no JSON").unwrap_err();
        assert_eq!(err, DispatchSkip::NoOpCall);
    }

    #[test]
    fn dispatch_unknown_op_with_mock() {
        let mock = MockDispatcher::new(&["gcd"]);
        let mut session = ExpertSession::new(mock);
        let err = session
            .dispatch(r#"{"op":"unknown_op","args":{}}"#)
            .unwrap_err();
        assert_eq!(err, DispatchSkip::UnknownOp("unknown_op".into()));
    }

    #[test]
    fn dispatch_expert_declined_with_mock() {
        let mock = MockDispatcher::new(&["gcd"]).with_decline("gcd");
        let mut session = ExpertSession::new(mock);
        let err = session
            .dispatch(r#"{"op":"gcd","args":{"a":1}}"#)
            .unwrap_err();
        assert!(matches!(
            err,
            DispatchSkip::ExpertDeclined { ref op, .. } if op == "gcd"
        ));
    }

    #[test]
    fn dispatch_forwards_args_verbatim_to_dispatcher() {
        // Verify that whatever JSON args the parser produces are passed
        // through unchanged to the dispatcher.
        let mock = MockDispatcher::new(&["echo"]).with_response("echo", serde_json::json!(true));
        let mut session = ExpertSession::new(mock);
        let _ = session
            .dispatch(r#"{"op":"echo","args":{"nested":{"k":[1,2,3]},"s":"日本語"}}"#)
            .expect("dispatch");

        let calls = session.registry().calls();
        assert_eq!(calls.len(), 1);
        let (op, args) = &calls[0];
        assert_eq!(op, "echo");
        assert_eq!(args["nested"]["k"], serde_json::json!([1, 2, 3]));
        assert_eq!(args["s"], serde_json::json!("日本語"));
    }
}
