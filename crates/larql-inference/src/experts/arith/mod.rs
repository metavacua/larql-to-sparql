//! Arithmetic Virtual Expert (AVE) — instance #1 of [`VirtualExpert`].
//!
//! Spec: `docs/specs/virtual-experts/arithmetic-virtual-expert.md`.
//!
//! The model is an I/O system, not a calculator: it supplies digit
//! decomposition, an involuntary engagement signal, operand extraction, a
//! magnitude prior and a fluent readout — it structurally cannot supply the
//! serial algorithm. So **fired ⇒ dispatch, always**: the gate fires, the
//! payload is extracted (symbolically or via model rewrite), the ALU computes
//! exactly, and the answer is forced back through the sampler with
//! schedule-end termination. The model's own arithmetic output is consumed
//! only as a verification prior.

pub mod alu;
// force_decode_kquant/_streaming/_backend all take &Tokenizer/&VectorIndex
// in every fn -- native-only.
#[cfg(not(target_arch = "wasm32"))]
pub mod drive;
pub mod extract;
pub mod gate;
pub mod verify;

use serde::Serialize;
#[cfg(not(target_arch = "wasm32"))]
use {
    crate::vindex::generate_kquant_cpu_cached, larql_models::ModelWeights,
    larql_vindex::VectorIndex, tokenizers::Tokenizer,
};

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

use super::virtual_expert::{
    DriveSchedule, ExtractMiss, Fire, ResidualTap, Verdict, VirtualExpert,
};
use alu::{BigInt, Expr};
#[cfg(not(target_arch = "wasm32"))]
use drive::TerminationCause;

/// Compute result plus the operand width that scopes the verify prior.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArithAnswer {
    pub value: BigInt,
    pub max_operand_digits: usize,
}

/// The arithmetic expert: tier-0 symbolic gate always on; tier-1 engagement
/// probe when a per-checkpoint artifact is loaded.
#[derive(Debug, Clone, Default)]
pub struct ArithmeticExpert {
    pub probe: Option<gate::RidgeProbe>,
}

impl ArithmeticExpert {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_probe(probe: gate::RidgeProbe) -> Self {
        ArithmeticExpert { probe: Some(probe) }
    }
}

impl VirtualExpert for ArithmeticExpert {
    type Payload = Expr;
    type Answer = ArithAnswer;

    fn name(&self) -> &'static str {
        "arith"
    }

    fn gate(&self, tap: Option<&ResidualTap>, prompt_text: &str) -> Fire {
        gate::gate(self.probe.as_ref(), tap, prompt_text)
    }

    fn extract(&self, prompt_text: &str, rewrite: Option<&str>) -> Result<Expr, ExtractMiss> {
        match rewrite {
            Some(r) => extract::parse_rewrite(r)
                .ok_or_else(|| ExtractMiss(format!("unparseable rewrite: {:?}", r.trim()))),
            None => extract::find_expression(prompt_text)
                .ok_or_else(|| ExtractMiss("no explicit expression on prompt surface".into())),
        }
    }

    fn compute(&self, payload: &Expr) -> ArithAnswer {
        ArithAnswer {
            value: payload.eval(),
            max_operand_digits: payload.max_operand_digits(),
        }
    }

    fn drive(&self, answer: &ArithAnswer) -> DriveSchedule {
        // Leading space: the answer rides the position the model was about
        // to emit into (typically after "=" or a question mark).
        DriveSchedule {
            text: format!(" {}", answer.value),
        }
    }

    fn verify(&self, answer: &ArithAnswer, native: Option<&str>) -> Verdict {
        verify::magnitude_prior(&answer.value, native, answer.max_operand_digits)
    }
}

/// Which arm of the state machine handled the item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AvePath {
    /// No fire — native path untouched.
    Native,
    /// Fired, symbolic extract, forced decode.
    ForcedExplicit,
    /// Fired, rewrite extract, forced decode.
    ForcedRewrite,
    /// Fired but extraction missed — native + `extract_miss` flag.
    NativeExtractMiss,
}

/// Controller options.
#[derive(Debug, Clone)]
pub struct AveOptions {
    /// Token budget for the native path (no fire / extract miss).
    pub max_native_tokens: usize,
    /// Disguised path: allow the 2-shot rewrite when symbolic extract misses.
    pub enable_rewrite: bool,
    /// Token budget for the rewrite segment (~2× the native answer is the
    /// measured floor; budget is a hard cap, not a target).
    pub rewrite_max_tokens: usize,
}

impl Default for AveOptions {
    fn default() -> Self {
        AveOptions {
            max_native_tokens: 64,
            enable_rewrite: true,
            rewrite_max_tokens: 48,
        }
    }
}

/// Per-item telemetry (mandatory — the A10 lesson: per-item logging turns a
/// rerun into a grep). Every field feeds the batch-level mutual-consistency
/// check.
#[derive(Debug, Clone, Serialize)]
pub struct AveTelemetry {
    pub fire: String,
    pub path: AvePath,
    pub expression: Option<String>,
    pub alu_result: Option<String>,
    pub emitted: String,
    pub termination: String,
    pub verify: String,
    pub flags: Vec<String>,
    /// Tokens spent on the rewrite segment (disguised path only).
    pub rewrite_tokens: usize,
    /// Tokens emitted on the answer segment (forced or native).
    pub answer_tokens: usize,
}

/// Outcome of one controller run.
#[derive(Debug, Clone)]
pub struct AveOutcome {
    pub path: AvePath,
    pub fire: Fire,
    /// Exact ALU answer, when the dispatch path ran.
    pub answer: Option<String>,
    /// Full emitted string (forced schedule or native generation).
    pub emitted: String,
    pub telemetry: AveTelemetry,
}

/// State machine (spec §7) over the CPU Q4_K decode path:
///
/// ```text
/// IDLE → (prompt pass; tap)
///   ├─ no fire ────────────→ NATIVE (untouched)
///   └─ fire (T0|T1) → EXTRACT
///         ├─ symbolic ok ──→ COMPUTE → DRIVE(forced) → TERMINATE → VERIFY? → IDLE
///         ├─ rewrite ok ───→ COMPUTE → DRIVE → TERMINATE → VERIFY? → IDLE
///         └─ extract miss ─→ NATIVE + flag
/// ```
///
/// `tap` is the caller's residual capture for the tier-1 probe (in
/// production a free read off the prompt pass; `None` runs tier-0 only).
/// The verify leg runs only when a native answer happens to exist — the
/// dispatch path never spends tokens producing one (`Verdict::Skipped`).
#[cfg(not(target_arch = "wasm32"))]
pub fn ave_generate_kquant(
    expert: &ArithmeticExpert,
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    index: &VectorIndex,
    prompt: &str,
    tap: Option<&ResidualTap>,
    opts: &AveOptions,
) -> Result<AveOutcome, String> {
    let prompt_ids = tokenizer
        .encode(prompt, true)
        .map_err(|e| format!("tokenize prompt: {e}"))?
        .get_ids()
        .to_vec();

    let fire = expert.gate(tap, prompt);
    if !fire.fired() {
        let (emitted, n) = run_native(weights, tokenizer, index, &prompt_ids, opts);
        return Ok(outcome_native(AvePath::Native, fire, emitted, n, vec![]));
    }

    // EXTRACT: symbolic first (zero tokens), rewrite fallback if enabled.
    let mut rewrite_tokens = 0usize;
    let (expr, path) = match expert.extract(prompt, None) {
        Ok(expr) => (expr, AvePath::ForcedExplicit),
        Err(_) if opts.enable_rewrite => {
            let rp = extract::rewrite_prompt(prompt);
            let rids = tokenizer
                .encode(rp.as_str(), true)
                .map_err(|e| format!("tokenize rewrite prompt: {e}"))?
                .get_ids()
                .to_vec();
            let rew = generate_kquant_cpu_cached(
                weights,
                tokenizer,
                &rids,
                opts.rewrite_max_tokens,
                index,
            );
            rewrite_tokens = rew.len();
            let rew_text: String = rew.iter().map(|(t, _)| t.as_str()).collect();
            match expert.extract(prompt, Some(&rew_text)) {
                Ok(expr) => (expr, AvePath::ForcedRewrite),
                Err(miss) => {
                    let (emitted, n) = run_native(weights, tokenizer, index, &prompt_ids, opts);
                    let mut out = outcome_native(
                        AvePath::NativeExtractMiss,
                        fire,
                        emitted,
                        n,
                        vec!["extract_miss".into(), miss.0],
                    );
                    out.telemetry.rewrite_tokens = rewrite_tokens;
                    return Ok(out);
                }
            }
        }
        Err(miss) => {
            let (emitted, n) = run_native(weights, tokenizer, index, &prompt_ids, opts);
            return Ok(outcome_native(
                AvePath::NativeExtractMiss,
                fire,
                emitted,
                n,
                vec!["extract_miss".into(), miss.0],
            ));
        }
    };

    // COMPUTE → DRIVE(forced, schedule) → TERMINATE.
    let answer = expert.compute(&expr);
    let schedule = expert.drive(&answer);
    let schedule_ids = schedule.forced_ids(tokenizer);
    let fd = drive::force_decode_kquant(weights, tokenizer, index, &prompt_ids, &schedule_ids);

    // VERIFY?: no native answer was produced on this path — prior skipped.
    let verdict = expert.verify(&answer, None);

    let mut flags = Vec::new();
    if matches!(fd.cause, TerminationCause::EarlyStop { .. }) {
        flags.push("early_stop".into());
    }
    if let Verdict::Suspect(_) = &verdict {
        flags.push("extract_suspect".into());
    }

    let telemetry = AveTelemetry {
        fire: fire.label(),
        path,
        expression: Some(expr.to_string()),
        alu_result: Some(answer.value.to_string()),
        emitted: fd.emitted.clone(),
        termination: fd.cause.label(),
        verify: verdict.label(),
        flags,
        rewrite_tokens,
        answer_tokens: fd.ids.len(),
    };
    Ok(AveOutcome {
        path,
        fire,
        answer: Some(answer.value.to_string()),
        emitted: fd.emitted,
        telemetry,
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn run_native(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    index: &VectorIndex,
    prompt_ids: &[u32],
    opts: &AveOptions,
) -> (String, usize) {
    let out = generate_kquant_cpu_cached(
        weights,
        tokenizer,
        prompt_ids,
        opts.max_native_tokens,
        index,
    );
    let n = out.len();
    (out.into_iter().map(|(t, _)| t).collect(), n)
}

#[cfg(not(target_arch = "wasm32"))]
fn outcome_native(
    path: AvePath,
    fire: Fire,
    emitted: String,
    answer_tokens: usize,
    flags: Vec<String>,
) -> AveOutcome {
    AveOutcome {
        path,
        fire,
        answer: None,
        emitted: emitted.clone(),
        telemetry: AveTelemetry {
            fire: fire.label(),
            path,
            expression: None,
            alu_result: None,
            emitted,
            termination: "native".into(),
            verify: Verdict::Skipped.label(),
            flags,
            rewrite_tokens: 0,
            answer_tokens,
        },
    }
}

/// Batch-level mutual-consistency check (spec §7): the controller should
/// assert `fleet ≈ fire·dispatch + (1−fire)·native` and alarm on violation —
/// table arithmetic is a control surface. Returns the residual.
pub fn decomposition_residual(
    fleet_accuracy: f64,
    fire_rate: f64,
    dispatch_accuracy: f64,
    native_accuracy_unfired: f64,
) -> f64 {
    fleet_accuracy - (fire_rate * dispatch_accuracy + (1.0 - fire_rate) * native_accuracy_unfired)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{
        make_test_q4k_vindex, make_test_q4k_weights, synthetic_tokenizer_json,
    };

    /// Fixture tokenizer with `[UNK]` mapped to id 0 (a real vocab slot), so
    /// free-text prompts — which the WordLevel fixture can't represent —
    /// still encode to in-range ids and the forward pass runs.
    fn fixture() -> (ModelWeights, VectorIndex, Tokenizer) {
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let tokenizer =
            Tokenizer::from_bytes(synthetic_tokenizer_json(weights.vocab_size).as_bytes())
                .expect("fixture tokenizer");
        (weights, index, tokenizer)
    }

    /// Probe that fires on any matching tap (threshold below any score).
    fn always_fire_probe(dim: usize) -> gate::RidgeProbe {
        gate::RidgeProbe {
            model: "fixture".into(),
            layer: 8,
            weights: vec![0.0; dim],
            bias: 1.0,
            threshold: 0.5,
        }
    }

    #[test]
    fn expert_trait_explicit_pipeline_is_exact() {
        let ave = ArithmeticExpert::new();
        assert_eq!(ave.name(), "arith");
        let fire = ave.gate(None, "123456 + 654321 =");
        assert_eq!(fire, Fire::Tier0);
        let expr = ave.extract("123456 + 654321 =", None).expect("extract");
        let answer = ave.compute(&expr);
        assert_eq!(answer.value.to_string(), "777777");
        assert_eq!(answer.max_operand_digits, 6);
        assert_eq!(ave.drive(&answer).text, " 777777");
        assert_eq!(ave.verify(&answer, Some("777,777")), Verdict::Consistent);
        assert_eq!(ave.verify(&answer, None), Verdict::Skipped);
    }

    #[test]
    fn extract_miss_carries_a_reason() {
        let ave = ArithmeticExpert::new();
        let miss = ave.extract("no math here", None).expect_err("miss");
        assert!(miss.0.contains("no explicit expression"));
        let miss = ave
            .extract("ignored", Some("I will not rewrite that"))
            .expect_err("miss");
        assert!(miss.0.contains("unparseable rewrite"));
    }

    #[test]
    fn controller_no_fire_takes_native_path() {
        let (mut weights, index, tokenizer) = fixture();
        let ave = ArithmeticExpert::new();
        let opts = AveOptions {
            max_native_tokens: 3,
            ..AveOptions::default()
        };
        // "[1] [2]" carries digits but no operator — must not fire.
        let out = ave_generate_kquant(
            &ave,
            &mut weights,
            &tokenizer,
            &index,
            "[1] [2]",
            None,
            &opts,
        )
        .expect("run");
        assert_eq!(out.path, AvePath::Native);
        assert_eq!(out.fire, Fire::No);
        assert!(out.answer.is_none());
        assert_eq!(out.telemetry.termination, "native");
    }

    #[test]
    fn controller_tier1_fire_with_unparseable_rewrite_flags_extract_miss() {
        let (mut weights, index, tokenizer) = fixture();
        let dim = weights.hidden_size;
        let ave = ArithmeticExpert::with_probe(always_fire_probe(dim));
        let tap = ResidualTap::single(8, vec![0.0; dim]);
        let opts = AveOptions {
            max_native_tokens: 2,
            rewrite_max_tokens: 2,
            enable_rewrite: true,
        };
        // No explicit expression; probe fires; the fixture model's rewrite
        // output ("[N]" tokens) is unparseable → NATIVE + extract_miss flag.
        let out = ave_generate_kquant(
            &ave,
            &mut weights,
            &tokenizer,
            &index,
            "[1] [2]",
            Some(&tap),
            &opts,
        )
        .expect("run");
        assert_eq!(out.path, AvePath::NativeExtractMiss);
        assert!(matches!(out.fire, Fire::Tier1(_)));
        assert!(out.telemetry.flags.iter().any(|f| f == "extract_miss"));
        assert!(out.telemetry.rewrite_tokens <= 2);
    }

    #[test]
    fn controller_tier1_fire_with_rewrite_disabled_falls_native() {
        let (mut weights, index, tokenizer) = fixture();
        let dim = weights.hidden_size;
        let ave = ArithmeticExpert::with_probe(always_fire_probe(dim));
        let tap = ResidualTap::single(8, vec![0.0; dim]);
        let opts = AveOptions {
            max_native_tokens: 2,
            enable_rewrite: false,
            ..AveOptions::default()
        };
        let out = ave_generate_kquant(
            &ave,
            &mut weights,
            &tokenizer,
            &index,
            "[1] [2]",
            Some(&tap),
            &opts,
        )
        .expect("run");
        assert_eq!(out.path, AvePath::NativeExtractMiss);
        assert_eq!(out.telemetry.rewrite_tokens, 0);
    }

    #[test]
    fn controller_explicit_fire_forces_the_schedule() {
        let (mut weights, index, tokenizer) = fixture();
        let ave = ArithmeticExpert::new();
        // Tier-0 fires on the prompt text; the WordLevel fixture tokenizer
        // can't encode " 19", so the schedule is empty — the state machine
        // still walks COMPUTE → DRIVE → TERMINATE and reports exactly that.
        let out = ave_generate_kquant(
            &ave,
            &mut weights,
            &tokenizer,
            &index,
            "12 + 7 =",
            None,
            &AveOptions::default(),
        )
        .expect("run");
        assert_eq!(out.path, AvePath::ForcedExplicit);
        assert_eq!(out.fire, Fire::Tier0);
        assert_eq!(out.answer.as_deref(), Some("19"));
        assert_eq!(out.telemetry.expression.as_deref(), Some("12 + 7"));
        assert_eq!(out.telemetry.termination, "schedule_end");
        assert_eq!(out.telemetry.verify, "skipped");
    }

    #[test]
    fn decomposition_residual_is_zero_when_table_is_consistent() {
        // fleet = fire·dispatch + (1−fire)·native exactly.
        let r = decomposition_residual(0.92, 0.9, 1.0, 0.2);
        assert!(r.abs() < 1e-12, "residual {r}");
        // And alarms (nonzero) when the table is inconsistent.
        assert!(decomposition_residual(0.5, 0.9, 1.0, 0.2).abs() > 0.1);
    }

    #[test]
    fn telemetry_serializes_for_per_item_logs() {
        let t = AveTelemetry {
            fire: "tier0".into(),
            path: AvePath::ForcedExplicit,
            expression: Some("12 + 7".into()),
            alu_result: Some("19".into()),
            emitted: " 19".into(),
            termination: "schedule_end".into(),
            verify: "skipped".into(),
            flags: vec![],
            rewrite_tokens: 0,
            answer_tokens: 2,
        };
        let json = serde_json::to_string(&t).expect("json");
        assert!(json.contains("\"forced_explicit\""));
        assert!(json.contains("schedule_end"));
    }
}
