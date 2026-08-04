//! Pure logic for `larql dec-bench drift` — the C6 wire-fidelity gate
//! (docs/dec-funnel.md §3 DEC-1A: "i8/Q8K arms run `larql shannon
//! verify`-style bits/char scoring vs the f32-wire baseline; drift gate
//! pre-set at 0.5%").
//!
//! Everything transport-shaped lives in `drift_runtime.rs`; this module owns
//! the scoring math (bits from raw logits, mirroring `larql shannon`), the
//! arm plan (baseline-first, deduped), the drift/gate arithmetic, and the
//! output records (JSON run record + `dec/*` pulse lines).
//!
//! DETERMINISM: the number is a teacher-forced NLL over a pinned corpus —
//! exact math given weights + wire, no sampling, no timing in the metric.
//! Thermal state, battery, and machine load CANNOT move it, so this
//! instrument is battery-safe (unlike the `dec-bench replay` / `bench`
//! timing instruments, which need the cool-machine protocol).

use serde::Serialize;

use super::replay::{WireArm, WireSpec};
use larql_inference::{WireFormat, WirePreference};

/// Pre-registered C6 gate (docs/dec-funnel.md §3 DEC-1A), matching the
/// shannon-verify CI threshold.
pub const DEFAULT_DRIFT_GATE_PCT: f64 = 0.5;

const LN_2: f64 = std::f64::consts::LN_2;

// ── Scoring math ─────────────────────────────────────────────────────────────

/// `-log2 p(target)` from one raw lm_head score row, applying the arch's
/// logits scaling and final softcap. Mirrors `larql shannon`'s
/// `bits_for_raw_row` so the drift number is a shannon-style bits/char.
pub fn bits_for_target(
    raw: &[f32],
    target: usize,
    logits_scaling: f32,
    softcap: Option<f32>,
) -> Result<f64, String> {
    if target >= raw.len() {
        return Err(format!(
            "target token {target} out of vocab (len {})",
            raw.len()
        ));
    }
    let inv_scale = 1.0 / logits_scaling;
    let transform = |v: f32| {
        let mut logit = v * inv_scale;
        if let Some(cap) = softcap {
            logit = (logit / cap).tanh() * cap;
        }
        logit
    };

    let max_logit = raw
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .map(transform)
        .fold(None, |acc: Option<f32>, v| {
            Some(acc.map_or(v, |m| m.max(v)))
        })
        .ok_or_else(|| "all logits were non-finite".to_string())?;

    let exp_sum: f64 = raw
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .map(|v| ((transform(v) - max_logit) as f64).exp())
        .sum();
    if !raw[target].is_finite() {
        return Err(format!("target token {target} has a non-finite logit"));
    }
    let target_logit = transform(raw[target]);
    let logsumexp = max_logit as f64 + exp_sum.ln();
    Ok((logsumexp - target_logit as f64) / LN_2)
}

// ── Corpus handling ──────────────────────────────────────────────────────────

/// Strip CR (CRLF → LF), matching `larql shannon verify`'s cross-engine
/// tokenization normalization, then truncate to `limit_bytes` on a UTF-8
/// boundary when set.
pub fn normalize_corpus(raw: &str, limit_bytes: Option<usize>) -> String {
    let mut text: String = raw.chars().filter(|c| *c != '\r').collect();
    if let Some(limit) = limit_bytes {
        if text.len() > limit {
            let mut end = limit;
            while end > 0 && !text.is_char_boundary(end) {
                end -= 1;
            }
            text.truncate(end);
        }
    }
    text
}

// ── Arm plan ─────────────────────────────────────────────────────────────────

/// How a [`WireSpec`] arm maps onto a production client connection: plain
/// arms keep the historical wire (f32 request frames, `Accept` per arm),
/// pairs set the directions independently, and the q8k arm rides its own
/// endpoint via the q8k dispatch path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArmConnection {
    pub wire_in: WireFormat,
    pub wire_out: WirePreference,
    pub use_q8k: bool,
}

pub fn arm_connection(spec: WireSpec) -> ArmConnection {
    match spec {
        WireSpec::Plain(WireArm::Q8k) => ArmConnection {
            wire_in: WireFormat::F32,
            wire_out: WirePreference::F32,
            use_q8k: true,
        },
        WireSpec::Plain(arm) => ArmConnection {
            wire_in: WireFormat::F32,
            wire_out: match arm {
                WireArm::F32 => WirePreference::F32,
                WireArm::F16 => WirePreference::F16,
                WireArm::I8 => WirePreference::I8,
                WireArm::Q8k => unreachable!("handled above"),
            },
            use_q8k: false,
        },
        WireSpec::Pair { input, output } => ArmConnection {
            wire_in: input,
            wire_out: match output {
                WireFormat::F32 => WirePreference::F32,
                WireFormat::F16 => WirePreference::F16,
                WireFormat::I8 => WirePreference::I8,
            },
            use_q8k: false,
        },
    }
}

/// The scoring plan: the f32/f32 baseline ALWAYS runs first (in-run
/// baseline — never a stored one: same machine, same run), followed by the
/// requested arms in order, deduplicated. A requested plain `f32` is
/// satisfied by the baseline itself.
pub fn drift_plan(requested: &[WireSpec]) -> Vec<WireSpec> {
    let baseline = WireSpec::Plain(WireArm::F32);
    let mut plan = vec![baseline];
    for &spec in requested {
        if !plan.contains(&spec) {
            plan.push(spec);
        }
    }
    plan
}

/// Human label for a direction's codec fidelity. Labels only — the numbers
/// speak; these just remind the reader which arms are lossy in which
/// direction.
pub fn fidelity_label(dtype: &str) -> &'static str {
    match dtype {
        "f32" => "exact",
        "f16" => "near-lossless",
        "i8" => "lossy",
        "q8k" => "lossy (block-q8)",
        _ => "unknown",
    }
}

/// Derive the served return format from mean response bytes per call.
/// The `Accept` chain the production client sends allows server fallback
/// (e.g. i8 → f32 when the server lacks `LARQL_I8_WIRE`), so the requested
/// out-label alone could mislabel an arm; the observed body size per value
/// classifies what was actually served. Coarse thresholds — headers are
/// small relative to `hidden` on real models.
pub fn served_format_guess(bytes_per_call: f64, hidden: usize) -> &'static str {
    if hidden == 0 || !bytes_per_call.is_finite() || bytes_per_call <= 0.0 {
        return "unknown";
    }
    let per_value = bytes_per_call / hidden as f64;
    if per_value >= 3.0 {
        "f32"
    } else if per_value >= 1.5 {
        "f16"
    } else {
        "i8"
    }
}

// ── Drift + gate ─────────────────────────────────────────────────────────────

/// Signed drift of an arm's bits/char vs the in-run baseline, in percent.
pub fn drift_pct(arm_bpc: f64, baseline_bpc: f64) -> f64 {
    if baseline_bpc == 0.0 {
        return if arm_bpc == 0.0 { 0.0 } else { f64::INFINITY };
    }
    (arm_bpc - baseline_bpc) / baseline_bpc * 100.0
}

/// Gate verdict: |drift| within the gate passes (inclusive — the gate is a
/// pre-registered bound, landing exactly on it is inside the bound).
pub fn gate_pass(drift_pct: f64, gate_pct: f64) -> bool {
    drift_pct.is_finite() && drift_pct.abs() <= gate_pct
}

/// Validate `--gate-pct`: any finite non-negative percentage.
pub fn validate_gate_pct(gate_pct: f64) -> Result<(), String> {
    if gate_pct.is_finite() && gate_pct >= 0.0 {
        Ok(())
    } else {
        Err(format!(
            "--gate-pct must be a finite percentage >= 0, got {gate_pct}"
        ))
    }
}

// ── Records ──────────────────────────────────────────────────────────────────

/// One scored wire arm.
#[derive(Debug, Clone, Serialize)]
pub struct DriftArmResult {
    /// Combined arm label (`dec/wire_format` axis continuity).
    pub wire_format: String,
    pub wire_format_code: u32,
    /// Request-direction dtype actually put on the wire.
    pub wire_in: String,
    /// Return-direction dtype requested (`Accept`).
    pub wire_out: String,
    pub wire_in_fidelity: String,
    pub wire_out_fidelity: String,
    pub tokens_scored: usize,
    pub total_bits: f64,
    pub bits_per_token: f64,
    pub bits_per_char: f64,
    /// Signed bits/char drift vs the in-run f32/f32 baseline, percent.
    pub drift_pct: f64,
    pub pass: bool,
    pub is_baseline: bool,
    /// Mean request body bytes per FFN call (observed).
    pub wire_bytes_per_call_in: f64,
    /// Mean response body bytes per FFN call (observed).
    pub wire_bytes_per_call_out: f64,
    /// Served return format derived from `wire_bytes_per_call_out` — flags
    /// silent server-side Accept fallback (e.g. i8 requested, f32 served).
    pub served_wire_out_guess: String,
    /// Wall time for the arm — informational only; the metric itself is
    /// timing-independent.
    pub elapsed_secs: f64,
}

impl DriftArmResult {
    /// Build an arm record from its scored totals; drift/pass are computed
    /// against `baseline_bpc` at the caller's `gate_pct`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        spec: WireSpec,
        tokens_scored: usize,
        total_bits: f64,
        corpus_chars: usize,
        baseline_bpc: Option<f64>,
        gate_pct: f64,
        wire_bytes_per_call_in: f64,
        wire_bytes_per_call_out: f64,
        hidden: usize,
        elapsed_secs: f64,
    ) -> Self {
        let bits_per_char = total_bits / corpus_chars.max(1) as f64;
        let bits_per_token = total_bits / tokens_scored.max(1) as f64;
        let (drift, is_baseline) = match baseline_bpc {
            Some(base) => (drift_pct(bits_per_char, base), false),
            None => (0.0, true),
        };
        // The q8k arm's response wire is f32 (its own endpoint returns f32
        // deltas); the request side is the lossy direction.
        let out_label = if spec.is_q8k() {
            "f32"
        } else {
            spec.out_label()
        };
        DriftArmResult {
            wire_format: spec.label(),
            wire_format_code: spec.code(),
            wire_in: spec.in_label().to_string(),
            wire_out: out_label.to_string(),
            wire_in_fidelity: fidelity_label(spec.in_label()).to_string(),
            wire_out_fidelity: fidelity_label(out_label).to_string(),
            tokens_scored,
            total_bits,
            bits_per_token,
            bits_per_char,
            drift_pct: drift,
            pass: is_baseline || gate_pass(drift, gate_pct),
            is_baseline,
            wire_bytes_per_call_in,
            wire_bytes_per_call_out,
            served_wire_out_guess: served_format_guess(wire_bytes_per_call_out, hidden).to_string(),
            elapsed_secs,
        }
    }
}

/// Full-fidelity JSON run record written by `--output-file`.
#[derive(Debug, Serialize)]
pub struct DecDriftJsonResult {
    /// Unix seconds, as a string (matches ADR-0012 bench JSON).
    pub timestamp: String,
    pub model: String,
    pub ffn_url: String,
    pub corpus: String,
    pub corpus_bytes: usize,
    pub corpus_chars: usize,
    pub tokens_scored: usize,
    /// Pre-registered drift gate, percent (C6, dec-funnel §3 DEC-1A).
    pub gate_pct: f64,
    pub baseline_wire: String,
    /// Why this instrument is battery-safe, embedded so a run record is
    /// self-describing.
    pub determinism_note: String,
    pub pass: bool,
    pub arms: Vec<DriftArmResult>,
}

pub const DETERMINISM_NOTE: &str = "teacher-forced NLL on a pinned corpus: exact math given \
     weights + wire, no sampling; thermal/load/battery cannot move the number";

/// Overall verdict: every requested arm within the gate.
pub fn overall_pass(arms: &[DriftArmResult]) -> bool {
    arms.iter().all(|a| a.pass)
}

// ── Pulse (dec/* JSONL) ──────────────────────────────────────────────────────

/// One `dec/*` pulse line per arm. `step` is the arm index in the plan.
/// Numeric fields become rig metric samples; string axes ride alongside
/// with numeric `_code` twins (same contract as the replay pulse).
pub fn drift_pulse_line(step: usize, a: &DriftArmResult, gate_pct: f64) -> serde_json::Value {
    let mut obj = serde_json::Map::new();
    obj.insert("step".into(), step.into());
    obj.insert("dec/wire_format".into(), a.wire_format.clone().into());
    obj.insert("dec/wire_format_code".into(), a.wire_format_code.into());
    obj.insert("dec/wire_in".into(), a.wire_in.clone().into());
    obj.insert("dec/wire_out".into(), a.wire_out.clone().into());
    obj.insert("dec/bits_per_char".into(), num(a.bits_per_char));
    obj.insert("dec/bits_per_token".into(), num(a.bits_per_token));
    obj.insert("dec/tokens_scored".into(), (a.tokens_scored as u64).into());
    obj.insert("dec/drift_pct".into(), num(a.drift_pct));
    obj.insert("dec/drift_gate".into(), num(gate_pct));
    obj.insert("dec/drift_pass".into(), num(if a.pass { 1.0 } else { 0.0 }));
    obj.insert(
        "dec/wire_bytes_call_in".into(),
        num(a.wire_bytes_per_call_in),
    );
    obj.insert(
        "dec/wire_bytes_call_out".into(),
        num(a.wire_bytes_per_call_out),
    );
    serde_json::Value::Object(obj)
}

fn num(v: f64) -> serde_json::Value {
    serde_json::Number::from_f64(v)
        .map(serde_json::Value::Number)
        .unwrap_or(serde_json::Value::Null)
}

// ── Table ────────────────────────────────────────────────────────────────────

/// Human-readable per-arm table (shannon-verify style).
pub fn format_drift_table(arms: &[DriftArmResult], gate_pct: f64) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{:<10} {:<14} {:>10} {:>12} {:>10} {:>10}  {}\n",
        "wire", "in -> out", "bits/char", "drift vs f32", "gate", "verdict", "served-out"
    ));
    out.push_str(&format!("{}\n", "-".repeat(84)));
    for a in arms {
        let drift_str = if a.is_baseline {
            "baseline".to_string()
        } else {
            format!("{:+.4}%", a.drift_pct)
        };
        let verdict = if a.is_baseline {
            "—"
        } else if a.pass {
            "PASS"
        } else {
            "FAIL"
        };
        out.push_str(&format!(
            "{:<10} {:<14} {:>10.4} {:>12} {:>9.2}% {:>10}  {}\n",
            a.wire_format,
            format!(
                "{} -> {}",
                fidelity_short(&a.wire_in),
                fidelity_short(&a.wire_out)
            ),
            a.bits_per_char,
            drift_str,
            gate_pct,
            verdict,
            a.served_wire_out_guess,
        ));
    }
    out
}

/// `dtype(fidelity-initial)` cell, e.g. `i8(L)` for lossy, `f16(~)` for
/// near-lossless, `f32(=)` for exact.
fn fidelity_short(dtype: &str) -> String {
    let tag = match fidelity_label(dtype) {
        "exact" => "=",
        "near-lossless" => "~",
        s if s.starts_with("lossy") => "L",
        _ => "?",
    };
    format!("{dtype}({tag})")
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── bits_for_target ──────────────────────────────────────────────────

    #[test]
    fn bits_uniform_two_way_is_one_bit() {
        let bits = bits_for_target(&[0.0, 0.0], 0, 1.0, None).unwrap();
        assert!((bits - 1.0).abs() < 1e-12, "got {bits}");
    }

    #[test]
    fn bits_uniform_four_way_is_two_bits() {
        let bits = bits_for_target(&[1.5, 1.5, 1.5, 1.5], 2, 1.0, None).unwrap();
        assert!((bits - 2.0).abs() < 1e-9, "got {bits}");
    }

    #[test]
    fn bits_dominant_target_near_zero() {
        let bits = bits_for_target(&[50.0, 0.0, 0.0], 0, 1.0, None).unwrap();
        assert!(bits < 1e-9, "got {bits}");
    }

    #[test]
    fn bits_matches_manual_softmax() {
        // p(target=1) for logits [1, 2, 3]: e^2 / (e^1 + e^2 + e^3)
        let p = (2.0f64).exp() / ((1.0f64).exp() + (2.0f64).exp() + (3.0f64).exp());
        let expected = -p.log2();
        let bits = bits_for_target(&[1.0, 2.0, 3.0], 1, 1.0, None).unwrap();
        assert!((bits - expected).abs() < 1e-6, "got {bits} want {expected}");
    }

    #[test]
    fn bits_applies_logits_scaling() {
        // Scaling by 2 halves every logit — [0, 2] scaled becomes [0, 1].
        let scaled = bits_for_target(&[0.0, 2.0], 1, 2.0, None).unwrap();
        let manual = bits_for_target(&[0.0, 1.0], 1, 1.0, None).unwrap();
        assert!((scaled - manual).abs() < 1e-9);
    }

    #[test]
    fn bits_applies_softcap() {
        // With cap=1 both logits saturate toward ±1, flattening the gap:
        // bits at the low-logit target shrink vs uncapped.
        let uncapped = bits_for_target(&[10.0, 0.0], 1, 1.0, None).unwrap();
        let capped = bits_for_target(&[10.0, 0.0], 1, 1.0, Some(1.0)).unwrap();
        assert!(
            capped < uncapped,
            "softcap should flatten: capped={capped} uncapped={uncapped}"
        );
    }

    #[test]
    fn bits_skips_non_finite_competitors() {
        // A NaN competitor is excluded from the normalizer.
        let with_nan = bits_for_target(&[0.0, 0.0, f32::NAN], 0, 1.0, None).unwrap();
        let without = bits_for_target(&[0.0, 0.0], 0, 1.0, None).unwrap();
        assert!((with_nan - without).abs() < 1e-12);
    }

    #[test]
    fn bits_errors_on_out_of_vocab_target() {
        let err = bits_for_target(&[0.0, 0.0], 5, 1.0, None).unwrap_err();
        assert!(err.contains("out of vocab"));
    }

    #[test]
    fn bits_errors_on_all_non_finite() {
        let err = bits_for_target(&[f32::NAN, f32::INFINITY], 0, 1.0, None).unwrap_err();
        assert!(err.contains("non-finite"));
    }

    #[test]
    fn bits_errors_on_non_finite_target() {
        let err = bits_for_target(&[0.0, f32::NAN], 1, 1.0, None).unwrap_err();
        assert!(err.contains("non-finite logit"));
    }

    // ── normalize_corpus ─────────────────────────────────────────────────

    #[test]
    fn normalize_strips_cr_and_truncates_on_utf8_boundary() {
        let text = "ab\r\ncd\r\né";
        let normalized = normalize_corpus(text, None);
        assert_eq!(normalized, "ab\ncd\né");
        // é is 2 bytes starting at index 6 of the normalized text;
        // a 7-byte limit falls mid-char and must back off to 6.
        let truncated = normalize_corpus(text, Some(7));
        assert_eq!(truncated, "ab\ncd\n");
    }

    #[test]
    fn normalize_no_limit_and_large_limit_are_identity() {
        assert_eq!(normalize_corpus("abc", None), "abc");
        assert_eq!(normalize_corpus("abc", Some(100)), "abc");
    }

    // ── arm plan ─────────────────────────────────────────────────────────

    fn spec(s: &str) -> WireSpec {
        WireSpec::parse(s).unwrap()
    }

    #[test]
    fn plan_prepends_baseline_and_dedupes() {
        let requested = vec![spec("f16"), spec("f32"), spec("f16"), spec("q8k")];
        let plan = drift_plan(&requested);
        assert_eq!(plan, vec![spec("f32"), spec("f16"), spec("q8k")]);
    }

    #[test]
    fn plan_on_empty_request_is_baseline_only() {
        assert_eq!(drift_plan(&[]), vec![spec("f32")]);
    }

    #[test]
    fn plan_keeps_pair_arms_distinct_from_plain() {
        let plan = drift_plan(&[spec("i8"), spec("i8/f32"), spec("f32/i8")]);
        assert_eq!(
            plan,
            vec![spec("f32"), spec("i8"), spec("i8/f32"), spec("f32/i8")]
        );
    }

    // ── arm_connection ───────────────────────────────────────────────────

    #[test]
    fn plain_arms_keep_f32_request_wire() {
        for (s, out) in [
            ("f32", WirePreference::F32),
            ("f16", WirePreference::F16),
            ("i8", WirePreference::I8),
        ] {
            let c = arm_connection(spec(s));
            assert_eq!(c.wire_in, WireFormat::F32, "{s}");
            assert_eq!(c.wire_out, out, "{s}");
            assert!(!c.use_q8k, "{s}");
        }
    }

    #[test]
    fn q8k_arm_uses_q8k_dispatch() {
        let c = arm_connection(spec("q8k"));
        assert!(c.use_q8k);
        assert_eq!(c.wire_in, WireFormat::F32);
    }

    #[test]
    fn pair_arms_set_both_directions() {
        let c = arm_connection(spec("f16/i8"));
        assert_eq!(c.wire_in, WireFormat::F16);
        assert_eq!(c.wire_out, WirePreference::I8);
        assert!(!c.use_q8k);
        let c = arm_connection(spec("i8/f32"));
        assert_eq!(c.wire_in, WireFormat::I8);
        assert_eq!(c.wire_out, WirePreference::F32);
    }

    // ── fidelity + served guess ──────────────────────────────────────────

    #[test]
    fn fidelity_labels() {
        assert_eq!(fidelity_label("f32"), "exact");
        assert_eq!(fidelity_label("f16"), "near-lossless");
        assert_eq!(fidelity_label("i8"), "lossy");
        assert_eq!(fidelity_label("q8k"), "lossy (block-q8)");
        assert_eq!(fidelity_label("nope"), "unknown");
    }

    #[test]
    fn served_guess_classifies_by_bytes_per_value() {
        let hidden = 1024;
        assert_eq!(served_format_guess(4.0 * 1024.0 + 24.0, hidden), "f32");
        assert_eq!(served_format_guess(2.0 * 1024.0 + 24.0, hidden), "f16");
        assert_eq!(served_format_guess(1.1 * 1024.0, hidden), "i8");
    }

    #[test]
    fn served_guess_handles_degenerate_inputs() {
        assert_eq!(served_format_guess(4096.0, 0), "unknown");
        assert_eq!(served_format_guess(0.0, 1024), "unknown");
        assert_eq!(served_format_guess(f64::NAN, 1024), "unknown");
        assert_eq!(served_format_guess(-1.0, 1024), "unknown");
    }

    // ── drift + gate ─────────────────────────────────────────────────────

    #[test]
    fn drift_is_signed_percent() {
        assert!((drift_pct(1.01, 1.0) - 1.0).abs() < 1e-9);
        assert!((drift_pct(0.99, 1.0) + 1.0).abs() < 1e-9);
        assert_eq!(drift_pct(1.0, 1.0), 0.0);
    }

    #[test]
    fn drift_guards_zero_baseline() {
        assert_eq!(drift_pct(0.0, 0.0), 0.0);
        assert!(drift_pct(0.5, 0.0).is_infinite());
    }

    #[test]
    fn gate_edge_cases() {
        // Exactly on the gate passes (inclusive bound).
        assert!(gate_pass(0.5, 0.5));
        assert!(gate_pass(-0.5, 0.5));
        assert!(!gate_pass(0.500001, 0.5));
        assert!(!gate_pass(-0.6, 0.5));
        // Non-finite drift always fails.
        assert!(!gate_pass(f64::INFINITY, 0.5));
        assert!(!gate_pass(f64::NAN, 0.5));
        // Zero gate: only exactly-zero drift passes.
        assert!(gate_pass(0.0, 0.0));
        assert!(!gate_pass(1e-9, 0.0));
    }

    #[test]
    fn gate_pct_validation() {
        assert!(validate_gate_pct(0.5).is_ok());
        assert!(validate_gate_pct(0.0).is_ok());
        assert!(validate_gate_pct(-0.1).is_err());
        assert!(validate_gate_pct(f64::NAN).is_err());
        assert!(validate_gate_pct(f64::INFINITY).is_err());
    }

    // ── DriftArmResult ───────────────────────────────────────────────────

    fn arm(spec_s: &str, total_bits: f64, baseline_bpc: Option<f64>) -> DriftArmResult {
        DriftArmResult::new(
            spec(spec_s),
            100,
            total_bits,
            1000,
            baseline_bpc,
            0.5,
            4096.0,
            2072.0,
            1024,
            12.0,
        )
    }

    #[test]
    fn baseline_arm_has_zero_drift_and_passes() {
        let a = arm("f32", 1000.0, None);
        assert!(a.is_baseline);
        assert_eq!(a.drift_pct, 0.0);
        assert!(a.pass);
        assert!((a.bits_per_char - 1.0).abs() < 1e-12);
        assert!((a.bits_per_token - 10.0).abs() < 1e-12);
    }

    #[test]
    fn arm_drift_and_verdict_computed_against_baseline() {
        // 1.004 bits/char vs baseline 1.0 → +0.4% → passes at 0.5.
        let a = arm("i8", 1004.0, Some(1.0));
        assert!(!a.is_baseline);
        assert!((a.drift_pct - 0.4).abs() < 1e-9);
        assert!(a.pass);
        // 1.006 → +0.6% → fails.
        let b = arm("i8", 1006.0, Some(1.0));
        assert!(!b.pass);
    }

    #[test]
    fn arm_records_direction_labels_and_served_guess() {
        let a = arm("f16/i8", 1000.0, Some(1.0));
        assert_eq!(a.wire_in, "f16");
        assert_eq!(a.wire_out, "i8");
        assert_eq!(a.wire_in_fidelity, "near-lossless");
        assert_eq!(a.wire_out_fidelity, "lossy");
        // 2072 bytes / 1024 hidden ≈ 2.02 per value → f16 served.
        assert_eq!(a.served_wire_out_guess, "f16");
    }

    #[test]
    fn q8k_arm_reports_f32_return_wire() {
        let a = arm("q8k", 1000.0, Some(1.0));
        assert_eq!(a.wire_in, "q8k");
        assert_eq!(a.wire_out, "f32");
        assert_eq!(a.wire_in_fidelity, "lossy (block-q8)");
        assert_eq!(a.wire_out_fidelity, "exact");
    }

    #[test]
    fn overall_pass_requires_every_arm() {
        let good = arm("f16", 1001.0, Some(1.0));
        let bad = arm("i8", 1100.0, Some(1.0));
        assert!(overall_pass(std::slice::from_ref(&good)));
        assert!(!overall_pass(&[good, bad]));
    }

    // ── pulse ────────────────────────────────────────────────────────────

    #[test]
    fn pulse_line_has_registry_keys() {
        let a = arm("i8", 1004.0, Some(1.0));
        let line = drift_pulse_line(2, &a, 0.5);
        assert_eq!(line["step"], 2);
        assert_eq!(line["dec/wire_format"], "i8");
        assert_eq!(line["dec/wire_format_code"], 2);
        assert_eq!(line["dec/wire_in"], "f32");
        assert_eq!(line["dec/wire_out"], "i8");
        assert!(line["dec/bits_per_char"].is_number());
        assert!(line["dec/bits_per_token"].is_number());
        assert_eq!(line["dec/tokens_scored"], 100);
        assert!((line["dec/drift_pct"].as_f64().unwrap() - 0.4).abs() < 1e-9);
        assert_eq!(line["dec/drift_gate"], 0.5);
        assert_eq!(line["dec/drift_pass"], 1.0);
        assert!(line["dec/wire_bytes_call_in"].is_number());
        assert!(line["dec/wire_bytes_call_out"].is_number());
    }

    #[test]
    fn pulse_line_failing_arm_reports_zero_pass() {
        let a = arm("i8", 1100.0, Some(1.0));
        let line = drift_pulse_line(0, &a, 0.5);
        assert_eq!(line["dec/drift_pass"], 0.0);
    }

    // ── JSON record ──────────────────────────────────────────────────────

    #[test]
    fn json_record_serializes_with_schema_fields() {
        let arms = vec![arm("f32", 1000.0, None), arm("i8", 1004.0, Some(1.0))];
        let pass = overall_pass(&arms);
        let record = DecDriftJsonResult {
            timestamp: "123".into(),
            model: "gemma4-26b-a4b-q4k".into(),
            ffn_url: "http://127.0.0.1:8080".into(),
            corpus: "tests/fixtures/shannon_frankenstein_2k.txt".into(),
            corpus_bytes: 2048,
            corpus_chars: 1000,
            tokens_scored: 100,
            gate_pct: 0.5,
            baseline_wire: "f32".into(),
            determinism_note: DETERMINISM_NOTE.into(),
            pass,
            arms,
        };
        let v = serde_json::to_value(&record).unwrap();
        assert_eq!(v["gate_pct"], 0.5);
        assert_eq!(v["baseline_wire"], "f32");
        assert_eq!(v["pass"], true);
        assert_eq!(v["arms"][0]["is_baseline"], true);
        assert_eq!(v["arms"][1]["wire_format"], "i8");
        assert!(v["determinism_note"]
            .as_str()
            .unwrap()
            .contains("teacher-forced"));
    }

    // ── table ────────────────────────────────────────────────────────────

    #[test]
    fn table_marks_baseline_pass_and_fail() {
        let arms = vec![
            arm("f32", 1000.0, None),
            arm("f16", 1001.0, Some(1.0)),
            arm("i8", 1100.0, Some(1.0)),
        ];
        let table = format_drift_table(&arms, 0.5);
        assert!(table.contains("baseline"));
        assert!(table.contains("PASS"));
        assert!(table.contains("FAIL"));
        // Direction fidelity cells: f32 exact, f16 near-lossless, i8 lossy.
        assert!(table.contains("f32(=)"));
        assert!(table.contains("f16(~)"));
        assert!(table.contains("i8(L)"));
    }

    #[test]
    fn default_gate_matches_preregistered_value() {
        assert_eq!(DEFAULT_DRIFT_GATE_PCT, 0.5);
    }
}
