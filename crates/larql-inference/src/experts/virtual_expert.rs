//! `VirtualExpert` — the gate / extract / compute / drive / verify decomposition
//! for experts that ride the forward pass without touching weights or routing.
//!
//! Spec: `docs/specs/virtual-experts/arithmetic-virtual-expert.md` (§8).
//! Instance #1 is [`crate::experts::arith::ArithmeticExpert`].
//!
//! Design constraints (baked in from the arithmetic_mechanism arc):
//! - the gate reads **exhaust, not intent** — an involuntary engagement signal
//!   in the residual stream, plus a symbolic scan of the prompt surface;
//! - the expert is **invisible to the model** — no weights touched, no model
//!   routing used;
//! - compute is **never** the model's — the model supplies I/O (extraction,
//!   readout, a magnitude prior), the expert supplies the algorithm.

#[cfg(not(target_arch = "wasm32"))]
use tokenizers::Tokenizer;

/// Read-only residual capture, last prompt token, at one or more layers.
/// In production this is a free read off the prompt forward pass; harnesses
/// may populate it with `crate::forward::capture_residuals` (whose
/// `Vec<(layer, residual)>` output converts directly via `From`).
///
/// Multi-layer so one capture can serve several experts/probes reading at
/// different depths — the tap is taken once per prompt pass, not per expert.
#[derive(Debug, Clone, Default)]
pub struct ResidualTap {
    layers: Vec<(usize, Vec<f32>)>,
}

impl ResidualTap {
    /// Tap with a single captured layer.
    pub fn single(layer: usize, residual: Vec<f32>) -> Self {
        ResidualTap {
            layers: vec![(layer, residual)],
        }
    }

    /// The residual captured at `layer`, if that layer was tapped.
    pub fn residual_at(&self, layer: usize) -> Option<&[f32]> {
        self.layers
            .iter()
            .find(|(l, _)| *l == layer)
            .map(|(_, r)| r.as_slice())
    }

    /// All captured `(layer, residual)` pairs.
    pub fn layers(&self) -> &[(usize, Vec<f32>)] {
        &self.layers
    }
}

impl From<Vec<(usize, Vec<f32>)>> for ResidualTap {
    fn from(layers: Vec<(usize, Vec<f32>)>) -> Self {
        ResidualTap { layers }
    }
}

/// Gate decision. A fire is a dispatch decision: fired ⇒ dispatch, always.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Fire {
    /// Native path untouched.
    No,
    /// Tier-0 symbolic scan fired (explicit math on the prompt surface).
    Tier0,
    /// Tier-1 engagement probe fired, with the probe score.
    Tier1(f32),
}

impl Fire {
    pub fn fired(&self) -> bool {
        !matches!(self, Fire::No)
    }

    /// Telemetry label ("no" | "tier0" | "tier1(score)").
    pub fn label(&self) -> String {
        match self {
            Fire::No => "no".to_string(),
            Fire::Tier0 => "tier0".to_string(),
            Fire::Tier1(s) => format!("tier1({s:.3})"),
        }
    }
}

/// Extraction failed; controller falls to native and flags `extract_miss`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("extract miss: {0}")]
pub struct ExtractMiss(pub String);

/// Verify-leg verdict. The native answer is a magnitude **prior**, not a
/// judge: `Suspect` flags a likely extraction bug, it never overrides the
/// exact compute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// No native answer available, or the prior is void (operand size past
    /// its measured envelope).
    Skipped,
    /// ALU result magnitude-consistent with the model's native answer.
    Consistent,
    /// Magnitude mismatch — flag `extract_suspect`.
    Suspect(String),
}

impl Verdict {
    /// Telemetry label.
    pub fn label(&self) -> String {
        match self {
            Verdict::Skipped => "skipped".to_string(),
            Verdict::Consistent => "consistent".to_string(),
            Verdict::Suspect(r) => format!("suspect: {r}"),
        }
    }
}

/// The answer text the controller forces at the sampler, one token per decode
/// step, then **terminates at schedule end** (delivery = 1.0 by construction —
/// the one observed delivery defect was post-schedule digit continuation).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DriveSchedule {
    pub text: String,
}

impl DriveSchedule {
    /// Tokenize the schedule text into the forced token sequence (no special
    /// tokens — the schedule rides the decode the model was emitting anyway).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn forced_ids(&self, tokenizer: &Tokenizer) -> Vec<u32> {
        tokenizer
            .encode(self.text.as_str(), false)
            .map(|e| e.get_ids().to_vec())
            .unwrap_or_default()
    }
}

/// An expert that gates on forward-pass exhaust, extracts a payload through
/// the model's I/O, computes externally and exactly, and drives the answer
/// back through the sampler.
pub trait VirtualExpert {
    /// What extraction produces (e.g. a parsed arithmetic expression).
    type Payload;
    /// What compute produces (exact, external — never the model's).
    type Answer;

    fn name(&self) -> &'static str;

    /// Gate on exhaust, not intent. `tap` is the residual capture for the
    /// tier-1 probe when one is loaded; tier-0 scans the prompt surface.
    fn gate(&self, tap: Option<&ResidualTap>, prompt_text: &str) -> Fire;

    /// Extract the payload: from the prompt surface (explicit path,
    /// `rewrite = None`) or from a model-emitted rewrite (disguised path).
    fn extract(
        &self,
        prompt_text: &str,
        rewrite: Option<&str>,
    ) -> Result<Self::Payload, ExtractMiss>;

    /// Exact external compute.
    fn compute(&self, payload: &Self::Payload) -> Self::Answer;

    /// Forced-decode schedule for the answer (default drive path).
    fn drive(&self, answer: &Self::Answer) -> DriveSchedule;

    /// Magnitude-prior check against the model's native answer, if one was
    /// produced. A prior, not a judge.
    fn verify(&self, answer: &Self::Answer, native: Option<&str>) -> Verdict;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{make_test_tokenizer, synthetic_tokenizer_json};

    #[test]
    fn fire_fired_and_labels() {
        assert!(!Fire::No.fired());
        assert!(Fire::Tier0.fired());
        assert!(Fire::Tier1(0.93).fired());
        assert_eq!(Fire::No.label(), "no");
        assert_eq!(Fire::Tier0.label(), "tier0");
        assert_eq!(Fire::Tier1(0.5).label(), "tier1(0.500)");
    }

    #[test]
    fn verdict_labels() {
        assert_eq!(Verdict::Skipped.label(), "skipped");
        assert_eq!(Verdict::Consistent.label(), "consistent");
        assert_eq!(
            Verdict::Suspect("digit count".into()).label(),
            "suspect: digit count"
        );
    }

    #[test]
    fn extract_miss_displays_reason() {
        let m = ExtractMiss("no expression".into());
        assert_eq!(m.to_string(), "extract miss: no expression");
    }

    #[test]
    fn drive_schedule_tokenizes_with_the_session_tokenizer() {
        // Null-pre-tokenizer fixture: "[N]" encodes to the single id N.
        let tok = Tokenizer::from_bytes(synthetic_tokenizer_json(16).as_bytes()).expect("tok");
        let sched = DriveSchedule {
            text: "[5]".to_string(),
        };
        assert_eq!(sched.forced_ids(&tok), vec![5]);
    }

    #[test]
    fn drive_schedule_empty_text_yields_empty_schedule() {
        let tok = make_test_tokenizer(8);
        let sched = DriveSchedule {
            text: String::new(),
        };
        assert!(sched.forced_ids(&tok).is_empty());
    }
}
