//! Gate for the arithmetic expert (spec §3). Two tiers:
//!
//! - **Tier 0 — symbolic.** The explicit-expression scanner from
//!   `extract.rs` run over the prompt surface. Sharing the scanner with the
//!   extractor makes "tier-0 fire ⇒ symbolic extract succeeds" true by
//!   construction (the A10 fire × extract invariant).
//! - **Tier 1 — engagement probe.** Ridge probe on the residual at the probe
//!   layer, last prompt token, reading arithmetic-engagement exhaust. Probe
//!   weights are per-checkpoint artifacts shipped alongside the vindex
//!   (see `probe_weights/README.md`); current weights are SUB-SPEC on
//!   sensitivity and hardening is required before the probe is sole trigger.
//!
//! Policy: tier-0 fire OR tier-1 fire ⇒ dispatch. No fire ⇒ native path
//! untouched.

use serde::{Deserialize, Serialize};

use crate::experts::virtual_expert::{Fire, ResidualTap};

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

use super::extract::find_expression;

/// Tier-0 symbolic scan: does the prompt surface carry an explicit integer
/// expression (operator adjacent to digit spans)?
pub fn tier0_fires(prompt_text: &str) -> bool {
    find_expression(prompt_text).is_some()
}

/// Ridge probe artifact: linear readout over the residual at `layer`, last
/// prompt token. Versioned per model checkpoint; re-fit per checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RidgeProbe {
    /// Model checkpoint the weights were fit on (informational).
    #[serde(default)]
    pub model: String,
    /// Residual layer the probe reads (L8 on Gemma-3-4b; treat as a depth
    /// fraction when porting — the relative-depth law is the ASSUMED part).
    pub layer: usize,
    pub weights: Vec<f32>,
    pub bias: f32,
    /// Fire when `score >= threshold`.
    pub threshold: f32,
}

impl RidgeProbe {
    /// Load a probe artifact (JSON) from disk.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        serde_json::from_slice(&bytes).map_err(|e| format!("parse {}: {e}", path.display()))
    }

    /// Probe score for a tap, or `None` when the tap doesn't carry this
    /// probe's layer or the dimension mismatches (must never fire on a
    /// mismatched tap).
    pub fn score(&self, tap: &ResidualTap) -> Option<f32> {
        let residual = tap.residual_at(self.layer)?;
        if residual.len() != self.weights.len() {
            return None;
        }
        let dot: f32 = self
            .weights
            .iter()
            .zip(residual.iter())
            .map(|(w, x)| w * x)
            .sum();
        Some(dot + self.bias)
    }
}

/// Combined gate policy. Tier 0 is checked first (cost ~0); the probe only
/// decides when the surface scan is silent and a tap was captured.
pub fn gate(probe: Option<&RidgeProbe>, tap: Option<&ResidualTap>, prompt_text: &str) -> Fire {
    if tier0_fires(prompt_text) {
        return Fire::Tier0;
    }
    if let (Some(probe), Some(tap)) = (probe, tap) {
        if let Some(score) = probe.score(tap) {
            if score >= probe.threshold {
                return Fire::Tier1(score);
            }
        }
    }
    Fire::No
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe(layer: usize, dim: usize, threshold: f32) -> RidgeProbe {
        // weights = [1, 0, 0, ...] → score = residual[0] + bias
        let mut weights = vec![0.0; dim];
        weights[0] = 1.0;
        RidgeProbe {
            model: "test".into(),
            layer,
            weights,
            bias: 0.0,
            threshold,
        }
    }

    fn tap(layer: usize, first: f32, dim: usize) -> ResidualTap {
        let mut residual = vec![0.0; dim];
        residual[0] = first;
        ResidualTap::single(layer, residual)
    }

    #[test]
    fn tier0_fires_on_explicit_math_only() {
        assert!(tier0_fires("123456 + 654321 ="));
        assert!(tier0_fires("what is 12345 * 6789?"));
        assert!(!tier0_fires("My phone number is 4415550172."));
        assert!(!tier0_fires("The meeting is on 2026-06-11."));
        assert!(!tier0_fires("What is the capital of France?"));
    }

    #[test]
    fn probe_scores_matching_tap() {
        let p = probe(8, 4, 0.5);
        assert_eq!(p.score(&tap(8, 0.9, 4)), Some(0.9));
    }

    #[test]
    fn probe_refuses_layer_or_dim_mismatch() {
        let p = probe(8, 4, 0.5);
        assert_eq!(p.score(&tap(9, 0.9, 4)), None, "wrong layer");
        assert_eq!(p.score(&tap(8, 0.9, 5)), None, "wrong dim");
    }

    #[test]
    fn probe_selects_its_layer_from_a_multi_layer_tap() {
        let p = probe(8, 4, 0.5);
        let mut r8 = vec![0.0; 4];
        r8[0] = 0.7;
        let tap = ResidualTap::from(vec![(4, vec![9.0; 4]), (8, r8), (16, vec![9.0; 4])]);
        assert_eq!(p.score(&tap), Some(0.7));
        assert_eq!(tap.residual_at(16), Some(&[9.0f32; 4][..]));
        assert_eq!(tap.residual_at(5), None);
        assert_eq!(tap.layers().len(), 3);
    }

    #[test]
    fn gate_prefers_tier0() {
        let p = probe(8, 4, 0.5);
        let t = tap(8, 0.9, 4);
        assert_eq!(gate(Some(&p), Some(&t), "12 + 7 ="), Fire::Tier0);
    }

    #[test]
    fn gate_tier1_fires_above_threshold() {
        let p = probe(8, 4, 0.5);
        let t = tap(8, 0.9, 4);
        match gate(Some(&p), Some(&t), "If you have seven apples...") {
            Fire::Tier1(s) => assert!((s - 0.9).abs() < 1e-6),
            other => panic!("expected tier1, got {other:?}"),
        }
    }

    #[test]
    fn gate_no_fire_below_threshold_or_without_tap() {
        let p = probe(8, 4, 0.5);
        let cold = tap(8, 0.1, 4);
        assert_eq!(gate(Some(&p), Some(&cold), "plain prose"), Fire::No);
        assert_eq!(gate(Some(&p), None, "plain prose"), Fire::No);
        assert_eq!(gate(None, None, "plain prose"), Fire::No);
    }

    #[test]
    fn probe_load_roundtrip_and_errors() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("probe.json");
        let p = probe(8, 3, 0.25);
        std::fs::write(&path, serde_json::to_vec(&p).expect("ser")).expect("write");
        let loaded = RidgeProbe::load(&path).expect("load");
        assert_eq!(loaded.layer, 8);
        assert_eq!(loaded.weights.len(), 3);
        assert!(RidgeProbe::load(&dir.path().join("missing.json")).is_err());
        std::fs::write(&path, b"not json").expect("write");
        assert!(RidgeProbe::load(&path).is_err());
    }
}
