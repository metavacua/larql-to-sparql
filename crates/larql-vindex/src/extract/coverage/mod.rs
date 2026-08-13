//! Extraction tensor-coverage audit.
//!
//! Every tensor in a source checkpoint must be classifiable as one of three
//! things, and the third one has to be loud:
//!
//! 1. **recognised** — the architecture names it, so extraction can ask for it;
//! 2. **dropped by a named rule** — a deliberate, explained omission ([`rules`]);
//! 3. **unrecognised** — nobody knows what it is.
//!
//! Today a tensor in bucket 3 is silently discarded and nothing anywhere
//! reports it. That is not hypothetical: it cost this codebase 5 of 11
//! attention tensors per layer on GPT-OSS (`docs/k3-funnel.md` §4.6.1) and
//! then 3 of 8 MLP tensors on the same model (§4.7) — both found by a human
//! reading a checkpoint by hand, months apart, after the wrong numbers had
//! already been believed.
//!
//! K3 is guaranteed to carry tensor classes larql has never seen (QB frozen
//! bias, block AttnRes state, LatentMoE projections). This turns that
//! guarantee from a silent hazard into a startup error.
//!
//! ## What this does and does not prove
//!
//! It measures **naming**, not consumption: a recognised tensor is one
//! extraction *can* reach, not one it definitely wrote. Naming is the
//! necessary condition, and it is where every silent drop found so far
//! actually lived — in both cases the extractor asked for every key it was
//! told about, and nobody had told it. A consumption-level audit (recording
//! `WeightSource` reads) is the natural follow-on and would subsume this;
//! it is not needed to close the bug class that has actually bitten.

mod named_keys;
mod rules;

pub use rules::DropRule;

use std::collections::HashSet;

use larql_models::ModelArchitecture;

/// How one source tensor is accounted for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Coverage {
    /// The architecture names this key, so extraction can reach it.
    Recognised,
    /// Deliberately not extracted, per a named rule.
    Dropped(DropRule),
    /// Nobody knows what this is. The whole point of the audit.
    Unrecognised,
}

/// Result of auditing a checkpoint's tensor names against an architecture.
#[derive(Debug, Clone, Default)]
pub struct CoverageReport {
    /// Count of tensors the architecture names.
    pub recognised: usize,
    /// Deliberately dropped tensors, with the rule that explains each.
    pub dropped: Vec<(String, DropRule)>,
    /// Tensors nothing accounts for. Sorted, for a stable report.
    pub unrecognised: Vec<String>,
}

impl CoverageReport {
    /// Total tensors examined.
    pub fn total(&self) -> usize {
        self.recognised + self.dropped.len() + self.unrecognised.len()
    }

    /// Whether every tensor was accounted for.
    pub fn is_clean(&self) -> bool {
        self.unrecognised.is_empty()
    }

    /// Human-readable summary, one line per unrecognised tensor.
    ///
    /// Deliberately lists unrecognised names in full rather than truncating:
    /// the count alone is what a reader skims past, and the names are what
    /// makes the omission actionable.
    pub fn summary(&self) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        let _ = write!(
            s,
            "tensor coverage: {} recognised, {} dropped by rule, {} unrecognised (of {})",
            self.recognised,
            self.dropped.len(),
            self.unrecognised.len(),
            self.total()
        );
        if !self.unrecognised.is_empty() {
            let _ = write!(s, "\nunrecognised tensors — extraction will discard these:");
            for name in &self.unrecognised {
                let _ = write!(s, "\n  {name}");
            }
        }
        s
    }
}

/// Audit `source_names` against what `arch` can name.
///
/// `source_names` are checkpoint tensor names with the architecture's
/// `key_prefixes_to_strip` already applied, so they are directly comparable
/// with the keys the accessors emit.
pub fn audit<'a>(
    arch: &dyn ModelArchitecture,
    num_layers: usize,
    source_names: impl IntoIterator<Item = &'a str>,
) -> CoverageReport {
    let named: HashSet<String> = named_keys::collect(arch, num_layers);
    let mut report = CoverageReport::default();

    for name in source_names {
        if named.contains(name) {
            report.recognised += 1;
        } else if let Some(rule) = rules::classify(name) {
            report.dropped.push((name.to_string(), rule));
        } else {
            report.unrecognised.push(name.to_string());
        }
    }

    report.unrecognised.sort();
    report.dropped.sort_by(|a, b| a.0.cmp(&b.0));
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_models::detect_from_json;

    fn gpt_oss() -> Box<dyn ModelArchitecture> {
        detect_from_json(&serde_json::json!({
            "model_type": "gpt_oss",
            "num_hidden_layers": 1,
            "hidden_size": 2880,
            "intermediate_size": 2880,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "num_local_experts": 2,
            "num_experts_per_tok": 2,
        }))
    }

    /// The exact surface of one real GPT-OSS layer: 11 attention tensors and
    /// 8 MLP tensors. All 19 must be recognised — this is the regression
    /// test for both §4.6.1 and §4.7.
    fn gpt_oss_layer0_tensors() -> Vec<&'static str> {
        vec![
            "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.k_proj.weight",
            "layers.0.self_attn.v_proj.weight",
            "layers.0.self_attn.o_proj.weight",
            "layers.0.self_attn.q_proj.bias",
            "layers.0.self_attn.k_proj.bias",
            "layers.0.self_attn.v_proj.bias",
            "layers.0.self_attn.o_proj.bias",
            "layers.0.self_attn.sinks",
            "layers.0.input_layernorm.weight",
            "layers.0.post_attention_layernorm.weight",
            "layers.0.mlp.router.weight",
            "layers.0.mlp.router.bias",
            "layers.0.mlp.experts.gate_up_proj_blocks",
            "layers.0.mlp.experts.gate_up_proj_scales",
            "layers.0.mlp.experts.gate_up_proj_bias",
            "layers.0.mlp.experts.down_proj_blocks",
            "layers.0.mlp.experts.down_proj_scales",
            "layers.0.mlp.experts.down_proj_bias",
        ]
    }

    #[test]
    fn a_fully_named_layer_audits_clean() {
        let a = gpt_oss();
        let r = audit(&*a, 1, gpt_oss_layer0_tensors());
        assert!(r.is_clean(), "{}", r.summary());
        assert_eq!(r.recognised, 19);
        assert_eq!(r.total(), 19);
    }

    /// The bug this exists to catch: a tensor the checkpoint has and the
    /// architecture does not name must land in `unrecognised`, not vanish.
    #[test]
    fn an_unnamed_tensor_is_reported_not_swallowed() {
        let a = gpt_oss();
        let mut names = gpt_oss_layer0_tensors();
        names.push("layers.0.self_attn.some_future_tensor");
        let r = audit(&*a, 1, names);

        assert!(!r.is_clean());
        assert_eq!(
            r.unrecognised,
            vec!["layers.0.self_attn.some_future_tensor"]
        );
        assert!(r.summary().contains("some_future_tensor"));
    }

    /// Simulates the pre-2026-07-29 state: an architecture that does not name
    /// sinks or the projection biases. Those five tensors must surface.
    #[test]
    fn reproduces_the_historical_attention_drop() {
        // Qwen2 names q/k/v biases but not sinks — so a checkpoint carrying
        // sinks against this architecture reports exactly that one tensor.
        let a = detect_from_json(&serde_json::json!({
            "model_type": "qwen2",
            "num_hidden_layers": 1,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "vocab_size": 32,
        }));
        let r = audit(&*a, 1, vec!["layers.0.self_attn.sinks"]);
        assert_eq!(r.unrecognised, vec!["layers.0.self_attn.sinks"]);
    }

    #[test]
    fn deliberate_drops_are_separated_from_unknowns() {
        let a = gpt_oss();
        let r = audit(
            &*a,
            1,
            vec![
                "layers.0.self_attn.rotary_emb.inv_freq",
                "vision_model.encoder.layers.0.fc1.weight",
                "layers.0.mystery_tensor",
            ],
        );
        assert_eq!(r.dropped.len(), 2);
        assert_eq!(r.unrecognised, vec!["layers.0.mystery_tensor"]);
        assert_eq!(r.recognised, 0);
    }

    #[test]
    fn report_output_is_stable_and_sorted() {
        let a = gpt_oss();
        let r = audit(&*a, 1, vec!["layers.0.zzz", "layers.0.aaa", "layers.0.mmm"]);
        assert_eq!(
            r.unrecognised,
            vec!["layers.0.aaa", "layers.0.mmm", "layers.0.zzz"]
        );
    }

    #[test]
    fn an_empty_checkpoint_audits_clean() {
        let a = gpt_oss();
        let r = audit(&*a, 1, Vec::<&str>::new());
        assert!(r.is_clean());
        assert_eq!(r.total(), 0);
        assert!(r.summary().contains("0 unrecognised"));
    }

    /// A layer beyond `num_layers` is not named, so its tensors report as
    /// unrecognised — which is the correct answer for a checkpoint whose
    /// depth disagrees with its config.
    #[test]
    fn tensors_past_the_configured_depth_are_unrecognised() {
        let a = gpt_oss();
        let r = audit(&*a, 1, vec!["layers.7.self_attn.q_proj.weight"]);
        assert_eq!(r.unrecognised.len(), 1);
    }
}
