//! Named rules for source tensors that are deliberately not extracted.
//!
//! The audit's whole value is that its third bucket — *unrecognised* — stays
//! small and loud. That only works if every legitimate omission is explained
//! by a rule with a name, so a reader can tell "we decided not to carry this"
//! from "nobody noticed this".
//!
//! A rule is a claim about a tensor class, and adding one is a decision. Do
//! not add a rule to quieten an unrecognised tensor you have not identified —
//! that converts this module back into the silent-drop mechanism it exists to
//! replace. If you don't know what a tensor is, it belongs in `unrecognised`.

/// Why a source tensor is deliberately not extracted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DropRule {
    /// Stable identifier, reported alongside the tensor.
    pub name: &'static str,
    /// What the rule claims, in one line.
    pub reason: &'static str,
}

/// Derived at load time from config, never read from the checkpoint.
///
/// RoPE inverse frequencies are a pure function of `rope_theta`, `head_dim`
/// and the scaling config. Some exports persist them as buffers; recomputing
/// is both cheaper and immune to a stale buffer disagreeing with the config.
const ROPE_INV_FREQ: DropRule = DropRule {
    name: "rope-derived",
    reason: "RoPE inverse frequencies are recomputed from config, not stored",
};

/// Vision/audio towers and their projectors.
///
/// The vindex format carries the language model. Multi-modal encoders are a
/// separate concern with their own protocol (`larql_models::multimodal`), and
/// a text vindex deliberately does not embed one.
const NON_TEXT_TOWER: DropRule = DropRule {
    name: "non-text-tower",
    reason: "vision/audio encoder and projector — not part of the language model",
};

/// Training-time state that a serving checkpoint sometimes still carries.
const TRAINING_STATE: DropRule = DropRule {
    name: "training-state",
    reason: "optimiser / bookkeeping state, not an inference weight",
};

/// Classify a source tensor name against the deliberate-drop rules.
///
/// `name` is the checkpoint's own name with architecture prefixes already
/// stripped, matching what [`super::named_keys::collect`] emits.
pub fn classify(name: &str) -> Option<DropRule> {
    // Segment-aware matching: split on '.' so a rule fires on a real path
    // segment rather than on a substring that happens to appear inside a
    // longer identifier.
    let segments: Vec<&str> = name.split('.').collect();
    let has = |seg: &str| segments.contains(&seg);

    if has("rotary_emb") || has("inv_freq") || name.ends_with(".inv_freq") {
        return Some(ROPE_INV_FREQ);
    }
    if has("vision_model")
        || has("vision_tower")
        || has("audio_tower")
        || has("multi_modal_projector")
    {
        return Some(NON_TEXT_TOWER);
    }
    if has("optimizer") || has("global_step") || has("training_state") {
        return Some(TRAINING_STATE);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_inverse_frequencies_are_derived() {
        assert_eq!(
            classify("model.layers.0.self_attn.rotary_emb.inv_freq")
                .unwrap()
                .name,
            "rope-derived"
        );
        assert_eq!(
            classify("rotary_emb.inv_freq").unwrap().name,
            "rope-derived"
        );
    }

    #[test]
    fn vision_and_audio_towers_are_named_drops() {
        for k in [
            "vision_model.encoder.layers.0.self_attn.q_proj.weight",
            "vision_tower.embeddings.patch_embedding.weight",
            "multi_modal_projector.linear.weight",
            "audio_tower.layers.3.fc1.weight",
        ] {
            assert_eq!(classify(k).unwrap().name, "non-text-tower", "for {k}");
        }
    }

    #[test]
    fn training_state_is_a_named_drop() {
        assert_eq!(
            classify("optimizer.state.step").unwrap().name,
            "training-state"
        );
    }

    /// A language-model weight must never be swallowed by a rule — that is
    /// exactly the silent drop this module exists to prevent.
    #[test]
    fn ordinary_language_model_weights_match_no_rule() {
        for k in [
            "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.sinks",
            "layers.0.mlp.router.bias",
            "layers.0.mlp.experts.gate_up_proj_bias",
            "model.embed_tokens.weight",
            "lm_head.weight",
        ] {
            assert!(classify(k).is_none(), "{k} was wrongly matched by a rule");
        }
    }

    /// Rules match whole path segments, so a tensor whose name merely
    /// *contains* a rule word is not swallowed. Without this, a future
    /// `vision_model_adapter` weight in the language model would vanish.
    #[test]
    fn rules_match_segments_not_substrings() {
        assert!(classify("layers.0.vision_model_adapter.weight").is_none());
        assert!(classify("layers.0.mlp.optimizer_gate.weight").is_none());
    }
}
