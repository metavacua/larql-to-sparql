//! Shared tensor-name fragments for model architectures.
//!
//! The sibling of [`crate::defaults`], for names rather than numbers:
//! where several architectures address the *same* HuggingFace tensor,
//! the string lives here once so `architectures/*.rs` can't drift on it.
//! A drifted key is silently destructive — a wrong name resolves to
//! `None` at extraction time and the tensor is quietly dropped rather
//! than erroring, which is how GPT-OSS lost five attention tensors per
//! layer for months before 2026-07-29.
//!
//! Architectures whose naming genuinely differs (fused QKV,
//! `c_attn`-style Conv1D layouts) build their own keys and should not
//! reach for these.

/// Standard HuggingFace QK-norm names.
///
/// Declared by Gemma 2/3/4, Qwen3, and OLMoE — five architectures, one
/// spelling, since the tensors come from `transformers`' per-head
/// RMSNorm on Q and K rather than anything family-specific.
pub(crate) mod qk_norm {
    /// Query-norm weight, relative to a layer prefix.
    const Q_SUFFIX: &str = "self_attn.q_norm.weight";
    /// Key-norm weight, relative to a layer prefix.
    const K_SUFFIX: &str = "self_attn.k_norm.weight";

    /// `<prefix>self_attn.q_norm.weight`.
    pub(crate) fn q(layer_prefix: &str) -> Option<String> {
        Some(format!("{layer_prefix}{Q_SUFFIX}"))
    }

    /// `<prefix>self_attn.k_norm.weight`.
    pub(crate) fn k(layer_prefix: &str) -> Option<String> {
        Some(format!("{layer_prefix}{K_SUFFIX}"))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        const PREFIX: &str = "model.layers.3.";

        #[test]
        fn each_builder_appends_its_suffix_to_the_prefix() {
            assert_eq!(q(PREFIX).unwrap(), "model.layers.3.self_attn.q_norm.weight");
            assert_eq!(k(PREFIX).unwrap(), "model.layers.3.self_attn.k_norm.weight");
        }

        #[test]
        fn q_and_k_do_not_collide() {
            assert_ne!(q(PREFIX), k(PREFIX));
        }
    }
}

/// Standard HuggingFace per-expert MoE names.
///
/// Declared by DeepSeek, Qwen3-MoE, and OLMoE. The router lives at
/// `mlp.gate.weight` and each expert's three projections are indexed by
/// position under `mlp.experts.{id}` — the layout `transformers` uses
/// for every unpacked MoE. Architectures that pack experts into one
/// tensor per layer (GPT-OSS's MXFP4 blocks) name them differently and
/// do not use this.
pub(crate) mod moe_experts {
    /// Router weight, relative to a layer prefix.
    const ROUTER_SUFFIX: &str = "mlp.gate.weight";

    /// `<prefix>mlp.gate.weight` — the per-layer expert router.
    pub(crate) fn router(layer_prefix: &str) -> Option<String> {
        Some(format!("{layer_prefix}{ROUTER_SUFFIX}"))
    }

    /// `<prefix>mlp.experts.<id>.gate_proj.weight`.
    pub(crate) fn gate_proj(layer_prefix: &str, expert_id: usize) -> Option<String> {
        Some(projection(layer_prefix, expert_id, "gate"))
    }

    /// `<prefix>mlp.experts.<id>.up_proj.weight`.
    pub(crate) fn up_proj(layer_prefix: &str, expert_id: usize) -> Option<String> {
        Some(projection(layer_prefix, expert_id, "up"))
    }

    /// `<prefix>mlp.experts.<id>.down_proj.weight`.
    pub(crate) fn down_proj(layer_prefix: &str, expert_id: usize) -> Option<String> {
        Some(projection(layer_prefix, expert_id, "down"))
    }

    /// The shape all three per-expert projections share.
    fn projection(layer_prefix: &str, expert_id: usize, which: &str) -> String {
        format!("{layer_prefix}mlp.experts.{expert_id}.{which}_proj.weight")
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        const PREFIX: &str = "model.layers.5.";

        #[test]
        fn router_is_the_layer_gate() {
            assert_eq!(router(PREFIX).unwrap(), "model.layers.5.mlp.gate.weight");
        }

        #[test]
        fn each_projection_indexes_by_expert_id() {
            assert_eq!(
                gate_proj(PREFIX, 12).unwrap(),
                "model.layers.5.mlp.experts.12.gate_proj.weight"
            );
            assert_eq!(
                up_proj(PREFIX, 12).unwrap(),
                "model.layers.5.mlp.experts.12.up_proj.weight"
            );
            assert_eq!(
                down_proj(PREFIX, 12).unwrap(),
                "model.layers.5.mlp.experts.12.down_proj.weight"
            );
        }

        #[test]
        fn expert_zero_is_not_special_cased() {
            // Off-by-one here would silently mis-address every expert.
            assert_eq!(
                gate_proj(PREFIX, 0).unwrap(),
                "model.layers.5.mlp.experts.0.gate_proj.weight"
            );
        }

        #[test]
        fn distinct_experts_get_distinct_keys() {
            assert_ne!(gate_proj(PREFIX, 0), gate_proj(PREFIX, 1));
        }

        #[test]
        fn the_three_projections_are_distinct() {
            let keys = [
                gate_proj(PREFIX, 3),
                up_proj(PREFIX, 3),
                down_proj(PREFIX, 3),
            ];
            let mut unique: Vec<_> = keys.iter().flatten().collect();
            unique.sort();
            unique.dedup();
            assert_eq!(unique.len(), 3);
        }

        #[test]
        fn the_router_is_not_an_expert_key() {
            assert_ne!(router(PREFIX), gate_proj(PREFIX, 0));
        }
    }
}

/// Names the MXFP4 loader gives to expert weights it has **dequantised**.
///
/// GPT-OSS ships its experts packed and fused, so there are no per-expert
/// tensors on disk. `loading::safetensors::load_mxfp4_expert_tensors`
/// dequantises them, de-interleaves gate from up, and stores each expert
/// separately under Mixtral's `w1`/`w2`/`w3` naming. Those synthesised keys
/// are the only handle a compute backend has on GPT-OSS expert weights, so
/// the convention has to be shared between the loader that writes them and
/// the architecture that advertises them — a drift here means the backend
/// asks for a key that exists under another name and silently gets nothing.
pub(crate) mod mxfp4_dequantised {
    /// Container segment the loader nests dequantised experts under.
    const EXPERTS_PREFIX: &str = "block_sparse_moe.experts";
    /// Mixtral's name for the gate projection.
    const GATE: &str = "w1";
    /// Mixtral's name for the down projection.
    const DOWN: &str = "w2";
    /// Mixtral's name for the up projection.
    const UP: &str = "w3";

    /// `<prefix>block_sparse_moe.experts.<id>.w1.weight` — gate.
    pub(crate) fn gate_proj(layer_prefix: &str, expert_id: usize) -> Option<String> {
        Some(projection(layer_prefix, expert_id, GATE))
    }

    /// `<prefix>block_sparse_moe.experts.<id>.w3.weight` — up.
    pub(crate) fn up_proj(layer_prefix: &str, expert_id: usize) -> Option<String> {
        Some(projection(layer_prefix, expert_id, UP))
    }

    /// `<prefix>block_sparse_moe.experts.<id>.w2.weight` — down.
    pub(crate) fn down_proj(layer_prefix: &str, expert_id: usize) -> Option<String> {
        Some(projection(layer_prefix, expert_id, DOWN))
    }

    /// The shape all three projections share. `layer_prefix` carries its own
    /// trailing separator, matching [`ModelArchitecture::layer_prefix`].
    pub(crate) fn projection(layer_prefix: &str, expert_id: usize, which: &str) -> String {
        format!("{layer_prefix}{EXPERTS_PREFIX}.{expert_id}.{which}.weight")
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        const PREFIX: &str = "layers.0.";

        #[test]
        fn projections_use_mixtral_w_naming() {
            assert_eq!(
                gate_proj(PREFIX, 0).unwrap(),
                "layers.0.block_sparse_moe.experts.0.w1.weight"
            );
            assert_eq!(
                up_proj(PREFIX, 0).unwrap(),
                "layers.0.block_sparse_moe.experts.0.w3.weight"
            );
            assert_eq!(
                down_proj(PREFIX, 0).unwrap(),
                "layers.0.block_sparse_moe.experts.0.w2.weight"
            );
        }

        /// `w2` is down and `w3` is up. Swapping them is a silent transpose
        /// mismatch that only shows up as wrong numbers.
        #[test]
        fn up_is_w3_and_down_is_w2() {
            assert!(up_proj(PREFIX, 1).unwrap().contains(".w3."));
            assert!(down_proj(PREFIX, 1).unwrap().contains(".w2."));
        }

        #[test]
        fn distinct_experts_get_distinct_keys() {
            assert_ne!(gate_proj(PREFIX, 0), gate_proj(PREFIX, 31));
        }

        #[test]
        fn the_three_projections_are_distinct() {
            let keys = [
                gate_proj(PREFIX, 4),
                up_proj(PREFIX, 4),
                down_proj(PREFIX, 4),
            ];
            let mut unique: Vec<_> = keys.iter().flatten().collect();
            unique.sort();
            unique.dedup();
            assert_eq!(unique.len(), 3);
        }
    }
}

/// Standard HuggingFace attention projection-bias names.
///
/// Declared by GPT-2, StarCoder2, Qwen2/2.5, and GPT-OSS. All four are
/// byte-identical because the names come from `transformers`'
/// `*Attention` module layout, not from anything model-specific.
pub(crate) mod attn_bias {
    /// Query-projection bias, relative to a layer prefix.
    const Q_SUFFIX: &str = "self_attn.q_proj.bias";
    /// Key-projection bias, relative to a layer prefix.
    const K_SUFFIX: &str = "self_attn.k_proj.bias";
    /// Value-projection bias, relative to a layer prefix.
    const V_SUFFIX: &str = "self_attn.v_proj.bias";
    /// Output-projection bias, relative to a layer prefix.
    const O_SUFFIX: &str = "self_attn.o_proj.bias";

    /// `<prefix>self_attn.q_proj.bias`.
    pub(crate) fn q(layer_prefix: &str) -> Option<String> {
        Some(format!("{layer_prefix}{Q_SUFFIX}"))
    }

    /// `<prefix>self_attn.k_proj.bias`.
    pub(crate) fn k(layer_prefix: &str) -> Option<String> {
        Some(format!("{layer_prefix}{K_SUFFIX}"))
    }

    /// `<prefix>self_attn.v_proj.bias`.
    pub(crate) fn v(layer_prefix: &str) -> Option<String> {
        Some(format!("{layer_prefix}{V_SUFFIX}"))
    }

    /// `<prefix>self_attn.o_proj.bias`.
    pub(crate) fn o(layer_prefix: &str) -> Option<String> {
        Some(format!("{layer_prefix}{O_SUFFIX}"))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        const PREFIX: &str = "model.layers.7.";

        #[test]
        fn each_builder_appends_its_suffix_to_the_prefix() {
            assert_eq!(q(PREFIX).unwrap(), "model.layers.7.self_attn.q_proj.bias");
            assert_eq!(k(PREFIX).unwrap(), "model.layers.7.self_attn.k_proj.bias");
            assert_eq!(v(PREFIX).unwrap(), "model.layers.7.self_attn.v_proj.bias");
            assert_eq!(o(PREFIX).unwrap(), "model.layers.7.self_attn.o_proj.bias");
        }

        #[test]
        fn an_empty_prefix_yields_the_bare_suffix() {
            // Architectures whose layers are top-level (`layers.N.`) pass
            // a prefix; a prefixless model must still get a usable key.
            assert_eq!(q("").unwrap(), Q_SUFFIX);
        }

        #[test]
        fn the_four_keys_are_distinct() {
            let keys = [q(PREFIX), k(PREFIX), v(PREFIX), o(PREFIX)];
            let mut unique: Vec<_> = keys.iter().flatten().collect();
            unique.sort();
            unique.dedup();
            assert_eq!(unique.len(), 4, "a copy-paste slip would collapse these");
        }
    }
}
