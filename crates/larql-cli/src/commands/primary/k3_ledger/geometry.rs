//! K3 checkpoint geometry — the measured shape of the model, and nothing derived.
//!
//! Pure: every field here comes from a real `config.json` field or a real
//! safetensors header entry. The arithmetic that *consumes* this lives in
//! `budget`, `touch`, `frontier` and `block`, so that each of those is a pure
//! function of a `K3Geometry` and can be unit-tested without a network.
//!
//! That split is deliberate. Every error this ladder caught in July 2026 was an
//! arithmetic error over correct measurements — a routed-zeroed constant reused
//! as a routed-side divisor, one variable serving as both union index and
//! amortisation divisor. Those are exactly the mistakes a unit test catches and
//! a script does not.

use serde::{Deserialize, Serialize};

/// Bytes per weight for MXFP4: 4-bit codeword plus one 8-bit block scale per 32.
///
/// Two conventions travel with any bit-width quoted against this (standing rule
/// R0): *payload* bits are the codeword alone, *all-in* bits include the scale.
/// MXFP4 is 4.0 payload / 4.25 all-in. They diverge by 0.25 bits, which is
/// immaterial at 4-bit and decisive at 1-bit.
pub const MXFP4_BYTES_PER_WEIGHT: f64 = 0.53125;
pub const MXFP4_PAYLOAD_BITS: f64 = 4.0;
pub const SCALE_BITS_PER_WEIGHT: f64 = 8.0 / 32.0;

/// The measured shape of a K3-family checkpoint.
///
/// Constructed by `fetch` from `config.json` plus two safetensors headers (one
/// KDA layer, one MLA layer). Never inferred: `n_kda_layers` and `n_mla_layers`
/// come from the config's explicit `kda_layers` / `full_attn_layers` lists, and
/// the per-layer parameter counts are summed from real tensor shapes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct K3Geometry {
    pub hidden_size: usize,
    pub latent_dim: usize,
    pub moe_intermediate: usize,
    pub n_layers: usize,
    pub n_dense_layers: usize,
    pub n_kda_layers: usize,
    pub n_mla_layers: usize,
    pub n_experts: usize,
    pub top_k: usize,
    /// Total MXFP4 bytes for one routed expert, all three branches.
    pub expert_bytes: u64,
    /// Per-branch bytes, in `w1` (gate), `w2` (down), `w3` (up) order.
    pub branch_bytes: [u64; 3],
    /// Non-expert parameters in one KDA layer (five [12288,7168] projections dominate).
    pub kda_layer_dense_params: u64,
    /// Non-expert parameters in one MLA layer (q_a/q_b/kv_a/kv_b compress the same role).
    pub mla_layer_dense_params: u64,
    /// Embedding + LM head parameters. Vision is EXCLUDED — see `vision_excluded`.
    pub embedding_params: u64,
    /// Always false-if-honest: the ledger sums `language_model` layers plus text
    /// embeddings only. `vision_tower` + `mm_projector` are ~2 GB BF16 by
    /// residual against the published repo size, under 2% of the dense term. A
    /// multimodal demo means recomputing this, not adjusting it (R0).
    pub vision_included: bool,
}

impl K3Geometry {
    /// MoE layers — every layer but the leading dense ones.
    pub fn n_moe_layers(&self) -> usize {
        self.n_layers.saturating_sub(self.n_dense_layers)
    }

    /// Always-on parameters read once per forward pass, whatever it emits.
    pub fn dense_params(&self) -> u64 {
        self.n_kda_layers as u64 * self.kda_layer_dense_params
            + self.n_mla_layers as u64 * self.mla_layer_dense_params
            + self.embedding_params
    }

    /// Routed-expert parameters activated per token position.
    pub fn routed_activated_params(&self) -> u64 {
        // One expert's weights = its packed bytes at 2 weights/byte, summed over
        // branches. Derived from bytes rather than shapes so it stays correct if
        // the codec changes.
        let per_expert = (self.expert_bytes as f64 / MXFP4_BYTES_PER_WEIGHT) as u64;
        per_expert * self.top_k as u64 * self.n_moe_layers() as u64
    }

    /// The headline "activated parameters" figure.
    pub fn activated_params(&self) -> u64 {
        self.dense_params() + self.routed_activated_params()
    }

    /// Always-on bytes at a given bits-per-weight (all-in, scales included).
    pub fn dense_bytes(&self, all_in_bits: f64) -> f64 {
        self.dense_params() as f64 * all_in_bits / 8.0
    }

    /// Routed bytes per token position, raw, before any lever.
    pub fn routed_bytes_per_position(&self) -> f64 {
        (self.top_k * self.n_moe_layers()) as f64 * self.expert_bytes as f64
    }

    /// Distinct experts expected across `n` independent top-k draws.
    ///
    /// `E * (1 - (1 - k/E)^n)`. Depends on `k/E`, not on `n` alone — standing
    /// rule R2 exists because this was once transferred across models with
    /// different activation fractions.
    pub fn expert_union(&self, n: u32) -> f64 {
        let e = self.n_experts as f64;
        let k = self.top_k as f64;
        e * (1.0 - (1.0 - k / e).powi(n as i32))
    }

    /// Fraction of experts a single token activates.
    pub fn activation_fraction(&self) -> f64 {
        self.top_k as f64 / self.n_experts as f64
    }

    /// True when all three GLU branches are byte-identical, which caps the
    /// "drop one branch" lever at exactly 1.5x.
    pub fn branches_are_equal(&self) -> bool {
        self.branch_bytes[0] == self.branch_bytes[1] && self.branch_bytes[1] == self.branch_bytes[2]
    }

    /// Retention ceiling from folding the up branch away, when branches are equal.
    ///
    /// This is the *banked* routed-side floor: `w3` folds to per-feature scalars,
    /// `w1` and `w2` must still be read. Anything below it is row sparsity, which
    /// is a separate and currently unmeasured factor (owner DEC-8.1).
    pub fn up_fold_retention(&self) -> f64 {
        let total: u64 = self.branch_bytes.iter().sum();
        (total - self.branch_bytes[2]) as f64 / total as f64
    }
}

/// Effective bits per weight implied by a byte budget over a parameter count.
///
/// `convention` selects payload (codeword only) or all-in (scales included).
/// Never quote one without saying which — R0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BitConvention {
    /// Codeword width alone. Compare against a codec *called* "4-bit".
    Payload,
    /// Total wire bytes per weight. MXFP4 is 4.25 here, not 4.0.
    AllIn,
}

impl BitConvention {
    pub fn bits(self, byte_budget: f64, params: u64) -> f64 {
        if params == 0 {
            return f64::NAN;
        }
        let all_in = byte_budget * 8.0 / params as f64;
        match self {
            BitConvention::AllIn => all_in,
            BitConvention::Payload => all_in - SCALE_BITS_PER_WEIGHT,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            BitConvention::Payload => "payload",
            BitConvention::AllIn => "all-in",
        }
    }
}

#[cfg(test)]
pub(crate) fn k3_reference() -> K3Geometry {
    // The real moonshotai/Kimi-K3 geometry, measured 2026-07-31 from config.json
    // plus safetensors headers for tensor layers 49 (KDA) and 3 (MLA).
    K3Geometry {
        hidden_size: 7168,
        latent_dim: 3584,
        moe_intermediate: 3072,
        n_layers: 93,
        n_dense_layers: 1,
        n_kda_layers: 69,
        n_mla_layers: 24,
        n_experts: 896,
        top_k: 16,
        expert_bytes: 17_547_264,
        branch_bytes: [5_849_088, 5_849_088, 5_849_088],
        kda_layer_dense_params: 633_700_000,
        mla_layer_dense_params: 422_200_000,
        embedding_params: 2 * 163_840 * 7_168,
        vision_included: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "expected ~{b}, got {a}");
    }

    #[test]
    fn moe_layers_excludes_leading_dense() {
        assert_eq!(k3_reference().n_moe_layers(), 92);
    }

    #[test]
    fn kda_and_mla_layers_account_for_every_layer() {
        let g = k3_reference();
        assert_eq!(g.n_kda_layers + g.n_mla_layers, g.n_layers);
    }

    #[test]
    fn activated_params_match_the_published_figure() {
        // Published ~104.2B; measured 104.83B, agreeing to 0.6%.
        let g = k3_reference();
        approx(g.activated_params() as f64 / 1e9, 104.83, 0.15);
    }

    #[test]
    fn dense_half_is_the_larger_half() {
        // The finding that closed 20 tok/s single-token: no expert-side lever
        // touches the dense term, and the dense term is the bigger one.
        let g = k3_reference();
        assert!(
            g.dense_params() > g.routed_activated_params(),
            "dense {} should exceed routed {}",
            g.dense_params(),
            g.routed_activated_params()
        );
    }

    #[test]
    fn mla_layers_are_lighter_than_kda_layers() {
        // Extrapolating one KDA layer across all 93 over-counted by 4.7%.
        let g = k3_reference();
        assert!(g.mla_layer_dense_params < g.kda_layer_dense_params);
    }

    #[test]
    fn routed_bytes_per_position_is_the_known_figure() {
        approx(
            k3_reference().routed_bytes_per_position() / 1e9,
            25.83,
            0.01,
        );
    }

    #[test]
    fn branches_are_equal_so_the_fold_ceiling_is_exactly_two_thirds() {
        let g = k3_reference();
        assert!(g.branches_are_equal());
        approx(g.up_fold_retention(), 2.0 / 3.0, 1e-12);
    }

    #[test]
    fn union_depends_on_activation_fraction_not_count_alone() {
        // R2 in executable form: same draw count, different k/E, different union
        // ratio. Transferring a union number across models is what this forbids.
        let k3 = k3_reference();
        let mut gemma = k3_reference();
        gemma.n_experts = 128;
        gemma.top_k = 8;

        let ratio = |g: &K3Geometry, n: u32| g.expert_union(n) / (n as f64 * g.top_k as f64);
        assert!(
            ratio(&gemma, 64) < ratio(&k3, 64),
            "denser activation must amortise better at equal batch"
        );
    }

    #[test]
    fn union_is_bounded_by_the_bank_and_by_the_draws() {
        let g = k3_reference();
        approx(g.expert_union(1), g.top_k as f64, 1e-9);
        assert!(g.expert_union(100_000) <= g.n_experts as f64);
    }

    #[test]
    fn bit_conventions_differ_by_exactly_the_scale_overhead() {
        let g = k3_reference();
        let params = g.dense_params();
        let budget = g.dense_bytes(4.25);
        let all_in = BitConvention::AllIn.bits(budget, params);
        let payload = BitConvention::Payload.bits(budget, params);
        approx(all_in, 4.25, 1e-9);
        approx(all_in - payload, SCALE_BITS_PER_WEIGHT, 1e-12);
    }

    #[test]
    fn vision_is_excluded_by_construction() {
        assert!(!k3_reference().vision_included);
    }
}
