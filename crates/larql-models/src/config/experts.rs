//! Mixture-of-experts storage format and the two policies that decide
//! how an expert block computes: its gate shape and its router normalisation.

/// How expert weights are stored in a MoE model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExpertFormat {
    /// Per-expert separate tensors (Mixtral, DeepSeek).
    /// Keys: `experts.{id}.w1.weight`, `experts.{id}.w2.weight`, etc.
    PerExpert,
    /// Packed MXFP4 (GPT-OSS/OpenAI).
    /// All experts fused into one tensor with block quantization.
    /// Keys: `experts.gate_up_proj_blocks`, `experts.gate_up_proj_scales`, etc.
    PackedMxfp4,
    /// Packed BF16/F16 stacked tensors (Gemma 4 26B A4B).
    /// All experts fused into one tensor per projection, no quantization scales.
    /// Keys: `experts.gate_up_proj` [num_experts, 2*moe_intermediate, hidden],
    ///        `experts.down_proj`   [num_experts, hidden, moe_intermediate].
    PackedBF16,
}

/// Which rows of a fused `gate_up` operand belong to the gate branch and
/// which to the up branch.
///
/// **Orthogonal to [`ExpertFormat`], and deliberately so.** Two packed
/// architectures already disagree, so "packed" cannot imply a layout:
///
/// | family     | reference split                        | layout             |
/// |------------|----------------------------------------|--------------------|
/// | GraniteMoE | `hidden_states.chunk(2, dim=-1)`       | `ContiguousHalves` |
/// | GPT-OSS    | `gate_up[..., ::2], gate_up[..., 1::2]`| `Interleaved`      |
///
/// Reading one as the other does not fail — it silently mixes the two
/// branches and produces plausible garbage, which is why this is a
/// declaration rather than something a reader infers from a shape. It is
/// also distinct from [`MoeWeightLayout`](../../../larql_compute/index.html),
/// which describes down-projection *padding*; keeping the two names
/// semantically narrow is deliberate.
///
/// This describes the **checkpoint's** operand. A representation transform
/// (e.g. K3's MXFP4→Q6_K transcode) may canonicalise it on the way to an
/// execution operand; when that happens the transform owns the invariant,
/// not this declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateUpLayout {
    /// `[all gate rows | all up rows]` — the first half is gate.
    ContiguousHalves,
    /// `gate = rows[0], rows[2], …`, `up = rows[1], rows[3], …`.
    Interleaved,
}

/// Which branch of a fused `gate_up` operand a row belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateUpBranch {
    Gate,
    Up,
}

impl GateUpLayout {
    /// Row index, within one expert's fused `gate_up` operand, holding
    /// element `i` of `branch` — where the operand has `half` rows per
    /// branch.
    ///
    /// This is the **entire** shared surface between the reference tier and
    /// any execution path: an indexing rule, not an operation. Sharing a
    /// `compute_gate_up()` helper instead would let one addressing bug make
    /// oracle and engine agree with each other and disagree with reality.
    pub const fn row(self, branch: GateUpBranch, i: usize, half: usize) -> usize {
        match (self, branch) {
            (Self::ContiguousHalves, GateUpBranch::Gate) => i,
            (Self::ContiguousHalves, GateUpBranch::Up) => half + i,
            (Self::Interleaved, GateUpBranch::Gate) => 2 * i,
            (Self::Interleaved, GateUpBranch::Up) => 2 * i + 1,
        }
    }
}

/// How an expert's fused gate/up projection becomes the down projection's
/// input.
///
/// Most MoE models are a plain gated FFN. GPT-OSS is not, and the difference
/// is not cosmetic: it clamps both halves, scales the sigmoid argument, and
/// adds one to the up branch. Modelling that as "SiLU with extra steps" is how
/// a forward pass ends up plausibly wrong — hence an explicit policy rather
/// than an [`Activation`] variant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExpertGatePolicy {
    /// `activation(gate) * up` — Mixtral, Gemma 4, OLMoE, GraniteMoE.
    Gated,
    /// GPT-OSS's clamped GLU, from `GptOssExperts._apply_gate`:
    ///
    /// ```text
    /// g   = gate.clamp(max = limit)          // upper bound only
    /// u   = up.clamp(-limit, limit)          // symmetric
    /// glu = g * sigmoid(alpha * g)
    /// out = (u + 1) * glu
    /// ```
    ClampedGlu {
        /// Clamp bound (`swiglu_limit`, 7.0 on the released checkpoints).
        limit: f32,
        /// Multiplier on the sigmoid argument (1.702 in the reference).
        alpha: f32,
    },
}

/// How a router's top-k weights are normalised.
///
/// There are only two observable behaviours, and the difference is whether the
/// selected weights sum to 1. Getting it wrong rescales the entire expert
/// branch, which is a large error that still produces coherent-looking output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertRoutingPolicy {
    /// Softmax over **all** experts, then keep the top-k probabilities as they
    /// are. They sum to *less* than 1 — by whatever mass the unselected
    /// experts hold. Mixtral and OLMoE with `norm_topk_prob: false`.
    SoftmaxThenSelect,
    /// The selected weights are normalised to sum to 1.
    ///
    /// Two routes arrive here and they are algebraically identical, which is
    /// why one variant covers both: renormalising the top-k probabilities
    /// (`norm_topk_prob: true`, Gemma 4), or softmaxing over just the selected
    /// logits (GPT-OSS) —
    /// `softmax(l)_i / Σ_{j∈topk} softmax(l)_j = exp(l_i) / Σ_{j∈topk} exp(l_j)`.
    NormalisedOverSelected,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The indexing rule against both reference splits, and the property that
    /// makes a wrong declaration dangerous: each layout is a *permutation* of
    /// the same row block, so no bounds or shape check can catch the wrong
    /// one — every row is visited exactly once either way.
    #[test]
    fn gate_up_row_maps_each_branch_and_stays_a_permutation() {
        const HALF: usize = 4;

        // GraniteMoe: `hidden_states.chunk(2, dim=-1)` — leading half is gate.
        let c = GateUpLayout::ContiguousHalves;
        assert_eq!(c.row(GateUpBranch::Gate, 0, HALF), 0);
        assert_eq!(c.row(GateUpBranch::Gate, 3, HALF), 3);
        assert_eq!(c.row(GateUpBranch::Up, 0, HALF), 4);
        assert_eq!(c.row(GateUpBranch::Up, 3, HALF), 7);

        // GPT-OSS: `gate_up[..., ::2]` / `[..., 1::2]` — even gate, odd up.
        let i = GateUpLayout::Interleaved;
        assert_eq!(i.row(GateUpBranch::Gate, 0, HALF), 0);
        assert_eq!(i.row(GateUpBranch::Gate, 3, HALF), 6);
        assert_eq!(i.row(GateUpBranch::Up, 0, HALF), 1);
        assert_eq!(i.row(GateUpBranch::Up, 3, HALF), 7);

        for layout in [c, i] {
            let mut rows: Vec<usize> = (0..HALF)
                .flat_map(|k| {
                    [
                        layout.row(GateUpBranch::Gate, k, HALF),
                        layout.row(GateUpBranch::Up, k, HALF),
                    ]
                })
                .collect();
            rows.sort_unstable();
            assert_eq!(rows, (0..2 * HALF).collect::<Vec<_>>());
        }

        // The two disagree everywhere except the first gate row — which is
        // why a probe at index 0 alone would not distinguish them.
        assert_eq!(
            c.row(GateUpBranch::Gate, 0, HALF),
            i.row(GateUpBranch::Gate, 0, HALF)
        );
        assert_ne!(
            c.row(GateUpBranch::Up, 0, HALF),
            i.row(GateUpBranch::Up, 0, HALF)
        );
    }
}
