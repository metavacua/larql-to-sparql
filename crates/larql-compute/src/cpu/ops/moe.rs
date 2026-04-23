//! CPU-side MoE (Mixture-of-Experts) forward pass for hybrid models (Gemma 4 26B A4B).
//!
//! Called when a layer has `is_hybrid_moe() == true`. Computes the expert block
//! in parallel with the dense FFN and returns the expert contribution for summation.
//!
//! Flow (per Gemma 4 architecture):
//!   pre_experts_norm(h) → router_scale * h_norm → router_proj → softmax → top-k
//!   → for each selected expert: gate_proj, up_proj, SiLU(gate)*up, down_proj
//!   → weighted_sum(expert_outs * router_weights * per_expert_scale)
//!
//! Expert weights are stored as packed BF16: [num_experts, out_dim, in_dim].
//! We dequantize only the selected top-k expert slices on demand.

use crate::MoeLayerWeights;

/// Dequantize a BF16 byte slice to f32.
#[inline]
fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(2)
        .map(|b| f32::from_bits((u32::from(u8::from_le_bytes([b[0]])) | (u32::from(u8::from_le_bytes([b[1]])) << 8)) << 16))
        .collect()
}

/// Extract one expert's weight slice from packed BF16 tensor and dequantize to f32.
/// Packed layout: [num_experts, out_rows, in_cols] — expert `e` starts at byte
/// `e * out_rows * in_cols * 2`.
fn extract_expert_weights(
    packed: &[u8],
    expert_idx: usize,
    out_rows: usize,
    in_cols: usize,
) -> Vec<f32> {
    let bytes_per_expert = out_rows * in_cols * 2;
    let start = expert_idx * bytes_per_expert;
    let end = start + bytes_per_expert;
    bf16_to_f32(&packed[start..end])
}

/// RMSNorm: out[i] = x[i] / rms(x) * w[i] + w[i] * norm_offset
fn rms_norm(x: &[f32], w: &[f32], eps: f32, offset: f32) -> Vec<f32> {
    if w.is_empty() || x.is_empty() { return x.to_vec(); }
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    x.iter().zip(w.iter()).map(|(&xi, &wi)| xi / rms * (wi + offset)).collect()
}

/// SiLU activation: x * sigmoid(x)
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// GELU with tanh approximation (Gemma 4 expert FFN activation).
#[inline]
fn gelu_tanh(x: f32) -> f32 {
    let c = 0.797_884_6_f32;
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

/// Compute y = x @ W.T where W is [out_rows, in_cols] stored row-major.
fn matmul_vec(x: &[f32], w: &[f32], out_rows: usize, in_cols: usize) -> Vec<f32> {
    debug_assert_eq!(w.len(), out_rows * in_cols);
    debug_assert_eq!(x.len(), in_cols);
    (0..out_rows).map(|row| {
        let w_row = &w[row * in_cols..(row + 1) * in_cols];
        x.iter().zip(w_row.iter()).map(|(a, b)| a * b).sum()
    }).collect()
}

/// Softmax in-place.
fn softmax(v: &mut [f32]) {
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() { *x = (*x - max).exp(); sum += *x; }
    if sum > 0.0 { for x in v.iter_mut() { *x /= sum; } }
}

/// Top-k indices by value (descending). Returns (indices, values).
fn top_k(v: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
    let k = k.min(v.len());
    let mut indexed: Vec<(usize, f32)> = v.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    let indices: Vec<usize> = indexed.iter().map(|(i, _)| *i).collect();
    let values: Vec<f32> = indexed.iter().map(|(_, v)| *v).collect();
    (indices, values)
}

/// Run a single expert's gated FFN given a pre-normed input vector.
///
/// Returns the expert's output (not yet weighted by router probability).
/// `h_norm` must already be RMS-normed — use `run_single_expert_with_norm`
/// when you have the raw residual.
pub fn run_single_expert(
    h_norm: &[f32],
    experts_gate_up: &[u8],
    experts_down: &[u8],
    expert_idx: usize,
    inter: usize,
    activation: crate::Activation,
) -> Vec<f32> {
    let hidden = h_norm.len();
    if inter == 0 || hidden == 0 { return vec![0.0f32; hidden]; }

    let gate_up_w = extract_expert_weights(experts_gate_up, expert_idx, 2 * inter, hidden);
    let gate_w = &gate_up_w[..inter * hidden];
    let up_w = &gate_up_w[inter * hidden..];

    let gate_out = matmul_vec(h_norm, gate_w, inter, hidden);
    let up_out = matmul_vec(h_norm, up_w, inter, hidden);

    let hidden_state: Vec<f32> = gate_out.iter().zip(up_out.iter())
        .map(|(&g, &u)| match activation {
            crate::Activation::GeluTanh => gelu_tanh(g) * u,
            _ => silu(g) * u,
        })
        .collect();

    let down_w = extract_expert_weights(experts_down, expert_idx, hidden, inter);
    matmul_vec(&hidden_state, &down_w, hidden, inter)
}

/// Apply pre-experts norm then run a single expert. Used by the remote
/// expert server endpoint where the raw residual arrives from the client.
pub fn run_single_expert_with_norm(
    h: &[f32],
    experts_gate_up: &[u8],
    experts_down: &[u8],
    expert_idx: usize,
    inter: usize,
    pre_experts_norm: &[f32],
    norm_offset: f32,
    eps: f32,
    activation: crate::Activation,
) -> Vec<f32> {
    let h_norm = rms_norm(h, pre_experts_norm, eps, norm_offset);
    run_single_expert(&h_norm, experts_gate_up, experts_down, expert_idx, inter, activation)
}

/// Run the MoE expert block for one token.
///
/// `h` — residual stream at this layer (hidden_size f32 values).
/// Returns the expert block contribution to add to the dense FFN output.
/// If `moe` is missing required fields, returns a zero vector of hidden_size.
pub fn cpu_moe_forward(h: &[f32], moe: &MoeLayerWeights<'_>, norm_offset: f32, eps: f32) -> Vec<f32> {
    let hidden = h.len();
    let num_experts = moe.num_experts;
    let top_k_val = moe.top_k;
    let inter = moe.intermediate_size;

    if num_experts == 0 || top_k_val == 0 || inter == 0 {
        return vec![0.0f32; hidden];
    }
    if moe.router_proj.is_empty() || moe.experts_gate_up.is_empty() || moe.experts_down.is_empty() {
        return vec![0.0f32; hidden];
    }

    // 1. Pre-experts norm
    let h_norm = rms_norm(h, moe.pre_experts_norm, eps, norm_offset);

    // 2. Router scale (Gemma4TextRouter: scale input before projection)
    let h_scaled: Vec<f32> = if !moe.router_scale.is_empty() {
        h_norm.iter().zip(moe.router_scale.iter()).map(|(a, b)| a * b).collect()
    } else {
        h_norm.clone()
    };

    // 3. Router projection: [hidden] → [num_experts]
    let mut logits = matmul_vec(&h_scaled, moe.router_proj, num_experts, hidden);

    // 4. Softmax
    softmax(&mut logits);

    // 5. Top-k selection
    let (expert_indices, mut expert_weights) = top_k(&logits, top_k_val);

    // Debug: print routing per layer if MOE_DEBUG=1
    static DEBUG_LAYER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    if std::env::var("MOE_DEBUG").is_ok() {
        let layer_n = DEBUG_LAYER.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 30;
        let h_rms = (h.iter().map(|v| v*v).sum::<f32>() / h.len() as f32).sqrt();
        let hn_rms = (h_norm.iter().map(|v| v*v).sum::<f32>() / h_norm.len() as f32).sqrt();
        let hs_rms = (h_scaled.iter().map(|v| v*v).sum::<f32>() / h_scaled.len() as f32).sqrt();
        let logit_max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let logit_min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
        let pnorm_rms = (moe.pre_experts_norm.iter().map(|v| v*v).sum::<f32>() / moe.pre_experts_norm.len().max(1) as f32).sqrt();
        let rscale_rms = (moe.router_scale.iter().map(|v| v*v).sum::<f32>() / moe.router_scale.len().max(1) as f32).sqrt();
        eprintln!("[L{layer_n:02}] h_rms={h_rms:.2} hn_rms={hn_rms:.2} hs_rms={hs_rms:.2} | pnorm_rms={pnorm_rms:.2} rscale_rms={rscale_rms:.2} | logits [{logit_min:.3}..{logit_max:.3}] | experts:{expert_indices:?}");
    }

    // 6. Renormalize selected weights to sum to 1 (Gemma 4 gemma4_top_k_softmax).
    // After softmax over all 128 experts, the selected top-8 weights sum to
    // ~0.5-0.7, not 1.0.  Renormalising ensures the expert block contributes
    // at the correct scale.  Without this the expert residual is undersized
    // every layer and the model output is garbage.
    let weight_sum: f32 = expert_weights.iter().sum();
    if weight_sum > 0.0 {
        for w in &mut expert_weights { *w /= weight_sum; }
    }

    // 7. Per-expert output scale (Gemma 4 learned per-expert scale)
    if !moe.router_per_expert_scale.is_empty() {
        for (i, &ei) in expert_indices.iter().enumerate() {
            if ei < moe.router_per_expert_scale.len() {
                expert_weights[i] *= moe.router_per_expert_scale[ei];
            }
        }
    }

    // 7. Run each selected expert's gated FFN (BF16 dequant on demand)
    //    gate_up layout: [num_experts, 2*inter, hidden]  (gate rows first, then up rows)
    //    down layout:    [num_experts, hidden, inter]
    let mut expert_out = vec![0.0f32; hidden];
    for (rank, &ei) in expert_indices.iter().enumerate() {
        let weight = expert_weights[rank];
        if weight == 0.0 { continue; }

        // Extract gate+up weights for this expert: [2*inter, hidden]
        let gate_up_w = extract_expert_weights(moe.experts_gate_up, ei, 2 * inter, hidden);
        // gate: rows [0..inter], up: rows [inter..2*inter]
        let gate_w = &gate_up_w[..inter * hidden];
        let up_w = &gate_up_w[inter * hidden..];

        let gate_out = matmul_vec(&h_norm, gate_w, inter, hidden);
        let up_out = matmul_vec(&h_norm, up_w, inter, hidden);

        // Gated activation: ACT(gate) * up.  Gemma 4 uses GELU-tanh; Mixtral uses SiLU.
        let hidden_state: Vec<f32> = gate_out.iter().zip(up_out.iter())
            .map(|(&g, &u)| match moe.activation {
                crate::Activation::GeluTanh => gelu_tanh(g) * u,
                _ => silu(g) * u,
            })
            .collect();

        // Down projection: [inter] → [hidden]
        let down_w = extract_expert_weights(moe.experts_down, ei, hidden, inter);
        let expert_contribution = matmul_vec(&hidden_state, &down_w, hidden, inter);

        // Accumulate weighted
        for (acc, &val) in expert_out.iter_mut().zip(expert_contribution.iter()) {
            *acc += val * weight;
        }
    }

    // 8. Post-experts norm
    let result = rms_norm(&expert_out, moe.post_experts_norm, eps, norm_offset);

    if std::env::var("MOE_DEBUG").is_ok() {
        let pre_rms = (expert_out.iter().map(|v| v*v).sum::<f32>() / expert_out.len() as f32).sqrt();
        let post_rms = (result.iter().map(|v| v*v).sum::<f32>() / result.len() as f32).sqrt();
        let pnorm2_rms = (moe.post_experts_norm.iter().map(|v| v*v).sum::<f32>() / moe.post_experts_norm.len().max(1) as f32).sqrt();
        eprintln!("  pre_norm_rms={pre_rms:.3} post_norm2_rms={pnorm2_rms:.3} moe_out_rms={post_rms:.3}");
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_moe<'a>(
        hidden: usize, inter: usize, num_experts: usize, top_k: usize,
        gate_up: &'a [u8], down: &'a [u8], router: &'a [f32],
    ) -> MoeLayerWeights<'a> {
        MoeLayerWeights {
            experts_gate_up: gate_up,
            experts_down: down,
            router_proj: router,
            router_scale: &[],
            router_per_expert_scale: &[],
            pre_experts_norm: &[],
            post_ffn1_norm: &[],
            post_experts_norm: &[],
            num_experts,
            top_k,
            intermediate_size: inter,
            activation: crate::Activation::Silu,
        }
    }

    #[test]
    fn test_moe_zero_input_produces_zero() {
        let hidden = 8;
        let inter = 4;
        let num_experts = 4;
        let top_k = 2;

        // All-zero BF16 weights (value 0.0 in BF16 = 0x0000)
        let gate_up = vec![0u8; num_experts * 2 * inter * hidden * 2];
        let down = vec![0u8; num_experts * hidden * inter * 2];
        let router = vec![0.0f32; num_experts * hidden];

        let moe = make_moe(hidden, inter, num_experts, top_k, &gate_up, &down, &router);
        let h = vec![1.0f32; hidden];
        let out = cpu_moe_forward(&h, &moe, 0.0, 1e-6);
        assert_eq!(out.len(), hidden);
        assert!(out.iter().all(|v| v.abs() < 1e-5), "zero weights → zero output");
    }

    #[test]
    fn test_moe_identity_expert() {
        // Construct a single expert that acts as identity via gate≫0, up=1, down=identity
        // This verifies the full path runs without panics.
        let hidden = 4;
        let inter = 2;
        let num_experts = 2;
        let top_k = 1;

        // BF16 encoding of 1.0 = 0x3F80
        let one_bf16 = [0x80u8, 0x3Fu8];
        // BF16 encoding of 5.0 (large gate → SiLU ≈ 5) = 0x40A0
        let five_bf16 = [0xA0u8, 0x40u8];

        // gate_up: [num_experts, 2*inter, hidden] — expert 0: gate rows = 5.0, up rows = 1.0
        let mut gate_up = vec![0u8; num_experts * 2 * inter * hidden * 2];
        // Expert 0, gate rows (rows 0..inter): set to 5.0
        for row in 0..inter {
            for col in 0..hidden {
                let byte_off = (row * hidden + col) * 2;
                gate_up[byte_off] = five_bf16[0];
                gate_up[byte_off + 1] = five_bf16[1];
            }
        }
        // Expert 0, up rows (rows inter..2*inter): set to 1.0
        for row in inter..2*inter {
            for col in 0..hidden {
                let byte_off = (row * hidden + col) * 2;
                gate_up[byte_off] = one_bf16[0];
                gate_up[byte_off + 1] = one_bf16[1];
            }
        }

        // down: [num_experts, hidden, inter] — expert 0: 1.0 everywhere
        let mut down = vec![0u8; num_experts * hidden * inter * 2];
        for i in 0..(hidden * inter) {
            let byte_off = i * 2;
            down[byte_off] = one_bf16[0];
            down[byte_off + 1] = one_bf16[1];
        }

        // router: [num_experts, hidden] — expert 0 row has 1.0, expert 1 row has 0.0
        let mut router = vec![0.0f32; num_experts * hidden];
        for col in 0..hidden { router[col] = 1.0; } // expert 0 gets high logit

        let moe = make_moe(hidden, inter, num_experts, top_k, &gate_up, &down, &router);
        let h = vec![1.0f32; hidden];
        let out = cpu_moe_forward(&h, &moe, 0.0, 1e-6);
        assert_eq!(out.len(), hidden);
        // Output should be nonzero since gate activates
        assert!(out.iter().any(|v| v.abs() > 0.01), "expected nonzero output from identity-like expert");
    }
}
