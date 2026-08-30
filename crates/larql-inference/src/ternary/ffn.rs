//! The BitNet FFN block and its RMS-norm helper.
//!
//! Split out of the former single-file `ternary.rs`; [`super`] shows
//! how the pieces compose.

// Siblings resolve through the parent's re-exports.
use larql_compute::cpu::ops::ternary_matvec::{
    matvec_i2s_a8_f32_into, matvec_i2s_a8_into, quantize_activation_i8, BitLinearWeight,
};

/// One BitLinear-FFN block.  Holds three ternary weight tensors
/// (gate, up, down) and the two RMSnorm scales (input, post-attn).
///
/// Layer ordering (BitNet b1.58 architecture):
///
/// ```text
///   x        : input residual                                  [hidden]
///   x_norm   = rmsnorm(x, ffn_norm.weight, eps)                [hidden]
///   gate     = matvec_i2s(gate.weight, x_norm) (* gate_scale)   [inter]
///   up       = matvec_i2s(up.weight,   x_norm) (* up_scale)     [inter]
///   hid      = (gate * gate) * up                                [inter]
///   hid_norm = rmsnorm(hid, ffn_sub_norm.weight, eps)            [inter]
///   y        = matvec_i2s(down.weight, hid_norm) (* down_scale)  [hidden]
///   x_out    = x + y                                              [hidden]
/// ```
///
/// `gate_scale`, `up_scale`, and `down_scale` are baked into the
/// [`BitLinearWeight::channel_scales`] of each tensor, so the
/// matvec call already returns scaled outputs.
pub struct BitNetFfn {
    pub gate: BitLinearWeight,
    pub up: BitLinearWeight,
    pub down: BitLinearWeight,
    /// Per-channel weight for the input RMSnorm (`ffn_norm.weight`),
    /// length = `hidden_size`.
    pub ffn_norm: Vec<f32>,
    /// Per-channel weight for the post-gate-up RMSnorm
    /// (`ffn_sub_norm.weight`), length = `intermediate_size`.
    pub ffn_sub_norm: Vec<f32>,
    /// RMSnorm epsilon (typically 1e-5).
    pub eps: f32,
}

impl BitNetFfn {
    /// Run one forward step: `x_out = x + ffn(rmsnorm(x))`.
    ///
    /// Allocates two scratch buffers (gate and hid).  For
    /// per-token-allocations-matter callers, see
    /// [`Self::forward_into`].
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let hidden = x.len();
        let inter = self.gate.rows;
        let mut gate = vec![0.0f32; inter];
        let mut up = vec![0.0f32; inter];
        let mut hid = vec![0.0f32; inter];
        let mut y = vec![0.0f32; hidden];
        self.forward_into(x, &mut gate, &mut up, &mut hid, &mut y);
        // Residual addition: y already holds the FFN output.
        for (yo, xi) in y.iter_mut().zip(x.iter()) {
            *yo += xi;
        }
        y
    }

    /// In-place variant that uses caller-provided scratch buffers.
    ///
    /// `gate`, `up`, `hid` must each be length `intermediate_size`.
    /// `y` must be length `hidden_size`.  All four buffers are
    /// overwritten.  Caller is responsible for the residual-add
    /// step (we leave it out so the caller can choose whether to
    /// add to `x` or to a pre-existing accumulator).
    pub fn forward_into(
        &self,
        x: &[f32],
        gate: &mut [f32],
        up: &mut [f32],
        hid: &mut [f32],
        y: &mut [f32],
    ) {
        let hidden = x.len();
        let inter = self.gate.rows;
        debug_assert_eq!(self.up.rows, inter);
        debug_assert_eq!(self.down.cols, inter);
        debug_assert_eq!(self.down.rows, hidden);
        debug_assert_eq!(gate.len(), inter);
        debug_assert_eq!(up.len(), inter);
        debug_assert_eq!(hid.len(), inter);
        debug_assert_eq!(y.len(), hidden);
        debug_assert_eq!(self.ffn_norm.len(), hidden);
        debug_assert_eq!(self.ffn_sub_norm.len(), inter);

        // 1. Input RMSnorm.  We do this in-place into the gate
        //    buffer (we'll overwrite gate immediately below) just
        //    so we don't allocate a third hidden-sized scratch.
        let mut x_norm = vec![0.0f32; hidden];
        rmsnorm_into(x, &self.ffn_norm, self.eps, &mut x_norm);

        // 2. gate = ternary(gate.weight) · x_norm
        //    up   = ternary(up.weight)   · x_norm
        //    Both projections share x_norm — quantise it to int8 once (A8)
        //    and feed both matvecs, instead of re-quantising per call.
        let (x_i8, x_scale) = quantize_activation_i8(&x_norm);
        matvec_i2s_a8_into(&self.gate, &x_i8, x_scale, gate).expect("gate shape");
        matvec_i2s_a8_into(&self.up, &x_i8, x_scale, up).expect("up shape");

        // 3. Squared-ReLU activation (BitNet b1.58 spec) +
        //    element-wise multiply with up.
        for ((g, u), h) in gate.iter().zip(up.iter()).zip(hid.iter_mut()) {
            let relu = g.max(0.0);
            *h = relu * relu * u;
        }

        // 4. Post-gate-up RMSnorm.
        let mut hid_norm = vec![0.0f32; inter];
        rmsnorm_into(hid, &self.ffn_sub_norm, self.eps, &mut hid_norm);

        // 5. y = ternary(down.weight) · hid_norm
        matvec_i2s_a8_f32_into(&self.down, &hid_norm, y).expect("down shape");
    }
}

/// RMS normalisation: `out[i] = (x[i] / rms(x)) * weight[i]`.
///
/// `rms(x) = sqrt(mean(x_i^2) + eps)`.  Standard transformer
/// formulation; BitNet b1.58 uses RMSnorm rather than LayerNorm
/// throughout.
pub fn rmsnorm_into(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), weight.len());
    debug_assert_eq!(out.len(), x.len());
    let mut ss = 0.0f64;
    for &v in x {
        ss += (v as f64) * (v as f64);
    }
    let inv = (1.0 / (ss / (x.len() as f64) + eps as f64).sqrt()) as f32;
    for ((o, &xi), &wi) in out.iter_mut().zip(x.iter()).zip(weight.iter()) {
        *o = xi * inv * wi;
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
//  predict_bitnet — full BitNet 1.58 forward pass
// ─────────────────────────────────────────────────────────────────────────────
//
// Closes the wiring deferred from BUG-infer-deadlock §5.4: end-to-end
// inference against I2_S native ternary weights, no f16/f32 weight
// materialisation anywhere.  Self-contained — does not touch the
// dense `predict()` path or `ModelWeights`.
//
// Inputs:
//   - BitnetModel: per-layer BitLinear weights + RMSnorm scales +
//     embed table + lm_head, plus a few model dims (head_dim,
//     n_q_heads, n_kv_heads, rope_base).
//   - tokenizer: standard HF-style tokeniser used to decode top-K
//     output token ids back into strings.
//   - token_ids: prefill tokens (seq_len up to ~1k for one-shot infer).
//   - top_k: how many top predictions to emit.
//
// Output: top-K (token_string, probability) pairs for the next token
// after the last input token.
//
// The forward pass:
//   1. h = embed[token_ids] * embed_scale         [seq_len, hidden]
//   2. for each layer L:
//      a. x_norm = rmsnorm(h, attn_norm[L])
//      b. q[i,j] = matvec_i2s(W_q[L], x_norm[i])     for i in 0..seq_len
//         k[i,j] = matvec_i2s(W_k[L], x_norm[i])
//         v[i,j] = matvec_i2s(W_v[L], x_norm[i])
//      c. apply RoPE to q, k
//      d. per-head causal-masked scaled-dot-product attention
//         (no KV cache — one-shot prefill is the use case).
//      e. attn_out[i] = matvec_i2s(W_o[L], rmsnorm(attn_pool[i], attn_sub_norm[L]))
//      f. h += attn_out
//      g. x_norm = rmsnorm(h, ffn_norm[L])
//      h. ffn_out[i] = BitNetFfn.forward(x_norm[i]) — already includes residual
//      i. h becomes the FFN output (BitNetFfn applies residual internally,
//         but here we feed x_norm rather than h, so we add the residual
//         at the call site instead).
//   3. h_final = rmsnorm(h[-1], output_norm)
//   4. logits = lm_head @ h_final
//   5. Top-K softmax → predictions.
