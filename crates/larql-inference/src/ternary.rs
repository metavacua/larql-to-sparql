//! BitNet 1.58 native-ternary inference building blocks
//! (BUG-infer-deadlock §5.4).
//!
//! This module assembles the ternary matvec kernel from
//! `larql_compute::cpu::ops::ternary_matvec` into the higher-level
//! pieces a BitNet forward pass needs:
//!
//! - [`BitNetFfn`]: a complete FFN block — RMSnorm → gate / up
//!   BitLinear projections → squared-ReLU activation (BitNet b1.58
//!   uses ReLU², not SwiGLU) → element-wise multiply →
//!   post-FFN-norm → down BitLinear → residual addition.
//! - [`BitNetAttention`]: the attention-side companion — q/k/v/o
//!   BitLinear projections wired around the existing attention
//!   kernel (RoPE + softmax) from [`super::attention`].  Uses
//!   `attn_sub_norm.weight` between QK and the projection (BitNet
//!   b1.58 architecture has an extra norm there).
//!
//! Both structs hold their weights as [`BitLinearWeight`] (typed
//! ternary container with per-channel scale).  Forward methods
//! produce f32 activations.  No f16 / f32 weight tensors are ever
//! materialised — the entire arithmetic stays in i32 trit
//! accumulation + one f32 scale per output channel.
//!
//! Wiring this into the workspace's existing predict / infer_patched
//! path is a separate piece (it requires a parallel
//! `ModelWeights::Bitnet { ... }` variant + a dispatch hook in
//! `forward::layer`).  This module ships the math and the typed
//! interface; the loader / dispatch wiring is mechanical follow-up.
//!
//! ## Why this exists
//!
//! Production triage (`BUG-infer-deadlock.md` §3.3) showed that even
//! after the deadlock + OOM fixes land, BitNet b1.58 2 B 4 T still
//! consumes ~5 GB of RSS because the convert path dequantizes I2_S
//! weights to f16 at vindex build time.  The model's whole point is
//! that ternary weights need no per-element fp arithmetic: a 2-bpw
//! native path drops the runtime working set to ~1.4 GB.  This
//! module provides the math; replacing `--f16` in the convert path
//! is what closes out §5.4 end-to-end.
//!
//! ## Relationship to the engine machinery (why a separate stack)
//!
//! This module deliberately sits *beside* `attention/`, `kv_dispatch`,
//! `forward/`, and the StatePolicy engine rather than threading the
//! BitNet forward through them.  The reasoning, component by
//! component:
//!
//! - **RoPE** is reused directly (`super::attention::rope`) — the
//!   seam is clean and the math is identical, so there's no reason
//!   to fork it.
//! - **RMSNorm** is NOT reused from `larql_compute::residual`.  That
//!   crate's `rms_norm*` operate on `Array2<f32>` and allocate a new
//!   array per call; the BitNet forward runs a per-token,
//!   allocation-free `&[f32] -> &mut [f32]` norm in the hot inner
//!   loop (`rmsnorm_into`).  Routing it through the allocating
//!   `Array2` form would add an allocation per position per layer.
//!   The numerics are the same (verified: same eps, same
//!   `x/rms*weight`); the divergence is purely the alloc-free
//!   buffer shape.  If `residual` grows an `_into` variant this can
//!   collapse to a one-line call.
//! - **GQA attention** is computed inline here rather than via
//!   `attention::{gqa,decode}` because BitNet inserts an EXTRA norm
//!   (`attn_sub_norm`) between the QK product and the output
//!   projection — a sub-layer norm the standard attention path has
//!   no hook for.  The Q/K/V/O projections are themselves ternary
//!   (`BitLinearWeight`), not dense, so the projection step can't
//!   call the dense attention kernels regardless.
//! - **FFN** is squared-ReLU (ReLU²), not SwiGLU, and its
//!   projections are ternary with a mid-FFN sub-norm
//!   (`ffn_sub_norm`).  Neither shape matches the existing FFN
//!   forward.
//! - **KV cache / decode_step / generate / sampling** are the
//!   genuinely-forkable parts.  They are reimplemented here as a
//!   self-contained single-shot-prefill + greedy/temperature decode
//!   so the BitNet path is verifiable on its own (it was qualified
//!   end-to-end against the real 2 B model: "capital of France" ->
//!   Paris 94.5%).  Status (branch feat/quant-ternary-a8): the
//!   int8-quantized-activation (A8) kernel + NEON sign-select now
//!   EXIST in `larql_compute::cpu::ops::ternary_matvec`, and this
//!   forward already runs on them (`matvec_i2s_a8_f32_into`). What
//!   remains is the *shared* path — dispatch through the
//!   `QuantFormat`/`FormatRoute` registry (no ternary variant yet)
//!   and a shared KV-cache (`larql_kv::KvCache`) in place of the
//!   bespoke `BitnetKvCache`.
//!
//! Net: the legitimate BitNet-specific divergences are the two
//! sub-norms and the ReLU² FFN over ternary projections.  The
//! reimplemented KV/sampling is a maintenance cost acknowledged
//! here. Its blocking precondition — the quantized-activation kernel
//! — is now met, so folding the decode loop onto the forward-pass
//! spine + a `KvEngine` impl is live roadmap work (ROADMAP "BitNet
//! b1.58 integration hardening"), no longer blocked on a missing
//! kernel. Until a second consumer exists it may stay isolated under
//! the no-premature-extraction rule; the point is the decision is now
//! explicit, not deferred to a non-existent dependency.
//!
//! ## I2_S layout (dual representation — intentional)
//!
//! There are TWO I2_S byte layouts in play, and they are not the
//! same on purpose:
//!   - `larql_models::quant::ggml::tq::dequantize_i2_s` decodes the
//!     STRIDED microsoft layout (128-elem / 32-byte blocks) read
//!     from the source GGUF.
//!   - the runtime kernel + the keep-quant writer use a CONTIGUOUS
//!     per-row layout (4 trits/byte, sequential).  The writer
//!     re-packs from the strided source into this form so the hot
//!     `matvec_i2s_f32` loop never has to handle the strided
//!     addressing.
//!
//! Conflating the two scrambles every weight; see the decode-fix PR
//! and the format spec for the authoritative description.

use larql_compute::cpu::ops::ternary_matvec::{
    matvec_i2s_a8_f32_into, matvec_i2s_a8_into, quantize_activation_i8, BitLinearWeight,
};
use ndarray::{Array1, Array2, ArrayView2};

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

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

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny synthetic BitLinearWeight from a list of (row, col, trit)
    /// triples plus per-row scales.
    fn build_weight(
        rows: usize,
        cols: usize,
        trits: &[(usize, usize, i8)],
        scales: Vec<f32>,
    ) -> BitLinearWeight {
        assert!(cols.is_multiple_of(4));
        let mut bytes = vec![0u8; rows * cols / 4];
        for &(r, c, t) in trits {
            let bits: u8 = match t {
                1 => 0b01,
                -1 => 0b10,
                _ => 0b00,
            };
            let byte_idx = r * (cols / 4) + c / 4;
            let slot = (c % 4) as u8;
            bytes[byte_idx] |= bits << (2 * slot);
        }
        BitLinearWeight::new(rows, cols, bytes, scales).unwrap()
    }

    /// Parity gate for the A8 wiring: the production FFN forward (now the
    /// int8-activation A8 path) must track a full f32-activation reference
    /// FFN within int8 tolerance. Validates that swapping `matvec_i2s_f32`
    /// → `matvec_i2s_a8_f32` across the forward didn't shift the numerics
    /// beyond the intended activation-quantisation error.
    #[test]
    fn ffn_a8_forward_matches_f32_reference_within_tolerance() {
        use larql_compute::cpu::ops::ternary_matvec::matvec_i2s_f32_into;

        fn rand_w(rows: usize, cols: usize, seed: u64) -> BitLinearWeight {
            let mut s = seed;
            let mut bytes = vec![0u8; rows * cols / 4];
            for b in bytes.iter_mut() {
                let mut bv = 0u8;
                for slot in 0..4 {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let code = match (s >> 33) % 3 {
                        0 => 0b00u8,
                        1 => 0b01,
                        _ => 0b10,
                    };
                    bv |= code << (2 * slot);
                }
                *b = bv;
            }
            let scales = (0..rows).map(|r| 0.05 + (r % 5) as f32 * 0.01).collect();
            BitLinearWeight::new(rows, cols, bytes, scales).unwrap()
        }
        fn synth(n: usize, seed: u64) -> Vec<f32> {
            let mut s = seed;
            (0..n)
                .map(|_| {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
                })
                .collect()
        }
        fn cosine(a: &[f32], b: &[f32]) -> f32 {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            dot / (na * nb)
        }

        let (hidden, inter) = (256usize, 512usize);
        let ffn = BitNetFfn {
            gate: rand_w(inter, hidden, 1),
            up: rand_w(inter, hidden, 2),
            down: rand_w(hidden, inter, 3),
            ffn_norm: vec![1.0; hidden],
            ffn_sub_norm: vec![1.0; inter],
            eps: 1e-5,
        };
        let x = synth(hidden, 42);

        // Production (A8) forward.
        let mut gate = vec![0.0; inter];
        let mut up = vec![0.0; inter];
        let mut hid = vec![0.0; inter];
        let mut y_a8 = vec![0.0; hidden];
        ffn.forward_into(&x, &mut gate, &mut up, &mut hid, &mut y_a8);

        // f32-activation reference forward (same math, f32 kernel).
        let mut x_norm = vec![0.0; hidden];
        rmsnorm_into(&x, &ffn.ffn_norm, ffn.eps, &mut x_norm);
        let mut g = vec![0.0; inter];
        let mut u = vec![0.0; inter];
        matvec_i2s_f32_into(&ffn.gate, &x_norm, &mut g).unwrap();
        matvec_i2s_f32_into(&ffn.up, &x_norm, &mut u).unwrap();
        let hid_f: Vec<f32> = g
            .iter()
            .zip(&u)
            .map(|(gv, uv)| {
                let r = gv.max(0.0);
                r * r * uv
            })
            .collect();
        let mut hid_norm = vec![0.0; inter];
        rmsnorm_into(&hid_f, &ffn.ffn_sub_norm, ffn.eps, &mut hid_norm);
        let mut y_f32 = vec![0.0; hidden];
        matvec_i2s_f32_into(&ffn.down, &hid_norm, &mut y_f32).unwrap();

        let cos = cosine(&y_a8, &y_f32);
        assert!(
            cos > 0.999,
            "A8 FFN forward vs f32 reference cosine {cos} < 0.999"
        );
    }

    #[test]
    fn rmsnorm_zero_input_zero_output() {
        let x = vec![0.0f32; 8];
        let w = vec![1.0f32; 8];
        let mut out = vec![0.0f32; 8];
        rmsnorm_into(&x, &w, 1e-6, &mut out);
        // 0 / sqrt(0 + 1e-6) = 0; output is 0.
        assert!(out.iter().all(|&v| v.abs() < 1e-3));
    }

    #[test]
    fn rmsnorm_with_unit_weight_normalises() {
        // Input with rms = 2 → after norm rms should be ~1.
        let x = vec![2.0f32, 2.0, 2.0, 2.0]; // rms = 2.0
        let w = vec![1.0f32; 4];
        let mut out = vec![0.0f32; 4];
        rmsnorm_into(&x, &w, 0.0, &mut out);
        let post_rms = (out.iter().map(|v| v * v).sum::<f32>() / (out.len() as f32)).sqrt();
        assert!(
            (post_rms - 1.0).abs() < 1e-5,
            "post-norm rms should be ~1, got {post_rms}"
        );
    }

    #[test]
    fn rmsnorm_weight_scales_per_channel() {
        let x = vec![1.0f32; 4];
        let w = vec![2.0f32, 0.5, 1.0, -1.0];
        let mut out = vec![0.0f32; 4];
        rmsnorm_into(&x, &w, 0.0, &mut out);
        // rms(x) = 1, so normalised x = x.  Output = 1 * w.
        assert!((out[0] - 2.0).abs() < 1e-5);
        assert!((out[1] - 0.5).abs() < 1e-5);
        assert!((out[2] - 1.0).abs() < 1e-5);
        assert!((out[3] - (-1.0)).abs() < 1e-5);
    }

    /// Synthetic BitNet FFN with a single non-zero gate trit at
    /// position 0 of intermediate.  Verify the squared-ReLU
    /// activation: a positive activation squares, a negative
    /// activation zeros out.
    #[test]
    fn bitnet_ffn_squared_relu_zeros_negative_gates() {
        let hidden = 4;
        let inter = 4;
        // gate[0,0] = +1: gate output = +x_norm[0] * scale.
        // Other gate rows = 0.
        let gate = build_weight(inter, hidden, &[(0, 0, 1)], vec![1.0; inter]);
        // up[0,0] = +1: up output = x_norm[0].  Other up rows = 0.
        let up = build_weight(inter, hidden, &[(0, 0, 1)], vec![1.0; inter]);
        // down[0,0] = +1: y[0] = hid[0].  Other down rows = 0.
        let down = build_weight(hidden, inter, &[(0, 0, 1)], vec![1.0; hidden]);

        let ffn = BitNetFfn {
            gate,
            up,
            down,
            ffn_norm: vec![1.0; hidden],
            ffn_sub_norm: vec![1.0; inter],
            eps: 1e-5,
        };

        // Positive input: gate output > 0, ReLU keeps it, square it.
        // Activation flow:
        //   x = [4, 0, 0, 0] (rms = 2)
        //   x_norm = [2, 0, 0, 0]
        //   gate output[0] = 2; up output[0] = 2.
        //   hid[0] = relu(2)^2 * 2 = 4 * 2 = 8.
        //   ffn_sub_norm: rms(hid) = sqrt(8^2 / 4) = 4; hid_norm[0] = 8/4 = 2.
        //   y[0] = 2.
        //   x_out[0] = 4 + 2 = 6.
        let x_pos = vec![4.0f32, 0.0, 0.0, 0.0];
        let out_pos = ffn.forward(&x_pos);
        assert!(
            (out_pos[0] - 6.0).abs() < 1e-3,
            "positive input: expected x_out[0]=6, got {}",
            out_pos[0]
        );

        // Negative input: gate output < 0, ReLU zeros it, hid = 0,
        // y = 0, residual passes through.
        let x_neg = vec![-4.0f32, 0.0, 0.0, 0.0];
        let out_neg = ffn.forward(&x_neg);
        assert!(
            (out_neg[0] - (-4.0)).abs() < 1e-3,
            "negative input: ReLU should zero gate, residual passthrough; got {}",
            out_neg[0]
        );
    }

    /// `forward_into` and `forward` agree (the convenience method
    /// composes the in-place one + adds the residual).
    #[test]
    fn forward_and_forward_into_agree() {
        let hidden = 4;
        let inter = 8;
        let gate = build_weight(
            inter,
            hidden,
            &[(0, 0, 1), (1, 1, -1), (2, 2, 1), (3, 3, 1), (4, 0, 1)],
            vec![0.5; inter],
        );
        let up = build_weight(
            inter,
            hidden,
            &[(0, 0, 1), (1, 0, 1), (2, 1, 1), (3, 2, -1), (4, 3, 1)],
            vec![0.5; inter],
        );
        let down = build_weight(
            hidden,
            inter,
            &[(0, 0, 1), (1, 1, 1), (2, 2, 1), (3, 4, -1)],
            vec![0.7; hidden],
        );

        let ffn = BitNetFfn {
            gate,
            up,
            down,
            ffn_norm: vec![1.0, 1.5, 0.8, 1.2],
            ffn_sub_norm: vec![1.0; inter],
            eps: 1e-6,
        };
        let x = vec![0.7f32, -0.3, 0.5, -0.1];

        let out_a = ffn.forward(&x);

        let mut gate_buf = vec![0.0; inter];
        let mut up_buf = vec![0.0; inter];
        let mut hid_buf = vec![0.0; inter];
        let mut y_buf = vec![0.0; hidden];
        ffn.forward_into(&x, &mut gate_buf, &mut up_buf, &mut hid_buf, &mut y_buf);
        // forward() also adds the residual; forward_into() does not.
        for (b, xi) in y_buf.iter_mut().zip(x.iter()) {
            *b += xi;
        }

        for (a, b) in out_a.iter().zip(y_buf.iter()) {
            assert!((a - b).abs() < 1e-5, "forward {a} vs into+resid {b}");
        }
    }
}

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

/// Complete BitNet 1.58 model — every tensor needed for a forward
/// pass.  Built by `larql-vindex::extract::bitnet_loader` from a
/// `--keep-quant` vindex; feed into [`predict_bitnet`].
pub struct BitnetModel {
    /// Per-layer BitLinear projections + RMSnorm weights.
    pub layers: Vec<BitnetLayer>,
    /// Token embedding table, shape [vocab, hidden], f32.
    /// (Source GGUF has it in F16; we expand on load — the embed
    /// table is small relative to the BitLinear weights.)
    pub embed: Array2<f32>,
    /// Optional embed scale (most BitNet builds = 1.0).
    pub embed_scale: f32,
    /// Output RMSnorm weight, length = hidden_size.
    pub output_norm: Vec<f32>,
    /// LM head matrix, shape [vocab, hidden], f32.  Often tied to
    /// embed; supplied separately so the loader can decide.
    pub lm_head: Array2<f32>,
    /// RMSnorm epsilon used everywhere.
    pub eps: f32,
    /// Per-head dimension (= hidden / n_q_heads typically).
    pub head_dim: usize,
    /// Number of query heads.
    pub n_q_heads: usize,
    /// Number of key/value heads (GQA: usually < n_q_heads).
    pub n_kv_heads: usize,
    /// RoPE base (theta) — read from GGUF metadata.
    pub rope_base: f64,
}

/// One transformer block's worth of BitLinear weights + norms.
pub struct BitnetLayer {
    pub attn_norm: Vec<f32>,     // input RMSnorm, length = hidden
    pub attn_q: BitLinearWeight, // [hidden, hidden] (q heads x head_dim packed)
    pub attn_k: BitLinearWeight, // [n_kv_heads * head_dim, hidden]
    pub attn_v: BitLinearWeight, // [n_kv_heads * head_dim, hidden]
    pub attn_sub_norm: Vec<f32>, // post-attn RMSnorm, length = hidden
    pub attn_o: BitLinearWeight, // [hidden, hidden]
    pub ffn: BitNetFfn,          // self-contained FFN block
}

/// One top-K prediction.
#[derive(Debug, Clone, PartialEq)]
pub struct TernaryPrediction {
    pub token: String,
    pub probability: f64,
}

/// Run a full BitNet forward pass and return top-K next-token
/// predictions for the position immediately after `token_ids`.
///
/// Single-shot prefill, no KV cache.  Adequate for pg_infer's
/// `infer()` SQL surface (one-shot per call) and for the bug
/// report's repro path (a single `/v1/infer` from curl).
///
/// Memory profile at BitNet b1.58 2 B 4 T:
///   - weights resident:  ~1.1 GB (the I2_S bytes + scales + norms)
///   - per-call working:  ~10 MB (h, q/k/v, scratch buffers)
///
/// Compare to the dense f16 path's ~5 GB resident — that's the
/// architectural goal closed by this commit.
///
/// Native-only: takes `&larql_vindex::tokenizers::Tokenizer` (see the
/// `tokenizers` native-only note in `layer_graph/generate/detok.rs`).
#[cfg(not(target_arch = "wasm32"))]
pub fn predict_bitnet(
    model: &BitnetModel,
    tokenizer: &larql_vindex::tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
) -> Vec<TernaryPrediction> {
    if token_ids.is_empty() {
        return Vec::new();
    }
    let seq_len = token_ids.len();
    let hidden = model.embed.shape()[1];
    let head_dim = model.head_dim;
    let n_q_heads = model.n_q_heads;
    let n_kv_heads = model.n_kv_heads;
    debug_assert!(n_q_heads >= n_kv_heads, "GQA: n_q_heads >= n_kv_heads");
    debug_assert_eq!(
        n_q_heads * head_dim,
        hidden,
        "hidden = n_q_heads * head_dim"
    );

    // 1. Embed lookup -> residual stream h: [seq_len, hidden].
    let mut h = Array2::<f32>::zeros((seq_len, hidden));
    for (i, &tok) in token_ids.iter().enumerate() {
        let row = model.embed.row(tok as usize % model.embed.shape()[0]);
        let mut h_row = h.row_mut(i);
        for (dst, &src) in h_row.iter_mut().zip(row.iter()) {
            *dst = src * model.embed_scale;
        }
    }

    // Scratch buffers reused across layers.
    let mut x_norm = Array2::<f32>::zeros((seq_len, hidden));
    let mut q = Array2::<f32>::zeros((seq_len, n_q_heads * head_dim));
    let mut k = Array2::<f32>::zeros((seq_len, n_kv_heads * head_dim));
    let mut v = Array2::<f32>::zeros((seq_len, n_kv_heads * head_dim));
    let mut attn_pool = Array2::<f32>::zeros((seq_len, hidden));
    let mut attn_pool_norm = Array2::<f32>::zeros((seq_len, hidden));
    let mut attn_out = Array2::<f32>::zeros((seq_len, hidden));
    let mut ffn_x_norm = Array2::<f32>::zeros((seq_len, hidden));
    let mut ffn_gate = vec![0.0f32; model.layers[0].ffn.gate.rows];
    let mut ffn_up = vec![0.0f32; model.layers[0].ffn.up.rows];
    let mut ffn_hid = vec![0.0f32; model.layers[0].ffn.gate.rows];
    let mut ffn_out_row = vec![0.0f32; hidden];

    for layer in &model.layers {
        // a. attn input norm.
        for i in 0..seq_len {
            rmsnorm_into(
                h.row(i).as_slice().unwrap(),
                &layer.attn_norm,
                model.eps,
                x_norm.row_mut(i).as_slice_mut().unwrap(),
            );
        }

        // b. Q/K/V projections via ternary matvec, per token. Q/K/V share
        //    x_norm.row(i), so quantise it to int8 once (A8) and reuse.
        for i in 0..seq_len {
            let (x_i8, x_scale) = quantize_activation_i8(x_norm.row(i).as_slice().unwrap());
            matvec_i2s_a8_into(
                &layer.attn_q,
                &x_i8,
                x_scale,
                q.row_mut(i).as_slice_mut().unwrap(),
            )
            .expect("attn_q shape");
            matvec_i2s_a8_into(
                &layer.attn_k,
                &x_i8,
                x_scale,
                k.row_mut(i).as_slice_mut().unwrap(),
            )
            .expect("attn_k shape");
            matvec_i2s_a8_into(
                &layer.attn_v,
                &x_i8,
                x_scale,
                v.row_mut(i).as_slice_mut().unwrap(),
            )
            .expect("attn_v shape");
        }

        // c. RoPE on Q (per q-head) and K (per kv-head).
        let q_rotated =
            larql_compute::attention::rope::apply_rope(&q, n_q_heads, head_dim, model.rope_base);
        let k_rotated =
            larql_compute::attention::rope::apply_rope(&k, n_kv_heads, head_dim, model.rope_base);

        // d. Per-head causal-masked scaled-dot-product attention.
        attn_pool.fill(0.0);
        scaled_dot_product_attention_gqa(
            q_rotated.view(),
            k_rotated.view(),
            v.view(),
            n_q_heads,
            n_kv_heads,
            head_dim,
            attn_pool.view_mut(),
        );

        // e. Post-attn norm + output projection.
        for i in 0..seq_len {
            rmsnorm_into(
                attn_pool.row(i).as_slice().unwrap(),
                &layer.attn_sub_norm,
                model.eps,
                attn_pool_norm.row_mut(i).as_slice_mut().unwrap(),
            );
            matvec_i2s_a8_f32_into(
                &layer.attn_o,
                attn_pool_norm.row(i).as_slice().unwrap(),
                attn_out.row_mut(i).as_slice_mut().unwrap(),
            )
            .expect("attn_o shape");
        }

        // f. residual h += attn_out
        h += &attn_out;

        // g. FFN: per-token RMSnorm + BitNetFfn.forward_into +
        //    residual.  We call forward_into so the per-token gate /
        //    up / hid scratch stays out of the hot allocator.
        for i in 0..seq_len {
            rmsnorm_into(
                h.row(i).as_slice().unwrap(),
                &layer.ffn.ffn_norm,
                model.eps,
                ffn_x_norm.row_mut(i).as_slice_mut().unwrap(),
            );
            // BitNetFfn.forward_into expects the *un-normed* x in its
            // signature so it can run its own input norm.  Since we
            // already did that here (so the same x_norm could be
            // reused), we replicate the rest of forward_into manually
            // to skip the redundant RMSnorm step.
            ffn_forward_after_input_norm(
                &layer.ffn,
                ffn_x_norm.row(i).as_slice().unwrap(),
                model.eps,
                &mut ffn_gate,
                &mut ffn_up,
                &mut ffn_hid,
                &mut ffn_out_row,
            );
            for (dst, &src) in h.row_mut(i).iter_mut().zip(ffn_out_row.iter()) {
                *dst += src;
            }
        }
    }

    // Final norm + lm_head over the LAST token only.
    let last_h = h.row(seq_len - 1).to_owned();
    let mut h_final = vec![0.0f32; hidden];
    rmsnorm_into(
        last_h.as_slice().unwrap(),
        &model.output_norm,
        model.eps,
        &mut h_final,
    );
    let h_final_arr = Array1::from(h_final);
    let logits = model.lm_head.dot(&h_final_arr);

    // Top-K softmax.  Stable softmax: subtract max before exp.
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut probs: Vec<(usize, f64)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, ((v - max_logit) as f64).exp()))
        .collect();
    let sum: f64 = probs.iter().map(|(_, p)| p).sum();
    if sum > 0.0 {
        for (_, p) in probs.iter_mut() {
            *p /= sum;
        }
    }
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    probs
        .into_iter()
        .take(top_k)
        .filter_map(|(token_id, prob)| {
            tokenizer
                .id_to_token(token_id as u32)
                .map(|s| TernaryPrediction {
                    token: s,
                    probability: prob,
                })
        })
        .collect()
}

/// FFN forward pass that skips the input RMSnorm (caller already did
/// it).  Used by `predict_bitnet` to avoid double-norming when we
/// pre-compute the input norm once per layer.
fn ffn_forward_after_input_norm(
    ffn: &BitNetFfn,
    x_norm: &[f32],
    eps: f32,
    gate: &mut [f32],
    up: &mut [f32],
    hid: &mut [f32],
    y: &mut [f32],
) {
    let inter = ffn.gate.rows;
    debug_assert_eq!(gate.len(), inter);
    debug_assert_eq!(up.len(), inter);
    debug_assert_eq!(hid.len(), inter);

    // gate / up projections. Both share x_norm — quantise to int8 once (A8).
    let (x_i8, x_scale) = quantize_activation_i8(x_norm);
    matvec_i2s_a8_into(&ffn.gate, &x_i8, x_scale, gate).expect("gate shape");
    matvec_i2s_a8_into(&ffn.up, &x_i8, x_scale, up).expect("up shape");

    // Squared-ReLU activation.
    for ((g, u), h) in gate.iter().zip(up.iter()).zip(hid.iter_mut()) {
        let relu = g.max(0.0);
        *h = relu * relu * u;
    }

    // Post-gate-up norm.
    let mut hid_norm = vec![0.0f32; inter];
    rmsnorm_into(hid, &ffn.ffn_sub_norm, eps, &mut hid_norm);

    // Down projection.
    matvec_i2s_a8_f32_into(&ffn.down, &hid_norm, y).expect("down shape");
}

/// Causal-masked scaled-dot-product attention with GQA support.
///
/// `q` is `[seq_len, n_q_heads * head_dim]`, `k` and `v` are
/// `[seq_len, n_kv_heads * head_dim]`.  Each q-head maps to k/v
/// head `head_idx % n_kv_heads` (standard GQA); when `n_kv_heads
/// == n_q_heads` this is plain MHA.
///
/// Output is written to `out` (shape `[seq_len, hidden]` where
/// `hidden = n_q_heads * head_dim`).  Mask is causal: position `i`
/// only attends to positions `0..=i`.
fn scaled_dot_product_attention_gqa(
    q: ArrayView2<f32>,
    k: ArrayView2<f32>,
    v: ArrayView2<f32>,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    mut out: ndarray::ArrayViewMut2<f32>,
) {
    let seq_len = q.shape()[0];
    let scale = 1.0 / (head_dim as f32).sqrt();
    let groups = n_q_heads / n_kv_heads.max(1);

    out.fill(0.0);

    for h_q in 0..n_q_heads {
        let h_kv = h_q / groups.max(1);
        let q_off = h_q * head_dim;
        let kv_off = h_kv * head_dim;

        // For each query position, compute attention over all
        // earlier (and self) key positions.
        for i in 0..seq_len {
            // 1. scores[j] = (q[i, q_head] · k[j, kv_head]) * scale
            let mut scores = vec![f32::NEG_INFINITY; seq_len];
            for j in 0..=i {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[(i, q_off + d)] * k[(j, kv_off + d)];
                }
                scores[j] = dot * scale;
            }

            // 2. Stable softmax over the unmasked (j ≤ i) prefix.
            let max_score = scores[..=i]
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum = 0.0f32;
            for s in scores[..=i].iter_mut() {
                *s = (*s - max_score).exp();
                sum += *s;
            }
            if sum > 0.0 {
                for s in scores[..=i].iter_mut() {
                    *s /= sum;
                }
            }

            // 3. out[i, q_head] += sum_j weights[j] * v[j, kv_head]
            for j in 0..=i {
                let w = scores[j];
                if w == 0.0 {
                    continue;
                }
                for d in 0..head_dim {
                    out[(i, q_off + d)] += w * v[(j, kv_off + d)];
                }
            }
        }
    }
}

#[cfg(test)]
mod predict_tests {
    use super::*;

    /// End-to-end smoke test: build a tiny synthetic BitNet model
    /// (1 layer, hidden=4, vocab=8, head_dim=4, 1 head) and confirm
    /// `predict_bitnet` produces a top-K of the right shape with
    /// probabilities summing to ~1.
    #[test]
    fn predict_bitnet_runs_end_to_end_on_synthetic_model() {
        let hidden = 4;
        let inter = 4;
        let vocab = 8;
        let n_heads = 1;
        let head_dim = hidden / n_heads;

        // Tiny tokeniser stub: HF Tokenizer with byte-level vocab.
        // We don't actually need it to decode meaningfully — the
        // test asserts shape + numerical sanity, not which tokens
        // come out.
        let tok_json = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7},"merges":[]}}"#;
        let tokenizer =
            larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap();

        // Trivial weights: zero everywhere; predict_bitnet should
        // still produce a uniform-ish distribution and not crash.
        let mk_w = |rows: usize, cols: usize| {
            BitLinearWeight::new(rows, cols, vec![0u8; rows * cols / 4], vec![0.1f32; rows])
                .unwrap()
        };

        let layer = BitnetLayer {
            attn_norm: vec![1.0; hidden],
            attn_q: mk_w(hidden, hidden),
            attn_k: mk_w(hidden, hidden),
            attn_v: mk_w(hidden, hidden),
            attn_sub_norm: vec![1.0; hidden],
            attn_o: mk_w(hidden, hidden),
            ffn: BitNetFfn {
                gate: mk_w(inter, hidden),
                up: mk_w(inter, hidden),
                down: mk_w(hidden, inter),
                ffn_norm: vec![1.0; hidden],
                ffn_sub_norm: vec![1.0; inter],
                eps: 1e-5,
            },
        };

        let model = BitnetModel {
            layers: vec![layer],
            embed: Array2::from_shape_fn((vocab, hidden), |(i, j)| {
                ((i * 7 + j * 3) as f32 % 5.0) - 2.0
            }),
            embed_scale: 1.0,
            output_norm: vec![1.0; hidden],
            lm_head: Array2::from_shape_fn((vocab, hidden), |(i, j)| {
                ((i * 11 + j * 5) as f32 % 4.0) - 1.5
            }),
            eps: 1e-5,
            head_dim,
            n_q_heads: n_heads,
            n_kv_heads: n_heads,
            rope_base: 10000.0,
        };

        let token_ids = vec![0u32, 1, 2, 3];
        let preds = predict_bitnet(&model, &tokenizer, &token_ids, 4);
        assert_eq!(preds.len(), 4, "top_k=4 should return 4 predictions");

        // Probabilities must form a valid prefix of a softmax
        // distribution: each in [0, 1], sorted descending, summing
        // to <= 1 (we only return top-K).
        let mut prev = 1.0_f64;
        let mut sum = 0.0_f64;
        for p in &preds {
            assert!(p.probability >= 0.0 && p.probability <= 1.0);
            assert!(p.probability <= prev + 1e-9);
            prev = p.probability;
            sum += p.probability;
        }
        assert!(sum <= 1.0 + 1e-6, "top-K sum {sum} > 1");
    }

    /// Empty token_ids returns no predictions.
    #[test]
    fn predict_bitnet_empty_tokens_returns_empty() {
        let tok_json =
            r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
        let tokenizer =
            larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap();

        let mk_w = |rows: usize, cols: usize| {
            BitLinearWeight::new(rows, cols, vec![0u8; rows * cols / 4], vec![0.0; rows]).unwrap()
        };
        let layer = BitnetLayer {
            attn_norm: vec![1.0; 4],
            attn_q: mk_w(4, 4),
            attn_k: mk_w(4, 4),
            attn_v: mk_w(4, 4),
            attn_sub_norm: vec![1.0; 4],
            attn_o: mk_w(4, 4),
            ffn: BitNetFfn {
                gate: mk_w(4, 4),
                up: mk_w(4, 4),
                down: mk_w(4, 4),
                ffn_norm: vec![1.0; 4],
                ffn_sub_norm: vec![1.0; 4],
                eps: 1e-5,
            },
        };
        let model = BitnetModel {
            layers: vec![layer],
            embed: Array2::zeros((4, 4)),
            embed_scale: 1.0,
            output_norm: vec![1.0; 4],
            lm_head: Array2::zeros((4, 4)),
            eps: 1e-5,
            head_dim: 4,
            n_q_heads: 1,
            n_kv_heads: 1,
            rope_base: 10000.0,
        };
        let preds = predict_bitnet(&model, &tokenizer, &[], 5);
        assert!(preds.is_empty());
    }

    /// Causal mask self-test: position 0 can only attend to itself,
    /// so its attention output must equal v[0] (after the implicit
    /// softmax-of-one-element).
    #[test]
    fn scaled_dot_product_attention_position_zero_is_self_attended() {
        let n_heads = 1;
        let head_dim = 4;
        let q = Array2::from_shape_vec((1, head_dim), vec![1.0, 0.5, -0.5, 0.25]).unwrap();
        let k = q.clone();
        let v = Array2::from_shape_vec((1, head_dim), vec![3.0, -1.0, 2.5, 0.0]).unwrap();
        let mut out = Array2::<f32>::zeros((1, head_dim));
        scaled_dot_product_attention_gqa(
            q.view(),
            k.view(),
            v.view(),
            n_heads,
            n_heads,
            head_dim,
            out.view_mut(),
        );
        for (a, b) in out.row(0).iter().zip(v.row(0).iter()) {
            assert!((a - b).abs() < 1e-5, "expected v, got {a} vs {b}");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  KV-cached decode
// ─────────────────────────────────────────────────────────────────────────────
//
// Closes the "single-shot prefill, no KV cache" gap from the
// v0.3.0-bitnet release notes.  Three-piece API:
//
//   prefill(model, tokens) -> (BitnetKvCache, last_logits)
//      Run prefill over the full prompt, accumulating per-layer K/V
//      across all positions.  Returns the cache + the next-token
//      logits at the last prompt position.
//
//   decode_step(model, cache, new_token) -> next_logits
//      Append one token: project Q/K/V for it, append K/V to the
//      cache, run causal-masked attention against the entire cached
//      history (not against the new K/V alone — that would lose
//      context), apply the rest of the layer stack, return logits.
//
//   generate(model, tokenizer, prompt, max_new_tokens, sampler) -> String
//      Convenience wrapper that runs prefill then decodes step by
//      step until either (a) max_new_tokens reached or (b) the
//      sampler returns a stop signal.
//
// The cache is opaque to callers — they hold it across decode_step
// calls but never mutate it directly.  Per-layer storage is
// `Vec<Array2<f32>>` (one [past_len, n_kv_heads * head_dim] tensor
// per layer for K and V respectively); the cache grows by one row
// per decode_step.

/// Per-layer K/V projections accumulated across all positions seen
/// so far.  Held by the caller across decode steps.
#[derive(Clone)]
pub struct BitnetKvCache {
    /// `k[layer]` is `[past_len, n_kv_heads * head_dim]` f32.
    /// Always RoPE-applied at the position the row represents.
    pub k: Vec<Array2<f32>>,
    /// `v[layer]` is `[past_len, n_kv_heads * head_dim]` f32.
    pub v: Vec<Array2<f32>>,
    /// Number of positions accumulated so far.  Equal to
    /// `k[0].shape()[0]` for every layer; tracked separately so we
    /// can construct an empty cache without choosing a layer count.
    pub seq_len: usize,
}

impl BitnetKvCache {
    /// Empty cache sized for `n_layers`.  Each per-layer `k`/`v`
    /// starts with zero rows; rows are appended one-at-a-time as
    /// decode_step or prefill runs.
    pub fn new(n_layers: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        let kv_width = n_kv_heads * head_dim;
        Self {
            k: (0..n_layers)
                .map(|_| Array2::zeros((0, kv_width)))
                .collect(),
            v: (0..n_layers)
                .map(|_| Array2::zeros((0, kv_width)))
                .collect(),
            seq_len: 0,
        }
    }
}

/// Run the full prompt through every layer, accumulating K/V into a
/// fresh cache.  Returns the cache + raw logits at the last position
/// (caller decides sampling / softmax / top-K).
///
/// Equivalent to `predict_bitnet` minus the top-K extraction.
pub fn prefill(model: &BitnetModel, token_ids: &[u32]) -> (BitnetKvCache, Vec<f32>) {
    let n_layers = model.layers.len();
    let mut cache = BitnetKvCache::new(n_layers, model.n_kv_heads, model.head_dim);
    if token_ids.is_empty() {
        let vocab = model.lm_head.shape()[0];
        return (cache, vec![0.0; vocab]);
    }
    let logits = run_full_forward(model, token_ids, Some(&mut cache), None);
    (cache, logits)
}

/// Append one new token to an existing cache and return the logits
/// for that position.  Caller picks the sampling strategy.
///
/// Internally: position = cache.seq_len; the new token's Q sees
/// causal-masked attention against the full cached K/V plus its own
/// row.
pub fn decode_step(model: &BitnetModel, cache: &mut BitnetKvCache, new_token: u32) -> Vec<f32> {
    let position = cache.seq_len;
    let hidden = model.embed.shape()[1];
    let head_dim = model.head_dim;
    let n_q_heads = model.n_q_heads;
    let n_kv_heads = model.n_kv_heads;
    let kv_width = n_kv_heads * head_dim;
    let q_width = n_q_heads * head_dim;
    debug_assert_eq!(q_width, hidden, "hidden = n_q_heads * head_dim");

    // 1. Embed the new token.
    let mut h = Array1::<f32>::zeros(hidden);
    let row = model.embed.row(new_token as usize % model.embed.shape()[0]);
    for (dst, &src) in h.iter_mut().zip(row.iter()) {
        *dst = src * model.embed_scale;
    }

    let mut x_norm = vec![0.0f32; hidden];
    let mut q = vec![0.0f32; q_width];
    let mut k = vec![0.0f32; kv_width];
    let mut v = vec![0.0f32; kv_width];
    let mut attn_pool = vec![0.0f32; hidden];
    let mut attn_pool_norm = vec![0.0f32; hidden];
    let mut attn_out = vec![0.0f32; hidden];
    let mut ffn_x_norm = vec![0.0f32; hidden];
    let mut ffn_gate = vec![0.0f32; model.layers[0].ffn.gate.rows];
    let mut ffn_up = vec![0.0f32; model.layers[0].ffn.up.rows];
    let mut ffn_hid = vec![0.0f32; model.layers[0].ffn.gate.rows];
    let mut ffn_out_row = vec![0.0f32; hidden];

    for (layer_idx, layer) in model.layers.iter().enumerate() {
        // a. attn_norm.
        rmsnorm_into(
            h.as_slice().unwrap(),
            &layer.attn_norm,
            model.eps,
            &mut x_norm,
        );

        // b. Q/K/V projections. Q/K/V share x_norm — quantise once (A8).
        let (x_i8, x_scale) = quantize_activation_i8(&x_norm);
        matvec_i2s_a8_into(&layer.attn_q, &x_i8, x_scale, &mut q).expect("attn_q shape");
        matvec_i2s_a8_into(&layer.attn_k, &x_i8, x_scale, &mut k).expect("attn_k shape");
        matvec_i2s_a8_into(&layer.attn_v, &x_i8, x_scale, &mut v).expect("attn_v shape");

        // c. RoPE on the new token's Q + K only.  The cached K
        //    already carries RoPE for positions 0..position-1.
        let q_arr = Array2::from_shape_vec((1, q_width), q.clone()).expect("q shape");
        let k_arr = Array2::from_shape_vec((1, kv_width), k.clone()).expect("k shape");
        let q_rotated = larql_compute::attention::rope::apply_rope_partial_at(
            &q_arr,
            n_q_heads,
            head_dim,
            model.rope_base,
            1.0,
            position,
        );
        let k_rotated = larql_compute::attention::rope::apply_rope_partial_at(
            &k_arr,
            n_kv_heads,
            head_dim,
            model.rope_base,
            1.0,
            position,
        );

        // d. Append K/V rows to the per-layer cache.  ndarray has no
        //    cheap append, so we rebuild — cache growth is O(n) total
        //    across n_layers per decode_step which is fine for our
        //    workloads (max_new_tokens typically ≤ 256, hidden ≤ 4k).
        let new_k_row = k_rotated.row(0).to_owned();
        let new_v_row = Array1::from(v.clone());
        cache.k[layer_idx] = stack_one_row(&cache.k[layer_idx], &new_k_row);
        cache.v[layer_idx] = stack_one_row(&cache.v[layer_idx], &new_v_row);

        // e. Causal-masked GQA attention: new Q vs cached K/V (which
        //    now includes our just-appended row).
        let q_view = q_rotated.row(0);
        attention_decode_into(
            q_view.as_slice().unwrap(),
            cache.k[layer_idx].view(),
            cache.v[layer_idx].view(),
            n_q_heads,
            n_kv_heads,
            head_dim,
            &mut attn_pool,
        );

        // f. Sub-norm + O projection.
        rmsnorm_into(
            &attn_pool,
            &layer.attn_sub_norm,
            model.eps,
            &mut attn_pool_norm,
        );
        matvec_i2s_a8_f32_into(&layer.attn_o, &attn_pool_norm, &mut attn_out)
            .expect("attn_o shape");

        // g. Residual + FFN + residual.
        for (dst, &src) in h.iter_mut().zip(attn_out.iter()) {
            *dst += src;
        }
        rmsnorm_into(
            h.as_slice().unwrap(),
            &layer.ffn.ffn_norm,
            model.eps,
            &mut ffn_x_norm,
        );
        ffn_forward_after_input_norm(
            &layer.ffn,
            &ffn_x_norm,
            model.eps,
            &mut ffn_gate,
            &mut ffn_up,
            &mut ffn_hid,
            &mut ffn_out_row,
        );
        for (dst, &src) in h.iter_mut().zip(ffn_out_row.iter()) {
            *dst += src;
        }
    }

    cache.seq_len += 1;

    // h_final = output_norm(h)
    let mut h_final = vec![0.0f32; hidden];
    rmsnorm_into(
        h.as_slice().unwrap(),
        &model.output_norm,
        model.eps,
        &mut h_final,
    );
    let h_arr = Array1::from(h_final);
    model.lm_head.dot(&h_arr).to_vec()
}

/// Generate up to `max_new_tokens` greedily from `prompt`.  Stops
/// early if `stop_token` is produced.  Returns the raw token-id
/// stream (caller decodes for surface form).
///
/// Backwards-compat shim around [`generate_sampled`] with
/// [`SamplingConfig::greedy`] — byte-for-byte identical output to
/// callers built before sampling existed.
///
/// Native-only: takes `&larql_vindex::tokenizers::Tokenizer`.
#[cfg(not(target_arch = "wasm32"))]
pub fn generate(
    model: &BitnetModel,
    tokenizer: &larql_vindex::tokenizers::Tokenizer,
    prompt_token_ids: &[u32],
    max_new_tokens: usize,
    stop_token: Option<u32>,
) -> Vec<u32> {
    let _ = tokenizer; // unused on the greedy path; kept for API stability
    generate_sampled(
        model,
        prompt_token_ids,
        max_new_tokens,
        crate::layer_graph::generate::SamplingConfig::greedy(),
        stop_token,
    )
}

/// Generate up to `max_new_tokens` from `prompt` using a configurable
/// sampler (temperature / top-k / top-p / repetition penalties /
/// seedable RNG).  See [`SamplingConfig`] for the knobs.
///
/// `stop_token` halts generation before the token would be emitted
/// (mirrors [`generate`]).  EOS detection beyond a single token id
/// (stop strings, multiple EOS ids) belongs in
/// [`generate_streaming_bitnet`] which threads the full
/// [`EosConfig`].
///
/// Native-only: uses `generate::Sampler`, which is itself native-only
/// (holds a `StdRng` — no OS entropy source on wasm32v1-none).
#[cfg(not(target_arch = "wasm32"))]
pub fn generate_sampled(
    model: &BitnetModel,
    prompt_token_ids: &[u32],
    max_new_tokens: usize,
    sampling: crate::layer_graph::generate::SamplingConfig,
    stop_token: Option<u32>,
) -> Vec<u32> {
    if prompt_token_ids.is_empty() || max_new_tokens == 0 {
        return Vec::new();
    }
    let mut sampler = crate::layer_graph::generate::Sampler::new(sampling);
    let (mut cache, last_logits) = prefill(model, prompt_token_ids);
    let mut generated = Vec::with_capacity(max_new_tokens);

    let Some(mut next) = sampler.sample_with_history(&last_logits, &generated) else {
        return generated;
    };
    for _ in 0..max_new_tokens {
        if let Some(stop) = stop_token {
            if next == stop {
                break;
            }
        }
        generated.push(next);
        let logits = decode_step(model, &mut cache, next);
        match sampler.sample_with_history(&logits, &generated) {
            Some(t) => next = t,
            None => break,
        }
    }
    generated
}
/// Decode-time attention: one Q-row vs the full cached K/V history.
///
/// `q` is `[n_q_heads * head_dim]`, `k` and `v` are `[seq_len,
/// n_kv_heads * head_dim]`.  Result is written to `out` (length
/// `n_q_heads * head_dim`).
fn attention_decode_into(
    q: &[f32],
    k: ArrayView2<f32>,
    v: ArrayView2<f32>,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    let seq_len = k.shape()[0];
    debug_assert_eq!(v.shape()[0], seq_len);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let groups = n_q_heads / n_kv_heads.max(1);

    for o in out.iter_mut() {
        *o = 0.0;
    }

    for h_q in 0..n_q_heads {
        let h_kv = h_q / groups.max(1);
        let q_off = h_q * head_dim;
        let kv_off = h_kv * head_dim;

        // Scores over the full cached K (no causal mask — position
        // is at the end, attends to all of 0..seq_len-1 + itself,
        // which equals the full prefix).
        let mut scores = vec![0.0f32; seq_len];
        for (j, score) in scores.iter_mut().enumerate() {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[q_off + d] * k[(j, kv_off + d)];
            }
            *score = dot * scale;
        }

        // Stable softmax.
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max_score).exp();
            sum += *s;
        }
        if sum > 0.0 {
            for s in scores.iter_mut() {
                *s /= sum;
            }
        }

        // out[q_head] = Σ_j w[j] * v[j, kv_head]
        for (j, &w) in scores.iter().enumerate() {
            if w == 0.0 {
                continue;
            }
            for d in 0..head_dim {
                out[q_off + d] += w * v[(j, kv_off + d)];
            }
        }
    }
}

/// Append one row to a 2D ndarray.  ndarray has no built-in append;
/// we rebuild and copy.
fn stack_one_row(prev: &Array2<f32>, new_row: &Array1<f32>) -> Array2<f32> {
    let cols = prev.shape()[1];
    debug_assert_eq!(new_row.len(), cols);
    let new_rows = prev.shape()[0] + 1;
    let mut out = Array2::<f32>::zeros((new_rows, cols));
    if !prev.is_empty() {
        out.slice_mut(ndarray::s![..new_rows - 1, ..]).assign(prev);
    }
    out.row_mut(new_rows - 1).assign(new_row);
    out
}

#[cfg(test)]
fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best {
            best = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

/// Shared-spine forward used by both `prefill` (when `cache=Some`) and
/// the legacy single-shot `predict_bitnet`.  When `cache` is `Some`,
/// per-layer K/V are pushed in (after RoPE) so a subsequent
/// `decode_step` can extend the sequence.
///
/// Returns logits at the last position only.
fn run_full_forward(
    model: &BitnetModel,
    token_ids: &[u32],
    mut cache: Option<&mut BitnetKvCache>,
    mut residuals: Option<&mut Vec<(usize, Vec<f32>)>>,
) -> Vec<f32> {
    let seq_len = token_ids.len();
    let hidden = model.embed.shape()[1];
    let head_dim = model.head_dim;
    let n_q_heads = model.n_q_heads;
    let n_kv_heads = model.n_kv_heads;
    debug_assert_eq!(n_q_heads * head_dim, hidden);

    let mut h = Array2::<f32>::zeros((seq_len, hidden));
    for (i, &tok) in token_ids.iter().enumerate() {
        let row = model.embed.row(tok as usize % model.embed.shape()[0]);
        let mut h_row = h.row_mut(i);
        for (dst, &src) in h_row.iter_mut().zip(row.iter()) {
            *dst = src * model.embed_scale;
        }
    }

    let mut x_norm = Array2::<f32>::zeros((seq_len, hidden));
    let mut q = Array2::<f32>::zeros((seq_len, n_q_heads * head_dim));
    let mut k = Array2::<f32>::zeros((seq_len, n_kv_heads * head_dim));
    let mut v = Array2::<f32>::zeros((seq_len, n_kv_heads * head_dim));
    let mut attn_pool = Array2::<f32>::zeros((seq_len, hidden));
    let mut attn_pool_norm = Array2::<f32>::zeros((seq_len, hidden));
    let mut attn_out = Array2::<f32>::zeros((seq_len, hidden));
    let mut ffn_x_norm = Array2::<f32>::zeros((seq_len, hidden));
    let mut ffn_gate = vec![0.0f32; model.layers[0].ffn.gate.rows];
    let mut ffn_up = vec![0.0f32; model.layers[0].ffn.up.rows];
    let mut ffn_hid = vec![0.0f32; model.layers[0].ffn.gate.rows];
    let mut ffn_out_row = vec![0.0f32; hidden];

    for (layer_idx, layer) in model.layers.iter().enumerate() {
        for i in 0..seq_len {
            rmsnorm_into(
                h.row(i).as_slice().unwrap(),
                &layer.attn_norm,
                model.eps,
                x_norm.row_mut(i).as_slice_mut().unwrap(),
            );
        }
        for i in 0..seq_len {
            // Q/K/V share x_norm.row(i) — quantise to int8 once (A8) and reuse.
            let (x_i8, x_scale) = quantize_activation_i8(x_norm.row(i).as_slice().unwrap());
            matvec_i2s_a8_into(
                &layer.attn_q,
                &x_i8,
                x_scale,
                q.row_mut(i).as_slice_mut().unwrap(),
            )
            .expect("attn_q shape");
            matvec_i2s_a8_into(
                &layer.attn_k,
                &x_i8,
                x_scale,
                k.row_mut(i).as_slice_mut().unwrap(),
            )
            .expect("attn_k shape");
            matvec_i2s_a8_into(
                &layer.attn_v,
                &x_i8,
                x_scale,
                v.row_mut(i).as_slice_mut().unwrap(),
            )
            .expect("attn_v shape");
        }

        let q_rot =
            larql_compute::attention::rope::apply_rope(&q, n_q_heads, head_dim, model.rope_base);
        let k_rot =
            larql_compute::attention::rope::apply_rope(&k, n_kv_heads, head_dim, model.rope_base);

        attn_pool.fill(0.0);
        scaled_dot_product_attention_gqa(
            q_rot.view(),
            k_rot.view(),
            v.view(),
            n_q_heads,
            n_kv_heads,
            head_dim,
            attn_pool.view_mut(),
        );

        // If a cache is being built, capture the prefill K/V for
        // this layer (post-RoPE for K, pre-anything for V).
        if let Some(c) = cache.as_deref_mut() {
            c.k[layer_idx] = k_rot.clone();
            c.v[layer_idx] = v.clone();
        }

        for i in 0..seq_len {
            rmsnorm_into(
                attn_pool.row(i).as_slice().unwrap(),
                &layer.attn_sub_norm,
                model.eps,
                attn_pool_norm.row_mut(i).as_slice_mut().unwrap(),
            );
            matvec_i2s_a8_f32_into(
                &layer.attn_o,
                attn_pool_norm.row(i).as_slice().unwrap(),
                attn_out.row_mut(i).as_slice_mut().unwrap(),
            )
            .expect("attn_o shape");
        }
        h += &attn_out;

        for i in 0..seq_len {
            rmsnorm_into(
                h.row(i).as_slice().unwrap(),
                &layer.ffn.ffn_norm,
                model.eps,
                ffn_x_norm.row_mut(i).as_slice_mut().unwrap(),
            );
            ffn_forward_after_input_norm(
                &layer.ffn,
                ffn_x_norm.row(i).as_slice().unwrap(),
                model.eps,
                &mut ffn_gate,
                &mut ffn_up,
                &mut ffn_hid,
                &mut ffn_out_row,
            );
            for (dst, &src) in h.row_mut(i).iter_mut().zip(ffn_out_row.iter()) {
                *dst += src;
            }
        }

        // Capture the last-token residual at this layer for walk
        // inference's KNN-store override.  Mirrors what the dense
        // `WalkFfn::take_residuals` produces — same semantic position
        // (post-FFN-residual at the last prompt token).
        if let Some(r) = residuals.as_deref_mut() {
            r.push((layer_idx, h.row(seq_len - 1).to_vec()));
        }
    }

    if let Some(c) = cache {
        c.seq_len = seq_len;
    }

    let last_h = h.row(seq_len - 1).to_owned();
    let mut h_final = vec![0.0f32; hidden];
    rmsnorm_into(
        last_h.as_slice().unwrap(),
        &model.output_norm,
        model.eps,
        &mut h_final,
    );
    let h_arr = Array1::from(h_final);
    model.lm_head.dot(&h_arr).to_vec()
}

#[cfg(test)]
mod kv_cache_tests {
    use super::*;
    use larql_compute::cpu::ops::ternary_matvec::BitLinearWeight;

    /// Reusable tiny model factory: hidden=4, vocab=8, 1 head, 1 layer.
    fn tiny_model() -> BitnetModel {
        let hidden = 4;
        let inter = 4;
        let vocab = 8;
        let n_heads = 1;
        let head_dim = hidden / n_heads;
        let mk_w = |rows: usize, cols: usize, scale: f32| {
            // Cycle through ternary patterns so the matvec output
            // varies meaningfully across rows.
            let mut bytes = vec![0u8; rows * cols / 4];
            for (i, b) in bytes.iter_mut().enumerate() {
                *b = match i % 4 {
                    0 => 0b01_10_00_01,
                    1 => 0b10_01_01_00,
                    2 => 0b00_01_10_01,
                    _ => 0b01_00_01_10,
                };
            }
            BitLinearWeight::new(rows, cols, bytes, vec![scale; rows]).unwrap()
        };
        let layer = BitnetLayer {
            attn_norm: vec![1.0; hidden],
            attn_q: mk_w(hidden, hidden, 0.3),
            attn_k: mk_w(hidden, hidden, 0.4),
            attn_v: mk_w(hidden, hidden, 0.5),
            attn_sub_norm: vec![1.0; hidden],
            attn_o: mk_w(hidden, hidden, 0.6),
            ffn: BitNetFfn {
                gate: mk_w(inter, hidden, 0.2),
                up: mk_w(inter, hidden, 0.3),
                down: mk_w(hidden, inter, 0.7),
                ffn_norm: vec![1.0; hidden],
                ffn_sub_norm: vec![1.0; inter],
                eps: 1e-5,
            },
        };
        BitnetModel {
            layers: vec![layer],
            embed: Array2::from_shape_fn((vocab, hidden), |(i, j)| {
                ((i * 7 + j * 3) as f32 % 5.0) - 2.0
            }),
            embed_scale: 1.0,
            output_norm: vec![1.0; hidden],
            lm_head: Array2::from_shape_fn((vocab, hidden), |(i, j)| {
                ((i * 11 + j * 5) as f32 % 4.0) - 1.5
            }),
            eps: 1e-5,
            head_dim,
            n_q_heads: n_heads,
            n_kv_heads: n_heads,
            rope_base: 10000.0,
        }
    }

    /// `prefill` should produce exactly the same logits as
    /// `predict_bitnet` produces internally for the last position.
    /// (predict_bitnet returns top-K after softmax; prefill returns
    /// raw logits, so we re-derive the softmax here for comparison.)
    #[test]
    fn prefill_logits_match_predict_bitnet_top1() {
        let model = tiny_model();
        let tok_json = r#"{"version":"1.0","model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7},"merges":[]},"added_tokens":[]}"#;
        let tokenizer =
            larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap();

        let tokens = vec![0u32, 1, 2, 3];
        let preds = predict_bitnet(&model, &tokenizer, &tokens, 1);
        let (_cache, logits) = prefill(&model, &tokens);
        let argmax_logit = argmax(&logits);
        let predicted = tokenizer.id_to_token(argmax_logit).unwrap();
        assert_eq!(preds[0].token, predicted);
    }

    /// Prefill cache should hold one row per token in K and V.
    #[test]
    fn prefill_populates_cache_rows() {
        let model = tiny_model();
        let tokens = vec![0u32, 1, 2, 3, 4];
        let (cache, _logits) = prefill(&model, &tokens);
        assert_eq!(cache.seq_len, tokens.len());
        for (k_layer, v_layer) in cache.k.iter().zip(cache.v.iter()) {
            assert_eq!(k_layer.shape()[0], tokens.len());
            assert_eq!(v_layer.shape()[0], tokens.len());
        }
    }

    /// A decode_step appends one row to each layer's K and V cache.
    #[test]
    fn decode_step_grows_cache_by_one() {
        let model = tiny_model();
        let tokens = vec![0u32, 1, 2];
        let (mut cache, _) = prefill(&model, &tokens);
        let before = cache.seq_len;
        let logits = decode_step(&model, &mut cache, 5);
        assert_eq!(cache.seq_len, before + 1);
        assert_eq!(cache.k[0].shape()[0], before + 1);
        assert_eq!(cache.v[0].shape()[0], before + 1);
        assert_eq!(logits.len(), model.lm_head.shape()[0]);
    }

    /// Greedy generate on a tiny model returns the requested number
    /// of tokens (or stops at stop_token).
    #[test]
    fn generate_produces_max_new_tokens() {
        let model = tiny_model();
        let tok_json = r#"{"version":"1.0","model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7},"merges":[]},"added_tokens":[]}"#;
        let tokenizer =
            larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap();
        let prompt = vec![0u32, 1];
        let out = generate(&model, &tokenizer, &prompt, 4, None);
        assert_eq!(out.len(), 4);
        for &id in &out {
            assert!(id < 8, "vocab=8");
        }
    }

    /// Decode equivalence: prefilling N tokens then decoding one
    /// must produce the same logits as prefilling all N+1 tokens
    /// for the last position.  This is the load-bearing correctness
    /// test for the cache.
    #[test]
    fn decode_step_matches_full_prefill_at_last_position() {
        let model = tiny_model();
        let tokens = vec![0u32, 1, 2, 3];

        // Path 1: prefill all then read last_logits.
        let (_, logits_full) = prefill(&model, &tokens);

        // Path 2: prefill the prefix, decode the last token, read
        // the resulting logits.
        let (mut cache, _) = prefill(&model, &tokens[..tokens.len() - 1]);
        let logits_decoded = decode_step(&model, &mut cache, *tokens.last().unwrap());

        // Equivalence within fp noise.  Tolerance is generous because
        // the decode path uses apply_rope_partial_at while the
        // prefill path uses apply_rope, which are subtly different
        // kernels but produce identical output at integer positions.
        assert_eq!(logits_full.len(), logits_decoded.len());
        for (i, (a, b)) in logits_full.iter().zip(logits_decoded.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff < 1e-3,
                "logit {i}: prefill={a} decoded={b} diff={diff}"
            );
        }
    }

    /// argmax is stable and returns the right index.
    #[test]
    fn argmax_picks_max() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0, 0.0, -1.0]), 0);
        // Ties resolve to the first occurrence (consistent with
        // strict `>` test).
        assert_eq!(argmax(&[2.0, 2.0, 2.0]), 0);
    }

    /// Empty prompt for generate: returns no new tokens.
    #[test]
    fn generate_empty_prompt_returns_empty() {
        let model = tiny_model();
        let tok_json =
            r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
        let tokenizer =
            larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap();
        let out = generate(&model, &tokenizer, &[], 5, None);
        assert!(out.is_empty());
    }

    /// Stop token short-circuits generation.
    #[test]
    fn generate_stops_on_stop_token() {
        let model = tiny_model();
        let tok_json = r#"{"version":"1.0","model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7},"merges":[]},"added_tokens":[]}"#;
        let tokenizer =
            larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap();
        let prompt = vec![0u32, 1];
        // Set stop_token = the next-token argmax for this tiny
        // model.  We compute it via prefill.
        let (_, logits) = prefill(&model, &prompt);
        let first_pred = argmax(&logits);
        let out = generate(&model, &tokenizer, &prompt, 10, Some(first_pred));
        // Generate breaks before pushing the stop token.
        assert!(
            !out.contains(&first_pred),
            "stop_token leaked into output: {out:?}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  load_bitnet_model — construct a BitnetModel from a `--keep-quant` vindex
// ─────────────────────────────────────────────────────────────────────────────

/// Errors surfaced by [`load_bitnet_model`].  Distinct from the
/// underlying VindexError so callers can pattern-match on
/// "tensor missing" cleanly.
#[derive(Debug)]
pub enum BitnetLoadError {
    Vindex(larql_vindex::VindexError),
    Model(larql_models::ModelError),
    NotBitnet(String),
    MissingTensor(String),
    Shape(String),
}

impl core::fmt::Display for BitnetLoadError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BitnetLoadError::Vindex(e) => write!(f, "vindex: {e}"),
            BitnetLoadError::Model(e) => write!(f, "model: {e}"),
            BitnetLoadError::NotBitnet(msg) => write!(f, "not a bitnet vindex: {msg}"),
            BitnetLoadError::MissingTensor(name) => {
                write!(f, "missing required tensor: {name}")
            }
            BitnetLoadError::Shape(msg) => write!(f, "shape mismatch: {msg}"),
        }
    }
}

// `std::error::Error` has no core/alloc equivalent -- same split as
// `InferenceError` in `error.rs`.
#[cfg(not(target_arch = "wasm32"))]
impl std::error::Error for BitnetLoadError {}

impl From<larql_vindex::VindexError> for BitnetLoadError {
    fn from(e: larql_vindex::VindexError) -> Self {
        BitnetLoadError::Vindex(e)
    }
}

impl From<larql_models::ModelError> for BitnetLoadError {
    fn from(e: larql_models::ModelError) -> Self {
        BitnetLoadError::Model(e)
    }
}

/// Load a complete `BitnetModel` from a `--keep-quant` vindex
/// directory.  Reads:
///
///   index.json         — config + bitnet_layout
///   embeddings.bin     — embed table (F16 or F32; auto-detected)
///   norms.bin (etc.)   — per-layer attn_norm / sub_norm / ffn_norm /
///                         sub_norm + output_norm (via the standard
///                         dense loader, with `skip_attn` and
///                         `skip_ffn` so the BitLinear projections
///                         aren't dequantised into f32)
///   bitnet/<...>.i2s   — ternary BitLinear bytes
///   bitnet/scales.f32  — concatenated per-channel scales
///
/// LM head: BitNet b1.58 ties output to `token_embd.weight`
/// (verified on `microsoft/bitnet-b1.58-2B-4T-gguf`); we use the
/// same `embed` matrix as `lm_head` when the dense loader didn't
/// surface a separate `lm_head.weight` (the dense loader's tying
/// logic already handles this).
///
/// # Errors
/// `BitnetLoadError::NotBitnet` when index.json lacks
/// `bitnet_layout`.  Other variants surface tensor-shape /
/// I/O failures from the underlying loaders.
///
/// Native-only: takes `&std::path::Path` and calls
/// `load_vindex_config`/`LoadWeightsOptions`/`load_model_weights_with_opts`,
/// all filesystem-only (no core/alloc equivalent).
#[cfg(not(target_arch = "wasm32"))]
pub fn load_bitnet_model(vindex_path: &std::path::Path) -> Result<BitnetModel, BitnetLoadError> {
    let config = larql_vindex::load_vindex_config(vindex_path)?;
    let layout = config.bitnet_layout.clone().ok_or_else(|| {
        BitnetLoadError::NotBitnet(format!(
            "{} has no bitnet_layout in index.json (rebuild with `--keep-quant`)",
            vindex_path.display()
        ))
    })?;

    // 1. Dense parts: embed, lm_head, per-layer norms, output_norm.
    //
    //    skip_attn + skip_ffn keeps load_model_weights_with_opts from
    //    expanding the I2_S BitLinears into f32 (which would defeat
    //    the entire memory-savings story).  RMSnorm vectors don't
    //    match either pattern so they survive.
    let opts = larql_vindex::LoadWeightsOptions {
        skip_attn: true,
        skip_ffn: true,
        skip_lm_head: false,
        skip_embed: false,
    };
    let mut callbacks = larql_vindex::SilentLoadCallbacks;
    let dense = larql_vindex::load_model_weights_with_opts(vindex_path, &mut callbacks, opts)?;

    // 2. Ternary BitLinears.
    let bitnet = larql_vindex::extract::bitnet_loader::load_bitnet_weights(vindex_path)?;

    // 3. Embed scale (vocabulary-aware) — the dense loader already
    //    yields `dense.embed` with `embed_scale` applied if relevant;
    //    keep ours at 1.0 because BitNet b1.58 doesn't scale embeddings.
    let embed_scale = 1.0;
    let embed = dense.embed.to_owned();
    let lm_head = dense.lm_head.to_owned();
    let hidden = config.hidden_size;
    let n_layers = config.num_layers;

    // 4. Per-layer norms + BitLinear projections.
    //
    //    Keys are HF-normalised — the GGUF loader rewrites every
    //    tensor name through GGUF_TO_HF_KEY_REPLACEMENTS before it
    //    reaches the vindex, and the keep-quant writer stamps the
    //    same HF names into `bitnet_layout`.  So:
    //      blk.N.attn_norm.weight   -> layers.N.input_layernorm.weight
    //      blk.N.ffn_norm.weight    -> layers.N.post_attention_layernorm.weight
    //      blk.N.attn_sub_norm      -> layers.N.attn_sub_norm.weight (no HF
    //                                  equivalent; only blk.->layers. applies)
    //      blk.N.ffn_sub_norm       -> layers.N.ffn_sub_norm.weight
    //      blk.N.attn_q.weight      -> layers.N.self_attn.q_proj.weight
    //      blk.N.ffn_gate.weight    -> layers.N.mlp.gate_proj.weight  (etc.)
    //    Using GGUF-native keys here was BUG-infer-deadlock bug #5
    //    (loader/vindex namespace mismatch) — it made every lookup miss.
    let inter = config.intermediate_size;
    let mut layers = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let attn_norm = take_norm(
            &dense,
            &format!("layers.{i}.input_layernorm.weight"),
            hidden,
        )?;
        let attn_sub_norm = take_norm(&dense, &format!("layers.{i}.attn_sub_norm.weight"), hidden)?;
        let ffn_norm = take_norm(
            &dense,
            &format!("layers.{i}.post_attention_layernorm.weight"),
            hidden,
        )?;
        let ffn_sub_norm = take_norm(&dense, &format!("layers.{i}.ffn_sub_norm.weight"), inter)?;

        let get_bitlinear = |suffix: &str| -> Result<BitLinearWeight, BitnetLoadError> {
            let key = format!("layers.{i}.{suffix}.weight");
            bitnet
                .tensors
                .get(&key)
                .cloned()
                .ok_or(BitnetLoadError::MissingTensor(key))
        };
        let attn_q = get_bitlinear("self_attn.q_proj")?;
        let attn_k = get_bitlinear("self_attn.k_proj")?;
        let attn_v = get_bitlinear("self_attn.v_proj")?;
        let attn_o = get_bitlinear("self_attn.o_proj")?;
        let ffn_gate = get_bitlinear("mlp.gate_proj")?;
        let ffn_up = get_bitlinear("mlp.up_proj")?;
        let ffn_down = get_bitlinear("mlp.down_proj")?;

        let ffn = BitNetFfn {
            gate: ffn_gate,
            up: ffn_up,
            down: ffn_down,
            ffn_norm,
            ffn_sub_norm,
            eps: layout.rms_eps,
        };

        layers.push(BitnetLayer {
            attn_norm,
            attn_q,
            attn_k,
            attn_v,
            attn_sub_norm,
            attn_o,
            ffn,
        });
    }

    // 5. output_norm.  GGUF `output_norm.weight` -> HF `norm.weight`.
    let output_norm = dense
        .vectors
        .get("norm.weight")
        .cloned()
        .ok_or_else(|| BitnetLoadError::MissingTensor("norm.weight".into()))?;

    Ok(BitnetModel {
        layers,
        embed,
        embed_scale,
        output_norm,
        lm_head,
        eps: layout.rms_eps,
        head_dim: layout.head_dim.max(1),
        n_q_heads: layout.n_q_heads.max(1),
        n_kv_heads: layout.n_kv_heads.max(1),
        rope_base: if layout.rope_base > 0.0 {
            layout.rope_base
        } else {
            10000.0
        },
    })
}

/// Pluck a 1D norm vector out of `dense.vectors` and validate its
/// length.
fn take_norm(
    dense: &larql_models::ModelWeights,
    key: &str,
    expected_len: usize,
) -> Result<Vec<f32>, BitnetLoadError> {
    let v = dense
        .vectors
        .get(key)
        .cloned()
        .ok_or_else(|| BitnetLoadError::MissingTensor(key.to_string()))?;
    if v.len() != expected_len {
        return Err(BitnetLoadError::Shape(format!(
            "{key}: len {} != expected {}",
            v.len(),
            expected_len
        )));
    }
    Ok(v)
}

// ─────────────────────────────────────────────────────────────────────────────
//  generate_streaming_bitnet — token-by-token callback with detok + EOS
// ─────────────────────────────────────────────────────────────────────────────

/// Streaming generation for BitNet.  Mirrors the callback shape of
/// `larql_inference::layer_graph::generate_streaming` so HTTP SSE
/// route handlers can treat dense and ternary models uniformly.
///
/// `on_token(id, surface_text, decode_ms)` fires once per generated
/// token, in order.  `surface_text` is the cumulative-decode delta
/// (HF leading-space semantics preserved via [`Detokenizer`]) and may
/// be empty for tokens that don't grow the decoded string (e.g.
/// reserved/special tokens with skip_special=true).
///
/// `eos` and `sampling` are honoured: stop strings are matched against
/// the *cumulative* decoded text, EOS token ids halt immediately, and
/// the sampler can carry temperature/top-K/top-p/penalties.
///
/// Returns the count of tokens emitted (excluding the prompt).  Errors
/// from the prefill (empty prompt, etc.) yield `0` with no callback
/// invocations \u2014 the route handler gets to decide whether to surface
/// that as 200 OK with an empty stream or a 4xx.
///
/// Native-only: takes `&larql_vindex::tokenizers::Tokenizer` and uses
/// `generate::Sampler` + `generate::Detokenizer`, both native-only.
#[cfg(not(target_arch = "wasm32"))]
pub fn generate_streaming_bitnet<F>(
    model: &BitnetModel,
    tokenizer: &larql_vindex::tokenizers::Tokenizer,
    prompt_token_ids: &[u32],
    max_new_tokens: usize,
    sampling: crate::layer_graph::generate::SamplingConfig,
    eos: &crate::layer_graph::generate::EosConfig,
    mut on_token: F,
) -> usize
where
    F: FnMut(u32, &str, f64),
{
    if prompt_token_ids.is_empty() || max_new_tokens == 0 {
        return 0;
    }
    let mut sampler = crate::layer_graph::generate::Sampler::new(sampling);
    let mut detok = crate::layer_graph::generate::Detokenizer::new(tokenizer);
    detok.seed(prompt_token_ids);

    let (mut cache, last_logits) = prefill(model, prompt_token_ids);

    let Some(mut next) = sampler.sample_with_history(&last_logits, &[]) else {
        return 0;
    };
    let mut emitted = 0usize;
    let mut history: Vec<u32> = Vec::with_capacity(max_new_tokens);

    for _ in 0..max_new_tokens {
        let step_start = std::time::Instant::now();

        // Stop on a *next* token that matches an EOS id before we
        // even decode it (cheap path; symmetric with the dense
        // generate_streaming).
        if eos.eos_token_ids.contains(&next) {
            break;
        }

        // Push to detokeniser, get the cumulative-decode delta.
        let delta = detok.push(next);

        // EOS by surface form: use the tokenizer-aware variant which
        // re-decodes without skip-special when the cleaned delta is
        // empty (catches end-of-turn markers etc.).
        if eos.is_eos_with_tokenizer(next, &delta, tokenizer) {
            break;
        }

        let elapsed = step_start.elapsed().as_secs_f64() * 1000.0;
        on_token(next, &delta, elapsed);
        emitted += 1;
        history.push(next);

        // Decode the next-token logits *after* emitting the current
        // token so the loop terminates cleanly without an extra
        // forward at the end.
        let logits = decode_step(model, &mut cache, next);
        match sampler.sample_with_history(&logits, &history) {
            Some(t) => next = t,
            None => break,
        }
    }
    emitted
}

#[cfg(test)]
mod streaming_tests {
    use super::*;
    use crate::layer_graph::generate::{EosConfig, SamplingConfig};
    use larql_compute::cpu::ops::ternary_matvec::BitLinearWeight;

    /// Same fixture as kv_cache_tests but inlined here so the streaming
    /// suite is self-contained.
    fn tiny_model() -> BitnetModel {
        let hidden = 4;
        let inter = 4;
        let vocab = 8;
        let n_heads = 1;
        let head_dim = hidden / n_heads;
        let mk_w = |rows: usize, cols: usize, scale: f32| {
            let mut bytes = vec![0u8; rows * cols / 4];
            for (i, b) in bytes.iter_mut().enumerate() {
                *b = match i % 4 {
                    0 => 0b01_10_00_01,
                    1 => 0b10_01_01_00,
                    2 => 0b00_01_10_01,
                    _ => 0b01_00_01_10,
                };
            }
            BitLinearWeight::new(rows, cols, bytes, vec![scale; rows]).unwrap()
        };
        let layer = BitnetLayer {
            attn_norm: vec![1.0; hidden],
            attn_q: mk_w(hidden, hidden, 0.3),
            attn_k: mk_w(hidden, hidden, 0.4),
            attn_v: mk_w(hidden, hidden, 0.5),
            attn_sub_norm: vec![1.0; hidden],
            attn_o: mk_w(hidden, hidden, 0.6),
            ffn: BitNetFfn {
                gate: mk_w(inter, hidden, 0.2),
                up: mk_w(inter, hidden, 0.3),
                down: mk_w(hidden, inter, 0.7),
                ffn_norm: vec![1.0; hidden],
                ffn_sub_norm: vec![1.0; inter],
                eps: 1e-5,
            },
        };
        BitnetModel {
            layers: vec![layer],
            embed: Array2::from_shape_fn((vocab, hidden), |(i, j)| {
                ((i * 7 + j * 3) as f32 % 5.0) - 2.0
            }),
            embed_scale: 1.0,
            output_norm: vec![1.0; hidden],
            lm_head: Array2::from_shape_fn((vocab, hidden), |(i, j)| {
                ((i * 11 + j * 5) as f32 % 4.0) - 1.5
            }),
            eps: 1e-5,
            head_dim,
            n_q_heads: n_heads,
            n_kv_heads: n_heads,
            rope_base: 10000.0,
        }
    }

    fn tiny_tokenizer() -> larql_vindex::tokenizers::Tokenizer {
        let tok_json = r#"{"version":"1.0","model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7},"merges":[]},"added_tokens":[]}"#;
        larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap()
    }

    /// Greedy sampling matches the legacy generate() path token-for-token.
    /// This is the load-bearing backwards-compat test \u2014 if generate()
    /// drifts from generate_sampled(SamplingConfig::greedy()) we want
    /// to know.
    #[test]
    fn greedy_generate_sampled_matches_legacy_generate() {
        let model = tiny_model();
        let tok = tiny_tokenizer();
        let prompt = vec![0u32, 1, 2];
        let legacy = generate(&model, &tok, &prompt, 5, None);
        let sampled = generate_sampled(&model, &prompt, 5, SamplingConfig::greedy(), None);
        assert_eq!(legacy, sampled);
    }

    /// Seeded temperature sampling is reproducible: same seed +
    /// same prompt = same token stream.
    #[test]
    fn seeded_sampling_is_deterministic() {
        let model = tiny_model();
        let prompt = vec![0u32, 1];
        let cfg = SamplingConfig::temperature(1.0).with_seed(42);
        let a = generate_sampled(&model, &prompt, 5, cfg, None);
        let b = generate_sampled(&model, &prompt, 5, cfg, None);
        assert_eq!(a, b);
        assert_eq!(a.len(), 5);
    }

    /// Distinct seeds must produce different streams (with overwhelming
    /// probability for vocab=8 over 5 tokens).  Lock guards against a
    /// regression where seeding silently no-ops.
    #[test]
    fn distinct_seeds_diverge() {
        let model = tiny_model();
        let prompt = vec![0u32, 1];
        let a = generate_sampled(
            &model,
            &prompt,
            5,
            SamplingConfig::temperature(1.5).with_seed(1),
            None,
        );
        let b = generate_sampled(
            &model,
            &prompt,
            5,
            SamplingConfig::temperature(1.5).with_seed(99999),
            None,
        );
        // 8^5 = 32768 distinct streams: ~1-in-32768 odds of accidental match.
        assert_ne!(a, b, "seeds {{1, 99999}} produced identical streams");
    }

    /// Sampling filters are applied: top_k=1 with high temperature
    /// produces a single deterministic stream (only one candidate
    /// survives top_k=1 truncation, so multinomial degenerates).
    /// Note: top_k=1 routes through the sampling code path, not the
    /// `is_greedy()` short-circuit, so it can diverge from raw argmax
    /// when ties exist in the logits — we only assert that successive
    /// runs with the same seed match.
    #[test]
    fn top_k_one_is_deterministic() {
        let model = tiny_model();
        let prompt = vec![0u32, 1];
        let cfg = SamplingConfig::temperature(2.0).with_top_k(1).with_seed(7);
        let a = generate_sampled(&model, &prompt, 4, cfg, None);
        let b = generate_sampled(&model, &prompt, 4, cfg, None);
        assert_eq!(a, b);
        assert_eq!(a.len(), 4);
    }

    /// Streaming callback fires once per emitted token with the
    /// cumulative-decode delta.
    #[test]
    fn streaming_callback_fires_per_token() {
        let model = tiny_model();
        let tok = tiny_tokenizer();
        let prompt = vec![0u32, 1];
        let mut events: Vec<(u32, String)> = Vec::new();
        let n = generate_streaming_bitnet(
            &model,
            &tok,
            &prompt,
            4,
            SamplingConfig::greedy(),
            &EosConfig::empty(),
            |id, text, _ms| events.push((id, text.to_string())),
        );
        assert_eq!(n, events.len());
        assert_eq!(n, 4, "no early EOS");
        for (id, text) in &events {
            assert!(!text.is_empty(), "empty delta for token {id}");
        }
        let concat: String = events.iter().map(|(_, s)| s.as_str()).collect();
        assert!(
            !concat.is_empty(),
            "concatenated stream surface form was empty"
        );
    }

    /// EOS token id halts the stream before emitting that token.
    #[test]
    fn streaming_stops_on_eos_id() {
        let model = tiny_model();
        let tok = tiny_tokenizer();
        let prompt = vec![0u32, 1];
        let baseline = generate_sampled(&model, &prompt, 1, SamplingConfig::greedy(), None);
        let first = baseline[0];

        let mut emitted = 0;
        let _ = generate_streaming_bitnet(
            &model,
            &tok,
            &prompt,
            10,
            SamplingConfig::greedy(),
            &EosConfig::empty().with_eos_id(first),
            |_, _, _| emitted += 1,
        );
        assert_eq!(emitted, 0, "first sampled token = EOS id, no emits");
    }

    /// Empty prompt: zero tokens emitted, no callback invocations.
    #[test]
    fn streaming_empty_prompt_emits_nothing() {
        let model = tiny_model();
        let tok = tiny_tokenizer();
        let mut count = 0;
        let n = generate_streaming_bitnet(
            &model,
            &tok,
            &[],
            5,
            SamplingConfig::greedy(),
            &EosConfig::empty(),
            |_, _, _| count += 1,
        );
        assert_eq!(n, 0);
        assert_eq!(count, 0);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Walk-mode for BitNet: residual capture + KNN-store override
// ─────────────────────────────────────────────────────────────────────────────
//
// The dense walk-FFN path (`infer_patched`) replaces the FFN with a
// gate-index lookup that captures per-layer residuals at the
// last-token position, then queries an optional `KnnStore` for a
// retrieval-augmented top-1 swap (cosine > KNN_COSINE_THRESHOLD).
//
// On a BitNet 1.58 model the FFN is already ternary and very cheap,
// so the *compute* benefit of walk-FFN sparse evaluation is dubious.
// What walk-mode still buys us:
//
//   1. Per-layer residual trace, used by LQL `INFER` / `EXPLAIN INFER`
//      display to show the cosine path through gate features.
//   2. The KNN-store override for retrieval-augmented top-1 swap.
//
// Both are independent of the FFN compute strategy: they only need
// the residual stream at each layer's exit.  So our BitNet walk
// implementation runs the standard ternary forward (via
// `run_full_forward`) with residual capture enabled, then
// post-processes the same way the dense path does.

/// Run a BitNet forward and return both top-K predictions and
/// per-layer residuals at the last-token position.  Equivalent to
/// `predict_bitnet` plus the residual capture from `WalkFfn`.
///
/// The residual at layer `i` is `h[seq_len - 1, :]` after that
/// layer's residual additions (post FFN-residual), matching the
/// semantic position used by the dense path's
/// `WalkFfn::take_residuals`.
///
/// Native-only: takes `&larql_vindex::tokenizers::Tokenizer`.
#[cfg(not(target_arch = "wasm32"))]
pub fn predict_bitnet_with_residuals(
    model: &BitnetModel,
    tokenizer: &larql_vindex::tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
) -> (Vec<TernaryPrediction>, Vec<(usize, Vec<f32>)>) {
    if token_ids.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut residuals: Vec<(usize, Vec<f32>)> = Vec::with_capacity(model.layers.len());
    let logits = run_full_forward(model, token_ids, None, Some(&mut residuals));
    let preds = softmax_topk(&logits, tokenizer, top_k);
    (preds, residuals)
}

/// BitNet walk-mode entry point.  Mirrors the shape of
/// `larql_inference::infer_patched`: takes a token-id slice + an
/// optional `KnnStore` and returns the same `InferPatchedResult`
/// envelope (predictions, model_top1, knn_override, residuals,
/// walk_ms) so the route handler can dispatch uniformly between
/// dense and ternary paths.
///
/// The "walk" of the name is partial here — we don't replace the
/// ternary FFN with a gate-index lookup (it's already cheap
/// enough that the sparse path doesn't pay).  We run the standard
/// BitNet forward with residual capture enabled, then apply the
/// same KNN-store override and produce the same output shape.
/// Future work: a true sparse FFN walk on BitNet would require
/// per-feature ternary access in `BitLinearWeight` to amortise
/// the down-projection over selected features only.
///
/// Native-only: takes `&larql_vindex::tokenizers::Tokenizer`, returns
/// `crate::forward::InferPatchedResult`, and calls
/// `crate::forward::apply_knn_override` — all native-only.
#[cfg(not(target_arch = "wasm32"))]
pub fn infer_bitnet_walk(
    model: &BitnetModel,
    tokenizer: &larql_vindex::tokenizers::Tokenizer,
    knn_store: Option<&larql_vindex::KnnStore>,
    token_ids: &[u32],
    top_k: usize,
) -> crate::forward::InferPatchedResult {
    let start = std::time::Instant::now();
    let (preds, residuals) = predict_bitnet_with_residuals(model, tokenizer, token_ids, top_k);
    let walk_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Convert TernaryPrediction -> (String, f64) for shape parity
    // with the dense path's InferPatchedResult.
    let raw: Vec<(String, f64)> = preds
        .into_iter()
        .map(|p| (p.token, p.probability))
        .collect();
    let model_top1 = raw.first().cloned();
    let (predictions, knn_override) =
        crate::forward::apply_knn_override(raw, &residuals, knn_store, top_k);

    crate::forward::InferPatchedResult {
        predictions,
        model_top1,
        knn_override,
        residuals,
        walk_ms,
    }
}

/// Stable softmax over `logits` followed by top-K selection by
/// probability.  Pulled out of `predict_bitnet` so
/// `predict_bitnet_with_residuals` can share the post-processing
/// without duplicating the loop.
///
/// Native-only: takes `&larql_vindex::tokenizers::Tokenizer` directly
/// (despite the name suggesting pure math) to resolve token ids to
/// surface strings.
#[cfg(not(target_arch = "wasm32"))]
fn softmax_topk(
    logits: &[f32],
    tokenizer: &larql_vindex::tokenizers::Tokenizer,
    top_k: usize,
) -> Vec<TernaryPrediction> {
    if logits.is_empty() || top_k == 0 {
        return Vec::new();
    }
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut probs: Vec<(usize, f64)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, ((v - max_logit) as f64).exp()))
        .collect();
    let sum: f64 = probs.iter().map(|(_, p)| p).sum();
    if sum > 0.0 {
        for (_, p) in probs.iter_mut() {
            *p /= sum;
        }
    }
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    probs
        .into_iter()
        .take(top_k)
        .filter_map(|(token_id, prob)| {
            tokenizer
                .id_to_token(token_id as u32)
                .map(|s| TernaryPrediction {
                    token: s,
                    probability: prob,
                })
        })
        .collect()
}

#[cfg(test)]
mod walk_tests {
    use super::*;
    use larql_compute::cpu::ops::ternary_matvec::BitLinearWeight;

    fn tiny_model() -> BitnetModel {
        let hidden = 4;
        let inter = 4;
        let vocab = 8;
        let n_heads = 1;
        let head_dim = hidden / n_heads;
        let mk_w = |rows: usize, cols: usize, scale: f32| {
            let mut bytes = vec![0u8; rows * cols / 4];
            for (i, b) in bytes.iter_mut().enumerate() {
                *b = match i % 4 {
                    0 => 0b01_10_00_01,
                    1 => 0b10_01_01_00,
                    2 => 0b00_01_10_01,
                    _ => 0b01_00_01_10,
                };
            }
            BitLinearWeight::new(rows, cols, bytes, vec![scale; rows]).unwrap()
        };
        let mk_layer = || BitnetLayer {
            attn_norm: vec![1.0; hidden],
            attn_q: mk_w(hidden, hidden, 0.3),
            attn_k: mk_w(hidden, hidden, 0.4),
            attn_v: mk_w(hidden, hidden, 0.5),
            attn_sub_norm: vec![1.0; hidden],
            attn_o: mk_w(hidden, hidden, 0.6),
            ffn: BitNetFfn {
                gate: mk_w(inter, hidden, 0.2),
                up: mk_w(inter, hidden, 0.3),
                down: mk_w(hidden, inter, 0.7),
                ffn_norm: vec![1.0; hidden],
                ffn_sub_norm: vec![1.0; inter],
                eps: 1e-5,
            },
        };
        BitnetModel {
            layers: vec![mk_layer(), mk_layer()],
            embed: Array2::from_shape_fn((vocab, hidden), |(i, j)| {
                ((i * 7 + j * 3) as f32 % 5.0) - 2.0
            }),
            embed_scale: 1.0,
            output_norm: vec![1.0; hidden],
            lm_head: Array2::from_shape_fn((vocab, hidden), |(i, j)| {
                ((i * 11 + j * 5) as f32 % 4.0) - 1.5
            }),
            eps: 1e-5,
            head_dim,
            n_q_heads: n_heads,
            n_kv_heads: n_heads,
            rope_base: 10000.0,
        }
    }

    fn tiny_tokenizer() -> larql_vindex::tokenizers::Tokenizer {
        let tok_json = r#"{"version":"1.0","model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7},"merges":[]},"added_tokens":[]}"#;
        larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap()
    }

    /// One residual per layer, captured at the last-token position.
    /// Width must match `hidden`.
    #[test]
    fn predict_bitnet_with_residuals_emits_one_per_layer() {
        let model = tiny_model();
        let tok = tiny_tokenizer();
        let tokens = vec![0u32, 1, 2];
        let (preds, residuals) = predict_bitnet_with_residuals(&model, &tok, &tokens, 3);
        assert_eq!(preds.len(), 3);
        assert_eq!(residuals.len(), model.layers.len());
        for (i, (layer_idx, r)) in residuals.iter().enumerate() {
            assert_eq!(*layer_idx, i, "layer index sequence");
            assert_eq!(r.len(), model.embed.shape()[1], "hidden width");
        }
    }

    /// Top-K from `predict_bitnet_with_residuals` matches the
    /// legacy `predict_bitnet` (same top-K tokens in the same order).
    /// Guards against drift in the shared softmax_topk helper.
    #[test]
    fn predict_with_residuals_matches_legacy_top_k() {
        let model = tiny_model();
        let tok = tiny_tokenizer();
        let tokens = vec![0u32, 1, 2, 3];
        let legacy = predict_bitnet(&model, &tok, &tokens, 5);
        let (with_res, _) = predict_bitnet_with_residuals(&model, &tok, &tokens, 5);
        assert_eq!(legacy.len(), with_res.len());
        for (a, b) in legacy.iter().zip(with_res.iter()) {
            assert_eq!(a.token, b.token);
            assert!(
                (a.probability - b.probability).abs() < 1e-9,
                "{} vs {}",
                a.probability,
                b.probability,
            );
        }
    }

    /// `infer_bitnet_walk` with no KNN store: knn_override is None,
    /// predictions == raw bitnet predictions, model_top1 = predictions[0].
    #[test]
    fn walk_without_knn_store_passes_predictions_through() {
        let model = tiny_model();
        let tok = tiny_tokenizer();
        let tokens = vec![0u32, 1, 2];
        let result = infer_bitnet_walk(&model, &tok, None, &tokens, 4);
        assert!(result.knn_override.is_none());
        assert_eq!(result.predictions.len(), 4);
        let raw = predict_bitnet(&model, &tok, &tokens, 4);
        assert_eq!(result.predictions.len(), raw.len());
        assert_eq!(result.model_top1.as_ref().unwrap().0, raw[0].token);
    }

    /// Empty tokens: walk returns empty everything, doesn't panic.
    #[test]
    fn walk_empty_tokens_returns_empty() {
        let model = tiny_model();
        let tok = tiny_tokenizer();
        let result = infer_bitnet_walk(&model, &tok, None, &[], 5);
        assert!(result.predictions.is_empty());
        assert!(result.residuals.is_empty());
        assert!(result.model_top1.is_none());
    }
}
