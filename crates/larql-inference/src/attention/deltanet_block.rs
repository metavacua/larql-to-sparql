//! Gated DeltaNet block forward (Phase C.4a of
//! `inference-qwen35-deltanet`).
//!
//! Integrates the C.1 building blocks (Conv1D-with-state, scalar
//! helpers, state cache) and the C.2 recurrence into a single-token
//! forward step for one Gated DeltaNet linear-attention layer.
//!
//! What's NOT included: the residual add, the post-attention
//! RMSNorm (`attn_post_norm`), and the FFN. Those live in the
//! per-layer router (Phase C.4b) which adds the residual + FFN
//! around this block.
//!
//! Math literal from `openspec/changes/inference-qwen35-deltanet/
//! design.md` §4:
//!
//! ```text
//! x_norm    = RMSNorm(x, attn_norm)
//! qkv_mixed = attn_qkv  @ x_norm           # [conv_dim]
//! z         = attn_gate @ x_norm           # [value_dim]
//! beta      = sigmoid(ssm_beta  @ x_norm)  # [n_v_heads]
//! alpha     = ssm_alpha @ x_norm           # [n_v_heads]
//! log_g     = ssm_a * softplus(alpha + ssm_dt)  # [n_v_heads]
//!
//! qkv_conv  = SiLU(Conv1D(qkv_mixed, ssm_conv1d, state.conv_state))
//! q_raw, k_raw, v_raw = split(qkv_conv, [key_dim, key_dim, value_dim])
//! q = L2Norm(reshape(q_raw, [head_v_dim, n_k_heads]))
//! k = L2Norm(reshape(k_raw, [head_v_dim, n_k_heads]))
//! v = reshape(v_raw, [head_v_dim, n_v_heads])
//!
//! o = delta_net_step(q, k, v, log_g, beta, state.recurrent_state)
//! o = RMSNorm(reshape(o, [value_dim]), ssm_norm) * SiLU(z)
//! y = ssm_out @ o                          # [hidden]
//! ```

use ndarray::{ArcArray2, Array1, Array2, Axis};

use super::deltanet_recurrence::delta_net_step;
use super::deltanet_state::{
    causal_conv1d_step, l2_normalize_per_head, sigmoid, softplus, DeltaNetLayerState,
};

/// One DeltaNet layer's weight tensors.
///
/// Stored as `ArcArray2<f32>` so cloning the struct (e.g. when
/// building a `Qwen35Weights` view) only bumps Arc refcounts — no
/// data copies. Shape convention: every projection has
/// `[out_features, in_features]` (matvec is `y = W @ x`).
///
/// 1-D tensors (norms, scalar per-head) are `Arc<[f32]>` for the
/// same cheap-clone reason; callers index with `.as_ref()` to get
/// `&[f32]`.
#[derive(Clone)]
pub struct DeltaNetLayerWeights {
    /// Pre-mixer RMSNorm weight `[hidden]`.
    pub attn_norm: std::sync::Arc<[f32]>,
    /// Fused QKV projection `[conv_dim, hidden]`.
    pub attn_qkv: ArcArray2<f32>,
    /// Z-gate projection `[value_dim, hidden]`.
    pub attn_gate: ArcArray2<f32>,
    /// Causal depthwise Conv1D weight `[d_conv, conv_dim]`.
    pub ssm_conv1d: ArcArray2<f32>,
    /// Per-head bias added to `alpha` before softplus `[n_v_heads]`.
    pub ssm_dt: std::sync::Arc<[f32]>,
    /// Per-head log-decay scalar `[n_v_heads]`.
    pub ssm_a: std::sync::Arc<[f32]>,
    /// Beta projection `[n_v_heads, hidden]`.
    pub ssm_beta: ArcArray2<f32>,
    /// Alpha projection `[n_v_heads, hidden]`.
    pub ssm_alpha: ArcArray2<f32>,
    /// Post-mixer per-head RMSNorm weight `[head_v_dim]`.
    pub ssm_norm: std::sync::Arc<[f32]>,
    /// Output projection `[hidden, value_dim]`.
    pub ssm_out: ArcArray2<f32>,

    /// Optional lazy-quantised versions of the big projections.
    /// When `Some`, matvec dispatches through `QuantTensor::matvec`
    /// and the corresponding dense field above is a 0×0 placeholder.
    /// Set by `load_qwen35_weights_lazy_*`; default `None` keeps the
    /// dense path.
    pub attn_qkv_quant: Option<larql_models::quant::lazy::QuantTensor>,
    pub attn_gate_quant: Option<larql_models::quant::lazy::QuantTensor>,
    pub ssm_out_quant: Option<larql_models::quant::lazy::QuantTensor>,
}

/// Per-layer shape constants. All architectures in scope (Qwen 3.6
/// dense + MoE) share the same DeltaNet shape, so this is a
/// per-model constant rather than a per-layer one — but the struct
/// stays per-block in case future variants differ.
#[derive(Clone, Copy, Debug)]
pub struct DeltaNetDims {
    /// Residual-stream width `[hidden]`. Qwen 3.6 27B: 5120.
    pub hidden: usize,
    /// Per-head dim (`ssm_state_size`). Qwen 3.6: 128.
    pub head_v_dim: usize,
    /// Number of V heads (`ssm_dt_rank`). Qwen 3.6: 48 / 32.
    pub n_v_heads: usize,
    /// Number of K heads (`ssm_group_count`). Qwen 3.6: 16.
    pub n_k_heads: usize,
    /// Conv1D window. Qwen 3.6: 4.
    pub d_conv: usize,
    /// RMSNorm epsilon. Qwen 3.6: 1e-6.
    pub eps: f32,
    /// GQA broadcast pattern between V and K heads when `n_v > n_k`.
    /// `true` = BLOCK (V head `h` reads K head `h / repeat_factor`,
    /// matching qwen3next's interleave-repeat in `ggml_reshape_4d`
    /// + `ggml_repeat_4d` + reshape). `false` = CYCLE (V head `h`
    /// reads K head `h % h_k`, matching qwen35moe's direct
    /// `ggml_repeat_4d` tile). Both reduce to identity when
    /// `n_v == n_k`. See `delta_net_step` for the elementwise-
    /// verified rationale.
    pub block_gqa: bool,
}

impl DeltaNetDims {
    /// Construct from an architecture handle. Reads the SSM metadata
    /// fields the `qwen35` / `qwen35moe` archs expose:
    /// - `hidden` ← `arch.config().hidden_size`
    /// - `head_v_dim` ← `arch.ssm_state_size()`
    /// - `n_v_heads` ← `arch.ssm_dt_rank()`
    /// - `n_k_heads` ← `arch.ssm_group_count()`
    /// - `d_conv` ← `arch.ssm_conv_kernel()`
    /// - `eps` ← provided by the caller (typically `1e-6`)
    ///
    /// Required by `qwen35_generate_with_sampling` (task 2c.a of
    /// `vindex-qwen35moe-reader`). Pure utility — every read goes
    /// through the existing `ModelArchitecture` trait methods that
    /// PR #167 wired up via the vindex loader's arch reconstruction.
    pub fn from_arch(arch: &dyn larql_models::ModelArchitecture, eps: f32) -> Self {
        // qwen3next uses interleave-repeat (BLOCK); qwen35moe uses
        // tile-repeat (CYCLE). Both are GQA broadcast patterns between
        // V and K heads; selecting the wrong one randomises the
        // recurrence output (verified elementwise vs llama-eval).
        let block_gqa = matches!(arch.family(), "qwen3next");
        Self {
            hidden: arch.config().hidden_size,
            head_v_dim: arch.ssm_state_size(),
            n_v_heads: arch.ssm_dt_rank(),
            n_k_heads: arch.ssm_group_count(),
            d_conv: arch.ssm_conv_kernel(),
            eps,
            block_gqa,
        }
    }

    /// `head_v_dim * n_k_heads`.
    #[inline]
    pub fn key_dim(&self) -> usize {
        self.head_v_dim * self.n_k_heads
    }

    /// `head_v_dim * n_v_heads`.
    #[inline]
    pub fn value_dim(&self) -> usize {
        self.head_v_dim * self.n_v_heads
    }

    /// `2 * key_dim + value_dim` (the Conv1D channel count).
    #[inline]
    pub fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }
}

/// One-token forward through the DeltaNet block. Mutates `state` in
/// place; returns the block's output, ready to be added back to the
/// residual stream by the caller.
///
/// `layer` is purely diagnostic — used to key the per-sub-op binary
/// dumps when `LARQL_QWEN35_DUMP_LAYER_BOUNDARY` is set. Pass any
/// value (e.g. `0`) when not dumping.
pub fn deltanet_block_step(
    x: &Array1<f32>,
    weights: &DeltaNetLayerWeights,
    dims: &DeltaNetDims,
    state: &mut DeltaNetLayerState,
    backend: Option<&dyn larql_compute::ComputeBackend>,
    sequence_pos: usize,
    layer: usize,
) -> Array1<f32> {
    deltanet_block_step_with_optional_projections(
        x,
        weights,
        dims,
        state,
        backend,
        sequence_pos,
        layer,
        None,
        None,
        false,
    )
}

/// Same body as `deltanet_block_step` but with the option to
/// pass pre-computed projections (`qkv_mixed`, `z`). When both
/// are `Some`, the function skips its own RMSNorm + projection
/// step entirely — used by `deltanet_block_prefill` to batch
/// the projections into one matmul each across all prompt
/// positions instead of `seq_len` separate matvecs.
///
/// When either is `None`, projections are computed as in the
/// single-token path (the per-row decode call site).
#[allow(clippy::too_many_arguments)]
pub fn deltanet_block_step_with_optional_projections(
    x: &Array1<f32>,
    weights: &DeltaNetLayerWeights,
    dims: &DeltaNetDims,
    state: &mut DeltaNetLayerState,
    backend: Option<&dyn larql_compute::ComputeBackend>,
    sequence_pos: usize,
    layer: usize,
    precomputed_qkv: Option<&[f32]>,
    precomputed_z: Option<&[f32]>,
    // When `true`, skip the final `ssm_out` projection and return
    // `o_flat` (shape `[value_dim]`) instead of `block_out` (shape
    // `[hidden]`). Used by `deltanet_block_prefill`'s batched
    // branch to collect per-position `o_flat`s then run one batched
    // `ssm_out.matmul()` instead of `seq_len` separate matvecs.
    skip_ssm_out: bool,
) -> Array1<f32> {
    debug_assert_eq!(x.len(), dims.hidden);
    debug_assert_eq!(weights.attn_norm.len(), dims.hidden);
    if weights.attn_qkv_quant.is_none() {
        debug_assert_eq!(weights.attn_qkv.shape(), [dims.conv_dim(), dims.hidden]);
    }
    if weights.attn_gate_quant.is_none() {
        debug_assert_eq!(weights.attn_gate.shape(), [dims.value_dim(), dims.hidden]);
    }
    debug_assert_eq!(weights.ssm_conv1d.shape(), [dims.d_conv, dims.conv_dim()]);
    debug_assert_eq!(weights.ssm_dt.len(), dims.n_v_heads);
    debug_assert_eq!(weights.ssm_a.len(), dims.n_v_heads);
    debug_assert_eq!(weights.ssm_beta.shape(), [dims.n_v_heads, dims.hidden]);
    debug_assert_eq!(weights.ssm_alpha.shape(), [dims.n_v_heads, dims.hidden]);
    debug_assert_eq!(weights.ssm_norm.len(), dims.head_v_dim);
    if weights.ssm_out_quant.is_none() {
        debug_assert_eq!(weights.ssm_out.shape(), [dims.hidden, dims.value_dim()]);
    }

    use crate::attention::fine_profile::*;
    use crate::time_section;

    // Per-sub-op dump helper: writes `<dir>/<token_tag>/<name>_l<NN>.bin`
    // in the same `[i64 ne[4]; f32 data]` format as the layer-boundary
    // dump in `qwen35_forward_step`, so each binary lines up with a
    // llama-eval-callback `LLAMA_DUMP_BIN_DIR` tensor. The format is
    // documented in `common/debug.cpp:188+` of the llama.cpp clone at
    // `/home/ianblenke/3rd-party/llama.cpp/`. Cheap when unset (single
    // `std::env::var` cache hit, no allocation).
    let dump_subop = |name: &str, data: &[f32]| {
        if let Ok(dir) = std::env::var("LARQL_QWEN35_DUMP_LAYER_BOUNDARY") {
            let token_tag =
                std::env::var("LARQL_QWEN35_DUMP_TOKEN_TAG").unwrap_or_else(|_| "tok".to_string());
            let layer_dir = format!("{dir}/{token_tag}");
            let _ = std::fs::create_dir_all(&layer_dir);
            let path = format!("{layer_dir}/{name}_l{layer:02}.bin");
            if let Ok(mut f) = std::fs::File::create(&path) {
                use std::io::Write;
                let ne: [i64; 4] = [data.len() as i64, 1, 1, 1];
                for n in ne {
                    let _ = f.write_all(&n.to_le_bytes());
                }
                for v in data {
                    let _ = f.write_all(&v.to_le_bytes());
                }
            }
        }
    };

    // 1. Pre-mixer RMSNorm.
    let x_norm = time_section!(DN_RMS_NORM_X, rms_norm_1d(x, &weights.attn_norm, dims.eps));
    dump_subop("dn_x_norm", x_norm.as_slice().unwrap_or(&[]));
    // Diagnostic dump for layer 0 tensor comparison vs
    // llama-eval-callback. Verified C.5a: x_norm matches llama.cpp
    // bit-exactly at positions 0, 1, 2, N-3, N-2, N-1 for
    // chat-prompt's first token (248045 = `<|im_start|>`).
    if std::env::var("LARQL_QWEN35_DUMP_L0").is_ok() {
        let n = x_norm.len();
        eprintln!(
            "x_norm:    first-3 = [{:.4}, {:.4}, {:.4}]  last-3 = [{:.4}, {:.4}, {:.4}]  l2={:.3}",
            x_norm[0],
            x_norm[1],
            x_norm[2],
            x_norm[n - 3],
            x_norm[n - 2],
            x_norm[n - 1],
            x_norm.iter().map(|v| v * v).sum::<f32>().sqrt(),
        );
    }

    // 2. Projections (matvec).
    // Matvecs go through `matvec_with_backend` when the lazy form is
    // populated; otherwise fall back to the dense f32 dot.
    //
    // Fast-path: when both attn_qkv and attn_gate are lazy Q4_K and a
    // backend exposes `qwen35_paired_q4k_matvec`, run them on the
    // same device-resident `x_norm` upload with a single sync —
    // saves one htod + one sync per linear layer.
    use crate::attention::gpu_tier::{self, GpuClass};
    use crate::attention::quant_dispatch::{ggml_type_to_quant_format, matvec_with_backend};
    use larql_compute::QuantFormat;
    // Phase E.7: separate the projection-class backend (attn_qkv,
    // attn_gate, ssm_out) from the recurrence-class backend
    // (conv1d, L2, recurrence, rms_norm_heads). They can be
    // independently disabled via `LARQL_QWEN35_GPU_NO_DN_PROJ=1`
    // and `LARQL_QWEN35_GPU_NO_DN_RECURRENCE=1` to push back to CPU.
    let dn_proj_backend = gpu_tier::backend_for(GpuClass::DnProj, backend);
    let dn_recurrence_backend = gpu_tier::backend_for(GpuClass::DnRecurrence, backend);
    let paired = match (
        dn_proj_backend,
        weights.attn_qkv_quant.as_ref(),
        weights.attn_gate_quant.as_ref(),
    ) {
        (Some(b), Some(qkv_q), Some(gate_q)) => {
            let qkv_fmt = ggml_type_to_quant_format(qkv_q.tensor_type());
            let gate_fmt = ggml_type_to_quant_format(gate_q.tensor_type());
            if matches!(qkv_fmt, Some(QuantFormat::Q4_K))
                && matches!(gate_fmt, Some(QuantFormat::Q4_K))
            {
                let x_slice = x_norm.as_slice().expect("Array1 contiguous");
                let qkv_shape = qkv_q.shape();
                let gate_shape = gate_q.shape();
                b.qwen35_paired_q4k_matvec(
                    qkv_q.raw_bytes(),
                    qkv_shape[0],
                    gate_q.raw_bytes(),
                    gate_shape[0],
                    x_slice,
                    qkv_shape[1],
                )
            } else {
                None
            }
        }
        _ => None,
    };
    let (qkv_mixed, z) = match (precomputed_qkv, precomputed_z) {
        (Some(qkv_slice), Some(z_slice)) => {
            // Pre-computed by the batched-matmul path in
            // `deltanet_block_prefill`. Skip the per-row projection
            // (incl. the paired Q4_K GPU dispatch). Diagnostics that
            // referenced `qkv_mixed` / `z` still work; `x_norm`-based
            // diagnostics still see the per-row x_norm computed above.
            (
                Array1::from(qkv_slice.to_vec()),
                Array1::from(z_slice.to_vec()),
            )
        }
        _ => time_section!(
            DN_QKV_GATE_PAIR,
            if let Some((qkv_out, gate_out)) = paired {
                (Array1::from(qkv_out), Array1::from(gate_out))
            } else {
                let qkv_mixed = if let Some(q) = weights.attn_qkv_quant.as_ref() {
                    matvec_with_backend(q, &x_norm, dn_proj_backend)
                } else {
                    weights.attn_qkv.dot(&x_norm)
                };
                let z = if let Some(q) = weights.attn_gate_quant.as_ref() {
                    matvec_with_backend(q, &x_norm, dn_proj_backend)
                } else {
                    weights.attn_gate.dot(&x_norm)
                };
                (qkv_mixed, z)
            }
        ),
    };
    dump_subop("dn_qkv_mixed", qkv_mixed.as_slice().unwrap_or(&[]));
    dump_subop("dn_z_gate", z.as_slice().unwrap_or(&[]));
    // Diagnostic: qkv_mixed has ~1-4% per-element noise vs llama.cpp
    // at C.5a. Likely Q5_K dequant rounding (weights are Q5_K-quant).
    if std::env::var("LARQL_QWEN35_DUMP_L0").is_ok() {
        let n = qkv_mixed.len();
        eprintln!(
            "qkv_mixed: first-3 = [{:.4}, {:.4}, {:.4}]  last-3 = [{:.4}, {:.4}, {:.4}]  l2={:.3}",
            qkv_mixed[0],
            qkv_mixed[1],
            qkv_mixed[2],
            qkv_mixed[n - 3],
            qkv_mixed[n - 2],
            qkv_mixed[n - 1],
            qkv_mixed.iter().map(|v| v * v).sum::<f32>().sqrt(),
        );
        // Z gate (pre-silu): raw first/last-3 + per-head l2 + silu magnitudes.
        eprintln!(
            "z(gate) raw: first-3 = [{:.4}, {:.4}, {:.4}]  last-3 = [{:.4}, {:.4}, {:.4}]  l2={:.3}  mid={:.4} {:.4} {:.4}",
            z[0], z[1], z[2],
            z[z.len() - 3], z[z.len() - 2], z[z.len() - 1],
            z.iter().map(|v| v * v).sum::<f32>().sqrt(),
            z[1024], z[2048], z[3072],
        );
        // Also dump attn_gate weight stats to see if weight itself is the bug.
        let attn_gate = &weights.attn_gate;
        let aw = attn_gate.as_slice().unwrap_or(&[]);
        if !aw.is_empty() {
            let max_abs = aw.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let mean_abs = aw.iter().map(|v| v.abs()).sum::<f32>() / aw.len() as f32;
            let l2 = aw.iter().map(|v| v * v).sum::<f32>().sqrt();
            eprintln!(
                "attn_gate.weight: shape={:?}  max|w|={:.4}  mean|w|={:.4}  frobenius={:.3}  first-3=[{:.4},{:.4},{:.4}]",
                attn_gate.shape(), max_abs, mean_abs, l2,
                aw[0], aw[1], aw[2],
            );
        }

        let mut z_l2: Vec<(usize, f32, f32)> = (0..dims.n_v_heads)
            .map(|h| {
                let off = h * dims.head_v_dim;
                let l2 = (0..dims.head_v_dim)
                    .map(|d| {
                        let v = z[off + d];
                        v * v
                    })
                    .sum::<f32>()
                    .sqrt();
                let silu_l2 = (0..dims.head_v_dim)
                    .map(|d| {
                        let v = z[off + d];
                        let s = v * sigmoid(v);
                        s * s
                    })
                    .sum::<f32>()
                    .sqrt();
                (h, l2, silu_l2)
            })
            .collect();
        z_l2.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        eprintln!(
            "z(gate):   top-5 silu-heads: {} (z={:.3}, silu={:.3}) {} ({:.3}, {:.3}) {} ({:.3}, {:.3}) {} ({:.3}, {:.3}) {} ({:.3}, {:.3})  bot-3: {} ({:.3}, {:.3}) {} ({:.3}, {:.3}) {} ({:.3}, {:.3})",
            z_l2[0].0, z_l2[0].1, z_l2[0].2,
            z_l2[1].0, z_l2[1].1, z_l2[1].2,
            z_l2[2].0, z_l2[2].1, z_l2[2].2,
            z_l2[3].0, z_l2[3].1, z_l2[3].2,
            z_l2[4].0, z_l2[4].1, z_l2[4].2,
            z_l2[z_l2.len() - 1].0, z_l2[z_l2.len() - 1].1, z_l2[z_l2.len() - 1].2,
            z_l2[z_l2.len() - 2].0, z_l2[z_l2.len() - 2].1, z_l2[z_l2.len() - 2].2,
            z_l2[z_l2.len() - 3].0, z_l2[z_l2.len() - 3].1, z_l2[z_l2.len() - 3].2,
        );
    }
    let (beta_raw, alpha_raw) = time_section!(DN_BETA_ALPHA_DOTS, {
        let b = weights.ssm_beta.dot(&x_norm);
        let a = weights.ssm_alpha.dot(&x_norm);
        (b, a)
    });

    let (beta, log_g) = time_section!(DN_SIGMOID_SOFTPLUS, {
        let beta: Array1<f32> = beta_raw.iter().map(|&v| sigmoid(v)).collect();
        let log_g: Array1<f32> = (0..dims.n_v_heads)
            .map(|h| weights.ssm_a[h] * softplus(alpha_raw[h] + weights.ssm_dt[h]))
            .collect();
        (beta, log_g)
    });

    // 3. Causal Conv1D-with-state, then SiLU element-wise.
    let mut qkv_conv = time_section!(
        DN_CONV1D,
        dn_recurrence_backend
            .and_then(|b| {
                let weight = weights.ssm_conv1d.as_slice()?;
                let conv_state = state.conv_state.as_slice_mut()?;
                let new = qkv_mixed.as_slice()?;
                b.qwen35_causal_conv1d_step(
                    weight,
                    conv_state,
                    new,
                    dims.d_conv,
                    dims.conv_dim(),
                    sequence_pos,
                )
            })
            .map(Array1::from)
            .unwrap_or_else(|| {
                causal_conv1d_step(weights.ssm_conv1d.view(), &mut state.conv_state, &qkv_mixed)
            })
    );
    time_section!(DN_SILU_QKV_CONV, {
        for v in qkv_conv.iter_mut() {
            *v = silu(*v);
        }
    });
    dump_subop("dn_qkv_conv_silu", qkv_conv.as_slice().unwrap_or(&[]));
    if std::env::var("LARQL_QWEN35_DUMP_L0").is_ok() {
        let n = qkv_conv.len();
        eprintln!(
            "qkv_conv:  first-3 = [{:.4}, {:.4}, {:.4}]  last-3 = [{:.4}, {:.4}, {:.4}]  l2={:.3}",
            qkv_conv[0],
            qkv_conv[1],
            qkv_conv[2],
            qkv_conv[n - 3],
            qkv_conv[n - 2],
            qkv_conv[n - 1],
            qkv_conv.iter().map(|v| v * v).sum::<f32>().sqrt(),
        );
    }

    let key_dim = dims.key_dim();
    let value_dim = dims.value_dim();

    // Phase E.6.A fast-path: if a CUDA backend exposes the fused
    // post-projection step, do conv1d → silu → split → reshape →
    // L2 Q/K → recurrence → reshape → rms_norm_heads → silu(z)*o
    // entirely on device with one sync at the end. The matvecs
    // (attn_qkv/attn_gate/ssm_out) still go through the existing
    // host-bouncing path — E.6.B absorbs them.
    //
    // Returns the same `o_flat` value the per-step path computes
    // just before `ssm_out`; we resume the unfused tail (output
    // projection) from there.
    // Phase E.6.A — fused device-resident DeltaNet post-projection
    // chain. Opt-in via `LARQL_QWEN35_E6A_FUSED=1`. Off by default
    // because the recurrence kernel matches CPU on a single call but
    // accumulated state drift across positions causes the multi-token
    // parity check to deviate from llama.cpp (single-call output
    // matches CPU to 1e-7; across 9 prompt + 5 decode tokens the
    // residual stream diverges enough to flip argmax — root cause
    // still under investigation, likely fp32 reduction-order
    // sensitivity compounding through the per-layer recurrent state).
    // The fused implementation is otherwise complete and ready for
    // E.6.B (absorbing the projection matvecs into the same device
    // chain), which is the actual throughput lever — the post-proj
    // hooks aren't the bottleneck on their own.
    let post_proj = if std::env::var("LARQL_QWEN35_E6A_FUSED").is_ok() {
        dn_recurrence_backend.and_then(|b| {
            let qkv_mixed_slice = qkv_mixed.as_slice()?;
            let log_g_slice = log_g.as_slice()?;
            let beta_slice = beta.as_slice()?;
            let z_slice = z.as_slice()?;
            let conv_w = weights.ssm_conv1d.as_slice()?;
            let conv_state_slice = state.conv_state.as_slice_mut()?;
            let rec_state_slice = state.recurrent_state.as_slice_mut()?;
            b.qwen35_deltanet_postproj_step(
                qkv_mixed_slice,
                conv_w,
                log_g_slice,
                beta_slice,
                z_slice,
                &weights.ssm_norm,
                conv_state_slice,
                rec_state_slice,
                dims.head_v_dim,
                dims.n_v_heads,
                dims.n_k_heads,
                dims.d_conv,
                crate::residual::DEFAULT_EPS as f32,
                sequence_pos,
                dims.block_gqa,
            )
        })
    } else {
        None
    };
    if let Some(o_flat_vec) = post_proj {
        let o_flat = Array1::from(o_flat_vec);
        // Honour the batched-prefill exit point: when the outer
        // caller is collecting `o_flat`s for a batched `ssm_out`
        // matmul, return the unprojected `o_flat` (shape [value_dim])
        // instead of running ssm_out here.
        if skip_ssm_out {
            return o_flat;
        }
        let block_out = if let Some(q) = weights.ssm_out_quant.as_ref() {
            matvec_with_backend(q, &o_flat, dn_proj_backend)
        } else {
            weights.ssm_out.dot(&o_flat)
        };
        return block_out;
    }

    // 4. Split QKV into Q, K, V slabs.
    let q_raw = qkv_conv.slice(ndarray::s![..key_dim]).to_owned();
    let k_raw = qkv_conv.slice(ndarray::s![key_dim..2 * key_dim]).to_owned();
    let v_raw = qkv_conv.slice(ndarray::s![2 * key_dim..]).to_owned();

    // Reshape Q/K to [head_v_dim, n_k_heads], V to [head_v_dim, n_v_heads].
    //
    // CONVENTION: the projection output is laid out head-major in flat
    // form — `q_raw[h * head_v_dim + d]` is head h, dim d (this matches
    // the PyTorch `.view(n_heads, head_dim)` interpretation used by
    // HF transformers). To get the `[head_v_dim, n_k_heads]` shape that
    // `l2_normalize_per_head` and `delta_net_step` expect (with d as
    // the outer axis), we reshape FIRST to `(n_k_heads, head_v_dim)`
    // — which is the natural row-major view of a head-major flat —
    // THEN transpose to `(head_v_dim, n_k_heads)`.
    //
    // The earlier `.into_shape_with_order((head_v_dim, n_k_heads))`
    // direct reshape was a bug: it interpreted head-major flat as
    // dim-major, scrambling head indices. This was latent until real
    // Qwen 3.6 weights were loaded (synthetic constant tensors mask it).
    // `reversed_axes().to_owned()` preserves transposed strides (same
    // gotcha as the loader's `t().to_owned()` fixed by E.6.A's
    // ssm_conv1d patch). `as_standard_layout().to_owned()` forces a
    // row-major copy so `as_slice()` returns `Some`, which lets the
    // L2-norm and deltanet-recurrence GPU hooks actually fire when
    // `LARQL_QWEN35_GPU=1`. Without this they silently fall back to
    // CPU regardless of backend availability. Layout convention
    // unchanged: `q[d, h]` is still head h, dim d.
    let (q, k, v) = time_section!(DN_SPLIT_RESHAPE, {
        let q = q_raw
            .into_shape_with_order((dims.n_k_heads, dims.head_v_dim))
            .expect("q reshape")
            .reversed_axes()
            .as_standard_layout()
            .to_owned();
        let k = k_raw
            .into_shape_with_order((dims.n_k_heads, dims.head_v_dim))
            .expect("k reshape")
            .reversed_axes()
            .as_standard_layout()
            .to_owned();
        let v = v_raw
            .into_shape_with_order((dims.n_v_heads, dims.head_v_dim))
            .expect("v reshape")
            .reversed_axes()
            .as_standard_layout()
            .to_owned();
        (q, k, v)
    });

    // 5+6+7. Fused L2 norm Q/K + delta-rule recurrence + rms_norm_heads
    // on the GPU when the backend exposes it. Saves ~3 stream syncs
    // per linear DeltaNet layer (vs the four-call per-step path
    // below). silu(qkv_conv) above and silu(z)*o below stay on host,
    // which keeps the recurrence input bit-exact to llama.cpp's CPU
    // libm-expf (the E.6.A.6 multi-position drift culprit).
    //
    // Returns `o_normed_flat[value_dim]` in head-major layout,
    // bypassing the per-step `o.t().to_owned()` + `rms_norm_heads`
    // pair below. If the backend hook is None / unavailable, falls
    // through to the legacy per-step path.
    let fused_o_normed = if std::env::var("LARQL_QWEN35_DN_FUSED_DISABLE").is_err() {
        time_section!(
            DN_RECURRENCE,
            dn_recurrence_backend.and_then(|b| {
                let q_slice = q.as_slice()?;
                let k_slice = k.as_slice()?;
                let v_slice = v.as_slice()?;
                let log_g_slice = log_g.as_slice()?;
                let beta_slice = beta.as_slice()?;
                let rec_state_slice = state.recurrent_state.as_slice_mut()?;
                b.qwen35_deltanet_recurrence_block(
                    q_slice,
                    k_slice,
                    v_slice,
                    log_g_slice,
                    beta_slice,
                    &weights.ssm_norm,
                    rec_state_slice,
                    dims.head_v_dim,
                    dims.n_v_heads,
                    dims.n_k_heads,
                    crate::residual::DEFAULT_EPS as f32,
                    sequence_pos,
                    dims.block_gqa,
                )
            })
        )
    } else {
        None
    };

    if let Some(o_normed) = fused_o_normed {
        let mut o_flat = Array1::from(o_normed);
        // silu(z) * o_normed on host (stays CPU to match the per-step
        // path's parity behaviour). The fused chain returns the
        // already-rms_norm-ed output; we just need the gate
        // multiply + the ssm_out matvec.
        time_section!(DN_SILU_Z, {
            for c in 0..value_dim {
                o_flat[c] *= silu(z[c]);
            }
        });
        // Honour the batched-prefill exit point: when the outer
        // caller is collecting `o_flat`s for a batched `ssm_out`
        // matmul (set by `deltanet_block_prefill`'s batched branch),
        // return the unprojected `o_flat` (shape [value_dim]) instead
        // of running ssm_out per-position.
        if skip_ssm_out {
            return o_flat;
        }
        let block_out = time_section!(
            DN_SSM_OUT,
            if let Some(q) = weights.ssm_out_quant.as_ref() {
                matvec_with_backend(q, &o_flat, dn_proj_backend)
            } else {
                weights.ssm_out.dot(&o_flat)
            }
        );
        return block_out;
    }

    // Legacy per-step path: backend hook unavailable or disabled.
    let (q, k) = time_section!(DN_L2_NORM, {
        let q = dn_recurrence_backend
            .and_then(|b| {
                let q_slice = q.as_slice()?;
                b.qwen35_l2_normalize_per_head(q_slice, dims.head_v_dim, dims.n_k_heads, 1e-6)
            })
            .and_then(|out| Array2::from_shape_vec((dims.head_v_dim, dims.n_k_heads), out).ok())
            .unwrap_or_else(|| l2_normalize_per_head(&q));
        let k = dn_recurrence_backend
            .and_then(|b| {
                let k_slice = k.as_slice()?;
                b.qwen35_l2_normalize_per_head(k_slice, dims.head_v_dim, dims.n_k_heads, 1e-6)
            })
            .and_then(|out| Array2::from_shape_vec((dims.head_v_dim, dims.n_k_heads), out).ok())
            .unwrap_or_else(|| l2_normalize_per_head(&k));
        (q, k)
    });

    // 6. Delta-rule recurrence.
    let o = time_section!(
        DN_RECURRENCE,
        if std::env::var("LARQL_QWEN35_PAPER_ORDER").is_err() {
            dn_recurrence_backend
                .and_then(|b| {
                    let q_slice = q.as_slice()?;
                    let k_slice = k.as_slice()?;
                    let v_slice = v.as_slice()?;
                    let log_g_slice = log_g.as_slice()?;
                    let beta_slice = beta.as_slice()?;
                    let recurrent_state = state.recurrent_state.as_slice_mut()?;
                    b.qwen35_deltanet_step(
                        q_slice,
                        k_slice,
                        v_slice,
                        log_g_slice,
                        beta_slice,
                        recurrent_state,
                        dims.head_v_dim,
                        dims.n_k_heads,
                        dims.n_v_heads,
                        sequence_pos,
                        dims.block_gqa,
                    )
                })
                .and_then(|out| {
                    ndarray::Array2::from_shape_vec((dims.head_v_dim, dims.n_v_heads), out).ok()
                })
                .unwrap_or_else(|| {
                    delta_net_step(
                        &q,
                        &k,
                        &v,
                        &log_g,
                        &beta,
                        &mut state.recurrent_state,
                        dims.block_gqa,
                    )
                })
        } else {
            delta_net_step(
                &q,
                &k,
                &v,
                &log_g,
                &beta,
                &mut state.recurrent_state,
                dims.block_gqa,
            )
        }
    );
    if std::env::var("LARQL_QWEN35_DUMP_L0").is_ok() {
        let mut per_head: Vec<(usize, f32)> = (0..dims.n_v_heads)
            .map(|h| {
                let l2 = (0..dims.head_v_dim)
                    .map(|d| {
                        let v = o[[d, h]];
                        v * v
                    })
                    .sum::<f32>()
                    .sqrt();
                (h, l2)
            })
            .collect();
        per_head.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        eprintln!(
            "o(recurrence) shape={:?}: first-3 = [{:.4}, {:.4}, {:.4}]  l2={:.3}  top-3 heads: {} ({:.3}), {} ({:.3}), {} ({:.3})  bot-3 heads: {} ({:.3}), {} ({:.3}), {} ({:.3})",
            o.shape(),
            o[[0, 0]],
            o[[1, 0]],
            o[[2, 0]],
            o.iter().map(|v| v * v).sum::<f32>().sqrt(),
            per_head[0].0, per_head[0].1,
            per_head[1].0, per_head[1].1,
            per_head[2].0, per_head[2].1,
            per_head[per_head.len() - 1].0, per_head[per_head.len() - 1].1,
            per_head[per_head.len() - 2].0, per_head[per_head.len() - 2].1,
            per_head[per_head.len() - 3].0, per_head[per_head.len() - 3].1,
        );
    }
    // o has shape [head_v_dim, n_v_heads] with `o[d, h]` = head h, dim d.
    // The downstream `rms_norm_heads` and element-wise `silu(z)` expect
    // **head-major** flat layout (`flat[h * head_v_dim + d]`), matching
    // HF Qwen3-Next's `out.reshape(B, S, -1)` of an underlying
    // `[..., n_v_heads, head_v_dim]` tensor.
    //
    // Phase C.5c verified: naïve `o.into_iter()` flatten produces
    // DIM-MAJOR flat which scrambles `rms_norm_heads` slices across
    // heads, producing a 46× amplification (l2: 0.6 → 27.4) where
    // llama.cpp produces 5× less. Transpose first so the flatten
    // yields head-major.
    let o_flat_pre: Array1<f32> = time_section!(DN_O_RESHAPE, {
        let r: Array1<f32> = o.t().to_owned().into_iter().collect();
        r
    });
    debug_assert_eq!(o_flat_pre.len(), value_dim);
    dump_subop("dn_o_recurrence", o_flat_pre.as_slice().unwrap_or(&[]));

    // 7. Per-head RMSNorm by ssm_norm (weight is [head_v_dim], shared
    //    across heads). Reshape to [1, value_dim] for rms_norm_heads.
    let mut o_flat = time_section!(DN_RMS_NORM_HEADS, {
        let gpu_o_normed = dn_recurrence_backend.and_then(|b| {
            let o_slice = o_flat_pre.as_slice()?;
            b.qwen35_rms_norm_heads(
                o_slice,
                &weights.ssm_norm,
                dims.n_v_heads,
                dims.head_v_dim,
                crate::residual::DEFAULT_EPS as f32,
            )
        });
        if let Some(out) = gpu_o_normed {
            Array1::from(out)
        } else {
            let o_2d = o_flat_pre
                .into_shape_with_order((1, value_dim))
                .expect("o reshape");
            let o_normed = crate::residual::rms_norm_heads(
                &o_2d,
                &weights.ssm_norm,
                dims.n_v_heads,
                dims.head_v_dim,
                0.0,
            );
            o_normed.index_axis_move(Axis(0), 0)
        }
    });

    dump_subop("dn_o_postnorm", o_flat.as_slice().unwrap_or(&[]));

    // 8. Multiply by SiLU(Z) element-wise.
    time_section!(DN_SILU_Z, {
        for c in 0..value_dim {
            o_flat[c] *= silu(z[c]);
        }
    });
    dump_subop("dn_o_gated", o_flat.as_slice().unwrap_or(&[]));
    if std::env::var("LARQL_QWEN35_DUMP_L0").is_ok() {
        let n = o_flat.len();
        // Optional: dump full final_out (and other vectors) to binary file
        // for elementwise diff vs llama-eval-callback's LLAMA_DUMP_BIN_DIR.
        if let Ok(dir) = std::env::var("LARQL_QWEN35_DUMP_BIN_DIR") {
            let _ = std::fs::create_dir_all(&dir);
            // Single-token, we just write [N] f32. Use the same convention as
            // llama-eval-callback so the diff tool reads both identically.
            let write = |name: &str, data: &[f32], ne0: i64| {
                let path = std::path::Path::new(&dir).join(name);
                if let Ok(mut f) = std::fs::File::create(&path) {
                    use std::io::Write;
                    let ne: [i64; 4] = [ne0, 1, 1, 1];
                    for n in ne {
                        f.write_all(&n.to_le_bytes()).ok();
                    }
                    for v in data {
                        f.write_all(&v.to_le_bytes()).ok();
                    }
                }
            };
            write(
                "our_final_out.bin",
                &o_flat.as_slice().unwrap_or(&[]),
                o_flat.len() as i64,
            );
            write("our_z.bin", &z.as_slice().unwrap_or(&[]), z.len() as i64);
            write(
                "our_x_norm.bin",
                &x_norm.as_slice().unwrap_or(&[]),
                x_norm.len() as i64,
            );
            write(
                "our_qkv_conv.bin",
                &qkv_conv.as_slice().unwrap_or(&[]),
                qkv_conv.len() as i64,
            );
            // Mark done so we don't re-dump on subsequent tokens.
            std::env::remove_var("LARQL_QWEN35_DUMP_BIN_DIR");
        }
        let mut per_head: Vec<(usize, f32)> = (0..dims.n_v_heads)
            .map(|h| {
                let off = h * dims.head_v_dim;
                let l2 = (0..dims.head_v_dim)
                    .map(|d| {
                        let v = o_flat[off + d];
                        v * v
                    })
                    .sum::<f32>()
                    .sqrt();
                (h, l2)
            })
            .collect();
        per_head.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        eprintln!(
            "final_out (pre-ssm_out): first-3 = [{:.4}, {:.4}, {:.4}]  last-3 = [{:.4}, {:.4}, {:.4}]  l2={:.3}  sum={:.4}  mid={:.4} {:.4} {:.4}  top-5 heads: {} ({:.3}) {} ({:.3}) {} ({:.3}) {} ({:.3}) {} ({:.3})  median head: {} ({:.3})  bot-3: {} ({:.3}) {} ({:.3}) {} ({:.3})",
            o_flat[0], o_flat[1], o_flat[2],
            o_flat[n - 3], o_flat[n - 2], o_flat[n - 1],
            o_flat.iter().map(|v| v * v).sum::<f32>().sqrt(),
            o_flat.iter().sum::<f32>(),
            o_flat[1024], o_flat[2048], o_flat[3072],
            per_head[0].0, per_head[0].1,
            per_head[1].0, per_head[1].1,
            per_head[2].0, per_head[2].1,
            per_head[3].0, per_head[3].1,
            per_head[4].0, per_head[4].1,
            per_head[per_head.len() / 2].0, per_head[per_head.len() / 2].1,
            per_head[per_head.len() - 1].0, per_head[per_head.len() - 1].1,
            per_head[per_head.len() - 2].0, per_head[per_head.len() - 2].1,
            per_head[per_head.len() - 3].0, per_head[per_head.len() - 3].1,
        );
    }

    // Batched-prefill exit point: return o_flat (shape [value_dim])
    // so `deltanet_block_prefill` can run one ssm_out.matmul()
    // across all `seq_len` o_flats instead of `seq_len` separate
    // matvecs.
    if skip_ssm_out {
        return o_flat;
    }

    // 9. Output projection (matvec).
    let block_out = time_section!(
        DN_SSM_OUT,
        if let Some(q) = weights.ssm_out_quant.as_ref() {
            matvec_with_backend(q, &o_flat, dn_proj_backend)
        } else {
            weights.ssm_out.dot(&o_flat)
        }
    );
    if std::env::var("LARQL_QWEN35_DUMP_L0").is_ok() {
        let n = block_out.len();
        eprintln!(
            "linear_attn_out:         first-3 = [{:.4}, {:.4}, {:.4}]  last-3 = [{:.4}, {:.4}, {:.4}]  l2={:.3}",
            block_out[0], block_out[1], block_out[2],
            block_out[n - 3], block_out[n - 2], block_out[n - 1],
            block_out.iter().map(|v| v * v).sum::<f32>().sqrt(),
        );
    }
    block_out
}

/// Multi-position sibling of `deltanet_block_step`. Processes
/// `seq_len = x_seq.nrows()` positions through one DeltaNet linear-
/// attention block, chaining the `state` forward exactly as a
/// per-position loop would.
///
/// Phase 4d-scaffold of the batched-prefill arc. Body is a per-row
/// loop today. The wrapper exists so the prefill flow's three
/// sibling entry points — `qwen35_attention_block_prefill` (4b-host,
/// PR #232), `swiglu_moe_lazy_prefill` (4c-scaffold, PR #233), and
/// this function — share the same multi-position signature shape:
///
/// ```text
/// pub fn deltanet_block_prefill(
///     x_seq: &Array2<f32>,
///     weights, dims,
///     state: &mut DeltaNetLayerState,
///     backend,
///     base_pos: usize,
///     layer: usize,
/// ) -> Array2<f32>
/// ```
///
/// Once Phase 4-final wires `qwen35_forward_prefill` to call the
/// three batched siblings together, only the bodies need to change
/// to deliver wall-time savings — the call sites stay put.
///
/// ## DeltaNet has sequential dependency
///
/// Unlike the MoE FFN (positions are independent) and the
/// full-attention block (positions all see the same K/V slab), the
/// DeltaNet recurrence is genuinely sequential: position `i`'s
/// recurrent state depends on position `i-1`'s update. So even the
/// internals follow-up can't simply parallelise over positions —
/// it'll need to either (a) reformulate as a prefix-scan, or (b)
/// batch the non-recurrent parts (projections, RMSNorm, gate/output
/// proj) and leave the recurrent rank-1 update as a sequential
/// inner loop. Option (b) is the practical first step; option (a)
/// is a research project.
///
/// ## Parity test
///
/// `deltanet_block_prefill_matches_per_position_loop` runs both
/// paths on a 3-position synthetic input and asserts:
///   - per-row output values match within `1e-5` absolute,
///   - final `recurrent_state` and `conv_state` match within
///     `1e-5` absolute.
#[allow(clippy::too_many_arguments)]
pub fn deltanet_block_prefill(
    x_seq: &Array2<f32>,
    weights: &DeltaNetLayerWeights,
    dims: &DeltaNetDims,
    state: &mut DeltaNetLayerState,
    backend: Option<&dyn larql_compute::ComputeBackend>,
    base_pos: usize,
    layer: usize,
) -> Array2<f32> {
    let seq_len = x_seq.nrows();
    let hidden = x_seq.ncols();
    let mut out = Array2::<f32>::zeros((seq_len, hidden));
    if seq_len == 0 {
        return out;
    }

    // Phase 4d-internals: batched projections for the linear-
    // attention block. RMSNorm + attn_qkv + attn_gate are the
    // pre-recurrent linear ops — none depend on prior state, so
    // they batch cleanly across the full prompt.
    //
    // Step B (Phase 4e follow-up): the prior `backend.is_none()` gate
    // here was a defensive hold against routing batched matmul through
    // the GPU backend that lacked a batched-matmul kernel. With
    // `matmul_with_backend` now routing Q4_K through `gemm_proj_seq`
    // (cuBLAS hgemm on cached f16 dequant), the GPU path produces a
    // batched matmul too — so the gate is dropped. Per-class GPU
    // residency for the projection matmuls is filtered through
    // `gpu_tier::backend_for(DnProj, ...)` so
    // `LARQL_QWEN35_GPU_NO_DN_PROJ=1` still pushes them back to CPU.
    let batched_projections = weights.attn_qkv_quant.is_some() && weights.attn_gate_quant.is_some();
    let batched_ssm_out = batched_projections && weights.ssm_out_quant.is_some();

    use crate::attention::gpu_tier::{self as dn_gpu_tier, GpuClass as DnGpuClass};
    use crate::attention::quant_dispatch::matmul_with_backend;
    let proj_backend = dn_gpu_tier::backend_for(DnGpuClass::DnProj, backend);
    if batched_projections {
        // 1. Batched pre-mixer RMSNorm of x_seq.
        let mut x_norm_all = Array2::<f32>::zeros((seq_len, hidden));
        crate::attention::deltanet_block::rms_norm_2d_into(
            x_seq.view(),
            &weights.attn_norm,
            dims.eps,
            x_norm_all.view_mut(),
        );

        // 2. Batched qkv_mixed = attn_qkv @ x_norm.
        let qkv_q = weights.attn_qkv_quant.as_ref().unwrap();
        let qkv_all = matmul_with_backend(qkv_q, &x_norm_all, proj_backend);

        // 3. Batched z = attn_gate @ x_norm.
        let gate_q = weights.attn_gate_quant.as_ref().unwrap();
        let z_all = matmul_with_backend(gate_q, &x_norm_all, proj_backend);

        // 4. Per-position recurrent inner — uses pre-computed
        //    projections so the per-row block_step skips its own
        //    RMSNorm + projection step. When `batched_ssm_out` is
        //    enabled the inner also skips the final ssm_out matvec
        //    and returns `o_flat` (shape [value_dim]); we collect
        //    those rows into `o_seq` for one batched matmul below.
        let value_dim = dims.value_dim();
        let mut o_seq: Option<Array2<f32>> = if batched_ssm_out {
            Some(Array2::<f32>::zeros((seq_len, value_dim)))
        } else {
            None
        };
        for i in 0..seq_len {
            let x_row: Array1<f32> = x_seq.row(i).to_owned();
            let qkv_row: Array1<f32> = qkv_all.row(i).to_owned();
            let z_row: Array1<f32> = z_all.row(i).to_owned();
            let y = deltanet_block_step_with_optional_projections(
                &x_row,
                weights,
                dims,
                state,
                backend,
                base_pos + i,
                layer,
                qkv_row.as_slice(),
                z_row.as_slice(),
                batched_ssm_out,
            );
            if let Some(ref mut buf) = o_seq {
                buf.row_mut(i).assign(&y);
            } else {
                out.row_mut(i).assign(&y);
            }
        }

        // 5. Batched ssm_out projection (when enabled). For
        //    Qwen3.6 35B-A3B: ssm_out `[hidden=2048, value_dim=6144]`
        //    Q4_K (~9 MB). Per-row form: seq_len × 9 MB reads at
        //    32K = 288 GB. Matmul form: cores × 9 MB ≈ 216 MB.
        if let Some(o_seq) = o_seq {
            let ssm_q = weights.ssm_out_quant.as_ref().unwrap();
            let block_out_seq = matmul_with_backend(ssm_q, &o_seq, proj_backend);
            out.assign(&block_out_seq);
        }
    } else {
        for i in 0..seq_len {
            let x_row: Array1<f32> = x_seq.row(i).to_owned();
            let y = deltanet_block_step(&x_row, weights, dims, state, backend, base_pos + i, layer);
            out.row_mut(i).assign(&y);
        }
    }
    out
}

#[inline]
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// RMSNorm on a single 1-D vector with weight broadcast across the
/// last axis. `(x / sqrt(mean(x²) + eps)) * weight`.
///
/// `pub(crate)` so the qwen35 full-attention block can reuse it for
/// its pre-norm step.
pub(crate) fn rms_norm_1d_pub(x: &Array1<f32>, weight: &[f32], eps: f32) -> Array1<f32> {
    rms_norm_1d(x, weight, eps)
}

fn rms_norm_1d(x: &Array1<f32>, weight: &[f32], eps: f32) -> Array1<f32> {
    debug_assert_eq!(x.len(), weight.len());
    let n = x.len();
    let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / n as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    Array1::from_iter(x.iter().zip(weight).map(|(&xv, &wv)| xv * inv * wv))
}

/// Batched in-place RMSNorm. Writes each row of `out` from the
/// corresponding row of `x`, using the same per-row formula as
/// `rms_norm_1d` (f32 mean-of-squares — bit-identical to looping
/// over `rms_norm_1d` per row).
///
/// Lifts the per-row pattern
/// ```text
/// for i in 0..seq_len {
///     let row = x.row(i).to_owned();           // alloc
///     let normed = rms_norm_1d_pub(&row, w, e); // alloc
///     out.row_mut(i).assign(&normed);
/// }
/// ```
/// out of `qwen35_forward_prefill`'s post-attention norm step.
/// For a 32K-token prefill with hidden=2048 and 40 layers that
/// removes ~64 GB of allocate-then-copy traffic from intermediate
/// Array1s — pure overhead, no compute change.
///
/// `pub(crate)` so the qwen35 forward and block modules can share
/// it.
pub(crate) fn rms_norm_2d_into(
    x: ndarray::ArrayView2<f32>,
    weight: &[f32],
    eps: f32,
    mut out: ndarray::ArrayViewMut2<f32>,
) {
    let seq_len = x.shape()[0];
    let hidden = x.shape()[1];
    debug_assert_eq!(weight.len(), hidden);
    debug_assert_eq!(out.shape(), x.shape());
    for i in 0..seq_len {
        let row = x.row(i);
        let mean_sq = row.iter().map(|&v| v * v).sum::<f32>() / hidden as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        let mut out_row = out.row_mut(i);
        for j in 0..hidden {
            out_row[j] = row[j] * inv * weight[j];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use std::sync::Arc;

    /// Convenience: build a `DeltaNetLayerWeights` from owned Vec /
    /// Array2 inputs, doing the Arc / ArcArray wrap inline. Reduces
    /// per-test boilerplate (10 fields × Arc-conversion).
    #[allow(clippy::too_many_arguments)]
    fn make_dn_weights(
        attn_norm: Vec<f32>,
        attn_qkv: Array2<f32>,
        attn_gate: Array2<f32>,
        ssm_conv1d: Array2<f32>,
        ssm_dt: Vec<f32>,
        ssm_a: Vec<f32>,
        ssm_beta: Array2<f32>,
        ssm_alpha: Array2<f32>,
        ssm_norm: Vec<f32>,
        ssm_out: Array2<f32>,
    ) -> DeltaNetLayerWeights {
        DeltaNetLayerWeights {
            attn_norm: Arc::from(attn_norm.as_slice()),
            attn_qkv: attn_qkv.into_shared(),
            attn_gate: attn_gate.into_shared(),
            ssm_conv1d: ssm_conv1d.into_shared(),
            ssm_dt: Arc::from(ssm_dt.as_slice()),
            ssm_a: Arc::from(ssm_a.as_slice()),
            ssm_beta: ssm_beta.into_shared(),
            ssm_alpha: ssm_alpha.into_shared(),
            ssm_norm: Arc::from(ssm_norm.as_slice()),
            ssm_out: ssm_out.into_shared(),
            attn_qkv_quant: None,
            attn_gate_quant: None,
            ssm_out_quant: None,
        }
    }

    /// Tiny end-to-end shape check: build a 1-head linear-attention
    /// layer with deterministic weights, feed it a known input, and
    /// verify the output is the right shape + the state was updated.
    /// Numerical correctness is the parity-vs-llama.cpp test's job
    /// (Phase C.5); here we just sanity-check that the pipeline
    /// composes correctly.
    fn make_tiny_dims() -> DeltaNetDims {
        DeltaNetDims {
            hidden: 4,
            head_v_dim: 2,
            n_v_heads: 1,
            n_k_heads: 1,
            d_conv: 2,
            eps: 1e-6,
            block_gqa: true,
        }
    }

    #[test]
    fn deltanet_block_step_output_has_hidden_shape() {
        let dims = make_tiny_dims();
        // conv_dim = 2*key_dim + value_dim = 2*2 + 2 = 6.
        let conv_dim = dims.conv_dim();
        let value_dim = dims.value_dim();
        let key_dim = dims.key_dim();

        // Small constant weights so we can predict shape + non-zero
        // output without overflowing.
        let weights = make_dn_weights(
            vec![1.0_f32; dims.hidden],
            Array2::from_elem((conv_dim, dims.hidden), 0.1_f32),
            Array2::from_elem((value_dim, dims.hidden), 0.1_f32),
            Array2::from_elem((dims.d_conv, conv_dim), 0.5_f32),
            vec![0.0_f32; dims.n_v_heads],
            vec![-1.0_f32; dims.n_v_heads], // moderate decay
            Array2::from_elem((dims.n_v_heads, dims.hidden), 0.1_f32),
            Array2::from_elem((dims.n_v_heads, dims.hidden), 0.1_f32),
            vec![1.0_f32; dims.head_v_dim],
            Array2::from_elem((dims.hidden, value_dim), 0.5_f32),
        );

        let mut state =
            DeltaNetLayerState::allocate(dims.d_conv, conv_dim, dims.head_v_dim, dims.n_v_heads);
        let x = Array1::from_elem(dims.hidden, 1.0_f32);

        let y = deltanet_block_step(&x, &weights, &dims, &mut state, None, 0, 0);
        assert_eq!(y.len(), dims.hidden);
        assert!(y.iter().all(|v| v.is_finite()));

        // Conv state should now hold the most recent qkv_mixed value
        // (since d_conv=2 → state_rows=1). For our constant inputs
        // qkv_mixed = attn_qkv @ x_norm; x_norm has unit RMS so its
        // norm scales x. We don't pin the value, just that it
        // moved off zero.
        let _ = key_dim;
        assert!(
            state.conv_state.iter().any(|v| v.abs() > 0.0),
            "conv_state should have absorbed the new token"
        );
        // Recurrent state likewise non-zero after step (rank-1 update
        // from k ⊗ d).
        assert!(
            state.recurrent_state.iter().any(|v| v.abs() > 0.0),
            "recurrent_state should reflect the rank-1 update"
        );
    }

    #[test]
    fn deltanet_block_step_zero_input_zero_output() {
        // x = 0 → x_norm = 0 (because RMS of zero vector with eps
        // gives x/sqrt(eps) * weight = 0). All projections = 0 →
        // conv input = 0, qkv_conv = SiLU(conv_out) where conv_out
        // depends on the (zero) state. First step: state was zero,
        // qkv_mixed = 0, so qkv_conv = 0. Recurrence inputs zero;
        // delta-rule rank-1 update is zero. Output projection of
        // zero = zero.
        let dims = make_tiny_dims();
        let conv_dim = dims.conv_dim();
        let value_dim = dims.value_dim();

        let weights = make_dn_weights(
            vec![1.0_f32; dims.hidden],
            Array2::zeros((conv_dim, dims.hidden)),
            Array2::zeros((value_dim, dims.hidden)),
            Array2::zeros((dims.d_conv, conv_dim)),
            vec![0.0_f32; dims.n_v_heads],
            vec![0.0_f32; dims.n_v_heads],
            Array2::zeros((dims.n_v_heads, dims.hidden)),
            Array2::zeros((dims.n_v_heads, dims.hidden)),
            vec![1.0_f32; dims.head_v_dim],
            Array2::zeros((dims.hidden, value_dim)),
        );

        let mut state =
            DeltaNetLayerState::allocate(dims.d_conv, conv_dim, dims.head_v_dim, dims.n_v_heads);
        let x = Array1::zeros(dims.hidden);

        let y = deltanet_block_step(&x, &weights, &dims, &mut state, None, 0, 0);
        for &v in y.iter() {
            assert!(v.abs() < 1e-5, "expected zero output, got {v}");
        }
    }

    #[test]
    fn deltanet_block_dims_helpers() {
        let dims = DeltaNetDims {
            hidden: 5120,
            head_v_dim: 128,
            n_v_heads: 48,
            n_k_heads: 16,
            d_conv: 4,
            eps: 1e-6,
            block_gqa: true,
        };
        // Qwen 3.6 27B sanity check.
        assert_eq!(dims.key_dim(), 128 * 16); // 2048
        assert_eq!(dims.value_dim(), 128 * 48); // 6144
        assert_eq!(dims.conv_dim(), 2 * 2048 + 6144); // 10240
    }

    /// `rms_norm_2d_into` must produce bit-identical results to
    /// running `rms_norm_1d_pub` per row + assigning. Locks the
    /// optimisation in `qwen35_forward_prefill` and
    /// `qwen35_attention_block_prefill`.
    #[test]
    fn rms_norm_2d_into_matches_per_row_loop() {
        let seq_len = 5usize;
        let hidden = 7usize;
        let eps = 1e-6_f32;
        let data: Vec<f32> = (0..seq_len * hidden)
            .map(|i| 0.1 + 0.05 * (i as f32) - 0.001 * (i as f32) * (i as f32))
            .collect();
        let x = Array2::from_shape_vec((seq_len, hidden), data).unwrap();
        let weight: Vec<f32> = (0..hidden).map(|i| 0.5 + 0.07 * i as f32).collect();

        // Per-row reference.
        let mut expected = Array2::<f32>::zeros((seq_len, hidden));
        for i in 0..seq_len {
            let row = x.row(i).to_owned();
            let normed = super::rms_norm_1d_pub(&row, &weight, eps);
            expected.row_mut(i).assign(&normed);
        }

        // Batched in-place.
        let mut got = Array2::<f32>::zeros((seq_len, hidden));
        super::rms_norm_2d_into(x.view(), &weight, eps, got.view_mut());

        // Bit-identical: same formula, same fp accumulation order.
        for i in 0..seq_len {
            for j in 0..hidden {
                assert_eq!(
                    got[[i, j]],
                    expected[[i, j]],
                    "rms_norm_2d_into[{i},{j}]: got={} expected={}",
                    got[[i, j]],
                    expected[[i, j]],
                );
            }
        }
    }

    #[test]
    fn rms_norm_1d_unit_weight_normalises() {
        let x = ndarray::array![3.0_f32, 4.0]; // RMS = sqrt((9+16)/2) = sqrt(12.5)
        let w = [1.0_f32, 1.0];
        let out = rms_norm_1d(&x, &w, 0.0);
        let rms = (12.5_f32).sqrt();
        assert!((out[0] - 3.0 / rms).abs() < 1e-5);
        assert!((out[1] - 4.0 / rms).abs() < 1e-5);
    }

    #[test]
    fn rms_norm_1d_applies_weight() {
        let x = ndarray::array![3.0_f32, 4.0];
        let w = [2.0_f32, 0.5];
        let out = rms_norm_1d(&x, &w, 0.0);
        let rms = (12.5_f32).sqrt();
        assert!((out[0] - 2.0 * 3.0 / rms).abs() < 1e-5);
        assert!((out[1] - 0.5 * 4.0 / rms).abs() < 1e-5);
    }

    #[test]
    fn deltanet_block_step_two_calls_state_carries_forward() {
        // Run the block twice with the same input. State should
        // accumulate: the second call's recurrent state has more
        // mass than the first.
        let dims = make_tiny_dims();
        let conv_dim = dims.conv_dim();
        let value_dim = dims.value_dim();

        let weights = make_dn_weights(
            vec![1.0_f32; dims.hidden],
            Array2::from_elem((conv_dim, dims.hidden), 0.1_f32),
            Array2::from_elem((value_dim, dims.hidden), 0.1_f32),
            Array2::from_elem((dims.d_conv, conv_dim), 0.5_f32),
            vec![0.0_f32; dims.n_v_heads],
            vec![0.0_f32; dims.n_v_heads], // no decay; state accumulates
            Array2::from_elem((dims.n_v_heads, dims.hidden), 0.1_f32),
            Array2::from_elem((dims.n_v_heads, dims.hidden), 0.0_f32),
            vec![1.0_f32; dims.head_v_dim],
            Array2::from_elem((dims.hidden, value_dim), 0.5_f32),
        );

        let mut state =
            DeltaNetLayerState::allocate(dims.d_conv, conv_dim, dims.head_v_dim, dims.n_v_heads);
        let x = Array1::from_elem(dims.hidden, 1.0_f32);

        let _y1 = deltanet_block_step(&x, &weights, &dims, &mut state, None, 0, 0);
        let state_mass_after_1: f32 = state.recurrent_state.iter().map(|&v| v.abs()).sum();
        let _y2 = deltanet_block_step(&x, &weights, &dims, &mut state, None, 1, 0);
        let state_mass_after_2: f32 = state.recurrent_state.iter().map(|&v| v.abs()).sum();

        // Without decay, the state should retain at least as much
        // mass after step 2 as after step 1 (and typically more —
        // rank-1 updates compound when inputs are non-orthogonal).
        assert!(
            state_mass_after_2 >= state_mass_after_1 - 1e-5,
            "state mass shrank without decay: {state_mass_after_1} → {state_mass_after_2}"
        );
    }

    /// Phase 4d-internals parity gate: the `batched_projections`
    /// branch of `deltanet_block_prefill` (CPU + quantised QKV/gate)
    /// must produce the same output as the per-row reference. With
    /// `from_f32_rows` the `QuantTensor::matmul` fast path falls
    /// through to per-row `matvec`, so this test exercises the
    /// wrapper logic that splits projections out of
    /// `deltanet_block_step_with_optional_projections`.
    #[test]
    fn deltanet_block_prefill_batched_projections_matches_per_position_loop() {
        use larql_models::quant::lazy::QuantTensor;
        let dims = make_tiny_dims();
        let conv_dim = dims.conv_dim();
        let value_dim = dims.value_dim();

        // Build weights with attn_qkv_quant / attn_gate_quant SET so
        // the batched_projections branch fires.
        let attn_qkv_dense = Array2::from_elem((conv_dim, dims.hidden), 0.1_f32).into_shared();
        let attn_gate_dense = Array2::from_elem((value_dim, dims.hidden), 0.1_f32).into_shared();
        let attn_qkv_quant = QuantTensor::from_f32_rows(
            conv_dim,
            dims.hidden,
            &vec![0.1_f32; conv_dim * dims.hidden],
        );
        let attn_gate_quant = QuantTensor::from_f32_rows(
            value_dim,
            dims.hidden,
            &vec![0.1_f32; value_dim * dims.hidden],
        );

        let weights = DeltaNetLayerWeights {
            attn_norm: Arc::from(vec![1.0_f32; dims.hidden].as_slice()),
            attn_qkv: attn_qkv_dense,
            attn_gate: attn_gate_dense,
            ssm_conv1d: Array2::from_elem((dims.d_conv, conv_dim), 0.5_f32).into_shared(),
            ssm_dt: Arc::from(vec![0.0_f32; dims.n_v_heads].as_slice()),
            ssm_a: Arc::from(vec![-0.5_f32; dims.n_v_heads].as_slice()),
            ssm_beta: Array2::from_elem((dims.n_v_heads, dims.hidden), 0.1_f32).into_shared(),
            ssm_alpha: Array2::from_elem((dims.n_v_heads, dims.hidden), 0.1_f32).into_shared(),
            ssm_norm: Arc::from(vec![1.0_f32; dims.head_v_dim].as_slice()),
            ssm_out: Array2::from_elem((dims.hidden, value_dim), 0.5_f32).into_shared(),
            attn_qkv_quant: Some(attn_qkv_quant),
            attn_gate_quant: Some(attn_gate_quant),
            ssm_out_quant: None,
        };

        let seq_len = 3usize;
        let x_seq_data = vec![
            1.0_f32, 0.5, -0.25, 0.75, 0.5, 0.5, 0.5, 0.5, -0.25, 0.75, 1.0, -0.5,
        ];
        let x_seq = Array2::from_shape_vec((seq_len, dims.hidden), x_seq_data).unwrap();

        // Reference: per-position loop (forces the same code path
        // via the else branch of `batched_projections`).
        let mut ref_state =
            DeltaNetLayerState::allocate(dims.d_conv, conv_dim, dims.head_v_dim, dims.n_v_heads);
        let mut expected = Array2::<f32>::zeros((seq_len, dims.hidden));
        for i in 0..seq_len {
            let x_row: Array1<f32> = x_seq.row(i).to_owned();
            let y = deltanet_block_step(&x_row, &weights, &dims, &mut ref_state, None, i, 0);
            expected.row_mut(i).assign(&y);
        }

        // Impl: batched (backend = None + both quant tensors set
        // → batched_projections branch fires).
        let mut got_state =
            DeltaNetLayerState::allocate(dims.d_conv, conv_dim, dims.head_v_dim, dims.n_v_heads);
        let got = deltanet_block_prefill(&x_seq, &weights, &dims, &mut got_state, None, 0, 0);

        for i in 0..seq_len {
            for h in 0..dims.hidden {
                let g = got[[i, h]];
                let e = expected[[i, h]];
                assert!(
                    (g - e).abs() < 1e-4,
                    "dn_prefill_batched[{i},{h}] got={} expected={}",
                    g,
                    e,
                );
            }
        }

        // Final state must match too — recurrent inner consumed the
        // pre-computed projections in the same order.
        for (i, (&g, &e)) in got_state
            .recurrent_state
            .iter()
            .zip(ref_state.recurrent_state.iter())
            .enumerate()
        {
            assert!(
                (g - e).abs() < 1e-4,
                "recurrent_state[{i}] got={} expected={}",
                g,
                e,
            );
        }
    }

    /// Phase 4d-internals + ssm_out parity gate: when ALL three
    /// projection quants (attn_qkv, attn_gate, ssm_out) are
    /// present, `deltanet_block_prefill` should:
    ///   1. Batch RMSNorm + attn_qkv + attn_gate upfront.
    ///   2. Loop the recurrent inner with `skip_ssm_out = true`,
    ///      collecting per-position `o_flat`s.
    ///   3. Run ONE batched `ssm_out.matmul()` at the end.
    ///
    /// Output must match the per-row reference. Tolerance is
    /// `< 1e-4` (fp accumulation order in matmul vs per-row
    /// matvec drifts slightly; same gate as the projections-only
    /// test).
    #[test]
    fn deltanet_block_prefill_batched_ssm_out_matches_per_position_loop() {
        use larql_models::quant::lazy::QuantTensor;
        let dims = make_tiny_dims();
        let conv_dim = dims.conv_dim();
        let value_dim = dims.value_dim();

        let attn_qkv_dense = Array2::from_elem((conv_dim, dims.hidden), 0.1_f32).into_shared();
        let attn_gate_dense = Array2::from_elem((value_dim, dims.hidden), 0.1_f32).into_shared();
        let ssm_out_dense = Array2::from_elem((dims.hidden, value_dim), 0.5_f32).into_shared();
        let attn_qkv_quant = QuantTensor::from_f32_rows(
            conv_dim,
            dims.hidden,
            &vec![0.1_f32; conv_dim * dims.hidden],
        );
        let attn_gate_quant = QuantTensor::from_f32_rows(
            value_dim,
            dims.hidden,
            &vec![0.1_f32; value_dim * dims.hidden],
        );
        let ssm_out_quant = QuantTensor::from_f32_rows(
            dims.hidden,
            value_dim,
            &vec![0.5_f32; dims.hidden * value_dim],
        );

        let weights = DeltaNetLayerWeights {
            attn_norm: Arc::from(vec![1.0_f32; dims.hidden].as_slice()),
            attn_qkv: attn_qkv_dense,
            attn_gate: attn_gate_dense,
            ssm_conv1d: Array2::from_elem((dims.d_conv, conv_dim), 0.5_f32).into_shared(),
            ssm_dt: Arc::from(vec![0.0_f32; dims.n_v_heads].as_slice()),
            ssm_a: Arc::from(vec![-0.5_f32; dims.n_v_heads].as_slice()),
            ssm_beta: Array2::from_elem((dims.n_v_heads, dims.hidden), 0.1_f32).into_shared(),
            ssm_alpha: Array2::from_elem((dims.n_v_heads, dims.hidden), 0.1_f32).into_shared(),
            ssm_norm: Arc::from(vec![1.0_f32; dims.head_v_dim].as_slice()),
            ssm_out: ssm_out_dense,
            attn_qkv_quant: Some(attn_qkv_quant),
            attn_gate_quant: Some(attn_gate_quant),
            ssm_out_quant: Some(ssm_out_quant),
        };

        let seq_len = 3usize;
        let x_seq_data = vec![
            1.0_f32, 0.5, -0.25, 0.75, 0.5, 0.5, 0.5, 0.5, -0.25, 0.75, 1.0, -0.5,
        ];
        let x_seq = Array2::from_shape_vec((seq_len, dims.hidden), x_seq_data).unwrap();

        // Reference: per-position loop with the same quantised
        // weights (forces the `else` branch via no-quant path
        // would be wrong — we want the SAME weights, just the
        // per-row code).
        let mut ref_state =
            DeltaNetLayerState::allocate(dims.d_conv, conv_dim, dims.head_v_dim, dims.n_v_heads);
        let mut expected = Array2::<f32>::zeros((seq_len, dims.hidden));
        for i in 0..seq_len {
            let x_row: Array1<f32> = x_seq.row(i).to_owned();
            let y = deltanet_block_step(&x_row, &weights, &dims, &mut ref_state, None, i, 0);
            expected.row_mut(i).assign(&y);
        }

        // Impl: batched (backend = None + all three quants set
        // → batched_projections AND batched_ssm_out both fire).
        let mut got_state =
            DeltaNetLayerState::allocate(dims.d_conv, conv_dim, dims.head_v_dim, dims.n_v_heads);
        let got = deltanet_block_prefill(&x_seq, &weights, &dims, &mut got_state, None, 0, 0);

        for i in 0..seq_len {
            for h in 0..dims.hidden {
                let g = got[[i, h]];
                let e = expected[[i, h]];
                assert!(
                    (g - e).abs() < 1e-4,
                    "dn_prefill_batched_ssm_out[{i},{h}] got={} expected={}",
                    g,
                    e,
                );
            }
        }

        // Recurrent state should be bit-identical (recurrent inner
        // doesn't touch ssm_out, so the batched vs per-row order
        // only differs in the post-recurrent projection).
        for (i, (&g, &e)) in got_state
            .recurrent_state
            .iter()
            .zip(ref_state.recurrent_state.iter())
            .enumerate()
        {
            assert!(
                (g - e).abs() < 1e-5,
                "recurrent_state[{i}] got={} expected={}",
                g,
                e,
            );
        }
    }

    /// Phase 4d-scaffold parity test: `deltanet_block_prefill` over
    /// `seq_len` positions must produce the same per-row outputs AND
    /// the same final `state` as running `deltanet_block_step` once
    /// per position with state chained forward.
    ///
    /// Locks the multi-position contract so the internals follow-up
    /// can swap the wrapper's per-row loop for batched projections
    /// (+ sequential recurrence) without silently drifting numerics.
    #[test]
    fn deltanet_block_prefill_matches_per_position_loop() {
        let dims = make_tiny_dims();
        let conv_dim = dims.conv_dim();
        let value_dim = dims.value_dim();

        let weights = make_dn_weights(
            vec![1.0_f32; dims.hidden],
            Array2::from_elem((conv_dim, dims.hidden), 0.1_f32),
            Array2::from_elem((value_dim, dims.hidden), 0.1_f32),
            Array2::from_elem((dims.d_conv, conv_dim), 0.5_f32),
            vec![0.0_f32; dims.n_v_heads],
            vec![-0.5_f32; dims.n_v_heads],
            Array2::from_elem((dims.n_v_heads, dims.hidden), 0.1_f32),
            Array2::from_elem((dims.n_v_heads, dims.hidden), 0.1_f32),
            vec![1.0_f32; dims.head_v_dim],
            Array2::from_elem((dims.hidden, value_dim), 0.5_f32),
        );

        // Three distinct input rows so the recurrent state's update
        // sequence actually varies between positions.
        let seq_len = 3usize;
        let x_seq_data = vec![
            1.0_f32, 0.5, -0.25, 0.75, // row 0
            0.5, 0.5, 0.5, 0.5, // row 1
            -0.25, 0.75, 1.0, -0.5, // row 2
        ];
        let x_seq = Array2::from_shape_vec((seq_len, dims.hidden), x_seq_data).unwrap();

        // Reference: per-position loop, chaining state.
        let mut ref_state =
            DeltaNetLayerState::allocate(dims.d_conv, conv_dim, dims.head_v_dim, dims.n_v_heads);
        let mut expected = Array2::<f32>::zeros((seq_len, dims.hidden));
        for i in 0..seq_len {
            let x_row: Array1<f32> = x_seq.row(i).to_owned();
            let y = deltanet_block_step(&x_row, &weights, &dims, &mut ref_state, None, i, 0);
            expected.row_mut(i).assign(&y);
        }

        // Impl: batched wrapper, fresh state.
        let mut got_state =
            DeltaNetLayerState::allocate(dims.d_conv, conv_dim, dims.head_v_dim, dims.n_v_heads);
        let got = deltanet_block_prefill(&x_seq, &weights, &dims, &mut got_state, None, 0, 0);

        // Per-row outputs match.
        assert_eq!(got.shape(), expected.shape());
        for i in 0..seq_len {
            for h in 0..dims.hidden {
                let g = got[[i, h]];
                let e = expected[[i, h]];
                assert!(
                    (g - e).abs() < 1e-5,
                    "dn_prefill[{i},{h}] got={} expected={}",
                    g,
                    e,
                );
            }
        }

        // Final state matches exactly — recurrent_state + conv_state.
        for (i, (&g, &e)) in got_state
            .recurrent_state
            .iter()
            .zip(ref_state.recurrent_state.iter())
            .enumerate()
        {
            assert!(
                (g - e).abs() < 1e-5,
                "recurrent_state[{i}] got={} expected={}",
                g,
                e,
            );
        }
        for (i, (&g, &e)) in got_state
            .conv_state
            .iter()
            .zip(ref_state.conv_state.iter())
            .enumerate()
        {
            assert!(
                (g - e).abs() < 1e-5,
                "conv_state[{i}] got={} expected={}",
                g,
                e,
            );
        }
    }
}
