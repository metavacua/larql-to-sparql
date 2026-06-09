//! End-to-end `qwen35_forward` glue (Phase C.4c of
//! `inference-qwen35-deltanet`).
//!
//! Single-token forward through the full Qwen 3.6 hybrid model:
//! embed → 64 layers (each is a hybrid linear / full-attn block with
//! a Gemma-2-style pre+post-norm sandwich and a SwiGLU FFN
//! suffix) → final_norm → lm_head → logits.
//!
//! Per design.md §6: the **FFN residual adds to the pre-post-norm
//! tensor**. So the layer flow is:
//!
//! ```text
//! attn_out = block(x)              # block does its own pre-norm
//! residual = x + attn_out          # residual add 1
//! ffn_in   = attn_post_norm(residual)  # pre-FFN norm
//! ffn_out  = swiglu(ffn_in)        # silu(gate @ x) * (up @ x) → down
//! final    = residual + ffn_out    # residual add 2 (to PRE-post-norm)
//! ```
//!
//! This is the load-bearing structural difference vs vanilla LLaMA
//! pre-norm-only layers; the FFN residual on the post-normed tensor
//! is a common bug per llama.cpp `qwen35.cpp:141–152`.
//!
//! Weight loading from a vindex / GGUF lives in Phase C.4d; this
//! module only provides the borrow-based forward function. Tests
//! exercise it on tiny synthetic 2-layer models.

use ndarray::{ArcArray2, Array1, Array2};
use std::sync::Arc;

use super::deltanet_block::{rms_norm_1d_pub, DeltaNetDims};
use super::deltanet_state::sigmoid;
use super::qwen35_block::{
    hybrid_layer_step, DeltaNetHybridCache, Qwen35AttentionDims, Qwen35LayerWeights,
};

/// Per-layer MoE FFN weights for `Qwen35MoeArch` (e.g.
/// Qwen3.6-35B-A3B). When a `Qwen35FullLayerWeights` has `moe =
/// Some(...)`, the forward step routes through `swiglu_moe_lazy`
/// instead of the dense `swiglu_ffn_lazy`.
///
/// Tensor layouts (matching the GGUF probe from F.0):
///   - `router`        f32, `[num_experts, hidden]`
///   - `gate_exps`     Q4_K, `[num_experts * expert_ffn_dim, hidden]`
///   - `up_exps`       Q4_K, `[num_experts * expert_ffn_dim, hidden]`
///   - `down_exps`     Q5_K, `[num_experts * hidden, expert_ffn_dim]`
///   - shexp_*         optional shared expert (always-on). Same
///                     layouts as a dense SwiGLU but with `expert_ffn_dim`
///                     (Qwen3.6-35B-A3B has one).
#[derive(Clone)]
pub struct Qwen35MoeFfnWeights {
    pub router: larql_models::quant::lazy::QuantTensor,
    pub gate_exps: larql_models::quant::lazy::QuantTensor,
    pub up_exps: larql_models::quant::lazy::QuantTensor,
    pub down_exps: larql_models::quant::lazy::QuantTensor,
    pub shexp_gate: Option<larql_models::quant::lazy::QuantTensor>,
    pub shexp_up: Option<larql_models::quant::lazy::QuantTensor>,
    pub shexp_down: Option<larql_models::quant::lazy::QuantTensor>,
    /// Per-feature sigmoid gate for the shared expert.
    /// `ffn_gate_inp_shexp.weight` (1D F32 [hidden]). Applied as
    /// `y_moe += sigmoid(x · shexp_gate_inp) * swiglu(shexp...)(x)`.
    /// Without this gate, the shared expert contribution is
    /// uncontrolled and produces noticeable token-distribution noise
    /// at every layer. Some Qwen MoE variants ship without it (None
    /// → unweighted addition fallback, the legacy path).
    pub shexp_gate_inp: Option<Arc<[f32]>>,
    pub num_experts: usize,
    pub top_k: usize,
}

/// Full weights for one layer (both kinds): the block weights
/// (linear or attention) PLUS the post-attention norm + SwiGLU FFN.
#[derive(Clone)]
pub struct Qwen35FullLayerWeights {
    /// The block-level weights — DeltaNet for linear layers,
    /// attention for full-attn layers. Use `Qwen35LayerWeights::Linear(...)`
    /// or `::Attention(...)` to construct.
    pub block: Qwen35LayerWeights,
    /// Post-attention RMSNorm weight `[hidden]`. Applied to
    /// (x + block_out) and fed into the FFN as its pre-norm. The
    /// FFN residual still uses the pre-post-norm tensor.
    pub attn_post_norm: Arc<[f32]>,
    /// SwiGLU gate projection `[ffn_dim, hidden]`.
    pub ffn_gate: ArcArray2<f32>,
    /// SwiGLU up projection `[ffn_dim, hidden]`.
    pub ffn_up: ArcArray2<f32>,
    /// SwiGLU down projection `[hidden, ffn_dim]`.
    pub ffn_down: ArcArray2<f32>,
    /// Optional lazy-quantised gate / up / down. When `Some`, the
    /// matvec dispatches through `QuantTensor::matvec` and the
    /// corresponding dense `ffn_*` field is unused (typically a 0×0
    /// placeholder). Populated by `load_qwen35_weights_lazy_ffn`.
    pub ffn_gate_quant: Option<larql_models::quant::lazy::QuantTensor>,
    pub ffn_up_quant: Option<larql_models::quant::lazy::QuantTensor>,
    pub ffn_down_quant: Option<larql_models::quant::lazy::QuantTensor>,
    /// MoE FFN. When `Some(_)`, the forward step dispatches through
    /// `swiglu_moe_lazy` and the dense `ffn_*` / `ffn_*_quant` fields
    /// are unused. Populated by `load_qwen35_moe_weights` for
    /// `Qwen35MoeArch` GGUFs.
    pub moe: Option<Qwen35MoeFfnWeights>,
}

/// Full Qwen 3.6 model weights — embed, every layer's weights, and
/// the output head.
#[derive(Clone)]
pub struct Qwen35Weights {
    /// Token embedding matrix `[vocab, hidden]`. The lookup uses
    /// `embed.row(token_id)`. May be a 0×0 placeholder when
    /// `embed_quant` is populated and the caller does row-lookup
    /// via `QuantTensor::row_to_f32`.
    pub embed: ArcArray2<f32>,
    /// Lazy-quantised embed (Q4_K typical). When `Some`, the
    /// forward uses `embed_quant.row_to_f32(token_id)` instead of
    /// the dense `embed` field. Saves ~4 GiB on Qwen3.6-27B.
    pub embed_quant: Option<larql_models::quant::lazy::QuantTensor>,
    /// Per-layer full weights, indexed 0..n_layer.
    pub layers: Vec<Qwen35FullLayerWeights>,
    /// Final RMSNorm weight `[hidden]`.
    pub final_norm: Arc<[f32]>,
    /// LM head projection `[vocab, hidden]`. Often tied to `embed`
    /// (the caller decides; this struct just holds whichever).
    /// May be a 0×0 placeholder when `lm_head_quant` is populated and
    /// the caller is using the lazy-quantised matvec path.
    pub lm_head: ArcArray2<f32>,
    /// Lazy-quantised LM head, if loaded via
    /// `larql_models::load_gguf_lazy_lm_head`. When `Some`, the
    /// forward dispatches the final logits matvec through
    /// `QuantTensor::matvec` and `lm_head` is unused.
    pub lm_head_quant: Option<larql_models::quant::lazy::QuantTensor>,
    /// FFN intermediate dim (for shape assertions in tests).
    pub ffn_dim: usize,
    /// Optional GPU (or other) quant-matvec backend. Phase E.1
    /// of `qwen35-gpu-forward`: when set, lazy-quant tensors
    /// route through `backend.quant_matvec(...)` before falling
    /// back to the CPU rayon path. None keeps the all-CPU
    /// behaviour shipped in Phase 2a-2d.
    pub backend: Option<std::sync::Arc<dyn larql_compute::ComputeBackend>>,
}

/// Batched prefill — process `prompt_ids` end-to-end through the
/// model, returning the logits for the **last** prompt token (the
/// first decode-step input). Mutates `hybrid_cache` exactly as if
/// `qwen35_forward_step` had been called once per token.
///
/// **Phase 4a infrastructure** (long-context arc, batched-prefill
/// sub-arc): this function exists to give downstream batching kernels
/// a single insertion point. The current body is the
/// straightforward per-token loop — bit-equivalent to the prior
/// inline loop in `qwen35_generate_with_sampling`. Subsequent PRs
/// in the arc replace this body with kernels that process N
/// positions per call:
///
/// - **4b**: batched full-attn prefill via the existing
///   `fused_prefill_attention_seq_device_into` CUDA kernel
///   (already in `cuda::attn`, just needs host plumbing).
/// - **4c**: batched MoE FFN router + expert dispatch (route N
///   tokens, bucket by expert, batched matmul per non-empty
///   bucket).
/// - **4d**: batched DeltaNet recurrence (chunked prefix-scan on
///   the recurrent state; needs new CUDA kernel work).
///
/// The single-token `qwen35_forward_step` stays as the decode-step
/// entry point — decode is fundamentally sequential (each token
/// depends on the last one's logits).
pub fn qwen35_forward_prefill(
    prompt_ids: &[u32],
    weights: &Qwen35Weights,
    dn_dims: &DeltaNetDims,
    attn_dims: &Qwen35AttentionDims,
    hybrid_cache: &mut DeltaNetHybridCache,
    eps: f32,
) -> Option<Array1<f32>> {
    if prompt_ids.is_empty() {
        return None;
    }
    // The batched path bypasses the per-token diagnostic dump /
    // trace env vars (`LARQL_QWEN35_DUMP_LAYER_BOUNDARY`,
    // `LARQL_QWEN35_TRACE`, `LARQL_QWEN35_DUMP_FINAL_BIN_DIR`,
    // `LARQL_QWEN35_DUMP_FINAL`, `LARQL_QWEN35_DUMP_L0`). Those are
    // per-token introspection tools; when any is set, fall back to
    // the per-token loop so they keep working unchanged.
    //
    // `LARQL_QWEN35_FORCE_PER_TOKEN_PREFILL=1` is the dedicated A/B
    // switch — set it to force the per-token loop without enabling
    // any diagnostic side-effects, for benching batched vs fallback.
    let dump_active = std::env::var("LARQL_QWEN35_DUMP_LAYER_BOUNDARY").is_ok()
        || std::env::var("LARQL_QWEN35_TRACE").is_ok()
        || std::env::var("LARQL_QWEN35_DUMP_FINAL_BIN_DIR").is_ok()
        || std::env::var("LARQL_QWEN35_DUMP_FINAL").is_ok()
        || std::env::var("LARQL_QWEN35_DUMP_L0").is_ok()
        || std::env::var("LARQL_QWEN35_FORCE_PER_TOKEN_PREFILL").is_ok();
    if dump_active {
        let mut last = None;
        for &tok in prompt_ids {
            last = Some(qwen35_forward_step(
                tok,
                weights,
                dn_dims,
                attn_dims,
                hybrid_cache,
                eps,
            ));
        }
        return last;
    }

    use crate::attention::qwen35_block::hybrid_layer_prefill;
    let backend = weights.backend.as_deref();
    let n_layers = weights.layers.len();
    debug_assert_eq!(n_layers, hybrid_cache.num_layers());
    let hidden = dn_dims.hidden;
    let seq_len = prompt_ids.len();
    let base_pos = hybrid_cache.next_position;

    // 1. Embed all prompt tokens into `x_seq: [seq_len, hidden]`.
    let mut x_seq = Array2::<f32>::zeros((seq_len, hidden));
    for (i, &token_id) in prompt_ids.iter().enumerate() {
        let row = if let Some(qt) = weights.embed_quant.as_ref() {
            qt.row_to_f32(token_id as usize)
                .expect("embed_quant row_to_f32")
        } else {
            weights.embed.row(token_id as usize).to_owned()
        };
        x_seq.row_mut(i).assign(&row);
    }

    // 2. Hybrid layer stack.
    //
    // The block dispatch goes through `hybrid_layer_prefill` (which
    // routes to `qwen35_attention_block_prefill` or
    // `deltanet_block_prefill`). The FFN side branches three ways
    // exactly like the per-token forward:
    //   - MoE (top-K experts + optional shared) → swiglu_moe_lazy_prefill
    //   - Lazy-quant dense → per-row swiglu_ffn_lazy
    //   - Dense fp32 fallback → per-row swiglu_ffn
    //
    // `swiglu_moe_lazy_prefill` (PR #233) has a batched signature
    // but its body is still a per-position loop — the internals
    // follow-up will fill in the scatter-gather expert dispatch.
    // The wiring here is what unlocks that swap.
    //
    // We need to mutably borrow `hybrid_cache.dn_state` and
    // `hybrid_cache.kv_layers` separately inside `hybrid_layer_prefill`;
    // split the borrow with destructuring so the borrow checker
    // accepts the two `&mut`s.
    let DeltaNetHybridCache {
        kv_layers,
        dn_state,
        layer_kinds,
        next_position: _,
        ..
    } = hybrid_cache;

    // Hoist the FFN-norm scratch buffer outside the layer loop:
    // `rms_norm_2d_into` overwrites every element each iteration, so
    // reusing a single allocation saves `n_layers` fresh Array2
    // allocations per prefill (each [seq_len, hidden] = 256 MB at
    // 32K × 2048).
    let mut ffn_in = Array2::<f32>::zeros((seq_len, hidden));

    for layer in 0..n_layers {
        let layer_w = &weights.layers[layer];
        let block_out = hybrid_layer_prefill(
            layer,
            &x_seq,
            &layer_w.block,
            dn_dims,
            attn_dims,
            dn_state,
            kv_layers,
            base_pos,
            backend,
            layer_kinds,
        );
        // 2b. Residual add 1: move `block_out` into `residual` and
        //     accumulate `x_seq` in place. Saves an Array2 alloc per
        //     layer vs `let residual = &x_seq + &block_out`.
        let mut residual = block_out;
        residual += &x_seq;
        // 2c. Post-attention RMSNorm — batched-in-place into the
        //     hoisted `ffn_in` scratch.
        crate::attention::deltanet_block::rms_norm_2d_into(
            residual.view(),
            &layer_w.attn_post_norm,
            eps,
            ffn_in.view_mut(),
        );
        // 2d. SwiGLU FFN.
        let ffn_out: Array2<f32> = if let Some(moe) = layer_w.moe.as_ref() {
            swiglu_moe_lazy_prefill(
                &ffn_in,
                &moe.router,
                &moe.gate_exps,
                &moe.up_exps,
                &moe.down_exps,
                moe.shexp_gate.as_ref(),
                moe.shexp_up.as_ref(),
                moe.shexp_down.as_ref(),
                moe.shexp_gate_inp.as_deref(),
                moe.num_experts,
                moe.top_k,
                backend,
            )
        } else if layer_w.ffn_gate_quant.is_some()
            || layer_w.ffn_up_quant.is_some()
            || layer_w.ffn_down_quant.is_some()
        {
            let mut out = Array2::<f32>::zeros((seq_len, hidden));
            for i in 0..seq_len {
                let row: Array1<f32> = ffn_in.row(i).to_owned();
                let y = swiglu_ffn_lazy(
                    &row,
                    layer_w.ffn_gate_quant.as_ref(),
                    layer_w.ffn_up_quant.as_ref(),
                    layer_w.ffn_down_quant.as_ref(),
                    &layer_w.ffn_gate,
                    &layer_w.ffn_up,
                    &layer_w.ffn_down,
                    backend,
                );
                out.row_mut(i).assign(&y);
            }
            out
        } else {
            let mut out = Array2::<f32>::zeros((seq_len, hidden));
            for i in 0..seq_len {
                let row: Array1<f32> = ffn_in.row(i).to_owned();
                let y = swiglu_ffn(
                    &row,
                    layer_w.ffn_gate.view(),
                    layer_w.ffn_up.view(),
                    layer_w.ffn_down.view(),
                );
                out.row_mut(i).assign(&y);
            }
            out
        };
        // 2e. Residual add 2: move `ffn_out` into `x_seq` and add
        //     `residual` in place. Saves another Array2 alloc vs
        //     `x_seq = &residual + &ffn_out`.
        let mut new_x = ffn_out;
        new_x += &residual;
        x_seq = new_x;
    }

    // 3. Final RMSNorm + LM head on the last position only —
    //    prefill returns just the logits for the next token after
    //    the prompt, matching the per-token loop's behaviour.
    let last_idx = seq_len - 1;
    let last_x: Array1<f32> = x_seq.row(last_idx).to_owned();
    let x_final = rms_norm_1d_pub(&last_x, &weights.final_norm, eps);
    let logits = if let Some(qt) = weights.lm_head_quant.as_ref() {
        let lm_backend = crate::attention::gpu_tier::backend_for(
            crate::attention::gpu_tier::GpuClass::LmHead,
            backend,
        );
        crate::attention::quant_dispatch::matvec_with_backend(qt, &x_final, lm_backend)
    } else {
        weights.lm_head.dot(&x_final)
    };

    // 4. Advance cursor by the full prompt length.
    hybrid_cache.next_position += seq_len;

    Some(logits)
}

/// One-token forward through the entire Qwen 3.6 model.
///
/// Returns the logits `[vocab]` for the next token after the input.
/// Mutates `hybrid_cache` (appends K/V or updates DeltaNet state per
/// layer) and advances `hybrid_cache.next_position`.
pub fn qwen35_forward_step(
    token_id: u32,
    weights: &Qwen35Weights,
    dn_dims: &DeltaNetDims,
    attn_dims: &Qwen35AttentionDims,
    hybrid_cache: &mut DeltaNetHybridCache,
    eps: f32,
) -> Array1<f32> {
    let n_layers = weights.layers.len();
    debug_assert_eq!(n_layers, hybrid_cache.num_layers());

    // Per-session token counter for the LARQL_QWEN35_DUMP_LAYER_BOUNDARY
    // diagnostic. Reset by setting `LARQL_QWEN35_DUMP_TOKEN_RESET=1`
    // before the first forward call.
    if std::env::var("LARQL_QWEN35_DUMP_LAYER_BOUNDARY").is_ok() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static TOK_COUNTER: AtomicUsize = AtomicUsize::new(0);
        if std::env::var("LARQL_QWEN35_DUMP_TOKEN_RESET").is_ok() {
            TOK_COUNTER.store(0, Ordering::Relaxed);
            std::env::remove_var("LARQL_QWEN35_DUMP_TOKEN_RESET");
        }
        let tok = TOK_COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::set_var("LARQL_QWEN35_DUMP_TOKEN_TAG", format!("tok{tok}"));
    }

    use crate::attention::fine_profile::{self, *};
    use crate::time_section;

    // 1. Embed lookup. Lazy-quant path takes precedence when set;
    //    falls back to dense `embed.row(token_id)` otherwise.
    let mut x: Array1<f32> = time_section!(
        EMBED,
        if let Some(qt) = weights.embed_quant.as_ref() {
            let [vocab, hidden] = qt.shape();
            debug_assert!((token_id as usize) < vocab);
            debug_assert_eq!(hidden, dn_dims.hidden);
            qt.row_to_f32(token_id as usize)
                .expect("embed_quant row_to_f32")
        } else {
            let vocab = weights.embed.shape()[0];
            let hidden = weights.embed.shape()[1];
            debug_assert!((token_id as usize) < vocab);
            debug_assert_eq!(hidden, dn_dims.hidden);
            weights.embed.row(token_id as usize).to_owned()
        }
    );

    // Optional per-layer trace, enabled via `LARQL_QWEN35_TRACE=1`.
    // Prints residual stream stats to stderr at each layer so a
    // single forward run can be bisected for "where does the
    // stream blow up / collapse." Cheap when disabled (the env
    // lookup is a thread-local cache hit) so leaving it in the
    // production forward is fine.
    let trace = std::env::var("LARQL_QWEN35_TRACE").is_ok();
    if trace {
        eprintln!(
            "    embed:                  l2={:.3} max_abs={:.3}",
            l2(&x),
            max_abs(&x),
        );
    }

    // 2. Hybrid layer stack.
    for layer in 0..n_layers {
        let layer_w = &weights.layers[layer];
        // 2a. Block forward (linear or full-attn). The block does
        // its own pre-norm internally (`attn_norm`).
        let backend = weights.backend.as_deref();
        // Linear-block timing is attributed by the per-section
        // accumulators inside `deltanet_block_step`; full-attn time
        // lands in ATTN_BLOCK so the totals reflect the actual block
        // kind. Match on the discriminant cheaply.
        let is_full_attn = matches!(
            &layer_w.block,
            crate::attention::qwen35_block::Qwen35LayerWeights::Attention(_)
        );
        let block_out = if is_full_attn {
            time_section!(
                ATTN_BLOCK,
                hybrid_layer_step(
                    layer,
                    &x,
                    &layer_w.block,
                    dn_dims,
                    attn_dims,
                    hybrid_cache,
                    backend,
                    hybrid_cache.next_position,
                )
            )
        } else {
            hybrid_layer_step(
                layer,
                &x,
                &layer_w.block,
                dn_dims,
                attn_dims,
                hybrid_cache,
                backend,
                hybrid_cache.next_position,
            )
        };
        // 2b. Residual add 1 — `residual = x + block_out`.
        let residual: Array1<f32> = time_section!(RESIDUAL_ADDS, &x + &block_out);
        // 2c. Post-attention RMSNorm (only on `has_post_norms`
        // architectures — Qwen 3.6 is one of them).
        let ffn_in = time_section!(
            FFN_RMS_NORM,
            rms_norm_1d_pub(&residual, &layer_w.attn_post_norm, eps)
        );
        // 2d. SwiGLU FFN. MoE arch takes precedence; then the
        //     lazy-quant dense path; then dense ndarray dot fallback.
        let ffn_out = if let Some(moe) = layer_w.moe.as_ref() {
            swiglu_moe_lazy(
                &ffn_in,
                &moe.router,
                &moe.gate_exps,
                &moe.up_exps,
                &moe.down_exps,
                moe.shexp_gate.as_ref(),
                moe.shexp_up.as_ref(),
                moe.shexp_down.as_ref(),
                moe.shexp_gate_inp.as_deref(),
                moe.num_experts,
                moe.top_k,
                weights.backend.as_deref(),
            )
        } else if layer_w.ffn_gate_quant.is_some()
            || layer_w.ffn_up_quant.is_some()
            || layer_w.ffn_down_quant.is_some()
        {
            swiglu_ffn_lazy(
                &ffn_in,
                layer_w.ffn_gate_quant.as_ref(),
                layer_w.ffn_up_quant.as_ref(),
                layer_w.ffn_down_quant.as_ref(),
                &layer_w.ffn_gate,
                &layer_w.ffn_up,
                &layer_w.ffn_down,
                weights.backend.as_deref(),
            )
        } else {
            swiglu_ffn(
                &ffn_in,
                layer_w.ffn_gate.view(),
                layer_w.ffn_up.view(),
                layer_w.ffn_down.view(),
            )
        };
        // 2e. Residual add 2 — `x = residual + ffn_out` (NOT
        // `ffn_in + ffn_out`; the FFN residual bypasses the
        // post-norm per design.md §6).
        x = time_section!(RESIDUAL_ADDS, &residual + &ffn_out);

        // Optional per-layer binary tensor dump for elementwise parity
        // diff vs llama-eval-callback (`LLAMA_DUMP_BIN_DIR` on llama.cpp
        // side). Triggered by `LARQL_QWEN35_DUMP_LAYER_BOUNDARY=<dir>` —
        // dumps `block_out_lN.bin`, `attn_residual_lN.bin`,
        // `ffn_in_lN.bin`, `ffn_out_lN.bin`, `x_after_lN.bin` for every
        // layer. Each call is per-token; the caller is responsible for
        // dispatching the right token (or pre-pending the token-id).
        if let Ok(dir) = std::env::var("LARQL_QWEN35_DUMP_LAYER_BOUNDARY") {
            let _ = std::fs::create_dir_all(&dir);
            let write = |name: &str, data: &[f32]| {
                use std::io::Write;
                let path = std::path::Path::new(&dir).join(name);
                if let Ok(mut f) = std::fs::File::create(&path) {
                    let ne: [i64; 4] = [data.len() as i64, 1, 1, 1];
                    for n in ne {
                        f.write_all(&n.to_le_bytes()).ok();
                    }
                    for v in data {
                        f.write_all(&v.to_le_bytes()).ok();
                    }
                }
            };
            // Use token-prefixed dir so successive prompt tokens don't
            // overwrite each other. The harness sets an outer env that
            // names the token index, e.g. `dir/tok0/`, `dir/tok1/`, ...
            let token_subdir =
                std::env::var("LARQL_QWEN35_DUMP_TOKEN_TAG").unwrap_or_else(|_| "tok".to_string());
            let layer_dir = format!("{}/{}", dir, token_subdir);
            let _ = std::fs::create_dir_all(&layer_dir);
            let path_for = |name: &str| format!("{}/{}", layer_dir, name);
            let write_at = |name: &str, data: &[f32]| {
                use std::io::Write;
                if let Ok(mut f) = std::fs::File::create(path_for(name)) {
                    let ne: [i64; 4] = [data.len() as i64, 1, 1, 1];
                    for n in ne {
                        f.write_all(&n.to_le_bytes()).ok();
                    }
                    for v in data {
                        f.write_all(&v.to_le_bytes()).ok();
                    }
                }
            };
            let _ = write; // suppress unused; we now use write_at
            write_at(
                &format!("block_out_l{layer:02}.bin"),
                block_out.as_slice().unwrap_or(&[]),
            );
            write_at(
                &format!("residual_l{layer:02}.bin"),
                residual.as_slice().unwrap_or(&[]),
            );
            write_at(
                &format!("ffn_in_l{layer:02}.bin"),
                ffn_in.as_slice().unwrap_or(&[]),
            );
            write_at(
                &format!("ffn_out_l{layer:02}.bin"),
                ffn_out.as_slice().unwrap_or(&[]),
            );
            write_at(
                &format!("x_after_l{layer:02}.bin"),
                x.as_slice().unwrap_or(&[]),
            );
        }

        if trace {
            let kind = matches!(
                layer_w.block,
                crate::attention::qwen35_block::Qwen35LayerWeights::Linear(_)
            );
            eprintln!(
                "  L{layer:02} {} block_l2={:.3} block_max={:.3}  ffn_l2={:.3} ffn_max={:.3}  x_l2={:.3} x_max={:.3}",
                if kind { "lin" } else { "atn" },
                l2(&block_out),
                max_abs(&block_out),
                l2(&ffn_out),
                max_abs(&ffn_out),
                l2(&x),
                max_abs(&x),
            );
        }
    }

    // 3. Final norm + lm_head.
    let x_final = time_section!(FINAL_NORM, rms_norm_1d_pub(&x, &weights.final_norm, eps));
    // Optional final-norm + logits dump for elementwise parity diff
    // vs llama.cpp's `result_norm.bin` / `result_output.bin`. Triggered
    // by `LARQL_QWEN35_DUMP_FINAL_BIN_DIR=<dir>` (single-shot, dumps
    // each call but the harness will only call once for the last
    // prompt token). Caller is responsible for invoking at the right
    // step.
    if let Ok(dir) = std::env::var("LARQL_QWEN35_DUMP_FINAL_BIN_DIR") {
        let _ = std::fs::create_dir_all(&dir);
        let tag =
            std::env::var("LARQL_QWEN35_DUMP_TOKEN_TAG").unwrap_or_else(|_| "tok".to_string());
        let write = |name: &str, data: &[f32]| {
            use std::io::Write;
            let path = std::path::Path::new(&dir).join(format!("{tag}_{name}"));
            if let Ok(mut f) = std::fs::File::create(&path) {
                let ne: [i64; 4] = [data.len() as i64, 1, 1, 1];
                for n in ne {
                    f.write_all(&n.to_le_bytes()).ok();
                }
                for v in data {
                    f.write_all(&v.to_le_bytes()).ok();
                }
            }
        };
        write("x_final.bin", x_final.as_slice().unwrap_or(&[]));
    }
    if std::env::var("LARQL_QWEN35_DUMP_FINAL").is_ok() {
        let n = x_final.len();
        eprintln!(
            "x_final: first-3 = [{:.4}, {:.4}, {:.4}]  last-3 = [{:.4}, {:.4}, {:.4}]  l2={:.3}",
            x_final[0],
            x_final[1],
            x_final[2],
            x_final[n - 3],
            x_final[n - 2],
            x_final[n - 1],
            x_final.iter().map(|v| v * v).sum::<f32>().sqrt(),
        );
    }
    // Final logits matvec. Prefer the lazy-quantised path when
    // populated; falls back to dense f32 dot otherwise. When a
    // GPU backend is attached, the lazy path routes through
    // `matvec_with_backend` for cuBLAS-style GEMV.
    let logits = time_section!(
        LM_HEAD,
        if let Some(qt) = weights.lm_head_quant.as_ref() {
            // Phase E.7: per-class GPU residency. `LARQL_QWEN35_GPU_NO_LM_HEAD=1`
            // forces the final logits matvec onto CPU rayon.
            let lm_backend = crate::attention::gpu_tier::backend_for(
                crate::attention::gpu_tier::GpuClass::LmHead,
                weights.backend.as_deref(),
            );
            crate::attention::quant_dispatch::matvec_with_backend(qt, &x_final, lm_backend)
        } else {
            weights.lm_head.dot(&x_final)
        }
    );
    if let Ok(dir) = std::env::var("LARQL_QWEN35_DUMP_FINAL_BIN_DIR") {
        let tag =
            std::env::var("LARQL_QWEN35_DUMP_TOKEN_TAG").unwrap_or_else(|_| "tok".to_string());
        let path = std::path::Path::new(&dir).join(format!("{tag}_logits.bin"));
        if let Ok(mut f) = std::fs::File::create(&path) {
            use std::io::Write;
            let data = logits.as_slice().unwrap_or(&[]);
            let ne: [i64; 4] = [data.len() as i64, 1, 1, 1];
            for n in ne {
                f.write_all(&n.to_le_bytes()).ok();
            }
            for v in data {
                f.write_all(&v.to_le_bytes()).ok();
            }
        }
    }

    if trace {
        eprintln!(
            "    final_norm(x):          l2={:.3} max_abs={:.3}",
            l2(&x_final),
            max_abs(&x_final),
        );
        eprintln!(
            "    logits:                 l2={:.3} max_abs={:.3}",
            l2(&logits),
            max_abs(&logits),
        );
    }

    // 4. Advance position for the next token's RoPE.
    hybrid_cache.next_position += 1;

    if fine_profile::enabled() {
        fine_profile::dump_and_reset();
    }

    logits
}

#[inline]
fn l2(a: &Array1<f32>) -> f32 {
    a.iter().map(|&v| v * v).sum::<f32>().sqrt()
}

#[inline]
fn max_abs(a: &Array1<f32>) -> f32 {
    a.iter().fold(0.0_f32, |acc, &v| acc.max(v.abs()))
}

/// SwiGLU FFN: `down @ (silu(gate @ x) * (up @ x))`.
///
/// - `gate`: `[ffn_dim, hidden]`
/// - `up`:   `[ffn_dim, hidden]`
/// - `down`: `[hidden, ffn_dim]`
fn swiglu_ffn(
    x: &Array1<f32>,
    gate: ndarray::ArrayView2<f32>,
    up: ndarray::ArrayView2<f32>,
    down: ndarray::ArrayView2<f32>,
) -> Array1<f32> {
    let g = gate.dot(x); // [ffn_dim]
    let u = up.dot(x); // [ffn_dim]
    let mut inter = Array1::<f32>::zeros(g.len());
    for i in 0..g.len() {
        inter[i] = (g[i] * sigmoid(g[i])) * u[i]; // silu(g) * u
    }
    down.dot(&inter) // [hidden]
}

/// SwiGLU FFN with per-projection lazy-quantised fallback. Each of
/// the three projections takes the lazy form when `Some`, falling
/// back to the dense `ArcArray2<f32>` when `None`. Allows the bench
/// to compare configurations where only some projections are lazy
/// (or to drop in AVX2-accelerated `QuantTensor::matvec` in a
/// follow-up without touching the call sites).
#[allow(clippy::too_many_arguments)]
/// MoE FFN forward for Qwen3.6-35B-A3B and similar architectures.
///
/// Layout matches the Qwen35MoeArch GGUFs:
///   - `router`        `[num_experts, hidden]` f32 — produces a
///                     `[num_experts]` vector of routing logits.
///   - `gate_exps`     `[num_experts * ffn_dim, hidden]` Q4_K, 3D-packed
///   - `up_exps`       same shape, same quant
///   - `down_exps`     `[num_experts * hidden, ffn_dim]` Q5_K, 3D-packed
///   - shared expert (optional): the always-on dense SwiGLU with its
///     own `[ffn_dim, hidden]` gate/up and `[hidden, ffn_dim]` down.
///     If `shexp_*` is `None` this term is dropped (some Qwen MoE
///     variants don't have a shared expert).
///
/// Algorithm (per token):
///   1. `logits = router @ x`                                      // [num_experts]
///   2. `(idx, top_logits) = top_k(logits, top_k)`
///   3. `w = softmax(top_logits)`                                  // [top_k]
///   4. for each chosen expert `e_i`:
///        `y_i = swiglu(gate_exps[e_i] @ x, up_exps[e_i] @ x) @ down_exps[e_i]`
///   5. `y_moe = sum_i (w_i * y_i)`
///   6. `y_shared = swiglu(shexp_gate @ x, shexp_up @ x) @ shexp_down`
///   7. return `y_moe + y_shared`
///
/// Each expert dispatch reuses the existing `swiglu_ffn_lazy` path
/// (which already plumbs through gpu_tier::Ffn / paired matvec / the
/// device-resident fused block when the backend supports it). The
/// per-expert weight slicing is via `QuantTensor::expert_slice` —
/// zero-copy `Arc<[u8]>` views, F.1.
#[allow(clippy::too_many_arguments)]
fn swiglu_moe_lazy(
    x: &Array1<f32>,
    router: &larql_models::quant::lazy::QuantTensor,
    gate_exps: &larql_models::quant::lazy::QuantTensor,
    up_exps: &larql_models::quant::lazy::QuantTensor,
    down_exps: &larql_models::quant::lazy::QuantTensor,
    shexp_gate: Option<&larql_models::quant::lazy::QuantTensor>,
    shexp_up: Option<&larql_models::quant::lazy::QuantTensor>,
    shexp_down: Option<&larql_models::quant::lazy::QuantTensor>,
    shexp_gate_inp: Option<&[f32]>,
    num_experts: usize,
    top_k: usize,
    backend: Option<&dyn larql_compute::ComputeBackend>,
) -> Array1<f32> {
    swiglu_moe_lazy_with_optional_logits(
        x,
        None,
        router,
        gate_exps,
        up_exps,
        down_exps,
        shexp_gate,
        shexp_up,
        shexp_down,
        shexp_gate_inp,
        num_experts,
        top_k,
        backend,
    )
}

/// Same body as `swiglu_moe_lazy` but with the option to pass
/// pre-computed router logits. When `Some(logits)`, the router
/// matvec is skipped — used by `swiglu_moe_lazy_prefill` to batch
/// the router into one matmul across all prompt positions instead
/// of `seq_len` separate matvecs.
#[allow(clippy::too_many_arguments)]
fn swiglu_moe_lazy_with_optional_logits(
    x: &Array1<f32>,
    precomputed_logits: Option<&[f32]>,
    router: &larql_models::quant::lazy::QuantTensor,
    gate_exps: &larql_models::quant::lazy::QuantTensor,
    up_exps: &larql_models::quant::lazy::QuantTensor,
    down_exps: &larql_models::quant::lazy::QuantTensor,
    shexp_gate: Option<&larql_models::quant::lazy::QuantTensor>,
    shexp_up: Option<&larql_models::quant::lazy::QuantTensor>,
    shexp_down: Option<&larql_models::quant::lazy::QuantTensor>,
    shexp_gate_inp: Option<&[f32]>,
    num_experts: usize,
    top_k: usize,
    backend: Option<&dyn larql_compute::ComputeBackend>,
) -> Array1<f32> {
    use crate::attention::gpu_tier::{self, GpuClass};
    use crate::attention::quant_dispatch::matvec_with_backend;
    use ndarray::ArcArray2;

    // The FFN GPU residency knob covers MoE too — pushing FFN to CPU
    // means all expert dispatches run CPU rayon (which is the
    // VRAM-minimal mode larql's value prop targets).
    let backend = gpu_tier::backend_for(GpuClass::Ffn, backend);

    // 1. Router logits — pre-computed by the batched-matmul path
    //    in `swiglu_moe_lazy_prefill`, or computed here per-token.
    let logits: Array1<f32> = match precomputed_logits {
        Some(slice) => {
            debug_assert_eq!(slice.len(), num_experts);
            Array1::from(slice.to_vec())
        }
        None => matvec_with_backend(router, x, backend),
    };
    debug_assert_eq!(logits.len(), num_experts);

    // 2. Top-K selection on host. num_experts is typically 128-256 so a
    //    full sort is cheap; top_k is 8 so we don't bother with a
    //    proper partial-sort heap.
    let mut idx_logit: Vec<(usize, f32)> =
        logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    idx_logit.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top: Vec<(usize, f32)> = idx_logit.into_iter().take(top_k).collect();

    if std::env::var("LARQL_QWEN35_MOE_DUMP").is_ok() {
        eprintln!(
            "MOE topk: experts={:?} logits_top={:?}",
            top.iter().map(|x| x.0).collect::<Vec<_>>(),
            top.iter().map(|x| x.1).collect::<Vec<_>>(),
        );
    }

    // 3. Softmax over the top-K logits (numerically stable).
    let max_logit = top
        .iter()
        .map(|&(_, l)| l)
        .fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = top.iter().map(|&(_, l)| (l - max_logit).exp()).collect();
    let sum_exp: f32 = exps.iter().sum();
    let weights: Vec<f32> = exps.iter().map(|&e| e / sum_exp).collect();

    if std::env::var("LARQL_QWEN35_MOE_DUMP").is_ok() {
        eprintln!("MOE weights: {:?} (sum={:.4})", weights, sum_exp,);
    }

    // 4. Per-expert SwiGLU. Each expert_slice is a zero-copy view of
    //    the 3D-packed parent tensor (F.1).
    let hidden = x.len();
    let mut y_moe = Array1::<f32>::zeros(hidden);
    // Cheap empty placeholders for the dense-fallback slots in
    // swiglu_ffn_lazy — we never use them when the lazy form is set.
    let empty_2d: ArcArray2<f32> =
        ArcArray2::from_shape_vec((0, 0), vec![]).expect("0x0 array is always valid");

    // Phase G.1: pre-slice all top-K experts and issue
    // MADV_WILLNEED hints up front. The kernel starts paging in
    // every selected expert's gate/up/down byte range while we
    // dispatch the first expert's compute. No-op for heap-backed
    // tensors; real win for mmap-backed views (the load_gguf_lazy
    // path). Plus the shared expert, which is always-on.
    let expert_slices: Vec<(
        usize,
        larql_models::quant::lazy::QuantTensor,
        larql_models::quant::lazy::QuantTensor,
        larql_models::quant::lazy::QuantTensor,
    )> = top
        .iter()
        .map(|&(expert_id, _)| {
            let gate_e = gate_exps
                .expert_slice(expert_id, num_experts)
                .expect("expert_slice gate");
            let up_e = up_exps
                .expert_slice(expert_id, num_experts)
                .expect("expert_slice up");
            let down_e = down_exps
                .expert_slice(expert_id, num_experts)
                .expect("expert_slice down");
            gate_e.prefetch_willneed();
            up_e.prefetch_willneed();
            down_e.prefetch_willneed();
            (expert_id, gate_e, up_e, down_e)
        })
        .collect();
    if let (Some(g), Some(u), Some(d)) = (shexp_gate, shexp_up, shexp_down) {
        g.prefetch_willneed();
        u.prefetch_willneed();
        d.prefetch_willneed();
    }

    // Phase G.3 (`cuda-moe-multistream`): when the backend exposes
    // the parallel-expert MoE dispatch, batch all top-K experts onto
    // its worker stream pool so the 8 expert gate/up/down chains can
    // overlap on the GPU. Falls back to the per-expert sequential
    // loop when the backend can't serve the shape (CPU, or any expert
    // not in Q4_K/Q5_K).
    //
    // History — Phase G.2 (reverted): tried `rayon::par_iter` over the
    // experts. Regressed throughput because each expert's internal
    // matvec already saturates the rayon pool and the single CUDA
    // stream serialised all 8 expert kernels anyway. The multi-stream
    // path attacks the CUDA-side bottleneck directly.
    // Note (explored 2026-05-21, reverted): tried batching the shared
    // always-on expert as a 9th slot in the same `qwen35_moe_ffn_batch`
    // call so it could overlap on its own CUDA worker stream alongside
    // the top-K experts. Output collapsed to a single `"` token
    // greedy-decode loop with VRAM 9.2 GiB vs the 11.0 GiB baseline,
    // suggesting the weight cache state was somehow off when the
    // shared expert's tensor bytes joined the batch (the per-expert
    // slices vs the standalone shared tensor may key-collide in the
    // content-fingerprint cache, or another subtle issue). Left as a
    // future investigation; the shared expert still runs through the
    // sequential `swiglu_ffn_lazy` path below — that's only a few ms
    // per layer and the +18% MoE batch win (PR #218) already lands.
    let batch_outputs: Option<Vec<Array1<f32>>> = match backend {
        Some(b) => {
            use crate::attention::quant_dispatch::ggml_type_to_quant_format;
            let mut specs: Vec<larql_compute::MoeFfnExpert<'_>> =
                Vec::with_capacity(expert_slices.len());
            let mut ffn_dim_ok = None;
            let mut all_ok = true;
            for (_expert_id, gate_e, up_e, down_e) in expert_slices.iter() {
                let gfmt = ggml_type_to_quant_format(gate_e.tensor_type());
                let ufmt = ggml_type_to_quant_format(up_e.tensor_type());
                let dfmt = ggml_type_to_quant_format(down_e.tensor_type());
                let (Some(gf), Some(uf), Some(df)) = (gfmt, ufmt, dfmt) else {
                    all_ok = false;
                    break;
                };
                let g_rows = gate_e.shape()[0];
                let u_rows = up_e.shape()[0];
                if g_rows != u_rows {
                    all_ok = false;
                    break;
                }
                if let Some(prev) = ffn_dim_ok {
                    if prev != g_rows {
                        all_ok = false;
                        break;
                    }
                } else {
                    ffn_dim_ok = Some(g_rows);
                }
                specs.push(larql_compute::MoeFfnExpert {
                    gate_data: gate_e.raw_bytes(),
                    gate_format: gf,
                    up_data: up_e.raw_bytes(),
                    up_format: uf,
                    down_data: down_e.raw_bytes(),
                    down_format: df,
                });
            }
            if all_ok {
                if let Some(ffn_dim) = ffn_dim_ok {
                    let x_slice = x.as_slice().expect("Array1 contiguous");
                    b.qwen35_moe_ffn_batch(x_slice, hidden, ffn_dim, &specs)
                        .map(|outputs| outputs.into_iter().map(Array1::from).collect())
                } else {
                    None
                }
            } else {
                None
            }
        }
        None => None,
    };

    if let Some(per_expert) = batch_outputs {
        for (i, y_i) in per_expert.iter().enumerate() {
            let w = weights[i];
            for h in 0..hidden {
                y_moe[h] += w * y_i[h];
            }
        }
    } else {
        // CPU-FFN path (no GPU backend, or backend declined the batched
        // MoE call): parallelise *across* the top-K experts on rayon.
        // Inner matvecs detect the outer rayon scope via
        // `rayon::current_thread_index()` and skip their own par_iter
        // — each expert thread runs its 3 matvecs serially.
        //
        // Microbench (`q4k_q8k_kernel::moe_step_par_outer`): 3× faster
        // than the prior 24 × inner-par_iter pattern (2.72 ms → 0.89 ms
        // per step). The prior PR #217 `par_iter` attempt failed
        // because the inner par_iter was still active and the CUDA
        // stream serialised anyway; this iteration only kicks in when
        // there's no GPU backend (the inner par_iter is the inherited
        // CPU path, now gated by the rayon-context check).
        // Cross-expert batching (par over `(expert × row)` pairs)
        // explored 2026-05-21 and reverted: hardware bench showed it
        // matches the par-outer pattern at ~10.2 t/s (no measurable
        // gain). The workload is already memory-bandwidth-saturated
        // with 8 expert workers on the 24-core Threadripper PRO —
        // adding more parallel work items doesn't unlock more
        // bandwidth, and rayon overhead per row-dot work item eats
        // any compute headroom. Kept the `cpu_moe_ffn_batched` helper
        // for the future (might help on bigger CCX-per-socket parts)
        // but not active. See the function's body for the design.
        use rayon::prelude::*;
        let per_expert: Vec<Array1<f32>> = expert_slices
            .par_iter()
            .map(|(_expert_id, gate_e, up_e, down_e)| {
                swiglu_ffn_lazy(
                    x,
                    Some(gate_e),
                    Some(up_e),
                    Some(down_e),
                    &empty_2d,
                    &empty_2d,
                    &empty_2d,
                    backend,
                )
            })
            .collect();
        for (i, y_i) in per_expert.iter().enumerate() {
            let w = weights[i];
            for h in 0..hidden {
                y_moe[h] += w * y_i[h];
            }
        }
    }

    // 5. Shared expert (always-on, sigmoid-gated). Some Qwen MoE
    //    variants ship without one — `None` skips the whole branch.
    //    When the per-feature `shexp_gate_inp` is present (Qwen 3.6
    //    35B-A3B + Coder-Next both ship one as
    //    `ffn_gate_inp_shexp.weight`, 1D F32 [hidden]), apply the
    //    HF Qwen3MoeSparseMoeBlock pattern:
    //
    //        gate_scalar = sigmoid(x · shexp_gate_inp)
    //        y_moe += gate_scalar * swiglu(shexp_gate, shexp_up, shexp_down)(x)
    //
    //    Without the sigmoid scaling, the shared expert dumps an
    //    over-amplified contribution into the residual every layer,
    //    producing visible token-distribution noise on Qwen3-Coder-
    //    Next greedy completions (verified empirically).
    //
    if let (Some(g), Some(u), Some(d)) = (shexp_gate, shexp_up, shexp_down) {
        let y_shared = swiglu_ffn_lazy(
            x,
            Some(g),
            Some(u),
            Some(d),
            &empty_2d,
            &empty_2d,
            &empty_2d,
            backend,
        );
        let gate_scalar = match shexp_gate_inp {
            Some(w) => {
                let dot: f32 = x.iter().zip(w).map(|(a, b)| a * b).sum();
                1.0 / (1.0 + (-dot).exp())
            }
            None => 1.0, // legacy unweighted addition for arches without an explicit gate
        };
        for h in 0..hidden {
            y_moe[h] += gate_scalar * y_shared[h];
        }
    }

    y_moe
}

/// Multi-position MoE-FFN sibling of `swiglu_moe_lazy`. Processes
/// `seq_len = x_seq.nrows()` token positions through the same MoE
/// SwiGLU as the per-token path.
///
/// Phase 4c-scaffold: the body is a per-position loop over
/// `swiglu_moe_lazy`. The wrapper exists so that
/// `qwen35_forward_prefill` (Phase 4a) and `qwen35_attention_block_prefill`
/// (Phase 4b-host) have a matching MoE entry point to integrate
/// against — once the batched router + scatter-gather expert
/// dispatch lands in a follow-up, only this function's body
/// changes; every call site keeps working.
///
/// The parity test
/// `swiglu_moe_lazy_prefill_matches_per_position_loop` is the
/// safety gate: it locks the multi-position contract so the
/// follow-up can swap the body without silently breaking shape /
/// numerics.
#[allow(clippy::too_many_arguments)]
fn swiglu_moe_lazy_prefill(
    x_seq: &Array2<f32>,
    router: &larql_models::quant::lazy::QuantTensor,
    gate_exps: &larql_models::quant::lazy::QuantTensor,
    up_exps: &larql_models::quant::lazy::QuantTensor,
    down_exps: &larql_models::quant::lazy::QuantTensor,
    shexp_gate: Option<&larql_models::quant::lazy::QuantTensor>,
    shexp_up: Option<&larql_models::quant::lazy::QuantTensor>,
    shexp_down: Option<&larql_models::quant::lazy::QuantTensor>,
    shexp_gate_inp: Option<&[f32]>,
    num_experts: usize,
    top_k: usize,
    backend: Option<&dyn larql_compute::ComputeBackend>,
) -> Array2<f32> {
    // CPU mode (no GPU backend): use scatter-gather. Per-expert
    // SwiGLU runs as one batched matmul against the gathered
    // bucket of positions that picked it, instead of `bucket_size`
    // separate matvecs. Same L3-reuse argument as PR #239's
    // attention projections but with per-position routing.
    //
    // GPU mode: preserve the existing per-row loop, which
    // dispatches the 8-expert MoE batched kernel
    // (`qwen35_moe_ffn_batch`) per token. Scatter-gather would
    // bypass that GPU kernel, regressing the CUDA path.
    if backend.is_none() {
        return swiglu_moe_lazy_prefill_scatter_gather(
            x_seq,
            router,
            gate_exps,
            up_exps,
            down_exps,
            shexp_gate,
            shexp_up,
            shexp_down,
            shexp_gate_inp,
            num_experts,
            top_k,
        );
    }

    let seq_len = x_seq.nrows();
    let hidden = x_seq.ncols();
    let mut out = Array2::<f32>::zeros((seq_len, hidden));

    // Batched router (CPU matmul, but cheap and uniform across
    // backends). Each per-row call inside the loop then skips its
    // own router matvec via `swiglu_moe_lazy_with_optional_logits`.
    let precomputed_logits: Option<Array2<f32>> = router.matmul(x_seq).ok();

    for i in 0..seq_len {
        let x_row: Array1<f32> = x_seq.row(i).to_owned();
        let logits_row: Option<Array1<f32>> =
            precomputed_logits.as_ref().map(|all| all.row(i).to_owned());
        let logits_slice = logits_row.as_ref().and_then(|a| a.as_slice());
        let y = swiglu_moe_lazy_with_optional_logits(
            &x_row,
            logits_slice,
            router,
            gate_exps,
            up_exps,
            down_exps,
            shexp_gate,
            shexp_up,
            shexp_down,
            shexp_gate_inp,
            num_experts,
            top_k,
            backend,
        );
        out.row_mut(i).assign(&y);
    }
    out
}

/// CPU-mode scatter-gather MoE for prefill.
///
/// The classical MoE batching pattern: for each unique expert
/// picked by the per-position top-K, gather the activations of
/// the positions that picked it into a contiguous bucket and
/// run ONE batched SwiGLU on that bucket (gate/up/down each as
/// a single `QuantTensor::matmul` call). Outputs scatter back to
/// `y_moe` with the routing weights.
///
/// For Qwen3.6 35B-A3B (num_experts=128, top_k=8) at 32K prefill,
/// each expert gets ~2K positions on average. Per-expert gate
/// `[ffn_dim=512, hidden=2048]` Q4_K (~0.6 MB) is read once per
/// rayon worker per bucket instead of `bucket_size` times in the
/// per-row form — the same L3-reuse win as the attention
/// projections, applied per-expert.
///
/// Parity vs the per-row reference is approximate (not bit-equal)
/// because the order of partial sums in `y_moe[s, h]` differs:
/// per-row visits experts in top-K rank order; scatter-gather
/// visits them in expert-id order. Floating-point addition is
/// associative-up-to-rounding so per-element drift is bounded.
/// The parity test `swiglu_moe_lazy_prefill_matches_per_position_loop`
/// allows `1e-4` relative tolerance.
#[allow(clippy::too_many_arguments)]
fn swiglu_moe_lazy_prefill_scatter_gather(
    x_seq: &Array2<f32>,
    router: &larql_models::quant::lazy::QuantTensor,
    gate_exps: &larql_models::quant::lazy::QuantTensor,
    up_exps: &larql_models::quant::lazy::QuantTensor,
    down_exps: &larql_models::quant::lazy::QuantTensor,
    shexp_gate: Option<&larql_models::quant::lazy::QuantTensor>,
    shexp_up: Option<&larql_models::quant::lazy::QuantTensor>,
    shexp_down: Option<&larql_models::quant::lazy::QuantTensor>,
    shexp_gate_inp: Option<&[f32]>,
    num_experts: usize,
    top_k: usize,
) -> Array2<f32> {
    let seq_len = x_seq.nrows();
    let hidden = x_seq.ncols();
    let mut y_moe = Array2::<f32>::zeros((seq_len, hidden));
    if seq_len == 0 {
        return y_moe;
    }

    // Step 1: batched router → [seq_len, num_experts].
    let logits_all = match router.matmul(x_seq) {
        Ok(l) => l,
        Err(_) => {
            // Defensive: fall back to per-row router (shouldn't
            // happen given matmul's fallback covers all formats).
            return swiglu_moe_lazy_prefill_per_row(
                x_seq,
                router,
                gate_exps,
                up_exps,
                down_exps,
                shexp_gate,
                shexp_up,
                shexp_down,
                shexp_gate_inp,
                num_experts,
                top_k,
            );
        }
    };

    // Step 2: per-position top-K + bucketing.
    // Each bucket entry is `(position, routing_weight)`.
    let mut buckets: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_experts];
    for s in 0..seq_len {
        let logits_row = logits_all.row(s);
        let mut idx_logit: Vec<(usize, f32)> = logits_row
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        idx_logit.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top: Vec<(usize, f32)> = idx_logit.into_iter().take(top_k).collect();
        let max_logit = top
            .iter()
            .map(|&(_, l)| l)
            .fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = top.iter().map(|&(_, l)| (l - max_logit).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        for (i, &(expert_id, _)) in top.iter().enumerate() {
            let w = exps[i] / sum_exp;
            buckets[expert_id].push((s, w));
        }
    }

    // Step 3: per-expert batched SwiGLU. Parallelise across
    // experts — the gather + 3 matmuls per expert are independent
    // until the scatter step (which writes to shared `y_moe` and
    // stays sequential). `QuantTensor::matmul` detects the outer
    // rayon scope via `current_thread_index()` and falls back to
    // serial inner iteration, so nested-rayon contention is
    // avoided.
    //
    // For Qwen3.6 35B-A3B (128 experts) on a 24-core machine,
    // each worker handles ~5-6 experts serially; their per-expert
    // working set (~3 × 0.6 MB = ~1.8 MB) is well within per-core
    // L2 (~1 MB) / shared L3 (~32+ MB). The gather + matmul +
    // intermediate compute parallelism is what's new here vs
    // PR #241's sequential expert loop.
    use rayon::prelude::*;
    let per_expert_outputs: Vec<(usize, Array2<f32>)> = buckets
        .par_iter()
        .enumerate()
        .filter(|(_, positions)| !positions.is_empty())
        .map(|(expert_id, positions)| {
            let gate_e = gate_exps
                .expert_slice(expert_id, num_experts)
                .expect("gate_exps expert_slice");
            let up_e = up_exps
                .expert_slice(expert_id, num_experts)
                .expect("up_exps expert_slice");
            let down_e = down_exps
                .expert_slice(expert_id, num_experts)
                .expect("down_exps expert_slice");

            // Gather: build [bucket_size, hidden] from x_seq.
            let bucket_size = positions.len();
            let mut bucket_x = Array2::<f32>::zeros((bucket_size, hidden));
            for (b, &(s, _)) in positions.iter().enumerate() {
                bucket_x.row_mut(b).assign(&x_seq.row(s));
            }

            // Three batched matmuls: gate, up, down.
            let g = gate_e.matmul(&bucket_x).expect("gate_e matmul");
            let u = up_e.matmul(&bucket_x).expect("up_e matmul");

            // SiLU(g) * u — elementwise [bucket_size, ffn_dim].
            let ffn_dim = g.ncols();
            let mut inter = Array2::<f32>::zeros((bucket_size, ffn_dim));
            for b in 0..bucket_size {
                for j in 0..ffn_dim {
                    let gv = g[[b, j]];
                    let uv = u[[b, j]];
                    inter[[b, j]] = (gv * sigmoid(gv)) * uv;
                }
            }

            let d = down_e.matmul(&inter).expect("down_e matmul"); // [bucket_size, hidden]
            (expert_id, d)
        })
        .collect();

    // Sequential scatter: each position can receive contributions
    // from up to `top_k` experts; using `+=` here on `y_moe`
    // requires exclusive access to avoid write races.
    for (expert_id, d) in per_expert_outputs.iter() {
        let positions = &buckets[*expert_id];
        for (b, &(s, w)) in positions.iter().enumerate() {
            for h in 0..hidden {
                y_moe[[s, h]] += w * d[[b, h]];
            }
        }
    }

    // Step 4: shared expert (always-on, optionally sigmoid-gated).
    if let (Some(sg), Some(su), Some(sd)) = (shexp_gate, shexp_up, shexp_down) {
        let g = sg.matmul(x_seq).expect("shexp_gate matmul");
        let u = su.matmul(x_seq).expect("shexp_up matmul");
        let ffn_dim = g.ncols();
        let mut inter = Array2::<f32>::zeros((seq_len, ffn_dim));
        for s in 0..seq_len {
            for j in 0..ffn_dim {
                let gv = g[[s, j]];
                let uv = u[[s, j]];
                inter[[s, j]] = (gv * sigmoid(gv)) * uv;
            }
        }
        let d = sd.matmul(&inter).expect("shexp_down matmul");
        // Per-position sigmoid gate scalar.
        for s in 0..seq_len {
            let gate_scalar = match shexp_gate_inp {
                Some(w) => {
                    let dot: f32 = x_seq.row(s).iter().zip(w.iter()).map(|(a, b)| a * b).sum();
                    1.0 / (1.0 + (-dot).exp())
                }
                None => 1.0,
            };
            for h in 0..hidden {
                y_moe[[s, h]] += gate_scalar * d[[s, h]];
            }
        }
    }

    y_moe
}

/// Per-row reference path retained as the fallback when matmul
/// errors (defensive — currently unreachable since `matmul` has
/// its own per-row fallback).
#[allow(clippy::too_many_arguments)]
fn swiglu_moe_lazy_prefill_per_row(
    x_seq: &Array2<f32>,
    router: &larql_models::quant::lazy::QuantTensor,
    gate_exps: &larql_models::quant::lazy::QuantTensor,
    up_exps: &larql_models::quant::lazy::QuantTensor,
    down_exps: &larql_models::quant::lazy::QuantTensor,
    shexp_gate: Option<&larql_models::quant::lazy::QuantTensor>,
    shexp_up: Option<&larql_models::quant::lazy::QuantTensor>,
    shexp_down: Option<&larql_models::quant::lazy::QuantTensor>,
    shexp_gate_inp: Option<&[f32]>,
    num_experts: usize,
    top_k: usize,
) -> Array2<f32> {
    let seq_len = x_seq.nrows();
    let hidden = x_seq.ncols();
    let mut out = Array2::<f32>::zeros((seq_len, hidden));
    for i in 0..seq_len {
        let x_row: Array1<f32> = x_seq.row(i).to_owned();
        let y = swiglu_moe_lazy(
            &x_row,
            router,
            gate_exps,
            up_exps,
            down_exps,
            shexp_gate,
            shexp_up,
            shexp_down,
            shexp_gate_inp,
            num_experts,
            top_k,
            None,
        );
        out.row_mut(i).assign(&y);
    }
    out
}

/// Cross-expert batched CPU MoE FFN. Replaces the per-expert
/// `swiglu_ffn_lazy` loop with two big par_iter passes — one over
/// `(expert × ffn_row)` for fused gate+up+silu, one over
/// `(expert × hidden_row)` for the down matvec. Distributes work
/// across all rayon threads (24 physical cores on the bench host)
/// instead of letting 16 of them sit idle while 8 expert workers
/// each chew through their own matvec.
///
/// Hardware bench (2026-05-21, Threadripper PRO 5965WX): matched
/// the par-outer pattern at ~10.2 t/s — **no measurable gain**.
/// The workload is already memory-bandwidth-saturated with 8 expert
/// workers, and rayon's per-work-item overhead eats any compute
/// headroom from finer-grained parallelism. Function kept as
/// dead code so future bench on bigger CCX-per-socket parts can
/// retry without re-deriving the design.
///
/// Returns `Some(weighted_sum [hidden])` (already softmax-weighted
/// per expert) when every expert's gate/up/down is a supported
/// quant format (Q4_K / Q5_K / Q8_0) with matching shape and the
/// activation length is a Q8_K multiple of 256. `None` falls back
/// to the per-expert par-outer loop.
#[allow(dead_code)]
fn cpu_moe_ffn_batched(
    expert_slices: &[(
        usize,
        larql_models::quant::lazy::QuantTensor,
        larql_models::quant::lazy::QuantTensor,
        larql_models::quant::lazy::QuantTensor,
    )],
    x: &Array1<f32>,
    weights: &[f32],
    hidden: usize,
) -> Option<Vec<f32>> {
    use larql_models::quant::ggml::{
        q4k_q8k_row_dot, q5k_q8k_row_dot, q8_0_q8k_row_dot, quantize_to_q8_k,
    };
    use rayon::prelude::*;

    if expert_slices.is_empty() {
        return None;
    }
    let n_experts = expert_slices.len();
    let ffn_dim = expert_slices[0].1.shape()[0];
    let cols = expert_slices[0].1.shape()[1];
    if cols != hidden || !hidden.is_multiple_of(256) {
        return None;
    }
    // All experts must share gate/up/down shapes, row_bytes, and
    // tensor_type. (Real Qwen MoE GGUFs satisfy this — same dtype
    // across the 3D expert axis.)
    let g_type = expert_slices[0].1.tensor_type();
    let u_type = expert_slices[0].2.tensor_type();
    let d_type = expert_slices[0].3.tensor_type();
    let g_rb = expert_slices[0].1.row_bytes();
    let u_rb = expert_slices[0].2.row_bytes();
    let d_rb = expert_slices[0].3.row_bytes();
    for e in expert_slices.iter() {
        if e.1.shape() != [ffn_dim, hidden]
            || e.2.shape() != [ffn_dim, hidden]
            || e.3.shape() != [hidden, ffn_dim]
            || e.1.tensor_type() != g_type
            || e.2.tensor_type() != u_type
            || e.3.tensor_type() != d_type
        {
            return None;
        }
    }

    // Type→row-dot function pointer. Bail to per-expert fallback for
    // any unsupported format.
    type RowDot = fn(&[u8], &[u8]) -> Result<f32, larql_models::ModelError>;
    let pick = |t: u32| -> Option<RowDot> {
        match t {
            larql_models::quant::ggml::TYPE_Q4_K => Some(q4k_q8k_row_dot as RowDot),
            larql_models::quant::ggml::TYPE_Q5_K => Some(q5k_q8k_row_dot as RowDot),
            larql_models::quant::ggml::TYPE_Q8_0 => Some(q8_0_q8k_row_dot as RowDot),
            _ => None,
        }
    };
    let g_dot = pick(g_type)?;
    let u_dot = pick(u_type)?;
    let d_dot = pick(d_type)?;

    // Phase 0: pre-quantise x once into Q8_K bytes. Same Q8_K is
    // used by every expert's gate AND up matvec.
    let x_slice = x.as_slice()?;
    let q8k_x = quantize_to_q8_k(x_slice);

    // Phase 1: cross-expert gate+up+silu, fully parallel over
    // `n_experts * ffn_dim` work items. Each work item reads one
    // row of gate bytes, one row of up bytes, and the shared Q8_K
    // activation slice — all hot in L1.
    #[inline(always)]
    fn silu(g: f32) -> f32 {
        g / (1.0 + (-g).exp())
    }
    let mut inters: Vec<f32> = vec![0.0_f32; n_experts * ffn_dim];
    inters.par_iter_mut().enumerate().for_each(|(idx, slot)| {
        let e = idx / ffn_dim;
        let r = idx % ffn_dim;
        let gate_bytes = expert_slices[e].1.raw_bytes_view();
        let up_bytes = expert_slices[e].2.raw_bytes_view();
        let g_row = &gate_bytes[r * g_rb..(r + 1) * g_rb];
        let u_row = &up_bytes[r * u_rb..(r + 1) * u_rb];
        let g = g_dot(g_row, &q8k_x).expect("gate row_dot");
        let u = u_dot(u_row, &q8k_x).expect("up row_dot");
        *slot = silu(g) * u;
    });

    // Phase 2: per-expert Q8_K of its intermediate, plus down
    // matvec parallel over `n_experts * hidden` work items.
    let q8k_inters: Vec<Vec<u8>> = (0..n_experts)
        .map(|e| {
            let inter_slice = &inters[e * ffn_dim..(e + 1) * ffn_dim];
            quantize_to_q8_k(inter_slice)
        })
        .collect();
    let mut outputs: Vec<f32> = vec![0.0_f32; n_experts * hidden];
    outputs.par_iter_mut().enumerate().for_each(|(idx, slot)| {
        let e = idx / hidden;
        let r = idx % hidden;
        let down_bytes = expert_slices[e].3.raw_bytes_view();
        let row = &down_bytes[r * d_rb..(r + 1) * d_rb];
        *slot = d_dot(row, &q8k_inters[e]).expect("down row_dot");
    });

    // Phase 3: weighted sum into a [hidden] vector. Sequential —
    // n_experts × hidden is small.
    let mut y = vec![0.0_f32; hidden];
    for e in 0..n_experts {
        let w = weights[e];
        let base = e * hidden;
        for h in 0..hidden {
            y[h] += w * outputs[base + h];
        }
    }
    Some(y)
}

fn swiglu_ffn_lazy(
    x: &Array1<f32>,
    gate_q: Option<&larql_models::quant::lazy::QuantTensor>,
    up_q: Option<&larql_models::quant::lazy::QuantTensor>,
    down_q: Option<&larql_models::quant::lazy::QuantTensor>,
    gate_dense: &ArcArray2<f32>,
    up_dense: &ArcArray2<f32>,
    down_dense: &ArcArray2<f32>,
    backend: Option<&dyn larql_compute::ComputeBackend>,
) -> Array1<f32> {
    use crate::attention::fine_profile::*;
    use crate::attention::gpu_tier::{self, GpuClass};
    use crate::attention::quant_dispatch::{ggml_type_to_quant_format, matvec_with_backend};
    use crate::time_section;
    use larql_compute::QuantFormat;

    // Phase E.7: per-class GPU residency. When the FFN class is
    // disabled (`LARQL_QWEN35_GPU_NO_FFN=1`), drop the backend so
    // every matvec in this block falls back to the CPU rayon path.
    let backend = gpu_tier::backend_for(GpuClass::Ffn, backend);

    // Fastest path: fully device-resident FFN block. Requires all
    // three projections (gate, up, down) to be lazy-quant in a
    // GPU-supported format. On Qwen3.6-27B-Q4_K_S `gate`/`up` are
    // Q4_K and `down` is Q5_K, so the backend must accept a mixed
    // pair. Saves 2 dtohs (g, u to host), 1 htod (inter back to
    // device), and 1 CPU silu loop per FFN layer.
    if let (Some(b), Some(gq), Some(uq), Some(dq)) = (backend, gate_q, up_q, down_q) {
        let gfmt = ggml_type_to_quant_format(gq.tensor_type());
        let ufmt = ggml_type_to_quant_format(uq.tensor_type());
        let dfmt = ggml_type_to_quant_format(dq.tensor_type());
        let gpu_ok =
            |f: Option<QuantFormat>| matches!(f, Some(QuantFormat::Q4_K | QuantFormat::Q5_K));
        if gpu_ok(gfmt) && gpu_ok(ufmt) && gpu_ok(dfmt) {
            let x_slice = x.as_slice().expect("Array1 contiguous");
            let g_shape = gq.shape();
            let u_shape = uq.shape();
            let d_shape = dq.shape();
            let block = time_section!(
                FFN_GATE_UP_PAIR,
                b.qwen35_ffn_lazy_block(
                    x_slice,
                    gq.raw_bytes(),
                    gfmt.unwrap(),
                    g_shape[0],
                    uq.raw_bytes(),
                    ufmt.unwrap(),
                    u_shape[0],
                    dq.raw_bytes(),
                    dfmt.unwrap(),
                    d_shape[0],
                    g_shape[1],
                )
            );
            if let Some(out) = block {
                return Array1::from(out);
            }
        }
    }

    // Fused (gate, up, SwiGLU) CPU path — when backend is absent
    // (CPU-FFN mode), gate_q and up_q have matching shape/format, and
    // the activation is Q8_K-compatible (cols multiple of 256). One
    // row-loop pass over gate+up+silu instead of three sequential
    // passes — fewer intermediate allocations and better L1/L2
    // residency (gate row, up row, x-Q8K all hot per iteration vs
    // gate-all-rows then up-all-rows).
    let fused = if backend.is_none() {
        match (gate_q, up_q) {
            (Some(gq), Some(uq)) => time_section!(
                FFN_GATE_UP_PAIR,
                gq.fused_gate_up_silu(uq, x).ok().flatten()
            ),
            _ => None,
        }
    } else {
        None
    };
    let inter = if let Some(inter_fused) = fused {
        inter_fused
    } else {
        let (g, u) = time_section!(FFN_GATE_UP_PAIR, {
            let paired = match (backend, gate_q, up_q) {
                (Some(b), Some(gq), Some(uq)) => {
                    let gfmt = ggml_type_to_quant_format(gq.tensor_type());
                    let ufmt = ggml_type_to_quant_format(uq.tensor_type());
                    if matches!(gfmt, Some(QuantFormat::Q4_K))
                        && matches!(ufmt, Some(QuantFormat::Q4_K))
                    {
                        let x_slice = x.as_slice().expect("Array1 contiguous");
                        let g_shape = gq.shape();
                        let u_shape = uq.shape();
                        b.qwen35_paired_q4k_matvec(
                            gq.raw_bytes(),
                            g_shape[0],
                            uq.raw_bytes(),
                            u_shape[0],
                            x_slice,
                            g_shape[1],
                        )
                    } else {
                        None
                    }
                }
                _ => None,
            };
            if let Some((gv, uv)) = paired {
                (Array1::from(gv), Array1::from(uv))
            } else {
                let g = match gate_q {
                    Some(q) => matvec_with_backend(q, x, backend),
                    None => gate_dense.dot(x),
                };
                let u = match up_q {
                    Some(q) => matvec_with_backend(q, x, backend),
                    None => up_dense.dot(x),
                };
                (g, u)
            }
        });
        time_section!(FFN_SILU_LOOP, {
            let mut inter = Array1::<f32>::zeros(g.len());
            for i in 0..g.len() {
                inter[i] = (g[i] * sigmoid(g[i])) * u[i];
            }
            inter
        })
    };
    time_section!(
        FFN_DOWN,
        match down_q {
            Some(q) => matvec_with_backend(q, &inter, backend),
            None => down_dense.dot(&inter),
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::deltanet_block::DeltaNetLayerWeights;
    use crate::attention::qwen35_block::{DeltaNetHybridCache, Qwen35AttentionLayerWeights};
    use ndarray::Array2;

    fn tiny_dn_dims() -> DeltaNetDims {
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

    fn tiny_attn_dims() -> Qwen35AttentionDims {
        Qwen35AttentionDims {
            hidden: 4,
            n_head: 2,
            n_head_kv: 1,
            head_dim: 2,
            rotary_dim: 2,
            rope_base: 10_000.0,
            eps: 1e-6,
        }
    }

    /// Build a constant-filled tiny DeltaNet weights bundle. ArcArray
    /// + Arc<[f32]> are cheap-to-clone refcounted owners.
    fn make_dn_weights(dn_dims: &DeltaNetDims) -> DeltaNetLayerWeights {
        let conv_dim = dn_dims.conv_dim();
        let value_dim = dn_dims.value_dim();
        DeltaNetLayerWeights {
            attn_norm: Arc::from(vec![1.0_f32; dn_dims.hidden].as_slice()),
            attn_qkv: Array2::from_elem((conv_dim, dn_dims.hidden), 0.1_f32).into_shared(),
            attn_gate: Array2::from_elem((value_dim, dn_dims.hidden), 0.1_f32).into_shared(),
            ssm_conv1d: Array2::from_elem((dn_dims.d_conv, conv_dim), 0.5_f32).into_shared(),
            ssm_dt: Arc::from(vec![0.0_f32; dn_dims.n_v_heads].as_slice()),
            ssm_a: Arc::from(vec![-1.0_f32; dn_dims.n_v_heads].as_slice()),
            ssm_beta: Array2::from_elem((dn_dims.n_v_heads, dn_dims.hidden), 0.1_f32).into_shared(),
            ssm_alpha: Array2::from_elem((dn_dims.n_v_heads, dn_dims.hidden), 0.1_f32)
                .into_shared(),
            ssm_norm: Arc::from(vec![1.0_f32; dn_dims.head_v_dim].as_slice()),
            ssm_out: Array2::from_elem((dn_dims.hidden, value_dim), 0.5_f32).into_shared(),
            attn_qkv_quant: None,
            attn_gate_quant: None,
            ssm_out_quant: None,
        }
    }

    fn make_attn_weights(attn_dims: &Qwen35AttentionDims) -> Qwen35AttentionLayerWeights {
        Qwen35AttentionLayerWeights {
            attn_norm: Arc::from(vec![1.0_f32; attn_dims.hidden].as_slice()),
            attn_q: Array2::from_elem((attn_dims.fused_q_dim(), attn_dims.hidden), 0.1_f32)
                .into_shared(),
            attn_k: Array2::from_elem((attn_dims.kv_dim(), attn_dims.hidden), 0.1_f32)
                .into_shared(),
            attn_v: Array2::from_elem((attn_dims.kv_dim(), attn_dims.hidden), 0.1_f32)
                .into_shared(),
            attn_q_norm: Arc::from(vec![1.0_f32; attn_dims.head_dim].as_slice()),
            attn_k_norm: Arc::from(vec![1.0_f32; attn_dims.head_dim].as_slice()),
            attn_output: Array2::from_elem((attn_dims.hidden, attn_dims.q_dim()), 0.5_f32)
                .into_shared(),
            attn_q_quant: None,
            attn_k_quant: None,
            attn_v_quant: None,
            attn_output_quant: None,
        }
    }

    /// Build a per-layer suffix (attn_post_norm + SwiGLU FFN trio).
    fn make_suffix(
        hidden: usize,
        ffn_dim: usize,
    ) -> (
        Arc<[f32]>,
        ndarray::ArcArray2<f32>,
        ndarray::ArcArray2<f32>,
        ndarray::ArcArray2<f32>,
    ) {
        (
            Arc::from(vec![1.0_f32; hidden].as_slice()),
            Array2::from_elem((ffn_dim, hidden), 0.1_f32).into_shared(),
            Array2::from_elem((ffn_dim, hidden), 0.1_f32).into_shared(),
            Array2::from_elem((hidden, ffn_dim), 0.5_f32).into_shared(),
        )
    }

    #[test]
    fn swiglu_ffn_zero_input_zero_output() {
        let hidden = 4;
        let ffn_dim = 8;
        let x = Array1::<f32>::zeros(hidden);
        let gate = Array2::from_elem((ffn_dim, hidden), 1.0_f32);
        let up = Array2::from_elem((ffn_dim, hidden), 1.0_f32);
        let down = Array2::from_elem((hidden, ffn_dim), 1.0_f32);
        let y = swiglu_ffn(&x, gate.view(), up.view(), down.view());
        // gate @ 0 = 0, silu(0)=0, up @ 0 = 0, inter = 0, down @ 0 = 0.
        for &v in y.iter() {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn swiglu_ffn_matches_definition_unit_vectors() {
        // hidden=2, ffn_dim=2. gate = I (identity), up = I, down = I.
        // y = down @ (silu(gate @ x) * (up @ x))
        //   = silu(x) * x
        //
        // For x = [1, 0]: silu(1)*1 + silu(0)*0 = silu(1) at index 0.
        let gate = ndarray::array![[1.0_f32, 0.0], [0.0, 1.0]];
        let up = ndarray::array![[1.0_f32, 0.0], [0.0, 1.0]];
        let down = ndarray::array![[1.0_f32, 0.0], [0.0, 1.0]];
        let x = ndarray::array![1.0_f32, 0.0];
        let y = swiglu_ffn(&x, gate.view(), up.view(), down.view());
        let expected = 1.0 * sigmoid(1.0); // silu(1)
        assert!((y[0] - expected).abs() < 1e-5);
        assert!(y[1].abs() < 1e-5);
    }

    /// Parity test for `swiglu_moe_lazy` against a hand-rolled scalar
    /// reference. Tiny setup: hidden=3, ffn_dim=4, num_experts=4,
    /// top_k=2. F32-typed QuantTensors so the GPU fast-paths
    /// fall through to the per-call `matvec_with_backend` fallback
    /// (which routes f32 through `QuantTensor::matvec`).
    #[test]
    fn swiglu_moe_lazy_matches_reference() {
        use larql_models::quant::lazy::QuantTensor;

        let hidden = 3usize;
        let ffn_dim = 4usize;
        let num_experts = 4usize;
        let top_k = 2usize;

        // Reproducible synthetic weights — each tensor gets a distinct
        // per-element pattern so the reference and the impl exercise
        // every slot.
        let mk = |n: usize, seed: f32| -> Vec<f32> {
            (0..n)
                .map(|i| ((i as f32) * 0.07 + seed).sin() * 0.3)
                .collect()
        };
        let router_v = mk(num_experts * hidden, 0.1);
        let gate_exps_v = mk(num_experts * ffn_dim * hidden, 0.2);
        let up_exps_v = mk(num_experts * ffn_dim * hidden, 0.3);
        let down_exps_v = mk(num_experts * hidden * ffn_dim, 0.4);
        let shexp_g_v = mk(ffn_dim * hidden, 0.5);
        let shexp_u_v = mk(ffn_dim * hidden, 0.6);
        let shexp_d_v = mk(hidden * ffn_dim, 0.7);

        let router = QuantTensor::from_f32_rows(num_experts, hidden, &router_v);
        // 3D-packed layout per F.1: rows = num_experts * out_dim.
        let gate_exps = QuantTensor::from_f32_rows(num_experts * ffn_dim, hidden, &gate_exps_v);
        let up_exps = QuantTensor::from_f32_rows(num_experts * ffn_dim, hidden, &up_exps_v);
        let down_exps = QuantTensor::from_f32_rows(num_experts * hidden, ffn_dim, &down_exps_v);
        let shexp_g = QuantTensor::from_f32_rows(ffn_dim, hidden, &shexp_g_v);
        let shexp_u = QuantTensor::from_f32_rows(ffn_dim, hidden, &shexp_u_v);
        let shexp_d = QuantTensor::from_f32_rows(hidden, ffn_dim, &shexp_d_v);

        let x = Array1::from(vec![0.4_f32, -0.2, 0.15]);

        // --- Reference (scalar) ---
        // Router logits.
        let mut logits = vec![0.0_f32; num_experts];
        for e in 0..num_experts {
            let mut acc = 0.0_f32;
            for c in 0..hidden {
                acc += router_v[e * hidden + c] * x[c];
            }
            logits[e] = acc;
        }
        // Top-K (full sort is fine here).
        let mut idx: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top: Vec<(usize, f32)> = idx.into_iter().take(top_k).collect();
        // Softmax.
        let max_l = top
            .iter()
            .map(|&(_, l)| l)
            .fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = top.iter().map(|&(_, l)| (l - max_l).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        let w: Vec<f32> = exps.iter().map(|&e| e / sum_exp).collect();

        let ref_swiglu = |gate: &[f32], up: &[f32], down: &[f32]| -> Vec<f32> {
            // gate @ x and up @ x : [ffn_dim]
            let mut g = vec![0.0_f32; ffn_dim];
            let mut u = vec![0.0_f32; ffn_dim];
            for r in 0..ffn_dim {
                let mut ag = 0.0;
                let mut au = 0.0;
                for c in 0..hidden {
                    ag += gate[r * hidden + c] * x[c];
                    au += up[r * hidden + c] * x[c];
                }
                g[r] = ag;
                u[r] = au;
            }
            // SiLU(g) * u
            let inter: Vec<f32> = g
                .iter()
                .zip(u.iter())
                .map(|(&gi, &ui)| (gi * sigmoid(gi)) * ui)
                .collect();
            // down @ inter : [hidden]
            let mut y = vec![0.0_f32; hidden];
            for r in 0..hidden {
                let mut a = 0.0;
                for c in 0..ffn_dim {
                    a += down[r * ffn_dim + c] * inter[c];
                }
                y[r] = a;
            }
            y
        };

        let mut expected = vec![0.0_f32; hidden];
        let bytes_gate_per_expert = ffn_dim * hidden;
        let bytes_down_per_expert = hidden * ffn_dim;
        for (i, &(e, _)) in top.iter().enumerate() {
            let g0 = e * bytes_gate_per_expert;
            let g1 = g0 + bytes_gate_per_expert;
            let d0 = e * bytes_down_per_expert;
            let d1 = d0 + bytes_down_per_expert;
            let y_i = ref_swiglu(
                &gate_exps_v[g0..g1],
                &up_exps_v[g0..g1],
                &down_exps_v[d0..d1],
            );
            for h in 0..hidden {
                expected[h] += w[i] * y_i[h];
            }
        }
        // Shared expert.
        let y_sh = ref_swiglu(&shexp_g_v, &shexp_u_v, &shexp_d_v);
        for h in 0..hidden {
            expected[h] += y_sh[h];
        }

        // --- Impl ---
        let got = swiglu_moe_lazy(
            &x,
            &router,
            &gate_exps,
            &up_exps,
            &down_exps,
            Some(&shexp_g),
            Some(&shexp_u),
            Some(&shexp_d),
            None, // no per-feature sigmoid gate → unweighted shexp (legacy)
            num_experts,
            top_k,
            None, // CPU path through QuantTensor::matvec
        );

        assert_eq!(got.len(), hidden);
        for h in 0..hidden {
            assert!(
                (got[h] - expected[h]).abs() < 1e-5,
                "moe[{h}] got={} expected={}",
                got[h],
                expected[h],
            );
        }
    }

    /// Shared expert is optional — if all three shexp params are
    /// `None`, the output is just the weighted top-K sum.
    #[test]
    fn swiglu_moe_lazy_without_shared_expert() {
        use larql_models::quant::lazy::QuantTensor;
        let hidden = 2usize;
        let ffn_dim = 2usize;
        let num_experts = 2usize;
        let top_k = 1usize;

        // Router heavily favours expert 0 for x = [1, 0].
        let router_v = vec![
            1.0_f32, 0.0, // expert 0 row
            -1.0, 0.0, // expert 1 row
        ];
        // Expert 0: identity-ish. Expert 1: distinct but should be unused with top_k=1.
        let gate_exps_v = vec![1.0_f32, 0.0, 0.0, 1.0, 5.0, 5.0, 5.0, 5.0];
        let up_exps_v = vec![1.0_f32, 0.0, 0.0, 1.0, 5.0, 5.0, 5.0, 5.0];
        let down_exps_v = vec![1.0_f32, 0.0, 0.0, 1.0, 5.0, 5.0, 5.0, 5.0];

        let router = QuantTensor::from_f32_rows(num_experts, hidden, &router_v);
        let gate_exps = QuantTensor::from_f32_rows(num_experts * ffn_dim, hidden, &gate_exps_v);
        let up_exps = QuantTensor::from_f32_rows(num_experts * ffn_dim, hidden, &up_exps_v);
        let down_exps = QuantTensor::from_f32_rows(num_experts * hidden, ffn_dim, &down_exps_v);
        let x = Array1::from(vec![1.0_f32, 0.0]);

        let got = swiglu_moe_lazy(
            &x,
            &router,
            &gate_exps,
            &up_exps,
            &down_exps,
            None,
            None,
            None,
            None, // no shexp_gate_inp — branch is dead when shexp_* are None anyway
            num_experts,
            top_k,
            None,
        );

        // top_k=1 → softmax over one logit = [1.0]. Expert 0 SwiGLU
        // with identity gate/up/down: y = silu(x) * x.
        // x = [1, 0] → y = [silu(1)*1, 0].
        let expected_y0 = sigmoid(1.0); // 1.0 * sigmoid(1.0)
        assert!(
            (got[0] - expected_y0).abs() < 1e-5,
            "got={} exp={}",
            got[0],
            expected_y0
        );
        assert!(got[1].abs() < 1e-5, "got[1]={}", got[1]);
    }

    /// Phase 4c-scaffold parity test: `swiglu_moe_lazy_prefill` over
    /// `seq_len` positions must produce the same output as running
    /// `swiglu_moe_lazy` once per position. Locks the multi-position
    /// contract so the follow-up (batched router + scatter-gather
    /// expert dispatch) can swap the wrapper's body without silently
    /// changing numerics. Uses the same synthetic-QuantTensor pattern
    /// as `swiglu_moe_lazy_without_shared_expert`.
    #[test]
    fn swiglu_moe_lazy_prefill_matches_per_position_loop() {
        use larql_models::quant::lazy::QuantTensor;
        let hidden = 2usize;
        let ffn_dim = 2usize;
        let num_experts = 2usize;
        let top_k = 1usize;
        let seq_len = 3usize;

        // Router that picks expert 0 when x[0] dominates, expert 1
        // when x[1] dominates. With our 3 test rows we exercise both.
        let router_v = vec![
            1.0_f32, 0.0, // expert 0
            0.0, 1.0, // expert 1
        ];
        // Two distinct experts so the per-position selection actually
        // routes different rows to different code paths.
        let gate_exps_v = vec![1.0_f32, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0];
        let up_exps_v = vec![1.0_f32, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0];
        let down_exps_v = vec![1.0_f32, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0];

        let router = QuantTensor::from_f32_rows(num_experts, hidden, &router_v);
        let gate_exps = QuantTensor::from_f32_rows(num_experts * ffn_dim, hidden, &gate_exps_v);
        let up_exps = QuantTensor::from_f32_rows(num_experts * ffn_dim, hidden, &up_exps_v);
        let down_exps = QuantTensor::from_f32_rows(num_experts * hidden, ffn_dim, &down_exps_v);

        // Mix of x rows: expert-0 favouring, expert-1 favouring, mixed.
        let x_seq =
            Array2::from_shape_vec((seq_len, hidden), vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5]).unwrap();

        // Per-position loop reference.
        let mut expected = Array2::<f32>::zeros((seq_len, hidden));
        for i in 0..seq_len {
            let x_row: Array1<f32> = x_seq.row(i).to_owned();
            let y = swiglu_moe_lazy(
                &x_row,
                &router,
                &gate_exps,
                &up_exps,
                &down_exps,
                None,
                None,
                None,
                None,
                num_experts,
                top_k,
                None,
            );
            expected.row_mut(i).assign(&y);
        }

        // Batched wrapper.
        let got = swiglu_moe_lazy_prefill(
            &x_seq,
            &router,
            &gate_exps,
            &up_exps,
            &down_exps,
            None,
            None,
            None,
            None,
            num_experts,
            top_k,
            None,
        );

        assert_eq!(got.shape(), expected.shape());
        for i in 0..seq_len {
            for h in 0..hidden {
                let g = got[[i, h]];
                let e = expected[[i, h]];
                assert!(
                    (g - e).abs() < 1e-5,
                    "moe_prefill[{i},{h}] got={} expected={}",
                    g,
                    e,
                );
            }
        }
    }

    /// Phase 4-final integration parity test:
    /// `qwen35_forward_prefill` over a multi-token prompt must
    /// produce the same final logits as running
    /// `qwen35_forward_step` once per token in sequence.
    ///
    /// This is the integration-moment safety gate — every batched
    /// sibling (`qwen35_attention_block_prefill`,
    /// `swiglu_moe_lazy_prefill`, `deltanet_block_prefill`) is
    /// individually parity-tested against its per-token version,
    /// but the wiring that composes them needs its own end-to-end
    /// gate. Without this test, an off-by-one in `base_pos` or a
    /// mis-handled per-row RMSNorm could silently drift.
    ///
    /// Two-layer tiny model (linear + attention, dense FFN), three-
    /// token prompt. Asserts:
    ///   - Same final logits (within `1e-4` absolute + `1e-4 * |val|`
    ///     relative tolerance — accumulated fp arithmetic).
    ///   - Same `cache.next_position` after both runs.
    ///   - Same KV-cache contents and DeltaNet state at the end.
    #[test]
    fn qwen35_forward_prefill_matches_per_token_loop() {
        let dn_dims = tiny_dn_dims();
        let attn_dims = tiny_attn_dims();
        let hidden = dn_dims.hidden;
        let vocab = 5;
        let ffn_dim = 8;
        let layer_kinds = vec![true, false];

        let dn_weights = make_dn_weights(&dn_dims);
        let attn_weights = make_attn_weights(&attn_dims);
        let (post0, gate0, up0, down0) = make_suffix(hidden, ffn_dim);
        let (post1, gate1, up1, down1) = make_suffix(hidden, ffn_dim);
        let make_weights = || Qwen35Weights {
            embed: Array2::from_elem((vocab, hidden), 0.5_f32).into_shared(),
            embed_quant: None,
            layers: vec![
                Qwen35FullLayerWeights {
                    block: Qwen35LayerWeights::Linear(dn_weights.clone()),
                    attn_post_norm: post0.clone(),
                    ffn_gate: gate0.clone(),
                    ffn_up: up0.clone(),
                    ffn_down: down0.clone(),
                    ffn_gate_quant: None,
                    ffn_up_quant: None,
                    ffn_down_quant: None,
                    moe: None,
                },
                Qwen35FullLayerWeights {
                    block: Qwen35LayerWeights::Attention(attn_weights.clone()),
                    attn_post_norm: post1.clone(),
                    ffn_gate: gate1.clone(),
                    ffn_up: up1.clone(),
                    ffn_down: down1.clone(),
                    ffn_gate_quant: None,
                    ffn_up_quant: None,
                    ffn_down_quant: None,
                    moe: None,
                },
            ],
            final_norm: Arc::from(vec![1.0_f32; hidden].as_slice()),
            lm_head: Array2::from_elem((vocab, hidden), 0.5_f32).into_shared(),
            lm_head_quant: None,
            ffn_dim,
            backend: None,
        };

        let prompt = [2u32, 0u32, 3u32];

        // --- Reference: per-token loop ---
        let mut ref_cache = DeltaNetHybridCache::allocate(
            &layer_kinds,
            attn_dims.kv_dim(),
            dn_dims.d_conv,
            dn_dims.conv_dim(),
            dn_dims.head_v_dim,
            dn_dims.n_v_heads,
        );
        let ref_weights = make_weights();
        let mut ref_logits: Option<Array1<f32>> = None;
        for &tok in &prompt {
            ref_logits = Some(qwen35_forward_step(
                tok,
                &ref_weights,
                &dn_dims,
                &attn_dims,
                &mut ref_cache,
                1e-6,
            ));
        }
        let ref_logits = ref_logits.expect("non-empty prompt");

        // --- Impl: batched prefill ---
        let mut got_cache = DeltaNetHybridCache::allocate(
            &layer_kinds,
            attn_dims.kv_dim(),
            dn_dims.d_conv,
            dn_dims.conv_dim(),
            dn_dims.head_v_dim,
            dn_dims.n_v_heads,
        );
        let got_weights = make_weights();
        let got_logits = qwen35_forward_prefill(
            &prompt,
            &got_weights,
            &dn_dims,
            &attn_dims,
            &mut got_cache,
            1e-6,
        )
        .expect("non-empty prompt");

        // 1. Cache cursor advanced equally.
        assert_eq!(
            got_cache.next_position, ref_cache.next_position,
            "next_position drift"
        );

        // 2. Final logits parity.
        assert_eq!(got_logits.len(), ref_logits.len());
        for i in 0..ref_logits.len() {
            let g = got_logits[i];
            let r = ref_logits[i];
            let tol = 1e-4 + 1e-4 * r.abs();
            assert!(
                (g - r).abs() < tol,
                "logits[{i}] got={} ref={} tol={}",
                g,
                r,
                tol,
            );
        }

        // 3. KV cache parity for the full-attn layer.
        let (got_k, got_v) = got_cache.kv_layers[1].as_ref().unwrap();
        let (ref_k, ref_v) = ref_cache.kv_layers[1].as_ref().unwrap();
        assert_eq!(got_k.shape(), ref_k.shape());
        assert_eq!(got_v.shape(), ref_v.shape());
        for ((g, r), name) in [(got_k, ref_k), (got_v, ref_v)]
            .iter()
            .zip(["K", "V"].iter())
        {
            for ((idx, &gv), &rv) in g.iter().enumerate().zip(r.iter()) {
                assert!(
                    (gv - rv).abs() < 1e-5,
                    "{name}_cache[{idx}] got={} ref={}",
                    gv,
                    rv,
                );
            }
        }

        // 4. DeltaNet state parity for the linear layer.
        let got_dn = got_cache.dn_state.layers[0].as_ref().unwrap();
        let ref_dn = ref_cache.dn_state.layers[0].as_ref().unwrap();
        for (i, (&g, &r)) in got_dn
            .recurrent_state
            .iter()
            .zip(ref_dn.recurrent_state.iter())
            .enumerate()
        {
            assert!(
                (g - r).abs() < 1e-5,
                "dn recurrent[{i}] got={} ref={}",
                g,
                r,
            );
        }
        for (i, (&g, &r)) in got_dn
            .conv_state
            .iter()
            .zip(ref_dn.conv_state.iter())
            .enumerate()
        {
            assert!((g - r).abs() < 1e-5, "dn conv[{i}] got={} ref={}", g, r);
        }
    }

    #[test]
    fn qwen35_forward_step_returns_vocab_shape() {
        // 2-layer model: layer 0 linear, layer 1 attention.
        let dn_dims = tiny_dn_dims();
        let attn_dims = tiny_attn_dims();
        let hidden = dn_dims.hidden;
        let vocab = 5;
        let ffn_dim = 8;

        let layer_kinds = vec![true, false];

        let mut cache = DeltaNetHybridCache::allocate(
            &layer_kinds,
            attn_dims.kv_dim(),
            dn_dims.d_conv,
            dn_dims.conv_dim(),
            dn_dims.head_v_dim,
            dn_dims.n_v_heads,
        );

        let dn_weights = make_dn_weights(&dn_dims);
        let attn_weights = make_attn_weights(&attn_dims);
        let (post0, gate0, up0, down0) = make_suffix(hidden, ffn_dim);
        let (post1, gate1, up1, down1) = make_suffix(hidden, ffn_dim);

        let weights = Qwen35Weights {
            embed: Array2::from_elem((vocab, hidden), 0.5_f32).into_shared(),
            embed_quant: None,
            layers: vec![
                Qwen35FullLayerWeights {
                    block: Qwen35LayerWeights::Linear(dn_weights),
                    attn_post_norm: post0,
                    ffn_gate: gate0,
                    ffn_up: up0,
                    ffn_down: down0,
                    ffn_gate_quant: None,
                    ffn_up_quant: None,
                    ffn_down_quant: None,
                    moe: None,
                },
                Qwen35FullLayerWeights {
                    block: Qwen35LayerWeights::Attention(attn_weights),
                    attn_post_norm: post1,
                    ffn_gate: gate1,
                    ffn_up: up1,
                    ffn_down: down1,
                    ffn_gate_quant: None,
                    ffn_up_quant: None,
                    ffn_down_quant: None,
                    moe: None,
                },
            ],
            final_norm: Arc::from(vec![1.0_f32; hidden].as_slice()),
            lm_head: Array2::from_elem((vocab, hidden), 0.5_f32).into_shared(),
            lm_head_quant: None,
            ffn_dim,
            backend: None,
        };

        let token = 2u32;
        let logits = qwen35_forward_step(token, &weights, &dn_dims, &attn_dims, &mut cache, 1e-6);
        assert_eq!(logits.len(), vocab);
        assert!(logits.iter().all(|v| v.is_finite()));
        assert_eq!(cache.next_position, 1);
    }

    #[test]
    fn qwen35_forward_step_two_calls_advances_position_and_grows_cache() {
        let dn_dims = tiny_dn_dims();
        let attn_dims = tiny_attn_dims();
        let hidden = dn_dims.hidden;
        let vocab = 5;
        let ffn_dim = 8;

        // Single full-attention layer so we can observe KV growth.
        let layer_kinds = vec![false];

        let mut cache = DeltaNetHybridCache::allocate(
            &layer_kinds,
            attn_dims.kv_dim(),
            dn_dims.d_conv,
            dn_dims.conv_dim(),
            dn_dims.head_v_dim,
            dn_dims.n_v_heads,
        );

        let attn_weights = make_attn_weights(&attn_dims);
        let (post, gate, up, down) = make_suffix(hidden, ffn_dim);

        let weights = Qwen35Weights {
            embed: Array2::from_elem((vocab, hidden), 0.5_f32).into_shared(),
            embed_quant: None,
            layers: vec![Qwen35FullLayerWeights {
                block: Qwen35LayerWeights::Attention(attn_weights),
                attn_post_norm: post,
                ffn_gate: gate,
                ffn_up: up,
                ffn_down: down,
                ffn_gate_quant: None,
                ffn_up_quant: None,
                ffn_down_quant: None,
                moe: None,
            }],
            final_norm: Arc::from(vec![1.0_f32; hidden].as_slice()),
            lm_head: Array2::from_elem((vocab, hidden), 0.5_f32).into_shared(),
            lm_head_quant: None,
            ffn_dim,
            backend: None,
        };

        let _ = qwen35_forward_step(0, &weights, &dn_dims, &attn_dims, &mut cache, 1e-6);
        assert_eq!(cache.next_position, 1);
        let (k_after_1, _) = cache.kv_layers[0].as_ref().unwrap();
        assert_eq!(k_after_1.shape()[0], 1);

        let _ = qwen35_forward_step(1, &weights, &dn_dims, &attn_dims, &mut cache, 1e-6);
        assert_eq!(cache.next_position, 2);
        let (k_after_2, _) = cache.kv_layers[0].as_ref().unwrap();
        assert_eq!(k_after_2.shape()[0], 2);
    }

    #[test]
    fn qwen35_forward_step_dn_state_carries_across_calls() {
        // Single linear layer — verify DeltaNet state moves off zero
        // and accumulates across calls.
        let dn_dims = tiny_dn_dims();
        let attn_dims = tiny_attn_dims();
        let hidden = dn_dims.hidden;
        let vocab = 5;
        let ffn_dim = 8;

        let layer_kinds = vec![true];

        let mut cache = DeltaNetHybridCache::allocate(
            &layer_kinds,
            attn_dims.kv_dim(),
            dn_dims.d_conv,
            dn_dims.conv_dim(),
            dn_dims.head_v_dim,
            dn_dims.n_v_heads,
        );

        let dn_weights = make_dn_weights(&dn_dims);
        let (post, gate, up, down) = make_suffix(hidden, ffn_dim);

        let weights = Qwen35Weights {
            embed: Array2::from_elem((vocab, hidden), 0.5_f32).into_shared(),
            embed_quant: None,
            layers: vec![Qwen35FullLayerWeights {
                block: Qwen35LayerWeights::Linear(dn_weights),
                attn_post_norm: post,
                ffn_gate_quant: None,
                ffn_up_quant: None,
                ffn_down_quant: None,
                ffn_gate: gate,
                ffn_up: up,
                ffn_down: down,
                moe: None,
            }],
            final_norm: Arc::from(vec![1.0_f32; hidden].as_slice()),
            lm_head: Array2::from_elem((vocab, hidden), 0.5_f32).into_shared(),
            lm_head_quant: None,
            ffn_dim,
            backend: None,
        };

        let _ = qwen35_forward_step(2, &weights, &dn_dims, &attn_dims, &mut cache, 1e-6);
        let mass_after_1: f32 = cache.dn_state.layers[0]
            .as_ref()
            .unwrap()
            .recurrent_state
            .iter()
            .map(|&v| v.abs())
            .sum();
        assert!(
            mass_after_1 > 0.0,
            "recurrent_state should be non-zero after one forward step"
        );

        let _ = qwen35_forward_step(2, &weights, &dn_dims, &attn_dims, &mut cache, 1e-6);
        let mass_after_2: f32 = cache.dn_state.layers[0]
            .as_ref()
            .unwrap()
            .recurrent_state
            .iter()
            .map(|&v| v.abs())
            .sum();
        assert!(
            mass_after_2 >= 0.0,
            "recurrent_state still finite after two steps"
        );
    }
}
