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

// The former single-file module put these in scope for every item and
// every test. Re-exported at the parent so the split modules and the
// tests lifted out of it still see one consistent namespace, rather than
// each file re-deriving the same import list.
#[allow(unused_imports)]
pub(crate) use larql_compute::cpu::ops::ternary_matvec::{
    matvec_i2s_a8_f32_into, matvec_i2s_a8_into, quantize_activation_i8, BitLinearWeight,
};
#[allow(unused_imports)]
pub(crate) use ndarray::{Array1, Array2, ArrayView2};

mod ffn;
mod kv_cache;
mod load;
mod predict;
mod streaming;

pub use ffn::{rmsnorm_into, BitNetFfn};
pub use kv_cache::{decode_step, generate, generate_sampled, prefill, BitnetKvCache};
pub use load::{load_bitnet_model, BitnetLoadError};
pub use predict::{predict_bitnet, BitnetLayer, BitnetModel, TernaryPrediction};
pub use streaming::{generate_streaming_bitnet, infer_bitnet_walk, predict_bitnet_with_residuals};

// Tests live beside the module they exercise; each was lifted verbatim
// out of the former single-file `ternary.rs`, so `super::` still means
// this module.
#[cfg(test)]
mod kv_cache_tests;
#[cfg(test)]
mod predict_tests;
#[cfg(test)]
mod streaming_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod walk_tests;
