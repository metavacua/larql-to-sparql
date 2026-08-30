//! GPU expert dispatch for per-layer Q4_K MoE models (§5.12).
//!
//! Called when a MoE layer's expert weights are in `QuantFormat::Q4_K`
//! (per-layer files, not BF16 blob). The router runs on CPU (cheap: 2816×128
//! matmul), expert FFNs run on GPU using existing Q4_K shaders.
//!
//! Flow per MoE layer (after the standard GPU commit for `h_post_attn`):
//!
//! 1. CPU: model-declared expert input + router policy + softmax + top-K.
//! 2. CPU→GPU: write the K selected experts' gate / up / down byte slices
//!    DIRECTLY into pre-allocated Metal staging buffers (one memcpy each).
//! 3. GPU: `q4k_ffn_gate_up` over all K experts in one dispatch.
//! 4. GPU: K × `geglu_gelu_tanh` — one per expert at strided act_buf offset
//!    `e × inter_padded` so down's `K = inter_padded` reads see zero padding.
//! 5. GPU: K × `q4k_matvec` for expert down projections.
//! 6. Commit + wait (one GPU sync per MoE layer).
//! 7. CPU: read back K × hidden expert outputs, weighted sum → `moe_out`.
//!
//! Phase 2 (2026-04-26): all scratch is pre-allocated once per decode call
//! via `MoeScratch::new(...)` and reused across every MoE layer. Previously
//! each layer called `bufs.output(...)` ~10 times (~120ms allocation overhead
//! per token at 30 MoE layers on M3 Max). Buffer sizes are constant per model
//! — `(top_k, hidden, inter_padded)` — so the buffers can stay live for the
//! whole decode and serve every layer's expert routing.

mod dense;
mod dispatch;
mod entry;
mod experts;
mod scratch;

pub use scratch::MoeScratch;
