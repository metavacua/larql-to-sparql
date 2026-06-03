# ADR-017: Shader retention under model agnosticity

**Status**: Accepted
**Date**: 2026-05-09
**Context**: A May 2026 cleanup pass nearly deleted ~10 Metal shaders that were marked DEAD or TEST-ONLY because grep didn't find Gemma-specific dispatch sites. Closer inspection (`shader-inventory.md`) showed those shaders are **production-load-bearing for non-Gemma model families** — Llama, Mistral, GPT-2, StarCoder2 all use kernels that look unused if you only check the Gemma path. The same audit also questioned the 2026-05-09 NR2 deletion: NR2 lost on Gemma 3 4B (K=2560) but its multi-row pattern could plausibly have won on larger K dimensions in other model families.

larql-compute is supposed to support Gemma 2/3/4, Llama 1/2/3, Mistral, DeepSeek, Qwen, Mixtral, GPT-OSS, GPT-2, Granite, StarCoder2, and the LARQL custom interpretability architecture (TinyModel). The shader directory is the **capability surface** for those families, not a Gemma-3-4B-perf scratchpad.

## Decision

**Shaders are not deleted purely on the basis of failed Gemma A/B.** Each opt-in or experimental shader gets a *retention rationale* doc-block at the top of its file documenting:

1. **What was tried**: model, hardware, date, kernel-isolated and end-to-end deltas.
2. **Why it stays**: the specific model-family / hardware / scenario where it could plausibly win.
3. **Re-validation gate**: what would constitute a positive A/B that justifies promotion.
4. **Deletion criterion**: what would justify removal (typically "multi-K multi-hardware multi-arch bench all show null").

Retention rationale template (drop into the shader's top doc-comment):

```rust
//! ## Retention rationale (post 2026-05-09 model-agnosticity audit)
//!
//! - **Tried**: <model> on <hardware> on <date>. Result: <iso win>, <end-to-end>.
//! - **Status**: Opt-in via `<env-var>`. Not deleted despite Gemma A/B loss because:
//!   - <Specific scenario A: e.g. "larger K dimensions could win">
//!   - <Specific scenario B: e.g. "M5+/A19 silicon ALU/bandwidth shift">
//!   - <Specific scenario C: e.g. "non-mixed-quant layout">
//! - **Re-validation gate**: <what A/B would promote it>
//! - **Deletion criterion**: <what would justify removal>
```

## Decision rules for shader deletion

A shader can be deleted when **all** of the following are true:

1. **Zero production dispatch sites** in `metal/decode/`, `metal/ops/`, `metal/stages/`, `metal/trait_impl/`, `metal/prefill.rs`, `metal/decode_hybrid.rs`.
2. **Zero env-var gate** that would route to it.
3. **Zero parity test** that exercises the kernel on any model family.
4. **No active or open ROADMAP track** that depends on it.
5. **No defensible model-family-or-hardware revival story** — i.e. the kernel is functionally subsumed by another shader for every architecture larql supports.

Under these rules, the only verifiable orphan in the 2026-05-09 audit was `q4k_qkv_proj_v2` (deleted same day, see `shader-inventory.md` §A).

The "ADR-015 graveyard" candidates (`q4k_ffn_gate_up_coop`, `q4k_ffn_gate_up_f16acc`, the `NORMED_SHADER` variant of `q4k_q6k_qkv_proj`) all have plausible non-Gemma revival stories and stay opt-in with retention rationale doc-blocks added.

## Decision rules for shader addition

When a new shader is added under `crates/larql-compute/src/metal/shaders/`:

1. The shader's top doc-comment **must** declare its applicable model families and op-class.
2. If opt-in (gated by an env-var), the env-var must be declared in `crates/larql-compute/src/options.rs` with a doc-comment explaining the trade.
3. The shader must have **at least one parity test** in `crates/larql-compute/tests/` against a known-good reference (CPU implementation, llama.cpp output, or another existing shader).
4. The shader must be added to `docs/shader-inventory.md` and (if introducing a new dispatch path) `docs/architecture-shader-map.md`.

## Consequences

- **Shader directory is the capability surface**, not a scratchpad. Adding shaders is cheap; removing them is deliberate.
- **Cross-model parity tests become more important.** Currently `crates/larql-compute/tests/` is mostly Gemma-3-4B; gaps exist for Llama 2, Mistral, StarCoder2 (`shader-inventory.md` §D). Filling those gaps is a follow-up project.
- **Retention rationale doc-blocks become a code-review checklist.** A new opt-in shader without a retention rationale block fails review. A deletion PR without satisfying the 5 deletion rules fails review.
- **The 2026-05-09 NR2 deletion stands** as a decision predating this ADR, but is acknowledged as inconsistent with the rule. The implementation is in git history (`git log -- crates/larql-compute/src/metal/shaders/q4k_ffn_gate_up_nr2.rs`) and can be restored if a non-Gemma scenario reveals it as a win.

## Related

- `crates/larql-compute/docs/shader-inventory.md` — per-shader retention rationale and applicability.
- `crates/larql-compute/docs/architecture-shader-map.md` — architecture → shader dispatch map.
- ADR-015 (Isolated vs batched kernel perf) — the iso-vs-batched lesson; **complementary to this ADR**, not superseded. Direction-mismatch in batched diag is still a hard kill (per ADR-015), but Gemma-end-to-end null with direction-match is no longer grounds for deletion under this ADR.
- ADR-018 (Architecture → shader routing pattern) — companion ADR documenting the dispatch-decision pattern.
- `feedback_metal_dispatch_fusion_parallelism.md` (memory) — the TG-count-collapse fusion failure mode.
- `project_prefill_matmul_falsified.md` (memory) — D-PREFILL-MM closure record.
