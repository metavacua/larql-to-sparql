# ADR-018: Architecture → shader routing pattern

**Status**: Accepted
**Date**: 2026-05-09
**Context**: `larql-compute` supports multiple model architectures (Gemma 2/3/4, Llama 1/2/3, Mistral, DeepSeek, Qwen, Mixtral, GPT-OSS, GPT-2, Granite, StarCoder2, TinyModel). Each architecture's compute path is the cumulative result of dozens of dispatch-time predicate checks scattered across `metal/decode/`, `metal/stages/`, `metal/ops/`, `metal/prefill.rs`, `metal/decode_hybrid.rs`. There's no single "for Gemma 3, dispatch X / Y / Z" table. A new contributor adding a model architecture has to grep predicate logic to figure out which kernels run for that family.

## Decision

**Per-architecture dispatch is documented, not refactored** (for now). The bridge between `larql-models/architectures/{family}.rs` and `larql-compute/src/metal/shaders/*.rs` lives in `crates/larql-compute/docs/architecture-shader-map.md`. Each architecture has a row showing which Metal shaders it dispatches at each pipeline stage.

**The current dispatch-time predicate pattern is kept** because:

1. The predicates encode *capabilities*, not architectures (e.g. "this layer has QK-norm" not "this is Gemma 3"). New architectures with the same capability set automatically pick up the right shader without code changes — the trait method `attn_q_norm_key().is_some()` is the routing key, not the family name string.
2. Refactoring to explicit per-arch dispatch modules (e.g. `metal/architectures/gemma3.rs`) would freeze the capability decomposition into class boundaries that don't actually match how the codebase has organically separated concerns. The capability-predicate pattern is more flexible.
3. The cost of the current pattern is **discoverability**, which is solved by documentation (the architecture-shader-map doc), not by code restructuring.

## When to add a new architecture

The procedure (also documented in `architecture-shader-map.md`):

1. Add the trait impl: `crates/larql-models/src/architectures/{family}.rs` overriding `ModelArchitecture` methods that differ from defaults. Most non-Gemma archs override only ~5-10 methods (the family name, position embed key if any, FFN type if non-gated, norm type if not RMS).
2. Add detection logic in `crates/larql-models/src/detect.rs` if the architecture isn't covered by an existing `family` string.
3. Verify dispatch-time predicates handle the new architecture's capabilities. The capabilities that drive dispatch are:
   - `norm_type()` → RmsNorm vs LayerNorm
   - `ffn_type()` → Gated vs Standard
   - `activation()` → SiLU vs GeluTanh vs GeluErf
   - `attn_q_norm_key().is_some()` → QK-norm pre-attn or not
   - `has_v_norm` → V-norm pre-attn (Gemma 4)
   - `is_global_layer(layer)` → per-layer attention pattern (Gemma 4)
   - `has_post_norms` → post-attn / post-FFN norm pipeline (Gemma 3/4)
   - `is_sliding_window_layer(layer)` → sliding window vs full attention (Mistral, Gemma 4)
   - quant format per weight (`Q4_K`, `Q4_KF`, `Q4_0`, `Q6_K`, `Q8_0`, `f16`, `f32`)

   If a new architecture has a capability **not** currently covered by these predicates, that's the design decision: either add a new predicate (preferred — keeps the capability-based pattern) or add architecture-specific code paths (last resort).

4. Add an entry to `crates/larql-inference/tests/test_logits_goldens.rs` with golden tokens for the new arch.
5. Run `./target/release/larql bench <vindex>` to verify dispatch works end-to-end.
6. **Add a row to `architecture-shader-map.md`** documenting which shaders the new architecture's dispatch chain hits.

## Consequences

- **Adding an architecture is a documentation update and a trait impl, not a code reorg.** The Metal kernels mostly just work via the capability-predicate path.
- **Architecture-specific kernels are rare.** When they're needed (e.g. `q4k_q6k_qkv_proj` for the Gemma-3/4-with-ollama-extract layout), the dispatch lives in the existing `metal/decode/encode_qkv.rs` site behind a format-and-norm predicate. The shader file goes in `metal/shaders/` as usual; ADR-017 governs its retention.
- **The `architecture-shader-map.md` doc must stay current.** New shaders or new dispatch decisions must update it. This is enforced by the doc being the only place where the per-arch pipeline is human-readable.

## Alternative considered: per-arch dispatch modules

Considered: refactor `metal/decode/`, `metal/stages/`, `metal/ops/` into per-architecture submodules (`metal/architectures/gemma3.rs`, `metal/architectures/llama.rs`, etc.) where each file wires up the full pipeline for one family.

**Rejected** because:

1. **Capability sharing.** Gemma 3 / Gemma 4 share most of their pipeline; Llama / Mistral / Qwen share most of theirs; StarCoder2 / GPT-2 share theirs. Per-arch modules would either duplicate or require re-introducing capability-class abstractions that already exist via the trait predicates.
2. **Adding a new model family becomes a code-write project.** Currently a new Llama-shaped model "just works" by inheriting Llama defaults from the trait. Per-arch modules would require writing a new dispatch module per family.
3. **The current predicates are battle-tested.** Refactoring carries a non-trivial regression risk for the production paths.

The per-arch refactor is **deferred indefinitely** — possibly revisited if/when a contributor finds the predicate pattern hard to extend to a new architecture (e.g. DeepSeek-V3's MLA + 256-expert MoE may stress the current pattern).

## Future direction

Potential follow-ups (not committed by this ADR):

- **Cross-architecture parity tests at the per-shader level** — currently per-shader tests in `crates/larql-compute/tests/` are mostly Gemma-only. Adding Llama 2 / Mistral parity tests per shader would catch model-specific regressions earlier. Tracked as roadmap item.
- **Capability-predicate documentation** — a doc that enumerates every dispatch-time predicate in the codebase and which architectures hit which branches. Currently this requires grepping. Useful when the predicate pattern becomes hard to discover.
- **`metal/architectures/` per-arch modules** — only if predicate proliferation makes the current dispatch unmaintainable.

## Related

- `crates/larql-compute/docs/architecture-shader-map.md` — the per-architecture map this ADR is the policy for.
- `crates/larql-compute/docs/shader-inventory.md` — per-shader retention rationale.
- `crates/larql-models/docs/architecture-trait.md` — `ModelArchitecture` trait reference.
- ADR-017 (Shader retention under model agnosticity) — governs deletions; this ADR governs additions and dispatch routing.
