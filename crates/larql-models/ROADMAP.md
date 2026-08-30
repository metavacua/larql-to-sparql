# Roadmap — larql-models

For shipped work, see [CHANGELOG.md](CHANGELOG.md).

## Current state (verified 2026-08-04)

**Architectures — 17 modules** in `src/architectures/`: `bitnet`, `deepseek`,
`deepseek_v4`, `gemma2`, `gemma3`, `gemma4`, `generic`, `gpt2`, `gpt_oss`,
`granite`, `llama`, `mistral`, `mixtral`, `olmoe`, `qwen`, `starcoder2`,
`tinymodel`. (The long-standing "12 architectures" figure in this file dated
from the 2026-04-07 coverage milestone and was five behind.)

**Loading.** Safetensors mmap + HF cache resolution, and GGUF with
Q4_0/Q4_1/Q8_0/F16/BF16 dequantization. Config parsing is alias-driven
(`detect/config_io.rs::CONFIG_KEY_*_ALIASES`) so `rope_scaling`, `norm_eps`,
and the GPT-2 legacy names (`n_embd`/`n_layer`/`n_head`/`n_inner`) all resolve
without per-arch special cases. Shared numerical defaults live in one
`defaults` module (`DEFAULT_NORM_EPS`, `ROPE_BASE_GEMMA`, `ROPE_BASE_DEFAULT`)
— drift between the parser fallback, the trait default, and the per-arch
fallback was the mechanism of a real bits/char divergence, so keep them
single-sourced.

**Multi-modal.** Trait surface (`multimodal.rs`), generic vision tower
(`encoders/vision_tower.rs`, arch-agnostic across SigLIP / SigLIP2 / ViT), and
two connectors (`connectors/projector.rs`, `connectors/mlp_connector.rs`).
Gemma 3 and Granite Vision are the two wired protocols.

**Validation.** `ModelArchitecture::validate()` plus the parallel
`detect_*_validated` / `load_model_dir*_validated` / `load_gguf_validated`
APIs; the permissive inspection APIs are retained deliberately.

**Tests.** 620 tests, all passing (`cargo test -p larql-models`).

**Coverage.** Enforced: 90% per file, 94% on the included total, 88% on the
whole-crate total. `src/test_fixtures.rs` is excluded — it is `test-utils`
support code with no production callers, exercised transitively by four
downstream crates, and measuring it here always lands near 30%. Two debt
baselines remain (`loading/gguf.rs`, `loading/loading/safetensors/`), both about
1pp short and both needing MXFP4 / packed-BF16 fixture tests to clear. See the
`policy_note` in `coverage-policy.json`.

**Hardening posture.** The 2026-05-28 whole-codebase review found **no
reachable panics** in this crate — the strongest result of any crate in the
sweep. Two non-defect notes stand: the TQ1_0 codec path is unverified (its two
round-trip tests are `#[ignore]`'d because the internal codec pairing does not
yet round-trip — that needs a production fix), and the positional QKVO
`attn_data[1]/[2]` convention shared with `larql-kv` is a typed-contract
candidate.

---

## Open work

Recommended next sequence — items 1–3 confirmed still open on 2026-08-04
(`loading/loading/safetensors/` has no `wte`/`wpe`/`c_attn` mapping; no Phi module
exists):
- **GPT-2 raw-safetensors tensor-key renaming.** Config parses cleanly
  now; tensor loading needs the `wte` / `wpe` / `h.N.attn.c_attn` /
  `h.N.mlp.c_fc` → canonical mapping in `loading/loading/safetensors/` (the
  existing `gpt2.rs` arch assumes GGUF→HF normalisation has already run).
- **Granite-4 MoE validator relaxation** so `granite-4.0-micro` loads —
  the dense Granite-4 model carries hybrid MoE *flags* without expert
  tensors, which the current validator rejects.
- Add Phi-3 / Phi-4 architecture support. Low effort, exercises the
  validation path, expands coverage without changing the trait.
- Use validated loading/detection APIs at downstream inference/extraction boundaries.
- Defer large loading changes until after architecture coverage. ADR-008 defines the additive lazy/quantized weight API shape.

## P0: Code Quality

### Downstream validation rollout
**Effort**: Medium
**Status**: Not started

`larql-models` now exposes validated APIs. Update downstream inference, vindex extraction, CLI, and server entry points to use `detect_*_validated` or `load_*_validated` where invalid configs should fail fast.

### Deterministic HuggingFace cache resolution
**Effort**: Low
**Status**: Not started

`loading/loading/safetensors/::resolve_model_path` scans cached snapshot
directories and returns the first snapshot with safetensors. `read_dir` order
is not stable and the resolver ignores `refs/main`, so the same model ID can
resolve to an old or arbitrary cached revision. Prefer the commit recorded in
`models--.../refs/main` when no explicit revision is provided, then fall back
to a deterministic snapshot ordering.

### Architecture capability contracts
**Effort**: Medium  
**Status**: Not started

Detection currently says which family a config belongs to, but it does not
state which downstream surfaces are actually implemented for that family.
Add an explicit capability contract so extraction, vindex weight writing,
inference, trace, and prompt rendering can fail loudly instead of accepting an
architecture whose tensors are not consumed by the active path.

Immediate driver: DeepSeek is correctly detected as MoE + MLA and exposes
`mla_*` tensor keys, but vindex writers and inference paths currently consume
standard Q/K/V/O attention tensors only. Either implement the MLA extraction
and forward contract, or report it as unsupported at the boundary.

### Note on quant/dequant crate split
**Decision**: `larql-models/quant/` is **format deserialization** (GGUF/safetensors → f32). `larql-compute` has **compute operations** (quantized matvec, Metal shaders). The split is correct. The `f16_to_f32` copies in `larql-compute/cpu/ops/q4k_matvec.rs` and `q6k_matvec.rs` are intentional — CPU reference impls for Metal shader testing, isolated by design. `larql-compute` is dev-only dep; don't flip that direction.

## P1: Architecture Coverage

### Phi-3 / Phi-4
**Effort**: Low  
**Status**: Not started

Similar to Llama with some attention differences (partial RoPE, SuRoPE). Most trait defaults apply.

### Command R / Cohere
**Effort**: Medium  
**Status**: Not started

Different attention key pattern, different norm placement.

### Mamba / state-space models
**Effort**: Large  
**Status**: Research

Would require extending the trait beyond transformer assumptions (no attention keys, no KV cache). May warrant a separate trait hierarchy.

## P2: Loading Improvements

### Streaming safetensors loading
**Effort**: Medium  
**Status**: Not started

Current loader mmaps shards but eagerly converts retained dense tensors into f32 `ModelWeights`; packed BF16 expert tensors are already retained as mmap byte ranges. For 70B+ models, per-layer/lazy loading would reduce peak memory further. Already have mmap infrastructure — extend to lazy loading with `Arc<Mmap>` references and explicit tensor lifetimes.

Design direction: ADR-008 proposes additive `LazyModelWeights` / `load_model_dir_lazy(_validated)` APIs rather than overloading eager `ModelWeights`.

### GGUF quantized inference (skip dequant)
**Effort**: Large  
**Status**: Not started

Currently GGUF tensors are dequantized to f32 during loading. For Q4_K/Q6_K formats, keep data in quantized form and pass directly to `larql-compute` quantized kernels. Requires a `QuantizedWeights` variant alongside `ModelWeights`.

Design direction: ADR-008 proposes additive `QuantizedModelWeights` / `load_gguf_quantized(_validated)` APIs that preserve GGML type ids and byte ranges.

### MLX npz/safetensors hybrid
**Effort**: Low  
**Status**: Partial (MLX safetensors work, npz not yet)

Apple MLX models sometimes use `.npz` format. Add npz parsing alongside safetensors.

## P3: Trait Evolution

### Per-layer FFN type
**Effort**: Low  
**Status**: Not started

Some models (e.g., future MoE variants) may have different FFN types per layer (dense for early layers, MoE for later). Add `ffn_type_for_layer(layer)` method.

### Attention pattern abstraction
**Effort**: Medium  
**Status**: Research

Current sliding window is boolean per layer. Future models may have more complex patterns (local + global hybrid, dilated attention, prefix caching hints). Consider a richer `AttentionPattern` enum.
