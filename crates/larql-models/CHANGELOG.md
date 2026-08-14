# Changelog — larql-models

All notable changes to `larql-models` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/) conventions
with dated entries (`YYYY-MM-DD`) instead of semantic versions during the
pre-1.0 phase. Forward-looking work lives in [`ROADMAP.md`](ROADMAP.md).

Entries were migrated from `ROADMAP.md` on 2026-08-04 and preserve the date and
voice they were originally written in.

## [2026-05-28] — Hardening findings from the whole-codebase review

From the whole-codebase review ([`docs/audits/codebase-review-2026-05-28.md`](../../docs/audits/codebase-review-2026-05-28.md)): exceptionally hardened — **no reachable panics found**. Two cosmetic notes only:

- Unverified TQ1_0 codec path (correctness note, not a reachable defect today).
- Size-truncation note in loading. Plus the positional QKVO `attn_data[1]/[2]` convention shared with larql-kv (typed-contract candidate).

## [2026-05-25] — Multi-modal Phase 2 (PR #144)

- **architectures/granite.rs**: `GraniteVisionMultiModal` protocol —
  SigLIP2 encoder, `PerTile{729}` budget, AnyRes tile counts `[1..6]`,
  `precomputed_scaling = None`. Gated by `has_vision_config` on
  `ModelConfig` (set by parser from `config.json` `vision_config` presence).
- **encoders/vision_tower.rs**: `VisionConfig` extended with `hidden_act`
  (default `gelu_pytorch_tanh`) and `norm_type` (default `layer_norm`)
  for SigLIP2 parametrization. `is_siglip2()` helper.
- **connectors/mlp_connector.rs**: `MlpConnectorWeights` (fc1/fc2 weight
  + bias), `load_mlp_connector_from_safetensors`. 2-layer MLP GELU
  connector for Granite Vision.
- **config.rs**: `has_vision_config: bool` field on `ModelConfig`.
- **detect/parser.rs**: sets `has_vision_config` from
  `config.get("vision_config").is_some()`.

## [2026-05-24] — Multi-modal Phase 1 (PR #143)

- **multimodal.rs**: `ModalEncoder`, `Connector`, `MultiModalProtocol`,
  `PlaceholderProtocol`, `TokenBudget` (Fixed / PerTile / Dynamic),
  `PrecomputedScaling`, `Modality`, `ModalInput`. `ModelArchitecture`
  gains `multimodal()` default None; Gemma3Arch overrides with verified
  token IDs.
- **encoders/vision_tower.rs**: `VisionConfig` (generic, parsed from
  `vision_config` in config.json), `VisionWeights`, `VisionLayerWeights`,
  `ProjWithBias`, `LayerNormWeights`, `load_vision_tower_from_safetensors`.
  Arch-agnostic: same struct works for SigLIP, SigLIP2, ViT.
- **connectors/projector.rs**: `ProjectorWeights`,
  `load_projector_from_safetensors`. Loads `multi_modal_projector.*`
  tensors (projection matrix + optional norm weight).
- Design doc: `docs/multi-modal.md`. ADR: `docs/adr/0023-multimodal-engine-seam.md`.

## [2026-05-16] — Config-loading correctness pass

Cross-engine Shannon verify (`larql shannon verify`) on Linux + macOS
revealed four config-loading defects in `larql-models` that drove the
LARQL Rust forward path off by 5.4 % (Gemma 3 4B) to 8.2 % (Mistral 7B)
bits/char relative to HF transformers. All four are now fixed in the
loader itself — env-var diagnostics stay in tree but production runs
need zero overrides:

| # | Bug | Models affected | Fix site |
|--:|---|---|---|
| 1 | `rms_norm_eps` from config.json was never read; trait default 1e-6 used everywhere | Mistral 7B, Llama 3.2, Gemma 3 4B | `parser.rs` parses `rms_norm_eps` / `layer_norm_eps` / `layer_norm_epsilon` / `norm_epsilon` (StarCoder2) into `ModelConfig.norm_eps`; default `norm_eps()` reads it |
| 2 | Per-layer-type `rope_scaling` (Gemma 3 structured `{full_attention, sliding_attention}` form) was not honoured | Gemma 3 4B | `RopeScaling.gemma3_global_only`; `Gemma3Arch::rope_position_divisor_for_layer` returns `factor` on full-attention layers only |
| 3 | `rope_scaling = llama3` (wavelength-dependent per-channel factors) was not implemented | Llama 3.2 1B | New `Llama3RopeScaling` type in `config.rs`; `LlamaArch::llama3_rope_scaling` returns parsed params; `attention/rope.rs::Llama3Scaling::apply` mirrors HF's `_compute_llama3_parameters` |
| 4 | `norm_epsilon` alias not recognised (StarCoder2's name for `rms_norm_eps`) | StarCoder2 3B | Added to the alias list in `parser.rs` |

Plus GPT-2 config aliases (`n_embd` / `n_layer` / `n_head` / `n_inner`)
parsed via the new alias-list machinery in
`detect/config_io.rs::CONFIG_KEY_*_ALIASES`. Loader path now resolves
`openai-community/gpt2`; raw-safetensors tensor-key renaming
(`wte`/`wpe`/`c_attn` → canonical) is a separate scope kept for the
GPT-2 safetensors loader item below.

Shared numerical defaults moved to a new `defaults` module
(`DEFAULT_NORM_EPS`, `ROPE_BASE_GEMMA`, `ROPE_BASE_DEFAULT`) so the
parser fallback, the trait default, and the per-arch fallback all
reference the same value — the drift between these three sites was
the mechanism of bug 1.

Verification:
`scripts/diagnose_models.py` (multi-arch sweep across SmolLM2-135M,
Llama-3.2-1B, Qwen3-0.6B, Gemma-2-2B, StarCoder2-3B, Mistral-7B-v0.1,
Gemma-3-4B-it) reports 7/9 PASS at <0.5 % threshold with **zero env
vars set**. The two ERR rows are pre-existing issues unrelated to this
work (Granite-4.0-micro MoE validator strictness; Gemma-4 not yet
supported by HF transformers in `.venv`).

CI gate at `.github/workflows/shannon-verify.yml` runs
`larql shannon verify HuggingFaceTB/SmolLM2-135M --engines hf` on
every PR + push to main.

Diagnostic doc:
[`docs/diagnoses/shannon-cross-engine-divergence.md`](../../docs/diagnoses/shannon-cross-engine-divergence.md).

## [2026-04-26 → 2026-05-07] — Quality pass and its two follow-ups

The 2026-04-26 quality pass closed the known P0 items for `larql-models`: walk-only filtering, silent dtype reporting, quant test gaps, loader string constants, MXFP4 consolidation, config validation adoption, clippy, examples, benchmark coverage, and coverage refresh are complete. The 2026-04-30 follow-up fixed packed BF16 expert ownership, GGUF matrix layout/config-default handling, and refreshed coverage to the current baseline.

The 2026-05-07 follow-up fixed small-vocab GGUF handling, explicit embedding
orientation, missing GGUF attention metadata defaults, and checked mmap packed
byte ranges. It also added targeted regression coverage and refreshed CI to
run rustfmt plus a crate-scoped coverage summary.

## Completed milestones (migrated table)

This table came across from `ROADMAP.md` unchanged. It predates the dated-entry
convention above and overlaps the entry immediately preceding it; it is kept
because it is the only per-item record of the 2026-03 foundation work.

| Item | Date | Impact |
|------|------|--------|
| ModelArchitecture trait | 2026-03 | Foundation — 83 methods with defaults |
| Gemma 2/3 support | 2026-03 | QK-norm, softcapping, sliding window |
| Llama/Mistral/Qwen/DeepSeek | 2026-03 | Core architecture coverage |
| Mixtral MoE (PerExpert) | 2026-03 | Expert key patterns |
| GPT-OSS (PackedMxfp4) | 2026-03 | MXFP4 dequantization, packed expert keys |
| Granite (scaling multipliers) | 2026-03 | Embedding/residual/attention/logits scaling |
| StarCoder2 | 2026-03 | LayerNorm, bias, GELU |
| GGUF loading | 2026-03 | Q4_0/Q4_1/Q8_0/F16/BF16 dequantization |
| Safetensors mmap + HF cache | 2026-03 | Zero-copy loading, cache resolution |
| drop_ffn_weights | 2026-04 | Walk-only mode saves ~13GB |
| Gemma 4 architecture | 2026-04 | Per-layer geometry, PLE, KV sharing, V-norm, layer scalars |
| Gemma 4 31B + E2B configs | 2026-04 | Both variants tested with real config.json |
| Gemma4Arch re-export | 2026-04-07 | Public API complete |
| v_shares_k from config | 2026-04-07 | Uses attention_k_eq_v flag instead of hardcoded false |
| Gemma 3 qk_norm_weight_offset | 2026-04-07 | Was missing (Gemma 2 had it, Gemma 3 didn't) |
| Architecture coverage milestone | 2026-04-07 | All 12 architectures tested: Gemma 2/3/4, Llama, Mistral, Mixtral, Qwen, DeepSeek, GPT-OSS, Granite, StarCoder2, Generic |
| GGML quant test gaps closed (51 tests) | 2026-04-26 | q4k_row_dot NEON≡scalar, q4k/q6k scaled_add correctness, Q4_K known nonzero values |
| Silent dtype skip fixed | 2026-04-26 | `skipped_tensors` field on ModelWeights; UnsupportedDtype collected, other errors bubbled |
| normalize_key_pub removed | 2026-04-26 | Dead wrapper gone; `normalize_key` is `pub(crate)` |
| Config alias constants | 2026-04-26 | `NUM_EXPERTS_KEYS`, `NUM_EXPERTS_PER_TOK_KEYS`, `field_u64` helper in `detect.rs` |
| MXFP4 consolidation | 2026-04-26 | `split_gate_up_experts` in `quant/mxfp4.rs`; loader thinned + renamed |
| Walk-only loader fixes | 2026-04-26 | GGUF filtering, GPT-OSS MXFP4 predicate-aware expansion, StarCoder2 c_fc/c_proj classification |
| Loader magic-string cleanup | 2026-04-26 | Centralized GGUF metadata/key rewrites, MXFP4 suffixes, HF cache path fragments, packed expert keys |
| Config validation | 2026-04-26 | `ModelArchitecture::validate()` with centralized diagnostic fields; catches dimensions, head geometry, RoPE values, per-layer metadata, KV sharing, and MoE inconsistencies |
| Validation adoption in larql-models APIs | 2026-04-26 | Added `detect_*_validated`, `load_model_dir*_validated`, and `load_gguf_validated` while preserving permissive inspection APIs |
| Detection hardening for invalid configs | 2026-04-26 | Malformed zero-head configs and short Gemma 4 `layer_types` no longer panic before validation |
| Lazy/quantized weight API design | 2026-04-26 | ADR-008 defines additive `LazyModelWeights` and `QuantizedModelWeights` direction for larger loading work |
| Coverage baseline refresh | 2026-04-26 | 274 tests; 88.02% line / 86.29% function coverage |
| Clippy clean (zero warnings) | 2026-04-26 | lib + examples + tests all pass `-D warnings` |
| Criterion benchmark suite | 2026-04-26 | `cargo bench -p larql-models --bench models` covers detection, validation, key mapping, FFN classification, synthetic loading, and GGML dequant |
| Documentation refresh | 2026-04-26 | README, roadmap, performance notes, loading/quant docs, and ADRs updated for validation and current metrics |
| Example suite (3 demos) | 2026-04-07 | architecture_demo (all 12), demo_tensor_keys (all 12), demo_loading |
| Packed BF16 mmap retention | 2026-04-30 | Gemma 4 A4B packed BF16 expert tensors are retained as mmap byte ranges instead of heap-cloned raw bytes |
| GGUF loader correctness fixes | 2026-04-30 | 2D tensors load as standard `[rows, cols]`; absent optional RoPE/vocab metadata falls back through architecture/tokenizer defaults |
| Coverage baseline refresh | 2026-04-30 | 282 tests; 81.41% line / 82.06% function coverage |
| GGUF loader regression fixes | 2026-05-07 | Small vocab metadata, shape-derived vocab fallback, missing KV/head-dim defaults, checked packed mmap ranges |
| Coverage baseline refresh | 2026-05-07 | 286 tests; 77.86% line / 78.30% function coverage |
