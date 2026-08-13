# Metal kernel capability table (Phase B ground truth)

Status: **audit complete, dispatcher not yet changed.** 2026-08-09.

This document is the authority the Phase B dispatcher work will consume. It
records, per kernel entry point: the formats it decodes, its reduction unit,
its legal dimensions (and whether they are asserted or assumed), tail
handling, auxiliary buffers, dispatch geometry, and what selects it. Route
tables and a wiring-status inventory follow, then ranked findings.

Provenance: six parallel per-family audits over every `.rs`-embedded MSL
shader in `crates/larql-compute-metal/src/shaders/` plus all dispatch sites.
Every ranked finding carries an explicit status:

- **VERIFIED** — re-checked by hand against the source before ranking.
- **AUDIT-FINDING** — reported by the auditing pass with file:line
  citations, not independently re-checked. **Re-cite before acting on or
  quoting one as fact.**

Column key
- **red.** — reduction unit: `sg` = one simdgroup per row + `simd_sum`;
  `tg` = threadgroup-cooperative; `thr` = per-thread, no reduction;
  `tile` = 2-D threadgroup tile.
- **r/TG, thr/TG** — rows per threadgroup, threads per threadgroup.
- **legal K** — inner-dimension constraint. `a` = asserted host-side,
  `s` = silently assumed (truncating division or unchecked scratch bound).
- **wired** — `prod` (production dispatch), `diag` (bench/diag only),
  `dead` (no dispatch site at all), `opt-in` (env-gated off by default).

## 1. Quant matvec family

| kernel | formats (W · X) | red. | r/TG·thr/TG | legal K | tail | wired |
|---|---|---|---|---|---|---|
| `q4k_matvec` | Q4_K 144B · f32 | sg | 4·128 | %256 s | rows masked; K tail dropped | prod (opt-out variant) |
| `q4k_matvec_8sg` | Q4_K 144B · f32 | sg | 8·256 | %256 s | same | **prod default** |
| `q4k_matvec_stride32` | Q4_K 144B · f32 | sg | 8·256 | %256 **a** (`trait_impl/matmul.rs:473`) | host rejects bad K | prod (vindex lm-head knn) |
| `q4k_matmul` | Q4_K 144B · f32 [M,K] | sg | 4r×4c·128 | %256 s | best row/M tails of the set | plumbing dead: `Pipelines.q4k_matmul` never read |
| `q6k_matvec` | Q6_K 210B planar · f32 | sg | 4·128 | %256 s | rows masked; K dropped | prod default |
| `q6k_matvec_8sg` | Q6_K 210B planar · f32 | sg | 8·256 | %256 s | same | opt-in `LARQL_Q6K_8SG=1` |
| `mxfp4_matvec` | MXFP4 16B/32 + ext e8m0 scales · f32 | sg | 4·128 | %32 **a** | host rejects bad K | no non-test caller (K1 rung) |
| `q4_matvec_v4` | Q4_0 18B · **Q8 int8 + ext scales** | tg stage + sg | 8·256 | %32 s; **K ≤ 8192 hard (tg scratch), unchecked** | rows masked after barrier (safe) | prod (Q4_0 route) |
| `q4_vecmat` | Q4_0 18B · f32 (row-scatter) | thr | — | %32 s; **OOB read on misaligned K** | out tail masked | prod (`q4_vecmat` trait) |
| `q4_sparse_matvec` | Q4_0 18B · Q8 + ext scales, row-gather | thr | — | %32 s; **indices unbounded → OOB** | tid masked | **dead** (no pipeline in src/) |
| `q4_f32_matvec` | Q4_0 18B · f32 | thr | — | %32 s | rows masked; K dropped | prod (Q4_0 FFN down) |
| `q8_matvec` | **Q8_0 split: int8 rows + ext f32 weight scales** · Q8 + ext scales | tg stage + sg | 8·256 | %32 s; **K ≤ 8192 hard, unchecked** | rows masked after barrier | prod (O-proj Q8_0 arm) |

Notes
- `q8_matvec` is the only matvec with an **external weight-scale buffer**
  (buffer 2) — the reason Q8_0 is `ScaleStorage::External` in the Phase A
  contract.
- Geometry is handle-driven at every site **except**: the `q6k_matvec`
  trait dispatch hardcodes 4sg constants while the bound pipeline alias can
  resolve to 8sg under `LARQL_Q6K_8SG=1` (finding F5), and `q8_matvec`'s two
  production sites hardcode `8`/`256` (currently matching).

## 2. QKV / projection family

| kernel | formats (W · X) | red. | r/TG·thr/TG | legal K | wired |
|---|---|---|---|---|---|
| `q4k_qkv_proj` | Q4_K×3 · f32 | sg | 8·256 | %256 s | prod (UniformQ4K) |
| `q4k_proj` | Q4_K · f32 | sg | 8·256 | %256 s | **dead** (registered, never dispatched) |
| `q4k_q6k_qkv_proj` | Q4_K Q/K + Q6_K V · f32 | sg | 4·128 | %256 s (shared stride deriv.) | prod (MixedQ4kQ6kV, decode only) |
| `q4k_q6k_qkv_proj_normed` | same + inline RMS | tg then sg | 4·128 | %256 s | opt-in `LARQL_QKV_FUSED=1` |
| `q4kf_qkv_proj` | "Q4_KF" (**144B in shader**) · f32 | sg | 4·64 (2r×2sg) | %256 s | prod (UniformQ4Kf; synthetic-only in practice) |
| `q4kf_proj` | same | sg | 4·64 | %256 s | prod (PerProjection Q4_KF arm; hybrid O-proj) |
| `q8_qkv_proj` | int8 rows + 3 ext weight-scale bufs · Q8 + ext scales | tg stage + sg | 8·256 | %32 s; **K ≤ 8192 hard**; **early-return-before-barrier UB when rows%8≠0** | prod (fused Q8 route) |
| `q8_proj_rope` | int8 + ext scales (no RoPE despite name) | sg | 8·256 | same hazards | **dead** (no pipeline field) |
| `quantize_q8` | f32 → int8 + per-32 ext scales | thr | — | %32 s; **host div_ceil vs shader trunc → unwritten tail scale** | prod (Q8 input path) |
| `qk_norm` / `qk_norm_qk` | f32 norm | tg tree ≤512 | 1 head/TG | tg_w pow2 (host-ensured) | prod (unfused chain) |
| `qk_norm_rope_fused` | f32 norm + RoPE | tg tree | 1 head/TG | tg_w pow2; odd rdim drops last elem | **prod default** |
| `rope_at_pos` / `rope_at_pos_batched_qk` | f32 in-place | thr | — | — | prod |
| `rope_apply`, `rope_at_pos_batched` | f32 | thr | — | — | **dead / test-only** |
| `v_norm` (scalar) | f32, param-free | **thr, O(len²)** | — | **aliased in-place race (unfixed twin of batched fix)** | prod (hybrid path only) |
| `v_norm_batched` | f32 | tg tree ≤512 | 1 head/TG | safe aliasing | prod (main decode) |

## 3. Attention family

| kernel | role | softmax | scratch | hard limits | sinks | softcap | wired |
|---|---|---|---|---|---|---|---|
| `attn_fused` | decode, full fusion (norm+RoPE+append+attend) | 2-pass | tg_q/tg_k/tg_red/tg_scores ≈6KB; fenced (F11) | head≤256 a, span≤1024 a; GQA div s | yes | yes (F8) | opt-in `LARQL_FUSED_ATTN=1` |
| `kv_append_attend_fused` | decode default (append+attend) | 2-pass | fenced (reference impl) | head≤256 a, span≤1024 a; GQA s | yes | yes (F8) | **prod default** |
| `kv_attention` / `_long` | decode fallback attend | 2-pass | fenced | span ≤1024 / ≤4096 a (F14) | yes (F7) | yes (F8) | prod (fallback + KV-shared) |
| `kv_cache_append` | flat copy | — | — | — | — | — | prod (unfused chain) |
| `fused_attention` | prefill, one TG per (head,pos) | 2-pass serial tid0 scan | ~19KB, no reuse | head≤512 a, seq≤4096 a (F14); TG hardcoded 256 both sides | yes | yes (clamped) | prod (always, prefill) |
| `causal_attention` | bench-only, single head, no GQA | 2-pass recompute O(seq²·d²) | none | — | no | no | bench (`full_layer_direct`) |

Decode fallback chain: `attn_fused` → `kv_append_attend_fused` →
shared-source attend-only → `kv_cache_append` + `kv_attention[_long]`.
KV cache is f32 position-major everywhere; there is no f16 KV path.

## 4. FFN family

| kernel | formats | act. fused | red. | r/TG·thr/TG | wired |
|---|---|---|---|---|---|
| `q4k_ffn_gate_up` | Q4_K×2 · f32 | none | sg | 4·128 | prod (opt-out variant) |
| `q4k_ffn_gate_up_8sg` | Q4_K×2 · f32 | none | sg | 8·256 | **prod default** |
| `q4k_ffn_gate_up_coop` | Q4_K×2 · f32 | none | tg+sg | 4·128 | opt-in; **divergent-barrier UB on odd K/256 or N%4≠0 [real shapes hit both]** |
| `q4k_ffn_gate_up_f16acc` | Q4_K×2 · f32→half dot | none | sg | 4·128 | opt-in; needs `LARQL_GATE_UP_8SG=0` too (alone = no-op) |
| `q4kf_ffn_gate_up` | 144B llama.cpp loop ×2 · f32 | none | sg | 4·64 | prod (Q4_KF gate) |
| `q4k_geglu_silu_down` | Q4_K down · gate+up f32 | SiLU | sg | 8·256 | prod (fused_down default on, inter≤16384) |
| `q4k_geglu_gelu_tanh_down` | Q4_K down | GELU-tanh **NO CLAMP** | sg | 8·256 | prod (same gate) |
| `q6k_geglu_silu_down` | Q6_K planar down | SiLU | tg stage + sg | 4·128 | **dead in prod** (routing requires GeluTanh) |
| `q6k_geglu_gelu_tanh_down` | Q6_K planar down | GELU-tanh **NO CLAMP** | tg+sg | 4·128 | opt-in `LARQL_FUSED_Q6K_DOWN=1` (recorded broken) |
| `q6k_geglu_gelu_tanh_down_cached` | Q6_K | GELU-tanh **NO CLAMP**; **caches nothing (verbatim copy)** | tg+sg | 4·128 | **dead** (flag dispatches the non-cached one) |
| `geglu_silu` / `geglu_gelu_tanh` | f32 elementwise | SiLU / GELU-tanh **clamped ±15** | thr | — | prod (the separated fallback) |
| `silu` / `gelu_tanh` | f32 elementwise, non-gated | clamped | thr | — | prod (Standard FFN) |

All gate+up and fused-down kernels: K %256 silently assumed, asserted only
in tests. `GeluExact`/`ReLU` panic loudly at both encoders (good).

## 5. MoE / experts family

Production MoE experts do **not** use the grouped kernels: the live path is
`q4k_ffn_gate_up` (all K experts as one tall matrix) → `geglu_gelu_tanh`
per expert → **per-expert `q4k_matvec` loop** for down. Routing (softmax,
top-K, weighted sum) is entirely CPU; expert selection is materialized as
weight-byte memcpys. No routing data reaches the GPU.

| kernel | status |
|---|---|
| `q4k_grouped_experts`, `q6k_grouped_experts` | pipelines built; **test/diag only** — the per-expert loop they were written to replace still runs |
| `mxfp4g_*` (7 arms) | diag tournament only; `_affine`/`_nox` are ceiling probes that **deliberately compute wrong answers** |
| `gate_knn_score`, `gate_knn_score_q8` | **fully dead**; q8 variant has a latent unaligned load |
| `turboquant_encode/decode_4bit` | **fully dead**; divergent-barrier UB + unset threadgroup binding + device-byte race if ever wired |

## 6. Norms / residual / GEMV family

| kernel | op | red. | wired |
|---|---|---|---|
| `rms_norm` | RMS ×(w+offset) | tg coop, 1 vec/TG, tg_p[8] | prod (everywhere) |
| `rms_norm_q8` | RMS + Q8 quantise | tg coop; **block-scale wrong unless tg%32=0 ∧ len%tg=0** | prod (Q8 input path) |
| `residual_add` | a + b_scale·b | thr | prod (universal fallback) |
| `residual_norm` | add+RMS, no b_scale | tg coop | **dead** (superseded by `_store`) |
| `residual_norm_q8` | add+RMS+Q8, b_scale | tg coop; same block-scale hazard | prod (non-kquant FFN) |
| `residual_norm_store` | add+RMS+raw store, b_scale | tg coop | prod (kquant FFN; D-RMS-FUSE) |
| `post_attn_residual_norm_store` | triple fusion (Gemma post-norms) | tg coop ×2 | **prod default**; **no b_scale and no residual_multiplier guard** |
| `post_ffn_norm_residual_add` | double fusion | tg coop | prod default (guarded); **PLE reuse bypasses the guard** |
| `scale_vector` | ×scalar in-place | thr | prod (layer_scalar) |
| `residual_copy` | copy | thr | **fully dead** (no pipeline) |
| `layer_norm` / `_no_bias` | LayerNorm ± bias | **thr, O(N²)** | prod — **input norm only, decode only** |
| `ple_gate_apply` | gelu(gate)·ple_input | thr (clamped) | prod (Gemma 4 E2B) |
| `f32_gemv` / `f16_gemv` | GEMV f32/f16 W | sg 8·256, K arbitrary (real tail loops) | prod (LM head, PLE) |
| `f32_argmax_partial` / `f32_topk_partial` | 2-phase argmax / top-8 | sg + tg fan-in, TG=256 fixed | prod (top-1/top-k LM head) |
| `sgemm` / `sgemm_transb` | GEMM 32×32 tile | tile, TG=(32,32) exact | prod (matmul trait, threshold-gated) |

## 7. Route tables (the three-plus mechanisms Phase B unifies)

### QKV — three disagreeing mechanisms

1. **Decode** (`pick_qkv_route`, `stages/qkv_proj.rs:61-68`):
   Q4_K×3→UniformQ4K; Q4_KF×3→UniformQ4Kf; (Q4_K,Q4_K,Q6_K)→Mixed;
   else PerProjection. Q8_0×3 is a *hand-written bypass* before the route
   table; if it ever reached PerProjection it would panic.
2. **Prefill** (`ops/full_pipeline/stages.rs:70-96`): `all_same_format`
   gate + two-arm match. **The mixed kernel is unreachable at prefill** —
   Gemma's Q4_K/Q4_K/Q6_K silently degrades to three per-proj dispatches.
3. **Hybrid** (`decode_hybrid.rs:145-149`) — **VERIFIED**: selects on
   `wq` **alone**. `wq=Q4_K, wv=Q6_K` (production Gemma shape) dispatches
   the uniform Q4_K kernel, striding V at 144B against 210B blocks.
   Production-reachable via `layer_graph/hybrid.rs:148`.

PerProjection arms: Q4_KF→`q4kf_proj` (fallback `q4k_matvec`);
Q4_K→`q4k_matvec`; Q6_K→`q6k_matvec`; Q4_0→`q4_matvec`+Q8 input;
**Q8_0→panic; I2S→panic; BF16/F16/F32→silent no-op** (no dispatch, stale
output buffer).

### FFN decode — branches on `gate.format()` only — **VERIFIED**

`Q4_KF` → q4kf chain; `is_kquant_family()` → Q4_K chain; else → Q4_0 chain.
Consequences: **Q6_K gate decoded as Q4_K (silent corruption); Q8_0 gate
decoded as Q4_0 (silent corruption); `up.format()` never read** (mixed
gate/up silently mis-decoded); float gate → Q4_0 decoder on float bytes.
Down format *is* honoured via `qmv::encode` — with the Q8_0/I2S panic and
float silent-no-op arms above. Prefill honours all three formats
independently (correct) but has no fused gate+up and near-no fused down.

Profile-split (`LARQL_PROFILE_SPLIT=1`) hardcodes 8sg gate+up (ignores
coop/4sg/f16acc flags) and drops both fused-down guards — split-mode
measurements are not the production configuration.

### O-projection (fixed in Phase A)

Dispatches on `wo.format()`: packed four → `o_proj::encode`; Q8_0 →
legacy `q8_matvec` (the one external-weight-scale consumer); anything else
panics loudly.

### MoE modes

`split_mode = moe_fn ∧ moe_collect_fn`; per-layer defer when layer has moe;
local CPU `cpu_moe_forward` only when no moe_fn and not remote. `has_moe`
is **model-level** but layer_scalar application is layer-level: dense
layers of a hybrid MoE model **never get their layer_scalar applied**
(`decode/mod.rs:789` vs `:832-841`).

### Norms

`norm_type` is consulted **only** for the decode input norm. Post-attn,
pre-FFN, post-FFN and the whole prefill path are unconditionally RMS — a
LayerNorm arch gets LayerNorm at layer input and RMS everywhere else, no
guard. `post_attn_norm_bias` is never bound by any dispatch site.

## 8. Ranked findings

Correctness, live path:
- **F1. Fused GELU-tanh down kernels lack the ±15 clamp** their separated
  twins carry (Apple tanh → NaN for |y|≳44). Mechanically explains both
  recorded fused-path NaN incidents. One line per kernel (×3). **VERIFIED**
- **F2. Hybrid QKV selects on `wq` alone** — Gemma Q4_K/Q6_K-V corrupted
  on a production-reachable path. Same defect class Phase A fixed for the
  O-projection. **VERIFIED**
- **F3. FFN decode branch misroutes Q6_K and Q8_0 gates** into wrong
  decoders, silently; `up.format()` never read. **VERIFIED**
- **F4. Float weights (BF16/F16/F32) in a matvec position = silent no-op**
  with stale scratch as output — worse than the panic arms. **AUDIT-FINDING**
- **F5. `q6k_matvec` trait dispatch hardcodes 4sg geometry** while the
  pipeline alias can be 8sg → half the rows unwritten under
  `LARQL_Q6K_8SG=1`. **AUDIT-FINDING**
- **F6. `quantize_q8` host `div_ceil` vs shader truncation** — unwritten
  tail scale read downstream when K%32≠0. **AUDIT-FINDING**
- **F7. Attention sinks silently dropped** on every non-fused decode
  fallback (span>1024, head>256, KV-shared): different softmax
  denominator, no guard. Softcap exists only at prefill (F8) — a
  capped arch decodes uncapped. **FIXED (slice 3)**: sinks and softcap
  travel on `FullPipelineLayer` (`attn_sinks` / `attn_softcap`, set at
  the arch boundary) and every decode attention kernel — fused and
  fallback — applies both, parity-pinned against the CPU reference
  with a cap-actually-applied control.
- **F9. Dense layers of hybrid-MoE models never get `layer_scalar`.** **AUDIT-FINDING**
- **F10. MoE staging loop missing `valid_count >= top_k` guard** (its
  sibling has one) — release-mode buffer overflow on excess indices. **AUDIT-FINDING**

Latent (opt-in paths, unshipped shapes, or UB not yet observed):
- **F11. `attn_fused` tg_red reuse unfenced** — the exact race class fixed
  in its two siblings; masked by the kernel being opt-in. **AUDIT-FINDING**
- **F12. `q4k_ffn_gate_up_coop` divergent barriers** on odd K/256 and
  N%4≠0 — both occur in shipped shapes; kernel is opt-in. **AUDIT-FINDING**
- **F13. `q8_qkv_proj`/`q8_proj_rope` early-return before barrier** when
  rows%8≠0; plus the K≤8192 threadgroup-scratch ceiling shared with
  `q4_matvec_v4`/`q8_matvec` — unchecked everywhere. **AUDIT-FINDING**
- **F14. `fused_attention` unasserted head≤512 / seq≤4096** (scratch
  indexed by absolute position — windows don't shrink footprint);
  `kv_attention_long` unguarded past 4096 (alloc coincidence). **AUDIT-FINDING**
- **F15. Q4_KF host constant 160B vs shader 144B** — `packed_block_layout`
  disagrees with both Q4_KF shaders by 16B/superblock. **AUDIT-FINDING**
- **F16. `gate_out`/`up_out` sized `inter` but fused Q4_K down reads
  `inter_padded`** — OOB read for any inter%256≠0 (all shipped shapes
  aligned, so latent). Plus no single `inter` vs `inter_padded` convention
  across the five down dispatch variants. **AUDIT-FINDING**
- **F17. K-alignment validation is the exception**: 2 of ~30 quant kernels
  check it; the rest silently truncate. GQA divisibility asserted nowhere. **AUDIT-FINDING**
- **F18. Granite guard asymmetry**: `post_attn_residual_norm_store` and
  the PLE reuse of `post_ffn_norm_residual_add` have no
  `residual_multiplier == 1.0` guard; their siblings do. **AUDIT-FINDING**
- **F19. `q8_qkv_proj` prefill site over-dispatches 8×**
  (`total_rows` TGs instead of `div_ceil(total_rows, 8)`). **AUDIT-FINDING**
- **F20. `append_and_attend` legacy API can overflow `tg_scores[1024]`**
  (passes `None` for the long pipeline, hardcodes window 0). **AUDIT-FINDING**

Hygiene / inventory:
- **F21. Dead-but-compiled set**: `q4k_proj`, `q8_proj_rope`,
  `residual_copy`, `residual_norm`, `q4_sparse_matvec`,
  `rope_apply`, `rope_at_pos_batched`, `q6k_geglu_silu_down`,
  `q6k_..._cached` (which also caches nothing, contradicting its 38-line
  doc), both `gate_knn_score*`, both `turboquant_*`, all 7 `mxfp4g_*`
  arms (2 of which deliberately compute wrong answers), grouped-experts
  pair, `q4k_matmul` router plumbing. All reachable by name in one Metal
  library — the pipeline-selection hazard `shaders/mod.rs:9-16` warns
  about.
- **F22. Doc/code mismatches**: `LARQL_FUSED_Q6K_DOWN` doc names the
  cached kernel, dispatches the other; `LARQL_F16_ACC` co-requirement
  undocumented; `q4k_matvec_stride32` "not wired" doc is stale;
  `Q4K_GU_MAX_K` test comment references a removed constant.

## 9. What the Phase B dispatcher consumes

The separations Phase A established — representation → auxiliary storage →
executable kernel geometry — mean the capability authority for a kernel is:

```
kernel:      MSL name + KernelHandle (geometry travels with the pipeline)
formats:     (weight format[s], input encoding) — exact, no family tests
legal dims:  K alignment, K ceiling, row constraints, GQA divisibility —
             checked at dispatch, not assumed
aux:         binding table incl. external scale operands (Phase A's
             QuantAux answers "does one exist"; the table answers "which
             index wants it")
features:    sinks / softcap / window / norm-type support — a fallback
             may not silently drop one
fallback:    the chain, with the invariant that a fallback preserves the
             feature set or refuses loudly
```

The route tables in §7 are today's behaviour, not the target: the target
is one table the QKV/FFN/O-proj/attention dispatchers all consult, with
`is_kquant_family()`-style family tests replaced by per-operand format
rows, and every `s` (silently assumed) entry in §1–§6 either promoted to
an `a` (asserted) or encoded as a capability bound the dispatcher checks
before selecting the kernel.
