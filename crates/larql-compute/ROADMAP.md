# Roadmap — larql-compute

For shipped work, see [CHANGELOG.md](CHANGELOG.md). Kernel-level performance
notes live in [PERFORMANCE.md](PERFORMANCE.md); ADRs in [docs/adr/](docs/adr/).

## Current state (verified 2026-08-04)

**Tests.** 871 lib tests passing (`cargo test -p larql-compute --lib`);
127 source files.

**Coverage.** Enforced: 90% per file, 95% crate total, 8 debt baselines.

**Layout.** `backend` (trait + dispatch), `async_compute_backend`, `attention`,
`cpu`, `ffn`, `forward`, `kquant_forward`, `kv_dispatch`, `pipeline`,
`quant_route`, plus the multi-modal `encoders` / `connectors` pair.
Metal-specific code lives in the sibling `larql-compute-metal` crate.

**The bandwidth ceiling that frames every open perf item.** Measured
2026-07-29 on M3 Max: the **CPU cluster attains 127 GB/s** (saturating at two
threads) against a **GPU 367 GB/s** — 31% vs 92% of the 400 GB/s SoC spec. That
2.9× structural advantage is not closable by any CPU kernel. Read the roofline
writeup before proposing a CPU-side bandwidth win:
[`docs/diagnoses/memory-bandwidth-roofline.md`](../../docs/diagnoses/memory-bandwidth-roofline.md).

**Performance figures are in the changelog, not here.** This section used to
carry a "Current state (2026-05-04, M3 Max, real vindex)" table that stayed
frozen for three months while the crate's newest content advanced to
2026-07-29. Those tables now live in [`CHANGELOG.md`](CHANGELOG.md) under their
measurement dates, where a reader can see how old they are. Two conclusions
from them are still load-bearing and worth restating:

- **Don't chase the historical "81–84 tok/s".** It was real wall-clock
  throughput on a model producing wrong logits — a dispatch-geometry bug left
  ~75% of output rows unwritten. Reverting to recover it re-introduces the bug.
- **Both big FFN kernels are bandwidth-bound near LPDDR5X peak** (q6k_matvec
  311 GB/s, q4k_ffn_gate_up 274 GB/s). The earlier "compute-bound at 103 GB/s"
  diagnosis was a profiler artifact — `measure_batched` created a fresh command
  buffer per call, undercounting throughput 2–4×.

---

## P0: Metal kernel capability audit findings (2026-08-09)

**PROGRAMME CLOSED 2026-08-09** — all 22 findings resolved across PRs
#231 (audit doc + F1–F4), #233 (guard batch), #234 (slice 1: one QKV
plan for decode/prefill/hybrid), #235 (F15 verdict: Q4_KF is a
kernel-route tag over 144-byte GGUF blocks; F16; F22), #236 (slice 2:
per-operand FFN everywhere incl. profile-split), #237 (slice 3: sinks
and softcap travel with every attention path). Only F17's per-kernel
tail and F21 (retention-gated) remain open below.

Performance: **neutral** — canonical baseline in
[`CHANGELOG.md`](CHANGELOG.md) 2026-08-09 (10 paired alternating
rounds on gemma4-26b-a4b: median paired Δ +0.54% decode p50, GPU
fraction unchanged at 95.2%).

Three kernels found broken-since-written along the way:
`q4k_ffn_gate_up_coop` (parity-mixed scale staging, cos 0.52 vs CPU —
fixed), the fused-Q6K-down **decode integration** (all-NaN while the
kernel passes isolation parity at the exact shape — hard-disabled,
`#[ignore]`d reproducer, open thread below), and the `attn_fused`
test runner (scalar bound into the inv_freq table slot — fixed).

**Open threads out of the programme:** the fused-Q6K-down integration
NaN root-cause; `norm_type` honoured only at the decode input norm
(LayerNorm archs silently get RMS for post-attn/pre-FFN/post-FFN and
all of prefill — no guard); `MAX_FUSED_GEGLU_DOWN_INTER` possibly
removable now the tanh clamp landed (untested); the bench's
EOS-vs-GPU-fallback ambiguity (instrument note below); production MoE
still routes on CPU with per-expert down loops (perf, deliberately
sequenced after dispatch authority).

Source of truth: [`docs/metal-kernel-capabilities.md`](../../docs/metal-kernel-capabilities.md)
— the Phase B per-kernel capability table (formats, reduction unit, legal
dims asserted-vs-assumed, aux buffers, dispatch geometry, fallback chain)
built from a six-family audit of every MSL entry point in
`larql-compute-metal`. Finding numbers below (F1–F22) are that document's.

**Fix checklist** (live correctness first):

- [x] **F1** (fixed c3688d38) clamp tanh arg ±15 in `q4k_geglu_gelu_tanh_down`,
      `q6k_geglu_gelu_tanh_down`, `q6k_..._cached` — matches `geglu.rs:39`;
      explains both recorded fused-path NaN incidents.
- [x] **F2** (fixed ac8168ec) hybrid QKV (`decode_hybrid.rs:145`) selects kernel on `wq`
      alone — route (Q4_K,Q4_K,Q6_K) to the mixed kernel, refuse other
      mismatches loudly.
- [x] **F3** (fixed 0f5a73e0) FFN decode branch keys on `gate.format()` family only —
      Q6_K gate mis-decoded as Q4_K, Q8_0 as Q4_0, `up.format()` never
      read. Route Q6_K via per-format matvecs; refuse Q8_0/float/mixed
      loudly.
- [x] **F4** (fixed bc2b429f) float weights (BF16/F16/F32) in `quant_matvec::encode` are a
      silent no-op with stale scratch output — panic loudly.
- [x] **F5** (fixed 4408d18a) `q6k_matvec` trait dispatch hardcodes 4sg geometry against a
      pipeline alias that can be 8sg — read the `KernelHandle`.
- [x] **F6** (fixed 4408d18a) `quantize_q8` host `div_ceil` vs shader truncating `K/32` —
      unwritten tail scale; make the kernel handle partial blocks.
- [x] **F7** (fixed, slice 3: feature-aware attention) sinks silently dropped by `kv_attention`/`_long` fallbacks —
      add sink buffers (join max + denominator) so fallback preserves the
      feature set.
- [x] **F8** (fixed, slice 3: feature-aware attention) softcap exists only in prefill `fused_attention` — decode
      refuses or supports; no silent cap-drop.
- [x] **F9** (fixed 4408d18a) dense layers of hybrid-MoE models never get `layer_scalar`
      (`decode/mod.rs:789` model-level branch vs layer-level apply).
- [x] **F10** (fixed 4408d18a) `gpu_moe_dispatch_with_scratch` staging loop missing
      `valid_count >= top_k` guard (sibling has it) — release-mode
      buffer overflow.
- [x] **F11** (fixed 4408d18a) `attn_fused` tg_red max→sum reuse unfenced + `tg_q` handoff
      fenced `mem_device`-only + ropes at occupancy not stream position.
- [x] **F12** (fixed 4408d18a) `q4k_ffn_gate_up_coop` divergent threadgroup barriers on odd
      `K/256` and `N%4 != 0` (both occur in shipped shapes). The fix's
      new parity test also found the kernel numerically broken outright
      (16-slot scale staging mixed superblock parities; cos 0.52 vs
      CPU) — rewritten with parity-indexed staging, pinned even+odd.
- [x] **F13** (fixed 4408d18a) `q8_qkv_proj`/`q8_proj_rope` early-return before barrier;
      K ≤ 8192 threadgroup-scratch ceiling (shared with `q4_matvec_v4`,
      `q8_matvec`) asserted nowhere.
- [x] **F14** (fixed 4408d18a) `fused_attention` head≤512 / seq≤4096 unasserted;
      `kv_attention_long` unguarded past its 4096 scratch.
- [x] **F15** (fixed on fix/metal-audit-f15-f16-f22) `Q4_KF_BLOCK_BYTES = 160` (host) vs 144 (both Q4_KF
      shaders) — establish which is real, fix the other.
- [x] **F16** (fixed on fix/metal-audit-f15-f16-f22) `gate_out`/`up_out` sized `inter` while fused Q4_K down
      reads `inter_padded`; unify the `inter` vs `inter_padded` K
      convention across the five down dispatch variants.
- [ ] **F17** (narrowed by the selector slices: `plan_qkv` asserts
      hidden%256 at the fused f32 QKV dispatch, GQA divisibility is
      asserted in `stages/attention.rs`, and the K≤8192 staging
      ceilings are asserted at every production site — remaining scope
      is the per-kernel K%256/%32 truncation checks in the matvec/FFN
      dispatchers) K-alignment asserted in 2 of ~30 quant kernels; GQA
      divisibility asserted nowhere — add dispatch-time checks.
- [x] **F18** (fixed 4408d18a) `residual_multiplier == 1.0` guard missing on
      `post_attn_residual_norm_store` selection and the PLE reuse of
      `post_ffn_norm_residual_add` (siblings have it).
- [x] **F19** (fixed 4408d18a) `encode_fused_q8` prefill site dispatches `total_rows` TGs
      instead of `div_ceil(total_rows, 8)` — 8× over-dispatch.
- [x] **F20** (fixed 4408d18a) legacy `append_and_attend` passes `None` for the long
      pipeline → `tg_scores[1024]` overflow past span 1024; hardcodes
      window 0.
- [ ] **F21** dead-but-compiled inventory (see capability doc §8) — per
      the shader-retention policy each needs a revival-story check before
      deletion, not a bulk purge; `q6k_..._cached` additionally caches
      nothing despite its module doc.
- [x] **F22** (fixed on fix/metal-audit-f15-f16-f22) doc/code mismatches: `LARQL_FUSED_Q6K_DOWN` names the wrong
      kernel, `LARQL_F16_ACC` co-requirement undocumented, stride32
      "not wired" doc stale, `Q4K_GU_MAX_K` test comment references a
      removed constant.

**New finding (2026-08-09, slice 2):** the fused Q6_K down decode
integration produces all-NaN output even though the kernel now passes
isolation parity at the exact decode shape (planar bytes, 512→256) —
the historical "layout drift" hypothesis is falsified; the defect is in
what the decode dispatch binds or sequences. `LARQL_FUSED_Q6K_DOWN` is
hard-disabled at both dispatch sites; reproducer test
`decode_token_with_q4k_ffn_and_fused_q6k_down_option` is `#[ignore]`d
with the finding. Root-cause before re-engaging.

**Instrument note (2026-08-09, post-Phase-B benchmarking):** `larql
bench` reports `early stop @N/50 (EOS or GPU fallback)` — it cannot
distinguish a legitimate EOS from a silent GPU fallback. For a harness
increasingly used as a correctness instrument that ambiguity is too
coarse: granite-4.1-8b stops at 4/50 on Metal on both pre- and
post-audit builds and the bench cannot say which of the two reasons
applies. Surface the stop reason explicitly (token id vs backend
fallback flag).

The structural target (capability doc §9): one authority table consumed by
the QKV/FFN/O-proj/attention dispatchers — per-operand format rows instead
of `is_kquant_family()` tests, every silently-assumed dim promoted to a
checked bound, feature sets (sinks/softcap/window/norm-type) that
fallbacks must preserve or refuse.

---

## Open defects

From the 2026-05-28 whole-codebase review
([`docs/audits/codebase-review-2026-05-28.md`](../../docs/audits/codebase-review-2026-05-28.md)).
Full text of the finding is in [`CHANGELOG.md`](CHANGELOG.md) 2026-05-28.

The MoE panic sites here are gated on the same fix as
`larql-inference`'s P0: `FfnBackend::forward` (declared in this crate at
`src/ffn.rs:39`) returns an infallible `Array2<f32>`, so implementations have
no way to report failure and must abort. Changing that signature is the
highest-leverage fix in the workspace — it unblocks both crates at once.

---

## Open: head-major K/V layout for decode attention

**The single highest-leverage change the `kvperf-1` data points at**, because
it sits *below* every KV engine rather than inside one. All six cached engines
share one marginal cost per context token (7.72-8.30 µs, a 7.5% spread across
six mechanisms that could hardly be more different) — they all end up in this
kernel, and it runs at roughly 44% of attainable memory bandwidth. See
[`larql-kv/docs/decode-cost-model.md`](../larql-kv/docs/decode-cost-model.md) §4.

Today K/V is `[L, num_kv * head_dim]` — head-**minor**:

```text
each head's gemv reads head_dim floats with a num_kv * head_dim stride
with GQA, the `reps` query heads sharing a KV head each stream it again
```

On qwen3-0.6b that is a 512 B window with a 4096 B stride, read twice
(`reps = 2`). **Hypothesis (untested):** `[kv_head][L][head_dim]` makes each
head's read contiguous *and* lets the sharing q-heads share one pass, roughly
halving the O(context) term that dominates every engine.

A bench exists — `benches/decode_attention_layout.rs` — comparing four layouts
doing identical arithmetic, so any difference is memory behaviour. It includes
a `head_major_spin` variant specifically so the parallel primitive is not
confounded with the layout (production uses the spin pool; a rayon candidate
would otherwise be compared on two axes at once).

**Measurement protocol, because this one has already bitten:** an earlier
version timed one layer against one reused K/V buffer. At L=1024 that is 8.4 MB
— cache-resident — and it clocked 90 µs against the ~301 µs the end-to-end
slope implies, reporting essentially the roofline. It was measuring a regime
decode never runs in, **and the layout ranking inverted between that regime and
the real one.** Each iteration now walks a full stack of distinct per-layer K/V
buffers, as one decode step does.

Two further conditions before believing a result:

- **AC power.** On battery macOS throttles enough to invent or erase this
  effect, and criterion runs the variants sequentially — deepening throttle
  mid-run penalises later variants and corrupts the *ranking*, which is the
  entire output.
- **A drift control.** Measure `head_minor` twice, first and last. If those
  two disagree the machine moved under the run and the ordering is not
  trustworthy regardless of what it says.

## Open: compute modularity and model-agnostic cleanup

**Status**: Started 2026-05-08.

Review focus: modularity, magic strings / magic numbers, and model-family
agnosticity in `larql-compute`.

Findings now tracked here so follow-up work does not live only in review notes:

- [x] Centralize CPU MoE env-var names and parsing in `src/options.rs`, with
  namespaced `LARQL_*` names and temporary compatibility for legacy
  `SKIP_MOE` / `MOE_DEBUG`.
- [x] Make unsupported CPU MoE dequant formats return a typed error internally
  instead of silently becoming an empty expert vector.
- [x] Add crate-specific fast and integration test targets:
  `make larql-compute-test-fast` now runs library/unit tests only for the
  inner loop, while `make larql-compute-test-integration` walks the heavier
  integration binaries explicitly.
- [x] Harden cross-platform Metal gates in compute, inference, vindex benches,
  and CLI call sites so non-macOS `--all-features` builds do not import the
  macOS-only `metal::` module.
- [x] Add `.github/workflows/larql-compute.yml` for Linux, Windows, and macOS
  fast compute checks, including non-macOS `--all-features` coverage and macOS
  `--features gpu` coverage.
- [x] Add compute coverage targets and a 90%-default per-file policy with
  current debt baselines in `crates/larql-compute/coverage-policy.json`.
- [x] Raise default-feature line coverage from 56.76% to 93.59% with targeted
  tests for backend default contracts, compute option parsing, Q4K kernels, and
  MoE helpers. Remaining per-file debt is now concentrated in
  `cpu/ops/moe/expert.rs` and `cpu/ops/moe/forward.rs`.
- [x] Replace remaining direct `std::env::var` reads in Metal decode/stage
  code with the same options surface, so env names and parsing live in
  `src/options.rs`.
- [x] Thread explicit options through backend construction for non-debug
  behavior, keeping env-backed debug toggles as a compatibility bridge.
  Shipped 2026-05-09 (M1): `BackendOptions` + `MetalBackend::with_options`,
  `default_backend_with_options` in lib.rs. Env reads still flow through
  `BackendOptions::from_env` for compatibility.
- [ ] Split `FullPipelineLayer` into smaller architecture and runtime specs.
  Compatibility views now exist for `AttentionSpec`, `FfnSpec`, `MoeSpec`,
  `LayerWeights`, `LayerNorms`, and remote-FFN settings; remaining work is to
  migrate inference/vindex/backend call sites off the flat field bag and then
  make the wrapper private or narrower.
  **Compute-side prep shipped 2026-05-09 (M2 partial)**: the four
  hottest decode entry points (`encode_input_norm_and_qkv`,
  `encode_q4k_qkv`, `encode_q4k_input_norm`, `encode_q4_0_norm_and_qkv`,
  `encode_attention_block`, `encode_post_ffn_residual`) now read
  `layer.<flat-field>` through `layer.weights().attention`,
  `layer.attention_spec()`, `layer.norms()` views. Pattern
  established: `let view = layer.method();` at the top of each
  function, then `view.field` throughout. Remaining work for vindex
  / inference call sites + final wrapper-narrowing is unchanged.
  Also fixed an additional Q8_0 hybrid-path dispatch-geometry hardcode
  in `encode_q4_0_norm_and_qkv` (same bug class as decode_hybrid Q4_K).
- [x] Promote Gemma-style MoE assumptions into an explicit `MoeRoutingPolicy`:
  router input source, router norm policy, selected-weight renormalization,
  per-expert scale policy, post-expert norm policy, softmax/top-k semantics,
  and expert down-padding layout.
- [x] Group `MetalBackend` kernel handles into cohesive registries
  (`AttentionKernels`, `FfnKernels`, `NormKernels`, `QuantKernels`) and pass a
  compact dispatch context into full-pipeline execution instead of large
  positional argument lists. Shipped 2026-05-09 (M3) — 57 fields moved off
  `MetalBackend` into the four registries; ~150 call sites updated. End-to-end
  tok/s within noise of pre-M3 baseline (86.1 vs ~83 tok/s on Gemma 3 4B
  Q4_K v2; the registry pattern is zero-cost — `&self.norms.foo` lowers to
  the same load as `&self.foo`).
- [x] Replace hard-coded QKV quant-format branches with a descriptor/table keyed
  by `(q_format, k_format, v_format)` so new model-format combinations do not
  add another inline conditional. Shipped 2026-05-09 (M4): `QkvFormatRoute`
  enum + `pick_qkv_route` table in `metal/stages/qkv_proj.rs`; encode_qkv +
  decode_hybrid both read the route through it.
- [x] Fix the trait-level `QuantMatVec` dispatch for `QuantFormat::Q8_0`.
  Shipped 2026-05-09 (M-Q8): split the `Q4_0 | Q8_0` arms in
  `backend/quant_matvec.rs::quant_matvec` and `quant_matvec_q8_input`; added
  a `q8_matvec` trait method (default `None`); the Metal-side
  `metal/stages/quant_matvec.rs::encode` panics loud on Q8_0 with a clear
  message pointing at `quant.q8_matvec_pipeline`. Production decode reaches
  Q8_0 weights via dedicated kernels (`q8_qkv_proj_pipeline` /
  `q8_matvec_pipeline`), so the dead generic dispatch was lying rather than
  load-bearing. 3 new unit tests pin the contract (Q4_0 still routes through
  `q4_matvec`, Q8_0 returns `None` not silent garbage, float-input formats
  stay `None`).
- [x] Align Metal FFN activation handling with public `pipeline::Activation`;
  unsupported activations should fail explicitly or use a known fallback.
  Shipped 2026-05-09 (M5): `metal_supports_activation` /
  `assert_metal_activation_supported` in `metal/stages/ffn.rs`; the three
  silent `_ => silu` fallback sites now panic clearly on `GeluExact` / `ReLU`.
- [x] Name and validate remaining format/kernel constants (`256`, `144`, `210`,
  `8192`, `16384`, debug layer modulo values) at the format or kernel-descriptor
  boundary instead of scattering them through CPU/Metal paths. Shipped
  2026-05-09 (M6): cpu/ops sites pull `Q4_K_BLOCK_BYTES` /
  `Q6_K_BLOCK_BYTES` / `Q4_K_BLOCK_ELEMS` from `larql_models::quant::ggml`;
  `MAX_FUSED_GEGLU_DOWN_INTER = 16384` named in `metal/decode/encode_ffn.rs`.

Acceptance: adding a new supported architecture or quant layout should require
declaring its layer specs and format routes, not editing Gemma-specific branches
inside hot-path execution code.

---



## Open: Metal MoE expert kernel — accuracy bug at inter=704

**Status**: Open as of 2026-04-30. Workaround in place (CPU experts default).

The Metal MoE expert dispatch produces numerically wrong outputs for
Gemma 4 26B-A4B-it's MoE shape (`inter=704`, `hidden=2816`, `top_k=8`).
Affects all three Metal entry points equally:

- `MetalBackend::gpu_moe_dispatch_with_scratch` (in-process MoE decode path)
- `MetalBackend::run_experts_preselected_metal` (server old path — byte-copy + one big dispatch)
- `MetalBackend::run_experts_prestaged_metal` (server new path — pre-cached per-expert buffers + per-expert dispatch)

Symptoms (vs CPU reference per `LARQL_METAL_VS_CPU_DEBUG=1` in
`larql-server::routes::expert::run_experts_metal_batch`):

| Layer | K | max\|Δ\| | \|metal\| | \|cpu\| | cos |
|-------|---|----------|-----------|---------|-----|
| L00   | 2 | 5.5e-2   | 0.011     | 0.015   | 0.72 |
| L02   | 6 | 5.6e+0   | 0.74      | 0.97    | 0.76 |
| L05   | 3 | 5.0e+0   | 0.29      | 0.35    | 0.81 |

Pattern: cos ≈ 0.7 every layer, |metal| ≈ 70% of |cpu|. Not just a scaling
bug (cos < 1.0 means direction is wrong too) but consistent across calls.
End-to-end output: `"What is the capital of France?"` → "answer is in the
context of France" via Metal vs "**Paris**" via CPU.

**Same shaders are correct for dense FFN.** `q4k_ffn_gate_up`,
`geglu_gelu_tanh`, `q4k_matvec` all pass per-layer parity at cos ≥ 0.9999
on Gemma 3 4B (inter=10240) and Gemma 4 31B dense (inter=21504). The bug
is specific to the MoE dispatch pattern at inter=704 — possibly the
small inter / unusual padding ratio (inter_padded=768, so 64 trailing
zeros per slot in act_buf), or something about the per-expert offset
math when N = K × inter is moderate and K > 1.

**Workaround** (`larql-server`): default to CPU expert dispatch even on
`--features metal-experts` builds. `LARQL_USE_METAL_EXPERTS=1` opts back
in for kernel-debug runs.

**To fix:**

1. Extend `larql parity --component moe-expert` with a `metal` backend
   (call `run_experts_preselected_metal` with K=1) so CPU vs Metal can be
   diffed for a single expert with synthetic input. Establishes whether
   the bug is single-expert or multi-expert.
2. If single-expert: bisect the kernel chain — gate-only → gate+act →
   gate+act+down — to localise which stage diverges.
3. If multi-expert only: investigate the `q4k_ffn_gate_up` dispatch when
   `n_rows = K × inter` for small inter; check that per-row weight pointer
   math doesn't lose precision or step into a tile-boundary edge case.

Once fixed, expect the gRPC grid to jump from 3.5 tok/s → ~9-11 tok/s
(measured during the bug investigation: server compute is 95% of token
time, Metal experts give 3-4× speedup vs CPU experts).

---

## Open: Per-layer backend shape contract

**Status**: Planned as of 2026-05-02.

`FullPipelineLayer` already carries per-layer attention geometry, RoPE, norms,
FFN type, and activation. The backend APIs should make that the only shape
contract. Several decode/prefill signatures still accept scalar
`num_q_heads`, `num_kv_heads`, `head_dim`, `q_dim`, `kv_dim`, and `rope_base`
values that are usually first-layer defaults. That creates a fallback path
where heterogeneous architectures can allocate uniform KV/cache state.

Work items:

- [ ] Replace uniform `create_kv_cache(num_layers, max_seq, num_kv_heads,
  head_dim)` fallbacks in decode paths with per-layer cache construction from
  `layers`.
- [ ] Introduce a compact decode shape/context struct, or derive all shape
  values inside the backend from `FullPipelineLayer`, to reduce parameter drift.
- [ ] Add tests covering mixed per-layer KV/head geometry without requiring
  caller-side preallocation.
- [ ] Keep scalar helpers only for legacy/uniform compatibility and mark them
  clearly as such.

Acceptance: callers should not need to know whether a model has uniform,
sliding/global, or otherwise heterogeneous attention geometry before invoking a
backend decode path.



## P0: Production gap closers

Remaining gap: **~1.17×** (~88 vs ~103 tok/s, ~1.66 ms/tok) post 2026-05-09
QKV defuse. Was ~1.18× pre-defuse, ~1.30× pre 2026-05-02 dispatch
geometry fix. The historical diagnosis below was on the pre-fix baseline —
kept for context. Acceptance criterion (`~85 tok/s, ~1.16×`) is effectively
met; remaining work is closing the last ~14% via the multi-TG `attn_fused`
retry (D-ATTN-MTG below) or accepting the M3 Max ceiling for matvec
without `simdgroup_matrix`.

### Open decode-side levers (post 2026-05-02)

| # | Lever | Estimated win | Status | File / approach |
|---|---|---|---|---|
| **D-ATTN-MTG** | Multi-TG `attn_fused` retry — preserve 12 TGs while fusing qk_norm_rope + kv_append + attend | 0.2–0.4 ms/tok within the 3.48 ms attention bucket | Open. First attempt regressed −1.45 ms because the merge collapsed TG count 12→8; the multi-TG-per-head variant (split QKV+attend across 2 TGs/head, total ≥12) is untried. ADR-015 § "Lesson — diagnostic order" applies. | `metal/shaders/attn_fused.rs` rewrite; gated on `LARQL_FUSED_ATTN=1` until verified |
| **D-FFN-PROFILE** | Split `encode_ffn` profiler boundary (gate_up vs activation+down) | Diagnostic, not perf. | **SHIPPED 2026-05-04.** `LARQL_PROFILE_SPLIT=1` + `--profile` bench now shows three separate GPU buckets per step. Measured on Gemma 3 4B (10-token steady state): **attn=3.3ms (34%), gate+up=3.5ms (36%), act+down=2.8ms (29%)** — all three roughly equal thirds. Gate+up is the largest single kernel. See `metal/decode/encode_ffn.rs` (split helpers) + `profile.rs` (GateUp/Down stages) + `bench_cmd.rs` (sub-rows). | `metal/decode/encode_ffn.rs` + `metal/decode/profile.rs` |
| **D-FFN-FUSE** | Q6_K geglu+down fusion with cheaper-activation variant | ~0.2 ms/tok | **BLOCKED — all-NaN bug with production weights.** Kernel passes unit parity tests (synthetic data, production geometry). On real vindex decode: `down_out` = all 2560 NaN even in a fresh encoder with valid gate/up inputs (max±12). Metal API validation reports no errors. Bug not found by static analysis. Possible cause: interaction between production Q6_K block values and the fused kernel's inner-loop accumulation. Needs Metal shader debugger. Wired behind `LARQL_FUSED_Q6K_DOWN=1` (opt-in, broken). | `metal/shaders/q6k_geglu_down.rs` + `encode_ffn.rs` |
| **D-PREFILL-MM** | Wire `q4k_matmul` into FFN gate/up/down + QKV (prefill only) | Was estimated 3–4× prefill speedup. **FALSIFIED twice end-to-end.** | **CLOSED 2026-05-09.** Wiring tried at O-proj (2026-04-28: −10% on long prompts), at FFN gate+up (2026-04-28: 2933 → 3268 ms on 340 tokens), and re-validated FFN gate+up under post-defuse state (2026-05-09: 5–7% regression at 10/50/150-token prompts). Diagnosis: the `q4k_matmul` kernel is bandwidth-bound; the [seq_len × hidden] X working set thrashes GPU L1 on long prompts and the dequant amortisation gain is paid back in DRAM↔L1 traffic. See **D-PREFILL-MM2** below for the kernel-rewrite track. The current `q4k_matmul` shader + `MetalBackend::q4k_matmul` method + parity tests stay shipped for re-validation on future hardware; production prefill stays per-position matvec. | (closed) |
| **D-PREFILL-MM2** | Rewrite `q4k_matmul` using Apple `simdgroup_matrix` intrinsics (mirrors llama.cpp's `kernel_mul_mm_*`) | Closes the 4–14× prefill gap to ollama (per [llama-cpp-comparison.md](docs/llama-cpp-comparison.md) §2). | **Open.** Confirmed via dylib-symbol diff that llama.cpp's prefill matmul uses `simdgroup_matrix` 8×8 register-tile fused-multiply-add — the same hardware feature that's load-bearing for their flash-attn. Our scalar-accumulator `q4k_matmul` can't hold the working set in registers on long prompts. M3 Max satisfies the `MTLGPUFamilyMetal3` device-family check; this is purely a kernel-implementation gap. **Multi-day kernel research project.** | New shader `metal/shaders/q4k_matmul_simdgroup.rs`; gated on `LARQL_PREFILL_MATMUL2=1` until verified; existing `q4k_matmul` retained as fallback |
| **D-RMS-FUSE** | RMS-norm pre-fusion with surrounding scalar mul/add (mirrors llama.cpp's `kernel_rms_norm_mul_f32` / `kernel_rms_norm_mul_add_f32`) | Predicted ~0.1-0.2 ms decode on non-Gemma. **Measured null end-to-end** (within drift). | **IMPLEMENTED + FALSIFIED 2026-05-09 (Phase 1).** Most of `_mul_f32` / `_mul_add_f32` was already shipped: our `rms_norm` shader already does norm + scalar mul, and `residual_norm_store` / `post_attn_residual_norm_store` / `post_ffn_norm_residual_add` already fuse norm + residual add at all the load-bearing boundaries except the post-FFN→next-layer-input step on non-Gemma archs. Phase 1 implemented that last gap via `LARQL_FUSED_PRELAYER_NORM=1` opt-in (Llama 2 7B / Mistral 7B). Parity preserved (bit-identical output across 3 archs). End-to-end A/B: Llama 2 7B `+0.0 tok/s within drift`, Mistral 7B `−0.3 tok/s within drift`. The dispatch-overhead saving from skipping rms_norm in layer N+1 is offset by the heavier `residual_norm_store` (does the RMS reduction inline) replacing the lighter `residual_add` in layer N. Per ADR-015 magnitude-compression at the extreme. | `metal/decode/encode_post_ffn.rs` (`PreLayerNormFusion` struct + branch), `metal/decode/encode_qkv.rs` (`input_already_normed` skip flag), `metal/decode/mod.rs` (call sites + `prelayer_norm_active` plumbing). Kept opt-in per ADR-017 retention rules; revival scenario: different K dimensions or future hardware where the rms_norm dispatch overhead vs `residual_norm_store` extra reduction balance shifts. |
| **D-GEMMA4-E2B** | Investigate Gemma 4 E2B decode 30× pathology surfaced by 2026-05-09 cross-arch bench | (diagnosed) | **DIAGNOSED 2026-05-09.** Cross-arch bench captured `gemma4-e2b-q4k` at 4205 ms/tok GPU fwd vs Gemma 3 4B at 140 ms/tok. **Root cause**: E2B has Per-Layer Embeddings (PLE, `hidden_size_per_layer_input: 256`) which **the Metal pipeline does not implement**. `larql-inference/src/layer_graph/generate/gpu/` explicitly checks `weights.arch.has_per_layer_embeddings()` and routes to `generate_via_cpu_q4k` when true — the entire decode path falls back to CPU. The `LARQL_GPU_TIMING=1` line never fires for E2B because Metal's `decode_token_with_moe_split_fn` is never called. Documented in `gpu.rs` comment: "Without this routing the model produces multilingual gibberish." Diagnosis confirmed by reading the dispatch code and checking E2B's vindex `index.json` (`per_layer_embed_dim: 256`). Llama 2 / Mistral / Gemma 4 31B don't have PLE so route through Metal normally. | Closes the bug as "expected behaviour given PLE-not-in-Metal limitation". The actual perf fix is **D-METAL-PLE** (next row) which implements PLE on Metal. |
| **D-METAL-PLE** | Implement Per-Layer Embeddings in the Metal decode path | Brings Gemma 4 E2B / future-PLE-models from CPU fallback (~1670 ms/tok) onto Metal (target: ~10-20 ms/tok at E2B's compute scale = **80-150× speedup for E2B**). | **IMPLEMENTED 2026-05-10, but E2B end-to-end blocked on a SECOND missing feature (D-METAL-KV-SHARED below).** Wired per-layer PLE block as 4 dispatches: f32_gemv (gate proj) → new `ple_gate_apply` shader (fused gelu_tanh + multiply against precomputed per-layer-input) → f32_gemv (output proj) → reuses `post_ffn_norm_residual_add` (in-place via `h_post_attn`/`new_h` aliasing) for the closing rms_norm + residual add. Precompute stays on CPU (`precompute_per_layer_inputs`); inference uploads the `[num_layers × ple_dim]` table once per generation step via new `MetalBackend::prepare_ple_inputs`. PLE block fires inside `decode_token_with_moe_split_fn` after `encode_post_ffn_residual` and reuses `gate_out_scratch` / `down_out` (dead by post-FFN) for scratch, so no new persistent buffers. Routing: when `LARQL_METAL_PLE=1`, gpu.rs runs prefill via per-token `decode_token` calls and per-step decode with PLE upload before each call (batched-prefill via `dispatch_full_pipeline` is a follow-up after parity passes). Kernel-level parity for `ple_gate_apply` passes 5/5 against CPU reference (`tests/test_kernel_per_layer_embed.rs`, max abs diff < 1e-5). **End-to-end E2B output is still gibberish with `LARQL_METAL_PLE=1`** because Metal also lacks KV-cache sharing — see D-METAL-KV-SHARED below. PLE wiring is complete and correct in isolation; will deliver the predicted speedup once KV sharing lands. | Files: `metal/shaders/per_layer_embed.rs`, `metal/decode/encode_ple.rs`, `metal/decode/mod.rs` (call site), `metal/mod.rs` (`PleInputBuffer` + `prepare_ple_inputs` / `clear_ple_inputs` / `ple_inputs_snapshot`), `metal/ffn_kernels.rs` (`ple_gate_apply_pipeline`), `pipeline.rs` (`PleSpec` + 3 PLE fields on `FullPipelineLayer`), `larql-inference/src/layer_graph/pipeline_layer.rs` (populate from arch keys), `larql-inference/src/layer_graph/generate/gpu/mod.rs` (per-token PLE upload + routing). E2B fixture added to `scripts/bench-cross-arch.sh`. Gated on `LARQL_METAL_PLE=1`; flip the default once D-METAL-KV-SHARED also lands and end-to-end parity passes. |
| **D-METAL-KV-SHARED** | Implement cross-layer KV cache sharing in the Metal decode path | E2B's last 20 of 35 layers reuse K/V from earlier "source" layers (last non-shared sliding / global layer of the same type). Without this, layers 15-34 compute their own (wrong) K/V → attention is broken from L15 → garbage output. Same mechanism likely needed for any future KV-shared model. | **PARTIALLY IMPLEMENTED 2026-05-10 — routing correct, source-cache contents byte-identical to CPU, but L15+ shared-layer dispatch chain still produces wrong residuals.** See "D-METAL-KV-SHARED bisection 2026-05-10" section below for the bisection record and remaining diagnostic plan. | (see section below) |
| **D-CROSS-PARITY** | Cross-architecture decode bench (Gemma 3 4B, Gemma 4 31B dense, Gemma 4 26B A4B MoE, Llama 2 7B, Mistral 7B) | Operationalises ADR-017 model-agnosticity check; surfaces thermal artifacts (every-arch-regresses-simultaneously signature). | **SHIPPED 2026-05-09 (Phase 1).** `scripts/bench-cross-arch.sh` + `bench/baselines/cross-arch/` + `make bench-cross-arch[ ARGS=--save-baseline\|--compare]`. Captured first under-load run; cool-machine baseline pending. **Phase 2 (per-shader)** still open — current sweep is end-to-end-only; per-shader parity tests (e.g. dispatching `q4k_ffn_gate_up_8sg` against synthetic input on multiple K dimensions to surface kernel-specific arch sensitivities) would catch deeper regressions earlier. | `scripts/bench-cross-arch.sh`, `bench/baselines/cross-arch/README.md`. Phase 2: parametrised tests in `crates/larql-compute/tests/`. |
| **D-ROPE-VARIANTS** | Add `rope_neox` and `rope_multi` shader variants | Coverage for non-split-half RoPE models (GPT-NeoX, Pythia, some Falcon, multi-frequency long-context) | Open. We have one `rope` shader (split-half). llama.cpp ships `kernel_rope_norm_*`, `kernel_rope_neox_*`, `kernel_rope_multi_*`, `kernel_rope_vision_*`. Currently no architectures in `larql-models/architectures/` need NeoX (all use split-half), so this is **deferred until a NeoX-using model is brought into scope**. TODO documented in `metal/shaders/rope.rs`. | `metal/shaders/rope.rs` extension. See [llama-cpp-comparison.md](docs/llama-cpp-comparison.md) §3 |

**Sequencing rationale (updated 2026-05-09)**: D-FFN-PROFILE shipped; data
shows all three buckets roughly equal thirds (~34/36/29%). Gate+up is the
largest but already bandwidth-bound at 74% LPDDR5X peak — no headroom left.
D-FFN-FUSE targets act+down (~0.24 ms from GEGLU dispatch overhead) but is
blocked by an unexplained production NaN. D-PREFILL-MM is closed (twice-
falsified) and superseded by D-PREFILL-MM2. The 2026-05-09 dylib-symbol
diff against llama.cpp ([llama-cpp-comparison.md](docs/llama-cpp-comparison.md))
confirms three concrete kernel-architecture gaps: flash attention (covered
by D-ATTN-MTG), `simdgroup_matrix` prefill matmul (D-PREFILL-MM2), and
scalar-op RMS-norm pre-fusion (D-RMS-FUSE).

**Open levers ordered by leverage and risk**:

- **High-impact, multi-day kernel research:** D-PREFILL-MM2 (closes
  4–14× prefill gap), D-ATTN-MTG (+5–8 tok/s decode).
- **Concrete agnosticity bug:** D-GEMMA4-E2B (real 30× per-arch slowdown
  needs investigation).
- **Smaller perf wins:** D-RMS-FUSE (~0.1 ms decode), D-FFN-FUSE
  (~0.2 ms but NaN-blocked).
- **Infrastructure:** D-CROSS-PARITY (per-shader cross-model tests),
  D-ROPE-VARIANTS (deferred until needed).

**Recommended next pull**: D-GEMMA4-E2B (concrete bug, bounded
investigation, immediately empirically grounded — the 30× slowdown
surfaced by the cross-arch bench is the kind of thing that should be
investigated before more multi-day kernel work goes in). After that,
D-CROSS-PARITY to harden the test surface, then D-PREFILL-MM2 or
D-ATTN-MTG as the bigger swing.

**Documentation entry points** for new contributors:

- [docs/shader-inventory.md](docs/shader-inventory.md) — per-shader retention rationale and applicability across model families.
- [docs/architecture-shader-map.md](docs/architecture-shader-map.md) — which Metal shaders each model architecture dispatches.
- [docs/llama-cpp-comparison.md](docs/llama-cpp-comparison.md) — kernel-architecture comparison vs llama.cpp; documents the three load-bearing gaps.
- [docs/adr/017-shader-retention-model-agnosticity.md](docs/adr/017-shader-retention-model-agnosticity.md) — when to add/keep/delete shaders.
- [docs/adr/018-architecture-shader-routing.md](docs/adr/018-architecture-shader-routing.md) — when to add a new architecture.

### D-METAL-KV-SHARED bisection 2026-05-10

End-to-end E2B output was still gibberish ("တ initialState eTo aud ανhiokin
initial است류 …") after the routing-and-plumbing landed. This section is the
diagnostic record.

**What was implemented**

Field plumbing — `kv_shared_source: Option<usize>` added to
`FullPipelineLayer`, populated in `pipeline_layer.rs::build_arch_params`
from `arch.kv_shared_source_layer(layer)`. Three literal-constructor
sites updated.

Dispatch — in `metal/decode/encode_attn.rs`:

- `did_fused_attn` and `use_fused_kv_aa` are gated on
  `kv_shared_source.is_none()`. Shared layers can't use the fused
  attn / fused kv_append_attend kernels because both write the
  layer's own K/V cache and would corrupt the source's cache.
- Shared layers fall through to a new `else if let Some(src) =
  kv_shared_source` branch that picks `kv_attention` or
  `kv_attention_long` based on `attention_span(t_val, window_size)`,
  binds `kv_cache.layers[src].k_cache` / `v_cache`, and passes
  `t_val = source.current_len`. Source's `current_len` has already
  incremented past the new token by the time the shared layer runs
  (layers process in order, `current_len += 1` fires at the end of
  source's Step 4).
- `current_len += 1` for the shared layer is gated on
  `kv_shared_source.is_none()`. Shared layers leave their own
  cache pointer at 0 forever; the source's pointer is the
  source-of-truth.
- Q is computed at the shared layer (own `W_q`, own
  `q_norm_weight`, RoPE'd at `pos = source.current_len.saturating_sub(1)`).
  K/V projections still run at the shared layer but the results
  are discarded — wasted work that's harmless and was preserved
  for first-pass simplicity.

The routing path was sanity-checked with a `LARQL_KV_SHARED_DEBUG=1`
print (since removed) — source mapping is correct (sliding L15-L18
→ L13, global L19 → L14, etc.), `src.current_len` advances 1, 2, 3, …
per decode_token call, `t_val` matches.

**Bisection**

Two diagnostic env vars added to `metal/decode/mod.rs` for the
bisection (kept; trivial to use again):

- `LARQL_KV_CACHE_DUMP_DIR=<dir>` — at end of each `decode_token` call,
  dumps `metal_L{NN}_K_cache.f32` / `_V_cache.f32` for every layer
  with `current_len > 0`. Last call wins (file overwritten).
- `LARQL_PERCALL_LAYER_DUMP_DIR=<dir>` — at end of each call, dumps
  `metal_call{NNN}_h_final.f32` (residual after L34 +
  layer_scalar), `metal_call{NNN}_x_input.f32` (the per-token
  embed input), and per-layer K caches with the call counter in
  the filename. Counter is process-global (`CALL_COUNT`
  `AtomicUsize`).

**Bisection results** (E2B prompt "Hi" with chat-template wrap →
seq_len=29 prompt tokens, `-n 2`):

| stage | result |
|---|---|
| Token IDs identical CPU vs Metal | ✓ (`token_ids.len() == 29` in both via `LARQL_TOKEN_DEBUG=1`) |
| Per-call input embed `metal_call{n}_x_input` | ✓ cosine 1.0000 vs CPU's `cpu_h_embed.f32` row n for all prefill positions 0-28 |
| Source K cache contents at L0, L13, L14 | ✓ cosine 1.0000 for all 29 prompt positions vs CPU's `cpu_L0_k_out_after_rope.f32` |
| Source V cache contents at L13 | ✓ cosine 1.0000 for all 29 prompt positions vs CPU's `cpu_L0_v_out.f32` |
| Per-call `h_final` (after L34, before final_norm) | ✗ cosine **0.18-0.48** vs CPU's `cpu_layer_34.f32[pos]` at every prompt position |

So everything that's a function of source layers' weights, RoPE,
qk_norm, and V-norm produces byte-perfect cache contents. The bug
is downstream of the source-layer write.

A telling artifact: Metal's `h_final` magnitude across all 29
prompt positions is roughly **constant** at ~17, while CPU's
varies 9-39. The Metal residual stream is being **compressed** —
losing per-position information — which is the signature of
attention not differentiating positions (uniform softmax over
identical-magnitude logits). This is consistent with shared
layers' attention dispatch returning a near-constant output
regardless of Q.

**What's been ruled out**

- Source layer correctness: K/V cache is byte-identical CPU vs
  Metal. Source projections (W_q, W_k, W_v), qk_norm, RoPE,
  V-norm all produce identical bytes through L14.
- Routing: `kv_shared_source` populates correctly,
  `t_val`/`pos` math is correct, layer→source mapping is
  correct, `current_len` advances correctly.
- Tokenization: `encode_prompt` produces identical
  `token_ids` in both code paths (verified via
  `LARQL_TOKEN_DEBUG=1`).
- Embed: `embed_tokens_pub` produces identical embeds at every
  position (cosine 1.000).
- Q at shared layer: computed with the layer's own W_q against
  correct h_input. RoPE'd at `pos = N` (matches source's K
  rotation at position N).
- PLE math: kernel-level parity passes (5/5 tests at <1e-5 abs
  diff vs CPU), and turning PLE off does not change the gibberish
  symptom in any way that reveals PLE as the culprit.
- Fused-kernel interaction: same gibberish under
  `LARQL_FUSED_KV_APPEND_ATTEND=0`,
  `LARQL_FUSED_QK_NORM_ROPE=0`, `LARQL_FUSED_POST_ATTN_NORM=0`,
  `LARQL_FUSED_POST_FFN_NORM=0`, `LARQL_FUSED_PRELAYER_NORM=0`
  — all simultaneously and individually. Bug isn't in any
  fused kernel.

**What's left to check** (next session's plan)

1. **Stage-level dump inside the per-call loop.** Extend
   `LARQL_PERCALL_LAYER_DUMP_DIR` to also capture L15..L34 stage
   outputs (`q_out_after_rope`, `attn_out_buf`, `o_out_buf`,
   `h_post_attn`, post-PLE `h`, post-`layer_scalar` `h`) at the
   end of each call, gated on a layer-allowlist env var.
   Compare to CPU's `cpu_L{N}_*.f32` for the same prompt position.
   The first stage that diverges is the bug.
2. **Memory hazard tracking on the source cache.** Try inserting
   `enc.memory_barrier_with_resources(&[source_k_cache,
   source_v_cache], MTLBarrierScope::Buffers)` at the start of
   the shared-layer attend dispatch. Apple's compute encoder
   auto-tracks RAW hazards across dispatches in the same encoder,
   but the source's `kv_append_attend_fused` writes via
   `threadgroup_barrier(mem_device)` rather than ending its
   dispatch — possible the auto-barrier doesn't see the writes.
3. **Force unfused source path.** Make source layers (those that
   are themselves a `kv_shared_source` for some later layer) use
   the unfused `kv_cache_append` + `kv_attention` chain instead
   of `kv_append_attend_fused`. This forces a hard dispatch
   boundary so the writes are flushed before any read by a
   shared layer. Slower but eliminates the hazard hypothesis.
4. **Single-position end-to-end check.** With a 1-token raw
   prompt (set `LARQL_RAW_PROMPT=1`), the shared-layer attention
   collapses to attend(Q, K[0..1], V[0..1]) — softmax over a
   single position is identity, attn_out = V[0]. Compare
   Metal's L15 attn_out to CPU's L15 attn_out at this position.
   Diverging there isolates the bug to Q computation or
   attention output write; matching there pushes it downstream
   (FFN, PLE, layer_scalar, or the next shared layer's input).

**Tests landed for regression protection**

- `crates/larql-inference/src/layer_graph/pipeline_layer.rs::tests`
  (3 new tests, all green):
  - `build_arch_params_populates_kv_shared_source_for_e2b_like_arch`
    pins the `kv_shared_source` field's contract end-to-end:
    builds a synthetic Gemma-4-E2B-shaped arch via
    `detect_from_json` (4 layers, 2 sliding non-shared + 2
    shared) and asserts that L0/L1 → None, L2 → Some(0),
    L3 → Some(1).  Catches any future regression where someone
    removes the
    `kv_shared_source: arch.kv_shared_source_layer(layer)`
    line in `build_arch_params`.
  - `build_arch_params_populates_ple_fields_for_e2b_like_arch`
    pins the `ple_input_gate` / `ple_projection` /
    `ple_post_norm` field contract.  Builds the synthetic
    arch with PLE keys and tensors, runs `build_arch_params`
    for each layer, asserts all three slices populate and
    `ple_spec()` returns Some.
  - `build_arch_params_leaves_ple_and_kv_shared_fields_none_for_tinymodel`
    pins the negative case: non-PLE / non-shared archs leave
    every new field as None.  Prevents spurious population.
- `crates/larql-compute-metal/tests/test_kernel_per_layer_embed.rs`
  (5 tests, all green): `ple_gate_apply` shader vs CPU
  reference at <1e-5 abs diff — smooth inputs, zero inputs,
  large negatives, E2B production shape.

These tests don't catch the current bug (which is a
dispatch-level issue beyond field plumbing), but they do
prevent the entire field-plumbing regression class — the
"someone refactored the trait and silently dropped a populate
line" failure mode that's hit larql-models / larql-inference
multiple times during this work.

**Out of scope for this section but flagged**

- Per-layer Metal-vs-CPU stage parity tests (D-METAL-KV-SHARED
  Phase B test infrastructure): take a tiny synthetic
  Gemma-4-E2B-shaped vindex with 4-layer / hidden=8 weights,
  run a 3-token prompt through both paths, assert
  byte-equivalence for every per-stage scratch buffer at every
  layer.  Would have caught the current bug in CI.  Blocked on
  not yet having a small-vindex builder; cross-arch bench
  fixture is the closest existing analogue but uses real
  E2B-scale models.

### Decode gap diagnosis (2026-04-28, 3-iter median)

Measured per-stage on `gemma3-4b-q4k-v2.vindex`, 50-token decode after 5 warmup, ollama gemma3:4b reference on same machine:

| stage | LARQL | Ollama (est.) | gap (ms) | gap (% of total) |
|---|---|---|---|---|
| **GPU forward** (34 layers) | **11.91 ms** | **~8.5 ms** | **3.41 ms** | **90% of gap** |
| **lm_head** (262K × 2560) | **1.89 ms** | **~1.5 ms** | 0.39 ms | 10% of gap |
| embed + final_norm + detok | <0.05 ms | <0.15 ms | ~0 | ~0% |
| **total** | **13.16 ms/tok = 76 tok/s** | **10.15 ms/tok = 99 tok/s** | **3.01 ms** | **1.30×** |

The gap is **almost entirely in the GPU forward**. Within GPU forward (~0.35 ms/layer × 34 layers):

| kernel | shape | batched GB/s | est. share | utilisation |
|---|---|---|---|---|
| `q4k_ffn_gate_up` (fused gate+up) | 10240 × 2560 | **274 GB/s** | ~31% (~3.7 ms) | bandwidth-bound, **74% of LPDDR5X peak** |
| `q6k_matvec` (down) | 2560 × 10240 | **311 GB/s** | ~19% (~2.3 ms) | bandwidth-bound, **84% of peak** |
| Wo + QKV + attn + 4× RMS norms | mixed | mixed | ~50% (~5.9 ms) | mixed, presumed near-peak |
| **GPU fwd total** | — | — | 100% (~11.9 ms) | — |

**lm_head**: `f32_gemv` runs at 374 GB/s — within 1% of LPDDR5X peak (370 GB/s). Bandwidth is NOT the bottleneck there; remaining gap is CPU-side readback + size-K heap.

⚠ The earlier "103 GB/s ALU-bound on q4k_ffn_gate_up" diagnosis was a **profiler bug** — the "batched" measurement was creating a fresh cmd buffer per call (with commit+wait per call) instead of running `n_layers` dispatches in ONE cmd buffer. The per-call overhead dominated, undercounting throughput 2-4×. Fixed 2026-04-28 in `metal/diag/kernel_profile.rs::measure_single_cmdbuf_batched`. With the fix, both big FFN kernels are bandwidth-bound at 74-84% of LPDDR5X peak — no compute-bound headroom.

Reproduction: `cargo run --release --features gpu -p larql-compute-metal --bin larql -- bench output/gemma3-4b-q4k-v2.vindex --backends metal --ollama gemma3:4b --tokens 50 --warmup 5` on a quiet system. Per-kernel detail: `cargo run --release --features gpu -p larql-compute-metal --example diag_profile_kernels`.

### Decode kernel optimization — the path forward (2026-04-28, revised)

**Both big FFN kernels are already bandwidth-bound near LPDDR5X peak.** The earlier "compute-bound, ALU-throttled" framing was a profiler artifact. The remaining 3 ms gap to ollama isn't sitting in any single kernel with obvious headroom — it's distributed across the dispatch pipeline.

#### Track A — profiler harness fixed ✓ (2026-04-28, done)

`metal/diag/kernel_profile.rs` now uses `measure_single_cmdbuf_batched` for q6k_matvec and q4k_ffn_gate_up. Old `measure_batched` is kept (with a "DON'T USE for kernel throughput" doc note) for callers who genuinely want per-call cmd-buffer overhead. **Follow-up**: same fix for q4k_matvec (Wo) and any future kernels added.

#### Track B — `q4k_ffn_gate_up_f16acc` SHIPPED 2026-04-28 (opt-in, no end-to-end win on this hardware)

`metal/shaders/q4k_ffn_gate_up_f16acc.rs` — variant with f16 inner accumulators (per-superblock dot product). Outer accumulator and `sumy` stay f32. Safe because Q4_K nibbles are 0..15 (exact in f16) and RMS-normed X has |x| < ~5, so the 16-FMA partial sum stays well under f16 max (65504).

**Measured 2026-04-28**:

| measurement | f32 (default) | f16 acc | delta |
|---|---|---|---|
| Kernel isolated (N=10240, K=2560) | 0.607 ms | 0.340 ms | **1.79× kernel speedup** |
| End-to-end decode, **thermally loaded** GPU | 16.40 ms/tok | 13.34 ms/tok | +23% (apparent) |
| End-to-end decode, **quiet** GPU | 12.95 ms/tok | 13.06 ms/tok | **at parity (~1% slower)** |
| Numerical drift (max abs in kernel output) | — | 0.155 (≈1.5% relative) | — |
| Output text on 10-prompt corpus | bit-identical to f16 | bit-identical to f32 | full parity ✓ |

**The end-to-end perf win does not reproduce on a quiet GPU.** Initial 5-iter measurement showed +23% throughput, but that was on a thermally-loaded system where the f32 kernel was throttling. On a quiet system both paths run at the same wall-clock — f16 freed ALU cycles get absorbed into pipeline stalls or thermal headroom the surrounding kernels reclaim. The 1.79× kernel speedup is real in isolation; it doesn't translate to end-to-end decode improvement because the kernel was already bandwidth-bound (274 GB/s, 74% of LPDDR5X peak), not ALU-bound.

**Numerical parity is solid**: 10-prompt greedy-decode sweep (knowledge / code / math / creative / translation, 32 tokens each) — all outputs bit-identical between f32 and f16 paths. The 1.5% per-call drift never crosses a top-1 token boundary in the validated corpus.

**Status: kept as `LARQL_F16_ACC=1` opt-in.** Default stays f32. Useful as future-proofing if (a) hardware changes the ALU/bandwidth balance, (b) a future kernel re-fuses the path so ALU becomes the bottleneck, or (c) a sustained-load workload benefits from less thermal pressure. Not promoted to default because there's no measurable steady-state win to justify the precision risk on unvalidated workloads.

**Lesson for future kernel work**: the kernel-isolated profiler can be misleading. A 1.79× isolated speedup ≠ 1.79× end-to-end if the kernel was bandwidth-bound or part of a longer pipeline where other resources serialise the GPU. Always validate end-to-end on a quiet system before adopting.

#### Track C — `q4k_ffn_gate_up_nr2` candidate ROUND-TRIPPED 2026-05-02, REMOVED 2026-05-09

NR2 (2 rows / simdgroup variant of `q4k_ffn_gate_up`) was a strong isolated profiler candidate after the 8sg landing — `diag_profile_kernels` showed:

| | iso ms | iso GB/s | **batched ms** | **batched GB/s** |
|---|---|---|---|---|
| 8sg (default) | 0.591 | 51.4 | 0.106 | **278.9** |
| NR2 (candidate) | 0.401 | 76.8 | 0.110 | **267.0** |

End-to-end A/B (warmup 8, decode 30, quiet GPU, three runs, 2026-05-02):

| config | tok/s | GPU fwd | lm_head | output |
|---|---|---|---|---|
| baseline (8sg) | **75.9** | **11.19 ms** | 2.99 ms | "Paris" ✓ |
| NR2 (`LARQL_GATE_UP_NR2=1`) | 72.9 | 11.80 ms (**+0.62 ms**) | 2.96 ms | "Paris" ✓ |

NR2 wins isolated by 1.47× and **loses batched by 4%**. End-to-end tracks the batched number, not the isolated one — the 1.47× iso win was dispatch-overhead amortisation that disappears once n_layers calls share one cmd buffer. **Initial outcome (2026-05-02): not promoted; kept as opt-in.**

**Re-benched 2026-05-09** alongside the QKV-gap investigation:

| config | tok/s | GPU fwd |
|---|---|---|
| 8sg baseline (run 1) | 86.3 | 11.71 ms |
| NR2 (`LARQL_GATE_UP_NR2=1`) | 83.4 | 12.05 ms (**+0.34 ms**) |
| 8sg baseline (run 2, drift check) | 86.2 | 11.64 ms |

The 0.07 ms baseline drift between runs 1 and 2 confirms the −0.34 ms NR2 regression is real signal, not thermal noise (unlike the f16_acc story where a +23% measurement turned out to be thermal artifact). Direction matched the batched-diag prediction; magnitude was ~3× the predicted delta but the ordering held — exactly the contract ADR-015 makes.

**Outcome (2026-05-09): kernel and `LARQL_GATE_UP_NR2` env var removed.** The candidate had two independent benches against it; leaving it as opt-in dead code was inviting re-litigation. Shader (`shaders/q4k_ffn_gate_up_nr2.rs`), pipeline field, env-var constant, dispatch-branch in `encode_ffn`, diag-profile entry, and shader-bench inventory entry all deleted. Historical record preserved here, in `PERFORMANCE.md`, and in ADR-015.

**Same A/B run also confirmed** that the v5 stride-32 lm_head is the *fast* path, not just the correct one: `LARQL_LM_HEAD_STRIDE32=0` regressed lm_head 2.99 → 4.08 ms (+1.09 ms, 75.9 → 69.5 tok/s). The "+0.7 ms cost" framing in PERFORMANCE.md is relative to the pre-fix broken-output kernel, not the current fallback. No tradeoff to chase.

#### Iso-vs-batched pattern, third confirmed instance

`f16_acc` (2026-04-28) + `attn_fused` (2026-05-01) + `nr2` (2026-05-02) all showed isolated wins that failed end-to-end. Pattern pinned in `docs/adr/015-isolated-vs-batched-kernel-perf.md`. **Promotion criterion going forward**: a candidate must win the *batched* `diag_profile_kernels` column AND end-to-end bench. Isolated-only wins do not justify a session of end-to-end measurement on their own — three sessions burned on this so far.

#### Track D — QKV defuse PROMOTED 2026-05-09 (+1.6 tok/s, magnitude undershoot)

The fused `q4k_q6k_qkv_proj_normed` kernel had been the production default for Gemma 3/4 layers since the QKV-stage fusion landed earlier in the year. It saved ~0.24 ms/tok of dispatch overhead but ran at 199 GB/s vs the non-fused `q4k_q6k_qkv_proj`'s 287 GB/s — the per-TG H + norm_w reread (3× redundant device traffic across the 4 simdgroups) was costing ~1.46 ms/tok in the per-kernel batched diag.

Diag prediction: defusing recovers 1.46 ms − 0.24 ms = **−1.22 ms/tok** end-to-end.

End-to-end A/B (warmup 8, n=100, drift 0.02 ms):

| config | tok/s | mean ms | GPU fwd | output |
|---|---|---|---|---|
| baseline (fused, default) | 84.8 / 85.0 | 11.79 / 11.77 | 11.79 / 11.86 | "Paris" ✓ |
| defused (`LARQL_QKV_DEFUSED=1` at the time) | **86.5** | **11.57** | **11.52** | "Paris" ✓ |

**Measured Δ: +1.6 tok/s, −0.30 ms/tok GPU fwd** — direction matched the diag prediction but **magnitude was 18% of predicted**.

**Promoted to default** 2026-05-09; env var renamed to `LARQL_QKV_FUSED=1` (opts back in to fused). Fused shader + `encode_normed_q4k_q6k_qkv` retained as the opt-in fallback. The +1.6 tok/s landing matches the +1.6-tok/s gap that existed at the time of investigation between us and the next-best decode milestone.

**Why magnitude undershot.** The non-fused kernel's 287 GB/s peak is a single-kernel-isolated number. In the production decode pipeline, the candidate shares the LPDDR5X bus with attention, FFN gate+up, FFN down, lm_head — all also bandwidth-bound. The candidate doesn't get to *claim* its kernel-isolated headroom because the surrounding pipeline already saturates the same bus. Predicted batched-diag deltas should be discounted ~3-5× for bandwidth-headroom candidates; cycle-headroom or dispatch-count candidates don't show this discount as severely. Pinned as the first magnitude-compression case study in ADR-015.

**Cross-arch validation 2026-05-09 (post-thermal-cooldown)**: A/B on Gemma 4 26B A4B (60 layers, hidden=5376, MoE) confirms defuse is also net-positive on the larger MoE arch:

| config | tok/s | GPU fwd | mean ms |
|---|---|---|---|
| defused (current default, run 1) | 14.3 | 62.90 ms | 69.92 |
| Fused (`LARQL_QKV_FUSED=1`) | 13.9 | 65.91 ms | 71.82 |
| defused (drift check, run 2) | 13.9 | 64.53 ms | 71.94 |

**Δ defused vs fused: +0.4 tok/s, −2.19 ms GPU fwd** (median across both defused runs vs fused). Drift between defused runs is 1.6 ms; signal is 2.19 ms — at the noise floor but directionally consistent with the Gemma 3 4B finding (+1.6 tok/s defused). **No arch-specific dispatch path needed** — the same defused default works for both 4B and 26B. (Smaller win on 26B reflects the larger absolute GPU-fwd budget; the fixed dispatch-overhead saving is a smaller fraction of the total.) Same A/B also confirmed there's no QKV-defuse-driven regression on 26B; the earlier "2.4× regression vs 19.4 tok/s baseline" was a thermal artifact from sustained load (`feedback_thermal_perf_artifacts.md` memory entry).

#### Remaining decode gap (after f16 acc, attn_fused, NR2 ruled out; QKV defuse landed)

Decode at ~86 tok/s vs ollama ~99-101 tok/s steady-state, ~1.16-1.18×. The "isolated-only" candidates are exhausted on the FFN gate+up path. Remaining options, ordered:

1. **Multi-TG `attn_fused` retry** — the standalone `qk_norm_rope_fused` runs 12 TGs; the fused variant collapses to 8 because of register pressure. A multi-TG-per-head fused variant (split the QKV+attend work across 2 TGs, keep total ≥12) would preserve parallelism while saving the dispatch. **This is the one remaining iso-win-prone candidate that is *also* batched-friendly** — the dispatch saving lives in the cmd-buffer count, not the per-kernel ALU. Estimated +0.2–0.4 ms recovery if successful.
2. ~~f16 lm_head wiring~~ — **falsified 2026-05-09**. Production lm_head is already on `q4k_matvec` (1.85 ms/tok) since the 2026-05-02 dispatch-geometry fix; f16_gemv fallback measures 3.88 ms/tok, so wiring f16 ahead of Q4_K would regress lm_head by ~2 ms/tok. The "+0.7 ms paid for v5 stride-32 correctness" framing was itself stale — stride-32 is no longer the production path, the dispatch fix superseded it. 31B memory-footprint concern (5.6 GB f32 clone) is a separate issue addressable independently.
3. **Wire `ProfileTimings.gate_up_ms` / `down_ms` producer** (#12 in P0 structural cleanup) — without it, the remaining ~2 ms in GPU fwd is unattributed. Diagnostic, not perf — but it points the next swing.
4. **Apply f16 to other Q4_K matvecs** (Wo, QKV) — same diagnosis applies as #2: Q4_K is already the fast path on these. Drop unless a specific dispatch-geometry bug surfaces.
5. **Dispatch overhead reduction** (~100-dispatch gap to ollama) — closing this means more aggressive kernel fusion. The fused FFN gate+up + GEGLU + down for Q6_K models was tried (#1 below) and regressed — re-enable might require a cheaper activation variant.
6. **Accept ~1.17× as the M3 Max ceiling** for our pipeline architecture. ollama's hand-tuned llama.cpp kernels have years of tuning; closing the last 15% likely requires fundamental architecture changes (`simdgroup_matrix` for matvec).

**Effort**: multi-TG attn_fused retry is ~2 days (split the kernel, keep parity tests, batched bench). f16 lm_head wiring is ~half a day. Other tracks are larger.

#### Acceptance criterion

**Close 1.5 ms of the 3 ms decode gap to reach ~12 tok/s (~85 tok/s, 1.16× of ollama)**. Closing the full 3 ms requires `simdgroup_matrix` for matvec (no llama.cpp precedent for matvec — they use it for matmul/prefill only). Above that ceiling we're chasing Apple-specific intrinsics not exposed publicly.

**Status (2026-05-09)**: QKV defuse landed +1.6–1.8 tok/s → **88 tok/s, 1.17× of ollama** on a fair-thermal session bench. **Acceptance threshold reached** on the tok/s axis (target was ~85), one notch off on the gap-ratio axis (target 1.16×, ollama drift is itself ~±15% session-to-session so this is effectively at the threshold). Closing the remaining ~14% means **multi-TG `attn_fused` retry** (~+0.2-0.4 ms, ~2 days work) — the only known-viable swing left. ~~f16 lm_head wiring~~ was on the table but **falsified 2026-05-09**: production lm_head is already on `q4k_matvec` at 1.85 ms/tok, and f16_gemv fallback measures 3.88 ms/tok — wiring f16 ahead of Q4_K would *regress* lm_head by ~2 ms/tok. The "v5 stride-32 correctness tax" the f16 plan referenced was itself stale: the 2026-05-02 dispatch-geometry fix removed that tax. See `project_f16_gemv_wiring_todo.md` memory entry for the falsification record.

### #0 — Decode kernel optimisation (NEW, 2026-04-28)

See "Decode kernel optimization" section above. Replaces the older "#6 — Q4_K kernel optimization" P0 entry below; that entry now serves as the historical record of what was tried and ruled out.



### Prefill: per-position matvec → matmul (4-14× gap, biggest end-to-end win)

**Measured 2026-04-27** (gemma3-4b-q4k-v2.vindex). The gap **scales with prompt length**:

| prompt length | larql prefill | ollama prefill | gap |
|---|---|---|---|
| 18 tok (chat) | 196 ms (10.9 ms/tok) | 50 ms (2.8 ms/tok) | **3.9×** |
| 340 tok (long) | 2933 ms (8.6 ms/tok) | 210 ms (0.62 ms/tok) | **14×** |

The widening ratio is the smoking gun: larql is per-position linear (`prefill ≈ seq_len × decode_per_tok`); ollama is sublinear via gemm. Decode itself (seq=1) is only ~1.17× behind (was 1.30× when this diagnosis was written; subsequent dispatch-geometry fix and QKV defuse closed it).

**Root cause** (verified 2026-04-27 by reading `metal/ops/full_pipeline/dispatch.rs`): `prefill_q4 → dispatch_full_pipeline` IS already wired and IS allocating `[seq_len × hidden]` buffers, but every per-stage compute step issues per-position matvec dispatches. For an 18-token × 34-layer prefill that's ~600+ matvec calls vs ollama's ~34 gemm calls per stage.

**The earlier "wire dispatch_prefill" suggestion was wrong** — `metal/prefill.rs::dispatch_prefill` is dead code; production already goes through `prefill_q4`. Infrastructure isn't missing, the kernel approach is.

**Three actionable wins, ordered by effort × impact:**

1. **Encoder coalescing** — **SHIPPED 2026-04-27**, marginal impact.
   Hoisted `cmd.new_compute_command_encoder()` out of the per-position loops in `dispatch.rs::399` (O proj) and `stages.rs::97`, `:174` (input_norm + QKV). One encoder per stage instead of `seq_len` of them. **Measured: saves ~5% on long prompts, within noise on short prompts.** The 5 µs × seq_len savings is real but dwarfed by per-dispatch kernel compute time. No regression on decode (seq=1 path runs the loop once, identical semantics). 135 Metal tests still pass.

2. **Q4_K threadgroup memory reuse across positions** (M, 2-3 days, ~20-30% on long prompts — speculative)
   The current matvec loads the same Q4_K weight rows from device memory once per position dispatch. Cache one super-block of weights in threadgroup memory and run all `seq_len` positions through it before advancing rows. Same matvec primitive, reordered loops. Closes a chunk without writing new shaders. **Caveat**: the gate+up kernel is already compute-bound (272 GB/s, ALU-limited dequant), so weight-side caching may not help much; output-side caching across positions might.

3. **Q4_K matmul (gemm) kernel** — **SHIPPED 2026-04-27** (kernel + parity tests; not yet wired into prefill).
   `crates/larql-compute-metal/src/shaders/q4k_matmul.rs` — amortises Q4_K dequant across `COLS_PER_TG=4` positions per super-block. Same `ROWS_PER_TG=4` simdgroup geometry as `q4k_matvec`, plus a per-thread `acc[4]` accumulator array (16 bytes register footprint, fits comfortably). 5 parity tests in `tests/test_kernel_q4k_matmul.rs` assert bit-equivalence with stacked matvec calls across basic / seq_len=1 / ragged-tail / production shapes. Perf spot-check (`tests/test_kernel_q4k_matmul_perf.rs`, gated on `LARQL_PERF_SPOT_CHECK=1`) on N=2560, K=8192, M=18: **3.82× speedup** (4.99 ms stacked matvec → 1.31 ms matmul). At full closure that's ~196 ms → ~51 ms prefill on Gemma 3 4B (ollama parity).

   **Wiring status — partial 2026-04-27**: Wired into the O projection site (`dispatch.rs::5. O projection`). Added `q4k_matmul: Option<&KernelHandle>` to `quant_matvec::Pipelines`; threaded through `dispatch_full_pipeline` signature and all callers. Branches on `seq_len > 1 && format == Q4_K && pipeline.is_some()` and falls back to per-position matvec otherwise. Decode (seq=1) keeps the matvec path, decode tests (135 lib) all pass.

   **Measured impact of partial wiring**: WITHIN NOISE. Short prompt 196 → 203 ms; long prompt 2933 → 3006 ms; decode 13.78 → 13.45 ms/tok. O projection is only ~1/7 of the per-position Q4_K work in prefill — the 3.8× kernel speedup applied to one site saves ~2 ms on an 18-tok prompt, below the ±5% prefill noise floor. The kernel works, but a single call site doesn't show in the headline number.

   **Open — full wiring** (the actual perf delivery):
   - `metal/stages/ffn.rs::76,135,172`: FFN gate, up, and down matvec loops. Each is a clean per-position Q4_K matvec — direct matmul swap, no fused-kernel complications. Combined ~3× the work of O proj; should be the largest measurable win.
   - `metal/ops/full_pipeline/stages.rs::97` (QKV f32 path): fused `q4kf_qkv_proj` / `q4k_qkv_proj` kernels do Q+K+V in one dispatch per position. Either (a) write a fused Q+K+V matmul kernel (mirrors the per-position fused convention, biggest one-time effort), or (b) fall back to per-projection matmul (3 calls per layer, simpler but loses the per-position fusion win). Bench-test both to decide.
   - `metal/ops/full_pipeline/stages.rs::174` (Q8 path): same pattern; Q8 has its own fused QKV kernel.

   Once gate/up/down + QKV are all wired, total Q4_K per-position dispatches drop from ~7×seq_len per layer to ~5 per layer (matmul replaces gate/up/down/QKV; activation + residual stay per-position because they're not matmuls). At that point the 3.8× kernel speedup should translate to a ~3× prefill improvement, closing most of the 4-14× gap.

   For the long-haul (matching ollama on 340-token prompts): the current matmul uses simdgroup-sum reduction; a future step is `simdgroup_matrix` operations (the existing P2 entry below). The current kernel is "matvec amortised", not true gemm — but the perf headroom from amortisation alone is enough to close the short-prompt gap if all sites are wired.

**What landed in #1 (for future-me)**: encoder coalescing at three sites (`dispatch.rs::5. O projection`, `stages.rs::QKV f32 path`, `stages.rs::QKV Q8 path`). The FFN stage was already coalesced — `ffn::encode_gated/encode_standard` take a single encoder and iterate per-position dispatches inside. `residual::encode_post_attn/post_ffn` similarly. So the only remaining waste was at the dispatch.rs/stages.rs level.

**Bench reproduction**:
- Short: `larql bench <vindex> --backends metal --ollama gemma3:4b --tokens 100 --warmup 8`
- Long: same with `--prompt "<340+ token prompt>"` to surface the full gap.

### q6k_matvec ROWS_PER_TG shader/dispatch mismatch — **FIXED (2026-04-26)**

**Root cause of the "regression" to 68-70 tok/s:** the shader constant
`Q6K_ROWS_PER_TG` and the Rust dispatch constant `ROWS_PER_TG` were mismatched:

- **Shader:** `Q6K_ROWS_PER_TG = 2` → `row_idx = tg_id * 2 + sg_id` (sg_id 0..3 = 4 rows per TG)
- **Rust dispatch (HEAD):** `ROWS_PER_TG = 4` → dispatched ceil(N/4) = 640 TGs

With this mismatch, maximum covered row = 639 × 2 + 3 = **1281 of 2560**. Rows 1282–2559 received **zeros** — a silent correctness bug in the FFN down projection for dense models. Model output was degraded but simple prompts (e.g. "Paris") survived because the residual stream carried enough signal.

The stash that fixed the dispatch to `ROWS_PER_TG = 2` made the output correct but dispatched 1280 TGs — 2× more work than necessary (each row computed by two adjacent simdgroups due to the overlap in the formula).

**Fix:** set both constants to `4`: shader `Q6K_ROWS_PER_TG = 4` and Rust `ROWS_PER_TG = 4`. Each TG covers 4 non-overlapping rows (sg_id 0..3), dispatches 640 TGs, correct output, optimal throughput.

**Result:** 78.7 tok/s, GPU fwd 10.8ms — **correct and faster than the broken HEAD**.

### P0 correctness blockers — status (2026-04-26)

1. **✅ q6k_matvec ROWS_PER_TG mismatch** — FIXED. Shader and Rust constants both set
   to 4. All 2560 rows now covered; dense model back to 78.7 tok/s. See entry above.

2. **✅ Mixed Q4_K/Q6_K QKV fused V path** — resolved 2026-04-26 (stale entry).
   The named test `q4k_q6k_qkv_proj_normed_matches_separate_norm_and_proj`
   passes against `q6k_matvec` at the original 512-hidden test geometry
   AND at production hidden=2560 (10 super-blocks/row). Added
   `q4k_q6k_qkv_proj_normed_matches_at_production_hidden` regression
   test pinning the larger shape so any future drift is caught at
   production K, not via a model-output bug report.

3. **MoE GPU dispatch: activation scratch not padded to `inter_padded` (open).**
   `gpu_moe_dispatch` dispatches expert down with `K = inter_padded` but the activation
   buffer is sized and offset-indexed with `inter`. For `moe_intermediate_size=704`
   (`inter_padded=768`), the down projection reads 64 floats beyond each expert's
   activation slice. Fix: allocate `top_k × inter_padded × 4` bytes, zero-fill padded
   tail, offset per expert by `inter_padded` (not `inter`).

4. **MoE GPU parity test coverage thin (open).**
   Existing tests cover CPU routing and prefill shape/finiteness but not
   `gpu_moe_dispatch` correctness for Q4_K experts, padded intermediates, or
   `valid_count < top_k`.

| Source | Gap | Status |
|---|---|---|
| **Kernel compute** | **~2.0ms** | gate+up compute-bound (K=2560 ALU-limited); open |
| **lm_head overhead** | **~0.5ms** | GPU argmax_partial (top_k=1) + GPU topk_partial K_TOPK=8 (top_k=5) shipped 2026-04-26 (`f32_topk_partial` shader, `f16_gemv_topk` / `q4_matvec_topk` wired into `lm_head_knn_backend`) |
| **Dispatch overhead** | **~0.5ms** | Mostly closed (374 vs Ollama ~272 dispatches) |

**Achievable targets:**
- Close kernel compute gap → **~87 tok/s**
- Close lm_head gap → **~85 tok/s**
- Close all remaining → **~95 tok/s** (~Ollama parity)

**Key findings from per-kernel profiler (`diag_profile_kernels`):**
- Gate+up is **COMPUTE-BOUND** at 272 GB/s (K=2560, 0.5625 B/elem, dequant-limited).
  Float4 dual-sub-block approach was tried and regressed — complex addressing offsets
  gains from ILP. Format-compatible vectorization remains the unsolved problem.
- q6k_matvec (down) is **bandwidth-bound** at ~315 GB/s (K=10240, 0.82 B/elem).
  ROWS_PER_TG=2 (64 threads/TG) improved it by ~5% via better occupancy.
- lm_head f32_gemv is near peak at 370 GB/s — the overhead is CPU-side (readback,
  sort). `f32_gemv_topk1` GPU argmax ships the fix for top_k=1 callers.

### #1 — Q6_K fused activation+down (closed — wrong fix, correct diagnosis)

**Status:** Benchmarked (2026-04-25). Not viable. Routing reverted.
Root cause of original regression identified and documented.

**What was tried:** Added threadgroup-memory caching of `gate`/`up`
per super-block so all 4 simdgroups in a TG share one device load
(128 threads × 2 values each). All 5 parity tests pass. But
`larql bench gemma3-4b-q4k-v2` showed 61–62 tok/s — identical to
the unfused-TG-cache attempt and identical to the regression without
TG caching. TG caching had zero effect.

**Root cause (corrected):** bandwidth was never the bottleneck.
gate/up = 80 KB total per dispatch — well within M3 Max GPU L2 cache.
All 640 TGs share the same gate/up data → L2 cache-hits from TG 2
onward. The real regression is GELU-tanh recomputation:

- Separated path: `geglu_gelu_tanh` kernel runs 10,240 threads,
  each computing one `tanh(gate[i])`. Total: 10,240 `tanh` calls.
- Fused path: inner loop computes `tanh(gate[i])` for every output
  row independently. At N=2560 output rows: 2,560 × 10,240 =
  **26.2 M `tanh` calls** — 2560× more than separated.

`tanh` is a transcendental function; GPU ALU cost dominates. The
saved dispatch + buffer round-trip (~0.2 ms) doesn't offset the
extra 2560× `tanh` work at production shape.

**Q4_K fusion wins for a different reason:** the all-Q4_K model
uses SiLU (`x/(1+exp(-x))`), not GELU-tanh. SiLU is cheaper than
`tanh`, so the recomputation overhead is smaller relative to the
heavier Q4_K dequant per cell.

**Remaining Q6_K opportunity:** optimise `q6k_matvec` throughput
directly (P0 #5 below) — currently 79 GE/s vs Q4_K 105 GE/s.
Alternatively: precompute `act[]` via a fast batch activation and
pass a float input to a future `q6k_matvec_f32in` kernel (avoids
the per-row `tanh` recomputation entirely while still fusing
dispatch). ~50 LOC new shader.

### #2 — Single encoder per token (done — was already implemented)

**Status:** The decode loop already uses ONE encoder for ALL 34 layers
(non-MoE path). The ROADMAP item was mislabelled — the actual overhead
is per-`dispatch_thread_groups` call (~5-8µs each), not per-encoder.
Current dispatch count: ~14 dispatches/layer × 34 = 476 dispatches/tok
= ~2.4-3.8ms of dispatch overhead. Reducing requires kernel fusion.

### #3 — Fused `rms_norm + QKV projection` for Q4_K/Q6_K path (open)

**Estimated gain: ~0.2ms/tok (1 saved dispatch × 34 layers × 5-8µs).**
Currently `encode_input_norm_and_qkv` runs two dispatches per layer:
`rms_norm_pipeline` → f32 norm_out buffer → `q4k_q6k_qkv_proj`.
The norm_out write/read is L2-cached (10 KB), so main saving is the
dispatch. A fused `rms_norm_q4k_q6k_qkv` shader:
- Phase 1 (all 128 threads cooperate): reduce `||h||²` / hidden
- Phase 2 (each simdgroup independently): matvec with inline `h[i] / rms * w[i]`
Effort: ~200 LOC MSL (cooperative reduction + two-format Q4K/Q6K paths).
The revised estimate is ~0.2ms (not 0.4ms — norm_out is L2-cached).

### #4 — LM head wrapper overhead (partial — heap done 2026-04-25)

**Remaining gain: ~0.1ms.** `backend_lm_head_topk`:
- ~~partial-sort 262k → top-k~~ → **min-heap done**: avoids 2MB Vec allocation,
  saves ~0.1ms (observed lm_head 2.38 → 2.27ms).
- GPU dispatch+commit+wait: ~200µs — reducible with async readback.
- Buffer readback (1 MB): ~150µs — async pipelining needed.
- Remaining overhead after heap: ~0.35ms.
The GPU kernel itself (1.55ms) is the irreducible floor.

### #5 — `q6k_matvec` full rewrite (done 2026-04-25)

**Total gain: ~3ms/tok / ~20% / +10 tok/s** (62→72 tok/s), in two phases:

**Phase A — 4-element batching** (+7 tok/s, 62→69):
Scalar inner loop used `(i & 3u) << 1u` — a runtime shift the GPU can't hoist.
Restructured to 4-element groups with compile-time hi2 shifts (0,2,4,6), 16
preloaded scales, and ROWS_PER_TG=8. All tests pass.

**Phase B — inter-superblock interleaving + X preload + deferred scale** (+3 tok/s, 69→72):
Adapted the llama.cpp `kernel_mul_mv_q6_K_f32_impl` strategy to LARQL's linear
Q6_K layout (GGUF's transposed layout can't be ported directly — different format):
- `ix = lane & 1` → adjacent lanes process alternate superblocks, letting DRAM
  serve two memory banks in parallel.
- `xl[16]` preloaded before weight reads → X fetches overlap weight byte loads.
- Deferred scale: `acc += d*sc * (unscaled_sum_4_elems)` — 4× fewer scale mults.
- ROWS_PER_TG dropped from 8→4 (128 threads/TG) → halved register pressure,
  2× more concurrent TGs, better latency hiding on LPDDR5X.
Effective Q6_K bandwidth: ~322 GB/s (up from ~294 GB/s).

### #5b — `q4k_matvec` llama.cpp-style rewrite (open — see #6)

Folded into #6 below with updated size estimate.

---

### q6k_matvec ROWS_PER_TG — correctness fix (2026-04-26)

**Corrected to ROWS_PER_TG=4** for both shader and Rust dispatch constant. See "P0
correctness blockers" entry above for full diagnosis. The previous ROWS_PER_TG=2
ship note was based on a mismatch that appeared to gain performance by skipping half
the rows — real performance at correct ROWS_PER_TG=4 is **78.7 tok/s, GPU fwd 10.8ms**,
better than any previous measurement.

### f32_gemv_topk1 GPU argmax (done 2026-04-26, infrastructure)

New `MatMul::f32_gemv_topk1` trait method: runs gemv + GPU argmax in one command
buffer, reads back only 8KB (1024 partial results) instead of 1MB (262K scores).
Saves ~0.33ms for top_k=1 callers. Implemented on MetalBackend. Main decode loop
uses the KNN lm_head path (top_k=5 → KNN fires first), so this doesn't yet
benefit the bench. Useful for non-KNN models and future greedy-decode APIs.

### Q4_K `sumy` precompute (2026-04-26, measured 2026-04-27 — no measurable gain)

Separated the X-sum used in the min-correction term from the FMA dot-product
loop in `q4k_matvec` and `q4k_ffn_gate_up`. Previously both shared one loop
(`dot_acc` and `sum_acc` accumulated together); now a dedicated `sumy` pass
runs first, leaving the dot loop as a pure FMA chain the compiler can
schedule without interleaved additions. Applied to both the standalone matvec
and the fused gate+up shader.

**Measured 2026-04-27 on the all-Q4_K extract (`gemma3-4b-q4k-downq4k`),
3 runs each, identical bench setup:**

| Shader form | Run 1 | Run 2 | Run 3 | GPU fwd |
|---|---|---|---|---|
| With `sumy` precompute (split loops) | 71.7 | 72.3 | 72.1 | 12.67–12.74 ms |
| Without (combined `dot_acc` / `sum_acc`) | 72.4 | 71.6 | 72.9 | 12.62–12.77 ms |

Difference is within run-to-run variance — the Apple Silicon shader compiler
schedules the combined loop just as well as the split form. Kept the split
version anyway since it's cleaner code for future readers; no perf regression
either direction. Worth flagging that this micro-optimisation didn't pan out
so future "split the FMA chain from the sum" attempts know the answer.

### #6 — Q4_K kernel optimization (explored 2026-04-26, blocked by ALU bound)

**Tried:** (a) inter-superblock interleaving (ix=lane&1 stride-2, already applied).
(b) 2 rows per simdgroup + 64 threads/TG (REGRESSED: halves total wavefronts,
  hurts more than X-sharing helps for K=2560).
(c) llama.cpp uint16 `float4` trick — INCOMPATIBLE: llama.cpp uses a
  transposed nibble layout (qs[b] lo=elem b, hi=elem b+32) while LARQL uses
  linear (qs[b] lo=elem 2b, hi=elem 2b+1). The uint16 accumulation trick only
  works for the transposed layout.

**Root cause unchanged:** K=2560 fits in GPU L1 cache (1440 bytes/row). The
weight read bottleneck is not the X reads but the ~89 MB/layer weight data,
and the main gap vs Ollama is in ALL-operations bandwidth (322 vs ~414 GB/s).

**Remaining Q4_K opportunity:** `sumy[]` precomputation (saves 16 additions
per superblock for the min correction term) and profiling to understand the
full ~2ms kernel gap. For K=8192 (Wo, 4608 bytes/row = DRAM-bound),
inter-superblock interleaving at stride 2 is already applied; stride-4
(ix=lane/8) would add more DRAM bank parallelism.

**Root cause of limited gain:** All Q4_K matvecs in Gemma 3 4B use K=2560 as
input dimension (hidden size). K=2560 → 10 superblocks × 144 bytes = 1440 bytes
per row — fits entirely in GPU L1 cache. The old lane-stride approach had 22/32
idle lanes for K=2560, but L1-cached superblock data hid that inefficiency. The
inter-superblock optimization helps primarily when K is large enough that
superblock data spills to DRAM — which is why Q6_K down (K=10240, 8400 bytes/row,
21.5 MB total) got a much larger gain.

**Potential remaining Q4_K gains:** The llama.cpp approach uses `yl[]/yh[]`
preloading + `float4 acc1/acc2` vectorized accumulation. For the output dimension
(N=10240 for gate/up), more TGs may help via better GPU saturation. But the
fundamental bottleneck for Q4_K with K=2560 is now something else.

**Estimated gain: ~1.0–1.5ms/tok.** The Q4_K kernel handles:
- Wq (8192×2560) + Wk (4096×2560) + Wv fused QKV: 26.3 MB/layer × 34 = 895 MB
- Wo (2560×8192): 11.8 MB/layer × 34 = 401 MB
- W gate+up (10240×2560 ×2, fused): 29.5 MB/layer × 34 = 1003 MB
- **Total Q4_K data: ~2300 MB/token** (vs Q6_K's 1023 MB — more than double)

The old sub-block-stride kernel hasn't been touched. Applying the same
inter-superblock + preload + deferred-scale treatment as Q6_K should
close a proportionally larger gap.

**llama.cpp Q4_K algorithm** (`kernel_mul_mv_q4_K_f32_impl`):
```
ix = tiisg / 8     → 0..3: which of 4 parallel superblock groups
it = tiisg % 8     → 0..7: position within the group
iq = it / 4        → 0 or 1: low or high sub-block
ir = it % 4        → 0..3: which of 4 groups within sub-block

for (ib = ix; ib < nb; ib += 4):   // stride 4, processes 4 superblocks at once
    yl[16], yh[16] = preload X values for this superblock
    sumy[4]        = precompute X sums (for the min correction term)
    for row in 0..nr0:             // nr0=2: 2 rows per simdgroup
        float4 acc1, acc2 = { 0 }  // vectorized accumulation
        FOR_UNROLL (i=0..3):
            acc1[0..3], acc2[0..3] += nibble × yl/yh
        sumf[row] += d × (acc1 scale corrections) - dmin × (sumy correction)
```

Key differences from LARQL's current `q4k_matvec`:
1. **4 parallel superblock groups** (ix=0..3): all 4 groups run simultaneously,
   4× as many concurrent DRAM reads vs LARQL's 1 per stride.
2. **`yl[16]/yh[16]` preloaded**: X reads issued before weight bytes.
3. **`sumy[4]` precomputed**: the `Σ x[i]` term for min correction is
   accumulated once per superblock per ix-group, not per nibble.
4. **`float4 acc1/acc2`**: 4-wide vectorized accumulation — compiler can emit
   packed FMAs for 4× instruction-level throughput.
5. **2 rows per simdgroup** (`nr0=2`): both rows share the same superblock
   reads, amortising preload cost across 2 outputs.

**LARQL's Q4_K format matches GGUF** (same 144-byte block structure: d/dmin
f16 + 12-byte packed scales/mins + 128 bytes of 4-bit nibbles). llama.cpp's
algorithm can be ported directly without format translation.

**Effort:** ~200 LOC MSL. Need to adapt the `yl[]/yh[]` preload pattern
for LARQL's block layout, handle the `fused_q4k_qkv` path (3 output
matrices), and update `q4k_ffn_gate_up` to use the same interleaving.

### #7 — Dispatch fusion: consolidate per-layer ops (open)

**Estimated gain: ~1.0ms/tok** (saves ~200 dispatches at ~5µs each).

Current per-layer dispatch count (~14 for Gemma 3 4B):
1. `rms_norm` (input norm)
2. `q4k_q6k_qkv_proj` (QKV projection)
3. `qk_norm` — Q heads
4. `qk_norm` — K heads
5. `rope_at_pos_batched` — Q heads
6. `rope_at_pos_batched` — K heads
7. `kv_append`
8. `kv_attend`
9. `o_proj` (O projection)
10. `residual_norm` (post-attention residual + FFN norm)
11. `q4k_ffn_gate_up` (fused gate+up)
12. `geglu_gelu_tanh` (activation)
13. `q6k_matvec` (FFN down)
14. `residual_add` (post-FFN)

Three fusions with clear wins (each saves 34 dispatches = ~0.17ms):

**7a — Fused QK-norm Q+K** ✅ done 2026-04-25 (+0.17ms recovered):
New `qk_norm_qk` shader dispatches total_heads = q_heads + kv_heads in one
call; TG index selects Q buffer + q_weight vs K buffer + k_weight.

**7b — Fused RoPE Q+K** ✅ done 2026-04-25 (+0.17ms recovered):
New `rope_at_pos_batched_qk` shader: grid `(rope_pairs, q_heads+kv_heads, 1)`;
thread `h < num_q` selects Q buffer, else K buffer.

**7c — Fused input norm + QKV projection** ✅ done 2026-04-25:
New `q4k_q6k_qkv_proj_normed` kernel: all 128 threads cooperatively reduce
`||h||²` in Phase 1 (barrier), then each simdgroup runs its matvec with inline
`h[i] * rms * (offset + norm_w[i])`. Fires when format is Q4_K Q/K + Q6_K V,
standard RMS norm, no bias (Gemma 3 4B production).

**7e — Fused residual_norm + residual_add** ✅ done 2026-04-25:
New `residual_norm_store` kernel writes both `ffn_norm_out` (normed FFN input)
and `h_post_attn` (raw sum for post-FFN add) in one pass. Replaces the
`residual_norm + residual_add` two-dispatch pair in the Q4_K hot path.

**7d — Fused GEGLU + down** (~0.17ms):
Dispatches 12+13 can be merged for Q4_K down (already done). For Q6_K down,
fusion was attempted but regressed due to GELU-tanh recomputation cost
(see #1 closed). Not viable unless activation is precomputed separately.

---


## P0: Structural cleanup (open)

From the 2026-04-25 codebase review. Most ship in the same time
window as the perf wins above; some unblock cleaner perf work.

### #6 — Magic-string kernel names on non-tiled shaders (DONE)

Added `ShaderKernel` trait + `get_shader_pipeline::<T>()` to
`kernel/traits.rs`; 31 magic strings eliminated. Each shader now
exports a compile-time `NAME` constant — renaming a shader causes a
compile error rather than a silent runtime panic.

### #7 — `QuantFormat` pattern-match spread (partial — classifiers shipped 2026-04-27)

**Classifier helpers shipped:** `QuantFormat::is_q4k_family()` /
`is_q4kf()` / `is_legacy_q8()` on `pipeline.rs`. The most-duplicated
predicate (`format == Q4_K || == Q4_KF || == Q6_K`, repeated verbatim
in `decode/mod.rs` ×2 and `decode_hybrid.rs` ×1) collapses to a single
method call. Adding a future Q4_K-style format updates one classifier,
not 3+ OR-chains. Pinned by `quant_format_classifiers` test.

**CPU-side `FormatRoute` registry SHIPPED (2026-07-22, DEC prep).**
`src/quant_route.rs`: a per-format dispatch record (`dequant_padded` /
`q8k_matvec` / `quant_matmul` fn pointers) resolved by
`QuantFormat::route()` — one match arm per format. The string-keyed CPU
dispatch sites (`kquant_forward::dequant::dequantize_matrix`,
`q4k_q8k_matvec_parallel`, `ffn::weight::{quant_matmul, block_bytes}`,
`kquant_forward::cached`'s direct-matvec gates and down-stride lookup)
now resolve `tag → QuantFormat → FormatRoute` instead of each holding a
local `"Q4_K" | "Q6_K"` match. Unknown-format contract documented on the
module and enforced: seam parsers error loudly, capability probes return
`None`/`false` and fall back, kernel entry points panic instead of
silently zero-filling (`q4k_q8k_matvec_parallel`'s `_ => return` bug —
dec-readiness §1 class — is fixed, with `should_panic` regressions).
Adding MXFP4 (DEC-6a) = one variant + one route arm + kernels; formats
with side metadata follow the `ternary_matvec` parallel-path template.

**Metal-side route DEFERRED.** The original intent (`F32Input { … }` /
`Q8Input { … }` with the `match QuantFormat::*` confined to one
constructor in `metal/stages/quant_matvec.rs`) remains open — the Metal
crate keeps its own enum dispatch (`pick_qkv_route` already anticipates
an FP4 family). Revisit when a new format actually lands a shader.

### #8 — `Pipelines` struct asymmetry (DONE)

All fields in `metal/stages/quant_matvec.rs::Pipelines` now use
`&KernelHandle`; geometry drift is now a compile error rather than
a silent dispatch mismatch. ~100 LOC mechanical migration across
callsites.

### #9 — `FullPipelineLayer` 63 pub fields (partial — `Default` shipped 2026-04-27)

**Test ergonomics fix shipped:** `FullPipelineLayer` and `QuantWeight` now
implement `Default`, so test code uses
`FullPipelineLayer { wq, ..Default::default() }` instead of spelling out 30
fields. The pre-existing `minimal_layer` helper collapsed from 30 lines to
10. New `default_layer_accepts_local_borrows_via_spread` test pins the
pattern for future tests (verifies `..Default::default()` reborrows the
`'static` defaults at the caller's stack-local lifetime — typical Rust
HRTB territory but worth a test since it's a non-obvious property).

**Full sub-struct split DEFERRED.** The roadmap intent
(`LayerWeights` / `LayerNorms` / `LayerArchParams` / optional `MoeBlock`)
is a 30+ caller-file refactor. Doing it concurrently with the in-flight
MoE struct refactor (ongoing in this branch) risks merge conflicts on
`pipeline.rs`. Pick this back up once MoE work settles. The `Default`
impl removes the immediate test pain — that was the user-visible cost
of #9.

### #10 — `dispatch_full_pipeline` 30+ params (open)

Even after stage extraction the signature is unreadable. Same
`Pipelines`-struct treatment as `stages/quant_matvec.rs` — bundle
the pipelines and norms into a `FullPipelineRefs<'_>` context.

### #11 — `compare_*.rs` examples consolidation (open)

5 `compare_*.rs` files (~1400 LOC) overlap heavily. Particularly
`compare_decode` (195) and `compare_pipeline` (240). Consolidate to
one with subcommand flags.

### #12 — `ProfileTimings` producer (open)

`ProfileTimings` struct + `format_summary` shipped (2026-04-25) but
no code populates `gate_up_ms` / `down_ms`. Wire commit/wait
boundaries through `decode_token_with_moe_fn` — completes the
diagnostic that replaced the deleted 567-LOC `decode_profile.rs`.

---

## P0.5: Gemma 4 26B A4B correctness

### ✅ CPU MoE decode interleave — DONE (2026-04-20)
GPU dense FFN + CPU MoE per layer. See `metal/decode/moe_combine.rs`
for the outer combine math (HF Gemma 4 has three post-FFN norms per
MoE layer: `_1` on dense, `_2` on MoE, and un-suffixed outer on the
sum — only the un-suffixed one gets `layer_scalar` applied to the
whole layer output after residual add).

### ✅ Full end-to-end correctness — DONE (2026-04-24)
Four coordinated fixes were needed (earlier "working" claim was only
approximate — the Metal output was degenerate token repetition on a
cold vindex). All verified against HF bf16 via layer-by-layer
residual-cosine diff in `metal/decode/diag.rs::ResidualDump` +
`/tmp/hf_residuals.py` + `/tmp/diff_residuals.py`.

1. **Row-padded Q4_K / Q6_K storage** for matrices whose inner dim
   isn't a multiple of 256. 26B A4B's `intermediate_size=2112` gives
   8.25 super-blocks per row; old extraction stored contiguously and
   the matvec shader read wrong bytes for every `down_proj` row past
   row 0. `pad_rows_to_256` in `larql-vindex/format/weights/write.rs`
   per-row-pads to the next 256-multiple; runtime dispatches
   `down_proj` with `K = inter_padded` (see `metal/decode/mod.rs`).
   `act_buf` allocated to `inter_padded * 4` bytes and zero-inited so
   the trailing columns contribute nothing. Aligned models
   (`inter_padded == inter`) are unchanged.
2. **Parameter-free router RMSNorm** — HF `Gemma4TextRouter.norm` has
   `with_scale=False`; no weight tensor exists on disk. Trait method
   `moe_router_norm_parameter_free()` + `rms_norm_no_weight` branch
   in `cpu/ops/moe/forward.rs`. Also added `router.scale *
   hidden_size^-0.5` multiplier (HF's `scalar_root_size`).
3. **Outer `post_feedforward_layernorm.weight`** (un-suffixed) added
   to extraction + wired through `FullPipelineLayer.moe_outer_post_norm`.
   Distinct from the `_1` dense-branch norm that was previously being
   double-applied.
4. **`layer_scalar` applied to the whole layer output** after residual
   add (`new_h *= layer_scalar`) — matches HF's `hidden_states *=
   self.layer_scalar` at the end of `Gemma4TextDecoderLayer.forward`.
   Prior code folded it into the outer-norm scale (14× magnitude
   error, collapsed the model to degenerate output).

Artifacts for future regression checks:
- `crates/larql-cli/examples/patch_down_proj.rs` — surgical vindex
  patcher (re-quantises `down_proj` rows with per-row padding).
  Avoids re-extracting 42 GB when the extraction side is fixed.
- `crates/larql-compute/src/metal/decode/diag.rs::ResidualDump` —
  env-gated (`LARQL_DUMP_RESIDUALS=<path>`) binary dump of every
  layer's `layer_in` / `h_post_attn` / `layer_out` for HF-ref diff.
- `crates/larql-inference/tests/test_arch_golden.rs` — architecture
  regression suite with one `#[test]` per `(arch × backend)`,
  skip-if-missing for vindexes. Caught the broken output immediately
  and flagged which architecture-specific change broke it.

### Batched MoE prefill — **SHIPPED (2026-04-26)**

Replaced the O(seq_len × num_layers) token-by-token decode loop with a
batched approach: `dispatch_full_pipeline` now accepts an optional
`moe_fn: Option<&mut dyn FnMut(usize, &[f32], &mut [f32])>` callback.
When the callback is present and a layer has MoE, the function commits
the GPU command buffer after that layer's dense FFN, calls the closure
(which runs CPU experts for all seq_len positions and applies outer norm
+ layer_scalar), then restarts the command buffer for the next layer.

**Measured on Gemma 4 26B A4B (5-token prompt, 15 warmup / 30 tokens, M3 Max):**

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Prefill | 1889ms | 1297ms | **−31%** |
| Decode GPU fwd | 334ms/tok | 246ms/tok | **−26%** |
| Decode tok/s | 2.9 | **3.9** | **+35%** |

**Why:** 5-token prefill now uses 26 GPU commits (one per layer) vs 130
(5 positions × 26 layers). Batching all positions per layer also improves
weight cache utilisation. GPU layer_scalar skipped for MoE layers in the
dispatch; the callback applies it correctly after combining dense + MoE.
`kv_copy::populate_kv_one_layer` added for per-layer KV cache population.

### GPU expert dispatch — Phase 2: pre-allocated staging buffers (DONE; baseline corrected 2026-05-02)

**Status**: SHIPPED. `MoeScratch::new` (in `metal/moe_dispatch/`)
pre-allocates all expert staging buffers once per model shape and caches
by `(top_k, hidden, intermediate_size)` on the backend. Per-layer
`gpu_moe_dispatch_with_scratch` only memcpys expert bytes into existing
buffer contents — no `bufs.output(...)` calls in the hot path.

**Measured 2026-05-02 (post Phase 2 + dispatch-geometry fix)**:
- 26B A4B Metal: **19.4 tok/s** (was 5.1 pre-2026-05-02 — bug-locked under
  the dispatch-geometry mismatch in the same `moe_dispatch/` sites; the
  "Phase 1 shipped 5.1 tok/s" baseline was attributing the bug-locked
  number to Phase 1, which was wrong).
- GPU-only ceiling (`LARQL_SKIP_MOE=1`): **56.8 tok/s**.
- Remaining headroom (19.4 → 56.8): genuine expert dispatch work
  (240/token = 8 experts × 30 layers × 1 fused gate+up + 1 GEGLU + 1 down)
  + 30 commit/wait syncs. Real shader/dispatch work, not allocation.

The pre-2026-05-02 "Phase 2 expected ~4× gain" estimate happened to
match the actual 5.1 → 19.4 perf jump — not because Phase 2 was the
load-bearing fix, but because the dispatch-geometry mismatch was masking
the same ~4× of real perf as 240 broken expert dispatches. With both
fixes in, the new ceiling estimate for 26B A4B is ~25–30 tok/s if the
expert-dispatch fusion levers in `larql-server/ROADMAP.md§F-LOCAL-MOE`
land.

**Scope (single landing):**

1. **Pre-allocate persistent staging buffers** in `decode_token_q4k_moe`
   (`metal/moe_dispatch/`). Sizes are constants of `(top_k, inter_padded,
   hidden, row_bytes, down_row_bytes)` — known once per decode, not per layer.
   Buffers to pre-allocate (all `StorageModeShared` so CPU writes via
   `buffer.contents()`):
   - `gate_buf`: `top_k × inter × row_bytes`
   - `up_buf`: `top_k × inter × row_bytes`
   - `down_bufs`: `top_k` × `[hidden × down_row_bytes]` (per-expert; experts
     come from different mmap pages, so K independent staging buffers — not
     a single concatenated one).
   - `g_out`, `u_out`: `top_k × inter × 4`
   - `act_buf`: `top_k × inter_padded × 4`, zero-initialised once
   - `expert_outs`: `top_k × hidden × 4`

   `gpu_moe_dispatch` becomes `gpu_moe_dispatch_with_scratch(scratch, ...)`;
   the per-call body just memcpys expert bytes into the existing buffer
   contents and dispatches. No `self.bufs.output(...)` calls inside the
   per-layer hot path.

2. **Fix activation-stride bug** (P0 correctness blocker #3 in this file).
   Today: `act_buf` allocated at `valid_count × inter_padded × 4`, but the
   geglu kernel writes linearly at stride `inter`. For
   `moe_intermediate_size` not a multiple of 256 (e.g. Gemma 4 26B's 2112 →
   inter_padded 2304), expert `e>0` reads stale/garbage floats. Fix:
   dispatch `geglu_gelu_tanh` per expert with `g_out`/`u_out` linear offset
   `e × inter × 4` and `act_buf` strided offset `e × inter_padded × 4`. K
   extra dispatches per layer (top_k=8 → 8 small dispatches) but each is
   ~5µs — negligible vs the ~120ms allocation overhead this PR removes.
   Alternative: stride-aware kernel — defer if perf demands it post-bench.

3. **Borrow expert slices instead of `to_vec()`** (host-copy churn). Today
   `larql-inference::layer_graph::generate::gpu` allocates two
   `Vec<u8>` per expert per layer (~2.2 MB heap-copy × 30 layers × 8 experts
   per token). Change `get_expert: impl Fn(layer, expert) -> Option<(Vec<u8>,
   Vec<u8>)>` to return `Option<(&[u8], &[u8])>`. Lifetime-bound to the
   weights mmap — borrow lasts only across the `gpu_moe_dispatch` call.
   Updates `decode_token_q4k_moe` signature + the inference-side caller.

4. **Add parity test** `gpu_moe_dispatch` Q4_K experts with
   - aligned `inter` (e.g. 768),
   - misaligned `inter` requiring padding (e.g. 704),
   - `valid_count < top_k` (some experts return None),
   against CPU MoE reference.

**Acceptance criteria**:
- `cargo test -p larql-compute --features gpu` green (existing + new parity).
- `larql bench gemma4-26b-a4b` ≥ 15 tok/s (3× from baseline 5.1).
- No regression on `larql bench gemma3-4b-q4k-v2` (dense path untouched).

**Out of scope for this PR**: dense kernel optimisation, fused
QKV V-path correctness blocker (#2), the expert-bytes-→-Metal-buffer copy
itself (already a single memcpy via `contents()` ptr; can't shrink further
without DMA-side weights, which is a larger refactor).


**Root cause of remaining gap.** `gpu_moe_dispatch` calls `self.bufs.output()` ~10 times per
MoE layer to allocate gate, up, per-expert-down, activation, and output Metal buffers.
With 30 MoE layers × ~10 allocations = 300 Metal buffer allocations per decode token,
each allocation of a 1–9 MB `StorageModeShared` buffer costs ~0.4ms on M3 Max.
**Total: ~120ms/token in allocation overhead** (measured: 194ms total − ~40ms compute − ~30ms syncs).

There is also avoidable host-copy churn before those Metal allocations:
`larql-inference::layer_graph::generate::gpu` calls
`weights.get_layer_entry_bytes(...)?` and immediately `to_vec()`s both
expert slices before `gpu_moe_dispatch` copies them into Metal staging.
For Gemma 4 26B A4B, this is 30 layers × top_k=8 × roughly 2.2MB of
heap copies per decode token. Phase 2 should change the API to pass
borrowed mmap slices (`&[u8]`) through the closure and copy exactly once
into reusable Metal buffers.

**Fix.** Pre-allocate all staging buffers once before the layer loop in
`decode_token_q4k_moe` (in `metal/moe_dispatch/`), identical to the pattern that
eliminated 550→20 allocations in `decode_token_with_moe_fn` (see ship log below):

```
Pre-allocated once:
  gate_buf:     [top_k × inter × row_bytes]  (gate Q4K staging)
  up_buf:       [top_k × inter × row_bytes]  (up Q4K staging)
  down_bufs:    [top_k] × [hidden × down_row_bytes]  (per-expert down Q4K staging)
  act_buf:      [top_k × inter × 4]  (f32 activations after GELU)
  expert_outs:  [top_k × hidden × 4]  (f32 expert outputs)
```

Sizes are constant per model (determined by `moe.intermediate_size`, `moe.top_k`,
`hidden`). The pre-allocated buffers are reused for all 30 layers via write-in-place
to `buffer.contents()` pointers.

**Effort**: ~1 session. No new shaders needed — just restructure the buffer lifecycle
in `decode_token_q4k_moe`.

### Fix `dispatch_full_pipeline` layer_scalar
**Effort**: Low
**Status**: Not started — current models (Gemma 3 4B) not affected

`dispatch_full_pipeline` applies `layer_scalar` to `h_bufs[l+1]`
(full residual = `h_post_attn + ffn_delta`) instead of just the FFN
delta. Correct formula: `h_post_attn + scalar * ffn_delta`.

Fix: pass `(scale_pipeline, scalar)` into
`residual::encode_post_ffn`, apply scalar to the normed FFN buffer
before the residual add. Call sites: `ops/full_pipeline/`,
`tests/test_metal_shaders.rs:2696,2748` — add `None` for non-scaling.

Not urgent: Gemma 3 4B has `layer_scalar = 0.0` (no scaling); Gemma 4
26B uses the MoE callback path which applies layer_scalar correctly.

## P1: Production Hardening

### Streaming prefill
**Effort**: Medium  
**Status**: Prefill pipeline exists but uses CPU for KV cache population

The `prefill_q4` GPU pipeline runs the forward pass. KV cache is populated via CPU `prefill_with_kv` afterward. Integrate KV cache writes into the GPU pipeline to eliminate the CPU roundtrip.

### Dynamic KV cache sizing
**Effort**: Low  
**Status**: Fixed at 4096 max_seq

Current KV cache allocates for 4096 tokens at creation. Need dynamic growth or configurable max_seq for long-context inference.

---

## P1.5: Platform expansion

**Prerequisite: performance parity with Ollama on Metal first.**
These items are sequenced after the Metal gap closes (~1.0× vs Ollama),
so platform users start with a competitive baseline.

### Linux support
**Effort**: Medium  
**Status**: In progress — compute crate CI added; downstream crate CI still open.

larql-compute has a CPU baseline plus a macOS-only Metal backend. The
`ComputeBackend` trait and CPU fallback compile without Metal at the trait
level. Current guardrail: tests, examples, benches, and the exported
`metal::` module are gated on `#[cfg(all(feature = "gpu", target_os =
"macos"))]`, so Linux `--all-features` checks do not try to import Metal.
Done for `larql-compute`: Linux CI installs OpenBLAS, checks default and
all-feature builds, runs clippy, and runs the fast compute test split. Gaps:

- `larql-cli` / `larql-inference`: a small number of `metal`-feature
  entrypoints have been guarded, but they still need native Linux CI coverage.
- Add build-system CI for the downstream CPU stack (`larql-inference`,
  `larql-cli`) once the OpenBLAS setup is proven stable in compute CI.

Expected result: `cargo build -p larql-cli` (no features) works on
Ubuntu 22.04 / 24.04 x86_64 and aarch64, with CPU-only decode.

### Windows support
**Effort**: Medium  
**Status**: In progress — compute crate CI added; full CLI/inference support still open.

Similar to Linux plus:
- Path handling: a small number of `std::fs::File::create` /
  `PathBuf::join` calls use `/tmp/` or Unix paths — audit and fix.
- Symbol visibility: `extern "C"` symbols from BLAS need checked on
  MSVC with vcpkg OpenBLAS. `larql-compute` CI now exercises this first.
- Extend Windows CI beyond compute after native OpenBLAS linking is confirmed.

Expected result: `cargo build -p larql-cli` works on Windows 11
x86_64 (MSVC toolchain) with CPU-only decode.

### CUDA backend (G-ladder; see ADR-0024)
**Effort**: Large  
**Status**: G0 decided 2026-07-24 (ADR-0024): sibling crate
        `larql-compute-cuda`, cudarc + NVRTC runtime-compiled kernels,
        phase-1 ports the Metal f32-FMA reference semantics (~24 shader
        files), dp4a int8 as opt-in phase 2. NOTE the earlier framing here
        ("re-land from earlier PR") was investigated and corrected: PR #53
        contained cuBLAS-SGEMV-over-CPU-dequant only — no custom kernels,
        stub DecodeBackend. Nothing to cherry-pick; its no-SDK dynamic-
        loading property is inherited via cudarc instead.

An earlier PR implemented CUDA kernels but was not merged. Current
`ComputeBackend` trait supports the interface; the Metal decode loop
(`decode_token_with_moe_fn`) provides the implementation template.

Scope to re-land:
1. `cuda::` module gated on `--features cuda` (mirrors `metal::` module).
2. Buffer management via `cuMemAlloc` / `cuMemcpy` under unified-memory
   or explicit device buffers.
3. Kernel ports: `q4k_matvec`, `q6k_matvec`, fused attention (FlashAttention
   or a clean CUDA port of the Metal `kv_attention` kernel), `rms_norm`.
4. `DecodeBackend` impl wired into `decode_token_with_moe_fn`.
5. `larql bench --backends cuda` path in the CLI.

Target: competitive with llama.cpp on a single A100 / H100 for
Gemma 3 4B and Gemma 4 27B (the models already validated on Metal).

## P2: Research

### Q4_K FFN pipeline (end-to-end) — DONE
**Effort**: Medium  
**Status**: ✅ Complete (2026-04-07)

Vindex loader (`load_interleaved_q4k`), inference wiring (`predict_honest` prefers Q4_K FFN), and format tag propagation through `FullPipelineLayer` all wired. When `interleaved_kquant.bin` exists, Q4_K format flows through to compute shader dispatch.

### simdgroup_multiply_accumulate for tiled matmul
**Effort**: Large  
**Status**: Research

Apple Silicon has dedicated matrix hardware. For batch inference (seq>1), tiled Q4_K matmul using simdgroup_matrix operations could significantly speed up prefill. Not useful for seq=1 decode (matvec, not matmul).

### Fused layer kernel
**Effort**: Large  
**Status**: Research

Single kernel per layer: norm → QKV → attention → O → residual → norm → FFN → residual. Eliminates ALL inter-op dispatch overhead. Requires careful register management and threadgroup synchronization.
