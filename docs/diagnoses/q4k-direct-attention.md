# Q4K-direct attention — recon map (task #16, step 1)

Deliverable for Active Sequence item **5b** / task **#16, step 1**: *recon, not
build*. Map where the decode-attention ~28% is paid, what the half-existing
`q4_attention_proj` covers **and doesn't**, and the parity baseline — so step 2
opens as a scoped clean-swap (or a clearly-flagged "kernel is only projections")
instead of a reconstruction. Source-verified 2026-05-30 against the
`fix/moe-setup-pure-moe` tree.

Companion docs: `docs/diagnoses/remote-moe-bottlenecks.md` (the decode-stage
split that gives the 28%), `docs/diagnoses/walk-ffn-performance.md` (why this is
the only speed lever left).

---

## Bottom line up front

1. **It is a projection swap, not a function swap.** Of the attention block, only
   the **four Q/K/V/O projections** are Q4K-accelerable. RoPE, QK/V-norm, the GQA
   decode step (scores + softmax + weighted-V), the KV-concat memcpy, and the
   residual all **stay f32 and unchanged** — and must, for parity. So the build is
   "swap the four `dot_proj_gpu` f32-BLAS calls for Q4K-direct matvecs reading the
   index bytes, leave everything else byte-identical."

2. **The docs name the wrong function for the 28%.** The roadmap/diagnosis say
   "replace the f32 `run_attention_with_kv_backend`." That function is the
   **prefill / KV-populate** path (TTFT, ~42 ms/prompt-token). The steady-state
   **decode 28%** is paid in a *different* function, the per-token
   **`run_attention_block_decode_step_backend`**. They are parallel edits sharing
   the same projection helper — but if step 2 naively edits the named function it
   optimizes prefill and leaves the 28% untouched. **Target the decode-step
   function for the 28%; do the prefill one too for the TTFT win.**

3. **`q4_attention_proj` is the wrong primitive and is not really tested.** It
   guards on `supports_quant(Q4_K)` but calls `q4_matvec` — the **Q4_0**
   legacy-block kernel — and its only test feeds synthetic **Q4_0** bytes and
   asserts nothing numeric. Production attention weights are stored **Q4_K**
   (super-block). Q4_K bytes through the Q4_0 kernel = wrong byte stride = garbage.
   **Do not wire `q4_attention_proj`.** The correct, production-grade primitive
   already exists: `CpuBackend::q4k_matvec` (f32-input, rayon-parallel,
   parity-tested vs dequant→matmul). Prefer `quant_matvec(format, …)` so the
   per-matrix format tag (Q4_K vs Q6_K) dispatches correctly.

4. **The 28% is unsplit and context-flattered.** `record_attn` wraps the *whole*
   attention block, so the projection-vs-GQA share inside the 28% is **currently
   unmeasured**. And the 28% was measured at short context (33-token prompt + 12
   decode); the GQA step + KV-concat grow **linearly with context length** and are
   **not** Q4K-accelerable, so the addressable fraction *shrinks* as context grows.
   **Cheap next probe before building: a projection-vs-GQA split timer** — sizes
   the win and avoids the #24 build-then-measure trap.

---

## Where the 28% is paid — function map

Measured split (`docs/diagnoses/remote-moe-bottlenecks.md`, `LARQL_DECODE_STAGES=1`,
26B, standard engine, 2 shards, localhost, prefill + 12 decode tokens):
attention **1051 ms / ~28%**, all client-side f32 BLAS.

The recording site and the dispatch chain (source-verified):

```
kv_decode_step_via_dispatch                       larql-inference/src/kv_dispatch/helpers.rs:135
  per layer:
    let _t = Instant::now()
    backend.attention_step(...)                   helpers.rs:156   ← the timed call
    record_attn(_t.elapsed())                     helpers.rs:164   ← the 28% accumulator
        │
        └─ CpuBackend::attention_step             larql-compute/src/kv_dispatch/cpu.rs:274
             └─ run_attention_block_decode_step_backend   larql-compute/src/attention/decode.rs:111  ★ the 28% lives here
```

Prefill (TTFT, a *different* function — the one the docs name):

```
backend.attention_prefill                         larql-compute/src/kv_dispatch/cpu.rs:304
  └─ run_attention_with_kv_backend                larql-compute/src/attention/gpu.rs:158
```

Both call the same f32 projection helper:

```
dot_proj_gpu(a, b, backend)                       larql-compute/src/backend/helpers.rs:12
  = backend.matmul_transb(a, b)  (or ndarray a·bᵀ)   — f32 × f32 ONLY; requires b already dequantized to f32
```

**The tax.** Before either path runs, the engine pre-dequantizes all Q/K/V/O Q4K
bytes → f32 into `weights.tensors`:

```
ensure_attn_tensors_dequantised(weights, index)   larql-inference/src/vindex/dequant.rs:35
  per layer: dequantize_matrix(index.attn_kquant_layer_data(layer)[i], fmt, rows, cols) → weights.tensors
  called from: larql-kv/src/engines/standard.rs:336,381 (prefill/decode setup), run_cmd_image.rs:253
```

This is the up-front "dequantize all layers to f32" that contributes to the ~6.8 s
load tax *and* forces every decode token to read f32 weights from RAM. The CPU
`attention_step` even documents it (cpu.rs:283-289): *"CpuBackend reads f32
attention tensors out of `weights.tensors` … expected to have already populated
… via `ensure_attn_tensors_dequantised`. Until phase-3 CPU Q4K matvec kernels
land, the `index` parameter is accepted … but not consumed."*

**Clean wiring point:** `attention_step` already receives the Q4K `index` — it
just drops it. The direct path reads `index.attn_kquant_layer_data(layer)` (the
exact `&[u8]` slices) instead of `weights.tensors`, and `ensure_attn_tensors_…`
goes away (nothing left to dequantize → also kills that slice of the load tax).

---

## Decode-step anatomy — accelerable vs stays-f32

`run_attention_block_decode_step_backend` (decode.rs:111), in order, with what a
Q4K-direct path changes (✅ swap) vs preserves byte-for-byte (➖ keep):

| Step | decode.rs | Cost shape (per token) | Q4K-direct? |
|---|---|---|---|
| input RMS-norm | :136 `apply_norm` | O(hidden) | ➖ keep f32 |
| **Q proj** | :145 `dot_proj_gpu(h_norm, w_q)` | hidden × q_dim matvec | ✅ **swap → q4k_matvec** |
| Q bias | :146 | O(q_dim) | ➖ keep |
| QK-norm (Gemma) | :159 `rms_norm_heads` | O(q_dim) | ➖ keep |
| RoPE(Q) | :171 `apply_rope_partial_at_full` | O(q_dim) | ➖ keep |
| **K proj** | :191 `dot_proj_gpu(h_norm, w_k)` | hidden × kv_dim | ✅ **swap** |
| **V proj** | :192 `dot_proj_gpu(h_norm, w_v)` | hidden × kv_dim | ✅ **swap** |
| K/V bias, V-norm, K-norm | :193-214 | O(kv_dim) | ➖ keep |
| RoPE(K) | :215 | O(kv_dim) | ➖ keep |
| **KV-concat memcpy** | :227-248 | **O(cached_len × kv_dim)** — grows | ➖ keep f32 |
| **GQA decode step** | :251 `gqa_attention_decode_step` (decode.rs:29) | **O(cached_len × head_dim × num_q)** — grows | ➖ keep f32 |
| **O proj** | :255 `dot_proj_gpu(attn_out, w_o)` | q_dim × hidden | ✅ **swap** |
| O bias, post-norm, residual | :256-280 | O(hidden) | ➖ keep |

The function's own comment (decode.rs:104-108) already asserts the thesis:
*"GQA softmax + weighted-V stays on CPU — that's O(cached_len × head_dim × num_q)
per step and **rarely the bottleneck vs the hidden×hidden projection gemms**."*
True at short context; see the caveat below.

### Dimensional estimate (Gemma-4-26B-A4B: hidden=2816, num_q=16, head_dim=256 sliding / 512 global, num_kv=8 sliding / 4 global, 30 layers)

Projection weights per layer (the ✅ rows):

| | sliding layer | global layer |
|---|---|---|
| Q `[q_dim, hidden]` | [4096, 2816] = 11.5M | [8192, 2816] = 23.1M |
| K `[kv_dim, hidden]` | [2048, 2816] = 5.8M | [2048, 2816] = 5.8M |
| V `[kv_dim, hidden]` | [2048, 2816] = 5.8M | [2048, 2816] = 5.8M |
| O `[hidden, q_dim]` | [2816, 4096] = 11.5M | [2816, 8192] = 23.1M |
| **total / layer** | **~34.6M** | **~57.7M** |

Per-token weight **bandwidth** (the bound — these are matvecs at seq_len=1):
- **f32 today** (post-dequant, read from RAM): ~138 MB/layer (sliding) → ~4 GB/token across 30 layers for attention weights alone.
- **Q4K-direct** (~4.5 bits/wt ≈ 0.56 B/wt + scales): ~18 MB/layer → ~540 MB/token. **≈ 7× less projection bandwidth** — the lever.

GQA decode step (➖, f32, **grows with cached_len**), per layer:
`cached_len × head_dim × num_q × 2`. At cached_len=33 ≈ 0.27M (sliding) / 0.54M
(global) ops — negligible vs a 11–23M-weight projection. At cached_len=4096 ≈
33.5M (sliding) — **comparable to a whole projection**, and unaccelerated.

**Reading:** at the measured short context the projections dominate the 28%, so
Q4K-direct addresses ~most of it. At long context the f32 GQA + KV-concat eat a
rising, *non-addressable* share. The 28% is a short-context number — it flatters
the win. Pin the pre-committed end-to-end bar at a representative context length,
and **measure the projection-vs-GQA split first** (below).

---

## `q4_attention_proj` — what it covers and doesn't

`q4_attention_proj` (gpu.rs:300-324): **one projection matvec, nothing else.**
Per sequence row: quantize input → Q8, call `backend.q4_matvec(...)`, write the
output row. Does **not** do: input/QK/V-norm, RoPE, GQA/softmax/mask/softcap,
bias, KV-append, O-projection orchestration, residual. So it is at most a drop-in
for *one of* the four `dot_proj_gpu` calls — never the block.

**Two defects make it the wrong primitive to wire as-is:**

1. **Format mismatch.** It guards `supports_quant(QuantFormat::Q4_K)` (gpu.rs:307)
   but then calls `backend.q4_matvec(...)` (gpu.rs:317) — the **Q4_0** kernel
   (`quant_matvec.rs:127`; the dispatcher routes Q4_K→`q4k_matvec`, Q4_0→`q4_matvec`,
   `quant_matvec.rs:57-63`). On `CpuBackend`, `supports_quant(Q4_K)` is **true**
   (cpu/mod.rs:120-126), so the guard passes and Q4_K super-block bytes would be
   fed to the Q4_0 18-byte-block kernel → wrong stride → garbage. This is the exact
   bug class pinned by `q8_0_weights_do_not_silently_route_through_q4_kernel`
   (quant_matvec.rs:324).
2. **Not actually tested.** Its only test (gpu.rs:504, "works_with_cpu_backend")
   **builds a synthetic Q4_0 buffer** and asserts only no-panic + shape *if*
   `Some` — no numeric check, no Q4_K. "CPU-tested" in the roadmap overstates it.

**Use instead** (already production-grade on `CpuBackend`):
- `q4k_matvec(q4k_data, x_f32, num_rows, hidden)` — Q4_K super-block × **f32**
  input (no Q8 pre-quant), rayon-parallel for 2560–8192-row decode shapes
  (cpu/mod.rs:75-92), **parity-tested vs dequant→matmul** (q4_common.rs:1284).
- `q4k_dual_matvec(q4k_a, q4k_b, x, …)` — fused two-weight, one shared input
  (cpu/mod.rs:104). K and V share `h_norm` and have identical `[2048,2816]` shape
  on Gemma-4 → **fuse K+V in one call**.
- `quant_matvec(format, weights, x_f32, num_rows, hidden)` (quant_matvec.rs:49) —
  the format-dispatched front door; routes Q4_K/Q4_KF→`q4k_matvec`, Q6_K→`q6k_matvec`.
  **Use this** so per-matrix format tags handle themselves (see parity baseline).

The raw bytes are one call away: `index.attn_kquant_layer_data(layer) →
Option<[(&[u8], &str); 4]>` for (Q,K,V,O), each a `(bytes, format_tag)`
(larql-vindex `index/storage/attn.rs`), mmap'd from `attn_weights_kquant.bin` +
manifest. Same source `ensure_attn_tensors_dequantised` already reads — the
direct path just skips the dequant and matvecs the bytes.

---

## Parity baseline — Q4K-direct vs Q4K-dequant (NOT vs f32-from-f32)

The gate is **engineering parity, before any timing number.** Both sides start
from the **same Q4K bytes**; the only difference is whether we round-trip through
f32 first:

- **Reference (current):** `attn_kquant_layer_data` bytes → `dequantize_matrix`
  → f32 tensor → `dot_proj_gpu` (f32 BLAS). = today's decode output.
- **Candidate (direct):** same bytes → `quant_matvec(format, bytes, h_norm, …)`
  (Q4_K×f32 matvec). 

This isolates **exactly the dequant-tax removal**, holding weight quantization
constant — it does **not** conflate with quant *error* (which a vs-f32-from-f32
baseline would). Expect a tiny, bounded gap from accumulation order (and, if the
direct kernel ever takes Q8 input rather than f32, from activation quant — but
`q4k_matvec` takes f32, so for Q4_K there is *no* activation-quant term; the gap
should be float-summation noise only).

**Report:** per-token **distribution** (KL / NLL across a held text), plus the
**worst single token** (attention feeds everything downstream — a mean hides the
tail; cf. the #26 lesson where mean NLL inverted the ship decision). Run on the
real **26B** vindex. Gate: Q4K-direct ≈ Q4K-dequant within float noise across all
tokens, worst-token included — **then** the timing question opens.

**Per-matrix format — RESOLVED on the real 26B** (`output/gemma4-26b-a4b-q4k.vindex/attn_weights_q4k_manifest.json`,
120 entries = 30 layers × 4): **Q = Q4_K, K = Q4_K, V = Q6_K, O = Q4_K**, uniform
across all 30 layers (it's **V** that's the high-precision Q6_K outlier, not O).
Two build consequences:
- **Per-matrix dispatch is mandatory** — `quant_matvec(fmt, bytes, x_f32, rows, cols)`
  routes Q4_K→`q4k_matvec` and Q6_K→`q6k_matvec` (both f32-input, both on
  `CpuBackend`). `dequantize_matrix` already dispatches on `attn[i].1` the same
  way, so the direct path mirrors it exactly.
- **K+V `q4k_dual_matvec` fusion is DEAD here** — K (Q4_K) and V (Q6_K) share the
  input and shape `[2048,2816]` but differ in format, so they can't fuse. Build
  three plain `quant_matvec` calls (Q, K via q4k; V via q6k) + O; drop the fusion
  from the build shape below.

---

## Step 2 — gate results (2026-05-30): both GREEN → build

Two cheap probes (real production functions, synthetic same-size f32 weights →
faithful bandwidth, no model load; `CpuBackend`, the no-`--metal` path the 28%
was measured on). These are the #24-trap guard — run *before* building the path.

### Gate 1 — projection-vs-GQA split (`chris-experiments/larql_probes/examples/q4k_attention/attn_proj_vs_gqa_split.rs`)

How much of the attention block is the Q4K-accelerable projections vs the
unaccelerated f32 GQA, across a cached_len sweep at 26B dims. **Projection cost
is flat & bandwidth-bound (~40 ms/token blended, constant); GQA grows with
cached_len.** Blended per-token (25 sliding + 5 global, W=1024 sliding cap):

| ctx | proj ms | gqa ms | **proj %** |
|---:|---:|---:|---:|
| 32 | 40.0 | 0.68 | **98.3%** |
| 128 | 41.6 | 1.1 | **97.4%** |
| 512 | 39.5 | 3.4 | **92.1%** |
| 1024 | 40.7 | 8.4 | **82.8%** |
| 2048 | 40.7 | 13.8 | **74.8%** |
| 4096 | 40.6 | 22.8 | **64.1%** |
| 8192 | 40.6 | 40.2 | **50.3%** |

**Reading:** at the band where the 28% was measured (ctx ≈ 32–128) projections
are **97–98%** of the attention block — Q4K-direct addresses ~all of the 28%.
With the sliding-window cap, projections stay ≥64% out to 4K and ~50% at 8K (the
no-cap upper bound on GQA crosses over earlier, ~2.4K). The GQA residual is small
at working context and is the only part Q4K-direct can't touch.

### Gate 2 — f32 BLAS vs Q4K-direct on the projection (`chris-experiments/larql_probes/examples/q4k_attention/attn_proj_f32_vs_q4k.rs`)

The decisive question: does `q4k_matvec` actually beat Apple AMX/Accelerate f32
sgemm, or does AMX throughput eat the 7× bandwidth cut? Per-projection, same
matrix, q4k weights quantized once outside the timed loop:

| | f32 ms | q4k ms | **speedup** |
|---|---:|---:|---:|
| sliding block (Q+K+V+O) | 1.225 | 0.594 | **2.06×** |
| global block (Q+K+V+O) | 2.073 | 0.824 | **2.51×** |

**Reading:** Q4K-direct wins **2.06–2.51×** on the projection — the bandwidth cut
nets out (realized speedup < the 7× byte ratio because AMX f32 is already
bandwidth-efficient, but a solid 2–2.5× regardless). The f32-vs-q4k rel-err
(~2e-2) is just the Q4_K weight-quant error as a wiring sanity check — **not the
parity gate** (which is Q4K-direct vs Q4K-*dequant*, same bytes, expected
≈float-noise since today's path dequantizes the *same* Q4_K bytes).
**⚠ Do not cite the 1.99e-2 as a fidelity number.** It's byte-identical across
all eight projections (Q/K/V/O × sliding/global) because the synthetic `fill()`
draws every weight matrix from the *same* position-cyclic value distribution
regardless of shape — so the Q4_K per-superblock min/max (hence quant error) is
identical by construction. It confirms the wiring produces sane correlated
output and nothing more; real 26B per-matrix quant error varies with the actual
weight distribution and is measured by the parity gate, not here.

### Verdict

**Greenlight step-2 build.** Both axes favorable: projections dominate the block
at working context (Gate 1) *and* the Q4K kernel beats f32 BLAS 2–2.5× (Gate 2).
Amdahl: projections ≈ 97% × 28% ≈ **~27% of decode**; a 2.1–2.5× cut there → the
block ~halves at short context → **~14–16% net decode improvement from attention
alone** (before lm_head/FFN). **⚠ The ~14–16% is the SHORT-CONTEXT end of its own
range, not a flat number.** It is `97% × 28% × (1 − 1/2.1)` at ctx 32–128; the
projection share *decays* with cached_len (Gate 1: 83% at 1024, 64% at 4096), so
the realized net win shrinks as context grows. **Set the pre-committed ship-gate
bar at a *representative* context length, not the band that flatters it** — else
we re-run the #26 mean-NLL-inverts trap in a different costume (a number measured
where it's largest, used as if it held everywhere). **These are also ISOLATED
kernel timings** — per the #24 lesson
(`feedback_isolated_vs_batched_kernel_profile`), the *ship* gate stays
**end-to-end net decode tok/s > dense on the real 26B vindex at a representative
context**; the isolated evidence just says it's worth building and is very
unlikely to be a wash.

## Step 3 — Q4K-direct decode path landed; parity GREEN (2026-05-30)

Built and unit-parity-validated. Opt-in while the end-to-end ship gate is pending.

- **New CPU function** `run_attention_block_decode_step_q4k_direct`
  (`attention/decode.rs`) + `q4k_direct_proj` helper. Mirrors the f32
  `run_attention_block_decode_step_backend` **byte-for-byte** except the four
  projections: `dot_proj_gpu` (f32 BLAS on pre-dequantised `weights.tensors`) →
  `quant_matvec(fmt, bytes, x_f32, rows, cols)` reading
  `resolve_attn_weights(index, layer)` (per-matrix format dispatch:
  Q/K/O→`q4k_matvec`, **V→`q6k_matvec`**). Norms/RoPE/GQA/concat/biases/residual
  unchanged.
- **Wired opt-in** in `CpuBackend::attention_step` behind
  `LARQL_Q4K_DIRECT_ATTN=1`, with **per-layer f32 fallback** (any layer whose
  index lacks Q4K bytes / unsupported format silently uses the f32 path). Default
  off → zero behaviour change until flipped. The previously-ignored `index` arg
  is now consumed.
- **Parity gate ✅** (`larql-inference` `vindex::dequant`, **two tests**): Q4K-direct
  vs Q4K-**dequant** (same bytes, both paths) agree **< 1e-3 max-abs** on h/k/v
  across both layers, output non-degenerate. The spine — parity before timing —
  green at unit level. (Reference built by stripping attn tensors then
  re-inserting `dequantise(quantise(original))`, so the f32 path carries the same
  weight-quant error the candidate does; isolates exactly the dequant-tax removal.)
  - `q4k_direct_decode_step_matches_q4k_dequant` — all-Q4_K.
  - `q4k_direct_decode_step_matches_dequant_with_q6k_v` — **mixed, V=Q6_K** (real
    26B layout). The shared `make_test_q4k_vindex` is all-Q4_K, so it never
    exercises the `q6k_matvec` dispatch the real V hits; this test builds a mixed
    index so Q6_K flows through **both** sides (candidate `q6k_matvec`, reference
    Q6_K dequant) — closing the format-coverage gap, not just an all-Q4_K green.
  - `q4k_direct_decode_multistep_parity_compounds_within_noise` — **multi-step /
    compounding**. The two above are single-step (`kv_entry=None`); the real run
    accumulates a KV cache where the two paths' caches drift *cumulatively*. This
    drives 8 sequential steps, each path carrying its own growing cache (mixed
    V=Q6_K), and checks the post-attention hidden stays bounded. **Worst drift
    over 8 steps = 1.9e-6** — float-noise scale, bounded, ~500× below any
    argmax-flip risk → compounding does NOT blow up with cache depth. (The 64-tok
    26B run is the full multi-step test; this localises a compounding bug to the
    unit level first.)
  - *Framing guard:* the original all-Q4_K test was **narrow, not wrong** — a
    genuine Q4_K-direct≈Q4K-dequant result with a coverage hole (it never touched
    Q6_K, the format every real V uses). The mixed + multi-step tests **extend
    coverage; they do not correct an error.** Don't let a future summary collapse
    this into "the original test was broken" — it was a real result, and that
    distinction is what lets us trust the rest of the suite.
- **Tests/lint:** larql-compute 97 attention + 44 kv_dispatch pass; clippy clean
  on both crates; larql-kv/larql-cli build.
- **Note — separate existing path:** cpu.rs already has a `cached_decode_step_q4k`
  / `CpuQ4kCacheHandle` Q4K pipeline (the `prefill_quant`/cached-decode route).
  This new function targets the **standard-engine `attention_step` path** that
  carries the *measured* 28% (`kv_decode_step_via_dispatch`), so it's additive,
  not a duplicate. Consolidating the two is a possible later cleanup.

**Remaining for step 3 → ship:**
1. **End-to-end on the real 26B** (the ship gate): run `--moe-shards` decode with
   `LARQL_Q4K_DIRECT_ATTN=1` vs off — confirm output parity *and* net decode
   tok/s > the f32 path, **at a representative context length** (not just the
   short band — proj share decays, §Step 2).
2. **Drop the `ensure_attn_tensors_dequantised` precondition** on this path once
   end-to-end holds → reclaims the up-front dequant slice of the ~6.8 s load tax
   (nothing left to dequantise for attn).
3. **Prefill twin** (`run_attention_with_kv_backend`, `q4k_matmul` amortised) for
   the TTFT win — lower priority than the per-token decode path.

### End-to-end run recipe (the ship gate) — to be driven on the real 26B

The decode 28% lives on the KV-cached `--engine standard` `--moe-shards` path
(`kv_decode_step_via_dispatch → attention_step`, where `record_attn` measured it);
the flag fires there. Run flag-OFF then flag-ON, **everything else identical**.

```sh
# Representative context — NOT the ~30-token band that flatters proj-share.
# Use a ~1024-token prompt so decode runs at cached_len ≈ 1K (Gate 1: proj ~83%).
PROMPT="$(cat long_prompt_~1k_tokens.txt)"
SHARDS="0-127=http://localhost:8080"   # your existing expert server(s)
VINDEX=output/gemma4-26b-a4b-q4k.vindex

# Baseline (f32 dequant path)
LARQL_DECODE_STAGES=1 \
  larql run "$VINDEX" --moe-shards "$SHARDS" --engine standard --max-tokens 64 "$PROMPT"

# Candidate (Q4K-direct) — identical + the one flag
LARQL_Q4K_DIRECT_ATTN=1 LARQL_DECODE_STAGES=1 \
  larql run "$VINDEX" --moe-shards "$SHARDS" --engine standard --max-tokens 64 "$PROMPT"
```

Capture, one pass each (repeat 2–3× for variance):
1. **Flag-fired check (do this FIRST):** the `[stages] attn: … ms` line, on vs off.
   Expect attn ms to **drop ~2×** (Gate 2). If it doesn't move, the flag isn't on
   this decode path — STOP and report *that* (it's the finding), don't read tok/s
   as "no win."
2. **Parity:** the generated text on vs off must be **byte-identical**. This holds
   only if decode is deterministic — `larql run` exposes **no** temperature/sampling
   flag and defaults to **greedy argmax** (generation picks `logits.max_by(...)` /
   `SamplingConfig::greedy()`), so it is deterministic by default; confirm no
   sampling line in the banner and **note "greedy" in the captured results**.
   With greedy + <1e-3 per-step parity → identical argmax. If it diverges anyway,
   capture both outputs + the first divergent token — a parity signal to
   investigate, not a pass (and not a sampling phantom, since there's no sampling).
3. **tok/s:** the honest **decode** tok/s (inter-token, not total/n) from the
   banner, on vs off — the ship number. Note TTFT separately (prefill twin not
   built → TTFT shouldn't move; if it does, flag it).
4. **Real Amdahl denominator — attn as a FRACTION of total decode, not just
   absolute ms.** Gate 1's ~83%-of-block / ~28%-of-decode were *synthetic-weight,
   no-network* microbenches. This run is the first measurement of attention's
   *real* share with real weights + real MoE shard round-trips in the wall-clock.
   The standard engine is **network-bound** (C1 rollup: "all seven within ~2.6×
   and network-bound"), so expert calls may make attention a *smaller* slice than
   28% end-to-end. Record `attn_ms / total_decode_ms` on vs off. If attn halves
   but is only ~15% of real decode, the net tok/s win is ~7-8%, not ~14-16% — and
   the honest ship framing becomes **"attention is no longer a meaningful decode
   cost"** rather than a specific tok/s headline. Capture the denominator so the
   projected ~14-16% never gets cited as the realized number.
5. **Context:** record the actual prompt token count so the number is pinned to a
   context, not assumed.
6. **Thermal:** if decode tok/s swings >10% run-to-run, suspect M3 Max throttling
   (`feedback_thermal_perf_artifacts`) — cool + repeat before trusting it.

**Ship verdict:** net decode tok/s(on) > tok/s(off) at the recorded context, with
identical (greedy) output text, attn-ms dropping, and the real attn-fraction
recorded → fill the §"Open questions" end-to-end line and the lever ships; then
drop `ensure_attn_tensors_dequantised` on this path (load-tax win). Frame the win
against the **real** decode denominator, not the synthetic 28%.

## Step-2 build shape

1. ✅ **Split timer + f32-vs-Q4K A/B** — done (Step 2 above, both GREEN). Optional
   follow-up confirmation: an in-decode `LARQL_DECODE_STAGES`-style projection-vs-GQA
   split on the *real* 26B vindex (the microbench used synthetic same-size weights);
   not blocking the build given the isolated evidence.
2. ✅ **Q4K-direct decode-step variant** — `run_attention_block_decode_step_q4k_direct`
   built (per-matrix `quant_matvec`, Q6_K for V, no dual-fusion).
3. ✅ **Wired** opt-in (`LARQL_Q4K_DIRECT_ATTN=1`) through the now-consumed `index`
   arg in `CpuBackend::attention_step`, per-layer f32 fallback. (Dropping the
   `ensure_attn_tensors_dequantised` precondition deferred to after end-to-end.)
4. ❌ **Prefill twin — FALSIFIED** by the prefill-shape gate (compute-bound gemm,
   q4k ~20× slower; no `q4k_matmul`, and even one wouldn't beat AMX f32 at
   seq_len=907). Do not build. See Open-questions "Prefill twin — GATED then
   FALSIFIED".
5. ✅ **Parity gate** (unit, < 1e-3) + ✅ **end-to-end ran** — flag fires, parity
   holds, net ≈ 0 at representative context (decode is expert/GQA-bound). Ship the
   decode path opt-in; don't headline a tok/s number.

## Open questions

- ✅ **Projection share of the 28%** — resolved (Gate 1): 97–98% at ctx 32–128.
- ✅ **Does Q4K-direct beat AMX f32 BLAS?** — resolved (Gate 2): yes, 2.06–2.51×.
- ✅ **Per-matrix format tags** on the 26B — resolved: Q/K/O = Q4_K, **V = Q6_K**
  (uniform across 30 layers). Per-matrix `quant_matvec(fmt,…)` dispatch required;
  K+V dual-fusion ruled out (format mismatch). Parity gate covers both formats.
- ⚠️ **CONSOLIDATION HAZARD — two CPU Q4K attention paths.** This new
  `run_attention_block_decode_step_q4k_direct` (standard-engine `attention_step`)
  and the pre-existing `cached_decode_step_q4k` / `CpuQ4kCacheHandle`
  (`kquant_forward`, the `prefill_quant` cached route) are **independent
  implementations of the same computation**. They MUST agree on RoPE / softcap /
  QK-V-norm / GQA handling, but nothing enforces it — the moment one gets a bug
  fix or a RoPE-scaling tweak the other doesn't, you get a silent cross-path
  divergence (the exact class as the 2026-05-28 prefill-RoPE bug). Acceptable for
  this unit (additive, default-off), but **consolidate to one path** before either
  is load-bearing — track it, don't let "additive, not duplicate" decay into "two
  paths nobody remembers must agree." A cross-path parity test would be the
  cheapest guard if consolidation is deferred.
- ✅ **End-to-end RAN on the real 26B (2026-05-31)** — flag fires, parity holds,
  net win is **marginal and context-dependent**. Localhost server `--experts
  0-127`, `--engine standard`, warm, interleaved OFF/ON.
  - **Unblock (the resident path).** The flag was first inert: the CPU moe-shards
    decode pre-dequantises attn then drives the *immutable* `generate_with_engine`
    → `decode_step` (index=None), so `attention_step` never got the index. The
    fix that *threads* it (`decode_step_quant`) needs `&mut weights` and collides
    with `RemoteMoeFfn` holding `&weights`. Resolved with a **resident-weights
    path** (`KvEngine::{prefill,decode_step}_resident`, `AnyEngine` forwarders,
    `generate_with_engine_resident`, CLI swap): since the caller already made
    weights f32-resident, the resident methods take **`&weights`** (no lazy
    dequant) and just thread `index` → both engine and FFN borrow `&weights`
    immutably, no conflict, no Arc. Flag-off ≡ old f32 path (index ignored).
  - **Short context (~14 cached_len, 6-tok prompt, 32 decode tok, n=6 paired):**
    attn-ms **1628→1413 (−13%)**, decode **8.22→8.60 tok/s = +4.6%** (ON>OFF every
    sample), output "Paris." both → real Q4K-direct-vs-f32 end-to-end parity ✓.
  - **Representative context (907-tok prompt, 16 decode tok, n=5 paired):** decode
    **OFF mean 4.80 / median 4.70 (4.6–5.2); ON mean 4.92 / median 5.00 (4.8–5.0)**
    — **net ≈ 0, within the ~6% run-to-run noise floor** (absolute rate also
    drifted 5.4→4.8 between sessions → thermal/load; one-sample numbers untrustworthy
    here). Direction is ambiguous-to-marginally-positive, magnitude negligible —
    NOT the "slightly negative" a single sample implied. The short-ctx +4.6%
    **washes out**: at 907-deep cache, per-token decode is dominated by the 30
    sequential expert round-trips **and** f32 GQA over the large KV cache — both
    untouched by Q4K-direct projection. **Confirms: measure at representative
    context** (short-ctx flattered, #26-class), and the engine is **MoE-network +
    GQA bound**, not attention-proj bound, at real context. Attention is no longer
    a meaningful *client decode* cost — the binding constraint is the expert path.
    (Strength note: short-ctx +4.6% is n=6 tight/every-sample; rep-ctx ~0 is n=5
    within noise — both measured, not inferred.)
  - **Prefill twin — GATED then FALSIFIED.** At 907 ctx prefill attention is 6288 ms
    of a 14739 ms TTFT (~43% of prefill), which *looked* like the better lever. But
    the prefill-shape Gate-2 (`chris-experiments/larql_probes/examples/q4k_attention/attn_prefill_f32_vs_q4k.rs`, seq_len=907)
    kills it: repeated per-position `q4k_matvec` (the only CPU path — **no
    `q4k_matmul`**) is **~20× SLOWER** than one f32 BLAS sgemm (sliding block 25.9
    vs 569 ms, 0.05×; global 42 vs 789 ms). Deeper reason: at prefill the projection
    is a **batched gemm = compute-bound** (AMX's home turf), not weight-bandwidth-
    bound — the exact regime where Q4K's bandwidth edge (which won decode at
    seq_len=1) **evaporates**. Even a hypothetical perfect `q4k_matmul` would be
    compute-bound + dequant overhead at this seq_len, so it couldn't beat AMX f32
    either; the "bandwidth floor" column is moot because we're not bandwidth-bound.
    **Q4K-direct is the wrong tool for prefill attention — do NOT build the twin.**
    The 43%-of-TTFT is real but needs a non-quant attack (the O(N²) GQA, or a
    better f32 gemm), not quantisation.
  - **Ship call:** the decode path is correct, parity-safe, free, **opt-in**
    (`LARQL_Q4K_DIRECT_ATTN`, default off) — keep it (removes attention as a
    client-compute variable; sets up dropping the attn dequant for the load-tax
    win) but **do not headline a decode tok/s number** (~0 at rep ctx). Net of the
    whole arc: **Q4K-direct wins ONLY in the bandwidth-bound decode-matvec regime,
    and even there the win washes out behind the expert/GQA bottleneck; at prefill
    (compute-bound gemm) it loses outright.** Throughput levers are now the
    **expert/network path** (45% of decode) and TTFT's O(N²) GQA — not Q4K attention.
