# Phase C.4–C.5 investigation summary

Status as of 2026-05-11 after ~35 sub-phases. **Phase C is DONE.**

## C.5j/C.5k: full greedy parity vs llama.cpp ✅

The first 5 greedy-decoded tokens from our Qwen3.6-27B Q4_K_S
forward now match `llama-eval-callback` exactly — GT rank 0 at
every step:

| step | gt_id | gt_logit | gt_rank | our_argmax | gt_text |
|------|-------|----------|---------|------------|---------|
| 0 | 248068 | 28.177 | 0 | 248068 | `<think>` |
| 1 | 271 | 24.776 | 0 | 271 | `\n\n` |
| 2 | 248069 | 25.470 | 0 | 248069 | `</think>` |
| 3 | 271 | 30.392 | 0 | 271 | `\n\n` |
| 4 | 9419 | 21.656 | 0 | 9419 | `Hello` |

The earlier "ground truth" (`|- [Start thinking]\nHere's a thinking`)
was a non-chat-template completion captured under different
sampling, not what a chat-tuned model produces for the
`<|im_start|>assistant\n` continuation. Stored GT in the parity
test has been corrected.

## C.5j: the two final bugs

Both surfaced by the elementwise binary tensor parity oracle
(added in C.5i — `LLAMA_DUMP_BIN_DIR` on llama.cpp, mirrored on
ours). Token-rank had been hiding them.

1. **Q6_K dequant layout** — our code wrote dequantized values
   sequentially per scale-subblock; llama.cpp uses an interleaved
   layout (`y[l]`, `y[l+32]`, `y[l+64]`, `y[l+96]` per
   `l in 0..32` of each half, with different scales). Hit
   `output.weight` (lm_head, Q6_K). Sorted-logit pearson with
   llama.cpp was 0.99 (correct value distribution) but elementwise
   pearson was 0.06 (rows permuted within each Q6_K super-block).
2. **DeltaNet recurrence order** — we used paper-order
   (sk-before-decay per Yang et al. Eq. 6). llama.cpp's actual
   kernel does **decay-first**. Switched. Token-8 LIN-layer
   block_out pearson rose from 0.83-0.97 to 0.995-0.998 at every
   layer; residual-stream L62 pearson 0.982 → 0.9985.

## C.5i breakthrough — CYCLE GQA is correct (elementwise parity)

Earlier sub-phases (C.4o → C.4u → C.5h) flipped CYCLE
(`kh = h % h_k`) vs BLOCK (`kh = h / repeat_factor`) GQA mapping
based on **token-rank** as the validator. C.5i instead extended
llama.cpp's `common/debug.cpp` to dump full f32 tensors to
`LLAMA_DUMP_BIN_DIR`, then compared token-0 layer-0 `final_out`
against ours **elementwise**:

| GQA mode | pearson r | max\|diff\| | l2 ours | l2 theirs | median per-head ratio |
|---|---|---|---|---|---|
| BLOCK | 0.7697 | 3.17 | 7.01 | 6.67 | 0.996 |
| **CYCLE** | **0.9999** | **0.006** | **6.67** | **6.67** | **0.998** |

CYCLE matches llama.cpp's `final_output-0` token-0 essentially
bit-exact (modulo Q5_K quant noise). Heads 12, 28, 10 (BLOCK
amplified 100×+) and heads 32, 36, 37 (BLOCK suppressed 100×) all
collapse to within 1% of llama.cpp under CYCLE. The C.5h reversion
was wrong: token-rank is masked by downstream bugs and missed this.

**Token-rank with CYCLE is the same as C.5h's measurement (~125k
step-0).** The remaining bug is downstream of layer 0 — almost
certainly in the recurrence algorithm order for non-empty state
(decay-first vs sk-before-decay, per llama.cpp's actual kernel) and
its compound effect across 64 layers.

## Where we are

- **Plumbing complete.** GGUF → Qwen35Weights bridge, tokenizer
  round-trip, prefill + decode all work end-to-end on real
  Qwen3.6 27B Q4_K_S in ~3 s/token CPU. 113 unit tests + 9
  real-GGUF env-gated tests pass.
- **Parity oracle operational.** `llama-eval-callback` is built;
  per-layer tensor comparison vs llama.cpp now drives bug
  bisection. Layer-0 `x_norm`, `qkv_conv`, and `final_out`
  (TOKEN 0 ONLY) all verified BIT-EXACT or near-bit-exact at
  sampled positions.
- **6 confirmed bug fixes landed** (PRs #60, #63, #74, #76, #79,
  plus diagnostic infra).
- **1 known remaining issue**: token 0 matches bit-exact, but
  **token 1+ `final_out` diverges ~20× from llama.cpp**. The bug
  is in inter-token state-handling. The most likely structural
  cause is per-token recurrent algorithm semantically diverging
  from llama.cpp's prefill-chunked algorithm, even though the
  Gated DeltaNet paper claims they're mathematically equivalent.

## Token-rank progression for step-0 GT (`|-` = 49143)

| State | step-0 GT rank | step-0 GT logit |
|---|---:|---:|
| Pre-C.4 fixes | 118,718 | 0.097 |
| After PR #69 (wrong double 1+w) | 16,054 | 4.289 |
| After PR #74 (revert 1+w, correct math) | 183,447 | -1.344 |
| After PR #76 (head-major flatten) | 149,333 | -0.913 |
| **After PR #79 (sk-before-decay)** | **101,839** | **-0.083** |

## Confirmed fixes (in `main`)

1. **PR #60** — DeltaNet `[s_v, h_v]` reshape was scrambling
   head/dim indices. Now: reshape via
   `(n_v_heads, head_v_dim).reversed_axes()`.
2. **PR #63** — `split_q_gate` assumed split-half layout; actual
   Qwen 3.6 layout is interleaved-per-head per llama.cpp
   `qwen35.cpp:220-244`.
3. **PR #69 (reverted in PR #74)** — Tried Gemma-style `(1+w)`
   RMSNorm offset at bridge load. **WRONG** — empirical inspection
   via `llama-eval-callback` showed GGUF stores `(1+w)` baked in
   (stored weights ≈ 1.0). PR #74 reverts the double-application.
4. **PR #76 (C.5c) — HEAD-MAJOR FLATTEN** of DeltaNet recurrence
   output. The naïve `o.into_iter()` flatten produced DIM-MAJOR
   flat layout, scrambling `rms_norm_heads` slices across multiple
   heads and producing 46× over-amplification. Transpose first
   so layout becomes head-major (matching HF Qwen3-Next's
   `out.reshape(B, S, -1)` of an `[..., n_v_heads, head_v_dim]`
   tensor). **Step-1 GT rank: 216,947 → 7,617 (top 3%).**

## Token-rank progression

For ground-truth token at step 1 (` [` = 498):

| State | step-1 GT rank | step-1 GT logit |
|---|---:|---:|
| Pre-C.4 fixes | n/a | n/a |
| After PR #64 (cycle GQA, wrong) | 119,184 | 0.075 |
| After PR #69 ((1+w) + block GQA) | 185,292 | -1.978 |
| After PR #74 (revert 1+w, keep block) | 216,947 | -2.045 |
| **After PR #76 (head-major flatten)** | **7,617** | **+3.505** |

## Parity oracle: verified bit-exact (or near) at layer 0

Using `llama-eval-callback` to dump tensors during a real Qwen3.6-27B
forward, then `LARQL_QWEN35_DUMP_L0=1` to dump ours:

| Tensor | Match status |
|---|---|
| `embed` (input embedding lookup) | implied bit-exact |
| `attn_norm` (x_norm output) | **BIT-EXACT** at first-3 + last-3 |
| `attn_qkv` matmul (qkv_mixed) | ~1-4% per-element Q5_K dequant noise |
| `conv1d + silu` (qkv_conv) | **near-bit-exact at f32 precision** |
| `recurrence` (o) | matches expected (small magnitude) |
| `ssm_norm + silu(z)` (final_out, pre-ssm_out) | **BIT-EXACT** at first-3 + last-3 |
| `ssm_out` matmul (linear_attn_out) | **3× too large** (16.7 vs ~5.7) ← bug |

## Hypotheses ruled out

- **lm_head row outlier** (C.4k) — row norms within population σ.
- **ssm_a sign error** (C.4s) — all 48 values negative as
  expected.
- **GGUF `ssm_a` storage** (C.4v) — pre-computed `-exp(A_log)`
  matches llama.cpp's direct multiplication.
- **HF chunkwise recurrence formula** (C.4w) — paper's recurrent
  form is correct at chunk_size=1.
- **`(1+w)` RMSNorm offset** (C.5a) — GGUF already pre-applies.
- **Embedding scale / softcapping** — verified absent in HF and
  llama.cpp.
- **CYCLE GQA in DeltaNet** (C.5h) — llama.cpp's fused kernel uses
  `ik1 = iv1 % nek1` (cycle), but applying that to our recurrence
  empirically regressed step-0 (101,839 → 125,425) and step-1
  (7,617 → 32,881) GT ranks. The model weights were trained with
  HF's `repeat_interleave` (block) layout; reconciliation with
  llama.cpp's `%` mapping lives in the GGUF tensor layout, not in
  the per-head recurrence loop. Block remains correct.

## Verified consistent with HF Qwen3-Next + llama.cpp

After reading both references:

- RMSNorm semantics (epsilon, mean, weight broadcast)
- Softplus / sigmoid pointwise math
- Attention scale = `1/sqrt(head_dim)`
- DeltaNet decay: `g_exp = exp(ssm_a * softplus(alpha + dt))`
- Conv1D time direction (weight[0] = oldest token)
- Conv state shift-and-insert pattern
- Residual add pattern (attn block to original x, FFN to residual)
- Pre-attention norm placement (inside block)
- Post-attention norm placement (between residual and FFN input)
- Final RMSNorm + lm_head sequence
- RoPE pairing convention (split-half for NEOX/IMROPE)
- MRoPE-text-only reduces to partial-RoPE at first `rotary_dim` dims
- `ggml_l2_norm` per-head normalization
- Q+gate split convention (PR #63)
- DeltaNet GQA block pattern (PR #76 / `repeat_interleave`)
- DeltaNet recurrence output flatten: HEAD-MAJOR (PR #76)
- Embedding lookup: row by token id, no scale

## Remaining bug investigation paths

The 3× `linear_attn_out` discrepancy with bit-exact `final_out`
input means the `ssm_out` matmul produces 3× larger output than
llama.cpp's. Candidate explanations to investigate:

1. **Q5_K dequant precision differences for `ssm_out.weight`** —
   middle-element noise that doesn't show in abbreviated first/last
   prints but compounds through the matmul.
2. **Matmul precision** — llama.cpp may use f16 intermediate; we
   use full f32 BLAS.
3. **A missing per-head scale factor** between recurrence output
   and final projection.
4. **A possible 3× normalization factor** we're missing.

The 3× ratio is consistent across `linear_attn_out` (l2 16.7 vs
~5.7) and `attn_norm-1` (l2 44.8 vs ~14.3), suggesting a single
multiplicative cause not compounding through layers.

## Diagnostics infrastructure

Env-gated, all checked in:

- `LARQL_QWEN35_GGUF=/path/to/gguf` enables 9 real-GGUF tests
- `LARQL_QWEN35_TRACE=1` per-layer residual stream l2 trace
- `LARQL_QWEN35_DUMP_L0=1` layer-0 tensor first/last-3 dumps
  (x_norm, qkv_mixed, qkv_conv, o(recurrence), final_out,
  linear_attn_out)
- `LARQL_QWEN35_DUMP_FINAL=1` x_final dump pre-lm_head

llama.cpp side:
- `llama-eval-callback` binary built and ready (in
  `~/3rd-party/llama.cpp/build/bin/`); dumps all tensors during
  a forward pass.

## Next session's concrete agenda

The parity oracle has reduced the bug from "somewhere in 64
layers" to "in the `ssm_out` matmul or its immediate inputs."
Pick one:

1. **Dump more positions of `final_out`** (e.g. positions 1024,
   2048, 3072, 5000) to verify bit-exact across the full 6144
   vector. If middle differs, the 3× is from Q5_K input noise
   amplified by matmul.

2. **Dump `ssm_out.weight` row norms** for a few rows; compare
   to magnitudes implied by llama.cpp's `linear_attn_out` values.

3. **Try a HIGHER-precision GGUF** (Q8_0 if available) — if the
   3× discrepancy disappears, the bug is Q5_K dequant precision
   (not a logic bug).

4. **Switch token-diff test to ground-truth feeding** for clean
   per-step parity comparison.
