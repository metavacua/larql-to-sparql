## Why

After PRs #138 / #139 / #140 / #142 / #143 took FFN, attention Q/K/V/O,
and matvec parallelism off the critical path, the remaining decode-step
wall-clock was dominated by `lm_head` — an f32 BLAS GEMV against
`weights.lm_head`, sized `vocab × hidden` (262144 × 2560 for Gemma 3 4B
= 671 M MACs per decoded token).

The vindex loader already reads the Q4_K bytes from `lm_head_q4.bin`
and dequantises them into the f32 `weights.lm_head` field for BLAS.
But `weights.lm_head_quant` (the `QuantTensor` representation that
already exists and has rayon-parallel AVX2 Q4_K × Q8_K matvec via
`QuantTensor::matvec`) was left as `None`.

This proposal populates `lm_head_quant` from the same on-disk bytes
during vindex load, then dispatches the final logit projection through
`QuantTensor::matvec` when available — skipping the f32 GEMV entirely.

## What This Change Ships

**Code:**
- `crates/larql-vindex/src/format/weights/load.rs`: after dequantising
  `lm_head_q4.bin` into f32, ALSO construct a `QuantTensor` from the
  same bytes and store it in `weights.lm_head_quant`. Trim trailing
  GGUF vocab-alignment padding rows (e.g. Gemma 3 4B's 262208 → 262144)
  before handing the bytes to `QuantTensor::from_raw`. Only populated
  when `hidden_size % 256 == 0` — the QuantTensor path doesn't yet
  model padded-col semantics that Gemma 3 1B (hidden=1152→1280) would
  need.
- `crates/larql-inference/src/forward/predict/dense.rs`: add
  `project_lm_head_last_row` helper that prefers
  `weights.lm_head_quant.matvec` when available, falling back to
  `dot_proj` over `weights.lm_head` otherwise. `full_vocab_probs` uses
  the new helper.
- `crates/larql-inference/src/forward/predict/raw.rs`: `hidden_to_raw_logits`
  also routes through the same helper for the chat-completion-on-MoE
  path. Both call sites get the direct path together.

**Capability deltas** (under `inference-residual-engine/`):
- The lm_head projection MUST dispatch through `QuantTensor::matvec`
  when `weights.lm_head_quant` is `Some`, preferring the direct Q4_K
  × Q8_K matvec over the f32 BLAS GEMV.
- Models whose `hidden_size` is not a multiple of 256 (Gemma 3 1B)
  SHALL keep using the f32 `weights.lm_head` field; the QuantTensor
  path is gated on Q8_K alignment.

## Bench (Gemma 3 4B Q4_K_M, 48-thread EPYC host)

| Stage | Before | After | Total speedup vs 0.117 baseline |
|---|---:|---:|---:|
| After #143 (rayon matvec) | 1.79 tok/s | — | 15.3× |
| **After this PR (direct lm_head)** | — | **3.96 tok/s** | **33.9×** |
| llama.cpp CPU reference | 14.1 tok/s | — | — |

End-to-end 1.79 → 3.96 tok/s is **2.22× from this PR alone**. lm_head
projection was the dominant remaining cost — `QuantTensor::matvec` is
already rayon-parallel + AVX2 Q4_K × Q8_K, so routing through it picks
up the same scale-out the FFN/attention paths got in #142/#143.

Output remains coherent on Gemma 3 4B:
> Okay, let's dive into the wonderful world of cats!…

## Out of Scope (Follow-Ups)

- Padded-col QuantTensor variant so Gemma 3 1B (hidden=1152→padded 1280)
  can also use the direct path. Today 1B keeps the f32 BLAS GEMV — its
  end-to-end was the gibberish-or-bust path until #136/#137 and isn't
  the bench target anyway.
- Overlap sampling with the next decode step. After this PR sampling
  becomes a larger fraction of wall-clock; pipelining could close
  another 10-20% before the ~3.6× gap to llama.cpp narrows further.
