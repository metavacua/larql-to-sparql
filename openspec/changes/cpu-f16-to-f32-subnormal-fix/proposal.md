## Why

`larql_compute::cpu::ops::q4_common::f16_to_f32` decoded every subnormal
f16 input as exactly 2× the correct value. The subnormal-exponent
formula was `127 - 14 - lz` (= `113 - lz`), but the correct value is
`112 - lz`. Latent for years because Q4_K super-block scales (`d`) are
typically NORMAL f16 — surfaced via the Q6_K matvec path when surfacing
the direct Q4_K × Q8_K attention arc on Gemma 3 4B.

The bug was masked by a test bug: `q4_common::tests::f16_to_f32_bit_exact_for_all_inputs`
defined a SHADOW `fn f16_to_f32` inside the test module which contained
the correct subnormal formula. The test compared the shadow against the
powi reference (both correct) instead of testing the outer (buggy)
function. With the shadow removed and the fix applied, the test
exhaustively verifies all 65,536 f16 inputs against the powi reference.

A parallel `f16_to_f32` in `larql_models::quant::half` already had the
correct formula (its own doc comment notes: "The previous normalisation
formula `127 - 15 + 1 - e` produced values exactly 2× too small for
every subnormal path — fine when all scales were normal floats (legacy
quant settings), catastrophic once k-quant super-block scales were
forced into f16 subnormal range by the corrected Q4_K/Q6_K scale
formulas."). The same correction now lands in `larql_compute`.

## Real-World Impact

Affected paths — every CPU K-quant matvec that reads f16 `d` from a
weight block and lands in subnormal range:

- `cpu::ops::q4k_q8k_dot::q4k_q8k_matvec_into` (and `_scalar`, `_avx2`)
- `cpu::ops::q4k_q8k_dot::q6k_q8k_matvec_into` (and `_scalar`, `_avx2`)
- `cpu::ops::q4k_q8k_dot::q4k_q8k_gate_up_into`
- All callers including `q4k_ffn_forward_layer_q8k` (used by
  `Q4kDirectFfn` from PR #139 and `walk_ffn_q8k` server route).

Q4_K scales typically land in normal f16 range, so PR #138 / #139 FFN
matvec was numerically slightly off but RMSnorm masked the error. Q6_K
on Gemma 3 4B V-projection (small magnitude weights) and FFN_DOWN
(small per-feature scales) pushed `d` into subnormal range — kernel
output was 2× the correct value. Tested empirically:

| Tensor | Before fix | After fix |
|---|---|---|
| Q6_K V matvec (Gemma 3 4B layer 0) row 0 first element | -2.700 | -1.350 |
| canonical dequant + dot reference | -1.350 | -1.350 |
| ratio kernel / reference | **2.0** | **1.0** |

Same 2× for FFN_DOWN Q6_K. Now matches reference within Q8_K activation
quantisation noise (≤ 1.5 % rel).

## What This Change Ships

**Code:**
- `crates/larql-compute/src/cpu/ops/q4_common.rs`: subnormal exponent
  formula corrected from `(127u32 - 14 - lz) << 23` to
  `(112u32 - lz) << 23`.
- Same file: remove the shadow `fn f16_to_f32` in the test module so
  `f16_to_f32_bit_exact_for_all_inputs` exercises the production
  function, not its own copy.

**Capability deltas** (under `compute-backend-traits/`):
- f16-to-f32 subnormal correctness invariant + bit-exact-vs-powi
  oracle.

## Out of Scope (Follow-Ups)

- Re-bench PRs #138 / #139 for Gemma 3 4B to confirm the FFN-side
  numerical drift was the cause of the otherwise-coherent-but-slightly
  off generations.
- Audit any other duplicated `f16_to_f32` / `f32_to_f16` definitions in
  the workspace for the same off-by-one. (A grep found only the two
  decoders.)
