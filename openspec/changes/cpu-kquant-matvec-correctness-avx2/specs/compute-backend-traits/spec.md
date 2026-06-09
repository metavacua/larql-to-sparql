## ADDED Requirements

### Requirement: CPU K-quant matvec MUST read llama.cpp wire format

CPU K-quant matvec MUST read the canonical llama.cpp wire format for
both Q4_K and Q6_K weights — the 256-element super-block / interleaved-
stride layout produced by `quantize_q*_k` and consumed by
`larql_models::quant::ggml::dequantize_q*_k`.

Both production Q6_K matvec entry points apply: the f32-input trait
body `CpuBackend::q6k_matvec` (backed by `cpu::ops::q6k_matvec::
dispatch`) and the Q8_K-input kernel `cpu::ops::q4k_q8k_dot::
q6k_q8k_matvec_into`. The same applies to Q4_K via
`CpuBackend::q4k_matvec` → `cpu::ops::q4k_matvec::dispatch` and
`cpu::ops::q4k_q8k_dot::q4k_q8k_matvec_into`.

#### Scenario: Q6_K matvec output matches canonical dequant + dot

- **WHEN** `q6k_q8k_matvec_scalar` runs on a Q6_K weight matrix produced
  by `quantize_q6_k` against a Q8_K-quantised activation
- **THEN** the result SHALL match `dequantize_q6_k(w) · x` row-wise
  within ≤ 1.5 % relative error (Q8_K activation noise envelope)
<!-- test: larql_compute::cpu::ops::q4k_q8k_dot::tests::q6k_q8k_matvec_matches_canonical_dequant -->

#### Scenario: Q4_K matvec output matches canonical dequant + dot

- **WHEN** `q4k_q8k_matvec_scalar` and the dispatched
  `q4k_q8k_matvec_into` run on a Q4_K weight matrix produced by
  `quantize_q4_k` against a Q8_K-quantised activation
- **THEN** both SHALL match `dequantize_q4_k(w) · x` row-wise within
  ≤ 1.5 % relative error, and the dispatched output SHALL be bit-exact
  vs the scalar reference
<!-- test: larql_compute::cpu::ops::q4k_q8k_dot::tests::q4k_q8k_matvec_matches_canonical_dequant -->

#### Scenario: `q6k_matvec::dispatch` matches canonical dequant

- **WHEN** the trait-dispatched f32-input `q6k_matvec::dispatch` runs on
  the same Q6_K weights
- **THEN** the result SHALL match `dequantize_q6_k(w) · x` row-wise
  within ≤ 1e-4 relative error (no activation quant noise on the f32
  path)
<!-- test: larql_compute::cpu::ops::q6k_matvec::tests::q6k_dispatch_matches_canonical_dequant -->

### Requirement: AVX2 Q6_K matvec dispatch on x86_64 SHALL be bit-exact vs scalar

On `x86_64` with the `avx2` feature, `q6k_q8k_matvec_into` SHALL
dispatch to an AVX2 implementation (`q6k_q8k_matvec_avx2`) that uses
the sign-trick `_mm256_sign_epi8` + `_mm256_maddubs_epi16` +
`_mm256_madd_epi16` pattern. The AVX2 output SHALL be bit-exact (same
f32 bit pattern) vs the scalar reference; reduction order across the
two paths SHALL be identical.

On all other targets (no AVX2 detected, or non-x86_64), dispatch SHALL
fall through to the scalar implementation.

#### Scenario: AVX2 output bit-exact vs scalar on x86_64

- **WHEN** `q6k_q8k_matvec_avx2` runs on x86_64-with-AVX2 alongside
  `q6k_q8k_matvec_scalar` on the same inputs
- **THEN** every output element's `to_bits()` SHALL be identical
<!-- test: larql_compute::cpu::ops::q4k_q8k_dot::tests::q6k_q8k_matvec_avx2_matches_scalar -->

#### Scenario: NEON Q6_K dispatch is not used on aarch64 until re-vectorised

- **WHEN** `q6k_q8k_matvec_into` runs on aarch64
- **THEN** dispatch SHALL fall through to the scalar implementation
  (the legacy `q6k_q8k_matvec_neon` reads an incompatible layout and is
  kept only as an `#[allow(dead_code)]` reference)
<!-- test: unbacked -->

### Requirement: Cross-path parity between f32 trait and Q8K AVX2 Q6_K matvec

The two production Q6_K matvec entry points SHALL agree on identical
Q6_K weights within Q8_K activation noise.

`q6k_matvec::dispatch` (f32-input scalar) and `q6k_q8k_matvec_into`
(Q8K-input AVX2) are both reached by production callers — attention V
projection / lm-head KNN through the trait; walk-ffn-q8k FFN_DOWN / MoE
experts through the Q8K path — so any future divergence in one alone
would silently corrupt output.

#### Scenario: f32 and Q8K Q6_K paths agree within Q8_K noise

- **WHEN** both `q6k_matvec::dispatch` (f32 input) and
  `q6k_q8k_matvec_into` (Q8_K input) run on the same Q6_K weight matrix
  and the same `x`
- **THEN** outputs SHALL agree row-wise within ≤ 1.5 % relative error
<!-- test: larql_compute::cpu::ops::q4k_q8k_dot::tests::q6k_two_production_paths_agree_within_q8k_noise -->

### Requirement: `quantize_q6_k` SHALL round-trip through canonical dequant

`cpu::ops::q4_common::quantize_q6_k` SHALL produce byte layout correctly
readable by the canonical `larql_models::quant::ggml::dequantize_q6_k`,
with element-wise reconstruction within Q6_K's expected quantisation
envelope and cosine similarity ≥ 0.9999 on smooth inputs.

This function is called by vindex extraction for every Q6_K weight
written — V projections under `write_q4k/attn.rs` and FFN_DOWN under
default `--down-q4k=false` per `write_q4k/ffn.rs` — so any layout drift
would silently corrupt every vindex Q6_K matvec downstream.

#### Scenario: Round-trip preserves element-wise magnitude and direction

- **WHEN** `dequantize_q6_k(quantize_q6_k(x))` runs on smooth input
- **THEN** per-element relative error SHALL be ≤ 5 % (or absolute ≤
  5e-3 near zero), and full-vector cosine similarity SHALL be ≥ 0.9999
<!-- test: larql_compute::cpu::ops::q4_common::tests::q6_k_quantize_dequantize_roundtrip_within_quant_eps -->

### Requirement: `walk_ffn_q8k` SHALL dispatch FFN_DOWN on per-tensor format

`larql_inference::vindex::q4k_ffn_forward_layer_q8k` SHALL dispatch the
down-projection matvec on the per-tensor format string returned by
`VectorIndex::interleaved_q4k_layer_data` at `ffn[2].1`.

This function is the CPU-fallback path for the `/v1/walk-ffn-q8k`
endpoint. Specifically:

- `"Q4_K"` → `q4k_q8k_matvec_into`
- `"Q6_K"` → `q6k_q8k_matvec_into`
- Any other format → fall through to the existing f32-dequant slow path

Vindex extraction defaults to Q6_K FFN_DOWN under
`format/weights/write_q4k/ffn.rs:74`'s `is_down && !opts.down_q4k`
gate. A blanket call to `q4k_q8k_matvec_into` on Q6_K bytes silently
mis-parses 210-byte Q6_K blocks as 144-byte Q4_K blocks and corrupts
every FFN delta through this endpoint.

#### Scenario: Q6_K FFN_DOWN routes through Q6_K kernel

- **WHEN** the vindex stores Q6_K bytes at `ffn[2].0` and
  `ffn[2].1 == "Q4_K"` / `"Q6_K"`
- **THEN** the dispatch SHALL pick the matching kernel; mis-routing
  Q6_K bytes through a Q4_K parser SHALL NOT occur
<!-- test: unbacked -->
