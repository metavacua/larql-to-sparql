## ADDED Requirements

### Requirement: with_q8k_for cache key MUST be content-aware

`with_q8k_for` in `larql_models::quant::lazy` SHALL not return stale
Q8_K bytes for an input `x` whose pointer and length happen to match a
previously-cached input but whose content differs. The cache key SHALL
include a content fingerprint (currently `x[0]` and `x[len-1]` as f32
bit patterns); a fingerprint mismatch SHALL invalidate the cache and
trigger re-quantisation.

This addresses the allocator-reuse hazard documented in the prior
`with_q8k_for` doc comment: when a caller drops `x` and a fresh
`Vec<f32>` lands at the same heap address (e.g., PR #144's lm_head
`last_2d.row(0).to_owned()` invoked once per decode step), the old
`(ptr, len)` key matches the new `x` but the cached Q8_K bytes
correspond to the OLD content.

#### Scenario: per-step lm_head matvec sees fresh quantisation each call

- **GIVEN** a `QuantTensor` (Q4_K) with `lm_head` shape (262144, 2560)
  loaded from a Gemma 3 4B vindex
- **WHEN** the caller invokes `QuantTensor::matvec` twice in succession,
  each time with a freshly-allocated `Array1<f32>` of length 2560 whose
  Vec storage happens to be reused at the same heap address (typical
  Rust allocator behaviour)
- **THEN** the second call SHALL re-quantise `x` (not return the first
  call's stale Q8_K bytes); the matvec output SHALL be deterministic
  for identical input content
<!-- test: unbacked -->

#### Scenario: MoE-style cache reuse still hits the cache

- **GIVEN** a single `Array1<f32>` activation `x` held across multiple
  sibling matvec calls (the MoE FFN pattern that originally motivated
  the cache)
- **WHEN** `with_q8k_for` is invoked multiple times back-to-back with
  the same `&x`
- **THEN** the second and subsequent calls SHALL hit the cache (no
  re-quantisation); pointer and content fingerprints both match
<!-- test: unbacked -->
