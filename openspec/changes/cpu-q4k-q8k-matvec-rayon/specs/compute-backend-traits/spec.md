## ADDED Requirements

### Requirement: AVX2 Q4_K / Q6_K × Q8_K matvec MUST scale across CPU threads

`q4k_q8k_matvec_avx2` and `q6k_q8k_matvec_avx2` SHALL dispatch the row
loop across multiple CPU threads via `rayon::par_chunks_mut` when
`rows >= MIN_PAR_ROWS` (currently 16). The parallel path SHALL chunk
rows into approximately `rows / rayon::current_num_threads()` per
chunk, with a minimum chunk size of 4 rows to amortise per-task
dispatch overhead.

For `rows < MIN_PAR_ROWS`, the kernels SHALL dispatch sequentially —
the per-task overhead of rayon does not amortise on small matvecs
(unit-test shapes 5..7 rows fall into this range).

#### Scenario: Production decode-step matvecs use the parallel path

- **GIVEN** Gemma 3 4B Q4_K decode shapes (Q proj `rows=2048`, K/V proj
  `rows=1024`, FFN gate/up `rows=10240`, FFN down `rows=2560`, O proj
  `rows=2560`, lm_head not affected by this kernel)
- **WHEN** `q4k_q8k_matvec_into` or `q6k_q8k_matvec_into` is called
  on x86_64 with AVX2 detected
- **THEN** the row loop SHALL execute in parallel across the host's
  rayon thread pool (chunk_rows = `rows / current_num_threads()`,
  min 4)
<!-- test: unbacked -->

### Requirement: Parallel matvec output MUST be bit-exact vs the scalar reference

The parallel AVX2 output SHALL be bit-exact (same f32 bit pattern) vs the scalar reference for both Q4_K and Q6_K matvec, for all `rows` and `cols` combinations including the unit-test shapes (`rows=5,7`, `cols=512`) and production shapes (`rows ∈ 1024..10240`, `cols ∈ 2048..2560`). Per-row reduction order is preserved by the parallel dispatch — each row's accumulator stays thread-local.

#### Scenario: AVX2 parallel output matches scalar bit-exactly

- **WHEN** `q4k_q8k_matvec_into` (AVX2, parallel for rows ≥ 16) and
  `q4k_q8k_matvec_scalar` (sequential reference) are called with the
  same `q8k_x`, `w`, `rows`, `cols`
- **THEN** every element of the AVX2 output SHALL have the same f32
  bit pattern as the corresponding scalar output
<!-- test: larql_compute::cpu::ops::q4k_q8k_dot::tests::q8k_matvec_avx2_matches_scalar -->

#### Scenario: Q6_K AVX2 parallel output matches scalar bit-exactly

- **WHEN** `q6k_q8k_matvec_into` (AVX2, parallel for rows ≥ 16) and
  `q6k_q8k_matvec_scalar` (sequential reference) are called with the
  same arguments
- **THEN** every element of the AVX2 output SHALL be bit-exact vs the
  scalar reference
<!-- test: larql_compute::cpu::ops::q4k_q8k_dot::tests::q6k_q8k_matvec_avx2_matches_scalar -->
