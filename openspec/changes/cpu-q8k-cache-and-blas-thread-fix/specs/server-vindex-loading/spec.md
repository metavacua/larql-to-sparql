## ADDED Requirements

### Requirement: larql-server MUST pin BLAS thread count to 1 at startup

`larql-server`'s `main()` SHALL set `OPENBLAS_NUM_THREADS=1`,
`OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1` env vars (if not already
set by the caller) and SHALL also call `openblas_set_num_threads(1)`
via FFI at process startup, before any compute can run. The hot-path
matvecs (FFN, attention Q/K/V/O, lm_head — PRs #142/#143/#144) are
all rayon-parallel AVX2 Q4_K × Q8_K; the remaining BLAS calls are
small per-head dots in `gqa_attention_decode_step` where BLAS's
fork-join overhead dominates and amortises poorly across 48 threads.

Caller-supplied env var overrides SHALL be honoured: a user setting
`OPENBLAS_NUM_THREADS=N` in the environment retains that value.

#### Scenario: Default startup yields BLAS_NUM_THREADS=1

- **GIVEN** a clean environment (no `OPENBLAS_NUM_THREADS` set)
- **WHEN** `larql-server` boots
- **THEN** the env var SHALL be 1 from process startup onwards; AND
  `openblas_set_num_threads(1)` SHALL have been called before the
  tokio runtime spawns any worker
<!-- test: unbacked -->

#### Scenario: Caller-supplied OPENBLAS_NUM_THREADS overrides the default

- **GIVEN** an environment with `OPENBLAS_NUM_THREADS=4` set by the
  caller
- **WHEN** `larql-server` boots
- **THEN** the env var SHALL remain `4`; the FFI call still runs but
  callers who explicitly want multi-threaded BLAS keep their setting
<!-- test: unbacked -->

#### Scenario: End-to-end decode runs at the post-rayon target

- **GIVEN** a Gemma 3 4B Q4_K_M vindex
- **WHEN** the server serves `/v1/chat/completions` with a 150-token
  max completion at temperature 0
- **THEN** decode SHALL complete at ≥ 8 tok/s on a 48-thread EPYC host
  (empirical measurement: 9.81 tok/s); AND output SHALL be coherent
  English text matching the structure produced by the same vindex on
  llama.cpp at the same temperature
<!-- test: unbacked -->
