# larql-compute benchmarks

Three Criterion benches, each scoped to one concern. Run any with:

```
cargo bench -p larql-compute --bench <name> --features metal
```

Reports land under `target/criterion/<bench>/` as HTML + raw JSON.

## The three benches

| Bench | Surface | Scope |
|---|---|---|
| **`quant_matvec`** | quantised matvec | Q4_0 / Q4_K / Q4_KF / Q6_K × {decode_2560, prefill_10240, lm_head_262144} × {cpu, metal}. The headline regression-detector — would have caught the `q4_matvec_v4` 75 %-row drop (4× cliff at `metal/lm_head_262144`) at PR time. |
| **`matmul`** | dense f32 / specialised gemv | CPU vs Metal `matmul_transb` at three shapes; Metal-only `f32_gemv` at the lm-head shape (row-per-simdgroup specialised kernel). |
| **`linalg`** | linear-algebra primitives | CPU-only Cholesky factor + solve, ridge-regression decomposition (the closed-form solve under `larql_vindex::memit_solve`). |

Adding a new format: add a `QuantFormat` variant + match arm in
`quant_matvec.rs`'s `bench_format` body. The cell shows up in the
HTML report alongside the existing formats automatically.

## Regression gating

Three Make targets wrap the suite:

```
make bench-compute   # run the primary quant_matvec bench with Metal
make bench-save      # record current results as the `main` baseline
make bench-check     # re-run; fail if any cell regressed past Criterion's noise threshold
```

`bench-save` and `bench-check` call `scripts/bench-regress.sh`, which gates
all three compute benches by default. Tunables:

| Env var | Default | Effect |
|---|---|---|
| `BASELINE_NAME` | `main` | Criterion baseline name |
| `THRESHOLD` | `0.10` | Per-cell regression threshold (informational; Criterion does its own significance check) |
| `BENCHES` | `quant_matvec matmul linalg` | Subset to run; pass e.g. `BENCHES=quant_matvec` to focus |
| `FEATURES` | `--features metal` | Cargo features for the bench build |

CI starter at `.github/workflows/bench-regress.yml` (saves baseline
on `main` pushes, runs `make bench-check` on PRs, treats a cold
cache as neutral).

## Why three benches and not one?

Each covers a *different layer of the abstraction stack*:

- `quant_matvec` measures **kernel** throughput (one matvec, one
  format). Catches kernel regressions in isolation.
- `matmul` measures **dense linear algebra** throughput. Distinct
  from quantised matvec — `matmul_transb` is the building block for
  prefill, `f32_gemv` is the lm-head fallback when the Q4 path can't
  be used.
- `linalg` measures **linear-algebra primitives** with no GPU surface.
  Cholesky + ridge solve are the closed-form operations under
  MEMIT-style weight edits.

For *full-pipeline* throughput (whole-decode-token, generation tok/s),
use `examples/compare_*` — those are end-to-end benchmarks that the
kernel-level criterion suite intentionally doesn't cover.

## Metal shader diagnostics

For a Metal shader inventory plus direct isolated/batched GPU timings,
use:

```
cargo run --release --features metal -p larql-compute --example diag_shader_bench
cargo run --release --features metal -p larql-compute --example diag_shader_bench -- --profile gemma3 --json /tmp/larql-shaders.json
cargo run --release --features metal -p larql-compute --example diag_shader_bench -- --profile gemma3 --compare /tmp/larql-shaders.json --threshold 5
```

The shader bench is diagnostic rather than Criterion-based. Treat the
batched column as the promotion signal; isolated timings include
per-call command-buffer overhead and can make candidate kernels look
better than they are in decode. `--compare` reads a prior JSON file
from this tool and reports per-kernel `batched_ms` deltas.
