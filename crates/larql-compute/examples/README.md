# larql-compute examples

Examples in three groups. Run any with:

```
cargo run --release --features metal -p larql-compute --example <name>
```

## Demos — show the API

| Example | What it does |
|---|---|
| `demo_basic` | Auto-detects the best backend, calls `matmul_transb` and a Q4 matvec. The 5-line "hello, world" of the crate. |
| `demo_architecture` | Guided tour of the major design points — `ComputeBackend` trait, `KernelHandle`, `quant_matvec`, `Capability`. Useful as a code-driven crate intro. |
| `demo_ridge_solve` | `ridge_decomposition_solve` — the closed-form ridge solve that underlies MEMIT-style weight edits. Linalg-side, no Metal needed. |

## Compares — full-pipeline benchmarks

End-to-end decode/generation throughput. Different surface from `benches/quant_matvec.rs`
(which measures kernel-level throughput). Run with `--release --features metal`.

| Example | What it measures |
|---|---|
| `compare_decode` | Q4_K decode latency through `decode_token` with KV cache. The production decode path. |
| `compare_formats` | Q4_KF (pre-baked scales) vs Q4_K vs Q8 — quant-format tradeoff. |
| `compare_generation` | End-to-end token generation throughput — the headline tok/s figure. |
| `compare_ollama` | Head-to-head LARQL vs Ollama on the same machine, same model. |
| `compare_pipeline` | Q4_K fused-QKV vs Q8 fused-QKV through `full_pipeline_q4`. |

For kernel-level throughput regressions, use the criterion bench suite:

```
make bench-compute   # quant_matvec Criterion bench with Metal
cargo bench -p larql-compute --bench matmul
cargo bench -p larql-compute --bench linalg
```

## Diagnostics (`diag_*`) — investigate production issues

These are operational tools, not tutorials. They answer specific questions
about where time goes or why output diverges. They require `--features metal`
and a real vindex or production-shape synthetic data.

| Example | Question it answers |
|---|---|
| `diag_profile_kernels` | **Where does GPU time go per kernel?** Measures each production kernel (q6k_matvec, q4k_ffn_gate_up, QKV, lm_head) in isolation and batched (34× in one command buffer). Reports GB/s vs theoretical peak, revealing compute-bound vs bandwidth-bound. |
| `diag_shader_bench` | **Did a kernel performance change regress?** Runs smoke or Gemma-3-shaped shader microbenches, optionally writing/comparing JSON. |
| `diag_decode_pipeline` | **Which layer/stage first diverges from CPU?** Per-stage buffer reads with `LARQL_METAL_DUMP_LAYERS=<dir>` for bisecting CPU/Metal divergence. |

Usage:

```bash
# Per-kernel bandwidth profiler — runs 50 iterations per kernel, batched x34
cargo run --release --features metal -p larql-compute --example diag_profile_kernels

# Shader benchmark inventory / JSON comparison
cargo run --release --features metal -p larql-compute --example diag_shader_bench -- --profile smoke

# Decode pipeline stage bisect — dumps per-stage f32 files for diffing
LARQL_METAL_DUMP_LAYERS=/tmp/decode_dump \
cargo run --release --features metal -p larql-compute --example diag_decode_pipeline
```

### When to use each

| Symptom | Tool |
|---|---|
| Overall tok/s regressed | `larql bench` + criterion bench suite |
| Specific kernel slower than expected | `diag_profile_kernels` |
| Metal and CPU produce different outputs | `diag_decode_pipeline` + `larql-inference/tests/test_decode_stage_bisect.rs` |
| NaN appearing in decode | `LARQL_DECODE_DIAG_LAYER=<n>` env var in `decode/diag.rs` |
