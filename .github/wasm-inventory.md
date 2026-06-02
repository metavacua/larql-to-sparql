# WASM boundary inventory

Living document. Update after each CI run that changes the boundary.
Last scan: run 26594114167 (PR#70, 2026-05-28).

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | passes |
| ❌ᶜ | fails at compile time |
| ❌ʳ | fails at runtime |
| ⏸ | not yet attempted / setup failure |

Runtimes:
- **wasm32v1-none** — `cargo check --target wasm32v1-none` (WASM MVP 1.0; no std)
- **wasmi** — `cargo test --target wasm32v1-none` with `wasmi run --` as the test runner
- **node** — `cargo test --target wasm32-unknown-unknown` via `wasm-bindgen-test-runner` in Node.js 22
- **firefox** — same, headless Firefox (`--browser --headless --firefox`)
- **pyodide** — `cargo build --target wasm32-unknown-emscripten` (larql-python only)

## Status matrix

| crate | wasm32v1-none | wasmi | node | firefox | pyodide |
|-------|:---:|:---:|:---:|:---:|:---:|
| kv-cache-benchmark | ❌ᶜ A | ❌ᶜ A | ❌ᶜ E | ❌ᶜ E | — |
| larql-boundary | ❌ᶜ A | ❌ᶜ A | ❌ᶜ E | ❌ᶜ E | — |
| larql-cli | ❌ᶜ A | ❌ᶜ A | ❌ᶜ E | ❌ᶜ E | — |
| larql-compute | ❌ᶜ A+C | ❌ᶜ A+C | ❌ᶜ C+E | ❌ᶜ C+E | — |
| larql-core | ❌ᶜ A | ❌ᶜ A | ❌ᶜ B+E | ❌ᶜ B+E | — |
| larql-experts | ❌ᶜ A† | ❌ᶜ A† | ❌ᶜ E | ❌ᶜ E | — |
| larql-inference | ❌ᶜ A | ❌ᶜ A | ❌ᶜ D+E | ❌ᶜ D+E | — |
| larql-kv | ❌ᶜ A | ❌ᶜ A | ❌ᶜ D+E | ❌ᶜ D+E | — |
| larql-lql | ❌ᶜ A | ❌ᶜ A | ❌ᶜ D+E | ❌ᶜ D+E | — |
| larql-models | ❌ᶜ A | ❌ᶜ A | ❌ᶜ E | ❌ᶜ E | — |
| larql-python | ❌ᶜ A | ❌ᶜ A | ❌ᶜ E | ❌ᶜ E | ⏸ F |
| larql-router | ❌ᶜ A | ❌ᶜ A | ❌ᶜ D+E | ❌ᶜ D+E | — |
| larql-router-protocol | ❌ᶜ A | ❌ᶜ A | ❌ᶜ D+E | ❌ᶜ D+E | — |
| larql-server | ❌ᶜ A | ❌ᶜ A | ❌ᶜ D+E | ❌ᶜ D+E | — |
| larql-vindex | ❌ᶜ A | ❌ᶜ A | ❌ᶜ E | ❌ᶜ E | — |
| model-compute | ❌ᶜ A | ❌ᶜ A | ❌ᶜ E | ❌ᶜ E | — |

† larql-experts additionally needs `#[global_allocator]` and `#[panic_handler]`.

> **Note on node/firefox status:** The CI run used before this redesign built to
> wasm32-wasip1 (not Node.js). That run showed larql-boundary, larql-experts,
> larql-models, model-compute as passing the old wasm32-wasip1 node column.
> Status above reflects the corrected wasm-bindgen-test-runner design.

---

## Blocking categories

### A — std dependency (blocks wasm32v1-none and wasmi)

wasm32v1-none is a no_std-only target. Every crate currently links std
directly or through transitive deps (serde, thiserror, etc.).

**What is needed per crate:**
- `#![no_std]` at crate root
- `extern crate alloc;` for heap allocation
- `#[global_allocator]` (larql-experts, any cdylib)
- `#[panic_handler]` (larql-experts, any no_std binary)
- Feature-gate any remaining std items under `#[cfg(feature = "std")]`

### B — `reqwest::blocking` import (blocks wasm32-wasip1 / node)

**File:** `crates/larql-core/src/engine/http_provider.rs:2`

```rust
use reqwest::blocking::Client;   // ← needs cfg gate
```

reqwest gates its blocking module with `#[cfg(not(target_arch = "wasm32"))]`.
The import must be wrapped accordingly.

### C — C FFI / build.rs (blocks wasm32-wasip1 / node / wasm32v1-none)

**Files:**
- `crates/larql-compute/build.rs`
- `crates/larql-compute/csrc/q4_dot.c`

Error observed (wasm32-wasip1 target): `'bits/libc-header-start.h' file not found`

The C kernel compilation cannot cross-compile to WASM without WASI-SDK headers.
The `build.rs` must skip C compilation when `CARGO_CFG_TARGET_ARCH == "wasm32"`.

### D — tokio wasm restrictions (blocks wasm32-wasip1 / node)

**File:** `tokio-1.52.3/src/lib.rs:478` (via `crates/larql-inference/Cargo.toml`)

Error: `Only features sync,macros,io-util,rt,time are supported on wasm.`

tokio's full feature set is incompatible with wasm32-wasip1. Crates that pull in
tokio with `features = ["full"]` (larql-inference, larql-server, larql-router,
larql-router-protocol, larql-lql, larql-kv) need a wasm-compatible subset, or the
tokio dependency must be feature-gated.

### E — getrandom "js" feature (blocks wasm32-unknown-unknown / node / firefox)

**Dep:** `getrandom 0.2.17` (transitive through rand, ring, or directly)

Error: `the wasm*-unknown-unknown targets are not supported by default, you may
need to enable the "js" feature. See https://docs.rs/getrandom/#webassembly-support`

Any crate whose dep tree includes getrandom must enable
`getrandom = { features = ["js"] }` when targeting wasm32-unknown-unknown, OR
route around getrandom entirely on that target.

### F — Emscripten toolchain (blocks pyodide / wasm32-unknown-emscripten)

larql-python builds with maturin (PyO3). Emscripten cross-compilation for pyodide
requires the full emsdk toolchain and a matching PyO3 version that supports
wasm32-unknown-emscripten. Neither is currently configured.

---

## Native open issues

| Job | Observed in run | Root cause | Fix location |
|-----|-----------------|-----------|--------------|
| `native · larql-compute · windows` | 26594097762 | `assertion failed: max_diff(&routed, &fallback) < 1e-5` at `crates/larql-compute/src/backend/helpers.rs:79`. Test: `backend::helpers::tests::dot_proj_gpu_some_backend_matches_fallback`. Numerical precision difference between Windows OpenBLAS and macOS Accelerate. Passed in run 26594114167 — potentially flaky. | `crates/larql-compute/src/backend/helpers.rs` — widen tolerance for non-Accelerate BLAS |

---

## Remediation priority

1. **Category A** (no_std) — blocks all WASM columns for all 16 crates. Highest leverage.
2. **Category E** (getrandom js) — blocks node/firefox once A is resolved.
3. **Category B** (reqwest::blocking) — single-file fix, unblocks larql-core node/firefox.
4. **Category C** (C FFI) — build.rs change, unblocks larql-compute WASM builds.
5. **Category D** (tokio) — requires either feature split or async redesign per crate.
6. **Category F** (pyodide) — requires emsdk setup and PyO3 emscripten support.
