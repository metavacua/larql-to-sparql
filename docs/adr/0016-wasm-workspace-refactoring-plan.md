# ADR-0016 — larql-wasm Workspace Refactoring Plan

**Status:** Proposed  
**Date:** 2026-06-06  
**Depends on:** ADR-0013 (LQL ↔ wasm32v1-none correspondence)

---

## Context

The `crates/larql-wasm/` directory is a separate 36-member Cargo workspace.
It serves as a **refactoring sandbox**: results can be verified against the root
workspace implementation without touching the working production code.

The workspace currently has four crate families:

| Family | Count | Purpose |
|--------|-------|---------|
| `*-wasm32v1-none` | 15 | Pure engine implementations — no `cfg` gates, compile to any target |
| `*-interface` | 15 | IO stratification layer — `cfg`-gated wrappers over `*-wasm32v1-none` |
| Bridge | 3 | `larql-bridge`, `larql-bridge-browser`, `larql-bridge-native` — JS/native entrypoints |
| Utilities | 3 | `larql-wasm-math`, `larql-wasm-certify`, `larql-vindex-mvv-cdylib` |

The `*-wasm32v1-none` crates currently depend **only** on each other — never on
`-interface` crates or root workspace crates. This has been verified (June 2026).

The `*-interface` crates are currently copies of the root workspace crates. The goal
of this refactoring plan is to **invert** that relationship: `*-interface` should
depend on `*-wasm32v1-none` (the pure implementation) and add `cfg`-gated IO wrappers
on top — it must not duplicate source from `*-wasm32v1-none`.

---

## Two-Crate Invariant

The entire refactoring is governed by one invariant with two halves:

**`*-wasm32v1-none` crates: unconditionally pure.**
These crates contain **no `cfg` gates of any kind**. They compile to any target —
`wasm32v1-none`, native, `wasm32-unknown-unknown`, anything — without modification.
Any code that requires a `cfg` gate to compile does not belong in these crates.
This is what makes formal ¬L∧¬M certification possible: the certifier sees a fixed,
unconditional call graph with no conditional branches at the type system level.

**`*-interface` crates: explicit IO stratification.**
These crates depend on `*-wasm32v1-none` for all pure logic. They add
`#[cfg(feature = "...")]`-gated wrappers for each external resource tier the crate
needs. They do not duplicate source from `*-wasm32v1-none`. The feature gates are
stratified by external resource type, not by target architecture.

---

## IO Stratification Ladder

The refactoring stratifies code by **external resource**, not by target architecture.
The architecture target is a consequence of which resources the code requires.

### Resource stratification table

| Feature gate | External resource | Practical examples | Minimum deployment target |
|---|---|---|---|
| (none — pure) | None | dot product, gate scoring, dequant arithmetic, tokenizer lookup | `wasm32v1-none` — any target |
| `cpu-cache` | CPU cache / SIMD registers | NEON/AVX blocked matmul, prefetch, cache-line-aware loads | native; `wasm32-unknown-unknown` with SIMD proposal |
| `ram` | OS-backed virtual memory | large mmap-adjacent allocations, VirtualAlloc-scale heap growth | native; `wasm32-wasip1` |
| `browser-js` | Browser JS host | `wasm-bindgen`, JS callbacks, `window`, `fetch`, DOM | `wasm32-unknown-unknown` + `wasm-bindgen` |
| `posix` | POSIX libc | clock, env vars, thread-local storage, `pthreads` | `wasm32-unknown-emscripten`; native |
| `filesystem` | Persistent storage | `mmap`, `std::fs`, vindex file reads, patch writes, WASI `fd_read` | `wasm32-wasip1`; native |
| `network` | Network stack | HTTP client, gRPC, TCP/UDP, SPARQL endpoint, peer sync | native server |
| `gpu` | GPU compute device | Metal kernels, WGPU shaders, WebGPU compute | native (Metal/Vulkan/DX12); browser (WebGPU) |

### Deployment shell feature sets

| Deployment target | Feature gates enabled |
|---|---|
| browser wasm32v1-none (certified dialect) | (none — pure only) |
| browser + JS interop | `browser-js` |
| browser + WebGPU | `browser-js` + `gpu` |
| WASI server | `filesystem` |
| native CLI | `cpu-cache` + `ram` + `filesystem` |
| native server | `cpu-cache` + `ram` + `filesystem` + `network` |
| native GPU server | `cpu-cache` + `ram` + `filesystem` + `network` + `gpu` |

### Compilation probe order

Each probe target is also a gate: code that fails to compile at level N requires a
resource in the feature set at level N+1 or above.

```
wasm32v1-none (pure: no feature gates)
    ↓
wasm32-unknown-unknown + wasm-bindgen  (adds: browser-js)
    ↓
wasm32-unknown-emscripten              (adds: posix)
    ↓
wasm32-wasip1                          (adds: filesystem)
    ↓
native                                 (adds: cpu-cache, ram, network, gpu)
```

---

## Phase 0 — Completed (June 2026)

- Verified: all `*-wasm32v1-none` crates depend only on each other
- Verified: all `-interface` crates are byte-for-byte copies of native
- MVV (Minimal-Viable Vindex) gate-KNN kernel certified: 0 imports, 0 `call_indirect`
- ADR-0013 written: browser LQL dialect defined (WALK, DESCRIBE, SELECT, INSERT/KNN)

No changes to the workspace structure were made in Phase 0.

---

## Phase 1 — Interface Inversion

**Goal:** Make `-interface` crates the IO stratification layer over `*-wasm32v1-none`.

**Steps per crate pair (e.g. `larql-compute-interface` ↔ `larql-compute-wasm32v1-none`):**

1. Add `larql-compute-wasm32v1-none` as a dependency in `larql-compute-interface/Cargo.toml`.
   Remove deps on other `-interface` crates; replace with deps on `*-wasm32v1-none` crates.
2. Rewrite `larql-compute-interface/src/` to re-export the pure implementation from
   `larql-compute-wasm32v1-none`. Do **not** copy source — re-export via `pub use`.
3. Add `#[cfg(feature = "...")]`-gated IO wrappers for each resource tier this crate
   touches (see resource stratification table above). Only gates that apply to this
   crate's actual OS surface are needed.
4. Verify: `cargo build --target wasm32v1-none -p larql-compute-interface` passes
   (the pure re-export path must compile unconditionally).
5. Verify: `cargo test -p larql-compute-interface` (native, all features) passes.

**Parity check after all crates are inverted:**
```bash
# Every native test must produce the same result as before inversion
cargo test --manifest-path crates/larql-wasm/Cargo.toml
```

**What changes:** `-interface` crates no longer contain duplicated implementation code.
They become re-export + IO-wrapper crates. Their source legitimately diverges from
`*-wasm32v1-none` source — diff-based parity checks are not meaningful after inversion.

**What does not change:** `*-wasm32v1-none` crate sources are untouched in Phase 1.
They must remain gate-free throughout.

---

## Phase 2 — Parity Verification

After Phase 1, verify that the inverted `-interface` crates are behaviorally identical
to the pre-inversion state. Source diffs are **expected and intentional** — the interface
crate re-exports rather than duplicates. Parity is verified by behavior:

```bash
# All native tests must produce identical results to pre-inversion
cargo test --manifest-path crates/larql-wasm/Cargo.toml

# The pure re-export path must compile under wasm32v1-none for every interface crate
for crate in model-compute larql-models larql-compute larql-vindex larql-core \
             larql-inference larql-kv larql-lql larql-boundary larql-router-protocol \
             larql-router larql-server larql-cli larql-python kv-cache-benchmark; do
  cargo build \
    --manifest-path crates/larql-wasm/Cargo.toml \
    --target wasm32v1-none \
    -p ${crate}-interface 2>&1 | grep -E "^error" && echo "FAIL: ${crate}" || echo "OK: ${crate}"
done
```

Any native test regression is a Phase 1 error — investigate before proceeding to Phase 3.

---

## Phase 3 — wasm-bindgen Probe

**Target:** `wasm32-unknown-unknown` with `wasm-bindgen`

**Purpose:** Browser JS interop layer — the most OS-dependency-free browser target.
`wasm32-unknown-unknown` has no implicit OS imports; `wasm-bindgen` provides the
JS ↔ Rust bridge via explicit `extern "C"` bindings.

**Probe:** Attempt to compile the browser LQL dialect evaluator (WALK, DESCRIBE,
SELECT, INSERT/KNN) targeting `wasm32-unknown-unknown` with wasm-bindgen:

```bash
cargo build \
  --manifest-path crates/larql-wasm/Cargo.toml \
  -p larql-lql-wasm32v1-none \
  --target wasm32-unknown-unknown \
  --features wasm-bindgen
```

Any compilation error identifies code that requires OS imports not present in
`wasm32-unknown-unknown`. That code belongs in Phase 4 (emscripten) or later.

---

## Phase 4 — Emscripten Probe

**Target:** `wasm32-unknown-emscripten`

**Purpose:** POSIX-over-WASM layer. Emscripten provides a POSIX compatibility shim
that translates `libc` calls to browser equivalents. Code that requires `libc` but
not a real filesystem passes at this level.

**Probe:** Compile under emscripten; identify code that passes Phase 3 but fails
under `wasm32-unknown-unknown` (OS-import code that emscripten can satisfy).

---

## Phase 5 — WASI Probe

**Target:** `wasm32-wasip1`

**Purpose:** WASI filesystem imports. The `wasm32-wasip1` target provides explicit
WASI host imports for filesystem access. Code that requires real filesystem access
(reading vindex files, writing patches) passes at this level.

The `larql-experts` workspace already targets `wasm32-wasip1` — that baseline shows
which patterns work at this level.

---

## Phase 6 — Native Stratification

Code that does not compile under any WASM32 target belongs in the native execution
context. Stratify native-only code into:

| Category | Destination |
|----------|-------------|
| Filesystem IO (mmap vindex, read weights, write patches) | `larql-vindex`, `larql-cli`, `larql-server` |
| Network IO (USE REMOTE, HTTP API, gRPC) | `larql-server`, `larql-router` |
| GPU (Metal, WGPU) | `larql-compute`, future `larql-gpu` |
| Process management (daemon, signal handling) | `larql-server` |

The design goal: **all shared library crates (`larql-vindex`, `larql-core`,
`larql-inference`) must compile under `wasm32v1-none` after Phase 6**. OS-specific
code lives only in `larql-cli` and `larql-server`.

---

## End State

After all 6 phases:

**The `*-wasm32v1-none` crates ARE the canonical engine implementations.**

The migration runs in the promotion direction:

1. Original native crates (`larql-vindex`, `larql-inference`, `larql-lql`, etc.) → **deleted**
2. `*-wasm32v1-none` crates → **promoted** into the root workspace as the new canonical engine
3. `-interface` crates → **remain as the IO adaptation layer** (or are absorbed into deployment shells — this architectural decision is deferred; it becomes tractable once the pure/IO boundary is established)

**`*-wasm32v1-none` crates contain no `cfg` gates.** They compile unconditionally to any target — native, `wasm32v1-none`, or anything else. They are pure by construction, not by conditional compilation.

**`-interface` crates carry the `cfg`-gated IO stratification**, stratified by external resource type:

| Feature gate | Resource | Examples |
|---|---|---|
| `cpu` | CPU computation only | arithmetic, dot product, gate scoring |
| `cpu-cache` | Cache-aware / SIMD | blocked matmul, NEON/AVX paths |
| `ram` | Heap allocation | dynamic Vec, HashMap, allocator |
| `filesystem` | Persistent storage | mmap, std::fs, vindex file reads |
| `network` | External communication | HTTP, gRPC, SPARQL endpoint, peer sync |

Each deployment shell (cli, server, browser-js, client, peer) enables only the resource features appropriate for its IO surface.

**larql-wasm workspace collapses from ~36 members to 3:**
- `larql-wasm-math` — FloatExt + FnvHasher, pure arithmetic utilities
- `larql-vindex-mvv-cdylib` — MVV certification artifact (minimum certifiable kernel)
- `larql-wasm-certify` — certifier tool

**Root workspace after promotion:**
- Engine: promoted `*-wasm32v1-none` crates — unconditionally pure, compile to any target
- IO adaptation: `-interface` crates (or their contents, absorbed into deployment shells)
- Deployment shells: cli, server, browser-js, client, peer — each composes the IO features it needs

**This enables:**
- Browser deployment of the certified LQL dialect (ADR-0013) via the `cpu`+`ram`-only interface feature set
- WASI-based deployment via the `filesystem` feature set
- Native server deployment via the full `filesystem`+`network` feature set
- Formal certification of the engine stack as ¬L∧¬M (certifier targets the `*-wasm32v1-none` cdylibs)

---

## Consequences

**Positive.**

- The 6-phase stratification provides an objective criterion for every line of code:
  it either compiles at the target level or it doesn't. No architectural judgment
  calls needed during stratification.
- Phase 1 (interface inversion) is independent of the root workspace. It cannot
  break production builds.
- Each phase is independently verifiable by a CI job targeting the appropriate WASM
  target.

**Negative.**

- Phase 1 requires touching every `-interface` crate (11 crates). This is
  mechanical but time-consuming.
- Phase 6 may require moving code from `larql-vindex` into `larql-cli` — a breaking
  change for any code that imports `larql-vindex` and uses its filesystem-dependent APIs.

**Not in scope.**

- The `larql-experts` workspace (already on `wasm32-wasip1`) is not part of this plan.
- GPU shader compilation for WASM (`wgpu`/WebGPU) is covered by ADR-0017.
- The `larql-python` cdylib is native-only; no WASM target for Python bindings is planned.
