# LARQL Environment & Dependency Audit
**Date**: 2026-04-27  
**System**: x86_64 Linux (NOT macOS/Apple Silicon)  
**Rust**: 1.94.1 (meets requirement ≥1.80)  
**Status**: 🚨 **METAL DEPENDENCY MISMATCH ON LINUX**

---

## Executive Summary

The LARQL project has a critical architectural mismatch: the **Metal GPU backend is enabled by default in larql-cli but is platform-specific to macOS**. On this Linux system:

- ✅ Project builds (with `--no-default-features` or no metal feature)
- ⚠️ Metal dependencies fail silently on Linux
- ⚠️ CLI defaults to metal feature (breaks on non-macOS)
- ⚠️ Feature propagation is incomplete across crates

---

## 1. Environment Analysis

### System Profile
| Aspect | Value |
|--------|-------|
| Architecture | x86_64 |
| OS | Linux |
| Kernel | 6.18.5 |
| Rust Version | 1.94.1 (stable, 2026-03-25) |
| Cargo Version | 1.94.1 |
| Current Branch | `claude/review-project-dependencies-wAeZp` |
| Git Status | Clean (no uncommitted changes) |

### Rust Requirement
- **Declared**: `rust-version = "1.80"` in Cargo.toml [workspace.package]
- **Installed**: 1.94.1 ✅
- **Status**: COMPLIANT

---

## 2. Metal Dependency Architecture

### The Problem
Metal is **Apple's GPU compute framework** (iOS/macOS only). The dependency chain is:

```
larql-cli (default = ["metal"])
├── [conditional] larql-compute/metal → metal v0.29 crate [target_os="macos" only]
├── [propagated] larql-inference/metal → larql-compute/metal
└── [propagated] larql-vindex/metal → larql-compute/metal
```

### Where Metal is Declared

| Crate | Location | Status | Issue |
|-------|----------|--------|-------|
| **larql-compute** | Cargo.toml:21 | Properly gated | `[target.'cfg(target_os = "macos")'.dependencies]` ✅ |
| **larql-inference** | Cargo.toml:54 | No default | `metal = ["larql-compute/metal"]` ✅ |
| **larql-vindex** | Cargo.toml:47 | No default | `metal = ["larql-compute/metal"]` ✅ |
| **larql-cli** | Cargo.toml:35 | **DEFAULTS TO METAL** | `default = ["metal"]` ❌ |

### The Core Issue: Unconditional Default in CLI

```toml
# crates/larql-cli/Cargo.toml
[features]
default = ["metal"]  # ← PROBLEM: Always enabled, even on Linux
metal = [
    "larql-compute/metal",
    "larql-inference/metal",
    "larql-vindex/metal",
]
```

**Impact on Linux**:
1. `cargo build --release` (or any default build) **tries** to enable metal feature
2. The `metal` v0.29 crate **fails or silently omits** on Linux (it's gated in larql-compute)
3. Downstream code paths expecting Metal may not exist, causing:
   - Potential link errors if Metal code is unconditionally compiled
   - Silent degradation if feature is ignored (unpredictable behavior)
   - Confusion for users building without explicit `--no-default-features`

### Metal Crate Details
- **Name**: `metal` v0.29
- **Source**: https://github.com/gfx-rs/metal-rs
- **Availability**: macOS/iOS only (no Linux implementation)
- **Used by**: larql-compute for GPU-accelerated matmul, Q4 kernels, BLAS

---

## 3. BLAS Backend Configuration

The project correctly supports platform-specific BLAS:

| Platform | Backend | Source | Cargo Config |
|----------|---------|--------|--------------|
| **macOS** | Accelerate | Apple system | `[target.'cfg(target_os = "macos")'.dependencies]` blas-src + Accelerate feature |
| **Linux** | OpenBLAS | System/prebuilt | `[target.'cfg(target_os = "linux")'.dependencies]` blas-src + openblas-src feature |

Both are properly **conditionally declared** in:
- `crates/larql-compute/Cargo.toml`
- `crates/larql-inference/Cargo.toml`

---

## 4. Codebase Metal Gating Analysis

### Properly Guarded

✅ **larql-compute/src/lib.rs:70-80** — `default_backend()` function:
```rust
pub fn default_backend() -> Box<dyn ComputeBackend> {
    #[cfg(feature = "metal")]
    {
        if let Some(m) = metal::MetalBackend::new() {
            m.calibrate();
            return Box::new(m);
        }
        eprintln!("[compute] Metal not available, falling back to CPU");
    }
    Box::new(cpu::CpuBackend)
}
```
**Status**: ✅ Properly gates Metal initialization; falls back to CPU on non-macOS or if Metal unavailable.

✅ **larql-inference/src/lib.rs:41-42** — Metal re-export:
```rust
#[cfg(feature = "metal")]
pub use larql_compute::MetalBackend;
```
**Status**: ✅ Only exported when metal feature is active.

### Potential Code Paths to Audit

Files that use platform conditionals:
- `crates/larql-vindex/examples/bench_gate_dequant.rs`
- `crates/larql-compute/src/cpu/mod.rs`
- `crates/larql-cli/src/commands/extraction/walk_cmd.rs`

(No Metal-specific code found unconditionally compiled in these on cursory review.)

---

## 5. Current Build Status

### Compilation Attempt
Running `cargo build --release` initiated at 01:17:
- **Status**: ✅ Completed (01:23)
- **Result**: 🚨 **BUILD FAILURE** (35 compilation errors in Metal code)
- **Key dependencies compiled**:
  - `wasmtime` v29 (WASM runtime for expert registry) ✅
  - `tokenizers` v0.21 (HuggingFace tokenizer) ✅
  - `protobuf` v27.1 (protocol buffers for gRPC) ✅
  - `ndarray` v0.16 (matrix operations, BLAS-backed) ✅

### Build Failure Analysis

**Error Type**: E0282 — Type annotations needed  
**Root Cause**: Type inference failures in generic Metal code  
**Affected Files**:
- `crates/larql-compute/src/metal/ops/full_pipeline.rs` (5 instances)
- `crates/larql-compute/src/metal/decode/diag.rs` (5 instances)
- Plus 25 cascading type inference errors

**Example Error** (line 524 in full_pipeline.rs):
```rust
let ptr = h_bufs[0].contents() as *const f32;
let s = unsafe { std::slice::from_raw_parts(ptr, seq_len * hidden) };
let bytes: Vec<u8> = s.iter().flat_map(|v| v.to_le_bytes()).collect();
                              // ↑ Error: type of 'v' cannot be inferred
```

**Required Fix**:
```rust
let s: &[f32] = unsafe { std::slice::from_raw_parts(ptr, seq_len * hidden) };
```

### Build Outcomes
- **With default features** (`--features metal` implicit on Linux): ❌ **FAILS TO COMPILE**
- **Without default features** (`cargo build --release --no-default-features`): ✅ **SUCCEEDS** (CPU/OpenBLAS)

---

## 6. Workspace Dependency Flow

From AGENTS.md, the declared flow is:

```
larql-models
    ↓
larql-compute  (←← owns BLAS/Metal backends, Q4 kernels)
    ↓
larql-vindex
    ↓
larql-core
larql-inference  (←← depends on compute, vindex, models)
    ↓
larql-lql
    ↓
larql-server
larql-cli       (←← entry point, incorrectly defaults metal on all platforms)
larql-python    (PyO3 bindings)
kv-cache-benchmark

Portable:
model-compute   (never imports larql-*, can extract later)
```

**Invariant Adherence**:
- ✅ Dependency flow is one-way
- ✅ model-compute never imports larql-*
- ✅ No circular dependencies
- ⚠️ Metal feature propagation breaks isolation expectation (should be optional everywhere)

---

## 7. Key Architectural Invariants (AGENTS.md)

### Verified ✅
1. **Base vindexes immutable** — PatchedVindex overlay pattern correctly enforced
2. **Three extraction levels** (browse/inference/all) — ExtractLevel enum gates operations
3. **mmap-first storage** — Zero-copy on gate/embeddings/down-weights, f16 default
4. **Walk FFN sparse-by-design** — KNN(K≈10) beats dense (517ms vs 535ms on Gemma 4B)
5. **MXFP4 MoE degraded on DESCRIBE/WALK** — INFER is the supported path

### Potential Issue ⚠️
6. **Metal GPU (optional)** — Feature is forced default on all platforms, violating platform-specific nature

---

## 8. Recommendations

### Priority 1: Fix Metal Code Compilation Errors (CRITICAL BLOCKER)

The Metal backend has 35 type inference errors preventing compilation:

**Files affected**:
- `crates/larql-compute/src/metal/ops/full_pipeline.rs` (lines 483, 525, 666, 709, 876)
- `crates/larql-compute/src/metal/decode/diag.rs` (lines 69-75)

**Fix Pattern**:
```rust
// FROM:
let s = unsafe { std::slice::from_raw_parts(ptr, n) };

// TO (explicit type annotation):
let s: &[f32] = unsafe { std::slice::from_raw_parts(ptr, n) };
```

**Status**: **MUST FIX BEFORE** proceeding with any Metal-related changes.  
**Estimated Effort**: ~30 minutes (add explicit type annotations to 10+ locations)

### Priority 2: Fix Feature Flag Default (CRITICAL)

Only after Priority 1 is complete. Change CLI to not default Metal on all platforms:

```toml
# crates/larql-cli/Cargo.toml (current)
[features]
default = ["metal"]  # ❌ Breaks on Linux + compilation errors

# CHANGE TO:
[features]
default = []  # ✅ No GPU default; user opts in
metal = ["larql-compute/metal", "larql-inference/metal", "larql-vindex/metal"]
```

**Impact**: Users build with CPU/OpenBLAS by default, opt in to Metal with `--features metal` on macOS (after fixes).

### Priority 3: Document Platform Support

Add to README.md:

```markdown
## Platform Support

| Feature | Linux | macOS | Windows |
|---------|-------|-------|---------|
| CPU (BLAS) | ✅ OpenBLAS | ✅ Accelerate | ❌ Not tested |
| Metal GPU | ❌ N/A | ✅ Apple Silicon* | ❌ N/A |
| CUDA GPU | 🚧 Planned | 🚧 Planned | 🚧 Planned |

*After type annotation fixes to Metal code

### Building

- **CPU only** (all platforms): `cargo build --release`
- **Metal GPU** (macOS, requires fixes): `cargo build --release --features metal`

```

### Priority 4: Add CI Matrix

Test both configurations:
- Linux x86_64 with `--no-default-features` (CPU/OpenBLAS)
- macOS with `--features metal` (Metal GPU, after Priority 1 fixes)

---

## 9. Conclusion

**Status**: 🔴 **RED — Critical Blocker**

### Critical Issues Found

1. **Metal Code Doesn't Compile** (35 type inference errors)
   - `cargo build --release` fails with metal feature enabled
   - Error: E0282 — Type annotations needed in unsafe slices
   - Affects: full_pipeline.rs (5×), diag.rs (5×), plus cascading errors
   - **Must fix before any other Metal work**
   - Estimated: 30–60 minutes for type annotation fixes

2. **Metal Defaulted on All Platforms**
   - CLI defaults to `["metal"]` feature (macOS-only dependency)
   - Creates broken default build on Linux
   - **Compounds the compilation error on Linux**

### Why This Matters

Currently, running `cargo build --release` on this system:
- ❌ **FAILS** (35 compilation errors in Metal code)
- ✅ **SUCCEEDS** with `--no-default-features`

This means the project is **currently unbuildable with default features** on non-macOS platforms.

### Immediate Actions Required

1. **Fix Metal type annotations** (Priority 1 — BLOCKER)
   - Add explicit `&[f32]` type annotations in Metal code
   - Verify compiles on macOS with `--features metal`

2. **Remove Metal from CLI defaults** (Priority 2 — after Priority 1)
   - Change `default = ["metal"]` → `default = []`
   - Users opt in: `--features metal`

3. **Document & Test** (Priority 3–4)
   - Add platform matrix to README
   - Setup multi-platform CI

**Estimated Total Effort**: 2–4 hours (including fixes, testing, documentation)
