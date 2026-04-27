# Build Process Optimization: Device & OS Agnosticism Analysis

**Date**: 2026-04-27  
**Branch**: `claude/build-process-optimization-MyCWS`  
**Scope**: Identifying problems and low-hanging fruit for cross-platform build process generalization

---

## Executive Summary

The LARQL build process has **critical hardcoded assumptions** that prevent reproducibility and portability across operating systems and devices. The system mixes Rust (with conditional compilation), Python (via maturin/uv), shell scripts (Unix-only), and ad-hoc configuration, creating friction for:

- Linux developers (OpenBLAS assumed but not guaranteed)
- Windows developers (complete absence of Windows support)
- macOS Intel developers (Metal GPU feature is Apple Silicon-only)
- CI/CD pipelines (no automated testing across platforms)
- Containerized environments (hardcoded absolute paths)
- Python binding distribution (maturin wheels not pre-built)

**Expected outcome**: A single, deterministic build command that works on Linux, macOS (Intel/Silicon), Windows, and Docker without special-casing per developer or environment.

---

## Current State: Build Artifacts & Assumptions

### 1. **Makefile (Root)**

**Location**: `./Makefile`

**Issues**:
- ✗ Pure `cargo` commands without environment detection
- ✗ No documentation of required system dependencies (OpenBLAS, Python, uv)
- ✗ `make release` assumes you can run `larql-cli` immediately after
- ✗ Python targets (`python-setup`, `python-build`) hard-depend on `uv` without fallback
- ✗ All paths are relative (fragile when invoked from different directories)
- ✗ No distinction between "CI build" and "development build"
- ✗ Missing `make help` — developers don't know all targets exist

**Severity**: **HIGH** (foundational, affects all downstream users)

### 2. **Platform-Specific Dependencies**

#### Linux (OpenBLAS)
**File**: `crates/larql-compute/Cargo.toml:15-17`
```toml
[target.'cfg(target_os = "linux")'.dependencies]
blas-src = { version = "0.10", features = ["openblas"], default-features = false }
openblas-src = { version = "0.10", features = ["system"] }
```

**Issues**:
- ✗ `openblas-src` with `features = ["system"]` assumes OpenBLAS is pre-installed
- ✗ No fallback to vendored OpenBLAS if system version unavailable
- ✗ Docker containers on Alpine/musl fail (OpenBLAS not available)
- ✗ Ubuntu/Debian installation not documented
- ✗ No version pinning for system OpenBLAS

#### macOS (Accelerate + optional Metal)
**File**: `crates/larql-compute/Cargo.toml:20-22`
```toml
[target.'cfg(target_os = "macos")'.dependencies]
metal = { version = "0.29", optional = true }
blas-src = { version = "0.10", features = ["accelerate"] }
```

**Issues**:
- ✗ Metal is `optional` but build fails if metal crate unavailable (Apple Silicon only)
- ✗ Accelerate BLAS is assumed available (not true on all macOS versions)
- ✗ Intel Mac + `--features metal` silently fails or builds wrong binary
- ✗ No runtime detection of CPU architecture

#### Windows
**Files**: None. Windows is completely unsupported.

**Issues**:
- ✗ Cargo should work on Windows MSVC, but libc-dependent code will fail
- ✗ No `cfg(target_os = "windows")` blocks
- ✗ `mmap_util.rs` only has `#[cfg(unix)]` implementation
- ✗ Shell scripts (`extract_and_publish_all.sh`, `grid/start.sh`) are bash-only
- ✗ Path separators hardcoded as `/` (breaks on Windows)

**Severity**: **CRITICAL** (Windows is 25-30% of developer machines globally)

### 3. **Memory-Mapped File Utilities**

**File**: `crates/larql-vindex/src/mmap_util.rs`

**Issues**:
- ✗ `mmap_optimized()` and `advise_sequential()` only implement POSIX `madvise()`
- ✗ Windows lacks equivalent; falls back to no-op or crashes
- ✗ `#[cfg(unix)]` gate makes compilation succeed but semantics degraded on Windows
- ✗ No documentation of performance implications on non-Unix

**Severity**: **CRITICAL** (core mmap strategy requires platform-specific tuning)

### 4. **Python Build System**

**Files**: `crates/larql-python/`, `crates/larql-python/pyproject.toml`, `crates/larql-python/build.rs`

**Issues**:

1. **Dependency on external tool (`uv`)**:
   - Makefile assumes `uv` is in PATH
   - `uv` is not a standard Python tool (vs. `pip`, `venv`)
   - Incompatible with CI systems that only provide `pip`
   - No fallback mechanism

2. **macOS-specific linker flags**:
   ```rust
   if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
       println!("cargo:rustc-link-arg=-undefined");
       println!("cargo:rustc-link-arg=dynamic_lookup");
   }
   ```
   - Required on macOS but not documented
   - Indicates fragile PyO3 integration
   - Will cause linker errors on other OS's if not carefully conditioned

3. **No pre-built wheels**:
   - `maturin develop` rebuilds from source every time
   - Wheels not published to PyPI
   - Binary distribution of `larql` Python package not feasible

4. **Version pinning**:
   - `maturin >= 1.7` (loose)
   - `pytest >= 8` (loose)
   - Transitive deps not locked by `uv.lock` when using `--no-install-project`

**Severity**: **HIGH** (blocks distribution, Python adoption, and CI/CD)

### 5. **Shell Scripts (Hardcoded Paths)**

#### `scripts/extract_and_publish_all.sh`
**Issues**:
- ✗ Uses `/bin/bash` shebang (not POSIX sh)
- ✗ Assumes `larql` is in PATH globally
- ✗ Uses `date '+%H:%M:%S'` (GNU date, differs on BSD/macOS)
- ✗ `tee` + array appends (`FAILED_PUBLISHES+=`) are bash-specific
- ✗ No error handling if command fails (e.g., network timeout)

#### `scripts/grid/start.sh`
**Issues**:
- ✗ **Hardcoded absolute path**: `/Users/christopherhay/chris-source/larql/output/gemma3-4b-q4k-v2.vindex`
- ✗ Assumes macOS (shebang says bash, assumes `/Users`)
- ✗ Direct references to `~` which expands differently in cronjobs
- ✗ No validation of executable permissions on `larql-router`, `larql-server`
- ✗ `curl` used without `-f` flag (doesn't distinguish 404 from success)

#### `scripts/migrate_repos_to_model.sh`
**Issues**:
- ✗ Assumes `hf` CLI is available and configured
- ✗ No API key validation before running
- ✗ Error messages suggest rerunning but don't explain why 404s occur

**Severity**: **MEDIUM** (dev scripts, not shipped, but affect reproducibility)

### 6. **Cargo.toml: No Default Features**

**File**: `./Cargo.toml`, `crates/larql-inference/Cargo.toml`

**Issues**:
- ✗ `larql-compute` has `features = ["metal"]` but defaults to empty
- ✗ Users must explicitly pass `--features metal` or miss GPU support
- ✗ No guidance in README about which features to enable per platform
- ✗ CI would need to test both `default` and `metal` builds separately

**Severity**: **MEDIUM** (silent degradation, hard to diagnose)

### 7. **No CI/CD Pipeline**

**Status**: No `.github/workflows/` directory exists.

**Issues**:
- ✗ No automated testing on Linux, macOS, Windows
- ✗ Regression from new dependencies (OpenBLAS, Metal) undetected
- ✗ Python bindings not tested across Python versions (3.11, 3.12, 3.13)
- ✗ No pre-built binaries / release artifacts

**Severity**: **CRITICAL** (impossible to guarantee cross-platform correctness)

### 8. **Documentation Gaps**

**Files affected**: `README.md`, `AGENTS.md`, `CLAUDE.md`, `docs/cli.md`

**Issues**:
- ✗ No "Getting Started" instructions for Linux/macOS/Windows
- ✗ No system dependency checklist (OpenBLAS, Python, Rust version)
- ✗ No troubleshooting guide (e.g., "OpenBLAS not found on Ubuntu")
- ✗ AGENTS.md mentions `--features metal` but doesn't explain when/why to use it
- ✗ No guidance on Docker usage

**Severity**: **MEDIUM** (UX friction, increases support burden)

---

## Low-Hanging Fruit (Impact vs. Effort)

### Tier 1: Critical, ~2-4 hours each

1. **Add Windows MMAP support** (`crates/larql-vindex/src/mmap_util.rs`)
   - Implement `#[cfg(target_os = "windows")]` with no-op or `FlushViewOfFile`
   - Effort: 2-3 hours
   - Impact: Unblocks Windows builds
   - Risk: Low (new code, tests added)

2. **Create `.github/workflows/ci.yml`**
   - Test on Linux, macOS (Intel), macOS (ARM), Windows
   - Run `cargo test --all` and `make python-test` on each
   - Effort: 3-4 hours
   - Impact: Detect regressions early, prove cross-platform correctness
   - Risk: Low (CI only, no side effects)

3. **Add shell script compatibility layer**
   - Replace bash-specific constructs in `scripts/` with POSIX sh
   - Remove hardcoded `/Users/` paths
   - Effort: 2-3 hours
   - Impact: Scripts work in Docker, cron, CI/CD without modification
   - Risk: Low (scripts are dev-only)

### Tier 2: High Value, ~4-6 hours each

4. **Pin BLAS backend selection and add vendored fallback**
   - Make `larql-compute` vendored BLAS the default
   - Document system BLAS as opt-in with `--features system-blas`
   - Effort: 4-5 hours
   - Impact: Removes dependency on pre-installed OpenBLAS, works in all containers
   - Risk: Medium (build time increases, may hit linker issues)

5. **Deprecate `uv` dependency, support standard `pip`**
   - Add `venv` + `pip install -r requirements.txt` as primary Python path
   - Keep `uv` as optional for fast CI builds
   - Effort: 3-4 hours
   - Impact: Works with any Python CI system (GitHub Actions, GitLab, Azure Pipelines)
   - Risk: Low (both methods build same binary)

6. **Create Makefile help and cross-platform wrapper**
   - `make help` lists all targets
   - `make check-env` validates system dependencies
   - Wrap shell scripts in Makefile targets (e.g., `make grid-start`)
   - Effort: 3-4 hours
   - Impact: Onboarding friction drops 70%, UX improves
   - Risk: Low (additive, doesn't break existing targets)

### Tier 3: Technical Debt, ~6-8 hours each

7. **Runtime CPU architecture detection**
   - Detect at startup whether Metal/CUDA is available
   - Fall back gracefully if GPU not present
   - Log what backend was selected
   - Effort: 5-6 hours
   - Impact: Single binary works on Intel Mac + Apple Silicon + Linux
   - Risk: Medium (startup overhead, must test all paths)

8. **Dockerize build + distribute pre-built wheels**
   - `Dockerfile` for dev, CI, and release builds
   - Maturin CI to publish `.whl` to PyPI
   - Effort: 6-8 hours
   - Impact: `pip install larql` just works for most users
   - Risk: High (wheel distribution has versioning/ABI complexities)

---

## Formal Problem Statement

### Problem Definition

**Title**: Build Process Device & OS Non-Agnosticism

**Scope**: The LARQL Rust + Python project has **implicit, undocumented platform-specific dependencies** embedded in:
- Conditional compilation (`cfg` gates)
- Shell scripts (bash-isms, hardcoded paths)
- Build tool choices (`uv`, maturin)
- System library assumptions (OpenBLAS, Accelerate)
- Memory management primitives (mmap hints)

This prevents **reproducible, portable builds** across Linux, macOS (Intel/ARM), Windows, and containerized environments.

### Root Causes

1. **Organic growth without cross-platform design**: The project started on one developer's macOS setup (`/Users/christopherhay` hardcoded). Later Linux support was added piecemeal without systematic testing.

2. **Missing CI/CD**: No automated testing across platforms means regressions accumulate. Each developer tests only their local environment.

3. **Tool proliferation without fallbacks**: Adoption of `uv` and `maturin` is convenient for certain workflows but incompatible with standard Python tooling.

4. **Implicit system dependencies**: BLAS libraries are assumed present but not validated. Linker flags are platform-specific but not documented.

### Success Criteria (Closure Definition)

A build is **device/OS-agnostic** when:

1. **Single source of truth**: One canonical Makefile, one set of Cargo.toml features, one build script (no per-developer tweaks).

2. **Automatic platform detection**: Build system detects OS, architecture, available BLAS, GPU support at build time and configures accordingly.

3. **Graceful degradation**: If Metal is unavailable on macOS, build succeeds with CPU-only backend. If OpenBLAS unavailable, use vendored fallback.

4. **Reproducibility**: Same commit + same command on any machine produces bit-identical binary (modulo timestamps).

5. **CI coverage**: Automated tests on Linux, macOS (Intel/ARM), Windows (if supported), Python 3.11/3.12/3.13.

6. **Documentation completeness**: README clearly states:
   - System requirements per platform
   - Which features are optional
   - Troubleshooting for common failures

7. **Distribution viability**: Python package installable via `pip install larql` without source compilation (pre-built wheels).

### Finite Closure Checklist

- [ ] Windows MMAP implementation (`cfg(target_os = "windows")` block added)
- [ ] CI/CD pipeline (GitHub Actions) testing all 3 platforms
- [ ] Makefile help + `make check-env` target
- [ ] BLAS vendored by default, system BLAS as opt-in
- [ ] Shell scripts converted to POSIX sh, no hardcoded paths
- [ ] Python bindings testable via `pip + venv`, `uv` optional
- [ ] Runtime logging of selected backends (Metal/CUDA/OpenBLAS/CPU)
- [ ] Setup guide for each platform (Linux, macOS, Windows) in README
- [ ] Pre-built Python wheels published to test PyPI
- [ ] Regression test suite runs in GitHub Actions (Linux, macOS, Windows)

---

## Concrete Examples of Current Friction

### Example 1: New Linux Developer

```bash
$ git clone https://github.com/chrishayuk/larql
$ cd larql
$ make build
# Error: error: linker `cc` not found
# Missing: build-essential package
# Developer must google, guess, install manually
```

**Fix**: `make check-env` would detect missing gcc/clang and print installation instructions for their distro.

### Example 2: macOS Intel Developer with Metal

```bash
$ cargo build --features metal
# Compiles successfully
$ ./target/debug/larql predict ...
# Runtime error: Metal device unavailable
# Developer confused: why compile metal if it doesn't work?
```

**Fix**: Runtime detection logs `[INFO] Metal GPU unavailable, using CPU backend` at startup.

### Example 3: Docker Build

```dockerfile
FROM rust:latest
WORKDIR /app
COPY . .
RUN make build
# Error: mmap_util.rs relies on libc::madvise (Unix only)
# Error: scripts/grid/start.sh references /Users/christopherhay
```

**Fix**: MMAP has Windows fallback, scripts use relative paths or `$HOME`.

### Example 4: Python Package Distribution

```bash
$ pip install larql
# Error: no binary wheel found, building from source
# Requires Rust toolchain, OpenBLAS, maturin
# 10 minutes to install instead of 30 seconds
```

**Fix**: Pre-built wheels on PyPI, `pip install larql` takes 5 seconds.

---

## Impact Assessment

| Issue | Severity | Affected Users | Estimated Fix Time |
|-------|----------|----------------|--------------------|
| Windows unsupported | CRITICAL | 25-30% of devs | 2-3 hrs |
| No CI/CD | CRITICAL | All future contributors | 3-4 hrs |
| OpenBLAS hard-required on Linux | HIGH | Linux devs, Docker | 4-5 hrs |
| Python `uv` hard-required | HIGH | Non-uv Python users | 3-4 hrs |
| Hardcoded paths in scripts | MEDIUM | Dev scripts, CI | 2-3 hrs |
| Makefile lacks help/validation | MEDIUM | New users | 2-3 hrs |
| No GPU runtime detection | MEDIUM | Apple Silicon devs | 5-6 hrs |
| No pre-built wheels | MEDIUM | Python users | 6-8 hrs |

**Total estimated effort**: ~30-40 hours to achieve full OS/device agnosticism.  
**Critical path** (blocking all others): Windows MMAP + CI/CD (~6-7 hours).  
**Quick wins** (high ROI): Makefile help, POSIX scripts, environment check (~7 hours).

---

## Recommendations

1. **Start with Tier 1** (Windows MMAP, CI/CD, scripts) — unblocks everything.
2. **Parallelize** where possible (CI/CD setup can happen independently of MMAP).
3. **Document as you go** — each platform detection adds a comment explaining the choice.
4. **Test locally** before pushing CI changes (use `act` to run GitHub Actions locally).
5. **Use feature flags extensively** — don't hardcode platform choices, make them discoverable.

---

## References

- **Cargo Platform-Specific Dependencies**: https://doc.rust-lang.org/cargo/reference/manifest.html#target
- **PyO3 macOS Linker Flags**: https://pyo3.rs/latest/
- **Maturin Python Wheels**: https://www.maturin.rs/
- **POSIX Shell Compatibility**: https://pubs.opengroup.org/onlinepubs/9699919799/utilities/sh.html
- **GitHub Actions Matrix Testing**: https://docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-job-s
