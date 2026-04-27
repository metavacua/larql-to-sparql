# Build Optimization Roadmap: Execution Plan

**Status**: Planning Phase  
**Branch**: `claude/build-process-optimization-MyCWS`  
**Last Updated**: 2026-04-27

---

## Overview

This document translates the **problem statement** from `BUILD_PROCESS_ANALYSIS.md` into concrete, sequenced work items with acceptance criteria, dependencies, and risk mitigation.

**Goal**: Ship a fully cross-platform build process that works on Linux, macOS, Windows, and in containers without special-casing per developer or CI system.

---

## Phase 1: Foundation (Weeks 1-2)

### Task 1.1: Windows MMAP Fallback Implementation

**File**: `crates/larql-vindex/src/mmap_util.rs`

**Changes**:
```rust
// Add Windows implementation alongside Unix
#[cfg(target_os = "windows")]
pub unsafe fn mmap_demand_paged(file: &std::fs::File) -> Result<memmap2::Mmap, std::io::Error> {
    // Windows: memmap2::Mmap has no MADV_RANDOM equivalent.
    // Strategy: Just map normally. Windows VM doesn't track RSS by mmap,
    // so prefaulting has no benefit. Pages fault on access as usual.
    memmap2::Mmap::map(file)
}

#[cfg(target_os = "windows")]
pub fn advise_sequential(mmap: &memmap2::Mmap) {
    // Windows: No MADV_SEQUENTIAL. Prefaulting isn't beneficial because
    // ReadFile() already initiates async readahead on sequential access.
    // This is a no-op, which is safe and correct.
}
```

**Acceptance Criteria**:
- [ ] Code compiles on Windows MSVC
- [ ] Unit test passes: `cargo test -p larql-vindex --lib mmap_util`
- [ ] No performance regression on Linux/macOS (madvise still called)
- [ ] Comment explains why Windows versions are no-ops
- [ ] Tested on Windows 10/11 MSVC build

**Risk**: LOW  
**Effort**: 2-3 hours  
**Blocker for**: Windows builds

---

### Task 1.2: Create GitHub Actions CI Workflow

**File**: `.github/workflows/ci.yml` (new)

**Content** (sketch):
```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-12, macos-14, windows-latest]
        rust: [stable, 1.80]  # MSRV
        include:
          - os: ubuntu-latest
            cargo_args: ""
          - os: macos-12
            cargo_args: ""
          - os: macos-14
            cargo_args: "--features metal"
          - os: windows-latest
            cargo_args: ""
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          toolchain: ${{ matrix.rust }}
      - run: cargo test --workspace ${{ matrix.cargo_args }}
      - run: cargo test --doc

  python:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.11", "3.12", "3.13"]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install maturin pytest numpy
      - run: cd crates/larql-python && pip install -e .
      - run: cd crates/larql-python && pytest tests/

  fmt:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt,clippy
      - run: cargo fmt --all -- --check
      - run: cargo clippy --workspace --tests -- -D warnings
```

**Acceptance Criteria**:
- [ ] CI runs on all 3 platforms successfully
- [ ] Matrix covers: Linux (OpenBLAS), macOS Intel (Accelerate), macOS ARM (Metal), Windows (default BLAS)
- [ ] Python tests run on 3.11, 3.12, 3.13
- [ ] Failure on any platform blocks merging
- [ ] All existing tests pass

**Risk**: LOW (CI-only, no side effects)  
**Effort**: 3-4 hours  
**Blocker for**: Everything (enables continuous validation)

---

### Task 1.3: Repair Shell Scripts (POSIX Compliance)

**Files**: 
- `scripts/extract_and_publish_all.sh`
- `scripts/grid/start.sh`
- `scripts/grid/stop.sh`
- `scripts/migrate_repos_to_model.sh`

**Changes**:

1. **Shebang**: Change `#!/usr/bin/env bash` → `#!/usr/bin/env sh`
2. **Remove bash-isms**:
   - Replace `+=` array append with temp file
   - Replace `${BASH_SOURCE[0]}` with `"$0"`
   - Replace `set -euo pipefail` → `set -eu` (POSIX only has `-e`, `-u`, `-x`)
   - Use `printf` instead of `echo -e`
3. **Remove hardcoded paths**:
   - Replace `/Users/christopherhay` with template `${LARQL_OUTPUT_DIR:-./output}`
   - Provide setup instructions for each platform in README
4. **Use POSIX date**:
   - Replace `date '+%H:%M:%S'` with something portable

**Example fix** (`scripts/grid/start.sh`):
```bash
#!/usr/bin/env sh
set -eu

# Allow override via env var
VINDEX="${LARQL_VINDEX:-${1:-.}}"
ROUTER_HTTP=9090
ROUTER_GRPC=50052
ROUTER_HOST=127.0.0.1
BIN_DIR="$(cd "$(dirname "$0")/../.." && pwd)/target/release"
PID_DIR="$(cd "$(dirname "$0")" && pwd)/.pids"

if [ ! -d "$VINDEX" ]; then
  echo "error: vindex not found at $VINDEX" >&2
  exit 1
fi
```

**Acceptance Criteria**:
- [ ] All scripts run under `sh` (not just `bash`)
- [ ] Scripts work in CI/Docker without modification
- [ ] No hardcoded `/Users/` or developer-specific paths
- [ ] README documents required env vars (e.g., `LARQL_VINDEX`, `LARQL_OUTPUT_DIR`)
- [ ] Scripts tested in both Linux and macOS

**Risk**: LOW (dev scripts only, no core impact)  
**Effort**: 2-3 hours  
**Blocker for**: Docker builds, CI/CD

---

## Phase 2: Validation & Ergonomics (Weeks 2-3)

### Task 2.1: Makefile Enhancement (`make help`, `make check-env`)

**File**: `Makefile` (modified)

**New targets**:

1. **`make help`**:
   ```makefile
   .PHONY: help
   help:
   	@echo "LARQL Build Targets:"
   	@echo ""
   	@echo "Build & Test:"
   	@echo "  make build       - Debug build (cargo build --workspace)"
   	@echo "  make release     - Release build (optimized)"
   	@echo "  make check       - Check compilation (no build)"
   	@echo "  make test        - Run all tests"
   	@echo "  make ci          - Full CI: fmt-check, lint, test"
   	@echo ""
   	@echo "Code Quality:"
   	@echo "  make fmt         - Auto-format all code"
   	@echo "  make fmt-check   - Check formatting (CI mode)"
   	@echo "  make lint        - Run clippy linter"
   	@echo ""
   	@echo "Python Bindings:"
   	@echo "  make python-setup   - Create .venv, install dev deps"
   	@echo "  make python-build   - Build PyO3 extension"
   	@echo "  make python-test    - Run Python tests"
   	@echo ""
   	@echo "Demos & Benchmarks:"
   	@echo "  make demos       - Run example scripts"
   	@echo "  make bench       - Run core benchmarks"
   	@echo ""
   	@echo "Environment:"
   	@echo "  make check-env   - Validate system dependencies"
   	@echo "  make clean       - Remove build artifacts"
   	@echo ""
   	@echo "Features:"
   	@echo "  METAL=1 make build      - Build with Metal GPU (macOS only)"
   	@echo ""
   ```

2. **`make check-env`**:
   ```makefile
   .PHONY: check-env
   check-env:
   	@echo "Checking build environment..."
   	@command -v cargo >/dev/null || (echo "✗ cargo not found"; exit 1)
   	@echo "✓ cargo: $$(cargo --version)"
   	@rustc --version
   	@echo ""
   	@echo "Optional dependencies:"
   	@if command -v python3 >/dev/null; then \
   		echo "✓ python3: $$(python3 --version)"; \
   	else \
   		echo "⚠ python3 not found (needed for Python bindings)"; \
   	fi
   	@if command -v uv >/dev/null; then \
   		echo "✓ uv: $$(uv --version)"; \
   	else \
   		echo "⚠ uv not found (will use pip instead)"; \
   	fi
   	@echo ""
   	@echo "Platform-specific checks:"
   	@if [ "$$(uname -s)" = "Linux" ]; then \
   		echo "Linux detected"; \
   		if pkg-config --exists blas; then \
   			echo "✓ OpenBLAS found: $$(pkg-config --modversion blas)"; \
   		else \
   			echo "⚠ OpenBLAS not found (will use vendored fallback)"; \
   		fi; \
   	elif [ "$$(uname -s)" = "Darwin" ]; then \
   		echo "macOS detected"; \
   		echo "✓ Accelerate framework available"; \
   		sysctl -n sysctl.proc_translated 2>/dev/null | grep -q 1 && \
		echo "  (Running via Rosetta translation)" || echo "  (Native CPU arch)"; \
	fi
   ```

3. **Optional `METAL` flag**:
   ```makefile
   METAL ?= 0
   ifeq ($(METAL), 1)
   	CARGO_FEATURES += metal
   endif
   
   build:
   	cargo build --workspace $(if $(CARGO_FEATURES),--features $(CARGO_FEATURES))
   ```

**Acceptance Criteria**:
- [ ] `make help` displays all targets with descriptions
- [ ] `make check-env` validates: cargo, Rust version, Python (optional), OpenBLAS (optional)
- [ ] `make check-env` produces actionable output (✓ or ⚠)
- [ ] `METAL=1 make build` compiles with Metal on macOS
- [ ] Tested on Linux, macOS, Windows

**Risk**: LOW (additive, doesn't break existing targets)  
**Effort**: 2-3 hours

---

### Task 2.2: Setup Documentation (Platform-Specific Guides)

**Files**: `README.md`, `docs/SETUP.md` (new)

**Content** (`docs/SETUP.md`):

#### Linux (Ubuntu/Debian)
```markdown
## Linux Setup

### System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  pkg-config \
  libopenblas-dev \
  python3.11 python3.11-dev \
  python3.11-venv
```

### Build

```bash
make check-env   # Verify dependencies
make build       # Compile
make test        # Run tests
```

If OpenBLAS is unavailable, the build will automatically use vendored OpenBLAS (slower, but works).
```

#### macOS
```markdown
## macOS Setup

### System Dependencies

```bash
# Install Xcode Command Line Tools
xcode-select --install

# (optional) Homebrew for Python 3.11+
brew install python@3.11
```

Apple Silicon (M1/M2/M3):
```bash
make build --features metal   # GPU-accelerated inference
```

Intel Mac:
```bash
make build   # CPU only (Metal not available)
```

### Build

```bash
make check-env   # Verify Xcode tools
make build       # Compile
make test        # Run tests
```
```

#### Windows
```markdown
## Windows Setup

### System Dependencies

Install **Visual Studio Build Tools** (MSVC toolchain):
- Download from https://visualstudio.microsoft.com/downloads/
- Select "C++ build tools"
- or: Install full Visual Studio 2022

Install **Rust** (if not already installed):
```bash
rustup-init.exe
```

Install **Python 3.11+** (optional, for Python bindings):
- https://www.python.org/downloads/

### Build

```bash
cargo build --release
cargo test
```

Note: Python bindings on Windows require PyO3 + maturin. See `crates/larql-python/README.md`.
```

**Acceptance Criteria**:
- [ ] Setup guide for each platform (Linux, macOS, Windows)
- [ ] Commands are copy-paste ready
- [ ] Covers both standard and optional dependencies
- [ ] Includes troubleshooting for common errors
- [ ] README links to `docs/SETUP.md`

**Risk**: LOW (documentation only)  
**Effort**: 2-3 hours

---

## Phase 3: Dependency Management (Weeks 3-4)

### Task 3.1: BLAS Vendoring Strategy

**Files**: `crates/larql-compute/Cargo.toml`, `crates/larql-inference/Cargo.toml`

**Goal**: Make vendored BLAS the default. System BLAS becomes opt-in.

**Current**:
```toml
[target.'cfg(target_os = "linux")'.dependencies]
blas-src = { version = "0.10", features = ["openblas"] }
openblas-src = { version = "0.10", features = ["system"] }
```

**Desired**:
```toml
# Default: vendored
[target.'cfg(target_os = "linux")'.dependencies]
blas-src = { version = "0.10", features = ["openblas"] }
openblas-src = { version = "0.10" }  # no "system" feature

# Opt-in: system BLAS
[features]
system-blas = ["dep:system-blas"]

[target.'cfg(target_os = "linux")'.dependencies.system-blas]
optional = true
version = "0.10"
features = ["openblas", "system"]
```

**Acceptance Criteria**:
- [ ] `cargo build` uses vendored BLAS by default (compiles anywhere)
- [ ] `cargo build --features system-blas` uses system OpenBLAS (if available)
- [ ] CI tests both `default` and `system-blas` builds on Linux
- [ ] Build time documented in CI logs
- [ ] Regression test: ensure vendored and system BLAS give same results (numerical)

**Risk**: MEDIUM (build time increases, may hit compile failures)  
**Effort**: 4-5 hours

---

### Task 3.2: Python Tooling Flexibility

**Files**: `crates/larql-python/`, `Makefile`, `docs/SETUP.md`

**Goal**: Support both `uv` and standard `pip/venv`, with `uv` as optional fast-path.

**Changes**:

1. **Fallback in Makefile**:
   ```makefile
   .PHONY: python-setup
   python-setup:
   	if command -v uv >/dev/null 2>&1; then \
   		cd crates/larql-python && uv sync --no-install-project --group dev; \
   	else \
   		cd crates/larql-python && python3 -m venv .venv; \
   		. .venv/bin/activate; \
   		pip install -U pip setuptools maturin pytest numpy; \
   	fi
   
   .PHONY: python-build
   python-build: python-setup
   	cd crates/larql-python && \
   	if command -v uv >/dev/null 2>&1; then \
   		uv run --no-sync maturin develop --release; \
   	else \
   		. .venv/bin/activate; \
   		maturin develop --release; \
   	fi
   ```

2. **Document both paths in README**.

3. **CI**: Test both uv and pip paths separately.

**Acceptance Criteria**:
- [ ] `make python-build` works with `uv` (if installed)
- [ ] `make python-build` works with `pip/venv` (if uv not available)
- [ ] CI tests both paths
- [ ] `uv.lock` is optional (nice-to-have, not required)

**Risk**: LOW (both paths produce same binary)  
**Effort**: 2-3 hours

---

## Phase 4: Runtime Intelligence (Weeks 4-5)

### Task 4.1: Runtime GPU Backend Detection & Logging

**Files**: 
- `crates/larql-models/src/backend.rs` (new module)
- `crates/larql-cli/src/main.rs` (integrate logging)
- `crates/larql-server/src/main.rs` (integrate logging)

**Sketch**:
```rust
// larql-models/src/backend.rs (new)
#[derive(Debug, Clone, Copy)]
pub enum ComputeBackend {
    MetalGPU,
    CUDAGPU,
    CPUWithBLAS,
    CPUFallback,
}

impl ComputeBackend {
    pub fn detect() -> Self {
        #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "metal"))]
        if is_metal_available() {
            return ComputeBackend::MetalGPU;
        }
        
        #[cfg(feature = "cuda")]
        if is_cuda_available() {
            return ComputeBackend::CUDAGPU;
        }
        
        if has_blas() {
            ComputeBackend::CPUWithBLAS
        } else {
            ComputeBackend::CPUFallback
        }
    }
    
    pub fn describe(&self) -> &'static str {
        match self {
            ComputeBackend::MetalGPU => "Metal GPU (Apple Silicon)",
            ComputeBackend::CUDAGPU => "NVIDIA CUDA",
            ComputeBackend::CPUWithBLAS => "CPU (BLAS-accelerated)",
            ComputeBackend::CPUFallback => "CPU (no BLAS)",
        }
    }
}

fn is_metal_available() -> bool {
    // Try to create a Metal device at startup
    // Return false if unavailable (e.g., running on Intel Mac)
    todo!()
}

fn has_blas() -> bool {
    // Simple check: try to call a dummy BLAS operation
    // or check for OpenBLAS library symbols
    todo!()
}
```

**Integration** (`larql-cli/src/main.rs`):
```rust
fn main() {
    let backend = larql_models::ComputeBackend::detect();
    eprintln!("[INFO] Using {} backend", backend.describe());
    // ... rest of main
}
```

**Acceptance Criteria**:
- [ ] `larql` CLI logs selected backend at startup
- [ ] Logs appear in stderr so they don't pollute JSON output
- [ ] Detection is fast (<100ms)
- [ ] Tested on: macOS Intel (should pick CPU), macOS ARM (should pick Metal if available), Linux (should pick BLAS)

**Risk**: MEDIUM (detection logic must be robust, tested on all platforms)  
**Effort**: 4-5 hours

---

## Phase 5: Distribution (Weeks 5-6)

### Task 5.1: Pre-Built Python Wheels (TestPyPI)

**Goal**: Publish maturin-built wheels to TestPyPI, then PyPI.

**Files**: `.github/workflows/publish.yml` (new)

**Workflow** (sketch):
```yaml
name: Publish

on:
  push:
    tags:
      - "v*"

jobs:
  build-wheels:
    strategy:
      matrix:
        include:
          - os: ubuntu-latest
            python-version: "3.11"
          - os: macos-latest
            python-version: "3.11"
          - os: windows-latest
            python-version: "3.11"
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: PyO3/maturin-action@v1
        with:
          python-version: ${{ matrix.python-version }}
          maturin-version: latest
          command: build
          args: --release --out dist -p larql-python
      - uses: actions/upload-artifact@v4
        with:
          name: wheels
          path: dist/*.whl

  publish:
    runs-on: ubuntu-latest
    needs: build-wheels
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: wheels
          path: dist
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          repository-url: https://test.pypi.org/legacy/
          password: ${{ secrets.TEST_PYPI_API_TOKEN }}
```

**Acceptance Criteria**:
- [ ] Wheels built for Linux, macOS (x86_64 + arm64), Windows
- [ ] Wheels published to TestPyPI on tag push
- [ ] `pip install larql --index-url https://test.pypi.org/simple/` installs pre-built wheel
- [ ] No source compilation required
- [ ] Test against Python 3.11, 3.12, 3.13

**Risk**: HIGH (wheel distribution, ABI compatibility, versioning)  
**Effort**: 6-8 hours (includes testing in different environments)

---

## Success Metrics

### By End of Phase 1 (Week 2)
- [ ] Windows MMAP code compiles and tests pass
- [ ] CI/CD pipeline runs on all platforms
- [ ] Shell scripts are POSIX-compliant

### By End of Phase 2 (Week 3)
- [ ] `make help` and `make check-env` working
- [ ] Setup guides for all platforms
- [ ] No hardcoded paths in codebase

### By End of Phase 3 (Week 4)
- [ ] BLAS vendored by default, system BLAS opt-in
- [ ] Python builds work with pip/venv (not just uv)
- [ ] CI tests multiple Python versions

### By End of Phase 4 (Week 5)
- [ ] Runtime backend detection logging implemented
- [ ] Verified on all platforms

### By End of Phase 5 (Week 6)
- [ ] Pre-built wheels published to TestPyPI
- [ ] `pip install larql` works for supported platforms

---

## Dependencies Between Tasks

```
1.1 Windows MMAP ────┐
1.2 CI/CD Pipeline ──┤─→ 2.1 Makefile Help ─→ 2.2 Setup Docs
1.3 POSIX Scripts ───┘    3.1 BLAS Vendor  ─→ 4.1 Backend Detection
                          3.2 Python Tools  → 5.1 Pre-Built Wheels
```

**Critical Path**: Tasks 1.2 and 1.1 (CI must run before optimization is validated)

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| CI config errors block all development | Use `act` locally before merging; keep CI changes small |
| BLAS vendoring increases build time 10x | Document in CI logs; consider ccache for local builds |
| Python wheels break ABI compatibility | Test wheels in fresh venvs; publish to TestPyPI first |
| Windows MMAP performance regression | Benchmark mmap operations on all platforms before/after |
| Backwards compatibility breaks | All changes are additive; document deprecations |

---

## Testing Plan

### Unit Tests
```bash
cargo test --workspace
cargo test -p larql-python
```

### Integration Tests
```bash
make ci              # Full CI suite
make check-env       # Environment validation
```

### Platform-Specific
- **Linux**: GitHub Actions (Ubuntu 22.04, 24.04)
- **macOS**: GitHub Actions (Intel, ARM64)
- **Windows**: GitHub Actions (Windows Server 2022)

### Manual Testing Checklist
- [ ] First-time setup on clean Linux VM
- [ ] First-time setup on clean macOS (Intel)
- [ ] First-time setup on clean macOS (ARM)
- [ ] Python bindings import and run on all platforms

---

## Rollout Strategy

1. **Phase 1 & 2** (weeks 1-3): Merge as feature branches, get review feedback
2. **Phase 3** (week 4): Merge dependency changes, monitor build times
3. **Phase 4** (week 5): Runtime detection merged before wheels
4. **Phase 5** (week 6): Wheel publishing, with tags triggering CI

**Publication**: Full `v0.2.0` release after all phases complete.

---

## Sign-Off

- **Owner**: [Your name]
- **Reviewers**: [TBD]
- **Target Completion**: 6 weeks from start
- **Iteration Frequency**: Weekly sync on progress
