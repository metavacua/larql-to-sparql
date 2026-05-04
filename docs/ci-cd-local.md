# Local CI/CD Testing Guide

This document explains how to run the project's CI/CD checks locally before pushing to GitHub, ensuring consistency between your local environment and the remote CI/CD pipeline.

## Quick Start

### Run the current platform's full test suite:

```bash
make platform-test
```

Or equivalently:

```bash
./scripts/ci/comprehensive.sh
```

### Run a specific platform's tests:

```bash
make platform-test-ubuntu      # Ubuntu (Phase 1)
make platform-test-chromeos    # ChromeOS (Phase 2a)
make platform-test-android     # Android (Phase 2b)
make platform-test-macos       # macOS (Phase 3)
```

## What's Tested

Each platform script runs:

1. **Rust toolchain setup** (1.88.0)
2. **Compilation** (`cargo build --release`)
3. **Tests** (`cargo test --workspace`)
4. **Python bindings** (Ubuntu and macOS only):
   - Dependency setup (`uv sync`)
   - Extension build (`maturin develop --release`)
   - Test execution (`pytest`)

## Platforms and Status

| Platform | Status | CI/CD Job | Local Script |
|----------|--------|-----------|--------------|
| Ubuntu 24.04 | ✓ Phase 1 | `.github/workflows/cross-platform-build.yml` → `build-ubuntu` | `./scripts/ci/build-ubuntu.sh` |
| ChromeOS 24.04 | ✓ Phase 2a | `.github/workflows/cross-platform-build.yml` → `build-chromeos` | `./scripts/ci/build-chromeos.sh` |
| Android (aarch64 + armv7) | ✓ Phase 2b | `.github/workflows/cross-platform-build.yml` → `build-android` | `./scripts/ci/build-android.sh` |
| macOS (Intel + ARM) | ✓ Phase 3 | `.github/workflows/cross-platform-build.yml` → `build-macos` | `./scripts/ci/build-macos.sh` |

## Prerequisites

### All Platforms

- **Rust 1.88.0+**: Install via [rustup](https://rustup.rs/)
  ```bash
  rustup toolchain install 1.88.0
  rustup default 1.88.0
  ```

- **Cargo** (comes with Rust)

### Ubuntu / ChromeOS (Crostini)

- **Python 3.12+**: Required for Python bindings test
  ```bash
  sudo apt-get install python3.12 python3.12-venv
  ```

- **uv**: Package manager for Python bindings
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- **Build tools**:
  ```bash
  sudo apt-get install build-essential
  ```

### macOS

- **Xcode Command Line Tools**:
  ```bash
  xcode-select --install
  ```

- **Python 3.12+**: Via Homebrew
  ```bash
  brew install python@3.12
  ```

- **uv**: 
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- **Metal SDK**: Bundled with Xcode (ARM/Apple Silicon only)
  - No additional installation needed; auto-detected by build script
  - Intel Macs: Metal not available; builds and tests run without Metal feature

- **Rust 1.88.0+**: Install via [rustup](https://rustup.rs/)

### Android

- **Android NDK 27.0.11902837+**: Required for cross-compilation
  - Option 1: Install via Android Studio (Tools → SDK Manager → NDK)
    ```bash
    export ANDROID_NDK_HOME=$HOME/Android/sdk/ndk/27.0.11902837
    ```
  - Option 2: Download manually from [Android NDK Downloads](https://developer.android.com/ndk/downloads)
    ```bash
    export ANDROID_NDK_HOME=/path/to/android-ndk-r27
    ```
  - Option 3: On GitHub Actions, NDK is pre-installed on ubuntu-24.04 runners

- **Rust Android targets**: Installed automatically via `rustup target add`
  - aarch64-linux-android (64-bit ARM)
  - armv7-linux-androideabi (32-bit ARM)

## Feature Flags by Platform

| Platform | Features | Notes |
|----------|----------|-------|
| Ubuntu | Default (no Metal) | Metal disabled; not available on Linux |
| ChromeOS | Default (no Metal) | Crostini Linux container; Metal not available |
| Android (aarch64 + armv7) | Default (no Metal) | Metal not available on Android; build-only |
| macOS (ARM) | `--features metal` | Metal GPU acceleration for Apple Silicon |
| macOS (Intel) | Default (no Metal) | Metal not available on Intel Macs |

## Examples

### Run Ubuntu tests locally

```bash
./scripts/ci/build-ubuntu.sh
```

**Output:**
```
=====================================
Ubuntu Platform Build & Test
=====================================
→ Installing Rust 1.88.0
✓ Rust toolchain 1.88.0 ready
→ Building project (cargo build --release)
✓ Build successful
→ Running workspace tests (cargo test --workspace)
✓ Tests passed
→ Testing Python bindings
→ Installing Python dependencies (uv sync)
✓ Dependencies synced
→ Building Python extension (maturin develop --release)
✓ Python extension built
→ Running pytest
✓ Python tests passed
=====================================
All checks passed ✓
=====================================
```

### Auto-detect platform and test

```bash
make platform-test
```

**On Linux:**
```
→ Running: ubuntu
✓ ubuntu build and test passed
```

**On macOS:**
```
→ Running: macos
✓ macos build and test passed
```

### Force a specific platform

```bash
PLATFORM=ubuntu ./scripts/ci/comprehensive.sh
```

## Troubleshooting

### Error: "Required command not found: cargo"

**Solution**: Install Rust via [rustup](https://rustup.rs/).

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### Error: "Required command not found: uv"

**Solution**: Install uv.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Error: "Required command not found: python3"

**Solution**: Install Python 3.12+.

- **Ubuntu**: `sudo apt-get install python3.12`
- **macOS**: `brew install python@3.12`

### Error: "maturin develop --release failed"

**Possible causes:**
1. Python version mismatch (need 3.12+)
2. Missing Rust toolchain component
3. C build tools not installed

**Solutions:**
- Verify Python version: `python3 --version`
- Reinstall toolchain: `rustup toolchain install 1.88.0 --profile minimal`
- Install build tools:
  - **Ubuntu**: `sudo apt-get install build-essential`
  - **macOS**: `xcode-select --install`

### Error: "Metal feature not available"

**Cause**: Trying to build with Metal on a non-ARM macOS or on Linux.

**Solution**: The scripts auto-detect architecture and feature flags. If you see this error on Intel macOS, the script should have disabled Metal automatically.

- On Intel macOS: Use `cargo build --release --no-default-features`
- On Linux: Metal is not supported; use default flags without Metal

### Error: "Android NDK not found"

**Cause**: Android NDK is not installed or ANDROID_NDK_HOME is not set.

**Solution**: Install Android NDK and set the environment variable.

1. **Via Android Studio** (recommended):
   - Open Android Studio → Tools → SDK Manager → SDK Tools
   - Check "NDK (Side by side)" and select version 27.0.11902837
   - Click "Apply" and wait for installation
   - Set: `export ANDROID_NDK_HOME=$HOME/Android/sdk/ndk/27.0.11902837`

2. **Manual download**:
   - Download NDK from [Android NDK Downloads](https://developer.android.com/ndk/downloads)
   - Extract: `unzip android-ndk-r27-*.zip`
   - Set: `export ANDROID_NDK_HOME=/path/to/android-ndk-r27`

3. **GitHub Actions** (CI):
   - NDK is pre-installed at `/opt/android/ndk/27.0.11902837`
   - Auto-detected by the build script

### Error: "could not compile for Android target"

**Possible causes:**
1. Rust Android targets not installed
2. Toolchain outdated
3. Platform-specific dependency issue

**Solutions:**
- Add Android targets: `rustup target add aarch64-linux-android armv7-linux-androideabi`
- Reinstall toolchain: `rustup toolchain install 1.88.0 --profile minimal`
- Check if your crate has Android-incompatible dependencies
- Run with verbose output: `VERBOSE=1 ./scripts/ci/build-android.sh`

### Error: "xcrun: error" on macOS

**Cause**: Xcode Command Line Tools not installed or updated.

**Solution**: 
```bash
xcode-select --install
# Or if already installed, reset the path:
sudo xcode-select --reset
```

### Error: "Could not find Metal library" on macOS

**Cause**: Trying to build with Metal feature on Intel Mac or without Xcode SDK.

**Solution**: The script auto-detects architecture:
- ARM (Apple Silicon): Metal is enabled by default and available
- Intel: Metal feature is not available; builds run without it
- If you see this error despite running the script, ensure Xcode is fully installed:
  ```bash
  xcode-select --install
  xcode-select --reset
  ```

### Error: "Tests failed"

1. **Check the error output** for the specific failing test
2. **Run a single test** for faster iteration:
   ```bash
   cargo test -p <crate> <test_name> -- --nocapture
   ```
3. **Run tests with features**: If testing a specific platform's features
   ```bash
   cargo test --features metal -p larql-inference
   ```

## CI/CD Workflow Reference

### GitHub Actions workflows

- **`.github/workflows/validate.yml`**: Deterministic checks (licenses, commits, changelog, SemVer)
- **`.github/workflows/quality.yml`**: Code scanning (clippy, tests, audit, deny, doc, examples, python, proto-lint)
- **`.github/workflows/cross-platform-build.yml`**: Cross-platform builds (Phase 1: Ubuntu; Phase 2+: Android/ChromeOS/macOS)

### Local Makefile targets

| Target | Equivalent CI Job | Purpose |
|--------|------------------|---------|
| `make ci` | `quality.yml` (partial) | Format check, lint, test |
| `make platform-test` | `cross-platform-build.yml` | Platform-specific build + test |
| `make platform-test-ubuntu` | `cross-platform-build.yml::build-ubuntu` | Ubuntu-only build + test |
| `make python-test` | `quality.yml::python` | Python bindings test |

## Next Steps

### Phase 1 (Ubuntu, complete)
- Run `make platform-test` regularly before pushing
- Scripts mirror GitHub Actions for consistency

### Phase 2a (ChromeOS, complete)
- ChromeOS build script implemented in `./scripts/ci/build-chromeos.sh`
- GitHub Actions job enabled in `.github/workflows/cross-platform-build.yml`
- Targets x86_64-unknown-linux-gnu (Crostini Linux container)
- Build process identical to Ubuntu since both target the same Linux environment

### Phase 2b (Android, complete)
- Android build script implemented in `./scripts/ci/build-android.sh`
- GitHub Actions job enabled in `.github/workflows/cross-platform-build.yml`
- Cross-compiles for dual targets: aarch64-linux-android and armv7-linux-androideabi
- Build-only validation; no runtime testing or Python bindings
- Android NDK auto-detected or can be installed separately

### Phase 3 (macOS, complete)
- macOS build script implemented in `./scripts/ci/build-macos.sh`
- GitHub Actions jobs enabled in `.github/workflows/cross-platform-build.yml` for ARM and Intel
- Architecture-specific builds:
  * macOS 13 (Intel): Builds and tests without Metal
  * macOS 15 (Apple Silicon): Builds and tests both with and without Metal
- Full test suite and Python bindings validation on both architectures

## Development Workflow

**Before committing:**

```bash
# 1. Run code quality checks (existing)
make ci

# 2. Run platform-specific tests (new)
make platform-test

# 3. If all pass, commit and push
git commit -am "..."
git push
```

**Before opening a pull request:**

```bash
# Ensure all checks pass locally
make ci
make platform-test

# Or run validate.yml locally (deterministic checks)
# (See docs/specs/compliance-pipeline.md for details)
```

## See Also

- [AGENTS.md](../AGENTS.md): Project architecture and workspace layout
- [Makefile](../Makefile): Available build targets
- [GitHub Workflows](./.github/workflows/): CI/CD definitions
- [docs/specs/compliance-pipeline.md](./specs/compliance-pipeline.md): Validation and release pipeline
