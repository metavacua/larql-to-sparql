#!/usr/bin/env bash
# SPDX-FileCopyrightText: Contributors to the larql-to-sparql project
# SPDX-License-Identifier: Apache-2.0
#
# macOS platform build and test script (Phase 3).
#
# Handles both Intel (x86_64) and ARM (Apple Silicon) architectures.
# Architecture-specific behavior:
#   - ARM (Apple Silicon): Builds with and without Metal GPU support
#   - Intel (x86_64): Builds without Metal (Metal not available on Intel)
#
# Both architectures run full test suites and validate Python bindings.
#
# Assumptions:
#   - Rust is installed and on PATH (via rustup)
#   - Python 3.12+ is available
#   - uv package manager is installed
#   - Xcode Command Line Tools are installed
#   - Running on macOS (detected via 'uname -s')
#
# Usage:
#   ./scripts/ci/build-macos.sh
#
# Environment variables:
#   VERBOSE: Enable verbose output (set to 1)
#   RUST_TOOLCHAIN: Rust version to use (auto-detected from CI env, optional locally)
#   FORCE_NO_METAL: Force build without Metal even on ARM (for testing, optional)

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Color output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Pinned Rust toolchain (matches .github/workflows/cross-platform-build.yml)
readonly RUST_TOOLCHAIN="${RUST_TOOLCHAIN:-1.88.0}"

# Utility functions
log_header() {
  echo -e "${BLUE}========================================${NC}"
  echo -e "${BLUE}$1${NC}"
  echo -e "${BLUE}========================================${NC}"
}

log_step() {
  echo -e "${YELLOW}→ $1${NC}"
}

log_success() {
  echo -e "${GREEN}✓ $1${NC}"
}

log_error() {
  echo -e "${RED}✗ $1${NC}"
}

check_command() {
  if ! command -v "$1" &>/dev/null; then
    log_error "Required command not found: $1"
    exit 1
  fi
}

# Detect CPU architecture
detect_architecture() {
  local arch
  arch="$(uname -m)"

  case "${arch}" in
    arm64)
      echo "arm64"
      ;;
    x86_64)
      echo "intel"
      ;;
    *)
      log_error "Unsupported macOS architecture: ${arch}"
      exit 1
      ;;
  esac
}

# Check prerequisites
check_prerequisites() {
  log_header "Checking prerequisites"

  check_command cargo
  check_command rustup
  check_command python3
  check_command uv

  log_success "All prerequisites found"
}

# Install/verify Rust toolchain
setup_rust_toolchain() {
  log_header "Setting up Rust toolchain"

  log_step "Installing Rust ${RUST_TOOLCHAIN}"
  rustup toolchain install "${RUST_TOOLCHAIN}" \
    --profile minimal \
    --no-self-update

  rustup default "${RUST_TOOLCHAIN}"

  log_success "Rust toolchain ${RUST_TOOLCHAIN} ready"
  cargo --version
  rustc --version
}

# Build the project
# Arguments: $1 = architecture (arm64 or intel), $2 = feature flags (empty or --features metal)
build_for_arch() {
  local arch="$1"
  local features="$2"
  local build_flags="--release --workspace"

  if [[ -n "${features}" ]]; then
    build_flags="${build_flags} ${features}"
  fi

  log_header "Building for ${arch}${features:+ with $features}"

  cd "${REPO_ROOT}"

  if cargo build ${build_flags}; then
    log_success "Build successful for ${arch}${features:+ with $features}"
  else
    log_error "Build failed for ${arch}${features:+ with $features}"
    return 1
  fi
}

# Run tests
# Arguments: $1 = architecture (arm64 or intel), $2 = feature flags (empty or --features metal)
run_tests_for_arch() {
  local arch="$1"
  local features="$2"
  local test_flags="--workspace --no-fail-fast"

  if [[ -n "${features}" ]]; then
    test_flags="${test_flags} ${features}"
  fi

  log_header "Running tests for ${arch}${features:+ with $features}"

  cd "${REPO_ROOT}"

  if cargo test ${test_flags}; then
    log_success "Tests passed for ${arch}${features:+ with $features}"
  else
    log_error "Tests failed for ${arch}${features:+ with $features}"
    return 1
  fi
}

# Test Python bindings
test_python_bindings() {
  log_header "Testing Python bindings"

  cd "${REPO_ROOT}/crates/larql-python"

  log_step "Installing Python dependencies (uv sync)"
  if uv sync --no-install-project --group dev; then
    log_success "Dependencies synced"
  else
    log_error "Failed to sync dependencies"
    return 1
  fi

  log_step "Building Python extension (maturin develop --release)"
  if uv run --no-sync maturin develop --release; then
    log_success "Python extension built"
  else
    log_error "Failed to build Python extension"
    return 1
  fi

  log_step "Running pytest"
  if uv run --no-sync pytest tests/ -v; then
    log_success "Python tests passed"
  else
    log_error "Python tests failed"
    return 1
  fi

  cd "${REPO_ROOT}"
}

# Main
main() {
  if [[ "${VERBOSE:-0}" == "1" ]]; then
    set -x
  fi

  local arch
  arch="$(detect_architecture)"

  log_header "macOS Platform Build & Test (Phase 3)"
  echo "Architecture: ${arch}"
  echo ""

  check_prerequisites
  setup_rust_toolchain

  local build_success=true

  case "${arch}" in
    arm64)
      echo "Target: Apple Silicon (ARM64) with and without Metal GPU support"
      echo ""

      # Build without Metal (baseline for OS independence)
      if ! build_for_arch "arm64" ""; then
        build_success=false
      fi

      # Run tests without Metal
      if ! run_tests_for_arch "arm64" ""; then
        build_success=false
      fi

      # Build with Metal (ARM-specific GPU support)
      if ! build_for_arch "arm64" "--features metal"; then
        build_success=false
      fi

      # Run tests with Metal
      if ! run_tests_for_arch "arm64" "--features metal"; then
        build_success=false
      fi
      ;;

    intel)
      echo "Target: Intel (x86_64) without Metal"
      echo ""

      # Build without Metal (only supported configuration on Intel)
      if ! build_for_arch "intel" ""; then
        build_success=false
      fi

      # Run tests without Metal
      if ! run_tests_for_arch "intel" ""; then
        build_success=false
      fi
      ;;
  esac

  if [[ "${build_success}" == "false" ]]; then
    log_error "One or more build/test steps failed"
    exit 1
  fi

  # Python bindings test (both architectures)
  if ! test_python_bindings; then
    log_error "Python bindings test failed"
    exit 1
  fi

  log_header "All checks passed ✓"
  echo -e "${GREEN}macOS build and test suite completed successfully.${NC}"
}

main "$@"
