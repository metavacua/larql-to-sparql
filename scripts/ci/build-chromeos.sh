#!/usr/bin/env bash
# SPDX-FileCopyrightText: Contributors to the larql-to-sparql project
# SPDX-License-Identifier: Apache-2.0
#
# ChromeOS platform build and test script (Phase 2a).
#
# ChromeOS runs a Linux container (Crostini) targeting x86_64-unknown-linux-gnu.
# The build process is nearly identical to Ubuntu since both target the same
# Linux environment. This script validates that the project builds and tests
# successfully on ChromeOS.
#
# Assumptions:
#   - Rust is installed and on PATH
#   - Python 3.12+ is available (for Python bindings test)
#   - uv package manager is installed
#   - Running on ChromeOS with Linux container (Crostini) enabled
#
# Usage:
#   ./scripts/ci/build-chromeos.sh
#
# Environment variables:
#   VERBOSE: Enable verbose output (set to 1)
#   RUST_TOOLCHAIN: Rust version to use (auto-detected from CI env, optional locally)

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
build() {
  log_header "Building project (cargo build --release)"

  cd "${REPO_ROOT}"

  if cargo build --release --workspace; then
    log_success "Build successful"
  else
    log_error "Build failed"
    exit 1
  fi
}

# Run tests
run_tests() {
  log_header "Running workspace tests (cargo test --workspace)"

  cd "${REPO_ROOT}"

  if cargo test --workspace --no-fail-fast; then
    log_success "Tests passed"
  else
    log_error "Tests failed"
    exit 1
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
    exit 1
  fi

  log_step "Building Python extension (maturin develop --release)"
  if uv run --no-sync maturin develop --release; then
    log_success "Python extension built"
  else
    log_error "Failed to build Python extension"
    exit 1
  fi

  log_step "Running pytest"
  if uv run --no-sync pytest tests/ -v; then
    log_success "Python tests passed"
  else
    log_error "Python tests failed"
    exit 1
  fi

  cd "${REPO_ROOT}"
}

# Main
main() {
  if [[ "${VERBOSE:-0}" == "1" ]]; then
    set -x
  fi

  log_header "ChromeOS Platform Build & Test (Phase 2a)"
  echo "Target: x86_64-unknown-linux-gnu (Crostini Linux container)"
  echo ""

  check_prerequisites
  setup_rust_toolchain
  build
  run_tests
  test_python_bindings

  log_header "All checks passed ✓"
  echo -e "${GREEN}ChromeOS build and test suite completed successfully.${NC}"
}

main "$@"
