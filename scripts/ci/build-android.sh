#!/usr/bin/env bash
# SPDX-FileCopyrightText: Contributors to the larql-to-sparql project
# SPDX-License-Identifier: Apache-2.0
#
# Android platform build and cross-compilation script (Phase 2b).
#
# This script cross-compiles the larql workspace for Android targets using the
# Android NDK. It validates that the project compiles for:
#   - aarch64-linux-android (primary 64-bit ARM target)
#   - armv7-linux-androideabi (secondary 32-bit ARM target)
#
# Note: Python bindings (larql-python) are skipped on Android. Compilation only;
# no runtime testing via emulator (test surface limited on Android).
#
# Assumptions:
#   - Rust is installed and on PATH
#   - Android NDK can be auto-detected or is provided via ANDROID_NDK_HOME
#   - rustup is available for adding Android targets
#
# Usage:
#   ./scripts/ci/build-android.sh
#
# Environment variables:
#   ANDROID_NDK_HOME: Path to Android NDK (auto-detected if not set)
#   ANDROID_NDK_VERSION: NDK version to download (default: 27.0.11902837)
#   RUST_TOOLCHAIN: Rust version to use (auto-detected from CI env, optional locally)
#   VERBOSE: Enable verbose output (set to 1)

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Color output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Pinned Rust toolchain (matches .github/workflows/validate.yml and quality.yml)
readonly RUST_TOOLCHAIN="${RUST_TOOLCHAIN:-1.92.0}"

# Android NDK configuration
readonly ANDROID_NDK_VERSION="${ANDROID_NDK_VERSION:-27.0.11902837}"
readonly ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-}"

# Android targets to build
declare -a ANDROID_TARGETS=(
  "aarch64-linux-android"
  "armv7-linux-androideabi"
)

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

# Detect or download Android NDK
setup_android_ndk() {
  log_header "Setting up Android NDK"

  if [[ -n "${ANDROID_NDK_HOME}" ]] && [[ -d "${ANDROID_NDK_HOME}" ]]; then
    log_success "Android NDK found at: ${ANDROID_NDK_HOME}"
    return 0
  fi

  # Try to detect NDK in common locations
  if [[ -d "${HOME}/Android/sdk/ndk/${ANDROID_NDK_VERSION}" ]]; then
    export ANDROID_NDK_HOME="${HOME}/Android/sdk/ndk/${ANDROID_NDK_VERSION}"
    log_success "Android NDK auto-detected at: ${ANDROID_NDK_HOME}"
    return 0
  fi

  # On CI systems, NDK might be pre-installed
  if [[ -d "/opt/android/ndk/${ANDROID_NDK_VERSION}" ]]; then
    export ANDROID_NDK_HOME="/opt/android/ndk/${ANDROID_NDK_VERSION}"
    log_success "Android NDK found at: ${ANDROID_NDK_HOME}"
    return 0
  fi

  # If not found and not in CI, provide helpful message
  log_error "Android NDK not found"
  echo ""
  echo "To set up Android NDK for local development:"
  echo ""
  echo "  Option 1: Use Android Studio"
  echo "    - Install via Android Studio's SDK Manager (Tools → SDK Manager → NDK)"
  echo "    - Set: export ANDROID_NDK_HOME=\$HOME/Android/sdk/ndk/${ANDROID_NDK_VERSION}"
  echo ""
  echo "  Option 2: Download manually"
  echo "    - Download NDK ${ANDROID_NDK_VERSION} from:"
  echo "      https://developer.android.com/ndk/downloads"
  echo "    - Extract and set: export ANDROID_NDK_HOME=/path/to/ndk/${ANDROID_NDK_VERSION}"
  echo ""
  echo "  Option 3: GitHub Actions (CI) environment"
  echo "    - NDK is pre-installed on ubuntu-24.04 runners"
  echo "    - Detected automatically via /opt/android/ndk path"
  echo ""
  exit 1
}

# Install Rust Android targets
install_android_targets() {
  log_header "Installing Rust Android targets"

  for target in "${ANDROID_TARGETS[@]}"; do
    log_step "Adding target: ${target}"
    rustup target add "${target}"
  done

  log_success "All Android targets installed"
}

# Build for Android targets
build_android_targets() {
  log_header "Building for Android targets"

  cd "${REPO_ROOT}"

  local build_success=true
  for target in "${ANDROID_TARGETS[@]}"; do
    log_step "Building for ${target}"

    if cargo build --release --workspace --target "${target}" --no-fail-fast; then
      log_success "Build successful for ${target}"
    else
      log_error "Build failed for ${target}"
      build_success=false
    fi
  done

  if [[ "${build_success}" == "false" ]]; then
    log_error "One or more Android targets failed to build"
    exit 1
  fi

  log_success "All Android targets built successfully"
}

# Main
main() {
  if [[ "${VERBOSE:-0}" == "1" ]]; then
    set -x
  fi

  log_header "Android Platform Build & Cross-Compile (Phase 2b)"
  echo "Targets: aarch64-linux-android (primary), armv7-linux-androideabi (secondary)"
  echo "Note: Python bindings skipped on Android (no runtime test surface)"
  echo ""

  check_prerequisites
  setup_rust_toolchain
  setup_android_ndk
  install_android_targets
  build_android_targets

  log_header "All checks passed ✓"
  echo -e "${GREEN}Android build and cross-compilation suite completed successfully.${NC}"
}

main "$@"
