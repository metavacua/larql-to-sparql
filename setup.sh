#!/usr/bin/env bash
# SPDX-FileCopyrightText: Contributors to the larql-to-sparql project
# SPDX-License-Identifier: Apache-2.0
#
# setup.sh: Install all system and toolchain dependencies for the LARQL project.
#
# This script sets up a fresh development environment for building and testing
# the LARQL Rust + Python (maturin/PyO3) workspace. It documents every implicit
# dependency and ensures reproducible builds on Ubuntu and macOS.
#
# Usage:
#   ./setup.sh              # Auto-detect OS and install
#   ./setup.sh --help       # Show this message
#   ./setup.sh --verify     # Check if dependencies are installed (no changes)

set -euo pipefail

# Color output for readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

# Detect Linux distribution
detect_linux_distro() {
    if command -v lsb_release &> /dev/null; then
        lsb_release -si | tr '[:upper:]' '[:lower:]'
    elif [[ -f /etc/os-release ]]; then
        awk -F'=' '/^ID=/ {gsub(/"/, ""); print tolower($2)}' /etc/os-release
    else
        echo "unknown"
    fi
}

# Ubuntu system dependencies
install_ubuntu_deps() {
    log_info "Installing Ubuntu system dependencies..."

    local deps=(
        # Build essentials
        "build-essential"
        "pkg-config"
        "curl"
        "git"

        # OpenSSL (required by many Rust crates)
        "libssl-dev"

        # Python development
        "python3"
        "python3-dev"
        "python3-pip"

        # For cross-compilation (Android)
        "clang"
        "llvm"
    )

    log_info "Updating package manager..."
    sudo apt-get update

    local missing_deps=()
    for dep in "${deps[@]}"; do
        if ! dpkg -l "$dep" 2>/dev/null | grep -q "^ii"; then
            missing_deps+=("$dep")
        fi
    done

    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_info "Installing missing packages: ${missing_deps[*]}"
        sudo apt-get install -y "${missing_deps[@]}"
        log_success "System dependencies installed"
    else
        log_success "All system dependencies already installed"
    fi
}

# macOS system dependencies
install_macos_deps() {
    log_info "Installing macOS system dependencies..."

    log_info "Checking Xcode Command Line Tools..."
    if ! xcode-select --print-path &> /dev/null; then
        log_warn "Xcode Command Line Tools not found. Starting installer..."
        xcode-select --install
        log_error "Please complete the Xcode Command Line Tools installation in the popup window, then re-run this script."
        return 1
    fi

    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        log_error "Homebrew not found. Please install it first:"
        echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        return 1
    fi

    local deps=(
        # Build essentials
        "pkg-config"

        # OpenSSL
        "openssl"

        # Python
        "python@3.12"
    )

    local missing_deps=()
    for dep in "${deps[@]}"; do
        if ! brew list "$dep" &> /dev/null; then
            missing_deps+=("$dep")
        fi
    done

    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_info "Installing missing Homebrew packages: ${missing_deps[*]}"
        brew install "${missing_deps[@]}"
        log_success "Homebrew dependencies installed"
    else
        log_success "All Homebrew dependencies already installed"
    fi
}

# Install Rust (if not already installed)
install_rust() {
    if command -v rustup &> /dev/null; then
        log_info "Rust/rustup already installed, updating..."
        rustup self update
        rustup update
    else
        log_info "Installing Rust via rustup..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        # shellcheck source=/dev/null
        source "$HOME/.cargo/env"
    fi

    log_success "Rust installed: $(rustc --version)"
}

# Install uv (Python package manager)
install_uv() {
    if command -v uv &> /dev/null; then
        log_success "uv already installed: $(uv --version)"
        return 0
    fi

    log_info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Update PATH if needed
    export PATH="$HOME/.local/bin:$PATH"

    log_success "uv installed: $(uv --version)"
}

# Verify all dependencies
verify_dependencies() {
    log_info "Verifying all dependencies..."

    local missing=0

    # Check Rust
    if command -v rustc &> /dev/null; then
        log_success "Rust: $(rustc --version)"
    else
        log_error "Rust/rustc not found"
        missing=$((missing + 1))
    fi

    if command -v cargo &> /dev/null; then
        log_success "Cargo: $(cargo --version)"
    else
        log_error "Cargo not found"
        missing=$((missing + 1))
    fi

    # Check Python
    if command -v python3 &> /dev/null; then
        log_success "Python: $(python3 --version)"
    else
        log_error "Python3 not found"
        missing=$((missing + 1))
    fi

    # Check uv
    if command -v uv &> /dev/null; then
        log_success "uv: $(uv --version)"
    else
        log_error "uv not found"
        missing=$((missing + 1))
    fi

    # Check pkg-config (for system library detection)
    if command -v pkg-config &> /dev/null; then
        log_success "pkg-config: $(pkg-config --version)"
    else
        log_warn "pkg-config not found (some builds may fail)"
        missing=$((missing + 1))
    fi

    # Check OpenSSL
    if pkg-config --exists openssl 2>/dev/null || command -v openssl &> /dev/null; then
        log_success "OpenSSL found"
    else
        log_warn "OpenSSL not found (may cause build failures)"
        missing=$((missing + 1))
    fi

    if [[ $missing -eq 0 ]]; then
        log_success "All dependencies verified!"
        return 0
    else
        log_error "$missing dependency/dependencies missing"
        return 1
    fi
}

# Show help
show_help() {
    cat << EOF
Usage: ./setup.sh [OPTIONS]

Setup script to install all dependencies for the LARQL project.

OPTIONS:
    --help      Show this help message
    --verify    Verify dependencies without installing
    --ubuntu    Force Ubuntu setup (for WSL/Linux Subsystem)
    --macos     Force macOS setup

DEPENDENCIES INSTALLED:

Ubuntu/Debian:
  - build-essential, pkg-config, curl, git
  - libssl-dev (OpenSSL)
  - python3, python3-dev, python3-pip
  - clang, llvm (for Android cross-compilation)
  - Rust via rustup
  - uv (Python package manager)

macOS:
  - Xcode Command Line Tools
  - Homebrew packages: openssl, python@3.12
  - Rust via rustup
  - uv (Python package manager)

DOCUMENTATION:

For more information on system requirements, see:
  README.md :: System Requirements

For CI/CD pipeline details, see:
  .github/workflows/
  docs/specs/compliance-pipeline.md (if present)

EOF
}

# Main
main() {
    local mode="setup"
    local os_override=""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help)
                show_help
                exit 0
                ;;
            --verify)
                mode="verify"
                shift
                ;;
            --ubuntu)
                os_override="linux"
                shift
                ;;
            --macos)
                os_override="macos"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    log_info "LARQL Development Environment Setup"
    echo ""

    # Detect OS
    local os="${os_override:-$(detect_os)}"

    if [[ "$os" == "unknown" ]]; then
        log_error "Unable to detect operating system"
        exit 1
    fi

    log_info "Detected OS: $os"

    if [[ "$mode" == "verify" ]]; then
        verify_dependencies
        exit $?
    fi

    # Install dependencies based on OS
    case "$os" in
        linux)
            local distro=$(detect_linux_distro)
            log_info "Detected Linux distribution: $distro"

            if [[ "$distro" == "ubuntu" ]] || [[ "$distro" == "debian" ]]; then
                install_ubuntu_deps
            else
                log_error "This script is tested on Ubuntu/Debian"
                log_info "For $distro, install: build-essential pkg-config libssl-dev python3-dev clang"
                exit 1
            fi
            ;;
        macos)
            install_macos_deps
            ;;
        *)
            log_error "Unsupported OS: $os"
            exit 1
            ;;
    esac

    echo ""

    # Install Rust (both platforms)
    install_rust

    echo ""

    # Install uv (both platforms)
    install_uv

    echo ""

    # Verify everything
    if verify_dependencies; then
        log_success "Setup complete! You're ready to build LARQL."
        echo ""
        echo "Next steps:"
        echo "  cargo build --release"
        echo "  cargo test --workspace"
        echo ""
        echo "For Python bindings:"
        echo "  cd crates/larql-python"
        echo "  uv sync --group dev"
        echo "  uv run maturin develop --release"
        echo "  uv run pytest tests/ -v"
        return 0
    else
        log_error "Setup completed but some dependencies are missing. Please check above."
        return 1
    fi
}

main "$@"
