#!/usr/bin/env bash
# SPDX-FileCopyrightText: Contributors to the larql-to-sparql project
# SPDX-License-Identifier: Apache-2.0
#
# Development environment setup script (repository-agnostic).
#
# This script installs and configures CI/CD tools and development dependencies.
# It works standalone or within any Rust/Python project, autodetecting project
# structure to optionally configure project-specific environments.
#
# Usage:
#   ./scripts/setup-dev-env.sh                 # Full setup
#   ./scripts/setup-dev-env.sh --help          # Show help
#   ./scripts/setup-dev-env.sh --minimal       # Skip optional tools
#   ./scripts/setup-dev-env.sh --platform-only # Android NDK + cross-compile targets only
#   ./scripts/setup-dev-env.sh --no-project    # Skip project-specific setup

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

RUST_TOOLCHAIN="1.88.0"
REUSE_TOOL_VERSION="6.2.0"
COCOGITTO_VERSION="7.0.0"
GIT_CLIFF_VERSION="2.13.1"
PYTHON_VERSION="3.12"
NDK_VERSION="r27"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PROJECT_SETUP=true

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Flags
MINIMAL_MODE=false
PLATFORM_ONLY_MODE=false
VERBOSE=false

# =============================================================================
# Utility Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[!]${NC} $*"
}

log_error() {
    echo -e "${RED}[✗]${NC} $*"
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

check_cmd() {
    if command_exists "$1"; then
        log_success "$1 is installed"
        return 0
    else
        log_warning "$1 is not installed"
        return 1
    fi
}

print_help() {
    cat <<EOF
CI/CD Development Environment Setup (Repository-Agnostic)

Usage: $0 [OPTIONS]

Options:
  --help              Show this help message
  --minimal           Skip optional tools (cargo-audit, cargo-deny, etc.)
  --platform-only     Setup Android NDK and cross-compile targets only
  --verbose           Enable verbose output
  --no-hooks          Skip pre-commit hook installation
  --no-python         Skip Python environment setup
  --no-project        Skip project-specific setup (larql-python, etc.)

Configuration:
  Rust toolchain:     $RUST_TOOLCHAIN
  Python version:     $PYTHON_VERSION
  NDK version:        $NDK_VERSION

This script performs:
  1. Rust toolchain installation and setup
  2. Python environment setup (optional, can skip with --no-python)
  3. Pre-commit hook configuration (optional, can skip with --no-hooks)
  4. CI/CD tool installation (cocogitto, git-cliff, reuse, etc.)
  5. Optional: Android NDK for cross-platform builds
  6. Optional: Project-specific Python bindings (if crates/*/pyproject.toml found)
  7. Validation of the complete setup

The script is repository-agnostic and works in any directory. It autodetects
project structure and optionally configures project-specific environments.

EOF
}

# =============================================================================
# Argument Parsing
# =============================================================================

NO_HOOKS=false
NO_PYTHON=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help)
            print_help
            exit 0
            ;;
        --minimal)
            MINIMAL_MODE=true
            log_info "Minimal mode: skipping optional tools"
            shift
            ;;
        --platform-only)
            PLATFORM_ONLY_MODE=true
            log_info "Platform-only mode: setting up Android NDK only"
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --no-hooks)
            NO_HOOKS=true
            shift
            ;;
        --no-python)
            NO_PYTHON=true
            shift
            ;;
        --no-project)
            PROJECT_SETUP=false
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            print_help
            exit 1
            ;;
    esac
done

# =============================================================================
# Phase 1: Rust Toolchain
# =============================================================================

phase_rust_toolchain() {
    log_info "Phase 1: Installing Rust toolchain..."

    if ! command_exists rustup; then
        log_error "rustup is not installed. Install from https://rustup.rs/"
        exit 1
    fi

    log_info "Installing Rust $RUST_TOOLCHAIN..."
    rustup toolchain install "$RUST_TOOLCHAIN" \
        --profile minimal \
        --no-self-update

    rustup default "$RUST_TOOLCHAIN"

    log_success "Rust toolchain configured to $RUST_TOOLCHAIN"

    # Verify installation
    log_info "Verifying Rust installation..."
    rustc --version
    cargo --version

    # Install additional Rust components needed for CI/CD
    log_info "Installing Rust components..."
    rustup component add clippy --toolchain "$RUST_TOOLCHAIN"
    log_success "Clippy component installed"
}

# =============================================================================
# Phase 2: Python Environment (Optional)
# =============================================================================

phase_python_setup() {
    if [[ "$NO_PYTHON" == true ]]; then
        log_info "Skipping Python setup (--no-python)"
        return 0
    fi

    log_info "Phase 2: Setting up Python environment..."

    if ! command_exists python3; then
        log_error "python3 is not installed"
        return 1
    fi

    # Verify Python version matches requirement
    local py_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)
    log_info "Python version: $py_version"
    if [[ "$py_version" != "$PYTHON_VERSION" ]]; then
        log_warning "Python version mismatch: expected $PYTHON_VERSION, found $py_version"
    fi

    # Check if uv is installed
    if ! command_exists uv; then
        log_warning "uv is not installed. Installing..."
        python3 -m pip install --user uv
        export PATH="$HOME/.local/bin:$PATH"
        log_success "uv installed"
    else
        log_success "uv is installed"
    fi

    # Setup project-specific Python environment if available
    if [[ "$PROJECT_SETUP" == true ]]; then
        phase_python_project_setup
    fi
}

# =============================================================================
# Phase 2b: Project-Specific Python Setup (Optional)
# =============================================================================

phase_python_project_setup() {
    # Detect and configure project-specific Python environments
    local python_projects=()

    # Look for Python projects with uv configuration
    if [[ -f "$REPO_ROOT/pyproject.toml" ]]; then
        python_projects+=("$REPO_ROOT")
    fi

    # Look for Python crates (like larql-python)
    if [[ -d "$REPO_ROOT/crates" ]]; then
        while IFS= read -r -d '' crate_dir; do
            if [[ -f "$crate_dir/pyproject.toml" ]]; then
                python_projects+=("$crate_dir")
            fi
        done < <(find "$REPO_ROOT/crates" -maxdepth 1 -type d -print0 2>/dev/null)
    fi

    if [[ ${#python_projects[@]} -eq 0 ]]; then
        log_info "No Python projects detected in repository"
        return 0
    fi

    # Setup each detected Python project
    for project_dir in "${python_projects[@]}"; do
        project_name=$(basename "$project_dir")
        log_info "Setting up Python environment for $project_name..."

        if [[ -f "$project_dir/uv.lock" ]] || [[ -f "$project_dir/pyproject.toml" ]]; then
            cd "$project_dir"
            uv sync --no-install-project --group dev 2>/dev/null || true
            cd "$REPO_ROOT"
            log_success "Python environment ready for $project_name"
        fi
    done
}

# =============================================================================
# Phase 3: Pre-commit Hooks
# =============================================================================

phase_precommit_hooks() {
    if [[ "$NO_HOOKS" == true ]]; then
        log_info "Skipping pre-commit hook installation (--no-hooks)"
        return 0
    fi

    log_info "Phase 3: Configuring pre-commit hooks..."

    if ! command_exists pre-commit; then
        log_warning "pre-commit is not installed. Installing via pip3..."
        python3 -m pip install --user pre-commit
        export PATH="$HOME/.local/bin:$PATH"
    fi

    log_info "Installing pre-commit hooks..."
    pre-commit install --install-hooks
    pre-commit install --hook-type commit-msg

    log_success "Pre-commit hooks installed"
}

# =============================================================================
# Phase 4: CI/CD Tools
# =============================================================================

phase_cicd_tools() {
    log_info "Phase 4: Installing CI/CD tools..."

    # cocogitto (Conventional Commits validation)
    if ! command_exists cog; then
        log_info "Installing cocogitto $COCOGITTO_VERSION..."
        case "$(uname -s)" in
            Linux)
                if [[ "$(uname -m)" == "x86_64" ]]; then
                    curl -sSL "https://github.com/cocogitto/cocogitto/releases/download/${COCOGITTO_VERSION}/cocogitto-${COCOGITTO_VERSION}-x86_64-unknown-linux-musl.tar.gz" \
                        | tar -xz -C /tmp cocogitto-${COCOGITTO_VERSION}-x86_64-unknown-linux-musl/cog
                    sudo mv /tmp/cocogitto-${COCOGITTO_VERSION}-x86_64-unknown-linux-musl/cog /usr/local/bin/
                else
                    log_warning "Non-x86_64 Linux detected; installing cocogitto via cargo..."
                    cargo install cocogitto --version "$COCOGITTO_VERSION"
                fi
                ;;
            Darwin)
                log_warning "macOS detected; installing cocogitto via cargo..."
                cargo install cocogitto --version "$COCOGITTO_VERSION"
                ;;
            *)
                log_warning "Unsupported OS for cocogitto binary; install manually or use cargo"
                return 1
                ;;
        esac
        log_success "cocogitto installed"
    else
        log_success "cocogitto is installed"
    fi

    # git-cliff (changelog generation)
    if ! command_exists git-cliff; then
        log_info "Installing git-cliff $GIT_CLIFF_VERSION..."
        case "$(uname -s)" in
            Linux)
                if [[ "$(uname -m)" == "x86_64" ]]; then
                    curl -sSL "https://github.com/orhun/git-cliff/releases/download/v${GIT_CLIFF_VERSION}/git-cliff-${GIT_CLIFF_VERSION}-x86_64-unknown-linux-musl.tar.gz" \
                        | tar -xz -C /tmp
                    sudo mv "/tmp/git-cliff-${GIT_CLIFF_VERSION}/git-cliff" /usr/local/bin/
                else
                    log_warning "Non-x86_64 Linux detected; installing git-cliff via cargo..."
                    cargo install git-cliff --version "$GIT_CLIFF_VERSION"
                fi
                ;;
            Darwin)
                log_warning "macOS detected; installing git-cliff via cargo..."
                cargo install git-cliff
                ;;
            *)
                log_warning "Unsupported OS for git-cliff binary; install manually or use cargo"
                return 1
                ;;
        esac
        log_success "git-cliff installed"
    else
        log_success "git-cliff is installed"
    fi

    # REUSE tool (license compliance)
    if ! command_exists reuse; then
        log_info "Installing REUSE tool $REUSE_TOOL_VERSION..."
        python3 -m pip install --user reuse
        export PATH="$HOME/.local/bin:$PATH"
        log_success "REUSE tool installed"
    else
        log_success "REUSE tool is installed"
    fi

    if [[ "$MINIMAL_MODE" == false ]]; then
        log_info "Installing optional CI/CD scanning tools..."

        # cargo-audit (vulnerability scanning)
        log_info "Installing cargo-audit..."
        cargo install cargo-audit
        log_success "cargo-audit installed"

        # cargo-deny (dependency policy)
        log_info "Installing cargo-deny..."
        cargo install cargo-deny
        log_success "cargo-deny installed"

        # cargo-msrv (MSRV verification)
        log_info "Installing cargo-msrv..."
        cargo install cargo-msrv
        log_success "cargo-msrv installed"

        # cargo-mutants (mutation testing)
        log_info "Installing cargo-mutants..."
        cargo install cargo-mutants
        log_success "cargo-mutants installed"

        # clippy-sarif (SARIF output)
        log_info "Installing clippy-sarif..."
        cargo install clippy-sarif
        log_success "clippy-sarif installed"

        # buf (protobuf linting)
        if ! command_exists buf; then
            log_info "Installing buf..."
            case "$(uname -s)" in
                Linux)
                    curl -sSL "https://github.com/bufbuild/buf/releases/download/v1.45.0/buf-$(uname -s)-$(uname -m)" \
                        -o /tmp/buf
                    sudo mv /tmp/buf /usr/local/bin/
                    sudo chmod +x /usr/local/bin/buf
                    ;;
                Darwin)
                    brew install bufbuild/buf/buf
                    ;;
                *)
                    log_warning "Cannot install buf on this OS"
                    ;;
            esac
            log_success "buf installed"
        fi
    fi
}

# =============================================================================
# Phase 5: Platform Setup (Android NDK)
# =============================================================================

phase_platform_setup() {
    log_info "Phase 5: Setting up platform targets..."

    # Android NDK setup
    if [[ "$PLATFORM_ONLY_MODE" == true ]] || [[ "$MINIMAL_MODE" == false ]]; then
        log_info "Setting up Android NDK support..."

        if ! command_exists ndk-build; then
            log_warning "Android NDK not detected in PATH"
            log_info "To set up Android NDK:"
            log_info "  1. Download NDK r27 from https://developer.android.com/ndk"
            log_info "  2. Extract to a known location"
            log_info "  3. Set ANDROID_NDK_HOME environment variable"
            log_info "  4. Run: ./scripts/setup-dev-env.sh --platform-only"
        else
            log_success "Android NDK is available"
        fi
    fi

    # Add Android targets for cross-compilation
    if [[ "$PLATFORM_ONLY_MODE" == true ]] || [[ "$MINIMAL_MODE" == false ]]; then
        log_info "Adding Android compilation targets..."
        rustup target add aarch64-linux-android
        rustup target add armv7-linux-androideabi
        log_success "Android targets added"
    fi
}

# =============================================================================
# Phase 6: Validation
# =============================================================================

phase_validation() {
    log_info "Phase 6: Validating setup..."

    local failures=0

    # Check Rust setup
    if check_cmd cargo; then
        local rust_version=$(cargo --version | awk '{print $2}')
        log_success "Cargo version: $rust_version"
    else
        ((failures++))
    fi

    if check_cmd rustc; then
        local rustc_version=$(rustc --version | awk '{print $2}')
        log_success "Rustc version: $rustc_version"
    else
        ((failures++))
    fi

    if check_cmd rustfmt; then
        log_success "rustfmt is available"
    else
        ((failures++))
    fi

    if check_cmd clippy-driver; then
        log_success "clippy is available"
    else
        ((failures++))
    fi

    # Check Python setup (if not skipped)
    if [[ "$NO_PYTHON" == false ]]; then
        if check_cmd python3; then
            local py_version=$(python3 --version 2>&1 | awk '{print $2}')
            log_success "Python version: $py_version"
        else
            ((failures++))
        fi

        if check_cmd uv; then
            log_success "uv is available"
        else
            ((failures++))
        fi
    fi

    # Check pre-commit (if not skipped)
    if [[ "$NO_HOOKS" == false ]]; then
        if check_cmd pre-commit; then
            log_success "pre-commit is available"
        else
            ((failures++))
        fi
    fi

    # Check CI/CD tools
    if check_cmd cog; then
        log_success "cocogitto is available"
    else
        ((failures++))
    fi

    if check_cmd git-cliff; then
        log_success "git-cliff is available"
    else
        ((failures++))
    fi

    if check_cmd reuse; then
        log_success "REUSE tool is available"
    else
        ((failures++))
    fi

    # Check optional tools
    if [[ "$MINIMAL_MODE" == false ]]; then
        if ! check_cmd cargo-audit; then
            ((failures++))
        fi
        if ! check_cmd cargo-deny; then
            ((failures++))
        fi
    fi

    # Check that we're in the right directory
    if [[ -f "$REPO_ROOT/Cargo.toml" ]]; then
        log_success "Repository root detected: $REPO_ROOT"
    else
        log_error "Not in the repository root"
        ((failures++))
    fi

    if [[ $failures -eq 0 ]]; then
        log_success "All checks passed!"
        return 0
    else
        log_warning "$failures check(s) failed"
        return 1
    fi
}

# =============================================================================
# Quick Test
# =============================================================================

phase_quick_test() {
    log_info "Phase 7: Running quick verification tests..."

    log_info "Testing cargo fmt..."
    if cargo fmt --all -- --check > /dev/null 2>&1; then
        log_success "Code formatting check passed"
    else
        log_warning "Code formatting check failed (expected for new repos)"
    fi

    log_info "Testing cargo clippy..."
    if cargo clippy --workspace --tests -- -D warnings > /dev/null 2>&1; then
        log_success "Clippy check passed"
    else
        log_warning "Clippy check failed (this is expected for initial setup)"
    fi
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  CI/CD Development Environment Setup                           ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo

    if [[ "$PLATFORM_ONLY_MODE" == true ]]; then
        log_info "Running in platform-only mode"
        phase_platform_setup
    else
        phase_rust_toolchain
        phase_python_setup
        phase_precommit_hooks
        phase_cicd_tools
        phase_platform_setup
    fi

    phase_validation
    if [[ "$PLATFORM_ONLY_MODE" == false ]]; then
        phase_quick_test
    fi

    echo
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  Setup Complete!                                               ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo

    log_success "Development environment is ready"
    log_info "Common next steps:"

    if [[ -f "$REPO_ROOT/Makefile" ]]; then
        echo "  • make ci               (run all local checks)"
        echo "  • make test             (run full test suite)"
    fi

    if [[ -f "$REPO_ROOT/.git" ]]; then
        echo "  • pre-commit run --all-files  (validate all files)"
        echo "  • cog check HEAD~N..HEAD      (validate commit history)"
    fi

    if [[ -d "$REPO_ROOT/crates" ]]; then
        echo "  • cargo build --release       (build release binary)"
        echo "  • cargo test --workspace      (run workspace tests)"
    fi
    echo
}

# Trap errors
trap 'log_error "Setup failed at line $LINENO"; exit 1' ERR

# Run main
main "$@"
