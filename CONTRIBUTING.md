<!--
SPDX-License-Identifier: CC-BY-SA-4.0
Copyright 2026 Ian Douglas Lawrence Norman McLean

With attribution to Chris Hay for LARQL:
https://github.com/chrishayuk/chuk-larql-rs

This documentation is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
https://creativecommons.org/licenses/by-sa/4.0/
-->

# Contributing to larql-to-sparql

This document provides guidance for contributors to the larql-to-sparql repository.

## Setup & Environment

### Prerequisites

- **Rust**: 1.86 or later (MSRV enforced in `Cargo.toml`)
- **Python**: 3.11+ (for larql-python bindings)
- **Build dependencies**:
  - Linux: OpenBLAS development libraries (`libblas-dev`, `liblapack-dev`)
  - macOS: Metal framework (built-in on Apple Silicon)
  - Windows: MSVC toolchain

### Getting Started

```bash
# Clone the repository
git clone https://github.com/metavacua/larql-to-sparql.git
cd larql-to-sparql

# Check environment and dependencies
make check-env

# Build the workspace
cargo build --release

# Run tests
cargo test --workspace
```

### Python Environment Setup

For work on `larql-python` bindings:

```bash
cd crates/larql-python
uv sync --no-install-project --group dev     # Create .venv and install dev deps
uv run --no-sync maturin develop --release   # Build PyO3 extension into .venv
uv run --no-sync pytest tests/               # Run Python binding tests
```

Or via the Makefile:
```bash
make python-setup   # Create environment
make python-build   # Build extension
make python-test    # Run tests
make python-clean   # Clean environment
```

## Testing Before PR

All changes must pass the CI checks before merging:

```bash
# Format check
cargo fmt --check --all

# Lint check
cargo clippy --workspace --tests -- -D warnings

# Run all tests
cargo test --workspace
```

Or use the shorthand:
```bash
make ci   # Runs fmt-check, clippy, and tests
```

### Python-Specific Tests

If your changes touch `larql-python`:
```bash
make python-test
```

### Full Workflow Tests

For changes affecting model extraction, compilation, or inference:
```bash
# Include this in your commit message or run before PR:
# Full round-trip: extract → insert → compile → infer
```

## Commit Message Style

We encourage (but do not enforce) Conventional Commits for clarity:

```
feat(larql-lql): Add support for LIMIT clause in SELECT
fix(larql-compute): Handle negative strides in BLAS kernels
docs: Update inference engine architecture guide
refactor(larql-vindex): Extract patch logic into separate module
test(larql-inference): Add edge case tests for Metal GPU backend
```

### Commit Message Format

- **Type**: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`
- **Scope**: (optional) crate or module name in parentheses
- **Message**: Brief imperative description (max 72 characters)
- **Body**: (optional) Detailed explanation if needed
- **Issue reference**: (optional) Closes #123, Relates to #456

Examples:
```
feat(larql-cli): Add --output-format JSON flag

Allows users to export vindex metadata as JSON instead of pretty-printed text.
Useful for downstream processing pipelines.

Closes #123
```

## Versioning & Changelog

### Independent Per-Crate Versioning

Each crate maintains its own version in `Cargo.toml`. All crates start at `0.1.0`:

- `0.1.0` - Initial release (unstable API, frequent changes expected)
- `0.2.0` - API-breaking change, new features, or significant refactor
- `1.0.0` - Stable API (rare for young projects)

When publishing a crate (upstream for Chris Hay), the maintainer bumps the version and updates the changelog.

### Changelog Format (Keep a Changelog v1.1)

Each crate has a `CHANGELOG.md` in [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format:

```markdown
## [Unreleased]

### Added
- New feature or capability

### Changed
- Behavior change or enhancement to existing feature

### Deprecated
- Feature marked for future removal

### Removed
- Removed feature or API

### Fixed
- Bug fix

### Security
- Security vulnerability fix

## [0.1.0] - 2026-04-28

### Added
- Initial release
```

**Maintainers**: When releasing a crate version:
1. Move items from `[Unreleased]` to a dated version section
2. Update version in `Cargo.toml`
3. Commit and tag: `git tag v<crate>-<version>`

Example: `git tag vlarql-compute-0.2.0`

## License Information

### Source Code

**License**: Apache 2.0 (all source code)

- All `.rs` files must include at the top:
  ```rust
  // SPDX-License-Identifier: Apache-2.0
  ```

- All `.py` files must include at the top:
  ```python
  # SPDX-License-Identifier: Apache-2.0
  ```

- Build scripts (`build.rs`, `build.py`) must include the same header

No copyright attribution or contributor notes are required in source headers. The Apache 2.0 license applies uniformly to all source code, whether original or modified during the fork.

### Documentation

**License**: Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0)

- New documentation files (README, guides, CONTRIBUTING.md, etc.) are licensed under CC BY-SA 4.0
- Include the following header in new documentation files:
  ```markdown
  <!--
  SPDX-License-Identifier: CC-BY-SA-4.0
  Copyright 2026 Contributors to larql-to-sparql

  This documentation is licensed under the Creative Commons Attribution-ShareAlike 4.0 License.
  -->
  ```

## Workspace Structure

Refer to [AGENTS.md](AGENTS.md) for:
- Detailed workspace layout and dependency hierarchy
- Key architectural invariants
- Where to find specification documents
- Python bindings documentation

## CI/CD and Workflows

See [.github/workflows/](.github/workflows/) for:
- Fast-path CI (quick fmt/lint/test checks)
- Full CI (all tests, including optional platforms)
- Nightly and experimental workflows

## Issues and Discussions

- **Bug reports**: GitHub Issues
- **Feature requests**: GitHub Issues with `enhancement` label
- **Architecture questions**: GitHub Discussions

When opening an issue, please include:
- Minimal reproducible example (for bugs)
- Current behavior vs. expected behavior
- Rust version and platform (output of `rustc --version` and `uname -a`)

## Code Review

All pull requests require at least one review before merge. Reviewers will check:

- ✓ Tests pass (CI must be green)
- ✓ Code follows Rust conventions (clippy, fmt)
- ✓ SPDX headers present in new files
- ✓ Changelog entries in relevant `CHANGELOG.md` files
- ✓ No introduced regressions
- ✓ Architecture integrity (no dependency cycle violations)

## Questions?

For questions about:
- **Using larql**: Check [docs/](docs/) and [README.md](README.md)
- **Development setup**: See "Setup & Environment" above
- **Code contribution guidelines**: See "Testing Before PR" above
- **License compliance**: See "License Information" above

Happy contributing!
