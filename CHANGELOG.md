<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Ian Douglas Lawrence Norman McLean
SPDX-License-Identifier: Apache-2.0
-->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).
## [Unreleased]

### Added

- Add Gemma 4 GGUF support + fix column-major loading and Q4_K dequantization (#1)
- Add deterministic changelog and SemVer preflight checks
- Cross-platform CI/CD foundation (Phase 1)
- Implement Android (Phase 2b) cross-platform CI/CD support
- Implement ChromeOS (Phase 2a) cross-platform CI/CD support
- Implement macOS (Phase 3) cross-platform CI/CD support
- **BREAKING:** Implement unified CI/CD pipeline with license verification gate

### Fixed

- Linux support — conditional BLAS and Q4 scalar fallback
- Linux/WSL2 support + temperature parameter
- Acknowledge known pre-existing unmaintained dependency advisories
- Add Android NDK setup to cross-platform-build workflow
- Address PR #29 review feedback and surface known build issues
- Address cargo deny failures with explicit path dependency versions and wasmtime upgrade
- Address code review feedback on CI scripts
- Address review feedback and CI environment realities
- Align license enforcement with audited multi-license tree
- Bump pinned versions and drop fmt CI duplication
- Bump toolchain to 1.88 and unpin scanner-tool versions
- Configure Android cross-compilation with linker and PATH setup
- Continue inlining format string variables in larql-lql
- Correct CHANGELOG.md structure and formatting
- Correct cog.toml schema, workflow flags, and review feedback
- Correct invalid format strings with double colons
- Correct tool release URLs and pre-commit hook wiring
- Drop §4(b) per-file re-walk; rely on REUSE.toml manifest
- Format helpers.rs error message to single line
- Inline additional format arguments and apply rustfmt
- Inline additional format arguments in model-compute
- Inline format arguments to resolve clippy warnings
- Inline format string variables across CLI and server examples
- Inline format string variables in embed_demo and server_demo examples
- Inline format strings in describe.rs
- Inline remaining format arguments in larql-inference
- Inline variables in format strings (clippy uninlined_format_args)
- Resolve clippy uninlined_format_args warnings across all crates
- Resolve clippy warnings blocking CI
- Resolve compilation failures with metal feature gating and wasmtime 36.0.7
- Resolve remaining clippy uninlined_format_args errors
- Resolve remaining format string warnings in examples and source code
- Resolve three CI failures from Linux incompatibilities
- Revert manual CHANGELOG edit; let git-cliff regenerate from commits
- Scope cron to advisory scanners and harden SARIF upload
- Upgrade rust-version to 1.91 to support wasmtime 43.0.2 security fixes


