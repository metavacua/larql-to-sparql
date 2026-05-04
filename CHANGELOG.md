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

### Fixed

- Add Android NDK setup to cross-platform-build workflow
- Address code review feedback on CI scripts
- Address review feedback and CI environment realities
- Align license enforcement with audited multi-license tree
- Bump pinned versions and drop fmt CI duplication
- Bump toolchain to 1.88 and unpin scanner-tool versions
- Configure Android cross-compilation with linker and PATH setup
- Correct CHANGELOG.md structure and formatting
- Correct cog.toml schema, workflow flags, and review feedback
- Correct tool release URLs and pre-commit hook wiring
- Drop §4(b) per-file re-walk; rely on REUSE.toml manifest
- Linux support — conditional BLAS and Q4 scalar fallback
- Linux/WSL2 support + temperature parameter
- Revert manual CHANGELOG edit; let git-cliff regenerate from commits
- Scope cron to advisory scanners and harden SARIF upload


