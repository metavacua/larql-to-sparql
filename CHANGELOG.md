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
- Add cross-platform CI/CD foundation (Phase 1): GitHub Actions workflow and local test scripts for Ubuntu, with Phase 2-3 skeletons for Android/ChromeOS/macOS

### Fixed

- Linux support — conditional BLAS and Q4 scalar fallback
- Linux/WSL2 support + temperature parameter
- Address review feedback and CI environment realities
- Align license enforcement with audited multi-license tree
- Bump pinned versions and drop fmt CI duplication
- Bump toolchain to 1.88 and unpin scanner-tool versions
- Correct cog.toml schema, workflow flags, and review feedback
- Correct tool release URLs and pre-commit hook wiring
- Drop §4(b) per-file re-walk; rely on REUSE.toml manifest
- Scope cron to advisory scanners and harden SARIF upload


