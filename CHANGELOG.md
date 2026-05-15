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
- DeepSeekV4Arch — V4 tensor naming (no model. prefix, ffn, w1/w2/w3)
- MXFP4-aware streaming gate_vectors path
- Add Nix flake for reproducible builds; track Cargo.lock (#34)
- Add deterministic changelog and SemVer preflight checks
- Add modular Nix flake with demos, OCI containers, and model catalog
- Cap down_meta feature count via LARQL_SUMMARY_FEATURES_PER_EXPERT
- Cross-platform CI/CD foundation (Phase 1)
- Implement Android (Phase 2b) cross-platform CI/CD support
- Implement ChromeOS (Phase 2a) cross-platform CI/CD support
- Implement macOS (Phase 3) cross-platform CI/CD support
- Metadata-only resolve_hf_vindex (no eager binary downloads)
- Per-expert dequantization for DeepSeek-V4 layout
- Per-expert top-K SVD summary tier for many-experts MoE
- Support F8_E4M3 / F8_E5M2 / F8_E8M0 / I8 dtypes

### Fixed

- Rust toolchain upgrade to 1.92.0 for wasmtime 44.0.1+ (wasm-jit) compatibility
- Linux support — conditional BLAS and Q4 scalar fallback
- Linux/WSL2 support + temperature parameter
- MSRV truth, OpenBLAS, version-preflight, changelog
- Add Android NDK setup to cross-platform-build workflow
- Address code review feedback on CI scripts
- Address review feedback and CI environment realities
- Align license enforcement with audited multi-license tree
- Allow pulling vindexes from HF model repos
- Bump pinned versions and drop fmt CI duplication
- Bump toolchain to 1.88 and unpin scanner-tool versions
- Configure Android cross-compilation with linker and PATH setup
- Correct CHANGELOG.md structure and formatting
- Correct cog.toml schema, workflow flags, and review feedback
- Correct tool release URLs and pre-commit hook wiring
- Drop bogus `hidden_size % head_dim == 0` invariant
- Drop §4(b) per-file re-walk; rely on REUSE.toml manifest
- Error on missing config.json / required topology fields (#22)
- Gate metal-only code behind target_os = "macos" (#48)
- Gate metal-only code behind target_os = "macos" so the workspace builds on Linux
- Narrow `cog check` PR range to first-parent + no-merges
- Pull Q4K vindex weight artifacts
- Restore cfg-gated imports removed by PR #48
- Restore deleted extract/build.rs and align stale test/example initializers
- Restore extract/build.rs and align stale test/example initializers (#46)
- Restore extract/build.rs lost in d3a8bc6 + reconcile API drift
- Revert manual CHANGELOG edit; let git-cliff regenerate from commits
- Scope cron to advisory scanners and harden SARIF upload
- Silence unused cfg param in validate_one_layer
- Unblock CI tests broken by e67b4f3
- Unblock cargo, bump wasmtime past CVEs, require MSRV
- Use checked_div for head_dim derivation
- Use checked_div for head_dim derivation
- Use checked_div for head_dim derivation (#50)
- Use matmul_transb for MoE expert scoring


