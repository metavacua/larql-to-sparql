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
- Enable aarch64-linux-android cross-compilation
- Implement Android (Phase 2b) cross-platform CI/CD support
- Implement ChromeOS (Phase 2a) cross-platform CI/CD support
- Implement macOS (Phase 3) cross-platform CI/CD support
- Metadata-only resolve_hf_vindex (no eager binary downloads)
- Per-expert dequantization for DeepSeek-V4 layout
- Per-expert top-K SVD summary tier for many-experts MoE
- Support F8_E4M3 / F8_E5M2 / F8_E8M0 / I8 dtypes
- Wasmi migration, arm32 atomics, REUSE compliance

### Changed

- Move Python testing to per-crate workflow, fix cargo-deny wildcards

### Fixed

- Linux support — conditional BLAS and Q4 scalar fallback
- Linux/WSL2 support + temperature parameter
- Add -C link-arg=-static to eliminate Android PT_INTERP
- Add Android NDK setup to cross-platform-build workflow
- Add arithmetic overflow fix to changelog
- Add missing down_meta.bin header in test fixture
- Address code review feedback on CI scripts
- Address review feedback and CI environment realities
- Address second-wave windows-fix CI failures
- Align license enforcement with audited multi-license tree
- Allow pulling vindexes from HF model repos
- Apply rustfmt and fix clippy::unnecessary_sort_by
- Bump pinned versions and drop fmt CI duplication
- Bump toolchain to 1.88 and unpin scanner-tool versions
- Configure Android cross-compilation with linker and PATH setup
- Configure BLAS for Android in larql-inference and larql-kv
- Configure larql-compute BLAS for Android cross-compilation
- Correct CHANGELOG.md structure and formatting
- Drop bogus `hidden_size % head_dim == 0` invariant
- Drop §4(b) per-file re-walk; rely on REUSE.toml manifest
- Error on missing config.json / required topology fields (#22)
- Gate UDS listener bind behind cfg(unix)
- Gate UDS shard transport behind cfg(unix)
- Gate forward_raw_logits imports alongside their sole user
- Gate metal-only code behind target_os = "macos" (#48)
- Gate metal-only code behind target_os = "macos" so the workspace builds on Linux
- Gate orphan items in vindex test + cover second lql bench
- Gate sdot on dotprod feature and add QEMU emulation for tests
- Gate trace_final_residual_matches_raw_forward_logits
- Install chromite (not depot_tools) to provide cros_sdk in CI
- Pin evalexpr to v11.3.1 (MIT) to avoid AGPL-3.0 at v12
- Pass --sdk-version to cros_sdk when running outside a ChromiumOS tree
- Prevent arithmetic overflow in lm_head vocab calculation on 32-bit platforms
- Pull Q4K vindex weight artifacts
- Remove BLIS dependency due to yanked transitive versions
- Resolve three review issues in ChromeOS/crosh CI integration
- Restore cfg-gated imports removed by PR #48
- Restore deleted extract/build.rs and align stale test/example initializers
- Restore extract/build.rs and align stale test/example initializers (#46)
- Restore extract/build.rs lost in d3a8bc6 + reconcile API drift
- Revert manual CHANGELOG edit; let git-cliff regenerate from commits
- Scope cron to advisory scanners and harden SARIF upload
- Silence unused cfg param in validate_one_layer
- Six platform-specific test/build failures on windows-latest
- Skip BLAS entirely for Android cross-compilation
- Skip default features check for Android in larql-compute
- Skip default features check for Android in larql-core
- Sync CHANGELOG.md unreleased block with git-cliff
- Unblock CI tests broken by e67b4f3
- Update runtime to use Engine with Config
- Use BLIS (pure-Rust BLAS) for Android cross-compilation
- Use blas-src netlib feature for Android BLAS
- Use checked_div for head_dim derivation
- Use checked_div for head_dim derivation
- Use checked_div for head_dim derivation (#50)
- Use matmul_transb for MoE expert scoring
- Use netlib (pure-Rust BLAS) for Android builds


