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
- Add larql-wasm crate and serial-vs-parallel CI/CD pipeline
- Add modular Nix flake with demos, OCI containers, and model catalog
- Cap down_meta feature count via LARQL_SUMMARY_FEATURES_PER_EXPERT
- Cross-platform CI/CD foundation (Phase 1)
- Enable aarch64-linux-android cross-compilation
- Implement Android (Phase 2b) cross-platform CI/CD support
- Implement ChromeOS (Phase 2a) cross-platform CI/CD support
- Implement macOS (Phase 3) cross-platform CI/CD support
- Metadata-only resolve_hf_vindex (no eager binary downloads)
- Per-crate WASM compatibility matrix (16 × serial/parallel × node/firefox)
- Per-expert dequantization for DeepSeek-V4 layout
- Per-expert top-K SVD summary tier for many-experts MoE
- Support F8_E4M3 / F8_E5M2 / F8_E8M0 / I8 dtypes
- Wasmi migration, arm32 atomics, REUSE compliance

### Changed

- Move Python testing to per-crate workflow, fix cargo-deny wildcards
- Replace per-crate native workflows with forensic WASM matrix

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
- Apply code review suggestions
- Apply rustfmt and fix clippy::unnecessary_sort_by
- Build Graph per parallel worker to satisfy Sync bound
- Bump pinned versions and drop fmt CI duplication
- Bump toolchain to 1.88 and unpin scanner-tool versions
- Configure Android cross-compilation with linker and PATH setup
- Configure BLAS for Android in larql-inference and larql-kv
- Configure larql-compute BLAS for Android cross-compilation
- Correct CHANGELOG.md structure and formatting
- Correct test JSON format and parallel build-std arg passing
- Drop bogus `hidden_size % head_dim == 0` invariant
- Drop §4(b) per-file re-walk; rely on REUSE.toml manifest
- Error on missing config.json / required topology fields (#22)
- Gate UDS listener bind behind cfg(unix)
- Gate UDS shard transport behind cfg(unix)
- Gate forward_raw_logits imports alongside their sole user
- Gate metal-only code behind target_os = "macos" (#48)
- Gate metal-only code behind target_os = "macos" so the workspace builds on Linux
- Gate orphan items in vindex test + cover second lql bench
- Gate source-level tonic/tokio/axum types and native-only server modules behind `#[cfg(not(target_arch = "wasm32"))]`; larql-router-protocol, larql-router, and larql-server now pass `cargo check --target wasm32-unknown-unknown`
- Gate source-level reqwest::blocking/hf-hub references in larql-vindex (huggingface module), larql-inference (remote/moe_remote FFN modules), and larql-lql (Remote backend variant, executor/remote module); promote larql-compute from Blocked to Passing in wasm CI matrix
- Gate larql-core HttpProvider (reqwest::blocking) behind not(wasm32) in addition to http feature; gate vindexfile huggingface path resolver behind not(wasm32)
- Gate sdot on dotprod feature and add QEMU emulation for tests
- Gate trace_final_residual_matches_raw_forward_logits
- Move wasm-pack flags before crate path for latest wasm-pack compat
- Pin evalexpr to v11.3.1 (MIT) to avoid AGPL-3.0 at v12
- Prevent arithmetic overflow in lm_head vocab calculation on 32-bit platforms
- Pull Q4K vindex weight artifacts
- Remove BLIS dependency due to yanked transitive versions
- Remove draft guard and add rust-src for parallel build
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
- Switch browser target from Chrome to Firefox
- Unblock CI tests broken by e67b4f3
- Update runtime to use Engine with Config
- Use --features=parallel (equals) to avoid wasm-pack path ambiguity
- Use BLIS (pure-Rust BLAS) for Android cross-compilation
- Use blas-src netlib feature for Android BLAS
- Use checked_div for head_dim derivation
- Use checked_div for head_dim derivation
- Use checked_div for head_dim derivation (#50)
- Use matmul_transb for MoE expert scoring
- Use netlib (pure-Rust BLAS) for Android builds
- Use single-line wasm-pack commands to avoid backslash continuation


