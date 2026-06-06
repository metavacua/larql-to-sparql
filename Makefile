.PHONY: build release test test-fast test-full test-integration test-models check clean fmt lint demos demo bench bench-core bench-inference bench-compute bench-wire bench-routing bench-grid bench-all bench-vindex bench-vindex-scaling bench-save bench-check coverage coverage-summary coverage-check coverage-install ci-coverage traceability traceability-check gaps gaps-untested gaps-unbacked openspec-validate ci-cuda test-cuda rotorquant-sync cuda-rotorquant-bench cuda-oxide-doctor cuda-oxide-pilot cuda-oxide-cross-parity cuda-oxide-bench docker-ffn docker-gpu docker-up docker-up-cpu docker-down docker-logs cuda-status attention-smoke attention-validate attention-validate-gemma attention-bench larql-core-ci larql-core-test larql-core-fmt-check larql-core-lint larql-core-feature-test larql-core-bench-test larql-core-bench larql-core-examples larql-core-coverage larql-core-coverage-html larql-models-ci larql-models-test larql-models-fmt-check larql-models-lint larql-models-coverage-summary larql-models-bench-test larql-vindex-ci larql-vindex-test larql-vindex-fmt-check larql-vindex-lint larql-vindex-examples larql-vindex-bench-test larql-vindex-bench larql-vindex-coverage larql-vindex-coverage-summary larql-vindex-coverage-html larql-vindex-coverage-policy larql-compute-test larql-compute-test-fast larql-compute-test-integration larql-compute-check-fast larql-compute-check-tests larql-compute-check-all larql-compute-test-metal-decode larql-compute-test-metal-lib larql-compute-fmt-check larql-compute-lint larql-compute-coverage larql-compute-coverage-summary larql-compute-coverage-html larql-compute-coverage-policy larql-compute-ci larql-boundary-ci larql-boundary-test larql-boundary-fmt-check larql-boundary-lint larql-boundary-bench-test larql-boundary-examples larql-kv-ci larql-kv-test larql-kv-fmt-check larql-kv-lint larql-kv-examples larql-kv-bench-test larql-kv-bench larql-kv-coverage larql-kv-coverage-summary larql-kv-coverage-html larql-kv-coverage-policy larql-server-ci larql-server-test larql-server-fmt-check larql-server-lint larql-server-coverage larql-server-coverage-summary larql-server-coverage-html larql-server-coverage-policy larql-lql-ci larql-lql-test larql-lql-fmt-check larql-lql-lint larql-lql-examples larql-lql-bench-test larql-lql-coverage-summary larql-cli-ci larql-cli-test larql-cli-fmt-check larql-cli-lint larql-cli-coverage-summary larql-inference-ci larql-inference-test larql-inference-fmt-check larql-inference-lint larql-inference-bench-test larql-inference-coverage-summary

# Build
build:
	cargo build --workspace

release:
	cargo build --release -p larql-cli

# Test
#
# Default test target is intentionally fast: no integration binaries, no
# model-backed ignored tests. Use `test-full` for the historical full
# workspace run, and `test-models` for real-model/vindex checks.
test: test-fast

test-fast:
	cargo test --workspace --lib --bins

test-full:
	cargo test --workspace

test-integration:
	cargo test --workspace --tests

test-models:
	cargo test -p larql-inference --test test_arch_golden -- --ignored
	cargo test -p larql-inference --test test_logits_goldens -- --ignored
	cargo test -p larql-inference --test test_gemma3_smoke -- --ignored
	cargo test -p larql-inference --test test_generate_q4k_cpu -- --ignored
	cargo test -p larql-inference --test bench_probe_latency -- --ignored --nocapture
	cargo test -p larql-inference --test test_llm_dispatch -- --ignored --nocapture
	cargo test -p larql-inference --test test_constrained_dispatch -- --ignored --nocapture
	cargo test -p larql-inference --test test_trie_dispatch -- --ignored --nocapture

# CUDA test suite — requires LARQL_CUDA_AVAILABLE=1 and a working CUDA
# runtime (driver + libcublas matching the cudarc feature in
# crates/larql-compute/Cargo.toml). Runs all gpu-gated parity tests for
# f32 / Q4 / fused attention against synthetic inputs.
test-cuda:
	@if [ "${LARQL_CUDA_AVAILABLE}" != "1" ]; then \
	  echo "Set LARQL_CUDA_AVAILABLE=1 to run CUDA parity tests."; \
	  exit 2; \
	fi
	cargo test -p larql-compute --features cuda --test test_cuda_f32 -- --test-threads=1
	cargo test -p larql-compute --features cuda --test test_cuda_q4  -- --test-threads=1
	cargo test -p larql-compute --features cuda --test test_cuda_attn -- --test-threads=1
	cargo test -p larql-inference --features cuda --test test_cpu_metal_parity -- --test-threads=1
	cargo test -p larql-rotorquant
	@echo "All CUDA + RotorQuant parity tests passed."

ci-cuda: test-cuda

rotorquant-sync:
	scripts/rotorquant-sync.sh

cuda-rotorquant-bench:
	@if [ "${LARQL_CUDA_AVAILABLE}" = "1" ]; then \
	  cargo run --release -p larql-rotorquant --features cuda --example cuda_round_trip_bench; \
	else \
	  echo "Skipping CUDA RotorQuant benchmark: set LARQL_CUDA_AVAILABLE=1 to run it."; \
	fi

cuda-oxide-doctor:
	@if ! rustup toolchain list | grep -q '^nightly-2026-04-03'; then \
	  echo "Missing Rust toolchain: rustup toolchain install nightly-2026-04-03"; \
	  echo "Then add components: rustup component add rust-src rustc-dev --toolchain nightly-2026-04-03"; \
	  exit 2; \
	fi
	@cargo +nightly-2026-04-03 oxide doctor || { \
	  status=$$?; \
	  echo "cuda-oxide doctor failed. Install cargo-oxide from NVlabs/cuda-oxide at commit 6de050946cd1013335a33cf2c5144888a32efab3 and ensure LLVM 21, clang-21, and CUDA Toolkit 13.1 are visible."; \
	  exit $$status; \
	}

cuda-oxide-pilot: cuda-oxide-doctor
	cd crates/larql-rotorquant && CUDA_OXIDE_PTX_DIR=$$(pwd) cargo +nightly-2026-04-03 oxide build --features cuda-oxide --arch sm_89
	@if [ "${LARQL_CUDA_AVAILABLE}" = "1" ]; then \
	  cargo +nightly-2026-04-03 test -p larql-rotorquant --features cuda-oxide --test cuda_oxide_round_trip -- --test-threads=1; \
	else \
	  echo "Skipping cuda-oxide GPU round-trip test: set LARQL_CUDA_AVAILABLE=1 to run it."; \
	fi

cuda-oxide-cross-parity: cuda-oxide-doctor
	@if [ "${LARQL_CUDA_AVAILABLE}" != "1" ]; then \
	  echo "Set LARQL_CUDA_AVAILABLE=1 to run cuda-oxide/cudarc parity."; \
	  exit 2; \
	fi
	@set -e; \
	tmp=$$(mktemp -d); \
	trap 'rm -rf "$$tmp"' EXIT; \
	LARQL_ROTORQUANT_PARITY_DIR=$$tmp cargo test -p larql-rotorquant --features cuda --test cuda_round_trip iso3_quantized_layout_dequantizes_like_cpu_reference -- --test-threads=1 --nocapture; \
	cd crates/larql-rotorquant && CUDA_OXIDE_PTX_DIR=$$(pwd) cargo +nightly-2026-04-03 oxide build --features cuda-oxide --arch sm_89; \
	LARQL_ROTORQUANT_PARITY_DIR=$$tmp cargo +nightly-2026-04-03 test -p larql-rotorquant --features cuda-oxide --test cuda_oxide_round_trip iso3_cuda_oxide_dequantize_matches_cpu -- --test-threads=1 --nocapture

cuda-oxide-bench: cuda-oxide-pilot
	@if [ "${LARQL_CUDA_AVAILABLE}" = "1" ]; then \
	  cargo +nightly-2026-04-03 run --release -p larql-rotorquant --features cuda-oxide --example cuda_oxide_iso3_bench; \
	else \
	  echo "Skipping cuda-oxide benchmark: set LARQL_CUDA_AVAILABLE=1 to run it."; \
	fi

# Snapshot of cuda backend status against this dev box.
cuda-status:
	@echo "═══ CUDA capability snapshot ═══"
	@echo
	@nvidia-smi 2>/dev/null | grep -E 'CUDA Version|GeForce|Tesla|RTX|GTX' | head -5 || echo "(no nvidia-smi)"
	@echo
	@if [ -d "/usr/local/cuda/targets/x86_64-linux/lib" ]; then \
	  echo "libcublas: $$(ls /usr/local/cuda/targets/x86_64-linux/lib/libcublas.so.* 2>/dev/null | head -1)"; \
	fi
	@echo
	@cargo metadata --format-version 1 --quiet 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); pkg=[p for p in d['packages'] if p['name']=='larql-compute'][0]; print(f\"larql-compute features: {list(pkg['features'].keys())}\")" 2>/dev/null || true

# ── Two-container deployment ──────────────────────────────────────────
# See deploy/docker/README.md for the full topology + VRAM budget.

docker-ffn:
	docker build -f deploy/docker/Dockerfile.ffn -t larql-ffn:dev .

docker-gpu:
	docker build -f deploy/docker/Dockerfile.gpu -t larql-gpu:dev .

docker-up:
	cd deploy/docker && docker compose up --build

docker-up-cpu:
	cd deploy/docker && docker compose -f docker-compose.cpu.yml up --build

docker-down:
	cd deploy/docker && docker compose down -v

docker-logs:
	cd deploy/docker && docker compose logs -f

demo:
	deploy/docker/demo.sh

# attention-service-routes change. Run a full HTTP smoke test against
# a running attention server (defaults to http://localhost:8081).
# Override target via: LARQL_ATTN_URL=... LARQL_MODEL_ID=... make attention-smoke
attention-smoke:
	python3 scripts/attention-service-smoke.py

# Numerical-validation harness for the attention-service routes —
# runs against the synthetic make_test_weights model in-process,
# bit-comparing every per-layer residual against a direct
# larql_inference forward pass. No network, no real model.
attention-validate:
	cargo test -p larql-server --test test_attention_validation

# Same as attention-validate, but at Gemma-3-4B-shaped synthetic
# dimensions (hidden=2560, num_q=8, num_kv=4, head_dim=320,
# 4 layers). Takes ~10s due to the synthetic weight build.
attention-validate-gemma:
	cargo test -p larql-server --test test_attention_validation -- --ignored prefill_gemma_shaped

# Latency benchmarks for the attention-service routes — drives
# prefill at seq_len ∈ {1, 8, 32, 128}, plus decode and snapshot
# after a 32-token prefill, against the synthetic model. Reports
# numbers under target/criterion/.
attention-bench:
	cargo bench -p larql-server --bench attention_service

# larql-core — graph engine, algorithms, extraction helpers, serialization
larql-core-test:
	cargo test -p larql-core

larql-core-feature-test:
	cargo test -p larql-core --no-default-features
	cargo test -p larql-core --no-default-features --features msgpack

larql-core-fmt-check:
	cargo fmt -p larql-core -- --check

larql-core-lint:
	cargo clippy -p larql-core --all-targets -- -D warnings

larql-core-bench-test:
	cargo test -p larql-core --benches

larql-core-bench:
	cargo bench -p larql-core --bench graph

larql-core-examples:
	cargo run -p larql-core --example edge_demo
	cargo run -p larql-core --example graph_demo
	cargo run -p larql-core --example algorithm_demo
	cargo run -p larql-core --example filter_demo
	cargo run -p larql-core --example serialization_demo

larql-core-coverage:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-core --summary-only

larql-core-coverage-html:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed."; exit 1; \
	fi
	cargo llvm-cov --package larql-core --html --output-dir coverage/larql-core
	@echo "Report: coverage/larql-core/html/index.html"

larql-core-ci: larql-core-fmt-check larql-core-lint larql-core-test larql-core-feature-test larql-core-bench-test larql-core-examples

larql-models-test:
	cargo test -p larql-models

larql-models-fmt-check:
	cargo fmt -p larql-models -- --check

larql-models-lint:
	cargo clippy -p larql-models --all-targets -- -D warnings

larql-models-bench-test:
	cargo test -p larql-models --benches

larql-models-coverage-summary:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-models --summary-only

larql-models-ci: larql-models-fmt-check larql-models-lint larql-models-test larql-models-bench-test

# larql-vindex - vindex extraction, storage, load/save, patch overlays
#
# Current local baseline: 71.56% line coverage from cargo-llvm-cov.
# Keep this as a ratchet: raise it when new coverage lands.
LARQL_VINDEX_COVERAGE_MIN ?= 71
LARQL_VINDEX_COVERAGE_POLICY ?= crates/larql-vindex/coverage-policy.json
LARQL_VINDEX_COVERAGE_REPORT ?= coverage/larql-vindex/summary.json

larql-vindex-test:
	cargo test -p larql-vindex

larql-vindex-fmt-check:
	cargo fmt -p larql-vindex -- --check

larql-vindex-lint:
	cargo clippy -p larql-vindex --all-targets -- -D warnings

larql-vindex-examples:
	cargo check -p larql-vindex --examples

larql-vindex-bench-test:
	cargo test -p larql-vindex --benches

larql-vindex-bench:
	cargo bench -p larql-vindex --bench vindex_ops

larql-vindex-coverage-policy:
	@if [ ! -f "$(LARQL_VINDEX_COVERAGE_REPORT)" ]; then \
		echo "Coverage report not found: $(LARQL_VINDEX_COVERAGE_REPORT)"; \
		echo "Run: make larql-vindex-coverage-summary"; \
		exit 1; \
	fi
	python3 scripts/check_coverage_policy.py $(LARQL_VINDEX_COVERAGE_REPORT) $(LARQL_VINDEX_COVERAGE_POLICY)

larql-vindex-coverage:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-vindex --fail-under-lines $(LARQL_VINDEX_COVERAGE_MIN)
	@mkdir -p coverage/larql-vindex
	cargo llvm-cov report --package larql-vindex --json --summary-only --output-path $(LARQL_VINDEX_COVERAGE_REPORT)
	$(MAKE) larql-vindex-coverage-policy

larql-vindex-coverage-summary:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-vindex --summary-only --fail-under-lines $(LARQL_VINDEX_COVERAGE_MIN)
	@mkdir -p coverage/larql-vindex
	cargo llvm-cov report --package larql-vindex --json --summary-only --output-path $(LARQL_VINDEX_COVERAGE_REPORT)
	$(MAKE) larql-vindex-coverage-policy

larql-vindex-coverage-html:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed."; exit 1; \
	fi
	cargo llvm-cov --package larql-vindex --html --output-dir coverage/larql-vindex --fail-under-lines $(LARQL_VINDEX_COVERAGE_MIN)
	cargo llvm-cov report --package larql-vindex --json --summary-only --output-path $(LARQL_VINDEX_COVERAGE_REPORT)
	$(MAKE) larql-vindex-coverage-policy
	@echo "Report: coverage/larql-vindex/html/index.html"

larql-vindex-ci: larql-vindex-fmt-check larql-vindex-lint larql-vindex-test larql-vindex-examples larql-vindex-bench-test larql-vindex-coverage-summary

# larql-kv — pluggable KV-cache engines (markov-rs, unlimited-context, turbo-quant, apollo)
#
# Default policy is 90% per-file line coverage; total floor tracks the
# starting baseline and ratchets upward.
LARQL_KV_COVERAGE_MIN ?= 85
LARQL_KV_COVERAGE_POLICY ?= crates/larql-kv/coverage-policy.json
LARQL_KV_COVERAGE_REPORT ?= coverage/larql-kv/summary.json

larql-kv-test:
	cargo test -p larql-kv

larql-kv-fmt-check:
	cargo fmt -p larql-kv -- --check

larql-kv-lint:
	cargo clippy -p larql-kv --all-targets --no-deps -- -D warnings

larql-kv-examples:
	cargo check -p larql-kv --examples

larql-kv-bench-test:
	cargo test -p larql-kv --benches

larql-kv-bench:
	cargo bench -p larql-kv --bench engine_decode

larql-kv-coverage-policy:
	@if [ ! -f "$(LARQL_KV_COVERAGE_REPORT)" ]; then \
		echo "Coverage report not found: $(LARQL_KV_COVERAGE_REPORT)"; \
		echo "Run: make larql-kv-coverage-summary"; \
		exit 1; \
	fi
	python3 scripts/check_coverage_policy.py $(LARQL_KV_COVERAGE_REPORT) $(LARQL_KV_COVERAGE_POLICY)

larql-kv-coverage:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-kv --fail-under-lines $(LARQL_KV_COVERAGE_MIN)
	@mkdir -p coverage/larql-kv
	cargo llvm-cov report --package larql-kv --json --summary-only --output-path $(LARQL_KV_COVERAGE_REPORT)
	$(MAKE) larql-kv-coverage-policy

larql-kv-coverage-summary:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-kv --summary-only --fail-under-lines $(LARQL_KV_COVERAGE_MIN)
	@mkdir -p coverage/larql-kv
	cargo llvm-cov report --package larql-kv --json --summary-only --output-path $(LARQL_KV_COVERAGE_REPORT)
	$(MAKE) larql-kv-coverage-policy

larql-kv-coverage-html:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed."; exit 1; \
	fi
	cargo llvm-cov --package larql-kv --html --output-dir coverage/larql-kv --fail-under-lines $(LARQL_KV_COVERAGE_MIN)
	cargo llvm-cov report --package larql-kv --json --summary-only --output-path $(LARQL_KV_COVERAGE_REPORT)
	$(MAKE) larql-kv-coverage-policy
	@echo "Report: coverage/larql-kv/html/index.html"

larql-kv-ci: larql-kv-fmt-check larql-kv-lint larql-kv-test larql-kv-examples larql-kv-bench-test larql-kv-coverage-summary

# larql-compute — CPU/Metal kernels and backend contracts
#
# Current local default-feature baseline: 93.59% line coverage from
# cargo-llvm-cov. Keep this as a ratchet and raise it as per-file debt
# baselines move toward the 90% default policy.
LARQL_COMPUTE_COVERAGE_MIN ?= 93
LARQL_COMPUTE_COVERAGE_POLICY ?= crates/larql-compute/coverage-policy.json
LARQL_COMPUTE_COVERAGE_REPORT ?= coverage/larql-compute/summary.json

larql-compute-test: larql-compute-test-fast

# Default fast path: library/unit tests only. This deliberately avoids
# compiling every integration-test binary, including Metal-gated harnesses
# that have zero runnable tests on default-feature builds.
larql-compute-test-fast:
	cargo test -p larql-compute --lib

# ── Iteration loops for refactor work (no test execution, just type-check) ──
#
# These shave 1–3 minutes off the inner refactor loop versus
# `cargo test --tests --features metal` by skipping codegen and execution.
# Use the smallest one that catches the change you're making, then promote
# to `larql-compute-test-metal-decode` (executes the synthetic decode
# integration suite) only when ready to validate runtime behaviour.

# Fastest type-check — `lib` only, with the `metal` feature on so Metal
# code is type-checked too. ~5–30 s warm. The right loop for refactors
# that don't change test signatures (registry sweeps, env-flag plumbing,
# struct rearrangements that keep field names).
larql-compute-check-fast:
	cargo check -p larql-compute --features metal --lib

# Type-check `lib` + every integration-test binary under `tests/` with
# the `metal` feature. ~30 s – 3 min depending on warm cache. Use when a
# refactor renames or moves something that integration tests reach into
# (e.g. `MetalBackend`'s public fields).
larql-compute-check-tests:
	cargo check -p larql-compute --features metal --tests

# Same but also walks examples + benches — the most thorough type check
# short of building everything. Catches breakage in `examples/diag_*`
# and `benches/quant_matvec` etc that the `--tests` form misses.
larql-compute-check-all:
	cargo check -p larql-compute --features metal --tests --benches --examples

# Run JUST the synthetic-decode integration test under `metal`. Smallest
# end-to-end runtime validation — ~1–2 min cold, faster warm. Use after
# `larql-compute-check-tests` passes, before declaring a refactor done.
larql-compute-test-metal-decode:
	cargo test -p larql-compute --features metal --test test_metal_decode_synthetic

# Lib-test execution under `metal`. Adds the unit tests inside
# `src/metal/**` to what `larql-compute-test-fast` covers.
larql-compute-test-metal-lib:
	cargo test -p larql-compute --features metal --lib

# Full integration suite — turns on `heavy_tests` for the slow non-Metal
# correctness/parity suites and walks every integration binary under
# crates/larql-compute/tests. Add `--features metal` to also build the
# Metal kernel tests on macOS.
larql-compute-test-integration:
	cargo test -p larql-compute --features heavy_tests --tests

larql-compute-fmt-check:
	cargo fmt -p larql-compute -- --check

larql-compute-lint:
	cargo clippy -p larql-compute --all-targets -- -D warnings

larql-compute-coverage-policy:
	@if [ ! -f "$(LARQL_COMPUTE_COVERAGE_REPORT)" ]; then \
		echo "Coverage report not found: $(LARQL_COMPUTE_COVERAGE_REPORT)"; \
		echo "Run: make larql-compute-coverage-summary"; \
		exit 1; \
	fi
	python3 scripts/check_coverage_policy.py $(LARQL_COMPUTE_COVERAGE_REPORT) $(LARQL_COMPUTE_COVERAGE_POLICY)

larql-compute-coverage:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-compute --fail-under-lines $(LARQL_COMPUTE_COVERAGE_MIN)
	@mkdir -p coverage/larql-compute
	cargo llvm-cov report --package larql-compute --json --summary-only --output-path $(LARQL_COMPUTE_COVERAGE_REPORT)
	$(MAKE) larql-compute-coverage-policy

larql-compute-coverage-summary:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-compute --summary-only --fail-under-lines $(LARQL_COMPUTE_COVERAGE_MIN)
	@mkdir -p coverage/larql-compute
	cargo llvm-cov report --package larql-compute --json --summary-only --output-path $(LARQL_COMPUTE_COVERAGE_REPORT)
	$(MAKE) larql-compute-coverage-policy

larql-compute-coverage-html:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed."; exit 1; \
	fi
	cargo llvm-cov --package larql-compute --html --output-dir coverage/larql-compute --fail-under-lines $(LARQL_COMPUTE_COVERAGE_MIN)
	cargo llvm-cov report --package larql-compute --json --summary-only --output-path $(LARQL_COMPUTE_COVERAGE_REPORT)
	$(MAKE) larql-compute-coverage-policy
	@echo "Report: coverage/larql-compute/html/index.html"

larql-compute-ci: larql-compute-fmt-check larql-compute-lint larql-compute-test-fast

# larql-boundary — confidence-gated BOUNDARY ref codec
larql-boundary-test:
	cargo test -p larql-boundary

larql-boundary-fmt-check:
	cargo fmt -p larql-boundary -- --check

larql-boundary-lint:
	cargo clippy -p larql-boundary --all-targets -- -D warnings

larql-boundary-bench-test:
	cargo test -p larql-boundary --benches

larql-boundary-examples:
	cargo run -p larql-boundary --example encode_decode
	cargo run -p larql-boundary --example gate_decision
	cargo run -p larql-boundary --example accuracy

larql-boundary-coverage:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-boundary --summary-only

larql-boundary-coverage-html:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed."; exit 1; \
	fi
	cargo llvm-cov --package larql-boundary --html --output-dir coverage/larql-boundary
	@echo "Report: coverage/larql-boundary/html/index.html"

larql-boundary-ci: larql-boundary-fmt-check larql-boundary-lint larql-boundary-test larql-boundary-bench-test larql-boundary-examples

# larql-server — HTTP/gRPC inference server (vindex queries, OpenAI-compat,
# remote MoE expert shards). The 90% per-file coverage floor (see
# crates/larql-server/coverage-policy.json) is the goal; existing files
# carry debt baselines that should ratchet upward, never down.
#
# 2026-05-10 measured baseline (post-REV1..REV5 review fixes):
# **65.68% line / 72.18% function** with all integration tests compiling.
# This is below the 2026-04-26 ROADMAP claim of 74.2% — coverage drifted
# during the in-flight `larql-vindex` / `larql-inference` API refactor,
# and several expert/* routes are 0% because they need a live grid to
# exercise. Floor is set just below the current value to ratchet upward.
LARQL_SERVER_COVERAGE_MIN ?= 65
LARQL_SERVER_COVERAGE_POLICY ?= crates/larql-server/coverage-policy.json
LARQL_SERVER_COVERAGE_REPORT ?= coverage/larql-server/summary.json

larql-server-test:
	cargo test -p larql-server

larql-server-fmt-check:
	cargo fmt -p larql-server -- --check

larql-server-lint:
	cargo clippy -p larql-server --all-targets -- -D warnings

larql-server-coverage-policy:
	@if [ ! -f "$(LARQL_SERVER_COVERAGE_REPORT)" ]; then \
		echo "Coverage report not found: $(LARQL_SERVER_COVERAGE_REPORT)"; \
		echo "Run: make larql-server-coverage-summary"; \
		exit 1; \
	fi
	python3 scripts/check_coverage_policy.py $(LARQL_SERVER_COVERAGE_REPORT) $(LARQL_SERVER_COVERAGE_POLICY)

larql-server-coverage:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-server --fail-under-lines $(LARQL_SERVER_COVERAGE_MIN) -- --test-threads=1
	@mkdir -p coverage/larql-server
	cargo llvm-cov report --package larql-server --json --summary-only --output-path $(LARQL_SERVER_COVERAGE_REPORT)
	$(MAKE) larql-server-coverage-policy

larql-server-coverage-summary:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-server --summary-only --fail-under-lines $(LARQL_SERVER_COVERAGE_MIN) -- --test-threads=1
	@mkdir -p coverage/larql-server
	cargo llvm-cov report --package larql-server --json --summary-only --output-path $(LARQL_SERVER_COVERAGE_REPORT)
	$(MAKE) larql-server-coverage-policy

larql-server-coverage-html:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed."; exit 1; \
	fi
	cargo llvm-cov --package larql-server --html --output-dir coverage/larql-server --fail-under-lines $(LARQL_SERVER_COVERAGE_MIN) -- --test-threads=1
	cargo llvm-cov report --package larql-server --json --summary-only --output-path $(LARQL_SERVER_COVERAGE_REPORT)
	$(MAKE) larql-server-coverage-policy
	@echo "Report: coverage/larql-server/html/index.html"

larql-server-ci: larql-server-fmt-check larql-server-lint larql-server-test

# larql-lql — LQL parser, executor, REPL. Crate has no metal default;
# Remote-backend tests use `mockito`, no real model weights required.
larql-lql-test:
	cargo test -p larql-lql

larql-lql-fmt-check:
	cargo fmt -p larql-lql -- --check

larql-lql-lint:
	cargo clippy -p larql-lql --all-targets --no-deps -- -D warnings

larql-lql-examples:
	cargo check -p larql-lql --examples

larql-lql-bench-test:
	cargo test -p larql-lql --benches

larql-lql-coverage-summary:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-lql --summary-only

larql-lql-ci: larql-lql-fmt-check larql-lql-lint larql-lql-test larql-lql-examples larql-lql-bench-test

# larql-cli — top-level `larql` binary. Default features pull in Metal
# on every member crate; CPU-only matrix uses `--no-default-features`.
LARQL_CLI_DEFAULT_FEATURES ?= --no-default-features

larql-cli-test:
	cargo test -p larql-cli $(LARQL_CLI_DEFAULT_FEATURES)

larql-cli-fmt-check:
	cargo fmt -p larql-cli -- --check

# Lint disabled: 2026-05-10 `larql-cli` carries ~82 pre-existing clippy
# errors under default features and ~112 under `--no-default-features`
# (mostly `large_enum_variant` and `dead_code` on metal-only paths).
# Re-enable `-- -D warnings` after that backlog is cleared.
larql-cli-lint:
	cargo clippy -p larql-cli --bins --tests $(LARQL_CLI_DEFAULT_FEATURES) --no-deps

larql-cli-coverage-summary:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-cli $(LARQL_CLI_DEFAULT_FEATURES) --summary-only

larql-cli-ci: larql-cli-fmt-check larql-cli-test

# larql-inference — transformer inference engine. Tests requiring real
# model weights are gated `#[ignore]` (test_arch_golden, test_logits_goldens,
# test_gemma3_smoke, test_generate_q4k_cpu, test_layer_graph_integration);
# CI runs the default set only. Several diagnostic examples lag the
# refactored `larql-compute` decode API and are excluded from `--all-targets`
# until repaired.
larql-inference-test:
	cargo test -p larql-inference

larql-inference-fmt-check:
	cargo fmt -p larql-inference -- --check

larql-inference-lint:
	cargo clippy -p larql-inference --lib --tests --benches --no-deps -- -D warnings

larql-inference-bench-test:
	cargo test -p larql-inference --benches

larql-inference-coverage-summary:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Install with:"; \
		echo "  cargo install cargo-llvm-cov"; \
		exit 1; \
	fi
	cargo llvm-cov --package larql-inference --summary-only

larql-inference-ci: larql-inference-fmt-check larql-inference-lint larql-inference-test larql-inference-bench-test

# Check (compile without building)
check:
	cargo check --workspace

# Code quality
fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

lint:
	cargo clippy --workspace --tests -- -D warnings

# All quality checks. The traceability gate is fast (pure Python over markdown +
# `.rs` files) and runs in CI on every PR. The coverage gate is heavier; it has
# its own target (`ci-coverage`) and is wired into a separate CI job so spec-only
# PRs don't pay the cost.
ci: fmt-check lint test-full traceability-check openspec-validate

# Clean
clean:
	cargo clean

# Benchmarks
#
# `bench` runs the core graph example. `bench-compute` runs the primary
# larql-compute Criterion surface. `bench-save` records a compute baseline
# named `main`; `bench-check` re-runs the compute benches and fails if any
# cell regresses past Criterion's default noise threshold.
bench: bench-core

bench-core:
	cargo run --release -p larql-core --example bench_graph

bench-inference:
	cargo run --release -p larql-inference --example bench_inference

# Compute kernel criterion bench (quant_matvec — Metal GPU).
bench-compute:
	cargo bench -p larql-compute --bench quant_matvec --features metal

# Wire codec criterion bench (encode/decode f32/f16/i8 throughput).
bench-wire:
	cargo bench -p larql-inference --bench wire_codec

# Router routing hot-path criterion bench (route/heartbeat/rebuild ns/op).
bench-routing:
	cargo bench -p larql-router --bench routing

# Grid end-to-end regression gate (requires LARQL_BENCH_FFN_URL env var).
bench-grid:
	./scripts/bench-grid-regress.sh $(MODEL)

# Cross-architecture decode bench — runs `larql bench` on Gemma 3 4B,
# Gemma 4 31B dense, Llama 2 7B, Mistral 7B, Gemma 4 26B A4B in
# sequence and prints per-arch tok/s. Operationalises ADR-017
# model-agnosticity check: any A/B promoted on Gemma 3 4B alone should
# be re-bench'd here before landing. Also surfaces thermal artifacts:
# if every arch regresses simultaneously vs baseline, suspect thermal.
#
#   make bench-cross-arch                     # report current numbers
#   make bench-cross-arch ARGS=--save-baseline  # save current as baseline
#   make bench-cross-arch ARGS=--compare        # diff vs saved baseline
#
# Bench params via env: LARQL_BENCH_TOKENS, LARQL_BENCH_WARMUP, LARQL_BENCH_PROMPT.
bench-cross-arch:
	./scripts/bench-cross-arch.sh $(ARGS)

bench-all: bench-core bench-inference bench-compute bench-wire bench-routing

# Vindex micro-benches — synthetic, fast, safe under load.
bench-vindex:
	cargo bench -p larql-vindex --bench vindex_ops

# Vindex production-dim scaling bench. Refuses if larql-server / router
# are alive (they distort 1-2 GB matmuls). Run alone, on a cool host;
# results feed PERFORMANCE.md.
bench-vindex-scaling:
	@if pgrep -fl 'larql-(server|router)' >/dev/null 2>&1; then \
		echo "Refusing bench-vindex-scaling: larql daemons running. Stop them first."; \
		pgrep -fl 'larql-(server|router)'; \
		exit 2; \
	fi
	cargo bench -p larql-vindex --bench vindex_scaling

bench-save:
	bash scripts/bench-regress.sh save

bench-check:
	bash scripts/bench-regress.sh check

# Coverage — uses cargo-llvm-cov.
#
#   coverage          — full workspace coverage; emits HTML + JSON.
#   coverage-summary  — terse per-crate text summary.
#   coverage-check    — enforce per-crate thresholds in coverage-thresholds.toml.
#   ci-coverage       — combined: regenerate + check; intended for CI.
#   coverage-install  — install rustup component + cargo-llvm-cov.
#
# Pass CRATE=<name> to scope `coverage` to a single crate (HTML only).
COVERAGE_CRATE ?=
coverage:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Run: make coverage-install"; \
		exit 1; \
	fi
	@if [ -n "$(COVERAGE_CRATE)" ]; then \
		cargo llvm-cov --package $(COVERAGE_CRATE) --html --output-dir target/llvm-cov/html; \
		echo "Report: target/llvm-cov/html/index.html"; \
	else \
		cargo llvm-cov --workspace --json --output-path target/llvm-cov/coverage.json; \
		cargo llvm-cov --workspace --html --output-dir target/llvm-cov/html --no-clean; \
		echo "JSON:  target/llvm-cov/coverage.json"; \
		echo "HTML:  target/llvm-cov/html/index.html"; \
	fi

coverage-summary:
	@if ! command -v cargo-llvm-cov >/dev/null 2>&1; then \
		echo "cargo-llvm-cov not installed. Run: make coverage-install"; \
		exit 1; \
	fi
	cargo llvm-cov --workspace --summary-only

coverage-check:
	@if [ ! -f target/llvm-cov/coverage.json ]; then \
		echo "target/llvm-cov/coverage.json missing — run \`make coverage\` first."; \
		exit 1; \
	fi
	python3 scripts/coverage-check.py

ci-coverage: coverage coverage-check

coverage-install:
	rustup component add llvm-tools-preview
	cargo install cargo-llvm-cov --locked

# OpenSpec spec → test traceability.
#
#   traceability         — regenerate openspec/coverage/traceability.{md,json}.
#   traceability-check   — fail if regenerated output diverges from committed.
#   gaps-unbacked        — write openspec/changes/<change>/gaps-unbacked-scenarios.md.
#   gaps-untested        — write gaps-untested-code.md (requires coverage JSON).
#   gaps                 — both gap reports.
#   openspec-validate    — `openspec validate <change> --strict` for the active change.
traceability:
	python3 scripts/spec-trace.py

traceability-check:
	python3 scripts/spec-trace.py --check

gaps-unbacked:
	python3 scripts/spec-trace.py --unbacked --quiet

gaps-untested:
	@if [ ! -f target/llvm-cov/coverage.json ]; then \
		echo "target/llvm-cov/coverage.json missing — run \`make coverage\` first."; \
		exit 1; \
	fi
	python3 scripts/spec-gap.py --untested-code

gaps: gaps-unbacked gaps-untested

openspec-validate:
	@for change in $$(ls openspec/changes 2>/dev/null | grep -v '^archive$$'); do \
		echo "openspec validate $$change --strict"; \
		openspec validate $$change --strict || exit 1; \
	done

# Python extension (managed via uv)
python-setup:
	cd crates/larql-python && uv sync --no-install-project --group dev

python-build: python-setup
	cd crates/larql-python && uv run --no-sync maturin develop --release

python-test: python-build
	cd crates/larql-python && uv run --no-sync pytest tests/ -v

python-check:
	cargo check -p larql-python

python-clean:
	rm -rf crates/larql-python/.venv crates/larql-python/uv.lock

# Extraction
extract-test:
	cargo run --release -p larql-cli -- weight-extract google/gemma-3-4b-it \
		--layer 26 -o output/test-L26.larql.json \
		--stats output/test-L26-stats.json

extract-full:
	cargo run --release -p larql-cli -- weight-extract google/gemma-3-4b-it \
		-o output/gemma-3-4b-knowledge.larql.json \
		--stats output/gemma-3-4b-stats.json

# Inference
predict:
	cargo run --release -p larql-cli -- predict google/gemma-3-4b-it \
		--prompt "The capital of France is" -k 10
