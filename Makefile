.PHONY: build release test check clean fmt lint demos help check-env

# Detect OS for platform-specific behavior
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    DETECTED_OS := macos
endif
ifeq ($(UNAME_S),Linux)
    DETECTED_OS := linux
endif

# Help target (always first for discoverability)
help:
	@echo "LARQL Build System — Cross-Platform Rust + Python"
	@echo ""
	@echo "Core Targets:"
	@echo "  make check        Compile all crates (cargo check)"
	@echo "  make build        Build all crates"
	@echo "  make release      Build release binary (larql-cli)"
	@echo "  make test         Run all tests"
	@echo "  make clean        Clean build artifacts"
	@echo ""
	@echo "Code Quality:"
	@echo "  make fmt          Format all code (cargo fmt)"
	@echo "  make fmt-check    Check formatting without changes"
	@echo "  make lint         Run clippy linting (warnings as errors)"
	@echo "  make ci           Run fmt-check, lint, and test (like GitHub Actions)"
	@echo ""
	@echo "Python Bindings (requires uv or pip + venv):"
	@echo "  make python-check Build check for Python extension"
	@echo "  make python-setup Setup Python dev environment"
	@echo "  make python-build Build Python extension"
	@echo "  make python-test  Run Python tests"
	@echo "  make python-clean Clean Python build artifacts"
	@echo ""
	@echo "Examples & Benchmarks:"
	@echo "  make demos        Run all Rust demos"
	@echo "  make demos-inference  Run inference demo"
	@echo "  make bench        Run core benchmarks"
	@echo "  make bench-all    Run all benchmarks"
	@echo ""
	@echo "Extraction & Inference (requires models):"
	@echo "  make extract-test    Extract layer 26 from Gemma 3 4B"
	@echo "  make extract-full    Extract full Gemma 3 4B weights"
	@echo "  make predict         Run inference on Gemma 3 4B"
	@echo ""
	@echo "Environment:"
	@echo "  make check-env    Validate build dependencies"
	@echo "  make help         Show this help message"
	@echo ""
	@echo "Platform Info: $(DETECTED_OS)"

# Check environment (validate build dependencies)
check-env:
	@echo "Checking LARQL build environment..."
	@echo ""
	@echo "Required Tools:"
	@command -v cargo >/dev/null 2>&1 && echo "  ✓ cargo" || echo "  ✗ cargo (Rust required)"
	@command -v rustc >/dev/null 2>&1 && echo "  ✓ rustc" || echo "  ✗ rustc (Rust required)"
	@command -v rustfmt >/dev/null 2>&1 && echo "  ✓ rustfmt" || echo "  ✗ rustfmt (cargo install)"
	@echo ""
	@echo "Python Extension Tools:"
	@command -v python3 >/dev/null 2>&1 && echo "  ✓ python3" || echo "  ⚠ python3 (needed for Python bindings)"
	@if command -v uv >/dev/null 2>&1; then echo "  ✓ uv (Python package manager)"; else echo "  ⚠ uv (fallback to pip+venv)"; fi
	@echo ""
	@echo "Platform-Specific:"
	@if [ "$(DETECTED_OS)" = "macos" ]; then \
		echo "  macOS Detected:"; \
		uname -m | grep -q "arm64" && echo "    ✓ Metal framework available (Apple Silicon)" || echo "    ℹ Metal framework not available on this architecture (Intel Mac)"; \
	elif [ "$(DETECTED_OS)" = "linux" ]; then \
		echo "  Linux Detected:"; \
		pkg-config --cflags openblas >/dev/null 2>&1 && echo "    ✓ OpenBLAS (system)" || echo "    ℹ OpenBLAS (will be vendored if missing)"; \
	fi
	@echo ""
	@echo "✓ Build environment validated"


build:
	cargo build --workspace

release:
	cargo build --release -p larql-cli

# Test
test:
	cargo test --workspace

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

# All quality checks
ci: fmt-check lint test

# Clean
clean:
	cargo clean

# Demos
demos:
	cargo run --release -p larql-models --example architecture_demo
	cargo run --release -p larql-core --example graph_demo
	cargo run --release -p larql-core --example edge_demo
	cargo run --release -p larql-core --example serialization_demo
	cargo run --release -p larql-core --example algorithm_demo

demos-inference:
	cargo run --release -p larql-inference --example inference_demo

# Benchmarks
bench: bench-core

bench-core:
	cargo run --release -p larql-core --example bench_graph

bench-inference:
	cargo run --release -p larql-inference --example bench_inference

bench-all: bench-core bench-inference

# Python extension (managed via uv or pip)
python-setup:
	@if command -v uv >/dev/null 2>&1; then \
		echo "Using uv for Python environment..."; \
		cd crates/larql-python && uv sync --no-install-project --group dev; \
	else \
		echo "uv not found, using pip + venv..."; \
		cd crates/larql-python && python3 -m venv .venv && . .venv/bin/activate && pip install -e .[dev] maturin; \
	fi

python-build: python-setup
	@if command -v uv >/dev/null 2>&1; then \
		cd crates/larql-python && uv run --no-sync maturin develop --release; \
	else \
		cd crates/larql-python && . .venv/bin/activate && maturin develop --release; \
	fi

python-test: python-build
	@if command -v uv >/dev/null 2>&1; then \
		cd crates/larql-python && uv run --no-sync pytest tests/ -v; \
	else \
		cd crates/larql-python && . .venv/bin/activate && pytest tests/ -v; \
	fi

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
