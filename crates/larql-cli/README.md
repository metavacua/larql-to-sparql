# larql-cli

The `larql` command-line interface — a single binary that drives the whole
toolchain: vindex extraction and inspection, the LQL REPL, HuggingFace
Hub sync, and the HTTP/gRPC server.

Most commands are thin wrappers around the workspace crates: `larql-vindex`
(extract / build), `larql-models` (load weights), `larql-inference` (predict
/ walk), `larql-lql` (parser + executor), `larql-server` (serve).

```bash
# Build a standalone .vindex from a HuggingFace-layout model
cargo run --release -p larql-cli -- extract-index \
    --model google/gemma-3-4b-it \
    --output output/gemma3-4b.vindex

# Query it through LQL
cargo run --release -p larql-cli -- lql \
    'USE "output/gemma3-4b.vindex"; INFER "The capital of France is" TOP 5;'

# Or open the REPL
cargo run --release -p larql-cli -- repl

# Serve over HTTP/gRPC
cargo run --release -p larql-cli -- serve --dir output/ --port 8080

# Quantise an existing vindex (FP4 or GGML Q4_K_M) — see docs/specs/quantize-cli-spec.md
cargo run --release -p larql-cli -- convert quantize fp4 \
    --input  output/gemma3-4b.vindex \
    --output output/gemma3-4b-fp4.vindex
cargo run --release -p larql-cli -- convert quantize q4k \
    --input  output/gemma3-4b.vindex \
    --output output/gemma3-4b-q4k.vindex

# Engine diagnostic — print which kernel paths the loader picks for a
# vindex, validate Q4_K/Q6_K strides, and (with --probe) run a real
# forward pass and print per-stage timings.
cargo run --release --features metal -p larql-cli -- diag \
    output/gemma3-4b-q4k-v2.vindex --probe --probe-tokens 50
```

See [`docs/cli.md`](../../docs/cli.md) for the full command reference.

## Command families

| Family | Commands | What they do |
|---|---|---|
| **Vindex lifecycle** | `extract-index`, `build`, `slice`, `publish`, `pull`, `compile`, `convert`, `verify`, `hf` | Extract, build from a Vindexfile, **carve deployment slices** (`client`/`attn`/`embed`/`server`/`browse`/`router`), **publish** (full + 5 default slice siblings + collections to HF with SHA256-skip-if-unchanged), **pull** (with sibling hints, `--preset`, `--all-slices`, `--collection`), bake patches into weights, convert GGUF↔vindex↔safetensors, checksum, low-level HF helper |
| **Diagnostics** | `bench`, `diag`, `parity`, `verify`, `stats`, `validate` | `bench` runs end-to-end decode throughput; `diag <vindex> [--probe]` reports which kernel paths the loader will pick (lm_head fast/slow, attn fused/per-proj), validates Q4_K/Q6_K manifest strides against canonical 144-byte GGUF layout, and surfaces the silent-slowdown classes (stale 148-byte stride, `vocab_size=0`) at a glance |
| **LQL** | `repl`, `lql`, `query`, `describe`, `filter`, `merge`, `validate`, `stats` | Interactive REPL + one-shot LQL, plus lower-level graph utilities |
| **Weight-space extraction** | `weight-extract`, `attention-extract`, `vector-extract`, `index-gates`, `qk-templates`, `qk-rank`, `qk-modes`, `ov-gate`, `circuit-discover`, `fingerprint-extract` | Pull edges / templates / circuits from the model weights — zero forward passes |
| **Forward-pass analysis** | `predict`, `walk`, `residuals`, `attention-capture`, `extract-routes`, `trajectory-trace`, `bfs` | Run the model and capture residuals, attention patterns, trajectories |
| **Benchmarks & tests** | `ffn-bench`, `ffn-throughput`, `ffn-bottleneck`, `ffn-overlap`, `attn-bottleneck`, `kg-bench`, `projection-test`, `bottleneck-test`, `embedding-jump` | Correctness probes and throughput benchmarks used to validate the architecture |
| **Server** | `serve` | HTTP + gRPC vindex server; auth, TLS, rate limiting, CORS |

Each subcommand has `--help`; most also surface as LQL statements through
`larql-lql`, so the REPL and the CLI share the same semantics.

## Layout

- `src/main.rs` — clap dispatch + legacy argv trampoline (`larql walk` → `larql dev walk`)
- `src/commands/primary/` — primary user verbs: `run`, `chat`, `pull`, `model`, `link`, `list`, `show`, `slice`, `publish`, `rm`, `bench`, `shannon`, `serve` glue
- `src/commands/extraction/` — vindex build/extract/compile/convert/verify
- `src/commands/diagnostics/` — `parity` (cross-backend numerical diff)
- `src/commands/query/` — legacy graph-file query surface (`query`, `describe`, `stats`, `validate`, `merge`, `filter`)
- `src/commands/dev/` — research / interpretability tools surfaced under `larql dev <subcmd>`

The CLI has no feature flags of its own — Metal, CUDA, and BLAS variants
are selected through the upstream `larql-compute` / `larql-inference`
features on the workspace build.
