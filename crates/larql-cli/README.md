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
```

See [`docs/cli.md`](../../docs/cli.md) for the full command reference.

## Command families

| Family | Commands | What they do |
|---|---|---|
| **Vindex lifecycle** | `extract-index`, `build`, `compile`, `convert`, `verify`, `hf` | Extract, build from a Vindexfile, bake patches into weights, convert GGUF↔vindex↔safetensors, checksum, push/pull HuggingFace Hub |
| **LQL** | `repl`, `lql`, `query`, `describe`, `filter`, `merge`, `validate`, `stats` | Interactive REPL + one-shot LQL, plus lower-level graph utilities |
| **Weight-space extraction** | `weight-extract`, `attention-extract`, `vector-extract`, `index-gates`, `qk-templates`, `qk-rank`, `qk-modes`, `ov-gate`, `circuit-discover`, `fingerprint-extract` | Pull edges / templates / circuits from the model weights — zero forward passes |
| **Forward-pass analysis** | `predict`, `walk`, `residuals`, `attention-capture`, `extract-routes`, `trajectory-trace`, `bfs` | Run the model and capture residuals, attention patterns, trajectories |
| **Benchmarks & tests** | `ffn-bench`, `ffn-throughput`, `ffn-bottleneck`, `ffn-overlap`, `attn-bottleneck`, `kg-bench`, `projection-test`, `bottleneck-test`, `embedding-jump` | Correctness probes and throughput benchmarks used to validate the architecture |
| **Server** | `serve` | HTTP + gRPC vindex server; auth, TLS, rate limiting, CORS |

Each subcommand has `--help`; most also surface as LQL statements through
`larql-lql`, so the REPL and the CLI share the same semantics.

## Layout

- `src/main.rs` — clap dispatch
- `src/commands/extraction/` — extraction + analysis subcommands (most of the binary)
- `src/commands/query/` — graph query subcommands (`query`, `describe`, `stats`, etc.)

The CLI has no feature flags of its own — Metal, CUDA, and BLAS variants
are selected through the upstream `larql-compute` / `larql-inference`
features on the workspace build.
