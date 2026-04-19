# LARQL CLI Reference

```
larql <COMMAND> [OPTIONS]
```

## Primary commands

Ollama-style day-to-day verbs. Models can be referenced by cache
shorthand (`gemma-3-4b-it-vindex`), `owner/name`, `hf://owner/name`, or
a local directory path — see [Model resolution](#model-resolution) below.

| Command | Description |
|---|---|
| `run <model> [prompt]` | Run inference. One-shot if prompt given; chat loop if not. |
| `chat <model>` | Alias for `run <model>` with no prompt. |
| `pull <model>` | Download a vindex from HuggingFace and cache locally. |
| `list` | Show cached vindexes (model, size, layers, hidden). |
| `show <model>` | Vindex metadata and file inventory. |
| `rm <model>` | Evict a cached vindex. |
| `serve <model>` | Serve a vindex over HTTP + gRPC. |

## Build / extract

| Command | Description |
|---|---|
| `extract <model-id>` | Build a .vindex from a HuggingFace model (safetensors/GGUF/MLX → queryable). |
| `extract-index` | Backwards-compat alias of `extract`. |
| `build` | Build a custom vindex from a Vindexfile (FROM + PATCH + INSERT). |
| `compile` | Compile vindex patches into model weights (AOT). |
| `convert` | Convert between formats (GGUF ↔ vindex, safetensors → vindex). |
| `hf` | HuggingFace Hub: publish a vindex. |
| `verify` | Verify vindex file integrity (SHA256 checksums). |

## LQL

| Command | Description |
|---|---|
| `repl` | Launch the LQL interactive REPL. |
| `lql '<stmt>'` | Execute a one-shot LQL statement. |

## Research / interpretability tools — `larql dev <subcmd>`

All extraction / probing / benchmark tooling lives under `larql dev`.
The pre-redesign top-level invocations (`larql walk …`,
`larql weight-extract …`, etc.) are rewritten to `larql dev <name>`
transparently by an argv trampoline, so existing scripts continue to
work.

```
larql dev --help
larql dev walk --index X.vindex --prompt "..." --predict
```

See [Research commands (dev)](#research-commands-dev) below for the full
list.

## Model resolution

The `<model>` argument on `run`, `chat`, `show`, `rm`, and `pull`
resolves in this order:

1. **`hf://owner/name[@rev]`** — download (if not cached) via HF hub API,
   return the cache path.
2. **Existing local directory** — use as-is.
3. **`owner/name`** — cache lookup first; fall back to HF download.
4. **Plain name** — search the cache for a unique
   `datasets--<*>--<name>` entry. Ambiguous shorthands error out and
   list candidates.

`rm` never downloads — it only resolves against the cache.

### `larql run`

One-shot inference or interactive chat.

```
larql run <MODEL> [PROMPT] [OPTIONS]
```

| Flag | Description | Default |
|---|---|---|
| `<MODEL>` | Vindex dir, `hf://owner/name`, `owner/name`, or cache shorthand | — |
| `[PROMPT]` | Prompt text; omit to enter chat mode | — |
| `-n, --top <N>` | Number of predictions to show | 10 |
| `--ffn <URL>` | Route FFN to a remote `larql-server` (`http://host:port`). Attention stays local, each layer's FFN call lands on the server. | — |
| `--ffn-timeout-secs <N>` | HTTP timeout for `--ffn` | 60 |
| `-v, --verbose` | Verbose load / timing output | false |

Examples:

```bash
larql run gemma-3-4b-it-vindex "The capital of France is"
larql run chrishayuk/gemma-3-4b-it-vindex           # chat mode
larql run hf://chrishayuk/gemma-3-4b-it-vindex      # explicit HF
larql run gemma4-31b.vindex --ffn http://server:8080 "…"
```

### `larql chat`

Interactive chat. Alias for `run <model>` with no prompt.

```
larql chat <MODEL> [OPTIONS]
```

Same flag set as `run`, minus the positional prompt.

### `larql pull`

Download a vindex from HuggingFace into the HF hub cache
(`~/.cache/huggingface/hub/`).

```
larql pull <MODEL>
```

Accepts `hf://owner/name`, `owner/name`, or a local path (no-op). Prints
the resolved cache directory and basic metadata.

### `larql list`

Show every cached vindex, one row per entry.

```
larql list
```

Columns: `MODEL`, `SIZE (MB)`, `LAYERS`, `HIDDEN`. Scans the HF hub
cache for `datasets--<owner>--<name>/snapshots/<sha>/index.json`.

### `larql show`

Vindex metadata plus file inventory.

```
larql show <MODEL>
```

Prints layer count, hidden size, dtype, quant format, and each file in
the vindex with size. Resolves the same way as `run`.

### `larql rm`

Remove a cached vindex. Cache-only — never downloads.

```
larql rm <MODEL> [-y]
```

Accepts `owner/name` or cache shorthand. Prompts for confirmation unless
`-y` is passed.

### `larql serve`

Serve a vindex over HTTP. Loads a vindex into memory and exposes REST endpoints for knowledge queries, feature walks, edge selection, and patch management.

```
larql serve <VINDEX_PATH> [OPTIONS]
larql serve --dir <DIR> [OPTIONS]
```

| Flag | Description | Default |
|---|---|---|
| `<VINDEX_PATH>` | Path to .vindex directory or `hf://` URL | — |
| `--dir <DIR>` | Serve all .vindex directories in folder | — |
| `--port <PORT>` | Listen port | 8080 |
| `--host <HOST>` | Bind address | 0.0.0.0 |
| `--no-infer` | Disable inference endpoint (browse-only, saves memory) | false |
| `--ffn-only` | Run as an FFN-service endpoint for `larql run --ffn URL` clients. Implies `--no-infer`; advertises `mode: ffn-service` in `/v1/stats`. | false |
| `--cors` | Enable CORS headers for browser access | false |
| `--api-key <KEY>` | Require Bearer token auth (health exempt) | — |
| `--rate-limit <SPEC>` | Per-IP rate limit (e.g., "100/min", "10/sec") | — |
| `--max-concurrent <N>` | Max concurrent requests | 100 |
| `--cache-ttl <SECS>` | Cache TTL for DESCRIBE results (0 = disabled) | 0 |
| `--grpc-port <PORT>` | Enable gRPC server on this port | — |
| `--tls-cert <PATH>` | TLS certificate for HTTPS | — |
| `--tls-key <PATH>` | TLS private key for HTTPS | — |
| `--log-level <LEVEL>` | Logging level | info |

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| GET | `/v1/describe?entity=France` | Knowledge edges (with probe relation labels) |
| GET | `/v1/walk?prompt=...&top=5` | Feature scan for a prompt |
| POST | `/v1/select` | SQL-style edge query |
| POST | `/v1/infer` | Full forward pass (walk/dense/compare) |
| GET | `/v1/relations` | List known relation types |
| GET | `/v1/stats` | Model and index statistics |
| POST | `/v1/patches/apply` | Apply a patch in-memory |
| GET | `/v1/patches` | List active patches |
| DELETE | `/v1/patches/{name}` | Remove a patch |
| POST | `/v1/walk-ffn` | Decoupled inference. Two modes — see below. |
| WS | `/v1/stream` | WebSocket streaming (layer-by-layer DESCRIBE) |
| GET | `/v1/health` | Health check (auth exempt) |
| GET | `/v1/models` | List loaded models |

**`POST /v1/walk-ffn`** has two modes:

- **Features-only (default).** Client POSTs a `[hidden]` residual; server
  runs gate KNN only and returns feature indices + scores. Client still
  needs `up_features.bin` + `down_features.bin` locally to compute the
  FFN output.
- **Full-output (`"full_output": true`).** Client POSTs a
  `[seq_len × hidden]` row-major residual plus `"seq_len": N`; server
  runs the architecture-correct `WalkFfn` path (gate KNN → activation →
  up gather → down projection) and returns the hidden-size FFN output
  for each requested layer. This is what the `larql run --ffn URL`
  client uses — the server holds all FFN weights, the client holds only
  attention.

Example (full-output):
```bash
curl -X POST http://server:8080/v1/walk-ffn \
  -H 'Content-Type: application/json' \
  -d '{
        "layers": [0, 1, 2],
        "residual": [/* seq_len * hidden floats */],
        "seq_len": 4,
        "full_output": true
      }'
```

Response shape: `{ "results": [{ "layer": N, "output": [...], "seq_len": 4 }, ...] }`.

The gRPC `WalkFfn` RPC mirrors the HTTP endpoint — see `vindex.proto`.

**Multi-model:** When using `--dir`, each model gets its own namespace: `/v1/{model_id}/describe`, etc.

**Examples:**

```bash
# Development — single model
larql serve output/gemma3-4b-v2.vindex --port 8080

# Multi-model server
larql serve --dir ./vindexes/ --port 8080

# From HuggingFace
larql serve "hf://chrishayuk/gemma-3-4b-it-vindex" --port 8080

# Browse-only with CORS for web clients
larql serve output/gemma3-4b-v2.vindex --no-infer --cors --port 8080

# With auth + HTTPS
larql serve output/gemma3-4b.vindex --api-key "sk-abc123" --tls-cert cert.pem --tls-key key.pem

# With rate limiting + DESCRIBE cache
larql serve output/gemma3-4b.vindex --rate-limit "100/min" --cache-ttl 300

# Query from the REPL
larql repl
> USE REMOTE "http://localhost:8080";
> DESCRIBE "France";
> INFER "The capital of France is" TOP 5;
> APPLY PATCH "local-facts.vlp";    -- stays client-side
```

**Sessions:** Clients can send `X-Session-Id` header to isolate patches per session. Without the header, patches apply globally.

See `crates/larql-server/README.md` for full API documentation.

### `larql repl`

Launch the LQL interactive REPL. Tab completion, history, multi-line input.

```bash
larql repl
```

```sql
larql> USE "gemma3-4b.vindex";
larql> DESCRIBE "France";
larql> WALK "The capital of France is" TOP 5;
larql> INSERT INTO EDGES (entity, relation, target) VALUES ("John", "lives-in", "London");
larql> SAVE PATCH "edits.vlp";
```

Direct model access (no extraction needed):
```sql
larql> USE MODEL "google/gemma-3-4b-it";
larql> INFER "The capital of France is" TOP 5;
-- WALK/DESCRIBE/SELECT require EXTRACT into a vindex first
```

Remote mode:
```sql
larql> USE REMOTE "http://localhost:8080";
larql> DESCRIBE "France";
```

### `larql lql`

Execute a single LQL statement from the command line.

```bash
larql lql 'USE "gemma3-4b.vindex"; DESCRIBE "France";'
larql lql 'USE "gemma3-4b.vindex"; WALK "Einstein" TOP 10;'
```

## Research commands (dev)

These commands live under `larql dev <subcmd>`. They predate the REPL
and vindex format and remain available for low-level extraction,
debugging, and interpretability research.

The pre-redesign top-level invocations — `larql walk …`,
`larql weight-extract …`, `larql qk-templates …`, etc. — are still
accepted and rewritten to `larql dev <name>` transparently so existing
scripts keep working. Running any of them with `--help` prints the new
`Usage: larql dev <subcmd> …` form to signal the canonical home.

### `larql dev weight-extract`

Extract edges from FFN weight matrices. Zero forward passes. Pure matrix multiplication.

```
larql dev weight-extract <MODEL> --output <OUTPUT> [OPTIONS]
```

| Argument/Flag | Description |
|---|---|
| `<MODEL>` | Model path or HuggingFace model ID (e.g. `google/gemma-3-4b-it`) |
| `-o, --output <OUTPUT>` | Output file (`.larql.json` or `.larql.bin`) |
| `-l, --layer <LAYER>` | Single layer to walk. Default: all layers |
| `--top-k <TOP_K>` | Top-k tokens per feature [default: 5] |
| `--min-score <MIN_SCORE>` | Minimum raw activation score for top-k selection [default: 0.02] |
| `--min-confidence <MIN_CONFIDENCE>` | Minimum normalized confidence [0-1] to keep an edge [default: 0.0] |
| `--stats <STATS>` | Write layer statistics to a separate JSON file |

**Model resolution:** Accepts a local directory path or a HuggingFace model ID. Model IDs are resolved from the HuggingFace cache at `~/.cache/huggingface/hub/` (or `$HF_HOME/hub/`).

**Resume:** If the output file already exists, completed layers are detected from edge metadata and skipped. Saves after each layer. Safe to interrupt and re-run.

**Confidence scoring:** Each edge gets a normalized confidence `c` in [0, 1], computed as `(c_in × c_out) / max(c_in × c_out)` per layer. Raw scores `c_in` (input selectivity) and `c_out` (output strength) are stored in metadata.

**Examples:**

```bash
# Full extraction
larql dev weight-extract google/gemma-3-4b-it -o knowledge.larql.json

# Single layer test
larql dev weight-extract google/gemma-3-4b-it --layer 26 -o L26.larql.json

# Filtered extraction with stats
larql dev weight-extract google/gemma-3-4b-it \
    -o knowledge.larql.json \
    --min-confidence 0.1 \
    --stats stats.json

# MessagePack output (smaller, faster)
larql dev weight-extract google/gemma-3-4b-it -o knowledge.larql.bin
```

### `larql dev attention-extract`

Extract routing edges from attention OV circuits. Zero forward passes.

```
larql dev attention-extract <MODEL> --output <OUTPUT> [OPTIONS]
```

| Argument/Flag | Description |
|---|---|
| `<MODEL>` | Model path or HuggingFace model ID |
| `-o, --output <OUTPUT>` | Output file (`.larql.json` or `.larql.bin`) |
| `-l, --layer <LAYER>` | Single layer to walk. Default: all layers |
| `--top-k <TOP_K>` | Top-k tokens per head [default: 3] |
| `--min-score <MIN_SCORE>` | Minimum score [default: 0.0] |

**How it works:** For each attention head, computes the OV circuit (`O_h @ V_h`), projects all vocab tokens through it, finds the most amplified inputs, and decodes what output tokens each produces.

**Resume:** Same as `weight-extract` — detects completed layers and skips them.

**Examples:**

```bash
larql dev attention-extract google/gemma-3-4b-it -o attention.larql.json
larql dev attention-extract google/gemma-3-4b-it --layer 12 -o attention-L12.larql.json
```

### `larql dev predict`

Run a full transformer forward pass from extracted safetensors weights and return top-k next-token predictions. Pure Rust inference — no MLX, no PyTorch.

```
larql dev predict <MODEL> --prompt <PROMPT> [OPTIONS]
```

| Argument/Flag | Description |
|---|---|
| `<MODEL>` | Model path or HuggingFace model ID |
| `-p, --prompt <PROMPT>` | Prompt text to predict the next token for |
| `-k, --top-k <TOP_K>` | Number of top predictions to show [default: 10] |

**How it works:** Loads all safetensors weights, tokenizes the prompt (with BOS token), runs the full forward pass through all layers (embedding, attention with RoPE, GQA, QK norm, FFN with SiLU gating, all layer norms), projects the final residual against the embedding matrix, and returns softmax probabilities.

**Architecture awareness:** Uses the `ModelArchitecture` trait to handle model-specific behavior. Gemma 3 gets +1 norm offset, sqrt(hidden) embedding scale, QK normalization, and 4 norms per layer. Llama/generic gets standard behavior.

**Performance:** ~800ms per query for Gemma 3 4B on Apple Silicon (BLAS-accelerated, 34 layers, 6 tokens).

**Examples:**

```bash
# Basic prediction
larql dev predict google/gemma-3-4b-it --prompt "The capital of France is" -k 5
# 1. Paris (99.67%)

# Factual queries
larql dev predict google/gemma-3-4b-it --prompt "The largest planet is" -k 3
# 1. Jupiter (99.86%)

# Works with any HuggingFace model in cache
larql dev predict google/gemma-3-4b-it -p "Water freezes at" -k 10
```

For day-to-day inference against a `.vindex`, use
[`larql run`](#larql-run) — same forward pass, slimmer flag surface,
ollama-style ergonomics.

### `larql dev index-gates`

Build a precomputed gate index for graph-based FFN. Offline step — run once per model. Eliminates the gate matmul at inference time.

```
larql dev index-gates <MODEL> --output <OUTPUT> [OPTIONS]
```

| Flag | Description |
|---|---|
| `<MODEL>` | Model path or HuggingFace model ID |
| `-o, --output <OUTPUT>` | Output index file (`.gate-index.jsonl`) |
| `--features-per-token <N>` | Features to index per token per layer [default: 100] |
| `--top-tokens <N>` | Top tokens to match at runtime [default: 10] |
| `--layers <LAYERS>` | Layers to index (e.g. `0-33` or `26,27,28`). Default: all |

**Examples:**

```bash
larql dev index-gates google/gemma-3-4b-it -o gates.gate-index.jsonl
larql dev index-gates google/gemma-3-4b-it -o gates.gate-index.jsonl --layers 24-33
```

### `larql dev extract-routes`

Extract attention routing patterns from forward passes. Captures which FFN features activate for each entity/relation combination.

```
larql dev extract-routes <MODEL> --output <OUTPUT> [OPTIONS]
```

| Flag | Description |
|---|---|
| `<MODEL>` | Model path or HuggingFace model ID |
| `-o, --output <OUTPUT>` | Output JSON file for the routing table |
| `--top-k <N>` | Top features to capture per layer per forward pass [default: 50] |
| `--min-activation <F>` | Minimum absolute activation to record [default: 1.0] |
| `-e, --entities <ENTITIES>` | Comma-separated entities (overrides defaults) |
| `--layers <LAYERS>` | Comma-separated layers to capture. Default: all |

**Examples:**

```bash
larql dev extract-routes google/gemma-3-4b-it -o routes.json
larql dev extract-routes google/gemma-3-4b-it -o routes.json --entities "France,Germany,Japan" --layers 25,26,27
```

### `larql dev walk`

Walk the model as a local vector index — gate KNN followed by down token lookup. No forward pass needed when using a `.vindex`. This is the research-grade inference path with the full flag surface; for everyday use prefer [`larql run`](#larql-run).

```
larql dev walk --prompt <PROMPT> [OPTIONS]
```

| Flag | Description |
|---|---|
| `-p, --prompt <PROMPT>` | Prompt text to walk through the model |
| `--index <PATH>` | Path to a `.vindex` directory (self-contained, no model needed) |
| `-m, --model <MODEL>` | Model path or HuggingFace model ID |
| `--gate-vectors <PATH>` | Path to extracted ffn_gate vectors (alternative to `--index`) |
| `--down-vectors <PATH>` | Path to extracted ffn_down vectors (alternative to `--index`) |
| `-k, --top-k <N>` | Top-K features per layer for gate KNN [default: 10] |
| `-l, --layers <LAYERS>` | Layers to walk. Comma-separated or range. Default: all |
| `--predict-top-k <N>` | Number of top predictions to show [default: 10] |
| `--predict` | Run full forward pass with walk FFN and show predictions (requires `--model`) |
| `--compare` | Compare walk FFN predictions against dense ground truth (requires `--model`). Incompatible with `--ffn-remote`. |
| `--down-top-k <N>` | Number of down tokens to show per feature [default: 5] |
| `-v, --verbose` | Show verbose loading and timing info |
| `--ffn-remote <URL>` | Route FFN to a remote `larql-server` via `POST /v1/walk-ffn` (`full_output: true`). Attention still runs locally; all layers are sent in a single binary batch round trip (`application/x-larql-ffn`, little-endian f32). Falls back to JSON if the server does not support binary. Same wire protocol that [`larql run --ffn`](#larql-run) uses. |
| `--ffn-remote-timeout-secs <N>` | Per-request HTTP timeout for `--ffn-remote` [default: 60] |

**Examples:**

```bash
# Walk with a pre-built .vindex
larql dev walk --prompt "The capital of France is" --index model.vindex

# Walk with loose vector files
larql dev walk --prompt "The capital of France is" \
    --gate-vectors vectors/ffn_gate.vectors.jsonl \
    --down-vectors vectors/ffn_down.vectors.jsonl

# Walk + compare against ground truth
larql dev walk --prompt "The capital of France is" --index model.vindex --model google/gemma-3-4b-it --compare

# Walk + FFN on a remote server (Act 2 of the demo)
larql dev walk --prompt "The capital of France is" --index client.vindex \
    --model google/gemma-3-4b-it --predict \
    --ffn-remote http://server:8080
```

### `larql dev attention-capture`

Capture and compare attention patterns across multiple prompts. Shows which heads attend similarly or differently.

```
larql dev attention-capture <MODEL> --prompts <PROMPTS> [OPTIONS]
```

| Flag | Description |
|---|---|
| `<MODEL>` | Model path or HuggingFace model ID |
| `-p, --prompts <PROMPTS>` | Prompts to compare (comma-separated) |
| `-l, --layers <LAYERS>` | Layers to capture. Comma-separated or range. Default: all |
| `--threshold <F>` | Attention threshold — only show heads with max attention above this [default: 0.1] |
| `-v, --verbose` | Show verbose per-head details |

**Examples:**

```bash
larql dev attention-capture google/gemma-3-4b-it \
    --prompts "The capital of France is,The capital of Germany is,The capital of Japan is"

larql dev attention-capture google/gemma-3-4b-it \
    --prompts "The capital of France is,The language of France is" \
    --layers 20-33 --threshold 0.2
```

### `larql dev qk-templates`

Extract attention template circuits from QK weight decomposition. Identifies which heads are "fixed" (same pattern regardless of entity) vs "variable".

```
larql dev qk-templates <MODEL> [OPTIONS]
```

| Flag | Description |
|---|---|
| `<MODEL>` | Model path or HuggingFace model ID |
| `-t, --templates <TEMPLATES>` | Template prompts (format: `relation:prompt`, comma-separated). Uses built-in templates if omitted |
| `-l, --layers <LAYERS>` | Layers to analyze. Default: all |
| `--threshold <F>` | Correlation threshold below which a head is "variable" [default: 0.95] |
| `--top-components <N>` | Number of top SVD components to show per head [default: 5] |

**Examples:**

```bash
larql dev qk-templates google/gemma-3-4b-it
larql dev qk-templates google/gemma-3-4b-it --layers 20-33 --threshold 0.90
```

### `larql dev ov-gate`

Map attention OV circuits to FFN gate features. Shows what each attention head activates in the next layer's FFN.

```
larql dev ov-gate <MODEL> [OPTIONS]
```

| Flag | Description |
|---|---|
| `<MODEL>` | Model path or HuggingFace model ID |
| `-l, --layers <LAYERS>` | Layers to analyze. Default: all |
| `-k, --top-k <N>` | Top-K gate features to show per head [default: 10] |
| `--heads <HEADS>` | Only show specific heads (for focused analysis) |
| `-v, --verbose` | Show verbose per-feature details |

**Examples:**

```bash
larql dev ov-gate google/gemma-3-4b-it --layers 25,26,27
larql dev ov-gate google/gemma-3-4b-it --layers 26 --heads 0,1,2 -k 20 -v
```

### `larql dev vector-extract`

Extract full weight vectors to intermediate NDJSON files.

```
larql dev vector-extract <MODEL> --output <OUTPUT> [OPTIONS]
```

| Flag | Description |
|---|---|
| `<MODEL>` | Model path or HuggingFace model ID |
| `-o, --output <OUTPUT>` | Output directory for `.vectors.jsonl` files |
| `--components <COMPONENTS>` | Components to extract (comma-separated): `ffn_down`, `ffn_gate`, `ffn_up`, `attn_ov`, `attn_qk`, `embeddings` |
| `--layers <LAYERS>` | Layers to extract (comma-separated). Default: all |
| `--top-k <TOP_K>` | Top-k tokens for metadata per vector [default: 10] |
| `--resume` | Resume from existing output files |

**Examples:**

```bash
# Extract all components
larql dev vector-extract google/gemma-3-4b-it -o vectors/

# Extract only FFN down projections from layers 25-33
larql dev vector-extract google/gemma-3-4b-it -o vectors/ \
    --components ffn_down --layers 25,26,27,28,29,30,31,32,33
```

### `larql dev residuals capture`

Capture residual stream vectors for entities via forward passes. The residuals are the hidden state at a specific layer — the signal that the next layer's features actually see during inference.

```
larql dev residuals capture <MODEL> --entities <ENTITIES> --output <OUTPUT> [OPTIONS]
```

| Flag | Description |
|---|---|
| `<MODEL>` | Model path or HuggingFace model ID |
| `-e, --entities <ENTITIES>` | Comma-separated entities, or path to a text file (one per line) |
| `-l, --layer <LAYER>` | Layer(s) to capture at. Can specify multiple times. [default: 25] |
| `--all-layers` | Capture at every layer |
| `-o, --output <OUTPUT>` | Output directory for NDJSON files |
| `--template <TEMPLATE>` | Prompt template. `{entity}` is replaced. Default: bare entity name |
| `--activations` | Also capture sparse FFN activations (top-K features per layer) |
| `--activation-top-k <N>` | Number of top features to record per layer [default: 50] |

**How it works:** Tokenizes each entity, runs a full forward pass through the transformer up to the target layer(s), and saves the last-token hidden state as a vector in NDJSON format.

**Use case:** Capture real residual vectors for a small set of entities (50–100). These can be used as query vectors for vindex gate KNN lookups to discover factual edges without additional forward passes.

**Examples:**

```bash
# L25 residuals for seed entities
larql dev residuals capture google/gemma-3-4b-it \
    --entities "France,Germany,Japan,Mozart,Einstein" \
    --layer 25 -o residuals-L25.vectors.ndjson

# Multiple layers
larql dev residuals capture google/gemma-3-4b-it \
    --entities entities.txt \
    --layer 25 --layer 26 --layer 29 \
    -o residuals.vectors.ndjson

# Full trajectory (all layers)
larql dev residuals capture google/gemma-3-4b-it \
    --entities "France" --all-layers \
    -o residuals-full.vectors.ndjson

# With prompt template
larql dev residuals capture google/gemma-3-4b-it \
    --entities "France,Germany" \
    --layer 25 \
    --template "The capital of {entity} is" \
    -o residuals-capital.vectors.ndjson
```

**Output format:** Same NDJSON as `vector-extract`:

```json
{"_header": true, "component": "residuals", "model": "google/gemma-3-4b-it", "dimension": 2560}
{"id": "France_L25", "layer": 25, "feature": 0, "vector": [...], "top_token": "Paris", "c_score": 12.4, ...}
```

### `larql dev bfs`

BFS extraction from a running model endpoint.

```
larql dev bfs --seeds <SEEDS> --templates <TEMPLATES> --output <OUTPUT> [OPTIONS]
```

| Flag | Description |
|---|---|
| `-s, --seeds <SEEDS>` | Comma-separated seed entities |
| `-t, --templates <TEMPLATES>` | Path to templates JSON file |
| `-o, --output <OUTPUT>` | Output file (`.larql.json` or `.larql.bin`) |
| `-e, --endpoint <ENDPOINT>` | Model endpoint URL [default: `http://localhost:11434/v1`] |
| `-m, --model <MODEL>` | Model name for the endpoint |
| `--mock` | Use mock provider instead of HTTP |
| `--mock-knowledge <PATH>` | Path to mock knowledge JSON (with `--mock`) |
| `--max-depth <N>` | Maximum BFS depth [default: 3] |
| `--max-entities <N>` | Maximum entities to probe [default: 1000] |
| `--min-confidence <F>` | Minimum edge confidence [default: 0.3] |
| `--resume <PATH>` | Resume from a checkpoint file |

**Requires:** Templates JSON file defining prompt templates for each relation. See [format.md](format.md) for template format.

**Examples:**

```bash
# Against Ollama
larql dev bfs \
    --seeds "France,Germany,Japan" \
    --templates templates.json \
    --endpoint http://localhost:11434/v1 \
    --model gemma3:4b-it \
    -o knowledge.larql.json

# With mock provider
larql dev bfs \
    --seeds "France,Germany" \
    --templates templates.json \
    --mock --mock-knowledge mock.json \
    -o knowledge.larql.json
```

## Build commands

Top-level verbs for producing, converting, and publishing vindexes.

### `larql extract-index`

Build a `.vindex` — the model decompiled to a standalone vector index.
Can be queried with [`larql run`](#larql-run) or
[`larql dev walk`](#larql-dev-walk) without the original model.

`larql extract` is the canonical form; `larql extract-index` is kept as
a backwards-compatible alias.

```
larql extract-index [MODEL] --output <OUTPUT> [OPTIONS]
```

| Flag | Description | Default |
|---|---|---|
| `<MODEL>` | Model path or HuggingFace model ID (not needed with `--from-vectors`) | — |
| `-o, --output <OUTPUT>` | Output path for the `.vindex` directory | — |
| `--level <LEVEL>` | `browse` / `attention` / `inference` / `all` — strict increasing tiers. See below. | `inference` |
| `--f32` | Opt out of f16 on side-channel tensors. Rarely wanted — doubles file sizes. | off (f16) |
| `--quant <FORMAT>` | Inline-quantise forward-pass weights: `none` or `q4k`. `q4k` emits Q4_K/Q6_K Ollama-compatible blocks; implies `--level all` + f16 side-channels. | `none` |
| `--compact` | Skip `up_weights.bin` + `down_weights.bin`; FFN weights live only in feature-major files. `WalkFfn`-only. | off |
| `--drop-gate-vectors` | Skip `gate_vectors.bin` entirely; loader rebuilds gate from `interleaved_q4k.bin` at load. Only with `--quant q4k`. | off |
| `--down-q4k` | Quantise FFN down-proj as Q4_K instead of Q6_K. Saves ~1.8 GB on 31B, cuts down-matmul cost ~1.5–1.7× at decode. Introduces ~2.5× more probability-redistribution noise (top-1 + top-5 preserved). Validated by `walk_correctness`, which auto-relaxes its prob-delta gate from 0.02 to 0.035 when it detects Q4_K down. Only with `--quant q4k`. | off |
| `--from-vectors <PATH>` | Build from already-extracted NDJSON vector files instead of model weights | — |
| `--down-top-k <N>` | Top-K tokens per feature in down metadata | 10 |
| `--include-weights` | Alias for `--level all` (deprecated — use `--level` directly) | — |
| `--resume` | Skip stages that already have output files | off |

**Extract tiers (`--level`).** Each tier is a strict superset of the
previous:

| Tier | Adds | Enables |
|---|---|---|
| `browse` | gate + embed + down_meta + tokenizer | WALK / DESCRIBE / SELECT (no forward pass) |
| `attention` | + attention + norms | client-side of `run --ffn URL` (Act 2 demo) |
| **`inference` (default)** | + FFN up/down | full local forward pass (INFER) |
| `all` | + lm_head + COMPILE extras | COMPILE |

**Examples:**

```bash
# Default — inference-ready, f16 (~6 GB for 4B)
larql extract google/gemma-3-4b-it -o gemma3-4b.vindex

# Browse-only (no forward pass, ~3 GB)
larql extract google/gemma-3-4b-it -o gemma3-4b.vindex --level browse

# Attention-only slice for Act 2 of the demo — carve out the
# client-side half of an FFN-over-HTTP pair
larql extract google/gemma-4-31b-it -o gemma4-31b.client.vindex --level attention

# All (+ lm_head for COMPILE)
larql extract google/gemma-3-4b-it -o gemma3-4b.vindex --level all

# Q4_K/Q6_K quantised inline — smallest disk footprint, Ollama-compat
larql extract google/gemma-3-4b-it -o gemma3-4b.vindex --quant q4k

# Maximum size reduction on Q4K — drop the redundant f16 gate, rebuild
# from the Q4K at load time
larql extract google/gemma-3-4b-it -o gemma3-4b.vindex \
  --quant q4k --drop-gate-vectors

# All-Q4K FFN — gate + up + down all Q4_K, faster down projection at
# decode (~1.5–1.7× on CPU), ~30 MB/layer smaller. Trades ~2.5× more
# softmax-probability drift; validate with `walk_correctness`.
larql extract google/gemma-4-31b-it -o gemma4-31b.vindex \
  --quant q4k --down-q4k

# Legacy name still works
larql extract-index google/gemma-3-4b-it -o gemma3-4b.vindex

# Build from pre-extracted vectors
larql extract -o gemma3-4b.vindex --from-vectors vectors/

# Resume an interrupted build
larql extract google/gemma-3-4b-it -o gemma3-4b.vindex --resume
```

**`--quant q4k` details:**

- Q/K/O + gate/up: Q4_K (148 bytes per 256 values)
- V + down: Q6_K (210 bytes per 256 values), or Q4_K with `--down-q4k`
- `--level browse` is implicitly promoted to `--level all` — the Q4_K
  writer materialises all of attention, FFN, norms, and `lm_head` in
  one pass, so a browse-only Q4_K vindex would be incoherent.
- Side-channel tensors that Q4_K doesn't quantise — `gate_vectors.bin`
  and `embeddings.bin` — are stored at f16 by default under `--quant q4k`.
  Leaving them at f32 pairs 4-bit main weights with 32-bit lookup
  tables, roughly doubling the vindex footprint for no accuracy gain.
- V-shares-K fallback: on Gemma 4 31B global layers where `v_proj` is
  absent, K's bytes are stored in V's slot (still tagged `Q6_K`, still
  keyed by the V tensor name) so downstream 4-per-layer indexing
  stays valid.
- `VindexConfig.quant = Q4k` is written to `index.json`; loaders
  should dispatch on this field rather than sniffing filenames.

### `larql build`

Build a custom model from a Vindexfile (declarative: FROM + PATCH + INSERT).

```
larql build [DIR] [OPTIONS]
```

| Flag | Description |
|---|---|
| `[DIR]` | Directory containing the Vindexfile [default: `.`] |
| `--stage <NAME>` | Build stage (e.g. `dev`, `prod`, `edge`) |
| `-o, --output <PATH>` | Output directory for the built vindex [default: `./build/vindex/`] |
| `--compile <FORMAT>` | Compile output to a model format after building |

**Examples:**

```bash
# Build from Vindexfile in current directory
larql build .

# Build a specific stage
larql build . --stage prod

# Build to a custom output path
larql build . --output custom.vindex
```

### `larql convert`

Convert between model formats.

```
larql convert <SUBCOMMAND>
```

| Subcommand | Description |
|---|---|
| `gguf-to-vindex` | Convert a GGUF model to a vindex (dequantized to f32) |
| `safetensors-to-vindex` | Convert safetensors model to a vindex |
| `gguf-info` | Show GGUF file metadata and detected architecture |

**Examples:**

```bash
# Convert GGUF to vindex
larql convert gguf-to-vindex model-Q4_K_M.gguf -o model.vindex --f16

# Show GGUF metadata
larql convert gguf-info model-Q4_K_M.gguf

# Convert safetensors to vindex
larql convert safetensors-to-vindex ./model/ -o model.vindex --level inference --f16
```

Supported GGUF quantization types for reading: F32, F16, BF16, Q4_0, Q4_1, Q8_0. All tensors are dequantized to f32 during conversion.

### `larql hf`

HuggingFace Hub: download or publish vindexes.

```
larql hf <SUBCOMMAND>
```

| Subcommand | Description |
|---|---|
| `download` | Download a vindex from HuggingFace to local cache |
| `publish` | Upload a local vindex to HuggingFace |

**Examples:**

```bash
# Download a vindex
larql hf download chrishayuk/gemma-3-4b-it-vindex

# Download to a specific directory
larql hf download chrishayuk/gemma-3-4b-it-vindex -o ./gemma3-4b.vindex

# Download a specific version
larql hf download chrishayuk/gemma-3-4b-it-vindex --revision v2.0

# Publish a vindex
larql hf publish ./gemma3-4b.vindex --repo chrishayuk/gemma-3-4b-it-vindex
```

HuggingFace vindexes can also be used directly from the REPL:

```sql
larql> USE "hf://chrishayuk/gemma-3-4b-it-vindex";
larql> DESCRIBE "France";
```

Publishing requires `HF_TOKEN` environment variable or `huggingface-cli login`.

### `larql verify`

Verify vindex file integrity against stored SHA256 checksums.

```
larql verify <VINDEX>
```

**Example:**

```bash
larql verify gemma3-4b.vindex
# gate_vectors.bin ... OK (1.66 GB)
# embeddings.bin ... OK (1.25 GB)
# down_meta.bin ... OK (2.0 MB)
# All 3 files verified.
```

## Graph-file commands

These operate on the NDJSON/MessagePack knowledge-graph files produced
by `larql dev weight-extract` / `larql dev bfs` — the pre-vindex output
format. For a vindex, use LQL (`DESCRIBE` / `SELECT` / `STATS` in the
REPL) instead.

### `larql query`

Select edges from a subject, optionally filtered by relation.

```
larql query --graph <GRAPH> <SUBJECT> [RELATION]
```

```bash
larql query --graph knowledge.larql.json France
larql query --graph knowledge.larql.json France capital-of
```

### `larql describe`

Show all outgoing and incoming edges for an entity.

```
larql describe --graph <GRAPH> <ENTITY>
```

```bash
larql describe --graph knowledge.larql.json France
```

### `larql stats`

Show graph statistics: entity count, edge count, relation count, connected components, average degree, average confidence, source distribution.

```
larql stats <GRAPH>
```

```bash
larql stats knowledge.larql.json
```

### `larql validate`

Check a graph file for issues: zero-confidence edges, self-loops, empty subjects/objects.

```
larql validate <GRAPH>
```

```bash
larql validate knowledge.larql.json
```

### `larql merge`

Merge multiple graph files into one.

```
larql merge <INPUT>... --output <OUTPUT> [OPTIONS]
```

| Flag | Description |
|---|---|
| `<INPUT>...` | Input graph files to merge (at least 2) |
| `-o, --output <OUTPUT>` | Output merged graph file |
| `--strategy <STRATEGY>` | Merge strategy: `union`, `max_confidence`, `source_priority` [default: `union`] |

**Examples:**

```bash
larql merge weights.larql.json attention.larql.json -o combined.larql.json
larql merge weights.larql.json bfs.larql.json -o combined.larql.json --strategy max_confidence
```

## Templates format

Used by `larql bfs`. A JSON array of prompt templates:

```json
[
  {
    "relation": "capital-of",
    "template": "The capital of {subject} is",
    "multi_token": true,
    "stop_tokens": [".", "\n", ",", ";"]
  }
]
```

| Field | Type | Description |
|---|---|---|
| `relation` | string | Relation name for edges produced by this template |
| `template` | string | Prompt text. `{subject}` is replaced with the entity name |
| `multi_token` | bool | Chain multiple forward passes for multi-token answers |
| `reverse_template` | string? | Optional reverse probe (`{object}` placeholder) |
| `stop_tokens` | char[] | Characters that terminate multi-token chaining |

## Mock knowledge format

Used by `larql bfs --mock`. A JSON array:

```json
[
  {"prompt": "The capital of France is", "answer": "Paris", "probability": 0.89}
]
```

## Distribution: `slice`, `publish`, `pull`

The three commands form the distribution pipeline: extract once, carve
the deployment variants you need, push them to HuggingFace as sibling
repos under a shared collection, pull them on the consumer side with
progress + resume.

Architectural rationale (why each slice exists, how collections get
composed, SHA256 skip semantics, empty-gate loader for attention-only
clients) is in
[`docs/adr/0007-vindex-distribution.md`](adr/0007-vindex-distribution.md).
Command reference below.

### `slice`

Carve a built vindex into deployment variants. Pure file I/O + `index.json`
rewrite — no re-extract.

| Flag | Description | Default |
|---|---|---|
| `<SRC>` | Source vindex: directory, `hf://owner/name`, cache shorthand | — |
| `-o, --output <DST>` | Destination directory. Must not exist unless `--force`. | — |
| `--preset <NAME>` | `client`, `server`, `browse`, `router`, `all` | — |
| `--parts <list>` | Explicit parts (embed, norms, attn, gate, down_meta, ffn, lm_head, router, tokenizer, manifest, labels, readme). `index.json` is always copied. | — |
| `--force` | Overwrite `<DST>` if it exists | false |
| `--dry-run` | Preview what would be copied | false |

Slice sizes on Gemma 4 31B Q4_K: `client` 7.4 GB, `server` 27 GB,
`browse` 16 GB, full 32 GB.

### `publish`

Upload the full vindex plus sibling slices to HuggingFace and file them
into three nested collections.

| Flag | Description | Default |
|---|---|---|
| `<SRC>` | Source vindex | — |
| `--repo <OWNER/NAME>` | HF repo ID for the full vindex. Siblings derive from `--slice-repo-template`. | required |
| `--full` / `--no-full` | Upload the full vindex | `--full` |
| `--slices <list>` | Presets to upload alongside. `none` to skip. | `client,server,browse` |
| `--slice-repo-template <T>` | `{repo}` → `--repo`, `{preset}` → preset. | `{repo}-{preset}` |
| `--collections <list>` | `model`, `family`, `library`. `none` to skip. | `model,family,library` |
| `--model-title <T>` | Override per-model collection title | derived |
| `--family <NAME>` | Override family collection group | derived |
| `--library-title <T>` | Override library collection title | `LARQL Vindex Library` |
| `--force-upload` | Re-upload every file; ignore SHA256 skip | false |
| `--tmp-dir <DIR>` | Staging directory for slice carving | system temp |
| `--dry-run` | Preview, no HF writes | false |

**Skip-if-unchanged** (default on): before each upload the client fetches
the repo's LFS file index and compares `lfs.oid` against the local
SHA256. Matches skip. Small json files always re-upload. Re-publishing
a 27 GB server slice where nothing changed transfers only a few KB.

**Streaming + progress**: uploads stream the file via a counting
`Read` adapter that ticks a per-file indicatif bar as bytes flow out.
No whole-file-into-RAM pre-read. An interrupted `publish` restarts via
the SHA256 skip on completed files; the interrupted file re-uploads.

### `pull`

Download a vindex (or a slice, or a whole collection) with per-file
progress bars. hf-hub handles `.incomplete` partial-file resume
internally.

| Flag | Description | Default |
|---|---|---|
| `<MODEL>` | `hf://owner/name[@rev]`, `owner/name`, or local path. Omit with `--collection`. | — |
| `--preset <NAME>` | Pull `{repo}-{preset}` instead of the named repo. | — |
| `--all-slices` | Full + every default sibling (`-client`, `-server`, `-browse`). Missing siblings warn, don't fail. | false |
| `--collection <SLUG\|URL>` | Pull every dataset in an HF collection. | — |
| `--sibling-template <T>` | Must match `publish --slice-repo-template`. | `{repo}-{preset}` |

After a plain `pull <repo>`, `larql` HEAD-probes for standard siblings
and prints an "also available" hint if any exist — so the sliced layout
is self-announcing from a single repo URL.

```bash
larql pull chrishayuk/gemma-4-31b-it-vindex
# → progress bars per file, then:
#   Also available on HuggingFace:
#     --preset client   → hf://chrishayuk/gemma-4-31b-it-vindex-client
#     --preset server   → hf://chrishayuk/gemma-4-31b-it-vindex-server
#     --preset browse   → hf://chrishayuk/gemma-4-31b-it-vindex-browse
#   Use `larql pull <repo> --all-slices` to grab them all.
```

Requires `HF_TOKEN` or `~/.huggingface/token` only for private repos
and collections; public pulls work unauthenticated.
