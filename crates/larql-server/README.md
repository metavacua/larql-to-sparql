# larql-server

HTTP / gRPC / Unix-socket server for vindex knowledge queries and inference,
plus the per-expert backend for distributed MoE generation. Loads a vindex
and serves it over the network. No GPU, no ML framework, no Python. One
binary.

```bash
larql-server output/gemma3-4b-v2.vindex --port 8080
# Serving google/gemma-3-4b-it (348K features, 1967 probe-confirmed)
# Listening: http://0.0.0.0:8080
```

```bash
curl "http://localhost:8080/v1/describe?entity=France"
# {"entity":"France","edges":[{"relation":"capital","target":"Paris","gate_score":1436.9,"layer":27,"source":"probe"}, ...]}
```

For Gemma 4 26B-A4B and other hybrid-MoE models, this server is also the
**remote expert** that the inference client calls per layer. End-to-end
~18.3 tok/s on M3 Max with one local gRPC shard, ~17.3 tok/s with two local
shards (see `Remote MoE shard topology` below for setup, and `ROADMAP.md
→ F-FLY` for multi-host deployment).

The collect + fire halves of the gRPC dispatch are now both parallel across
shards (`std::thread::scope` + `rayon::par_iter`, 2026-05-02) — see
`ROADMAP.md → F-COLLECT`. On loopback the win is below noise (single
machine, P-core saturation), but at multi-host LAN/cross-region scale this
becomes the load-bearing primitive: parallel collect turns
`collect ≈ N × RTT × layers` into `collect ≈ max(RTT) × layers`.

## What this is

larql-server is the production face of the LARQL research thesis: that
transformer FFN layers are compilable knowledge databases, that training is
slow compilation, and that inference should be restructured around graph
walks rather than monolithic matrix multiplication. As new LARQL paradigms
become real, this is where they become network-addressable APIs.

That gives the roadmap two tracks:

- **Parity** — the server features any 2026 developer expects: OpenAI-compat
  endpoints, stateful sessions, streaming, structured output, LoRA
  hot-loading, prefix-caching for chat. Parity work is *defensive*: it
  removes reasons-to-leave so the paradigm is reachable from the existing
  ecosystem (Cursor, Continue, LangChain, OpenAI SDK, eval harnesses) without
  asking anyone to adopt a weird API first.
- **Paradigm** — capabilities that are unique to this substrate:
  DESCRIBE / WALK / SELECT over the indexed knowledge graph, patch overlays
  that edit model behaviour without retraining, residual-addressed FFN
  execution, remote MoE expert shards as routable compute assets, and
  federated knowledge graphs across multiple vindexes. Paradigm work is
  *offensive*: it's the reason to stay once parity gets you in the door.

Parity work is in service of paradigm work, not in competition with vLLM.
The bar for parity is "what someone expects when they plug in their existing
OpenAI client", not "every GPU-cluster optimisation vLLM ships". Once that
bar is cleared, the question shifts from "why use larql instead of X" to
"why *wouldn't* I use larql, given it does what X does *and* exposes the
model as a queryable knowledge graph I can edit at runtime".

> **For the framing in one place:** see [`THESIS.md`](./THESIS.md) for
> why this is built as a *reference implementation* and what success
> looks like (citations and pattern diffusion, not GitHub stars).

## Features

- **OpenAI-compatible API** — `GET /v1/models`, `POST /v1/embeddings` (with `encoding_format: "base64"`), `POST /v1/completions`, `POST /v1/chat/completions` with SSE streaming, structured outputs (`response_format: json_object | json_schema`), function calling (`tools` + `tool_choice`), tool-result replay (`role: "tool"`), repetition penalties (`frequency_penalty` / `presence_penalty`), and top-k logprobs all live. Existing `openai` Python/JS SDKs work unmodified — chat templates auto-detected from the model family (Gemma / Llama / ChatML / Mistral / plain)
- **Browse endpoints** — DESCRIBE, WALK, SELECT, RELATIONS, STATS (no weights needed)
- **Inference** — full forward pass with WalkFfn (weights lazy-loaded on first request)
- **Remote MoE expert** — `/v1/experts/layer-batch` (residual once + K experts), gRPC streaming with overlap, f16 wire opt-in, UDS transport for same-host shards
- **Relation labels** — probe-confirmed labels from `feature_labels.json` in DESCRIBE responses
- **Patch overlay** — apply knowledge patches via API without modifying base files
- **Multi-model serving** — serve multiple vindexes from a directory
- **HuggingFace support** — load vindexes directly from `hf://` paths
- **API key auth** — optional Bearer token authentication
- **TLS** — native HTTPS via rustls
- **Concurrency limit** — configurable max concurrent requests
- **CORS** — enable for browser-based clients
- **REPL integration** — `USE REMOTE "http://..."` in the LQL REPL

## Quickstart

```bash
# Build
cargo build --release -p larql-server

# Serve a local vindex
larql-server output/gemma3-4b-v2.vindex --port 8080

# Or via the CLI wrapper
larql serve output/gemma3-4b-v2.vindex --port 8080

# From HuggingFace
larql serve "hf://chrishayuk/gemma-3-4b-it-vindex" --port 8080

# Multi-model
larql serve --dir ./vindexes/ --port 8080

# With auth + TLS
larql serve output/gemma3-4b-v2.vindex --api-key "sk-abc123" --tls-cert cert.pem --tls-key key.pem
```

### Quickstart with the OpenAI SDK

larql-server speaks the OpenAI API. Point any existing `openai`
Python or JS client at the larql `base_url` and it works unmodified.
The full surface — `/v1/models`, `/v1/embeddings` (`encoding_format:
"base64"`), `/v1/completions`, `/v1/chat/completions` with SSE
streaming, structured outputs (`response_format: json_object` /
`json_schema`), function calling (`tools` + `tool_choice`),
multi-turn tool-result replay, repetition penalties, and top-k
logprobs — is live. Chat completions auto-detect the chat template
from the model family (Gemma / Llama / ChatML / Mistral / plain).

**Python:**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-anything",  # SDK requires non-empty; matched against --api-key if set
)

# /v1/models
for m in client.models.list().data:
    print(m.id, m.owned_by)

# /v1/embeddings (single + batched)
emb = client.embeddings.create(model="gemma-3-4b", input="France")
batch = client.embeddings.create(
    model="gemma-3-4b",
    input=["France", "Germany", "Japan"],
)

# /v1/completions
resp = client.completions.create(
    model="gemma-3-4b",
    prompt="The capital of France is",
    max_tokens=10,
    temperature=0.0,
)
print(resp.choices[0].text)

# /v1/chat/completions
chat = client.chat.completions.create(
    model="gemma-3-4b",
    messages=[
        {"role": "system", "content": "You are concise."},
        {"role": "user",   "content": "What is the capital of France?"},
    ],
    max_tokens=10,
)
print(chat.choices[0].message.content)

# Embeddings as base64 (~33% smaller wire)
emb_b64 = client.embeddings.create(
    model="gemma-3-4b",
    input="France",
    encoding_format="base64",
)

# Structured outputs — strict JSON Schema
person = client.chat.completions.create(
    model="gemma-3-4b",
    messages=[{"role": "user", "content": "Describe Alice, age 30, who is admin."}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "Person",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age":  {"type": "integer"},
                    "role": {"type": "string", "enum": ["user", "admin", "guest"]},
                },
                "required": ["name", "age", "role"],
            },
        },
    },
)
import json
data = json.loads(person.choices[0].message.content)  # guaranteed to match schema

# Function calling
weather = client.chat.completions.create(
    model="gemma-3-4b",
    messages=[{"role": "user", "content": "Weather in Tokyo?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }],
)
call = weather.choices[0].message.tool_calls[0]
# call.function.name, call.function.arguments  ('{"location":"Tokyo"}')

# Multi-turn tool-result replay: feed the call + the tool's result back in
chat2 = client.chat.completions.create(
    model="gemma-3-4b",
    messages=[
        {"role": "user", "content": "Weather in Tokyo?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": call.id, "type": "function",
             "function": {"name": call.function.name, "arguments": call.function.arguments}}
        ]},
        {"role": "tool", "tool_call_id": call.id, "content": "21 C, sunny"},
    ],
    max_tokens=32,
)

# Sampling + repetition penalties + logprobs
sampled = client.chat.completions.create(
    model="gemma-3-4b",
    messages=[{"role": "user", "content": "Once upon a time"}],
    max_tokens=20,
    temperature=0.8,
    top_p=0.9,
    seed=42,
    frequency_penalty=0.5,  # subtract freq * count(token) from each logit
    presence_penalty=0.3,   # subtract presence for any token already seen
    logprobs=True,
    top_logprobs=3,
)
# sampled.choices[0].logprobs.content[i].{token, logprob, top_logprobs}
```

#### Structured outputs and tool calling

Constrained decoding is built on a **schema-typed JSON FSM** that
masks the LM head per token. The same engine drives all three modes:

| Request                                    | Schema the FSM enforces                                       |
|--------------------------------------------|---------------------------------------------------------------|
| `response_format: {type: "json_object"}`   | any structurally-valid JSON object                            |
| `response_format: {type: "json_schema"}`   | `json_schema.schema` parsed to AST (strict mode supported)    |
| `tools: [...]`, `tool_choice: "auto"`      | discriminated `OneOf` of `{name=Const, arguments=<args>}`     |
| `tool_choice: {type:"function", function:{name}}` | single-tool branch from the union                       |

Schema parser supports `type` (incl. `["string","null"]`), `properties`,
`required`, `additionalProperties`, `items`, `minItems`/`maxItems`,
`enum`, `const`, `oneOf` / `anyOf`, `minLength` / `maxLength`,
`minimum` / `maximum`, plus integer-vs-number. `$ref`, `pattern`,
`format`, `allOf`, `not` return 400 with a clear message — no silent
relaxation. Sampling fields are honoured under the mask
(`temperature`, `top_p`, `seed`, `frequency_penalty`,
`presence_penalty`); pass `temperature: 0` (default) for deterministic
output. Tools + `stream=true` emits the tool call as a single delta
chunk followed by `finish_reason: "tool_calls"` (per-token argument
streaming is a future tightening).

**JS:**

```js
import OpenAI from "openai";
const client = new OpenAI({
  baseURL: "http://localhost:8080/v1",
  apiKey: "sk-anything",
});
const models = await client.models.list();
const emb    = await client.embeddings.create({ model: "gemma-3-4b", input: "France" });
const resp   = await client.completions.create({
  model: "gemma-3-4b",
  prompt: "The capital of France is",
  max_tokens: 10,
});
const chat = await client.chat.completions.create({
  model: "gemma-3-4b",
  messages: [
    { role: "system", content: "You are concise." },
    { role: "user",   content: "Capital of France?" },
  ],
  max_tokens: 10,
});
```

**curl:**

```bash
curl http://localhost:8080/v1/models

curl -X POST http://localhost:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model": "gemma-3-4b", "input": "France"}'

curl -X POST http://localhost:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma-3-4b",
    "prompt": "The capital of France is",
    "max_tokens": 5
  }'

curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma-3-4b",
    "messages": [
      {"role": "system", "content": "You are concise."},
      {"role": "user",   "content": "Capital of France?"}
    ],
    "max_tokens": 5
  }'
```

For an end-to-end live walkthrough that boots an in-process server
and exercises every endpoint with a real vindex:

```bash
# f16 vindex (fastest, KV-cached attention):
cargo run --release -p larql-server --example openai_demo -- \
  output/gemma3-4b-f16.vindex

# Q4_K vindex (also produces real output; per-step Q4_K decode is
# O(N²) so high `max_tokens` runs are slow on CPU):
cargo run --release -p larql-server --example openai_demo -- \
  output/gemma3-4b-q4k-streaming.vindex
```

Both produce intelligible output ("The capital of France is" → "
Paris.") — generation runs through `larql_inference::layer_graph::generate`
which auto-dispatches to the KV-cached f16 path or the per-step Q4_K
CPU path based on the loaded vindex format.

## CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `<VINDEX_PATH>` | Path to .vindex directory or `hf://` URL | — |
| `--dir <DIR>` | Serve all .vindex directories in folder | — |
| `--port <PORT>` | Listen port | 8080 |
| `--host <HOST>` | Bind address | 0.0.0.0 |
| `--no-infer` | Disable `/v1/infer` (browse-only, saves no memory directly — `walk-ffn` still loads weights lazily; pair with `--warmup-walk-ffn` to pay that cost at boot). | false |
| `--ffn-only` | Run as an FFN-service endpoint for `RemoteWalkBackend` clients. Skips the f16→f32 gate warmup (10× smaller startup RSS on 31B Q4_K) | false |
| `--embed-only` | Run as an embed-service endpoint (ADR-0008). Loads only embeddings + lm_head + tokenizer; skips all FFN and attention weights. Enables `/v1/embed`, `/v1/logits`, `/v1/token/*`. Advertises `mode: embed-service`. | false |
| `--layers <START-END>` | Serve only this layer range (inclusive). Out-of-range requests return HTTP 400. Pages outside the range are never touched. | all |
| `--experts <START-END>` | (MoE) Serve only this expert ID range (inclusive). Used to shard the expert bank across machines: `larql-server <vindex> --experts 0-63` on host A, `--experts 64-127` on host B. Requests for out-of-range expert IDs are rejected with HTTP 400. The remote-MoE inference client (`RemoteMoeBackend` in larql-inference) handles per-expert routing across shards. See "Remote MoE shard topology" below. | all |
| `--units <PATH>` | (MoE, fine-grained alternative to `--experts`) JSON manifest specifying per-`(layer, expert)` ownership for non-uniform shard layouts (e.g., layer-0 split into 4 shards but layer-29 into 2). Format: `{"layer_experts": {"0": [[0,31]], "1": [[0,15],[64,79]], ...}}`. Mutually exclusive with `--experts`. | — |
| `--uds-path <PATH>` | Bind a Unix domain socket alongside the TCP listener for same-host MoE shard clients. Skips the kernel TCP stack, ~50 µs/call faster on loopback (~3% end-to-end). Pre-existing socket files are unlinked at startup. Clients reach the shard via a `unix:///path/to/sock` URL in `--moe-shards`. | — |
| `--max-gate-cache-layers <N>` | LRU cap on decoded f16 gate layers. `0` = unlimited. Each decoded layer is ~433 MB on 31B. | 0 |
| `--max-q4k-cache-layers <N>` | LRU cap on the legacy `q4k_ffn_layer` whole-layer dequant cache. `0` = unlimited. Recommended `1` (or 0 once the vindex has W2 feature-major down — see `--feature-major-down` at extract time). | 0 |
| `--hnsw` | Use HNSW for gate KNN instead of brute-force matmul. Approximate (recall 80–95%); wins for high-feature MoE (e.g. 64-expert: ~230 → 60 ms/layer). Net loss for dense ≤ 10K-feature models — leave off. | false |
| `--hnsw-ef-search <N>` | HNSW beam width. Higher = better recall, slower search. | 200 |
| `--warmup-hnsw` | Eager-build HNSW for every owned layer at boot (rayon-parallel). Trades ~700 ms of boot for 76 ms × N lazy first-query cost. Requires `--hnsw`. | false |
| `--warmup-walk-ffn` | Pre-load inference weights + prefetch all owned-layer Q4K mmap pages at boot. Cuts first `/v1/walk-ffn` from ~1.3 s to ~13 ms. Costs ~1.3 s boot delay + 3 GB pre-allocated f32 gate cache. Recommended for grid shards under steady-state load. | false |
| `--release-mmap-after-request` | `madvise(MADV_DONTNEED)` on all mmaps after each walk-ffn request. Linux: immediate RSS drop. Darwin: advisory. | false |
| `--join <URL>` | Join a router grid via gRPC (see `larql-router --grid-port`). Comma-separate multiple routers; each gets an independent announce stream. Pair with `--public-url` so the router knows where to send clients. | — |
| `--grid-key <KEY>` | Shared secret matching the router's `--grid-key`. Required when the router enforces grid auth. Reads `LARQL_GRID_KEY` env. | — |
| `--public-url <URL>` | HTTP URL clients should use to reach this server, advertised when joining the grid (e.g. `http://shard-a:9181`). Required with `--join`. | — |
| `--cors` | Enable CORS headers | false |
| `--api-key <KEY>` | Require Bearer token auth (health exempt) | — |
| `--rate-limit <SPEC>` | Per-IP rate limit (e.g., "100/min", "10/sec") | — |
| `--trust-forwarded-for` | Use the first `X-Forwarded-For` IP for rate limiting. Enable only behind a trusted reverse proxy. | false |
| `--max-concurrent <N>` | Max concurrent requests | 100 |
| `--cache-ttl <SECS>` | Cache TTL for DESCRIBE results (0 = disabled) | 0 |
| `--grpc-port <PORT>` | Enable gRPC server on this port (separate from the router-announce gRPC) | — |
| `--tls-cert <PATH>` | TLS certificate for HTTPS | — |
| `--tls-key <PATH>` | TLS private key for HTTPS | — |
| `--log-level <LEVEL>` | Logging level | info |

### Environment variables

The server and inference client share a small set of env-var knobs for
tuning the MoE remote-expert path. Most have data-driven defaults from
the 2026-05-01 perf session — see `ROADMAP.md` for measurement context.

| Var | Effect | Default |
|---|---|---|
| `LARQL_MOE_NO_SPLIT=1` | Disable the gRPC streaming overlap (fire MoE → encode dense FFN → collect). Default-on (overlap) is reliably ~12% faster steady-state on M3 Max loopback; opt out only if a new hardware/driver combo regresses. | overlap on |
| `LARQL_MOE_WIRE_F16=1` | Use the `/v1/experts/layer-batch-f16` endpoint and ship residual + response as f16 (5.5 KB vs 11 KB). Loopback: within noise. LAN (1 Gbps): expected +3-5%. | f32 |
| `LARQL_MOE_TIMING=1` | Per-token MoE timing summary on stderr: route+fire / collect / server compute estimate / network estimate per layer + per-token totals. | off |
| `LARQL_HTTP_TIMING=1` | Per-call HTTP/UDS breakdown on stderr: encode / send_total / recv_body / decode µs. Server-side `[handle_layer_batch]` reports decode / spawn_overhead / compute / encode. | off |
| `LARQL_KERNEL_TIMING=1` | Per-expert kernel breakdown on stderr: gate / up / activation / act_q8k / down µs (compute-side). | off |
| `LARQL_MOE_FWD_TIMING=1` | Per-layer `cpu_moe_forward` breakdown: pre_par / q8k_quant / par_iter / sum / post_norm / total µs. | off |
| `LARQL_DISABLE_Q4K_DIRECT=1` | Fall back to BLAS-on-cached-f32 instead of the SDOT direct-Q4K matvec kernel. Kernel-debug A/B only. | direct-Q4K on |
| `LARQL_MOE_CACHE_ENTRIES=N` | Capacity of the f32 dequant cache (per server). Default 256 entries (~6 GB on Gemma 4 26B-A4B Q4_K). Mostly inert when direct-Q4K is on; matters for the BF16 fallback path. | 256 |
| `LARQL_GRID_KEY=<key>` | Same as `--grid-key`. | — |

### Memory bounds — cheat sheet

Measured on Gemma 4 31B Q4_K (macOS, CPU). See ADR-0005 for details.

| Flags | Startup RSS | After 3 requests |
|---|---|---|
| default | 55 GB | 55 GB |
| `--ffn-only` | 5.6 GB | 23 GB |
| `--ffn-only --max-gate-cache-layers 4` | 5.6 GB | 23 GB |
| `... --release-mmap-after-request` | 5.6 GB | 23 GB stable (Linux: ~6 GB) |
| `... --layers 0-19` (sharding) | 5.6 GB | ~8 GB |

`--layers` is the strict bound. The other flags target individual growth
modes and compose cleanly (`--ffn-only` skips startup warmup,
`--max-gate-cache-layers` caps decoded heap, `--release-mmap-after-request`
hints the kernel to drop mmap pages).

## Examples and Benchmarks

All examples compile with:

```bash
cargo check -p larql-server --examples
```

Synthetic demos do not require a real vindex:

```bash
cargo run -p larql-server --example server_demo
cargo run -p larql-server --example embed_demo
```

The OpenAI-compat live demo boots an in-process server and exercises
`/v1/models`, `/v1/embeddings`, `/v1/completions` against a real
vindex (no port binding, no external HTTP client):

```bash
cargo run --release -p larql-server --example openai_demo -- \
  output/gemma3-4b-q4k-streaming.vindex
```

Synthetic release benchmark, captured 2026-04-26 (re-validated
2026-05-01 post Q1 cleanup — within noise):

```bash
cargo run -p larql-server --example server_bench --release
```

| Operation | Result |
|---|---:|
| `gate_knn` L0 top-5 | 0.022 ms/op |
| `walk` 8 layers top-5 | 0.203 ms/op |
| `walk-ffn` single layer | 0.032 ms/op |
| `walk-ffn` batched 8 layers | 0.321 ms/op |
| `describe` simulation | 0.298 ms/op |
| `relations` simulation | 0.399 ms/op |
| `embed` 512-token prefill | 0.115 ms/op |
| `logits` dot, 1024 vocab × 256 hidden | 0.221 ms/op |
| **OpenAI envelopes (encode-only):** | |
| `/v1/models` JSON serialize | 0.001 ms/op (1.02 M ops/s) |
| `/v1/embeddings` single (hidden=256) | 0.008 ms/op |
| `/v1/embeddings` batch=8 (hidden=256) | 0.074 ms/op |
| `/v1/completions` serialize | 0.001 ms/op (723 K ops/s) |
| `/v1/chat/completions` serialize | 0.002 ms/op (635 K ops/s) |
| `/v1/chat/completions` Gemma render (3 turns) | 0.000 ms/op (5.7 M ops/s) |
| **Constrained decoding (slice 4 fixed cost):** | |
| FSM step `Schema::Any` (~50-char object) | 0.001 ms/op (1.01 M ops/s) |
| FSM step strict Person schema | 0.002 ms/op (652 K ops/s) |
| `parse_schema` Person (strict) | 0.001 ms/op (832 K ops/s) |
| `synth_tools_schema` 2-function union | 0.004 ms/op (263 K ops/s) |
| FSM tool-call OneOf (commit on `name`) | 0.025 ms/op (40 K ops/s) |
| **Sampler extras (F18, F19, slice 4.10):** | |
| Sampler with frequency_penalty (history N=8, vocab=256) | 0.001 ms/op (797 K ops/s) |
| Sampler with temperature + top-p (no penalty) | 0.006 ms/op (171 K ops/s) |

These numbers measure in-process synthetic index operations, not network
latency or real model weight paging. For a live vindex, use:

```bash
cargo run --release -p larql-server --example bench_embed_server -- \
  output/gemma3-4b-q4k-streaming.vindex

# Optional logits projection bench:
cargo run --release -p larql-server --example bench_embed_server -- \
  output/gemma3-4b-q4k-streaming.vindex --logits
```

Live embed numbers (2026-05-01, ADR-0008 f16 mmap, Gemma 3 4B, 262144 ×
2560 vocab × hidden):

| Operation | Result |
|---|---:|
| f16 embed 1 token — L1 hit | **4.3 ns/op** (232 M ops/s) |
| f16 embed 1 token — mmap decode (L1 miss) | 3.22 µs/op |
| f16 embed 32 tokens (prefill) | 59 µs/op |
| f16 embed 128 tokens (prefill) | 239 µs/op |
| f16 embed 512 tokens (prefill) | 1.10 ms/op |
| Logits projection (full vocab, CPU) | 335 ms (Metal: ~0.67 ms) |
| RSS, `--embed-only` (f32 heap) | ~2.9 GB |
| **RSS, `--embed-only` (f16 mmap + L1)** | **~1.6 GB** (48% reduction) |

For a hybrid-MoE vindex (Gemma 4 26B-A4B etc.), `bench_expert_server`
exercises the per-expert HTTP path end-to-end:

```bash
cargo run --release -p larql-server --example bench_expert_server -- \
  output/gemma4-26b-a4b-q4k.vindex
```

Flags (all combinable):

| Flag | Effect |
|------|--------|
| `--ffn-only` | Skip the f16 gate-vector warmup (faster boot, lazy decode). |
| `--two-shard` | Spin up 2 in-process shards instead of 1. |
| `--uds` | Bind a Unix domain socket alongside TCP and route the bench client through it (compares ~150 µs/call savings vs TCP loopback). |
| `--wire f32\|f16` | Wire format for the layer-batch endpoint. f16 halves wire bytes; on loopback the f32↔f16 conversion CPU cancels the saving (use on real LAN). Default f32. |

Reference numbers on M3 Max (single in-process shard, layer 15, top-K=8;
30-layer sweep is 1 decode-step's worth of MoE blocks):

| Config | `forward_moe` warm | 30-layer sweep |
|---|---|---|
| TCP HTTP + f32 (default) | **0.78 ms** | **23.24 ms** (0.77 ms/layer) |
| `cpu_moe_forward` floor (no HTTP) | 0.34 ms | — |
| UDS + f32 | 0.74 ms | 21.4 ms ← best on loopback |
| TCP HTTP + f16 | 1.05 ms | 29.6 ms (f16 conv CPU dominates on loopback) |
| UDS + f16 | 0.71 ms | 21.7 ms |

Full perf snapshot (per-layer breakdown, RSS, vindex load time, etc.)
is in `ROADMAP.md` → "Live perf snapshot → Remote MoE expert path".
The numbers above are the 2026-05-01 baseline (re-validated post Q1
cleanup); the ROADMAP also tracks the historical progression
(4.86 ms → 1.91 ms → 0.78 ms `forward_moe` warm across the 2026-04-26
+ 2026-05-01 sessions).

## Recommended setups

### Layer-range sharding (dense + MoE attention/router)

Two shards, one router:

```bash
# Router (advertises a gRPC grid port for shards to register against):
larql-router --grid-port 50051 --port 9090 --grid-key SECRET

# Shard A — layers 0..14:
larql-server <vindex> --layers 0-14 --port 8881 --no-infer \
  --join http://router-host:50051 --public-url http://shard-a:8881 \
  --grid-key SECRET

# Shard B — layers 15..29:
larql-server <vindex> --layers 15-29 --port 8882 --no-infer \
  --join http://router-host:50051 --public-url http://shard-b:8882 \
  --grid-key SECRET
```

Clients POST to `http://router:9090/v1/walk-ffn` with `{model_id, residual,
layers, top_k}`; the router fans out to the owning shards and merges results.

### Remote MoE shard topology

For hybrid-MoE models (e.g. Gemma 4 26B-A4B's 128 experts × 30 layers),
shard the expert bank across processes / hosts. Each shard mmaps the full
vindex but only the configured experts are reachable; the inference client
runs attention + dense FFN + the router locally, then POSTs the
post-attention residual + selected expert IDs to the owning shard(s).

#### Two-shard split (production-ready)

```bash
# Shard A — experts 0..63, HTTP + gRPC + UDS bound for same-host clients
larql-server output/gemma4-26b-a4b-q4k.vindex \
  --experts 0-63 --port 8881 --grpc-port 9081 \
  --uds-path /tmp/larql-moe-a.sock --warmup-walk-ffn

# Shard B — experts 64..127
larql-server output/gemma4-26b-a4b-q4k.vindex \
  --experts 64-127 --port 8882 --grpc-port 9082 \
  --uds-path /tmp/larql-moe-b.sock --warmup-walk-ffn
```

```bash
# Inference client — gRPC + SPLIT overlap default-on
larql run output/gemma4-26b-a4b-q4k.vindex \
  --moe-shards "0-63=grpc://localhost:9081,64-127=grpc://localhost:9082" \
  "Write a 100-word poem about computers." --max-tokens 100
# → ~19.7 tok/s steady-state (M3 Max, single shard collocated with client)
```

Per-shard URL scheme decides transport:
- `grpc://host:port` — persistent HTTP/2 channel; enables fire/collect
  streaming overlap with dense FFN GPU compute (default-on; ~12% faster
  than unary). Set `LARQL_MOE_NO_SPLIT=1` to opt out.
- `http://host:port` — TCP/HTTP; goes through the
  `/v1/experts/layer-batch` endpoint (one residual + K experts per call).
  TCP_NODELAY is set on accepted connections by default.
- `unix:///abs/path/to/sock` — manual HTTP/1.1 over a Unix domain socket;
  ~50 µs/call faster than TCP loopback (~3% end-to-end). Same wire
  format as the TCP HTTP path, identical correctness, smaller per-call
  cost. Same-host only.

#### Wire formats

| Endpoint | Content-Type | Use |
|---|---|---|
| `POST /v1/experts/layer-batch` | `application/x-larql-experts-layer` | Default. f32 residual + K (expert_id, weight) pairs → one router-weighted-sum vector. Server applies pre_experts_norm + Q8_K quantisation once and shares across the K experts. Saves K-1 redundant per-call work vs the legacy `/v1/expert/batch`. |
| `POST /v1/experts/layer-batch-f16` | `application/x-larql-experts-layer-f16` | Same shape with f16 residual + response. Halves wire bytes; opt-in with `LARQL_MOE_WIRE_F16=1` for LAN deployments where bandwidth matters more than the 9 µs/call f32↔f16 conversion CPU. |
| `POST /v1/expert/batch` (legacy) | `application/x-larql-expert` | Pre-2026-05-01 path: K (layer, expert_id, residual) items per call. Still served for back-compat. |

#### Performance reference (M3 Max, single local shard, Gemma 4 26B-A4B)

End-to-end `larql run` decode tok/s, 100-token poem, 3-run average.
Each row uses the indicated transport for `--moe-shards`. Wire format
is f32 unless noted; SPLIT (overlap with dense FFN GPU compute) is
default-on for `grpc://` shards.

| Transport | Wire | tok/s |
|---|---|---|
| `http://` (TCP HTTP, layer-batch endpoint) | f32 | **17.8** |
| `grpc://` + `LARQL_MOE_NO_SPLIT=1` (unary) | f32 | 17.7 |
| **`grpc://` + SPLIT overlap (default)** | f32 | **19.7** |
| `unix:///path/to/sock` (UDS HTTP/1.1) | f32 | 18.2 |

End-to-end ~19.7 tok/s = ~64 ms/tok, of which ~23 ms is MoE (30 layers
× ~0.77 ms/layer per the post-cleanup re-validation) and ~41 ms is
attention + dense FFN + lm_head + sampling on the client side.

For per-call latency breakdowns of each transport / wire combination,
see the `bench_expert_server` table in **Examples and Benchmarks**
above (those are micro-benchmark numbers — synthetic input, no decode
loop). The two reference tables agree within run-to-run noise.

For multi-host topologies (LAN-class RTT ≥ 100 µs), see
`ROADMAP.md → F-FLY` for the planned fly.io validation. The TCP
HTTP / UDS / f16-wire choices behave very differently on real
networks vs loopback.

### Per-layer FFN format

MoE vindexes store expert weights as per-layer Q4_K files
(`layers/layer_{L:02}.weights`); the legacy `experts_packed.bin` BF16
monolith is no longer written. To migrate an old MoE vindex in place:

```bash
cargo run --release -p larql-cli --example convert_moe_to_per_layer -- \
  output/<vindex>
# Then strip `packed_bf16` rows from weight_manifest.json and rm experts_packed.bin.
```

The loader (`format/weights/load.rs:614`) auto-detects the layout via
`index.json`'s `"ffn_layout": "per_layer"`. Both old and new vindexes are
supported through the same code path.

## API Endpoints

### Knowledge Endpoints (browse-only)

#### GET /v1/describe

Query all knowledge edges for an entity. Edges include probe-confirmed relation labels when `feature_labels.json` is present in the vindex.

```
GET /v1/describe?entity=France
GET /v1/describe?entity=France&band=all&verbose=true&limit=10&min_score=5.0
```

```json
{
  "entity": "France",
  "model": "google/gemma-3-4b-it",
  "edges": [
    {"relation": "capital", "target": "Paris", "gate_score": 1436.9, "layer": 27, "source": "probe", "also": ["Berlin", "Tokyo"]},
    {"target": "French", "gate_score": 35.2, "layer": 24},
    {"target": "Europe", "gate_score": 14.4, "layer": 25}
  ],
  "latency_ms": 12.3
}
```

| Param | Default | Description |
|-------|---------|-------------|
| `entity` | required | Entity name |
| `band` | `knowledge` | Layer band: syntax, knowledge, output, all |
| `verbose` | false | Include layer_min, layer_max, count per edge |
| `limit` | 20 | Max edges |
| `min_score` | 5.0 | Minimum gate score |

#### GET /v1/walk

Feature scan — which features fire for a prompt.

```
GET /v1/walk?prompt=The+capital+of+France+is&top=5
GET /v1/walk?prompt=Einstein&top=10&layers=24-33
```

```json
{
  "prompt": "The capital of France is",
  "hits": [
    {"layer": 27, "feature": 9515, "gate_score": 1436.9, "target": "Paris"},
    {"layer": 24, "feature": 4532, "gate_score": 26.1, "target": "French"}
  ],
  "latency_ms": 0.4
}
```

#### POST /v1/select

SQL-style edge query over down-projection metadata.

```json
POST /v1/select
{"entity": "France", "limit": 10, "order_by": "c_score", "order": "desc"}
```

```json
{
  "edges": [
    {"layer": 26, "feature": 8821, "target": "Paris", "c_score": 0.95}
  ],
  "total": 94,
  "latency_ms": 5.2
}
```

#### GET /v1/relations

List top tokens across knowledge layers.

```json
{
  "relations": [
    {"name": "Paris", "count": 94, "example": "Berlin"},
    {"name": "French", "count": 51, "example": "German"}
  ],
  "total": 512
}
```

#### GET /v1/stats

Model and index statistics, plus live W2 / Q4K cache state for
operator verification (see ROADMAP for the W2 retrofit story).

```json
{
  "model": "google/gemma-3-4b-it",
  "family": "gemma3",
  "layers": 34,
  "features": 348160,
  "hidden_size": 2560,
  "layer_bands": {"syntax": [0, 13], "knowledge": [14, 27], "output": [28, 33]},
  "loaded": {"browse": true, "inference": true},
  "q4k_ffn": {
    "cache_slots": 0,
    "cache_bytes": 0,
    "feature_major_down": true
  }
}
```

The `q4k_ffn` block lets operators confirm the W2 feature-major
down path is active (`feature_major_down: true` after extracting
with `--feature-major-down` or retrofitting via
`larql convert add-feature-major-down`). The legacy
`q4k_ffn_layer` cache should stay at `cache_slots: 0` in
production; non-zero indicates either (a) the W2 file is missing,
or (b) the workload is hitting the sparse walk path which
prefers the cache fallback when W2 isn't loaded.

#### POST /v1/warmup

Pre-touch the lazy state that `walk-ffn` would otherwise pay on first
request. Same code path as the `--warmup-walk-ffn` boot flag, exposed
over HTTP so operators can re-warm a running server without restart.

```bash
# default — warm everything (weights + every owned layer's Q4K mmap)
curl -X POST http://localhost:8080/v1/warmup

# selective — only mmap-prefetch specific layers, skip weights
curl -X POST http://localhost:8080/v1/warmup \
     -H 'content-type: application/json' \
     -d '{"layers": [14, 22, 28], "skip_weights": true}'
```

| Field | Default | Description |
|-------|---------|-------------|
| `layers` | every owned layer | Layers to `madvise WILLNEED` |
| `skip_weights` | false | Skip the `get_or_load_weights` call (only mmap prefetch). Use after the weights are already loaded. |
| `warmup_hnsw` | false | Eager-build HNSW for every owned layer at this call. Requires `--hnsw` at boot. |

```json
{
  "model": "google/gemma-3-4b-it",
  "weights_loaded": true,
  "weights_load_ms": 1266,
  "layers_prefetched": 30,
  "prefetch_ms": 13,
  "hnsw_built": false,
  "hnsw_warmup_ms": 0,
  "total_ms": 1279
}
```

Measured impact (Gemma 26B-A4B, M3 Max): first `/v1/walk-ffn`
**1247 ms → 12.6 ms (99×)**. Costs ~1.3 s + 3.2 GB pre-allocated f32
gate cache.

### Inference Endpoint

#### POST /v1/infer

Full forward pass with attention weights. Model weights are lazy-loaded on first request. Requires a vindex built with `--include-weights` (or extract level `inference`/`all`). Disabled when `--no-infer` is set.

```json
POST /v1/infer
{"prompt": "The capital of France is", "top": 5, "mode": "walk"}
```

| Field | Default | Description |
|-------|---------|-------------|
| `prompt` | required | Input text |
| `top` | 5 | Top-K predictions |
| `mode` | `walk` | `walk` (vindex FFN), `dense` (original FFN), `compare` (both) |

**Walk mode response:**

```json
{
  "prompt": "The capital of France is",
  "predictions": [
    {"token": "Paris", "probability": 0.9791},
    {"token": "the", "probability": 0.0042}
  ],
  "mode": "walk",
  "latency_ms": 210
}
```

**Compare mode response:**

```json
{
  "prompt": "The capital of France is",
  "walk": [{"token": "Paris", "probability": 0.9791}],
  "walk_ms": 210,
  "dense": [{"token": "Paris", "probability": 0.9801}],
  "dense_ms": 180,
  "latency_ms": 420
}
```

### Streaming Endpoint

#### WS /v1/stream

WebSocket endpoint for layer-by-layer streaming DESCRIBE. Client sends a JSON message, server streams per-layer results.

```
→ {"type": "describe", "entity": "France", "band": "all"}
← {"type": "layer", "layer": 14, "edges": []}
← {"type": "layer", "layer": 15, "edges": [{"target": "French", "gate_score": 35.2}]}
← {"type": "layer", "layer": 27, "edges": [{"relation": "capital", "target": "Paris", "gate_score": 1436.9, "source": "probe"}]}
← {"type": "done", "entity": "France", "total_edges": 6, "latency_ms": 12.3}
```

### Decoupled Inference Endpoint

#### POST /v1/walk-ffn

Client sends a residual vector, server runs gate KNN and returns selected features. Enables distributed inference where the client runs attention locally.

**Single layer:**

```json
POST /v1/walk-ffn
{"layer": 26, "residual": [0.12, -0.34, ...], "top_k": 8092}
→ {"layer": 26, "features": [9515, 4532, ...], "scores": [1436.9, 26.1, ...], "latency_ms": 0.01}
```

**Batched (all layers in one round-trip):**

```json
POST /v1/walk-ffn
{"layers": [0, 1, ..., 33], "residual": [0.12, -0.34, ...]}
→ {"results": [{"layer": 0, "features": [...], "scores": [...]}, ...], "latency_ms": 0.3}
```

**Full-output mode** — returns the computed FFN output vector (gate KNN → up
gather → down projection). Requires model weights (`--ffn-only` is sufficient).

```json
POST /v1/walk-ffn
Content-Type: application/json
{"layer": 26, "residual": [...], "seq_len": 1, "full_output": true}
→ {"layer": 26, "output": [...], "seq_len": 1, "latency_ms": 8.1}
```

**Binary wire format** (`Content-Type: application/x-larql-ffn`) — eliminates
JSON float serialization overhead. Only supported with `full_output: true`.

```
Single-layer request:
  [4: layer u32 LE][4: seq_len u32][4: flags u32 (bit0=1)][4: top_k u32][residual f32[] LE]

Single-layer response:
  [4: layer u32 LE][4: seq_len u32][4: latency f32][output f32[] LE]

Batch request:  [4: BATCH_MARKER=0xFFFFFFFF][4: num_layers u32][layers u32[] LE]...
Batch response: [4: BATCH_MARKER][4: num_results u32][4: latency f32]
                per result: [4: layer][4: seq_len][4: num_floats][output f32[] LE]
```

Performance vs JSON (Gemma 3 4B, hidden_size=3072, seq_len=1): ~33% smaller
requests, ~0.5 ms/hop faster.

`RemoteWalkBackend` in `larql-inference` uses binary format automatically and
exposes `forward_all_layers()` for a batched single-round-trip forward pass.

### Remote MoE Expert Endpoints

Used by `RemoteMoeBackend` in `larql-inference` when the inference client
runs attention + dense FFN + router locally and dispatches per-layer
top-K expert work to one or more shard servers. See
`Remote MoE shard topology` above for the deployment picture.

#### POST /v1/experts/layer-batch

**Binary wire** (`Content-Type: application/x-larql-experts-layer`).
Single residual + K (expert_id, weight) pairs for one layer. Server
applies pre_experts_norm once, quantises h_norm to Q8_K once, fans out
the K expert kernels with the shared activation via rayon, returns the
router-weighted sum.

```
Request:  [4: layer u32 LE][4: hidden u32][4: K u32]
          + hidden × f32  (residual, sent ONCE per call)
          + K × [4: expert_id u32, 4: weight f32]

Response: [4: hidden u32 LE][4: latency_ms f32]
          + hidden × f32  (router-weighted sum across K experts)
```

Replaces the legacy `/v1/expert/batch` (which shipped K identical residual
copies on the wire). Saves ~2.6 MB/token of redundant residual data plus
the K-1 redundant pre_experts_norm + Q8_K quantisations on the server.

#### POST /v1/experts/layer-batch-f16

Same shape as `layer-batch` but residual + response use IEEE-754 binary16.
Halves wire bytes (~5.5 KB request + 5.5 KB response vs 11+11 KB f32).
f16 quant noise is well below the Q8_K activation quantisation already
applied in the SDOT path; end-to-end accuracy unchanged.

Opt-in via `LARQL_MOE_WIRE_F16=1` on the client (server always exposes
both endpoints). Loopback: within noise (CPU conversion cancels wire
saving). LAN (1 Gbps): expected +3-5%.

#### POST /v1/expert/batch (legacy)

Pre-2026-05-01 wire format: `application/x-larql-expert` carrying N items
each with `(layer, expert_id, residual)`. Still served for back-compat.
New deployments should use `layer-batch` for the per-call wire savings.

#### POST /v1/expert/{layer}/{expert_id}

JSON-only single-expert dispatch. Diagnostic / smoke-test path.

### Embed Service Endpoints (ADR-0008)

Enabled on every server (including `--ffn-only` and default mode). The primary use case is `--embed-only`: offload the static embedding table and lm_head to a dedicated small server, shrinking the attention-only client from ~7 GB to ~1.9 GB on 31B models.

```bash
# Start an embed-only server
larql-server output/gemma3-4b-v2.vindex --embed-only --port 8082

# Serving google/gemma-3-4b-it — mode: embed-service
# Loaded: embeddings (1.3 GB), lm_head (tied), tokenizer
# Listening: http://0.0.0.0:8082
```

#### POST /v1/embed

Convert token IDs to scaled initial residual vectors.

```json
POST /v1/embed
{"token_ids": [1, 5432, 235, 1234]}
```

```json
{
  "residual": [[0.12, -0.03, ...], [0.45, 0.01, ...]],
  "seq_len": 4,
  "hidden_size": 2560,
  "latency_ms": 0.02
}
```

Binary wire format (`Content-Type: application/x-larql-ffn`):

```
Request:  [num_tokens u32 LE][token_id u32 LE × N]
Response: [seq_len u32 LE][hidden_size u32 LE][residuals f32[] LE]
```

**Measured (Gemma 3 4B, hidden=2560):** encode request 17 ns, encode response 1.5 µs.
Binary is 6.7× faster and 3× smaller than JSON for the embed response. Use binary on the decode hot path.

#### POST /v1/logits

Project a final residual through lm_head to get token probabilities. Accepts JSON or binary input.

```json
POST /v1/logits
{"residual": [0.12, -0.03, ...], "top_k": 5, "temperature": 1.0}
```

```json
{
  "top_k": [
    {"token_id": 9515, "token": "Paris", "prob": 0.801},
    {"token_id": 235,  "token": "the",   "prob": 0.042}
  ],
  "latency_ms": 2.1
}
```

Binary input (`Content-Type: application/x-larql-ffn`): raw `[f32 × hidden_size]` little-endian bytes.

Performance (measured, Gemma 3 4B): ~14ms CPU (BLAS), ~0.67ms Metal (Apple Silicon f32_gemv).

#### GET /v1/token/encode

```
GET /v1/token/encode?text=Paris
→ {"token_ids": [9515], "text": "Paris"}
```

#### GET /v1/token/decode

```
GET /v1/token/decode?ids=9515,235,1234
→ {"text": "Paris the model", "token_ids": [9515, 235, 1234]}
```

#### Memory footprint — embed-only server

Measured on Gemma 3 4B Q4K (macOS, release build). See ADR-0008 for full benchmark output.

| Model | Disk (f16) | RSS (f32 heap) | Total RSS (with tokenizer) |
|-------|-----------|----------------|---------------------------|
| Gemma 3 4B | 1.34 GB | 2.69 GB | ~2.9 GB |
| Gemma 4 31B | 2.67 GB | 5.37 GB | ~5.6 GB |
| Llama 3 70B | 2.10 GB | 4.20 GB | ~4.5 GB |

The current implementation decodes f16→f32 at load time (doubles RSS vs disk).
A future f16-at-rest path will halve this — tracked in ADR-0008 open questions.

The tokenizer alone takes ~244 MB for the Gemma 262K-vocab BPE model.

### gRPC

All endpoints are available over gRPC using Protocol Buffers. Enable with `--grpc-port`:

```bash
larql serve output/gemma3-4b-v2.vindex --port 8080 --grpc-port 50051
```

Proto definition: `proto/vindex.proto`. Services: `Describe`, `Walk`, `Select`, `Infer`, `GetRelations`, `GetStats`, `WalkFfn`, `Health`, `StreamDescribe` (server-streaming).

gRPC runs alongside HTTP — both ports active simultaneously.

### Patch Endpoints

#### POST /v1/patches/apply

Apply a patch in-memory (does not modify base files). Session-aware: include `X-Session-Id` header to scope patches to a session.

```json
POST /v1/patches/apply
{"patch": {"version": 1, "base_model": "...", "operations": [...]}}
```

```bash
# Session-scoped patch
curl -H "X-Session-Id: sess-a" -X POST http://localhost:8080/v1/patches/apply -d '{"patch": {...}}'
```

#### GET /v1/patches

List active patches (session-aware).

#### DELETE /v1/patches/{name}

Remove a patch by description (session-aware).

### Management Endpoints

#### GET /v1/health

Always accessible (exempt from API key auth).

```json
{"status": "ok", "uptime_seconds": 3600, "requests_served": 12450}
```

#### GET /v1/models

OpenAI-compatible shape (works with the `openai` Python/JS SDK as-is).
Larql-specific fields (`path`, `features`, `loaded`) are present as
extras — OpenAI clients ignore them.

```json
{
  "object": "list",
  "data": [
    {
      "id": "gemma-3-4b-it",
      "object": "model",
      "created": 1746094800,
      "owned_by": "larql",
      "path": "/v1",
      "features": 348160,
      "loaded": true
    }
  ]
}
```

### OpenAI-compatible Endpoints (N0 slice 1)

These endpoints conform to the OpenAI API shape so existing
`openai` Python/JS SDKs work unmodified:

```python
from openai import OpenAI
client = OpenAI(base_url="http://larql:8080/v1", api_key="sk-...")

# /v1/models
models = client.models.list()

# /v1/embeddings
emb = client.embeddings.create(model="gemma-3-4b", input="hello world")

# /v1/completions
resp = client.completions.create(
    model="gemma-3-4b",
    prompt="The capital of France is",
    max_tokens=10,
)
```

#### POST /v1/embeddings

Mean-pooled static-embedding lookup. All four `input` variants
supported: `string`, `string[]`, `int[]` (pre-tokenised), `int[][]`
(pre-tokenised batched).

```json
POST /v1/embeddings
{"model": "gemma-3-4b-it", "input": "France"}

→ {
  "object": "list",
  "data": [{"object": "embedding", "embedding": [0.12, ...], "index": 0}],
  "model": "gemma-3-4b-it",
  "usage": {"prompt_tokens": 1, "total_tokens": 1}
}
```

Note: results are *lookup-pooled* — they're a mean over the
input-token static embeddings, not a contrastively-trained sentence
encoder. Useful as a baseline; not competitive with dedicated
embedding models for retrieval ranking.

`encoding_format: "base64"` returns each vector as a base64-encoded
little-endian f32 byte string (~33% smaller wire than the JSON float
array form).

#### POST /v1/completions

Non-streaming text completions.

```json
POST /v1/completions
{
  "model": "gemma-3-4b-it",
  "prompt": "The capital of France is",
  "max_tokens": 10,
  "temperature": 0.7
}

→ {
  "id": "cmpl-abc123...",
  "object": "text_completion",
  "created": 1746094800,
  "model": "gemma-3-4b-it",
  "choices": [{
    "text": " Paris.",
    "index": 0,
    "finish_reason": "stop",
    "logprobs": null
  }],
  "usage": {"prompt_tokens": 6, "completion_tokens": 2, "total_tokens": 8}
}
```

Live: SSE streaming via `stream: true` (one chunk per token,
terminated by `data: [DONE]`); `temperature`, `top_p`, `seed`,
`stop`, `frequency_penalty`, `presence_penalty` all honoured by the
sampler; `logprobs: int` populates `choices[i].logprobs` with
per-token entries (top-k alternatives are placeholder until the
inference layer surfaces them — F18 follow-up); KV-cached generation
on f16 vindexes (Q4_K vindexes use the per-step CPU fallback).
Limitations: `n>1` → 400 (single completion per prompt); echo +
batched prompts disallowed in stream mode.

#### POST /v1/chat/completions

Multi-turn chat with chat-template rendering. Messages are rendered to
the model's native template (Gemma `<start_of_turn>` / Llama 3 header
tags / ChatML `<|im_start|>` / Mistral `[INST]` / plain) auto-detected
from the model family or id, then run through the same generation
loop as `/v1/completions`.

```json
POST /v1/chat/completions
{
  "model": "gemma-3-4b-it",
  "messages": [
    {"role": "system", "content": "You are concise."},
    {"role": "user",   "content": "What is the capital of France?"}
  ],
  "max_tokens": 10,
  "temperature": 0.0
}

→ {
  "id": "chatcmpl-abc123...",
  "object": "chat.completion",
  "created": 1746094800,
  "model": "gemma-3-4b-it",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": " Paris."},
    "finish_reason": "stop",
    "logprobs": null
  }],
  "usage": {"prompt_tokens": 16, "completion_tokens": 2, "total_tokens": 18}
}
```

When `tools` is on the request, the response shape switches to the
tool-calls form: `message.content: null`, `tool_calls: [{id, type:
"function", function: {name, arguments}}]`, `finish_reason:
"tool_calls"`. `arguments` is JSON-stringified (OpenAI's wire shape).

Live: SSE streaming, sampling fields (`temperature`, `top_p`, `seed`,
`stop`, `frequency_penalty`, `presence_penalty`) honoured by the
sampler — including under the constrained-decoding mask, constrained
decoding via `response_format: json_object | json_schema` and `tools`
/ `tool_choice` (see "Structured outputs and tool calling" in the
Quickstart section above), tool-result replay via `role: "tool"`
messages, top-k logprobs scaffolding (`logprobs: true` + `top_logprobs`).

Limitations: `n>1` → 400; tools + `stream=true` emits the call as a
single delta chunk rather than per-token argument streaming
(per-token tightening pending); `top_logprobs` returns picked-token
entries only — full top-K alternatives need inference work (F18
follow-up).

Coming next:
- **N0.3** Responses API (`/v1/responses`) — pairs with N1 stateful sessions

## Authentication

When `--api-key` is set, all endpoints (except `/v1/health`) require a Bearer token:

```bash
larql serve output/gemma3-4b-v2.vindex --api-key "sk-abc123"
```

```bash
curl -H "Authorization: Bearer sk-abc123" "http://localhost:8080/v1/describe?entity=France"
```

Requests without a valid token receive 401 Unauthorized.

## Rate Limiting

Per-IP token bucket rate limiting. Supports `N/sec`, `N/min`, `N/hour` formats. `/v1/health` is exempt.

```bash
larql serve output/gemma3-4b-v2.vindex --rate-limit "100/min"
```

Excess requests receive `429 Too Many Requests`. By default the limiter uses
the socket peer address and ignores client-supplied `X-Forwarded-For`. Behind a
trusted reverse proxy, add `--trust-forwarded-for` so the first forwarded IP is
used as the bucket key; the proxy must strip untrusted forwarding headers.

## DESCRIBE Cache

Cache DESCRIBE responses in memory with a configurable TTL. Useful for popular entities queried repeatedly.

```bash
larql serve output/gemma3-4b-v2.vindex --cache-ttl 300  # 5 minute cache
```

Cache keys include: model ID, entity, band, limit, min_score. Expired entries are evicted automatically.

## Sessions

Per-session patch isolation via the `X-Session-Id` header. Each session gets its own PatchedVindex overlay — patches applied to one session don't affect others or the global state.

```bash
# Session A applies a medical patch
curl -H "X-Session-Id: sess-a" \
     -X POST http://localhost:8080/v1/patches/apply \
     -d '{"patch": {"version": 1, ...}}'

# Session A sees patched edges
curl -H "X-Session-Id: sess-a" "http://localhost:8080/v1/describe?entity=aspirin"

# Session B (or no header) sees unpatched edges
curl "http://localhost:8080/v1/describe?entity=aspirin"
```

Sessions expire after 1 hour of inactivity. Without an `X-Session-Id` header, patches go to the global (shared) state.

## Error Codes

| Status | When |
|--------|------|
| 200 | Success |
| 400 | Bad request (empty prompt, missing required params) |
| 401 | Unauthorized (missing/invalid API key) |
| 404 | Model not found (multi-model), patch not found |
| 429 | Rate limit exceeded |
| 503 | Inference unavailable (`--no-infer` or no model weights) |
| 500 | Internal server error |

There are **two error envelope shapes**, split by endpoint family
(documented in `docs/server-spec.md §8.3.1`):

- **LARQL paradigm endpoints** (`/v1/describe`, `/v1/walk`,
  `/v1/select`, `/v1/relations`, `/v1/stats`, `/v1/infer`,
  `/v1/patches/*`, `/v1/walk-ffn`, `/v1/insert`, `/v1/embed`,
  `/v1/logits`, `/v1/token/*`) — flat: `{"error": "message"}`.
- **OpenAI-compatible endpoints** (`/v1/embeddings`,
  `/v1/completions`, `/v1/chat/completions`) — nested
  `{"error": {"message", "type", "param", "code"}}` so the OpenAI
  Python and JS SDKs parse errors without special-casing.

Canonical OpenAI `type` values: `invalid_request_error` (400),
`not_found_error` (404), `server_error` (500),
`service_unavailable_error` (503). `param` and `code` are always
present (possibly `null`).

## Layer Bands

Models are divided into three functional bands based on architecture:

| Band | Gemma 3 4B | What it encodes |
|------|-----------|-----------------|
| `syntax` | L0-13 | Morphology, grammar, function words |
| `knowledge` | L14-27 | Factual relations (capital, language, etc.) |
| `output` | L28-33 | Answer formatting, token selection |

DESCRIBE defaults to `band=knowledge`. Use `band=all` to scan everything.

## Probe Labels

If the vindex contains `feature_labels.json`, probe-confirmed relation labels appear in DESCRIBE and WALK responses:

```json
{"relation": "capital", "target": "Paris", "gate_score": 1436.9, "layer": 27, "source": "probe"}
```

Format of `feature_labels.json`: `{"L27_F9515": "capital", "L24_F4532": "language", ...}`

SELECT also supports filtering by relation: `{"relation": "capital"}` returns only edges with that probe label.

## REPL Integration

The LQL REPL connects to a remote server transparently:

```sql
USE REMOTE "http://localhost:8080";
-- Connected: google/gemma-3-4b-it (34 layers, 348160 features)

DESCRIBE "France";
WALK "Einstein" TOP 10;
INFER "The capital of France is" TOP 5;
STATS;
SHOW RELATIONS;

-- Mutations forwarded to server as patches
INSERT ("aspirin", "treats", "headache");

-- Local patches stay client-side (server never sees them)
APPLY PATCH "private-facts.vlp";
DESCRIBE "aspirin";          -- merges server + local patch edges
SHOW PATCHES;                -- shows local patches only
```

## Multi-Model Serving

When using `--dir`, each vindex gets its own namespace:

```bash
larql serve --dir ./vindexes/ --port 8080
# /v1/gemma-3-4b-it/describe, /v1/llama-3-8b/describe, ...
```

```
GET /v1/gemma-3-4b-it/describe?entity=France
GET /v1/llama-3-8b/describe?entity=France
```

## Crate Structure

```
larql-server/
├── Cargo.toml
├── README.md
├── ROADMAP.md
├── examples/
│   ├── server_demo.rs          Synthetic vindex API demo (no real model)
│   ├── embed_demo.rs           Synthetic embed/logits/token demo
│   ├── openai_demo.rs          Live OpenAI-compat walkthrough — boots an
│   │                           in-process server with the given vindex and
│   │                           exercises /v1/models, /v1/embeddings, /v1/completions
│   ├── server_bench.rs         Synthetic endpoint latency benchmarks
│   ├── bench_embed_server.rs   Live vindex embed-service benchmark
│   └── bench_expert_server.rs  Live MoE expert benchmark (cpu_moe_forward
│                               floor + forward_moe HTTP RTT + 30-layer sweep)
├── docs/
│   ├── server-spec.md          Full endpoint reference + wire formats
│   └── router-spec.md          larql-router (grid coordinator) spec
├── proto/                      gRPC service definitions
├── build.rs                    Proto compilation (bundled protoc)
├── tests/
│   ├── common/                 Shared synthetic vindex/tokenizer fixtures
│   ├── test_http_*.rs          HTTP route integration tests
│   ├── test_grpc.rs            Direct gRPC handler tests
│   ├── test_expert_endpoint.rs Per-expert MoE endpoint tests
│   └── test_unit_*.rs          Focused unit tests (band_utils, state,
│                               protocol parsing)
└── src/
    ├── main.rs                 Thin entry: parse Cli, init tracing, hand off
    │                           to bootstrap::serve. ~26 LOC.
    ├── lib.rs                  Crate-public exports
    ├── bootstrap.rs            Cli struct + serve(): vindex load, warmups,
    │                           listener setup (TCP + optional UDS via
    │                           --uds-path, TCP_NODELAY on accepted conns,
    │                           TLS, gRPC, grid announce).
    ├── state.rs                AppState: loaded models, probe labels, lazy
    │                           weights, expert_filter / unit_filter
    ├── error.rs                ServerError → HTTP status codes
    ├── env_flags.rs            Single source of truth for LARQL_* env knobs
    │                           (cached presence accessors via OnceLock)
    ├── wire.rs                 Shared has_content_type() helper for routes
    │                           that accept both binary and JSON bodies
    ├── http.rs                 Shared HTTP route + content-type constants
    │                           (BINARY_FFN_*, JSON_CONTENT_TYPE,
    │                           REQUEST_BODY_LIMIT_*, BEARER_PREFIX, …)
    ├── auth.rs                 API key Bearer token middleware
    ├── ratelimit.rs            Per-IP token bucket rate limiting
    ├── cache.rs                TTL cache for DESCRIBE results
    ├── session.rs              Per-session PatchedVindex isolation
    ├── etag.rs                 ETag generation for CDN caching
    ├── ffn_l2_cache.rs         Per-model FFN L2 score cache
    ├── embed_store.rs          mmap-backed f16 embedding lookup (--embed-only)
    ├── band_utils.rs           Layer band parsing + filter helpers
    ├── announce.rs             Grid `--join` announce + heartbeat loop
    ├── grpc.rs                 gRPC service (tonic, all browse/infer endpoints)
    ├── grpc_expert.rs          gRPC MoE expert dispatch (used with grpc:// shards)
    └── routes/
        ├── mod.rs              Router setup (single + multi-model)
        ├── describe.rs         GET /v1/describe (cached, ETag, relation labels)
        ├── walk.rs             GET /v1/walk (with relation labels)
        ├── select.rs           POST /v1/select (relation filter)
        ├── relations.rs        GET /v1/relations
        ├── stats.rs            GET /v1/stats
        ├── infer.rs            POST /v1/infer (walk/dense/compare)
        ├── explain.rs          POST /v1/explain-infer (per-layer attention/FFN)
        ├── stream.rs           WS /v1/stream (layer-by-layer streaming)
        ├── walk_ffn.rs         POST /v1/walk-ffn (decoupled FFN dispatch)
        ├── expert/             MoE expert dispatch — split by concern
        │   ├── mod.rs          Re-exports + shared request/response types
        │   ├── single.rs       run_expert + handle_expert
        │   │                   (POST /v1/expert/{layer}/{id})
        │   ├── batch_legacy.rs handle_expert_batch
        │   │                   (POST /v1/expert/batch — pre-2026-05-01 wire)
        │   ├── layer_batch.rs  handle_experts_layer_batch{,_f16}
        │   │                   (POST /v1/experts/layer-batch[-f16])
        │   ├── cpu.rs          run_experts_cpu_batch (rayon CPU dispatch)
        │   ├── metal.rs        run_experts_metal_batch
        │   │                   (#[cfg(all(feature = "metal-experts", target_os = "macos"))])
        │   └── warmup.rs       warmup_hnsw_unit_cache,
        │                       warmup_metal_expert_cache
        ├── topology.rs         GET /v1/expert/topology (shard advertisement)
        ├── embed.rs            POST /v1/embed, /v1/logits, /v1/token/*
        ├── insert.rs           POST /v1/insert (knowledge mutation)
        ├── patches.rs          POST/GET/DELETE /v1/patches (session-aware)
        ├── warmup.rs           POST /v1/warmup (manual weight + mmap warmup)
        ├── health.rs           GET /v1/health
        └── models.rs           GET /v1/models
```

## Dependencies

- `larql-vindex` — vector index loading, gate KNN, walk, patches
- `larql-inference` — forward pass, WalkFfn, dense FFN
- `axum` — HTTP framework
- `axum-server` — TLS support (rustls)
- `tokio` — async runtime
- `tonic` + `prost` — gRPC server and protobuf
- `tower` — concurrency limit middleware
- `tower-http` — CORS, tracing middleware
- `clap` — CLI argument parsing

## Testing

```bash
# Unit + integration tests (~660 tests across lib + 14 test files; all green)
cargo test -p larql-server

# Or via Make (workspace conventions match the other crates):
make larql-server-test
make larql-server-fmt-check
make larql-server-lint                    # clippy --all-targets -D warnings
make larql-server-ci                      # fmt-check + lint + test

# Coverage (cargo-llvm-cov + per-file 90% policy)
make larql-server-coverage-summary        # text summary + summary.json + policy
make larql-server-coverage-html           # writes coverage/larql-server/html/
make larql-server-coverage-policy         # re-run policy check on existing report

# 2026-05-10 measured baseline: 65.68% line / 72.18% function. Per-file
# debt + the 90% default floor live in `coverage-policy.json`; baselines
# only ratchet upward. New / split files must hit 90% on first commit.

# Synthetic demos (no real vindex)
cargo run -p larql-server --example server_demo
cargo run -p larql-server --example embed_demo

# Synthetic endpoint latency benchmark
cargo run -p larql-server --example server_bench --release

# Live OpenAI-compat walkthrough — boots in-process server and
# exercises /v1/models, /v1/embeddings, /v1/completions
cargo run --release -p larql-server --example openai_demo -- \
  output/gemma3-4b-q4k-streaming.vindex

# Live embed benchmark (requires a real vindex)
cargo run --release -p larql-server --example bench_embed_server -- \
  output/gemma3-4b-q4k-streaming.vindex

# Live MoE expert benchmark — measures cpu_moe_forward floor + forward_moe
# HTTP RTT + 30-layer sweep against a real hybrid-MoE vindex
cargo run --release -p larql-server --example bench_expert_server -- \
  output/gemma4-26b-a4b-q4k.vindex

# Router/grid route-table checks
cargo test -p larql-router
```

Per-call timing for the MoE remote-expert path is opt-in via env var:

```bash
# Server-side per-handler breakdown (decode / spawn_overhead / compute / encode)
LARQL_HTTP_TIMING=1 ./target/release/larql-server <vindex> --uds-path /tmp/m.sock

# Client-side per-call breakdown (encode / send_total / recv_body / decode)
LARQL_HTTP_TIMING=1 ./target/release/larql run <vindex> \
  --moe-shards "0-127=unix:///tmp/m.sock" "test" --max-tokens 30

# Per-layer MoE summary (route+fire / collect / server compute estimate / network)
LARQL_MOE_TIMING=1 ./target/release/larql run <vindex> \
  --moe-shards "0-127=grpc://localhost:9081" "test" --max-tokens 30
```

## Deployment

### Docker

```dockerfile
FROM rust:1.82-slim AS builder
WORKDIR /build
COPY . .
RUN cargo build --release -p larql-server

FROM debian:bookworm-slim
COPY --from=builder /build/target/release/larql-server /usr/local/bin/
EXPOSE 8080
ENTRYPOINT ["larql-server"]
```

```bash
docker run -v ./vindexes:/data -p 8080:8080 larql-server /data/gemma3-4b.vindex
```

### Systemd

```ini
[Unit]
Description=LARQL Vindex Server
After=network.target

[Service]
ExecStart=/usr/local/bin/larql-server /data/gemma3-4b.vindex --port 8080
Restart=always
MemoryMax=8G

[Install]
WantedBy=multi-user.target
```

Browse-only (f16): ~3 GB RAM. No GPU needed.

### Multi-host MoE shard topology (fly.io / similar)

Distributing a hybrid-MoE model across multiple VMs for production
serving is on the roadmap as `F-FLY` (see `ROADMAP.md` for VM-sizing
considerations, vindex distribution strategy, and the open questions on
which CPU optimisations win on real LAN-class RTT). Concrete recipe TBD;
the building blocks (sharding flags, gRPC streaming with overlap, f16
wire opt-in for bandwidth-constrained links) are all in place from the
2026-05-01 perf session.

## What's coming

The full forward-looking work is in `ROADMAP.md`. Grouped by track (see
"What this is" above):

### Parity track (clears the bar so the paradigm is reachable)

- **N0. OpenAI API compatibility** — `/v1/chat/completions`,
  `/v1/completions`, `/v1/responses` (stateful), `/v1/embeddings`
  (OpenAI-shape wrapper), `/v1/models`. Streaming via SSE, tool calls,
  JSON-schema `response_format`. Once landed, every existing OpenAI
  client (Python `openai` SDK, JS `openai`, LangChain, LlamaIndex,
  Cursor, Continue, Aider, eval harnesses, dashboards) becomes a larql
  client unmodified. Highest-leverage parity item — it's the adapter
  layer the rest of the ecosystem speaks.
- **N1. Stateful chat sessions** — KV-cache as a first-class resource
  (`POST /v1/sessions`, `/v1/sessions/{id}/append`). Today every
  `/v1/infer` re-prefills from scratch; with sessions the KV-cache stays
  resident across turns. Pairs with N0.3 (Responses API).
- **N2. Async batch inference job queue** — `/v1/jobs` for
  throughput-bound workloads (RAG document processing, evals, embedding
  pre-compute) that don't share the SLO of real-time chat.
- **N3. LoRA / adapter hot-loading per session** — multi-tenant serving,
  hundreds of adapters in RAM next to one base model.
- **F2. Streaming HTTP infer (SSE)**, **F7. KV-cache prefix sharing**,
  **F17. Structured-output / grammar-constrained generation** — the
  remaining table-stakes any 2026 chat client expects.

### Paradigm track (the reason to stay once parity gets you in the door)

- **Already shipped** — DESCRIBE / WALK / SELECT over the indexed
  knowledge graph, patch overlays (`/v1/patches/apply`), residual-addressed
  FFN execution (`/v1/walk-ffn`), remote MoE expert shards as routable
  compute assets (`/v1/experts/layer-batch`, gRPC streaming overlap, UDS
  same-host transport, f16 wire opt-in), embed-only / FFN-only mode splits,
  CPU-first multi-host shard topology.
- **N4. Multimodal API surface** — vision tower endpoint for Gemma 3/4 +
  Llama 3.2 vision variants. The vindex extractor already handles the
  weights; only the API surface is missing.
- **N5. Federated knowledge graph over multiple vindexes** — ask
  "describe France using Gemma's knowledge AND Llama's knowledge AND a
  custom vindex" in one call, with per-edge model attribution and
  confidence-weighted merge. No other LLM serving stack can do this; it
  falls out of the substrate. Pairs with the LQL `USE REMOTE` /
  `DESCRIBE … USING gemma, llama` syntax already hinted in the REPL.
- **N6. Live blue-green vindex deployment** — load v2 alongside v1,
  weighted traffic ramp, side-by-side metrics for canary rollout. Possible
  because vindexes are static artefacts, not in-process model state.
- **F-FLY. Remote multi-shard deployment on fly.io** — validation that
  the 2026-05-01 HTTP perf optimisations translate to real LAN-class RTT.
  Loopback can't tell us how f16 wire / TCP_NODELAY behave on a real
  network.

A code-quality cleanup pass (Q1.1–Q1.10 — split `routes/expert.rs`,
centralise env flags, lift remaining magic numbers) is also queued.

## License

Apache-2.0
