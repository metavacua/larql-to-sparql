# ADR-0008 — Remote Embeddings and lm_head Service

**Status:** Implemented (Phase 1 + f16 store + CDN endpoint)  
**Depends on:** ADR-0003 (FFN Router), ADR-0004 (FFN Grid), ADR-0006 (Q4K Remote Path)

---

## Problem

Every client in the current architecture holds three components locally:

1. **Attention weights** — dynamic, must be local, irreducible
2. **Embeddings** — static lookup table, 262K × hidden_size
3. **lm_head** — static projection, tied to embeddings in most models

Components 2 and 3 are pure static lookups. They require no computation
beyond a table lookup (embed) and a single matmul (lm_head). Yet they
consume 2–5GB of client RAM depending on model size — comparable to or
larger than the attention weights themselves.

Moving embeddings and lm_head to a dedicated server reduces the client
to attention-only. The client holds only what is genuinely dynamic.

---

## The Decomposition

```
Component        Type              Size (31B)   Location
─────────────────────────────────────────────────────────
Embeddings       static lookup     2.7 GB       → embed server
lm_head          static matmul     2.7 GB       → embed server (tied)
Attention        dynamic compute   1.9 GB       client (irreducible)
FFN              knowledge graph   31.0 GB      FFN grid
─────────────────────────────────────────────────────────
Client today:    7.3 GB
Client after:    1.9 GB            74% reduction
```

For a phone or Raspberry Pi, 1.9GB is achievable. 7.3GB is not.

---

## Architecture

```
┌──────────────────────────────────────────────┐
│  Client (attention-only, 1.9GB)              │
│                                              │
│  For each token:                             │
│  1. POST embed_server /v1/embed              │
│     {token_ids} → {residual_0}               │
│                                              │
│  2. For layer in 0..num_layers:              │
│     a. residual = attention(residual)        │
│     b. residual += grid.ffn(residual, layer) │
│                                              │
│  3. POST embed_server /v1/logits             │
│     {residual_final} → {top_k_tokens}        │
└──────────┬──────────────────┬────────────────┘
           │                  │
           ▼                  ▼
┌────────────────┐    ┌────────────────────────┐
│  Embed Server  │    │      FFN Grid          │
│                │    │                        │
│  embeddings    │    │  layer-sharded shards  │
│  lm_head       │    │  self-assembling       │
│  tokenizer     │    │  no GPU                │
│                │    │                        │
│  2.7GB mmap    │    │  ~11GB per shard       │
│  pure lookup   │    │  pure lookup           │
└────────────────┘    └────────────────────────┘
```

Three network calls per token:
1. Embed (token_ids → residual_0)
2. FFN grid (residual per layer, batched)
3. Logits (residual_final → top_k)

The embed calls are trivially fast — pure table lookup and one matmul
against a static matrix. Latency is dominated by the FFN grid call.

---

## Embed Server API

### POST /v1/embed

Convert token IDs to initial residual vector.

**Request:**
```json
{
  "token_ids": [1, 5432, 235, 1234],
  "seq_len": 4
}
```

**Response:**
```json
{
  "residual": [[f32 × hidden_size], ...],
  "seq_len": 4,
  "hidden_size": 5376,
  "latency_ms": 0.1
}
```

Wire format: binary by default (same codec as FFN grid).
Each embedding vector is hidden_size × f32 = 5376 × 4 = 21.5KB per token.
For seq_len=1 (decode): 21.5KB request payload.

**Implementation:** direct mmap index into `embeddings.bin`.
No compute. One pointer offset per token_id.

---

### POST /v1/logits

Project final residual through lm_head to get token probabilities.

**Request (JSON):**
```json
{
  "residual": [f32 × hidden_size],
  "top_k": 5,
  "temperature": 1.0
}
```

**Request (binary, Content-Type: application/x-larql-ffn):**
```
[f32 × hidden_size]  — final residual, one position
```

**Response:**
```json
{
  "top_k": [
    {"token_id": 9515, "token": "Paris", "prob": 0.801},
    {"token_id": 235,  "token": "the",   "prob": 0.042},
    ...
  ],
  "latency_ms": 2.1
}
```

Or raw logits mode for beam search / sampling:

**Response (binary):**
```
[f32 × vocab_size]  — full logit vector
```

**Implementation:** single matmul — `residual @ lm_head.T`.
For Gemma 4 31B: `[5376] @ [262208 × 5376]` = 262208 dot products.
On CPU: ~2ms. On Metal: ~0.1ms.

This is the Metal lm_head work already done on the local path —
same kernel, now exposed as a server endpoint.

---

### GET /v1/embed/{token_id}

CDN-cacheable single-token embedding lookup.

```
GET /v1/embed/9515
→ [f32 × hidden_size]  (binary, 10 KB for hidden=2560)

GET /v1/embed/9515  (Accept: application/json)
→ {"token_id": 9515, "embedding": [f32, ...], "hidden_size": 2560}
```

Response headers:
```
Cache-Control: public, max-age=31536000, immutable
Content-Type: application/x-larql-ffn
Vary: Accept
```

The token_id is a 32-bit integer key; the embedding is a deterministic
function of the model weights. Responses are immutably cacheable — a CDN
can serve repeated decode-step lookups for high-frequency tokens (the, a,
in, …) without the request reaching the embed server at all.

Implemented. Binary by default; `Accept: application/json` for human-readable.

---

### GET /v1/token/encode

```
GET /v1/token/encode?text=Paris
→ {"token_ids": [9515], "text": "Paris"}
```

---

### GET /v1/token/decode

```
GET /v1/token/decode?ids=9515,235,1234
→ {"text": "Paris the model"}
```

Useful for clients that don't want to bundle the tokenizer locally.

---

### GET /v1/stats

```json
{
  "model": "google/gemma-4-31B-it",
  "hidden_size": 5376,
  "vocab_size": 262208,
  "embed_size_gb": 2.7,
  "lm_head_tied": true,
  "mode": "embed-service",
  "loaded": {
    "embeddings": true,
    "lm_head": true,
    "tokenizer": true
  },
  "memory_mb": 5400
}
```

---

## CLI

```bash
# Start embed server
$ larql-server output/gemma4-31b-q4k.vindex \
    --embed-only \
    --port 8082 \
    --host 0.0.0.0

# Output:
LARQL Embed Server v0.4.1
  Model:      google/gemma-4-31B-it
  Vocab:      262,208 tokens
  Hidden:     5,376
  Embeddings: 2.7 GB  (mmap)
  lm_head:    2.7 GB  (tied, mmap)
  Tokenizer:  loaded
  Mode:       embed-service
  Listening:  http://0.0.0.0:8082
  Ready.
```

```bash
# Client — attention-only mode with remote embed + FFN grid
$ larql-cli predict \
    --model google/gemma-4-31B-it \
    --vindex output/gemma4-31b-q4k.vindex \
    --embed grpc://embed-server:8082 \
    --ffn grpc://router:50051 \
    --attention-only \
    --prompt "The capital of France is"
```

---

## Vindex Slice: embed

New slice type for `larql slice` and `larql publish`:

```
embed slice contents:
  embeddings.bin     (vocab × hidden, f16)
  lm_head.bin        (same as embeddings if tied, symlink or copy)
  tokenizer.json
  index.json         (model metadata only)
```

Size estimates:

```
Model              embed slice
───────────────────────────────
Gemma 3 4B         1.3 GB
Gemma 4 31B        2.7 GB
Llama 3 70B        2.1 GB
Kimi-K2 1T         ~2.3 GB
```

---

## Grid Registration

The embed server joins the grid the same way FFN servers do —
via the gRPC `GridService.Join` stream. It announces a different
capability:

```protobuf
message AnnounceMsg {
  string model_id    = 1;
  string listen_url  = 2;
  string capability  = 3;  // "ffn" | "embed" | "full"
  uint32 layer_start = 4;  // 0 for embed servers (ignored)
  uint32 layer_end   = 5;  // 0 for embed servers (ignored)
  uint64 ram_bytes   = 6;
}
```

---

## Client Forward Pass (attention-only mode)

```rust
pub async fn predict_attention_only(
    &self,
    token_ids: &[u32],
    embed_backend: &RemoteEmbedBackend,
    ffn_backend: &RemoteWalkBackend,
) -> Result<Vec<TokenProb>> {

    // 1. Get initial residual from embed server
    let mut residual = embed_backend.embed(token_ids).await?;

    // 2. Attention + remote FFN for each layer
    for layer in 0..self.num_layers {
        // Local attention (weights are resident)
        residual = self.run_attention(residual, layer)?;

        // Remote FFN (batched in practice)
        let delta = ffn_backend.walk_layer(residual, layer).await?;
        residual = residual + delta;
    }

    // 3. Get logits from embed server
    let top_k = embed_backend.logits(&residual, 5).await?;

    Ok(top_k)
}
```

---

## Memory Profile

```
Mode                    Client RAM    Servers needed
────────────────────────────────────────────────────
Full local              7.3 GB        none
Remote FFN              4.6 GB        FFN grid
Remote FFN + embed      1.9 GB        FFN grid + embed server
Attention-only client   1.9 GB        FFN grid + embed server
```

---

## Measured Performance (Gemma 3 4B, M-series Mac, release build)

Benchmarked via `cargo run --release -p larql-server --example bench_embed_server`.

### Load time

```
Component               Time      RSS after load
─────────────────────────────────────────────────
Baseline                —           3 MB
Tokenizer               ~690ms    244 MB   (HuggingFace BPE, 262K vocab)
embeddings.bin (f16→f32) ~1165ms  2833 MB   (1.34 GB f16 → 2.69 GB f32)
─────────────────────────────────────────────────
Total startup           ~1.9s    ~2.9 GB RSS
```

Throughput: 1.15 GB/s read + decode from disk (f16→f32 path).

### Embed lookup (per-request)

```
Operation                  Latency        Throughput
──────────────────────────────────────────────────────
Single token — row access  0.7 ns/op      1.4B ops/s  (pure pointer dereference)
Single token — Vec copy    1.6 µs/op      611K ops/s  (10 KB memcpy + scale)
Prefill 32 tokens          87 µs/op       11K ops/s
Prefill 128 tokens         297 µs/op      3.4K ops/s
Prefill 512 tokens         1.37 ms/op     730 ops/s
```

Embed lookup is **O(seq_len × hidden)** — pure memcpy + scalar multiply. No
computation. The 1.6 µs single-token cost is dominated by 2560 × 4 = 10 KB
memory bandwidth at ~6 GB/s.

### Tokenizer

```
Operation               Latency       Throughput
──────────────────────────────────────────────────
Encode 1 word           2.9 µs/op     348K ops/s
Encode 5 words          5.2 µs/op     191K ops/s
Encode 15 words         9.5 µs/op     105K ops/s
Decode 1 token id       617 ns/op     1.6M ops/s
Decode 5 token ids      1.9 µs/op     531K ops/s
```

### Binary wire format

```
Operation                            Latency
──────────────────────────────────────────────
Encode embed request (1 token)       17 ns
Encode embed request (512 tokens)    243 ns
Decode embed request (1 token)       18 ns
Encode embed response (1×2560 f32)   1.5 µs
Encode logits request (2560 f32)     306 ns
```

### JSON vs binary — embed response

```
Format          Latency      Size (1×2560 floats)
──────────────────────────────────────────────────
Binary          1.5 µs       10.2 KB  (exact f32 bytes)
JSON            10.1 µs      ~30 KB   (float text repr)
──────────────────────────────────────────────────
Binary speedup  6.7×         3× smaller
```

Use binary (`Content-Type: application/x-larql-ffn`) for the embed endpoint
on the hot decode path. JSON is fine for logits (one call per token, 0.5 µs).

### Logits projection (lm_head matmul)

```
Config          Latency      Notes
──────────────────────────────────────────────────
CPU naive       ~336ms       262208 × 2560 dot products
BLAS gemv       ~14ms        @ ~50 GFLOP/s
Metal gemv      ~0.67ms      @ ~2 TFLOP/s  (Apple Silicon)
```

The Metal path (`f32_gemv` on `ComputeBackend`) is already implemented in the
local lm_head (layer_graph/generate.rs). The embed server reuses the same
`logits_to_predictions_pub` call which dispatches via the same backend.

### Memory footprint — Gemma 3 4B

```
Mode                        RSS       What's loaded
──────────────────────────────────────────────────────────────────────────
--embed-only (f16 store)    ~1.5 GB   tokenizer (244 MB) + embeddings f16 mmap (1.34 GB)
--embed-only (f32 fallback) ~2.9 GB   tokenizer (244 MB) + embeddings f32 heap (2.69 GB)
--ffn-only                  ~3.6 GB   gate_vectors + interleaved_q4k + attn + norms
full                        ~6.3 GB   all of the above
```

**f16-at-rest store (implemented):** `EmbedStoreF16` mmaps `embeddings.bin`
as raw f16 bytes (1.34 GB) and decodes per-lookup. An L1 hot-vocab cache
(5 000 entries, ~50 MB) holds the top-N tokens as f32; the first 5 000 tokens
accessed are cached forever. On Gemma 3 4B this cuts embed-server RSS from
~2.9 GB to ~1.5 GB. Falls back to the f32 heap copy if the file is f32-encoded.

---

## Latency Budget (full pipeline)

```
Operation               Latency    Notes
──────────────────────────────────────────────────────
embed call (binary)     ~1.6µs     row copy, seq_len=1 (decode step)
embed call (binary)     ~87µs      seq_len=32 (short prefill)
attention (per layer)   ~0.3ms     local, Q4K dequant
FFN grid (34 layers)    ~58ms      one batched round trip
logits call (Metal)     ~0.67ms    f32_gemv on Apple Silicon
──────────────────────────────────────────────────────
Total per token         ~62ms      ~16 tok/s (FFN grid dominates)
```

The embed and logits calls are negligible vs FFN grid latency. The bottleneck
is network RTT + FFN compute. When the speculation error experiment proves
parallel layer walks, FFN grid drops from ~58ms to sub-10ms — at that point
embed + logits overhead (~2ms total) becomes the next target.

---

## Implementation Plan

### Phase 1 — Embed server (2 days)

- `--embed-only` flag on `larql-server`
- Skip attention weights and FFN weights at load time
- `POST /v1/embed` endpoint — mmap lookup into embeddings.bin
- `POST /v1/logits` endpoint — reuse Metal lm_head kernel
- `GET /v1/token/encode` and `/decode`
- `GET /v1/stats` with embed-service mode

### Phase 2 — Client attention-only mode (2 days)

- `RemoteEmbedBackend` in `larql-inference`
- `--embed URL` flag on `larql-cli predict`
- `predict_attention_only` forward pass
- Skip loading embeddings.bin and lm_head locally

### Phase 3 — Grid registration (1 day)

- `capability` field in `AnnounceMsg`
- Router maintains embed server registry
- Router proxies `/v1/embed` and `/v1/logits` to registered embed server
- Client uses single router endpoint for both services

### Phase 4 — Embed slice (1 day)

- `embed` preset in `larql slice`
- `larql publish --slices embed` support
- Model card template for embed repos

### Phase 5 — Token cache (1 day)

- Top-1000 token cache in embed server process
- Benchmark hit rate on natural language decode

---

## Open Questions

1. **Tied weights.** Most modern models tie embedding and lm_head weights.
   If tied, `lm_head.bin` is a symlink to `embeddings.bin` — no extra
   storage. If not tied (some fine-tuned variants), lm_head is a separate
   file. The server handles both; `index.json` declares `lm_head_tied: bool`.

2. **Batch embed for prefill.** During prefill, all token embeddings are
   needed at once. One call with `seq_len=N` returns N residuals. The
   server handles this as N parallel mmap lookups — trivially fast.

3. **KV cache interaction.** If the client holds a KV cache for decode,
   the embed server is called once per new token only. The KV cache stays
   local. No interaction.

4. **Streaming decode.** For streaming generation, the client calls embed
   once for the prompt, then once per generated token. The hot token cache
   means most decode-step embed calls return in microseconds.

5. **Multi-model embed server.** One embed server can serve multiple models
   if they share a vocabulary (e.g. all Gemma 4 variants use the same
   tokenizer). The server loads one embeddings.bin per model. Routing by
   `model_id` in the request header.
