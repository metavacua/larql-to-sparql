# ADR-0006 — Q4_K Dense-Remote FFN Path

**Status:** Implemented
**Depends on:** ADR-0002 (Activation Cache), ADR-0005 (Memory Bounds)

---

## Context

ADR-0002 established `RemoteWalkBackend`: a client POSTs a residual to
`/v1/walk-ffn`, the server runs the architecture-correct walk, and returns
the FFN output. That landed for float vindexes (`extract --quant none`).

On a quantised vindex (`extract --quant q4k`), both ends failed silently:

- **Server side.** `get_or_load_weights` called `load_model_weights_with_opts`
  which hard-rejects Q4_K (`"vindex is quantised (q4k) — call load_attn_q4k +
  load_interleaved_q4k instead"`). The handler returned HTTP 503. But that
  only fired for full-output requests; features-only requests succeeded.

- **Client side.** `larql run --ffn URL` short-circuited into `run_predict_q4k`
  before checking `ffn_remote`. The q4k path runs a fully-local forward,
  dequantising every layer's attention *and* FFN per step. The `--ffn` flag
  was silently ignored — the client loaded the FFN weights locally, computed
  the forward locally, and never hit the server. Log output said
  `Backend: CPU (Accelerate + dequantise-per-layer)` with no hint that a
  remote URL had been given.

The Q4_K path is the interesting one — it's the configuration that lets a
31B model fit in 8 GB RSS on a laptop (ADR-0005). Making the demo filmable
required both ends to work with quantised vindexes.

---

## Decision

Treat quantised FFN as a separate forward-pass layout, symmetric on both
ends: each side dequantises the pieces it owns, one layer at a time.

- **Client:** local attention (dequant per layer from `attn_weights_q4k.bin`),
  remote FFN (residual over HTTP, no local FFN weights).
- **Server:** no attention weights, local FFN (dequant per layer from
  `interleaved_q4k.bin`).

Eagerly materialising the full model as f32 is not viable — 31B Q4_K
(~33 GB on disk) expands to ~127 GB of f32. Per-layer dequant keeps
working-set at ~1.8 GB per side per layer (the 31B down_proj is the
largest matrix).

---

## Architecture

```
Client (laptop)                          Server (--ffn-only)
─────────────────                        ─────────────────────
load_model_weights_q4k                   load_model_weights_q4k
  + attn_weights_q4k.bin mmap            load_interleaved_q4k.bin mmap
  (no FFN weights loaded)                (no attn weights loaded)
           │
           │ for each layer:
           │   1. dequant Q/K/V/O locally
           │   2. run attention on residual
           │   3. POST /v1/walk-ffn (residual, layer, full_output: true)
           │        ────────────────────────────────►
           │                                         │ for each requested layer:
           │                                         │   1. dequant gate/up/down
           │                                         │   2. apply activation gate
           │                                         │   3. down projection
           │                                         │   ← return FFN output
           │        ◄────────────────────────────────┘
           │   4. add to residual
           │   5. drop the layer's dequanted tensors
```

Per forward pass, 60 HTTP round trips (one per layer). On localhost the
round trip is dominated by CPU dequant time on the server; on a LAN it
becomes RTT-bound — exactly the profile ADR-0003 (router) is designed to
improve via batching.

---

## Client Path

`crates/larql-inference/src/vindex/q4k_forward.rs::predict_q4k_with_ffn`
mirrors the existing `predict_q4k` but delegates the FFN step to any
`FfnBackend` — typically `RemoteWalkBackend`.

Differences from `predict_q4k`:

| Step | `predict_q4k` (local) | `predict_q4k_with_ffn` (remote) |
|---|---|---|
| Load | embed + norms via `load_model_weights_q4k` | same |
| Attn Q/K/V/O | dequant per layer from q4k mmap, insert into `weights.tensors` | same |
| FFN gate/up/down | dequant per layer, insert into `weights.tensors` | **skip** |
| Layer forward | `run_layer_with_ffn(..., WeightFfn { weights })` | `run_layer_with_ffn(..., &remote_backend)` |
| Cleanup | remove Q/K/V/O *and* FFN tensors after layer | remove Q/K/V/O only |
| Peak heap | ~1.8 GB/layer (attn + FFN) | ~0.4 GB/layer (attn only) |

`crates/larql-cli/src/commands/extraction/walk_cmd.rs::run_predict_q4k_remote`
is the CLI glue. It connects to the remote URL via `RemoteWalkBackend`,
builds a fresh `VectorIndex` with only the attention Q4_K mmap loaded
(deliberately omitting `load_interleaved_q4k` — the FFN lives on the
server), and calls `predict_q4k_with_ffn`.

The output label is `walk (q4k + ffn remote)`. If a user sees `walk (q4k)`
after passing `--ffn`, that's the old silent-fallback bug and is a test
regression.

---

## Server Path

`crates/larql-server/src/state.rs::get_or_load_weights` branches on
`config.quant == QuantFormat::Q4k`:

```rust
let weights = if self.config.quant == larql_vindex::QuantFormat::Q4k {
    larql_vindex::load_model_weights_q4k(&self.path, &mut cb)?
} else {
    larql_vindex::load_model_weights_with_opts(&self.path, &mut cb, opts)?
};
```

The Q4_K loader produces a `ModelWeights` with **empty `tensors`** — embed,
norms, and lm_head are loaded, but attention and FFN slots stay uninstalled.
That's fine: the walk-ffn handler never touches attention (the client ran
it), and the Q4_K handler path we added next doesn't use `weights.tensors`
for FFN either.

`crates/larql-server/src/routes/walk_ffn.rs::run_full_output` branches on
the same condition:

```rust
let walk_ffn = if is_q4k { None }
               else { Some(WalkFfn::new_unlimited(weights, &*patched)) };
```

For each requested layer:

```rust
let out = if let Some(ref wf) = walk_ffn {
    wf.forward(layer, &x)                         // float path
} else {
    q4k_ffn_forward_layer(&*weights.arch,         // q4k path
                          patched.base(), layer, &x)
};
```

`q4k_ffn_forward_layer` (new, in `q4k_forward.rs`) takes the architecture
trait object, the underlying `VectorIndex`, the layer index, and the
residual. It:

1. Reads `index.interleaved_q4k_layer_data(layer)` → `[gate, up, down]`
   byte ranges + per-matrix format tags (`"Q4_K"` or `"Q6_K"`).
2. Calls `dequantize_matrix` on each (reusing the existing helper).
3. Applies the architecture's activation via `silu_gate_up` or
   `gelu_tanh_gate_up` (picked from `arch.activation()`).
4. Returns `down @ activation`.

No allocations outside the three dequantised matrices. The caller drops
the output and moves on; the dequant is redone on the next request for
the same layer. For the demo this is acceptable — the per-layer dequant
(~1.4 GB allocated, ~10 ms of CPU) is smaller than the HTTP round trip.

---

## L2 Cache Interaction

`FfnL2Cache` (ADR-0002) still applies on the q4k server path. The cache
key is derived from gate-KNN feature IDs, which doesn't care about the
weight representation. A hit short-circuits the dequant → FFN pipeline
entirely. A miss populates the cache with the output computed via
`q4k_ffn_forward_layer`.

Patch safety (`has_overrides_at(layer)`) also works unchanged — if any
INSERT patches the layer, the cache is bypassed and a fresh dequant
happens every call.

---

## Measured Parity

Local and remote produce the same argmax on Gemma 4 31B Q4_K:

```
Prompt: "The capital of France is"

local  (walk (q4k)):              Paris  99.36%
remote (walk (q4k + ffn remote)): Paris  99.36%
                                  ───────
                                  identical top-5

client RSS:    8.1 GB       (attn mmap + embed + faulted gate pages)
server RSS:    5.6 GB startup, ~23 GB after req (ADR-0005 bounds apply)
forward pass:  20 s CPU     (dominated by server-side dequant)
```

Latency on localhost is the same as local Q4_K forward (within noise)
because the bottleneck is per-layer dequant, not network.

---

## Implementation Files

| File | Role |
|---|---|
| `crates/larql-inference/src/vindex/q4k_forward.rs` | `predict_q4k_with_ffn`, `q4k_ffn_forward_layer` |
| `crates/larql-inference/src/vindex/mod.rs` | Re-exports |
| `crates/larql-cli/src/commands/extraction/walk_cmd.rs` | `run_predict_q4k_remote`; routes `args.ffn_remote.is_some()` for q4k |
| `crates/larql-server/src/state.rs` | Q4_K branch in `get_or_load_weights` |
| `crates/larql-server/src/routes/walk_ffn.rs` | `is_q4k` branch in `run_full_output` |

---

## Trade-offs

- **Per-request dequant cost.** No layer-level cache on the server's dequant
  output. For single-client demos this is fine; for multi-client steady
  state, a per-layer dequant LRU (parallel to ADR-0005's gate cache) would
  pay back.
- **One layer, one round trip.** The `/v1/walk-ffn` call is per-layer. The
  router (ADR-0003) is where batching should live; making the per-layer
  RPC chattier at this tier would duplicate that effort.
- **No f16 equivalent on q4k.** The path assumes dequant-to-f32. Metal Q4
  shaders exist in `larql-compute` and are wired into `predict_q4k_metal`;
  exposing them to the remote path is a separate ADR (would change the wire
  format to raw quantised blocks, not f32 residuals).

---

## Open Questions

1. **Dequant cache on the server.** Adding `q4k_ffn_cache` (keyed by layer,
   LRU-bounded) would avoid re-dequantising hot layers across requests.
   Parallel to `f16_decode_cache` in ADR-0005. Defer until measured under
   realistic multi-client load.

2. **Metal/GPU q4k path for remote FFN.** Currently CPU-only. The server's
   `WalkFfn::forward` routing ladder has a Q4_K interleaved path gated on
   `backend.has_q4()` (Metal). Extending `q4k_ffn_forward_layer` to use it
   would cut dequant time from ~10 ms to ~1 ms per layer on M-series Macs.
   Needs the GPU gate-KNN crash on 31B (ADR-0005 §3) resolved first.

3. **Wire format.** Today: f32 residual in, f32 FFN output back. For
   LAN-distributed setups the ~5 KB payloads are trivial; for MoE/expert
   fan-out across WAN, quantised residuals (i8 + scales) would help. Out
   of scope here; see ADR-0003 §Wire format discussion.
