# Remote-MoE bottleneck diagnosis (2026-05-29)

CPU remote-MoE decode of Gemma-4-26B-A4B (2 expert shards on localhost, no
`--metal`, `--engine standard`). Goal: find where per-token wall-time goes and
whether the path is network- or compute-bound.

## Method

- Wall-clock differencing to separate the fixed cost (model load + prefill) from
  per-token decode: `(t(max_tokens=9) − t(max_tokens=1)) / 8`.
- Prompt-length differencing to separate model-load from prefill: same
  `max_tokens=1`, a 2-token prompt vs a 32-token prompt.
- Localhost round-trip floor: `curl` to `/v1/health`.
- Honest in-banner timing added: a timestamp per emitted token → TTFT (prefill)
  = start→first-emit; decode = inter-emit intervals.

## Breakdown (per token, 26B, localhost, standard engine)

| Stage | Cost | Notes |
|---|---:|---|
| **Model load** | **~6.8 s** | one-time; before generation. 1.5 GB `embeddings.bin` + Q4K attn/FFN reads + up-front dequant-to-f32 |
| **Prefill (TTFT)** | **~1.4 s** for 33-token prompt | **~42 ms / prompt-token** (attention O(N²) + dense FFN + expert dispatch for all positions) |
| **Decode** | **~6 tok/s = ~165 ms / token** | steady-state; KV-cached (new token only) + 30 sequential expert round-trips |
| Network (localhost) | **~12 ms / token** | 30 layers × ~0.35 ms RTT — **negligible** |

## Findings

1. **Network is NOT the bottleneck on localhost.** A `/v1/health` round-trip is
   0.35 ms; 30 sequential layer round-trips ≈ 12 ms, vs ~165 ms/token decode. The
   `forward_moe_seq` dispatch is one batch POST **per shard in parallel within a
   layer**, but **layers are sequential** (layer L+1 needs L's output) — so it's
   30 serial round-trips/token, each dominated by **server-side expert compute**
   (dequant + top-8 matmul), not wire time.

2. **Decode is compute-bound, ~6 tok/s** — not network-bound. The wire is ~660 KB
   sent+recv per token (tiny). Going faster means faster CPU kernels (client
   attention + dense FFN; server expert dequant/matmul), not fewer round-trips.

3. **Model load (~6.8 s) dominates one-shot / interactive latency.** Amortized to
   zero in a persistent server. Contributors: 1.5 GB embeddings load, Q4K bin
   reads, and the engine path's up-front "dequantize all layers to f32" (so the
   `WeightFfn`-based attention + dense `h1` can run). A Q4K-direct attention path
   (à la `prefill_quant`) or lazy/mmap embeddings would cut this.

4. **The old "decode: X tok/s" banner was misleading** — it synthesized
   `decode_ms` from *total* time (load + prefill + decode) / n, so for short runs
   it reported ~3–4.4 tok/s when **true steady-state decode is ~6 tok/s**. Fixed:
   the CLI now times TTFT (prefill) and decode (inter-token) separately. The
   per-engine tok/s figures recorded earlier in the larql-kv roadmap were that
   conflated `total/n` number — they rank the engines correctly but **understate
   absolute decode and compress the spread** (the constant load term dominates).

## Implications by deployment

- **Localhost / single box:** compute-bound; the multi-shard split buys parallel
  server expert compute but the client still serializes 30 layers. Optimize CPU
  kernels; the network has huge headroom.
- **Real LAN/WAN:** the 30 *sequential* layer round-trips scale with RTT — at
  10 ms RTT that's 300 ms/token of pure latency, which **would** dominate. There,
  multi-layer pipelining / speculative prefetch (batching across layers) is the
  lever — not relevant on localhost.

## Decode-stage split (measured 2026-05-29, `LARQL_DECODE_STAGES=1`)

Per-stage timers (`decode_stages`), accumulated over prefill + 12 decode tokens
on the 26B (standard engine, 2 shards, localhost). Full 4-way client/server split:

| Stage | Time | Share | Side |
|---|---:|---:|---|
| **remote experts** (`forward_moe_seq`: server dequant+matmul + wire) | **1539 ms** | **~41%** | server |
| **attention** (`backend.attention_*`, f32 BLAS) | **1051 ms** | **~28%** | client f32 |
| client dense FFN (`h1`, f32 `run_ffn` via `WeightFfn`) | 471 ms | ~13% | client f32 |
| lm_head (`logits_to_predictions_pub`, f32 vocab proj) | 446 ms | ~12% | client f32 |
| everything else (router, combine, embed, norms) — by subtraction | ~190 ms | ~5% | client |

Reading it:
- **~53% of decode is recoverable client-side f32 compute** (attention 28 + dense
  13 + lm_head 12), all on the dequant-to-f32 BLAS path. **Attention is the #1
  target**, not dense FFN.
- **~41% is server-side expert compute** (Q4K dequant + top-8 matmul, parallel
  across 2 shards; localhost wire negligible) — bandwidth-bound on the shards.
- Ready Q4K-direct kernels for the client stages: **`q4_attention_proj`**
  (`attention/gpu.rs`, Q4K Q/K/V/O via `q4_matvec`, CPU-tested) for attention;
  **`WalkFfn`** (Q4K-direct dense FFN from the index) for `h1`; lm_head needs a
  Q4K vocab-projection path (TBD). Reclaiming all three would also remove the
  ~6.8 s up-front dequant-to-f32 load tax (nothing left to dequantize).
- So the path to faster decode is **both**: (a) Q4K-direct client kernels
  (attention → dense → lm_head, ranked by win) to reclaim the ~53% f32 tax, and
  (b) reduce the server expert cost (more shards / FP4 / hash-routing) for the ~41%.

> ⚠️ **Retracted 2026-08-22 — do not implement step 1 as written.** Both of
> its named pieces were falsified by `q4k-direct-attention.md`, which is the
> follow-up study to this one:
> - **`q4_attention_proj` must NOT be wired.** It feeds Q4_K bytes through
>   the Q4_0 kernel — wrong byte stride, so the output is garbage rather
>   than merely slow. It is still live and exported
>   (`larql-compute/src/attention/gpu.rs`, re-exported from
>   `attention/mod.rs`), so nothing stops someone following this line.
> - **`run_attention_with_kv_backend` is the wrong function for the 28%.**
>   That is the prefill / KV-populate path, not decode; replacing it
>   optimises a stage this diagnosis did not measure.
>
> `crates/larql-kv/ROADMAP.md` still repeats the same recommendation and
> needs the same annotation. What survived from the Q4-direct programme is
> narrow: it wins only in bandwidth-bound decode-matvec, and the follow-up
> explicitly declines to headline a tok/s number for it.

**Implementation sequence for item 5b (ranked by client win):**
1. **Attention (28%)** — ⚠️ **superseded, see the retraction above.** As
   originally written: a Q4K-direct path reading attn bytes from the index via
   `q4_attention_proj`, replacing the f32 `run_attention_with_kv_backend`. Biggest
   win + (with the others) kills the dequant-all load tax. Parity-critical rework
   of the attention path → verify byte-parity on the 26B before flipping.
2. **dense FFN (13%)** — ⚠️ **`WalkFfn` is the wrong kernel** (tried 2026-05-29,
   reverted): its `forward` always routes through `walk_ffn_sparse` (per-position
   gate-KNN walk), so even "dense" config ran **~8.5× slower** than f32 BLAS
   (dense FFN 331 → 3986 ms, decode 7.4 → 2.0 tok/s). The genuine Q4K-direct dense
   kernel is **`kquant_ffn_forward_layer_q8k`** (NEON Q4K×Q8K, no dequant) — needs
   a thin `FfnBackend` wrapper. **Low ROI though:** f32 BLAS dense FFN is already
   competitive and it's only ~13%; deprioritized below attention.
3. **lm_head (12%)** — Q4K vocab projection from the loaded lm_head Q4K bytes
   (per-token cost, so pure decode win).

## 80 tok/s gap (12.5 ms/token vs ~127 ms/token today)

- **~4× from client kernels**: Q4K-direct attention/FFN/lm_head removes most of
  the ~60% f32 tax → roughly 7.9 → ~20-25 tok/s.
- **Then the DDR5 bandwidth wall (~22 tok/s single-box for A4B Q4)**: past this
  needs distributing expert bandwidth (the grid, which raises the per-machine
  ceiling) **and/or** the compounding stack (hash-routing 5× + FP4 2×, V1/V2,
  unproven to compound per ADR-015).
- **On real LAN/WAN**: the 30 *sequential* layer round-trips scale with RTT
  (10 ms → 300 ms/token), which would dominate → needs multi-layer prefetch.

80 tok/s on the **26B** therefore requires the compounding technique stack +
grid + client-kernel work — it is *not* reachable by kernel optimisation alone
(bandwidth caps a single box at ~22 tok/s). For a **4B-class** model it is already
near (Metal 104, CPU 28) because per-token bytes are ~6× smaller.

## Actionable next steps (not yet done)

1. **Re-measure per-engine decode with the honest banner** and update the
   larql-kv roadmap table (true decode, not `total/n`).
2. **Cut model load**: lazy/mmap embeddings; avoid the up-front dequant-all by
   giving the engine a Q4K-direct attention path (reuse `prefill_quant`'s
   tensors) instead of resident f32.
3. **Split decode client-vs-server** with per-stage instrumentation (attention /
   dense-FFN / expert-dispatch timers) to know whether the client local compute
   or the server expert compute dominates the 165 ms.
4. **For LAN/WAN**: multi-layer expert prefetch to hide sequential round-trips.
