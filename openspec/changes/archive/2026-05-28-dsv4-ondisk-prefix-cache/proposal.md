> **Update (2026-05-28, during P3):** investigating the cached forward
> showed the Zero-SWA reuse path needs a dedicated "recompute-mode"
> attention (decouple raw/compressed position + custom causal mask +
> suppress re-compress) — large and correctness-critical. We pivoted to
> **Full-SWA first**: the wire format serializes the *complete* cache
> (incl. `raw`/`pending_cur`), so a prefix hit simply *continues
> prefilling* at `H` via the existing cached forward — transparent by
> construction. Zero-SWA (drop `raw`, recompute the `n_win·L` tail)
> remains a future storage optimization. P1 (serialize) and P2 (store)
> are unchanged in shape; the wire format just gained the `raw`/`pending`
> fields. See `tasks.md` §4.

## Why

DSv4-Flash decode is now at the resident-Q4_K weight-bandwidth floor
(P1–P8 quantized every Q4_K-in-GGUF decode weight). The remaining lever
on the project's agentic / long-context driving goal is **not** raw
decode tok/s but **prefix reuse**: re-running the same long prompt
prefix (system prompt, retrieved documents, prior turns) re-prefills it
from scratch every request. At a 1M-token context that prefill is the
dominant cost.

The DeepSeek-V4 paper (§3.5.2) addresses exactly this with an on-disk KV
cache for shared-prefix requests: store the **compressed** CSA/HCA KV
entries to disk and, on a prefix hit, read them back instead of
recomputing. For the uncompressed sliding-window (SWA) entries it offers
three strategies; the most storage-efficient — **Zero-SWA** — stores no
SWA entries at all and reconstructs the SWA tail by recomputing only the
last `n_win·L` tokens (for DSv4-Flash, 128·43 ≈ 5 504 tokens) regardless
of prefix length, because each layer's SWA KV depends only on the most
recent `n_win` tokens of the previous layer.

This is the same direction as the recorded oMLX SSD cold-tier reference
and the VibeServe hybrid-prefix-cache roadmap item (#205), and it serves
the driving goal (CPU-FFN / GPU-attention, models > VRAM, SSD tier).

larql has **no** persistence for the DSv4 KV cache today
(`DsV4LayerHcaCache` is in-memory only) and the server's existing
kv-snapshot path is not DSv4-aware. This is greenfield.

## What Changes

Foundation-first, **DSv4-local single-process** (the server-integration
and the Periodic/Full-SWA strategies are explicit non-goals for v1):

- **Serialization wire format** for `DsV4LayerHcaCache` — serialize the
  `compressed` CSA/HCA KV (and the indexer's compressed KV), plus the
  `compress_ratio` and the chunk-`overlap_state`, to a versioned binary
  blob. The `raw` SWA cache and `pending_cur` are **not** serialized
  (Zero-SWA). Round-trips losslessly.
- **Prefix-keyed on-disk store** — a content-addressed store keyed by a
  hash of the prompt token-id prefix at compression-chunk boundaries
  (granularity `lcm(m, m') = lcm(4, 128) = 128` tokens, the paper's
  block size), mapping `prefix_hash → per-layer compressed-KV blobs`.
  Write-through on prefill; longest-prefix lookup on a new request.
- **Zero-SWA prefill reuse** — on a prefix hit, load the compressed KV
  for the hit prefix into each layer's `DsV4LayerHcaCache.compressed`,
  then recompute only the last `n_win·L`-token window through the layers
  to restore the SWA `raw` tail and the `pending_cur`/`overlap_state`,
  rather than re-prefilling the whole prefix.

No existing behavior changes when the cache is disabled (the default):
the prefill path is unchanged. This is purely additive.

## Capabilities

### New Capabilities
- `dsv4-ondisk-prefix-cache`: DSv4-Flash persists its **compressed**
  CSA/HCA KV entries to an on-disk, prefix-hash-keyed store and, on a
  shared-prefix hit, reconstructs the per-layer cache (Zero-SWA: load
  compressed entries + recompute the `n_win·L` SWA tail) instead of
  re-prefilling the prefix. Covers the serialization wire format, the
  prefix-keyed store with chunk-boundary keying, and the Zero-SWA
  prefill-reuse path.

### Modified Capabilities
None. `dsv4-quant-residency` (resident weights) and
`inference-attention-and-kv` (the in-memory cache types) are consumed
unchanged; this change only adds persistence + a reuse entry point.
