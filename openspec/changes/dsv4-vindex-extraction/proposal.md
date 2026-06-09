## Why

`larql-server` serves models **from a vindex** (the project's weight
format: gate vectors, Q4_K/Q6_K shards, embeddings, tokenizer). To serve
DeepSeek-V4-Flash through the standard vindex path — rather than the
GGUF-direct route scoped in `dsv4-server-serving` (#392) — DSv4 needs a
vindex extraction: GGUF → vindex. There is none today; `larql-vindex`
has no `deepseek_v4` handling, and the capabilities gate doesn't even
recognize V4.

A gap analysis (2026-05-28) shows the vindex format does **not** model
DSv4's architecture and must be extended:

- The generic attention writer assumes a 4-tensor **Q/K/V/O** layout.
  DSv4 has **low-rank Q** (`attn_q_a`/`attn_q_b`), a **latent KV**
  (`attn_kv_latent`), and a **grouped low-rank O** (`attn_output_a/b`,
  g=8) — net-new attention storage.
- **HCA** (per-`m`/`m'` KV compression + sliding window), the **lightning
  indexer** (top-k sparse selection), and **mHC** (4-stream
  manifold-constrained hyper-connections) have **no vindex
  representation** at all — net-new metadata + weight storage.
- Per-layer **attention-variant dispatch** (compress_ratio 0/1/4) has no
  field in `VindexLayerInfo`.
- Only the **routed MoE experts + shared expert** map cleanly onto the
  existing generic MoE extraction.

So this is a **format extension**, not a reuse of the generic Q4_K path.
Realistic size: ~700–1000 LoC + tests, phased.

(Alternative on record: `dsv4-server-serving` (#392) serves DSv4 directly
from its GGUF, reusing the already-built resident-Q4_K engine + on-disk
prefix cache with no format work. This change is the architecturally
"pure" route — DSv4 in the same format as every other served model — at
materially higher cost. The phasing below front-loads the decisive
unknowns so the investment can be re-evaluated after Phase 1.)

## What Changes

`deepseek_v4`-only, additive — no existing arch's extraction changes.

- **`DeepSeekV4Arch` extraction methods**: flesh out the stub
  (`larql-models`) so the extraction pipeline can resolve DSv4's tensor
  keys (low-rank Q/KV, grouped O, HCA compressor, indexer, mHC, hash
  router) — the per-arch hook the generic pipeline already calls.
- **`VindexModelConfig` + `VindexLayerInfo` extensions**: DSv4 metadata —
  `compress_ratio` per layer, `n_hc`, indexer dims, FP8-KV flag, YARN/
  partial-RoPE config.
- **New DSv4 attention storage**: a `attn_weights_dsv4.bin` (+ manifest)
  for low-rank Q/KV + grouped O, since the Q/K/V/O writer can't represent
  them.
- **HCA + indexer + mHC weight extraction**: Q4_K writers + manifests for
  the compressor (`attn_compress_*`), indexer (`indexer.*`), and the mHC
  bookends (`hc_*`). Indexer top-k stays runtime-dynamic (weights only,
  no precomputed masks).
- **Routed/shared MoE**: reuse the generic MoE extraction (the one part
  that already fits), incl. the first-3-layer hash routing table.
- **Capabilities gate**: recognize `deepseek_v4` as extractable (it is
  not classic MLA — its KV is latent but its serving path differs from
  V2/V3).

## Capabilities

### New Capabilities
- `dsv4-vindex-extraction`: convert a DeepSeek-V4-Flash GGUF into the
  larql vindex format — extending the vindex format + extraction pipeline
  to represent DSv4's low-rank/latent attention, grouped output
  projection, HCA compressor, lightning indexer, mHC residual streams,
  and hash/sqrt-softplus MoE, alongside the existing generic MoE
  extraction. Produces a vindex a (future) DSv4 vindex reader can serve.

### Modified Capabilities
None functionally — extraction for existing arches is unchanged. The
vindex format gains DSv4-only fields/files that other arches ignore.
