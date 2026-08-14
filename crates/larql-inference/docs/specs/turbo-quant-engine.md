# TurboQuantEngine — Specification

**Status:** ✅ Shipped. W2 (hot K/V cache) + W1-GPU step 6 wired
2026-05-17. Decode 19.6 → 33.0 tok/s on Metal post-W1-GPU.
Append-only codec path across all decode paths (dispatch + CPU +
legacy) 2026-05-21; 85.0 tok/s on Gemma 3 4B Q4K, 50-token decode.
Executor-routed decode path made append-only + unit-sigma
codebooks (per-block-dim scaling) 2026-07-30.
**Audience:** LARQL contributors.

> **State Policy slot**: `(canonical = quantised K/V (in-place,
> destructive codec), derivative = ∅, contract =
> codec_bounded_state — codec round-trip ≥ cos 0.995 at 4-bit;
> output KL observed, not bounded)`. K/V is **canonical** here, not
> derivative — the codec is destructive, so the engine can't drop
> the K/V shadow under W10. Result: 85.0 tok/s vs `standard`'s
> 97.6 (-12.9%). This is the empirical evidence that
> canonical-vs-derivative is the dominant tok/s lever on the
> trait — same compression contract as `MarkovResidualCodecEngine`,
> different §2 slot, 13% perf gap.
>
> **Task #31** proposes a derivative-K/V `TurboQuant` sibling
> (residuals canonical, quantised K/V reconstructable on demand).
> If the prediction holds, the new engine joins
> `MarkovResidualCodecEngine` et al. at 98+ tok/s. If it doesn't,
> [state-policy.md §3.1](../../../larql-kv/docs/state-policy.md)'s
> claim is falsified and the W10 perf shape needs a different
> explanation.

---

## 1. Purpose

`TurboQuantEngine` is a compressed-K/V engine: every K/V row is
stored as a WHT (Walsh-Hadamard Transform) rotation + Lloyd-Max
scalar quantisation at 3 or 4 bits per coordinate, plus a stored
scalar norm. The codec is stateless (deterministic transform of
each vector) and gives ~3.9× compression at 4 bits vs raw f32 K/V
with per-row round-trip cosine ≈ 0.9954 at 4 bits (Gaussian
simulation with the unit-sigma codebooks, 2026-07-30; the
mis-scaled pre-2026-07-30 codebooks measured ≈ 0.991).

Use case: long-context decode on memory-constrained machines where
the K/V cache is the binding constraint. `TurboQuantEngine`'s hot
state on Gemma 3 4B (window=512, 34 layers) is ~0.6 MB vs
`markov_residual`'s 10.8 MB — 18× smaller — for similar accuracy.

The engine is **lossy by design**. Don't use it where bit-identity
to `StandardEngine` is required. See §2.1 for the actual contract.

---

## 2. Contract

### 2.1 Accuracy contract — `codec_bounded_state`

> For any K/V row `(k, v)`, the codec round-trip
> `decode(encode(k))` satisfies `cosine(k, decoded_k) ≥ 0.995` at
> 4 bits on unit-norm vectors with random direction and block dim
> ≥ 64; `≥ 0.982` at 3 bits. (Per-vector variance at block dim 32
> lowers the pinned single-vector floor there to 0.975.) Mean
> round-trip cosine measured 2026-07-30 with the unit-sigma
> codebooks: 0.9954 at 4 bits, 0.9830 at 3 bits, stable across
> block dims 32–256.

This is the `codec_bounded_state` contract kind
([state-policy.md §2.3](../../../larql-kv/docs/state-policy.md)):
a **per-row state-distortion** bound, not a hidden-state or output
bound. End-to-end accuracy (KL on the output distribution,
hidden-state cosine to `StandardEngine`) is observed but not
bounded by spec — the engine's identity is "WHT + Lloyd-Max with
these bit-widths," not "any encoding satisfying KL ≤ X." That is
why the slot is not `bounded_KL`: `bounded_KL` requires an
output-side ε calibrated on a corpus (as `markov-rs-codec` does),
which this engine has never earned.

### 2.2 Memory contract

> Hot state: `O(num_layers × num_kv_rows × bytes_per_row)` where
> `bytes_per_row = num_kv_heads × (4 (stored norm) + ceil(head_dim
> × bits / 8))` (norm + packed indices per head).

Measured on the 10-token bench (Gemma 3 4B, window=512, bits=4):
**0.6 MB hot**, cold tier unused at this prompt length — 18× less
than `markov_residual`'s 10.8 MB. Compression ratio vs raw f32 K/V
is ~3.9× at 4 bits / ~5.0× at 3 bits.

### 2.3 Per-layer codec cycle (append-only)

Every decode step, for every layer:

1. **Decompress** the layer's stored `CompressedLayer` →
   `(K_prior, V_prior)` (full prior K/V as f32) for attention.
2. **Encode only the new K/V row** head-by-head and **append** its
   packed bytes onto the existing compressed buffers
   (`CompressedLayer::append_row`).

Old rows' bytes are never re-encoded. This is a correctness
invariant, not just a perf one: `encode` stores its input's norm,
and reconstruction of a unit vector comes back slightly short of
unit norm, so a decompress → re-encode cycle multiplies every
stored norm by that ratio and compounds it across steps (the
executor-routed decode path violated this until 2026-07-30 —
pinned by `executor_decode_is_append_only_first_row_stable`).
The per-step decompress is the remaining inner-loop codec cost.

---

## 3. Codec

WHT (Walsh-Hadamard rotation) over head_dim followed by Lloyd-Max
scalar quantisation. The rotation step disperses the input's energy
across coordinates so per-coordinate quantisation has nearly-uniform
error.

| Operation | Implementation |
|---|---|
| WHT | `crates/larql-kv/src/engines/turbo_quant/rotation.rs` — in-place butterfly, O(d log d) |
| Codebook | `engines/turbo_quant/codebooks.rs` — one unit-sigma (N(0,1)) Lloyd-Max table per bit-width, scaled by 1/√d per block dim at encode/decode (2026-07-30; replaces the per-(dim, bits) tables, whose sigma was √2 too small and which silently fell back to the D256 table for d ∉ {128, 256}) |
| Bit packing | `engines/turbo_quant/packing.rs` — 3-bit and 4-bit variants |

The codec is **scalar f32** today; SIMD vectorisation (NEON/AVX2)
of the rotation step is a P1 follow-up — would close most of the
remaining gap between TurboQuant (33 tok/s) and the cached-K/V
engines (58 tok/s).

---

## 4. W1-GPU integration (2026-05-17)

`try_prefill_via_dispatch` + `decode_step_via_dispatch` route per-
layer compute through the Metal fused kernel via
`KvDispatch::coarse_*_with_state`. State capture gives per-layer
new K/V rows directly; the engine handles the compression cycle
on CPU after the Metal kernel commits.

Bench result: **19.6 → 33.0 tok/s on Metal (+68%)**. The smaller
speedup vs markov_residual (+115%) reflects the codec encode/decode
work in the inner loop that markov_residual doesn't pay.

---

## 5. Implementation

| Concern | Location |
|---|---|
| Engine + `KvEngine` impl | `crates/larql-kv/src/engines/turbo_quant/engine.rs` |
| `TurboQuant` codec | same file, `pub struct TurboQuant { bits: u8 }` |
| `CompressedLayer` | same file — per-layer compressed K + V bytes |
| W1-GPU dispatch helpers | `engine.rs::try_prefill_via_dispatch` + `decode_step_via_dispatch` |

---

## 6. P1 follow-ups (from ROADMAP)

- ~~**Incremental encode of the new K/V row only** (W3)~~: shipped —
  every decode path is append-only (§2.3); the executor-routed path
  was the last holdout (2026-07-30).
- **SIMD WHT + Lloyd-Max** (W4): scalar f32 today. NEON on Apple
  Silicon, AVX2 on x86_64. ~2-4× on the codec step.
- **Compressed-domain attention** (research): WHT preserves dot
  products under rotation; Q @ K can be computed in the WHT domain
  without decompressing the full prior K/V. Skips ~80% of the
  decompress work.

---

## 7. Non-goals

- **Bit-identity to `StandardEngine`.** The engine is lossy by
  design. Don't claim "exact under contract" — `TurboQuant`'s
  contract is the per-row cosine bound in §2.1.
- **Stateful codec (e.g. predictive coding).** Each K/V row is
  encoded independently; no inter-row prediction. This makes the
  decompress path O(1) per row instead of O(seq_len).
- **Cold tier.** TurboQuant compresses the *hot* K/V in-place;
  there is no separate cold-tier eviction. For windowed bounded
  memory + compressed cold tier, use `markov_residual_codec`.
