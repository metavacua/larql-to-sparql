# larql-boundary

Confidence-gated BOUNDARY ref codec for [LARQL](https://github.com/chrishayuk/chuk-larql-rs).

Transforms transformer final-layer residuals into compact, contract-bearing protocol
objects. Compressed when the boundary is confident. Exact when fragile.

```
KV cache for the present.
Residual boundaries for memory.
```

---

## What it does

A transformer running over a long document produces a KV cache that grows without bound.
`larql-boundary` provides the cold-storage half of a two-tier memory model:

```
recent tokens  →  hot KV window    (normal attention, fast)
older context  →  BOUNDARY refs    (compressed residual checkpoints, compact)
```

Each BOUNDARY ref is the final-layer residual at the end of a 512-token chunk —
2564 bytes at 2× compression, with an explicit behavioural contract describing what
the compression preserved.

---

## Three phases

| Phase | Module | What it does |
|-------|--------|--------------|
| 1 | [`codec`] | Compress / decompress residual `f32` slices |
| 2 | [`metadata`] | Compute per-boundary confidence fields from logit slices |
| 3 | [`gate`] | Per-boundary decision: compress, bf16 fallback, or cold replay |

---

## Quick start

```rust
use larql_boundary::{codec, gate, metadata};
use larql_boundary::gate::{BoundaryDecision, BoundaryGateConfig};

// ── Phase 1: encode a residual ─────────────────────────────────────
// int8_clip3σ: 2564 bytes for d=2560, vs 5120 for bf16 (2× compression)
let residual = vec![0.1f32; 2560];
let payload = codec::int8::encode(&residual);
let decoded  = codec::int8::decode(&payload);
assert_eq!(decoded.len(), residual.len());

// ── Phase 2: compute metadata ──────────────────────────────────────
// Caller runs lm_head(final_norm(raw_residual)) and provides the logits.
// This crate never touches model weights.
let raw_logits = vec![0.0f32; 262_145]; // Gemma 3 vocab size
let hat_logits = raw_logits.clone();    // from compressed residual forward pass
let mut meta = metadata::compute(&raw_logits, Some(&hat_logits));

// ── Phase 3: gate decision ─────────────────────────────────────────
// Exp 44 calibrated config (set calibration_mode=false after Exp 44).
let config = BoundaryGateConfig {
    calibration_mode: false,
    min_log_prob_margin: 2.16,  // Exp 44 Track A: log-prob margin units
    min_top1_prob: 0.5,
    ..Default::default()
};
let decision = gate::apply(&mut meta, &config);
match decision {
    BoundaryDecision::CompressedOk { .. } => { /* emit int8 frame */ }
    BoundaryDecision::UseBf16             => { /* emit bf16 frame  */ }
    _                                     => { /* cold replay / reject */ }
}
```

---

## Codec schemes

| Scheme | Bytes (d=2560) | Ratio | Contract |
|--------|----------------|-------|----------|
| `codec::bf16` | 5 120 | 1× | Exact |
| **`codec::int8`** | **2 564** | **2×** | **D- (ArgmaxNearEquivalent)** |

**Why σ-clipping instead of absmax?**  
Gemma-class residuals have absmax/σ ≈ 92×. Absmax quantisation wastes 99% of
int8's 256 levels on a handful of extreme values. 3σ-clipping concentrates all
256 levels on the ±3σ band where the prediction-relevant geometry lives.

---

## Accuracy contract

The contract is **top-1 token preservation**, not residual MSE.

```
int8_clip3σ (Exp 43, 30 prompts, layer 33, Gemma 3 4B):
  top-1 agreement:  98.7% mean  /  93.3% min
  top-5 agreement:  100%
  KL divergence:    ~2.0 nats
  Contract:         D- (ArgmaxNearEquivalentHighMargin)
```

Residual MSE (~300 for the non-outlier elements) does not predict downstream
quality. A boundary that compresses poorly in MSE may still preserve top-1;
a boundary that looks clean in MSE may cascade if the model was uncertain.
See the `accuracy` example for a demonstration.

---

## Gate calibration (Exp 44 Track A)

396 boundary positions from Frankenstein, Gemma 3 4B, 90 continuation tests (20 tokens):

| Threshold (log-prob) | Accept rate | Early-div (≤5 tokens) | Total div (20 tokens) | System compression |
|---------------------:|------------:|----------------------:|----------------------:|-------------------:|
| 0.00 | 100% | 10.0% | 21.1% | 2.00× |
| 1.35 | 82.2% | 8.1% | 21.1% | 1.82× |
| **2.16** | **68.9%** | **4.8%** | **19.8%** | **1.69×** |
| 4.05 | 47.8% | 4.7% | 21.3% | 1.48× |
| 6.75 | 14.4% | 0.0% | 5.8% | 1.14× |

**Recommended threshold: 2.16** — meets accept ≥ 50% and early-div < 5%.

Note on the flat region: total divergence is ~20% at every threshold below 5.4.
Raising the threshold from 0 to 2.16 reduces early-div (first 5 tokens) but not
total divergence over 20 tokens. The early-div filter is doing all the safety work;
cascade compounds after the first wrong token.

**What D-@high actually guarantees:** the first ~5 generated tokens are safe at the
4.8% point-estimate level (95% CI ≈ 1.6%–10.7%, n=62 accepted boundaries). Total-window
safety is NOT contracted. Use D-@high for boundary-to-fresh-decode; not for long
uninterrupted continuation.

---

## Performance (M3 Max, release build)

| Operation | Time | Notes |
|-----------|------|-------|
| `bf16::encode` d=2560 | **1.2 µs** | Bit manipulation, memory-bound |
| `bf16::decode` d=2560 | **0.27 µs** | Shift + store |
| `int8::encode` d=2560 | **4.6 µs** | σ + clamp + quantize |
| `int8::decode` d=2560 | **0.23 µs** | Multiply by scale |
| `metadata::compute` (no hat) | **517 µs** | log_softmax over 262K vocab |
| `metadata::compute` (with hat) | **660 µs** | + one extra lm_head pass |

`metadata::compute` is the bottleneck at 517 µs — driven by log_softmax over
262 145 vocabulary elements. At 512-token chunk stride and 50 tok/s decode, a new
boundary arrives every ~10 seconds. 517 µs is 0.005% of that budget.

All codec operations (encode/decode) are well under the model forward pass time
(~150 ms on CPU per token) and are never on the critical path.

---

## Examples

Three examples, each demonstrating a different aspect:

```sh
# Phase 1: compression ratio and residual MSE
cargo run -p larql-boundary --example encode_decode

# Phase 3: gate decisions for the four Exp 43 boundary patterns
cargo run -p larql-boundary --example gate_decision

# Phase 1+2: why MSE is the wrong metric; top-1/KL/gate pipeline
cargo run -p larql-boundary --example accuracy
```

**`encode_decode`** output:
```
bf16:           5120 bytes  (1.0× vs bf16)  MSE = 3.61
int8_clip3σ:    2564 bytes  (2.0× vs bf16)  MSE = 4109583 ← outlier-dominated
non-outlier MSE (excl. 2 outlier elements) = 306
```

**`gate_decision`** output:
```
tech_3 (confident)  margin=9.00  agreement=Agrees    fragile=false  →  ✓ compressed
qa_2   (fragile)    margin=0.10  agreement=Agrees    fragile=true   →  ~ bf16 fallback
codec disagrees                  agreement=Disagrees              →  ~ bf16 fallback
not checked                      agreement=NotChecked             →  ~ bf16 fallback
calibration mode (default)                                        →  ~ bf16 fallback
```

**`accuracy`** output:
```
Phase 1: int8_clip3σ = 2564 bytes, 2.0×, non-outlier MSE = 300
         ⚠ MSE is NOT the contract. See Phase 2.

Phase 2 (Exp 43, 30 prompts): top-1=98.7% mean, KL=2.0 nats

Confident (~tech_3)  top-1 ✓  gate: COMPRESS ✓  margin=9.00 ≥ 2.16
Fragile   (~qa_2)    top-1 ✗  gate: bf16 fallback ✓  codec_fragile=true
```

---

## Tests

```sh
cargo test -p larql-boundary              # 25 unit + 9 integration + 1 doc
cargo test -p larql-boundary --benches    # benchmark compile + smoke
```

Coverage (cargo-llvm-cov): **functions 100% · lines 98% · regions 97%**

---

## CI

```sh
make larql-boundary-ci      # fmt + clippy + tests + benches + examples
make larql-boundary-coverage
```

GitHub Actions: `.github/workflows/larql-boundary.yml`  
Platforms: **Linux · Windows · macOS** (all in CI)

---

## Design notes

**Model-agnostic.** This crate takes `&[f32]` slices only. No model weights,
no inference backend, no MLX dependency. `larql-inference` runs the forward
pass and provides logit slices; `larql-boundary` decides what to do with them.

**No `unsafe`.** Pure Rust with `to_le_bytes()` / `from_le_bytes()` for endian
safety on all platforms.

**Two fragility types** — do not conflate them:

- *Codec fragility* (`codec_fragile = true`): the codec changed the top-1 token.
  Hard reject. Switching to bf16 eliminates this entirely.
- *Boundary fragility* (`boundary_fragile = true`): the *uncompressed* model is
  uncertain at this position. Bf16 is also fragile here. The gate falls back, but
  even the fallback frame carries risk.

**Calibration mode** (`calibration_mode = true`, the default): gate ignores all
thresholds, always falls back to bf16, emits `BoundaryContract::Calibrating`.
All metadata is logged for telemetry. Set to `false` only after running
`~/chris-source/chris-experiments/shannon/44_boundary_gate_calibration/calibrate.py`.

---

## Protocol spec

Full spec:  
`~/chris-source/chris-experiments/shannon/43_residual_stream_codec/BOUNDARY_REF_PROTOCOL.md`

Calibration:  
`~/chris-source/chris-experiments/shannon/44_boundary_gate_calibration/`

Residual codec characterisation:  
`~/chris-source/chris-experiments/shannon/43_residual_stream_codec/SPEC.md`
