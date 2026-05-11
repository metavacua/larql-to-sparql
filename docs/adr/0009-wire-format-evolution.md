# ADR-0009 — Wire Format Evolution: f16 Default + i8 Quantised Residuals

**Status:** Accepted — **implemented 2026-05-07** (GT1 f16 + GT2 i8)  
**Depends on:** ADR-0006 (Q4_K Remote FFN), ADR-0003 (FFN Router)  
**Closes:** ADR-0006 Open Question 3 ("Wire format")

---

## Context

ADR-0006 established the binary wire format (`application/x-larql-ffn`) with
f32 residuals on both directions. ADR-0006 §Trade-offs explicitly noted:

> "For LAN-distributed setups the ~5 KB payloads are trivial; for MoE/expert
> fan-out across WAN, quantised residuals (i8 + scales) would help. Out of
> scope here."

That deferral was correct for the initial demo. The grid is now validated on
2-shard LAN and the next target is multi-host / cross-region (ADR-0004
G-SCALE). At Gemma 4 26B hidden_size=5120 and seq_len=1, one f32 residual is
5120 × 4 = **20 KB** per direction per layer. For a 30-layer sweep with
per-layer serial dispatch that is **1.2 MB/token** on wire. On a 10 Mbps
internet link that is **960 ms of wire time per token** before RTT overhead.

Two changes address this:

1. **f16 default** — halves wire bytes with zero accuracy loss (IEEE 754 f16
   has sufficient range for residual values; dequant round-trip is lossless
   at the logit level for all tested models).
2. **i8 quantised residuals** — 75% bandwidth reduction with minimal accuracy
   cost, opt-in until validated across all model families.

Both are purely wire-level: server and client negotiate via
`Accept`/`Content-Type` headers. The router passes content-type through
transparently (no change needed).

---

## Decision

Add two new content-types alongside the existing f32 format. All three
coexist; the server falls back gracefully when the client does not advertise
a preference.

| Content-Type | Dtype | Bytes/value | Change |
|---|---|---|---|
| `application/x-larql-ffn` | f32 LE | 4 | existing (unchanged) |
| `application/x-larql-ffn-f16` | f16 LE (IEEE 754) | 2 | new — **default for grid** |
| `application/x-larql-ffn-i8` | i8 symmetric | 1 + 8 byte header | new — opt-in |

**f16 becomes the default** for the `RemoteWalkBackend` and `RemoteMoeBackend`
clients (all grid traffic). Non-grid HTTP clients that omit `Accept` continue
to receive f32 (no breaking change). The opt-out is `LARQL_F16_WIRE=0`.

**i8 is opt-in** via `LARQL_I8_WIRE=1` or `--wire i8` CLI flag. It requires
both ends to support the quantised format; the server rejects i8 requests from
clients that have not set it explicitly.

---

## Wire Layout

### f16 (`application/x-larql-ffn-f16`)

Identical byte structure to the f32 format (ADR-0003 §Wire) except every
`f32` field in the residual/output payload is replaced with `u16` LE (IEEE
754 half-precision). Header fields (`layer`, `seq_len`, `flags`, `top_k`,
`latency_ms`) remain f32/u32 LE — only the residual and output float arrays
are converted.

Single-layer request:
```
[layer u32 LE][seq_len u32 LE][flags u32 LE][top_k u32 LE]
[residual f16[] LE — seq_len × hidden_size u16 values]
```

Single-layer response:
```
[layer u32 LE][seq_len u32 LE][latency_ms f32 LE]
[output f16[] LE — seq_len × hidden_size u16 values]
```

Batch variants follow the same pattern as ADR-0003 §Wire with f16 arrays.

### i8 symmetric (`application/x-larql-ffn-i8`)

Per position in the sequence, a scale and zero_point are emitted before the
quantised bytes. This allows per-position scales which handle the high
dynamic-range variance of early-layer residuals.

Single-layer request:
```
[layer u32 LE][seq_len u32 LE][flags u32 LE][top_k u32 LE]
Per position (seq_len times):
  [scale f32 LE][zero_point f32 LE][data i8[] — hidden_size bytes]
```

Single-layer response: same structure.

Quantisation: symmetric (zero_point = 0.0), scale = max(|x|) / 127.0.
Dequantise: x̂ = i8_val × scale.

---

## Negotiation Protocol

Client sends `Accept: application/x-larql-ffn-f16` (or `-i8`).
Server inspects `Accept` header, selects the highest-precision format it
supports that the client accepts. Server responds with the chosen
`Content-Type`.

If the client sends no `Accept` header (non-grid HTTP client), the server
returns f32 (existing behaviour, no change).

If the client sends `Accept: application/x-larql-ffn-i8` but the server
does not support i8 (e.g. older deployment), the server falls back to f16
if the client also listed it, then f32.

---

## Accuracy Validation

The wire format is architecture-agnostic: it carries raw float arrays with no
model-specific structure. Validation must be run against each model family
before changing defaults for that family.

Before making f16 the default, run against the target vindex:

```bash
larql bench <model.vindex> --ffn URL --steps 50 \
            --wire f32,f16 --assert-topk-match 5
```

Expected: identical top-5 tokens across f32 and f16 for all 50 decode steps.
This must pass for every model family in use (dense, MoE, Q4K, f16, etc.)
before f16 is made the default for that path.

For i8, run the same with `--wire f32,i8 --assert-topk-match 1` (top-1 only,
some acceptable divergence at low-probability tokens).

The accuracy threshold may differ by model family and quantisation format:
- Dense f16 vindex: expect zero top-5 divergence on f16 wire.
- Q4K vindex: f16 wire is lossless (residuals are already f32 intermediates).
- High-variance architectures (e.g. early layers of very deep models): verify
  that residual norms at layers 0–2 are within f16 range (±65504). If not,
  per-layer f32 fallback for those layers is an option.

---

## Implementation

| File | Change |
|---|---|
| `crates/larql-server/src/wire.rs` | Add `F16_CT`, `I8_CT` constants; `fn preferred_response_ct(accept: &str) -> &str` |
| `crates/larql-server/src/env_flags.rs` | Add `F16_WIRE = "LARQL_F16_WIRE"`, `I8_WIRE = "LARQL_I8_WIRE"` |
| `crates/larql-server/src/routes/walk_ffn.rs` | Inspect Accept header; branch encode_binary_output to f16/i8 paths |
| `crates/larql-inference/src/ffn/remote/codec.rs` | Add `encode_f16_request`, `decode_f16_single/batch`, `encode_i8_request`, `decode_i8_single/batch` |
| `crates/larql-inference/src/ffn/remote/http.rs` | Set `Accept` header based on `WireFormat` enum; decode by response Content-Type |
| `crates/larql-inference/benches/wire_codec.rs` | New criterion bench: encode/decode throughput (MB/s) at hidden_size 2560/4096/5120, seq_len 1/32/256 |

---

## Trade-offs

- **f16 accuracy**: lossless for residuals in practice (values stay well within
  f16 range of ±65504 for tested architectures). Risk: very early layers
  (L0-L2) where residual norm can exceed 1000 in some architectures. Mitigated
  by the clamp-to-max behaviour of f16 overflow (not NaN for normal finite
  values). Must be validated per-architecture before enabling as default.
- **i8 accuracy**: expect <0.5% top-1 mismatch on Gemma 3/4. Higher mismatch
  on models with high-variance residuals (check with `--assert-topk-match 1`
  before enabling by default).
- **Router transparency**: router passes request/response bytes raw for
  single-shard routing. For multi-shard JSON fan-out, the router must decode
  JSON (no change needed — JSON path remains f32). Binary multi-shard is
  already rejected (ADR-0003).
- **Backward compatibility**: f32 remains available; non-upgraded clients
  continue to work without change.

---

## Open Questions

1. **Per-position vs per-tensor i8 scale.** Per-position (current decision)
   adds `seq_len × 8 bytes` overhead. For seq_len=1 this is 8 bytes (negligible);
   for seq_len=256 it is 2 KB on top of 256 × 5120 = 1.25 MB — still <1%.
   A per-tensor scale is simpler but loses quality on high-variance positions.
   Decision: per-position for now; revisit if prefill seq_len > 1024.

2. **MoE expert wire format.** Expert request/response bodies (`/v1/expert/batch`)
   are currently JSON. The i8 scheme applies equally but requires ADR-0009
   extension or a separate ADR. Defer until F9 (binary expert wire format) lands.
