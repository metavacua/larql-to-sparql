# ADR-0025: Dense-wire direction negotiation, timing trailer, and the fidelity gate

**Status:** accepted 2026-07-24 (implemented; four commits `3c4e7844` →
`400f7489`)
**Supersedes in part:** extends [ADR-0009](0009-wire-format-evolution.md)
(wire-format evolution) for the DEC-1A instrument requirements
(`docs/dec-funnel.md` v0.5 §3).

## Context

DEC-1A needs three wire capabilities the dense walk-ffn protocol lacked:
per-term timing (the composed model `T_step = T_queue + T_codec +
T_network + T_serve` must be fitted, not assumed), independently
compressed directions (the inbound residual and outbound FFN delta need
not share quantisation sensitivity — a mechanistic probe, not just
configs), and a fidelity gate that makes compressed arms claim-bearing.
The pre-registered consolidation trigger (hardening item 16) fired first:
all three land on a single-sourced codec.

## Decisions

### 1. One codec module (hardening #16)

The dense binary frame's encoder, decoder, and constants live in
`larql-inference` `ffn/remote/codec.rs` — the same single-source
discipline as the q8k/MoE codecs. `larql-server`'s `walk_ffn/binary.rs`
is a shim; `larql-router` imports the constants. Byte-identical wire is
pinned by encode→decode→re-encode tests; the allocation-bomb guards
(PR-104 lineage) moved verbatim.

### 2. Direction negotiation: `Content-Type` = inbound, `Accept` = return

HTTP already had the right vocabulary; the implementation now honours it
independently. Request bodies encode as f32 (`application/x-larql-ffn`),
f16 (`…-f16`), or i8 (`…-i8`) — the compressed *request* encodings are
new (the historical "f16 wire default" was return-direction only).
Response encoding follows `Accept` exactly as before, i8 responses still
gated by `LARQL_I8_WIRE`. CT detection is ordered (compressed suffixed
types before the base CT, which is their substring). Layouts mirror the
response forms: f16 = u16 LE halves; i8 = per-position
`[scale f32][zero f32][i8 × hidden]`.

Client API: `RemoteFfnConfig::with_wire_formats(in, out)` /
`LayerShardedBackend::connect_with_wire_formats(…)`; the symmetric
constructors delegate with `in = f32`, so existing callers are
byte-unchanged. dec-bench exposes the axis as `--wire f16/i8` pair
syntax; plain arms keep their historical semantics (f32 request +
compressed Accept) and truthfully record `wire_in = f32` — true
symmetric compression is `f16/f16`.

### 3. Timing: header-gated 8-byte response trailer

A request carrying `x-larql-timing: 1` receives, on the q8k walk-ffn and
multi-layer expert responses (the frames with no embedded latency; the
f32 walk-ffn response already embeds one), a trailer of
`TIMING_TRAILER_MAGIC` (`b"LQTM"`, u32 LE) + `serve_us` (f32 LE,
finite ≥ 0), appended *after* the unchanged encoder output.
`serve_us` clocks post-body-receive → response-encode-complete: a
duration, so no clock synchronisation exists anywhere in the protocol.

Mixed-version matrix (all pinned by tests): no header → byte-identical
wire; new client + old server → magic check fails → timing degrades to
`None`; the client strips the trailer *before* the production decoder
runs, so every length guard sees the pre-extension payload. The affected
decoders read declared entry counts and tolerate trailing bytes (also
pinned) — a mixed deployment cannot corrupt a decode.

dec-bench records the two-scoreboard schema per point: `encode_us`,
`serve_us`, `client_decode_us` measured; `transmit_us` derived as
`client_window − serve_us` (the driver's timer already excludes
encode/decode — subtracting them again would double-count), clamped at 0
and flagged; `queue_ms` reserved as DEC-2's axis. Tier-efficiency and
user-latency metrics are never collapsed into one figure.

### 4. Fidelity gate: `larql dec-bench drift`

Teacher-forced bits/char on a pinned corpus (default: the shannon-verify
Frankenstein carve) per wire config, against an **in-run f32/f32
baseline** — never a stored one — with a 0.5% drift gate (configurable)
and a non-zero exit code on failure (CI-usable). The scorer mirrors the
production streaming decode per token; a labeled arm that cannot get its
wire **errors loudly** (no silent fallback inside a measurement), and a
served-format classifier flags Accept fallbacks (e.g. i8 requested
without `LARQL_I8_WIRE` server-side) from the observed bytes-per-value.
Teacher-forced NLL is exact math given weights + wire: the instrument is
deterministic and battery/thermal-safe, unlike the timing instruments.

## Consequences

- DEC-1A runs with its final schema on day one; the asymmetric axis and
  the drift gate make every compressed arm's bandwidth number
  quality-bounded (C6) and every latency number decomposable (the fitted
  model DEC-2/DEC-CV consume).
- Where precision lives (residual vs delta) is now measurable — the
  input DEC-1B's compiled per-layer transport policy needs; that policy's
  carrier format is deliberately NOT designed until 1A's sensitivity
  data exists.
- Any future frame change starts from one module; the hand-sync failure
  mode item 16 tracked is structurally gone.
