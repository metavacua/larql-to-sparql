# Expert-serving review — 2026-07-23 (pre routed-experts replay arm)

Targeted review of the **expert-serving surface** (`larql-server` `routes/expert/*`,
the `moe_remote` codecs, the MoE routing computation, and `dec_bench`) ahead of
building the DEC routed-experts replay arm (ROADMAP hardening item 14 — the arm
that gates the C1-on-MoE verdict). Three parallel readers (server endpoints,
dec_bench/Endpoint seam, routing-capture path), verified findings with
`file:line` evidence only. Companion to
[`dec-readiness-review-2026-07-22.md`](dec-readiness-review-2026-07-22.md);
line numbers cite the tree at commit `141d3b1f`.

> **Remediation status (2026-07-23):** Phase A (server hardening, §1 items
> 1a–1e) ✅ — batch handlers 400 on unresolvable experts with run-count
> accounting; q8k shape validation; owned-entry stats probe; RifGuard +
> `requests_total` + layer-latency across the expert surface (closes
> ROADMAP item 13's class); `moe_q4k_direct` in the startup flag line.
> Perf pre-measurement batch ✅ — bulk LE codecs (byte-identical, re-encode
> pinned), codec work inside `spawn_blocking`, q8k per-task clones removed,
> stale parallelism docs corrected, `moe_batch_mode` cached. Phase B
> (capture sidecars + `--routing` + router twin-parity test, 97.9%
> coverage) ✅. Phase C (Endpoint seam + routed replay, per-point
> naive/union denominators, warmup non-zero guard, `client_rayon_threads`
> recorded) ✅ — closes ROADMAP hardening items 13 and 14. Phase D (the
> routed run) pending. Deferred to the tier-throughput track:
> expert-grouped scheduling (perf A2, ~4× byte ceiling at B=64 —
> pre-registered as a measurement caveat), semaphore unification,
> `Q8KActivation` borrow refactor, dense weight-range table.

## Verdict

The routed arm, built naively against today's endpoints, would produce a
corrupted batch curve **with no error signal**: batch expert handlers silently
skip unresolvable experts (partial sums, HTTP 200, *faster* than honest work),
the q8k multi-layer decoder accepts impossible shapes, and the movement-ratio
denominator is null on sharded servers. None of this affects the shipped
DEC-0 dense-arm numbers (walk-ffn path, single full-table loopback server).
The capture side needs new machinery: routing is never computed client-side
during capture, router weights are recorded nowhere in the codebase, and the
existing residual pool stores dense-prenormed rows the routed endpoints cannot
consume.

## 1. Server (fixes required before the arm — Phase A)

- **1a. Batch handlers silently skip unresolvable experts [HIGH]** —
  `routes/expert/cpu.rs:143,244` `else { return acc }` on `resolve_bytes`
  miss; no ownership check in any batch handler (`layer_batch.rs`,
  `multi_layer_batch.rs`, `batch_legacy.rs`) vs `single.rs:27-41` which 400s.
  Sharded replay → weighted sum over the owned subset only: wrong number,
  lower time, no error. Fix: run-count accounting + 400 on any unresolvable
  non-zero-weight expert (client partitions per shard, so server-side
  unresolvable is always a bug).
- **1b. q8k multi-layer decode accepts impossible hidden [MED]** —
  `multi_layer_wire.rs:239` floors `hidden/256`; no divisibility or
  model-hidden check in either multi-layer handler. Wrong-hidden replay
  "succeeds" with wrong row stride (garbage) or silent zeros.
- **1c. `/v1/stats` `ffn_weights.moe.per_expert_bytes` null on shards [MED]** —
  `stats.rs:107-110` probes expert entry 0 only; `--experts 64-127` shards
  have no entry 0. The movement-ratio denominator disappears on exactly the
  DEC-2 topologies. Probe the first *owned* entry.
- **1d. Expert surface invisible to drain/heartbeat/latency [MED, extends
  item 13]** — `RifGuard`/`requests_total`/`layer_latency_tracker.record`
  exist only on the f32 walk-ffn handler (`walk_ffn/handler.rs:56-57`,
  `core.rs:282`). A pure-expert shard heartbeats rate=0/latency=∅ while
  saturated, and drain can conclude idle mid-batch. Also: the
  `compute_semaphore` gates only layer-batch[-f16] (`layer_batch.rs:45-58`) —
  multi-layer/legacy/gRPC saturate via rayon oversubscription instead, so
  endpoint-vs-endpoint curves measure different queueing mechanisms; and
  layer-batch's reported `latency_ms` includes semaphore queue wait
  (`t_start` before acquire).
- **1e. MoE kernel-path observability [LOW]** — the direct-Q4K vs
  cached-dequant expert path flips silently on `LARQL_DISABLE_Q4K_DIRECT`,
  BF16 vindexes, or hidden%256≠0 (`expert/q4k.rs:16-21`) — a completely
  different byte-movement regime, previously visible only via timing-env
  stderr. Startup flag summary gains `moe_q4k_direct=`.

## 2. Facts the arm's design rests on (verified)

- **The server never re-routes**: every expert frame carries explicit
  `(expert_id, weight)` (`wire.rs:53-91`, `multi_layer_wire.rs:52-57`);
  weights on the wire are final post-renorm post-scale values. Replay
  bit-matches production routing without the server knowing the policy.
- **Production decode target**: `/v1/experts/multi-layer-batch-q8k` when
  hidden%256==0 and Q8K wire enabled (`backend.rs:719-756`), else
  `multi-layer-batch`; per-layer `layer-batch` for `forward_moe`. Legacy
  `/v1/expert/batch` is prefill-only (§4f, unchanged). The routed arm
  targets the multi-layer pair.
- **No cross-row weight sharing anywhere**: B same-layer tasks are processed
  independently (`multi_layer_batch.rs:63-85` par_iter); an expert in r rows
  is streamed r times. Within one task, K experts share pre-norm + one Q8K
  activation, each expert streamed once (`cpu.rs:61-99`). Batch-union
  amortisation is page-cache/DRAM-level only → the expected-bytes model is
  per-row naive (`Σ_r |E_r| × per_expert_bytes`); report the union
  (`|⋃E_r|`) alongside as the DEC-3 metrology bound. Prior V3 measurement
  (~124/128 experts per sequence) predicts near-total union at B≥16.
- **Caches**: the process-global f32 dequant FIFO is bypassed on the default
  Q4K-direct path (`f32.rs:54-57`) — no in-process cache fakes repeats on the
  26B; there is no cold-cache mode for expert weights (`packed_mmaps` never
  madvised) — replay measures warm page cache, which is the honest fits-in-RAM
  C1 regime but must be declared. Metal experts path (default-off) has a
  dead per-request buffer-cache population (`metal.rs:114-147`) and silent
  CPU fallback.
- **Timing**: multi-layer responses carry no server latency field
  (`multi_layer_wire.rs:140-152`) — the routed arm times client-side, like
  the q8k dense arm today.
- **Routing lives in two mirrored implementations**: policy-driven
  `moe_route_from_router_input` (`cpu/ops/moe/mod.rs:114-146`; used by CPU
  and Metal local paths — Metal routes on CPU at `moe_dispatch.rs:751-753`)
  and the client-side `MoeRouterWeights::route`
  (`ffn/moe_remote/router.rs:92-152`, Gemma-4 behaviour hard-coded).
  Numerically equivalent today **by convention only** — nothing pins them.
  The capture arm must add a twin-parity test.
- **`LARQL_MOE_DEBUG` is not a capture channel**: indices only (no weights),
  CPU path only, layer index reconstructed `mod 30`, unstructured stderr
  (`forward.rs:59-88`; `moe-routing/analyze.py` parses exactly this).

## 3. Capture/pool design constraints (Phase B)

- Pool rows are **dense-prenormed** (`remote_ffn.rs:466-468` applies
  `pre_feedforward_layernorm`); Gemma 4 MoE layers norm the expert block with
  a *different* weight (`gemma4.rs:347-356`) → routing is not derivable
  offline, and the routed endpoints want raw `h_post_attn` (f32 frame; server
  applies pre_experts_norm) or pre-experts-normed+quantised (q8k frame).
- **No routing runs during capture**: `ffn_is_remote` makes the local-expert
  branch unreachable (`moe_interleave.rs:156-167`). The sink push site
  (`remote_ffn.rs:463-468`) has raw `h_capture` + `weights` in scope —
  compute routing there via `MoeRouterWeights::route` (pure function; no
  kernel changes), and additionally store raw + pre-experts-normed rows for
  MoE layers.
- **Format evolution: additive sidecar, keep manifest v1** —
  `CapturePool::open` requires exact version equality
  (`capture_format.rs:167-172`); `CaptureManifest` tolerates unknown fields.
  `routing.bin` (`[prompt][step][layer][k×(u32,f32)]`, k=0 sentinel for
  non-MoE) + raw/normed planes as separate files; absence degrades to
  walk-ffn-only replay, so the existing 330M pool keeps serving the dense
  arms unchanged.
- Zero-weight pairs cost nothing server-side (`cpu.rs:139`) — capture strips
  them or accounting overcounts.

## 4. Endpoint seam (Phase C)

`WireArm::accept() → None` for Q8k plus `send_one`'s two-way branch
(`replay_runtime.rs:277-374`) are a latent endpoint enum — make it explicit:
`Endpoint` owns path, frame-builder, response decoder, `server_ms`
availability, and the denominator source. `weight_bytes_tok` moves from
run-level scalar (`output.rs:28-31`) to per-point (batch-dependent under
expert-union); pulse gains `dec/endpoint(_code)`, run record's hardcoded
`endpoint: "walk-ffn"` (`replay_runtime.rs:155`) becomes real. Batch-mode
frame encoding currently happens inside the timed window despite the comment
claiming otherwise (`replay_runtime.rs:235-239` pre-builds only `(layer,
rows)`) — fix or document while in there. New content-types need
`wire_label_for_content_type` entries. Replay must assert non-zero responses
(spot-check vs a CPU oracle) and verify shard ownership against topology
before any claim-bearing sharded run (§1a).

## 5. Run design notes (Phase D)

Warm-page-cache regime declared; first-touch iterations dropped or reported
separately; kernel path pinned via the startup flag line (§1e); both naive
and union denominators reported; `served_wire` degenerates to the endpoint's
fixed CT. C1-on-MoE pass criterion unchanged: step time sub-linear through
batch 32 on the multi-layer-batch-q8k arm.
