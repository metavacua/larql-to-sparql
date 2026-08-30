# DEC-readiness review — 2026-07-22

> **Remediation status (2026-07-22):** Batch A (§1a–§1d, §2a first item) is
> fixed — see [`ROADMAP.md`](../../ROADMAP.md) §"Codebase hardening" →
> "DEC-readiness review (2026-07-22)" for the per-item summary and the
> commit that landed it. Batch B items 7 (backend factory + capability
> dispatch, §3c) and 9 (`SKIP_MOE` name split, §3e) landed 2026-07-22
> alongside the larql-compute DEC prep (large-file module splits + the
> `FormatRoute` quant registry). The rest of Batches B and C (fleet/config
> landmines, structural pre-work) remain open, tracked in the same ROADMAP
> section.

Targeted review of the **DEC data-plane** (`larql-server`, `larql-router`,
`larql-inference` remote-FFN/MoE paths, `larql-cli` bench/dec_bench, the DEC
scripts) ahead of the DEC funnel programme ([`docs/dec-funnel.md`](../dec-funnel.md)).
Scope was deliberately narrow — the code the programme actually exercises when
it runs `larql-server` expert tiers, `larql-router`, and the new `dec-bench`
loadgen on **rented marketplace hosts** (Vast.ai x86 EPYC, not the M3 Max the
stack was developed on), against **models other than Gemma** (Inkling ~975B,
Kimi K3), over **network links with adversarial peers**.

Four parallel readers (security, hardcoding/config, modularity, performance),
each returning only findings verified against the code with `file:line`
evidence. This document is the canonical record; prioritized actions are
tracked in [`ROADMAP.md`](../../ROADMAP.md) §"Codebase hardening" under
"DEC-readiness review (2026-07-22)".

## Verdict

The stack is structurally sound and the wire decoders are mostly hardened —
but the review surfaced a **cluster of silent-corruption defects** that
matter more than usual here because the entire programme is a measurement
exercise: each produces plausible-looking output that is actually wrong, with
no error. The single dominant theme is *"produces a number, the number is a
lie"*, and it intersects the DEC-0 batch curve, the DEC-0.5 x86 tier, and the
Inkling/K3 extraction stages directly. Security exposure is one genuine
remote-abort bug (a regression from the codebase's own established guard
pattern) plus deployment-posture items that are real under the stated
threat model. Structural debt is real but lower-urgency — it's pre-work for
the G/C ladders and the K3 port, not a DEC-0 blocker.

The B-row **f32/f16/i8** serving path is genuinely well-built (one BLAS GEMM
over all rows, lock-free per-layer weight cache); it is specifically the
**Q8K** path — the wire DEC prefers — that does not batch.

---

## 1. Silent corruption — wrong numbers, no crash

The highest-priority cluster. None of these throw; all of them corrupt a DEC
measurement or a generation.

### 1a. Q8K has no batched compute on the server [perf + modularity, HIGH] — ✅ FIXED

`crates/larql-server/src/routes/walk_ffn/q8k.rs:199` maps each request entry
through `kquant_ffn_forward_layer_q8k`
(`crates/larql-inference/src/vindex/kquant_forward/walk_ffn.rs:84`), which is
strictly single-row (`Array2::from_shape_vec((1, …))`, matvec kernels). The
replay client sends B entries that all share one layer
(`dec_bench/replay.rs` `build_q8k_frame`), so a batch-64 Q8K request runs 64
independent matvecs, **re-streaming the layer's full Q4K gate/up/down bytes 64
times** while rayon workers contend for DRAM bandwidth.

Contrast the f32/f16/i8 arm: one frame with `seq_len=B` hits
`kquant_ffn_forward_layer` (`walk_ffn.rs:32`), which is `x.dot(w.t())` — a real
GEMM over the once-dequantised f32 cache, amortising weights across all rows.

**DEC impact:** the DEC-0/DEC-1 batch-64 curve on the Q8K arm is ~linear in B
by construction. C1 ("step time sub-linear in batch") fails on Q8K for a
serving-code reason, not physics; the C6 wire comparison is corrupted (Q8K
looks worse than its bandwidth advantage deserves).

**Fix direction:** on the Q8K handler, group same-layer entries and either
(a) dequantise each activation to f32 and run them as one batched GEMM through
the existing `kquant_ffn_forward_layer` path (preserves the Q8K *upload*
compression that is the C6 claim, fixes compute scaling), or (b) add a
multi-row Q4K×Q8K kernel that keeps a weight super-block resident across the B
rows. The wire already carries duplicate-layer entries in order
(`decode_q8k_batch_response_entries` exists for exactly this shape).

### 1b. x86 silently falls to scalar on two serving kernels [hardcoding, HIGH] — ✅ FIXED (observability)

`crates/larql-compute/src/cpu/ops/q4k_q8k_dot.rs`:
- `q4k_q8k_gate_up_into` (:1377) has **no x86 branch** — non-aarch64 runs
  `q4k_q8k_matvec_scalar` twice. This is the dense `/v1/walk-ffn-q8k` serving
  kernel (`kquant_forward/walk_ffn.rs:108`), whose doc-comment (:77) falsely
  claims "NEON/AVX2". The code's own comment (:1253) says AVX2 is ~12–16×
  faster than scalar.
- `q6k_q8k_matvec_into` (:2119) is NEON-or-scalar, no AVX2 — and
  `q4k_q8k_matvec_parallel` ("the single source for every quantized
  projection on the decode path") routes every Q6_K projection (attention V,
  lm_head on typical Q4K vindexes) through it.

**DEC impact:** DEC-0.5 (x86 kernel gate) is *defined* as measuring the x86
tier. It would produce numbers on a 10–16× handicapped kernel with no warning,
and sub-linearity conclusions would be drawn on it. Building the AVX2 kernels
is C-ladder work; the **silence** is the bug this review flags — the serving
path must log the selected kernel class once at startup and the false doc
comments must be corrected, so no DEC number is ever recorded on an unlogged
scalar fallback.

### 1c. Shard failure silently zero-fills the FFN output [hardcoding, HIGH] — ✅ FIXED

`crates/larql-inference/src/ffn/remote/sharded.rs:117` `forward_predispatch_all`:
an unowned layer (:129) or a panicked shard thread
(:136 `handle.join().unwrap_or_else(|_| vec![0.0; hidden])`) returns a zero
vector, and decode continues on a corrupted hidden state.

**DEC impact:** multi-hour rented session, one shard OOMs/restarts →
generation quality silently collapses while the bench harness keeps recording
tok/s. The MoE path already returns `Result`; the dense predispatch path
swallows it. Fix: propagate the error.

### 1d. Down-projection ignores its format tag on the Q8K fast path [hardcoding, HIGH] — ✅ FIXED

`crates/larql-inference/src/vindex/kquant_forward/walk_ffn.rs:135` feeds
`ffn[2].0` straight into the Q4_K-only `q4k_q8k_matvec_into` without consulting
the format tag `ffn[2].1` (the f32 fallback at :153 *does* use it via
`dequantize_matrix`); the doc comment at :78 staleley claims down goes through
the f32 dequant path.

**DEC impact:** a model whose down slab is not Q4_K (DEC-4 Inkling / DEC-6 Kimi
extraction with a different quant mix) decodes silent garbage rather than
erroring. Fix: gate on `ffn[2].1 == "Q4_K"`, fall back otherwise.

---

## 2. Security

### 2a. Allocation bomb in the multi-layer decoders [security, HIGH] — ✅ FIXED

`crates/larql-inference/src/ffn/moe_remote/multi_layer_wire.rs:105,146,211,254,292`
call `Vec::with_capacity(n)` directly from attacker-controlled `u32` length
fields with no bound against the actual body size. A 16-byte request
(`[n=1][layer=0][hidden=0xFFFFFFFF][ne=0]`) triggers a ~17 GB reservation →
`handle_alloc_error` process abort (uncatchable `SIGABRT`, kills the whole
server, all in-flight work dies).

Reachable **unauthenticated by default** at `POST /v1/experts/multi-layer-batch`
and `…-batch-q8k` (`routes/expert/multi_layer_batch.rs:55,126`). The 64 MB
`DefaultBodyLimit` does not help — the size comes from a header field, not the
body length. Client-side `decode_multi_layer_response` (:146) has the same bug:
a malicious shard can crash the router/inference client identically.

This is a **regression from the codebase's own guard pattern**: `codec.rs`,
`q8k_wire.rs`, and the expert `wire.rs` decoders all compute a
`max_possible_entries` / `want` bound before allocating, explicitly citing
"aborting the process — see PR 104 CI". The three multi-layer decoders never
got that treatment.

**Fix:** before each `Vec::with_capacity(n)`, bound `n` by
`remaining_bytes / min_element_size` (or compute a total `want` and reject
`bytes.len() < want`), mirroring `decode_expert_request`. Same for
`read_f32_slice`/`read_i16_slice`.

### 2b. Data-plane HTTP is unauthenticated unless `--api-key` is set [security, MED]

`bootstrap.rs:1145` attaches `auth::auth_middleware` only `if
cli.api_key.is_some()`. The `--grid-key`/`LARQL_GRID_KEY` guards **only** the
server→router announce bearer and the router's `join` check — it never guards
the server's own FFN/expert HTTP surface. On a Vast host with adversarial
peers, `/v1/walk-ffn`, `/v1/experts/*`, `/v1/expert/*` are open. In particular
`GET /v1/shard/{model_id}/{range}` (`routes/shard.rs:50`) **streams the entire
on-disk vindex directory as a tar to any caller** — full model-weight
exfiltration — with `model_id` discoverable via unauthenticated `/v1/models`
and `/v1/stats`.

**Fix direction:** make auth mandatory for the data plane in DEC deployments,
or bind these hosts to a private network; at minimum require the grid/api key
on `/v1/shard` and the expert endpoints.

### 2c. Router admin gRPC RPCs are unauthenticated [security, MED]

`crates/larql-router/src/grid/service.rs:386,397,455` — `status`,
`drain_server`, `assign_range` are unauthenticated even when `grid_key` is set
(only `join` checks it, :99; the module doc at :17 admits this). An adversarial
peer reaching the router's gRPC port can enumerate every replica (`status`),
force-drain/unassign shards (`drain_server`, grid-disruption DoS), or
manipulate topology (`assign_range`). `server_id` is guessable —
`alloc_server_id` is `srv-<unix_secs>-<n>` with a small monotonic `n` (:78).
Fix: apply the bearer check to the admin RPCs, or enforce that the gRPC port is
never network-exposed.

### 2d. Grid-key comparison is non-constant-time [security, LOW]

`crates/larql-router/src/grid/service.rs:105` uses `token != expected` — a
bytewise-timing side channel on the grid key. The HTTP path was deliberately
hardened for exactly this (`auth.rs` uses SHA-256 + `subtle::ConstantTimeEq`);
the gRPC grid path did not get it. Fix: hash-then-`ct_eq`.

### 2e. BF16-monolith expert stride overflow [security, LOW]

`crates/larql-server/src/routes/expert/single.rs:93` and `expert/cpu.rs:115` —
`expert_id * gu_stride` can wrap in release builds, reading a different
expert's weights (correctness/info) or panicking on a slice index (caught by
`spawn_blocking` → 500, no process crash). Only the legacy non-`per_layer_ffn`
BF16 path; the Q4K per-layer path DEC uses takes the bounds-checked
`get_layer_entry_bytes` branch. Fix: `checked_mul`/`checked_add`.

---

## 3. Fleet / config landmines (block the x86 + Linux arms)

### 3a. Grid join advertises `127.0.0.1` on the fleet [hardcoding, HIGH]

`crates/larql-server/src/bootstrap.rs:1222` — with default `--host 0.0.0.0`
and no `--public-url`, a `--join`'d server announces `listen_url =
http://127.0.0.1:<port>`. On a multi-host grid (DEC-1 grid arms, any x86
fleet) every shard advertises loopback and the router routes traffic to
itself. Fix: refuse `--join` with a wildcard host unless `--public-url` is set,
or detect the outbound-interface IP.

### 3b. CPU backend cannot run the remote-FFN decode loop [modularity, MED]

`generate_with_remote_ffn*` hard-fails when `decode_token_with_moe` returns
`None` (`grid/remote_ffn.rs:64,134,408,502`); `CpuBackend` implements
`DecodeBackend` with all defaults (`cpu/mod.rs:167`), so it returns `None`.
The remote-**MoE** CPU path dodges this via a completely different decode stack
(larql-kv engine + `RemoteMoeFfn`, `run_cmd.rs:751`) selected by an `if metal`
CLI branch — two divergent decode stacks chosen by a flag, not by capability.

**DEC impact:** `larql bench --ffn` (the DEC-0/DEC-1 single-stream *anchor*)
and `larql dec-bench capture` (`capture_runtime.rs:43`) only work on
macOS+Metal. The spec's "CPU attention until G-ladder" is true for *replay*
(model-free, portable pool) but **false for the anchor and capture** on Linux.
Near-term fix: make the Metal requirement explicit + documented (capture the
pool on a Mac, ship it — it is host-portable). Full fix (CPU fused decode) is
G-ladder-adjacent and larger; do not rush it.

### 3c. Backend construction is a 7-site copy-pasted `--metal: bool` + cfg block [modularity, MED]

Identical `if metal { #[cfg(all(feature="gpu", target_os="macos"))] … } else {
cpu }` at `run_cmd.rs:649,924`, `bench/remote_ffn_runtime.rs:77`,
`bench/local_runtime.rs:58`, `dec_bench/capture_runtime.rs:43`,
`shannon_cmd.rs:971`, `walk_cmd.rs:462`, plus behavioural branching on the same
bool (`run_cmd.rs:723,1289`). Adding `--cuda`/G3 today means touching every
site and every `metal: bool` arg struct. The `Capability` enum
(`larql-compute/src/backend/capability.rs:40`) already exists for exactly this.
Fix: one `backend_from_spec(&str) -> Box<dyn ComputeBackend>` factory + switch
`if metal` dispatch to capability probes — **before G1 lands**.

### 3d. `--metal` hardcoded in the DEC-0 script [hardcoding, HIGH]

`scripts/dec0-loopback.sh:80,97` hardcode `--metal` in the anchor and capture
arms; `dec-bench capture` hard-errors without the gpu feature on macOS
(`capture_runtime.rs:52`), so the script cannot run on the DEC-0.5 x86 box
without editing. Fix: a `DEC0_BACKEND` env defaulting per-platform (couples to
3c).

### 3e. `SKIP_MOE` vs `LARQL_SKIP_MOE` — two names, docs disagree [hardcoding, HIGH] — ✅ FIXED

Grid path reads unprefixed `SKIP_MOE` (`moe_remote/runtime.rs:64` →
`grid/config.rs:25`); local compute path reads `LARQL_SKIP_MOE`
(`compute/options.rs:70`). `README.md:649` documents `LARQL_SKIP_MOE`;
`docs/dec-funnel.md` DEC-0 says `SKIP_MOE`. The "SKIP_MOE ceiling 56.8" anchor
measures a different thing depending on which name the operator types, and
unprefixed `SKIP_MOE` can collide with a rented host's ambient env.
`SKIP_OUTER_NORM` and `DECODE_DEBUG` are also unprefixed. Fix: one prefixed
name, alias the old one with a loud warning.

### 3f. `--backends metal` default fails on Linux [hardcoding, MED]

`bench/args.rs:26` defaults `--backends` to `"metal"`; plain `larql bench
<model>` hard-errors on Linux (`bench/local_runtime.rs:66`). Fails loudly, but
every DEC runbook command needs `--backends cpu` appended on x86. Fix:
platform-conditional default (couples to 3c).

### 3g. `--default-timeout` values assume 26B-class latency + LAN [hardcoding, MED]

Shard HTTP 30s (`moe_remote/config.rs:47`), remote-FFN 60s (`remote/http.rs:101`),
server `--infer-timeout-secs` 60 (`bootstrap.rs:477`), router shard 120s
(`main.rs:196`). Inkling ~975B prefill or first-touch mmap page-fault storms on
a cold rented box can exceed these → spurious 504s/aborts mid-session. All
configurable; the *defaults* are the landmine. Note at DEC-4/5 provisioning.

### 3h. `grid_lan_runtime` run timeout is a no-op [hardcoding, MED]

`bench/grid_lan_runtime.rs:179` — `let _ = opts.timeout_secs; // future` while
`command.output()` blocks forever. A hung shard wedges the whole grid-LAN
matrix in an unattended multi-hour session.

### 3i. Grid auth optional = open grid port [security/hardcoding, MED]

`--grid-key` unset → "grid port is open to any server (development only)"
(`larql-router/src/main.rs:206`, `bootstrap.rs:695`). On rented boxes with
public IPs anyone can join/poison the shard map. Make the key mandatory
off-loopback. (Overlaps 2c.)

### 3j. Grid-LAN baselines keyed by model only [hardcoding, MED]

`scripts/bench-grid-regress.sh:35` — `bench/baselines/grid-<model>.json` has no
host/arch dimension; an EPYC run compared against an M3 Max baseline yields a
meaningless verdict, or silently seeds an EPYC baseline that later gates a Mac
run. Fix: add an arch/host tag to the baseline key.

---

## 4. Structural pre-work (schedule per-ladder, not a DEC-0 blocker)

### 4a. Dense-FFN vs MoE wire stacks are parallel implementations [modularity, MED-LARGE]

`ffn/remote/*` (dense) and `ffn/moe_remote/*` (MoE) share no transport, codec,
negotiation, or byte accounting. Dense is HTTP-only with Accept-header
negotiation; MoE has HTTP + UDS + gRPC with endpoint-per-format. Adding one
new activation wire format (the DEC-6a MXFP4/bf16 question) currently touches
~7 places. The UDS transport the DEC-0 loopback arms would benefit from exists
only on the MoE side. `call_q8k_layers` (`remote/http.rs:364`) never increments
the wire-byte atomics, so `bench --ffn` `wire_bytes_per_tok` undercounts on the
default Q8K predispatch path (dec-bench does its own counting, unaffected).
Consolidate at the first new wire format (DEC-6a decision point).

### 4b. Server-side expert dispatch is per-handler, three shapes [modularity, MED]

`q8k.rs:107` inline Metal-vs-CPU cfg block; `grpc_expert.rs:178` a second,
differently-shaped one; `expert/layer_batch.rs:105` + `expert/multi_layer_batch.rs:70`
hard-wire `run_experts_cpu_batch` with **no GPU path at all**. A CUDA expert
backend (G4, "mirroring metal-experts") would be spliced into each handler.
Extract one `run_experts(state, backend, …)` dispatcher — before G4.

### 4c. dec_bench `Endpoint` seam for the routed-experts arm [modularity, SMALL-MED]

Endpoint is conflated with wire format (`WireArm::Q8k.accept() → None` "is its
own endpoint"); `send_one` hard-branches on the two walk-ffn paths; the run
record hardcodes `endpoint: "walk-ffn"`. The capture pool stores only
pre-normed residuals — routed replay needs per-step per-layer `(expert_ids,
weights)`, which breaks the "replay is model-free" property. The
movement-ratio denominator is dense-only in the driver (the server side is
already ready — `/v1/stats` publishes `moe.per_expert_bytes`). Introduce an
`Endpoint` enum (path + frame-builder + response-decoder + denominator source)
before the routed-experts fast-follow (which gates the C1-on-MoE verdict).

### 4d. Walk-ffn binary frame re-declared per crate [modularity, SMALL]

Client encoder (`ffn/remote/codec.rs`) and server decoder
(`routes/walk_ffn/binary.rs`) are independent hand-maintained implementations
of one layout; the CT string `application/x-larql-ffn` is declared in **5**
places (`codec.rs`, server `types.rs`/`wire.rs`/`http.rs`, `larql-router/http.rs`);
`BATCH_MARKER` twice. Contrast: q8k and all MoE codecs are single-sourced in
larql-inference and imported by the server. Any DEC header extension must be
hand-synced across encoder/decoder + router. Consolidate opportunistically
with 4a.

### 4e. Duplicated MoE router-weight construction + combine math [modularity, SMALL]

`build_moe_router_weights` is a **private** helper (`kquant_forward/hidden.rs:93`);
the server hand-rolls the identical 10-field construction at
`routes/walk_ffn/core.rs:111`, plus 3 more sites in a server example. The
outer-norm + residual combine is duplicated inline (`core.rs:153` vs
`hidden.rs:121`) with no shared function or cross-test. K3's Stable LatentMoE
routing changes this structure — make the helper `pub` and share the combine
before the DEC-6b KDA/LatentMoE port.

### 4f. Vestigial / asymmetric bits [modularity, SMALL each]

- `/v1/expert/batch` labelled "pre-2026-05-01 legacy" (`routes/expert/mod.rs:12`)
  but still the only path used by `forward_moe_seq` prefill
  (`moe_remote/backend.rs:248`) and still mounted — a replay/CUDA implementer
  can target the wrong wire.
- `RemoteWalkBackend::forward_moe_full_layer` is the last JSON-float-array hot
  path (`remote/http.rs:535`); the server binary decoder can't express
  `moe_layer` (`binary.rs:76` hardcodes `false`).
- i8 wire is server-side **default-off** behind `LARQL_I8_WIRE` (`wire.rs:37`),
  while dec-bench's default sweep includes the i8 arm — a default DEC-1 run
  measures f32 on the i8 arm unless the rig sets the env var (`served_wire`
  records the fallback, so it is visible, but it is a plan-level footgun for
  the netem harness scripts).
- Q8K responses carry no server-latency field, so the Q8K arm loses the
  server/network decomposition all other DEC-1 arms have.
- `forward_predispatch_all_q8k` signals failure by inserting all-zero vectors
  and the caller heuristically re-dispatches on "any all-zero row"
  (`sharded.rs:199`, `grid/remote_ffn.rs:271`) — a legitimately all-zero FFN
  output silently switches wire paths.

### 4g. Twin-file duplication [hardcoding/modularity, LOW]

`larql-compute/src/kquant_forward/walk_ffn.rs` and
`larql-inference/src/vindex/kquant_forward/walk_ffn.rs` are ~160-line
near-identical copies (diff is only the index type) — the bug-fix-twice hazard
that produced the historical q4_common subnormal bug. `LayerLatencyTracker`
global `Mutex<HashMap>` is fine at DEC scale (tiny critical section).

---

## 5. Env-var inventory (serving path)

**Change numerical output:** `SKIP_MOE` (unprefixed), `LARQL_SKIP_MOE`,
`LARQL_MOE_TOP_K`, `LARQL_MOE_WIRE_F16`, `LARQL_I8_WIRE`,
`LARQL_F16_WIRE_DISABLE`, `LARQL_DISABLE_Q8K_WIRE`, `LARQL_DISABLE_Q4K_DIRECT` /
`LARQL_Q4K_DIRECT` / `LARQL_Q4K_DIRECT_FFN`, `LARQL_Q4K_ASM`,
`LARQL_USE_METAL_EXPERTS` / `LARQL_DISABLE_METAL_EXPERTS` /
`LARQL_USE_LEGACY_CPU`, `SKIP_OUTER_NORM` (unprefixed).

**Dispatch/perf:** `LARQL_MOE_BATCH_MODE`, `LARQL_COMPUTE_CONCURRENCY`
(**load-bearing, undocumented** — `layer_batch.rs:48`), `LARQL_MOE_NO_SPLIT`,
`LARQL_SPIN_POOL`, `RAYON_NUM_THREADS`, `LARQL_MOE_CACHE_ENTRIES`,
`LARQL_NO_WARMUP`.

**Diagnostics (stderr):** `LARQL_MOE_TIMING`, `LARQL_MOE_BYTES`,
`LARQL_MOE_SHARD_TIMING`, `LARQL_HTTP_TIMING`, `LARQL_KERNEL_TIMING`,
`LARQL_MOE_FWD_TIMING`, `LARQL_VERBOSE`, `LARQL_DECODE_STAGES`,
`LARQL_PROFILE_SPLIT`, `LARQL_METAL_VS_CPU_DEBUG`.

**Auth/scripts:** `LARQL_GRID_KEY`; `LARQL_BENCH_VINDEX`, `LARQL_BENCH_FFN_URL`,
`LARQL_TOK_PER_S_THRESHOLD`, `LARQL_P99_THRESHOLD`.

Undocumented-but-load-bearing: `LARQL_COMPUTE_CONCURRENCY`, `SKIP_MOE` under its
real (unprefixed) name, `LARQL_DISABLE_Q8K_WIRE`. (This overlaps the
2026-06-12 review's item-7 env-flag-registry action, which remains open.)

---

## Performance findings not already covered

- **No admission control on compute [perf, HIGH]:** every request is one
  `spawn_blocking` (`handler.rs:79,115`, `q8k.rs:51`) on the stock 512-thread
  blocking pool; 4 clients × 48-layer fan-out = ~192 simultaneous
  multithreaded-BLAS tasks on 8–16 cores. **DEC-2's "≥80% linear at N=4"
  (C3) is exactly where this shows, and the failure looks like tier
  saturation when it is scheduler oversubscription.** Fix: a
  `tokio::sync::Semaphore` sized ~physical cores gating the spawn_blocking
  bodies, and pin BLAS to 1 thread per call for the serving build
  (`OPENBLAS_NUM_THREADS=1`).
- **`--release-mmap-after-request` thrashes concurrent requests [perf, HIGH
  when set]:** `handler.rs:88`, `q8k.rs:183` call `release_mmap_pages()` after
  every request; with 4 clients each completing request evicts pages the other
  three are streaming → re-fault storm (the known madvise-churn class,
  per-request now), and in the q8k Metal path releases pages still referenced
  by zero-copy Metal buffers. Never release while `requests_in_flight > 1`;
  any DEC point run with this flag is measuring page faults — flag in run-record
  hygiene.
- **q8k endpoint invisible to drain / heartbeat / latency tracking [perf,
  MED]:** `q8k.rs:41` bumps only `bump_requests()` — no `RifGuard`,
  `requests_total`, or `layer_latency_tracker.record`. GT6 drain can conclude
  "no in-flight" during a q8k burst, and the C7 latency-EMA router
  (`HeartbeatMsg.layer_stats`) is blind to exactly the traffic DEC generates.
  Add the guard + counters + per-layer record.
- **Client fan-out spawns 30–48 OS threads per token, every token [perf,
  MED]:** `sharded.rs:119` (and mirrored `dec_bench/replay_runtime.rs`) use
  `std::thread::scope`/`s.spawn` per layer per step. Persistent worker pool or
  async `buffer_unordered`. The q8k grouping path already reduces spawns to one
  per shard — the f32 path should follow.
- **`model.patched` fair RwLock held across FFN compute [perf, MED]:** a
  queued patch/insert write (`patches.rs:155`) blocks all new readers across
  all clients until every in-flight batch-64 compute drains — correlated p99
  spikes; a landmine for the shared-tier demo (C3 is precisely concurrent
  patch + serve). Arc-swap/epoch snapshot of the overlay.
- **4–6 full-buffer passes per request [perf, MED]:** `req.residual.clone()`
  into Array2 (`core.rs:48,235`), `out.into_iter().collect()` (:284, vs the free
  `into_raw_vec_and_offset` used two lines away), per-element byte-push encode
  loops (`binary.rs:88,102,123,157`; f16 scalar per element), encode running on
  the async worker after `spawn_blocking` returns. Scales with B while the warm
  GEMM amortises — grows as a fraction of step time exactly where C1 is
  measured, and biases the f16/i8 arms of C6. Take residual by value, bulk
  `bytemuck::cast_slice` copies, chunked f16.

---

## What is already solid (verified)

- B-row **f32/f16/i8** path is genuinely vectorised: one reshape → one BLAS
  GEMM per projection over a lock-free per-layer f32 cache (single atomic load
  + `Arc::clone`), no per-row re-setup, no per-row locks, norm-free.
- Wire decoders (except the three multi-layer ones) are hardened:
  overflow-checked lengths, `num_entries` allocation bombs rejected, dedicated
  `reject_impossible_*` tests; `q8k.rs` validates block shape before any kernel
  touch.
- Compute is correctly off the async runtime; body reads bounded at 64 MB;
  `ConcurrencyLimitLayer` + `compute_semaphore` present.
- `auth.rs` HTTP bearer is constant-time SHA-256; no secrets committed; grid
  key never logged; scripts are `set -euo pipefail` and fully quoted; no
  request-derived filesystem paths or shell injection.
- The gRPC **proto is the single source of truth** (`tonic::include_proto!`);
  no re-declared proto shapes.
- MoE/q8k HTTP codecs are single-sourced and the loadgen reuses production
  codecs down to the truncation guards — the parity discipline is real.
- `/v1/stats` already publishes the routed-replay movement-ratio denominator
  (`ffn_weights.moe.per_expert_bytes` + `num_experts`/`top_k`).
- `DecodeBackend`/`Capability` is a workable CUDA seam; `metal-experts` server
  feature is a ready template for `cuda-experts` (G4).
- dec_bench pure/IO split with served-wire fallback recording — honest
  metrology baked into the structure.
- Weights lazy-load is single-flighted with a lock-free fast path; `RifGuard`
  is drop-based and saturating (on the f32 endpoint).

---

## Refuted / non-issues checked

- ~~No request-derived value builds a filesystem path or shell command anywhere
  in the reviewed surface (the 2026-06-12 `model_id` traversal item was already
  fixed; `/v1/shard` streams the server-controlled `model.path`).~~

  > **WRONG — corrected 2026-08-22. This closed a live finding.** The
  > 2026-06-12 item concerned the *download* side and was never fixed;
  > `git log --since=2026-06-12 -- crates/larql-server/src/shard_loader.rs`
  > was empty. `download_and_load_shard` joined `model_id` straight into
  > `PathBuf::from(store_path).join(model_id)` and `create_dir_all`'d it,
  > and that `model_id` arrives in the router's `AssignMsg` — remote
  > input, not local config. A peer able to reach the announce socket
  > could set `model_id` to `../../../../etc/cron.d` and choose where a
  > downloaded tar was unpacked.
  >
  > Fixed the same day: `validated_model_id` refuses anything that is not
  > a single path segment of `[A-Za-z0-9._-]` under 128 bytes, and
  > `shard_dest_path` returns `Option` so a caller cannot fall back to an
  > unvalidated path. Gated both ways — hostile ids refused, and real ids
  > still accepted and still inside the store, because a validator that
  > refused everything would have passed a one-sided test.
  >
  > The scoping error is the lesson: the earlier finding named one call
  > site, this review checked "the reviewed surface", and the sibling
  > path in another file was in neither. A traversal finding should be
  > closed against every use of the tainted value, not against the site
  > that raised it. Note `/v1/shard` streaming `model.path` was and
  > remains fine — that half was right.
- `gate_knn` clamps `top_k` from the wire (`gate_knn/mod.rs:94`,
  `dispatch.rs:283`) — no unbounded-heap DoS.
- Client response validation checks every shard output length against expected
  `hidden` — a malicious server can't corrupt shapes silently (aside from 2a).
- `LayerLatencyTracker` / `FfnL2Cache` locking is not a DEC-scale contention
  issue; replay's `top_k=0` bypasses the L2 cache entirely.
