# Vindex-as-FFN Lookup — Specification

**Status:** Draft.
**Source experiments:** Shannon Exp 51 (FFN-remote/attention-local split),
Exp 52 (vindex-as-FFN decision boundary), Exp 53 (sharded vindex deployment).
**Crates touched:** `larql-inference` (FFN backend, `WalkFfn` decorator),
`larql-vindex` (compiled-fact cache shape), `larql-server` (FFN-lookup
endpoint), `larql-router` (layer-fanout, already in place).
**Depends on:** ADR-0002 (FFN activation cache, L1/L2), ADR-0006 (Q4_K
dense-remote FFN), ADR-0003 (FFN router).

## 1. Purpose

Add a curated whole-FFN-output cache, keyed on cosine similarity between
the layer's RMSNorm'd input and a set of pre-compiled fact inputs, layered
on top of the existing `WalkFfn` backend. On a hit, return the stored
`mlp_out`. On a miss, fall through to `WalkFfn` unchanged.

Exp 52/53 established the empirical case:

```
hits at cos = 1.00
decoys cap at cos ≤ 0.39
gap is 0.6 wide and empty
0/6 false positives across decoys
```

The gap is not a tuning knob. The threshold (`tau`) is a safety floor the
operator sets once above the decoy ceiling and forgets. Loopback shard RTT
is 0.085 ms mean / 0.143 ms p99 at 10 KB/query — 0.2% of a 40 ms decode
step.

The mechanism is opt-in per layer, additive, and never replaces existing
paths. Every other code path in `larql-inference` continues unchanged.

## 2. Contract

### 2.1 Lookup contract

For a query input `q ∈ ℝ^d` (a `pre_feedforward_layernorm(h_attn)` slice —
see §4.1) at layer `L`, the lookup returns one of:

```
Hit(mlp_out)   if max_i cos(q, gate_input_i) >= tau
Miss           otherwise
```

The lookup is deterministic, stateless, and side-effect-free. Two queries
with identical bytes produce identical results; ordering of stored entries
does not affect the result.

### 2.2 Correctness contract

Generation under the lookup-enabled FFN backend must produce the same
first token as the `WalkFfn`-only reference path on at least the
calibration set. Exp 53 measured 10/10 facts preserved end-to-end. The
contract is:

```
greedy_top1_agreement(lookup_path, reference_walkffn_path)
    >= 99% on calibration corpus
    >= 95% on held-out prompts

early_div(lookup_path) <= early_div(reference_walkffn_path)
    over the first 5 generated tokens, on the same calibration corpus
```

Where `early_div(path)` is defined as: *the fraction of prompts in the
corpus where `path`'s greedy continuation diverges from the FP16
reference greedy continuation within the first 5 generated tokens.* It
is a non-negative scalar in `[0, 1]`; lower is better. The inequality
above is "the lookup path's divergence rate is no worse than the
WalkFfn-only path's at the same parameters."

"Calibration corpus" is whatever `tau` was chosen against (see §6).

### 2.3 Hit-rate contract

The hit-rate floor is **not** part of this spec. A vindex with no compiled
entries hits 0%; a fully compiled domain hits ~14–20% per ~7-token fact
prompt (Exp 52). The cache fires on answer-bearing positions only; this
is a feature, not a knob.

### 2.4 Miss path

A miss is silent. The miss path runs `WalkFfn` exactly as if the lookup
backend were not configured. No log spam, no retry, no degraded mode.

## 3. What this engine does NOT promise

Explicit non-contracts, so future contributors don't accidentally rely on
behaviour that was never in scope:

- **Replacement of `WalkFfn`.** The lookup is a decorator above `WalkFfn`,
  not a substitute. `WalkFfn` continues to be the FFN backend.
- **Wall-clock speedup at any hit rate.** §5.4 is the cost model; whether
  it pays back depends on `K` (sparse-walk top-K), N (compiled-fact
  count), and `tau` (admittance band). The spec defines correctness, not
  a perf claim.
- **Replacement of the L1/L2 activation cache.** The existing
  `FfnL1Cache` (sorted-feature-ID keyed) and L2 server cache continue to
  work. They catch a different thing (exact-activated-feature-set hits)
  and may overlap with this lookup; ordering is defined in §7.
- **Cross-architecture generality.** Calibration is per-architecture
  (§6, layer policy). A `tau` and layer choice good on Gemma 3 4B does
  not transfer mechanically to Llama or Mistral.
- **Cross-vindex state portability.** Compiled-fact entries reference a
  model's residual geometry. Entries compiled against vindex `A` are not
  meaningful when mounted against vindex `B`, even if `A` and `B` share
  hidden dim. Storage carries a model fingerprint; mismatched fingerprints
  are a hard refuse-to-load.
- **Training-time use.** Lookup is inference-only. Gradient flow through
  cached `mlp_out` rows is not supported and not planned.
- **Multi-tenant batching.** The lookup is single-position, single-tenant
  by §2.1. Batched-prefill consideration is out of scope (§4.5 below).
- **Attention-time interactions.** The cache is FFN-only. There is no
  parallel attention-output cache in this spec. The Markov-residual
  engine is the right home for residual-stream caching at attention scope.
- **Auto-tuning of `tau`.** §6 — the threshold is a safety floor with
  near-zero variance in measured behaviour across the [0.40, 0.97] range.
- **Cross-layer compiled chains.** Each compiled layer is independent.
  Exp 52 established 0/6 false positives across decoys at one layer; that
  generalises additively to multi-layer, but composing compiled chains
  for multi-step reasoning is a different problem.

## 4. Inputs

### 4.1 Query shape

The query is the FFN input — i.e. the output of the layer's pre-FFN
RMSNorm. This is the boundary identified in Exp 51:

- σ ≈ 0.55, absmax/σ ≈ 30 — bounded by RMSNorm construction
- BF16 transport at 174 KB / decode-step (full stack, 34 layers) is fine
  on any reasonable link
- int8 quantisation of this signal **fails** (top1 ≤ 83% at best,
  opposite to the final-layer Exp 43 result). Query stays in BF16 / f16
  on the wire

### 4.2 Index shape

A compiled-fact cache stores entries:

```
{ gate_input: [d_model],   # f16 or BF16, the same shape as the query
  gate_input_norm: f32,    # precomputed L2 norm for cosine
  mlp_out:    [d_model],   # f16 or BF16, the WalkFfn output for that input
  source_id:  optional fact / prompt id, for debugging only }
```

`gate_input_norm` is **precomputed at compile time and stored**. The
cost model in §5.4 assumes this; without it, every cosine sweep would
recompute N stored-side norms on top of the dot products, doubling the
sweep cost.

Layer-bound: each layer has its own index. Cross-layer entries are not
permitted; lookup at L_query never reads entries compiled at L_other.

### 4.3 Storage

Two acceptable layouts; the implementer picks one:

- **Sidecar mmap.** New optional file in the vindex (e.g.
  `compiled_ffn_cache.bin` + `compiled_ffn_cache.meta.json`), one row per
  entry, mmap'd on load. Hardlinked through `COMPILE CURRENT INTO VINDEX`
  unchanged.
- **Patch-resident.** Entries live inside the active patch (`.vlp`) and
  are baked in by `COMPILE CURRENT INTO VINDEX`.

Either way, base files remain immutable. INSERT/UPDATE of compiled-fact
entries goes through the standard patch overlay. A model fingerprint
header is required (§3, cross-vindex non-portability).

### 4.4 Decode-only scope

The lookup is consulted only when `seq_len == 1`. Prefill (`seq_len > 1`)
**bypasses** the lookup, matching the existing L1/L2 FFN-output cache
convention.

### 4.5 Batched-prefill consideration

Batched prefill across multiple positions is out of scope; the cache key
is per-position and the win is per-position. If a future caller wants
batched lookup, this spec does not preclude it but does not specify it.

## 5. Relationship to `WalkFfn` and the existing L1/L2 caches

This is the section the user-feedback cycle turned up as load-bearing.
The implementation must understand it, and the spec must say it
explicitly: *the FFN backend below this lookup is `WalkFfn`, not a dense
matmul.*

### 5.1 What `WalkFfn` already does

`WalkFfn` is the FFN backend in `larql-inference` since the April 2026
vindex unification. Its decode-time hot path is `walk_ffn_sparse`:

```
1. gate_knn(layer, residual)  ->  top-K (feature_id, gate_score) pairs
2. for each of K features:
     up_score   = dot(up_row(feat),   residual)
     activated  = activation(gate_score) * up_score
     mlp_out   += activated * down_row(feat)
```

When `K >= num_features`, the per-feature loop is mathematically
equivalent to three dense matmuls; that branch routes through BLAS gemm
or Q4K direct matmul. Otherwise the SAXPY loop is the path.

There is **no separate "dense FFN" backend** to fall back to. The miss
path is "run `WalkFfn` with the configured `K`," which may itself be
sparse (typical decode), full-K dense (specific configurations), or
overrides-aware (mid-INSERT session).

### 5.2 What the existing L1/L2 cache already catches

`FfnL1Cache` (in `crates/larql-inference/src/vindex/l1_cache.rs`) is an
in-process whole-FFN-output cache, keyed on the **sorted top-K feature
IDs** for the sparse path or on a **quantised-residual hash** for dense
paths. The quantised-residual key catches paraphrase pairs at
`cos >= 0.999` (i16 ulp-equivalent) — already.

L2 is the `larql-server` cross-client equivalent.

### 5.3 What this lookup adds on top

Different key, different reach:

```
L1/L2 hit          : exact activated feature set (sparse)
                     OR cos >= 0.999 (dense, via i16 quantisation)
Compiled-fact hit  : cos >= tau against any stored gate_input, where
                     tau in [0.40, 0.97]
```

The new value is **paraphrases under cos < 0.999**: prompts that activate
slightly different feature sets but still semantically retrieve the same
fact. Exp 52 measured paraphrases at cos 0.73–0.99 — squarely in the
range L1's i16 quantisation does not collapse.

The lookup is therefore not "another whole-output cache"; it is a
broader-reach cache, deliberately at the cost of a curated index instead
of opportunistic warming.

### 5.4 Cost model and break-even hit rate

Per-decode-step, per-compiled-layer (assuming precomputed stored-side
`gate_input_norm` per §4.2):

```
Cost(WalkFfn, sparse-K)
    = gate_knn(d_model, num_features)           # KNN over gate vectors
    + 2 * K * d_model                           # K up-dots + K down-saxpys
    + K * activation                            # SiLU/GELU per feature

Cost(lookup, N entries)
    = N * d_model                               # N cosine dot products
    + N                                         # N divides by stored norm
    + 1                                         # argmax + threshold
    + (hit) ? d_model     : 0                   # mlp_out memcpy on hit
    + (miss) ? Cost(WalkFfn, K) : 0             # fall-through on miss
```

The lookup *always* pays the cosine sweep cost, regardless of hit/miss.
Expected per-step cost over hit rate `h` (ignoring `gate_knn`, the `+N`
divides, the activation cost, and constants — keeping only the
d_model-scaled terms):

```
E[Cost(lookup)]   = N*d_model
                  + h     * d_model              # hit: mlp_out memcpy
                  + (1-h) * 2*K*d_model          # miss: full WalkFfn

E[Cost(WalkFfn)]  = 2*K*d_model
```

Net win condition `E[Cost(lookup)] < E[Cost(WalkFfn)]`:

```
N*d_model + h*d_model + (1-h)*2*K*d_model   <   2*K*d_model
N + h + 2K - 2Kh                            <   2K              (÷ d_model)
N + h - 2Kh                                 <   0
N + h*(1 - 2K)                              <   0
N                                           <   h * (2K - 1)
```

For non-trivial `K`, `2K - 1 ≈ 2K`:

```
N  <  2 * h * K
```

The hit rate `h` is in the inequality — it does not factor out. **At
typical `h`, the break-even is much tighter than `N < 2K`.**

Plugging Exp 52's measured fact-domain hit rate (`h ≈ 0.20`):

| K_layer | Crossover (h = 0.20) | Crossover (full-hit, h = 1.0) |
|--------:|---------------------:|------------------------------:|
| 8092    | N < 3,237            | N < 16,184                    |
| 1024    | N < 410              | N < 2,048                     |
| 256     | N < 102              | N < 512                       |
| 64      | N < 26               | N < 128                       |
| 10      | N < 4                | N < 20                        |

So at K=10 with a 20% hit rate, more than 4 compiled facts is
wall-clock-negative against `WalkFfn` itself. The qualitative conclusion
holds — full-K and high-K configurations are friendly to large
compiled-fact corpora; low-K configurations are not — but the
quantitative threshold is roughly 5× tighter at `h = 0.20` than the
naïve `N < 2K` calculation suggests.

The implementation **should refuse to enable** the lookup when

```
N > 2 * h_ref * K_layer
```

at engine construction, with `h_ref = 0.20` baked in as the reference
hit rate from Exp 52 fact-domain measurements. The inequality is
strict (`>`, not `≥`): the rule refuses configurations that are
strictly *wall-clock-negative* against `WalkFfn`-only. Net-zero
configurations sitting exactly at break-even are admitted, on the
assumption that the operator chose lookup for reasons beyond wall-clock
(paraphrase reach above the L1 i16 threshold, deterministic answer-
shape on compiled facts, network-shard offloading via §9). The
refusal message must surface the inequality, the chosen `h_ref`, and a
one-line reminder that operators measuring a higher `h` in their own
deployment should raise `h_ref` accordingly. Operators with stronger-than-Exp-52
hit-rate evidence may override; the override is a deliberate "I have
my own h measurement" knob, not "I know this costs more wall-clock"
(which is a different override and not in this spec).

If `h` is genuinely unknown for a deployment domain — i.e. no measured
hit rate exists yet — the only safe `h_ref` is the Exp 52 rate. Picking
a higher `h_ref` to admit a configuration on faith is the failure mode
this guard is meant to prevent.

### 5.5 Cache stacking order

When all three layers (L1, this lookup, `WalkFfn`) are wired:

```
1. FfnL1Cache.get(residual_or_features)          # cheapest, exact-match
2. compiled_fact_lookup(query, layer)            # cosine, may hit broader
3. WalkFfn.forward(residual, layer)              # fallback compute
```

L1 is checked first because it is exact, ulp-cheap, and already keyed on
data the layer just computed. Compiled-fact lookup is only consulted if
L1 missed. On compiled-fact hit, the `mlp_out` is written into L1 as if
it had come from `WalkFfn`, so subsequent identical residuals in this
session bypass even the cosine sweep.

L2 (server-process) sits in the same position one tier out; `larql-server`
checks L2 before delegating to its in-process pipeline.

## 6. Threshold (`tau`)

`tau ∈ [0.40, 0.97]` — every value in that range produces equivalent
behaviour on Exp 52's matrix for decoy rejection (0/6 false positives
across the band). The operational choice inside the band is **whether to
admit paraphrases** (cos 0.73–0.99 in Exp 52):

```
tau = 0.50   :  admits all measured paraphrases, rejects all measured decoys
                with a 0.11 buffer above the decoy ceiling. DEFAULT.
tau = 0.73   :  the explicit paraphrase-floor; same admittance as 0.50
                but with no decoy buffer.
tau = 0.97   :  exact-prompt-only mode. Excludes all paraphrases except
                those at near-identity. Use when paraphrase admittance is
                undesirable for a particular deployment.
```

The default is **0.50**, not 0.97. The earlier framing ("0.97 is the
safety floor") was inverted: 0.97 is the most restrictive number in the
empty band, and choosing it as default silently excludes the paraphrase
behaviour Exp 52 demonstrated as safe. 0.50 sits in the middle of the
empty `[0.40, 0.73]` decoy-to-paraphrase gap and admits paraphrases out
of the box.

`tau` is set per layer-index (not per query, not per session). It lives
in the index metadata and is fixed at compile time. Operators wishing
exact-only behaviour write `tau = 0.97` into the index metadata; this is
a deployment knob, not a correctness knob.

There is no runtime `tau` override. The metadata value is final for the
life of the loaded index. Re-tuning `tau` is a recompile, not a flag.

## 7. Layer policy

Default compiled layer: **L26** on Gemma 3 4B (the answer-crystallisation
layer identified in MI01 / Exp 52). Per-architecture defaults are model
config:

```
gemma-3-4b: L26
gemma-3-1b: TBD (precondition: re-run Exp 52 calibration)
llama-2-7b: TBD
mistral-7b: TBD
```

Multiple compiled layers are allowed; each has its own `tau` and its own
index. Inter-layer interactions are not part of this spec — Exp 52
established cross-shard independence (0/6 false positives across decoys),
which generalises to cross-layer.

## 8. Backend wiring

A new variant in `larql-inference::ffn::FfnBackend`:

```
FfnBackend::CompiledLookupOrWalk {
    inner: Box<FfnBackend>,        # the WalkFfn-based fallback
    cache: Arc<CompiledFfnCache>,  # the per-layer compiled-fact index
    tau: f32,
}
```

`inner` will, in practice, be a `WalkFfn`-backed variant. The wrapping
shape matters: this is a decorator above any existing backend. It does
not invent a new compute path.

Decode-time call sites:

1. `query = pre_feedforward_layernorm(h_attn)`
2. `l1_cache.get(...)` — existing L1 path, unchanged
3. on L1 miss: `cache.lookup(query, layer_id)` → `Hit | Miss`
4. on `Hit`: write `mlp_out` into L1 and return it (residual add at the
   call site as today)
5. on `Miss`: `inner.forward(h_attn, layer_id)` — i.e. `WalkFfn`
6. Prefill (`seq_len > 1`) **bypasses** steps 3–5 entirely

## 9. Network mode

For sharded deployments, the lookup runs on a remote `larql-server` shard.

### 9.1 Endpoint

`POST /v1/ffn-lookup` on `larql-server`:

```
request:
  layer:    u32
  query:    [d_model] BF16     # zero-copy
  tau:      f32 (optional; index default if omitted)

response (hit):
  status:   "hit"
  mlp_out:  [d_model] BF16
  cos:      f32                # debug field

response (miss):
  status:   "miss"
  best_cos: f32                # debug field
```

The endpoint is independent of `/v1/walk-ffn`. A shard may expose either
or both; clients do not rebalance based on which one is faster.

### 9.2 Router fanout

`larql-router` already implements layer-range fanout for `/v1/walk-ffn`
(see `crates/larql-server/docs/router-spec.md`). The same shard map
applies to `/v1/ffn-lookup`: requests for layer `L` go to the shard
owning `L`'s range. No new routing logic; only a new endpoint name.

### 9.3 Wire format

- BF16 query and reply (Exp 51 — int8 of the query fails)
- ~10 KB/query observed for d_model=2560 + headers (Exp 53)
- Encoding follows the same envelope as `/v1/walk-ffn` for shared
  serialiser code

### 9.4 Latency budget

```
loopback TCP   ≈ 0.085 ms mean,  0.143 ms p99
LAN 10 GbE     ≈ 0.05  ms est.
LAN 100 Mb     ≈ 0.2   ms est.
distant LAN    ≤ 1.0   ms (≈ 1–3% of a 40–100 ms decode step)
```

This spec's deployment envelope is **single-LAN**. Distant-LAN and WAN
deployments are out of scope; they need query batching and a different
SLA model than the one the experiments calibrated against.

## 10. COMPILE integration

A new LQL clause or `COMPILE` mode that records compiled-fact entries
during `COMPILE INTO VINDEX`:

```sql
-- conceptual; exact grammar TBD in larql-lql/docs/spec.md
COMPILE FACT "<prompt>" => "<answer>" AT LAYER L26 WITH TAU 0.50
```

The compiler:

1. Runs the model on `<prompt>` to last-position
2. Captures `gate_input = pre_feedforward_layernorm(h_attn)` at layer L26
3. Captures `mlp_out = WalkFfn(gate_input)` at layer L26 (NB: `WalkFfn`,
   not a dense matmul; whatever `K` the layer is configured at)
4. Writes `(gate_input, mlp_out)` into the active patch's compiled-fact
   cache for L26

`COMPILE CURRENT INTO VINDEX` bakes the patch into a new vindex by
hardlinking unchanged base files and writing the compiled-fact cache as
a new sidecar. No existing extracted weight is rewritten.

Note: §5.4 implies `K` selection at the time of COMPILE is load-bearing.
If the operator compiles at K=8092 and serves at K=10, hits are still
correct (the stored `mlp_out` is the K=8092 output) but the lookup
becomes wall-clock-negative against the K=10 walk well before the
naïve `N < 2K` would suggest — at `h_ref = 0.20`, the K=10 crossover
is `N < 4` (§5.4 table). The compiler must record the `K` used at
compile time. The lookup should warn (not refuse) when the serve-time
`K` differs, with the warning citing the corrected break-even
`N < 2 * h_ref * K_layer` against the *serve-time* K, not the
compile-time K.

## 11. Test plan

Required tests for this spec to be marked Implemented:

```
1. Unit: cache.lookup(query) returns Hit at cos=1.00 on a stored entry,
   Miss at cos<tau on a decoy.
2. Unit: empty cache → 100% Miss.
3. Unit: tau=0.40 and tau=0.73 produce identical greedy top-1 on the
   Exp 52 fact set (validates the empty-decoy-to-paraphrase gap).
4. Unit: tau=0.97 admits same-prompt only, rejects 7/10 paraphrases that
   tau=0.50 admits (validates §6's "exact-prompt-only mode").
5. Integration: WalkFfn-only reference vs lookup-enabled backend agree
   on top-1 for at least 10 compiled facts and 20 unrelated prompts on
   gemma-3-4b. (Exp 53: 10/10.)
6. Paraphrase regression: cache compiled on "the capital of France is"
   serves a hit for "France's capital is" at default tau=0.50, and
   the served `mlp_out` produces top-1 = " Paris" downstream. Documents
   the spec's broader-reach property over the L1 i16 hash.
7. Integration: bf16 transport over 127.0.0.1 produces same result as
   in-process lookup. (Exp 53.)
8. Property: prefill is unaffected — cache is never consulted at
   seq_len > 1.
9. Property: miss path bytes-equivalent to the unwrapped WalkFfn backend
   at the same K (no extra memory traffic on miss beyond the cosine
   sweep itself).
10. Property: the cost-model refusal in §5.4 fires when
    N > 2 * h_ref * K_layer (with h_ref = 0.20 baked in). Specifically:
    at K_layer=10, refusal fires for N >= 5; at K_layer=1024, refusal
    fires for N >= 410; at K_layer=8092, refusal fires for N >= 3238.
    Operator override (with their own measured h) produces a single
    warning log line surfacing the chosen h and the resulting threshold,
    then proceeds.
11. Property: model fingerprint mismatch on storage load is a hard
    refuse (not a warning), per §3 cross-vindex non-portability.
```

## 12. References

- `~/chris-source/chris-experiments/shannon/52_vindex_as_ffn/RESULTS.md`
- `~/chris-source/chris-experiments/shannon/53_sharded_vindex/README.md`
- `~/chris-source/chris-experiments/shannon/51_ffn_remote_split/README.md`
- `crates/larql-server/docs/server-spec.md` (existing transport)
- `crates/larql-server/docs/router-spec.md` (existing fanout)
- `crates/larql-inference/src/vindex/walk_ffn/mod.rs` — WalkFfn routing
  table and the actual decode path this spec sits on top of
- `crates/larql-inference/src/vindex/walk_ffn/sparse.rs` — the SAXPY
  hot path with full-K gemv fast-path
- `crates/larql-inference/src/vindex/l1_cache.rs` — existing whole-output
  cache (sorted-feature-ID + quantised-residual keys)
- `docs/adr/0002-ffn-activation-cache.md` (related but distinct cache;
  L1/L2 is keyed on activated feature IDs, not on cosine similarity to
  stored gate inputs)
