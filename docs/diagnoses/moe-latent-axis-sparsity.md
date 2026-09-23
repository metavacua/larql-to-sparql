# MoE latent-axis sparsity — sparsify the shared expert input, not the row population

**Status:** 2026-08-01 — **static and blocked forms REFUTED; per-channel form
alive but re-priced to a ≤1.04× ceiling at the fidelity gate.**
**Kernel hook:** `crates/larql-compute/src/cpu/ops/moe/latent_mask.rs` (opt-in,
default off = byte-identical), wired into `cpu_moe_forward` and
`ExpertWeightFfn::moe_block`.
**Instruments:** `scripts/moe_latent_reference_olmoe.py` (HF reference gate),
`scripts/moe_latent_block_partition.py` (per-layer block partitioning).
**Artifacts:** `bench/aim-validation/moe-latent/step2_block_partition.log`
**Model:** OLMoE-1B-7B-0125-Instruct, 16 MoE layers, hidden 2048, 64 experts,
top-8. Corpus Frankenstein.

## Question

[`walk-ffn-r4-zeroout.md`](walk-ffn-r4-zeroout.md) refuted sparsifying the
*nonlinear population* — FFN rows, the `m` axis. The other axis of the
factorisation is the **shared interface feeding** that population.

Every selected expert reads the same input vector, and expert `gate_up` is laid
out `[2·inter, hidden]`. Masking hidden channel `j` therefore removes column `j`
from **every active expert's gate AND up matrix** — one decision amortised
across `2·K` large projections, where a row decision buys one row in one expert.
It also sidesteps the cost problem that motivated ANN: the input is already
computed by the architecture, so scoring its channels needs no second dense
projection.

**Faithfulness:** the mask is applied *after* routing, so the trained router
still selects the same experts with the same weights. All `K` experts and all
`m` neurons per expert are retained. Masking a channel is exactly equivalent to
not reading that weight column, so zeroing is the faithful semantics — the
ablation *is* the lever.

## Ceiling, computed before measuring (R4 discipline)

Against the DEC-8.6 ledger (dense 29.86 / routed 25.83 GB/token, routed = 46.4%
of the per-token read), input-side masking scales the gate+up two-thirds of
expert bytes and leaves `down` fixed:

| retention | routed GB/tok | end-to-end |
|---|---|---|
| 0.75 | 21.52 | 1.084× |
| 0.50 | 17.22 | 1.183× |
| 0.00 | 8.61 | 1.448× ← hard cap |

Neither this nor the both-sides variant alone carries 5.8 → 10 tok/s.

## Result 1 — the representational signal is real

Reference forward (HF transformers, bf16), 512 tokens:

| retention | bits/token | rel | vs 0.5% shannon gate |
|---|---|---|---|
| 1.0 (base) | 2.420 | — | — |
| 0.875 | 2.431 | **+0.48%** | PASS → ceiling **1.040×** |
| 0.750 | 2.452 | +1.32% | 3× over |
| 0.625 | 2.463 | +1.78% | 4× over |
| 0.500 | 2.653 | **+9.66%** | **19× over** → 1.183× |
| random 0.500 | 15.843 | +555% | — |

**Magnitude vs random separates by 9.7% against 555%.** Channel importance on
the shared expert input is real and heavily concentrated, and `|z|` finds it for
free. The premise that would justify a new weight format survives.

**But the headline was an artifact.** larql's own (known-divergent) OLMoE
forward reported r=0.5 as *free* (−0.013 bits/token); the reference says
**+0.234 (+9.66%)** — an order of magnitude understated, with a sharp knee
between 0.625 and 0.5 that larql's forward sat on the wrong side of. Running the
reference gate before kernel work is what caught this.

**Consequence — the same non-overlap shape as R4, milder.** At the project's own
≤0.5% bits gate the viable retention is r ≈ 0.875, ceiling **1.040×**, and that
is the *ideal*: even a perfect kernel gives g = 0.875 ⇒ ~1.08× full-expert ⇒
1.04× end-to-end, before any capture loss (R4's kernels captured 23–40%).

**Both-sides masking is dead as specified:** the same mask on the output costs
+1.755 bits at r=0.5 vs +0.035 input-side. Input and output channel importance
are different sets, so the "one coherent latent subspace" assumption fails.

## Result 2 — static channel sets refuted (per layer, correct denominator)

Per-channel survival must be computed **per layer** — channel `j` in layer 4 and
layer 15 are unrelated coordinates, so a channel-indexed histogram summed across
layers is uniform by construction. And the denominator must be the **token
count**, recoverable exactly at fixed retention via `sum(v) = tokens × keep`.
Both were wrong in first passes; the invariants `sum(v) == tokens·keep`,
`max(v) ≤ tokens`, `mean(f) == keep/n` catch them for free.

Corrected: median survival 0.467, **96.7% of layer-channel pairs in the 20–80%
band**, 3 pairs below 0.01. Static core sizes per layer of 2048:

| q | C_q (min–max across layers) |
|---|---|
| 0.99 | 1–5 |
| 0.90 | 6–33 |
| 0.80 | 27–120 |

Pooled, a static core at P ≥ 0.8 covers only **6.6% of the 50% budget**. The
static-core-plus-dynamic-fringe hybrid exists but is negligible.

## Result 3 — blocked latent sparsity refuted (per layer, vs random control)

An arbitrary per-channel mask is realisable only in an **input-major** layout,
where each channel is a contiguous `2m` slab. In an **output-major** layout it
strides within rows and saves nothing, so blocks are required — and channel
order is free, since `z' = Pz`, `W' = W·Pᵀ` leaves the model exactly unchanged.
So "not spatially clustered" in the checkpoint's arbitrary order proves nothing.

Per-layer partitions were therefore learned against the **exact induced loss**
`L_t = Σ_kept(16 − s_{t,g}) + Σ_dropped s_{t,g}`, which prices within-block
disagreement *and* inter-block competition under the fixed budget (pairwise
disagreement `D_ij` only seeds it). Frozen on calibration, evaluated on a
disjoint contiguous held-out span.

Held-out bits/token vs dense 3.823:

| r | block=1 | native | D-clust | refined | random ×3 (sd) |
|---|---|---|---|---|---|
| 0.875 | **−0.18%** | +8.71% | +5.24% | +4.65% | +5.39% (.026) |
| 0.750 | +0.59% | +23.11% | +17.52% | +17.87% | +18.02% (.038) |
| 0.625 | +2.72% | +43.15% | +39.22% | +40.65% | +40.91% (.113) |
| 0.500 | +7.94% | +71.20% | +78.47% | +76.86% | +71.25% (.080) |

**Learned partitions are indistinguishable from random permutations at every
retention, and significantly worse at r=0.5 (~2.7 sd).** The random control is
load-bearing: without it, "D-cluster cuts +8.71% → +5.24%" reads as a 40% win,
when it is entirely explained by *any* permutation beating the native order —
which is itself worse than random.

**Mechanism, from held-out `L_t`.** At r=0.5 calibration `L_t` improves
824.0 → 714.0 (13.3%) but held-out only 824.2 → 811.1 (**1.6%**): the partition
overfit the calibration masks. And the 1.6% that does generalise buys nothing —
clustered partitions have *lower* held-out `L_t` than native yet *worse*
bits/token, so Hamming agreement does not track the unequal functional cost of
channel errors.

**No fragile-layer escape hatch:** per-layer held-out `L_t` is uniform (r=0.5:
mean 811.1, worst layer 818.4, +0.9%). Blocking fails uniformly across all 16
layers, closing the "channel-granular for a few fragile layers" hybrid too.

## Status of the four routes

| route | status |
|---|---|
| Dynamic row/output sparsity | refuted (R4) |
| Static latent-channel sets | refuted (per-layer, correct denominator) |
| Blocked latent sparsity | refuted (per-layer, exact-loss, vs random) |
| Dynamic per-channel, input-major packed kernel | **alive, untested** |

## Caveats

**OLMoE ≠ K3.** Pre-registered before running: OLMoE routes experts on the raw
residual, K3 on a *trained* rank-reduced latent, which should hold **less**
redundancy. A positive here was only ever an optimistic proxy; the block and
static negatives are the more transferable halves.

**Still over-constrained.** The probe applies one **shared** mask, so every
active expert gets the same channels — but experts share input *values*, not
*sensitivity*. The faithful K3 unit is the expert×channel slab `(i,j)` scored
`|p_i z_j|·√(‖W_g[i,:,j]‖² + ‖W_u[i,:,j]‖²)`; column norms are static, so
scoring costs ~K·ℓ ≈ 57k scalar ops at K3 dims. Separate masks do not break the
input-major kernel thesis — each expert owns its matrix anyway, so slabs stay
contiguous. The open objective is **minimise total expert-channel slabs read
subject to fidelity**, allocated across layers × tokens × experts × channels;
this probe fixed three of those four and varied only channel rank.

**No kernel, no timing.** Every speedup here is a byte-ledger projection. Per
standing rule **R11** ([`dec-funnel.md`](../dec-funnel.md) §"Standing rules") —
a structural reduction is a claim about a kernel, not about a matrix — the
1.04× must be priced on the kernel that would run it before it is believed.
