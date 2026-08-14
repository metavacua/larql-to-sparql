# R4 zero-out — is sparse FFN execution cheaper than dense when routing is free?

**Status:** RESOLVED 2026-08-01 — **all-layer dynamic row sparsity REFUTED.**
**Rule applied:** R4 (`docs/dec-funnel.md` §"Standing rules") — before testing a
lever, set the bytes it targets to zero.
**Harness:** `chris-experiments/larql_probes/examples/walk_ffn/walk_ffn_r4_zeroout.rs`
**Artifacts:** `bench/aim-validation/r4_zeroout_ac_paired.txt`, `..._v2.txt`
**Model:** Gemma 3 4B (`output/gemma3-4b-q4k-v2.vindex`), 34 layers,
10 240 features/layer, decode shape (seq_len = 1).

## Question

The 2026-05-29 scissors measured gate-KNN sparse FFN at ~3× *slower* than dense
at every K, because selection pays a full O(N·d) gate projection. Every
candidate router — HNSW/ANN, LSH, static pools — is an attempt to shrink that
selection term. R4 asks the prior question:

> With perfect routes supplied for **free**, is the remaining sparse execution
> cheaper than dense execution?

If not, no router can rescue it, and the ANN thread closes without repairing a
single graph defect.

## Two levers, separated

A known route does two things in this codebase, and collapsing them yields an
uninterpretable number:

1. **Selection disappears** — the O(N·d) gate sweep becomes O(K·d).
2. **A different kernel unlocks** — `sparse.rs:158` fires the contiguous-gather
   kernel (`sparse:gather_q4k`) *only* when the route is known in advance AND
   unranked. Production, which must search, can never reach it.

`rank_within_pool = true` on a K-sized pool keeps the production
`parallel_q4k_down` kernel with a known route, isolating lever 1 from lever 2.
A first version of this harness missed that and compared arms running
**different kernels**; the executed-kernel names in the runtime trace caught it.

## Arms

| Arm | Pool | Rank | Kernel | Selection cost |
|---|---|---|---|---|
| `dense` | — | — | dense FFN | none |
| `exact-pool` | all N | yes | `parallel_q4k_down` | N row-dots |
| `oracle-par` | top-K | yes | `parallel_q4k_down` | K row-dots |
| `oracle-gath` | top-K | no | `gather_q4k` | K row-dots |

`oracle-par` vs `dense` is the decision. The oracle arms still pay the K gate
row-dots, correctly: the sparse kernel consumes `gate_score` as its activation
input and any real router emits candidate IDs, not exact scores — that is
execution work every arm must do. Only the search over N disappears.

**Guardrail:** routes are captured from `exact-pool`'s own runtime trace in
executed visit order and replayed in that order, so the oracle arms issue the
identical scattered row reads a real stage-1 router would produce. The search is
zeroed; the gather disorder is not.

## Method

Paired and interleaved. All four arms run back-to-back inside each of 16
repeats in a **rotated order**; the statistic is the distribution of per-repeat
paired ratios (`dense/arm` within the same repeat), so thermal drift cancels
rather than being averaged. Dense wall-time per repeat is a throttle sentinel,
gated on **both** head-to-tail drift and IQR/median — a drift-only gate passed a
cell whose dense arm spiked 4× on one repeat. The harness refuses to run off AC
power or at 1-min load ≥ 3.0.

## Result (AC, load 2.32, all cells inside the sentinel; replicated twice)

`oracle-par` parity ✓ in every cell — same feature set, residual within 1e-4
relative, same token. The zero-out changed timing only.

| band | K | paired ratio vs dense | row-count ceiling | % of ceiling | top-1 |
|---|---|---|---|---|---|
| all | 2048 | **1.160** [1.157–1.172] | 5.00× | 23% | **✗** |
| all | 4096 | **0.837** [0.832–0.842] | 2.50× | 34% | ✓ |
| all | 6144 | **0.670** [0.661–0.683] | 1.67× | 40% | ✓ |
| last-4 | 2048 | 1.025 [1.013–1.039] | — | — | ✓ |
| last-4 | 4096 | 0.992 [0.982–1.034] | — | — | ✓ |
| last-9 | 2048 | 1.083 [1.053–1.096] | — | — | ✓ |
| last-9 | 4096 | 0.987 [0.963–1.006] | — | — | ✓ |

## Verdict

**All-layer row sparsity is refuted at accuracy-viable density.** All three
pre-registered closure conditions hold: K=4096 is *slower*, not break-even;
K=2048 is the only clear speed win; K=2048 fails top-1 against dense. There is
**no overlap between the speed-viable and accuracy-viable regions.**

**The mechanism is the kernel, not the router.** With routing free, the sparse
path captures only **23–40% of its row-count reduction**. At K=4096 it drops
60% of rows and still loses, i.e. it is >2.5× less efficient per row than dense.
The capture fraction *rises* as fewer rows are dropped (23 → 34 → 40%), which is
the signature of fixed per-row overhead in the kernel rather than anything about
*which* rows were selected.

**Precise claim.** Selection was **not the binding limiter at accuracy-viable
density** — it plainly had a large cost (~3 ms/layer), it just wasn't sufficient
to remove. The maximum recoverable routing benefit does not bridge the
sparse-kernel deficit. Merely tying dense at K=4096 needs ~1/0.843 ≈ **18.6%
execution improvement with routing already free**, which is why tuning one
gather implementation cannot close it; only a different execution model could.

**Scope.** This refutes a *compute*-side claim (rows resident, select a subset,
do arithmetic). It does **not** close DEC-8.1, a *read*-side claim about bytes
never fetched, which has a different denominator and a far milder retention
target. What R4 kills there is the *mechanism*: runtime scattered gather cannot
be how a read-side saving is realised, because the gather costs more than it
saves once the rows are in hand.

**Narrow bands** are a tradeoff, not a free win: the effect appears only at
K=2048 (20% retention), exactly where the 2026-05-29 scissors put the last-9
band at KL 0.46. Best cell is 1.083× with a real accuracy cost, measured against
a one-token top-1 check that is not a fidelity gate.

**Secondary findings.** The contiguous-gather kernel is *slower* than the
parallel kernel in 6 of 7 cells, including the faithful K its docstring claims to
fix; and it diverges numerically from the parallel path by 1.5–5.1e-3 relative.
These are **two separate issues** — the arithmetic contract must be named before
anyone optimises the performance, or an optimisation could bless the wrong
reference path.

## Caveats

The comparison is *within* the WalkFfn backend family — `dense` is WalkFfn's
dense mode, not production's FFN kernel, and `predict_with_ffn` is not the
production decode path (8.4 tok/s here vs C10's 30.9).

**The bias direction is cell-dependent, not uniform.** Large fixed per-token
overhead common to both arms drags every ratio toward 1.0. For losing cells the
true FFN-term ratio is *worse* than measured, so closure strengthens. For
winning cells the true FFN-term gain is *larger* than measured, opposed by the
dense-mode-is-slow effect — net sign unresolved. If narrow bands are ever
reopened, report the FFN-term ratio directly, not whole-forward.

## Consequence

The ANN/HNSW thread closes **as specified**: a graph that approximates
gate-top-K more cheaply addresses only selection cost, and exact gate-top-K
already sits on the wrong quality/performance frontier. A useful ANN would have
to search a representation correlated with *contribution*, not gate similarity.

Note the naming: `oracle-par` is an oracle for the **cost** of the current
selection policy, not for **which rows should be selected**. Fitting the three
all-layer cells gives `T_sparse/T_dense ≈ 0.547 + 0.000154·K`, crossing 1.0 at
**K\* ≈ 2944**. So a better *quality* selector preserving fidelity at K ≲ 2900
would still be interesting — that is ~1.4× better utility per selected row, not
an order of magnitude. What is refuted is that making the *current* selector
free is sufficient.
