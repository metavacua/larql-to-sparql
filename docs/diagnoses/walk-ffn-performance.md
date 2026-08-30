# WalkFfn performance analysis (2026-05-29)

`WalkFfn` is the "model-is-the-database" sparse-FFN inference path — gate-KNN
top-K feature selection → decode only the selected Q4K rows → down. The thesis
is "touch fewer weights per token." This note measures whether that speedup
actually materialises, and where `WalkFfn`'s time goes.

## Routing (two regimes)

`WalkFfn::forward` (routing ladder, `walk_ffn/mod.rs`):
- **Sparse** (`config.is_sparse(layer)`, i.e. `--top-k K`): `walk_ffn_sparse`
  — gate-KNN top-K, decode K rows, down. The intended "touch fewer weights" path.
- **Dense** (`k = MAX`, default `new_unlimited`): falls to route 4
  `walk_ffn_kquant_native` → `kquant_matmul_transb` (full gate/up/down).

## The dense kernel

`kquant_matmul_transb` (CPU, `larql-vindex/index/compute/kquant_dispatch.rs`):
**rayon-over-W-rows + NEON per-row Q4K×Q8 dot** — a *matvec* shape, not a
blocked BLAS `sgemm`. The `backend` hook is unwired (`let _ = backend`; CPU only).

- Competitive for **decode** (seq_len=1 → genuine matvec, bandwidth-bound, Q4K's
  4× size advantage helps).
- **Loses badly for prefill** (seq_len>1 → matmul; a row-parallel matvec without
  gemm cache-blocking is ~12× slower than BLAS `sgemm`, measured: MoE dense `h1`
  331 ms f32-BLAS → 3986 ms WalkFfn-native over prefill+12 decode).

## Sparsity buys nothing end-to-end (measured)

`larql dev walk --predict` on `gemma3-4b-q4k` (10240 features/layer), 9 tokens:

| K | wall (9 tok) |
|---|---:|
| dense (MAX) | 28.84 s |
| top-k 2048 (20%) | 29.28 s |
| top-k 512 (5%) | 28.64 s |

**No speedup from sparsity.** Root cause: `walk --predict` runs at **~0.5 tok/s
— the non-KV-cached full-recompute path** (re-forwards the whole growing
sequence every token). In that regime attention + lm_head over O(N) positions
dominate, and the FFN — sparse or dense — is a masked fraction. The
"touch-fewer-weights" win has nothing to bite on.

## Conclusions

1. **The sparse-FFN speedup is not observable in the only end-to-end WalkFfn
   harness** (`walk --predict`), because that path is full-recompute (0.5 tok/s).
   The benefit, if real, is invisible behind attention + lm_head re-compute.
2. **The dense Q4K kernel is matvec-shaped, not a gemm** — it cannot compete with
   BLAS `sgemm` for prefill, and the `ComputeBackend` hook is unwired.
3. To actually validate "touch fewer weights," two things are needed:
   - an **FFN microbench** (isolate `WalkFfn.forward` at varying K, decode shape
     seq_len=1) — the only way to see the sparse row-decode win without the
     full-recompute mask;
   - a **KV-cached WalkFfn decode path** so the FFN is a real fraction of wall
     time end-to-end (today the cached path is the `StandardEngine` Q4K route,
     which doesn't expose K).
4. For dense use (e.g. a hybrid-MoE dense `h1`), `WalkFfn` is the wrong tool — use
   BLAS f32 or wire `kquant_matmul_transb` to a blocked gemm / the backend.

## FFN microbench — the missing instrument (built 2026-05-29)

`chris-experiments/larql_probes/examples/walk_ffn/walk_ffn_microbench.rs` isolates `WalkFfn::forward` at **seq_len = 1**
(decode shape), no attention/lm_head, across K. gemma3-4b-q4k, layer 17, 10240
features, 300 iters:

| route | µs/call | vs dense |
|---|---:|---:|
| **dense** (k=MAX, `kquant_native` contiguous matvec) | **2021** | 1.0× |
| sparse k=2048 (20%) | 7647 | **3.8× slower** |
| sparse k=512 (5%) | 6511 | 3.2× slower |
| sparse k=128 (1%) | 6092 | 3.0× slower |
| sparse k=32 (0.3%) | 6034 | 3.0× slower |

**The sparse walk is ~3× slower than dense at *every* K, and barely improves as K
shrinks** (7647 → 6034 from 20% → 0.3%). The cost is **fixed, not
K-proportional** — so the "touch fewer weights" lever (small K) doesn't move it.

**Why:** to choose the top-K features, gate-KNN **scores all 10240 features (a
full gate projection)** — that ranking cost alone already exceeds the entire
dense Q4K matvec (which has no selection, contiguous access, NEON+rayon). The
only thing sparsity saves is the down-projection rows (K vs all), a small slice
swamped by (a) the full-gate ranking and (b) cache-unfriendly sparse gather. So
**at inference, the retrieval costs more than the dense compute it avoids.**

## Cheap routing flips it — the thesis holds (built 2026-05-29, task #18)

The fix flagged above was implemented and measured. A **precomputed route**
(`WalkFfnConfig::with_pool_per_layer(..).with_precomputed_routing(true)`) takes
the K features from a fixed per-layer pool — modelling hash routing (Exp 27,
token-deterministic) — and computes the gate score for **only those K features**
(`local_pool_gate_knn`, O(K) Q4K row-dots), *never* doing the full gate
projection. Same harness, same K sweep, head-to-head with gate-KNN:

| route | k=2048 (20%) | k=512 (5%) | k=128 (1%) | k=32 (0.3%) |
|---|---:|---:|---:|---:|
| dense (k=MAX) | **2039 µs** (1.0×) | — | — | — |
| gate-KNN (full projection) | 8488 µs | 7748 µs | 6919 µs | 6294 µs |
| **cheap-route (O(K) selection)** | **1557 µs** | **517 µs** | **245 µs** | **138 µs** |
| cheap-route vs dense | **1.3× faster** | **3.9×** | **8.3×** | **14.8×** |

Two things change at once:
1. **Sparse beats dense at every K** (1.3× → 14.8× as K shrinks).
2. **The cost is K-proportional again** — 1557 → 138 µs (an 11× drop for 64×
   fewer features). Gate-KNN was *flat* (8488 → 6294) because the full gate
   projection swamped everything; cheap routing removes that floor, so the
   "touch fewer weights" lever finally bites.

## Bottom line (the thesis holds — gate-KNN routing was the saboteur)

For **gemma3-4b at seq_len=1 on CPU**, sparse FFN retrieval **is faster than
dense — up to ~15× at K=32 — *provided routing is cheap*.** The earlier "3×
slower, K-independent" result was **not a failure of the model-is-the-database
thesis**; it was the cost of the *router*. Gate-KNN selection ranks all
features (a full gate projection over `num_features`) that alone exceeds the
entire dense matvec, and it's K-independent, so sparsity has nothing to bite on.
Replace it with an O(K) precomputed/hash route and "touch fewer weights = faster"
is restored, cleanly K-proportional.

**Implication for the inference path:** the win is gated on the *routing
mechanism*, not the sparsity. Production WalkFfn must route via a near-free
selector (hash routing / precomputed per-token pools, Exp 27) — **not** gate-KNN
— for the sparse path to pay off. But speed is only half the story; the price in
predictive quality is measured next.

## Accuracy frontier — speed is cheap, accuracy is not (built 2026-05-29, task #19)

`chris-experiments/larql_probes/examples/walk_ffn/walk_ffn_accuracy.rs` runs a full forward (attention dequantised to f32
up front via `insert_q4k_layer_tensors`, so the **FFN router is the only
variable**) and scores the last-token next-token distribution against dense in
the Shannon discipline — **KL in bits, top-1 agreement, q@p_argmax** — never
cosine. Three FFN routers: dense (ground truth), gate-KNN (content-addressed,
slow), cheap-route (precomputed strided pool, the O(K) router from task #18).

**Path is faithful (control).** Full-K parity vs the dense native path: forced
sparse-walk KL ≤ 0.006 bits, cheap-route-over-all-features KL = **0.0000** — so
`local_pool_gate_knn` computes the true gate and the walk reproduces dense. Any
divergence below is real, not a kernel bug.

**Sparsity at *every* layer collapses the model** (gemma3-4b, 34 layers, 4
prompts):

| router | k=2048 (20%) | k=512 (5%) | k=128 (1%) | k=32 (0.3%) |
|---|---:|---:|---:|---:|
| gate-KNN (smart) | 8.4 bits / 0% | 14.1 / 0% | 24.2 / 0% | 27.2 / 0% |
| cheap-route (dumb) | 37.1 / 0% | 37.5 / 0% | 33.0 / 0% | 35.6 / 0% |

Even the *smart* router at 20%-of-features-per-layer is unusable (0% top-1). This
is the stacked-ablation collapse signature: per-layer error that's
individually tiny (Exp 27: single-layer L0 top-2048 KL ≈ 0.03) **compounds
multiplicatively across depth** and breaks `final_norm` + `lm_head`.

**Confined to a thin late band it survives** — the hourglass (dense early, sparse
late), K=512:

| sparse depth | gate-KNN | cheap-route |
|---|---:|---:|
| last 4/34 | KL 0.71 / 50% | KL 1.75 / 75% |
| last 9/34 | **KL 0.46 / 75%** | KL 10.1 / 25% |
| last 17/34 | KL 5.3 / 25% | KL 15.8 / 0% |

Two results (4 prompts → top-1% is coarse, KL is the reliable signal):
1. **Sparsity is only survivable in a thin late band.** Even gate-KNN holds at
   the last ~9/34 layers (KL 0.46) but breaks by 17 (KL 5.3). Whole-model sparse
   WalkFfn is off the table; a late-layer sparse band is the only viable shape.
2. **The cheap (content-blind) route degrades far faster with depth than
   gate-KNN** — fine at 4 layers (KL 1.75) but collapsed by 9 (KL 10.1) where
   gate-KNN still holds KL 0.46. Routing *quality* buys depth.

## Bottom line — the speed/accuracy scissors

- **Speed (task #18):** cheap routing beats dense up to ~15×; gate-KNN is ~3×
  *slower* than dense. Routing cost dominates.
- **Accuracy (task #19):** gate-KNN tolerates a ~9-layer sparse band; the cheap
  strided route only ~4. Routing *quality* dominates.

These pull in opposite directions: the router that's fast is too dumb to go deep,
and the router that goes deep is too slow to win. The resolution is a router that
is both O(K)-cheap *and* good — measured next.

## Static-importance routing closes the gap for a shallow band (task #20)

A **static-importance pool** (`static_importance_pool` in the accuracy example:
top-K features per layer by ‖down_row‖ — the features that move the residual
most *when active*) is content-blind but **informed**, and exactly as cheap as
the strided route (precomputed once, no gate projection). Added as a third
hourglass router, K=512:

| sparse band | gate-KNN (smart, slow) | strided (dumb, fast) | **static-imp (informed, fast)** |
|---|---:|---:|---:|
| last 4/34 | KL 0.71 / 50% | 1.75 / 75% | **0.71 / 100% / q@p 0.86** |
| last 9/34 | KL 0.46 / 75% | 10.1 / 25% | 2.76 / 50% |
| last 17/34 | 5.3 / 25% | 15.8 / 0% | 21.5 / 0% |

(4 prompts → top-1% is coarse; KL is the reliable signal.)

- **Shallow band (last 4/34): static-importance *ties* gate-KNN on KL (0.7086 vs
  0.7098) and beats it on top-1/q@p.** A cheap, content-blind, O(K) route reaches
  the content-addressed router's accuracy — at cheap-route speed. **For a thin
  late band, the cheap+good router already exists.**
- **Medium band (last 9/34): content-addressing starts to pay.** static-imp
  (KL 2.76) crushes strided (10.1) but does not reach gate-KNN (0.46). Beyond ~4
  layers, *which* features fire is input-dependent and a static pool leaves
  accuracy on the table.
- **Deep band (last 17/34): all size-K routing collapses** — sparsity simply
  isn't viable that deep at K=512.

## Bottom line (resolved for the shallow band; open for deeper)

The speed/accuracy scissors closes — partially:
- **A thin late sparse band (~4/34 layers) is a solved win:** static-importance
  routing is O(K)-cheap (so sparse beats dense, task #18) *and* matches gate-KNN
  accuracy (KL ≈ 0.71). Ship-able today via `with_precomputed_routing` + a
  ‖down_row‖ pool. The catch: a 4-layer band is a small slice of the model, so
  the end-to-end FFN-time win is modest.
- **A deeper band (~9/34) still needs cheap *content-addressed* routing** —
  static-imp's 2.76 vs gate-KNN's 0.46 is the gap input-blindness can't close.
  The real candidates (per-position, which `forward(layer, x)` doesn't yet
  expose): **cell-conditional pools** (LSH/cluster the residual offline → O(1)
  cell lookup → tiny within-pool rank; the `pool_per_layer` design note) and
  **hash routing** (Exp 27 — but token-determinism fades by L3, mismatched with
  the *late* band, so it likely needs a residual key, not a token key). Building
  per-position routing is the gating infra for both.

So WalkFfn sparse is real but bounded: **cheap routing + a shallow late band is
deployable now; widening the band is an open routing-research problem, not a
kernel one.** Secondary levers unchanged: larger models (down a bigger share),
GPU, blocked-gemm dense baseline.

## Two-stage routing: ranking within a static pool can't widen the band (task #21)

The cheapest content-addressed router needs no offline clustering: take an
informed **static candidate pool** (top-P by ‖down_row‖, P ≫ K) and rank it
**per-position by the real gate score** down to K (`rank_within_pool` —
`local_pool_gate_knn` over the pool is O(P), still no full projection; the
residual is already in the per-position loop, so the ranking *is* per-position
content-addressing). If gate-KNN's true top-K live inside the static top-P, this
recovers gate-KNN accuracy at O(P) ≪ O(num_features). Sweep P at K=512:

| sparse band | gate-KNN | static-imp | 2-stage P=2048 | P=4096 | P=8192 |
|---|---:|---:|---:|---:|---:|
| last 4/34 | 0.71 | 0.71 | 0.80 | 0.92 | 5.68 |
| last 9/34 | **0.46** | 2.76 | 2.74 | **2.56** | 3.32 |
| last 17/34 | 5.3 | 21.5 | 21.3 | 18.4 | 22.8 |

(KL bits; 4 prompts — KL is the reliable signal.)

**It does not work — and the argument is the gap, nothing else.** Every
static-pool variant, at every P, sits in the **2–3 band at 9 layers while
gate-KNN alone drops to 0.46** (~4×). Ranking within the static pool ≈ the static
pool itself (2.74 vs 2.76). The natural reading is that the features gate-KNN
picks for a given input are *not in* the static top-P, so ranking can't conjure
them — but the P-sweep alone can't prove that: a competing reading is that the
cheap Q4K within-pool score is too noisy to rank usefully, collapsing toward
no-ranking regardless of the pool. The de-confound below separates them.

(An earlier draft leaned on the P-non-monotonicity — "if the score were
gate-KNN's, KL couldn't *rise* at high P, so the scores differ." **Retracted:**
KL is measured at the model output, several nonlinear layers downstream of this
FFN. The endpoint pin is real — at P = num_features the f32 two-stage *is*
gate-KNN exactly — but a strictly-better FFN output-vector approximation does
**not** imply monotone downstream KL through the remaining stack, so
KL-non-monotonicity in P was never a signal about score-source, in either
column. The gap, not the wobble, is the argument.)

## De-confound: full-precision ranking on the static pool (the candidate set IS the wall)

One config flip separates (a) from (b): rank the *same* static pool by
gate-KNN's **own full-precision f32 gate score** — exactly what
`pool_restricted_gate_knn` does (`gate_scores_batch_backend` = f32 gemv, filtered
to the pool), i.e. `precomputed_routing = false`. This differs from the Q4K
two-stage only in score precision, and from the gate-KNN baseline only in the
pool restriction. Plateau ≈ static-imp → (a); drop toward 0.46 → (b), and a
static pool + dequantized score would widen the band with no clustering at all.

9-layer band, K=512 (gate-KNN baseline 0.46):

| P | 2-stage (Q4K score) | de-confound (f32 score) |
|---|---:|---:|
| 2048 | 2.74 | 3.44 |
| 4096 | 2.56 | **2.01** |
| 8192 | 3.32 | 2.75 |

**Reading (a) is confirmed: the candidate set is the wall.** With gate-KNN's
*exact* full-precision score, ranking the static top-P-by-‖down_row‖ pool still
plateaus at ~2.0 (best, P=4096) and stays ~4× off gate-KNN's 0.46. The whole
static-pool family — both precisions, every P — lives in the 2–3 band; gate-KNN
alone reaches 0.46. So the features gate-KNN selects for these inputs are
genuinely *not in* the static pool, and a better ranking score can't put them
there. The score-precision effect (b) is **inconclusive at n=4** — f32 helps at
P=4096 (2.01 < 2.56) but hurts at P=2048 (3.44 > 2.74) — and, more to the point,
**immaterial to the decision**: every static variant is ~4× off regardless.

**Decision this settles:** there is **no cheaper-than-#22 shortcut.** A static
pool + a better (dequantized) ranking score does *not* widen the band — so the
content-addressed *candidate set* (task #22, residual-keyed IVF/LSH cells:
cluster residuals offline, each cell → a small precomputed pool, O(1) cell lookup
at inference) is genuinely required, not a confounded premise. Note #22 sidesteps
the score-noise issue by construction — if its cell pools are built as gate-KNN
unions sized ~K and used directly (no within-cell ranking), it needs no cheap
gate score at all. Infra already landed: `WalkFfnConfig::rank_within_pool` + the
per-position pool path compute the route from the in-loop residual, so a
residual-cell router drops into the same hook.

### n≈30 confirmation — gap holds, but the prize is smaller than n=4 implied

The n=4 hourglass numbers above are a small sample (top-1 bounces 25/50/75%).
Re-running just the load-bearing comparison — gate-KNN vs the best static pool
(f32-ranked, P=4096) at the 9-layer band — over **30 prompts**:

| config (sparse last 9/34) | mean | median | min | max |
|---|---:|---:|---:|---:|
| gate-KNN (content-addressed) | 2.43 | 1.30 | 0.008 | 14.53 |
| static pool f32 P=4096 | 4.85 | 3.74 | 0.026 | 15.68 |

Mean gap **1.99×**, median gap **2.86×**; gate-KNN beats static on **21/30**
prompts. Two corrections to the n=4 story:
- **The gap is real, not a mirage.** Content-addressing wins ~2–3× and on
  21/30 prompts. **#22's premise (a content-addressed candidate set beats a
  static one) holds** at n=30 — resourcing it is justified.
- **But n=4 oversold both sides.** Gate-KNN's 9-layer accuracy was "0.46" on the
  lucky 4-prompt draw; at n=30 it is **median 1.30 / mean 2.43 bits** (max 14.5 —
  some prompts collapse even for gate-KNN). So the *ceiling* #22 chases at 9
  layers is itself only mediocre, and widening to 9 layers is a **real but modest
  win**, not the dramatic one 0.46 implied. The cleanly-good, low-variance band
  remains the shallow #20 one (KL 0.71). Treat "9-layer band" claims elsewhere in
  this doc that cite KL 0.46 as the optimistic n=4 figure; median 1.30 (n=30) is
  the honest gate-KNN 9-layer number.

(De-confound + the n-bump check proposed in peer review of the #21 writeup. The
review also caught that the round-1 P-non-monotonicity argument was unsound —
downstream KL needn't be monotone in a strictly-better FFN approximation — so
that argument is retired here; the gap carries the conclusion on its own.)

### #20 deployment constraint — keep P modest at the shallow band

The 4-layer band (the shippable #20 win) shows a sharp **large-P blow-up**:
KL 0.71 at small P but **5.68 (Q4K) / 8.40 (f32) at P=8192**. Over-provisioning
the candidate pool is *not* benign at the deployable band — it actively wrecks
accuracy. So the #20 deployment note is concrete: at the 4-layer band keep the
pool modest (≤ ~4096); large pools degrade it well below the dense baseline.

## Residual-cell router closes the gap — content-addressed AND cheap (task #22)

The static pool's wall is that its candidate set is input-blind. The fix is an
IVF-style **residual-cell index** (`CellRouter`): offline, run dense forward over
a calibration corpus, record each band-layer FFN-input residual + its gate-KNN
top-K; k-means the residuals into C cells per layer; each cell's pool = the
most-frequent gate-KNN features across its members (capped). At inference, the
per-position residual picks its nearest cell (O(C·hidden)) and that cell's pool
is the candidate set — **content-addressed** (cell depends on the residual) but
cheap (no full O(num_features) gate projection). Built and run in
`chris-experiments/larql_probes/examples/walk_ffn/walk_ffn_cell_router.rs` (C=64, calibration 24 prose prompts, 9-layer
band, K=512). **All KL is measured against *dense* (KL 0 = dense); gate-KNN is
itself a lossy top-K truncation of the gate projection, not a floor.** Lower =
closer to dense.

| router | in-dist median (n=30) | OOD median (n=18) |
|---|---:|---:|
| gate-KNN (full projection) | 1.30 | 1.23 |
| cell-router, full cell pool | 0.88 | 0.43 |
| cell-router, ranked → K=512 | 0.79 | 0.40 |
| static pool P=2048 | 6.09 | 1.38 |

Unpaired medians (the table) suggest cell-router beats gate-KNN — but they
mislead under a heavy tail, and an **absolute cross-distribution comparison is
confounded by set difficulty** (every method's median *fell* OOD except
gate-KNN's, so the OOD set is simply easier / more peaked — you can't read overfit
off absolute KL across two difficulties). The difficulty-controlled test is the
**paired Wilcoxon signed-rank** on per-prompt deltas (same prompt, same
difficulty), and disaggregated by sub-distribution because code / non-English /
prose behave nothing alike:

| comparison (Δ = a − b, neg ⇒ a closer to dense) | in-dist (n=30) | OOD code (6) | OOD intl (6) | OOD prose (6) | OOD agg (18) |
|---|---|---|---|---|---|
| cell-full vs gate-KNN | +0.07, p=0.64 | −0.65, p=0.094 | −0.24, p=0.29 | +0.04, p=1.0 | −0.29, p=0.074 |
| cell-rank→K vs gate-KNN | −0.02, p=0.58 | −0.49, p=0.036 | **+0.74**, p=0.83 | −0.01, p=1.0 | −0.04, p=0.21 |
| cell-full vs static | −3.09, **p=0.0001** | +0.00, p=0.68 | −3.80, p=0.036 | −0.90, p=0.036 | −1.05, **p=0.0023** |

> **Read the per-category OOD p-values as direction-of-effect, not significance.**
> At n=6 the exact two-sided Wilcoxon floor is p = 2/2⁶ = **0.031** (all six
> deltas same-sign), so "p=0.036" means "all six pointed one way at the smallest
> n that can say anything" — suggestive, not robust. Only the **in-dist (n=30)**
> p-values carry real weight; they are **bold**.

Controlled conclusions:
1. **cell-router matches gate-KNN — in *and* out of distribution.** Not
   significantly different in-dist (p=0.64) *or* OOD-aggregate (p=0.074). The
   "beats the ceiling" median (0.88 vs 1.30) was a heavy-tail artifact; per-prompt
   they tie. The honest accuracy claim is parity: **cell-router ≈ gate-KNN at
   ~150× cheaper routing.**
2. **The OOD-superiority claim does not survive disaggregation.** The aggregate
   edge (p=0.074) is direction-carried by **code**, a **wash on prose** (p=1.0),
   and mixed on non-English. An earlier *un*balanced 18-prompt OOD set gave
   p=0.026; rebalancing dropped it to 0.074 — composition-sensitive. So
   "cell-router beats gate-KNN on unfamiliar residuals" is a **code-specific,
   small-n hypothesis, not a result** (task #23).
3. **rank→K is *not* equivalent to the full pool OOD — the pool's breadth is the
   robustness.** In-distribution the two track (rank→K is the fair matched-K
   control there). But OOD they diverge: non-English cell-full median 0.43 →
   rank→K 1.98 (delta vs gate-KNN flips to **+0.74 — worse**); code 0.49 → 0.84.
   Ranking the cell pool down to K=512 throws away exactly the features that carry
   unfamiliar residuals — the freq-union admits breadth the per-token ranker
   discards. **Consequence: if rank→K is shipped as the cheap variant, it may shed
   the only edge cell-routing has.** matched-K is an in-dist control, not a free
   OOD cost reduction.
4. **Overfit-collapse is refuted — for the *full pool* specifically.** A router
   memorising calibration neighbourhoods would *collapse* on OOD residuals
   (KL ≫ gate-KNN); the full-pool cell-router ties-or-beats gate-KNN OOD and never
   blows up (on the paired deltas, not the difficulty-confounded absolute
   comparison). But the mechanism is narrower than "the cells generalise": rank→K
   — *same* centroids, *same* calibration — flips to worse-than-gate-KNN on
   non-English (#3). So what generalises is the **unranked breadth**, not the
   routing per se: the router holds OOD *provided you keep the whole pool*, which
   is closer to "don't prune" than "the cells transfer." rank→K is the cautionary
   counterexample. (Relatedly, gate-KNN's *median* doesn't improve on the easier
   OOD set while everyone else's does — a per-token-noise floor that doesn't shrink
   with task ease; its OOD *mean* 3.36 vs median 1.23 is a few blow-ups at n=18,
   not general degradation.)

**The robust, everywhere claim is cost + parity**, not any OOD edge: cell-router
**matches gate-KNN accuracy** (in- and out-of-distribution) at ~150× cheaper
routing, and **beats static — established in-distribution (p=0.0001, n=30, the
load-bearing number).** The cell≫static *OOD* result needs the same evenhanded
disaggregation as the adversarial comparison: it is **a wash on code** (Δ=0.00,
p=0.68), directional on prose/non-English, and its aggregate (p=0.0023) is partly
**static face-planting on non-English** (median 8.21 / mean 11.25 — catastrophic
on an otherwise *easy* set). So: cell≫static is rock-solid in-distribution and
directional OOD-except-code; the in-dist n=30 result carries it. Mean cell pool
**1097 feats (10.7% of 10240)**, route cost
O(C·hidden + |pool|) ≈ 64·2560 + 1097 — **no full gate projection** (~150× fewer
routing MACs than gate-KNN's O(num_features·hidden) gemv). Crucially this moves
routing **back below the FFN matmul** (gate-KNN's routing was ~7–10× its own
K=512 FFN work — routing-dominated, hence ~3× slower than dense; cell-router's is
~1/16 of it — FFN-dominated again), which is why end-to-end is *expected* to land
well — but that is still a projection until the KV-cache decode loop and Metal
path confirm it (task #23).

## End-to-end decode falsification — KL parity ≠ token faithfulness (task #23)

KL is a proxy; the pre-committed bar is **top-1 next-token agreement vs dense ≥
90%** (teacher-forced on dense's own greedy stream — what actually shows in
generations). At K=512 it **fails at every band**: 9-layer gate-KNN 60% /
cell-router 62.5% / rank→K 55%; 4-layer gate-KNN 75% / static-imp 57.5%. Crucial
read: **gate-KNN — the accuracy ceiling — fails too**, so a median 0.8–1.3 bit KL
flips ~25–40% of argmax picks *regardless of router*. The cell-router still does
its job (matches gate-KNN), but K=512 sparsity is not generation-faithful.

**K-vs-agreement sweep (gate-KNN, the ceiling):**

| | K=512 | K=1024 | K=2048 | K=4096 |
|---|---|---|---|---|
| in-dist, last 4 | 66.7 | 72.2 | 83.3 | **100.0** |
| in-dist, last 9 | 61.1 | 72.2 | 80.6 | **97.2** |
| OOD, last 4 | 72.2 | 75.0 | 75.0 | 86.1 |
| OOD, last 9 | 61.1 | 66.7 | 66.7 | 88.9 |

Faithfulness arrives at **K≈4096 (40% of 10240 feats)** in-distribution (OOD
~87–89%, near). The verdict is **not** "sparse FFN doesn't generate" — it's
that the faithful-K regime is **kernel-bound, not FLOP-bound**:
- K=4096 is 40% of dense's FLOPs → at dense's ~0.20µs/feat it *should* be ~800µs
  vs dense 2021µs (**2.5× faster**).
- But the measured cheap-route at K=2048 is **1557µs** (ideal ~400µs) — **~4×
  per-row overhead** from scattered gather + per-row Q4K dispatch. Extrapolated,
  K=4096 lands ~3000µs — slower than dense — *entirely* from that overhead.

**So the faithful-K speedup exists in the FLOPs and is squandered by the kernel.**
The optimization (task #24) — **gather the selected K rows contiguous, then run
the kernel** — was built and measured (`chris-experiments/larql_probes/examples/walk_ffn/walk_ffn_gather_gemm.rs`), and it
**flips faithful-K from slower-than-dense to faster** (isolated FFN, seq_len=1):

| K | scattered (current) | gather Q4K + fused | dense |
|---|---|---|---|
| 2048 (20%) | 1563µs (1.40×) | **1284µs (1.70×)** | 2181µs |
| 4096 (40%, faithful) | 2713µs (**0.80×, slower**) | **1600µs (1.36×)** | 2181µs |

The *implementation* was the whole story: gathering Q4K **bytes** contiguous and
reusing the fused NEON row-dot (no f32 materialization) gives 1.36× at K=4096; a
naive **f32-dequant→BLAS-gemv** gather was *0.12×* (16701µs) — Q4K→f32 dequant +
per-row alloc dominated. So the lesson is "gather bytes, keep the fused kernel,"
not "dequant and gemm."

Honest scope: the 1.4× is **isolated FFN**; end-to-end the band is 9/34 layers
(25 stay dense) + attention + lm_head, so by Amdahl the net decode speedup is a
fraction of it (plausibly single-digit %) — and OOD faithfulness may want K>4096.
Per-row honesty: the kernel doesn't make rows *cheaper*, it stops them being
catastrophically expensive — dense ≈ 0.21µs/row, scattered ≈ 0.66µs/row (3×),
gather ≈ 0.39µs/row (1.8×); gather wins only by touching 40% of the rows, not by
matching dense's per-row rate.

**⚠️ Wiring blocked — transposed down (found wiring it into `walk_ffn_sparse`).**
gate/up are feature-major Q4K (`[intermediate × hidden]`, row = feature →
gatherable), but **down is stored *transposed* `[hidden × intermediate]` Q6_K**,
so a feature's down vector is a strided *column*, not a gatherable row. The
gather microbench read the transposed down with row striding: **the timing is
representative (work magnitude is right) but the down *values* are wrong.**
Realising it correctly needs the **feature-major down sidecar**
(`down_features_kquant.bin`) — *absent* on current vindexes — exposed on
`GateIndex`. So the wiring is reverted (the method `gather_q4k_accumulate` is
retained, unwired, with the caveat); `walk_ffn_sparse` stays on the correct
scalar paths. **This does not affect the #22/#23 results** — those loaded native
f32 down (feature-major), so the guard declined and they used the correct path.

So the verdict is narrower than first stated: the gather kernel needs a
feature-major quantised down (gate/up are already feature-major). That was then
**validated in-memory** (build the feature-major Q4K down via
`kquant_ffn_layer(2)` → re-quantize, gather it correctly):

| K | scattered | gather (correct feature-major Q4K down) | down requant err |
|---|---|---|---|
| 2048 | 1.31× | **1.69×** | 6e-3 |
| 4096 (faithful) | 0.76× | **1.29×** | 6e-3 |

The win **survives correctness** — 1.29× at faithful K, `|err|/‖ref‖ = 6e-3`
(just the Q4K-vs-Q6K precision drop on the re-quantized down; a feature-major
*Q6_K* down would be zero-added-error, slightly bigger). The earlier 1.36×/1.4×
was the transposed-down (wrong values); the honest correct number is **1.29×
isolated FFN.**

### Built, wired, and validated end-to-end (task #25)

The productionization landed: (a) **writer** `examples/build_down_features_q4k.rs`
re-quantizes the feature-major f32 down (`kquant_ffn_layer(2)`) to Q4K and emits
`down_features_kquant.bin` (501 MB) + manifest; (b) `down_features_q4k_layer_data`
is now on the `QuantizedFfnAccess`/`GateIndex` trait; (c) `gather_q4k_accumulate`
gathers gate+up+down contiguous (down from the sidecar) and **recomputes gate** so
a known-pool route skips the scattered `local_pool_gate_knn`; (d) it's wired into
`walk_ffn_sparse` as a fast path for precomputed/cell routes (declines safely
without the sidecar — a test asserts this). Trace confirms `sparse:gather_q4k`
fires; the standalone correct kernel is **1.29–1.31× faster than dense** at
faithful K, err 6e-3.

**Honest caveat on the wired single-layer microbench (0.76× at K=4096):** that
number is *not* representative — `walk_ffn_sparse` issues
`madvise(MADV_WILLNEED)` on **layer+1**'s ~44 MB every position, which is the
right thing in real sequential decode (you use it next) but pure waste in a
single-layer bench hammering one layer (it readaheads an unused layer 200×,
competing for bandwidth). So the isolated bench can't fairly measure the wired
path. **The only fair test is end-to-end decode** — the remaining piece:
**measure net decode tok/s > dense as a pre-committed bar** (the 1.29× isolated
FFN dilutes by Amdahl across 25 dense layers + attention + lm_head to plausibly
single-digit % net, so the honest target is *net-positive*, not "1.29×
survives"), starting with the **4-layer static band**, top-1 agreement alongside.

### End-to-end decode measurement — the win does not survive the full forward

Pre-committed bar: net forward tok/s **> dense**. Measured three ways
(`chris-experiments/larql_probes/examples/walk_ffn/walk_ffn_decode_timing.rs`), each hitting a distinct confound — and
**none clears the bar**:

| measurement | result | confound |
|---|---|---|
| isolated FFN, seq_len=1 (`gather_q4k_gemm`) | **1.29× faster** ✓ | the clean kernel number |
| full forward, seq_len=9 | 0.82× / 0.72× ✗ | **prefill**: dense uses batched BLAS gemm over 9 positions; per-position gather loses |
| full forward, seq_len=1 (decode shape) | **0.15× / 0.11×** ✗✗ | full-forward gather is ~6× *slower* despite correct output — and it is **not** the kernel (isolated is 1.29×) and **not** memory pressure (137 GB RAM, 3.9 GB resident) |

The seq_len=1 collapse (the gather firing per trace, output top-1 ✓, yet ~550 ms
per sparse layer vs the isolated kernel's 1.6 ms) is an unresolved
full-forward-context cost — candidates: the per-layer `madvise(WILLNEED)` prefetch
churning alongside the sidecar's demand-paged mmap, rayon contention with
attention, or per-iteration `WalkFfn` rebuild. **It is infra, not the kernel**,
but it means the gather as wired through `predict_with_ffn` is far slower
end-to-end, not faster.

**Honest conclusion for the speed thesis.** The gather kernel wins in isolation
(1.29×, correct) but the win **does not translate to an end-to-end decode speedup
on the available infrastructure**: it loses at prefill (gemm), and the only
decode-shape path (`predict_with_ffn`, no KV cache, f32-materialized attention)
collapses to 0.15× for full-forward-context reasons that are real but not the
kernel. A *fair* number needs a purpose-built **KV-cached, Q4K-attention,
swappable-FFN decode loop** that does not exist — a substantial build whose
upside is Amdahl-bounded to single-digit % even in the best case. So the
end-to-end ship decision is **negative on current infra / not worth the build**
at the measured/bounded upside. The kernel, sidecar, and wiring stand as
validated components; the end-to-end payoff does not.

## The one speed lever that survives: uniform 3-bit FFN (not graded, not sparse)

Sparsity is closed (do-less flips tokens). The remaining doors are *fewer bytes
per feature* or *more work per byte*. Tested the most graph-native:
**graded precision** — keep all features, spend bits by ‖down_row‖ importance
(`chris-experiments/larql_probes/examples/walk_ffn/walk_ffn_graded_precision.rs`, block-wise quantiser validated: uniform
4-bit = KL 0.011 vs f32, matching real Q4K). KL vs f32 reference, 4 prompts
(in-dist + code + non-English):

| schedule | avg bits | bw vs Q4 | KL (bits) | top-1 |
|---|---:|---:|---:|---:|
| uniform 4-bit | 4.00 | 1.00× | 0.011 | 100% |
| **uniform 3-bit** | 3.00 | **0.75×** | **0.052** | **100%** |
| uniform 2-bit | 2.00 | 0.50× | 18.06 | 0% |
| head10/4 tail90/3 | 3.10 | 0.78× | 0.070 | 100% |
| head40/4 tail60/2 | 2.80 | 0.70× | 11.75 | 0% |

**Graded precision buys nothing — the precision floor is universal at 3 bits.**
2-bit collapses for *every* feature group (even behind an 8-bit head); 3-bit is
near-lossless for *all* features uniformly. So there's no importance gradient to
exploit: ‖down_row‖ ranks the down-projection magnitude but not **gating
sensitivity** — a 2-bit gate/up corrupts *which* features fire, and that's
precision-critical everywhere, not just the head. `uniform 3-bit` dominates every
graded schedule (lower bandwidth *and* lower KL).

**Single-step:** uniform 3-bit FFN looks near-lossless (KL 0.052, 100% top-1,
n=4) at 0.75× FFN bandwidth (~16% total → ~1.19× decode if it holds). **But the
single-step number oversold it.**

**⚠️ Generation drift overturns the single-step story.** Greedy-decoding 10
prompts to 32 tokens, sim-Q4 vs Q3 (`chris-experiments/larql_probes/examples/walk_ffn/walk_ffn_drift.rs`): **per-step
argmax flip rate 19.1%, mean first-divergence token 4/32, 0/10 exact match.** The
"100% top-1" was an n=4 artifact; KL 0.05 bits *sounds* tiny but LM distributions
are full of near-ties, so a 0.05-bit perturbation flips ~19% of argmaxes →
generations fully diverge. So **uniform-3 is near-lossless by *KL* but not by
*generation*** — it materially changes outputs. This is the depth-compounding
lesson on the sequence axis: single-step KL/top-1 can't see drift.

**The three-way per-token NLL adjudicator decides it — and the mean would have
lied.** f32 / Q4 / Q3 teacher-forced on entropic prose
(`chris-experiments/larql_probes/examples/walk_ffn/walk_ffn_nll.rs`, 74 positions):

| arm | mean | **median** | p90 | p99 | max | mean Δ vs f32 | p90 Δ | p99 Δ | worst-token Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| f32 | 4.24 | **2.21** | 11.4 | 20.3 | 31.9 | — | — | — | — |
| Q4 | 4.49 | **2.39** | 10.8 | 19.6 | 39.9 | +0.25 | +1.16 | +2.68 | +7.95 |
| Q3 | 4.23 | **2.50** | 9.9 | 20.3 | 25.0 | −0.01 | +2.45 | +3.93 | +5.50 |

flip rate Q3-vs-Q4 = **27%**. The **mean** ladder doesn't just miss the cost — it
*inverts* the decision: f32 → Q4 → Q3 = 4.24 → 4.49 → **4.23** says Q3 ≈ f32,
*better* than Q4 → would ship Q3 with a confident "beats Q4." **That's an
artifact:** one Q4 catastrophic token (worst-Δ +7.95) carries Q4's whole average.

**The load-bearing finding is the median ladder: 2.21 < 2.39 < 2.50, monotone in
precision.** This is the robust statistic — full sample behind it — and it says
on the *typical* token Q3 is the most degraded of the three. **That alone decides
#26.** The fatter Q3 tail (p90 Δ +2.45 vs Q4 +1.16) is *corroborating, not the
clincher* — tails at n=74/one-passage are exactly where confidence is lowest (one
or two positions swing p99), so the tail agreeing is reassuring but the median
carries the call. Signature: **flip-high (27%) + median NLL monotone-elevated ⇒
real cost** (tail consistent).

**Decision (#26): uniform-3 is a real quality degradation, not benign chaos — do
not ship it as a free 25%.** Two claims, kept separate:
- **Direction — settled:** Q3 is a measurable cost beyond Q4. Robust on the
  median ladder; survives even if a multi-passage rerun wobbles the tail. The
  build/no-build call needs only this, and it holds.
- **Magnitude — not settled:** "~0.12 bits/token median over Q4 / ~0.26 mean" is
  a one-passage point estimate; wants the multi-passage rerun before it's a
  number to quote. Don't let the unsettled magnitude leak back into doubt about
  the settled direction.

The single-step KL (0.052) *hid* the cost, the *mean* NLL *inverted* it; only the
per-token distribution got it right. The mechanism (gate ≥ 3, down ≥ 2 bit
floors) stands; the *free-bandwidth* claim does not. FFN-only; 2-bit is the cliff.

This is the lever that survived the whole arc: not "touch fewer weights" (the FFN
is dense), but "stream each weight in fewer bits" (the FFN is 3-bit-tolerant).

### Mechanism: the gate is precision-critical, the down is forgiving

*Why* is the floor universal at 3 bits, with no down-norm gradient to grade
along? Because the spendable variance lives on the **gate-vs-down axis**, not the
importance axis. Grading by *component* (uniform per gate/up/down) confirms it:

| gate/up/down bits | bw vs Q4 | KL (bits) | top-1 |
|---|---:|---:|---:|
| 3 / 3 / 3 (uniform-3) | 0.75× | 0.052 | 100% |
| 3 / 3 / **2** | 0.67× | 1.16 | 75% |
| 3 / 3 / **1** | 0.58× | 18.6 | 0% |
| **2** / 2 / 3 | 0.58× | 15.6 | 0% |
| 4 / 4 / 2 | 0.83× | 0.67 | 100% |

**`gate@2` collapses (KL 15.6, routing breaks); `down@2` survives (KL 1.16,
alive).** A corrupted gate/up flips *which* features fire — a discrete routing
error with no magnitude tail to grade. The down-projection is magnitude-only and
tolerates 2 bits (floor at 1). So the precision floors are **gate ≥ 3 bits, down
≥ 2 bits** — a real statement about where the FFN's precision-critical surface
lives. The bandwidth payoff is bounded though: the down's extra bit isn't free
(`g3/d2` = 0.67× but KL 1.16, degraded; `g4/d2` = lossless but 0.83×), so
**uniform-3 (0.75× @ KL 0.052) stays the near-lossless sweet spot** and the
asymmetry only extends the frontier toward 0.67× if degradation is acceptable.

## Resolution — the WalkFfn sparse thesis, settled

The full arc (tasks #17–#22), gemma3-4b, KL bits:
- **Speed (#18):** cheap routing beats dense up to ~15×; gate-KNN ~3× slower.
- **Accuracy ceiling (#19):** whole-model sparsity collapses; only a late band
  survives.
- **Shallow band (#20):** static-importance routing ties gate-KNN at ~4 layers —
  ships today (keep pool ≤4096).
- **Static shortcut for deeper (#21):** ruled out — a static candidate set caps
  accuracy regardless of P or score precision (de-confounded).
- **Content-addressed candidate set (#22):** the residual-cell router **matches
  gate-KNN accuracy in *and* out of distribution** (Wilcoxon: in-dist p=0.64,
  OOD-agg p=0.074 — both ties), at **~150× cheaper routing**, and **beats static
  in-distribution** (p=0.0001, n=30 — the load-bearing number). A suggestive OOD
  edge over gate-KNN is code-specific and small-n — a hypothesis, not a result.
  Overfit-collapse refuted. ⚠️ Two caveats: (a) rank→K ≠ full pool *OOD* — the
  pool's breadth is the robustness, so the cheap matched-K variant may shed the
  edge; (b) cell≫static OOD is a wash on code and partly static face-planting on
  non-English — the in-dist result carries it.

**Net:** sparse WalkFfn is viable as a late-layer band with a residual-cell
router — content-addressed routing that **matches gate-KNN accuracy without the
full gate projection (~150× cheaper routing)**. The robust, everywhere-true,
no-statistics-required claim is the **cost win + parity**: cell-router = gate-KNN
accuracy at a fraction of the routing MACs, and **≫ static in-distribution**
(p=0.0001, n=30). The cell-beats-gate-KNN-OOD edge is a code-specific, larger-n
hypothesis, not a headline. Remaining to productionise (task #23): **confirm
end-to-end** first — the win is so far isolated-FFN (KL) + a routing-MAC
projection, not a measured decode tok/s with KV cache. Measure **both full-pool
and rank→K** for tok/s **and top-1 agreement vs dense** (a 0.4–0.8 bit median KL
can flip a non-trivial fraction of argmax picks; agreement is what shows in
generations) — if full-pool (1097) costs barely more than rank→K (512) once the
FFN matmul dominates routing, full-pool is strictly better (keeps OOD breadth)
and rank→K was a false economy. Then persist the `CellRouter` to the vindex
(offline build at index time); wire the
cell lookup on the Metal path; larger n on the cell-vs-gate-KNN OOD margin. The
infra (`CellRouter` in `WalkFfnConfig`, per-position lookup in `walk_ffn_sparse`)
is landed and tested.

## Metric discipline — the invariant behind four falsifications

Across this arc, four cheap metrics each flattered a result the harder measurement
falsified — **cosine → single-step KL → end-to-end → mean NLL** — and every one
failed *in the same direction*. The slogan "averages hide the tail" is true but
under-specified. The predictive invariant:

> **An expectation lies when the use is a repeated extremum.** Generation is an
> argmax (or sample) walked thousands of times — a max-operation, dominated by
> its tail by construction. Cosine, single-step KL, and mean NLL are all
> expectations; the thing shipped runs an argmax in a loop. The metric has to
> match the operation at the point of use.

This is *predictive*, not just a catalogue: it says in advance which future cheap
metrics will lie — **any expectation, whenever the use is a repeated extremum.**
Greedy decode is repeated argmax, so the right statistic is the per-token
distribution *including where the argmax goes wrong* (flip rate, median ladder,
worst-token tail), never the central tendency. The mean-NLL inversion at #26 is
the textbook case: one catastrophic token carried the average and reversed the
ship/no-ship call; the median (robust) and flip rate (the argmax itself) decided
it correctly. Carry this into the next arc: **match the statistic to the operation
at the point of use; distrust every expectation whose use is an extremum.**

## Where the speed budget should go (the cleared board)

The whole speed thread is, in effect, an expensive but rigorous proof of where
*not* to spend — and by elimination, where to. Cleared:
- **Sparsity** — the FFN is dense (K≈4096 for token faithfulness); the gather
  kernel wins isolated (1.29×) but not end-to-end.
- **Precision-thinning** — real floors (gate 3, down 2 bits) but uniform-3 is a
  measurable quality cost, not free.

What survives depends on **no thesis at all** — just the bandwidth fundamentals on
weights you already stream:
- **Q4K-direct attention** (~28% of decode, client-facing, removes the f32 tax) —
  the next arc. Needs no sparse-FFN or precision claim to be true.
- **Blocked-gemm prefill kernel** (the matvec→gemm shape, the 4–14× prefill gap).
- **lm_head Q4K** (~12%, per-token).
- **For serving:** batch-union gemm (gemm shape + amortised bytes on correlated
  batches) and **grid distribution** (the only ceiling-breaker past the per-box
  bandwidth wall).

**The clean factoring for stakeholders:** the graph FFN's contribution is
**capacity/accuracy** (the cell-router matching gate-KNN cheaply, #22) — *not*
speed. One mechanism for capacity, the bandwidth fundamentals for speed, no
overlap to defend. That is a cleaner story than "the FFN gives us a bit of both,"
and the falsification arc is what earned the right to state it plainly.

## #27 Temporal cursor reuse — the first POSITIVE signal (delta-walk is live)

The weight axis (#17–#26) was "touch less" and it's mined out. This probed the
TEMPORAL axis dense BLAS structurally can't see (no cursor, no delta between
tokens). Guardrail honored: token-to-token at FIXED layer, last-position residual
over real history (`predict_with_ffn_trace` on teacher-forced prefixes =
KV-cached decode step), never within-prefill cross-position (that would be the
spatial cosine wearing a temporal label). `chris-experiments/larql_probes/examples/walk_ffn/walk_ffn_temporal_reuse.rs`,
6 entropic passages, per-zone distribution (median / p10 / worst), not mean:

| zone | residual cosine (med/p10/worst) | pool Jaccard (med/p10/worst) | delta TwoNN |
|---|---|---|---:|
| pre-commit L0–4 | 0.764 / 0.073 / −0.04 | 0.19 / 0.13 / 0.11 | 5.8 |
| **highway L5–20** | **0.997 / 0.969 / 0.916** | 0.61 / 0.50 / 0.39 | **22.3** |
| retrieval L21–29 | 0.987 / 0.976 / 0.957 | 0.44 / 0.32 / 0.19 | 22.1 |
| format L30–33 | 0.978 / 0.968 / 0.952 | 0.39 / 0.31 / 0.22 | 22.2 |

Pre-committed reading resolves:
- **Delta-walk LIVE.** Highway delta TwoNN **22.3 ≤ 30** (lands on ~22 = the known
  intrinsic state dim). And it's **22 across L5–L33** (29 of 34 layers), even
  where pools churn more — so delta-walk **decouples from pool stability**: it
  rides the low-rank residual *delta* regardless of route movement.
- **Route reuse OUT** — pool Jaccard 0.61 < 0.9 (pools churn ~40% token-to-token).
- **Output reuse tail-blocked** — median cosine 0.997 (cursor barely moves;
  *higher* than the pre-registered 0.95–0.98) but worst 0.916; naive
  contribution-reuse drifts on the ~10% of steps that move. Delta-walk is the
  tail-safe form.
- **Pre-commit (L0–4) is a different regime** — residual unstable token-to-token
  (cosine 0.76), not yet on the manifold.

**This is the first positive in the speed program, and it's dense-inexpressible**
— a matmul engine has no cursor and no delta to exploit. But the honest gate:
**TwoNN ~22 is the ENABLING metric (necessary), not the win (sufficient).** The
delta being low-rank means you *can* compute the layer's action in a ~22-dim
subspace — but the FFN is nonlinear (gate/gelu/up/down), so realizing it needs
the highway to be near-linear enough that `f(base+δ) ≈ f(base) + J·δ` with a
precomputed low-rank Jacobian-subspace `J`. The prototype (#28) carries the real
risk — that the linearized action is both **cheaper than the matvec** AND
**faithful** — and it must clear the *same* per-token-distribution + worst-token
NLL/drift gate that killed Q3, not single-step KL. Pre-committed bar met → build
the subspace-projection delta-walk prototype; then gate it like everything else.

## #28 stage 1 — delta-walk KILLED by a script (amplitude ≠ rank)

#27's "delta is ~22-dim" was the enabling metric but it equalled the *state* dim
— a warning, not encouragement: a delta spanning the same manifold as the state
is a *full-amplitude* move, not a thin perturbation. Low-rank ≠ small. So before
any kernel, the cheap falsification: amplitude `‖δ‖/‖base‖` + full-Jacobian
linearization error `‖f(base+δ)−(f(base)+Jδ)‖/‖f(base+δ)‖` (finite-diff JVP),
**targeting the FFN-INPUT residual (post-attn-norm) — what the FFN actually
sees — not #27's layer-input residual.** `chris-experiments/larql_probes/examples/walk_ffn/walk_ffn_delta_walk.rs`:

| zone | ‖δ‖/‖base‖ med/p90/worst | lin-error med/p90/worst |
|---|---|---|
| highway L5–20 | **0.854** / 1.02 / 1.30 | **0.691** / 1.10 / 2.39 |
| retrieval L21–29 | 0.901 / 1.13 / 1.45 | 0.592 / 0.90 / 1.49 |
| format L30–33 | 0.920 / 1.12 / 1.34 | 0.488 / 0.69 / 1.24 |

**Both kill bars (amp ≳0.20, lin-error ≳0.15) blown out 4–5×.** The FFN-input
delta is ~0.85× the base magnitude (near-full-amplitude), and the **exact**
Jacobian linearization is **69% wrong** on the highway. No low-rank approximation
beats a 69%-error full Jacobian. **Delta-walk dead — killed for a script, no
kernel** (the #24 build-then-measure trap avoided by front-loading the probe).

**Why #27 looked positive — the load-bearing correction:** #27 measured the
*layer-input* residual (the residual *stream*) — cosine 0.997, smooth, real. But
the FFN sees the *post-attention-norm* residual, and **attention injects a
full-amplitude token-specific update every step** between them. The cursor is
smooth; the FFN rides cursor + per-token attention injection, which jumps. The
smooth object and the expensive op's input are *different objects* — the speed
technique needed the latter smooth, and it isn't. The rank measurement alone
couldn't see this; the amplitude probe did, for one forward pass.

## Temporal axis: CLOSED — for this architecture, measured

The temporal axis closes too — not because there's no temporal structure (the
residual stream genuinely is a smooth cursor) but because **the expensive
operation's input is not the smooth thing.** Qualifier kept honestly: the kill is
that *attention's per-step write is full-amplitude relative to the FFN input* —
an empirical fact about **this model's residual scaling (Gemma 3 4B, its norm
placement)**, not a theorem. Almost certainly general, but a *measured* scaling
fact — so "temporal axis closed" is correct *for this architecture measured*, not
"no temporal technique can work on any model." A model whose attention writes are
small relative to the FFN-input scale could reopen it; this one doesn't.

Combined with the weight axis (#17–#26), every graph-FFN-thesis speed lever is now
falsified for this model. What remains depends on no thesis: **Q4K-direct
attention** (~28%, the next arc), blocked-gemm prefill, lm_head Q4K; and for
serving, batch-union gemm + grid distribution. The graph FFN's contribution is
**capacity** (cell-router #22), not speed.

## The proxy is not the thing — two distinct disciplines, not one

The five falsifications are *not* five instances of one pattern; that's the
collapse to resist. The first four — cosine→KL, KL→end-to-end, single-step→drift,
mean→distribution — are all **"the average hid the tail"**: the decision lived in
a tail the central tendency averaged away, caught by refusing the average and
reading the distribution at the point of use. But **#28 is a different shape.**
The kill was visible in the *central tendency* (median amplitude 0.85) — not a
tail problem at all. It was **"the right statistic on the wrong object"**: #27's
cosine 0.997 was *correct about the residual stream* and *irrelevant to the FFN*,
because the FFN consumes a different vector (post-attention). Two distinct
disciplines, failing differently and caught differently — one needs the
*distribution*, the other needs the *right input vector*. The unifying principle
sits one level above both: **the proxy is not the thing.** Keep the two distinct;
merging them loses that #28 would have sailed past any distribution check on the
stream — you had to measure the FFN's actual input to see it.

## Strategic close — discipline as the asset (the framing that's earned)

Stated flat, "we exhaustively falsified the graph-FFN speed thesis" reads as a
defeat. It isn't one, and the accurate framing matters:

- **The capacity result is real and stands.** The graph-FFN buys *capacity* — the
  cell-router matching gate-KNN at ~150× cheaper routing (#22), the KG/context
  retrieval endpoints, the compression. None of that was touched by the speed
  falsifications; they're a different axis.
- **The speed program refused to let an elegant idea coast on the capacity win.**
  A weaker program ships the 1.29× isolated gather (#24) or the "near-lossless"
  Q3 (#26) and discovers the problem in production. This one found *separate* ways
  each plausible win would have been wrong — **before building the thing**: the
  gather died at end-to-end (memory/prefetch), Q3 died at the per-token
  distribution (mean *inverted* the decision), delta-walk died at the amplitude
  probe (one forward pass, no kernel). Each falsification got *cheaper*, which is
  the discipline compounding.
- **The factoring is now a conclusion, not a framing choice:** capacity = graph
  FFN, speed = bandwidth fundamentals, forced by elimination on both axes.

So the message for stakeholders is not "the FFN idea didn't pan out for speed" —
it's "**the capacity result is real, and we held the speed claims to a bar that
killed the ones that wouldn't survive contact.**" That discipline is the asset:
it's the thing that makes the *capacity* numbers trustworthy too, because the same
team that killed three plausible-but-wrong speed wins is the one reporting the
capacity wins. Next: Q4K-direct attention, no thesis riding on it.
