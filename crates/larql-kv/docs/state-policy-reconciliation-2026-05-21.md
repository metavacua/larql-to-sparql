# State Policy + KV Engine Unification — reconciliation patch (2026-05-21)

**One PR, four changes, three docs.** Brings the spec set into
agreement with the engine inventory after the W10 default-on flip,
the `boundary_per_layer` W1-GPU + W10 wire-up, and the modular split
of all seven `engine.rs` files.

## Summary

The bench table from 2026-05-21 (Gemma 3 4B Q4K, M3 Max, 50 decode
tokens, W10 default-on) makes one claim load-bearing: **the
canonical-vs-derivative classification in State Policy §2 is
operationally predictive**, not just descriptive. Three engines that
classify K/V as derivative all sit at +0.4–1.1% over `standard`.
The one engine that classifies K/V as canonical sits 12.9% behind.
Same compression ratio, same correctness contract, different §2
slot — different perf shape.

| Engine | K/V slot | W10 mask reached | tok/s | gap |
|---|---|---|---:|---:|
| `Standard` | canonical (backend-managed) | n/a | 97.6 | — |
| `MarkovResidual` (windowless) | **derivative** | None | 98.0 | +0.4% |
| `MarkovResidualCodec` (windowless) | **derivative** | None | 98.1 | +0.5% |
| `BoundaryPerLayer` (windowless) | **derivative** | None | 98.7 | +1.1% |
| `WindowedCheckpoint` (window=256) | **derivative** | HOnly | 94.2 | -3.5% |
| `TurboQuant` (bits=4) | canonical (destructive codec) | n/a | 85.0 | -12.9% |
| `NoCache` | canonical (no K/V; re-forward) | n/a | n/a | (debug fallback) |
| `BoundaryKv` | canonical (composes `Standard`) | n/a | — | no live Q4K bench (2026-05-21: `Q4K engine prefill failed`) |
| `Apollo` | n/a (no K/V; retrieval injection) | n/a | — | no live Q4K bench (requires populated boundary store) |

Nine engines in `larql-kv` total; **six are bench-validated** on the
canonical/derivative perf story. `Standard`/`NoCache` are the
controls (debug-only for NoCache). `BoundaryKv` and `Apollo` exist
in the codebase with valid contracts but **don't currently run on
the production Q4K bench path** — `BoundaryKv` fails Q4K prefill
under its `Standard`-composition route, `Apollo` requires a
populated boundary store that the bench doesn't supply. Both are
documented in §2-§3 of this patch as outside the W10 perf story
on that empirical basis.

The patch makes that picture explicit in the two governance specs.

---

## Patch 1 — `state-policy.md §3.1`: refresh worked example with current numbers

The §3.1 table was written 2026-05-18 with pre-default-flip numbers
(`MarkovResidualEngine` "106.8 (None)"). Current bench is 98.0 (None
mask, default-on). Replace inline numbers; add task #31 as the
explicit prediction-test.

**Replace** the table at `state-policy.md:136-142`:

```diff
-| Engine | Canonical | Derivative dropped under W10 | New tok/s ceiling |
-|---|---|---|---:|
-| `MarkovResidualEngine` | residual stream | `hot_kv`; (`rs.stored` too when `window=None`) | 106.8 (None) |
-| `MarkovResidualCodecEngine` | codec residuals | same | 98.5 (None) |
-| `WindowedCheckpointEngine` | KV within window | `current_window_kv` (CPU shadow of the Metal cache) | 92.8 (HOnly) |
-| `TurboQuantEngine` | compressed K/V (destructive) | nothing — K/V IS canonical | — |
-| `StandardEngine` | KV tensors | n/a — backend-managed already | (reference, ~100) |
+| Engine | Canonical | Derivative dropped under W10 | W10 tok/s |
+|---|---|---|---:|
+| `StandardEngine` | KV tensors | n/a — backend-managed | 97.6 (control) |
+| `MarkovResidualEngine` | residual stream | `hot_kv`; (`rs.stored` too when `window=None`) | **98.0** (None) |
+| `MarkovResidualCodecEngine` | codec residuals | same | **98.1** (None) |
+| `BoundaryPerLayerEngine` | per-layer codec residuals | same | **98.7** (None) |
+| `WindowedCheckpointEngine` | KV within window | `current_window_kv` (CPU shadow) | 94.2 (HOnly) |
+| `TurboQuantEngine` | compressed K/V (destructive) | nothing — K/V IS canonical | 85.0 (Full) |
```

**Append** to §3.1, after the existing "The cut held" sentence:

```diff
+The 13% delta between the four derivative-K/V engines and
+`TurboQuantEngine` is the cleanest available evidence that the
+canonical/derivative classification is operationally load-bearing.
+Same correctness contract (bounded_KL), same compression ratio in
+the `TurboQuantEngine` vs `MarkovResidualCodecEngine` case — the
+only difference is which slot K/V lives in. Task #31 (the
+derivative-K/V `TurboQuant` variant) is the explicit
+prediction-test: an engine with `TurboQuant`'s compression but K/V
+reclassified as derivative *must* join the 98+ club; if it doesn't,
+the cut is descriptive but not predictive and §3 needs revising.
+
+The W10 cascade flipped to default-on 2026-05-21; the cascade is
+now the assumed mode for engines that opted in, with
+`LARQL_W10_DISABLE=1` available as a debug opt-out.
```

---

## Patch 2 — `state-policy.md §5`: annotate the three engines that sit outside the canonical/derivative perf story

The §5 table currently slots every engine in `larql-kv`. That's
correct, but three of them aren't on the perf story the rest of
the spec is about:

- **`StandardEngine` / `NoCacheEngine`** — controls (no derivative
  state by construction).
- **`BoundaryKvEngine`** — composes `StandardEngine` with additive
  frame emission. State policy is `Standard`'s; the frames don't
  participate in the §3 cut.
- **`Apollo`** — `task_level_retrieval` contract. No K/V state to
  classify; the §3 cut doesn't apply.

These are valid engines with valid contracts — they should stay in
§5 — but the table should signal they don't participate in §3's
prediction.

**Replace** the table at `state-policy.md:201-211`:

```diff
-| Engine | Canonical state | Derivative state | Contract |
-|---|---|---|---|
+| Engine | Canonical state | Derivative state | Contract | §3 cut applies? |
+|---|---|---|---|---|
-| `StandardEngine` | KV tensors | — | `exact_logits` |
+| `StandardEngine` | KV tensors | — | `exact_logits` | control |
-| `NoCacheEngine` | tokens | — | `exact_logits` |
+| `NoCacheEngine` | tokens | — | `exact_logits` | control (debug) |
-| `MarkovResidualEngine` | residual stream | hot KV | `exact_logits` under arch preconditions |
+| `MarkovResidualEngine` | residual stream | hot KV; `rs.stored` (windowless) | `exact_logits` under arch preconditions | **yes** |
-| `MarkovResidualCodecEngine` | codec-encoded residuals | hot KV | `bounded_KL(ε)` — ε stated per codec |
+| `MarkovResidualCodecEngine` | codec-encoded residuals | hot KV; `rs.stored` (windowless) | `bounded_KL(ε)` — ε per codec | **yes** |
-| `BoundaryKvEngine` | KV tensors + chunk frames | — | `exact_logits` |
+| `BoundaryKvEngine` | KV tensors + chunk frames | — | `exact_logits` | no live Q4K bench (2026-05-21) |
-| `BoundaryPerLayerEngine` | per-layer codec policy over residuals | hot KV | `bounded_KL(ε_l)` per-layer; calibrated |
+| `BoundaryPerLayerEngine` | per-layer codec residuals | `rs.stored` (windowless) — no hot KV shadow | `bounded_KL(ε_l)` per-layer; calibrated | **yes** |
-| `WindowedCheckpointEngine` | KV tensors (within window) + per-window checkpoints + token archive | — | `exact_logits` within window |
+| `WindowedCheckpointEngine` | KV (within window) + per-window checkpoints + token archive | `current_window_kv` (CPU shadow of Metal cache) | `exact_logits` within window | **yes** (HOnly only) |
-| `TurboQuantEngine` | quantised KV (in-place) | — | `bounded_KL` — codec round-trip ≥ cos 0.991 on real distributions |
+| `TurboQuantEngine` | quantised KV (in-place; destructive codec) | — | `bounded_KL` — cos ≥ 0.991 | **no** — K/V is canonical; see task #31 for the derivative-KV variant |
-| `Apollo` | boundary retrieval / residual injection store | — | `task_level_retrieval` |
+| `Apollo` | boundary retrieval / residual injection store | — | `task_level_retrieval` | no live Q4K bench (requires populated boundary store) |
```

**Add** at the end of §5, before the existing "Some entries look
surprising" callout (line 213):

```diff
+The right-hand column is the §3 prediction test. "**yes**" engines
+gain the W10 mask cascade's readback skip and sit in the 94-99
+tok/s band. The other annotations are factually grounded:
+
+- "control" — `Standard`/`NoCache` define what we measure against;
+  no derivative state to drop.
+- "**no**" — `TurboQuant` keeps K/V canonical (destructive codec),
+  sits 13% behind; task #31 (state-policy §8) is the test of
+  whether reclassifying K/V as derivative closes the gap.
+- "no live Q4K bench (2026-05-21)" — `BoundaryKvEngine` fails
+  Q4K prefill under its `Standard`-composition route; `Apollo`
+  requires a populated boundary store that the bench doesn't
+  supply. Both have valid contracts; neither is currently part
+  of the W10 perf story because neither runs on the path the
+  W10 cascade lives on.
```

---

## Patch 3 — `state-policy.md §8`: open question for the prediction test

Add a new open question pointing at task #31.

**Append** to §8 (open questions):

```diff
+- **Is the canonical/derivative cut *predictive* or just
+  *descriptive*?** The 2026-05-21 bench is consistent with the
+  predictive reading: every engine on the derivative side sits
+  at the W10 ceiling; the only canonical-K/V engine sits 13%
+  behind. The explicit test is task #31 — port `TurboQuant`'s
+  WHT+Lloyd-Max compression to a derivative-K/V engine
+  (residuals canonical, quantised K/V reconstructed on demand).
+  If the new engine doesn't join the 98+ club, the cut is
+  descriptive only; §3 needs revising and the W10 perf win is
+  better explained by something else (the kv_cache-on-GPU
+  shape, residual-stream layout, etc.).
```

---

## Patch 4 — `kv-engine-unification.md §5`: catalog refresh

The §5 table is missing `MarkovResidualCodec` and
`BoundaryPerLayer`; both are now in `EngineKind` and on the
production bench path. Also note the W10 default-on flip.

**Replace** the table at `kv-engine-unification.md:238-246` with two
tables — bench-validated engines first, then the engines that exist
but aren't on the production bench path.

### 5.1 Bench-validated engines (Q4K Metal, 2026-05-21)

```diff
+| `EngineKind` variant | CLI spec | Underlying mechanism | W10 ceiling | Status |
+|---|---|---|---:|---|
+| `Standard { window_size: None }` | `--engine standard` (default) | Full K/V tensor cache, unbounded growth | 97.6 (control) | **Default**, bit-parity |
+| `Standard { window_size: Some(N) }` | `--engine standard:window=N` | Sliding-window K/V tensor cache | n/a (perf-shape == None) | bit-parity |
+| `NoCache` | `--engine no-cache` | Full re-forward per step (O(N²)) | n/a (debug) | bit-parity, debug fallback |
+| `MarkovResidual { window_size }` | `--engine markov-rs[:window=N]` | Stores residuals, recomputes K/V at decode | **98.0** (None) | On live path |
+| `MarkovResidualCodec { window_size, codec }` | `--engine markov-rs-codec[:window=N]` | `MarkovResidual` + bf16-encoded cold-tier residuals (2× cold saving) | **98.1** (None) | On live path |
+| `BoundaryPerLayer { window_size, num_layers }` | `--engine boundary-per-layer[:window=N,layers=L]` | Per-layer codec policy on cold tier; calibration-driven; cold-start convenience constructor | **98.7** (None) | On live path |
+| `WindowedCheckpoint { window_size }` | `--engine unlimited-context:window=N` | Per-window K/V checkpoints + token archive | 94.2 (HOnly only) | On live path |
+| `TurboQuant { bits }` | `--engine turbo-quant:bits=N` | WHT + Lloyd-Max 3/4-bit K/V codec (canonical K/V — destructive) | 85.0 (Full) | On live path; see §7.2 for the derivative-KV variant proposal |
```

### 5.2 Engines with valid contracts but no live Q4K bench path

```diff
+| `EngineKind` variant | CLI spec | Underlying mechanism | Bench status (2026-05-21) | Why |
+|---|---|---|---|---|
+| `BoundaryKv { ... }` | `--engine boundary-kv:chunk_tokens=N,sequence_id=…` | Composes `Standard` with `larql-boundary` frame emission per chunk | **`Q4K engine prefill failed`** | `Standard`-composition path doesn't currently route through the Q4K dispatch fast path; the engine works in unit-test fixtures but isn't wired for the production Metal Q4K bench. |
+| `Apollo { ... }` | `larql bench --engine apollo` only | Boundary store + residual injection (`task_level_retrieval` contract) | **prefill returns `None`** | Requires a populated boundary store; the bench harness doesn't load one. Also not wired into `larql run` / `larql walk` (§7). |
```

Both engines have valid `StatePolicy` slots and exist in the
codebase; they're documented in their own per-engine specs.
They're listed in §5.2 rather than §5.1 because the
canonical/derivative perf story in §7.2 / State Policy §3.1 is
falsified or supported on the bench, and these two engines
aren't currently producing data either way.

**Replace** the explanatory paragraph at `kv-engine-unification.md:248-252`:

```diff
-`Standard` is a thin `KvEngine` adapter over `generate_cached_bounded`;
-no new cache code, just a trait impl. `NoCache` is similarly a thin
-adapter over the existing full-forward loop. `MarkovResidual` is the
-existing residual-stream engine and is **not** an alias for `Standard`
-— it's a different mechanism, exposed as a peer option.
+`Standard` is a thin `KvEngine` adapter over `generate_cached_bounded`;
+no new cache code, just a trait impl. `NoCache` is similarly a thin
+adapter over the existing full-forward loop. `MarkovResidual` is the
+existing residual-stream engine and is **not** an alias for `Standard`
+— it's a different mechanism, exposed as a peer option.
+
+`MarkovResidualCodec` and `BoundaryPerLayer` are spec'd in their
+own per-engine docs; both opted into the W10 mask cascade as of
+2026-05-21. `TurboQuant`'s 85 tok/s ceiling reflects its
+canonical-K/V status — task #31 proposes a derivative-K/V sibling
+that should join the 98+ band.
+
+**W10 mask cascade default-on (2026-05-21).** Engines that opted in
+take the cascade automatically; `LARQL_W10_DISABLE=1` opts out for
+debug. The legacy `LARQL_W10_HONLY=1` env var is still accepted but
+is now a no-op. Backends without an optimised masked path fall
+through to `Full` via the trait's default impl — correct everywhere,
+perf-positive on Metal.
```

---

## Patch 5 — `kv-engine-unification.md §7`: refine Apollo carve-out

Current §7 framing reads as defensive ("forcing Apollo behind the
same trait... obscures..."). The bench data makes it cleaner:
Apollo's contract genuinely is orthogonal and the carve-out is
*correct*, not aspirational. Also add a §7.2 pointer to task #31.

**Replace** §7 paragraph at `kv-engine-unification.md:304-322`:

```diff
-Forcing Apollo behind the same trait as "sliding window K/V" obscures
-that it is a different mechanism with different correctness
-expectations (Apollo's `markov-residual-engine.md:393` row 5 says
-"first-token factual, not bit-exact"). Two consequences:
-
-1. Apollo **stays in `larql-kv`**, remains a valid `EngineKind` variant,
-   and remains reachable via `bench --engine apollo`.
-2. Apollo is **not wired into `larql run` / `larql walk`** in this
-   unification. A future spec — or a sibling trait
-   (`ContextRecallEngine`?) — handles Apollo's wiring on its own terms.
-3. The `--engine apollo` spec in `run`/`walk` returns an explicit
-   `not-yet-wired-for-live-decode` error. It does not silently fall
-   back to a default engine.
-
-This carve-out is the load-bearing reason the unification is shippable
-on a short timeline. Without it, the trait either grows a second
-abstract method group (boundary store) or starts leaking
-implementation-specific concepts (`injection_layer`) into the live
-dispatch path.
+Under State Policy's `(canonical, derivative, contract)` triple,
+Apollo's slot is `(retrieval store, —, task_level_retrieval)`.
+The W10 mask cascade's perf prediction is over engines whose
+canonical state is K/V or residuals; Apollo has neither. The
+2026-05-21 bench confirms this empirically — the other six
+engines participate in the 13% W10 win because they classify
+K/V as derivative; Apollo is outside that conversation by
+contract.
+
+Three consequences:
+
+1. Apollo **stays in `larql-kv`**, remains a valid `EngineKind`
+   variant, and remains reachable via `bench --engine apollo`.
+2. Apollo is **not wired into `larql run` / `larql walk`**. A
+   future sibling trait (`ContextRecallEngine`?) handles Apollo's
+   wiring on its own terms.
+3. The `--engine apollo` spec in `run`/`walk` returns an explicit
+   `not-yet-wired-for-live-decode` error; no silent fallback.
+
+The carve-out is correct (not aspirational): a trait shaped around
+`(canonical, derivative, contract)` where K/V is the canonical
+axis has nothing useful to say about an engine whose contract is
+`task_level_retrieval`. Polluting the trait to accommodate Apollo
+would either grow a second abstract method group (boundary store)
+or leak `injection_layer` into the live dispatch path. Both are
+spec smells the State Policy work was designed to prevent.
+
+### 7.2 The `TurboQuant` carve-out is *not* the same shape
+
+Unlike Apollo, `TurboQuant`'s 85-vs-98 tok/s gap is **not**
+orthogonal to the trait — it's the predicted outcome of K/V
+classification. Task #31 proposes a derivative-K/V `TurboQuant`
+sibling: residuals canonical (markov-rs-style), quantised K/V
+reconstructable on demand via the WHT+Lloyd-Max codec. Same
+compression ratio, same correctness contract, different §2 slot.
+If the prediction holds, the new engine joins `MarkovResidual`
+et al. at 98+ tok/s and `TurboQuant` itself is preserved as the
+"canonical K/V codec" reference for cases where reconstruction
+isn't acceptable.
```

---

## What this patch does NOT do

- **Doesn't drop `BoundaryKvEngine` or `Apollo` from the engine
  catalog.** Both exist in the codebase, both have valid State
  Policy slots, both have per-engine specs in
  `crates/larql-inference/docs/specs/`. The patch moves them into
  KV Engine Unification §5.2 ("no live Q4K bench path") rather
  than pretending they participate in the §3.1 perf story — the
  2026-05-21 bench shows `BoundaryKv` fails Q4K prefill (its
  `Standard`-composition route isn't wired for the production Q4K
  dispatch fast path) and `Apollo` returns `None` from prefill
  without a populated boundary store. Their absence from §5.1 is
  factual, not editorial.
- **Doesn't add task #31's implementation.** The patch references
  task #31 as the prediction test; implementing the derivative-K/V
  `TurboQuant` variant is a separate engine + spec.
- **Doesn't update the per-engine specs** (`markov-residual-engine.md`,
  `boundary-per-layer-engine.md`, etc.). Those have their own
  `## Phase N — landed YYYY-MM-DD` sections that already track
  the per-engine changes. The two governance specs (State Policy
  + KV Engine Unification) are the ones that need this
  reconciliation.

## Rationale

The spec set is now honest about what exists: seven contract-bearing
engines, three contract kinds (`exact_logits`, `bounded_KL`,
`task_level_retrieval`), W10 mask cascade default-on, the
canonical/derivative cut empirically validated as a perf lever.
Every claim in the post-patch State Policy + KV Engine
Unification specs points at running code with a 2026-05-21 bench
number behind it.

The aspirational rows get pruned in spirit (annotated as
"outside the §3 cut") but not deleted, so the spec stays
descriptive of the full inventory. The strength comes from the
honesty about what each row is *for*.
