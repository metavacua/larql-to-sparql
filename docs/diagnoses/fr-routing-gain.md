# FR routing — the end-to-end gain (and the two-tier caveat), quantified

**Date:** 2026-06-07. **Status:** ran (`chris-experiments/larql_probes/examples/fleet_routing/fr_routing_gain.rs`). Gemma-3-4B Q4K vindex, 20 novel facts installed at L26, three query slices, all three router modes on the **same** forward passes. Answers "does the FR work give a new gain?"

## Headline

**Yes — a large correctness gain on the KNN fact-injection path, ~zero latency cost. Not a throughput gain.** With only 20 installed facts, the **legacy top-1+0.75 router confident-wrongs 100% of non-installed entities** (it injects a stored fact into queries it should leave alone). The **FR1 verified router fixes this entirely (0% → 100% distractor-safe)** while preserving installed-fact recall, at **~13 µs/call** — negligible against a ~10-40 ms decode. The gain grows with store size (more facts → more near-rank-1 collisions → legacy degrades; verified stays safe).

## Results (20 installed, 20 distractors, 5 aliases)

| mode | CORRECT (want fact) | DISTRACTOR-safe (want NO override) | ALIAS (want fact) | override µs/call |
|---|---|---|---|---|
| **legacy** (top-1 + 0.75) | 20/20 (100%) | **0/20 (0%)** | 5/5 (100%) | 22 |
| **verified** (FR1) | 20/20 (100%) | **20/20 (100%)** | 0/5 (0%) | 13 |
| **two_tier** (FR2) | 20/20 (100%) | 0/20 (0%) | 5/5 (100%) | 18 |

- **CORRECT** — query about an installed entity; want the installed fact. All three: 100% (no regression).
- **DISTRACTOR** — query about a non-installed entity the model knows; the right move is *no override*. Legacy fires on **every** one (near-rank-1 cosine collides with some stored key > 0.75) → confident-wrong. Verified rejects (the distractor isn't a named installed entity) → the model answers itself.
- **ALIAS** — historical name of an installed entity; want the installed fact. Legacy lands them (the alias's nearest key happens to be the right canonical here); verified abstains (alias name ∉ prompt); two-tier recovers via the activation fallback.

## What this says about "the gain"

1. **FR1 verified is the production win.** Legacy KNN injection is **unsafe at any realistic store size** — 20 facts already corrupt 100% of unrelated queries. Verified makes the feature trustworthy (distractor-safe 0→100%, facts preserved) for **free** (13 µs, a 0.05% sidecar). This is the gain: it's what lets KNN fact-injection actually ship.
2. **No tok/s change.** The override is a post-logits sidecar; verify adds a top-k + a prompt-substring check (µs). For normal generation (no KnnStore) it's a no-op. There is no throughput gain and no measurable cost.
3. **Two-tier (FR2) is a targeted alias tool, NOT a general default — quantified.** Its fallback has no entity-name guard, so on distractors it confident-wrongs exactly like legacy (0/20). Use `ROUTE VERIFY FALLBACK` **only when the query is known to be an alias/paraphrase of a stored entity**; for open inference use `ROUTE VERIFY`. This is the E16 caveat, now measured: the fallback buys alias coverage at the full distractor-error cost.
4. **FR4 is not a speed gain.** It re-cut the compute→dispatch boundary (geometric/selection ride internal, only global-joint optimization routes external) — a criterion for future planner work, not a realized throughput change.

## Honest scope

- The gain is specific to the **KnnStore retrieval-override** path (knowledge editing / fact injection), not base generation.
- DISTRACTOR-safety = "override did not fire." A distractor's residual genuinely collides (near-rank-1) — legacy's 0/20 is the real near-rank-1 failure, not a contrived one.
- One model, capital relation, single-token facts. The two-tier distractor cost would be lower on a workload that is genuinely alias-only (no open distractors), which is the intended use of the opt-in clause.
- Latency is the override-step only (the dominating forward is identical across modes). Verified < legacy here is within noise; the point is both are µs.

## Bottom line

The FR work delivers a **decisive correctness gain** — it converts KNN fact-injection from "corrupts 100% of unrelated queries at 20 facts" to "100% distractor-safe" — at **no throughput cost**. It is not a tok/s gain and was never going to be; the override is a sidecar. The measurement also sharpened the FR2 guidance: two-tier fallback is an opt-in alias resolver, not a safe default.

**Artifacts:** `chris-experiments/larql_probes/examples/fleet_routing/fr_routing_gain.rs`.
