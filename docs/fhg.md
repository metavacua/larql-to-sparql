# FHG v0.1 — the Fourier heuristic graph: recovering a model's approximate graph as a spectrum

**Programme:** FHG-1 → FHG-6. FHG-1 and FHG-2 are the decisive pair; FHG-3/4/5 are conditional on them; FHG-6 exists so the programme cannot degenerate into a catalogue of failures.
**Scope:** behavioural, model-agnostic by construction. Falsifiable on Gemma 3 12B with the MLX instrument already built (`chris-experiments/fhg/`). No weights work, no kernel work, no larql code required until FHG-5.
**Status:** v0.1 — mirrored into the experiments registry as programme `fhg`. FHG-1 built and running 2026-08-05.
**Date:** 2026-08-05
**Predecessor:** the RSL authority/promotion branch, EXP-24 → EXP-43. FHG starts exactly where EXP-42 stopped.

---

## 1. Thesis

For a query that requires `A --r1--> B --r2--> C`, the model may not retrieve two discrete edges. Its answer may be reconstructed from several overlapping signals — the actual `r1` edge, the actual `r2` edge, a direct `A↔C` association, entity/value co-occurrence, relation-shaped wording, a recent mention of `B`, a competing path, and the query's phrasing — whose interference happens to land on `C`.

The claim is testable without opening the model. Treat the answer as a function of candidate heuristics, make each heuristic a binary factor, and compute the **Boolean Fourier spectrum** of that function. A stable graph operation and a bag of heuristics make different, pre-registerable predictions about where the spectral energy sits.

This is not a claim that internal neurons are Fourier modes. It is a claim that the model's graph-like *behaviour* admits a decomposition into heuristics and their interactions — and, if it does, that exact external edges can be shown to **simplify** that spectrum rather than merely replace it.

## 2. Why this is the branch that follows EXP-42

Three results from the predecessor branch set this up, and each maps onto a design decision here rather than onto a motivation paragraph.

**EXP-42 failed its two-hop pre-screen at every rung of a 64K → 16K ladder.** Its own verdict: *"the model cannot reliably traverse a two-edge graph it can read, at any length tested."* That search never reached a regime where the operation was clean, so it could not separate "promotion does not work" from "the underlying walk does not work." **FHG-1's ladder therefore runs upward**, starting at zero filler — the records and the query, nothing else — and climbing. Where the walk breaks is a measurement, not a blocked run.

**EXP-41 was voided by exactly two of this programme's factors.** A wrong-relation record (`x4`) that provably did *not* commit still moved the downstream answer in 3 of 4 cells, and an intervening conversational turn (a recency effect, `x6`) flipped the baseline walk. Those were reported as confounds that made the contrast unavailable. **In FHG-1 they are the signal**, with the placebo factor supplying the floor that says whether they are large.

**Greedy top-1 has repeatedly passed while the trajectory diverged** (R14, gate–claim congruence). FHG-1 scores a **teacher-forced margin at the answer's first token**, not a greedy read. Greedy is recorded, but only to support the one discrete claim the spectrum cannot make: *correct answers produced without the complete intended path*.

## 3. The ladder

| ID | Question | Instrument | Kill criterion | Depends on |
|---|---|---|---|---|
| **FHG-1** | Is the answer a stable operation over the required edges, or a weighted mixture of heuristics? | 7 binary factors → 128 arms, Walsh–Hadamard transform of the teacher-forced answer margin, 4 items × 4 phrasings. Placebo factor calibrates the null | Only `{x1,x2}` subsets carry energy; every nuisance coefficient sits at or below the placebo floor; spectra agree across phrasings → the bag-of-heuristics framing is **wrong** and the programme stops here | — |
| **FHG-2** | Does an exact intermediate restore a broken path? | Break the first edge; inject an exact value for `B` (never `C`) at the intermediate boundary. Arms: intact / ablated / prosthesis / corrupt prosthesis / sham / final-answer control | Correct `B` fails to restore, **or** a corrupt `B` is ignored rather than traversed → there is no usable interface at the choke point, and FHG-3/5 are dead | FHG-1 (needs a regime where the walk runs) |
| **FHG-3** | Does exactness *collapse* the heuristic spectrum? | Re-run FHG-1's matrix with `A→B` replaced by an exact external resolution of `B` | Nuisance energy and paraphrase disagreement do not fall → exactness buys accuracy only, not stability, and the "reduces interference downstream" story is dead | FHG-1 + FHG-2 |
| **FHG-1b** | Is it context *volume* or record-to-query *distance* that kills the walk? | The Stage A ladder with filler placed BETWEEN the records and the query, plus a 64K clean-room cell to test the extrapolation in §4g | Displacement degrades no faster than volume → EXP-42's collapse is unexplained by either axis and the corpus itself is the confound | FHG-1 Stage A (done) |
| **FHG-1c** | Are nonce labels inert addresses? | **G-NONCE**: a *population* of generated nonce graphs (not 4 hand-picked), then paired role swaps and permutations within a graph | Label identity contributes no more than the placebo → §4g's item effect was 4-sample noise | FHG-1 |
| **FHG-4** | Where does approximation become unsafe? | Competitor-pressure curves: 0/1/2/4/8 competitors, typed by topology (shared entity / relation / value / destination / lexical twin / paraphrase) | Degradation is flat or type-independent → pressure is not a usable planner input | FHG-1 |
| **FHG-5** | Selective exactness as an execution policy | Mixed graph, some edges approximation-permitted and some exactness-required. Compare all-neural / exact-at-choke-points / all-external / neural-with-post-hoc-verification | Exact-at-choke-points does not beat post-hoc verification → verify-after is sufficient and the planner is unnecessary complexity | FHG-2 + FHG-3 |
| **FHG-6** | Is approximation ever the *capability*? | Remove a semantically inferable edge (`Robin is a bird`, `birds usually fly`). Compare explicit-only traversal / neural approximation / neural proposal + exact verification | Neural approximation adds nothing over explicit traversal → the thesis is one-sided and the programme is only a failure catalogue | FHG-1 |

Minimal sequence: **FHG-1 → FHG-2 → FHG-3 → FHG-5**. FHG-4 and FHG-6 are cheap and can run in any gap.

## 4. FHG-1 in detail

### 4a. The factor matrix

Seven binary factors, each in a **fixed slot at a fixed position**. Only slot *content* changes between arms; every arm is token-identical in length, enforced at runtime by padding each slot to `max(len(present), len(absent))`. Without that, a factor's coefficient would absorb a position shift in every downstream slot.

| Factor | Present | Absent (matched control) |
|---|---|---|
| `x1` `edge_ab` | the real `A → B` edge | same relation wording, unbound entity + value |
| `x2` `edge_bc` | the real `B → C` edge | same relation wording, unbound sector + neutral region |
| `x3` `shortcut_ac` | a direct `A → C` locative claim | a contentless fact about `A` |
| `x4` `wrong_relation` | `A` paired with `B` under the **wrong** relation | same wrong relation, unbound pair |
| `x5` `competing_chain` | `A → D → E` via a *different* relation from the same `A` | same wording, unbound pair |
| `x6` `recent_b` | a recent bare mention of `B`, closest slot to the query | same sentence, unbound sector |
| `x7` `placebo` | an entity swap on an out-of-graph relation | the other out-of-graph entity |

Slot order is `x5, x3, x1, x4, x2, x7, x6` — the query follows slot 7, so `recent_b` sits closest to it. Recency is a positional claim and has to be positioned to be tested.

**Why the competing chain uses a different relation.** `A → D` under the *same* relation as `A → B` is a flat contradiction, and a contradiction is a different experiment. Using a second relation from the same source entity gives genuine shared-entity interference without asking the model to arbitrate inconsistent facts.

**Why absent states keep the wording.** Every absent state preserves the sentence template and breaks only the *binding*. If absent states were unrelated filler, `x1`'s coefficient would conflate "the edge exists" with "relation-shaped wording appears at all" — and relation-shaped wording is itself one of the heuristics under test. It has to sit in the mean, not in a factor.

**Why there is a 7th factor.** `x7` is the calibrated null floor, and the reason the matrix is 128 arms rather than 64. It is the *same kind of edit* as `x4` — swap which entity a record names — applied to content that touches no part of the graph. "Nuisance leakage is large" is only a claim relative to that floor. This is the standing lesson of R11/`uninformed control`: without a matched null, a nonzero coefficient is a number, not a finding.

### 4b. The score

Read at the answer's **first token**, teacher-forced:

```
margin_full = logP(C) − max over all other vocabulary
margin_set  = logP(C) − max over the other planted destinations
```

Each destination contributes the max logprob over its first-token variants (bare and space-prefixed) — a rule fixed in advance and applied identically to every candidate, so it cannot favour one. A gate asserts the three candidate destinations have **disjoint** first-token variant sets; if they collide the margin cannot separate them and the run aborts.

`margin_full` is the honest statistic (it can catch the model preferring a token outside the destination set entirely); `margin_set` answers "did it pick the right node *among nodes*." Both are recorded. This follows R12: name the metric's space. The space is the model's first-token distribution over the answer, not a similarity in some derived basis.

### 4b-bis. The decisive statistic is the interaction, not the class

Class energy is not enough, and quoting `required_only` as the headline would be a mistake. That class sums both main effects *with* the interaction, and each main effect has an innocent explanation that is not composition:

- `edge_bc` can raise `logP(C)` simply because the destination word is physically present in context;
- `edge_ab` can raise it by activating the intermediate;
- `shortcut_ac` can produce the right answer with no two-hop walk at all.

Only the **`x1 × x2` interaction** says the pair supplies something neither edge supplies alone:

```
Δcomposition = f(1,1) − f(1,0) − f(0,1) + f(0,0)
```

which is exactly `4·f̂({x1,x2})` when averaged over all 32 settings of the other factors. Two forms are reported: the **averaged** interaction (robust — it survives whatever else is in context) and the **clean-cell** interaction (the same 2×2 with every nuisance and the placebo off — interpretable, unaveraged, and what "the graph works" means in isolation).

Validated against a synthetic construction with a planted `4.0·x1·x2` term: recovered `+4.00`, sd `0.01`, positive in 8/8 spectra, with the main effects exactly at their analytic values.

**The phase-transition test, and why the raw pattern is not enough.** A large interaction can also arise from curvature in a response where every condition already succeeds. The sign of `margin_full` settles that, because `margin_full > 0` holds exactly when `C` is the argmax. Composition requires the clean cells to read `00 fail, 10 fail, 01 fail, 11 pass` — the pair crossing a boundary neither edge crosses.

But that bit is a knife edge. On synthetic *association-without-composition* data the pattern still read `...P` in 7 of 8 spectra purely because one cell landed on −0.0. So a cell counts as **resolved** only when `|margin| > placebo_max` — if an irrelevant matched edit could move it across zero, its side of zero is not evidence. `phase_transition_resolved` is the claim-bearing flag; `phase_transition` alone is not. With that rule the synthetic worlds separate 4/8 vs 0/8, while `required_only` energy was **0.86 in both** — which is precisely why the class fraction cannot be the headline.

### 4c. The transform and its calibration

Walsh–Hadamard over the 128 arms with `z_i = 2·x_i − 1`, so `f̂(S) = E_x[f(x)·Π_{i∈S} z_i]`, `f̂(∅)` is the mean, and Parseval holds. Energy is partitioned over the 127 non-constant subsets:

- `required_only` — `S ⊆ {x1,x2}` (3 subsets)
- `nuisance_only` — `S ⊆ {x3..x6}`, disjoint from the required pair (15)
- `mixed` — touches both (45)
- `placebo_touched` — contains `x7` (64)

**Those cardinalities are why raw class energy is not comparable across classes.** `placebo_touched` has 64 chances to accumulate energy against `nuisance_only`'s 15; comparing sums would flatter the null and understate leakage. Two per-subset calibrations are reported instead:

- **mean energy per subset** within each class;
- **`placebo_max_abs_coefficient`** — the largest `|f̂(S)|` any placebo-touched subset reaches. An irrelevant matched edit cannot do better than that, so a non-placebo coefficient above it is doing work the null cannot account for. The headline count is *how many of the 63 real subsets clear the placebo ceiling*.

Plus: interaction-order profile (energy by `|S|`), and **paraphrase spectral cosine** over the 127-vector between all phrasing pairs.

### 4d. Block invariance, and why the null must be dimension-matched

The strongest available FHG-1 result is not "required energy is largest" but a **contrast**: the required block holds still across items and phrasings while the nuisance block moves. A whole-vector cosine cannot show that — the required block dominates it and hides any nuisance drift — so the blocks are compared separately and the claim is the *gap* between them.

That comparison is only valid against a **dimension-matched null**. The blocks have very different sizes (required 3 coefficients, nuisance 15, placebo-touched 64), and cosine reproducibility under noise depends on dimensionality. Measured on this instrument, the 3-dimensional block's placebo null is `[−0.21, +0.28]` while the 15-dimensional block's is `[−0.11, +0.10]` — nearly 3× wider. Comparing raw cross-block cosines would therefore be systematically unfair to the required block. Each block instead gets its own null, built by drawing 2000 random subsets of the placebo coefficients **at that block's own k**, over the identical pairings, reported as a 95% interval.

**Two independent gates, because a low nuisance cosine is ambiguous on its own** — it can mean real coefficients whose signs change with context, or coefficients so small that noise sets their direction. Energy separates those. The pre-registered decision tree:

```
nuisance energy inside matched placebo energy null
    → no measurable nuisance contribution

energy clears the floor, cosine inside matched cosine null
    → measurable but unreproducible; no mixture-structure claim

energy clears, matched-null < nuisance cosine < required cosine
    → STRUCTURED, CONTEXT-DEPENDENT MIXTURE  (the claim)

nuisance cosine ≈ required cosine
    → one fixed heuristic recipe, not a varying mixture
```

### 4d. The screen is informative, not blocking

Stage A checks, at the reference cell (`x1=x2=1`, everything else off) and at each rung of the upward ladder: the direct read of `B`, the direct read of `C` from `B`, and the composed walk under all four phrasings.

In EXP-41/42 the reference walk was a *precondition* for an attribution claim, so its failure made the contrast unavailable. Here it is one cell of the matrix and the score is a continuous margin, so a failing screen means **low required-path energy** — a measurement, not a void. Stage B runs regardless; the screen conditions how it is read. This is the one place FHG deliberately departs from the predecessor branch's gate discipline, and the reason is that the object under test changed: EXP-42 was testing an intervention *on* the walk, FHG-1 is testing the walk itself.

### 4e. Gates

| Gate | Rule |
|---|---|
| token distinctness | the three candidate destinations must have disjoint first-token variant sets |
| query leakage | no phrasing may contain any planted value — enforced by code, not by reading them |
| slot width | any slot whose content exceeds its width aborts the run rather than shifting positions |
| determinism | the reference cell is rescored after the full 2048-cell sweep; a nonzero margin drift means cache restore is not bit-stable and small coefficients are unresolved |

## 4f. What FHG-1 can and cannot claim

Because `x2=0` replaces the correct `B→C` binding with a **template-matched neutral binding** rather than deleting the sentence, FHG-1 establishes:

> **binding-specific relational composition** — the two-hop result depends jointly on the two *compatible bindings*, not merely on both relation templates being present.

That is a better-controlled claim than deletion would give, because relation-shaped wording is held constant. It is also strictly weaker than necessity. **Necessity and external replacement belong to FHG-2**, and FHG-1 must not be written up as having shown them.

Separately, the Stage A ladder places filler *before* the records, so the records stay adjacent to the query at every rung. It therefore establishes tolerance to **preceding context volume**, not to **record-to-query distance** — and EXP-42 varied the latter, planting its spans at 15–54% through the corpus. The two axes are not interchangeable. The displacement ladder (filler *between* records and query) is **FHG-1b**, deliberately deferred until after FHG-2 rather than run as a side-branch.

## 4g. Stage A result (2026-08-05/06) — the baseline the thesis needs

64 reference cells: 4 filler levels × 4 items × 4 phrasings, every one with `x1=x2=1` and every nuisance and the placebo **off**. All 64 pass, all decisively.

**The composed operation survives long preceding context.** Mean margin 15.48 (filler 0) → 14.36 (1K) → 13.28 (4K) → 11.89 (16K); the weakest single cell at 16K is still +8.38 nats. The defensible statement is: *long preceding context does not inherently destroy this two-hop nonce composition*. Over the positive levels the decline is **consistent with an approximately logarithmic volume penalty** — `margin = 20.58 − 0.62·log₂(filler)` fitted on 1K/4K/16K only, since filler 0 cannot sit on a log axis. That fit predicts **~10.7 nats at 64K**, which is an extrapolation from three points and therefore a *falsifier to run* (FHG-1b), not a result. Either way the diagnosis of EXP-42 sharpens: context volume alone produces modest degradation, so its collapse must be carried by distance plus intervening semantic records.

**Three factors move a supposedly exact lookup, with every designed nuisance off.** The design is a complete balanced 4×4×4 with one observation per cell, so *every* component — main effects and all interactions — is estimable and the decomposition is exact:

| component | df | SS | MS | η² |
|---|---|---|---|---|
| item (which nonce names) | 3 | 121.77 | 40.59 | 0.294 |
| filler (0 → 16K) | 3 | 112.83 | 37.61 | 0.272 |
| phrasing | 3 | 88.87 | 29.62 | 0.214 |
| item × phrasing | 9 | 31.95 | 3.55 | 0.077 |
| filler × phrasing | 9 | 21.80 | 2.42 | 0.053 |
| filler × item | 9 | 18.80 | 2.09 | 0.045 |
| filler × item × phrasing | 27 | 18.63 | 0.69 | 0.045 |
| **total** | **63** | **414.64** | | **1.000** |

Item, filler and phrasing account descriptively for 29.4%, 27.2% and 21.4% of total variation; the remaining 22.0% belongs exactly to their two- and three-way interactions. **No independent error term exists**, because there is one observation per cell — so no F-tests are reported and none should be. (An earlier draft called the 54-df interaction pool a "residual" and tested against it. That was wrong: 9+9+9+27 = 54 is the complete, fully estimable interaction structure, not error, and pooled interactions are not a justified error distribution. What is genuinely unavailable is pure replication.)

The interactions are not debris — they answer design questions directly:

- **filler × item is small (η² 0.045)** → the context-volume tax is broadly uniform across bindings rather than name-specific.
- **item × phrasing is the largest interaction (η² 0.077, MS 3.55)** → there is no universally optimal phrasing across bindings, which constrains query compilation: a compiler cannot pick one canonical rendering and expect it to be best everywhere.
- **the three-way term is the weakest per degree of freedom (MS 0.69 vs 2.09–3.55)** → execution is *not* strongly reconstructed from the address × context × form conjunction. This cuts against the strongest bag-of-heuristics reading at Stage A level and is reported as such.

**Phrasing behaves like an execution plan.** The two phrasings that stage the intermediate explicitly — *"the code identifies a sector. in which region is that sector?"* — score 14.95 and 14.69; the two that nest both hops in one clause score 11.99 and 13.38. A ~3-nat spread. The architectural reading is that a decomposed query makes the intermediate node operationally explicit, where a nested one requires the model to construct and hold it while parsing a compound request. **This is directly relevant to LQL**: it suggests query compilation could improve reliability by transforming a nested operation into a staged traversal plan *before* any exact prosthesis is inserted. Caveat: two phrasings per apparent class, with clause count, length and wording all confounded. Stage B's coefficient-vector comparison across phrasings is far stronger evidence than four means.

**Nonce labels are not inert.** A 3.78-nat range across items of identical structure (`duskmoor` 15.96 vs `thessaly` 12.18) where only the entity, sector and region strings differ. Candidate causes — tokenisation length and segmentation, token frequency, residual morphology-like associations, similarity to other labels, addressing quality, position shifts from token width — are not separated here. Four hand-selected graphs cannot establish label invariance, which is why **FHG-1c (G-NONCE)** is a population experiment with paired role swaps, not a bigger hand-picked set. If it holds, the design rule follows: *identifiers supplied to neural graph operations have measurable address quality*, and LARQL should not rely on the raw textual identifier as the sole address — typed handles or internal IDs, with natural-language references resolved approximately on top.

**What this establishes for the thesis.** A clean relational graph operation is robustly present, *and* its confidence is continuously modulated by context volume, query form and binding identity — with all designed nuisance factors disabled. That is exactly the precondition the Fourier account needs: a stable underlying operation plus continuous modulation by incidental cues. Stage B asks whether that modulation decomposes.

The strongest available Stage B result is therefore not "required energy is largest". It is:

```
x1*x2 composition interaction   stable across items and phrasings
nuisance coefficient vector     changes with item and phrasing
placebo                         near zero
```

which would support: **the graph operation is invariant, while the heuristic mixture used to realise it is context-dependent.**

## 5. What settles it

**The theory loses if:** only the required edges affect the answer; nuisance coefficients sit at or below the placebo floor; spectra remain stable across phrasings; exact intermediate injection does not rescue a broken path; corrupt intermediate values are ignored rather than traversed.

**The theory gains strong support if:** correct answers arise from multiple substitute heuristic combinations; nuisance coefficients clear the placebo ceiling; equivalent phrasings use measurably different spectra; an exact intermediate restores downstream computation; and exact insertion *reduces* paraphrase and interference sensitivity without replacing the rest of the neural operation.

The most interesting single number is **nuisance leakage**: how much of the answer function is carried by variables that an exact graph query would ignore.

### 5a. The three worlds FHG-1 can separate

| | required interaction | nuisance vs placebo | single-edge cells | reading |
|---|---|---|---|---|
| **Symbolic-like graph** | strong, resolved phase transition | at or below placebo | fail | composes explicit bindings with little interference |
| **Fourier heuristic graph** | strong, resolved | several factors clear placebo; spectrum shifts with phrasing | fail | a genuine composed operation whose *expression* is modulated by approximate cues |
| **Association without composition** | weak or inconsistent | large shortcut / destination-presence effects | often succeed | apparent traversal is reconstructable from simpler associations |

The design distinguishes these from the *structure of the response function*, not from output accuracy — which is the whole point, since all three can produce the same greedy answer.

### 5b. The full conjunction required for the strong claim

No single number licenses the headline. All of the following must hold together:

```
composition     x1*x2 positive across every item x phrasing
                one-edge clean cells fail, both-edge cell passes
                interaction far above the matched placebo floor
                phase transition RESOLVED (no cell within placebo_max of zero)

required block  high energy
                high cross-item and cross-phrasing invariance vs its OWN k=3 null

nuisance block  energy above its matched placebo energy null
                reproducibility above its matched k=15 cosine null
                invariance materially BELOW the required block

placebo         no systematic behavioural effect

robustness      same qualitative result under raw margin, competitor-cancelled
                margin (immune to the neutral plant), and rank transform
```

Only then: **the relational operation is stable, but the incidental mechanisms modulating its execution are structured and context-dependent.**

## 6. Standing rules this programme inherits

- **R12** — name the metric's space and the search guarantee (§4b).
- **R14** — gate–claim congruence: a gate licenses claims only over the object it tests. FHG-1's screen licenses claims about the *reference cell*, not about the matrix.
- **Uninformed control** — a matched null is mandatory before any leakage claim (§4a, `x7`).
- **Read raw arms before the verdict line** — every derived verdict here (`passed`, `top1_label`, the class fractions) inherits its classifier's assumptions; the per-cell margins are kept in the JSON so they can be re-derived.
- **Fixture too small is an absent test** — 4 items × 4 phrasings, not one item. Item variation is *binding* variation with the template family frozen, because EXP-39/40 found which value you assert changes whether a write lands.

## 7. What this programme is not

It is not a claim about internal representations, and it must not be written up as one. The spectrum is over *behaviour as a function of injected context*. A heuristic-shaped spectrum says the model's answer is reconstructable from overlapping partial cues; it says nothing about whether any neuron computes one.

It is also not a replacement for the promotion branch. EXP-24 → EXP-43 asked whether an external write can take authority. FHG asks whether the thing the write would participate in is a graph at all.
