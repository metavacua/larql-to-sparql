# The authority control plane (EXP-26..38)

**Status:** layer-mechanism branch CLOSED 2026-08-05 by EXP-36 and EXP-37.
Registry `rsl-exp26`..`rsl-exp38`.
Instruments: `chris-experiments/rsl/exp26_*.py` .. `exp38_*.py`.
Model: `google/gemma-3-12b-it`, MLX bf16, 64K context, 8 global attention
layers at `[5, 11, 17, 23, 29, 35, 41, 47]` (48 layers, 5 sliding : 1 global).

## Why this exists

The programme opened as a graph-walk question: can a promoted chain preserve
identity so the model traverses several edges? The answer arrived quickly and
negatively, and it exposed a missing operation rather than a dead end:

```
external graph resolver returns 5824
promoted closure contains 5824
old source in context remains authoritative
model answers 7431
```

An externally resolved closure is worthless if a live source keeps control. So
EXP-29 onward is **the KV/attention control plane that graph walks require**,
not graph walking. The intended engine is:

```
RESOLVE       external graph traversal
CLOSE         construct authenticated closure
DEAUTHORIZE   stop conflicting live context dominating   <-- EXP-29..36 live here
PROMOTE       expose the resolved closure
EXECUTE       model performs the local semantic operation
VERIFY        check the result
```

## Headline

> **Query-time authority control and long-context correctness: yes, and cheaply,
> on one fact. Physical KV saving: no — the source's own K/V is never written,
> measured bit-exactly. Transferable certificate: no — the regime the whole
> branch characterises does not exist for the second fact tested.**

Read EXP-36 and EXP-37 before anything below them. Between them they close the
branch: EXP-37 settles authority-vs-deletion by direct measurement, and EXP-36
removes the precondition every layer result depends on.

## What is established

### Composition — EXP-26

Beyond depth 1 the contrast is **unmeasurable, not refuted**. The *reference*
walk (full access, nothing promoted or hidden) flips destination when one
unrelated 11-token span moves from 66% to 10% of the prefix, with the true and
decoy links at identical positions:

| hop | layout A | layout B |
|---|---|---|
| 1 | `7431` ✓ | `7431` ✓ |
| 2 | `North` ✓ | `South` ✗ |
| 3 | `Ostran` ✗ | `Corvin` ✗ |

Layout B's `South → Corvin` is a *correct* traversal of the wrong chain: the
relational operator works, the addressing does not. A 2K capability pre-screen
admitted hops 1–3 and **did not transfer to 64K**.

Single-edge promotion is solid and replicated four times: walk 0.0024/0.0012
bits, replay 0.0027/0.0023, payload and trajectory inside the 0.05 tolerance,
pad-only failing and a corrupt record steering the answer. **Stepwise walk ≡
closure replay at depth 1** — the stepwise machinery buys nothing there.

### Binding — EXP-27, EXP-28

`PROMOTE_AND_BIND` is **refuted**. Count-matched at four promoted records with
matched vocabulary and identical retirement, the only factor deciding the
outcome is whether a sub-threshold operand record was promoted:

| path | operand | flips | maxKL |
|---|---|---|---|
| broken | no | no | 0.0362 |
| coherent | no | no | 0.0531 |
| **broken** | **yes** | **YES** | **2.9071** |
| coherent | yes | yes | 7.3971 |
| coherent, *other entity* | its own | yes | 7.2189 |

Path coherence modulates magnitude, never the decision. An off-entity chain
carrying its own operand flips it too — the model reads a promoted terminal
value without checking the chain delivering it is the queried one.

Promotion cost has three tiers (EXP-28/29), all on the same question:

```
supply into a gap                   ~1 record
override an internal disposition    ~4 records + ~40 content tokens
override a LIVE readable source     never, at any size tested   <-- FALSE, EXP-36/38
```

Gross injected token count was constant by construction throughout (RoPE
parity), so none of this is token mass.

**The third tier does not survive EXP-36, and EXP-38 finishes it off.** On a
second fact from the same prefill, the same style of terse record overrides a
live, readable source unaided; and fact 1 itself — the fact "never" was measured
on — falls to a natural-language corroborator supplied at promotion time.

### Authority — EXP-29..EXP-34

**A live source cannot be overridden.** The same byte-identical record:

```
' Meridian.code=5824. '
  source retired  -> answers 5824, maxKL 14.7914
  source live     -> answers 7431, maxKL  0.0046
```

The record is **inert**, not outvoted — every override arm sits inside the
supply arms' own KL spread. Retirement is what makes promotion work.

**It is a per-layer gate, not a count** (EXP-31). Retiring global layer 29
*alone* flips it at 6.3298 bits with the source readable in all seven other
global layers. No other singleton comes close (0.0008–0.2312). Count-matched
triples behave oppositely: `[17,29,41]` flips at 7.5827, `[5,23,47]` does
nothing at 0.0013. Layer 29 is **sufficient but not necessary** — `[35,41,47]`
and `[5,11,17,23]` also flip without it, so at least three routes exist.

**It is acquired at query time, in a 2–4 token window at the chat turn
boundary** (EXP-33/34):

| route | phases needed | query window |
|---|---|---|
| `[29]` | Q only | suffix4 = idx 20–23 (`<end_of_turn> \n <start_of_turn> model`) |
| `[35,41,47]` | Q and A | suffix2 = idx 22–23; idx 22 alone suffices |

Injection-phase hiding is **irrelevant in both routes** — old and new records
may coexist at write time. Every prefix window fails at every length tested,
including `prefix16`, which hides the source across the entire semantic content
including the queried entity. **Authority is not selected at entity
resolution.** Against pure recency: masking idx 22 alone flips it while idx 23
— *closer* to generation — does not.

The two routes differ on three independent axes (layers, phases, window width),
which is strong evidence they are distinct mechanisms rather than two cuts
through one circuit.

**The sham control is load-bearing** (EXP-33): layer 29 retired with *no*
record present leaves the true answer intact in all eight phase configurations.
So layer 29 does not store the answer — it governs whether a *competing* record
can acquire authority.

### Persistence — EXP-35

Both arms teacher-forced to identical text, so only the mask differs:

| arm | mask during query A | B_plain | B_norecord |
|---|---|---|---|
| scoped | yes (idx 20–23, L29) | **promoted 5824** | source 7431 |
| unscoped | no | source 7431 | source 7431 |
| baseline (forced 7431) | no | source 7431 | source 7431 |

`unscoped` and `baseline` agree despite carrying *different* forced answers, so
the echo hypothesis is dead: only the four-token mask in the previous turn
moves the second turn. **The decision outlives the scope.**

But `B_norecord` in the scoped arm answers `7431` correctly — the source is
fully readable and still load-bearing. **This is direct evidence the span must
be retained.** What persists is standing *relative to the replacement*, not
availability.

Likely mechanism, and it is mundane: the query-A boundary tokens were encoded
while layer 29 was masked and their K/V persist; query B attends to *those*.
The source was never mutated. **EXP-37 confirms this by measurement.**

### Transfer — EXP-36. The precondition fails

The design had three outcomes. The run produced a fourth it did not enumerate:
**the contrast is unavailable**, the same class as EXP-26's composition result.

`k0` — nothing hidden, replacement injected — *already* answers the promoted
value. So all 12 layer arms and all 10 windows read `promoted` for free,
including `none` with zero tokens masked. The script computed
`solo = [all eight global layers]` and printed "same shape, different layer";
that verdict is wrong and the registry record corrects it. Nothing in the table
attributes to a layer. Data quality is not at issue — every continuation is
literally `South`, `ref` is `North`, and the sham arm (hide the source, no
record) returns `North`, so the source is intact and readable.

What it establishes is larger than what it set out to test: **override-a-live-
source is fact-dependent.** Layer 29, the two routes, the acquisition/
maintenance split and the turn-boundary window were all measured inside a
regime — contested source that wins by default — that does not exist for the
second fact. There is no demonstrated transferable certificate content, and
**the 256-subset lattice must not be run**: its stated precondition (open
question 1) has failed, and it would characterise one cell.

The missing gate was "is the record inert at `k0`?", the precondition the whole
attribution rests on. It has been added to the instrument, which now aborts.

### Localisation — EXP-37. Where the sticky state lives

Three results, strongest first.

**The source was never mutated — measured, not inferred.** The scoped and
unscoped caches after query A are bit-identical over the source span at every
global layer:

| rows | L5 | L11 | L17 | L23 | L29 | L35 | L41 | L47 |
|---|---|---|---|---|---|---|---|---|
| source | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| placebo (idx 16-19) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| boundary (idx 20-23) | 0 | 0 | 0 | 0 | 0 | K7.55 | K4.45 | K8.06 |
| model turn | 0 | 0 | 0 | 0 | 0 | K2.12 | K1.62 | K8.00 |

The source is 55.6K tokens from the query, far outside the 1024 sliding window,
so global layers are the *only* path to it — coverage is **complete** for this
claim even though it is partial for the carrier claim. With EXP-35's
`B_norecord`, the no-deletion result is now established twice by independent
means. **No amount of scoping frees the source.**

**The footprint matches the architecture exactly.** A layer-29 mask changes
that layer's *output*, not its K/V *write*, so only layers >29 can differ at
the boundary. Observed divergence is exactly `{35,41,47}` — EXP-33's second
route, reached by a completely independent instrument.

**The commit rides in V, not K, and is redundantly encoded.** Transplanting
rows between arms at fixed positions — nothing added or removed, RoPE
untouched, both row sets genuine model outputs at those exact positions:

| transplant | result |
|---|---|
| `identity_ctl` scoped ← scoped | promoted (round-trip neutral) |
| `suf_placebo` pre-mask rows | source (expected no-op) |
| **`suf_bnd`** unscoped ← scoped boundary | **promoted — sufficient** |
| **`suf_turn`** unscoped ← scoped model turn | **promoted — sufficient** |
| `nec_bnd`, `nec_turn` | promoted — **neither is necessary** |
| `suf_bnd_35` / `_41` / `_47` alone | source — no singleton suffices |
| `suf_bnd_Konly` | source |
| **`suf_bnd_Vonly`** | **promoted** |

Two independently sufficient carriers, neither necessary: the state propagates
forward from the boundary into the turn, so either region reconstitutes it. The
sufficient object is 4 positions x 3 layers x 8 kv-heads x 256 dims, V only —
**~49 KB**, small enough to store and move.

**The eviction test is uninformative and forms no part of this.** Evicting the
never-masked placebo window (`' one'`, `' word'`, `' only'`, `'.'`) flips the
*unscoped* arm to promoted on its own — removing any small window from global
attention de-authorises the source, which is precisely the intervention under
study. `turn_all8` moves both arms. Only the transplants carry the result,
exactly as the pre-registered coverage note anticipated: a negative eviction is
inconclusive, a positive transplant is not.

## Three primitives that must stay separate

```
AuthorityScope    temporarily decides which competing source wins
AuthorityCommit   continuation state materialised while the scope is open
Retire            certifies source material may be removed
```

EXP-35 demonstrates the first two interacting and provides evidence **against**
the third; EXP-37 closes it by measurement. *"The decision persists"* is not
*"the old evidence is dead."* Conflating them is the main hazard in this branch.

`AuthorityCommit` is real but **"small addressable record" is only half right.**
It is small (~49 KB) and transplantable, and it does survive the scope closing.
It is *not* removable by patching one site — substituting either carrier region
alone leaves the effect intact. It behaves as the leading edge of a contaminated
continuation, not as a discrete object with a defined lifetime. Any runtime that
wants to *revoke* a commit cannot do it by rewriting the boundary rows.

The certificate sketch below is retained only as a record of what the branch was
reaching for. **It has no measured content**: EXP-36 shows the regime it
describes is not general, so `excluded_layers` cannot be looked up, and nothing
establishes it can be measured per span at tolerable cost either.

```rust
AuthorityScope {              // NOT VALIDATED — see EXP-36
    source_span,
    excluded_layers,          // [29] or [35,41,47] on ONE fact
    acquisition_window,
    maintenance_window,
}
```

### Overridability — EXP-38. The tier is refuted; the predictor is not found

The corpus contained a natural experiment: `plant` drops decoy sentences for
`i > 0`, so `decoy_src[0]` is the one decoy value never planted in a
type-correct role — and fact 1 is the one resistant fact.

| arm | asserts | winner | continuation |
|---|---|---|---|
| `f1_ref` / `f1_k0` | `5824` | source | `7431` |
| `f2_ref` | `South` | source | `North` |
| **`f2_k0`** | `South` | **promoted** | `South` |
| `f3_ref` | `Corvin` | source | `Ilex` |
| **`f3_k0`** | `Corvin` | **promoted** | `Corvin` |
| `f4_ref` / `f4_k0` | `3` | promoted | `3` — **reference fails, unusable** |
| **`f1_ally`** | `5824` | **promoted** | `5824` |
| **`f1_ally_only`** | `5824` | **promoted** | `5824` |
| **`f2_novel`** | `Kelvar` | **promoted** | `Kelvar` |
| **`f2_novel_ally`** | `Kelvar` | **promoted** | `Kelvar` |
| `f2_hide_ally` | `South` | promoted | uninformative by design |

**EXP-29's third tier is refuted, not qualified, and twice over.** Facts 2 and 3
are overridden by a terse `key=value` record with nothing hidden and no
intervention at all. And fact 1 — *the exact fact EXP-29 measured and declared
un-overridable across k=0..3* — is overridden by a natural-language
corroborator, which `f1_ally_only` shows works **alone**, with no terse record
present.

**The ally is not necessary, so it is not the mechanism.** On fact 2 a record
asserting `Kelvar` — verified absent from the entire 64K corpus, with no
in-context support of any kind — wins outright. The ally fails necessity on the
very fact it was introduced to explain.

**It still correlates perfectly with the `k0` regime** across the three usable
facts, at depths 15%–41% and across both digit and word answers, so neither
depth nor answer type explains the split. Given necessity failed, that
correlation **must not be reported on its own**: whatever separates the regimes
is confounded with ally-presence here and is not isolated by this design.

**The result the graph-walk track needed:** a caller *can* manufacture authority
at promotion time. `f2_novel` wins with no corroborator at all, so the material
need not have been present when the context was built. For this class,
`RESOLVE → PROMOTE → EXECUTE` needs no `DEAUTHORIZE`, no certificate and no
masked forwards.

**Form confound, unresolved, and now the live question.** What rescues fact 1
differs from its terse record in *form* (a natural-language sentence vs
`' key=value. '`) as well as in provenance. The two missing cells are fact 1
with a **terse** corpus-absent record, and facts 2/3 with **natural-language**
records.

Fact 4's failure is itself a hint: its reference arm, with *no record injected*,
already answers `3`, because the corpus plants `' Corvin holds clearance level
3. '` and the question asks about Ilex. A type-correct in-context distractor
beats the true source with zero promotion involved — but it was not designed as
an arm and cannot carry weight.

## Open questions, in priority order

1. **Form or fact? (EXP-39.)** The only thing between here and a runtime rule.
   One prefill, 2×2: {fact 1, fact 2} × {terse, natural-language}, every arm
   asserting a **corpus-absent** value so ally-presence is held at zero and
   cannot confound the form contrast. If fact 1 flips under the verbose form
   only and fact 2 flips under both, the rule is "promote verbose assertions,
   not `key=value` records" — free. If fact 1 resists both with a corpus-absent
   value, provenance rather than form rescued it, the caller *cannot*
   manufacture authority for that class, and `DEAUTHORIZE` comes back for
   exactly the facts EXP-29 characterised. Add a fact-1 arm at higher record
   counts to check the tier boundary moved rather than vanished.
2. **Stickiness across a different operation.** EXP-35's query B repeats the
   same question. Nothing measured speaks to a *different* operation.
3. **Boundary vs recency.** The boundary tokens *are* the final tokens. Insert
   semantically empty padding between question content and boundary, then test
   a matched-distance off-boundary window.
4. **V-not-K** (EXP-37) is a direct hit on `rsl-4`'s premise (channel-specific
   liveness) and should be carried there.
5. ~~**The lattice.**~~ **Cancelled.** It was explicitly conditional on (1)
   transferring, and EXP-36 shows it does not. Running it would enumerate 256
   subsets of a single cell.

## Stopping condition

**Met, and the branch is stopped.** Authority-vs-deletion is settled (authority
only, twice, independently). The certificate axis is dead rather than
satisfied — EXP-36 removed its precondition. Do not continue into open-ended
layer archaeology; there is no reproducible certificate to find at this level.
The live work is question 1 above and then an end-to-end graph-walk workload.

**EXP-38 does not reopen it.** It shifts the engine question from *"which layers
must stop reading?"* to *"what does a promoted record have to look like?"* — a
promotion-side question with no attention intervention in it. If EXP-39 lands on
the favourable branch, `DEAUTHORIZE` leaves the critical path entirely for the
overridable class, and the layer machinery is retained only as a fallback for
whatever class turns out to need it.

## Programme split

**Graph/context** — authority scopes, boundary commits, transfer across
operations, external resolution, end-to-end closure execution. The plausible
shape:

```
external step -> authority commit -> local neural operation -> external step
```

rather than promoting a whole chain and hoping identity composes.

**KV compression** — separate track: canonical reconstruction, proven-dead
alternatives, durable closure sufficiency, cold replay. Authority control may
eventually assist compaction; it is not compaction.

## Reproduction notes

- Every experiment shares one corpus. `exp27`..`exp36` import `exp26`
  unchanged for corpus, exclusion path and scoring, so span offsets are
  byte-identical across runs.
- `k0`/no-retirement returns 0.0046 and `early_3` returns 0.3024 across six
  independent prefills — the frontier is prefill-stable, not noise.
- **Cost driver:** each masked forward pays a span-exclusion gather, ~4.3 GB of
  K/V copied at 64K. Keep the mask live over as few forwards as possible.
- **Do not swap the gather for an additive `-inf` mask.** Measured 1.21x faster
  but 0.19 max|logit| apart — the same order as a known precision confound,
  against a 0.05-bit tolerance. `exp26` keeps both paths and prints the parity
  number every run (gate G0).
- `tok.eos_token_id` is `<eos>`=1 but the model terminates turns with
  `<end_of_turn>`=106. Key termination on `tok.eos_token_ids`.
- Corpus filler must contain no answer token, including inflections. The
  inherited EXP-25 filler said "northern" while `North` was an answer.
- **Classify answers by whole first word, not by a two-character prefix.**
  `exp36` scored `North` vs `South` with `no`/`so`, which would have silently
  miscounted a continuation starting "Not" or "Sorry". It happened to be clean
  here; `E.answer_word` exists for this and `exp38` uses it.
- **Score an arm against what its record actually asserts, not against the value
  the fact's decoy happens to be.** `exp38`'s first run classified every arm
  against `f["decoy"]`, so its two novel-value arms — which assert a corpus-absent
  word and *got that word back* — were filed as `other` rather than as wins for the
  record. The derived flag `ally_suppliable` therefore came out `False` and the
  printed verdict claimed a caller "cannot manufacture authority this way", which
  is the exact opposite of what the continuations show. Same failure class as
  EXP-36's saturated table: **a derived boolean inherits every assumption in the
  classifier that fed it.** `run()` now takes an explicit `asserted=` value and the
  arm table prints what each record claims alongside the winner.
- **Gate the precondition, not just the reference.** `exp36` checked that the
  fact answers correctly with no record, but not that the record is *inert*
  with nothing hidden — and the latter is what every attribution arm assumes.
- **A placebo must be read on the arm that can move.** `exp37`'s placebo check
  originally asked whether the *scoped* arm stayed promoted; it was promoted
  already, so the check was vacuous. The informative arm was the unscoped one,
  and it failed.
- Thermal: prefill went 293s -> 711s across consecutive runs. Never reuse these
  harnesses for latency work without a cold machine.
