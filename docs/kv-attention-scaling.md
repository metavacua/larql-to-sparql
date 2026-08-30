# KV attention scaling — measurement schema

Schema and pre-registered predictions for the KV ladder's context-scaling
runs (KV-B1 clean e2e, KV-B2, KV-C, and the eventual `larql-kv` engine
tournament).

The purpose of fixing this before the runs is narrow: **a "tok/s vs context
length" curve is not interpretable on an architecture with mixed attention
layer classes**, and every model on the ladder has them. Recording the
decomposition at capture time costs nothing; reconstructing it afterwards is
impossible.

## Four things that must not be conflated

The ladder moves along four independent axes. Collapsing any two of them
produces a graph that looks like a result and isn't.

| Axis | What it is | Who changes it |
| --- | --- | --- |
| **Context** | Prompt + generated tokens the model is holding | The workload |
| **Effective span** | Rows the attention kernel actually reduces over, per layer | The architecture, via the sliding window |
| **Execution geometry** | How that reduction is scheduled — serial, seqpar, multi-TG | KV-B1, KV-B2 |
| **Representation** | Width and encoding of the KV rows themselves | KV-C, the `larql-kv` engines |

B1 and B2 move execution geometry only and are exactness-preserving. KV-C
moves representation and is not. The scoring rules differ accordingly —
see [Gate classes](#gate-classes).

## Why context ≠ span

`ops::kv_cache::attention_span(t, window_size)` bounds the span at the
layer's sliding window; only layers with `window_size == 0` grow with
context. So within one forward pass, at a given context depth, different
layers sit in different span tiers — and `ops::kv_seqpar::slices_for`
selects a *different threadgroup width per layer class* as a result.

A single "tok/s at 8K" number is therefore a weighted blend over layer
classes, with weights set by the architecture. Two consequences:

1. **The B1/B2 gain asymptotes.** As depth grows, sliding layers stop
   contributing new span; the incremental benefit comes only from the full
   layers. The curve tends to a ceiling set by the full-attention share of
   attention cost, not to an unbounded gain.
2. **Cross-model curves are not comparable** unless the layer-class mix is
   reported alongside, which is exactly what a cross-engine tournament
   needs them to be.

## Row schema

One row per (config, depth, layer class). Emit the class rows; the
per-token aggregate is derived, not measured separately.

| Field | Notes |
| --- | --- |
| `model` | Registry id |
| `context_length` | Prompt + generated, tokens |
| `generated_depth` | Tokens generated so far — span driver during decode |
| `layer_class` | `sliding` \| `full` |
| `layers_in_class` | Count, from the loaded config — not assumed |
| `window_size` | 0 for full layers |
| `head_dim` | Per class; drives the seqpar policy |
| `effective_span` | `attention_span(t, window_size)` for the class |
| `attention_kernel` | `serial` \| `seqpar` \| `seqpar_long` \| B2 variant |
| `seqpar_slices` | 0 when refused; records what the policy actually chose |
| `latency_per_token_ms` | |
| `tok_s` | |
| `gpu_occupancy_pre` | Mean and max, from the exclusivity check |

`seqpar_slices` is worth carrying explicitly: a run where the policy
refused (0) and a run where it chose 1 slice-equivalent look identical in
the aggregate and mean different things.

## Pre-registered prediction

Written before the deep-context blocks are re-run, so the shape is not
fitted afterwards.

For a model with `S` sliding layers at window `W` and `F` full layers:

```text
sliding layer span → min(depth, W)      saturates at W
full layer span    → depth              grows without bound
```

Therefore:

- Below `depth = W`, every layer is in the same regime and the B1 gain
  tracks the short-context number.
- Past `depth = W`, sliding layers pin to their tier while full layers
  climb through the tiers. The measured gain rises, then flattens toward
  the full-layer share.
- **The gain must not keep climbing linearly with context.** If it does,
  the layer-class mix or the window is not what the config says, and the
  decomposition is wrong before the performance claim is.

For gpt-oss-20b the expectation is an even split — 12 sliding at window
128, 12 full — which predicts a rise between depth 128 and roughly the
long tier, then a plateau. **Confirm this against the loaded config before
the run rather than assuming it**: dump `attn_spec.sliding_window` per
layer and record `layers_in_class` from what comes back. The prediction is
only falsifiable if the mix is read, not assumed.

## Gate classes

Two kinds of change move along this ladder and they take different
authorities. Do not borrow one for the other.

**Execution-order changes (KV-B1, KV-B2).** KV stays f32, slice and tile
partials accumulate in f32, the merge runs in f32. The only permitted
difference from the serial kernel is reassociation of the weighted-V sum.
Gated by `max_rel < 1e-4` against the serial f32 kernel, with negative
controls calibrated at ~1e-1 and bitwise determinism across repeats — see
`crates/larql-compute-metal/tests/test_kernel_kv_attention_seqpar.rs` and
its two siblings. That tolerance separates reassociation from *defect*. It
cannot separate reassociation from *approximation*, which is why no
representation-width change is allowed inside B2.

**Representation changes (KV-C, `larql-kv` engines).** An f16 KV cache
exceeds 1e-4 by construction, so the gates above cannot be reused and must
not be loosened to accommodate it. KV-C needs its own oracle: an f32-KV
reference scored in predictive units on the deployment path, with a
quality budget fixed **before** any latency is measured. Otherwise an
approximation win folds silently into the B2 number.

## Gate step: dump the layer specs first

Before any timed block, dump per layer and confirm the class mix against
the prediction above:

```text
layer   window_size   head_dim   q_heads   kv_heads
```

This is a gate, not a convenience. The predicted curve is only falsifiable
if the mix is read from the checkpoint; if it is assumed, a matching curve
confirms nothing and a mismatched one is unattributable.

## Blocks and arms

Measure the **default**, not `LARQL_KV_SEQPAR=auto`. The explicit-auto path
already has evidence; the shipping question is whether an unset env fires
the policy at head_dim 64, which is a different code path through
`kv_seqpar_from_env` → `SeqparRequest::Unset` → `default_is_auto`.

**This requires the enablement change applied first.**
`SEQPAR_DEFAULT_ON_HEAD_DIMS` ships empty, so on a clean checkout `Unset`
resolves exactly like `Off` and an `off / default` A/B would compare the
serial kernel against itself — a null result indistinguishable from a
negative one.

### Candidate tree

Build the candidate as "the enablement commit minus its evidence", all
gates green — not as a tree with two tests deliberately red:

1. `SEQPAR_DEFAULT_ON_HEAD_DIMS = &[64]`
2. Flip the two expectation tests
   (`nothing_defaults_on_until_the_gate_closes`,
   `the_default_list_is_empty_pending_the_gate`) to assert the enabled
   state.
3. `cargo test -p larql-compute-metal --lib ops::kv_seqpar` — green.
4. **Rebuild `--release`.** A stale `target/release/larql` is exactly how
   this experiment yields a beautifully reproducible null.

The candidate marker is then the shape of the diff, not a red suite:

```text
git diff:
  exactly 1 policy constant
  exactly 2 expectation changes
```

`unset_resolves_through_the_default_list` and
`explicit_off_stays_off_on_every_geometry` must stay unchanged and green
across both commits. If either needs editing, the change is to policy
*mechanics*, not to policy *evidence*, and does not belong in the
enablement commit.

### Invoking

Make the absence of the env var explicit rather than trusting shell state:

```bash
env -u LARQL_KV_SEQPAR LARQL_GPU_ROUTE=1 ./target/release/larql bench ...
```

The `off` arm is `LARQL_KV_SEQPAR=off` on the same binary.

Both of those, the bracket ordering, and the two arithmetic preconditions
are wrapped by [`scripts/kv-ladder-bracket.sh`](../scripts/kv-ladder-bracket.sh),
which is the supported way to run a block:

```bash
./scripts/kv-ladder-bracket.sh B bench/prompts/gpt-oss-kv-ladder-b.txt 300
```

The three ladder prompts are pinned in the repo, because context length is
the largest single term in the measurement and a prompt that lives only in
a session scratchpad makes the block unreproducible:

```text
A  bench/prompts/gpt-oss-kv-ladder-a.txt    ~36 tokens     147 chars
B  bench/prompts/gpt-oss-kv-ladder-b.txt    ~574 tokens   2296 chars
C  bench/prompts/gpt-oss-kv-ladder-c.txt   ~2024 tokens   8096 chars
```

Each block is `off / default / off` with a ~300 s rest before **every**
arm, not just before the block — see [The bracket](#the-bracket):

```text
PRE   handshake with every peer session; warm the page cache; warm to plateau
A     short context     off / default / off
B     ~574-token        off / default / off
C     ~2K regime        off / default / off        (blocked — see #229)
POST  confirm the foreign-process sampler read zero for the whole block
```

## Pre-registered decision rules

Fixed before the run so the outcome cannot be reinterpreted after it.

| Outcome | Action |
| --- | --- |
| Short + medium + deep all positive | Default head_dim 64 ON; close B1 |
| Short positive, deep flat | Still default ON, provided no meaningful regression and the short win is robust |
| Short positive, deep negative | Add a span restriction to the default; explicit `auto` stays available outside it |
| Any correctness or integration failure | Fix before defaulting |

A deep-context negative does not invalidate B1 — `slices_for` already takes
`span`, so the result would simply say the *default region* is narrower
than the capability.

## KV-B2 contract

Recorded here so the B2 kernel is specified before it is written, and so
the f32 invariant above has something concrete to attach to.

```text
grid.x = query head
grid.y = sequence tile
```

Each tile computes local `m` (max), `l` (exp sum), and `o[head_dim]`
(weighted-V partial). The structural difference from B1: sequence tiles no
longer share `tg_scores`, so **the softmax state must travel with the
partial output** rather than being resolved before accumulation. Merge is
online-softmax:

```text
tile state = { m, l, o[head_dim] }

merge(A, B):
    m = max(A.m, B.m)
    α = exp(A.m - m)
    β = exp(B.m - m)
    l = α*A.l + β*B.l
    o = α*A.o + β*B.o

final = o / l
```

All f32, per the invariant above. `examples/bench_attention_span` can then
answer whether B2 buys anything past B1's 1024-thread ceiling before any
e2e run is spent on it.

## Run hygiene

- Exclusive GPU, established by **handshake with every peer session**, not
  by a utilisation probe. See [Exclusivity is a handshake, not a
  probe](#exclusivity-is-a-handshake-not-a-probe) — a flat baseline arm
  does not establish that the candidate arm was uncontended, and neither
  does an idle reading taken before the block.
- **Bracket each candidate; do not interleave.** The unit is
  `off / candidate / off`, and the candidate counts only if the two
  brackets agree. Report the raw per-arm readings, not just the derived
  delta. See [The bracket](#the-bracket) for why `off/auto/off/auto` is
  not equivalent.
- Steady state: warmup 16, n 256. Short runs read slow.
- Run the control triple (`unset` / `off` / `auto`) once per session.
  `unset ≈ auto ≪ off` is what distinguishes "the default fired and helped"
  from "the default never fired"; an `off`-vs-`default` pair alone cannot
  tell those apart, and reads the second as a negative result.

### The bracket

Interleaving `off/auto/off/auto` has a positional bias that took a session
to see: the candidate always occupies positions 2 and 4, so under any
cumulative drift the candidate is systematically *later in the run* than
the baseline. The arm mean then carries the drift, and nothing in the
block reveals it.

The unit is therefore a bracket, and the block is warmed to plateau first
so that the bracket measures the candidate rather than the warm-up:

```text
warm to plateau
    ↓
off          ← opening bracket
    ↓
candidate
    ↓
off          ← closing bracket
```

**A candidate measurement is valid only when it sits between two mutually
agreeing controls.** If the brackets agree the candidate counts; if they
do not, the block invalidates itself and is not averaged. This is
stronger than interleaving because it validates the machine state
*locally around each candidate observation* rather than assuming drift is
monotonic and therefore cancellable.

#### What the bracket does not validate

The bracket validates the **machine**, not the **arm**. Two failure modes
pass it cleanly, and both produce a block that looks textbook:

- **Steady contention.** A peer holding the GPU for the whole block moves
  all three arms together, so the brackets agree and the block certifies a
  contended candidate. Only the handshake catches this.
- **A truncated candidate.** An arm that stops early still prints a row,
  with a mean over however few steps it took. If the arm that stopped is
  the *candidate*, the two baselines still agree with each other — the
  bracket check passes — and the block certifies a mean over a handful of
  steps as though it were a 256-step reading.

The second is quantifiable here rather than hypothetical. At the block-C
prompt the stop happens about 1 run in 7 ([#229](#229-blocks-block-c)), so
a three-arm C block has a **~37%** chance of containing a truncated arm and
a **~14%** chance that the truncated arm is the candidate — the case that
manufactures a protocol-blessed number from nothing. This is why
precondition 7 is listed separately from precondition 6 and is not implied
by it: **precondition 6 is an argument about the machine, and it cannot see
inside an arm.**

The instrument hid this. `bracket_rested.sh`, the script that produced the
readings recorded below, formatted the bench row as
`prefill / mean / tok-s` — dropping both `n_steps` and the note column — so
it was structurally incapable of displaying a truncated arm. The
replacement, [`scripts/kv-ladder-bracket.sh`](../scripts/kv-ladder-bracket.sh),
gates on the step count directly and voids the block when any arm falls
short of `n-1`, instead of relying on a human to read a note. Note the
general shape, which is the same one as the fixture rules above: *a gate
that is not printed is not a gate.*

#### The bracket is also counterbalanced, which `A/B/A/B` is not

There are two independent problems with interleaving and they fail in
opposite regimes. The note further down — that interleaving does not
protect against *non-monotonic* movement — is one. This is the other, and
it bites precisely when the drift is clean and monotonic, i.e. when the
block looks most trustworthy:

```text
A/B/A/B     A at positions 1,3 → mean 2      B at 2,4 → mean 3
            one full position of separation, always favouring A

A/B/A       A at positions 1,3 → mean 2      B at position 2 → mean 2
A/B/B/A     A at positions 1,4 → mean 2.5    B at 2,3 → mean 2.5
```

Under a drift that is a function of run position, `A/B/A/B` hands the
later arm a systematic penalty equal to the slope times the position gap.
The bracket is `A/B/A`, which is already balanced; `A/B/B/A` is the
four-run generalisation when two candidate readings are wanted.

This is not hypothetical here. The measured ramp on an exclusive, warmed
machine is ~0.44 ms/run over the first five runs, ~2.5% of an 18 ms base
per position — so an `off/default/off/default` block carries a ~2.5% bias
in whichever direction the ordering assigns it. Every 2026-08-15 block was
run that way, with `default` as the later arm, which means those blocks
**understate** the candidate and their deltas are floors rather than point
estimates.

**Counterbalance; do not numerically correct.** Fitting the ramp and
subtracting it is tempting and wrong on this machine — extending the same
series to eight runs gives slope 0.616 ms/run at R² 0.67, with residuals
reaching ±1.4 ms (±7%) and run 8 landing 8.9% *below* run 6. A drift that
badly modelled cannot be subtracted from an 11-25% effect without
inventing most of the correction. Counterbalancing needs no model of the
drift at all, which is exactly why it is the right instrument.

One caveat on what "position" means. In these series every run does the
same nominal GPU work, so run position and accumulated GPU work are
collinear and the data cannot separate them. Where the arms do *different*
amounts of work per run — a faster candidate does less — the two diverge,
and it is accumulated work that should be balanced, not the ordinal.

Block C on 2026-08-16 is the worked example:

```text
off      19.75
default  18.04
off      40.49
```

Without the closing bracket, `19.75 → 18.04` reads as a plausible ~9%
improvement. With it, the machine is visibly 105% different by the end of
the block and the candidate reading means nothing. An `off/default/off/
default` block would have averaged straight through that.

### Exclusivity is a handshake, not a probe

Three separate instruments failed to detect a peer session that was
holding the GPU:

- `pmset -g therm` — silent throughout.
- `ioreg` GPU Device Utilization, sampled *between* runs — 3-5%.
- `ps` — no foreign process at the sampled instants.

Sampling utilisation *during* a run does not fix this, because it cannot
separate your own dispatch from a peer's. Sample for foreign **processes**
during the run instead; that distinction is observable. And note the
watcher hazard: `pgrep -f larql` matches the watcher's own command line,
so use `pgrep -x` (executable name) and exclude the captured PID.

Even that is only a detector. The sufficient check is an explicit
handshake with every peer session before the block, because contention can
begin *after* a pre-block gate passes — measured 2026-08-16, where a peer
session's own idle gate passed at 01:27 and its GPU work started
afterwards. `ListAgents` shows peers that `ps` does not.

**State the handshake as "no load", not "no GPU".** Compilation counts.
[#198](https://github.com/chrishayuk/larql/issues/198) records the
synthetic prefill test going from 0/13 failures on an idle machine to 2/5
under a concurrent `cargo build` of an unrelated crate — CPU and linker
load, no GPU work at all. This was violated in the very session that
introduced the handshake: a 23-second debug build of `larql-cli` ran
against a peer's live block and voided all five of its completed runs.
"Code-only work" is a category error here; the thing being protected is
the machine, not the device.

**And the handshake has to cover cache state, not just load.** Whoever
runs second is cold no matter how idle the machine is, because the first
session's warm evicted theirs — the two working sets do not both fit. The
same 2026-08-16 exchange shows it from both sides: this session's runs
degraded across a series while the peer's *improved* across theirs
(16.827 → 17.458 tok/s, monotonic) as its working set re-warmed after
being evicted. Two sessions alternating on one box therefore produce drift
in whichever direction the last warm went, and each side reads it as a
property of its own workload. Warm explicitly after taking the machine,
and treat any series that begins immediately after a handover as suspect
until the plateau check (precondition 3) passes.

**The critical property: asymmetric contention and the power-cap
clip-the-faster-arm effect produce the same signature**, and no cheap gate
separates them. Both truncate the faster arm preferentially, both leave
the slower arm nearly flat, and both survive an idle check. Calibration
from a peer session on Muse-Glimmer-30B, same round, minutes apart, with
every cheap gate reading clean (GPU 5/5/3/3/3%, no competing process in
`ps`, 37 GB free, no swap, AC 94W):

```text
metal-lowered (NVFP4)   200 ms/token    5.003 tok/s
metal-lowered-mxfp4      72 ms/token   13.898 tok/s
```

2.8× apart on arms that differ only in kernel geometry.

The same session's three-round interleaved block, recorded here in full
because it is the best available teaching case — Muse-Glimmer-30B, commit
`1f172ed1`, `steady (last half)` = mean of the last 64 of 128 decode
tokens, 5-minute rest before the block and none between rounds:

```text
r1  metal-lowered        56 ms   17.723 tok/s
r1  metal-lowered-mxfp4  60 ms   16.791 tok/s     NVFP4 +5.5%
r2  metal-lowered        59 ms   17.062 tok/s
r2  metal-lowered-mxfp4  62 ms   16.136 tok/s     NVFP4 +5.7%
r3  metal-lowered        61 ms   16.465 tok/s
r3  metal-lowered-mxfp4  60 ms   16.575 tok/s     MXFP4 +0.7%  ← inversion

NVFP4  17.723 → 17.062 → 16.465   −7.1%, monotonic
MXFP4  16.791 → 16.136 → 16.575   −1.3%, non-monotonic
```

The block is void for **three independent reasons**, and interleaving
survived all three: a peer session's asymmetric load, a power-state
transition mid-block (r1 on battery at 85%, AC 94W and charging by r2),
and whatever sustained-load effect is also present. No gate that session
was running detected any of them.

**A gate stated in percent cannot be evaluated against a field quantised
coarser than the gate.** That harness prints ms as `{:.0}`, so at 56 ms one
count is 1.8% — a 1% agreement precondition is not decidable from the ms
column at all, and only the `{:.3}` tok/s column has the resolution to
test it. Check the printed precision of the field a precondition is
written against before trusting the precondition. `larql bench`'s
`ms/tok` is `{:.2}`, so at ~13.7 ms one count is 0.07% and the 1% gate is
decidable there — but that is a property of this harness, not a general
one.

### The instrument is not stable by default

Measured 2026-08-15 on the M3 Max, gpt-oss-20b. Both effects below are
larger than the ~11% being measured, and neither announces itself:
`pmset -g therm` stays silent, the GPU reads idle between runs, and memory
stays healthy throughout.

**Cold working set.** `larql bench` is one process per arm, and this model
is ~27.5 GB across the two containers (18 GB q4k spine + 9.5 GB routed
MXFP4). With the page cache evicted, each arm re-faults tens of GB and
consecutive runs warm progressively — an `off` arm read 17.40, 16.58,
16.05, 15.76, 15.51 ms across five identical runs. Warm explicitly before
timing anything:

```bash
find <spine> <routed> -type f -size +10M -exec cat {} + > /dev/null
```

**Sustained-load degradation, recoverable.** Past roughly 5–10 consecutive
runs the machine collapses — the same `off` arm walked 14.17 → 36.22 ms
over ten runs, a 2.5× loss, monotonic. A 5-minute rest fully restores it:
13.70, 13.72, 13.75, 13.75, 13.75, a 0.36% spread reproducing the
pre-degradation value. So this is pacing, not a leak, and the fix is rest
between blocks rather than a faster harness.

**The root cause is NOT established** — corrected 2026-08-16. This was
previously recorded here as "a 67W adapter negotiating 65W", stated as
the root cause. That attribution does not survive:

The collapse **reproduces on the 94W adapter** (96W rated). Ten
consecutive runs walked 15.4 s → 49.1 s, 3.2×, monotonic with no knee.

A peer session was live for part of that window, so the series was
reconstructed per-run against the peer's GPU windows before drawing any
conclusion. Per-run start times come from the printed generation times
plus a 5 s/run process-open estimate; the estimate is not free-floating —
it reproduces the observed 10:07:49 finish to within a second, which is
the check that makes the reconstruction usable:

```text
run    gen s   peer holding GPU?
  1     15.4   contended  2s of 15s
  2     16.0   CLEAN
  3     16.7   CLEAN
  4     18.5   CLEAN
  5     19.5   CLEAN
  6     21.4   CLEAN
  7     23.6   CLEAN
  8     28.8   contended 17s of 29s
  9     32.4   contended 32s of 32s
 10     49.1   CLEAN   ← the worst run of the series
```

**Six consecutive clean runs ramp +47%, and the single worst run is
clean.** Contention cannot produce that shape, so the degradation is real
and is not an artifact of the peer. It also reproduces on a
verified-exclusive, explicitly warmed machine: 17.50 / 17.56 / 18.12 /
17.66 / 19.64 ms per token over five runs, +12% by run 5, with a
foreign-process sampler reading zero for every sampled second and 92.9 GB
free.

What is established is the **operational shape**, not a mechanism:
degradation tracks accumulated GPU work, appears after roughly 30-40
seconds of heavy load, and a 300-second rest recovers it while 120 seconds
does not. Two candidate mechanisms remain open and are not separated by
anything measured here — the adapter is ruled out, but a co-resident peer
*working set* (as opposed to peer GPU work) is not, since page-cache
pressure persists whether or not the peer is dispatching. Rest-pacing is
justified by the shape alone. Do not name the mechanism in a commit
message.

Note also that a peer session pausing between its own blocks reproduces
"recovery on rest" exactly, so recovery-on-rest is **not** the falsifier
separating this from contention that it was once claimed to be. The
per-run overlay above is what separates them.

Adapter wattage is still worth checking, since an under-powered adapter is
a real and separate hazard — `pmset -g batt` says "AC Power" and is blind
to wattage:

```bash
ioreg -rn AppleSmartBattery | grep -o '"Watts"=[0-9]*'
system_profiler SPPowerDataType | grep -iE "wattage|Name:"
```

**The power cap clips the faster arm.** This is the part that matters for
A/B design: `default` does more work per unit time than `off`, so it draws
more instantaneous power and is the arm the cap truncates — variably, and
only downward. Same-arm spread therefore tracks how fast the arm is, not
how noisy the machine is, and it grows with work per token (0.18% at short
context, 1.91% at ~574, 5.43% at ~2024 while every `off` arm stayed under
0.4%). A power-limited A/B is biased *against* the faster arm, so it
understates rather than inflates — but it cannot produce a point estimate.

### Validity preconditions

A block counts only if all nine hold. Check them **before** computing any
delta:

1. **Adapter delivers the machine's rated wattage.** Necessary, not
   sufficient — see the correction above.
2. **Page cache explicitly warmed**, with the `find … -exec cat` above.
   High free memory is the symptom, not the reassurance.
3. **Warm to plateau, then confirm it**: repeat one arm until consecutive
   readings agree within ~1%. A still-improving or still-degrading series
   means the block is not yet runnable.
4. **Rest ~300 s before each block**, and keep the block to its three arms.
   120 s is known to be insufficient.
5. **Exclusivity by handshake with every peer session**, plus a during-run
   foreign-process sampler that reads zero for the whole block. A
   pre-block idle check is not sufficient.
6. **The two brackets agree within ~1%.** If they do not, the block is
   void — do **not** average across the disagreement.
7. **Every arm completed its full decode budget.** An arm that stopped
   early still prints a row, with a mean over however few steps it took.
   On this model a ~2K-token prompt does this intermittently — see
   [#229 blocks block C](#229-blocks-block-c).
8. **Every arm's `fp=` matches the one canonical trajectory for the
   prompt.** Completion is necessary and *not* sufficient: the ~0.5 ms/token
   broken state completes its full step budget and a step-count gate
   certifies it. Only the fingerprint says the arm executed the model.

   Note the word *one*. The sampler is greedy and constructs no RNG (see
   [A process sometimes begins broken](#a-process-sometimes-begins-broken--and-it-is-probably-229)),
   so a prompt has exactly one correct trajectory. Multiple
   plausible-latency fingerprints are therefore **evidence of corruption,
   not a set of acceptable outputs**, and this precondition must not be
   weakened into "belongs to the observed set" — that would launder the
   defect into the gate and certify whichever corrupt trajectory happened
   to appear during calibration.

   While the fault is under investigation it is fine to *report*
   fingerprint multiplicity without picking a winner. But before `fp=` is
   used as a performance gate, the canonical trajectory has to be
   established independently — from the fixed path once it is reproducible,
   or from an oracle that does not share the suspect code — never by
   majority vote among runs of the thing being tested.
9. **Every arm clears a physical plausibility floor.** ~2000 tok/s on a
   20B MoE is not a fast run, it is an absent one. This is deliberately
   redundant with 8: an impossible speed must invalidate an arm
   immediately, before anyone has to understand *why* it was impossible.
   Derive the floor from bytes-per-token against measured bandwidth, not
   from a tuned threshold.

Preconditions 8 and 9 exist because the failure they catch is invisible to
every other check. A broken-state arm has an exclusive GPU, sits between
agreeing brackets, and finishes all of its steps — so 5, 6 and 7 all pass
while the arm never ran the model.

Precondition 6 is the load-bearing one for *arithmetic*, and precondition
5 is the load-bearing one for *causality*. A mean computed over a
disagreeing pair will happily produce a plausible number; and a block
whose brackets happen to agree can still be measuring a peer session,
since contention that is steady across the block is invisible to the
bracket. The bracket catches drift, not steady contention. Only the
handshake catches steady contention.

### #229 blocks block C

Block C cannot be run until [#229](https://github.com/chrishayuk/larql/issues/229)
is fixed. At the ~2024-token prompt, `larql bench` intermittently reports
`no decode steps completed` — generation stops inside the first `warmup`
tokens. Measured 2026-08-16 on a verified-quiet machine: **1 failure in 7**
at `--warmup 16 -n 128`. The ~574-token block-B prompt has never done it.

Two properties make this worse than a lost run:

- **It is silent.** A run that stops early still prints a row. Only the
  note column distinguishes it, and only if you read the note.
- **The note is ambiguous at the protocol's own settings.** `measured_n =
  decode_ms.len() − warmup`, so at `--warmup 16` every stop inside the
  first 16 tokens collapses into the same message as a first-token EOS.
  Run the detector at `--warmup 0` instead: then `no decode steps` means
  token 1 exactly, and `early stop @k/N` gives the exact stop index. Note
  that a healthy run reads `early stop @(N−1)/N` — the first token comes
  from prefill's logits and is not counted in `decode_ms`, so being one
  short is normal and is **not** an early stop.

What is ruled out so far, by measurement rather than by reading:

- **Not the prompt alone.** `larql run` on the C prompt, greedy, nine
  consecutive runs: 9/9 byte-identical output. One prefill per process,
  fresh buffer pool.
- **Not the three obvious read-before-write hazards on pooled scratch.**
  `BufferCache::output` hands back recycled buffers without zeroing them,
  which makes any partial write a live hazard — but the lm_head's
  `norm_out` padding tail is explicitly zeroed (`decode/head.rs`),
  `f32_topk_partial` writes all `K_TOPK` slots unconditionally including
  in its ragged last threadgroup, and prefill's QKV/O slack is zeroed on
  every allocation (`full_pipeline/buffers.rs`).
- **Not stale KV across the bench's two `generate` calls.**
  `copy_persisted_tail` resets `current_len` and `abs_position` on every
  prefill.

The open lead is that `bench` prefills twice per process (a pre-warm
`generate_n(weights, 1)`, then the timed call) while `run` prefills once,
which is the difference between the deterministic case and the
intermittent one.

### A process sometimes begins broken — and it is probably #229

Found 2026-08-16. The wrong turn is recorded alongside the result, because
the wrong turn is the reusable part.

Chasing #229 with `--repeat` produced a clean 2x2 in (repeat count,
pooled-buffer contents) which said that `--repeat` corrupted state:
`--repeat 1` read a plausible 20.22 ms/token, `--repeat 3` and `8` read an
impossible ~0.5 ms/token, and zeroing the pool restored plausible numbers.
Every cell was consistent. It was wrong. **Four samples of an intermittent fault reproduced a perfect
interaction by chance** (the 2/4 rate in that control was itself small-N —
see [the tree control](#the-fault-is-on-main-and-predates-seqpar) for a
rate that means something), and the
2x2's neatness was the thing that made it convincing.

The fresh-process control falsified it — same command, same prompt,
`--repeat 1`, four separate processes:

```text
run 1   0.50 ms/tok   7 steps   fp=72f97535…   broken
run 2  17.69 ms/tok   7 steps   fp=8a2986f7…   sane latency
run 3  18.65 ms/tok   7 steps   fp=e9a47f8b…   sane latency, third output
run 4   0.64 ms/tok   7 steps   fp=72f97535…   broken
```

So the fault exists before `--repeat` enters the picture. It only ever
looked repeat-shaped because every repeat inherits its process's state,
which is why all eight repeats carried a single fingerprint.

Call it a **process-scoped startup fault**, and no more than that. What is
established is that the state differs *between* fresh processes and is
already present by the first timed generation. Where it is selected —
process start, backend construction, buffer acquisition, first Metal
encoding, or first use of a stale allocation — is exactly what is not yet
known, and "selected at process initialisation" quietly names a boundary
the evidence does not reach. Provenance has to identify it.

The sampler is not the explanation, which was checked rather than assumed:
`SamplingConfig::greedy()` sets `temperature 0.0` with no `top_k`/`top_p`,
`is_greedy()` is therefore true, and `Sampler::new` builds **no RNG at
all** on that path, so the `seed: None` field is inert. Argmax over a
differing top-K `hits` vector is the only way these outputs can diverge,
which puts the divergence upstream in the GPU compute.

**There are two distinct defects here, and they must be scored apart:**

- **The catastrophic short-circuit.** ~0.5 ms/token is ~2000 tok/s, which
  a 20B MoE cannot do; the arm is not executing the model. Its `fp=` moves
  with it, so the generated text differs too.
- **Output nondeterminism among the sane-latency runs.** Runs 2 and 3 both
  have entirely plausible latency and still disagree. Greedy sampling on an
  identical prompt must give identical output, so this is a defect and not
  noise. Counting the zeroed arm's fourth fingerprint, one prompt has
  produced four distinct outputs at these settings.

`LARQL_ZERO_POOLED_BUFFERS` looked decisive and is not, and the reason is
the same correction: that run was `--repeat 8`, so **it is one zeroed
process draw, not eight independent successes.** It shows that a single
zeroed process did not enter the catastrophic state across its repeats. It
does not establish that zeroing lowers the fresh-process fault rate, which
is a different claim and needs fresh processes to test. It did still show
two fingerprints within that one process, which is the main reason to
think these are two faults rather than one.

This strongly supports a common parent with [#229](#229-blocks-block-c)
rather than a separate bug: same prompt, same intermittency, and "no decode
steps" is the natural variant where a broken state crosses EOS on token 1
instead of producing wrong-but-running output. Both faces were then
reproduced on pristine `main` in the same sweep — see the tree control
below. It is not *proved* to be one mechanism: by the ledger's own rule the
faces stay separate until a single intervention removes them together.

The standing lead is the pool. `preallocate_kv_cache_per_layer_with_capacity`
replaces the cache outright, so `current_len` and `abs_position` *are*
fresh per call — position state is not the survivor. One layer down,
`LayerKVCache::new` draws `k_cache` and `v_cache` from
`BufferCache::output`, the **size-keyed recycling pool**. No `Drop` returns
KV buffers to it — the only recyclers are `ScratchGuard`, four
`moe_dispatch/dense` buffers and four `decode/head` buffers — so a KV cache
cannot receive an earlier *KV* buffer. What is not yet proven, and what
must be proven rather than read, is whether it can receive an unrelated
allocation of the same byte size. That needs provenance instrumentation on
`BufferCache::output(size)` (allocation id, size, new-vs-reused, previous
owner), not further inspection.

**`--repeat` is refused**, but not as a cause. A fault selected at process
init is exactly what repeats cannot sample: N repeats are N correlated
observations of one draw, multiplying a single outcome into a row count
that reads like replication.

**Consequences for the ladder.** No recorded reading used `--repeat`, and
the control at block B's exact configuration (`--warmup 16 -n 256`, no
repeat) completed all 255 steps at sane latency with a plausible
fingerprint, so block B landed in the good state. But every arm of every
block is an independent draw on this fault, which is a far sharper problem
than truncation: **a broken arm completes its full step budget**, so
precondition 7 certifies it. That is what forces preconditions 8 and 9
below.

### The fault is on main and predates seqpar

Run 2026-08-16 with `scripts/kv-seqpar-tree-control.sh`: one **fresh
process per row**, C prompt, `--warmup 0 -n 8`, arms interleaved, a
watchdog on every process. Integrity only — the machine was on battery for
the whole sweep, so no ms/token column below is a timing.

```text
tree                         draws  healthy  EARLY-STOP  CATASTROPHIC  HANG   sane-latency fps
branch, LARQL_KV_SEQPAR=off    8       7         1            0         0     8a29 x5  e9a4  0a75
branch, LARQL_KV_SEQPAR=auto   8       6         1            0         1     5eb5 x4  86d7  86b6
main 4e48d70e   (#259)         8       6         1            1         0     (no fp column on main)
main 9c13bf66   (#255, pre-#258 — no seqpar code exists)
                               8       5         1            2         0     (no fp column)
```

Read across the rows, not down the columns:

- **Every face that appeared did so on both kernels.** Early-stop is on
  serial and seqpar; three distinct sane-latency trajectories on each; the
  one hang was on seqpar and has not replicated. Nothing here is the
  attention kernel's.
- **Pristine `main` has early-stop *and* the catastrophic short-circuit**
  (0.48 ms/token, 2064 tok/s, "7 steps completed"). So does `9c13bf66`,
  which predates every line of KV-B1. **KV-B1 / seqpar and the `[64]`
  default flip are exonerated as the cause of #229.** The pool lead is
  untouched by this — a generic lifecycle bug predating seqpar fits — but
  the causal window is now "before #255", and the next move is to keep
  walking `main` back until a tree is clean.
- The catastrophic face has two prefill signatures on record: one process
  spent 37.4 s on the prompt and then decoded at nothing (`main` r6), one
  spent 15.7 s (`9c13bf66` r8, against ~33 s healthy). So the broken state
  can set in *during* the prompt pass, not only at the prompt→generation
  boundary. Instrument both.
- "Prefill" on this routed path is not a prefill kernel: `prefill_for_streaming`
  calls `decode_token_q4k_moe` once per prompt token, so the C prompt is
  ~2024 back-to-back decode steps through `decode/token.rs`. Every fault
  seen today landed inside those steps; a process that clears them decodes
  its budget cleanly. The open fork is **deep position** (KV/attention at
  ~2K) vs **deep history** (~2K sequential steps of pool churn), and the
  discriminator is prompt A with `-n 2048`.

**HANG is a fifth face.** The seqpar r3 process sat 6 min 55 s at 0 % CPU
in `-[_MTLCommandBuffer waitUntilCompleted]` from `decode/token.rs:761`,
inside `prefill_for_streaming`; the unified log shows no GPU reset or
fault; `kill -9` was the only exit. A hang prints **no row**, so it is
invisible to every row-based check — including preconditions 7–9 — and a
harness without a per-process wall-clock watchdog will simply stop. The
tree-control script kills at `HANG_SECS` (default 300), samples the stack
first, and writes a `HANG` row.

**PANIC is a sixth.** On the short prompt A with `-n 2048` (the
geometry-vs-history discriminator, 8 fresh processes, serial kernel), one
process died with rc=101 at `crates/larql-compute/src/forward/embed.rs:19` —
`weights.embed.row(tok_id as usize)` on a token id **past the end of the
vocabulary**. That is the "differing top-K `hits` vector" made concrete:
the corrupt state can emit *invalid token ids*. It strongly supports a
common token-selection-corruption parent for EARLY-STOP (the same
corruption landing on EOS instead of past the table) — a unified
hypothesis, not yet a proved identity; see the ledger rule below. The other
seven: six healthy through all 2047 steps with **one** fingerprint
(`78c080b5`), and one `early stop @2019/2048` carrying a different
fingerprint. So the fault also fires on a 36-token prompt after enough
steps — but at step ~2019 the KV position is ~2055, so **prompt A with
`-n 2048` cannot separate deep position from deep history**; on this path
they are collinear. Nor do the C-prompt "token 1" failures localise
anything: a fault at *any* of the 2024 prompt-pass steps surfaces at token
1. The instrument that splits this is the **first-divergence step** — dump
token ids and report "diverged from the canonical trajectory at step k".

The fault ledger is therefore six faces, counted separately, not one
mechanism until something proves it:

```text
HEALTHY
EARLY-STOP        stops at token 1..k; row printed
CATASTROPHIC      full step budget, impossible speed; row printed
NONDETERMINISTIC  sane latency, wrong trajectory; row printed, fp differs
HANG              no row; waitUntilCompleted never returns
PANIC             no row; rc=101, token id >= vocab reaches embed
```

**Mechanism named — 2026-08-16, later the same day.** Of 105
`wait_until_completed` sites in `larql-compute-metal`, exactly one looked at
the command buffer's status afterwards, and it treated `Error` as done. A
failed Metal command buffer returns from the wait like a finished one, and
after a GPU fault later buffers on the queue may be *ignored* — also
returning instantly. Adding a status check after the hot-path waits
(`cb_status.rs` in worktree `startup-fault-229`) and re-running the C-prompt
control produced, on the one EARLY-STOP process of eight:

```text
[metal] command buffer at decode/token step finished with status Error (#1):
  Caused GPU Address Fault Error (0000000b:kIOGPUCommandBufferCallbackErrorPageFault)
```

and nothing on any healthy process. So the parent is a **GPU page fault —
an out-of-bounds access by some kernel — that nothing checks**. It accounts
for every face from a different angle: garbage output whose argmax is EOS
(EARLY-STOP) or past the vocabulary (PANIC); ignored buffers after the
fault, "completing" at ~0.5 ms/step (CATASTROPHIC — and the ignored buffers
begin whenever the fault does, hence the fast, slow and very-slow prefill
signatures); a fault the GPU does not recover from (HANG, twice, both at
`decode/token.rs:761` inside the prompt pass); and, plausibly, #229's own
one-position-row NaN. What is not yet known is **which kernel and which
buffer** — Metal shader validation names both, and that is the run after
this one. Same-day controls: `LARQL_FUSED_DECODE_HEAD=0` (unfused head)
faults at the same rate, and main+#262 (`get_bytes` aliasing fix) faults
at the same rate — neither is the OOB.

The rule this leaves behind is simple: **every `wait_until_completed` must
be followed by a status check, and an `Error` must stop the step** — a
poisoned step must never be allowed to hand its output to the sampler.

**Root cause — 2026-08-16, 14:30.** Guarding the one unbounded GPU
indirection (`moe_descriptor_gather` reads `descs[selected_ids[slot]]`;
now clamps an id `>= num_experts` and counts it) made the page faults
disappear and replaced them with a *named* event: two of eight processes
printed `router emitted 40 expert id(s) >= num_experts` with **zero**
command-buffer errors. 40 = 10 layers × top-k 4: the residual goes NaN at
the same layer both times and every router below it emits `~0u`. The NaN
is computed, not stored (a scan of the MXFP4 container found no `0xFF`
E8M0 scale byte in any of 24 × 32 experts). Reading the decode path's KV
sizing then found it:

```text
decode/kv_setup.rs::kv_capacities_for_layers → kv_capacity_for_window(128, 4096) = 256
encode_kv_append                              → writes row current_len, no bound, no modulo
attention kernels                              → read absolute rows [T - window, T)
compact_kv_to_window                           → called ONLY by the KV-engine coarse_* path
```

So on the routed decode path every sliding layer (12 of 24 on gpt-oss)
gets a **256-row K and V buffer** and is written and read **past it from
position 256 onward**, by a margin that grows with position — ~3.5 MB per
buffer at a ~2K prompt. The overrun is self-consistent (the kernel reads
back what it wrote) until some other allocation shares that memory; then
the layer's K/V carry garbage, the residual goes NaN from that layer down,
the router emits ids past the table, and the gather page-faults. That is
every face, and every property: per-process (allocation layout),
position-dependent (overrun distance), the ~2K prompts, the B prompt's
relative quiet, the same entry layer twice, `--repeat` inheriting it, and
#229's own one-position-row NaN on the same path class. Pinned by
`decode_path_sizes_sliding_layers_for_the_full_max_seq_because_it_never_compacts`
(fails on the old sizing: "layer 0 (sliding_window 128) allocated 256 rows
for a 4096-row request").

Fix: the decode path allocates **every layer at the full requested
`max_seq`** (`kv_capacities_for_layers` and `reset_and_preallocate_kv_cache`);
the window-derived capacity stays available for the path that compacts.
Containment stays regardless: `cb_status::wait_checked` on all 101 waits
(a test forbids naked waits), the entry seam refuses a step whose buffer
failed, the prompt pass propagates a failed step instead of substituting
zeros, the gather guard, and `LARQL_CB_DIAG=1` for encoder status +
per-dispatch signposts. **Acceptance, 2026-08-16 15:09, fixed binary,
16 fresh C-prompt processes: 16/16 full completion, 0 command-buffer
errors, 0 gather-guard events, 0 hangs, 0 panics, no physical-floor
violation** (slowest row 98 ms/token under a concurrent Glimmer capture —
contention, not a fault), and **four fresh `larql run --metal` processes
on the C prompt produced byte-identical output** (`f82b05b5…`, 149 bytes).
Every criterion met; #229 moves from root-cause-found to fixed once the
worktree lands. `larql-compute-metal` lib tests 402/402, `larql-compute`
1056/1056, `larql-inference` 1499/1499 on the fix.

The lesson to keep is not "256 vs 4096". It is that **a residency policy
and the operation that makes it valid must be pinned together**: a
window-sized capacity is legal only on a path that compacts, and the pin
test now says so in the path's own words.

`fp=` hashes `(token text, prob bits)` per token, so it is stricter than
"same tokens": a last-ulp wobble that never moves the argmax reads as a
different fingerprint. Today's `off` arm gave three fingerprints, and one
of them (`e9a4`) is the doc's "run 3", which was a *different text*. Before
fp multiplicity is scored as a nondeterminism defect on its own, the report
should carry token-trajectory equality and prob-bit equality as separate
columns.

### Two blocks are labelled "block A" — neither is current

Commit `7e832261`'s message cites a short-context block reading
`11.97/10.84/12.01/10.86 ms/token` on "a verified-idle GPU". The
provisional table below, added one commit later in `68879332`, gives block
A as `12.14 12.15 / 10.88 10.90`. These are **two different blocks**, not
a transcription error — the session transcripts contain one set each — and
they were recorded under different conditions with the same label.

Neither is current, which is the useful resolution rather than adjudicating
the digits:

- Both predate the peer handshake, so neither satisfies precondition 5,
  and "verified-idle GPU" means the check now known to miss a peer.
- The `68879332` session failed precondition 1 for its whole duration
  (65W adapter).
- Both were `off/auto/off/auto`, so both carry the ordering bias described
  above, with the candidate as the later arm.
- Block A re-run under the bracket on 2026-08-16 is **void** (2.98%).

So the commit message's "clean short-context result" no longer holds under
the tightened preconditions, and **no short-context block currently
passes**. The lesson is the reusable part: a measurement quoted in a commit
message and the same measurement in a doc drift apart silently, and the
commit message is the artifact a reader treats as the justification for a
default flip. Quote block readings in one place and reference it from the
other.

## Provisional readings — 2026-08-15, VOID

Recorded because the prediction check pre-dates the clean run, which makes
the re-run a genuine replication rather than a first look. **These are not
results.** Blocks B and C fail precondition 4 on the `default` arm, and
precondition 1 failed for the whole session (65W adapter).

```
                     off (ms/tok)     default (ms/tok)   same-arm spread
A  ~36 + 256 tok     12.14  12.15     10.88  10.90       0.08% / 0.18%  PASS
B  ~574 + 256        13.73  13.69     10.98  11.19       0.29% / 1.91%  VOID
C  ~2024 + 256       18.19  18.26     13.67  12.93       0.38% / 5.43%  VOID
```

What survives without the arm means, as a description of the data rather
than an inference: the arms do not overlap in any block, and the *slowest*
`default` reading beats the *fastest* `off` reading by 10.2%, 18.3% and
24.9% respectively.

The prediction holds directionally — the gain grows with depth, as 12
sliding layers pinning at window 128 while 12 full layers keep climbing
predicts. The asymptote toward the full-attention share is **not** observed;
it is still rising at ~2K, so locating the plateau needs deeper context.

The token-by-token prefill phase is a second instrument on the same kernel
(`encode_attention_block` has one caller, `decode/token.rs`, and prefill
runs at decode rate — 19706 ms for ~2024 tokens). It averages over the
whole span ramp rather than sitting at final depth, so it should show a
*smaller* delta than decode at the same block, and it does: −10.8% at B and
−18.2% at C, with arms tight to 0.01–1.7%. Two instruments, same shape.

## Bracketed readings — 2026-08-16

First session on the 94W adapter, under the bracket protocol. **One block
passed.**

```text
                       off      default    off      brackets   verdict
A  ~36 + 256 tok      11.75      10.84     12.10      2.98%     VOID
B  ~574 + 256         13.75      10.98     13.77      0.15%     VALID
C  ~2024 + 256        19.75      18.04     40.49       105%     VOID
```

Blocks A and C were also taken before the peer handshake existed, so they
fail precondition 5 as well as precondition 6.

**Block B, the first valid block on this ladder:**

```text
serial (off)      13.75 / 13.77 ms      72.7 tok/s
seqpar (default)          10.98 ms      91.1 tok/s

+25.4% throughput, −20.2% latency
```

The two bracketing baselines differ by 0.15%, so the candidate sits inside
a demonstrably stable window. Read against the short-context readings, the
shape is the one the layer-class decomposition predicts — but note the
short-context row below comes from a **void** block A, so the comparison
is directional only:

```text
              serial      seqpar
short          ~83         ~92        (block A, VOID — directional only)
~574            72.7        91.1      (block B, VALID)
```

If that holds, the claim is not "91 tok/s" — it is that **seqpar holds
throughput roughly flat where the serial kernel falls away with context**,
i.e. it removes an increasing fraction of the context penalty. That is a
statement about the *slope*, which is the metric this ladder was set up to
move. It needs block A re-run under the full preconditions to be said at
all, and block C to be said about depth.

**Licensed — 2026-08-16, after #229 (#263).** The full ladder, one binary,
one session, `off / default / off` with 300 s rests, every arm at its full
255-step budget, both `off` arms of each block carrying one fingerprint:

```text
block  prompt          serial (off)    seqpar (default)   Δ throughput   brackets
A      ~36 + 256 tok      12.04 ms         10.80 ms          +11.5%        0.08%
B      ~574 + 256         13.71 ms         11.01 ms          +24.5%        0.51%
C      ~2024 + 256        18.11 ms         11.88 ms          +52.4%        0.55%
```

Serial degrades 12.0 → 13.7 → 18.1 ms/token with context; seqpar holds
10.8 → 11.0 → 11.9. That is the slope claim, and it is what the decision
table wanted: short, medium and deep all VALID, the candidate ahead in
every block, brackets agreeing. `SEQPAR_DEFAULT_ON_HEAD_DIMS = &[64]` is
therefore the shipping default for gpt-oss's geometry, and the two
expectation tests in `kv_seqpar/tests.rs` pin that state instead of the
pending one.

Recorded deviation: the session ran on the 65 W adapter with a full battery
(user's call), so precondition 1 was not met as written; the brackets
(≤0.55%) are the evidence that power did not move under the blocks. Block C
was runnable at all only because #229's KV overrun is fixed — its `off`
arms complete 255/255 where the same prompt used to stop at token 1.

## Glimmer geometry — the policy becomes a planner (2026-08-16, evening)

KV-B1's kernel ported into the VINDEX3 lowering unchanged (PR #265: tiered
short/long dispatch, seqpar behind the shared policy, a route witness
printed by `larql vindex3 exec --generate`). Its *policy* did not port:
`SEQPAR_DEFAULT_ON_HEAD_DIMS = &[64]` encodes span → threadgroup-width
tiers measured on gpt-oss (64 query heads, 8 KV heads, head_dim 64), and
Muse-Glimmer-30B is 32 query heads, 2 KV heads, head_dim 128 — the same
widths are half the slices, on half the head-level threadgroups, with 8
slices already the 1024-thread ceiling. So the policy moved into
`ops/attention_geometry.rs`: `choose_attention_geometry(request,
{head_dim, q_heads, kv_heads, span})` over **measured rows**, no model
names, serial wherever no row exists. The gpt-oss row reproduces the KV-B1
tiers exactly (pinned by test); the decode path and the lowering both call
the planner.

### The Glimmer surface

`scripts/glimmer-seqpar-surface.sh`, `bench/prompts/glimmer/span-*.ids`,
64 decode tokens, **300 s rest before every arm** — the first attempt ran
arms back-to-back and read 76 → 74 → 79 → 121 → 159 → 148 across nine
minutes: a monotone drift in *time* that a reader would score as a
monotone slice-count cliff. Rested, with serial brackets and the witness
confirming the kernel on every row (ms/token; identical token ids on every
arm at each context):

```text
ctx    serial   2     4     8    serial   bracket   verdict
512      73    70    69    68      77      5.5%    direction only
1024     88    83    83    79      89      1.1%    near-valid: 8 slices +12%
2048    116    96    92    85     113      2.6%    lower bounds +18 / +23 / +33%
4000    144     -   136   134     145      0.7%    VALID: 4 → +6.2%, 8 → +7.8%
```

(4000 rather than 4096: prompt + 64 generated tokens must stay within
`LONG_ATTENTION_SPAN`, the long kernels' threadgroup-scratch bound; past it
the lowering now refuses instead of overflowing, which is what the first
4096 arm did.)

Row: serial below 1024, `SeqPar { slices: 8 }` from 1024. Under the
default policy the witness splits at the boundary — 53196 serial (1023
positions × 52 layers) then 56628 seqpar — and the mixed trajectory is
byte-identical to both pure arms.

Two readings, kept apart. **B1 works at this geometry**: +12% at 1K, ≥+33%
at 2K, and 8 slices is best from 1K because 8 × 128 is the intra-TG ceiling.
**Past ~2K the family's gain collapses** (+33% → +8%) even though 39 of 52
layers stay at their 2048 window: the serial phase-3 walk is no longer
what dominates. The candidate — hypothesis, to be measured, not claimed —
is the 16:1 GQA read amplification: sixteen query-head threadgroups per KV
head each stream that head's whole K/V, ~2 GB/token at 4K, which no
intra-threadgroup slicing touches. The discriminating experiment is a
synthetic attention bench, 32 per-head threadgroups vs 2 KV-head groups
each serving 16 queries from one K/V stream, at 2K and 4K: if the grouped
form barely moves 2K and opens up 4K, that is the surface above, and a
GQA-group kernel is the next attention rung — followed by B2 (sequence
tiles across threadgroups) once B1's width is spent.

