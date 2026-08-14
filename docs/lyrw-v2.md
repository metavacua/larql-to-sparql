# LYRW v2 — the K3 routed-layer physical-layout gate

**Programme:** `lyrw` — six experiments deciding whether the K3 extraction can use today's per-layer weight format or needs one final revision first.
**Scope:** the *storage* half of the K3 work. The model-side half is [`k3-funnel.md`](k3-funnel.md); the transport half is [`dec-funnel.md`](dec-funnel.md); the production half is [`vindex-factory.md`](vindex-factory.md).
**Status:** v0.1 — draft. Three of the six experiments are materially narrowed by results already banked in `dec`; two are blocked; one blocker found while scoping outranks all six.
**Date:** 2026-08-01

---

## 0. The question, and what it is not

The per-layer LYRW format ([`format-spec.md` §5.12](../crates/larql-vindex/docs/format-spec.md)) already stores, per expert, two independently-addressed regions with their own offsets and lengths:

```
[entry e gate+up]   shape [2*inter, hidden]
[entry e down]      shape [hidden, inter]
```

The limitation is one line of design principle and one field in the header: **the whole layer file declares a single `quant_format`**, so the two regions cannot carry different representations without replacing the file.

So the programme does not need to ask "how should everything be sliced". It needs to answer three architectural questions —

1. should gate/up and down support different formats?
2. does physical expert ordering or grouping materially improve SSD behaviour?
3. are any inference modes that omit weights good enough to support?

— and leave everything else as runtime policy over the existing per-expert offset table.

**What this programme is a gate on.** Extraction *feasibility and operations*, not throughput. Standing rule **R4** ([`dec-funnel.md` §1](dec-funnel.md)) applied here: this whole programme targets routed expert bytes, which `dec8-6` measured at **25.83 GB of 55.69 GB** per-token touch (46.4 %). Zero them entirely and the dense half alone still caps the machine at **12.3 tok/s**. **No layout result in this document can decide a throughput target**, and any arm that reports itself in tok/s is answering a question it cannot answer. Judge these experiments in bytes-written, bytes-faulted, wall-clock-to-rebuild, and exactness — the units the operation actually has ([[feedback_metric_matches_operation]]).

---

## 1. What the banked evidence already settles

Four results from the `dec` programme land directly on this design. Three of them delete work.

### 1.1 The fidelity axis of Experiment 4 does not exist for K3 — **C1**

`dec8` measured that **MXFP4 → Q6_K transcode is exact**, over 1,032,192 superblocks of real K3 expert scale tensors ([[project_k3_kernel_ladder]]). fp4's value set doubled is `±{0,1,2,3,4,6,8,12}` — all integers inside Q6_K's signed 6-bit range — and both structural conditions hold with four exponents of headroom (max observed spread 2 against a limit of 6; `d = 2^(e_min−128)` representable in f16).

K3's routed experts are MXFP4 at source. Therefore, for K3:

| arm | gate/up | down | output vs Q0 |
|---|---|---|---|
| Q0 | Q6_K | Q6_K | — |
| Q1 | MXFP4 | Q6_K | **bit-identical** |
| Q2 | Q6_K | MXFP4 | **bit-identical** |
| Q3 | MXFP4 | MXFP4 | **bit-identical** |

**All four arms produce the same numbers.** The proposed battery — expert-output cosine, weighted latent aggregate, next-layer router agreement, teacher-forced BPB, token KL, greedy agreement, long-generation stability — has *nothing to measure* across Q0–Q3. Running it would produce four columns of zeros and read as a strong result.

What differs across those arms is **bytes and kernel η only**, and both are already instrumented: `larql k3-ledger ceilings` composes per-class time from banked `MeasuredEfficiency` ranges.

The arms that *do* have a fidelity axis are Q4/Q5 (an "approximate lower format"). But note what approximate means here: the source alphabet has 15 values and MXFP4 already spends exactly 4 payload bits on it. Going lower is **lossy requantisation of an already-4-bit tensor**, not a container choice — a different experiment with a different falsifier, and one `dec8`'s variable-rate census already priced at a 0.25-bit total prize ([[project_k3_exact_format_floor]]).

### 1.2 So the case for mixed-format is operational, not numerical — **C2**

Strip the fidelity axis and one motivation survives, but it is a good one:

> **Re-quantising one region without rewriting the layer file.**

At K3 scale that is not a nicety. A single routed layer file holds 896 experts:

| container | per expert | per layer | × 92 MoE layers |
|---|---:|---:|---:|
| MXFP4 (source density) | 17,547,264 B | **15.72 GB** | **1.45 TB** |
| Q6_K (cheapest exact + servable) | 27,095,040 B | **24.28 GB** | **2.23 TB** |

*(Cross-check: 1.45 TB routed + 0.112 TB always-resident dense = 1.56 TB, matching the published checkpoint size of 1.561 TB to three figures. The shapes are right.)*

Two consequences worth stating plainly:

- **Q6_K everywhere makes the routed vindex 43 % larger than the model it was derived from** — 2.23 TB against 1.561 TB. Same failure *shape* as `k3-funnel.md` Finding 2's 1.82 TB gate blow-up, arrived at honestly rather than by a silent no-op, but it is still a derived artifact exceeding its source.
- **Migrating one region across 92 layers is a 1.45–2.23 TB rewrite today.** With per-region format tags it is a 1.45 TB rewrite of two-thirds of the bytes, or 0.74 TB of one-third — and, more to the point, it can be done *per layer, incrementally, against a live index*.

That matters because MXFP4's kernel maturity is `Grouped`, below `is_servable()` ([[project_k3_kernel_ladder]]). The moment `mxfp4_grouped_experts` reaches `Dispatched`, you want to move gate/up (two-thirds of expert bytes) to MXFP4 and leave down on Q6_K until its kernel follows. **That migration is the entire product case for LYRW v2**, and it is invisible to any fidelity or single-run throughput measurement.

### 1.3 The byte split is exactly 2:1, and it is not 1:1 — **C3**

`dec8-0` read shard 50's header: per routed expert, `w1`/`w2`/`w3` are **5,849,088 B each, equal to the byte**. So:

```
gate/up region (w1 + w3) = 11,698,176 B   =  2/3 of expert bytes
down region    (w2)      =  5,849,088 B   =  1/3 of expert bytes
```

Any format lever on gate/up is worth **twice** the same lever on down. That orders the arms: Q1 (cheap gate/up) captures two-thirds of the available byte saving, Q2 captures one-third. It also caps branch-dropping at **1.5×, never 2–3×** — already banked, and the reason the "drop a branch" lever was found double-counted in the miss budget.

### 1.4 The repo already ships projection-asymmetric precision — and it does not apply to K3

[`format-spec.md` §5.10](../crates/larql-vindex/docs/format-spec.md) and [`fp4-precision-policy.md`](../crates/larql-vindex/docs/fp4-precision-policy.md) default the FP4 feature-row tier to `{gate: fp4, up: fp4, down: fp8}`, on exp-26 cross-model evidence that *down carries FFN's heaviest-tailed per-feature magnitude distribution*. `index.json` already has `fp4.projections.{gate|up|down}.precision` as the authoritative field.

So the engine already believes the two regions want different precision, has measured evidence for it, ships it in one storage tier, and **LYRW is the one tier that cannot express it.** That is a strong prior for L1.

**But it is inapplicable to K3.** A heavier tail argues for a *richer* container on down; K3's source is 4-bit everywhere, so there is nothing richer to buy — Q6_K down is exact, MXFP4 down is exact, and no container in between adds information. The exp-26 policy axis is real for models extracted from bf16/f16 sources and dead for K3.

**Consequence for the run order:** E4's fidelity arm belongs on a **bf16-source MoE** (Gemma 4 26B A4B, Qwen3-30B-A3B, OLMoE), not on K3. E4-on-K3 reduces to E1's byte/η arithmetic. These are two different experiments that the original plan ran as one.

> **To verify before citing:** exp 26's down-tail finding is recorded as "FFN" cross-model data. Whether the corpus included MoE *routed-expert* `w2` — as opposed to dense FFN down — is not established here, and E4's premise depends on it. Check the exp-26 source before the arm is built; if it is dense-only, the first E4 measurement is the per-projection tail on a routed bank, not the format A/B.

---

## 2. The blocker that outranks all six experiments — **C4**

**The writer cannot write a K3 routed layer, and no layout choice changes that.**

`write_layer_weights` (`crates/larql-vindex/src/format/weights/write_layers.rs:68`) takes `entries: &[LayerEntry]` — every expert's quantised bytes, fully materialised. `quantize_moe_entries` (`:209`) takes `gate_up_bf16: &[u8]` and `down_bf16: &[u8]` spanning **all** experts and returns `Vec<LayerEntry>`. At K3's routed layer:

| | |
|---|---:|
| input slice, bf16, all 896 experts | **39.4 GB** |
| f32 intermediate, if materialised per the current `bf16_bytes_to_f32` path | **118 GB** |
| output `Vec<LayerEntry>`, Q6_K | **24.3 GB** |

against 128 GB of RAM. The peak is not survivable, and it is a property of the API signature, not of the machine.

**It is also barely wired.** The only caller of either function outside tests is `crates/larql-cli/examples/convert_moe_to_per_layer.rs` — an *example* used to migrate an existing Gemma vindex. The main extraction path does not write per-layer MoE weights at all. So "can the K3 extraction use the current physical layout" currently has the answer **"the current physical layout has no K3-capable writer, in any variant."**

**The fix is layout-independent and should start now:** stream the write — emit the header, reserve the offset table, quantise and append one expert at a time, backpatch the table on close. That shape is identical under L0, L1 and L2, and it is a precondition for every arm in E1 and E2. Peak RAM becomes one expert (27 MB) plus the table (28 KB at 896 entries).

This is the long pole. Everything below assumes it lands first.

---

## 3. The second blocker: `format_version` is written and never checked — **C5**

```rust
// write_layers.rs:249
// format_version at [4..8] — currently ignored, forward-compatible
```

It is not forward-compatible; it is silently wrong-compatible. `parse_layer_weights_header` reads the magic, skips the version, then parses the offset table as `num_entries × 32` bytes unconditionally. A v2 file whose entry stride is anything other than 32 bytes will be parsed by today's reader into **garbage offsets that are still inside the file**, so it will not bounds-fail — it will hand `get_layer_entry_bytes` a plausible byte range from the wrong place, and the model will produce plausible wrong numbers.

Same class as the `XSTRIDE` and `BufferCache::get_bytes` bugs from the kernel ladder: *wrong stride yields a plausible number from the wrong expert, not a crash.*

**Fix before any v2 file exists anywhere**, including in a fixture: reject `format_version > FORMAT_VERSION` in the parser. There is exactly one production caller (`format/weights/load/q4k.rs:186`), which already treats `None` as "skip this layer", so the failure mode is a clean miss rather than a panic. That is a two-line change and a test, and it is cheap only *until* the first v2 file exists.

---

## 4. Test assets

| Asset | Scale | Answers | Status |
|---|---:|---|---|
| **Mini-K3 fixture** | ~1/168 expert area | format, loader, dispatch, manifest correctness | to build |
| **Byte-faithful K3 routed bank** | 1 layer | SSD, mmap, grouping, cache | to build; **1 layer, not 4** — see §4.2 |
| **Loaded bf16-source MoE** | Gemma 4 26B A4B | real scheduling, E3 A/B, E4's live fidelity axis | checkpoint present, **no vindex on this box** |
| **K3 trace pack** | selected real layers | fidelity, locality, omission | **blocked** — see §8 |

### 4.1 Mini-K3 fixture — dimensions corrected — **C9**

The proposed 1/16 downscale (hidden 448, latent 224, inter 192) **breaks Q4_K/Q6_K superblock alignment on every matrix.** `pad_cols_to_256` would pad `in_cols` 224 → 256 on gate/up (+14 %) and 192 → 256 on down (+33 %), so the fixture's byte ledger would be a padding artefact rather than a scaled K3.

And no proportional downscale fixes it. Alignment needs `latent/d ≡ 0 (mod 256)` and `inter/d ≡ 0 (mod 256)`; with `3584 = 2⁹·7` and `3072 = 2¹⁰·3` that admits only `d ∈ {1, 2}`, and `d = 2` is 6.07 GB per layer — not a fixture.

**Use 256-aligned dimensions that preserve the ratios that matter instead:**

```
hidden              512     (2:1 against latent — exactly K3's 7168:3584)
latent              256
expert intermediate 256     (K3 is 3584:3072 = 1:0.857; this is 1:1 — the one deviation)
experts             896     (unscaled — routing shape is the point)
selected             16     (unscaled)
layers                4     (8 if iteration cost allows)
```

| | mini | K3 | ratio |
|---|---:|---:|---:|
| per-expert Q6_K bytes | 161,280 | 27,095,040 | 1/168 |
| per-layer, 896 experts | **144.5 MB** | 24.28 GB | 1/168 |
| 4 layers | **578 MB** | — | — |
| gate/up : down | **2 : 1** | 2 : 1 | ✓ preserved exactly |
| padding | **zero** | zero | ✓ preserved |

The `w1 = w2 = w3` equal-bytes property — the thing that caps branch-dropping at 1.5× — is preserved exactly, which the 1/16 proposal would have destroyed via asymmetric padding.

It needs no training. Its purpose is deterministic numerical parity: same architecture graph, same top-16-of-896 routing, same shared latent projections, same expert entry structure, small enough to rewrite repeatedly.

**Standing caveat: never quote a byte or latency number from the mini fixture as a K3 number.** It is 1/168 of the area with a different latent:inter ratio and a working set that fits in cache. It answers correctness; it does not answer SSD.

### 4.2 Byte-faithful bank — the right device is attached — **C7, revised 2026-08-01**

Opaque or deterministic payloads at the real K3 expert byte sizes.

**Both constraints recorded here on 2026-08-01 are now dead.** `model-drive` is mounted at `/Volumes/model-drive`: a Ugreen NVMe enclosure on Thunderbolt/USB4 Bus 0, protocol PCI-Express, link negotiated at **40 Gb/s**, 2.0 TB APFS, **1.97 TB free**. So:

- **Disk is not a constraint.** Four Q6_K layers is 97 GB against 1.97 TB free — 5 % of the volume. Arm variants, MXFP4-density copies and permuted orderings all fit alongside. The earlier "start at one layer, 175 GB internal" scoping was a property of the machine that day, not of the experiment. Start at one layer if it answers the question, not because of space.
- **This is the external SSD, over the intended link.** `dec8-2` (USB4 random-read, no model) characterises this same device class, so an E2 replay run on `model-drive` **may** be labelled as the external-SSD result — the earlier "different device, must not be labelled as such" caveat applied only to an internal-NVMe fallback that is no longer necessary.

Two methodological notes that replace them, both about the *measurement* rather than the device:

- **The drive is nearly empty (23 GiB of 2 TB used).** SSD steady-state behaviour on a fresh volume is not the same as on a full one — garbage collection and write amplification both change. E2 measures reads, so the exposure is small, but a sustained-write phase (building four layers plus variants) should not be read as a *read* result.
- **Cold means cold.** A 24.3 GB layer against 128 GB of RAM is fully page-cached after one touch, so the 2/4/8/16 GB memory budgets have to be enforced against the unified buffer cache, not just declared. Without that, every arm after the first measures RAM.

Traces to replay (unchanged from the proposal, and the set is good): uniform top-16, Zipf, Gemma empirical, clustered co-activation, adversarial rotation, repeated conversation.

Measure: SSD bytes/token, useful vs amplified, page faults, read count and extent sizes, p50/p95/p99 layer latency, degradation onset, steady-state floor, next-layer prefetch effect, physical permutation effect.

**Test expert ordering independently of file grouping.** The offset table means physical order need not equal logical order, so `logical expert ID → physical offset` is a permutation the format already supports at zero cost. Compare ID order, frequency order, co-activation-cluster order, random order — *within* L0, before considering L3.

> **Run this under a within-run control.** `dec8-12` found the machine degrades after ~7 sustained censuses, with a control variable falling 0.89 → 0.06 while every cell moved together — blind averaging would have banked a value wrong by a factor *while presenting as more data*. A sustained SSD replay is exactly that regime. The control must have a working set **at least as large as the thing it protects** (a small sentinel admitted two runs where the large class had collapsed). And gate on `n ≥ 5 && relative SE ≤ 1 %`, never on observed spread — `max − min` is non-decreasing in `n`, so a spread gate rewards collecting less data.

---

## 5. Experiment 1 — the layout gate

### Candidates

| | | |
|---|---|---|
| **L0** | current format | one `quant_format` in the header; per-expert `(gate_up, down)` offsets |
| **L1** | mixed-format LYRW v2 | same file, same offset table, each region declares its own format |
| **L2** | projection-separated files | `layer_N.gate_up.weights` + `layer_N.down.weights` |
| **L3** | expert-group files | `layer_N.experts_000_031.weights`, … |

**Prior: L1.** It provides role-specific representation without doubling file count or weakening per-expert addressability, and §1.2 gives it a concrete operational payoff that no other arm has. L2 and L3 must beat it on measurement, not on tidiness.

**L1's wire shape**, stated so E1 has something concrete to build:

```
header:  magic, format_version = 2, default_format, num_entries, intermediate, hidden
entry:   gate_up_offset u64, gate_up_bytes u64, gate_up_format u32, _pad u32
         down_offset    u64, down_bytes    u64, down_format    u32, _pad u32
```

48 bytes per entry against v1's 32. At 896 experts the table grows 28.7 KB → 43.0 KB per layer — noise against a 24 GB file, and still one page-aligned read at startup. `default_format` keeps the header self-describing for tools that only want the layer's dominant representation; the per-region tags are authoritative. **This is precisely the stride change §3's missing version check would silently misparse.**

### Method

Run on the mini fixture, all arms carrying identical Q6_K bytes so that only the container varies.

Measure: exact output parity; loader complexity (LOC and branch count in the read path); mmap count; map/open latency; dispatch construction time; bytes faulted; sequential and random top-16 execution; and **whether a representation can be overridden or omitted through the manifest without rewriting the file** — the operational property from §1.2, which is the one L1 exists for.

### Decision rule (pre-registered)

Choose the **least fragmented layout** within:

- **2 %** of best warm execution
- **5 %** of best cold physical bytes read
- **exact** numerical parity — no tolerance; the arms carry identical bytes, so any difference is a bug

Do not promote L2 or L3 on conceptual tidiness. Do not promote L1 on a fidelity result — §1.1 says there isn't one; promote it on the migration property or not at all.

**Falsifier for L1:** if per-region tags cannot be threaded to the kernel without changing a `QuantMatVec` dispatch signature, or if the mixed path forces a per-expert branch inside the grouped-expert kernel (which would put a divergent branch in every threadgroup — the failure that killed zero-carried scales), then the format is buying an operation the compute path cannot execute, and L0 stands.

---

## 6. Experiments 2 and 3

**E2 — byte-faithful replay.** Assets and controls per §4.2.

*Decision rule:* do not bake a clustered physical ordering into the permanent index unless it delivers **≥ 10 % sustained improvement across multiple plausible traces** and does not regress uniform routing badly. Prior: per-expert offsets plus a runtime cache beat permanently encoding a locality assumption — and note that K3's quantile-balanced 16-of-896 router is *designed* to avoid the expert imbalance that would make locality pay, so Gemma's strong layer-specific collapse cannot be assumed to transfer ([[project_moe_routing_locality]], and `dec8-5`'s standing rule **R2**: never transfer a union ratio across activation fractions — Gemma's 6.25 % activation against K3's 1.79 %).

**E3 — loaded-model grouped A/B.** Arms A (production layout) / B (LYRW + grouped) / C (LYRW v2 + grouped) / D (split-file + grouped), holding representation, selected experts, routing weights, activation layout, command-buffer ownership and output reduction constant.

Two constraints from the bank:

- **No Gemma vindex exists on this box** — only the raw checkpoint. E3 needs an extract run first, or a different loaded MoE.
- **A Gemma A/B is a PROXY MECHANISM result.** Never attach a K3 throughput number to it. And per [[feedback_isolated_vs_batched_kernel_profile]], measure batched (`diag_profile_kernels`) — isolated single-dispatch benchmarking understated by 3.0–6.7× on this exact class of question, and `dec8-9` produced a phantom 5.10× that was 1.23× once both arms were batched identically.

---

## 7. Experiments 4 and 5

**E4 — projection-specific quantisation.** Re-scoped per §1.1 and §1.4 into two disjoint experiments:

- **E4a (K3):** no fidelity axis. Reduces to bytes × η per region, composed by `larql k3-ledger ceilings`. Largely already answered; run it as arithmetic, not as a sweep.
- **E4b (bf16-source MoE):** the live one. Does the exp-26 down-tail policy that ships for the FP4 feature tier reproduce on a *routed expert's* `w2`, and does it pay in a container? Arms Q4/Q5 (asymmetric approximate) against Q0. Subject to the §1.4 verification note.

*Decision rule:* add mixed-format complexity only when a mixed arm beats **both** uniform alternatives by ≥ 5 % projected whole-system gain at the same fidelity gate — **or** when the migration property of §1.2 is independently judged worth it. Those are two separate justifications and should be recorded separately, because only the second one applies to K3.

**E5 — weight omission.** The proposal's central correction is right and worth restating, because it is the kind of thing that gets relitigated: **dropping an expert's `w2` does not produce a cheaper partial expert, it produces no expert output.** Keeping gate/up while dropping down is strictly dominated by skipping the expert entirely, which also saves the gate/up read. So the useful modes are:

| | |
|---|---|
| **O0** | exact baseline, all 16 |
| **O1** | reduced routed top-K (16/12/8/4), with and without gate renormalisation |
| **O2** | cumulative gate-mass retention (99/97.5/95/90 %) |
| **O3** | shared-only layer — skip the routed branch, keep shared experts, residual, attention |
| **O4** | entire routed branch remote |
| **O5** | down-only lower precision — all experts execute, cheaper `w2` |

**O4's payload arithmetic is correct and decisive.** Whole-branch RPC moves one latent in and one aggregated latent out: `2 × 3584 × 2 B = 14,336 B`. Projection-split moves 16 intermediate vectors: `16 × 3072 × 2 B = 98,304 B` — **6.9× more**, before protocol overhead. Do not split local gate/up from remote down.

*Gate:* ≤ 0.5 % BPB regression for a production approximation; no pathological p99 KL; no collapse in next-layer routing agreement; no long-generation instability; gain recomposed against the whole-model byte ledger.

*Two priors that constrain O1/O2/O5, and should be read before designing them:* [[project_walkffn_speed_accuracy_scissors]] (20 %/layer all-layer → KL 8.4 bits, 0 % top-1) and [[project_r4_zeroout_sparse_ffn]] (kernel captured only 23–40 % of a row reduction even with routing free). And per [[feedback_stacked_zero_ablation]], **mean-ablate, never zero-ablate**, across stacked layers. O1/O2 also overlap the shipped C6 drift gate (`larql dec-bench drift`) — reuse it rather than building a second instrument.

---

## 8. Experiment 6 — the K3 routing trace is blocked — **C8**

This is the only experiment that cannot be done honestly with Gemma or synthetic routing, and it is also the one that cannot be scheduled.

Capturing per-token expert IDs requires applying `block_sparse_moe.gate.weight` to a real hidden state, which requires a K3 forward pass. There isn't one: R3/P2 is unbuilt, and `A_log`'s axis is a **fail-closed BLOCKER** that header shapes provably cannot settle ([[project_k3_exact_format_floor]]) — it needs an oracle, against an asymmetric fixture, before any KDA fidelity work.

**So E6 is downstream of the K3 adapter, full stop.** Its dependency is the ladder's long pole, not this programme's.

What follows for the other five: **nothing here may freeze a physical expert permutation, a permanent hot set, co-activation grouping, or a per-layer retention policy.** All four are locality claims, K3 locality is entirely unearned ([[project_k3_kernel_ladder]]: "K3a created none, the trace still owns reuse"), and E2's synthetic traces can characterise the *mechanism* without licensing a *policy*. Keep locality as runtime metadata over the offset table — which is also where the L1 prior lands.

---

## 9. Run order

### Now — blockers and the layout gate

| | | Depends on |
|---|---|---|
| 0a | **`format_version` rejection in the parser** (§3) | — |
| 0b | **Streaming layer writer** (§2) | — |
| 1 | Mini-K3 fixture at 512/256/256 × 896 (§4.1) | 0b |
| 2 | E1 — L0 / L1 / L2 (L3 only if L2 shows something) | 0a, 1 |
| 3 | E2 — byte-faithful layer(s) on `model-drive` (USB4, 40 Gb/s, 1.97 TB free) | 0b |
| 4 | E3 — loaded-MoE grouped A/B (needs a vindex extract first) | 2 |
| 5 | E4b — projection-specific on a bf16-source MoE | 4 |

### Before the K3 extraction

6. K3 adapter far enough for a forward pass (**owner: `k3-funnel` R3/P2**, `A_log` oracle first)
7. E6 locality/cache simulation on a real trace
8. E5 top-K / mass-retention / shared-only, via the C6 drift gate
9. Freeze the K3 vindex ABI

### Then extract once

```
attention / KDA / MLA
router and MoE control
shared latent projections            (routed_expert_down_proj 7168→3584,
shared experts                        routed_expert_up_proj  3584→7168)

routed/
  layer_N.weights
    entry e:
      gate_up  offset, bytes, format
      down     offset, bytes, format
```

Not: thousands of tiny expert files; not fixed hot/cold physical slices; not one permanently selected retention policy.

### Naming discipline

K3 has two things called "down" and two called "up" — the layer-level `routed_expert_down_proj` / `routed_expert_up_proj` (shared, resident, 7168↔3584) and the per-expert `w2` / `w3`. And two called "gate": `block_sparse_moe.gate.weight` (the router) and `w1` (the GLU gate branch). Every claim in this programme must say which ([[project_k3_ssd_miss_budget]]). In this document, **"down" unqualified means the per-expert `w2` region of a LYRW entry**, and the shared latent projections are always named in full.

---

## 10. Standing hypothesis

The programme will probably show:

- one file per routed layer remains correct;
- one entry per expert remains correct;
- gate and up stay fused;
- down stays independently addressable;
- **per-region format tags are the only format extension worth making** — and they are worth it for incremental migration under a maturing kernel set, not for fidelity or single-run throughput;
- expert locality stays runtime metadata;
- omission during local inference means "skip the routed branch", not a half-expert mode;
- remote deployment moves the whole routed computation, not individual experts across the network.

**And a caution on that list.** It is an organising frame that explains everything so far, which is exactly the kind of claim that needs its own falsification test rather than accumulating confirmations ([[feedback_organizing_vs_empirical_claims]]). Its sharpest falsifier is E1's L1 falsifier in §5: if per-region tags cannot reach the grouped-expert kernel without a divergent per-expert branch, the frame's central recommendation is unbuildable and L0 stands unchanged.
