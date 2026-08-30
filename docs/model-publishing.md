# Republishing models — the 2026-08 recovery, and why it was manual

**Publishing belongs to the Vindex Factory.** `larql recipe build` already
runs PREFLIGHT → RELEASE: fetch a pinned revision, extract, slice each
declared output, verify checksums, publish **private**, then flip to public
only after verification passes. That is
[`docs/vindex-factory.md`](vindex-factory.md) §7–§8, and it is a stronger
pipeline than anything done by hand.

This document is not an alternative process. It records a republish that
**could not use the factory**, what that cost, and what needs to exist so the
next one does.

---

## 1. Why this republish was manual

PR #207/#212 fixed the ggml nibble layout, which invalidated every vindex
written before 2026-08-07 that stores Q4_0 or Q6_K. Five models needed
re-extracting and republishing.

**No recipe exists for any of them**, and `chuk-vindex-recipes` — the data
plane, per [`vindex-factory.md`](vindex-factory.md) §3 — is not checked out
on this machine. `larql recipe build` had nothing to run, so the work went
through `larql extract` + `larql publish` directly.

That is the gap. Everything below is a consequence of it.

---

## 2. What the factory would have prevented

Not hypothetical — these happened.

**A partial upload went public.** HuggingFace rate-limited a 1.65 GB upload
(`503 SlowDown`, part 35 of 109) and the publish aborted. The repo was left
holding the *old* weight file with no replacement — still serving, still
public, now inconsistent with its own manifests. The factory publishes
private and only flips at RELEASE after VERIFY-B (§8: "nothing goes public
unverified"), so this failure would have been contained to a private repo.

**Stale files accumulated.** `publish` only ever added files, so renaming a
weight file left both generations live and the loader chose between them by
name. 4.4 GB of superseded bytes across six repos, removed by hand. Fixed in
PR #215, but the factory's checksum verification compares published bytes
against the manifest, which is a second net under the same failure.

**Nothing pinned the source revision.** The re-extractions used whatever
snapshot happened to be in the local HF cache. A recipe pins the revision, so
a rebuild is reproducible; a manual run is only as reproducible as the cache.

---

## 3. The audit rule

Factory-independent and worth keeping: how to tell whether an existing vindex
is affected.

A vindex is affected **iff both** hold:

1. it stores `Q6_K` or `Q4_0` blocks, **and**
2. it was written before **2026-08-07**.

`Q4_K` was never affected. Two traps:

- **`lm_head_q4.bin` is Q4_K** despite the name (`quantize_q4_k` in
  `write_kquant/lm_head.rs`). A blind "delete the old-named files" cleanup
  would destroy a good 378 MB tensor. Guard every delete on its replacement
  actually being present.
- **Two filename generations exist** — current `*_kquant.bin` and legacy
  `*_q4k.bin`. An audit globbing only the first reported `qwen3-0.6b-q4k` as
  carrying no quantised weights; it carries `interleaved_kquant.bin` and was
  fully broken. Match both, or list the directory.

`interleaved_*.bin` written in `q4k` mode always carries Q6_K — the layout is
`[gate Q4_K | up Q4_K | down Q6_K]` (`write_kquant/ffn.rs`). `attn_weights_*`
carries Q6_K for the V projection.

Confirming is cheap and unambiguous — an affected vindex emits obvious
garbage:

```
$ larql run gemma3-4b-q4k-v2.vindex "The capital of France is"
 shaker peč mixtoவர கருதப்படுகிறதுstehungjö às części ladder Vase znač
```

---

## 4. The manual fallback

Use only when no recipe exists. Prefer `larql recipe build`.

```bash
# 1. Extract to a NEW path — never overwrite the only working copy.
larql extract <hf-snapshot> --output <name>-v2.vindex --quant q4k --down-top-k 10

# 2. Verify it generates. "Re-extracted" is not "re-extracted correctly".
larql run <name>-v2.vindex "The capital of France is" --max-tokens 10

# 3. Publish (prunes stale remote files since PR #215).
larql publish <name>-v2.vindex --repo chrishayuk/<repo> --slices none

# 4. Verify the published file list — current names present, legacy gone.

# 5. Archive. COPYFILE_DISABLE stops macOS writing AppleDouble sidecars.
COPYFILE_DISABLE=1 rsync -a SRC/ /Volumes/chrishayuk/vindexes/<org>/<name>.vindex/

# 6. Byte-compare before deleting local — size equality misses SMB corruption.
cmp SRC/interleaved_kquant.bin DST/interleaved_kquant.bin
```

Naming follows [`card::naming`](../crates/larql-factory/src/card/), which the
factory and its generated cards share: `<model>-q4k-vindex` quantised,
`<model>-vindex` unquantised, `<repo>-<preset>` for slices.

Two repos predate that convention and have no parent —
`gemma-4-26b-a4b-it-vindex-expert-server` and
`gemma-4-26b-a4b-client-vindex-client`. Reproduce with `--no-full` rather
than creating parents that never existed.

---

## 5. Operational traps

**Piping to `tail` masks the exit code.** `larql extract ... | tail -40`
reports `tail`'s status. One 26B extraction "succeeded" in two minutes having
done nothing — the binary was missing. Use `set -o pipefail`, or don't pipe.

**macOS writes AppleDouble sidecars to SMB.** A 15-file vindex arrives as 30,
each `._name` an xattr sidecar. Harmless for loading, but they would be
uploaded if the archive were ever published from.

**`--quant q4k` did not imply `--level all`** despite its help saying so, so
`index.json` recorded `inference` while the writer emitted an `all` vindex.
Fixed in PR #215.

---

## 6. Status — 2026-08-08

| model | affected | re-extracted | published | archived |
|---|---|---|---|---|
| gemma-3-4b-it (+5 slices) | yes | ✅ | ✅ pruned | kept local (testing) |
| gemma-4-26b-a4b | yes | ✅ | ⬜ | kept local (testing) |
| granite-4.1-3b | yes | ✅ | ✅ auto-pruned | ✅ local deleted |
| granite-4.1-8b | yes | ✅ | 🔄 | ⬜ |
| granite-4.1-30b | yes | ✅ | ⬜ | ⬜ |
| qwen3-0.6b-q4k | yes | ✅ | ⬜ | ⬜ |
| qwen3-0.6b (dense) | no | — | ⬜ | ⬜ |
| bitnet-b1.58-2b (ternary) | no | — | ⬜ | ⬜ |
| gemma3-4b-f16 | no | — | ⬜ | kept local |

---

## 7. The road to recipe-driven — sequenced

Decided 2026-08-08: finish this recovery on the factory path rather than by
hand. The steps are ordered because each unblocks the next.

### R1 — cut `v0.2.0` *(blocks everything below)*

`v0.1.1` is dated 2026-07-27 and **predates the nibble-layout fix**. A recipe
pinning it would extract the broken layout, so the tag has to move before any
recipe can honestly reference it.

Needs #215 (publish prune + retry), #216 (models coverage), #217 (this doc)
merged first.

**Why the pin matters even though nothing enforces it.** `larql recipe build`
runs subprocesses of `std::env::current_exe()`, and no stage reads
`extractor.version` — so a build today would use the local binary and produce
*correct bytes*. But the recorded provenance would name a release that never
built them, which is precisely what §4.1 of
[`vindex-factory.md`](vindex-factory.md) says the pin exists to prevent:
"without a pinned upstream it is not reproducible and the provenance in the
vindex-v1 manifest is a lie." Correct bytes with false provenance are worse
than manual publishing with none.

*Worth fixing separately:* the driver should assert that its own version
matches `extractor.version`, or record the actual one. A pin nothing checks
is a comment.

### R2 — create `chuk-vindex-recipes`

§3.1 is RESOLVED — recipes, `policy/`, `verify/` prompt sets and the two
workflows live there, not in `larql`. The repo does not exist yet.

Seven validated recipes are ready to seed it (one per model in §6), each with
a pinned upstream revision and a distinct `build_id`.

### R3 — resolve the dotted-name conflict

`metadata.name` *is* the repo slug (`card::naming::hub_repo_name`), and
`is_kebab_case` rejects dots — but `granite-4.1-3b`, `qwen3-0.6b` and
`bitnet-b1.58-2b-4t` all carry dotted versions and are already published under
dotted names. Deriving the slug from the name would silently move
`granite-4.1-3b-q4k-vindex` to `granite-4-1-3b-q4k-vindex` and orphan the
existing repo.

Worked around by pinning literal `repo_template` values, which loses the
`{model_slug}` indirection. The real question: should `is_kebab_case` permit
dots in this position? Upstream model names routinely contain them.

### R4 — run the builds

`larql recipe build recipes/<model>.yaml` per model, which brings what the
manual path lacked: a pinned revision, checksum verification, and
private-until-verified.

### R5 — resolve PIN 13.1 explicitly

§13.1 is still formally open (dataset repos vs model repos). Every published
vindex is a **model** repo and the recipes say `repo_type: model`, so this is
already decided in practice — it should be written down before more artifacts
accumulate under an undocumented choice.

---

## 8. Beyond the recovery

**A nibble-layout version in the vindex format.** Nothing records which
layout a vindex's bytes are in, which is the entire reason a fixed reader
turns old bytes into fluent garbage rather than erroring. A version field —
with the loader refusing, or converting, on mismatch — is the durable fix.
Everything in §3 is a workaround for its absence, and it is the single most
valuable item on this page.

**An audit subcommand.** §3's rule is mechanical; `larql vindex doctor` could
answer it instead of a human globbing filenames and getting it wrong — which
happened, and reported a fully-broken vindex as unaffected.

**Model-card notes on affected repos.** Anyone who pulled before 2026-08-07
holds broken bytes and has no way to know.

**A verify harness with teeth.** §3.1's footnote 2 is honest that
`larql verify` is checksum integrity only; the reconstruction and logit-match
blocks in every recipe's `verify:` section are declared but not yet enforced.
Recipes asserting thresholds nothing checks is the same failure mode as R1's
unchecked version pin.
