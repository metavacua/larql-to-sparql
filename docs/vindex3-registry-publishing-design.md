# VINDEX3 production registry + publishing pipeline — grounding notes

Status: **investigation only, per explicit instruction — no code written.**
This document inventories what `larql publish` (and the surrounding
extract/encode/factory machinery) actually knows, writes, uploads, and
gets back from Hugging Face today, so the production-registry design
starts from what exists rather than from a second guess. It answers the
five grounding questions in order, then freezes the decisions this phase
needs before any implementation.

Companion docs: `docs/vindex3-registry-design.md` (rungs 1/2A/2B/2C —
the resolver contract this registry data will be resolved *through*);
`docs/vindex-factory.md` (an existing, much larger, still-aspirational
design this section's §0 explains the relationship to).

## 0. This is not the Vindex Factory, and shouldn't wait for it

`docs/vindex-factory.md` already specifies a recipe → GPU-worker-build →
verify → publish → release pipeline with GitHub-PR-as-approval,
content-addressed `build_id`s, and R2 mirroring. It is real design work,
but three facts matter for scoping *this* initiative:

1. **It targets VINDEX2.** Its recipe schema, manifest fields
   (`vindex_schema_version`), and reused tooling (`larql extract` /
   `larql slice`) are all V2-shaped; `slice` refuses VINDEX3 containers
   outright (rung-1 finding, unchanged).
2. **It is unimplemented and blocked.** None of its adoption gates
   (G0–G3, §12) are met. §14 names the blocker explicitly: "the rig
   currently has a mock Vast provider only" — there is no real GPU
   build worker yet. `project_vindex_factory.md`'s own status is
   "recipe-driven builds," not "shipped."
3. **Its own PIN 13.2 puts recipes in a *different* repo**
   (`chuk-vindex-recipes`), not this one — which sits in tension with
   the hypothesis this phase starts from (canonical registry
   source-controlled in the main repo, at least initially). That
   tension is real and named in §7 rather than silently resolved either
   way.

**Conclusion carried into this design**: the registry/publishing work
here is a narrower, achievable slice — a human runs `larql publish`
(as it exists, or with small additions) against an artifact they
already built by hand or via whatever pipeline; a *separate*,
lightweight step records the result as an official registry entry. It
reuses vindex-factory's *vocabulary* where it fits (§8's "nothing goes
public unverified," the two-phase private-then-verify-then-public
shape) but does not depend on its GPU-orchestration machinery, and does
not need to wait for it.

## 1. What `larql publish` proves today (Q1)

*As grounded, before the two fixes below shipped — the first and third
bullets are now out of date; see §7 item 3 for what changed.*

Traced end to end: `larql-cli publish_cmd::run` → `execute_step` →
`upload_dir` → `larql_vindex::publish_vindex_with_opts`.

- **The only thing `publish_vindex_with_opts` returns is a static
  browse URL** (`hf_repo_url(repo_type, repo_id)` — a string built from
  the repo id alone, no commit information). No commit SHA, no
  revision, nothing else.
- **Publishing is not one atomic commit.** Each file is uploaded via
  its own `POST /api/{type}s/{repo}/commit/{rev}` call
  (`upload_file_to_hf` → `upload_regular`/`upload_lfs`, confirmed by
  the crate's own mocked tests hitting that endpoint per file). A
  vindex with 20 files produces up to 20 separate commits on `main`,
  not one. There is no single "the commit this publish produced" —
  only "whatever `main` pointed at after the last file landed."
  Interrupted midway, a puller sees a partially-updated repo; nothing
  in `publish_vindex_with_opts` detects or reports that.
- **`larql publish` does not run cleanly on a VINDEX3 container by
  default.** `run()` unconditionally calls
  `larql_vindex::load_vindex_config(&src)` (a VINDEX2-only loader) the
  moment `--collections` is non-empty, which it is by default
  (`model,family,library`). Separately, the default `--slices`
  (`client,attn,embed,server,browse`) route through `slice_vindex`,
  which refuses VINDEX3 outright. **A VINDEX3 publish only works today
  with `larql publish <dir> --repo owner/name --slices none --collections
  none`** — undocumented, easy to get wrong, and the CLI gives no
  VINDEX3-aware guidance toward it.
- Nothing in the publish path calls any structural verification
  (`Vindex3Container::verify()`, `inspect_container`, or even V2's own
  `load_vindex_config` success) before or after upload. What goes up is
  whatever `enumerate_publishable_files` finds in the source directory,
  uploaded as-is.

## 2. What's known at build time and what's already thrown away (Q2)

- **VINDEX2 *does* have a populated provenance field** —
  `VindexConfig.source: VindexSource`, written during real extraction
  (`extract/build/index_json.rs`, not a test fixture):
  ```rust
  config.source = Some(crate::VindexSource {
      huggingface_repo: Some(model_name.to_string()),
      huggingface_revision: None,
      safetensors_sha256: None,
      extracted_at: chrono_now(),
      larql_version: env!("CARGO_PKG_VERSION").to_string(),
      // v1 provenance — populated once the extractor learns to
      // fetch the upstream commit hash + safetensors digests.
      base_model_sha: None,
      extractor_sha: None,
      base_safetensors_sha256: None,
  });
  ```
  `model_name` is `extract_index_cmd`'s raw `--model` CLI argument,
  **verbatim** — whatever string the human typed. If they typed
  `google/gemma-3-4b-it`, that lands correctly. If they typed a local
  checkpoint path (equally valid input), `huggingface_repo` silently
  holds a filesystem path mislabeled as a repo id. **`huggingface_revision`
  is hardcoded `None` unconditionally** — there is no code path that
  ever sets it, regardless of input. `base_model_sha` /
  `extractor_sha` / `base_safetensors_sha256` are `None` with a comment
  in the source admitting this is deliberately unfinished ("v1
  provenance — populated once the extractor learns to..."). **This gap
  predates VINDEX3 entirely** — it is not something this initiative
  introduces, and not something VINDEX2 ever closed either.
- **VINDEX3's `encode` takes no source information at all.**
  `EncodeArgs { artifacts: Vec<PathBuf>, output: PathBuf }` — checkpoint
  directories and an output path, full stop. By the time `larql
  vindex3 encode` runs, whatever HF repo/revision the checkpoint came
  from is not a parameter anywhere in the call. `Vindex3Index` itself
  has zero provenance fields (rung-1 finding, reconfirmed): VINDEX3
  captures *less* build-time provenance than VINDEX2's already-admitted-incomplete
  story, not more.
- **Mechanically recoverable, but only incidentally.** If the
  checkpoint directory handed to `encode`/`extract` is still inside the
  `hf-hub` cache tree (`~/.cache/huggingface/hub/models--{owner}--{name}/snapshots/{sha}/`),
  the repo and exact revision sha are recoverable by parsing that path
  structure — `resolve_hf_model_with_progress` (used for pulling
  upstream checkpoints) writes exactly this layout. But nothing
  guarantees the checkpoint dir handed to encode/extract still lives
  there — a human can (and often does) copy it elsewhere first, at
  which point the information is gone with no error, no warning.

## 3. Can a registry record be generated deterministically? (Q3)

Split by field:

| `RegistryVariant` field | Deterministic today? | What's missing |
|---|---|---|
| `artifact.repo` | Yes | Known at publish time (the `--repo` argument) |
| `artifact.revision` | **No, but cheaply fixable** | `publish_vindex_with_opts` never asks HF for the resulting commit sha. One `repo.info()` call (same call the VINDEX3 download branch already uses, §2C) right after the last upload returns `sha`, HF's current commit for `main`. Not implemented; genuinely one call away. |
| `source.repo` | **No** | Not a parameter to `encode`/`extract`'s VINDEX3 path at all (§2). Would need either a new explicit flag or reconstructing it from the hf-hub cache path (fragile). |
| `source.revision` | **No** | Same as above, plus V2's own equivalent field is hardcoded `None` — there's no working precedent to copy, only an admitted gap to close. |
| `abi` | N/A — not derivable, a policy declaration | This is the one field that is *supposed* to be a human/registry-authored fact, not extracted from the artifact — see design doc §3 on `Vindex3Abi` being new, non-container machinery. |

**Verdict**: `artifact.{repo,revision}` can be made fully mechanical
with a small addition (capture the post-upload commit sha). `source.*`
cannot be, today, for either container generation — closing that gap
is prerequisite work, not something the registry can paper over by
"deriving" a fact that isn't captured anywhere upstream. Any registry
authored before that gap closes will have hand-typed (or omitted)
`source` fields, which is exactly the "provenance bug waiting to
happen" the grounding question warned about.

## 4. Registry update semantics for existing names (Q4)

No existing mechanism to ground this in — there is no code today that
updates a registry entry, because there is no populated registry yet.
This is a genuine design decision, not a fact to discover. Recorded
here as the three cases needing an explicit answer before
implementation, not resolved:

```text
new model            "gpt-oss" doesn't exist in registry/models/ yet
new variant           "qwen3.8" exists; "27b-bf16" is a new key under it
new build, same name  "qwen3.8:27b-nvfp4" exists; the artifact it
                      points to needs to move to a newer pinned revision
```

The third case is the one worth deliberate design (per the grounding
request): a mutable public alias repointing at a new immutable
revision is a real, legitimate operation (a rebuild with a bug fix, a
better quantization), but silently overwriting `registry/models/qwen3.8.json`
in place loses the "what did this alias point to before, and why did it
change" history a reviewable PR is supposed to buy. Options worth
weighing when this gets designed for real: keep it a plain git-history
question (the PR diff *is* the record, `git log` on the file *is* the
audit trail — cheapest, and arguably sufficient given §0's decision to
keep this in-repo); or carry an explicit `previous_revision`/changelog
field in the JSON itself (belt-and-braces, but a schema field to keep
honest). Not decided here.

## 5. What should constitute "official"? (Q5)

Inventory of what already exists to build this checklist from, so it
reuses real mechanisms rather than inventing new ones:

```text
VINDEX3 structural validation      → format::vindex3::validate_downloaded_container
                                      (built in 2C: dispatches to
                                      Vindex3Container::open for
                                      routed-MoE, inspect_container for
                                      system-graph — the only two real
                                      "does this open" checks that exist)
supported ABI                      → registry::Vindex3Abi /
                                      lookup_claimed_variant (rung 1) —
                                      already gates at resolution time
pinned upstream source revision    → NOT CAPTURED (§2/§3) — prerequisite
                                      work, not a registry-side check
pinned HF artifact revision        → mechanically derivable, not yet
                                      wired (§3)
architecture admissibility         → `larql vindex3 plan` (pre-encode
                                      semantic representability) exists,
                                      but runs on a *checkpoint*, before
                                      encode — nothing re-runs it against
                                      a *published* container
execution verification             → exists only as ad hoc test
                                      harnesses (the V2↔V3 compose
                                      parity gate, the granite-4.1-3b
                                      real-model compose smoke) — no
                                      general-purpose "verify this
                                      published VINDEX3 model serves a
                                      correct forward pass" command a
                                      promotion step could shell out to
provenance complete                → blocked on §2/§3 closing
```

Vindex-factory §8 already named the right *shape* for a generation-
agnostic version of this ("nothing goes public unverified"; reconstruction
fidelity; logit match against a reference forward pass; manifest
integrity) — worth reusing that vocabulary for VINDEX3's checklist
rather than a fresh one, even though the *mechanism* (a GPU build
worker) doesn't apply here. Concretely, "execution verification" for
VINDEX3 today means adapting the existing ad hoc parity-harness pattern
into something a promotion command can invoke on demand, not building a
new verification engine.

**The honest floor today**, if "official" had to mean something this
week: structural validation + ABI compatibility are real, automatable,
already-built checks. Source/artifact provenance and execution
verification are not yet — an official registry entry promoted before
those close would have to carry hand-attested facts for those fields,
clearly distinguished from the mechanically-checked ones (never a
single unqualified "verified ✓" covering both).

## 6. The two-operation publish/promote split (design carried forward, not yet built)

Per the starting hypothesis, kept distinct and *not* implemented here:

```text
larql publish <vindex>              existing command, small additions:
    ├── upload VINDEX3 to HF          (unchanged mechanism)
    ├── pin artifact revision         (new: repo.info() after last upload)
    └── emit candidate registry record (new: print, don't write, the
                                         JSON shape a promotion step needs)

official registry promotion         new, separate command/step:
    ├── validate artifact              (validate_downloaded_container,
                                         re-pulled fresh, not the local
                                         staging dir — matches
                                         vindex-factory §8.2's
                                         verify-from-hub reasoning)
    ├── run VINDEX3 verification/
    │   admissibility gates            (§5 — whatever exists today)
    ├── verify immutable HF revision   (re-resolve, confirm still pinned)
    └── add/update registry entry      (writes registry/models/*.json)
```

Publishing an artifact and declaring it official stay two different
actions with two different authorities — the second is a human decision
(a PR merge, per §0's CODEOWNERS-as-gate precedent from vindex-factory
§6.2), not something `publish` triggers on its own success.

## 7. Open questions this phase freezes for decision before implementation

**Status (2026-08-23): all four decided — main repo confirmed, hand-attest
for milestone 1, both small publish fixes shipped, `attested_by` marker
deferred to schema design.** Decisions and what shipped are recorded
inline below.

1. **Registry repo location — confirm the tension with PIN 13.2.**
   The starting hypothesis (main repo, `registry/models/*.json`, schema
   colocated with `larql-vindex`) is the one this doc's §0 recommends —
   it's a genuinely different concept from vindex-factory's `recipes/`
   (build intent, pre-publication) even though PIN 13.2 resolved
   *that* content to a separate repo. Confirm this is a deliberate,
   accepted divergence, not an oversight.
   **Decided: main repo, deliberate divergence.**
2. **Does closing the source-provenance gap (§2/§3) block the first
   milestone, or does the first official entry ship with hand-attested
   `source.{repo,revision}` and a tracked follow-up?** Given the gap
   predates this initiative and affects VINDEX2 too, closing it
   properly (a new `encode`/`extract` flag, or hf-hub-cache-path
   recovery) is real, separate work. Recommend: hand-attest for the
   first milestone (§8), track closing it as explicit follow-up — don't
   let a pre-existing, unrelated gap block the first real loop.
   **Decided: don't block — hand-attest for milestone 1.**
3. **Should `larql publish` gain the `repo.info()`-after-upload +
   `--slices none --collections none`-by-default-for-V3 fixes now, as
   small, independently-useful corrections, ahead of the rest of the
   promotion design?** Both are small, mechanical, and reduce real
   friction (the pinned-revision fact and the "V3 publish just works"
   fact) regardless of how the rest of the pipeline shapes up.
   **Decided: yes, both now. Shipped:**
   - `publish_vindex`/`publish_vindex_with_opts` now return
     `PublishResult { url, revision }`: one extra `repo.info()` call
     (`fetch_repo_head_sha`, `publish/remote.rs`) fetches the repo's
     HEAD sha on `main` immediately after the last per-file upload
     commit lands. Hard error if the fetch fails — never falls back to
     a floating `"main"` reference. `larql publish`/`larql hf publish`
     now print the pinned revision. This is what makes a mechanically
     derived `RegistryArtifactRef::revision` possible at all (§3).
   - `larql publish` no longer crashes on a VINDEX3 container by
     default: it detects the container's generation up front
     (`detect_generation`) and, for V3, downgrades `--slices`/
     `--collections` to their `none` equivalent unless the caller
     passed them explicitly, printing why. VINDEX2 behaviour is
     unchanged. (`crates/larql-cli/src/commands/primary/publish_cmd/`.)
4. **What does "hand-attested until mechanized" look like in the
   schema?** — e.g. a `Provenance.attested_by: Option<String>`-style
   marker, or simply accepting the field as-is with a comment/CI note
   until the mechanized path lands. Avoid a silent, indistinguishable
   "looks the same as a verified fact" field.
   **Decided and built (R3A, 2026-08-24): a structural enum, not an
   optional field.** `Provenance.attestation: Attestation` where
   `Attestation` is `Mechanical | HandAttested { by: String }`
   (`#[serde(tag = "kind")]`, `crates/larql-vindex/src/registry/manifest.rs`).
   Rejected the `Option<String>` shape from the original framing:
   `None` reads as "not attested" *or* "mechanically verified"
   depending on which the reader assumes, exactly the silent
   ambiguity this question warned against. The enum forces every
   entry's JSON to name `"kind"` explicitly — nothing can omit the
   field and default to the safer-looking case. `validate()` also
   refuses a `HandAttested` naming no one (`by` empty or
   whitespace-only) — a structurally-present but empty attestation
   would be the same gap one field down.

## 8. Proposed first milestone target

Per the "target exactly one real model" instruction: **granite-4.1-3b**
is the strongest current candidate — it is the one model with a
complete, already-proven VINDEX3 end-to-end loop (real-model compose
smoke, `project_vi3_inf_runtime_ladder.md`: encoded, served, INFER/WALK
verified, container already sitting at `~/chris-models/granite-4.1-3b.vindex3`).
Using it means the first official entry proves the *pipeline*, not a
new model's VINDEX3 admissibility at the same time — matching
vindex-factory's own VF-1 reasoning ("you already have a known-good
published artifact... if the factory reproduces it, the factory
doesn't lie"). **Decided: granite-4.1-3b confirmed as the milestone-1
candidate.**

Success shape, unchanged from the instruction: `larql pull
<official-name>` + `larql serve <official-name>` resolve through a
real `registry/models/*.json` entry, the artifact lives on HF with a
pinned revision, the downloaded container validates, `/v1/runtime`
reports it — no test fixture anywhere in that loop.

**R3A execution (2026-08-24)**: published `larql/granite-4.1-3b`
(`granite-4.1-3b-deploy.vindex3`, the NVFP4 build — three local
candidates existed, `-deploy` chosen as newest/deploy-ready) via
`larql publish` — no slices, no collections, VINDEX3-safe defaults
downgrade both automatically. Publish target went through two
redirects before landing: `larql/*` was the original intent, but the
HF token's account had no orgs at publish time, so the fallback was
`chrishayuk/granite-4.1-3b`; the `larql` org was created mid-session
and publishing switched back to it before any data landed under the
personal namespace (the abandoned `chrishayuk/granite-4.1-3b` shell
was deleted, never held real bytes).

**A real bug found and fixed during this**: `create_hf_repo`
(`crates/larql-vindex/src/format/huggingface/publish/remote.rs`)
stripped the owner off `repo_id` for the repo *name* but never sent an
`organization` field in the `POST /api/repos/create` body — HF then
silently defaults repo creation to the *token's own* namespace,
regardless of what `--repo` named. This "worked" for every prior
`chrishayuk/*` publish purely because `chrishayuk` is the token
owner; it broke the very first `larql/*` publish attempt (repo
created under `chrishayuk`, the next preupload call against
`larql/granite-4.1-3b` 404'd on a repo that was never actually
created there). Fixed by deriving `organization` from `repo_id`'s
owner and always sending it when present; 2 new tests pin the fix
directly against the pure body-building function
(`create_hf_repo_body`, made `pub(super)` for the sibling test file
per this crate's established plain-file-testing convention).

Upstream source provenance for this entry:
`ibm-granite/granite-4.1-3b`, revision
`c0650403e44e78ec0262dab1c90914c65b196c4e` — recovered from the local
HF hub cache's `refs/main` for that repo, and cross-checked against
the container's own (out-of-schema, pre-dating this initiative)
`index.json.model`/`derived_from_model` fields, which matched
character-for-character. Marked `Attestation::HandAttested { by:
"chrishayuk" }` per §7 item 4 above: this correlation is a real,
verified fact, but nothing in the `encode` pipeline's own documented
contract guarantees it (the encode-time gap `Provenance` exists to
name) — the honest marker is hand-attested, not mechanical, matching
this section's own "milestone 1 may hand-attest, if visibly marked"
decision.
