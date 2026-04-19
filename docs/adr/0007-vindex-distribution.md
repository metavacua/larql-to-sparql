# ADR-0007 — Vindex Distribution: slice, publish, collections, skip-if-unchanged

**Status:** Implemented
**Depends on:** ADR-0005 (FFN-Service Memory Bounds), ADR-0006 (Q4_K Remote FFN)
**Relates to:** ADR-0003 (FFN Router), ADR-0004 (FFN Grid)

---

## Context

ADR-0005 and ADR-0006 produced a split-tier inference topology: clients
run attention locally and delegate FFN to a server, or to a router that
fans out to a shard grid. For that topology to be usable by anyone who
didn't extract the model themselves, the built artefacts have to be
distributable — ideally as HuggingFace repos that a laptop can pull in
pieces.

The minimum viable story is "upload one big vindex". That works for
`INFER` but wastes bandwidth: a client doesn't need FFN weights, a
server doesn't need attention weights, a browse-only consumer doesn't
need either. Pulling the full repo and discarding 60% of it on every
machine is the wrong shape.

Three concrete problems:

1. **One extract, many shapes.** The client/server split wants two
   disjoint subsets of the same source vindex, plus a third browse-only
   view for DESCRIBE/WALK users. Re-extracting from the safetensors
   source for each shape wastes compute and introduces drift.

2. **Discovery across four repos.** Publishing a full vindex + three
   slices means four separate HF repos. Without a landing page a user
   hitting `hf://chrishayuk/gemma-4-31b-it-vindex` has no way to
   discover the `-client` sibling, and it's not obvious which one they
   need.

3. **Re-publishing is expensive.** A 27 GB server slice uploaded via
   plain HTTP takes minutes. Most re-publishes are incremental — the
   gate vectors didn't change, only the index.json bumped a version.
   Re-transferring the whole payload every time is unnecessary.

---

## Decision

Three layered primitives, each composable on its own:

1. **`larql slice`** — carve a built vindex into deployment variants
   without re-extracting. Pure file I/O plus an `index.json` rewrite.

2. **`larql publish`** — upload the full vindex **and** every sibling
   slice to HuggingFace as separate repos, then file them into
   **collections** so discovery works from a single landing page.

3. **Skip-if-unchanged** — each upload compares the local SHA256
   against the remote `lfs.oid`. Files that already match skip the
   transfer entirely.

The three composition layers ship behind one command:

```bash
larql publish gemma4-31b.vindex --repo chrishayuk/gemma-4-31b-it-vindex
```

One invocation → four repos + three nested collections. Re-runs are
near-free when nothing changed.

---

## `larql slice` — deployment variants

`crates/larql-cli/src/commands/primary/slice_cmd.rs` exposes
`slice_vindex(src, dst, parts, force, dry_run) -> Result<SliceOutcome>`
as the testable core; `run()` is a thin CLI wrapper with progress
prints.

### Parts catalogue

Each part matches one or more filename patterns. `index.json` is always
copied regardless of the part set.

| Part | Files |
|---|---|
| `embed` | `embeddings.bin` |
| `norms` | `norms.bin` |
| `attn` | `attn_weights*.bin` (includes q4/q4k/q8 variants + manifests) |
| `gate` | `gate_vectors.bin`, `gate_vectors_q4.bin` |
| `down_meta` | `down_meta.bin`, `down_meta.jsonl` |
| `ffn` | `interleaved*.bin` + manifests, `up_weights.bin`, `down_weights.bin`, `up_features.bin`, `down_features.bin` |
| `lm_head` | `lm_head*.bin` |
| `router` | `router_weights.bin` |
| `tokenizer` | `tokenizer.json` |
| `manifest` | `weight_manifest.json` |
| `labels` | `feature_labels.json`, `feature_clusters.jsonl`, `relation_clusters.json` |
| `readme` | `README.md` |

### Presets

| Preset | Parts | Pairs with |
|---|---|---|
| `client` | embed + norms + attn + tokenizer + manifest + labels | `larql run --ffn URL` |
| `server` | embed + norms + gate + down_meta + ffn + tokenizer + manifest + labels | `larql serve --ffn-only` |
| `browse` | embed + gate + down_meta + tokenizer + labels + readme | DESCRIBE / WALK / SELECT (no forward pass) |
| `router` | router + tokenizer + manifest + labels + readme | MoE router (ADR-0003) |
| `all` | every part | full clone under a different name |

### `index.json` rewrite

On every slice the destination's `index.json` gets rewritten so
`extract_level` and `has_model_weights` match what's on disk:

- **`extract_level`** is set to the strongest tier actually present
  (`Browse` / `Attention` / `Inference` / `All`), and never higher than
  the source level. A client slice from an Inference-tier source thus
  downgrades to `Attention`.
- **`has_model_weights`** is true whenever attention OR FFN compute
  weights are kept. This is load-bearing: the Q4K loader
  (`load_model_weights_q4k`) refuses to open a vindex whose config
  advertises `has_model_weights: false`, so setting it correctly on
  an attention-only client slice is what lets `larql run --ffn URL`
  load it at all.

### The empty-gate loader relaxation

A `client`-preset slice contains no `gate_vectors.bin` and no
`interleaved_q4k.bin` — the client delegates gate-KNN to the server.
Before this change, `VectorIndex::load_vindex` rejected that layout
with:

```
parse error: neither gate_vectors.bin nor interleaved_q4k.bin present
```

`crates/larql-vindex/src/format/load.rs` now synthesises an empty
anonymous mmap with all-zero slices when both gate sources are absent.
`gate_knn` on that index returns an empty result — correct for
attention-only clients, which never call it. Tests in
`crates/larql-vindex/tests/test_vindex.rs ::
load_vindex_synthesises_empty_gate_when_both_sources_absent` pin the
behaviour.

### Sibling preset memory bounds (measured)

All on Gemma 4 31B Q4_K, macOS:

| Slice | On-disk | Pair command | Notes |
|---|---|---|---|
| full | 32 GB | `larql run` | baseline |
| `client` | 7.4 GB | `larql run --ffn URL` | 4.3× smaller than full |
| `server` | 27 GB | `larql serve --ffn-only` | no attention, still has embed+norms so the Q4K loader opens |
| `browse` | 16 GB | `larql lql 'DESCRIBE …'` | no FFN, no attention |

---

## `larql publish` — four repos + three collections

`crates/larql-cli/src/commands/primary/publish_cmd.rs` stages each
slice in a temp directory via `slice_vindex`, uploads via
`larql_vindex::publish_vindex_with_opts`, and finally calls
`larql_vindex::ensure_collection` for each requested collection level.

### Repo naming

Default template: `{repo}-{preset}`. The full vindex goes to `{repo}`.
For `chrishayuk/gemma-4-31b-it-vindex`:

```
chrishayuk/gemma-4-31b-it-vindex          (full)
chrishayuk/gemma-4-31b-it-vindex-client   (attention only)
chrishayuk/gemma-4-31b-it-vindex-server   (FFN only)
chrishayuk/gemma-4-31b-it-vindex-browse   (DESCRIBE / WALK)
```

Override with `--slice-repo-template "{repo}/{preset}"` (folder-style)
or `--slice-repo-template "{repo}_{preset}"` (underscore separator).
The templating supports any layout HF accepts.

### Collections

Three nested levels, all auto-derived from the vindex's `model` field:

| Level | Title | Holds |
|---|---|---|
| `model` | `Gemma 4 31B It — LARQL Vindex` | the four sibling repos for this model |
| `family` | `Gemma Family — LARQL Vindexes` | every model of this architecture you've published |
| `library` | `LARQL Vindex Library` | every vindex you've ever published |

The hierarchy isn't enforced by HF — the same repo appears in all three
collections. That's the point: someone landing on the family page sees
every Gemma you've uploaded; someone on the model page sees the four
deployment variants for one size.

### `ensure_collection` idempotency

`crates/larql-vindex/src/format/huggingface.rs::ensure_collection`:

```rust
pub fn ensure_collection(
    namespace: &str,
    title: &str,
    description: Option<&str>,
    items: &[CollectionItem],
) -> Result<String, VindexError>  // returns collection URL
```

1. `GET /api/users/{namespace}/collections?limit=100` — list existing
   collections.
2. Case-insensitive title match → reuse slug if found, otherwise
   `POST /api/collections` to create.
3. For each item: `POST /api/collections/{slug}/item`. HTTP 409
   ("already in collection") is treated as success.

Re-publishing the same vindex is safe: the `model` collection is found
by title, the four items are already present and yield 409s, the
family and library collections accrete entries as new models land.

### Model title / family derivation

The `model` field in `index.json` can be:

- `google/gemma-4-31b-it` (clean HF form)
- `/Users/.../models--google--gemma-4-31B-it/snapshots/abc/` (HF cache layout)
- `gemma-3-4b-it` (already short)

`short_model_name` handles all three, including the `models--{owner}--{name}`
prefix pattern that trips up a naive `rsplit('/')`. `default_model_title`
title-cases segments and `default_family` stops at the first digit-leading
segment (`Gemma 4 31B It` → family `Gemma`). Callers override with
`--model-title` / `--family` when the auto-derivation reads awkwardly
(e.g. "Gemma 4 31B **Instruct**" vs "...It").

---

## Skip-if-unchanged — SHA256 vs `lfs.oid`

`PublishOptions { skip_unchanged: bool }` drives per-file upload
decisions in `publish_vindex_with_opts`. When on (CLI default unless
`--force-upload`):

1. `fetch_remote_lfs_oids(repo, token)` hits
   `/api/datasets/{repo}/tree/main?recursive=true` and extracts every
   entry's `lfs.oid`. This field exists iff the file is
   LFS-tracked — i.e. a "big" binary like `gate_vectors.bin`.
2. Per file about to upload: compute local SHA256 via
   `format::checksums::sha256_file`. If the local hash equals the
   remote `lfs.oid`, call `PublishCallbacks::on_file_skipped(name,
   size, sha)` and move on.
3. Anything else (no remote entry, git-tracked without `lfs.oid`,
   tree API errored): upload normally.

### Why LFS-only

Small files (`index.json`, manifests) are git-tracked on HF. The git
blob SHA-1 format is `blob {size}\0{content}` hashed, which isn't
directly comparable to the file-content SHA256 without a separate
hash. Computing it is tractable but adds complexity for files that
total a few KB anyway — the win doesn't justify the code. Always
re-uploading them is cheap.

The practical payoff lands on the big stuff: re-publishing a 27 GB
server slice where nothing changed transfers only the manifests, not
the gate + interleaved FFN blobs.

### Graceful degradation

`fetch_remote_lfs_oids` returns `Ok(HashMap::new())` on:

- HTTP 404 (brand-new repo, just created, empty)
- JSON parse failure (HF API change, corruption)
- Network error

The upshot: if anything goes wrong with the index fetch, `publish`
silently falls back to "upload everything" rather than aborting.
Correctness is preserved at the cost of a wasted upload on a transient
failure.

---

## Flag surface summary

| Flag | Default | Effect |
|---|---|---|
| `--full` / `--no-full` | `--full` | Upload the full vindex to `--repo` |
| `--slices a,b,c` | `client,server,browse` | Which presets to upload as siblings; `none` to skip |
| `--slice-repo-template T` | `{repo}-{preset}` | Sibling naming; `{repo}` and `{preset}` substitute |
| `--collections a,b,c` | `model,family,library` | Which collections to create/update; `none` to skip |
| `--model-title T` | derived | Override the per-model collection title |
| `--family F` | derived | Override the family collection's grouping |
| `--library-title T` | `LARQL Vindex Library` | Override the top-level collection title |
| `--force-upload` | off | Bypass SHA256 skip; re-upload every file |
| `--tmp-dir D` | system temp | Where to stage intermediate slices |
| `--dry-run` | off | Print the plan; no repos created, no files uploaded |

---

## Implementation files

| File | Role |
|---|---|
| `crates/larql-cli/src/commands/primary/slice_cmd.rs` | `slice_vindex`, `Part`, `preset_parts`, CLI wrapper |
| `crates/larql-cli/src/commands/primary/publish_cmd.rs` | `larql publish`: slice orchestration, collection composition, skip plumbing |
| `crates/larql-cli/src/commands/extraction/hf_cmd.rs` | `larql hf publish` (simpler one-repo publish); shares `PublishCallbacks` |
| `crates/larql-vindex/src/format/huggingface.rs` | `publish_vindex`, `publish_vindex_with_opts`, `PublishOptions`, `fetch_remote_lfs_oids`, `ensure_collection`, `CollectionItem`, `PublishCallbacks::on_file_skipped` |
| `crates/larql-vindex/src/format/load.rs` | Empty-gate synthesis when both gate source files are absent |
| `crates/larql-vindex/src/format/checksums.rs` | `sha256_file` (reused from pre-existing checksum infra) |

---

## Open questions

1. **Git SHA-1 parity for small files.** Computing `git blob {size}\0{content}`
   SHA-1 locally would let us skip re-uploads of unchanged `index.json` too.
   The win is ~KB per small file. Deferred until measurement shows it
   matters — a diff of the HF git tree between publishes is usually more
   useful than skipping them.

2. **Collection description drift.** `ensure_collection` sets a description
   on create but doesn't reconcile on subsequent runs. If we want the
   description to track a field in the vindex config, the helper needs a
   `PATCH /api/collections/{slug}` call. Today's behaviour is fine for
   fire-and-forget publishes; the nit matters when collection descriptions
   evolve.

3. **Manifest-level skip.** Instead of per-file SHA256 compares we could
   publish a single `manifest.json` that lists every file's SHA256; a
   re-publish that finds the manifest unchanged could skip the whole
   repo. That's belt-and-suspenders on top of the current scheme and
   only matters under heavy re-publish load.

4. **Slicing a router vindex.** The `router` preset produces a <1 GB
   artefact for a dense model (no `router_weights.bin` present) — essentially
   empty. It's harmless but not useful. Either auto-skip the `router`
   slice when the source is dense, or keep the current explicit opt-in.
   Current decision: explicit opt-in (router isn't in the default slice
   list), but the slice itself shouldn't error if the source lacks router
   weights — an empty router slice is correct output for a dense model.

5. **Skip + collection ordering.** Collections are always updated, even
   when every file in every repo was skipped. That's intentional — the
   title might have changed, or the notes on items might have shifted.
   If it ever becomes expensive, add a `--skip-collection-update` flag.
