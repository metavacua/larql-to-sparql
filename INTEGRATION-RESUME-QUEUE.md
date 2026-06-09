# Integration Phase 0–1 — Merge stood up + dependency-ordered resolution queue (2026-06-08)

**Worktree:** `/home/metavacua/larql-integration-wt` (branch `integration/chrishayuk-base`, off
`upstream/main` @ cac5c8e0). **Merge:** `git merge ianblenke/main` → conflicted, **not committed**
(the PR #129 direction: chrishayuk base ← ianblenke head). Upstream commits untouched.

## Phase 0 results
- **Conflicts: 96** = 56 content (UU) + 35 modify/delete (21 UD + 14 DU) + 5 add-type (1 AA + 4 UA).
  Matches the pre-merge prediction (56/35/1) almost exactly.
- **Additive features: 138/138** ianblenke model files present in the merged tree; **only 1 conflicts**
  (`larql-models/src/quant/ggml/q5_k.rs`). DSv4 (2), Qwen3.5 (27), speculative (19) all landed
  essentially free. ✓ additive hypothesis confirmed empirically.
- Conflict sets saved: `/tmp/c-content.txt` (56), `/tmp/c-moddel.txt` (35), `/tmp/c-add.txt` (5),
  `/tmp/c-du.txt` (14), `/tmp/c-ud.txt` (21).

## Retarget map — the 35 non-survivors

### DU (14) — chrishayuk DELETED, ianblenke MODIFIED → retarget onto chrishayuk's consolidated home
| ianblenke file(s) | retarget destination (chrishayuk) | note |
|---|---|---|
| `kv-cache-benchmark/src/accuracy_suite/mod.rs` (+8 more `kv-cache-benchmark/*`) | `larql-kv/src/accuracy_suite/*` (+ `examples/`) | resolved decision; content reconcile |
| `larql-inference/src/vindex/q4k_forward/{dequant,hidden,mod}.rs` | `larql-compute/src/kquant_forward/` | substrate move (ADR-0022); confirm exact module per-file |
| `larql-models/src/loading/gguf.rs` | `larql-models/src/loading/gguf/{loader,parser,reader,writer,types,orient,constants}.rs` | GGUF modularization (8 modules) |
| `larql-server/src/routes/walk_ffn.rs` | `larql-compute/src/kquant_forward/walk_ffn.rs` | substrate move |

### UD (21) — chrishayuk MODIFIED, ianblenke DELETED → keep chrishayuk; verify no lost improvement
- `larql-vindex/src/extract/streaming/*` (10) + `format/weights/load/*` (4) — chrishayuk's
  streaming-extract + weights-load structure; keep chrishayuk; ianblenke's vindex work arrives via
  the UU content conflicts / additive files.
- `larql-inference/src/layer_graph/generate/gpu/*` (4) — chrishayuk's GPU generate path; keep;
  ianblenke's GPU work arrives via the additive CUDA / dsv4-gpu files.
- `larql-models/src/detect/{config_io,parser,tests}.rs` (3) — keep chrishayuk.

## Dependency-ordered resolution queue (sinks first)
1. **larql-models** — 7 UU + 4 UD + gguf-retarget (DU) + `q5_k.rs` (add). Foundation; gates features.
2. **larql-compute** — 7 UU. Substrate; receives the q4k_forward + walk_ffn retargets.
3. **larql-vindex** — 27 UU + 14 UD. Depends on compute/models.
4. **larql-inference** — 27 UU + gpu UD + q4k DU. Depends on vindex/compute/models.
5. **larql-server / larql-cli / larql-lql / larql-router-protocol** — ~6 + add-type. Downstream consumers.
6. **larql-kv accuracy reconciliation** — 9 kv-cache-benchmark DU → fold into `larql-kv/src/accuracy_suite/`. Last (consumer of the reconciled larql-kv).

Each conflict → **one additive atomic conventional resolution commit**. The merge's two parents
(`upstream/main`, `ianblenke/main`) stay pristine → bidirectional upstream sync preserved.

## Review gate (Phase 1 stop)
Stages 1–6 are the expensive tier (~92 reconciliations; surgery in the competing cores
larql-vindex/larql-inference). Proceed?
