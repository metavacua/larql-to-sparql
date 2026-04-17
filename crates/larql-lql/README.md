# larql-lql

The LQL parser, executor, and REPL — SQL-like queries against vindexes.

## What is LQL?

LQL (LARQL Query Language) is a query language for vindexes. It looks like
SQL but operates on a transformer's weight matrices: you `EXTRACT` a model
into a vindex, `WALK`/`DESCRIBE`/`SELECT` to interrogate it, `INSERT`/
`DELETE`/`UPDATE` to edit knowledge through the patch overlay, and
`COMPILE` back to safetensors / GGUF when you're done.

```rust
use larql_lql::{Session, parse, run_repl};

let mut session = Session::new();
let stmt = parse(r#"USE "gemma3-4b.vindex";"#)?;
session.execute(&stmt)?;

let stmt = parse(r#"INFER "The capital of France is" TOP 5;"#)?;
let lines = session.execute(&stmt)?;
for line in &lines { println!("{line}"); }
```

The same `Session::execute` path drives the REPL (`larql repl`), the
vindexfile build executor, and the in-process embedding (Python bindings,
larql-server).

## Statement families

| Family | Statements | Backend |
|---|---|---|
| **Lifecycle** | `EXTRACT`, `COMPILE`, `DIFF`, `USE` | vindex / model |
| **Query** | `WALK`, `INFER`, `SELECT`, `DESCRIBE`, `EXPLAIN` | vindex (model weights for INFER) |
| **Mutation** | `INSERT`, `DELETE`, `UPDATE`, `MERGE` | patch overlay (base files never modified) |
| **Patch** | `BEGIN PATCH`, `SAVE PATCH`, `APPLY PATCH`, `SHOW PATCHES`, `REMOVE PATCH` | patch session |
| **Trace** | `TRACE`, `EXPLAIN INFER` | full forward pass with attribution |
| **Introspection** | `SHOW {RELATIONS, LAYERS, FEATURES, MODELS, PATCHES}`, `STATS` | metadata |
| **Pipe** | `<stmt> \|> <stmt>` | composition |

The full grammar is in `docs/lql-spec.md`. The user-facing tutorial is in
`docs/lql-guide.md`.

## INSERT: two modes

`INSERT INTO EDGES` has two install modes. The default is **`KNN`** — a
retrieval-override (Architecture B): the residual at the install layer
is stored as a key in the `KnnStore` alongside the target token, and
`INFER` overrides the model's top-1 when a stored key matches at
`cos > 0.75`. Scales freely (validated at 25K edges) with no cross-fact
interference.

**`COMPOSE`** is the FFN-overlay install — a single-layer slot written
via the `install_compiled_slot` pipeline (gate × 30, up parallel,
down = target-embed-unit × `d_ref` × `alpha`). Features participate in
the forward pass and can chain for multi-hop, but have a Hopfield-style
cap at ~5–10 facts per layer under template-shared prompts. Validated
end-to-end by `refine_demo` (10/10 retrieval, 0/4 bleed on Gemma 3 4B).

```sql
-- Default: KNN mode, residual captured at knowledge.hi − 1.
INSERT INTO EDGES (entity, relation, target)
    VALUES ("Atlantis", "capital-of", "Poseidon");

-- COMPOSE with explicit layer, confidence, alpha.
INSERT INTO EDGES (entity, relation, target)
    VALUES ("Atlantis", "capital-of", "Poseidon")
    AT LAYER 24
    CONFIDENCE 0.95
    ALPHA 0.30
    MODE COMPOSE;
```

Optional clauses (all independent):

| Clause | Default | What it does |
|---|---|---|
| `AT LAYER N` | `knowledge.hi − 1` (L26 on Gemma 4B) | Pins the install layer |
| `CONFIDENCE c` | 0.9 (Compose) / 1.0 (Knn) | Stored on the feature / key |
| `ALPHA a` | 0.10 | Compose only — per-layer down-vector scale (validated range ~0.05–0.30) |
| `MODE {KNN,COMPOSE}` | KNN | Retrieval-override vs FFN-overlay install |

After a batch of `COMPOSE` installs, run `REBALANCE` to fixed-point the
down-vector magnitudes into the `[FLOOR, CEILING]` probability band
across all installed facts jointly — per-INSERT local balance is
greedy and breaks past N ≈ 5 on template-shared prompts.

## COMPILE INTO VINDEX

`COMPILE CURRENT INTO VINDEX "out.vindex"` produces a real standalone vindex
with the inserted facts baked into the canonical `down_weights.bin`. No
sidecar, no overlay, no special loader code — `USE "out.vindex"` and
`INFER` works like any other vindex.

End-to-end on Gemma 4B (COMPOSE mode install):

```
INSERT Atlantis → Poseidon MODE COMPOSE      (single-layer at L26, α=0.10)
COMPILE CURRENT INTO VINDEX "out.vindex"
USE "out.vindex"
INFER "The capital of Atlantis is"  → Pose 56.91% ✓
INFER "The capital of France is"   → Paris 67.34% ✓ (preserved)
```

The constellation is sitting in the canonical `down_weights.bin` columns,
which is exactly what `weight_manifest.json` references. A subsequent
`COMPILE INTO MODEL FORMAT safetensors` exports those bytes verbatim, so
loading the result in HuggingFace Transformers gives you the inserted
facts via standard `model.generate()` — no special loader code.

`COMPILE INTO VINDEX` accepts one optional clause:

```sql
COMPILE CURRENT INTO VINDEX "out.vindex"
    ON CONFLICT FAIL;
```

**`ON CONFLICT`** — how to resolve slots written by more than one patch.

| Strategy | Behaviour |
|---|---|
| `LAST_WINS` (default) | Last applied patch wins. |
| `HIGHEST_CONFIDENCE` | Accepted for forward compatibility. Currently resolves like `LAST_WINS` for down vectors — see spec §3.5. |
| `FAIL` | Abort if any slot has a conflicting write. |

> **Status:** validated end-to-end on Gemma 3 4B in `experiments/14_vindex_compilation`.
> 10/10 retrieval, 0/4 regression bleed, standalone baked vindex (no overlay needed at runtime).
> The online refine pass (Gram-Schmidt against cached decoy residuals) runs at INSERT time, so
> no compile-time refine step is needed — INSERT already handles bleed defense.

The full mechanism is documented in `docs/vindex-operations-spec.md` §1.6.

## COMPILE INTO MODEL (MEMIT)

`COMPILE CURRENT INTO MODEL "out/" FORMAT safetensors` applies the patch
overlay to actual model weights via MEMIT closed-form weight editing. The
output is a standard safetensors directory — no vindex, no overlay, no
special loader code.

```sql
INSERT INTO EDGES (entity, relation, target)
    VALUES ("Atlantis", "capital", "Poseidon");
COMPILE CURRENT INTO MODEL "gemma3-4b-edited/" FORMAT safetensors;
```

The MEMIT pipeline:
1. Estimates FFN activation covariance C at the install layer
2. Captures per-fact activations k* at the canonical prompt
3. Solves the closed-form weight edit: `dW = R^T S^-1 Q` where
   `S = K C^-1 K^T + lambda*I`
4. Applies dW to W_down and writes modified weights

Requires model weights in the vindex (`EXTRACT ... WITH ALL`).
Validated in Python at 200/200 (100%) with multi-layer MEMIT on v11.

**Opt-in.** The MEMIT pass is gated behind `LARQL_MEMIT_ENABLE=1`.
Default is off because MEMIT cross-hijacks native facts on Gemma 3 4B
at every layer tested (the hourglass plateau L6–L28 makes
template-sharing key vectors indistinguishable to the closed-form
solve). Without the env var, `COMPILE INTO MODEL` writes the raw
loaded weights unchanged — use the `COMPOSE` column-replace path
(`COMPILE INTO VINDEX`) for the default Gemma install pipeline.
Extra tuning knobs: `LARQL_MEMIT_RIDGE=<f>` (default `0.1`),
`LARQL_MEMIT_TARGET_DELTA=1` (gradient-optimised delta, slower but
scales to N=200+), `LARQL_MEMIT_SPREAD=<n>` (distribute each fact
across N consecutive layers).

## Building & Testing

```bash
cargo test -p larql-lql                                       # 317 tests
cargo test -p larql-lql --lib executor::tests                 # executor suite
cargo test -p larql-lql --lib parser::tests                   # parser unit tests

# Synthetic demos (run in CI, no model download)
cargo run -p larql-lql --example parser_demo                   # AST output, every statement type
cargo run -p larql-lql --example lql_demo                      # 61-row spec compliance grid
cargo run --release -p larql-lql --example compact_demo        # LSM storage-tier walkthrough: INSERT → COMPACT MINOR → SHOW COMPACT STATUS

# Model-dependent demos (skip if output/gemma3-4b-f16.vindex absent)
cargo run --release -p larql-lql --example compile_demo        # End-to-end COMPILE INTO VINDEX on real Gemma 4B
cargo run --release -p larql-lql --example refine_demo         # 10-fact INSERT + COMPILE (exp 14 reproduction, 10/10 retrieval + 0 bleed)
cargo run --release -p larql-lql --example trace_demo          # TRACE variants: residual decomposition, FOR <token>, DECOMPOSE, POSITIONS ALL SAVE

# Criterion benches (use --quick for a fast sweep)
cargo bench  -p larql-lql --bench parser                       # parse_single × 18, parse_batch
cargo bench  -p larql-lql --bench executor                     # SELECT, SHOW, DELETE, UPDATE, patch lifecycle
cargo bench  -p larql-lql --bench compile                      # COMPILE INTO VINDEX bake cost
```

### Test coverage (313 tests)

- **Parser** (`parser/tests.rs`, 146 tests): every `Statement` variant,
  every clause combination, plus negative tests for malformed input.
- **Executor — no-backend errors** (`executor/tests.rs`): every variant
  that needs a vindex returns `LqlError::NoBackend` cleanly when no
  `USE` has run. Includes `TRACE`, `REBALANCE`, `COMPACT {MINOR,MAJOR}`,
  `SHOW COMPACT STATUS`, `SHOW ENTITIES`, `REMOVE PATCH`, and `PIPE`
  error propagation.
- **Executor — Weight backend**: `USE MODEL` path with synthetic
  weights, validates which statements work without a vindex.
- **Executor — end-to-end on synthetic vindex**: builds a vindex on
  disk, runs `USE` against it, exercises `DELETE`, `UPDATE`,
  `BEGIN PATCH`, `SAVE PATCH`, auto-patch lifecycle, `MERGE`,
  `SHOW ENTITIES`, `SHOW COMPACT STATUS`, `COMPACT MINOR` (empty-L0
  path), `REBALANCE` (empty-installs no-op), `REMOVE PATCH` error
  handling, `PIPE` concatenation, and the `TRACE` model-weights-hint
  error.
- **Executor — COMPILE INTO VINDEX**: conflict detection (`ON CONFLICT
  FAIL`/`LAST_WINS`), down override baking, structural compile with no
  patches, plus 6 unit tests for `patch_down_weights` covering f32/f16
  dtypes, multiple-feature/multiple-layer overrides, shape mismatch
  errors, and missing-source error paths (live in
  `executor/lifecycle/compile/bake.rs`).
- **Executor — MEMIT + balance**: fact collection from patches,
  deduplication, template-matched decoys, relation template generation,
  `COMPILE INTO MODEL` requires model weights.
- **Executor — MemitStore wiring**: `memit_store_mut` persistence so
  `COMPACT MAJOR` cycles accumulate across calls.
- **INSERT install math**: 8 unit tests in
  `mutation/insert/compose.rs` for `unit_vector`, `median_or`,
  `compute_layer_median_norms`, and the end-to-end
  `install_compiled_slot` activation math (GATE_SCALE, alpha payload).

### Bench measurements (typical machine)

| Bench | Operation | Time |
|---|---|---|
| `parser` | parse `STATS;` | **102 ns** |
| `parser` | parse 100-statement batch | **78 µs** (1.28 M stmts/s) |
| `executor` | `SELECT * FROM EDGES LIMIT 5;` | 425 ns |
| `executor` | `STATS;` | 19 µs |
| `executor` | `DELETE … WHERE layer/feature` (USE+DELETE) | 98 µs |
| `executor` | `BEGIN PATCH → DELETE → SAVE PATCH` | 136 µs |
| `compile` | `COMPILE INTO VINDEX` (no patches) | **1.84 ms** |
| `compile` | `COMPILE INTO VINDEX` (with `down_weights.bin`) | **2.41 ms** |

Run `cargo bench -p larql-lql` (without `--quick`) for the full criterion
sample sizes — HTML reports go to `target/criterion/`.

## Architecture

```
src/
├── ast.rs                AST — one Statement enum per LQL verb
├── lexer.rs              Tokeniser (case-insensitive keywords)
├── error.rs              LqlError + LqlError::exec helper
├── repl.rs               Interactive REPL + batch mode
├── relations.rs          RelationClassifier (probe / cluster labels)
├── parser/
│   ├── mod.rs            Parser entry point + dispatch
│   ├── helpers.rs        parse_value, parse_conditions, parse_assignments
│   ├── lifecycle.rs      EXTRACT, COMPILE, DIFF, USE, COMPACT
│   ├── query.rs          WALK, INFER, SELECT, DESCRIBE, EXPLAIN
│   ├── mutation.rs       INSERT (ALPHA, MODE), DELETE, UPDATE, MERGE, REBALANCE
│   ├── patch.rs          BEGIN/SAVE/APPLY/REMOVE PATCH, DIFF INTO PATCH
│   ├── trace.rs          TRACE
│   ├── introspection.rs  SHOW {RELATIONS, LAYERS, FEATURES, ENTITIES,
│   │                          MODELS, PATCHES, COMPACT STATUS}, STATS
│   └── tests.rs          146 parser tests
└── executor/
    ├── mod.rs              Session + execute() dispatch + patch session helpers
    ├── backend.rs          Backend enum (Vindex/Weight/Remote) + require_* accessors
    ├── helpers.rs          format_number, format_bytes, dir_size, content/readable token
    ├── compact.rs          COMPACT MINOR / MAJOR (L0 → L1 → L2 promotion)
    ├── remote.rs           HTTP forwarding for the Remote backend
    ├── trace.rs            TRACE executor
    ├── introspection.rs    SHOW + STATS + SHOW COMPACT STATUS + SHOW ENTITIES
    ├── lifecycle/
    │   ├── mod.rs          submodule declarations
    │   ├── use_cmd.rs      USE {path | MODEL | REMOTE}
    │   ├── extract.rs      EXTRACT
    │   ├── stats.rs        STATS
    │   ├── diff.rs         DIFF [INTO PATCH]
    │   └── compile/
    │       ├── mod.rs          exec_compile dispatch + shared MEMIT fact collector
    │       ├── into_model.rs   COMPILE ... INTO MODEL (MEMIT-gated)
    │       ├── into_vindex.rs  COMPILE ... INTO VINDEX + collision detection
    │       └── bake.rs         patch_{down,gate,up}_weights + apply_memit_deltas + tests
    ├── query/
    │   ├── mod.rs          shared resolve_bands helper
    │   ├── walk.rs         WALK
    │   ├── infer.rs        INFER
    │   ├── describe.rs     DESCRIBE (with MoE router path + describe_* helpers)
    │   ├── select.rs       SELECT {EDGES, FEATURES, ENTITIES} + NEAREST TO
    │   ├── explain.rs      EXPLAIN WALK
    │   └── infer_trace.rs  EXPLAIN INFER (attention + logit-lens rendering)
    ├── mutation/
    │   ├── mod.rs          submodule declarations
    │   ├── delete.rs       DELETE
    │   ├── update.rs       UPDATE
    │   ├── merge.rs        MERGE
    │   ├── rebalance.rs    REBALANCE (global fixed-point balance)
    │   └── insert/
    │       ├── mod.rs      exec_insert orchestrator (plan → capture → install → balance)
    │       ├── knn.rs      MODE KNN (KnnStore retrieval override)
    │       ├── plan.rs     Phase 1 — target embed + layer selection
    │       ├── capture.rs  Phase 1b — canonical + decoy residual capture
    │       ├── compose.rs  Phase 2 — install_slots + cliff-breaker refine + tests
    │       └── balance.rs  Phase 3 — per-fact balance + cross-fact regression check
    └── tests.rs            93 executor integration tests (+ 17 in-module
                             unit tests across lifecycle/compile/bake,
                             lifecycle/compile/into_vindex, and
                             mutation/insert/compose)
```

## Public API

```rust
pub use ast::Statement;
pub use error::LqlError;
pub use executor::Session;
pub use parser::parse;
pub use repl::{run_batch, run_repl, run_statement};
```

The lexer, parser internals, and executor backend types are
`pub(crate)` — only the four entry points above are stable surface.
