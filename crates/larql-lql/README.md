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

## INSERT and the multi-layer constellation

`INSERT INTO EDGES` installs a multi-layer constellation by default — the
validated regime from `docs/training-free-insert.md` (8 layers × `alpha=0.25`).
Single-layer installs at this alpha don't move the logits enough; raising
alpha breaks neighbouring facts. There is no single-layer mode.

```sql
-- Default form: spans the upper half of the knowledge band.
INSERT INTO EDGES (entity, relation, target)
    VALUES ("Atlantis", "capital-of", "Poseidon");

-- Centered on a specific layer (8-layer span around L24, clamped to
-- valid range), with explicit confidence and alpha override.
INSERT INTO EDGES (entity, relation, target)
    VALUES ("Atlantis", "capital-of", "Poseidon")
    AT LAYER 24
    CONFIDENCE 0.95
    ALPHA 0.30;
```

The three optional clauses are independent and can be combined:

| Clause | Default | What it does |
|---|---|---|
| `AT LAYER N` | upper half of knowledge band | Centers the 8-layer constellation span on layer N |
| `CONFIDENCE c` | 0.9 | Stored on the inserted features |
| `ALPHA a` | 0.25 | Per-layer down-vector scale (validated range ~0.10–0.50) |

## COMPILE INTO VINDEX

`COMPILE CURRENT INTO VINDEX "out.vindex"` produces a real standalone vindex
with the inserted facts baked into the canonical `down_weights.bin`. No
sidecar, no overlay, no special loader code — `USE "out.vindex"` and
`INFER` works like any other vindex.

End-to-end on Gemma 4B:

```
INSERT Atlantis → Poseidon         (8 layers × alpha=0.25)
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

`COMPILE INTO VINDEX` accepts an optional `ON CONFLICT` clause for
choosing how to resolve `(layer, feature)` slots written by more than
one applied patch:

```sql
COMPILE CURRENT INTO VINDEX "out.vindex" ON CONFLICT FAIL;
```

| Strategy | Behaviour |
|---|---|
| `LAST_WINS` (default) | Last applied patch wins. |
| `HIGHEST_CONFIDENCE` | Accepted for forward compatibility. Currently resolves like `LAST_WINS` for down vectors — see spec §3.5. |
| `FAIL` | Abort if any slot has a conflicting write. |

The full mechanism is documented in `docs/vindex-operations-spec.md` §1.6.

## Building & Testing

```bash
cargo test -p larql-lql                                       # 242 tests
cargo test -p larql-lql --lib executor::tests                 # executor mutation pipeline
cargo test -p larql-lql --lib parser::tests                   # parser unit tests

# Demos
cargo run -p larql-lql --example parser_demo                   # AST output, every statement type
cargo run -p larql-lql --example lql_demo                      # 42-row spec compliance grid
cargo run --release -p larql-lql --example compile_demo        # End-to-end COMPILE INTO VINDEX
                                                                #   on real Gemma 4B (skips if absent)

# Criterion benches (use --quick for a fast sweep)
cargo bench  -p larql-lql --bench parser                       # parse_single × 18, parse_batch
cargo bench  -p larql-lql --bench executor                     # SELECT, SHOW, DELETE, UPDATE, patch lifecycle
cargo bench  -p larql-lql --bench compile                      # COMPILE INTO VINDEX bake cost
```

### Test coverage (242 tests)

- **Parser** (`parser/tests.rs`, 1,500+ lines): every statement type and
  every clause combination, plus negative tests for malformed input.
- **Executor — no-backend errors**: every statement type returns
  `LqlError::NoBackend` cleanly when no `USE` has run.
- **Executor — Weight backend**: `USE MODEL` path with synthetic weights,
  validates which statements work without a vindex.
- **Executor — mutation pipeline**: builds a synthetic vindex on disk,
  runs `USE` against it, exercises `DELETE`, `UPDATE`, `BEGIN PATCH`,
  `SAVE PATCH`, auto-patch lifecycle, and `MERGE` error paths.
- **`COMPILE INTO VINDEX` byte baker**: 6 unit tests for `patch_down_weights`
  covering f32/f16 dtypes, multiple-feature/multiple-layer overrides,
  shape mismatch errors, and missing-source error paths.

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
│   ├── lifecycle.rs      EXTRACT, COMPILE, DIFF, USE
│   ├── query.rs          WALK, INFER, SELECT, DESCRIBE, EXPLAIN
│   ├── mutation.rs       INSERT (with ALPHA), DELETE, UPDATE, MERGE
│   ├── patch.rs          BEGIN/SAVE/APPLY/REMOVE PATCH, DIFF INTO PATCH
│   ├── trace.rs          TRACE
│   ├── introspection.rs  SHOW {RELATIONS, LAYERS, FEATURES, MODELS, PATCHES}, STATS
│   └── tests.rs          1,500+ parser tests
└── executor/
    ├── mod.rs            Session, Backend (Vindex/Weight/Remote), execute dispatch
    ├── helpers.rs        format_number, format_bytes, dir_size, content/readable token
    ├── lifecycle.rs      USE, EXTRACT, COMPILE (incl. patch_down_weights baker), DIFF
    ├── query.rs          WALK, INFER, SELECT, DESCRIBE (with describe_* helpers)
    ├── mutation.rs       INSERT (constellation install), DELETE, UPDATE, MERGE
    ├── trace.rs          TRACE / EXPLAIN INFER (with build_attention_map etc.)
    ├── introspection.rs  SHOW + STATS
    ├── remote.rs         HTTP forwarding (remote_get_json / remote_post_json helpers)
    └── tests.rs          52 executor tests
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
