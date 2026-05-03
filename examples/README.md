# LARQL Examples

Practical examples demonstrating LARQL's Python API and LQL query language.

## Quick Start

### 1. Extract a Model

```bash
larql extract-index google/gemma-2-2b-it -o output/gemma-2-2b-it.vindex --level all --f16
```

### 2. Run an Example

```bash
# Knowledge query (no inference needed)
python3 examples/knowledge.py output/gemma-2-2b-it.vindex

# Insert knowledge facts
python3 examples/insert.py output/gemma-2-2b-it.vindex

# Run inference
python3 examples/infer.py output/gemma-2-2b-it.vindex

# LQL session
python3 examples/session.py output/gemma-2-2b-it.vindex

# **NEW: Full round-trip workflow**
python3 examples/round_trip_workflow.py output/gemma-2-2b-it.vindex output/round-trip-result
```

---

## Examples

### `knowledge.py` — Knowledge Queries (No Inference)

Demonstrates DESCRIBE, WALK, and relation introspection — zero-copy operations that don't require inference.

```python
vindex = larql.load("model.vindex")

# What does the model know about France?
edges = vindex.describe("France")
for e in edges[:5]:
    print(f"{e.relation} → {e.target} (score={e.gate_score:.0f})")
# Output:
#   capital → Paris (score=1437)
#   language → French (score=1289)
#   location → Europe (score=987)
```

**Use case:** Rapid knowledge exploration, zero latency.

---

### `insert.py` — Insert Knowledge (No Training)

Add new facts to the model without fine-tuning. Changes are stored as structural edits (patches).

```python
vindex.insert("Colchester", "country", "England")
vindex.insert("Colchester", "region", "Essex")

# Verify
edges = vindex.describe("Colchester")
```

**Use case:** Knowledge base corrections, domain-specific facts.

**How it works:**
1. Entity embedding + relation cluster centre → location in knowledge layer
2. Synthesize gate vector (encodes fact identity)
3. Copy target embedding to down vector (encodes target token)
4. Store as patch overlay (base weights untouched)

---

### `infer.py` — Inference (Rust Attention + Walk FFN)

Run end-to-end inference on prompts. Weights are mmap'd for memory efficiency.

```python
result = vindex.infer("The capital of France is", top_k_predictions=3)
# [("Paris", 0.805), ("London", 0.058), ("Berlin", 0.041)]
```

**Memory:** ~450 MB load RSS for 4B model (vs 18 GB native)  
**Speed:** ~100-200 ms/token (warm OS page cache)

---

### `session.py` — LQL Session

Use the Lazarus Query Language with direct numpy array access.

```python
session = larql.session("model.vindex")

# Execute LQL statements
result = session.query("DESCRIBE 'France'")

# Direct access to underlying vindex (numpy)
v = session.vindex
gates = v.gate_vectors(layer=24)  # numpy (10240, 2560)
```

---

### `round_trip_workflow.py` — Complete Workflow ⭐

**NEW:** End-to-end demonstration of structured knowledge editing.

The full pipeline: extract → insert → compile → infer → verify.

```bash
python3 examples/round_trip_workflow.py \
  output/gemma-2-2b-it.vindex \
  output/round-trip-result
```

**Output:**
- Baseline inference results (before editing)
- Inserted facts and their storage locations
- Compiled vindex (standalone, no patches needed)
- Inference results on edited model
- JSON report comparing before/after

**Demonstrates:**
1. Loading a vindex
2. Inserting knowledge (no training)
3. Compiling patches into a new vindex
4. Running inference on the edited model
5. Verifying that edits took effect

**Time:** ~2 minutes (inference level) to 5 minutes (full level)

---

## Performance Tips

### Fast Knowledge Queries
Use `browse` level (gate vectors + embeddings only):
```bash
larql extract-index google/gemma-2-2b-it \
  -o model.vindex \
  --level browse  # ~500 MB, <1s load
```

Then use DESCRIBE/WALK without inference overhead:
```python
edges = vindex.describe("France")  # ~5 ms
hits = vindex.walk("...", top_k=10)  # ~20 ms
```

### Heavy Inference
Use `all` level with `--f16`:
```bash
larql extract-index google/gemma-2-2b-it \
  -o model.vindex \
  --level all --f16  # ~1.2 GB, ~3s load
```

First call warms the OS page cache; subsequent calls are faster.

### Bulk Operations
For many insertions, batch them before compiling:
```python
for entity, relation, target in facts:
    vindex.insert(entity, relation, target)

# Then compile once
vindex.compile_vindex("result.vindex")
```

---

## CI/CD

The `round_trip_workflow.py` example is tested automatically in CI:

**Fast path (every PR):**
- LQL parsing/execution tests
- Python binding tests (synthetic vindex)
- Runs in ~30 seconds

**Full path (manual trigger / weekly):**
- Extract Gemma 2B-IT
- Run `round_trip_workflow.py`
- Generate and archive report
- Runs in ~45 minutes

Trigger full tests with:
1. `[test-full-workflow]` in commit message
2. GitHub Actions `workflow_dispatch` button

---

## Architecture Notes

### Vindex Files
The vindex is a directory of mmap'd binary files:
```
model.vindex/
  metadata.json        # Model name, layers, hidden size
  gate_weights.bin     # Gate vectors (mmap'd)
  embeddings.bin       # Token embeddings (mmap'd)
  down_weights.bin     # Down vectors (rewritten on compile)
  probe_metadata/      # Feature labels and confidence scores
```

### Patches
Edits are stored separately as JSON overlays (`.vlp` files):
```
model.vlp/
  L24.json            # Edits to layer 24 (gate + down)
  L25.json            # Edits to layer 25
```

No modification to base files — patches stack like layers.

### Compilation
`compile_vindex()` creates a new standalone vindex:
1. Hardlink base weight files (APFS fast path)
2. Rewrite `down_weights.bin` column-wise with patch data
3. Fold metadata
4. No dependencies on original patches

---

## Related Documentation

- **Full round-trip guide:** [docs/round-trip-workflow.md](../docs/round-trip-workflow.md)
- **Python API reference:** [docs/larql-python.md](../docs/larql-python.md)
- **LQL language spec:** [docs/specs/lql-spec.md](../docs/specs/lql-spec.md)
- **Vindex file format:** [docs/specs/vindex-format-spec.md](../docs/specs/vindex-format-spec.md)

