# LARQL Round-Trip Workflow

**Status:** Phase 1-3 implementation (extract + edit + compile + infer)  
**Model:** Gemma 2B-IT (Apache 2.0 licensed, ~2.2B parameters)  
**Estimated Time:** 2-5 minutes (fast path) to 30-60 minutes (full extraction + inference)

---

## Overview

The round-trip workflow demonstrates LARQL's core capability: **structured knowledge editing without fine-tuning**.

```
Extract Model
    ↓
Load Vindex (mmap'd, <1s)
    ↓
Insert Knowledge (0-copy patches)
    ↓
Compile to Model Weights (hardlink + rewrite down_weights)
    ↓
Run Inference (verify edits took effect)
    ↓
Compare Before/After
```

Each step is **reversible** — patches live in separate `.vlp` JSON files, base weights never change.

---

## Quick Start

### 1. Extract a Model (one-time, ~2-5 GB)

```bash
# Extract Gemma 2B-IT at browse level (gate + embeddings only, no inference weights)
larql extract-index google/gemma-2-2b-it \
  -o output/gemma-2-2b-it.vindex \
  --level browse \
  --f16

# Or with full weights (for inference + compilation)
larql extract-index google/gemma-2-2b-it \
  -o output/gemma-2-2b-it.vindex \
  --level all \
  --f16
```

**Levels:**
- `browse`: Gate vectors + embeddings. Fast queries (DESCRIBE, WALK). ~500 MB.
- `inference`: + FFN weights for inference. ~1.2 GB.
- `all`: Complete model. For compilation. ~2.2 GB.

### 2. Run the Round-Trip Example

```bash
# From the repo root
python3 examples/round_trip_workflow.py \
  output/gemma-2-2b-it.vindex \
  output/round-trip-result
```

**Output:**
- `output/round-trip-result/report.json` — JSON report with baseline vs. edited inferences
- Console output showing each phase

---

## What the Example Does

### Phase 1: Load & Baseline
Load the vindex and run inference on baseline prompts to establish expectations:
```
Q: "The capital of France is"  
A: "Paris" (78%)

Q: "Python was created by"  
A: "Guido" (65%)
```

### Phase 2: Insert Knowledge
Add new facts to the vindex without fine-tuning:
```python
vindex.insert("LARQL Test", "is_a", "knowledge_graph_system")
vindex.insert("LARQL", "language", "Rust")
```

Facts are stored as:
- **Gate vector** (2560 floats): encodes the entity + relation (synthesized from embeddings)
- **Down vector** (2560 floats): encodes the target token (copied from embedding)
- Both injected into the knowledge layer (~L24-27 for Gemma)

### Phase 3: Compile
Bake patches into a new standalone vindex:
```python
compiled_path = vindex.compile_vindex("output/edited-model.vindex")
```

This creates a new vindex where:
- Base weight files are hardlinked (APFS fast path on macOS, copy on Linux)
- `down_weights.bin` is rewritten column-wise with edits
- All `.vlp` patches are folded in
- New vindex is fully standalone (can be published, distributed, etc.)

### Phase 4: Verify
Run inference on the compiled vindex and check that edits took effect:
```
Q: "LARQL Test is_a"  
A: "knowledge_graph_system" (higher probability than baseline)
```

### Phase 5: Report
JSON report saved with:
- Baseline inferences (before editing)
- Compiled inferences (after editing)
- Side-by-side comparison
- Inserted facts + their storage locations

---

## Key Invariants

### Immutability
Base vindexes are **read-only**. All mutations go through `PatchedVindex` (overlay):

```python
# This auto-starts a patch
layer, feat = vindex.insert(entity, relation, target)

# Patches stack: base files + layer 1 + layer 2 + ...
vindex.apply_patch("medical.vlp")

# Bake down: new standalone vindex with all patches
vindex.compile_vindex("edited.vindex")
```

### Storage
Inserted facts are stored as **structural edits**:
- **Not** stored in a separate knowledge base
- **Not** requiring backprop/training
- **Are** stored as down_vector rewrites on the gate/down matrix pair

Position in the model: determined by:
1. Entity embedding → relation cluster centre → target location in knowledge layer
2. User-provided `layer` hint (if specified)
3. Available space in that layer's down_weights

### Inference Parity
Compiled (unedited) model should produce identical outputs to base model:
```python
baseline = vindex.infer("The capital of France is")
compiled_unedited = compiled_vindex_unedited.infer("The capital of France is")
assert baseline == compiled_unedited  # Structural edits preserve inference
```

---

## CI/CD Integration

### Fast Path (every PR, ~30 seconds)
**Goal:** Catch breaking changes in LQL parsing/execution.

```bash
cargo test -p larql-lql
pytest tests/test_vindex_bindings.py::TestSession  # LQL session tests
```

**Runs in:** ~30 seconds  
**Requires:** No model downloads  
**Blocks merge:** Yes (required)

### Full Path (manual trigger / weekly, ~45 minutes)
**Goal:** End-to-end verification — extract → insert → compile → infer.

**Trigger:**
1. Add `[test-full-workflow]` to commit message
2. Use GitHub Actions `workflow_dispatch` button
3. Scheduled weekly (Saturday 10 AM UTC)

```yaml
workflow_dispatch:
  inputs:
    run_full_test:
      description: 'Run full extraction + compilation test'
      required: false
      default: 'false'
```

**Runs in:** ~45 minutes  
**Requires:** HuggingFace hub access + 3-5 GB disk  
**Blocks merge:** No (informational only)  
**Artifacts:** Compiled vindex, report.json

---

## Python API Reference

### Load & Baseline

```python
import larql

vindex = larql.load("output/gemma-2-2b-it.vindex")
print(vindex)
# Vindex(google/gemma-2-2b-it, 26 layers, hidden_size=2304)

# Baseline inference
result = vindex.infer("The capital of France is", top_k_predictions=5)
# [("Paris", 0.78), ("London", 0.05), ...]
```

### Insert Knowledge

```python
# Insert a fact (no training required)
layer, feat = vindex.insert("Colchester", "country", "England")

# Insert with layer hint
layer, feat = vindex.insert("Colchester", "country", "England", layer=24)

# Verify via describe
edges = vindex.describe("Colchester")
for e in edges:
    print(f"{e.relation} → {e.target} (score={e.gate_score:.0f})")
```

### Compile

```python
# Compile to new standalone vindex
compiled_path = vindex.compile_vindex("output/edited-model.vindex")

# Load compiled vindex
vindex_compiled = larql.load(compiled_path)

# Inference on compiled model
result = vindex_compiled.infer("The capital of France is")
```

### LQL Queries

```python
# Or use LQL directly via session
session = larql.session("output/gemma-2-2b-it.vindex")

# Execute LQL statement
session.query("DESCRIBE 'France'")
session.query("INSERT 'Paris' RELATION 'capital' TARGET 'France' LAYER 25")
session.query("COMPILE CURRENT INTO VINDEX output/result.vindex")
```

---

## Troubleshooting

### Extraction Timeout
HuggingFace hub downloads can be slow. Increase the timeout:
```bash
timeout 120m larql extract-index google/gemma-2-2b-it -o model.vindex
```

Or download the model manually:
```bash
huggingface-cli download google/gemma-2-2b-it --local-dir gemma-model
larql extract-index ./gemma-model -o model.vindex
```

### Memory Issues During Compile
Compile is streaming (doesn't load entire model into RAM), but if you hit memory limits:
1. Use `--f16` during extraction (halves size, negligible accuracy loss)
2. Compile on a machine with 8+ GB RAM
3. Reduce the number of inserted facts

### Inference Differences
After compilation, some inferences may differ slightly due to:
1. **Quantization rounding:** If using `--f16`, some precision is lost
2. **Unrelated edits:** Other layers may have been edited
3. **Numerical instability:** Long sequences compound floating-point error

Run the baseline vs. compiled comparison to quantify drift:
```python
report = {
    "baseline": baseline_results,
    "compiled": compiled_results,
}
```

### Compilation Not Implemented
If `vindex.compile_vindex()` raises `NotImplementedError`:
1. Check that larql-python was built with the latest code
2. Rebuild: `cd crates/larql-python && uv run maturin develop --release`
3. Use LQL CLI instead: `larql lql "COMPILE CURRENT INTO VINDEX result.vindex"`

---

## Performance Expectations

### Load Time
| Extraction Level | Size | Load Time | Use Case |
|---|---|---|---|
| browse | ~500 MB | <1s | DESCRIBE, WALK, lightweight queries |
| inference | ~1.2 GB | ~2s (cold), <500ms (warm) | Full inference, editing |
| all | ~2.2 GB | ~3s (cold) | Compilation, AOT inference |

### Insertion Time
```
insert() call: ~10-50 ms (synthesize gate vector, write to patch)
describe() call: ~5-20 ms (embed, gate KNN, lookup metadata)
```

### Compilation Time
```
compile_vindex(): ~2-5 minutes (rewrite down_weights.bin column-wise)
```

### Inference Time
```
Cold (first call): 500-2000 ms (warm up OS page cache)
Warm (cached): 100-200 ms per token
```

---

## Advanced: Custom Models

### Extract a Different Model

```bash
larql extract-index meta-llama/Llama-2-7b-hf \
  -o output/llama-2-7b.vindex \
  --level all \
  --f16
```

### Supported Architectures
- ✅ Gemma (2B, 3 4B, 7B, etc.)
- ✅ Llama 2 (7B, 13B, 70B)
- ✅ Qwen
- ✅ Mistral (7B, 8x7B MoE)
- ✅ GPT-2 / GPT-3 (via GGUF conversion)

### Model-Specific Notes
- **Gemma:** Fast extraction (~3 min for 7B)
- **Llama 2:** Requires HF token for gated models
- **Qwen:** Good knowledge coverage, dense FFN
- **Mistral:** Sparse MoE, fast inference
- **GPT-2:** Smallest, good for testing

---

## Next Steps

1. **Run the example:** `python3 examples/round_trip_workflow.py`
2. **Publish your edited model:** `larql hf output/edited-model.vindex --repo-id your-username/gemma-2-2b-it-edited`
3. **Integrate into your pipeline:** Use the Python API in your application
4. **Contribute:** Share your workflows and findings in the discussions

---

## References

- **LQL Spec:** [docs/specs/lql-spec.md](specs/lql-spec.md)
- **Vindex Format:** [docs/specs/vindex-format-spec.md](specs/vindex-format-spec.md)
- **Python API:** [docs/larql-python.md](larql-python.md)
- **CLI Reference:** [docs/cli.md](cli.md)

