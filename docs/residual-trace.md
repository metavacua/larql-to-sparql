# Residual Stream Trace

The residual stream is the single wire through a transformer. Attention writes to it. FFN reads from it. The answer emerges from it. The trace is the complete record of what happened during inference — every waypoint, every contribution, queryable at any point.

## Architecture

```
Node:   residual state at (layer, position) — a 2560D vector
Edges:  attn_delta (what attention added) + ffn_delta (what FFN added)

residual[L] = residual[L-1] + attn_delta[L] + ffn_delta[L]
```

The trace is a DAG with `tokens x layers` nodes and two types of edges:
- **Vertical (FFN):** per-position knowledge retrieval
- **Horizontal (attention):** cross-position information routing

### Markov Property

Each layer's residual depends only on the previous layer's residual plus the current layer's deltas. Old token chains are complete and frozen — they never change. This makes the trace append-only.

## Implementation

### Rust Engine

```
crates/larql-inference/src/trace/
  mod.rs       — module root
  types.rs     — TraceNode, ResidualTrace, AnswerWaypoint, LayerSummary
  capture.rs   — decomposed forward pass recording attn/FFN deltas
  store.rs     — mmap'd append-only binary store (full chains)
  boundary.rs  — mmap'd boundary residual store (10 KB per window)
  context.rs   — tiered context store with critical layer deltas
  vocab.rs     — vocabulary projection helpers
```

### Python Bindings

```
crates/larql-python/src/trace_py.rs
  PyResidualTrace    — queryable trace object
  PyAnswerWaypoint   — per-layer answer state
  PyLayerSummary     — compact per-layer overview
  PyTraceStore       — mmap'd full chain reader
  PyBoundaryStore    — mmap'd boundary residual reader
  PyBoundaryWriter   — append-only boundary writer
```

## Usage

### LQL

```sql
-- Answer trajectory: track Paris through all 34 layers
TRACE "The capital of France is" FOR "Paris";

-- Attn vs FFN decomposition at the phase transition
TRACE "The capital of France is" DECOMPOSE LAYERS 22-27;

-- Full decomposition, all layers
TRACE "The capital of France is" DECOMPOSE;

-- Save to mmap'd file
TRACE "The capital of France is" SAVE "france.trace";

-- All positions, specific layer range, with answer tracking
TRACE "The capital of France is" FOR "Paris" LAYERS 20-33 POSITIONS ALL;
```

The LQL TRACE command uses the same backend as INFER — mutations via INSERT/DELETE are reflected in trace results.

### Python: Capture a trace

```python
import larql

wm = larql.WalkModel("gemma3-4b.vindex")
t = wm.trace("The capital of France is")
# ResidualTrace('The capital of France is', 6 tokens, 34 layers, 35 nodes)
```

### Python: Query the answer trajectory

```python
traj = t.answer_trajectory("Paris")
for w in traj:
    if w.layer >= 22:
        print(w)
# AnswerWaypoint(L22, rank=50,  prob=0.002, attn=22.2,  ffn=34.4)
# AnswerWaypoint(L23, rank=10,  prob=0.024, attn=-16.9, ffn=55.9)
# AnswerWaypoint(L24, rank=1,   prob=0.714, attn=105.7, ffn=24.4)
# AnswerWaypoint(L25, rank=1,   prob=0.997, attn=4.3,   ffn=94.4)
# AnswerWaypoint(L26, rank=1,   prob=0.999, attn=83.1,  ffn=18.7)
```

### Python: Inspect any layer

```python
t.top_k(24)        # [('Paris', 0.714), ('located', 0.133), ...]
t.rank_of("Paris", 23)  # 10
t.residual(24)      # [f32; 2560] — the raw vector
t.attn_delta(24)    # [f32; 2560] — what attention added
t.ffn_delta(24)     # [f32; 2560] — what FFN added
```

### Python: Multi-position trace

```python
t = wm.trace("The capital of France is", positions="all")
# 210 nodes (6 tokens x 35 layers)
t.residual(24, position=4)  # France's residual at L24
```

### Python: Save and mmap

```python
from larql._native import TraceStore

t.save("trace.bin")
store = TraceStore("trace.bin")
# TraceStore(6 tokens, 34 layers, 2560D, 6.5 MB)
store.residual(5, 25)   # token 5, layer 25 — zero-copy from mmap
```

### Python: Boundary store (production context)

```python
from larql._native import BoundaryWriter, BoundaryStore

writer = BoundaryWriter("context.bndx", hidden_size=2560, window_size=200)
writer.append(token_offset=0, window_tokens=200, residual=boundary_vec)
writer.finish()

store = BoundaryStore("context.bndx")
# BoundaryStore(1850 boundaries, 370000 tokens, 18.5 MB data, window=200)
store.residual(42)  # boundary 42 — zero-copy from mmap
```

## Key Findings

### The Phase Transition (L24-L26)

The answer does not exist in the residual stream until layer 24. Then it appears in a sudden phase transition:

| Layer | % top-1 | Mean P(answer) | Event |
|-------|---------|----------------|-------|
| L22 | 0% | 0.001 | Nothing |
| L23 | 3% | 0.010 | FFN fires (+56 logits) |
| **L24** | **30%** | **0.258** | **Attention fires (+106 logits)** |
| **L25** | **60%** | **0.514** | **FFN fires (+94 logits)** |
| **L26** | **84%** | **0.796** | Both fire together |
| L27 | 91% | 0.813 | Stabilizing |

Attention and FFN alternate: attention fires at even layers (L24, L26), FFN at every layer but especially odd layers (L23, L25, L27). A two-stroke engine.

### Attention vs FFN Decomposition

Across the full forward pass:
- **L0-L6:** FFN constructs, attention navigates (both positive, small)
- **L7-L13:** Attention positions, FFN neutral (encoding phase)
- **L14-L22:** Both start pushing answer, accelerating (knowledge layers)
- **L23-L27:** Alternating fire — phase transition
- **L28-L31:** FFN stabilizes, attention calibrates
- **L32:** Both destructive (confidence adjustment)
- **L33:** Recovery

FFN contributes ~60% of total answer signal, attention ~40%.

### Residual Stream Trajectory

The residual norm grows monotonically from 53 (embedding) to 66,638 (L33). The answer's rank trajectory:

```
Embedding:    rank 2     (latent knowledge in token embeddings)
L0-L13:       rank ~50K  (encoding — representations being built)
L14-L22:      rank ~600  (narrowing — knowledge layers positioning)
L23-L26:      rank 1     (phase transition — answer crystallizes)
L27-L33:      rank 1     (stable — refinement and calibration)
```

## Tiered Context Storage

The trace enables tiered storage for infinite context without KV cache:

| Tier | Content | Per window | 370K tokens | Compression | Accuracy |
|------|---------|-----------|-------------|-------------|----------|
| 1 | Boundary residual | 10 KB | 18.9 MB | 3,100x | 13/15 (no replay needed) |
| 3 | + deltas L23-27 | 110 KB | 199 MB | 282x | 13/15 (0.97 cos) |
| 4 | + deltas L23-33 | 230 KB | 416 MB | 135x | 15/15 (bit-perfect) |
| 4+int8 | Template + quantized residuals | 58 KB | 110 MB | 511x | 12/12 (0.9999 cos) |
| Hybrid | Tier 1 default + int8 Tier 4 | mixed | 55 MB | 1,012x | full |
| KV cache | Q,K per token per layer | ~30 MB/win | 56,000 MB | 1x | full |

### How Tier 4 Works

Tier 4 is not an approximation. It's the additive property of the residual stream:

```
residual[L33] = residual[L22] + Σ(attn_delta[l] + ffn_delta[l]) for l in 23..33
```

Store the L22 residual and all deltas for L23-L33. Reconstruct by addition. Mathematically exact.

### Template Compression

Entity trajectories have 0.99 cosine similarity. Store one template (mean delta per slot), then delta-encode each window's deviation:

```
stored[window] = template + quantize(actual - template)
```

At int8 quantization: 58 KB/window, 0.9999 cosine to ground truth.

### Template Compression Detail

Not all vectors compress equally:

| Vector type | Template cosine | Compresses well? |
|------------|----------------|-----------------|
| L22 boundary residual | 0.996 | Yes (12x) |
| Attention deltas L25,L27-L29 | 0.98+ | Yes (5-9x) |
| FFN deltas L24-L28 | 0.55 | No (1.2x) — entity-specific knowledge |
| Attention deltas L24,L26 | 0.64-0.81 | Partially |

The FFN deltas are where the knowledge lives. They can't be templated away — each entity activates different FFN features.

## File Formats

### Full Chain Store (.bin)

```
Header (64 bytes):   magic "TRAC", version, hidden_size, n_layers, n_tokens
Token chains:        contiguous, fixed-size
  Per token: (n_layers+1) x 3 x hidden_size x f32
  = 35 x 3 x 2560 x 4 = 1,075,200 bytes per token (~1 MB)
```

### Boundary Store (.bndx)

```
Header (64 bytes):   magic "BNDX", version, hidden_size, window_size, n_boundaries
Index:               n_boundaries x 16-byte entries (token_offset, window_tokens, data_offset)
Data:                n_boundaries x hidden_size x f32
  = 2560 x 4 = 10,240 bytes per boundary (~10 KB)
```

### Tiered Context Store (.ctxt)

```
Header (128 bytes):  magic "CTXT", version, hidden_size, n_layers, window_size,
                     tier, critical_layers[], n_boundaries
Index:               n_boundaries x 24-byte entries
Data:                variable per tier
  Tier 1: 1 vector per boundary (10 KB)
  Tier 2: 1 + n_critical vectors (60 KB for 5 critical layers)
  Tier 3: 1 + 2*n_critical vectors (110 KB for 5 critical layers)
```

All formats are append-only and mmap'd. Old data is frozen. The OS manages page residency — only active data is in RAM.

## Connection to Existing Work

The trace connects several proven findings:

- **Hourglass architecture:** encoder (L0-13), bottleneck (L13-14), decoder (L14-27), output (L28-33). The phase transition at L24-26 is the decoder-output boundary.
- **99% template-fixed attention:** confirmed by direct measurement. Attention patterns are a function of the template, not the entity.
- **FFN cached 4160x:** the FFN deltas in the trace are exactly what the vindex walk FFN computes. The trace stores the result; the vindex enables the lookup.
- **Gate vectors full rank:** the FFN uses all 2560 dimensions. Entity-specific FFN deltas can't be templated — they're the knowledge.
- **8 circuits / 192 OV edges:** the attention deltas at L24 and L26 (+106, +83 logits) are the circuit outputs. The two biggest attention events in the entire forward pass.
- **LQL integration:** TRACE uses the same WalkFfn backend as INFER — mutations via INSERT/DELETE are reflected in trace results. The knowledge graph and the inference trace are consistent.
