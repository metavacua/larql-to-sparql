# larql — Python Bindings

Python interface to the LARQL knowledge graph engine and vindex model format. Rust-powered via PyO3, with numpy array interop and MLX integration.

## Install

Managed via [uv](https://docs.astral.sh/uv/):

```bash
cd crates/larql-python
uv sync --no-install-project --group dev     # creates .venv + installs dev deps
uv run --no-sync maturin develop --release   # builds PyO3 extension into .venv
uv run --no-sync pytest tests/               # run binding tests
```

Optional extras (Apple Silicon only): `uv sync --no-install-project --group dev --extra mlx`.

## Quickstart

```python
import larql

# Load a vindex
vindex = larql.load("output/gemma3-4b-v2.vindex")

# Knowledge queries — instant, no inference
edges = vindex.describe("France")
for e in edges[:5]:
    print(f"  {e.relation} → {e.target}  score={e.gate_score:.0f}")

# Full inference — Rust attention + walk FFN
result = vindex.infer("The capital of France is")
# [("Paris", 0.805), ...]

# Insert knowledge — no training
vindex.insert("Colchester", "country", "England")

# Bulk gate vectors for research (SVD, PCA)
gates = vindex.gate_vectors(layer=26)     # numpy (10240, 2560)
```

## Inference — Three Paths

### 1. Pure Rust (`vindex.infer`)

Full forward pass in Rust. No MLX, no GPU, no dependencies.

```python
vindex = larql.load("model.vindex")
result = vindex.infer("The capital of France is")
# [("Paris", 0.805), ...]
```

### 2. MLX Generation (`larql.mlx.load`)

MLX handles generation (KV cache, sampling, chat). Weights loaded from vindex.

```python
import larql, mlx_lm

model, tokenizer = larql.mlx.load("model.vindex")
response = mlx_lm.generate(model, tokenizer, prompt="...", max_tokens=20)
```

### 3. Walk FFN (`larql.walk_ffn.load`)

MLX attention + Rust sparse FFN. FFN weights mmap'd — only touched pages loaded.
For models that don't fit in memory.

```python
from larql.walk_ffn import load
import mlx_lm

model, tokenizer = load("model.vindex", top_k=4096)
response = mlx_lm.generate(model, tokenizer, prompt="...", max_tokens=20)
# Walk FFN: 7.1 GB FFN weights handled by Rust (not in MLX memory)
```

### 4. WalkModel (zero-copy mmap)

Rust inference with mmap'd weights. Load RSS: ~450 MB for a 4B model (vs 18 GB heap).
For 120B models: ~1 GB RSS instead of 220 GB.

```python
wm = larql.WalkModel("model.vindex", top_k=4096)
result = wm.predict("The capital of France is")
# [("Paris", 0.498), ...]
```

### Memory & Performance (Gemma 3 4B, f32)

| Path | Load RSS | Inference | How |
|---|---|---|---|
| `WalkModel` / `vindex.infer()` | **+0 MB** (mmap) | 19s→13s (warms up) | Zero-copy mmap, OS pages on demand |
| `larql.mlx.load()` | +22 GB | 0.9s (GPU) | All weights in MLX/GPU memory |
| Native MLX | +8.6 GB | 0.9s (GPU) | Safetensors in GPU memory |

`vindex.infer()` uses mmap'd weights (lazy-loaded on first call, reused after).
The OS page cache warms up across calls — second call is faster, third faster still.

For 120B models: `WalkModel` ~1 GB load RSS vs native 220 GB.
With madvise prefetching, steady-state ~200-500ms/token after cache warms.

## LQL Session

```python
session = larql.session("model.vindex")
session.query("DESCRIBE 'France'")
session.query("WALK 'The capital of France is' TOP 10")
session.vindex.gate_vectors(layer=26)  # numpy access on same session
```

## API Reference

### Loading

| Function | Description |
|---|---|
| `larql.load(path)` | Load vindex, returns `Vindex` |
| `larql.session(path)` | LQL session with `.query()` and `.vindex` |
| `larql.mlx.load(path)` | MLX model from vindex (all weights in MLX) |
| `larql.walk_ffn.load(path, top_k)` | MLX attention + Rust FFN (mmap'd) |
| `larql.WalkModel(path, top_k)` | Rust inference with mmap'd weights |

### Vindex — Inference

| Method | Description |
|---|---|
| `infer(prompt, top_k_predictions=5)` | Full Rust forward pass, returns `[(token, prob)]`. Routes through `larql_inference::infer_patched` for byte-identical parity with LQL `SELECT ... INFER` (ADR 0001) |

### Vindex — Knowledge Queries

| Method | Description |
|---|---|
| `describe(entity, band="knowledge", verbose=False)` | Find all knowledge edges |
| `has_edge(entity, relation=None)` | Check if entity has edges |
| `get_target(entity, relation)` | Get target token for entity+relation |
| `relations()` | List all relation types with counts |
| `cluster_centre(relation)` | Relation direction vector as numpy |
| `typical_layer(relation)` | Most common layer for a relation |
| `stats()` | Model metadata as dict |

### Vindex — Feature Access

| Method | Returns |
|---|---|
| `embed(text)` | `numpy (hidden_size,)` — scaled, multi-token averaged |
| `gate_vector(layer, feature)` | `numpy (hidden_size,)` |
| `gate_vectors(layer)` | `numpy (num_features, hidden_size)` |
| `embedding(token_id)` | `numpy (hidden_size,)` — unscaled |
| `embedding_matrix()` | `numpy (vocab_size, hidden_size)` |
| `feature_meta(layer, feature)` | `FeatureMeta` or `None` |
| `feature(layer, feature)` | `dict` or `None` |
| `feature_label(layer, feature)` | `str` or `None` |
| `tokenize(text)` / `decode(ids)` | Tokenizer access |

### Vindex — KNN & Walk

| Method | Description |
|---|---|
| `gate_knn(layer, query_vector, top_k=10)` | Raw KNN with vector |
| `entity_knn(entity, layer, top_k=10)` | Embed entity then KNN |
| `walk(residual, layers=None, top_k=5)` | Walk with raw vector |
| `entity_walk(entity, layers=None, top_k=5)` | Walk with entity string |

### Vindex — Mutation

| Method | Description |
|---|---|
| `insert(entity, relation, target, layer=None, confidence=0.8)` | Insert knowledge edge |
| `delete(entity, relation=None, layer=None)` | Delete matching edges |

### WalkModel

| Method | Description |
|---|---|
| `WalkModel(path, top_k=8192)` | Load with mmap'd weights (zero-copy) |
| `predict(prompt, top_k_predictions=5)` | Full forward pass, returns `[(token, prob)]` |
| `ffn_layer(layer, x_bytes, seq_len)` | Per-layer sparse FFN (bytes in/out) |
| `num_layers`, `hidden_size`, `top_k` | Properties |

### Session

| Method | Description |
|---|---|
| `query(lql)` | Execute LQL, returns `list[str]` |
| `query_text(lql)` | Execute LQL, returns joined string |
| `vindex` | Access underlying `Vindex` |

### Types

| Type | Key Fields |
|---|---|
| `DescribeEdge` | `relation`, `target`, `gate_score`, `layer`, `feature`, `source`, `confidence`, `also` |
| `WalkHit` | `layer`, `feature`, `gate_score`, `top_token`, `target`, `meta` |
| `FeatureMeta` | `top_token`, `top_token_id`, `c_score`, `top_k` |
| `Relation` | `name`, `cluster_id`, `count`, `top_tokens` |

## Project Structure

```
crates/larql-python/
  src/
    lib.rs              # Module registration, graph bindings
    vindex.rs           # PyVindex: describe, insert, relations, infer
    session.rs          # PySession (LQL queries)
    walk.rs             # WalkModel: mmap'd weights, Rust walk FFN
  python/larql/
    __init__.py         # Clean Python API
    mlx.py              # MLX model loading from vindex (mmap)
    walk_ffn.py         # MLX attention + Rust walk FFN
  tests/
    test_bindings.py    # Synthetic vindex tests + real vindex integration
  examples/
    knowledge.py        # Describe, relations, steering
    insert.py           # Insert knowledge, no training
    session.py          # LQL session + numpy access
    infer.py            # Rust inference (vindex.infer / WalkModel)
    mlx_vindex.py       # MLX generation from vindex weights
  bench/
    bench_bindings.py   # Speed + memory benchmarks
```

### Running Tests

```bash
# Synthetic tests (run anywhere, no model files)
pytest crates/larql-python/tests/ -v

# With real vindex (integration tests for infer, WalkModel, MLX)
REAL_VINDEX_PATH=output/gemma3-4b-v2.vindex pytest crates/larql-python/tests/ -v
```

### Extracting a Vindex

```bash
# Browse level (knowledge queries only)
larql extract-index "google/gemma-3-4b-it" -o model.vindex

# All weights (for inference + MLX)
larql extract-index "google/gemma-3-4b-it" -o model.vindex --level all

# Half precision (recommended for MLX)
larql extract-index "google/gemma-3-4b-it" -o model.vindex --level all --f16
```
