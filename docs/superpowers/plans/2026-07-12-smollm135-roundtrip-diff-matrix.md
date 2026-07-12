# SmolLM2-135M Round-Trip Quantified-Difference Matrix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the pure measurement engine that computes **machine-checkable, quantified differences** between two model directories and enumerates the round-trip comparison matrix — emitting raw difference data with **no interpretation** (no expected/verdict/cause/meaning).

**Architecture:** Three stdlib-or-numpy Python modules under `scripts/lql_matrix/`, mirroring the `tensor_presence.py` precedent: `roundtrip_diff.py` (stdlib structural differ — manifest, safetensors header, JSON), `roundtrip_value.py` (numpy value-layer — tensor decode + metrics), `roundtrip_matrix.py` (active-model registry, comparison enumeration, top-level combine, JSONL/markdown emit). Every function is a pure computation over files/dicts, unit-tested with synthetic fixtures. **No larql runs in this plan** — it consumes model directories that a follow-on harness (out of scope here) produces.

**Tech Stack:** Python 3.12; `pytest` (test-only); `numpy` (value layer only — `roundtrip_diff.py` and `roundtrip_matrix.py` stay stdlib-only).

## Global Constraints

- **Python 3.12.** `roundtrip_diff.py`: **stdlib only, numpy-free** (`hashlib, struct, json, os, sys`) — importable without numpy, matching the `tensor_presence.py` precedent. `roundtrip_value.py`: stdlib + **numpy** (the value layer). `roundtrip_matrix.py`: stdlib for its own logic, but composes `roundtrip_value`, so it transitively requires numpy. `pytest` is test-only.
- **No larql code changes. No edits to `conformance.py`.** New files only.
- **Output is pure measurement:** every emitted field is a measured quantity or a machine-checkable predicate. **No** `expected`, `matches_expected`, `cause`, `verdict`, or meaning-named categories. Interpretation is **metalinguistic** — it lives at a different level (how *we* read the emitted relations), not inside this instrument.
- **No canonicalize-to-match / no reward-hacking.** Canonicalized comparison (if computed) is recorded as its own measured predicate alongside the raw one — never used to suppress a raw difference.
- Files live under `scripts/lql_matrix/`. Always `encoding="utf-8"`. Tolerant of missing/malformed artifacts — a differ function returns a structured "unreadable" record, it never crashes the run.
- Safetensors container (for the stdlib header parse): `[8-byte u64 LE header length][JSON header][data buffer]`; header maps `name → {dtype, shape, data_offsets:[begin,end]}` plus optional `__metadata__`.

---

## File Structure

- Create: `scripts/lql_matrix/roundtrip_diff.py` — stdlib structural differ.
- Create: `scripts/lql_matrix/roundtrip_diff_test.py` — pytest.
- Create: `scripts/lql_matrix/roundtrip_value.py` — numpy value layer.
- Create: `scripts/lql_matrix/roundtrip_value_test.py` — pytest.
- Create: `scripts/lql_matrix/roundtrip_matrix.py` — registry + enumeration + combine + emit.
- Create: `scripts/lql_matrix/roundtrip_matrix_test.py` — pytest.

---

### Task 1: File manifest + SHA256 (`roundtrip_diff.py`)

**Files:**
- Create: `scripts/lql_matrix/roundtrip_diff.py`
- Test: `scripts/lql_matrix/roundtrip_diff_test.py`

**Interfaces:**
- Produces: `sha256_file(path: str) -> str`; `file_manifest(model_dir: str) -> dict[str, dict]` returning `{filename: {"size": int, "sha256": str}}` for every regular file directly in `model_dir` (non-recursive).

- [ ] **Step 1: Write the failing test**

```python
# roundtrip_diff_test.py
import os, hashlib
import roundtrip_diff as D

def test_file_manifest_hashes_and_sizes(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"hello")
    (tmp_path / "b.json").write_bytes(b"{}")
    man = D.file_manifest(str(tmp_path))
    assert set(man) == {"a.bin", "b.json"}
    assert man["a.bin"]["size"] == 5
    assert man["a.bin"]["sha256"] == hashlib.sha256(b"hello").hexdigest()
    assert man["b.json"]["size"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd scripts/lql_matrix && python3 -m pytest roundtrip_diff_test.py::test_file_manifest_hashes_and_sizes -v`
Expected: FAIL with `ModuleNotFoundError` / `AttributeError: module 'roundtrip_diff' has no attribute 'file_manifest'`

- [ ] **Step 3: Write minimal implementation**

```python
# roundtrip_diff.py
"""Stdlib structural differ for model directories: file manifest, safetensors
header, JSON structural diff. Pure measurement — no interpretation."""
import hashlib
import json
import os
import struct


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_manifest(model_dir):
    out = {}
    for name in sorted(os.listdir(model_dir)):
        p = os.path.join(model_dir, name)
        if not os.path.isfile(p):
            continue
        out[name] = {"size": os.path.getsize(p), "sha256": sha256_file(p)}
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd scripts/lql_matrix && python3 -m pytest roundtrip_diff_test.py::test_file_manifest_hashes_and_sizes -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/lql_matrix/roundtrip_diff.py scripts/lql_matrix/roundtrip_diff_test.py
git commit -m "feat(roundtrip): file_manifest + sha256_file (stdlib structural differ)"
```

---

### Task 2: Manifest bijection (`roundtrip_diff.py`)

**Files:**
- Modify: `scripts/lql_matrix/roundtrip_diff.py`
- Test: `scripts/lql_matrix/roundtrip_diff_test.py`

**Interfaces:**
- Consumes: `file_manifest` output shape.
- Produces: `manifest_bijection(man_a: dict, man_b: dict) -> dict` returning `{"only_a": [str], "only_b": [str], "in_both": [str], "bijective": bool}` (sorted lists; `bijective` true iff no orphans).

- [ ] **Step 1: Write the failing test**

```python
def test_manifest_bijection_reports_orphans():
    a = {"model.safetensors": {}, "config.json": {}, "lm_head.safetensors": {}}
    b = {"model.safetensors": {}, "config.json": {}}
    bij = D.manifest_bijection(a, b)
    assert bij["only_a"] == ["lm_head.safetensors"]
    assert bij["only_b"] == []
    assert bij["in_both"] == ["config.json", "model.safetensors"]
    assert bij["bijective"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd scripts/lql_matrix && python3 -m pytest roundtrip_diff_test.py::test_manifest_bijection_reports_orphans -v`
Expected: FAIL with `AttributeError: ... 'manifest_bijection'`

- [ ] **Step 3: Write minimal implementation**

```python
def manifest_bijection(man_a, man_b):
    a, b = set(man_a), set(man_b)
    only_a = sorted(a - b)
    only_b = sorted(b - a)
    return {
        "only_a": only_a,
        "only_b": only_b,
        "in_both": sorted(a & b),
        "bijective": not only_a and not only_b,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd scripts/lql_matrix && python3 -m pytest roundtrip_diff_test.py::test_manifest_bijection_reports_orphans -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/lql_matrix/roundtrip_diff.py scripts/lql_matrix/roundtrip_diff_test.py
git commit -m "feat(roundtrip): manifest_bijection over file manifests"
```

---

### Task 3: Safetensors header parse (`roundtrip_diff.py`)

**Files:**
- Modify: `scripts/lql_matrix/roundtrip_diff.py`
- Test: `scripts/lql_matrix/roundtrip_diff_test.py`

**Interfaces:**
- Produces: `read_safetensors_header(path: str) -> dict` returning `{"tensors": {name: {"dtype": str, "shape": [int], "data_offsets": [int,int]}}, "metadata": dict|None, "order": [str]}` where `order` is tensor names sorted by their `data_offsets[0]`. On malformed input returns `{"error": str}` (never raises).

- [ ] **Step 1: Write the failing test**

```python
import struct, json as _json

def _write_safetensors(path, header):
    blob = _json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(b"\x00" * 64)  # dummy data buffer

def test_read_safetensors_header_parses_tensors_metadata_order(tmp_path):
    p = tmp_path / "model.safetensors"
    _write_safetensors(str(p), {
        "b.weight": {"dtype": "F32", "shape": [2], "data_offsets": [8, 16]},
        "a.weight": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 8]},
        "__metadata__": {"format": "pt"},
    })
    h = D.read_safetensors_header(str(p))
    assert h["tensors"]["a.weight"]["dtype"] == "BF16"
    assert h["metadata"] == {"format": "pt"}
    assert h["order"] == ["a.weight", "b.weight"]  # by data_offsets[0]

def test_read_safetensors_header_malformed_returns_error(tmp_path):
    p = tmp_path / "bad.safetensors"
    p.write_bytes(b"\x02\x00")  # too short for u64 length
    h = D.read_safetensors_header(str(p))
    assert "error" in h
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd scripts/lql_matrix && python3 -m pytest roundtrip_diff_test.py -k read_safetensors_header -v`
Expected: FAIL with `AttributeError: ... 'read_safetensors_header'`

- [ ] **Step 3: Write minimal implementation**

```python
def read_safetensors_header(path):
    try:
        with open(path, "rb") as f:
            raw_len = f.read(8)
            if len(raw_len) != 8:
                return {"error": "truncated header length"}
            (hlen,) = struct.unpack("<Q", raw_len)
            hdr = json.loads(f.read(hlen).decode("utf-8"))
    except (OSError, ValueError) as e:
        return {"error": f"{type(e).__name__}: {e}"}
    metadata = hdr.pop("__metadata__", None)
    tensors = {}
    for name, spec in hdr.items():
        tensors[name] = {
            "dtype": spec.get("dtype"),
            "shape": spec.get("shape"),
            "data_offsets": spec.get("data_offsets"),
        }
    order = sorted(tensors, key=lambda n: (tensors[n]["data_offsets"] or [0])[0])
    return {"tensors": tensors, "metadata": metadata, "order": order}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd scripts/lql_matrix && python3 -m pytest roundtrip_diff_test.py -k read_safetensors_header -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/lql_matrix/roundtrip_diff.py scripts/lql_matrix/roundtrip_diff_test.py
git commit -m "feat(roundtrip): stdlib safetensors header parse (tensors, metadata, order)"
```

---

### Task 4: Safetensors header diff (`roundtrip_diff.py`)

**Files:**
- Modify: `scripts/lql_matrix/roundtrip_diff.py`
- Test: `scripts/lql_matrix/roundtrip_diff_test.py`

**Interfaces:**
- Consumes: `read_safetensors_header` output.
- Produces: `header_diff(ha: dict, hb: dict) -> dict` returning
  `{"tensor_only_a": [str], "tensor_only_b": [str], "dtype_changes": {name: [a,b]}, "shape_changes": {name: [a,b]}, "order_equal": bool, "metadata_equal": bool, "metadata_a": dict|None, "metadata_b": dict|None}`. If either header carries `error`, returns `{"error_a"/"error_b": str}`.

- [ ] **Step 1: Write the failing test**

```python
def test_header_diff_reports_dtype_order_metadata_changes():
    ha = {"tensors": {"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
                      "lm": {"dtype": "F32", "shape": [4], "data_offsets": [16, 32]}},
          "metadata": {"format": "pt"}, "order": ["w", "lm"]}
    hb = {"tensors": {"w": {"dtype": "BF16", "shape": [4], "data_offsets": [8, 16]}},
          "metadata": None, "order": ["w"]}
    d = D.header_diff(ha, hb)
    assert d["tensor_only_a"] == ["lm"]
    assert d["tensor_only_b"] == []
    assert d["dtype_changes"] == {"w": ["F32", "BF16"]}
    assert d["order_equal"] is False
    assert d["metadata_equal"] is False

def test_header_diff_propagates_parse_error():
    d = D.header_diff({"error": "bad"}, {"tensors": {}, "metadata": None, "order": []})
    assert d["error_a"] == "bad"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd scripts/lql_matrix && python3 -m pytest roundtrip_diff_test.py -k header_diff -v`
Expected: FAIL with `AttributeError: ... 'header_diff'`

- [ ] **Step 3: Write minimal implementation**

```python
def header_diff(ha, hb):
    if "error" in ha or "error" in hb:
        return {"error_a": ha.get("error"), "error_b": hb.get("error")}
    ta, tb = ha["tensors"], hb["tensors"]
    a, b = set(ta), set(tb)
    dtype_changes, shape_changes = {}, {}
    for name in sorted(a & b):
        if ta[name]["dtype"] != tb[name]["dtype"]:
            dtype_changes[name] = [ta[name]["dtype"], tb[name]["dtype"]]
        if ta[name]["shape"] != tb[name]["shape"]:
            shape_changes[name] = [ta[name]["shape"], tb[name]["shape"]]
    return {
        "tensor_only_a": sorted(a - b),
        "tensor_only_b": sorted(b - a),
        "dtype_changes": dtype_changes,
        "shape_changes": shape_changes,
        "order_equal": ha["order"] == hb["order"],
        "metadata_equal": ha["metadata"] == hb["metadata"],
        "metadata_a": ha["metadata"],
        "metadata_b": hb["metadata"],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd scripts/lql_matrix && python3 -m pytest roundtrip_diff_test.py -k header_diff -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/lql_matrix/roundtrip_diff.py scripts/lql_matrix/roundtrip_diff_test.py
git commit -m "feat(roundtrip): safetensors header_diff (set/dtype/shape/order/metadata)"
```

---

### Task 5: JSON structural diff (`roundtrip_diff.py`)

**Files:**
- Modify: `scripts/lql_matrix/roundtrip_diff.py`
- Test: `scripts/lql_matrix/roundtrip_diff_test.py`

**Interfaces:**
- Produces: `json_structural_diff(path_a: str, path_b: str) -> dict` returning `{"only_a_paths": [str], "only_b_paths": [str], "changed": {dotted_path: [a_val, b_val]}, "byte_identical": bool}`. Dotted paths flatten nested objects (`text_config.rope_theta`). On unreadable/invalid JSON returns `{"error_a"/"error_b": str}`.

- [ ] **Step 1: Write the failing test**

```python
def test_json_structural_diff_flatten_and_byte_identical(tmp_path):
    a = tmp_path / "a.json"; b = tmp_path / "b.json"
    a.write_text('{"architectures": ["LlamaForCausalLM"], "n": {"x": 1, "y": 2}}')
    b.write_text('{"architectures": ["Gemma3ForCausalLM"], "n": {"x": 1}, "z": 3}')
    d = D.json_structural_diff(str(a), str(b))
    assert d["changed"]["architectures"] == [["LlamaForCausalLM"], ["Gemma3ForCausalLM"]]
    assert d["only_a_paths"] == ["n.y"]
    assert d["only_b_paths"] == ["z"]
    assert d["byte_identical"] is False

def test_json_structural_diff_identical(tmp_path):
    a = tmp_path / "a.json"; b = tmp_path / "b.json"
    a.write_text('{"k": 1}'); b.write_text('{"k": 1}')
    d = D.json_structural_diff(str(a), str(b))
    assert d["byte_identical"] is True
    assert d["changed"] == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd scripts/lql_matrix && python3 -m pytest roundtrip_diff_test.py -k json_structural -v`
Expected: FAIL with `AttributeError: ... 'json_structural_diff'`

- [ ] **Step 3: Write minimal implementation**

```python
def _flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    else:
        out[prefix] = obj
    return out


def json_structural_diff(path_a, path_b):
    try:
        raw_a = open(path_a, "rb").read()
        obj_a = json.loads(raw_a.decode("utf-8"))
    except (OSError, ValueError) as e:
        return {"error_a": f"{type(e).__name__}: {e}"}
    try:
        raw_b = open(path_b, "rb").read()
        obj_b = json.loads(raw_b.decode("utf-8"))
    except (OSError, ValueError) as e:
        return {"error_b": f"{type(e).__name__}: {e}"}
    fa, fb = _flatten(obj_a), _flatten(obj_b)
    a, b = set(fa), set(fb)
    changed = {p: [fa[p], fb[p]] for p in sorted(a & b) if fa[p] != fb[p]}
    return {
        "only_a_paths": sorted(a - b),
        "only_b_paths": sorted(b - a),
        "changed": changed,
        "byte_identical": raw_a == raw_b,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd scripts/lql_matrix && python3 -m pytest roundtrip_diff_test.py -k json_structural -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/lql_matrix/roundtrip_diff.py scripts/lql_matrix/roundtrip_diff_test.py
git commit -m "feat(roundtrip): json_structural_diff (flattened paths + byte-identical)"
```

---

### Task 6: Tensor decode (`roundtrip_value.py`)

**Files:**
- Create: `scripts/lql_matrix/roundtrip_value.py`
- Test: `scripts/lql_matrix/roundtrip_value_test.py`

**Interfaces:**
- Produces: `decode_tensor(data: bytes, dtype: str) -> numpy.ndarray` (float32), supporting `"F32"`, `"F16"`, `"BF16"`. Raises `ValueError` for unsupported dtype.

- [ ] **Step 1: Write the failing test**

```python
# roundtrip_value_test.py
import numpy as np
import roundtrip_value as V

def test_decode_f32_roundtrips():
    arr = np.array([1.5, -2.0, 0.0], dtype=np.float32)
    out = V.decode_tensor(arr.tobytes(), "F32")
    assert np.array_equal(out, arr)

def test_decode_bf16_upper16_bits():
    # bf16 of 1.0 is 0x3F80; little-endian bytes 80 3F
    data = np.array([0x3F80, 0xC000], dtype="<u2").tobytes()  # 1.0, -2.0
    out = V.decode_tensor(data, "BF16")
    assert out.dtype == np.float32
    assert np.allclose(out, [1.0, -2.0])

def test_decode_unsupported_raises():
    import pytest
    with pytest.raises(ValueError):
        V.decode_tensor(b"\x00", "Q4_K")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd scripts/lql_matrix && python3 -m pytest roundtrip_value_test.py -k decode -v`
Expected: FAIL with `ModuleNotFoundError: roundtrip_value`

- [ ] **Step 3: Write minimal implementation**

```python
# roundtrip_value.py
"""Numpy value layer: decode safetensors tensor bytes to float32 and compute
quantified per-tensor difference metrics. Pure measurement."""
import numpy as np


def decode_tensor(data, dtype):
    if dtype == "F32":
        return np.frombuffer(data, dtype="<f4").astype(np.float32, copy=True)
    if dtype == "F16":
        return np.frombuffer(data, dtype="<f2").astype(np.float32)
    if dtype == "BF16":
        u16 = np.frombuffer(data, dtype="<u2")
        u32 = (u16.astype(np.uint32) << 16)
        return u32.view(np.float32).astype(np.float32, copy=True)
    raise ValueError(f"unsupported dtype for decode: {dtype}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd scripts/lql_matrix && python3 -m pytest roundtrip_value_test.py -k decode -v`
Expected: PASS (all three)

- [ ] **Step 5: Commit**

```bash
git add scripts/lql_matrix/roundtrip_value.py scripts/lql_matrix/roundtrip_value_test.py
git commit -m "feat(roundtrip): numpy tensor decode (F32/F16/BF16 -> float32)"
```

---

### Task 7: Tensor value metrics (`roundtrip_value.py`)

**Files:**
- Modify: `scripts/lql_matrix/roundtrip_value.py`
- Test: `scripts/lql_matrix/roundtrip_value_test.py`

**Interfaces:**
- Produces: `tensor_value_metrics(a: numpy.ndarray, b: numpy.ndarray) -> dict` returning `{"comparable": bool, "n_total": int, "n_differing": int, "max_abs_diff": float, "l2": float}`. If shapes differ → `{"comparable": False, "shape_a": [...], "shape_b": [...]}`.

- [ ] **Step 1: Write the failing test**

```python
def test_tensor_value_metrics_exact_and_close():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert V.tensor_value_metrics(a, a) == {
        "comparable": True, "n_total": 3, "n_differing": 0,
        "max_abs_diff": 0.0, "l2": 0.0}
    b = np.array([1.0, 2.0, 3.5], dtype=np.float32)
    m = V.tensor_value_metrics(a, b)
    assert m["n_differing"] == 1
    assert abs(m["max_abs_diff"] - 0.5) < 1e-6

def test_tensor_value_metrics_shape_mismatch():
    a = np.zeros(3, dtype=np.float32); b = np.zeros(4, dtype=np.float32)
    m = V.tensor_value_metrics(a, b)
    assert m["comparable"] is False
    assert m["shape_a"] == [3] and m["shape_b"] == [4]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd scripts/lql_matrix && python3 -m pytest roundtrip_value_test.py -k value_metrics -v`
Expected: FAIL with `AttributeError: ... 'tensor_value_metrics'`

- [ ] **Step 3: Write minimal implementation**

```python
def tensor_value_metrics(a, b):
    if a.shape != b.shape:
        return {"comparable": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
    diff = a.astype(np.float64) - b.astype(np.float64)
    return {
        "comparable": True,
        "n_total": int(a.size),
        "n_differing": int(np.count_nonzero(a != b)),
        "max_abs_diff": float(np.max(np.abs(diff))) if a.size else 0.0,
        "l2": float(np.sqrt(np.sum(diff * diff))),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd scripts/lql_matrix && python3 -m pytest roundtrip_value_test.py -k value_metrics -v`
Expected: PASS (both)

- [ ] **Step 5: Commit**

```bash
git add scripts/lql_matrix/roundtrip_value.py scripts/lql_matrix/roundtrip_value_test.py
git commit -m "feat(roundtrip): tensor_value_metrics (n_differing/max_abs_diff/l2)"
```

---

### Task 8: Per-tensor value diff over a safetensors pair (`roundtrip_value.py`)

**Files:**
- Modify: `scripts/lql_matrix/roundtrip_value.py`
- Test: `scripts/lql_matrix/roundtrip_value_test.py`

**Interfaces:**
- Consumes: `roundtrip_diff.read_safetensors_header`; `decode_tensor`; `tensor_value_metrics`.
- Produces: `safetensors_value_diff(path_a: str, path_b: str) -> dict` returning `{name: <metrics>}` for every tensor present in **both** files (decoded to float32, so cross-dtype pairs are still value-comparable), plus `{"_bytes_equal": bool}` for the whole-file raw byte comparison. Unreadable → `{"error": str}`.

- [ ] **Step 1: Write the failing test**

```python
import struct, json as _json, roundtrip_diff as D2

def _st(path, tensors):  # tensors: {name: (dtype_str, np.ndarray)}
    header, buf, off = {}, bytearray(), 0
    for name, (dt, arr) in tensors.items():
        raw = arr.tobytes()
        header[name] = {"dtype": dt, "shape": list(arr.shape),
                        "data_offsets": [off, off + len(raw)]}
        buf += raw; off += len(raw)
    blob = _json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob))); f.write(blob); f.write(bytes(buf))

def test_safetensors_value_diff_cross_dtype(tmp_path):
    a = tmp_path / "a.safetensors"; b = tmp_path / "b.safetensors"
    v = np.array([1.0, 2.0], dtype=np.float32)
    _st(str(a), {"w": ("F32", v)})
    # bf16 of [1.0, 2.0] = 0x3F80, 0x4000
    bf = np.array([0x3F80, 0x4000], dtype="<u2")
    _st(str(b), {"w": ("BF16", bf)})
    d = V.safetensors_value_diff(str(a), str(b))
    assert d["w"]["comparable"] is True
    assert d["w"]["max_abs_diff"] == 0.0   # 1.0/2.0 exactly representable in bf16
    assert d["_bytes_equal"] is False       # F32 bytes != BF16 bytes
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd scripts/lql_matrix && python3 -m pytest roundtrip_value_test.py -k value_diff -v`
Expected: FAIL with `AttributeError: ... 'safetensors_value_diff'`

- [ ] **Step 3: Write minimal implementation**

```python
import struct
import roundtrip_diff as _D


def _tensor_bytes(path, spec, header_len):
    begin, end = spec["data_offsets"]
    with open(path, "rb") as f:
        f.seek(8 + header_len + begin)
        return f.read(end - begin)


def _header_len(path):
    with open(path, "rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


def safetensors_value_diff(path_a, path_b):
    ha, hb = _D.read_safetensors_header(path_a), _D.read_safetensors_header(path_b)
    if "error" in ha or "error" in hb:
        return {"error": ha.get("error") or hb.get("error")}
    la, lb = _header_len(path_a), _header_len(path_b)
    out = {}
    for name in sorted(set(ha["tensors"]) & set(hb["tensors"])):
        sa, sb = ha["tensors"][name], hb["tensors"][name]
        try:
            aa = decode_tensor(_tensor_bytes(path_a, sa, la), sa["dtype"]).reshape(sa["shape"])
            bb = decode_tensor(_tensor_bytes(path_b, sb, lb), sb["dtype"]).reshape(sb["shape"])
        except ValueError as e:
            out[name] = {"comparable": False, "decode_error": str(e)}
            continue
        out[name] = tensor_value_metrics(aa, bb)
    out["_bytes_equal"] = _D.sha256_file(path_a) == _D.sha256_file(path_b)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd scripts/lql_matrix && python3 -m pytest roundtrip_value_test.py -k value_diff -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/lql_matrix/roundtrip_value.py scripts/lql_matrix/roundtrip_value_test.py
git commit -m "feat(roundtrip): safetensors_value_diff over a file pair (cross-dtype value metrics)"
```

---

### Task 9: Active-model registry + comparison enumeration (`roundtrip_matrix.py`)

**Files:**
- Create: `scripts/lql_matrix/roundtrip_matrix.py`
- Test: `scripts/lql_matrix/roundtrip_matrix_test.py`

**Interfaces:**
- Produces: module-level `REGISTRY` (list of `{"id": str, "active": bool, "variants": [{"variant": str, "repo": str, "src_dtype": str}]}`); `active_variants(registry=REGISTRY) -> [(model_id, variant, repo, src_dtype)]`; `enumerate_comparisons(model_id, variant) -> [dict]` producing the comparison plan rows `{"model","variant","driver","mode","insert_form","comparison"}` for drivers `{lql, cli}`, modes `{A, B}`, insert forms `{knn, compose}` (B only), covering comparisons `input_vs_A`, `lqlA_vs_cliA`, `B_vs_A`, `lqlB_vs_cliB`.

- [ ] **Step 1: Write the failing test**

```python
# roundtrip_matrix_test.py
import roundtrip_matrix as M

def test_registry_defaults_to_smol135_only():
    active = M.active_variants()
    ids = {a[0] for a in active}
    assert ids == {"smol135"}
    variants = {a[1] for a in active}
    assert variants == {"instruct-bf16", "base-f32"}
    # everything else deactivated but present (just-works-on-re-add)
    assert any(m["id"] == "bitnet2b" and m["active"] is False for m in M.REGISTRY)

def test_enumerate_comparisons_covers_lattice():
    rows = M.enumerate_comparisons("smol135", "instruct-bf16")
    comps = {r["comparison"] for r in rows}
    assert {"input_vs_A", "lqlA_vs_cliA", "B_vs_A", "lqlB_vs_cliB"} <= comps
    # B rows carry an insert_form; A rows do not
    assert all(r["insert_form"] in ("knn", "compose")
               for r in rows if r["mode"] == "B")
    assert all(r["insert_form"] is None for r in rows if r["mode"] == "A")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd scripts/lql_matrix && python3 -m pytest roundtrip_matrix_test.py -v`
Expected: FAIL with `ModuleNotFoundError: roundtrip_matrix`

- [ ] **Step 3: Write minimal implementation**

```python
# roundtrip_matrix.py
"""Round-trip quantified-difference matrix: active-model registry, comparison
enumeration, and top-level combine over model-directory pairs. Emits raw
measured differences only — no interpretation."""
import json
import sys

import roundtrip_diff as D
import roundtrip_value as V

REGISTRY = [
    {"id": "smol135", "active": True, "variants": [
        {"variant": "instruct-bf16", "repo": "HuggingFaceTB/SmolLM2-135M-Instruct", "src_dtype": "BF16"},
        {"variant": "base-f32", "repo": "HuggingFaceTB/SmolLM2-135M", "src_dtype": "F32"},
    ]},
    {"id": "qwen05", "active": False, "variants": []},
    {"id": "smol360", "active": False, "variants": []},
    {"id": "qwen15", "active": False, "variants": []},
    {"id": "granite1b", "active": False, "variants": []},
    {"id": "bitnet2b", "active": False, "variants": []},
]

DRIVERS = ("lql", "cli")
INSERT_FORMS = ("knn", "compose")


def active_variants(registry=REGISTRY):
    out = []
    for m in registry:
        if not m["active"]:
            continue
        for v in m["variants"]:
            out.append((m["id"], v["variant"], v["repo"], v["src_dtype"]))
    return out


def enumerate_comparisons(model_id, variant):
    rows = []
    for driver in DRIVERS:
        rows.append({"model": model_id, "variant": variant, "driver": driver,
                     "mode": "A", "insert_form": None, "comparison": "input_vs_A"})
        for form in INSERT_FORMS:
            rows.append({"model": model_id, "variant": variant, "driver": driver,
                         "mode": "B", "insert_form": form, "comparison": "B_vs_A"})
    rows.append({"model": model_id, "variant": variant, "driver": "lql+cli",
                 "mode": "A", "insert_form": None, "comparison": "lqlA_vs_cliA"})
    for form in INSERT_FORMS:
        rows.append({"model": model_id, "variant": variant, "driver": "lql+cli",
                     "mode": "B", "insert_form": form, "comparison": "lqlB_vs_cliB"})
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd scripts/lql_matrix && python3 -m pytest roundtrip_matrix_test.py -v`
Expected: PASS (both)

- [ ] **Step 5: Commit**

```bash
git add scripts/lql_matrix/roundtrip_matrix.py scripts/lql_matrix/roundtrip_matrix_test.py
git commit -m "feat(roundtrip): active-model registry + comparison enumeration (smol135 only)"
```

---

### Task 10: Combine — quantified diff over a model-directory pair (`roundtrip_matrix.py`)

**Files:**
- Modify: `scripts/lql_matrix/roundtrip_matrix.py`
- Test: `scripts/lql_matrix/roundtrip_matrix_test.py`

**Interfaces:**
- Consumes: all `roundtrip_diff` + `roundtrip_value` functions.
- Produces: `diff_model_dirs(dir_a: str, dir_b: str) -> dict` returning
  `{"manifest": <bijection>, "files": {name: <per-file record>}}` where each per-file record is measured-only: for `*.safetensors` → `{"sha256_equal": bool, "header": <header_diff>, "values": <safetensors_value_diff>}`; for `*.json` → `{"sha256_equal": bool, "json": <json_structural_diff>}`; otherwise → `{"sha256_equal": bool, "size_a": int, "size_b": int}`. Only files `in_both` get content diffs.

- [ ] **Step 1: Write the failing test**

```python
def test_diff_model_dirs_combines_structural_and_value(tmp_path):
    import numpy as np, struct, json as _json
    da = tmp_path / "a"; db = tmp_path / "b"; da.mkdir(); db.mkdir()
    (da / "config.json").write_text('{"architectures": ["LlamaForCausalLM"]}')
    (db / "config.json").write_text('{"architectures": ["Gemma3ForCausalLM"]}')

    def _st(path, dt, arr):
        raw = arr.tobytes()
        hdr = {"w": {"dtype": dt, "shape": list(arr.shape), "data_offsets": [0, len(raw)]}}
        blob = _json.dumps(hdr).encode()
        path.write_bytes(struct.pack("<Q", len(blob)) + blob + raw)

    _st(da / "model.safetensors", "F32", np.array([1.0, 2.0], dtype=np.float32))
    _st(db / "model.safetensors", "BF16", np.array([0x3F80, 0x4000], dtype="<u2"))

    out = M.diff_model_dirs(str(da), str(db))
    assert out["manifest"]["bijective"] is True
    assert out["files"]["config.json"]["json"]["changed"]["architectures"] == \
        [["LlamaForCausalLM"], ["Gemma3ForCausalLM"]]
    st = out["files"]["model.safetensors"]
    assert st["sha256_equal"] is False
    assert st["header"]["dtype_changes"] == {"w": ["F32", "BF16"]}
    assert st["values"]["w"]["max_abs_diff"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd scripts/lql_matrix && python3 -m pytest roundtrip_matrix_test.py -k diff_model_dirs -v`
Expected: FAIL with `AttributeError: ... 'diff_model_dirs'`

- [ ] **Step 3: Write minimal implementation**

```python
import os


def diff_model_dirs(dir_a, dir_b):
    man_a, man_b = D.file_manifest(dir_a), D.file_manifest(dir_b)
    bij = D.manifest_bijection(man_a, man_b)
    files = {}
    for name in bij["in_both"]:
        pa, pb = os.path.join(dir_a, name), os.path.join(dir_b, name)
        sha_eq = man_a[name]["sha256"] == man_b[name]["sha256"]
        if name.endswith(".safetensors"):
            files[name] = {
                "sha256_equal": sha_eq,
                "header": D.header_diff(D.read_safetensors_header(pa),
                                        D.read_safetensors_header(pb)),
                "values": V.safetensors_value_diff(pa, pb),
            }
        elif name.endswith(".json"):
            files[name] = {"sha256_equal": sha_eq,
                           "json": D.json_structural_diff(pa, pb)}
        else:
            files[name] = {"sha256_equal": sha_eq,
                           "size_a": man_a[name]["size"], "size_b": man_b[name]["size"]}
    return {"manifest": bij, "files": files}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd scripts/lql_matrix && python3 -m pytest roundtrip_matrix_test.py -k diff_model_dirs -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/lql_matrix/roundtrip_matrix.py scripts/lql_matrix/roundtrip_matrix_test.py
git commit -m "feat(roundtrip): diff_model_dirs combine (manifest + header + json + values)"
```

---

### Task 11: Emit — JSONL rows + plain markdown render + `main()` (`roundtrip_matrix.py`)

**Files:**
- Modify: `scripts/lql_matrix/roundtrip_matrix.py`
- Test: `scripts/lql_matrix/roundtrip_matrix_test.py`

**Interfaces:**
- Produces: `to_rows(meta: dict, dir_diff: dict) -> [dict]` flattening `diff_model_dirs` output into one JSONL row per (file, measurement) carrying `meta` fields (`model, variant, driver, mode, insert_form, comparison, in_format_eq_out_format`) + the measured quantities, **no verdict fields**; `render_markdown(rows: list) -> str` producing a plain table of measured quantities; `main(argv)` reading `--a DIR --b DIR --meta JSON --out FILE.jsonl [--md FILE.md]` and writing outputs.

- [ ] **Step 1: Write the failing test**

```python
def test_to_rows_carries_meta_and_measurements_no_verdicts():
    dir_diff = {"manifest": {"bijective": True, "only_a": [], "only_b": []},
                "files": {"model.safetensors": {
                    "sha256_equal": False,
                    "header": {"dtype_changes": {"w": ["F32", "BF16"]}, "order_equal": True,
                               "metadata_equal": False, "tensor_only_a": [], "tensor_only_b": [],
                               "shape_changes": {}},
                    "values": {"w": {"comparable": True, "n_total": 2, "n_differing": 0,
                                     "max_abs_diff": 0.0, "l2": 0.0}, "_bytes_equal": False}}}}
    meta = {"model": "smol135", "variant": "base-f32", "driver": "cli",
            "mode": "A", "insert_form": None, "comparison": "input_vs_A",
            "in_format_eq_out_format": False}
    rows = M.to_rows(meta, dir_diff)
    r = [x for x in rows if x.get("file") == "model.safetensors"][0]
    assert r["model"] == "smol135" and r["comparison"] == "input_vs_A"
    assert r["sha256_equal"] is False
    assert r["in_format_eq_out_format"] is False
    # measured-only: no interpretation fields
    for banned in ("expected", "matches_expected", "cause", "verdict", "category"):
        assert banned not in r

def test_render_markdown_is_plain_table():
    rows = [{"model": "smol135", "variant": "base-f32", "comparison": "input_vs_A",
             "file": "model.safetensors", "sha256_equal": False}]
    md = M.render_markdown(rows)
    assert "model.safetensors" in md and "|" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd scripts/lql_matrix && python3 -m pytest roundtrip_matrix_test.py -k "to_rows or render_markdown" -v`
Expected: FAIL with `AttributeError: ... 'to_rows'`

- [ ] **Step 3: Write minimal implementation**

```python
def to_rows(meta, dir_diff):
    rows = []
    base = dict(meta)
    rows.append({**base, "file": "_manifest",
                 "bijective": dir_diff["manifest"]["bijective"],
                 "only_a": dir_diff["manifest"].get("only_a", []),
                 "only_b": dir_diff["manifest"].get("only_b", [])})
    for name, rec in dir_diff["files"].items():
        row = {**base, "file": name, "sha256_equal": rec.get("sha256_equal")}
        if "header" in rec:
            h = rec["header"]
            row["header_dtype_changes"] = h.get("dtype_changes", {})
            row["header_shape_changes"] = h.get("shape_changes", {})
            row["header_order_equal"] = h.get("order_equal")
            row["header_metadata_equal"] = h.get("metadata_equal")
            row["header_tensor_only_a"] = h.get("tensor_only_a", [])
            row["header_tensor_only_b"] = h.get("tensor_only_b", [])
            vals = rec.get("values", {})
            row["values_bytes_equal"] = vals.get("_bytes_equal")
            row["values_max_abs_diff"] = max(
                (m.get("max_abs_diff", 0.0) for k, m in vals.items()
                 if k != "_bytes_equal" and isinstance(m, dict) and m.get("comparable")),
                default=0.0)
            row["values_n_differing_total"] = sum(
                m.get("n_differing", 0) for k, m in vals.items()
                if k != "_bytes_equal" and isinstance(m, dict) and m.get("comparable"))
        if "json" in rec:
            j = rec["json"]
            row["json_changed"] = j.get("changed", {})
            row["json_only_a_paths"] = j.get("only_a_paths", [])
            row["json_only_b_paths"] = j.get("only_b_paths", [])
            row["json_byte_identical"] = j.get("byte_identical")
        rows.append(row)
    return rows


def render_markdown(rows):
    cols = ["model", "variant", "driver", "mode", "insert_form", "comparison",
            "file", "sha256_equal"]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--meta", required=True, help="JSON meta dict")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md")
    ns = ap.parse_args(argv)
    meta = json.loads(ns.meta)
    rows = to_rows(meta, diff_model_dirs(ns.a, ns.b))
    with open(ns.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    if ns.md:
        with open(ns.md, "w", encoding="utf-8") as f:
            f.write(render_markdown(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd scripts/lql_matrix && python3 -m pytest roundtrip_matrix_test.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add scripts/lql_matrix/roundtrip_matrix.py scripts/lql_matrix/roundtrip_matrix_test.py
git commit -m "feat(roundtrip): to_rows + render_markdown + main (measured-only emit)"
```

---

### Task 12: Full-suite green + end-to-end synthetic demonstration

**Files:**
- Test: all three test files.

- [ ] **Step 1: Run the whole round-trip suite**

Run: `cd scripts/lql_matrix && python3 -m pytest roundtrip_diff_test.py roundtrip_value_test.py roundtrip_matrix_test.py -v`
Expected: PASS, zero failures, zero collection errors.

- [ ] **Step 2: Drive `main()` on a synthetic F32-vs-BF16 model pair**

Build two tiny model dirs (as in the Task 10 test), then:
Run:
```bash
cd scripts/lql_matrix && python3 roundtrip_matrix.py \
  --a /tmp/rt_a --b /tmp/rt_b \
  --meta '{"model":"smol135","variant":"base-f32","driver":"cli","mode":"A","insert_form":null,"comparison":"input_vs_A","in_format_eq_out_format":false}' \
  --out /tmp/rt.jsonl --md /tmp/rt.md
```
Expected: `/tmp/rt.jsonl` has one row per file with measured quantities (`sha256_equal`, `header_dtype_changes`, `values_max_abs_diff`, `json_changed`), **no** `expected`/`verdict`/`cause` keys; `/tmp/rt.md` renders a plain table.

- [ ] **Step 3: Confirm measured-only invariant mechanically**

Run: `grep -E '"(expected|matches_expected|cause|verdict|category)"' /tmp/rt.jsonl; echo "exit=$?"`
Expected: no matches (grep exit 1) — the emit carries no interpretation fields.

- [ ] **Step 4: Commit the demonstration note**

```bash
git add -A && git commit -m "test(roundtrip): full suite green + synthetic end-to-end demonstration" --allow-empty
```

---

## Out of scope (deliberate — a separate follow-on plan)

This plan builds the **measurement engine only**. The following run larql and are a
distinct plan (`…-roundtrip-harness.md`), each producing the model-directory pairs
this engine consumes. **Venue is CI-first** (public repo ⇒ free unlimited Actions;
local box is underpowered and a local crash loses data — CI captures crashes):

- The **operation runner**: on a GitHub-hosted runner (ample resources, larql runs
  directly), invocations of `extract`, LQL `COMPILE … INTO MODEL … FORMAT safetensors`,
  `larql compile` CLI, and `INSERT … MODE {KNN,COMPOSE}` — capturing `input`/`A`/`B`
  model dirs per variant/driver/mode/insert-form, plus the `run_matrix.py` mechanical
  outcome for each driving statement (consumed, not re-classified — §6.4 of the spec).
  (An *optional*, non-preferred local run wraps `larql-probe safe --cpus N -- …`.)
- The **matrix driver** that walks `enumerate_comparisons`, invokes the runner,
  feeds pairs to `diff_model_dirs`, and aggregates the JSONL.
- The **CI workflow** `lql-roundtrip-catalogue.yml` (model-level, `roundtrip-*`
  artifacts excluded from the `results-*` glob).
- **Empirical grounding:** the first harness task is to **commit → push → open a PR**
  and let Actions run it on SmolLM2-135M — converting the compile-behavior questions
  (bf16 output, config rewrite, vindex participation, LQL-vs-CLI save path) into
  *measured* rows. No claims asserted in advance; the measurements come from a CI run,
  not a local spot-check.

## Interpretation is metalinguistic — a different level, not a later phase

Interpretation — reading meaning into the processes, states, values, functions,
and relations (verdicts, cause attribution, expected-vs-actual, meaning-named
categories) — is a **metalinguistic** act: it is about *how we read* the
object-level artifacts, performed by us, in the metalanguage, over the relations
this instrument emits.

This instrument (and the CI that runs it) is **object-level**: it executes
object-level processes and **measures** the resulting relations. It is a
measurement instrument, **not an interpreter**, and interpretation is therefore
not a downstream "phase" of the same pipeline — it is at a different level.

**Terminological hazard (kept straight deliberately):** `larql` *is* formally a
compiler and interpreter — it compiles models and interprets LQL — but that is an
object-level mechanical role the matrix *measures*. It is not the metalinguistic
interpretation deferred here, and the CI orchestration is neither. The matrix
measures what larql's compiler/interpreter *does*; assigning meaning to those
measurements happens only in the metalanguage, once the object-level relations
exist.

---

## Self-Review

**Spec coverage:** measurement engine covers spec §6.2 (structural stdlib + numpy
value), §6.3 (ladder rungs as measured quantities: sha256=rung1, header
order/dtype=rung2/3 structure, value metrics=rung4/5), §7 schema (measured-only
subset), manifest bijection, `in_format_eq_out_format` predicate. Spec §6.1
(active-model gate) → Task 9 registry. Spec §8 (CI) and the larql-driving portions
→ explicitly deferred to the follow-on harness plan. Interpretation (§4/§5/§10
verdicts/findings) → deferred to the data-gated phase, per the approved scope
correction.

**Placeholder scan:** no TBD/TODO; every code step is complete and runnable.

**Type consistency:** `file_manifest`→`manifest_bijection`→`diff_model_dirs`;
`read_safetensors_header`→`header_diff`/`safetensors_value_diff`; `decode_tensor`
+`tensor_value_metrics`→`safetensors_value_diff`; `enumerate_comparisons` meta
keys match `to_rows` `meta` consumption. Consistent.
