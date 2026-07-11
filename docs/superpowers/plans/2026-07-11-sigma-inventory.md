# σ-expansion Stage 1: Tensor-Presence Inventory (W₃ / Q8) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Append the *tensor-presence* column to the CI's observable algebra σ so the panic response becomes σ-measurable — i.e. observe, per `(model, level)`, which weight classes (attention vs FFN) the produced vindex actually contains, and prove `panic ⟺ (has_attn ∧ ¬has_ffn)`.

**Architecture:** A stdlib-only `tensor_presence.py` turns a produced vindex's file listing into a presence vector `{has_attn, has_ffn, ffn_unwrap_risk, …}`. The produce step captures a `listing.json` per leg (cheap; the manifest dir is already uploaded). A peer `inventory` CI job joins the presence vectors with the conformance panic outcomes and asserts the biconditional `ffn_unwrap_risk ⟺ panic@infer`, resolving the M3(a) duplicate-row (granite@attention panics; bitnet@attention refuses).

**Tech Stack:** Python 3.12 stdlib only; pytest 7.2.1; GitHub Actions YAML.

## Global Constraints

- `tensor_presence.py`: **Python 3.12 stdlib only** (`glob, json, re, sys, os, pathlib`). `pytest` is test-only.
- **No larql code changes.** Harness/CI only; the inventory consumes already-produced artifacts.
- Runs on **GitHub-hosted runners only** (isolated VMs) — no self-hosted, no local-device resource assumptions.
- Always `encoding="utf-8"`; tolerant of missing/malformed artifacts (never crash the checker).
- Vindex weight files (the presence witnesses): `attn_weights.bin` (attn), `up_weights.bin` + `down_weights.bin` (ffn), `gate_vectors.bin` (gate), `embeddings.bin`, `norms.bin`, `down_meta.bin`, `interleaved_kquant.bin` (q4k).
- `has_ffn := (up_weights.bin present ∧ down_weights.bin present)`; `ffn_unwrap_risk := has_attn ∧ ¬has_ffn`.
- Artifacts `retention-days: 1`.

---

## File Structure

- Create: `scripts/lql_matrix/tensor_presence.py` — the presence witness (pure `presence()` + `listing_of()` + `main()` + `resolve_q8()`).
- Create: `scripts/lql_matrix/tensor_presence_test.py` — pytest over synthetic listings.
- Modify: `.github/workflows/lql-strategy-matrix.yml` — capture `listing.json` in the produce step + add a peer `inventory` job.

`descriptor.py`/`conformance.py`/`gen_legs.py` are unchanged.

---

### Task 1: `presence()` + `listing_of()`

**Files:**
- Create: `scripts/lql_matrix/tensor_presence.py`
- Test: `scripts/lql_matrix/tensor_presence_test.py`

**Interfaces:**
- Produces: `presence(listing: dict) -> dict` (keys: `n_files:int, classes:dict, has_attn:bool, has_ffn:bool, has_gate:bool, ffn_unwrap_risk:bool`); `listing_of(vindex_dir: str) -> dict` (`{filename: size_bytes}`).

- [ ] **Step 1: Write the failing test**

```python
# scripts/lql_matrix/tensor_presence_test.py
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import tensor_presence as T

ATTN_ONLY = {"attn_weights.bin": 1000, "gate_vectors.bin": 500, "embeddings.bin": 200, "index.json": 10}
FULL      = {"attn_weights.bin": 1000, "up_weights.bin": 800, "down_weights.bin": 800,
             "gate_vectors.bin": 500, "embeddings.bin": 200}
EMPTY     = {"index.json": 10, "gate_vectors.bin": 0}   # 0-byte gate = absent


def test_presence_attention_only_flags_ffn_unwrap_risk():
    p = T.presence(ATTN_ONLY)
    assert p["has_attn"] is True and p["has_ffn"] is False
    assert p["ffn_unwrap_risk"] is True

def test_presence_full_has_ffn_no_risk():
    p = T.presence(FULL)
    assert p["has_attn"] and p["has_ffn"] and p["has_gate"]
    assert p["ffn_unwrap_risk"] is False

def test_presence_empty_no_attn_no_risk():
    p = T.presence(EMPTY)
    assert p["has_attn"] is False and p["ffn_unwrap_risk"] is False

def test_presence_tolerates_none_and_nonint():
    assert T.presence(None)["n_files"] == 0
    assert T.presence({"attn_weights.bin": "big"})["has_attn"] is False  # non-int size ignored
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd scripts/lql_matrix && python3 -m pytest tensor_presence_test.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tensor_presence'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/lql_matrix/tensor_presence.py
"""Tensor-presence witness for the σ-expansion (closes Q8). A produced vindex's
weight-file listing → a presence vector over weight classes. The presence of
attention vs FFN weight files at a level is the latent variable explaining why
some partial/hollow vindexes panic on INFER (guard passes, FFN absent) while
others refuse gracefully (no attention weights → guard refuses)."""
import glob
import json
import os
import re
import sys
from pathlib import Path

# vindex weight file → the class it witnesses
_CLASS = {
    "attn_weights.bin": "attn",
    "up_weights.bin": "ffn_up",
    "down_weights.bin": "ffn_down",
    "gate_vectors.bin": "gate",
    "embeddings.bin": "embed",
    "norms.bin": "norm",
    "down_meta.bin": "down_meta",
    "interleaved_kquant.bin": "kquant",
    "lm_head.bin": "lm_head",
}


def presence(listing):
    cls = {}
    for fn, sz in (listing or {}).items():
        c = _CLASS.get(fn)
        if c and isinstance(sz, int) and sz > 0:
            cls[c] = sz
    has_attn = "attn" in cls
    has_ffn = ("ffn_up" in cls) and ("ffn_down" in cls)
    return {
        "n_files": len(listing or {}),
        "classes": cls,
        "has_attn": has_attn,
        "has_ffn": has_ffn,
        "has_gate": "gate" in cls,
        "ffn_unwrap_risk": has_attn and not has_ffn,
    }


def listing_of(vindex_dir):
    try:
        return {f: os.path.getsize(os.path.join(vindex_dir, f))
                for f in os.listdir(vindex_dir)}
    except OSError:
        return {}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd scripts/lql_matrix && python3 -m pytest tensor_presence_test.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/lql_matrix/tensor_presence.py scripts/lql_matrix/tensor_presence_test.py
git commit -m "feat(inventory): presence() witness — attention vs FFN weight-file presence"
```

---

### Task 2: `load_listing()` + `main()`

**Files:**
- Modify: `scripts/lql_matrix/tensor_presence.py`
- Test: `scripts/lql_matrix/tensor_presence_test.py`

**Interfaces:**
- Produces: `load_listing(path)->dict`; `collect(results_glob)->dict[str,dict]` (maps `leg → presence`, keyed off `manifest-<leg>/listing.json`); `main()` CLI: `<results_glob> <out_json>`.

- [ ] **Step 1: Write the failing test**

```python
# append to tensor_presence_test.py
def test_collect_maps_leg_to_presence(tmp_path):
    d = tmp_path / "results-qwen05.native.attention" / "manifest-qwen05.native.attention"
    d.mkdir(parents=True)
    (d / "listing.json").write_text(json.dumps(ATTN_ONLY), encoding="utf-8")
    d2 = tmp_path / "results-qwen05.native.all" / "manifest-qwen05.native.all"
    d2.mkdir(parents=True)
    (d2 / "listing.json").write_text(json.dumps(FULL), encoding="utf-8")
    got = T.collect(str(tmp_path / "results-*/manifest-*/listing.json"))
    assert got["qwen05.native.attention"]["ffn_unwrap_risk"] is True
    assert got["qwen05.native.all"]["ffn_unwrap_risk"] is False

def test_load_listing_tolerates_missing(tmp_path):
    assert T.load_listing(str(tmp_path / "nope.json")) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd scripts/lql_matrix && python3 -m pytest tensor_presence_test.py -k "collect or load_listing" -v`
Expected: FAIL — `AttributeError: module 'tensor_presence' has no attribute 'collect'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to tensor_presence.py
def load_listing(path):
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


_LEG_RE = re.compile(r"manifest-(.+?)/listing\.json$")


def collect(results_glob):
    rows = {}
    seen = set()
    pats = sorted(glob.glob(results_glob)) + sorted(
        glob.glob("**/" + results_glob, recursive=True))
    for p in pats:
        if p in seen:
            continue
        seen.add(p)
        m = _LEG_RE.search(p)
        if m:
            rows[m.group(1)] = presence(load_listing(p))
    return rows


def main():
    args = sys.argv[1:]
    results_glob = args[0] if args else "artifacts/results-*/manifest-*/listing.json"
    out_json = args[1] if len(args) > 1 else "presence.json"
    rows = collect(results_glob)
    Path(out_json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"tensor-presence: {len(rows)} legs, "
          f"{sum(1 for r in rows.values() if r['ffn_unwrap_risk'])} at ffn_unwrap_risk")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd scripts/lql_matrix && python3 -m pytest tensor_presence_test.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add scripts/lql_matrix/tensor_presence.py scripts/lql_matrix/tensor_presence_test.py
git commit -m "feat(inventory): collect() + main — leg→presence over manifest listings"
```

---

### Task 3: `resolve_q8()` — join presence with panic, assert the biconditional

**Files:**
- Modify: `scripts/lql_matrix/tensor_presence.py`
- Test: `scripts/lql_matrix/tensor_presence_test.py`

**Interfaces:**
- Produces: `resolve_q8(presence: dict[str,dict], panic: dict[str,bool]) -> dict` returning `{rows: list, biconditional_holds: bool, counterexamples: list}` where `panic[leg]` is the observed `panic@infer` and the biconditional is `ffn_unwrap_risk(leg) ⟺ panic(leg)` over the shared legs.

- [ ] **Step 1: Write the failing test**

```python
# append to tensor_presence_test.py
def test_resolve_q8_biconditional_and_counterexample():
    pres = {
        "qwen05.native.attention":   T.presence(ATTN_ONLY),   # risk=True
        "qwen05.native.all":         T.presence(FULL),        # risk=False
        "bitnet2b.native.attention": T.presence(EMPTY),       # risk=False (no attn wt)
    }
    panic = {  # observed panic@infer (from conformance no-crash)
        "qwen05.native.attention":   True,
        "qwen05.native.all":         False,
        "bitnet2b.native.attention": False,   # refuses gracefully — matches risk=False
    }
    r = T.resolve_q8(pres, panic)
    assert r["biconditional_holds"] is True
    assert r["counterexamples"] == []
    # now inject a violation: risk=True but no panic
    panic["qwen05.native.attention"] = False
    r2 = T.resolve_q8(pres, panic)
    assert r2["biconditional_holds"] is False
    assert "qwen05.native.attention" in [c["leg"] for c in r2["counterexamples"]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd scripts/lql_matrix && python3 -m pytest tensor_presence_test.py -k resolve_q8 -v`
Expected: FAIL — `AttributeError: ... 'resolve_q8'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to tensor_presence.py
def resolve_q8(presence_by_leg, panic_by_leg):
    rows, counter = [], []
    for leg in sorted(set(presence_by_leg) & set(panic_by_leg)):
        risk = bool(presence_by_leg[leg]["ffn_unwrap_risk"])
        pan = bool(panic_by_leg[leg])
        agree = (risk == pan)
        rows.append({"leg": leg, "ffn_unwrap_risk": risk, "panic": pan, "agree": agree})
        if not agree:
            counter.append({"leg": leg, "ffn_unwrap_risk": risk, "panic": pan})
    return {"rows": rows, "biconditional_holds": not counter, "counterexamples": counter}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd scripts/lql_matrix && python3 -m pytest tensor_presence_test.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add scripts/lql_matrix/tensor_presence.py scripts/lql_matrix/tensor_presence_test.py
git commit -m "feat(inventory): resolve_q8 — assert ffn_unwrap_risk <=> panic@infer"
```

---

### Task 4: Capture `listing.json` in produce + add the `inventory` CI job

**Files:**
- Modify: `.github/workflows/lql-strategy-matrix.yml`

**Interfaces:**
- Consumes: `tensor_presence.py` (`main()` CLI; `collect`, `resolve_q8`), the matrix's uploaded `manifest-<leg>/` artifacts, and the `conformance` job's `conformance.json` (for the panic map).

- [ ] **Step 1: Capture the vindex file listing in the produce step**

In `.github/workflows/lql-strategy-matrix.yml`, in the "Produce vindex" step, immediately after the existing weight-family inventory block (the line `cp "$VINDEX/index.json" "$VINDEX/weight_manifest.json" "$MDIR/" 2>/dev/null || true`), add:

```bash
          # tensor-presence witness (Q8): filename -> size for every vindex file
          python3 -c "import json,os,sys; d=sys.argv[1]; print(json.dumps({f: os.path.getsize(os.path.join(d,f)) for f in os.listdir(d)}) if os.path.isdir(d) else '{}')" \
            "$VINDEX" > "$MDIR/listing.json" 2>/dev/null || echo '{}' > "$MDIR/listing.json"
```

(The `manifest-${NAME}/` directory is already in the "Upload leg results" `path:` list, so `listing.json` ships with it — no upload change needed.)

- [ ] **Step 2: Add the peer `inventory` job**

In `.github/workflows/lql-strategy-matrix.yml`, after the `conformance` job, add:

```yaml
  inventory:
    needs: [matrix, conformance]
    if: always()
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: actions/download-artifact@v4
        with:
          path: artifacts
      - name: Tensor-presence + Q8 resolution
        run: |
          python3 scripts/lql_matrix/tensor_presence.py \
            "artifacts/results-*/manifest-*/listing.json" presence.json
          python3 - <<'PY'
          import json, glob, sys
          sys.path.insert(0, "scripts/lql_matrix")
          import tensor_presence as T
          pres = json.load(open("presence.json"))
          # panic := any no-crash (SIGKILL/panic, ec=101) violation on the leg
          cj = glob.glob("artifacts/conformance/conformance.json")
          viol = json.load(open(cj[0]))["violations"] if cj else []
          panicked = {v["leg"] for v in viol if v["invariant"] == "no-crash"}
          # Q8 / M3(a) is the *native attention-level* asymmetry (granite panics, bitnet refuses).
          # Scope the biconditional to native legs; q4k legs panic by the distinct K6 hollow mechanism.
          native = {k: v for k, v in pres.items() if ".native." in k}
          panic = {leg: (leg in panicked) for leg in native}
          res = T.resolve_q8(native, panic)
          json.dump(res, open("q8.json", "w"), indent=2)
          L = ["# Q8 — panic ⟺ (has_attn ∧ ¬has_ffn)?", "",
               f"biconditional holds: **{res['biconditional_holds']}** · counterexamples: {len(res['counterexamples'])}", "",
               "| leg | ffn_unwrap_risk | panic | agree |", "|---|---|---|---|"]
          for r in res["rows"]:
              L.append(f"| `{r['leg']}` | {r['ffn_unwrap_risk']} | {r['panic']} | {'✅' if r['agree'] else '❌'} |")
          open("q8.md","w").write("\n".join(L)+"\n")
          print("\n".join(L))
          PY
          cat q8.md >> "$GITHUB_STEP_SUMMARY"
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: inventory
          retention-days: 1
          path: |
            presence.json
            q8.json
            q8.md
```

- [ ] **Step 3: Validate**

Run:
```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/lql-strategy-matrix.yml')); print('workflow valid')"
cd scripts/lql_matrix && python3 -m pytest tensor_presence_test.py -q
```
Expected: `workflow valid` and pytest all-pass.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/lql-strategy-matrix.yml
git commit -m "ci(inventory): capture vindex listing + peer inventory job resolving Q8"
```

---

## Verification (end-to-end, before pushing)

1. `cd scripts/lql_matrix && python3 -m pytest tensor_presence_test.py -v` → all pass.
2. Local witness on a real vindex (if one exists on the box), no model load:
   ```bash
   python3 -c "import sys; sys.path.insert(0,'scripts/lql_matrix'); import tensor_presence as T; print(T.presence(T.listing_of('/home/metavacua/work/vindexes/smollm2-360m-canonical.vindex')))"
   ```
   Expected: a full vindex → `has_attn=True, has_ffn=True, ffn_unwrap_risk=False`.
3. On the next full CI run, the `inventory` job's `q8.md` (scoped to native legs) should show `biconditional holds: True` — every attention-level dense leg `ffn_unwrap_risk=True ∧ panic=True`, and `bitnet2b.native.attention` `ffn_unwrap_risk=False ∧ panic=False` (the M3(a) duplicate-row broken by the `has_attn` column: bitnet@attention has no attention-weight file, so the guard refuses rather than unwrapping an absent FFN). A `❌` row is a real counterexample to investigate — and `presence.json` (all legs, including q4k) separately shows whether the q4k hollow panics share the `ffn_unwrap_risk` signature or need the distinct `hollow` column (K6).

## Sibling plans (separate Δ columns, not this plan)
- **W₅ binary-sweep (Q6)** — needs the candidate-refs input.
- **W₆ encoding (Q7)** — needs a TQ1_0 gguf (`llama-quantize`).
- **W₄ confound (Q9)** — uncertain constructibility of a `¬hollow ∧ FFN-absent` q4k vindex.
- Deferred (σ_device, blocked on larql-cli/repl resource discipline): **Q5 Φ_local, N4 contention**.
