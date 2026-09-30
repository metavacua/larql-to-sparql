# CI Conformance Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the lql-strategy-matrix from a pure discovery experiment into an empirical V&V oracle: a `conformance.py` + peer CI job that asserts principled no-expected-output invariants over the already-captured artifacts, plus a transformation/consumability coverage report that surfaces interface gaps (e.g. no compiled→GGUF/Ollama path, no cross-arch qwen↔bitnet transform).

**Architecture:** A standalone `scripts/lql_matrix/conformance.py` loads each leg's `results-*.jsonl` + `descriptor-*.json` + `produce-*.json`, runs an invariant registry (Completeness, No-crash, Descriptor self-match, Cross-check, Diagnostic), emits `conformance.md` + `conformance.json`, and exits non-zero only under `--strict`. `aggregate.py` stays descriptive; a peer `conformance` CI job (strict via `workflow_dispatch` input) runs the checker. A Transformation/Consumability coverage section (descriptive) surfaces gaps without running a server.

**Tech Stack:** Python 3.12 stdlib only (matches `run_matrix.py`/`aggregate.py`); pytest 7.2.1 for tests; GitHub Actions YAML.

## Global Constraints

- Python: **stdlib only** — no third-party imports in `conformance.py` (json, re, glob, sys, dataclasses, pathlib). `pytest` is test-only.
- **No larql code changes.** Harness/CI only. Read only the existing artifact fields.
- **Discovery preserved:** default run stays green; violations gate the run **only** under `--strict` (a `workflow_dispatch` boolean input). Resolves tracker #247.
- **No new capture:** `feature_count` is parsed from the `(L layers, F features` banner present in every cell's `stdout_head`; recipe `op`/`level`/`flags` come from `produce-<leg>.json`.
- Artifacts retention: **1 day** (`retention-days: 1`), consistent with the rest of the workflow.
- Encoding: always `encoding="utf-8"`; tolerant of missing/malformed artifacts (never crash the checker).
- Out of scope for v1: runtime `larql serve` + `/v1/chat` drive (Phase 2); model quality / output correctness; T3 browser/web-llm.

---

## File Structure

- Create: `scripts/lql_matrix/conformance.py` — the invariant registry, coverage report, gate. One responsibility: the oracle.
- Create: `scripts/lql_matrix/conformance_test.py` — pytest tests over synthetic in-memory `Leg` fixtures (no file I/O except the `load()` test).
- Modify: `.github/workflows/lql-strategy-matrix.yml` — add a peer `conformance` job + a `strict` `workflow_dispatch` input.
- Modify: `.github/workflows/lql-matrix-smoke.yml` — run `conformance_test.py` (fast, no larql) so the oracle is validated on a runner.

`descriptor.py`, `aggregate.py`, `run_matrix.py`, `gen_legs.py` are **unchanged** — `conformance.py` reads their existing outputs.

---

### Task 1: conformance.py skeleton — data model, loader, empty registry, CLI

**Files:**
- Create: `scripts/lql_matrix/conformance.py`
- Test: `scripts/lql_matrix/conformance_test.py`

**Interfaces:**
- Produces: `Leg` (dataclass: `name:str, meta:dict, cells:dict, descriptor:dict, produce:dict`); `Violation` (dataclass: `invariant:str, leg:str, cell:str, detail:str`); `load(results_glob:str)->dict[str,Leg]`; `INVARIANTS:list[callable]` (each `callable(dict[str,Leg])->list[Violation]`); `run(results_glob:str, out_md:str, out_json:str, strict:bool)->int`.

- [ ] **Step 1: Write the failing test**

```python
# scripts/lql_matrix/conformance_test.py
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import conformance as C


def make_leg(name="lg", cells=None, descriptor=None, produce=None, meta=None):
    return C.Leg(name=name, meta=meta or {"level": name},
                 cells=cells or {}, descriptor=descriptor or {},
                 produce=produce or {})


def test_load_reads_meta_cells_descriptor_produce(tmp_path):
    d = tmp_path / "results-lgA"
    d.mkdir()
    (d / "results-lgA.jsonl").write_text(
        json.dumps({"type": "meta", "level": "lgA"}) + "\n" +
        json.dumps({"level": "lgA", "id": "stats", "bucket": "ok",
                    "exit_code": 0, "stdout_head": "(24 layers, 5K features, m)"}) + "\n",
        encoding="utf-8")
    (d / "descriptor-lgA.json").write_text(json.dumps({"name": "lgA", "family": "qwen2"}), encoding="utf-8")
    (d / "produce-lgA.json").write_text(json.dumps({"name": "lgA", "op": "extract", "bucket": "ok"}), encoding="utf-8")
    legs = C.load(str(tmp_path / "results-*/results-*.jsonl"))
    assert set(legs) == {"lgA"}
    assert legs["lgA"].cells["stats"]["bucket"] == "ok"
    assert legs["lgA"].descriptor["family"] == "qwen2"
    assert legs["lgA"].produce["op"] == "extract"


def test_run_with_empty_registry_is_green(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "INVARIANTS", [])
    legs = {"lg": make_leg()}
    monkeypatch.setattr(C, "load", lambda g: legs)
    code = C.run("x", str(tmp_path / "c.md"), str(tmp_path / "c.json"), strict=True)
    assert code == 0
    assert json.loads((tmp_path / "c.json").read_text())["violations"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd scripts/lql_matrix && python3 -m pytest conformance_test.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'conformance'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/lql_matrix/conformance.py
"""Conformance oracle for the lql-strategy-matrix. Evaluates principled,
no-expected-output invariants over already-captured artifacts and emits a
report. Default exit 0 (discovery preserved); non-zero only under --strict."""
import glob
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Leg:
    name: str
    meta: dict = field(default_factory=dict)
    cells: dict = field(default_factory=dict)       # id -> row
    descriptor: dict = field(default_factory=dict)
    produce: dict = field(default_factory=dict)


@dataclass
class Violation:
    invariant: str
    leg: str
    cell: str
    detail: str


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def load(results_glob):
    legs = {}
    for rf in sorted(glob.glob(results_glob)):
        d = Path(rf).parent
        for line in Path(rf).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = r.get("level")
            if not name:
                continue
            lg = legs.setdefault(name, Leg(name=name))
            if r.get("type") == "meta":
                lg.meta = r
            else:
                lg.cells[r.get("id", "?")] = r
        # sidecars: descriptor-<name>.json / produce-<name>.json in the same dir
        for name, lg in legs.items():
            dp = d / f"descriptor-{name}.json"
            pp = d / f"produce-{name}.json"
            if dp.exists():
                lg.descriptor = _read_json(dp)
            if pp.exists():
                lg.produce = _read_json(pp)
    return legs


INVARIANTS = []


def run(results_glob, out_md, out_json, strict):
    legs = load(results_glob)
    violations = [v for inv in INVARIANTS for v in inv(legs)]
    Path(out_json).write_text(json.dumps(
        {"violations": [v.__dict__ for v in violations]}, indent=2), encoding="utf-8")
    Path(out_md).write_text(render(legs, violations), encoding="utf-8")
    print(f"conformance: {len(violations)} violation(s) across {len(legs)} legs "
          f"(strict={strict})")
    return 1 if (strict and violations) else 0


def render(legs, violations):
    L = ["# LQL Matrix — Conformance", "",
         f"Legs: {len(legs)} · Violations: {len(violations)}", ""]
    if violations:
        L += ["| invariant | leg | cell | detail |", "|---|---|---|---|"]
        for v in violations:
            L.append(f"| {v.invariant} | `{v.leg}` | {v.cell or '-'} | {v.detail} |")
    else:
        L.append("No invariant violations.")
    return "\n".join(L) + "\n"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    strict = "--strict" in sys.argv[1:]
    results_glob = args[0] if args else "artifacts/results-*/results-*.jsonl"
    out_md = args[1] if len(args) > 1 else "conformance.md"
    out_json = args[2] if len(args) > 2 else "conformance.json"
    sys.exit(run(results_glob, out_md, out_json, strict))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd scripts/lql_matrix && python3 -m pytest conformance_test.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/lql_matrix/conformance.py scripts/lql_matrix/conformance_test.py
git commit -m "feat(conformance): skeleton — Leg/Violation model, loader, empty registry, gate"
```

---

### Task 2: Completeness invariant (feature_count > 0)

**Files:**
- Modify: `scripts/lql_matrix/conformance.py`
- Test: `scripts/lql_matrix/conformance_test.py`

**Interfaces:**
- Produces: `feature_count(leg:Leg)->int|None` (parses the `(L layers, F features` banner from any cell's `stdout_head`; `K`/`M` suffix expanded); `inv_completeness(dict[str,Leg])->list[Violation]`. Appends `inv_completeness` to `INVARIANTS`.

- [ ] **Step 1: Write the failing test**

```python
# append to conformance_test.py
def _cell(cid, stdout="", bucket="ok", exit_code=0, err_signal=0, err_line="", stderr=""):
    return {"id": cid, "bucket": bucket, "exit_code": exit_code, "err_signal": err_signal,
            "err_line": err_line, "stdout_head": stdout, "stderr_head": stderr}


def test_feature_count_parses_banner_with_suffix():
    lg = make_leg(cells={"stats": _cell("stats", "Using: x (24 layers, 116.7K features, m)")})
    assert C.feature_count(lg) == 116700
    lg0 = make_leg(cells={"stats": _cell("stats", "Using: x (24 layers, 0 features, m)")})
    assert C.feature_count(lg0) == 0


def test_completeness_flags_hollow_vindex():
    hollow = make_leg("granite", cells={"stats": _cell("stats", "(24 layers, 0 features, m)")})
    healthy = make_leg("qwen", cells={"stats": _cell("stats", "(24 layers, 5K features, m)")})
    vs = C.inv_completeness({"granite": hollow, "qwen": healthy})
    assert [v.leg for v in vs] == ["granite"]
    assert vs[0].invariant == "completeness"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd scripts/lql_matrix && python3 -m pytest conformance_test.py -k "feature_count or completeness" -v`
Expected: FAIL — `AttributeError: module 'conformance' has no attribute 'feature_count'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to conformance.py (above `INVARIANTS = []`)
_FEAT = re.compile(r"\(\s*\d+\s+layers?,\s*([\d.]+)\s*([KM]?)\s+features", re.I)


def feature_count(leg):
    for row in leg.cells.values():
        m = _FEAT.search(row.get("stdout_head", "") or "")
        if m:
            n = float(m.group(1))
            n *= {"K": 1e3, "M": 1e6, "": 1}[m.group(2).upper()]
            return int(round(n))
    return None


def inv_completeness(legs):
    out = []
    for name, lg in legs.items():
        fc = feature_count(lg)
        if fc is None:
            out.append(Violation("completeness", name, "",
                                 "feature_count unknown (no STATS/banner — produce may have failed)"))
        elif fc == 0:
            out.append(Violation("completeness", name, "",
                                 "hollow vindex: 0 features (silent-incomplete extraction, cf #183)"))
    return out
```

Then change the registry line:

```python
INVARIANTS = [inv_completeness]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd scripts/lql_matrix && python3 -m pytest conformance_test.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add scripts/lql_matrix/conformance.py scripts/lql_matrix/conformance_test.py
git commit -m "feat(conformance): completeness invariant — flag hollow (0-feature) vindexes"
```

---

### Task 3: No-crash invariant (panic/crash in produce or any cell)

**Files:**
- Modify: `scripts/lql_matrix/conformance.py`
- Test: `scripts/lql_matrix/conformance_test.py`

**Interfaces:**
- Produces: `inv_no_crash(dict[str,Leg])->list[Violation]`. A cell/produce is a crash iff `bucket == "crash"` or `exit_code in (101, 134, 137, 139)`. Graceful in-band error (exit 0) or clean non-zero (e.g. 3) is NOT a crash. Appends to `INVARIANTS`.

- [ ] **Step 1: Write the failing test**

```python
# append to conformance_test.py
def test_no_crash_flags_panic_and_crash_not_graceful():
    legs = {
        "panic": make_leg("panic", cells={"infer.top5": _cell("infer.top5", bucket="err", exit_code=101)}),
        "crash": make_leg("crash", cells={"x": _cell("x", bucket="crash", exit_code=137)}),
        "graceful": make_leg("graceful", cells={"infer.top5": _cell("infer.top5", bucket="ok", exit_code=0, err_signal=1, err_line="Error: requires model weights")}),
        "cleanerr": make_leg("cleanerr", cells={"y": _cell("y", bucket="err", exit_code=3)}),
        "produce_panic": make_leg("pp", produce={"bucket": "crash", "exit_code": 139}),
    }
    vs = C.inv_no_crash(legs)
    flagged = sorted({v.leg for v in vs})
    assert flagged == ["crash", "panic", "produce_panic"]
    assert all(v.invariant == "no-crash" for v in vs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd scripts/lql_matrix && python3 -m pytest conformance_test.py -k no_crash -v`
Expected: FAIL — `AttributeError: ... 'inv_no_crash'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to conformance.py
_CRASH_CODES = {101, 134, 137, 139}


def _is_crash(row):
    return row.get("bucket") == "crash" or row.get("exit_code") in _CRASH_CODES


def inv_no_crash(legs):
    out = []
    for name, lg in legs.items():
        if lg.produce and _is_crash(lg.produce):
            out.append(Violation("no-crash", name, "produce",
                                 f"produce crashed (exit {lg.produce.get('exit_code')})"))
        for cid, row in lg.cells.items():
            if _is_crash(row):
                out.append(Violation("no-crash", name, cid,
                                     f"panic/crash (exit {row.get('exit_code')})"))
    return out
```

Then extend the registry:

```python
INVARIANTS = [inv_completeness, inv_no_crash]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd scripts/lql_matrix && python3 -m pytest conformance_test.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add scripts/lql_matrix/conformance.py scripts/lql_matrix/conformance_test.py
git commit -m "feat(conformance): no-crash invariant — panic/crash in produce or corpus is a violation"
```

---

### Task 4: Descriptor self-match invariant (produced == declared intent)

**Files:**
- Modify: `scripts/lql_matrix/conformance.py`
- Test: `scripts/lql_matrix/conformance_test.py`

**Interfaces:**
- Produces: `inv_descriptor_match(dict[str,Leg])->list[Violation]`. Violation when `descriptor.quant_match` is falsy, OR `descriptor.family` is falsy/`"generic"` (silent GenericArch fallback, cf #154). Legs with no descriptor (produce failed) are skipped here — No-crash/Completeness cover them. Appends to `INVARIANTS`.

- [ ] **Step 1: Write the failing test**

```python
# append to conformance_test.py
def test_descriptor_match_flags_quant_mismatch_and_generic_fallback():
    legs = {
        "ok": make_leg("ok", descriptor={"name": "ok", "quant_match": True, "family": "qwen2"}),
        "dequant": make_leg("dequant", descriptor={"name": "dequant", "quant_match": False,
                            "observed_quant": "none", "expect_quant": "q4k", "family": "qwen2"}),
        "generic": make_leg("generic", descriptor={"name": "generic", "quant_match": True, "family": "generic"}),
        "nodesc": make_leg("nodesc", descriptor={}),
    }
    vs = C.inv_descriptor_match(legs)
    assert sorted({v.leg for v in vs}) == ["dequant", "generic"]
    assert all(v.invariant == "descriptor-match" for v in vs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd scripts/lql_matrix && python3 -m pytest conformance_test.py -k descriptor_match -v`
Expected: FAIL — `AttributeError: ... 'inv_descriptor_match'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to conformance.py
def inv_descriptor_match(legs):
    out = []
    for name, lg in legs.items():
        d = lg.descriptor
        if not d or not d.get("produced", True) or d.get("error"):
            continue  # produce-side failure — covered by completeness/no-crash
        if not d.get("quant_match", True):
            out.append(Violation("descriptor-match", name, "",
                                 f"quant mismatch: produced {d.get('observed_quant')} != "
                                 f"expected {d.get('expect_quant')}"))
        fam = (d.get("family") or "").lower()
        if fam in ("", "generic"):
            out.append(Violation("descriptor-match", name, "",
                                 f"unrecognized/generic arch fallback (family={d.get('family')!r}, cf #154)"))
    return out
```

Then extend the registry:

```python
INVARIANTS = [inv_completeness, inv_no_crash, inv_descriptor_match]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd scripts/lql_matrix && python3 -m pytest conformance_test.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add scripts/lql_matrix/conformance.py scripts/lql_matrix/conformance_test.py
git commit -m "feat(conformance): descriptor self-match — quant mismatch + generic-arch fallback"
```

---

### Task 5: Cross-check invariant (SHOW LAYERS count == STATS count)

**Files:**
- Modify: `scripts/lql_matrix/conformance.py`
- Test: `scripts/lql_matrix/conformance_test.py`

**Interfaces:**
- Produces: `show_layers_total(leg:Leg)->int|None` (sums the per-layer feature column from the `show.layers` cell stdout); `inv_cross_check(dict[str,Leg])->list[Violation]`. Violation when `show_layers_total == 0` while `feature_count > 0` (the mmap SHOW LAYERS bug S1). Appends to `INVARIANTS`.

- [ ] **Step 1: Write the failing test**

```python
# append to conformance_test.py
SHOW_LAYERS_HOLLOW = ("Layer      Features  With Meta       Top Token\n"
                      "------------------------------------------------\n"
                      "L0                0          0\n"
                      "L1                0          0\n")
SHOW_LAYERS_OK = ("Layer      Features  With Meta       Top Token\n"
                  "------------------------------------------------\n"
                  "L0             2560          0\n"
                  "L1             2560          0\n")


def test_cross_check_flags_show_layers_zero_vs_stats_nonzero():
    bad = make_leg("bad", cells={
        "stats": _cell("stats", "(24 layers, 5K features, m)"),
        "show.layers": _cell("show.layers", SHOW_LAYERS_HOLLOW)})
    good = make_leg("good", cells={
        "stats": _cell("stats", "(24 layers, 5K features, m)"),
        "show.layers": _cell("show.layers", SHOW_LAYERS_OK)})
    vs = C.inv_cross_check({"bad": bad, "good": good})
    assert [v.leg for v in vs] == ["bad"]
    assert vs[0].invariant == "cross-check"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd scripts/lql_matrix && python3 -m pytest conformance_test.py -k cross_check -v`
Expected: FAIL — `AttributeError: ... 'show_layers_total'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to conformance.py
_LAYER_ROW = re.compile(r"^L\d+\s+(\d+)\s+\d+", re.M)


def show_layers_total(leg):
    row = leg.cells.get("show.layers")
    if not row:
        return None
    counts = _LAYER_ROW.findall(row.get("stdout_head", "") or "")
    return sum(int(c) for c in counts) if counts else None


def inv_cross_check(legs):
    out = []
    for name, lg in legs.items():
        sl = show_layers_total(lg)
        fc = feature_count(lg)
        if sl is not None and fc and sl == 0:
            out.append(Violation("cross-check", name, "show.layers",
                                 f"SHOW LAYERS reports 0 features but STATS reports {fc} "
                                 "(mmap heap-only accessor, cf SHOW LAYERS bug)"))
    return out
```

Then extend the registry:

```python
INVARIANTS = [inv_completeness, inv_no_crash, inv_descriptor_match, inv_cross_check]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd scripts/lql_matrix && python3 -m pytest conformance_test.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add scripts/lql_matrix/conformance.py scripts/lql_matrix/conformance_test.py
git commit -m "feat(conformance): cross-check — SHOW LAYERS count must equal STATS count"
```

---

### Task 6: Diagnostic invariant (flag-honored + message-attribution)

**Files:**
- Modify: `scripts/lql_matrix/conformance.py`
- Test: `scripts/lql_matrix/conformance_test.py`

**Interfaces:**
- Produces: `inv_diagnostic(dict[str,Leg])->list[Violation]` with two checks: (R-2 flag-honored) a `q4k` extract leg with `level in {browse,attention,inference}` whose `produce` output contains no warn/override/ignore text → the `--level` was silently overridden (S2); (R-3 message-attribution) any cell whose `err_line` names `--compact` while the recipe `flags` do not contain `compact` → misattribution (B2). Appends to `INVARIANTS`.

- [ ] **Step 1: Write the failing test**

```python
# append to conformance_test.py
def test_diagnostic_flag_honored_and_message_attribution():
    silent_override = make_leg("q4kbrowse",
        produce={"op": "extract", "level": "browse", "flags": "--quant q4k",
                 "stdout_head": "Extracting…", "stderr_head": ""})
    honored = make_leg("q4kall",
        produce={"op": "extract", "level": "all", "flags": "--quant q4k",
                 "stdout_head": "Extracting…", "stderr_head": ""})
    misattrib = make_leg("mis", produce={"op": "extract", "level": "all", "flags": ""},
        cells={"infer.top5": _cell("infer.top5", bucket="err", exit_code=101,
               err_line="panicked: FFN weight tensor missing — this is a `--compact` vindex")})
    vs = C.inv_diagnostic({"q4kbrowse": silent_override, "q4kall": honored, "mis": misattrib})
    invs = sorted((v.leg, v.invariant) for v in vs)
    assert ("mis", "diagnostic") in invs
    assert ("q4kbrowse", "diagnostic") in invs
    assert not any(v.leg == "q4kall" for v in vs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd scripts/lql_matrix && python3 -m pytest conformance_test.py -k diagnostic -v`
Expected: FAIL — `AttributeError: ... 'inv_diagnostic'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to conformance.py
_WARN = re.compile(r"warn|overrid|ignor", re.I)


def inv_diagnostic(legs):
    out = []
    for name, lg in legs.items():
        p = lg.produce
        # R-2: --quant q4k silently overrides a non-all --level
        if (p.get("op") == "extract" and "--quant q4k" in (p.get("flags") or "")
                and p.get("level") in {"browse", "attention", "inference"}):
            txt = (p.get("stdout_head", "") or "") + (p.get("stderr_head", "") or "")
            if not _WARN.search(txt):
                out.append(Violation("diagnostic", name, "produce",
                                     f"--level {p.get('level')} silently ignored under --quant q4k "
                                     "(no warn/override text; cf #208)"))
        # R-3: error text blames --compact but the recipe never passed it
        flags = (p.get("flags") or "")
        if "compact" not in flags:
            for cid, row in lg.cells.items():
                if "--compact" in (row.get("err_line", "") or ""):
                    out.append(Violation("diagnostic", name, cid,
                                         "error text misattributes to `--compact` (not in recipe flags)"))
                    break
    return out
```

Then extend the registry:

```python
INVARIANTS = [inv_completeness, inv_no_crash, inv_descriptor_match, inv_cross_check, inv_diagnostic]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd scripts/lql_matrix && python3 -m pytest conformance_test.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add scripts/lql_matrix/conformance.py scripts/lql_matrix/conformance_test.py
git commit -m "feat(conformance): diagnostic invariant — silent --level override + --compact misattribution"
```

---

### Task 7: Transformation & Consumability coverage report + strict gate wiring

**Files:**
- Modify: `scripts/lql_matrix/conformance.py`
- Test: `scripts/lql_matrix/conformance_test.py`

**Interfaces:**
- Produces: `transformation_report(dict[str,Leg])->list[str]` (markdown lines): a `source → produced` table (`op`/source-format → produced `family`/`quant`) and a declared-consumer reachability line that SURFACES gaps — larql-cli/larql-server reach any produced vindex; Ollama needs GGUF and there is **no compiled→GGUF export** (gap); cross-arch (produced family always == source) → no arch transform (gap). `render()` appends this section. This is descriptive (not a gating violation — no oracle for "should this transform exist"), per the gap-surfacing intent.

- [ ] **Step 1: Write the failing test**

```python
# append to conformance_test.py
def test_transformation_report_surfaces_gaps():
    legs = {
        "qwen.q4k": make_leg("qwen.q4k",
            produce={"op": "extract", "flags": "--quant q4k"},
            descriptor={"family": "qwen2", "observed_quant": "q4k"}),
        "bitnet.gguf": make_leg("bitnet.gguf",
            produce={"op": "gguf-to-vindex", "flags": ""},
            descriptor={"family": "bitnet", "observed_quant": "none"}),
    }
    md = "\n".join(C.transformation_report(legs))
    assert "Transformation" in md
    assert "qwen2" in md and "bitnet" in md
    # gaps surfaced verbatim
    assert "Ollama" in md and "GGUF" in md
    assert "no compiled" in md.lower() or "gap" in md.lower()


def test_run_strict_fails_on_violation(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "INVARIANTS", [lambda legs: [C.Violation("x", "lg", "", "boom")]])
    monkeypatch.setattr(C, "load", lambda g: {"lg": make_leg()})
    assert C.run("x", str(tmp_path/"c.md"), str(tmp_path/"c.json"), strict=True) == 1
    assert C.run("x", str(tmp_path/"c.md"), str(tmp_path/"c.json"), strict=False) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd scripts/lql_matrix && python3 -m pytest conformance_test.py -k "transformation or strict" -v`
Expected: FAIL — `AttributeError: ... 'transformation_report'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to conformance.py
_SRC_FMT = {"extract": "safetensors", "gguf-to-vindex": "gguf",
            "quantize-q4k": "safetensors", "quantize-fp4": "safetensors"}


def transformation_report(legs):
    L = ["## Transformation & consumability coverage", "",
         "| leg | source fmt | produced arch | produced quant |",
         "|---|---|---|---|"]
    for name in sorted(legs):
        lg = legs[name]
        src = _SRC_FMT.get(lg.produce.get("op"), "?")
        L.append(f"| `{name}` | {src} | {lg.descriptor.get('family','?')} | "
                 f"{lg.descriptor.get('observed_quant','?')} |")
    L += ["",
          "**Consumer reachability & gaps (static):**",
          "- larql-cli / larql-server (`/v1/chat`): reach any produced vindex directly.",
          "- **Ollama:** needs GGUF; larql `compile` emits safetensors and there is **no "
          "compiled→GGUF export** — Ollama is a **gap** (cf #181/N11).",
          "- **Cross-arch transform** (e.g. qwen↔bitnet): produced arch equals source arch in "
          "every leg — larql changes quant/format, **not architecture** — a **gap** for the "
          "'transform one arch into another' use case.", ""]
    return L
```

Then extend `render()` to append the section — replace the existing `render` body's final `return` with:

```python
def render(legs, violations):
    L = ["# LQL Matrix — Conformance", "",
         f"Legs: {len(legs)} · Violations: {len(violations)}", ""]
    if violations:
        L += ["| invariant | leg | cell | detail |", "|---|---|---|---|"]
        for v in violations:
            L.append(f"| {v.invariant} | `{v.leg}` | {v.cell or '-'} | {v.detail} |")
    else:
        L.append("No invariant violations.")
    L += [""] + transformation_report(legs)
    return "\n".join(L) + "\n"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd scripts/lql_matrix && python3 -m pytest conformance_test.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add scripts/lql_matrix/conformance.py scripts/lql_matrix/conformance_test.py
git commit -m "feat(conformance): transformation/consumability coverage report — surface Ollama/GGUF + cross-arch gaps"
```

---

### Task 8: Wire the conformance CI job + strict input; run tests in smoke

**Files:**
- Modify: `.github/workflows/lql-strategy-matrix.yml`
- Modify: `.github/workflows/lql-matrix-smoke.yml`

**Interfaces:**
- Consumes: `conformance.py` (`main()` CLI: `<results_glob> <out_md> <out_json> [--strict]`), `conformance_test.py`.

- [ ] **Step 1: Add the `strict` input to the strategy-matrix `workflow_dispatch`**

In `.github/workflows/lql-strategy-matrix.yml`, replace `workflow_dispatch: {}` with:

```yaml
  workflow_dispatch:
    inputs:
      strict:
        description: "Fail the run if any conformance invariant is violated"
        type: boolean
        default: false
```

- [ ] **Step 2: Add the peer `conformance` job**

In `.github/workflows/lql-strategy-matrix.yml`, after the `aggregate` job, add:

```yaml
  conformance:
    needs: matrix
    if: always()
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: actions/download-artifact@v4
        with:
          path: artifacts
      - name: Run conformance oracle
        run: |
          STRICT=""
          [ "${{ github.event.inputs.strict }}" = "true" ] && STRICT="--strict"
          python3 scripts/lql_matrix/conformance.py \
            "artifacts/results-*/results-*.jsonl" conformance.md conformance.json $STRICT
          echo "=== conformance.md ==="; cat conformance.md
          cat conformance.md >> "$GITHUB_STEP_SUMMARY"
      - name: Upload conformance report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: conformance
          retention-days: 1
          path: |
            conformance.md
            conformance.json
```

- [ ] **Step 3: Run the oracle's unit tests in the smoke workflow**

In `.github/workflows/lql-matrix-smoke.yml`, add this step before `Upload smoke artifacts`:

```yaml
      - name: Conformance oracle unit tests
        run: |
          python -m pip install --quiet pytest
          python -m pytest scripts/lql_matrix/conformance_test.py -v
```

- [ ] **Step 4: Validate both workflows parse**

Run:
```bash
python3 -c "import yaml; [yaml.safe_load(open(f)) for f in ['.github/workflows/lql-strategy-matrix.yml','.github/workflows/lql-matrix-smoke.yml']]; print('both workflows valid')"
cd scripts/lql_matrix && python3 -m pytest conformance_test.py -q
```
Expected: `both workflows valid` and pytest all-pass.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/lql-strategy-matrix.yml .github/workflows/lql-matrix-smoke.yml
git commit -m "ci(conformance): peer conformance job + strict dispatch input; oracle unit tests in smoke"
```

---

## Verification (end-to-end, before pushing)

1. `cd scripts/lql_matrix && python3 -m pytest conformance_test.py -v` → all pass.
2. Run the oracle against the **already-downloaded** expanded-run artifacts to confirm it catches the real findings:
   ```bash
   python3 scripts/lql_matrix/conformance.py \
     "/tmp/claude-1000/.../ci-expanded/results-*/results-*.jsonl" /tmp/conf.md /tmp/conf.json
   grep -E "granite.*hollow|attention.*panic|SHOW LAYERS|q4k.*silently|Ollama|cross-arch" /tmp/conf.md
   ```
   Expected: completeness violations for granite/bitnet legs, no-crash for `*.attention` + q4k-MoE legs, cross-check for the SHOW LAYERS case, diagnostic for q4k-browse + the `--compact` message, and the transformation report surfacing the Ollama/GGUF + cross-arch gaps.
3. Push the smoke branch first (`lql-matrix-smoke`) → confirm the oracle unit tests pass on a runner, then the real branch.

## v1 scope notes (deferred registry extensions — not silent drops)
The invariant registry is deliberately extensible; two spec-listed checks are deferred past v1 and slot in as additional functions later:
- **Cross-check structural equalities** — `q4k-inline ≡ q4k-posthoc` and `safetensors-to-vindex ≡ extract` (cross-leg descriptor equality). v1 ships only the internal-consistency cross-check (SHOW LAYERS == STATS, Task 5), which catches a real bug now; the cross-leg equalities are lower-value and need leg-pairing logic.
- **Diagnostic R-1 (exit-0-with-error)** — larql `lql` exits 0 even on in-band errors (#206). This is pervasive (fires on every graceful refusal) so it is NOT a gating invariant in v1; surface it as a descriptive count in a later report iteration rather than alarm on every refusal. "Success-honesty" (exit 0 + hollow) is already covered by the Completeness invariant.

## Phase 2 (follow-on, NOT this plan)
Runtime T1 consumability: a job that boots `larql serve <produced-vindex>` and drives `/v1/chat` (+ tool-call schema, SSE), asserting *boots→serves→well-formed response* (probes #245/#253/#266/#268). Separate subsystem (server-in-CI, containment) — its own spec/plan. web-llm/T3 remains parked on Q5/Q7.
