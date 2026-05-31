#!/usr/bin/env python3
"""
census.py — on-demand census of the wasm32v1-none gate boundary.

Produces two sets, just-in-time (nothing is persisted):

  SET A — GATED features: capabilities cut from the wasm32v1-none build
          (external crates that are native-only + the `std::` module surface).
  SET B — ACTIVE features: what the wasm32v1-none build actually uses
          (the few external crates it compiles + the `core::`/`alloc::` surface).

Comparing A and B is the point — it exposes the std→core/alloc moves already
made (B), the injective gaps with no portable codomain (A-only: io/fs/net/clock
/…), and the third-party surface that must live behind the bridge.

Two data sources, each authoritative for its half:
  * External crates: `cargo tree` per target (cargo resolves every cfg axis and
    the real feature set itself — no guessing).
  * stdlib surface: a source scan restricted to code actually compiled for
    wasm32v1-none. "Compiled" means surviving ALL cfg axes — target_arch,
    target_os, target_feature, unix/windows, and `feature=` resolved against the
    crate's default features — evaluated on both module gates (reachability) and
    every `use`'s enclosing-block cfgs. On `no_std` wasm32v1-none `std` does not
    exist, so any surviving `use std::` is a real anomaly and is reported.

Usage:
  python3 tools/census.py                 # aggregate over all kernel crates
  python3 tools/census.py <crate-base>... # specific crates (e.g. larql-core)
  python3 tools/census.py --paths         # also list the distinct symbol paths
  python3 tools/census.py --crates-only   # skip the stdlib scan (fast)
"""
from __future__ import annotations
import re
import subprocess
import sys
from collections import Counter, deque
from pathlib import Path

from wasm_gate_common import WASM_DIR, ROOT, MANIFEST, kernel_crates

# ── cfg evaluation for the wasm32v1-none target ──────────────────────────────
# Key/value facts true for `wasm32v1-none` (MVP: no atomics/simd128 features).
# Any (key, value) not listed is false — including every key=val for an unknown
# key — so this set alone decides target_* predicates.
_KV_TRUE = {
    ("target_arch", "wasm32"), ("target_family", "wasm"),
    ("target_pointer_width", "32"), ("target_endian", "little"),
    ("target_os", "none"), ("target_env", ""), ("target_vendor", "unknown"),
}


def _split_top(s: str) -> list[str]:
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur); cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


def _inner(pred: str, kw: str) -> str:
    return pred[len(kw) + 1: pred.rindex(")")]


def eval_cfg(pred: str, feats: set[str]) -> bool:
    """Evaluate a cfg predicate string for wasm32v1-none. Unknown keys/barewords
    are treated as false (conservative: the code is assumed NOT compiled)."""
    pred = pred.strip()
    if pred.startswith("all(") and pred.endswith(")"):
        return all(eval_cfg(p, feats) for p in _split_top(_inner(pred, "all")))
    if pred.startswith("any(") and pred.endswith(")"):
        return any(eval_cfg(p, feats) for p in _split_top(_inner(pred, "any")))
    if pred.startswith("not(") and pred.endswith(")"):
        return not eval_cfg(_inner(pred, "not"), feats)
    m = re.match(r'(\w+)\s*=\s*"([^"]*)"\s*$', pred)
    if m:
        k, v = m.group(1), m.group(2)
        if k == "feature":
            return v in feats
        if k == "target_feature":
            return False  # MVP: nothing enabled
        return (k, v) in _KV_TRUE  # unknown key → not present → false
    if pred == "debug_assertions":
        return True
    return False  # unix / windows / test / doc / any other bareword → false


def default_features(crate_dir: Path) -> set[str]:
    """Transitive closure of the crate's `default` feature (cfg(feature=…) only;
    `dep:`/`crate?/feat` activations are irrelevant to cfg evaluation)."""
    toml = crate_dir / "Cargo.toml"
    if not toml.exists():
        return set()
    text = toml.read_text()
    m = re.search(r'(?ms)^\[features\]\s*(.*?)(?=^\[|\Z)', text)
    if not m:
        return set()
    graph: dict[str, list[str]] = {}
    for line in m.group(1).splitlines():
        fm = re.match(r'\s*([\w-]+)\s*=\s*\[(.*)\]', line)
        if fm:
            subs = [s.strip().strip('"') for s in fm.group(2).split(",") if s.strip()]
            graph[fm.group(1)] = [s for s in subs if ":" not in s and "/" not in s]
    out, q = set(), deque(graph.get("default", []))
    while q:
        f = q.popleft()
        if f in out:
            continue
        out.add(f)
        q.extend(graph.get(f, []))
    return out


# ── source helpers ───────────────────────────────────────────────────────────
def _strip(line: str) -> str:
    line = re.sub(r'//.*', '', line)
    line = re.sub(r'"(?:\\.|[^"\\])*"', '""', line)
    return line


def _flatten_use(tree: str) -> list[str]:
    tree = re.sub(r'\s+as\s+\w+', '', tree.strip())
    i = tree.find("{")
    if i == -1:
        return [tree] if tree not in ("self", "*") else []
    prefix = tree[:i].rstrip(":").strip()
    depth, j = 0, i
    for k in range(i, len(tree)):
        if tree[k] == "{":
            depth += 1
        elif tree[k] == "}":
            depth -= 1
            if depth == 0:
                j = k; break
    out = []
    for member in _split_top(tree[i + 1:j]):
        for sub in _flatten_use(member):
            if sub in ("self", "*"):
                if prefix:
                    out.append(prefix)
                continue
            out.append(f"{prefix}::{sub}" if prefix else sub)
    return out


def _attr_cfgs(attr_lines: list[str]) -> list[str]:
    cfgs = []
    for a in attr_lines:
        m = re.search(r'#\[cfg\((.*)\)\]', a)
        if m:
            cfgs.append(m.group(1))
    return cfgs


def _module_dir(f: Path) -> Path:
    return f.parent if f.name in ("lib.rs", "main.rs", "mod.rs") else f.parent / f.stem


def reachable_files(src: Path, feats: set[str]) -> dict[Path, list[str]]:
    """Files compiled for wasm32v1-none, mapped to their (already-read) lines:
    BFS the `mod` tree from lib.rs, skipping any module whose cfg is false on
    this target. The lines are cached so the scan pass need not re-read them."""
    start = src / "lib.rs"
    if not start.exists():
        return {}
    seen: dict[Path, list[str]] = {}
    enqueued, q = {start}, deque([start])
    while q:
        f = q.popleft()
        d = _module_dir(f)
        lines = f.read_text().split("\n")
        seen[f] = lines
        i, n = 0, len(lines)
        while i < n:
            attrs = []
            while i < n and lines[i].lstrip().startswith("#["):
                attrs.append(lines[i]); i += 1
            if i >= n:
                break
            m = re.match(r'\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*;', lines[i])
            if m and all(eval_cfg(c, feats) for c in _attr_cfgs(attrs)):
                for cand in (d / f"{m.group(1)}.rs", d / m.group(1) / "mod.rs"):
                    if cand.exists() and cand not in enqueued:
                        enqueued.add(cand); q.append(cand)
            i += 1
    return seen


class Tally:
    """Accumulated stdlib-surface census results across scanned files."""
    def __init__(self):
        self.gated = Counter()      # 2-seg stdlib key -> gated use-site count
        self.active = Counter()     # 2-seg stdlib key -> active use-site count
        self.gated_paths = set()    # distinct full gated paths
        self.active_paths = set()   # distinct full active paths
        self.anomalies = []         # active `use std::` in reachable code (should be none)
        self.gated_sites = []       # (full_path, relpath, lineno) per gated use-site


def _scan_file(f: Path, lines: list[str], feats: set[str], local_core: bool, t: "Tally"):
    rel = str(f.relative_to(WASM_DIR))
    n, i, depth = len(lines), 0, 0
    stack: list[tuple[int, list[str]]] = []  # (depth outside block, cfgs) of enclosers
    pending: list[str] | None = None          # cfgs of a block whose `{` not seen yet
    attr_buf: list[str] = []
    while i < n:
        while stack and depth <= stack[-1][0]:
            stack.pop()
        raw = lines[i]
        s = raw.strip()
        if s.startswith("#["):                 # accumulate an attribute (maybe multi-line)
            buf = raw
            while buf.count("[") > buf.count("]") and i + 1 < n:
                i += 1; buf += lines[i]
            attr_buf.append(buf); i += 1
            continue
        if s.startswith("//"):                  # doc/comment — keep attr_buf alive
            i += 1
            continue
        if s == "":                              # blank breaks the attr/doc block
            attr_buf = []; i += 1
            continue
        cfgs = _attr_cfgs(attr_buf)
        if s.startswith("use ") or s.startswith("pub use "):
            buf = raw
            while ";" not in buf and i + 1 < n:
                i += 1; buf += " " + lines[i].strip()
            tree = re.sub(r'^\s*(pub\s+)?use\s+', '', buf).split(";")[0].strip()
            govern = cfgs + [c for _, cs in stack for c in cs]
            is_active = all(eval_cfg(c, feats) for c in govern)
            for p in _flatten_use(tree):
                sysroot = p.lstrip(":")
                root = sysroot.split("::")[0]
                if root == "core" and local_core and not p.startswith("::"):
                    continue                     # local module, not the sysroot crate
                if root not in ("std", "core", "alloc"):
                    continue
                key = "::".join(sysroot.split("::")[:2])
                if is_active:
                    t.active[key] += 1; t.active_paths.add(sysroot)
                    if root == "std":
                        t.anomalies.append(f"{rel}:{i+1}  {sysroot}")
                else:
                    t.gated[key] += 1; t.gated_paths.add(sysroot)
                    t.gated_sites.append((sysroot, rel, i + 1))
            attr_buf = []; i += 1
            continue
        # non-use item / code line: maintain the enclosing-cfg stack
        opens = _strip(raw).count("{") - _strip(raw).count("}")
        if cfgs:
            if opens > 0:
                stack.append((depth, cfgs))
            elif not s.endswith(";"):
                pending = cfgs                   # multi-line signature; attach at next `{`
        elif pending is not None:
            if opens > 0:
                stack.append((depth, pending)); pending = None
            elif s.endswith(";"):
                pending = None                   # statement item, never opened a block
        depth += opens
        attr_buf = []; i += 1


def scan_stdlib(crates: list[str]) -> "Tally":
    t = Tally()
    for base in crates:
        crate_dir = WASM_DIR / f"{base}-wasm32v1-none"
        src = crate_dir / "src"
        if not src.exists():
            continue
        feats = default_features(crate_dir)
        # `core` collides with a local module in crates that define `mod core`.
        local_core = (src / "core.rs").exists() or (src / "core").is_dir()
        for f, lines in reachable_files(src, feats).items():
            _scan_file(f, lines, feats, local_core, t)
    return t


# ── external crates via cargo tree ───────────────────────────────────────────
def crate_deps(crates: list[str], wasm: bool) -> set[str]:
    # One `cargo tree` for the whole crate set (the union is all we keep), not
    # one per crate — each invocation pays full cargo resolution startup.
    cmd = ["cargo", "tree", "--manifest-path", str(MANIFEST),
           "-e", "normal", "--prefix", "none"]
    for base in crates:
        cmd += ["-p", f"{base}-wasm32v1-none"]
    if wasm:
        cmd += ["--target", "wasm32v1-none"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    acc: set[str] = set()
    for line in r.stdout.splitlines():
        name = re.sub(r' v[0-9].*', '', line).strip()
        if not name or name.startswith("(") or name.endswith("-wasm32v1-none"):
            continue
        if name in ("larql-wasm-math", "larql-bridge"):
            continue
        acc.add(name)
    return acc


def kernel_crates_bases() -> list[str]:
    return [c[:-len("-wasm32v1-none")] for c in kernel_crates()]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    crates = args or kernel_crates_bases()
    show_paths = "--paths" in flags

    print(f"wasm32v1-none gate census over {len(crates)} crate(s): {', '.join(crates)}\n")

    wasm_deps = crate_deps(crates, wasm=True)
    native_deps = crate_deps(crates, wasm=False)
    gated_crates = sorted(native_deps - wasm_deps)
    print(f"== SET B: external crates COMPILED for wasm32v1-none ({len(wasm_deps)}) ==")
    print("  " + " ".join(sorted(wasm_deps)))
    print(f"\n== SET A: external crates NATIVE-ONLY (gated) ({len(gated_crates)}) ==")
    print("  " + " ".join(gated_crates))

    if "--crates-only" not in flags:
        t = scan_stdlib(crates)
        print(f"\n== SET A: gated stdlib roots ({sum(t.gated.values())} sites) ==")
        for k, c in t.gated.most_common():
            print(f"  {c:4d}  {k}")
        print(f"\n== SET B: active stdlib roots ({sum(t.active.values())} sites) ==")
        for k, c in t.active.most_common():
            print(f"  {c:4d}  {k}")
        if t.anomalies:
            print(f"\n!! INVARIANT VIOLATION: {len(t.anomalies)} active `use std::` on wasm "
                  f"(should be 0) — a real native leak or a cfg the evaluator missed:")
            for a in t.anomalies[:40]:
                print("  ", a)
        else:
            print("\n✅ invariant holds: no active `use std::` in wasm-reachable code")
        if show_paths:
            print("\n-- distinct gated stdlib paths --")
            for p in sorted(t.gated_paths):
                print("  A ", p)
            print("\n-- distinct active core::/alloc:: paths --")
            for p in sorted(t.active_paths):
                print("  B ", p)
        if "--sites" in flags:
            # Conversion worklist: every gated stdlib use grouped by full path.
            by_path: dict[str, list[str]] = {}
            for full, rel, ln in t.gated_sites:
                by_path.setdefault(full, []).append(f"{rel}:{ln}")
            print(f"\n-- gated stdlib use-sites ({len(t.gated_sites)} across "
                  f"{len(by_path)} symbols) --")
            for full in sorted(by_path):
                print(f"\n  {full}  ({len(by_path[full])})")
                for loc in sorted(by_path[full]):
                    print(f"      {loc}")


if __name__ == "__main__":
    main()
