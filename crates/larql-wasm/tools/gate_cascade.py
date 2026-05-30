#!/usr/bin/env python3
"""
gate_cascade.py — compiler-driven transitive gating.

autogate_fn gates items that *directly* name a native symbol. But a fn that
merely *calls* a gated fn (without naming any native symbol itself) is left
ungated and then fails with "unresolved import" / "cannot find function".

This tool closes the transitive cascade: it repeatedly runs
`cargo check --target wasm32v1-none --lib`, reads each "cannot find / unresolved"
error location, gates the enclosing item (fn / impl method / use / const /
static / type alias), and re-checks — until the only remaining errors are ones
it cannot resolve by gating (real portability work) or it reaches a fixpoint.

It NEVER gates an item that doesn't transitively depend on something native —
it only acts on actual compiler errors, so it cannot over-gate portable code
that compiles.

Usage:
  python3 tools/gate_cascade.py <crate> [--max N]
"""
from __future__ import annotations
import re
import subprocess
import sys
from pathlib import Path

WASM_DIR = Path(__file__).resolve().parents[1]
MANIFEST = WASM_DIR / "Cargo.toml"
GATE = '#[cfg(not(target_arch = "wasm32"))]'

ITEM_START = re.compile(
    r'^(\s*)(?:pub(?:\s*\([^)]*\))?\s+)?'
    r'(?:default\s+)?(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?(?:extern\s+"[^"]*"\s+)?'
    r'(fn|struct|enum|trait|impl|const|static|type|mod|use)\b'
)


def check(crate: str) -> tuple[bool, str]:
    r = subprocess.run(
        ["cargo", "check", "--manifest-path", str(MANIFEST), "-p", crate,
         "--target", "wasm32v1-none", "--lib", "--message-format=short"],
        capture_output=True, text=True, cwd=WASM_DIR.parent.parent,
    )
    return r.returncode == 0, r.stdout + r.stderr


# Error lines that indicate "this code path needs a native item that's gated".
# Error lines whose code path needs a native-only item that has been gated.
# Matches both `error[E####]:` and bare `error:` (macro-not-found has no code),
# across the phrasings the compiler uses for "this references something absent
# on wasm32v1-none": missing imports/items, missing trait items, no method /
# associated item, and unstable types (e.g. the built-in `f16`, available on
# native via the gated `half` crate).
ERR = re.compile(
    r'^([^:\s]+\.rs):(\d+):\d+:\s+error(?:\[E\d+\])?:\s+'
    r'('
    r'unresolved import'
    r'|cannot find (?:function|value|type|macro|module|trait|attribute|derive macro)'
    r'|no method named'
    r'|no function or associated item named'
    r'|not all trait items implemented'
    r'|the type `[^`]+` is unstable'
    r')'
)


def enclosing_item_line(lines: list[str], err_line0: int) -> int | None:
    """Walk upward from the error line to the nearest enclosing item start
    that is at the lowest indentation seen so far (so we gate the whole fn,
    not a nested block). Returns the line index to gate before (above the
    item's doc/attr block), or None."""
    # Find the item start at or above err_line0.
    i = err_line0
    target = None
    while i >= 0:
        m = ITEM_START.match(lines[i])
        if m:
            target = i
            break
        i -= 1
    if target is None:
        return None
    # Walk up over the item's doc comments / attributes.
    j = target - 1
    while j >= 0:
        s = lines[j].strip()
        if s.startswith('///') or s.startswith('//!') or s.startswith('#[') \
                or s.startswith('//') or s.endswith(']'):
            j -= 1
            continue
        break
    return j + 1


def already_gated(lines: list[str], insert_at: int) -> bool:
    return insert_at - 1 >= 0 and lines[insert_at - 1].strip() == GATE


def one_pass(crate: str) -> tuple[int, bool, str]:
    """Run one gating pass. Returns (items_gated, already_ok, check_output).
    The check output is returned so the caller can report/decide without a
    second `cargo check` (each check is a full wasm compile)."""
    ok, out = check(crate)
    if ok:
        return 0, True, out
    # Map file -> set of error line numbers (1-based).
    by_file: dict[str, set[int]] = {}
    for line in out.splitlines():
        m = ERR.match(line.strip())
        if not m:
            continue
        rel, lineno = m.group(1), int(m.group(2))
        # Resolve to absolute path under WASM_DIR.
        # short format gives path relative to the workspace dir.
        by_file.setdefault(rel, set()).add(lineno - 1)
    gated = 0
    for rel, errlines in by_file.items():
        path = (WASM_DIR.parent.parent / rel)
        if not path.exists():
            path = WASM_DIR / rel
        if not path.exists():
            continue
        lines = path.read_text().split('\n')
        insert_points = set()
        for el in errlines:
            ip = enclosing_item_line(lines, el)
            if ip is not None and not already_gated(lines, ip):
                insert_points.add(ip)
        for ip in sorted(insert_points, reverse=True):
            indent = re.match(r'(\s*)', lines[ip]).group(1)
            lines.insert(ip, f'{indent}{GATE}')
            gated += 1
        if insert_points:
            path.write_text('\n'.join(lines))
    return gated, False, out


def main(crate: str, max_iter: int) -> None:
    for it in range(1, max_iter + 1):
        # `out` here is the check run at the START of this pass (pre-gating).
        n, ok, out = one_pass(crate)
        if ok:
            print(f"  pass {it}: gated {n} items, 0 errors remain")
            print(f"✅ {crate} passes wasm32v1-none")
            return
        nerr = out.count('error[')
        print(f"  pass {it}: gated {n} items, {nerr} errors remain (pre-gating)")
        if n == 0:
            print(f"⚠️  fixpoint: {nerr} errors remain that gating cannot fix "
                  f"(need real portability work). Showing first few:")
            shown = 0
            for line in out.splitlines():
                if 'error' in line and '.rs' in line:
                    print(f"    {line.strip()[:110]}")
                    shown += 1
                    if shown >= 12:
                        break
            return
    print(f"reached max {max_iter} iterations")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    crate = sys.argv[1]
    max_iter = 20
    if '--max' in sys.argv:
        max_iter = int(sys.argv[sys.argv.index('--max') + 1])
    main(crate, max_iter)
