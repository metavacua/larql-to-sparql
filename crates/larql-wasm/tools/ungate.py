#!/usr/bin/env python3
"""
ungate.py — remove gates that the over-gate `prove` proved unnecessary.

`check_overgate.py prove` reports every `#[cfg(not(target_arch="wasm32"))]`
whose removal still leaves the crate compiling on wasm32 --lib. Those gates are
excess (the goal: gate ONLY what's needed). This tool removes them and then
re-verifies BOTH targets (`--lib`) still compile; if a removal breaks either
target it is restored, so the net result never regresses.

Process per crate:
  1. parse `prove` output for (file, 1-based line) of each unnecessary gate
  2. remove those exact gate lines (bottom-up per file)
  3. cargo check --lib on wasm32v1-none AND native
  4. if either fails, `git checkout` the crate (revert all removals) and report

Usage:
  python3 tools/ungate.py <crate-base> [crate-base ...]
"""
from __future__ import annotations
import re
import subprocess
import sys
from pathlib import Path

WASM_DIR = Path(__file__).resolve().parents[1]
ROOT = WASM_DIR.parent.parent
MANIFEST = WASM_DIR / "Cargo.toml"
GATE = '#[cfg(not(target_arch = "wasm32"))]'


def check(crate: str, target_wasm: bool) -> bool:
    cmd = ["cargo", "check", "--manifest-path", str(MANIFEST), "-p", crate, "--lib",
           "--message-format=short"]
    if target_wasm:
        cmd += ["--target", "wasm32v1-none"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    return r.returncode == 0


def prove_flagged(crate: str) -> list[tuple[str, int]]:
    """Return (relpath, lineno) pairs prove reports as unnecessary."""
    r = subprocess.run(
        ["python3", str(WASM_DIR / "tools" / "check_overgate.py"), "prove", crate],
        capture_output=True, text=True, cwd=ROOT,
    )
    out = []
    for line in r.stdout.splitlines():
        m = re.match(r'\s+(\S+\.rs):(\d+)\b', line)
        if m:
            out.append((m.group(1), int(m.group(2))))
    return out


def process(crate: str) -> None:
    flagged = prove_flagged(crate)
    if not flagged:
        print(f"✅ {crate}: prove found no unnecessary gates")
        return
    # Group by file; the reported line is the ITEM line — the gate is the
    # `#[cfg(not(...))]` immediately above it (possibly past doc/attr lines).
    by_file: dict[str, list[int]] = {}
    for rel, ln in flagged:
        by_file.setdefault(rel, []).append(ln)

    removed = 0
    for rel, lns in by_file.items():
        path = ROOT / rel
        if not path.exists():
            path = WASM_DIR / rel
        if not path.exists():
            continue
        lines = path.read_text().split('\n')
        gate_idxs = set()
        for ln in lns:
            # item is at ln-1 (0-based); walk up over doc/attr to find the gate
            i = ln - 2
            while i >= 0:
                s = lines[i].strip()
                if s == GATE:
                    gate_idxs.add(i)
                    break
                if s.startswith('///') or s.startswith('//') or s.startswith('#[') \
                        or s.endswith(']') or s == '':
                    i -= 1
                    continue
                break
        for gi in sorted(gate_idxs, reverse=True):
            del lines[gi]
            removed += 1
        if gate_idxs:
            path.write_text('\n'.join(lines))

    ok_w = check(crate, True)
    ok_n = check(crate, False)
    if ok_w and ok_n:
        print(f"✅ {crate}: removed {removed} unnecessary gate(s); both targets clean")
    else:
        # revert — a removal regressed something (e.g. a bin-only or test-only
        # need that --lib prove couldn't see)
        subprocess.run(["git", "checkout", "--", str((WASM_DIR / crate))], cwd=ROOT)
        print(f"↩️  {crate}: removals regressed (wasm={ok_w} native={ok_n}); reverted. "
              f"Gates are bin/test-only — keep them.")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for base in sys.argv[1:]:
        process(f"{base}-wasm32v1-none")
