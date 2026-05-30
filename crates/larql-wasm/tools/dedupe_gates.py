#!/usr/bin/env python3
"""
dedupe_gates.py — collapse consecutive duplicate native gates.

autogate/cascade passes can stack two identical
`#[cfg(not(target_arch = "wasm32"))]` lines on one item (when a doc comment or
attribute sits between an earlier gate and the item, the dup check misses it).
This removes any run of 2+ identical consecutive gate lines (ignoring blank
lines and intervening doc/attr lines is NOT done here — only *consecutive*
duplicates are collapsed, which is always safe).

Usage:
  python3 tools/dedupe_gates.py [crate ...]
"""
from __future__ import annotations
import sys

from wasm_gate_common import WASM_DIR, GATE, kernel_crates


def dedupe(path) -> int:
    lines = path.read_text().split('\n')
    out: list[str] = []
    removed = 0
    for line in lines:
        if line.strip() == GATE and out and out[-1].strip() == GATE:
            removed += 1
            continue
        out.append(line)
    if removed:
        path.write_text('\n'.join(out))
    return removed


if __name__ == '__main__':
    crates = sys.argv[1:] or kernel_crates()
    total = 0
    for crate in crates:
        src = WASM_DIR / crate / 'src'
        if not src.exists():
            continue
        c = 0
        for rs in sorted(src.rglob('*.rs')):
            c += dedupe(rs)
        if c:
            print(f'  {crate}: removed {c} duplicate gates')
        total += c
    print(f'\nTotal: {total} duplicate gates removed')
