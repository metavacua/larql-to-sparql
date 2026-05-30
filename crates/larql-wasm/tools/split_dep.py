#!/usr/bin/env python3
"""
split_dep.py — move a workspace dependency into a target-cfg split so it gets
std features on native and no_std on wasm32.

Some deps (thiserror) need their `std` feature on native (e.g. thiserror's
`std` feature special-cases PathBuf → `.display()` for `#[error("{0}")]`),
but must be `default-features = false` on wasm32v1-none.

This rewrites, in each kernel crate Cargo.toml:
    [dependencies]
    thiserror = { workspace = true }        # REMOVED
  →
    [target.'cfg(not(target_arch = "wasm32"))'.dependencies]
    thiserror = "2"                          # std (default features)
    [target.'cfg(target_arch = "wasm32")'.dependencies]
    thiserror = { version = "2", default-features = false }

Usage:
  python3 tools/split_dep.py <dep> <version> [crate ...]
  e.g. python3 tools/split_dep.py thiserror 2
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

WASM_DIR = Path(__file__).resolve().parents[1]

NOT_WASM = "[target.'cfg(not(target_arch = \"wasm32\"))'.dependencies]"
WASM = "[target.'cfg(target_arch = \"wasm32\")'.dependencies]"


def kernel_crates() -> list[str]:
    return sorted(
        d.name for d in WASM_DIR.iterdir()
        if d.is_dir() and d.name.endswith('-wasm32v1-none')
    )


def remove_from_deps(text: str, dep: str) -> tuple[str, bool]:
    """Remove `dep = ...` / `dep.workspace = true` lines from the plain
    [dependencies] section only (not target sections)."""
    lines = text.split('\n')
    out = []
    section = None
    removed = False
    for line in lines:
        s = line.strip()
        if s.startswith('[') and s.endswith(']'):
            section = s
            out.append(line)
            continue
        if section == '[dependencies]':
            # match `dep = ...` or `dep.workspace = true`
            if re.match(rf'^{re.escape(dep)}\s*(=|\.)', s):
                removed = True
                continue
        out.append(line)
    return '\n'.join(out), removed


def ensure_target_dep(text: str, section_header: str, dep_line: str) -> str:
    """Ensure `dep_line` exists under `section_header`, creating the section
    if needed."""
    dep_name = dep_line.split('=')[0].strip()
    if section_header in text:
        # Check if dep already present in that section
        # Find section body
        idx = text.index(section_header) + len(section_header)
        rest = text[idx:]
        nxt = re.search(r'\n\[', rest)
        body = rest[: nxt.start()] if nxt else rest
        if re.search(rf'^\s*{re.escape(dep_name)}\s*=', body, re.MULTILINE):
            return text  # already there
        return text.replace(section_header, f'{section_header}\n{dep_line}', 1)
    else:
        return text.rstrip() + f'\n\n{section_header}\n{dep_line}\n'


def process(crate: str, dep: str, version: str) -> bool:
    manifest = WASM_DIR / crate / 'Cargo.toml'
    if not manifest.exists():
        return False
    text = manifest.read_text()
    if dep not in text:
        return False

    new, removed = remove_from_deps(text, dep)
    if not removed:
        return False  # dep wasn't in plain [dependencies]; leave as-is

    new = ensure_target_dep(new, NOT_WASM, f'{dep} = "{version}"')
    new = ensure_target_dep(
        new, WASM,
        f'{dep} = {{ version = "{version}", default-features = false }}'
    )
    new = re.sub(r'\n{3,}', '\n\n', new)
    if new != text:
        manifest.write_text(new)
        return True
    return False


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    dep = sys.argv[1]
    version = sys.argv[2]
    crates = sys.argv[3:] or kernel_crates()
    n = 0
    for crate in crates:
        if process(crate, dep, version):
            print(f'  split {dep} in {crate}')
            n += 1
    print(f'\nTotal: {n} crates updated')
