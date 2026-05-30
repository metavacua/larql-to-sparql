#!/usr/bin/env python3
"""
normalize_prelude.py — enforce ONE canonical import strategy across kernel crates.

Problem this solves:
  Earlier remap rewrote `use std::X` → `use alloc::X` / `use hashbrown::X`
  UNCONDITIONALLY. That breaks native builds (no `extern crate alloc` on native,
  hashbrown is wasm32-only) and creates duplicate-import (E0252) errors.

Invariant enforced:
  1. Crate roots declare `extern crate alloc;` UNCONDITIONALLY (works on native
     via sysroot and on wasm32). `#![cfg_attr(target_arch="wasm32", no_std)]` stays.
  2. Each source file that references alloc/collection types gets exactly ONE
     canonical prelude block:

        #[allow(unused_imports)]
        use alloc::{boxed::Box, string::{String, ToString}, vec::Vec, vec, format,
                    borrow::{Cow, ToOwned}, rc::Rc, sync::Arc,
                    collections::{BTreeMap, BTreeSet, VecDeque, BinaryHeap}};
        #[cfg(target_arch = "wasm32")]
        #[allow(unused_imports)]
        use hashbrown::{HashMap, HashSet};
        #[cfg(not(target_arch = "wasm32"))]
        #[allow(unused_imports)]
        use std::collections::{HashMap, HashSet};

  3. ALL prior scattered/partial `use alloc::...`, `use hashbrown::...`, and
     gated `use std::collections::{HashMap,HashSet}` lines are stripped first.

  alloc:: works on BOTH targets, so prelude-type imports need NO cfg gate.
  Only HashMap/HashSet need a cfg split (hashbrown vs std::collections).

Usage:
  python3 tools/normalize_prelude.py [crate ...]    # default: all kernel crates
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

WASM_DIR = Path(__file__).resolve().parents[1]

CANONICAL = '''#[allow(unused_imports)]
use alloc::{boxed::Box, string::{String, ToString}, vec::Vec, vec, format, borrow::{Cow, ToOwned}, rc::Rc, sync::Arc, collections::{BTreeMap, BTreeSet, VecDeque, BinaryHeap}};
#[cfg(target_arch = "wasm32")]
#[allow(unused_imports)]
use hashbrown::{HashMap, HashSet};
#[cfg(not(target_arch = "wasm32"))]
#[allow(unused_imports)]
use std::collections::{HashMap, HashSet};
#[cfg(target_arch = "wasm32")]
#[allow(unused_imports)]
use larql_wasm_math::FloatExt as _;'''

PRELUDE_MARKER = "use alloc::{boxed::Box, string::{String, ToString}"

# Names whose presence means the file needs the prelude.
TRIGGER = re.compile(
    r'\b(String|Vec|Box|Rc|Arc|Cow|ToOwned|ToString|'
    r'BTreeMap|BTreeSet|VecDeque|BinaryHeap|HashMap|HashSet|format!|vec!)\b'
)


def strip_old_imports(text: str) -> str:
    """Remove all prior alloc/hashbrown/gated-collections import statements,
    including any immediately-preceding cfg/allow attributes."""
    lines = text.split('\n')
    out: list[str] = []
    i = 0
    n = len(lines)

    # Collection types the canonical prelude already provides (so any
    # `use std/alloc::collections::{...}` importing only these is redundant).
    PRELUDE_COLLECTIONS = {
        'HashMap', 'HashSet', 'BTreeMap', 'BTreeSet', 'VecDeque', 'BinaryHeap',
    }

    def redundant_collections_use(s: str) -> bool:
        m = re.match(r'(?:pub\s+)?use\s+(?:std|alloc)::collections::\{([^}]*)\};', s)
        if not m:
            return False
        items = {x.strip() for x in m.group(1).split(',') if x.strip()}
        # redundant iff every imported item is already provided by the prelude
        return bool(items) and items <= PRELUDE_COLLECTIONS

    def is_target_use(line: str) -> bool:
        s = line.strip()
        return (
            s.startswith('use alloc::') or
            s.startswith('use hashbrown::') or
            s.startswith('pub use alloc::') or
            s.startswith('pub use hashbrown::') or
            s.startswith('use larql_wasm_math::FloatExt') or
            # gated std::collections HashMap/HashSet imports we previously injected
            (s.startswith('use std::collections::{HashMap')) or
            (s.startswith('use std::collections::HashMap')) or
            (s.startswith('use std::collections::HashSet')) or
            # redundant `use {std,alloc}::collections::{...}` of prelude types
            redundant_collections_use(s)
        )

    # Attributes that, when stacked directly above a target use, should also drop.
    attr_re = re.compile(
        r'^\s*#\[(cfg\((not\()?target_arch\s*=\s*"wasm32"\)?\)|allow\(unused_imports\))\]\s*$'
    )

    while i < n:
        line = lines[i]

        # Look ahead: collect a run of attribute lines, then see if a target use follows.
        if attr_re.match(line):
            j = i
            attrs = []
            while j < n and attr_re.match(lines[j]):
                attrs.append(lines[j])
                j += 1
            if j < n and is_target_use(lines[j]):
                # Drop the attrs and the (possibly multi-line) use statement.
                k = j
                # consume until line containing ';'
                while k < n and ';' not in lines[k]:
                    k += 1
                i = k + 1
                continue
            else:
                # attrs belong to something else — keep them
                out.extend(attrs)
                i = j
                continue

        if is_target_use(line):
            # consume multi-line use until ';'
            k = i
            while k < n and ';' not in lines[k]:
                k += 1
            i = k + 1
            continue

        out.append(line)
        i += 1

    result = '\n'.join(out)
    # collapse 3+ blank lines
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result


def find_insert_point(lines: list[str]) -> int:
    """Insert after the module-level header — inner doc comments (`//!`), inner
    attributes (`#![`), `use`, and `extern crate` lines — but BEFORE any
    item-level doc comment (`///`) or outer attribute (`#[`), which belong to
    the following item and must not be separated from it."""
    last_header = 0
    i = 0
    n = len(lines)
    while i < n:
        s = lines[i].strip()
        if s == '':
            i += 1
            continue  # blanks don't extend the header; insert lands before them
        if s.startswith('//!') or s.startswith('#!['):
            last_header = i + 1
            i += 1
            continue
        if (s.startswith('use ') or s.startswith('pub use ') or
                s.startswith('extern crate')):
            # consume a possibly multi-line use/extern statement (until `;`)
            while i < n and ';' not in lines[i]:
                i += 1
            last_header = i + 1
            i += 1
            continue
        # `///` (outer doc), `#[` (outer attr), or any item → stop.
        break
    return last_header


def strip_inline_qualifiers(text: str) -> str:
    """Drop module qualifiers on HashMap/HashSet at inline (non-`use`) positions
    so they resolve to the cfg-split prelude imports (hashbrown on wasm32,
    std::collections on native).

    - `hashbrown::HashMap` → `HashMap` (remap left these inline; break native)
    - inline `std::collections::HashMap` → `HashMap` (remap only fixed `use`
      lines; inline type positions like `&std::collections::HashSet<T>` remain
      and break wasm32). `use std::collections::...` lines are preserved — they
      are the native half of the prelude split.
    """
    # hashbrown:: is only ever inline here (use-lines stripped earlier).
    text = re.sub(r'\bhashbrown::(HashMap|HashSet)\b', r'\1', text)

    out = []
    for line in text.split('\n'):
        s = line.lstrip()
        # `use {std,alloc}::sync::{Arc?, Rc?, ...native...}`: the prelude already
        # provides Arc/Rc, so drop them from the list to avoid an E0252 clash
        # (keeping any native sync types like Mutex/RwLock/OnceLock, which a
        # gate will cover). Drop the whole line if Arc/Rc were the only items.
        m = re.match(r'(\s*)((?:pub\s+)?use\s+(?:std|alloc)::sync)::\{([^}]*)\};\s*$', line)
        if m:
            indent, head, items = m.group(1), m.group(2), m.group(3)
            kept = [x.strip() for x in items.split(',')
                    if x.strip() and x.strip() not in ('Arc', 'Rc')]
            if not kept:
                continue  # only Arc/Rc — prelude covers them; drop line
            out.append(f'{indent}{head}::{{{", ".join(kept)}}};')
            continue
        # `use {std,alloc}::sync::Arc;` / `::Rc;` alone → drop (prelude covers).
        if re.match(r'\s*(?:pub\s+)?use\s+(?:std|alloc)::sync::(Arc|Rc);\s*$', line):
            continue
        # Preserve other `use ...` lines.
        if s.startswith('use ') or s.startswith('pub use '):
            out.append(line)
            continue
        out.append(re.sub(r'\bstd::collections::(HashMap|HashSet)\b', r'\1', line))
    return '\n'.join(out)


def fix_inner_doc_order(text: str) -> str:
    """Repair E0753: the early no_std insertion put `#[macro_use] extern crate
    alloc;` (an item) above the module's `//!` inner-doc block. Inner doc
    comments must precede all items, so move the extern-crate item below the
    leading `#![…]` / `//!` header region."""
    lines = text.split('\n')
    # Locate a `#[macro_use]?` + `extern crate alloc;` run near the top.
    ec_start = ec_end = None
    for i, l in enumerate(lines[:15]):
        if l.strip() == 'extern crate alloc;':
            ec_end = i
            ec_start = i
            if i > 0 and lines[i - 1].strip() == '#[macro_use]':
                ec_start = i - 1
            break
    if ec_start is None:
        return text
    # Is there a `//!` inner-doc line AFTER the extern crate? If not, fine.
    if not any(l.lstrip().startswith('//!') for l in lines[ec_end + 1: ec_end + 40]):
        return text
    block = lines[ec_start:ec_end + 1]
    rest = lines[:ec_start] + lines[ec_end + 1:]
    # Find where the leading header (#![…] inner attrs + //! docs + blanks) ends.
    insert_at = 0
    for i, l in enumerate(rest):
        s = l.strip()
        if s.startswith('#![') or s.startswith('//!') or s == '' or s.startswith('//'):
            insert_at = i + 1
            continue
        break
    repaired = rest[:insert_at] + block + rest[insert_at:]
    return '\n'.join(repaired)


def normalize_file(path: Path) -> bool:
    text = path.read_text()
    stripped = fix_inner_doc_order(text)
    stripped = strip_old_imports(stripped)
    stripped = strip_inline_qualifiers(stripped)

    needs = bool(TRIGGER.search(stripped))
    has = PRELUDE_MARKER in stripped

    if needs and not has:
        lines = stripped.split('\n')
        at = find_insert_point(lines)
        lines.insert(at, CANONICAL)
        stripped = '\n'.join(lines)

    if stripped != text:
        path.write_text(stripped)
        return True
    return False


def fix_extern_crate_alloc(root_file: Path) -> bool:
    """Make `extern crate alloc;` unconditional (drop the wasm32 cfg gate)."""
    if not root_file.exists():
        return False
    text = root_file.read_text()
    # Patterns: optional #[macro_use], #[cfg(target_arch="wasm32")], extern crate alloc;
    patt = re.compile(
        r'#\[cfg\(target_arch\s*=\s*"wasm32"\)\]\s*\n'
        r'(#\[macro_use\]\s*\n)?'
        r'extern crate alloc;'
    )
    def repl(m: re.Match) -> str:
        macro = m.group(1) or ''
        return f'{macro}extern crate alloc;'
    new = patt.sub(repl, text)
    if new != text:
        root_file.write_text(new)
        return True
    return False


def kernel_crates() -> list[str]:
    return sorted(
        d.name for d in WASM_DIR.iterdir()
        if d.is_dir() and d.name.endswith('-wasm32v1-none')
    )


if __name__ == '__main__':
    crates = sys.argv[1:] or kernel_crates()
    total_files = 0
    total_roots = 0
    for crate in crates:
        crate_dir = WASM_DIR / crate
        src = crate_dir / 'src'
        if not src.exists():
            continue
        # 1. unconditional extern crate alloc in lib.rs / main.rs
        for root_name in ('lib.rs', 'main.rs'):
            if fix_extern_crate_alloc(src / root_name):
                total_roots += 1
        # 2. normalize prelude in every file
        cnt = 0
        for rs in sorted(src.rglob('*.rs')):
            if normalize_file(rs):
                cnt += 1
        if cnt:
            print(f'  {crate}: normalized {cnt} files')
        total_files += cnt
    print(f'\nTotal: {total_files} files, {total_roots} crate roots updated')
