#!/usr/bin/env python3
"""
autogate_fn.py — gate exactly the functions/methods that use native-only
symbols, leaving portable functions ungated.

Honors the project rule: gate ONLY what cannot compile on wasm32v1-none.
A `fn` is gated iff its signature-or-body references a native symbol
(std::fs / std::io / std::path / std::net / std::thread / std::process /
std::env / std::sync::{Mutex,RwLock,OnceLock} / std::time::{Instant,SystemTime}
/ thread_local! / File / Command / TcpStream / …). Pure-compute fns are
left alone. Already-gated items are skipped.

It does NOT gate re-exports — run cargo check afterward and the compiler
points at any `pub use` of a now-gated item to fix by hand (cheap, few).

Usage:
  python3 tools/autogate_fn.py <crate> [crate ...]      # apply
  python3 tools/autogate_fn.py --dry <crate>            # preview only
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

WASM_DIR = Path(__file__).resolve().parents[1]
GATE = '#[cfg(not(target_arch = "wasm32"))]'

# Native-only symbols. A fn referencing any of these (outside comments) is gated.
NATIVE = re.compile(
    r'\bstd::(io|fs|net|thread|process|env|path|os|ffi)\b'
    r'|\bstd::sync::(Mutex|RwLock|OnceLock|Condvar|Barrier|mpsc)\b'
    r'|\bstd::time::(Instant|SystemTime)\b'
    r'|\bthread_local\s*!'
    r'|\b(PathBuf|Path)\b'
    r'|\b(File|OpenOptions|Command|TcpStream|TcpListener|UdpSocket)\b'
    r'|\bmemmap2?\b|\bMmap\b'
    r'|\b(ndarray|ArcArray\d?|Array\d)\b'
    r'|\b(safetensors|tokenizers|reqwest|rayon|libc|tokio|tonic|axum|prost'
    r'|wasmi|wasmtime|hf_hub|blas_src|openblas_src|pyo3|numpy|rustyline'
    r'|indicatif|minijinja)\b'
)

# A line that opens a function item (free fn or method).
FN_OPEN = re.compile(
    r'^(\s*)(?:pub(?:\s*\([^)]*\))?\s+)?(?:default\s+)?(?:const\s+)?'
    r'(?:async\s+)?(?:unsafe\s+)?(?:extern\s+"[^"]*"\s+)?fn\s+\w+'
)


def strip_strings_comments(s: str) -> str:
    """Crude: drop // line comments and "..." string contents so symbol
    matches don't trigger on docs or string literals."""
    s = re.sub(r'//.*', '', s)
    s = re.sub(r'"(?:\\.|[^"\\])*"', '""', s)
    return s


def find_item_span(lines: list[str], start: int) -> int:
    """Given a line index where a fn signature starts, return the index of the
    last line of the item (matching close brace, or the `;` for a trait-decl)."""
    # Walk to the first '{' or ';' that ends the signature.
    depth = 0
    i = start
    seen_brace = False
    while i < len(lines):
        code = strip_strings_comments(lines[i])
        for ch in code:
            if ch == '{':
                depth += 1
                seen_brace = True
            elif ch == '}':
                depth -= 1
        if seen_brace and depth == 0:
            return i
        if not seen_brace and ';' in code and depth == 0:
            return i  # e.g. trait method decl `fn foo();`
        i += 1
    return len(lines) - 1


def attr_block_start(lines: list[str], fn_line: int) -> int:
    """Walk upward over the fn's doc comments and attributes (including
    multi-line `#[cfg_attr(...)]`) to find where the gate must go — ABOVE the
    whole attribute/doc block, never inside a multi-line attribute."""
    i = fn_line - 1
    depth = 0  # net unclosed brackets while inside a multi-line attribute
    while i >= 0:
        line = lines[i]
        s = line.strip()
        opens = line.count('[') + line.count('(')
        closes = line.count(']') + line.count(')')
        if depth > 0:
            # inside a multi-line attribute — keep consuming until balanced
            depth += closes - opens
            i -= 1
            continue
        if s.startswith('///') or s.startswith('//!') or s.startswith('//') \
                or s.startswith('#['):
            i -= 1
            continue
        if (s.endswith(']') or s.endswith(')')) and closes > opens:
            # closing line of a multi-line attribute: enter it
            depth = closes - opens
            i -= 1
            continue
        break
    return i + 1


def already_gated(lines: list[str], insert_at: int) -> bool:
    # The item's attribute/doc block sits between insert_at and the fn line;
    # an existing gate may be anywhere in that block. Scan from insert_at down
    # over doc/attr lines, and also just above insert_at, for the gate.
    if insert_at - 1 >= 0 and 'cfg(not(target_arch = "wasm32"))' in lines[insert_at - 1]:
        return True
    j = insert_at
    while j < len(lines):
        s = lines[j].strip()
        if 'cfg(not(target_arch = "wasm32"))' in s:
            return True
        if s.startswith('///') or s.startswith('//!') or s.startswith('#[') \
                or s.startswith('//') or s == '' or s.endswith(']'):
            j += 1
            continue
        break  # reached the item itself
    return False


# A `use std::<native>` statement that must be gated (std absent on wasm32).
USE_NATIVE_STD = re.compile(
    r'^\s*(?:pub\s+)?use\s+std::(io|fs|net|thread|process|env|path|os|ffi)\b'
    r'|^\s*(?:pub\s+)?use\s+std::sync::(Mutex|RwLock|OnceLock|Condvar|Barrier|mpsc)\b'
    r'|^\s*(?:pub\s+)?use\s+std::time::(Instant|SystemTime)\b'
)

# A `use <native-crate>::...` statement for a crate absent on wasm32.
USE_NATIVE_DEP = re.compile(
    r'^\s*(?:pub\s+)?use\s+'
    r'(ndarray|memmap2|safetensors|tokenizers|reqwest|rayon|libc|tokio|tonic'
    r'|axum|axum_server|prost|wasmi|wasmtime|wasmtime_wasi|hf_hub|blas_src'
    r'|openblas_src|pyo3|numpy|rustyline|indicatif|minijinja|minijinja_contrib'
    r'|tower|tower_http|tracing|hyper|h2|rustls|ring|async_stream|tokio_stream'
    r'|prost_types|cblas_sys|utoipa|utoipa_swagger_ui)\b'
    r'|^\s*(?:pub\s+)?use\s+crate::expert_proto\b'
)


def process_file(path: Path, dry: bool) -> int:
    text = path.read_text()
    lines = text.split('\n')
    # Collect fn spans first (top-level and within impl blocks).
    inserts: list[int] = []
    i = 0
    n = len(lines)
    while i < n:
        # Gate standalone native `use` import lines (std::<native> or native crate).
        if USE_NATIVE_STD.match(lines[i]) or USE_NATIVE_DEP.match(lines[i]):
            if not (i > 0 and lines[i - 1].strip() == GATE):
                inserts.append(i)
            i += 1
            continue
        m = FN_OPEN.match(lines[i])
        if m:
            end = find_item_span(lines, i)
            body = strip_strings_comments('\n'.join(lines[i:end + 1]))
            if NATIVE.search(body):
                ins = attr_block_start(lines, i)
                if not already_gated(lines, ins):
                    inserts.append(ins)
            i = end + 1
            continue
        i += 1

    if not inserts:
        return 0

    # Insert from bottom up to keep indices valid.
    for ins in sorted(set(inserts), reverse=True):
        indent = re.match(r'(\s*)', lines[ins]).group(1)
        lines.insert(ins, f'{indent}{GATE}')

    if not dry:
        path.write_text('\n'.join(lines))
    return len(set(inserts))


def kernel_crates() -> list[str]:
    return sorted(d.name for d in WASM_DIR.iterdir()
                  if d.is_dir() and d.name.endswith('-wasm32v1-none'))


if __name__ == '__main__':
    args = sys.argv[1:]
    dry = '--dry' in args
    args = [a for a in args if a != '--dry']
    crates = args or kernel_crates()
    total = 0
    for crate in crates:
        src = WASM_DIR / crate / 'src'
        if not src.exists():
            continue
        cnt = 0
        for rs in sorted(src.rglob('*.rs')):
            cnt += process_file(rs, dry)
        if cnt:
            print(f'  {crate}: gated {cnt} fns{" (dry)" if dry else ""}')
        total += cnt
    print(f'\nTotal: {total} fns gated{" (dry run)" if dry else ""}')
