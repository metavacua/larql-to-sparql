#!/usr/bin/env python3
"""
check_overgate.py — detect `#[cfg(not(target_arch="wasm32"))]` gates that were
NOT necessary for the wasm32v1-none build.

Two modes:

  lint   (fast, heuristic): flag gated items whose first gated line does not
         reference any known-native symbol (std::io/fs/net/..., ndarray, memmap2,
         tokio, reqwest, …). These are *candidates* for over-gating.

  prove  (slow, definitive, per crate): for each gate site, remove the gate,
         run `cargo check --target wasm32v1-none --lib`, and if it STILL compiles
         the gate was unnecessary. Restores the file afterward. Reports each
         provably-unnecessary gate.

Usage:
  python3 tools/check_overgate.py lint  [crate ...]
  python3 tools/check_overgate.py prove <crate>
"""
from __future__ import annotations
import re
import subprocess
import sys
from pathlib import Path

WASM_DIR = Path(__file__).resolve().parents[1]
MANIFEST = WASM_DIR / "Cargo.toml"

GATE = '#[cfg(not(target_arch = "wasm32"))]'

# Symbols whose presence justifies a native-only gate.
NATIVE_SYMBOLS = (
    "std::io", "std::fs", "std::net", "std::thread", "std::process",
    "std::env", "std::path", "std::os", "std::sync::Mutex", "std::sync::RwLock",
    "std::sync::OnceLock", "std::sync::mpsc", "std::sync::Condvar",
    "std::time::Instant", "std::time::SystemTime", "std::ffi",
    "thread_local", "PathBuf", "Path",
    "ndarray", "memmap2", "Mmap", "ArcArray", "Array2", "safetensors",
    "tokenizers", "reqwest", "rayon", "libc", "tokio", "tonic", "axum",
    "prost", "wasmi", "wasmtime", "hf_hub", "hf-hub", "blas", "openblas",
    "pyo3", "numpy", "rustyline", "indicatif", "clap", "minijinja",
    "serde_json", "rand", "getrandom", "tower", "hyper", "rustls", "ring",
    "tracing", "expert_proto", "std::collections::HashMap", "std::error",
    "File", "OpenOptions", "Command", "TcpStream", "TcpListener",
    "Instant", "SystemTime", "Runtime", "block_on",
    "loading", "detect", "weights",  # gated sibling modules
)


def cargo_check(crate: str) -> bool:
    r = subprocess.run(
        ["cargo", "check", "--manifest-path", str(MANIFEST),
         "-p", crate, "--target", "wasm32v1-none", "--lib",
         "--message-format=short"],
        capture_output=True, text=True, cwd=WASM_DIR.parent.parent,
    )
    return r.returncode == 0


def kernel_crates() -> list[str]:
    return sorted(
        d.name for d in WASM_DIR.iterdir()
        if d.is_dir() and d.name.endswith('-wasm32v1-none')
    )


def gate_sites(rs: Path) -> list[int]:
    """Return line indices (0-based) of native-only gate attributes."""
    lines = rs.read_text().split('\n')
    return [i for i, l in enumerate(lines) if l.strip() == GATE]


def gated_context(lines: list[str], idx: int) -> str:
    """The gated item following the gate. Skips further outer attributes
    (`#[...]`), then captures the item — for a brace-opening item (fn/struct/
    enum-variant/impl/mod) it reads through the matching close brace so native
    types in the body are seen; otherwise it reads to the statement `;`."""
    out = []
    j = idx + 1
    # skip stacked attributes (e.g. #[allow(...)], #[error(...)] possibly multiline)
    while j < len(lines):
        s = lines[j].strip()
        out.append(lines[j])
        if s.startswith('#['):
            # consume a possibly multi-line attribute until its closing ']'
            depth = lines[j].count('[') - lines[j].count(']')
            while depth > 0 and j + 1 < len(lines):
                j += 1
                out.append(lines[j])
                depth += lines[j].count('[') - lines[j].count(']')
            j += 1
            continue
        break
    # now capture the item itself
    if j < len(lines):
        first = lines[j] if not out or out[-1] != lines[j] else ''
        if first:
            out.append(first)
        # read a brace block or up to ~8 lines / `;`
        depth = ' '.join(out).count('{') - ' '.join(out).count('}')
        k = j
        while k + 1 < len(lines) and (depth > 0 or
              not (lines[k].strip().endswith(';') or lines[k].strip().endswith('}'))) \
              and k - idx < 14:
            k += 1
            out.append(lines[k])
            depth += lines[k].count('{') - lines[k].count('}')
    return ' '.join(out)


def cmd_lint(crates: list[str]) -> None:
    suspicious = 0
    for crate in crates:
        src = WASM_DIR / crate / 'src'
        if not src.exists():
            continue
        for rs in sorted(src.rglob('*.rs')):
            lines = rs.read_text().split('\n')
            for i in gate_sites(rs):
                ctx = gated_context(lines, i)
                # Any gated `use std::...` is justified outright: std is wholly
                # unavailable on wasm32v1-none, so the gate is definitionally needed.
                if 'use std::' in ctx or 'use ::std::' in ctx:
                    continue
                if not any(sym in ctx for sym in NATIVE_SYMBOLS):
                    rel = rs.relative_to(WASM_DIR)
                    print(f"  ?? {rel}:{i+1}  {ctx.strip()[:90]}")
                    suspicious += 1
    if suspicious == 0:
        print("✅ lint: no obviously-unnecessary gates")
    else:
        print(f"\n{suspicious} gate(s) flagged — verify with `prove` (may be false positives)")


def cmd_prove(crate: str) -> None:
    src = WASM_DIR / crate / 'src'
    if not src.exists():
        print(f"no src for {crate}")
        return
    if not cargo_check(crate):
        print(f"❌ {crate} does not currently pass wasm32v1-none — fix it before proving.")
        return

    unnecessary = []
    for rs in sorted(src.rglob('*.rs')):
        original = rs.read_text()
        lines = original.split('\n')
        sites = [i for i, l in enumerate(lines) if l.strip() == GATE]
        if not sites:
            continue
        # Remove gates one at a time (re-read fresh each time).
        for site in sites:
            cur = original.split('\n')
            # find the matching line index in current text (stable: same as original)
            del cur[site]
            rs.write_text('\n'.join(cur))
            still_ok = cargo_check(crate)
            rs.write_text(original)  # restore
            if still_ok:
                ctx = gated_context(original.split('\n'), site)
                rel = rs.relative_to(WASM_DIR)
                unnecessary.append(f"  {rel}:{site+1}  {ctx.strip()[:90]}")

    if unnecessary:
        print(f"⚠️  {crate}: {len(unnecessary)} provably-unnecessary gate(s):")
        for u in unnecessary:
            print(u)
    else:
        print(f"✅ {crate}: every native gate is necessary for wasm32v1-none")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    mode = sys.argv[1]
    if mode == 'lint':
        cmd_lint(sys.argv[2:] or kernel_crates())
    elif mode == 'prove' and len(sys.argv) > 2:
        cmd_prove(sys.argv[2])
    else:
        print(__doc__)
        sys.exit(1)
