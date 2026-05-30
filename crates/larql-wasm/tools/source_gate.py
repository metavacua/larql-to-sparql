#!/usr/bin/env python3
"""
Source-level gating for wasm32v1-none kernel crates.

Strategy:
  Level 1 (coarse): Gate entire `mod X;` declarations in lib.rs for pure I/O modules.
  Level 2 (fine):   Gate `use X::Y;` statements for OS deps in mixed modules.
                    Gate `#[cfg(not(wasm32))]` on individual functions when return/param types reference OS deps.
  Remap:            Replace `std::X` → core::X / alloc::X in inline positions.

Usage:
  python3 tools/source_gate.py <crate_dir>
"""
from __future__ import annotations
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WASM_DIR = Path(__file__).resolve().parents[1]
MANIFEST = WASM_DIR / "Cargo.toml"

# Module names that are purely I/O → gate at lib.rs mod declaration level
IO_MODULES = {
    "loading", "io", "detect", "db", "store", "persistence",
    "http_provider", "checkpoint", "csv", "json", "format", "packed",
    "msgpack", "grpc", "routes", "bootstrap", "session", "shard_loader",
    "embed_store", "cache", "announce", "announce_service",
    "repl",  # REPL requires stdin/stdout
    "wasi_shim",
    "loader", "loader_jit",  # wasmi/wasmtime loaders
}

# Dep names that require OS/IO and must be gated at use statement level
GATED_DEPS = {
    "ndarray", "safetensors", "memmap2", "reqwest", "rayon", "libc",
    "tokio", "tonic", "axum", "axum_server", "prost", "async_stream",
    "futures", "wasmi", "wasmtime", "wasmtime_wasi",
    "clap", "indicatif", "minijinja", "minijinja_contrib",
    "half", "zip", "rustyline", "anyhow", "hf_hub",
    "blas_src", "openblas_src", "tower", "tower_http",
    "tracing", "tracing_subscriber", "async_stream",
    "rand", "rand_distr", "rand_chacha", "rand_core",
    "bytes", "hyper", "rustls", "h2",
    "sha2", "base64", "subtle",
    "utoipa", "utoipa_swagger_ui",
    "portable_atomic",
    "tokenizers",
    "serde_json",   # memchr dep requires std
    "pyo3", "numpy",
    "expert_proto",  # protobuf generated, needs tonic
    # std I/O modules that couldn't be remapped inline
    "std", "io", "fs", "net",
}

NOT_WASM_CFG = '#[cfg(not(target_arch = "wasm32"))]'


def is_gated(line: str) -> bool:
    """True if this line is already preceded by a wasm cfg gate."""
    return 'cfg(not(target_arch = "wasm32"))' in line


def gate_use_statement(text: str) -> str:
    """Add #[cfg(not(wasm32))] before use statements importing gated deps."""
    lines = text.split('\n')
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()

        # Skip already-gated lines
        if i > 0 and 'cfg(not(target_arch = "wasm32"))' in lines[i-1]:
            out.append(line)
            i += 1
            continue

        # Check if this is a use statement importing a gated dep
        use_match = re.match(r'(\s*)(pub\s+)?use\s+(\w+)', stripped)
        if use_match:
            dep = use_match.group(3)
            # Also check for use std::io::... patterns
            if (dep in GATED_DEPS or
                    (dep == 'std' and any(
                        f'std::{m}' in line for m in ['io', 'fs', 'net', 'thread',
                        'sync::Mutex', 'sync::RwLock', 'sync::OnceLock',
                        'process', 'env', 'time::Instant', 'time::SystemTime']))):
                indent = use_match.group(1)
                out.append(f'{indent}{NOT_WASM_CFG}')
                out.append(line)
                i += 1
                continue

        out.append(line)
        i += 1
    return '\n'.join(out)


def gate_io_modules_in_lib(lib_path: Path) -> bool:
    """Gate I/O module declarations in lib.rs / mod.rs."""
    if not lib_path.exists():
        return False
    text = lib_path.read_text()
    lines = text.split('\n')
    changed = False
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Already gated?
        if i > 0 and 'cfg(not(target_arch = "wasm32"))' in lines[i-1]:
            out.append(line)
            i += 1
            continue

        # Check for mod declaration
        mod_match = re.match(r'(\s*)(pub\s+)?mod\s+(\w+)\s*;', stripped)
        if mod_match:
            mod_name = mod_match.group(3)
            if mod_name in IO_MODULES:
                indent = re.match(r'(\s*)', line).group(1)
                out.append(f'{indent}{NOT_WASM_CFG}')
                changed = True

        out.append(line)
        i += 1

    if changed:
        lib_path.write_text('\n'.join(out))
    return changed


def process_crate(crate: str) -> int:
    """Apply source-level gates. Returns total changes."""
    crate_dir = WASM_DIR / crate
    if not crate_dir.exists():
        print(f"  ❌ {crate}: directory not found")
        return 0

    total = 0

    # 1. Gate I/O modules in lib.rs and mod.rs files
    for lib_file in (crate_dir / "src").rglob("*.rs"):
        if lib_file.name in ("lib.rs", "mod.rs"):
            if gate_io_modules_in_lib(lib_file):
                total += 1
                print(f"  Gated IO mods in {lib_file.relative_to(WASM_DIR)}")

    # 2. Gate use statements in all source files
    for rs in sorted((crate_dir / "src").rglob("*.rs")):
        text = rs.read_text()
        modified = gate_use_statement(text)
        if modified != text:
            rs.write_text(modified)
            total += 1

    return total


def check(crate: str) -> tuple[bool, int]:
    """Run cargo check. Return (passed, error_count)."""
    r = subprocess.run(
        ["cargo", "check",
         "--manifest-path", str(MANIFEST),
         "-p", crate,
         "--target", "wasm32v1-none",
         "--lib",
         "--message-format=short"],
        capture_output=True, text=True, cwd=ROOT
    )
    combined = r.stdout + r.stderr
    n = combined.count("error[")
    return r.returncode == 0, n


if __name__ == "__main__":
    crates = sys.argv[1:] or []
    if not crates:
        print("Usage: source_gate.py <crate1> [crate2 ...]")
        sys.exit(1)

    for crate in crates:
        print(f"\n── {crate} ──")
        _, before = check(crate)
        print(f"  Before: {before} errors")
        changes = process_crate(crate)
        print(f"  Applied {changes} changes")
        passed, after = check(crate)
        status = "✅" if passed else "❌"
        print(f"  {status} After: {after} errors")
