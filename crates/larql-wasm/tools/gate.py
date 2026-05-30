#!/usr/bin/env python3
"""
Minimum-gate tool for wasm32v1-none kernel crates.

Usage:
  python3 tools/gate.py check <crate>        -- show errors, categorised
  python3 tools/gate.py fix   <crate>        -- apply minimum fixes in-place
  python3 tools/gate.py verify <crate>       -- detect over-gating
  python3 tools/gate.py all                  -- fix + verify all kernel crates in dep order

Philosophy:
  REMAP first (std:: → core:: / alloc::), no #[cfg] needed.
  GATE only what is truly OS/IO-specific.
  VERIFY that no portable item was gated.
"""
from __future__ import annotations
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # larql-to-sparql/
WASM_DIR = Path(__file__).resolve().parents[1]  # larql-wasm/
MANIFEST = WASM_DIR / "Cargo.toml"

# ── Classification ────────────────────────────────────────────────────────────

# std modules whose items are available identically in core:: or alloc::
# Key = std path prefix, value = replacement prefix
REMAP: dict[str, str] = {
    "std::fmt":             "core::fmt",
    "std::cmp":             "core::cmp",
    "std::ops":             "core::ops",
    "std::str":             "core::str",
    "std::mem":             "core::mem",
    "std::ptr":             "core::ptr",
    "std::slice":           "core::slice",
    "std::marker":          "core::marker",
    "std::convert":         "core::convert",
    "std::option":          "core::option",
    "std::result":          "core::result",
    "std::iter":            "core::iter",
    "std::hash":            "core::hash",
    "std::cell":            "core::cell",
    "std::num":             "core::num",
    "std::hint":            "core::hint",
    "std::any":             "core::any",
    "std::panic":           "core::panic",
    "std::error::Error":    "core::error::Error",
    "std::error":           "core::error",
    # alloc equivalents
    "std::vec":             "alloc::vec",
    "std::string":          "alloc::string",
    "std::boxed":           "alloc::boxed",
    "std::rc":              "alloc::rc",
    "std::sync::Arc":       "alloc::sync::Arc",
    "std::borrow::Cow":     "alloc::borrow::Cow",
    "std::borrow::ToOwned": "alloc::borrow::ToOwned",
    "std::borrow":          "alloc::borrow",
    "std::collections::BTreeMap":  "alloc::collections::BTreeMap",
    "std::collections::BTreeSet":  "alloc::collections::BTreeSet",
    "std::collections::VecDeque":  "alloc::collections::VecDeque",
    "std::collections::BinaryHeap":"alloc::collections::BinaryHeap",
    "std::collections::btree_map": "alloc::collections::btree_map",
    "std::collections::btree_set": "alloc::collections::btree_set",
    # hashbrown replacements (HashMap/HashSet not in alloc)
    "std::collections::HashMap":   "hashbrown::HashMap",
    "std::collections::HashSet":   "hashbrown::HashSet",
    "std::collections::hash_map":  "hashbrown::hash_map",
    "std::collections::hash_set":  "hashbrown::hash_set",
    # format!/vec!/println! macros — alloc for format!/vec!, std only for println!
    # println! needs gating but format!/vec! are in alloc
}

# std paths that are genuinely OS/IO/hardware specific → must be gated
OS_PREFIXES: tuple[str, ...] = (
    "std::io",
    "std::fs",
    "std::net",
    "std::thread",
    "std::sync::Mutex",
    "std::sync::RwLock",
    "std::sync::OnceLock",
    "std::sync::Condvar",
    "std::sync::Barrier",
    "std::sync::mpsc",
    "std::process",
    "std::env",
    "std::time::Instant",
    "std::time::SystemTime",
    "std::ffi",
    "std::os",
    "std::path",         # PathBuf uses OS separators; gate by default
    "std::io::prelude",
)

# Portably available via core::sync::atomic (basic atomics work on wasm32)
PORTABLE_ATOMIC_TYPES = {"AtomicBool", "AtomicI8", "AtomicI16", "AtomicI32",
                          "AtomicU8", "AtomicU16", "AtomicU32", "AtomicUsize",
                          "AtomicIsize", "Ordering"}

KERNEL_CRATES = [
    "larql-boundary-wasm32v1-none",   # already passes
    "larql-models-wasm32v1-none",
    "larql-core-wasm32v1-none",
    "larql-compute-wasm32v1-none",
    "larql-kv-wasm32v1-none",
    "larql-vindex-wasm32v1-none",
    "larql-inference-wasm32v1-none",
    "larql-lql-wasm32v1-none",
    "larql-router-protocol-wasm32v1-none",
    "larql-router-wasm32v1-none",
    "larql-server-wasm32v1-none",
    "larql-cli-wasm32v1-none",
    "larql-python-wasm32v1-none",
    "model-compute-wasm32v1-none",
    "kv-cache-benchmark-wasm32v1-none",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def cargo_check(crate: str) -> tuple[bool, str]:
    """Run cargo check --lib --target wasm32v1-none. Return (passed, stderr)."""
    result = subprocess.run(
        ["cargo", "check",
         "--manifest-path", str(MANIFEST),
         "-p", crate,
         "--target", "wasm32v1-none",
         "--lib",
         "--message-format=short"],
        capture_output=True, text=True,
        cwd=ROOT
    )
    combined = result.stdout + result.stderr
    return result.returncode == 0, combined


def source_files(crate: str) -> list[Path]:
    crate_dir = WASM_DIR / crate
    return sorted((crate_dir / "src").rglob("*.rs"))


def classify_std_use(path: str) -> str:
    """Return 'remap:<target>', 'atomic', 'os', or 'unknown'."""
    # Check direct remaps
    for prefix, target in REMAP.items():
        if path == prefix or path.startswith(prefix + "::") or path.startswith(prefix + " "):
            return f"remap:{target}"
    # Check atomic portability
    for typ in PORTABLE_ATOMIC_TYPES:
        if f"atomic::{typ}" in path or f"Atomic" in path.split("::")[-1]:
            return "atomic"
    # Check OS prefixes
    for prefix in OS_PREFIXES:
        if path == prefix or path.startswith(prefix + "::"):
            return "os"
    return "unknown"


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_check(crate: str) -> None:
    """Show categorised errors."""
    passed, output = cargo_check(crate)
    if passed:
        print(f"✅ {crate} passes wasm32v1-none check")
        return
    # Categorise errors
    errors: dict[str, list[str]] = {"remap": [], "os": [], "atomic": [], "unknown": [], "other": []}
    for line in output.splitlines():
        m = re.search(r"cannot find.*`(std::[^`]+)`|use of unresolved.*`(std::[^`]+)`|unresolved.*`(std::[^`]+)`", line)
        if m:
            path = next(g for g in m.groups() if g)
            cat = classify_std_use(path)
            key = cat.split(":")[0] if ":" in cat else cat
            errors.setdefault(key, []).append(f"  {path}")
        elif "error" in line.lower():
            errors["other"].append(f"  {line[:120]}")

    print(f"❌ {crate}")
    for cat, items in errors.items():
        if items:
            uniq = sorted(set(items))[:10]
            print(f"  [{cat}] {len(set(items))} unique:")
            for i in uniq:
                print(i)


def cmd_fix(crate: str) -> int:
    """Apply minimum fixes. Returns count of changes made."""
    changes = 0
    crate_dir = WASM_DIR / crate

    for rs in source_files(crate):
        text = rs.read_text()
        modified = text

        # 1. Remap portable std:: → core:: / alloc:: / hashbrown::
        for std_prefix, target in sorted(REMAP.items(), key=lambda x: -len(x[0])):
            # use std::X::Y → use target::Y
            modified = modified.replace(f"use {std_prefix}::", f"use {target}::")
            # use std::X; → use target;
            modified = modified.replace(f"use {std_prefix};", f"use {target};")
            # inline std::X::Y( → target::Y(
            modified = modified.replace(f"{std_prefix}::", f"{target}::")

        # 2. Remap core::sync::atomic for portable atomics
        modified = modified.replace(
            "std::sync::atomic::", "core::sync::atomic::"
        )

        # 3. Fix HashMap import to use hashbrown (add extern if needed)
        if "hashbrown::" in modified and "use hashbrown" not in modified:
            # Already replaced but no import — the inline replacement is fine
            pass

        if modified != text:
            rs.write_text(modified)
            changes += 1

    if changes:
        print(f"  Remapped {changes} files in {crate}")
    return changes


def cmd_verify(crate: str) -> None:
    """Detect over-gating: find #[cfg(not(wasm32))] on items that are portable."""
    crate_dir = WASM_DIR / crate
    over_gated = []
    CFG_NOT_WASM = re.compile(
        r'#\[cfg\(not\(target_arch\s*=\s*"wasm32"\)\)\]'
    )
    for rs in source_files(crate):
        text = rs.read_text()
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if CFG_NOT_WASM.search(line):
                # Look at the next non-empty line for what's being gated
                next_lines = " ".join(lines[i+1:i+4]).strip()
                # Check if it contains OS-specific items
                has_os = any(os in next_lines for os in ["io::", "fs::", "net::",
                    "thread::", "Mutex", "RwLock", "process::", "env::", "Instant",
                    "SystemTime", "tokio", "axum", "tonic", "reqwest", "memmap",
                    "safetensors", "rayon", "libc", "pyo3", "clap", "rustyline",
                    "ndarray", "blas", "wasmi", "wasmtime", "tokenizers"])
                if not has_os:
                    rel = rs.relative_to(WASM_DIR)
                    ctx = next_lines[:80]
                    over_gated.append(f"  {rel}:{i+1}: possibly over-gated → {ctx}")

    if over_gated:
        print(f"⚠️  {crate}: {len(over_gated)} potentially over-gated items:")
        for item in over_gated[:15]:
            print(item)
    else:
        print(f"✅ {crate}: no over-gating detected")


def cmd_all() -> None:
    """Fix + verify all kernel crates in dependency order."""
    for crate in KERNEL_CRATES:
        passed, _ = cargo_check(crate)
        if passed:
            print(f"✅ {crate} — already passing, running verify only")
            cmd_verify(crate)
            continue
        print(f"\n── {crate} ──")
        changes = cmd_fix(crate)
        passed2, output = cargo_check(crate)
        if passed2:
            print(f"  ✅ now passes after {changes} file remaps")
            cmd_verify(crate)
        else:
            remaining = len([l for l in output.splitlines() if "error[" in l])
            print(f"  ❌ {remaining} errors remain after remapping — source-level gates needed")
            # Show which files still have issues
            files = set()
            for line in output.splitlines():
                m = re.search(r"--> ([^\s:]+\.rs)", line)
                if m and "registry" not in m.group(1):
                    files.add(m.group(1).split("/src/")[-1] if "/src/" in m.group(1) else m.group(1))
            if files:
                print(f"  Files needing source gates: {sorted(files)[:8]}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    crate = sys.argv[2] if len(sys.argv) > 2 else None

    if cmd == "check" and crate:
        cmd_check(crate)
    elif cmd == "fix" and crate:
        cmd_fix(crate)
        cmd_verify(crate)
    elif cmd == "verify" and crate:
        cmd_verify(crate)
    elif cmd == "all":
        cmd_all()
    else:
        print(__doc__)
        sys.exit(1)
