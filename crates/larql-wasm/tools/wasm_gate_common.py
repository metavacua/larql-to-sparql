#!/usr/bin/env python3
"""
wasm_gate_common.py — shared primitives for the wasm32v1-none gating tools.

Every tool in this directory needs the same handful of facts: where the nested
wasm workspace lives, which crates are the `-wasm32v1-none` kernels, the exact
spelling of the native gate attribute, and how to run a `cargo check` against a
kernel crate. They used to each carry private copies; this module is the single
source of truth so the layout/suffix/gate-spelling conventions live in one place.

Importable because the tools run as `python3 tools/<name>.py`, which puts this
directory on `sys.path[0]`.
"""
from __future__ import annotations
import subprocess
from pathlib import Path

# tools/ is two levels under the nested wasm workspace, which is two levels
# under the repo root.
WASM_DIR = Path(__file__).resolve().parents[1]
ROOT = WASM_DIR.parent.parent
MANIFEST = WASM_DIR / "Cargo.toml"

# The native-only gate. Spelled byte-for-byte the same everywhere — line-equality
# checks (dedupe/ungate) and substring scans (autogate's `already_gated`) both
# depend on this exact form, including the spaces around `=`.
GATE = '#[cfg(not(target_arch = "wasm32"))]'
GATE_SUBSTR = 'cfg(not(target_arch = "wasm32"))'


def kernel_crates() -> list[str]:
    """All `*-wasm32v1-none` kernel crate directory names, sorted."""
    return sorted(
        d.name for d in WASM_DIR.iterdir()
        if d.is_dir() and d.name.endswith('-wasm32v1-none')
    )


def cargo_check(crate: str, wasm: bool = True) -> tuple[bool, str]:
    """`cargo check -p <crate> --lib` (against wasm32v1-none when `wasm`, else
    native). Returns (compiled_ok, combined_stdout_stderr). Callers wanting only
    the boolean take `[0]`."""
    cmd = ["cargo", "check", "--manifest-path", str(MANIFEST), "-p", crate,
           "--lib", "--message-format=short"]
    if wasm:
        cmd += ["--target", "wasm32v1-none"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    return r.returncode == 0, r.stdout + r.stderr


def cargo_check_target(crate: str, target: str) -> tuple[bool | None, str]:
    """`cargo check -p <crate> --lib --target <target>`.

    Returns (result, combined_stdout_stderr) where result is:
      True  — compiled successfully
      False — compiled but produced errors
      None  — target toolchain unavailable (skip; do not count as a failure)

    Callers should treat None as "⏸" (not yet provable) rather than "❌"
    (proven to fail) — the distinction matters for GPU targets that require
    special toolchains (rust-gpu for spirv, ptx for nvptx64).
    """
    cmd = ["cargo", "check", "--manifest-path", str(MANIFEST), "-p", crate,
           "--lib", "--target", target, "--message-format=short"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    out = r.stdout + r.stderr
    # Distinguish "target not installed" from a real compile failure.
    if (
        "is not installed" in out
        or "target may not be installed" in out
        or "the target `" in out and "is not supported" in out
    ):
        return None, out
    return r.returncode == 0, out
