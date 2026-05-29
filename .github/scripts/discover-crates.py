#!/usr/bin/env python3
"""
Discovers all Cargo workspace crates (root workspace + nested workspaces) and
emits GitHub Actions strategy-matrix JSON on stdout.

Usage:
  python3 .github/scripts/discover-crates.py native    # 16 crates × 3 platforms
  python3 .github/scripts/discover-crates.py wasm      # 16 crates × 4 runtimes + pyodide
  python3 .github/scripts/discover-crates.py coverage  # 16 crates × ubuntu only

Requirements: Python 3.11+ (tomllib is stdlib).
"""
from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


# ── TOML helpers ─────────────────────────────────────────────────────────────


def _load(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _iter_dep_sections(data: dict[str, Any]) -> list[dict]:
    """Yield every dependency-map dict in the TOML (regular + target-specific)."""
    sections: list[dict] = []
    for key in ("dependencies", "dev-dependencies"):
        if key in data:
            sections.append(data[key])
    for cfg_block in data.get("target", {}).values():
        if isinstance(cfg_block, dict):
            for key in ("dependencies",):
                if key in cfg_block:
                    sections.append(cfg_block[key])
    return sections


def _local_path_deps(data: dict[str, Any], cargo_dir: Path) -> list[Path]:
    """Return resolved Cargo.toml paths for non-optional local path-dependencies."""
    results: list[Path] = []
    for section in _iter_dep_sections(data):
        for dep_val in section.values():
            if not isinstance(dep_val, dict):
                continue
            if "path" not in dep_val:
                continue
            if dep_val.get("optional", False):
                continue
            p = (cargo_dir / dep_val["path"] / "Cargo.toml").resolve()
            if p.exists():
                results.append(p)
    return results


def _check_recursive(
    cargo_toml: Path,
    predicate,
    visited: set[str] | None = None,
) -> bool:
    """Return True if predicate(data) holds for cargo_toml or any transitive
    non-optional local path dependency (cycle-safe)."""
    if visited is None:
        visited = set()
    key = str(cargo_toml.resolve())
    if key in visited:
        return False
    visited.add(key)
    try:
        data = _load(cargo_toml)
    except Exception:
        return False
    if predicate(data):
        return True
    for dep in _local_path_deps(data, cargo_toml.parent):
        if _check_recursive(dep, predicate, visited):
            return True
    return False


# ── Predicates ───────────────────────────────────────────────────────────────


def _needs_openblas(data: dict[str, Any]) -> bool:
    for section in _iter_dep_sections(data):
        if "openblas-src" in section:
            return True
    return False


def _needs_protoc(data: dict[str, Any]) -> bool:
    return "tonic-build" in data.get("build-dependencies", {})


def _has_metal_feature(data: dict[str, Any]) -> bool:
    return "metal" in data.get("features", {})


# ── Crate metadata ───────────────────────────────────────────────────────────


def _nested_any(ws_root: Path, predicate) -> bool:
    """True if any member crate of a nested workspace satisfies predicate."""
    try:
        ws_data = _load(ws_root / "Cargo.toml")
    except Exception:
        return False
    for member_glob in ws_data.get("workspace", {}).get("members", []):
        for member_dir in sorted(ws_root.glob(member_glob)):
            cargo_toml = member_dir / "Cargo.toml"
            if not cargo_toml.exists():
                continue
            try:
                if predicate(_load(cargo_toml)):
                    return True
            except Exception:
                pass
    return False


def build_crate_entry(cargo_toml: Path, *, nested: bool) -> dict[str, Any]:
    """Build the full metadata dict for one crate (or nested workspace)."""
    try:
        data = _load(cargo_toml)
    except Exception as exc:
        raise SystemExit(f"Cannot parse {cargo_toml}: {exc}") from exc

    name = data.get("package", {}).get("name", cargo_toml.parent.name)
    crate_path = str(cargo_toml.parent.relative_to(ROOT))

    if nested:
        has_metal = _nested_any(cargo_toml.parent, _has_metal_feature)
        # Nested workspace roots have no [dependencies], so direct checks are
        # sufficient — expert crates have no BLAS or protobuf deps.
        needs_openblas_val = False
        needs_protoc_val = False
        cargo_package = f"--manifest-path {crate_path}/Cargo.toml --workspace"
        fmt_flags = f"--manifest-path {crate_path}/Cargo.toml --all"
    else:
        has_metal = _has_metal_feature(data)
        needs_openblas_val = _check_recursive(cargo_toml, _needs_openblas)
        needs_protoc_val = _check_recursive(cargo_toml, _needs_protoc)
        cargo_package = f"-p {name}"
        fmt_flags = f"-p {name}"

    return {
        "name": name,
        "crate_path": crate_path,
        "nested": nested,
        "has_metal_feature": has_metal,
        "needs_openblas": needs_openblas_val,
        "needs_protoc_windows": needs_protoc_val,
        "cargo_package": cargo_package,
        "fmt_flags": fmt_flags,
    }


# ── Discovery ────────────────────────────────────────────────────────────────


def discover() -> list[dict[str, Any]]:
    """Return one entry per crate/nested-workspace, in workspace declaration order."""
    crates: list[dict[str, Any]] = []
    seen: set[str] = set()

    # Root workspace members
    root_data = _load(ROOT / "Cargo.toml")
    for member_glob in root_data.get("workspace", {}).get("members", []):
        for member_dir in sorted(ROOT.glob(member_glob)):
            cargo_toml = member_dir / "Cargo.toml"
            if not cargo_toml.exists():
                continue
            rel = str(member_dir.relative_to(ROOT))
            if rel in seen:
                continue
            # If this member is itself a workspace root, skip here; the nested-
            # workspace scan below will pick it up as a unit.
            try:
                m = _load(cargo_toml)
                if "workspace" in m and "members" in m.get("workspace", {}):
                    continue
            except Exception:
                pass
            seen.add(rel)
            crates.append(build_crate_entry(cargo_toml, nested=False))

    # Nested workspace roots (e.g. crates/larql-experts/)
    for cargo_toml in sorted(ROOT.rglob("Cargo.toml")):
        parts = cargo_toml.relative_to(ROOT).parts
        if cargo_toml == ROOT / "Cargo.toml":
            continue
        if any(p in ("target", ".git") for p in parts):
            continue
        try:
            data = _load(cargo_toml)
        except Exception:
            continue
        if "workspace" not in data or "members" not in data.get("workspace", {}):
            continue
        rel = str(cargo_toml.parent.relative_to(ROOT))
        if rel in seen:
            continue
        seen.add(rel)
        crates.append(build_crate_entry(cargo_toml, nested=True))

    return crates


# ── Matrix builders ──────────────────────────────────────────────────────────

NATIVE_PLATFORMS = [
    {
        "platform": "ubuntu",
        "runner": "ubuntu-24.04",
        "target": "x86_64-unknown-linux-gnu",
    },
    {
        "platform": "windows",
        "runner": "windows-latest",
        "target": "x86_64-pc-windows-msvc",
    },
    {
        "platform": "macos",
        "runner": "macos-15",
        "target": "aarch64-apple-darwin",
    },
]

WASM_RUNTIMES = ["wasm32v1-none", "wasmi", "node", "firefox"]


def native_matrix(crates: list[dict]) -> dict:
    # For native, only include crates NOT from crates/larql-wasm/ (only original crates + larql-experts).
    native_crates = [
        c for c in crates
        if not (c["crate_path"] == "crates/larql-wasm" or c["crate_path"].startswith("crates/larql-wasm/"))
    ]
    return {
        "include": [
            {
                **platform,
                "crate": c["name"],
                "crate_path": c["crate_path"],
                "nested": c["nested"],
                "has_metal_feature": c["has_metal_feature"],
                "needs_openblas": c["needs_openblas"],
                "needs_protoc_windows": c["needs_protoc_windows"],
                "cargo_package": c["cargo_package"],
                "fmt_flags": c["fmt_flags"],
            }
            for c in native_crates
            for platform in NATIVE_PLATFORMS
        ]
    }


def wasm_matrix(crates: list[dict]) -> dict:
    # For wasm, enumerate individual members of crates/larql-wasm/ with -wasm32v1-none or -interface suffix.
    # Exclude bridge infrastructure (larql-bridge, larql-bridge-native, larql-bridge-browser).
    wasm_crates: list[dict[str, Any]] = []

    # If larql-wasm nested workspace is present, enumerate its members individually
    wasm_workspace_root = ROOT / "crates" / "larql-wasm"
    if wasm_workspace_root.exists():
        try:
            ws_data = _load(wasm_workspace_root / "Cargo.toml")
            for member_glob in ws_data.get("workspace", {}).get("members", []):
                for member_dir in sorted(wasm_workspace_root.glob(member_glob)):
                    cargo_toml = member_dir / "Cargo.toml"
                    if not cargo_toml.exists():
                        continue
                    try:
                        member_data = _load(cargo_toml)
                        name = member_data.get("package", {}).get("name", member_dir.name)
                        # Only include -wasm32v1-none and -interface variants, skip bridge infrastructure
                        if name.endswith("-wasm32v1-none") or name.endswith("-interface"):
                            entry = build_crate_entry(cargo_toml, nested=False)
                            # Crates in crates/larql-wasm/ live in a nested workspace.
                            # cargo -p NAME fails from the repo root; must pin the manifest.
                            wasm_ws_rel = str(wasm_workspace_root.relative_to(ROOT))
                            entry["cargo_package"] = (
                                f"--manifest-path {wasm_ws_rel}/Cargo.toml -p {name}"
                            )
                            wasm_crates.append(entry)
                    except Exception:
                        pass
        except Exception:
            pass

    # -wasm32v1-none crates: all four runtimes (kernel portability + test execution)
    # -interface crates: only node + firefox (native std; wasm32v1-none/wasmi don't apply)
    KERNEL_RUNTIMES = ["wasm32v1-none", "wasmi", "node", "firefox"]
    INTERFACE_RUNTIMES = ["node", "firefox"]

    includes = []
    for c in wasm_crates:
        runtimes = KERNEL_RUNTIMES if c["name"].endswith("-wasm32v1-none") else INTERFACE_RUNTIMES
        for runtime in runtimes:
            includes.append({
                "crate": c["name"],
                "crate_path": c["crate_path"],
                "nested": c["nested"],
                "cargo_package": c["cargo_package"],
                "runtime": runtime,
            })
    # Extra pyodide/emscripten entry for larql-python-interface (pyo3 ABI)
    for c in wasm_crates:
        if c["name"] == "larql-python-interface":
            includes.append(
                {
                    "crate": c["name"],
                    "crate_path": c["crate_path"],
                    "nested": c["nested"],
                    "cargo_package": c["cargo_package"],
                    "runtime": "pyodide",
                    "wasm_target": "wasm32-unknown-emscripten",
                }
            )
    return {"include": includes}


def coverage_matrix(crates: list[dict]) -> dict:
    # For coverage, only include crates NOT from crates/larql-wasm/ (only original crates + larql-experts).
    coverage_crates = [
        c for c in crates
        if not (c["crate_path"] == "crates/larql-wasm" or c["crate_path"].startswith("crates/larql-wasm/"))
    ]
    return {
        "include": [
            {
                "crate": c["name"],
                "crate_path": c["crate_path"],
                "nested": c["nested"],
                "cargo_package": c["cargo_package"],
            }
            for c in coverage_crates
        ]
    }


# ── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "native"
    crates = discover()
    builders = {
        "native": native_matrix,
        "wasm": wasm_matrix,
        "coverage": coverage_matrix,
    }
    if mode not in builders:
        print(f"Unknown mode '{mode}'. Expected: native | wasm | coverage", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(builders[mode](crates)))


if __name__ == "__main__":
    main()
