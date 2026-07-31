//! Shared workspace utilities: cargo metadata + wasm32 build helpers.

use anyhow::{Context, Result};
use cargo_metadata::{Metadata, MetadataCommand, Package};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

pub fn meta() -> Result<Metadata> {
    Ok(MetadataCommand::new().exec()?)
}

/// Build a wasm32 cdylib; returns the .wasm artifact path, or None on build failure.
pub fn build_cdylib(crate_name: &str, crate_root: &Path) -> Result<Option<PathBuf>> {
    let output = Command::new("cargo")
        .args([
            "rustc",
            "--target",
            "wasm32-unknown-unknown",
            "--message-format",
            "json",
            "-p",
            crate_name,
            "--lib",
            "--crate-type",
            "cdylib",
        ])
        .current_dir(crate_root)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .context("cargo rustc --crate-type cdylib")?;

    if !output.status.success() {
        eprintln!(
            "  (cdylib build failed for {crate_name}, exit {})",
            output.status
        );
        return Ok(None);
    }

    for line in String::from_utf8_lossy(&output.stdout).lines() {
        if let Ok(msg) = serde_json::from_str::<serde_json::Value>(line) {
            if msg["reason"] == "compiler-artifact" {
                if let Some(filenames) = msg["filenames"].as_array() {
                    for f in filenames {
                        if let Some(s) = f.as_str() {
                            if s.ends_with(".wasm") {
                                return Ok(Some(PathBuf::from(s)));
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(None)
}

/// True if the crate declares [package.metadata.wasm].
pub fn has_wasm_meta(pkg: &Package) -> bool {
    pkg.metadata
        .as_object()
        .and_then(|m| m.get("wasm"))
        .is_some()
}

/// True if the crate declares a cdylib target in its Cargo.toml.
pub fn is_cdylib_crate(pkg: &Package) -> bool {
    pkg.targets.iter().any(|t| t.is_cdylib())
}

/// True if [package.metadata.wasm] split-candidate = true.
pub fn is_split_candidate(pkg: &Package) -> bool {
    pkg.metadata
        .as_object()
        .and_then(|m| m.get("wasm"))
        .and_then(|w| w.get("split-candidate"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}
