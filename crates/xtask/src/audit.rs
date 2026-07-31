//! Source-level module classifier.
//!
//! Reads src/lib.rs and partitions `pub mod` declarations into wasm32-accessible
//! (no immediately-preceding cfg-not-wasm32 gate) vs. native-only.

use anyhow::Result;
use std::path::{Path, PathBuf};

/// Classification of a crate's source modules.
#[derive(Debug, Default)]
pub struct Modules {
    /// pub mod declarations reachable from wasm32.
    pub accessible: Vec<String>,
    /// pub mod declarations behind #[cfg(not(target_arch = "wasm32"))].
    pub native_only: Vec<String>,
}

/// Classify modules for a single crate.
pub fn classify(_crate_name: &str, crate_root: &Path) -> Result<Modules> {
    let mut result = Modules::default();

    let lib_rs = crate_root.join("src/lib.rs");
    if !lib_rs.exists() {
        return Ok(result);
    }

    let src = std::fs::read_to_string(&lib_rs)?;
    let mut prev_was_cfg_gate = false;

    for line in src.lines() {
        let trimmed = line.trim();
        if trimmed == "#[cfg(not(target_arch = \"wasm32\"))]" {
            prev_was_cfg_gate = true;
            continue;
        }
        if let Some(mod_name) = trimmed
            .strip_prefix("pub mod ")
            .and_then(|s| s.strip_suffix(';'))
        {
            if prev_was_cfg_gate {
                result.native_only.push(mod_name.to_owned());
            } else {
                result.accessible.push(mod_name.to_owned());
            }
        }
        if !trimmed.is_empty() && !trimmed.starts_with("//") {
            prev_was_cfg_gate = false;
        }
    }

    Ok(result)
}

/// Return file paths that could contain module `name` (file or directory).
pub(crate) fn module_paths(src_dir: &Path, name: &str) -> Vec<PathBuf> {
    let mut paths = vec![];
    let file = src_dir.join(format!("{name}.rs"));
    if file.exists() {
        paths.push(file);
    }
    let dir = src_dir.join(name);
    if dir.is_dir() {
        collect_rs_files(&dir, &mut paths);
    }
    paths
}

pub(crate) fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_rs_files(&path, out);
            } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                out.push(path);
            }
        }
    }
}
