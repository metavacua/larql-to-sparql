//! On-disk Apollo store format.
//!
//! Mirrors the layout of `apollo-demo/apollo11_store/`:
//!
//! ```text
//! apollo11_store/
//! ├── manifest.json              # version, num_windows, crystal_layer, arch_config
//! ├── boundaries/
//! │   ├── window_000.npy         # shape (hidden,) f32 — single residual
//! │   ├── window_001.npy
//! │   └── ...
//! ├── boundary_residual.npy      # shape (1, 1, hidden) — most recent / active boundary
//! ├── window_token_lists.npz    # keyed by "0", "1", ... → u32 token arrays
//! └── entries.npz                # structured array of VecInjectEntry
//! ```
//!
//! Loading uses a handwritten `.npy` parser (see `npy.rs`) + the `zip` crate
//! for the `.npz` containers. No `ndarray-npy` dependency because its
//! current release (0.10) pins ndarray 0.17 and our workspace is on 0.16.

use std::io::Read;
use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::entry::VecInjectEntry;
use super::npy;

#[derive(Debug, Error)]
pub enum StoreLoadError {
    #[error("i/o error reading {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("json parse error in manifest: {0}")]
    Json(#[from] serde_json::Error),
    #[error("npy parse error in {path}: {source}")]
    Npy {
        path: String,
        #[source]
        source: npy::NpyError,
    },
    #[error("zip parse error in {path}: {source}")]
    Zip {
        path: String,
        #[source]
        source: zip::result::ZipError,
    },
    #[error("store missing required file: {0}")]
    MissingFile(String),
    #[error("manifest mismatch: {0}")]
    ManifestMismatch(String),
    #[error("structured-dtype parse error in {path}: {reason}")]
    StructuredDtype { path: String, reason: String },
}

/// Contents of `manifest.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreManifest {
    pub version: u32,
    pub num_entries: usize,
    pub num_windows: usize,
    pub num_tokens: usize,
    pub entries_per_window: usize,
    pub crystal_layer: usize,
    pub window_size: usize,
    pub arch_config: ArchConfig,
    pub has_residuals: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchConfig {
    pub retrieval_layer: usize,
    pub query_head: usize,
    pub injection_layer: usize,
    pub inject_coefficient: f32,
}

impl Default for ArchConfig {
    fn default() -> Self {
        // Apollo 11 defaults on Gemma 3 4B.
        Self {
            retrieval_layer: 29,
            query_head: 4,
            injection_layer: 30,
            inject_coefficient: 10.0,
        }
    }
}

/// In-memory representation of a loaded Apollo store.
#[derive(Debug)]
pub struct ApolloStore {
    pub manifest: StoreManifest,
    /// One residual vector per window at `crystal_layer`. `boundaries[i]`
    /// is a flat `(hidden,)` Vec for window i.
    pub boundaries: Vec<Vec<f32>>,
    /// `(1, 1, hidden)` — most recent / active boundary residual.
    /// Flattened to Vec<f32>.
    pub boundary_residual: Option<Vec<f32>>,
    /// Per-window token ID lists. `window_tokens[i]` has `window_size`
    /// entries (the last window may be shorter).
    pub window_tokens: Vec<Vec<u32>>,
    /// All vec_inject entries (flattened across windows).
    pub entries: Vec<VecInjectEntry>,
}

impl ApolloStore {
    /// Load an Apollo store from a directory.
    pub fn load(path: &Path) -> Result<Self, StoreLoadError> {
        let manifest = load_manifest(path)?;
        let boundaries = load_boundaries(path, manifest.num_windows)?;
        let boundary_residual = load_boundary_residual(path).ok();
        let window_tokens = load_window_tokens(path)?;
        let entries = load_entries(path)?;

        if boundaries.len() != manifest.num_windows {
            return Err(StoreLoadError::ManifestMismatch(format!(
                "manifest.num_windows={} but loaded {} boundaries",
                manifest.num_windows,
                boundaries.len(),
            )));
        }
        if entries.len() != manifest.num_entries {
            return Err(StoreLoadError::ManifestMismatch(format!(
                "manifest.num_entries={} but loaded {} entries",
                manifest.num_entries,
                entries.len(),
            )));
        }

        Ok(Self {
            manifest,
            boundaries,
            boundary_residual,
            window_tokens,
            entries,
        })
    }

    pub fn total_bytes(&self) -> usize {
        let boundary_bytes: usize = self.boundaries.iter().map(|b| b.len() * 4).sum();
        let boundary_residual_bytes = self
            .boundary_residual
            .as_ref()
            .map(|b| b.len() * 4)
            .unwrap_or(0);
        let token_bytes: usize = self.window_tokens.iter().map(|w| w.len() * 4).sum();
        let entry_bytes = self.entries.len() * std::mem::size_of::<VecInjectEntry>();
        boundary_bytes + boundary_residual_bytes + token_bytes + entry_bytes
    }

    pub fn hidden_size(&self) -> usize {
        self.boundaries.first().map(|b| b.len()).unwrap_or(0)
    }
}

// ── internals ────────────────────────────────────────────────────────────

fn read_file(path: &Path) -> Result<Vec<u8>, StoreLoadError> {
    std::fs::read(path).map_err(|source| StoreLoadError::Io {
        path: path.display().to_string(),
        source,
    })
}

fn load_manifest(path: &Path) -> Result<StoreManifest, StoreLoadError> {
    let bytes = read_file(&path.join("manifest.json"))?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn load_boundaries(path: &Path, num_windows: usize) -> Result<Vec<Vec<f32>>, StoreLoadError> {
    let dir = path.join("boundaries");
    let mut out = Vec::with_capacity(num_windows);
    for i in 0..num_windows {
        let p = dir.join(format!("window_{:03}.npy", i));
        let bytes = read_file(&p)?;
        let arr = npy::read_f32_1d(&bytes).map_err(|source| StoreLoadError::Npy {
            path: p.display().to_string(),
            source,
        })?;
        out.push(arr);
    }
    Ok(out)
}

fn load_boundary_residual(path: &Path) -> Result<Vec<f32>, StoreLoadError> {
    let p = path.join("boundary_residual.npy");
    let bytes = read_file(&p)?;
    let (flat, _shape) = npy::read_f32_flat(&bytes).map_err(|source| StoreLoadError::Npy {
        path: p.display().to_string(),
        source,
    })?;
    Ok(flat)
}

fn load_window_tokens(path: &Path) -> Result<Vec<Vec<u32>>, StoreLoadError> {
    let p = path.join("window_token_lists.npz");
    let file = std::fs::File::open(&p).map_err(|source| StoreLoadError::Io {
        path: p.display().to_string(),
        source,
    })?;
    let mut archive = zip::ZipArchive::new(file).map_err(|source| StoreLoadError::Zip {
        path: p.display().to_string(),
        source,
    })?;

    // Collect and numerically sort the members so returned Vec is indexable
    // by window_id. Member names are like "0.npy", "1.npy", ...
    let mut numbered: Vec<(usize, String)> = Vec::with_capacity(archive.len());
    for i in 0..archive.len() {
        let name = archive
            .by_index(i)
            .map_err(|source| StoreLoadError::Zip {
                path: p.display().to_string(),
                source,
            })?
            .name()
            .to_string();
        let trimmed = name.trim_end_matches(".npy");
        if let Ok(id) = trimmed.parse::<usize>() {
            numbered.push((id, name));
        }
    }
    numbered.sort_by_key(|(i, _)| *i);

    let mut out = Vec::with_capacity(numbered.len());
    for (_id, name) in numbered {
        let mut entry = archive
            .by_name(&name)
            .map_err(|source| StoreLoadError::Zip {
                path: format!("{}::{}", p.display(), name),
                source,
            })?;
        let mut buf = Vec::with_capacity(entry.size() as usize);
        entry.read_to_end(&mut buf).map_err(|source| StoreLoadError::Io {
            path: format!("{}::{}", p.display(), name),
            source,
        })?;
        let arr = npy::read_u32_1d(&buf).map_err(|source| StoreLoadError::Npy {
            path: format!("{}::{}", p.display(), name),
            source,
        })?;
        out.push(arr);
    }
    Ok(out)
}

fn load_entries(path: &Path) -> Result<Vec<VecInjectEntry>, StoreLoadError> {
    let p = path.join("entries.npz");
    let file = std::fs::File::open(&p).map_err(|source| StoreLoadError::Io {
        path: p.display().to_string(),
        source,
    })?;
    let mut archive = zip::ZipArchive::new(file).map_err(|source| StoreLoadError::Zip {
        path: p.display().to_string(),
        source,
    })?;

    // Find the first member whose name starts with "entries" (typically
    // "entries.npy" inside the zip).
    let member_name = {
        let mut found: Option<String> = None;
        for i in 0..archive.len() {
            let n = archive
                .by_index(i)
                .map_err(|source| StoreLoadError::Zip {
                    path: p.display().to_string(),
                    source,
                })?
                .name()
                .to_string();
            if n.starts_with("entries") {
                found = Some(n);
                break;
            }
        }
        found.ok_or_else(|| StoreLoadError::MissingFile("entries.npz::entries".into()))?
    };

    let mut entry = archive
        .by_name(&member_name)
        .map_err(|source| StoreLoadError::Zip {
            path: format!("{}::{}", p.display(), member_name),
            source,
        })?;
    let mut bytes = Vec::with_capacity(entry.size() as usize);
    entry.read_to_end(&mut bytes).map_err(|source| StoreLoadError::Io {
        path: member_name.clone(),
        source,
    })?;

    parse_structured_entries_npy(&bytes).map_err(|reason| StoreLoadError::StructuredDtype {
        path: format!("{}::{}", p.display(), member_name),
        reason,
    })
}

/// Parse a .npy file containing a structured-dtype array of `VecInjectEntry`.
///
/// Expected dtype (from the Python side):
///   (token_id: u32, coefficient: f32, window_id: u16,
///    position_in_window: u16, fact_id: u16)
/// Row size: 14 bytes, no padding (numpy packs structured dtypes tightly
/// when fields are already aligned).
fn parse_structured_entries_npy(bytes: &[u8]) -> Result<Vec<VecInjectEntry>, String> {
    let (header, data_off) = npy::parse_header(bytes).map_err(|e| e.to_string())?;

    for field in [
        "token_id",
        "coefficient",
        "window_id",
        "position_in_window",
        "fact_id",
    ] {
        if !header.descr.contains(field) {
            return Err(format!(
                "missing field '{field}' in descr: {}",
                header.descr
            ));
        }
    }
    if header.shape.len() != 1 {
        return Err(format!("expected 1D structured array, got shape {:?}", header.shape));
    }

    const ROW_SIZE: usize = 4 + 4 + 2 + 2 + 2;
    let n = header.shape[0];
    let data = &bytes[data_off..];
    let expected = n * ROW_SIZE;
    if data.len() != expected {
        return Err(format!(
            "data size {} != expected {} ({} rows × {} bytes)",
            data.len(),
            expected,
            n,
            ROW_SIZE,
        ));
    }

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let o = i * ROW_SIZE;
        out.push(VecInjectEntry {
            token_id: u32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]),
            coefficient: f32::from_le_bytes([
                data[o + 4],
                data[o + 5],
                data[o + 6],
                data[o + 7],
            ]),
            window_id: u16::from_le_bytes([data[o + 8], data[o + 9]]),
            position_in_window: u16::from_le_bytes([data[o + 10], data[o + 11]]),
            fact_id: u16::from_le_bytes([data[o + 12], data[o + 13]]),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_arch_config_matches_apollo11() {
        let cfg = ArchConfig::default();
        assert_eq!(cfg.retrieval_layer, 29);
        assert_eq!(cfg.query_head, 4);
        assert_eq!(cfg.injection_layer, 30);
        assert_eq!(cfg.inject_coefficient, 10.0);
    }

    #[test]
    fn load_missing_directory_errors() {
        let r = ApolloStore::load(Path::new("/tmp/apollo-does-not-exist"));
        assert!(matches!(r.unwrap_err(), StoreLoadError::Io { .. }));
    }
}
