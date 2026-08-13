//! Resolving a model reference to a directory on disk.
//!
//! Split out of `safetensors.rs` (ROADMAP H5b). This is the one concern in
//! that file with nothing to do with tensor formats — it answers "where is
//! the model", not "how is it encoded".

#[cfg(target_arch = "wasm32")]
use crate::prelude::*;
use std::path::PathBuf;

use crate::detect::ModelError;

use super::{CONFIG_JSON, MODEL_PREFIX, SAFETENSORS_EXT, SNAPSHOTS_DIR};

/// Return the HuggingFace hub cache directory, respecting env-var overrides.
///
/// Priority (matches Python `huggingface_hub`):
/// 1. `HF_HUB_CACHE` — exact cache dir
/// 2. `HF_HOME` — HF home; hub cache = `$HF_HOME/hub`
/// 3. `HOME` (Unix) / `USERPROFILE` (Windows) — `~/.cache/huggingface/hub`
fn hf_hub_cache() -> PathBuf {
    if let Ok(p) = std::env::var("HF_HUB_CACHE") {
        return PathBuf::from(p);
    }
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        return PathBuf::from(hf_home).join("hub");
    }
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    home.join(".cache").join("huggingface").join("hub")
}

/// Resolve a HuggingFace model ID or path to a local directory or GGUF file.
pub fn resolve_model_path(model: &str) -> Result<PathBuf, ModelError> {
    let path = PathBuf::from(model);
    if path.is_dir() {
        return Ok(path);
    }
    // Single GGUF file
    if path.is_file() && path.extension().is_some_and(|ext| ext == "gguf") {
        return Ok(path);
    }

    // Try HuggingFace cache — resolve location using the same env-var priority
    // as the Python huggingface_hub library: HF_HUB_CACHE > HF_HOME > home dir.
    let cache_name = format!("{MODEL_PREFIX}{}", model.replace('/', "--"));
    let hf_cache = hf_hub_cache().join(&cache_name).join(SNAPSHOTS_DIR);

    if hf_cache.is_dir() {
        // Find the snapshot that has actual model files (safetensors or config.json+weights)
        let mut best: Option<PathBuf> = None;
        if let Ok(entries) = std::fs::read_dir(&hf_cache) {
            for entry in entries.flatten() {
                let p = entry.path();
                if !p.is_dir() {
                    continue;
                }
                // Prefer snapshot with safetensors files
                let has_st = std::fs::read_dir(&p)
                    .ok()
                    .map(|rd| {
                        rd.flatten().any(|e| {
                            e.path()
                                .extension()
                                .is_some_and(|ext| ext == SAFETENSORS_EXT)
                        })
                    })
                    .unwrap_or(false);
                if has_st {
                    return Ok(p);
                }
                // Fallback: any snapshot with config.json
                if p.join(CONFIG_JSON).exists() {
                    best = Some(p);
                }
            }
        }
        if let Some(p) = best {
            return Ok(p);
        }
    }

    Err(ModelError::NotADirectory(path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::Mutex;
    use tempfile::TempDir;

    static HOME_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn resolve_model_path_existing_dir() {
        let dir = TempDir::new().unwrap();
        let result = resolve_model_path(dir.path().to_str().unwrap()).unwrap();
        assert_eq!(result, dir.path());
    }

    #[test]
    fn resolve_model_path_existing_gguf_file() {
        let dir = TempDir::new().unwrap();
        let gguf = dir.path().join("model.gguf");
        fs::write(&gguf, b"").unwrap();
        let result = resolve_model_path(gguf.to_str().unwrap()).unwrap();
        assert_eq!(result, gguf);
    }

    #[test]
    fn resolve_model_path_nonexistent_returns_error() {
        // Use a temp dir that we immediately drop, so the path is guaranteed
        // not to exist on any OS — no hardcoded Unix-style paths.
        let dir = TempDir::new().unwrap();
        let gone = dir.path().join("subdir_that_was_never_created");
        drop(dir);
        let result = resolve_model_path(gone.to_str().unwrap());
        assert!(result.is_err());
    }

    #[test]
    fn resolve_model_path_hf_cache_with_safetensors() {
        let _lock = HOME_LOCK.lock().unwrap();
        let home = TempDir::new().unwrap();
        let snapshot = home
            .path()
            .join(".cache")
            .join("huggingface")
            .join("hub")
            .join("models--org--name")
            .join("snapshots")
            .join("abc123");
        fs::create_dir_all(&snapshot).unwrap();
        fs::write(snapshot.join("model.safetensors"), b"").unwrap();
        std::env::set_var("HOME", home.path().to_str().unwrap());
        let result = resolve_model_path("org/name").unwrap();
        std::env::remove_var("HOME");
        assert_eq!(result, snapshot);
    }

    #[test]
    fn hf_hub_cache_honours_hf_hub_cache_env() {
        let _lock = HOME_LOCK.lock().unwrap();
        let tmp = TempDir::new().unwrap();
        let prior_hub = std::env::var_os("HF_HUB_CACHE");
        let prior_home_var = std::env::var_os("HF_HOME");
        std::env::set_var("HF_HUB_CACHE", tmp.path());
        // HF_HUB_CACHE wins over HF_HOME — clear it to avoid ambiguity.
        std::env::remove_var("HF_HOME");
        let got = hf_hub_cache();
        match prior_hub {
            Some(v) => std::env::set_var("HF_HUB_CACHE", v),
            None => std::env::remove_var("HF_HUB_CACHE"),
        }
        if let Some(v) = prior_home_var {
            std::env::set_var("HF_HOME", v);
        }
        assert_eq!(got, tmp.path());
    }

    #[test]
    fn hf_hub_cache_honours_hf_home_env_when_hub_cache_unset() {
        let _lock = HOME_LOCK.lock().unwrap();
        let tmp = TempDir::new().unwrap();
        let prior_hub = std::env::var_os("HF_HUB_CACHE");
        let prior_home_var = std::env::var_os("HF_HOME");
        // HF_HUB_CACHE must be unset for the HF_HOME branch to fire.
        std::env::remove_var("HF_HUB_CACHE");
        std::env::set_var("HF_HOME", tmp.path());
        let got = hf_hub_cache();
        match prior_hub {
            Some(v) => std::env::set_var("HF_HUB_CACHE", v),
            None => std::env::remove_var("HF_HUB_CACHE"),
        }
        match prior_home_var {
            Some(v) => std::env::set_var("HF_HOME", v),
            None => std::env::remove_var("HF_HOME"),
        }
        assert_eq!(got, tmp.path().join("hub"));
    }

    #[test]
    fn resolve_model_path_hf_cache_fallback_config_json() {
        let _lock = HOME_LOCK.lock().unwrap();
        let home = TempDir::new().unwrap();
        let snapshot = home
            .path()
            .join(".cache")
            .join("huggingface")
            .join("hub")
            .join("models--org--model")
            .join("snapshots")
            .join("def456");
        fs::create_dir_all(&snapshot).unwrap();
        fs::write(snapshot.join("config.json"), b"{}").unwrap();
        std::env::set_var("HOME", home.path().to_str().unwrap());
        let result = resolve_model_path("org/model").unwrap();
        std::env::remove_var("HOME");
        assert_eq!(result, snapshot);
    }
}
