//! The one place this binary knows which compute-backend crates it was
//! compiled with.
//!
//! Every command that used to inline the
//! `if metal { #[cfg(all(feature = "gpu", target_os = "macos"))] … }`
//! block now goes through [`backend_for_metal_flag`] (legacy `--metal`
//! bool surface) or [`backend_for_kind`]. Adding a backend (CUDA — the
//! DEC G-ladder) means adding one ctor entry to [`backend_registry`],
//! not editing every command.
//!
//! Semantics: an *explicitly requested* backend that is unavailable is a
//! loud error, never a silent CPU fallback — a DEC bench number must
//! not quietly land on the wrong substrate. (This tightens the old
//! run_cmd/bench sites, which fell back to CPU when Metal init failed;
//! the shannon/walk/local_runtime sites already hard-failed.)

use larql_compute::{backend_from_spec, BackendCtor, BackendKind, ComputeBackend};

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

#[cfg(all(feature = "gpu", target_os = "macos"))]
fn metal_ctor() -> Option<Box<dyn ComputeBackend>> {
    larql_compute_metal::metal_backend().map(|m| Box::new(m) as Box<dyn ComputeBackend>)
}

/// Constructors for the GPU backend crates compiled into this binary,
/// in `BackendKind::Auto` preference order.
pub fn backend_registry() -> Vec<(BackendKind, BackendCtor)> {
    #[cfg(all(feature = "gpu", target_os = "macos"))]
    {
        vec![(BackendKind::Metal, metal_ctor as BackendCtor)]
    }
    #[cfg(not(all(feature = "gpu", target_os = "macos")))]
    {
        Vec::new()
    }
}

/// Build the backend for a [`BackendKind`].
pub fn backend_for_kind(
    kind: BackendKind,
) -> Result<Box<dyn ComputeBackend>, Box<dyn core::error::Error>> {
    Ok(backend_from_spec(kind, &backend_registry())?)
}

/// Build the backend for a legacy `--metal` bool flag.
pub fn backend_for_metal_flag(
    metal: bool,
) -> Result<Box<dyn ComputeBackend>, Box<dyn core::error::Error>> {
    backend_for_kind(BackendKind::from_metal_flag(metal))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_flag_always_resolves() {
        let backend = backend_for_metal_flag(false).unwrap();
        assert!(backend.name().starts_with("cpu"));
    }

    #[cfg(not(all(feature = "gpu", target_os = "macos")))]
    #[test]
    fn metal_flag_errors_loudly_when_not_compiled_in() {
        // `unwrap_err` needs `Ok: Debug`, which `Box<dyn ComputeBackend>` isn't.
        let err = match backend_for_metal_flag(true) {
            Err(e) => e,
            Ok(_) => panic!("expected NotCompiledIn without the gpu feature"),
        };
        assert!(err.to_string().contains("metal"));
    }
}

#[cfg(all(feature = "gpu", target_os = "macos"))]
#[cfg(test)]
mod macos_tests {
    use super::*;

    /// On a gpu-feature macOS build the registry advertises Metal, and
    /// invoking its constructor is safe whether or not a device exists
    /// (`None` on headless CI, `Some` on an M-series Mac).
    #[test]
    fn registry_advertises_metal_and_ctor_is_callable() {
        let registry = backend_registry();
        assert_eq!(registry.len(), 1);
        assert_eq!(registry[0].0, BackendKind::Metal);
        if let Some(backend) = (registry[0].1)() {
            assert!(backend.name().contains("metal"));
        }
    }
}
