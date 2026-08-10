//! Backend construction — one factory replacing per-call-site cfg blocks.
//!
//! Before this module, every binary that wanted "Metal if asked, CPU
//! otherwise" repeated the same `if metal { #[cfg(...)] … }` block
//! (seven sites in larql-cli alone), each with its own error string and
//! its own idea of whether to fall back or hard-fail. Adding a backend
//! (CUDA, G-ladder) meant touching every site.
//!
//! `larql-compute` must not depend on any backend crate (ADR-019 — the
//! dep would be circular), so the factory takes the constructors by
//! injection: the *binary* builds a [`BackendCtor`] registry naming the
//! backend crates it was compiled with, and every construction site
//! collapses to one [`backend_from_spec`] call.
//!
//! ```rust,ignore
//! // In the binary, once:
//! fn registry() -> Vec<(BackendKind, BackendCtor)> {
//!     let mut r: Vec<(BackendKind, BackendCtor)> = Vec::new();
//!     #[cfg(all(feature = "gpu", target_os = "macos"))]
//!     r.push((BackendKind::Metal, || {
//!         larql_compute_metal::metal_backend()
//!             .map(|m| Box::new(m) as Box<dyn ComputeBackend>)
//!     }));
//!     r
//! }
//!
//! // At each construction site:
//! let backend = backend_from_spec(kind, &registry())?;
//! ```
//!
//! ## Capability probing after construction
//!
//! The factory returns `Box<dyn ComputeBackend>`; callers must not
//! branch on [`BackendKind`] afterwards. The canonical probe for "has a
//! fused quantised decode pipeline" (what the old `if metal` branches
//! actually meant) is
//! `supports(Capability::PrefillQ4) && supports(Capability::DecodeToken)`
//! — probed **on the constructed instance**, never on
//! [`crate::default_backend`] (which is always CPU and made the old
//! `metal_ready_for_q4` check vacuous).

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

use super::ComputeBackend;
// core::fmt/core::str::FromStr are the same items std re-exports --
// portable regardless of target, no cfg needed.
use core::fmt;
use core::str::FromStr;

/// Which backend a caller is asking for.
///
/// Parsed from user-facing strings (`--backend cpu|metal|cuda|vulkan|auto`,
/// `DEC0_BACKEND`); the legacy `--metal` bool maps via
/// [`BackendKind::from_metal_flag`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BackendKind {
    /// The always-available BLAS/NEON CPU backend in this crate.
    Cpu,
    /// `larql-compute-metal` (macOS + `gpu` feature).
    Metal,
    /// `larql-compute-cuda` (planned — G-ladder).
    Cuda,
    /// `larql-compute-vulkan` (planned).
    Vulkan,
    /// First registry entry whose constructor finds a device, else CPU.
    /// Never fails.
    Auto,
}

impl BackendKind {
    /// Canonical lowercase name, matching what [`FromStr`] accepts.
    pub fn as_str(&self) -> &'static str {
        match self {
            BackendKind::Cpu => "cpu",
            BackendKind::Metal => "metal",
            BackendKind::Cuda => "cuda",
            BackendKind::Vulkan => "vulkan",
            BackendKind::Auto => "auto",
        }
    }

    /// Bridge for the legacy `--metal: bool` CLI flags: `true` → `Metal`
    /// (requested explicitly, so an unavailable Metal is an error, not a
    /// silent CPU fallback), `false` → `Cpu`.
    pub fn from_metal_flag(metal: bool) -> Self {
        if metal {
            BackendKind::Metal
        } else {
            BackendKind::Cpu
        }
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Error from parsing a backend-kind string.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseBackendKindError(String);

impl fmt::Display for ParseBackendKindError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown backend `{}` (expected cpu, metal, cuda, vulkan, or auto)",
            self.0
        )
    }
}

impl std::error::Error for ParseBackendKindError {}

impl FromStr for BackendKind {
    type Err = ParseBackendKindError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "cpu" => Ok(BackendKind::Cpu),
            "metal" => Ok(BackendKind::Metal),
            "cuda" => Ok(BackendKind::Cuda),
            "vulkan" => Ok(BackendKind::Vulkan),
            "auto" => Ok(BackendKind::Auto),
            _ => Err(ParseBackendKindError(s.to_string())),
        }
    }
}

/// A backend constructor supplied by the binary. Returns `None` when the
/// crate is present but no usable device exists on this host (mirrors
/// `larql_compute_metal::metal_backend()`).
pub type BackendCtor = fn() -> Option<Box<dyn ComputeBackend>>;

/// Why a *named* backend request could not be satisfied.
///
/// `Auto` and `Cpu` never produce these — they always resolve to a
/// usable backend.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackendSelectError {
    /// No registry entry for the requested kind: the binary was built
    /// without that backend crate/feature.
    NotCompiledIn(BackendKind),
    /// The backend crate is compiled in but its constructor found no
    /// usable device on this host.
    NoDevice(BackendKind),
}

impl fmt::Display for BackendSelectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendSelectError::NotCompiledIn(kind) => write!(
                f,
                "backend `{kind}` is not compiled into this binary \
                 (rebuild with the matching feature — e.g. `--features gpu` \
                 for metal on an Apple-silicon Mac)"
            ),
            BackendSelectError::NoDevice(kind) => {
                write!(f, "backend `{kind}` found no usable device on this host")
            }
        }
    }
}

impl std::error::Error for BackendSelectError {}

/// Build the backend a caller asked for.
///
/// - `Cpu` → [`crate::cpu_backend`], always succeeds.
/// - `Auto` → first registry constructor that finds a device, else CPU;
///   always succeeds.
/// - A named GPU kind → its registry constructor, with a loud error if
///   the binary lacks the crate ([`BackendSelectError::NotCompiledIn`])
///   or the host lacks a device ([`BackendSelectError::NoDevice`]).
///   An explicit request never silently falls back to CPU — that would
///   put a bench number on the wrong substrate.
pub fn backend_from_spec(
    kind: BackendKind,
    registry: &[(BackendKind, BackendCtor)],
) -> Result<Box<dyn ComputeBackend>, BackendSelectError> {
    match kind {
        BackendKind::Cpu => Ok(crate::cpu_backend()),
        BackendKind::Auto => Ok(registry
            .iter()
            .find_map(|(_, ctor)| ctor())
            .unwrap_or_else(crate::cpu_backend)),
        requested => {
            let (_, ctor) = registry
                .iter()
                .find(|(k, _)| *k == requested)
                .ok_or(BackendSelectError::NotCompiledIn(requested))?;
            ctor().ok_or(BackendSelectError::NoDevice(requested))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuBackend;

    fn some_cpu_ctor() -> Option<Box<dyn ComputeBackend>> {
        Some(Box::new(CpuBackend))
    }

    fn no_device_ctor() -> Option<Box<dyn ComputeBackend>> {
        None
    }

    #[test]
    fn kind_parses_and_round_trips() {
        for kind in [
            BackendKind::Cpu,
            BackendKind::Metal,
            BackendKind::Cuda,
            BackendKind::Vulkan,
            BackendKind::Auto,
        ] {
            assert_eq!(kind.as_str().parse::<BackendKind>().unwrap(), kind);
        }
        assert_eq!("METAL".parse::<BackendKind>().unwrap(), BackendKind::Metal);
        let err = "tpu".parse::<BackendKind>().unwrap_err();
        assert!(err.to_string().contains("tpu"));
    }

    #[test]
    fn metal_flag_bridges_to_kind() {
        assert_eq!(BackendKind::from_metal_flag(true), BackendKind::Metal);
        assert_eq!(BackendKind::from_metal_flag(false), BackendKind::Cpu);
    }

    #[test]
    fn cpu_kind_ignores_registry() {
        let backend = backend_from_spec(BackendKind::Cpu, &[]).unwrap();
        assert!(backend.name().starts_with("cpu"));
    }

    #[test]
    fn named_kind_missing_from_registry_is_not_compiled_in() {
        // `unwrap_err` needs `Ok: Debug`, which `Box<dyn ComputeBackend>` isn't.
        let err = match backend_from_spec(BackendKind::Cuda, &[]) {
            Err(e) => e,
            Ok(_) => panic!("expected NotCompiledIn for an empty registry"),
        };
        assert_eq!(err, BackendSelectError::NotCompiledIn(BackendKind::Cuda));
        assert!(err.to_string().contains("cuda"));
    }

    #[test]
    fn named_kind_with_no_device_is_a_loud_error_not_a_cpu_fallback() {
        let registry: &[(BackendKind, BackendCtor)] = &[(BackendKind::Metal, no_device_ctor)];
        let err = match backend_from_spec(BackendKind::Metal, registry) {
            Err(e) => e,
            Ok(_) => panic!("expected NoDevice from a deviceless constructor"),
        };
        assert_eq!(err, BackendSelectError::NoDevice(BackendKind::Metal));
    }

    #[test]
    fn named_kind_with_device_constructs_it() {
        let registry: &[(BackendKind, BackendCtor)] = &[(BackendKind::Metal, some_cpu_ctor)];
        let backend = backend_from_spec(BackendKind::Metal, registry).unwrap();
        assert!(backend.name().starts_with("cpu")); // stub ctor hands back CPU
    }

    #[test]
    fn auto_falls_back_to_cpu_on_empty_registry() {
        let backend = backend_from_spec(BackendKind::Auto, &[]).unwrap();
        assert!(backend.name().starts_with("cpu"));
    }

    #[test]
    fn auto_skips_deviceless_entries_and_takes_first_live_one() {
        let registry: &[(BackendKind, BackendCtor)] = &[
            (BackendKind::Cuda, no_device_ctor),
            (BackendKind::Metal, some_cpu_ctor),
        ];
        let backend = backend_from_spec(BackendKind::Auto, registry).unwrap();
        assert!(backend.name().starts_with("cpu"));
    }
}
