#![warn(missing_docs)]
#![warn(unsafe_code)]

//! Platform detection and capability inference for LARQL.
//!
//! This crate provides a centralized, single-source-of-truth for platform-specific
//! behavior across LARQL. Instead of scattered `#[cfg(...)]` blocks throughout the
//! codebase, crates use `Platform::detect()` to query capabilities at runtime or
//! build time.
//!
//! # Examples
//!
//! ```ignore
//! use larql_platform::Platform;
//!
//! let platform = Platform::detect();
//! if platform.metal_available() {
//!     println!("Metal GPU is available on this macOS system");
//! }
//! ```
//!
//! # Feature Detection
//!
//! - **Metal GPU**: Available only on macOS
//! - **BLAS Backend**: Automatic selection (`Accelerate` on macOS, `OpenBLAS` on Linux)
//! - **Memory Mapping**: Advice hints available on Unix-like systems
//! - **CPU Architecture**: `x86_64` (AVX2), `aarch64` (NEON dotprod)

pub mod arch;
pub mod platform;

pub use arch::{Architecture, CpuFlags};
pub use platform::{BlasBackend, OperatingSystem, Platform};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_detection() {
        let platform = Platform::detect();
        // Platform detection should not panic
        assert!(!platform.target_os().is_empty());
        assert!(!platform.target_arch().is_empty());
    }

    #[test]
    fn test_cpu_flags_detection() {
        let _flags = CpuFlags::detect();
        // CPU flags should be detected without panicking
    }

    #[test]
    fn test_blas_backend_consistency() {
        let platform = Platform::detect();
        // BLAS backend selection should always return a valid backend
        let _backend = platform.blas_backend();
    }

    #[test]
    fn test_metal_availability_consistency() {
        let platform = Platform::detect();
        // Metal should only be available on macOS
        if platform.os() != OperatingSystem::MacOS {
            assert!(!platform.metal_available());
        }
    }

    #[test]
    fn test_mmap_advice_consistency() {
        let platform = Platform::detect();
        // MMAP advice only available on Unix-like systems
        #[cfg(not(unix))]
        {
            assert!(!platform.supports_mmap_advice());
        }
    }
}
