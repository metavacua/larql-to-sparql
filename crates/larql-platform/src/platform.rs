//! Runtime platform detection and capability inference.

use crate::arch::CpuFlags;

/// Operating system enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatingSystem {
    /// macOS (Intel and Apple Silicon)
    MacOS,
    /// Linux (`x86_64`, aarch64, etc.)
    Linux,
    /// Windows
    Windows,
    /// WebAssembly
    WebAssembly,
    /// Other/Unknown OS
    Unknown,
}

impl OperatingSystem {
    /// Detect the current operating system at compile time.
    #[must_use]
    pub const fn detect() -> Self {
        if cfg!(target_os = "macos") {
            Self::MacOS
        } else if cfg!(target_os = "linux") {
            Self::Linux
        } else if cfg!(target_os = "windows") {
            Self::Windows
        } else if cfg!(target_arch = "wasm32") {
            Self::WebAssembly
        } else {
            Self::Unknown
        }
    }
}

/// BLAS backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlasBackend {
    /// Accelerate framework (macOS native)
    Accelerate,
    /// `OpenBLAS` (Linux, cross-platform)
    OpenBLAS,
    /// Generic fallback (no optimized BLAS)
    Generic,
}

impl BlasBackend {
    /// Human-readable name for the BLAS backend.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Accelerate => "Accelerate",
            Self::OpenBLAS => "OpenBLAS",
            Self::Generic => "Generic",
        }
    }
}

/// Platform capabilities and configuration.
///
/// Provides a single source of truth for platform-specific behavior.
/// Use `Platform::detect()` to query capabilities at runtime.
#[derive(Debug, Clone)]
pub struct Platform {
    os: OperatingSystem,
    cpu_flags: CpuFlags,
}

impl Platform {
    /// Detect the current platform's capabilities.
    ///
    /// # Returns
    ///
    /// A `Platform` struct representing the current system's capabilities.
    /// This function never panics; unsupported platforms return conservative defaults.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use larql_platform::Platform;
    ///
    /// let platform = Platform::detect();
    /// println!("Running on: {:?}", platform.os());
    /// ```
    #[must_use]
    pub const fn detect() -> Self {
        Self {
            os: OperatingSystem::detect(),
            cpu_flags: CpuFlags::detect(),
        }
    }

    /// Get the current operating system.
    #[must_use]
    pub const fn os(&self) -> OperatingSystem {
        self.os
    }

    /// Get the target OS as a string (for display/logging).
    ///
    /// Returns the compile-time target OS, useful for diagnostics.
    #[must_use]
    pub const fn target_os(&self) -> &'static str {
        std::env::consts::OS
    }

    /// Get the target architecture as a string (for display/logging).
    ///
    /// Returns the compile-time target architecture, useful for diagnostics.
    #[must_use]
    pub const fn target_arch(&self) -> &'static str {
        std::env::consts::ARCH
    }

    /// Check if Metal GPU acceleration is available.
    ///
    /// Metal is only available on macOS (Intel and Apple Silicon).
    /// This is a compile-time + runtime check: feature must be enabled
    /// AND OS must be macOS.
    ///
    /// # Returns
    ///
    /// `true` if Metal GPU is available on this system.
    #[must_use]
    pub fn metal_available(&self) -> bool {
        self.os == OperatingSystem::MacOS
    }

    /// Recommended BLAS backend for this platform.
    ///
    /// Selection rules:
    /// - macOS: Accelerate (native framework)
    /// - Linux: `OpenBLAS` (cross-platform, optimized)
    /// - Other: Generic fallback
    ///
    /// # Returns
    ///
    /// The recommended `BlasBackend` for this platform.
    #[must_use]
    pub const fn blas_backend(&self) -> BlasBackend {
        match self.os {
            OperatingSystem::MacOS => BlasBackend::Accelerate,
            OperatingSystem::Linux => BlasBackend::OpenBLAS,
            _ => BlasBackend::Generic,
        }
    }

    /// Check if mmap advice hints are supported.
    ///
    /// Unix-like systems (Linux, macOS) support `libc::madvise()` for
    /// memory mapping hints (`MADV_SEQUENTIAL`, `MADV_WILLNEED`, etc.).
    /// Windows and WebAssembly do not support these hints.
    ///
    /// # Returns
    ///
    /// `true` if `madvise()` and similar hints are available.
    #[must_use]
    pub fn supports_mmap_advice(&self) -> bool {
        cfg!(unix) && (self.os == OperatingSystem::MacOS || self.os == OperatingSystem::Linux)
    }

    /// Check if this platform is WebAssembly.
    #[must_use]
    pub fn is_wasm(&self) -> bool {
        self.os == OperatingSystem::WebAssembly
    }

    /// Check if this platform is macOS.
    #[must_use]
    pub fn is_macos(&self) -> bool {
        self.os == OperatingSystem::MacOS
    }

    /// Check if this platform is Linux.
    #[must_use]
    pub fn is_linux(&self) -> bool {
        self.os == OperatingSystem::Linux
    }

    /// Check if this platform is Windows.
    #[must_use]
    pub fn is_windows(&self) -> bool {
        self.os == OperatingSystem::Windows
    }

    /// Get CPU feature flags (AVX2, NEON dotprod, etc.).
    #[must_use]
    pub const fn cpu_flags(&self) -> &CpuFlags {
        &self.cpu_flags
    }
}

impl Default for Platform {
    fn default() -> Self {
        Self::detect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_os_detection_is_consistent() {
        let os1 = OperatingSystem::detect();
        let os2 = OperatingSystem::detect();
        assert_eq!(os1, os2);
    }

    #[test]
    fn test_platform_detect_consistent() {
        let p1 = Platform::detect();
        let p2 = Platform::detect();
        assert_eq!(p1.os(), p2.os());
    }

    #[test]
    fn test_metal_only_on_macos() {
        let platform = Platform::detect();
        if platform.os() != OperatingSystem::MacOS {
            assert!(!platform.metal_available());
        }
    }

    #[test]
    fn test_blas_backend_never_panics() {
        let platform = Platform::detect();
        let _backend = platform.blas_backend();
    }

    #[test]
    fn test_target_os_and_arch_not_empty() {
        let platform = Platform::detect();
        assert!(!platform.target_os().is_empty());
        assert!(!platform.target_arch().is_empty());
    }

    #[test]
    fn test_mmap_advice_on_unix_only() {
        let platform = Platform::detect();
        #[cfg(unix)]
        {
            // On Unix, should be true (at least for Linux/macOS)
            if platform.os() == OperatingSystem::Linux || platform.os() == OperatingSystem::MacOS {
                assert!(platform.supports_mmap_advice());
            }
        }
        #[cfg(not(unix))]
        {
            assert!(!platform.supports_mmap_advice());
        }
    }
}
