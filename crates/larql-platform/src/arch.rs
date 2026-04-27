//! CPU architecture detection and feature flags.

/// CPU architecture enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    /// `x86_64` (Intel/AMD 64-bit)
    X86_64,
    /// ARM64 / `AArch64` (Apple Silicon, Raspberry Pi, etc.)
    AArch64,
    /// WebAssembly (`wasm32`)
    WebAssembly,
    /// Other/Unknown architecture
    Unknown,
}

impl Architecture {
    /// Detect the current CPU architecture at compile time.
    #[must_use]
    pub const fn detect() -> Self {
        if cfg!(target_arch = "x86_64") {
            Self::X86_64
        } else if cfg!(target_arch = "aarch64") {
            Self::AArch64
        } else if cfg!(target_arch = "wasm32") {
            Self::WebAssembly
        } else {
            Self::Unknown
        }
    }
}

/// CPU feature flags for compute optimization.
///
/// This struct represents available CPU features that LARQL can leverage
/// for performance optimization (e.g., vectorized operations).
#[derive(Debug, Clone)]
pub struct CpuFlags {
    /// AVX2 support (`x86_64` only)
    ///
    /// When true, compile with `-mavx2` for 256-bit SIMD operations.
    pub avx2: bool,

    /// NEON dotprod support (aarch64 only)
    ///
    /// When true, compile with `-march=armv8.2-a+dotprod` for 8-bit
    /// and 16-bit dot product instructions.
    pub dotprod: bool,
}

impl CpuFlags {
    /// Detect available CPU features for the current architecture.
    ///
    /// # Returns
    ///
    /// A `CpuFlags` struct with feature availability based on the target architecture.
    /// Compile-time detection; no runtime CPU feature detection is performed.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use larql_platform::CpuFlags;
    ///
    /// let flags = CpuFlags::detect();
    /// if flags.avx2 {
    ///     println!("AVX2 available, use vectorized operations");
    /// }
    /// ```
    #[must_use]
    pub const fn detect() -> Self {
        Self {
            avx2: cfg!(target_arch = "x86_64"),
            dotprod: cfg!(target_arch = "aarch64"),
        }
    }

    /// Check if any SIMD features are available.
    #[must_use]
    pub const fn has_simd(&self) -> bool {
        self.avx2 || self.dotprod
    }
}

impl Default for CpuFlags {
    fn default() -> Self {
        Self::detect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architecture_detection_consistent() {
        let arch1 = Architecture::detect();
        let arch2 = Architecture::detect();
        assert_eq!(arch1, arch2);
    }

    #[test]
    fn test_cpu_flags_detection_consistent() {
        let flags1 = CpuFlags::detect();
        let flags2 = CpuFlags::detect();
        assert_eq!(flags1.avx2, flags2.avx2);
        assert_eq!(flags1.dotprod, flags2.dotprod);
    }

    #[test]
    fn test_avx2_only_on_x86_64() {
        let _flags = CpuFlags::detect();
        #[cfg(not(target_arch = "x86_64"))]
        {
            assert!(!_flags.avx2);
        }
    }

    #[test]
    fn test_dotprod_only_on_aarch64() {
        let flags = CpuFlags::detect();
        #[cfg(not(target_arch = "aarch64"))]
        {
            assert!(!flags.dotprod);
        }
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn test_architecture_enum_consistency() {
        let arch = Architecture::detect();
        match arch {
            Architecture::X86_64 => assert!(cfg!(target_arch = "x86_64")),
            Architecture::AArch64 => assert!(cfg!(target_arch = "aarch64")),
            Architecture::WebAssembly => assert!(cfg!(target_arch = "wasm32")),
            Architecture::Unknown => {
                // Unknown is fine for unsupported architectures
                assert!(!cfg!(target_arch = "x86_64"));
                assert!(!cfg!(target_arch = "aarch64"));
                assert!(!cfg!(target_arch = "wasm32"));
            }
        }
    }
}
