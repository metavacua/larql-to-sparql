//! C compiler configuration for LARQL crates.
//!
//! Centralizes platform-specific compiler flags (CPU architecture optimizations,
//! etc.) to avoid duplication across build.rs scripts.

/// Get CPU architecture-specific compiler flags.
///
/// Returns GCC/Clang-compatible compiler flags based on the target architecture.
/// These flags are only appropriate for GCC/Clang-compatible compilers;
/// MSVC and other toolchains handle optimizations differently.
///
/// Returns:
/// - **`x86_64` (non-MSVC)**: `["-mavx2"]`
/// - **aarch64 (non-MSVC)**: `["-march=armv8.2-a+dotprod"]`
/// - **MSVC or other**: `[]` (empty, compiler handles optimizations internally)
///
/// # Examples
///
/// ```ignore
/// use larql_build::compiler::cpu_flags;
///
/// fn main() {
///     let flags = cpu_flags();
///     let mut build = cc::Build::new();
///     for flag in flags {
///         build.flag(flag);
///     }
///     build.file("csrc/matmul.c").opt_level(3).compile("matmul");
/// }
/// ```
///
/// # Notes
///
/// These flags are architecture-specific optimizations for GCC/Clang.
/// On MSVC, flags are omitted because MSVC has different flag syntax and
/// handles these optimizations through different mechanisms.
#[must_use]
pub fn cpu_flags() -> &'static [&'static str] {
    // Check if target is MSVC (e.g., x86_64-pc-windows-msvc)
    let is_msvc = std::env::var("CARGO_CFG_TARGET")
        .map(|t| t.contains("msvc"))
        .unwrap_or(false);

    if is_msvc {
        // MSVC uses different flag syntax; skip for now
        return &[];
    }

    match std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() {
        Ok("x86_64") => &["-mavx2"],
        Ok("aarch64") => &["-march=armv8.2-a+dotprod"],
        _ => &[],
    }
}

/// Get the recommended optimization level.
///
/// Returns 3 for aggressive optimization suitable for performance-critical code.
#[must_use]
pub const fn optimization_level() -> u32 {
    3
}

/// Configure a C compiler with LARQL-specific flags (using cc crate).
///
/// Sets platform-appropriate CPU optimization flags based on the target architecture
/// and compiler toolchain:
/// - **GCC/Clang on x86_64**: `-mavx2` (AVX2 vector instructions)
/// - **GCC/Clang on aarch64**: `-march=armv8.2-a+dotprod` (ARM NEON dot-product instructions)
/// - **MSVC**: No special flags (MSVC handles optimizations internally)
/// - **Other**: No special flags
///
/// Also sets optimization level to 3 (aggressive optimization).
///
/// # Arguments
///
/// * `build` - A `cc::Build` object to configure
///
/// # Examples
///
/// ```ignore
/// use larql_build::compiler::configure_c_compiler;
///
/// fn main() {
///     let mut build = cc::Build::new();
///     build.file("csrc/matmul.c");
///     configure_c_compiler(&mut build);
///     build.compile("matmul");
/// }
/// ```
///
/// # Notes
///
/// This function should be called in a crate's `build.rs` script.
/// Requires `cc = "1.0"` in `[build-dependencies]`.
///
/// To use the raw flags without cc, use [`cpu_flags`] instead.
pub fn configure_c_compiler(build: &mut cc::Build) {
    // Set optimization level: 3 = aggressive optimization
    build.opt_level(optimization_level());

    // Apply CPU-specific flags
    for flag in cpu_flags() {
        build.flag(flag);
    }
}

/// Register cargo rerun-if-changed triggers.
///
/// Tells cargo to rerun the build script if any of the specified files change.
/// Useful for tracking source files, headers, or other build inputs.
///
/// # Arguments
///
/// * `files` - Slice of file paths or directory patterns to watch
///
/// # Examples
///
/// ```ignore
/// use larql_build::set_rerun_triggers;
///
/// fn main() {
///     set_rerun_triggers(&[
///         "csrc",
///         "build.rs",
///         "Cargo.toml",
///     ]);
/// }
/// ```
///
/// # Notes
///
/// - Paths are relative to the crate root
/// - Use directory names to watch all files in that directory
/// - This is idempotent: calling multiple times is safe (deduped by cargo)
pub fn set_rerun_triggers(files: &[&str]) {
    for file in files {
        println!("cargo:rerun-if-changed={file}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_flags_not_null() {
        let flags = cpu_flags();
        for flag in flags {
            assert!(!flag.is_empty(), "Flag should not be empty");
        }
    }

    #[test]
    fn test_cpu_flags_consistent() {
        let flags1 = cpu_flags();
        let flags2 = cpu_flags();
        assert_eq!(flags1, flags2);
    }

    #[test]
    fn test_optimization_level() {
        assert_eq!(optimization_level(), 3);
    }

    #[test]
    fn test_set_rerun_triggers_does_not_panic() {
        set_rerun_triggers(&["csrc", "build.rs"]);
    }

    #[test]
    fn test_set_rerun_triggers_empty_slice() {
        set_rerun_triggers(&[]);
    }

    #[test]
    fn test_set_rerun_triggers_single_file() {
        set_rerun_triggers(&["Cargo.toml"]);
    }

    #[test]
    fn test_set_rerun_triggers_multiple_files() {
        set_rerun_triggers(&["a.c", "b.c", "c.h", "build.rs"]);
    }

    #[test]
    fn test_cpu_flags_consistency() {
        // Verify the function is deterministic
        let flags1 = cpu_flags();
        let flags2 = cpu_flags();
        assert_eq!(flags1, flags2, "cpu_flags should return consistent results");
    }
}
