//! C compiler configuration for LARQL crates.
//!
//! Centralizes platform-specific compiler flags (CPU architecture optimizations,
//! etc.) to avoid duplication across build.rs scripts.

/// Get CPU architecture-specific compiler flags.
///
/// Returns the appropriate compiler flags based on the target architecture:
/// - **`x86_64`**: `["-mavx2"]`
/// - **aarch64**: `["-march=armv8.2-a+dotprod"]`
/// - **Other**: `[]` (empty)
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
/// These flags are architecture-specific optimizations that improve performance
/// of CPU-intensive operations. They should be combined with `-O3` optimization.
#[must_use]
pub fn cpu_flags() -> &'static [&'static str] {
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
/// Sets platform-appropriate CPU optimization flags based on the target architecture:
/// - **`x86_64`**: `-mavx2` (AVX2 vector instructions)
/// - **aarch64**: `-march=armv8.2-a+dotprod` (ARM NEON dot-product instructions)
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
