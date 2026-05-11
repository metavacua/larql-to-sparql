//! Weight-file bakers for `COMPILE INTO VINDEX`: rewrite down / gate /
//! up / MEMIT-delta columns on disk so the compiled vindex is
//! self-contained and no runtime patch overlay is needed.
//!
//! Each verb-shaped baker lives in its own file:
//!
//!   - `down.rs`: `patch_down_weights` — column-replace into
//!     `down_weights.bin`.
//!   - `memit_apply.rs`: `apply_memit_deltas_to_down_weights` —
//!     additive ΔW into the same slab after column-replace runs.
//!   - `gate.rs`: `patch_gate_vectors` — row-write into
//!     `gate_vectors.bin` honouring per-layer offsets.
//!   - `up.rs`: `patch_up_weights` — manifest-driven row-write into
//!     whichever file `up_proj` lives in (per-layer).
//!
//! Shared helpers live in `mod.rs`: `copy_for_patch` (own-the-file
//! before seek-write) and `detect_down_dtype_bytes` (size → bytes-per-
//! element disambiguation).

use crate::error::LqlError;

mod down;
mod gate;
mod memit_apply;
mod up;

pub(super) use down::patch_down_weights;
pub(super) use gate::patch_gate_vectors;
pub(super) use memit_apply::apply_memit_deltas_to_down_weights;
pub(super) use up::patch_up_weights;

/// Bytes per `f32` element. Exposed as a named constant so the
/// dtype-detection arithmetic doesn't carry a magic literal across
/// every baker.
pub(super) const BYTES_PER_F32: usize = 4;
/// Bytes per `f16` element.
pub(super) const BYTES_PER_F16: usize = 2;

/// Replace `dst` with a fresh writable copy of `src`. Compile bakers
/// hard-link unchanging files in bulk; calling this before a seek-
/// write breaks the link so the source vindex is never mutated.
pub(super) fn copy_for_patch(src: &std::path::Path, dst: &std::path::Path) -> Result<(), LqlError> {
    let _ = std::fs::remove_file(dst);
    std::fs::copy(src, dst)
        .map_err(|e| LqlError::exec(format!("failed to copy {}", src.display()), e))?;
    Ok(())
}

/// Disambiguate `down_weights.bin`'s storage dtype from its file
/// size. The slab holds `total_elements` floats; if the byte count
/// matches `total × 4` it's f32, `total × 2` it's f16, anything else
/// is a corrupt or malformed vindex.
pub(super) fn detect_down_dtype_bytes(
    file_bytes: usize,
    total_elements: usize,
) -> Result<usize, LqlError> {
    if file_bytes == total_elements * BYTES_PER_F32 {
        Ok(BYTES_PER_F32)
    } else if file_bytes == total_elements * BYTES_PER_F16 {
        Ok(BYTES_PER_F16)
    } else {
        Err(LqlError::Execution(format!(
            "down_weights.bin size {file_bytes} matches neither f32 ({}) nor f16 ({})",
            total_elements * BYTES_PER_F32,
            total_elements * BYTES_PER_F16,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_dtype_recognises_f32() {
        let total = 1024;
        let size = total * BYTES_PER_F32;
        assert_eq!(detect_down_dtype_bytes(size, total).unwrap(), BYTES_PER_F32);
    }

    #[test]
    fn detect_dtype_recognises_f16() {
        let total = 1024;
        let size = total * BYTES_PER_F16;
        assert_eq!(detect_down_dtype_bytes(size, total).unwrap(), BYTES_PER_F16);
    }

    #[test]
    fn detect_dtype_rejects_mismatch() {
        // 100 bytes for 32 elements is neither f32 (128) nor f16 (64).
        let err = detect_down_dtype_bytes(100, 32).unwrap_err();
        assert!(err.to_string().contains("matches neither"));
    }

    #[test]
    fn copy_for_patch_creates_destination() {
        let tmp = std::env::temp_dir().join(format!(
            "larql_copy_for_patch_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        let src = tmp.join("src.bin");
        let dst = tmp.join("dst.bin");
        std::fs::write(&src, b"hello").unwrap();
        copy_for_patch(&src, &dst).unwrap();
        assert_eq!(std::fs::read(&dst).unwrap(), b"hello");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn copy_for_patch_replaces_existing_destination() {
        let tmp = std::env::temp_dir().join(format!(
            "larql_copy_for_patch_replace_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        let src = tmp.join("src.bin");
        let dst = tmp.join("dst.bin");
        std::fs::write(&src, b"new").unwrap();
        std::fs::write(&dst, b"stale-data").unwrap();
        copy_for_patch(&src, &dst).unwrap();
        assert_eq!(std::fs::read(&dst).unwrap(), b"new");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn copy_for_patch_errors_on_missing_source() {
        let tmp = std::env::temp_dir().join(format!(
            "larql_copy_for_patch_missing_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        let src = tmp.join("nonexistent.bin");
        let dst = tmp.join("dst.bin");
        let err = copy_for_patch(&src, &dst).unwrap_err();
        assert!(err.to_string().contains("failed to copy"));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn dtype_constants_are_canonical() {
        assert_eq!(BYTES_PER_F32, 4);
        assert_eq!(BYTES_PER_F16, 2);
    }
}
