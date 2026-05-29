//! Atomic write helpers for `COMPILE INTO {VINDEX, MODEL}`.
//!
//! Both compile targets produce a directory of files. The previous
//! implementation wrote into the user-supplied output path in place;
//! a SIGKILL or panic mid-bake left a half-baked vindex with a valid
//! `weight_manifest.json` and torn `down_weights.bin`. There was no
//! way to detect or recover the partial state.
//!
//! The pattern used here:
//!
//!   1. Compute a sibling staging directory: `<parent>/<basename>.tmp.<pid>`.
//!      Keeping it under the same parent keeps the final `rename` atomic
//!      (no cross-filesystem move).
//!   2. Reject `output_dir == source_path` up front so a stray invocation
//!      can't dissolve the source by hard-linking files into themselves.
//!   3. Run all writes against the staging directory.
//!   4. On success: remove any prior `output_dir`, then `rename` staging
//!      → output. This is the only step that mutates the user's intended
//!      destination.
//!   5. On error: remove the staging directory.

use std::path::{Path, PathBuf};

use crate::error::LqlError;

/// Decide whether two paths refer to the same on-disk directory.
///
/// Uses `canonicalize` when both paths exist (handles symlinks, `.`, `..`,
/// trailing slashes); falls back to `std::path::absolute` for paths that
/// haven't been created yet so the comparison still works in `compile`'s
/// "output doesn't exist yet" mode.
pub(super) fn paths_collide(a: &Path, b: &Path) -> Result<bool, LqlError> {
    let canon_a = canonical_or_absolute(a)?;
    let canon_b = canonical_or_absolute(b)?;
    Ok(canon_a == canon_b)
}

fn canonical_or_absolute(p: &Path) -> Result<PathBuf, LqlError> {
    if p.exists() {
        return p
            .canonicalize()
            .map_err(|e| LqlError::exec(format!("canonicalize {}", p.display()), e));
    }
    std::path::absolute(p).map_err(|e| LqlError::exec(format!("absolute path {}", p.display()), e))
}

/// Sibling staging directory for `final_dir`. Same parent → same
/// filesystem → atomic `rename`.
pub(super) fn staging_dir_for(final_dir: &Path) -> PathBuf {
    let basename = final_dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "compile".into());
    let parent = final_dir.parent().unwrap_or_else(|| Path::new("."));
    let staging = format!(".{}.tmp.{}", basename, std::process::id());
    parent.join(staging)
}

/// Run `work(staging_dir)` and atomically promote the staging directory
/// to `final_dir` on success. Cleans up the staging directory on error.
///
/// `source` is the read-side directory the compile is reading from; if
/// it canonicalises to the same path as `final_dir`, the call is rejected
/// before any writes happen.
pub(super) fn run_atomic_compile<F>(
    final_dir: &Path,
    source: &Path,
    work: F,
) -> Result<Vec<String>, LqlError>
where
    F: FnOnce(&Path) -> Result<Vec<String>, LqlError>,
{
    if paths_collide(source, final_dir)? {
        return Err(LqlError::Execution(format!(
            "COMPILE output {} resolves to the source vindex; refusing to clobber. \
             Pick a different destination.",
            final_dir.display()
        )));
    }

    // Fresh staging dir. Remove any leftover from a prior crashed run
    // (same PID is unlikely but `process::id()` recycles eventually).
    let staging = staging_dir_for(final_dir);
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging)
        .map_err(|e| LqlError::exec(format!("create staging dir {}", staging.display()), e))?;

    match work(&staging) {
        Ok(out) => {
            // Promote: drop any prior contents at final_dir, then move
            // staging into place atomically.
            let _ = std::fs::remove_dir_all(final_dir);
            std::fs::rename(&staging, final_dir).map_err(|e| {
                // If the rename fails, scrub the staging dir so we don't
                // leak a half-built tree at `<basename>.tmp.<pid>`.
                let _ = std::fs::remove_dir_all(&staging);
                LqlError::exec(
                    format!(
                        "promote staging {} → {}",
                        staging.display(),
                        final_dir.display()
                    ),
                    e,
                )
            })?;
            Ok(out)
        }
        Err(e) => {
            let _ = std::fs::remove_dir_all(&staging);
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_tmp(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "larql_atomic_{label}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ))
    }

    #[test]
    fn paths_collide_detects_existing_same_dir() {
        let a = unique_tmp("paths_collide");
        std::fs::create_dir_all(&a).unwrap();
        // `./a/.` resolves to the same canonical form as `a`.
        let b = a.join(".");
        assert!(paths_collide(&a, &b).unwrap());
        let _ = std::fs::remove_dir_all(&a);
    }

    #[test]
    fn paths_collide_distinguishes_different_dirs() {
        let a = unique_tmp("collide_diff_a");
        let b = unique_tmp("collide_diff_b");
        std::fs::create_dir_all(&a).unwrap();
        std::fs::create_dir_all(&b).unwrap();
        assert!(!paths_collide(&a, &b).unwrap());
        let _ = std::fs::remove_dir_all(&a);
        let _ = std::fs::remove_dir_all(&b);
    }

    #[test]
    fn paths_collide_works_when_output_does_not_exist() {
        let parent = unique_tmp("collide_nonexistent_parent");
        std::fs::create_dir_all(&parent).unwrap();
        let nonexistent = parent.join("not_yet_created");
        assert!(!paths_collide(&parent, &nonexistent).unwrap());
        // Same path under absolute resolution should still collide.
        assert!(paths_collide(&nonexistent, &nonexistent).unwrap());
        let _ = std::fs::remove_dir_all(&parent);
    }

    #[test]
    fn staging_dir_for_uses_sibling_under_same_parent() {
        let dir = unique_tmp("staging");
        let staging = staging_dir_for(&dir);
        assert_eq!(staging.parent(), dir.parent());
        let staging_name = staging.file_name().unwrap().to_string_lossy();
        let dir_name = dir.file_name().unwrap().to_string_lossy();
        assert!(
            staging_name.contains(dir_name.as_ref()),
            "staging name {staging_name:?} should mention {dir_name:?}"
        );
        assert!(staging_name.contains(".tmp."));
    }

    #[test]
    fn run_atomic_compile_promotes_staging_on_success() {
        let source = unique_tmp("atomic_ok_src");
        let final_dir = unique_tmp("atomic_ok_dst");
        std::fs::create_dir_all(&source).unwrap();

        let result = run_atomic_compile(&final_dir, &source, |staging| {
            assert!(staging.exists(), "work runs against existing staging dir");
            std::fs::write(staging.join("a.bin"), b"hello").unwrap();
            Ok(vec!["wrote a.bin".into()])
        });

        assert!(result.is_ok());
        assert!(final_dir.join("a.bin").exists(), "file promoted");
        assert!(!staging_dir_for(&final_dir).exists(), "staging removed");

        let _ = std::fs::remove_dir_all(&source);
        let _ = std::fs::remove_dir_all(&final_dir);
    }

    #[test]
    fn run_atomic_compile_cleans_up_staging_on_error() {
        let source = unique_tmp("atomic_err_src");
        let final_dir = unique_tmp("atomic_err_dst");
        std::fs::create_dir_all(&source).unwrap();

        let result = run_atomic_compile(&final_dir, &source, |staging| {
            std::fs::write(staging.join("partial.bin"), b"oops").unwrap();
            Err::<Vec<String>, _>(LqlError::Execution("simulated".into()))
        });

        assert!(result.is_err());
        assert!(!staging_dir_for(&final_dir).exists(), "staging removed");
        assert!(
            !final_dir.exists(),
            "no partial output at final destination"
        );

        let _ = std::fs::remove_dir_all(&source);
    }

    #[test]
    fn run_atomic_compile_rejects_output_equal_to_source() {
        let source = unique_tmp("atomic_same_path");
        std::fs::create_dir_all(&source).unwrap();

        let result =
            run_atomic_compile(&source, &source, |_staging| Ok(vec!["unreachable".into()]));

        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("resolves to the source vindex"),
            "unexpected error: {err}"
        );
        // Source must not have been touched.
        assert!(source.exists());

        let _ = std::fs::remove_dir_all(&source);
    }

    #[test]
    fn staging_dir_for_falls_back_when_basename_missing() {
        // Root path has no file_name — fallback to the literal "compile"
        // so we still produce a deterministic staging name.
        let staging = staging_dir_for(Path::new("/"));
        let name = staging.file_name().unwrap().to_string_lossy().to_string();
        assert!(name.contains("compile"));
        assert!(name.contains(".tmp."));
    }

    #[test]
    fn staging_dir_for_handles_relative_path_without_parent() {
        // `Path::new("foo").parent()` returns `Some("")`; we still
        // produce a usable staging dir by joining a sentinel name.
        let staging = staging_dir_for(Path::new("foo"));
        let name = staging.file_name().unwrap().to_string_lossy();
        assert!(name.contains("foo"));
        assert!(name.contains(".tmp."));
    }

    #[test]
    fn run_atomic_compile_propagates_work_error() {
        // The error variant should bubble unchanged (we don't swap it out
        // with a generic "compile failed" wrapper).
        let source = unique_tmp("atomic_propagates_src");
        let final_dir = unique_tmp("atomic_propagates_dst");
        std::fs::create_dir_all(&source).unwrap();

        let err = run_atomic_compile(&final_dir, &source, |_staging| {
            Err::<Vec<String>, _>(LqlError::Execution(
                "specific failure that callers will recognise".into(),
            ))
        })
        .unwrap_err();

        assert!(err.to_string().contains("specific failure"));

        let _ = std::fs::remove_dir_all(&source);
    }

    #[test]
    fn paths_collide_canonicalises_dot_segments() {
        // `/tmp/foo/./.` and `/tmp/foo` should collide once both
        // canonicalise via `std::path::absolute`.
        let dir = unique_tmp("paths_collide_dot");
        std::fs::create_dir_all(&dir).unwrap();
        let dotted = dir.join(".").join(".");
        assert!(paths_collide(&dir, &dotted).unwrap());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_atomic_compile_errors_when_staging_create_fails() {
        // Force `create_dir_all(staging)` to fail by parking the staging
        // dir under a path whose parent is a regular file. The failure
        // message must mention "staging dir" so the operator can tell
        // setup blew up rather than the bake itself.
        let source = unique_tmp("atomic_staging_fail_src");
        std::fs::create_dir_all(&source).unwrap();

        let parent_as_file = unique_tmp("atomic_parent_is_file");
        std::fs::write(&parent_as_file, b"definitely not a directory").unwrap();
        let final_dir = parent_as_file.join("never_created");

        let err = run_atomic_compile(&final_dir, &source, |_staging| {
            unreachable!("work should not run when staging cannot be created")
        })
        .unwrap_err();
        assert!(
            err.to_string().contains("staging dir"),
            "unexpected error: {err}"
        );

        let _ = std::fs::remove_file(&parent_as_file);
        let _ = std::fs::remove_dir_all(&source);
    }

    #[test]
    fn run_atomic_compile_replaces_existing_final_dir() {
        let source = unique_tmp("atomic_replace_src");
        let final_dir = unique_tmp("atomic_replace_dst");
        std::fs::create_dir_all(&source).unwrap();
        std::fs::create_dir_all(&final_dir).unwrap();
        std::fs::write(final_dir.join("stale.bin"), b"old").unwrap();

        let result = run_atomic_compile(&final_dir, &source, |staging| {
            std::fs::write(staging.join("fresh.bin"), b"new").unwrap();
            Ok(vec!["ok".into()])
        });

        assert!(result.is_ok());
        assert!(final_dir.join("fresh.bin").exists());
        assert!(
            !final_dir.join("stale.bin").exists(),
            "old contents removed"
        );

        let _ = std::fs::remove_dir_all(&source);
        let _ = std::fs::remove_dir_all(&final_dir);
    }
}
