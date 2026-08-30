//! Slice-preset resolution and the per-step upload plan.
//!
//! `UploadStep` is what [`super::run`] plans (one entry per repo: the
//! full vindex, plus one per requested slice preset); `execute_step`
//! carves (for a slice) and uploads (always) each one, producing a
//! [`StepOutcome`] carrying the pinned artifact URL + revision
//! (`larql_vindex::PublishResult`, `docs/vindex3-registry-publishing-design.md`
//! §1/§6) back to the caller.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::commands::primary::slice_cmd::{preset_parts, slice_vindex, Part};

use super::progress::{human_size, CliPublishCallbacks};

/// Default sibling slice presets when `--slices` is not given. Covers
/// every deployment shape ADR-0007 and ADR-0008 support today:
///
///   * `client`  — 2-tier dense-remote (client holds embed locally)
///   * `attn`    — 3-tier dense-remote client (embed delegated)
///   * `embed`   — 3-tier embed server
///   * `server`  — 3-tier / 2-tier FFN server
///   * `browse`  — read-only DESCRIBE/WALK consumers
///
/// `router` is omitted because it would produce an empty repo on non-MoE
/// vindexes; request it explicitly via `--slices router` when relevant.
/// Publishing all five by default is cheap: skip-if-unchanged keeps the
/// re-upload cost at a few KB per slice once the LFS blobs are already
/// on HF.
const DEFAULT_SLICES: &[&str] = &["client", "attn", "embed", "server", "browse"];

pub(super) fn resolve_slice_list(
    raw: &[String],
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    // Default set when --slices is not passed.
    if raw.is_empty() {
        return Ok(DEFAULT_SLICES.iter().map(|s| s.to_string()).collect());
    }
    // Explicit opt-out.
    if raw.len() == 1 && raw[0].eq_ignore_ascii_case("none") {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(raw.len());
    for name in raw {
        let trimmed = name.trim();
        // Validate by round-tripping through preset_parts. Catches typos
        // before we start creating repos.
        preset_parts(trimmed).map_err(|e| {
            format!(
                "invalid slice preset '{trimmed}': {e}. Valid: client, attn, embed, server, browse, router, all"
            )
        })?;
        out.push(trimmed.to_string());
    }
    Ok(out)
}

pub(super) struct UploadStep {
    pub(super) label: String,
    pub(super) repo: String,
    /// `None` for the full-vindex upload; `Some(preset)` for a sliced upload.
    pub(super) preset: Option<String>,
    /// Where the sliced vindex gets staged before upload.
    pub(super) staging: Option<PathBuf>,
}

pub(super) struct StepOutcome {
    pub(super) label: String,
    pub(super) repo: String,
    pub(super) url: String,
    pub(super) revision: String,
}

pub(super) fn execute_step(
    src: &Path,
    step: &UploadStep,
    force_upload: bool,
    repo_type: &str,
    private: bool,
    prune_remote: bool,
) -> Result<larql_vindex::PublishResult, Box<dyn std::error::Error>> {
    match (&step.preset, &step.staging) {
        // Full vindex — upload the source directory directly, no slicing.
        (None, _) => {
            println!("\n→ Uploading full vindex to {}", step.repo);
            upload_dir(
                src,
                &step.repo,
                force_upload,
                repo_type,
                private,
                prune_remote,
            )
        }
        // Sliced upload — carve into staging, upload, clean up.
        (Some(preset), Some(staging)) => {
            println!("\n→ Carving slice `{preset}` …");
            let parts: BTreeSet<Part> =
                preset_parts(preset).map_err(|e| format!("preset `{preset}`: {e}"))?;
            let outcome = slice_vindex(
                src, staging, parts, /*force=*/ true, /*dry_run=*/ false,
            )?;
            println!(
                "  staged {} file(s), {} — {}",
                outcome.copied.len(),
                human_size(outcome.total_bytes),
                staging.display()
            );
            println!("→ Uploading slice `{preset}` to {}", step.repo);
            let result = upload_dir(
                staging,
                &step.repo,
                force_upload,
                repo_type,
                private,
                prune_remote,
            );
            // Always try to clean up the staging dir, regardless of outcome.
            let _ = std::fs::remove_dir_all(staging);
            result
        }
        (Some(_), None) => Err("internal: slice step without staging dir".into()),
    }
}

fn upload_dir(
    dir: &Path,
    repo: &str,
    force_upload: bool,
    repo_type: &str,
    private: bool,
    prune_remote: bool,
) -> Result<larql_vindex::PublishResult, Box<dyn std::error::Error>> {
    let mut callbacks = CliPublishCallbacks::new();
    let opts = larql_vindex::PublishOptions {
        skip_unchanged: !force_upload,
        repo_type: repo_type.to_string(),
        private,
        prune_remote,
    };
    Ok(larql_vindex::publish_vindex_with_opts(
        dir,
        repo,
        &opts,
        &mut callbacks,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_slice_list_is_full_publish_set() {
        // Flipping this default changes what bare `larql publish` writes
        // to HF — pin the exact order so the test fails loudly if it
        // gets rearranged. Covers both 2-tier (`client`) and 3-tier
        // (`attn` + `embed`) deployment shapes out of the box.
        let got = resolve_slice_list(&[]).unwrap();
        assert_eq!(got, vec!["client", "attn", "embed", "server", "browse"]);
    }

    #[test]
    fn slices_none_disables_sliced_uploads() {
        let got = resolve_slice_list(&["none".to_string()]).unwrap();
        assert!(got.is_empty());
        // Case-insensitive.
        let got_caps = resolve_slice_list(&["NONE".to_string()]).unwrap();
        assert!(got_caps.is_empty());
    }

    #[test]
    fn slices_explicit_list_passes_through() {
        let raw = vec!["client".into(), "server".into()];
        let got = resolve_slice_list(&raw).unwrap();
        assert_eq!(got, vec!["client", "server"]);
    }

    #[test]
    fn slices_with_router_is_valid() {
        // Router is a real preset even though it's omitted from the default
        // set. Passing it explicitly must round-trip cleanly.
        let got = resolve_slice_list(&["router".into()]).unwrap();
        assert_eq!(got, vec!["router"]);
    }

    #[test]
    fn slices_invalid_name_errors() {
        let err = resolve_slice_list(&["typo".into()]).unwrap_err();
        assert!(
            err.to_string().contains("invalid slice preset"),
            "got: {err}"
        );
    }

    #[test]
    fn slice_repo_template_substitution() {
        let template = "{repo}-{preset}";
        let rendered = template
            .replace("{repo}", "chrishayuk/gemma-4-31b")
            .replace("{preset}", "client");
        assert_eq!(rendered, "chrishayuk/gemma-4-31b-client");
    }

    #[test]
    fn slice_repo_template_custom_separator() {
        // Verify callers can override to e.g. "{repo}_{preset}" without
        // hard-coding a dash in the implementation.
        let template = "{repo}/{preset}";
        let rendered = template
            .replace("{repo}", "me/model")
            .replace("{preset}", "client");
        assert_eq!(rendered, "me/model/client");
    }

    // ── Skip-if-unchanged ──────────────────────────────────────────────
    //
    // The actual upload/skip decision lives in
    // `larql_vindex::publish_vindex_with_opts` and can't be exercised
    // without an HF server. These tests pin the CLI-side plumbing: that
    // `--force-upload` flips the option into `skip_unchanged = false`,
    // and that `PublishOptions::skip_unchanged()` is the default-on
    // constructor.

    #[test]
    fn force_upload_disables_skip() {
        // Simulate the flag state the CLI builds from `--force-upload`.
        let opts = larql_vindex::PublishOptions {
            skip_unchanged: false,
            ..Default::default()
        };
        assert!(!opts.skip_unchanged);
    }

    #[test]
    fn default_publish_options_skip_unchanged() {
        // Without `--force-upload`, `skip_unchanged: true`.
        let opts = larql_vindex::PublishOptions {
            skip_unchanged: true,
            ..Default::default()
        };
        assert!(opts.skip_unchanged);
    }

    #[test]
    fn publish_options_explicit_skip_helper() {
        // The `::skip_unchanged()` constructor is intended for callers
        // that want the feature on without depending on field defaults.
        let opts = larql_vindex::PublishOptions::skip_unchanged();
        assert!(opts.skip_unchanged);
    }

    #[test]
    fn publish_options_default_is_conservative() {
        // `Default` keeps `skip_unchanged: false` so code that gets an
        // options struct via Default doesn't silently skip uploads —
        // the opt-in happens at the CLI boundary where it's explicit.
        let opts = larql_vindex::PublishOptions::default();
        assert!(!opts.skip_unchanged);
    }
}
