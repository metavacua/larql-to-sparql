//! `larql publish <SRC> --repo OWNER/NAME` — upload a vindex to HuggingFace,
//! optionally carving + uploading deployment slices to sibling repos in one
//! go.
//!
//! The default (`--all`) produces four repos from a single source vindex:
//!
//!   * `OWNER/NAME`         — the full vindex (INFER + DESCRIBE)
//!   * `OWNER/NAME-client`  — attention-only slice (pair with `run --ffn URL`)
//!   * `OWNER/NAME-server`  — FFN-only slice (pair with `serve --ffn-only`)
//!   * `OWNER/NAME-browse`  — gate + embed + down_meta (DESCRIBE/WALK only)
//!
//! The `router` preset is opt-in via `--slices` because dense vindexes don't
//! carry `router_weights.bin` and the resulting repo would be empty.
//!
//! Under the covers this is `larql slice` + `larql hf publish` bundled: each
//! slice is staged in a temp directory, uploaded to its sibling repo via
//! `larql_vindex::publish_vindex`, and then cleaned up.
//!
//! Requires `HF_TOKEN` (or `~/.huggingface/token`) just like `larql hf publish`.
//!
//! Module split (one concept per file, none over the 800-line cap):
//! - `mod.rs`        — CLI args, orchestration (`run`), the VINDEX3-safe-defaults
//!                      decision.
//! - `collections`   — collection/title derivation + the HF collection step.
//! - `upload`        — slice-preset resolution, the per-step upload plan,
//!                      `UploadStep`/`StepOutcome`.
//! - `progress`       — the indicatif progress-bar `PublishCallbacks` impl.

mod collections;
mod progress;
mod upload;

use larql_vindex::format::filenames::*;
use std::path::PathBuf;

use clap::Args;

use crate::commands::primary::cache;
use collections::{build_collections, default_family, default_model_title, namespace_of};
use upload::{execute_step, resolve_slice_list, StepOutcome, UploadStep};

/// The `--collections` default, as one comma-joined literal — the CLI
/// attribute below and [`collections_match_default`]'s "was this the
/// default, or an explicit request for the same three levels" check both
/// read this one constant (the latter via `.split(',')`) rather than
/// each spelling the list out separately, so the two can't drift apart.
pub(super) const DEFAULT_COLLECTIONS: &str = "model,family,library";

#[derive(Args)]
pub struct PublishArgs {
    /// Source vindex: directory, `hf://owner/name`, `owner/name`, or cache shorthand.
    pub source: String,

    /// HuggingFace repo ID for the full vindex (e.g. `chrishayuk/gemma-4-31b`).
    /// Sibling slice repos are named `<repo>-<preset>` by default.
    #[arg(long)]
    pub repo: String,

    /// Publish the full vindex to `--repo`. On by default; pair with
    /// `--no-full --slices client,server` to publish only the slices.
    #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
    pub full: bool,

    /// Shortcut: `--no-full` is the same as `--full false`.
    #[arg(long, conflicts_with = "full")]
    pub no_full: bool,

    /// Comma-separated slice presets to publish alongside the full vindex.
    /// Defaults to `client,attn,embed,server,browse` — covers both the
    /// 2-tier and 3-tier (ADR-0008) topologies in one run. Pass `none`
    /// to skip all slice uploads.
    #[arg(long, value_delimiter = ',')]
    pub slices: Vec<String>,

    /// Suffix template for sibling slice repos. `{repo}` is replaced with
    /// `--repo`; `{preset}` with the preset name. Default: `{repo}-{preset}`.
    #[arg(long, default_value = "{repo}-{preset}")]
    pub slice_repo_template: String,

    /// Directory to stage intermediate slices. Defaults to the system temp
    /// dir; each slice gets its own subdir and is cleaned up on success.
    #[arg(long)]
    pub tmp_dir: Option<PathBuf>,

    /// Preview the upload plan without creating repos or uploading files.
    #[arg(long)]
    pub dry_run: bool,

    /// Collection levels to create or update after the uploads land.
    /// Comma list of: `model` (per-model-size), `family` (per-architecture),
    /// `library` (one top-level "LARQL Vindex Library"). Default is all
    /// three. Pass `none` to skip collection creation entirely.
    #[arg(long, value_delimiter = ',', default_value = DEFAULT_COLLECTIONS)]
    pub collections: Vec<String>,

    /// Override the model title used in the per-model collection. Default
    /// is derived from the vindex config (e.g. `Gemma 4 31B`).
    #[arg(long)]
    pub model_title: Option<String>,

    /// Override the family name used in the family-level collection
    /// (e.g. `Gemma`). Default: prefix of the model id up to the first
    /// version/size token.
    #[arg(long)]
    pub family: Option<String>,

    /// Title for the library-level collection. Default matches the one
    /// in docs: "LARQL Vindex Library". Override if you want a namespaced
    /// variant.
    #[arg(long, default_value = "LARQL Vindex Library")]
    pub library_title: String,

    /// Force re-upload of every file even if the remote copy already
    /// matches the local SHA256. By default `publish` fetches the remote
    /// LFS file index and skips any file whose `lfs.oid` equals the
    /// local SHA256, which saves a full re-upload when nothing changed.
    ///
    /// Use this flag to bypass the skip and re-upload everything, e.g.
    /// if you suspect a prior upload was truncated.
    #[arg(long)]
    pub force_upload: bool,

    /// Keep remote files that no longer exist in the source vindex.
    ///
    /// By default `publish` mirrors the source: after uploading, any file
    /// on the Hub that the vindex no longer contains is deleted (except
    /// `README.md` and dot-files, which are authored separately). This
    /// matters when a weight file is renamed — leaving both generations in
    /// the repo lets the loader pick the stale one by name, which is how
    /// `gemma-3-4b-it-vindex` ended up serving a pre-ggml-layout weight
    /// after a republish on 2026-08-07.
    ///
    /// Pass this to keep extra files, e.g. when a repo intentionally
    /// carries hand-added assets alongside the vindex.
    #[arg(long)]
    pub no_prune: bool,

    /// HuggingFace repo type: `model` (default) or `dataset`.
    #[arg(long, default_value = "model")]
    pub repo_type: String,

    /// Create every repo (full + slices) private. Pair with `larql hf
    /// visibility --public` once verification passes — Vindex Factory's
    /// two-phase publish (docs/vindex-factory.md §8: "nothing goes
    /// public unverified"). Repos already on the Hub keep their
    /// existing visibility; this only affects repo *creation*.
    #[arg(long)]
    pub private: bool,
}

/// Whether `collections` is exactly [`DEFAULT_COLLECTIONS`] (case-insensitive,
/// order-sensitive — matches how clap parsed the flag). Used to tell
/// "omitted" from "explicitly requested the same three levels" as best
/// this CLI can (it can't, fully: clap bakes the default into the value
/// before this code ever sees it — see the V3-defaults comment in
/// [`run`]).
pub(super) fn collections_match_default(collections: &[String]) -> bool {
    let default = DEFAULT_COLLECTIONS.split(',');
    collections.len() == default.clone().count()
        && collections
            .iter()
            .zip(default)
            .all(|(a, b)| a.eq_ignore_ascii_case(b))
}

pub fn run(args: PublishArgs) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Resolve source.
    let src = cache::resolve_model(&args.source)?;
    if !src.is_dir() {
        return Err(format!("source vindex not a directory: {}", src.display()).into());
    }
    if !src.join(INDEX_JSON).exists() {
        return Err(format!("source vindex missing index.json: {}", src.display()).into());
    }

    // VINDEX3 has no slicing (`slice` refuses it outright) and no
    // config-derived collection titles (`load_vindex_config` is
    // VINDEX2-only) — the DEFAULT `--slices`/`--collections` used to
    // crash on a VINDEX3 source with no VINDEX3-specific guidance
    // toward the workaround (`docs/vindex3-registry-publishing-design.md`
    // §1). Detect once, up front, and only downgrade *defaults*: an
    // explicit, non-default `--slices` request still hits the same
    // refusal it always did — it asked for something VINDEX3 can't do
    // yet, which is a real answer, not a trap. `--collections` can't
    // make that same distinction (clap's `default_value` means an
    // explicit request for the default three levels is indistinguishable
    // from omitting the flag) — treated the same way regardless, which
    // is still strictly better than crashing either way.
    let is_v3 = matches!(
        larql_vindex::format::generation::detect_generation(&src)?,
        larql_vindex::format::generation::ContainerGeneration::V3
    );
    let slices_omitted = args.slices.is_empty();
    let collections_are_default = collections_match_default(&args.collections);
    if is_v3 && slices_omitted {
        println!(
            "VINDEX3 source: skipping the default slice presets (VINDEX3 containers can't be \
             sliced yet — pass --slices explicitly to see the refusal)."
        );
    }
    if is_v3 && collections_are_default {
        println!(
            "VINDEX3 source: skipping the default collection levels (needs a VINDEX3-aware \
             title deriver — pass --collections explicitly to see the refusal)."
        );
    }

    let publish_full = args.full && !args.no_full;
    let requested_slices = if is_v3 && slices_omitted {
        Vec::new()
    } else {
        resolve_slice_list(&args.slices)?
    };
    if !publish_full && requested_slices.is_empty() {
        return Err(
            "nothing to publish: `--no-full` requires at least one preset in `--slices`".into(),
        );
    }

    // 2. Build the upload plan.
    let mut plan: Vec<UploadStep> = Vec::new();
    if publish_full {
        plan.push(UploadStep {
            label: "full".into(),
            repo: args.repo.clone(),
            preset: None,
            staging: None,
        });
    }
    let staging_root = args.tmp_dir.clone().unwrap_or_else(std::env::temp_dir);
    for preset in &requested_slices {
        let repo = args
            .slice_repo_template
            .replace("{repo}", &args.repo)
            .replace("{preset}", preset);
        // Unique subdir per (pid, preset) so parallel invocations don't collide.
        let staging = staging_root.join(format!(
            "larql-publish-{}-{}-{}.vindex",
            args.repo.replace('/', "_"),
            preset,
            std::process::id()
        ));
        plan.push(UploadStep {
            label: preset.clone(),
            repo,
            preset: Some(preset.clone()),
            staging: Some(staging),
        });
    }

    // 3. Print the plan.
    println!("Source:    {}", src.display());
    println!("Upload plan ({} step(s)):", plan.len());
    for step in &plan {
        match &step.preset {
            None => println!("  full    → {}", step.repo),
            Some(p) => println!("  {p:<7} → {}", step.repo),
        }
    }
    let collection_levels = if is_v3 && collections_are_default {
        Vec::new()
    } else {
        collections::resolve_collection_list(&args.collections)?
    };
    if !collection_levels.is_empty() {
        let cfg = larql_vindex::load_vindex_config(&src)?;
        let model_title = args
            .model_title
            .clone()
            .unwrap_or_else(|| format!("{} — LARQL Vindex", default_model_title(&cfg.model)));
        let family = args
            .family
            .clone()
            .unwrap_or_else(|| default_family(&cfg.model));
        println!("Collections:");
        for level in &collection_levels {
            let title = match level.as_str() {
                "model" => model_title.clone(),
                "family" => format!("{family} Family — LARQL Vindexes"),
                "library" => args.library_title.clone(),
                _ => continue,
            };
            let namespace = namespace_of(&args.repo)?;
            println!("  {level:<8} {namespace}: {title}");
        }
    }
    if args.dry_run {
        println!("\n(dry run — no repos created, no files uploaded)");
        return Ok(());
    }

    // 4. Execute each step.
    let mut results: Vec<StepOutcome> = Vec::new();
    for step in plan {
        let result = execute_step(
            &src,
            &step,
            args.force_upload,
            &args.repo_type,
            args.private,
            !args.no_prune,
        )?;
        results.push(StepOutcome {
            label: step.label,
            repo: step.repo,
            url: result.url,
            revision: result.revision,
        });
    }

    // 5. Collection step — group the uploaded repos into HF collections.
    // `collection_levels` is the same V3-aware list computed for the
    // plan preview above (§ VINDEX3-safe defaults) — not recomputed,
    // so the preview and what actually runs can never disagree.
    let collection_urls = if !collection_levels.is_empty() {
        Some(build_collections(
            &src,
            &args,
            &results,
            &collection_levels,
        )?)
    } else {
        None
    };

    // 6. Summary.
    println!("\nPublished:");
    for r in &results {
        println!("  {:<8} {} → {} (@ {})", r.label, r.repo, r.url, r.revision);
    }
    if let Some(urls) = collection_urls {
        println!("\nCollections:");
        for (level, url) in &urls {
            println!("  {level:<8} {url}");
        }
    }
    println!("\nPull any of these with:");
    for r in &results {
        println!("  larql pull hf://{}@{}", r.repo, r.revision);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collections_match_default_accepts_the_default_case_insensitively() {
        let raw: Vec<String> = DEFAULT_COLLECTIONS
            .split(',')
            .map(|s| s.to_ascii_uppercase())
            .collect();
        assert!(collections_match_default(&raw));
    }

    #[test]
    fn collections_match_default_rejects_a_different_set() {
        assert!(!collections_match_default(&["library".to_string()]));
    }

    #[test]
    fn collections_match_default_rejects_a_reordering() {
        // Order-sensitive: matches how `resolve_collection_list` treats
        // the parsed flag value, not a set comparison.
        assert!(!collections_match_default(&[
            "library".to_string(),
            "family".to_string(),
            "model".to_string(),
        ]));
    }

    // ── VINDEX3-safe defaults (docs/vindex3-registry-publishing-design.md §1) ──
    //
    // `--dry-run` exercises exactly the code path that used to crash on a
    // VINDEX3 source (plan building + the collections preview) without
    // ever reaching `execute_step`, so these need no HF mocking at all —
    // a real fixture, resolved as a local directory (`cache::resolve_model`'s
    // own "already a directory" branch), is enough.

    fn dry_run_args(source: String, slices: Vec<String>, collections: Vec<String>) -> PublishArgs {
        PublishArgs {
            source,
            repo: "org/repo".to_string(),
            full: true,
            no_full: false,
            slices,
            slice_repo_template: "{repo}-{preset}".to_string(),
            tmp_dir: None,
            dry_run: true,
            collections,
            model_title: None,
            family: None,
            library_title: "LARQL Vindex Library".to_string(),
            force_upload: false,
            repo_type: "model".to_string(),
            private: false,
            no_prune: false,
        }
    }

    fn real_v3_fixture() -> tempfile::TempDir {
        let checkpoint = tempfile::tempdir().unwrap();
        let container = tempfile::tempdir().unwrap();
        larql_vindex::format::vindex3::fixtures::encode_fixture_container(
            larql_vindex::format::vindex3::fixtures::miniature_glimmer,
            checkpoint.path(),
            container.path(),
            "publish-fixture",
        );
        container
    }

    #[test]
    fn v3_source_with_default_slices_and_collections_does_not_crash() {
        // The defect this pins: bare `larql publish <v3-dir> --repo ...`
        // used to error out of `load_vindex_config` (collections preview)
        // or `slice_vindex` (default slices) — a VINDEX3 source needed
        // undocumented flags just to not crash.
        let fixture = real_v3_fixture();
        let args = dry_run_args(
            fixture.path().to_string_lossy().into_owned(),
            Vec::new(),
            vec!["model".into(), "family".into(), "library".into()],
        );
        run(args).expect("a VINDEX3 source with default slices/collections must not crash");
    }

    #[test]
    fn v2_source_with_default_slices_and_collections_is_unchanged() {
        // Regression guard: the VINDEX3 defaults fix must not touch V2's
        // existing behaviour — a bare-name VINDEX2 source with no model
        // config still can't render a collections preview (needs a real
        // `larql extract`-shaped index.json this minimal fixture doesn't
        // have), so this still fails, but via `load_vindex_config`'s
        // *existing* error, not a new one this fix introduced.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("index.json"), r#"{"version":2}"#).unwrap();
        let args = dry_run_args(
            dir.path().to_string_lossy().into_owned(),
            Vec::new(),
            vec!["model".into(), "family".into(), "library".into()],
        );
        let err = run(args).unwrap_err();
        // Not the VINDEX3-defaults skip message — this source is V2, so
        // the collections preview still runs and fails the same way it
        // always did (a config field this minimal fixture never sets).
        assert!(!err.to_string().contains("VINDEX3"), "{err}");
    }

    #[test]
    fn v3_source_with_explicit_nondefault_collections_still_refuses() {
        // The other half of "only downgrade defaults": an explicit,
        // non-default `--collections` request on a VINDEX3 source must
        // still hit today's real refusal (no VINDEX3-aware title
        // deriver exists yet) — silently downgrading an explicit ask
        // would hide that VINDEX3 collections don't work yet, not fix it.
        let fixture = real_v3_fixture();
        let args = dry_run_args(
            fixture.path().to_string_lossy().into_owned(),
            Vec::new(),
            vec!["library".into()],
        );
        run(args).expect_err("an explicit, non-default --collections must still refuse on V3");
    }
}
