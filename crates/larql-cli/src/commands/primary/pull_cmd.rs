//! `larql pull` — download a vindex (or a slice, or a whole collection)
//! and cache it locally, with ollama-style progress bars and free resume.
//!
//! Four resolution paths, in order of specificity:
//!
//!   1. `pull <repo>`                   — plain pull, one repo
//!   2. `pull <repo> --preset client`   — pull the `-client` sibling instead
//!   3. `pull <repo> --all-slices`      — pull full + default slice siblings
//!   4. `pull --collection <slug|url>`  — pull every dataset in a collection
//!
//! After a single-repo pull, `pull` probes HF for the standard sibling
//! suffixes and prints a hint if any exist — so the slice convention is
//! self-announcing. A user landing on `chrishayuk/gemma-4-31b-it-vindex`
//! discovers `-client` / `-server` / `-browse` without having to read a
//! README.
//!
//! Progress + resume: `indicatif::MultiProgress` gives one bar per file;
//! hf-hub 0.5 handles `.incomplete` partial-file resume internally so an
//! interrupted pull picks up where it left off on the next run.

use std::path::PathBuf;

use clap::Args;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

/// Default sibling presets to probe / pull when the caller doesn't pass
/// `--preset`. Matches the `publish` default set; the symmetry matters
/// so `publish` and `pull` stay in lock-step.
const DEFAULT_SIBLING_PRESETS: &[&str] = &["client", "attn", "embed", "server", "browse"];

/// Same sibling-naming template as `publish` so `pull` can reverse what
/// `publish` produced without a separate configuration handshake.
const DEFAULT_SIBLING_TEMPLATE: &str = "{repo}-{preset}";

#[derive(Args)]
pub struct PullArgs {
    /// `hf://owner/name[@rev]`, `owner/name`, or a local path. Omit when
    /// passing `--collection`.
    pub model: Option<String>,

    /// Pull a sibling slice instead of the named repo. Options: `client`,
    /// `server`, `browse`, `router`, `all`. Resolves via the
    /// `--sibling-template` (default `{repo}-{preset}`).
    #[arg(long)]
    pub preset: Option<String>,

    /// Pull the full vindex *and* every default slice sibling
    /// (`-client`, `-attn`, `-embed`, `-server`, `-browse`) in one
    /// command. Missing siblings are warned-about, not fatal.
    #[arg(long)]
    pub all_slices: bool,

    /// Pull every dataset item in an HF collection. Accepts the slug
    /// (`namespace/slug-id`) or the full
    /// `https://huggingface.co/collections/…` URL. Mutually exclusive
    /// with `<model>`.
    #[arg(long)]
    pub collection: Option<String>,

    /// Override the sibling-resolution template. `{repo}` and `{preset}`
    /// substitute. Must match whatever `larql publish` wrote — defaults
    /// align, override only if you changed `publish --slice-repo-template`.
    #[arg(long, default_value = DEFAULT_SIBLING_TEMPLATE)]
    pub sibling_template: String,
}

pub fn run(args: PullArgs) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(ref slug_or_url) = args.collection {
        return pull_collection(slug_or_url);
    }

    let model = args
        .model
        .as_deref()
        .ok_or_else(|| "pull needs <model> or --collection".to_string())?;

    if args.all_slices {
        return pull_all_slices(model, &args.sibling_template);
    }

    if let Some(ref preset) = args.preset {
        let sibling = render_sibling_repo(model, preset, &args.sibling_template)?;
        eprintln!("Resolving --preset {preset} → {sibling}");
        return pull_one(&sibling, /*print_siblings=*/ false);
    }

    pull_one(model, /*print_siblings=*/ true)
}

/// HuggingFace repos look like `owner/name` — exactly one `/`, neither
/// side empty, no leading `/`, no dot in the owner segment. Used by both
/// `render_sibling_repo` and `normalise_hf_path` so filesystem paths
/// never get confused for HF refs.
fn looks_like_hf_repo(s: &str) -> bool {
    if s.starts_with('/') {
        return false;
    }
    let mut parts = s.splitn(2, '/');
    let owner = parts.next().unwrap_or("");
    let name = parts.next().unwrap_or("");
    !owner.is_empty()
        && !name.is_empty()
        && !owner.contains('.')
        && !name.contains('/')
}

/// Render `{repo}-{preset}` (or the caller's override). Strips any
/// existing `hf://` prefix so the template operates on bare `owner/name`.
fn render_sibling_repo(
    model: &str,
    preset: &str,
    template: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let bare = model.trim_start_matches("hf://");
    if !looks_like_hf_repo(bare) {
        return Err(format!(
            "--preset needs an `owner/name` repo, not a local path: {model}"
        )
        .into());
    }
    Ok(template
        .replace("{repo}", bare)
        .replace("{preset}", preset))
}

/// `indicatif::ProgressBar` wrapper that implements hf-hub's `Progress`
/// trait. We can't use hf-hub's built-in `impl Progress for ProgressBar`
/// directly because hf-hub 0.5 pins indicatif 0.18 while the workspace
/// is on 0.17 — different types.
struct BarProgress(ProgressBar);

impl larql_vindex::DownloadProgress for BarProgress {
    fn init(&mut self, size: usize, filename: &str) {
        self.0.set_length(size as u64);
        self.0.set_style(
            ProgressStyle::with_template(
                "  {msg:28} [{elapsed_precise}] [{wide_bar:.cyan/blue}] \
                 {bytes:>10}/{total_bytes:<10} {bytes_per_sec:>10} ({eta})",
            )
            .unwrap()
            .progress_chars("#>-"),
        );
        let msg = if filename.len() > 28 {
            format!("…{}", &filename[filename.len() - 27..])
        } else {
            filename.to_string()
        };
        self.0.set_message(msg);
    }
    fn update(&mut self, size: usize) {
        self.0.inc(size as u64);
    }
    fn finish(&mut self) {
        self.0.finish();
    }
}

fn download_with_indicatif(hf_path: &str) -> Result<PathBuf, larql_vindex::VindexError> {
    let mp = MultiProgress::new();
    larql_vindex::resolve_hf_vindex_with_progress(hf_path, |_filename| {
        BarProgress(mp.add(ProgressBar::new(0)))
    })
}

/// Resolve + download a single repo, then optionally probe for siblings.
fn pull_one(model: &str, print_siblings: bool) -> Result<(), Box<dyn std::error::Error>> {
    let hf_path = normalise_hf_path(model)?;
    eprintln!("Pulling {hf_path}...");
    let cached: PathBuf = download_with_indicatif(&hf_path)?;
    eprintln!("Cached at: {}", cached.display());

    if let Ok(cfg) = larql_vindex::load_vindex_config(&cached) {
        eprintln!(
            "  {} layers, hidden_size={}, dtype={:?}, level={}",
            cfg.num_layers, cfg.hidden_size, cfg.dtype, cfg.extract_level
        );
    }

    if print_siblings {
        hint_siblings(model);
    }
    Ok(())
}

/// Pull every dataset item in an HF collection. A single-item failure
/// logs a warning but doesn't abort — one unavailable sibling shouldn't
/// fail the whole collection pull.
fn pull_collection(slug_or_url: &str) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Fetching collection: {slug_or_url}");
    let items = larql_vindex::fetch_collection_items(slug_or_url)?;
    let datasets: Vec<String> = items
        .into_iter()
        .filter(|(kind, _)| kind == "dataset")
        .map(|(_, id)| id)
        .collect();
    if datasets.is_empty() {
        eprintln!("  (no dataset items in collection)");
        return Ok(());
    }
    eprintln!("  Found {} dataset repo(s):", datasets.len());
    for id in &datasets {
        eprintln!("    {id}");
    }

    let mut ok = 0usize;
    let mut failed: Vec<(String, String)> = Vec::new();
    for id in datasets {
        let hf_path = format!("hf://{id}");
        match download_with_indicatif(&hf_path) {
            Ok(cached) => {
                eprintln!("  ✓ {id} → {}", cached.display());
                ok += 1;
            }
            Err(e) => {
                eprintln!("  ✗ {id}: {e}");
                failed.push((id, e.to_string()));
            }
        }
    }
    eprintln!("\nPulled {ok} of {} repos.", ok + failed.len());
    if !failed.is_empty() {
        eprintln!("Failures:");
        for (id, err) in &failed {
            eprintln!("  {id}: {err}");
        }
    }
    Ok(())
}

/// Pull the full repo + every default sibling preset. Missing siblings
/// log a warning; only the full repo is hard-required.
fn pull_all_slices(
    model: &str,
    template: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    pull_one(model, /*print_siblings=*/ false)?;
    for preset in DEFAULT_SIBLING_PRESETS {
        let sibling = match render_sibling_repo(model, preset, template) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  skip {preset}: {e}");
                continue;
            }
        };
        eprintln!("\n→ Pulling sibling `{preset}` ({sibling})");
        if let Err(e) = pull_one(&sibling, /*print_siblings=*/ false) {
            eprintln!("  skipped: {e}");
        }
    }
    Ok(())
}

/// After a successful pull, probe HF for standard sibling suffixes and
/// print what's available. Fail-silent — an HTTP error here shouldn't
/// mask the successful pull we just did.
fn hint_siblings(model: &str) {
    let bare = model.trim_start_matches("hf://");
    if !looks_like_hf_repo(bare) {
        return;
    }

    let (base, pulled_preset) = split_sibling_suffix(bare);
    let mut candidates: Vec<(String, String)> = Vec::new(); // (label, repo)
    if pulled_preset.is_some() {
        candidates.push(("full".into(), base.to_string()));
    }
    for preset in DEFAULT_SIBLING_PRESETS {
        if Some(*preset) == pulled_preset {
            continue;
        }
        candidates.push((preset.to_string(), format!("{base}-{preset}")));
    }

    let mut found: Vec<(String, String)> = Vec::new();
    for (label, repo) in &candidates {
        if let Ok(true) = larql_vindex::dataset_repo_exists(repo) {
            found.push((label.clone(), repo.clone()));
        }
    }
    if !found.is_empty() {
        eprintln!("\n  Also available on HuggingFace:");
        for (label, repo) in &found {
            eprintln!("    --preset {label:<8} → hf://{repo}");
        }
        eprintln!("  Use `larql pull <repo> --all-slices` to grab them all.");
    }
}

/// If `bare` ends in one of the known preset suffixes, return `(base,
/// Some(preset))`. Otherwise `(bare, None)`. Lets `hint_siblings`
/// suggest the full repo when the user pulled a specific slice directly.
fn split_sibling_suffix(bare: &str) -> (&str, Option<&'static str>) {
    for preset in DEFAULT_SIBLING_PRESETS {
        let suffix = format!("-{preset}");
        if let Some(base) = bare.strip_suffix(&suffix) {
            let preset_static: &'static str = preset;
            return (base, Some(preset_static));
        }
    }
    (bare, None)
}

fn normalise_hf_path(model: &str) -> Result<String, Box<dyn std::error::Error>> {
    if model.starts_with("hf://") {
        return Ok(model.to_string());
    }
    if looks_like_hf_repo(model) {
        return Ok(format!("hf://{model}"));
    }
    Err(format!(
        "pull expects `hf://owner/name` or `owner/name`, got: {model}"
    )
    .into())
}

// ─── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_sibling_uses_default_template() {
        let got = render_sibling_repo(
            "chrishayuk/gemma-4-31b-it-vindex",
            "client",
            DEFAULT_SIBLING_TEMPLATE,
        )
        .unwrap();
        assert_eq!(got, "chrishayuk/gemma-4-31b-it-vindex-client");
    }

    #[test]
    fn render_sibling_strips_hf_prefix() {
        let got = render_sibling_repo(
            "hf://chrishayuk/gemma-4-31b-it-vindex",
            "server",
            DEFAULT_SIBLING_TEMPLATE,
        )
        .unwrap();
        assert_eq!(got, "chrishayuk/gemma-4-31b-it-vindex-server");
    }

    #[test]
    fn render_sibling_custom_template() {
        let got = render_sibling_repo("me/model", "browse", "{repo}/{preset}").unwrap();
        assert_eq!(got, "me/model/browse");
    }

    #[test]
    fn render_sibling_rejects_local_path() {
        let err = render_sibling_repo(
            "/local/path/model.vindex",
            "client",
            DEFAULT_SIBLING_TEMPLATE,
        )
        .unwrap_err();
        assert!(err.to_string().contains("owner/name"), "got: {err}");
    }

    #[test]
    fn split_sibling_suffix_recognises_known_presets() {
        assert_eq!(
            split_sibling_suffix("chrishayuk/gemma-4-31b-it-vindex-client"),
            ("chrishayuk/gemma-4-31b-it-vindex", Some("client")),
        );
        assert_eq!(
            split_sibling_suffix("me/model-server"),
            ("me/model", Some("server")),
        );
        assert_eq!(
            split_sibling_suffix("me/model-browse"),
            ("me/model", Some("browse")),
        );
    }

    #[test]
    fn split_sibling_suffix_leaves_full_repo_untouched() {
        assert_eq!(
            split_sibling_suffix("chrishayuk/gemma-4-31b-it-vindex"),
            ("chrishayuk/gemma-4-31b-it-vindex", None),
        );
    }

    #[test]
    fn normalise_hf_path_accepts_hf_prefix_and_owner_name() {
        assert_eq!(
            normalise_hf_path("hf://me/model").unwrap(),
            "hf://me/model"
        );
        assert_eq!(normalise_hf_path("me/model").unwrap(), "hf://me/model");
    }

    #[test]
    fn normalise_hf_path_rejects_single_word() {
        assert!(normalise_hf_path("nomodel").is_err());
    }

    #[test]
    fn normalise_hf_path_rejects_local_path() {
        assert!(normalise_hf_path("/abs/path/model.vindex").is_err());
    }

    #[test]
    fn default_sibling_presets_match_publish_defaults() {
        // Symmetry guard: if publish's default slice set changes, pull
        // must change in lock-step so sibling hints don't go stale.
        // Keep in sync with `publish_cmd::DEFAULT_SLICES`.
        assert_eq!(
            DEFAULT_SIBLING_PRESETS,
            &["client", "attn", "embed", "server", "browse"]
        );
    }
}
