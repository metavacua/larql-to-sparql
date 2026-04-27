//! `larql link <path> [--as <name>]` — register a local vindex directory
//! with the cache so `larql list`/`run`/`show`/`rm` can find it by
//! shorthand.
//!
//! Creates a symlink:
//!
//! ```text
//! ~/.cache/larql/local/<name>.vindex  →  <absolute path>
//! ```
//!
//! The original directory is not moved or copied — the symlink just
//! advertises it. `larql rm <name>` unlinks without touching the
//! original.
//!
//! Name derivation:
//! - `--as <name>` wins if provided.
//! - Otherwise the basename of `<path>`, with a trailing `.vindex`
//!   stripped (so `output/gemma3-4b-f16.vindex` → `gemma3-4b-f16`).

use std::path::PathBuf;

use clap::Args;

use crate::commands::primary::cache;

#[derive(Args)]
pub struct LinkArgs {
    /// Path to a vindex directory (contains `index.json`).
    pub path: PathBuf,

    /// Override the registered name (defaults to the directory basename
    /// with any `.vindex` suffix stripped).
    #[arg(long = "as", value_name = "NAME")]
    pub as_name: Option<String>,

    /// Replace an existing link of the same name. Without this flag,
    /// linking over an existing entry errors out.
    #[arg(short = 'f', long)]
    pub force: bool,
}

pub fn run(args: LinkArgs) -> Result<(), Box<dyn std::error::Error>> {
    // Resolve target to an absolute path — symlinks without absolute
    // targets break the moment you cd elsewhere.
    let target = std::fs::canonicalize(&args.path).map_err(|e| {
        format!("could not resolve path `{}`: {e}", args.path.display())
    })?;
    if !target.is_dir() {
        return Err(format!("not a directory: {}", target.display()).into());
    }
    if !target.join("index.json").exists() {
        return Err(format!(
            "not a vindex: {} (no index.json)",
            target.display()
        )
        .into());
    }

    let name = match &args.as_name {
        Some(n) => n.clone(),
        None => {
            let base = target
                .file_name()
                .and_then(|n| n.to_str())
                .ok_or_else(|| format!("cannot derive name from path {}", target.display()))?;
            base.strip_suffix(".vindex").unwrap_or(base).to_string()
        }
    };
    validate_name(&name)?;

    let local_dir = cache::larql_local_dir()?;
    std::fs::create_dir_all(&local_dir)?;

    let link_path = local_dir.join(format!("{name}.vindex"));
    if link_path.exists() || link_path.is_symlink() {
        if !args.force {
            return Err(format!(
                "link already exists: {}\nRe-run with --force to replace.",
                link_path.display()
            )
            .into());
        }
        std::fs::remove_file(&link_path)
            .or_else(|_| std::fs::remove_dir_all(&link_path))?;
    }

    #[cfg(unix)]
    std::os::unix::fs::symlink(&target, &link_path)?;
    #[cfg(windows)]
    {
        // On Windows `symlink_dir` needs elevated privileges on older
        // builds; fall back to a junction with `std::fs::soft_link`
        // (deprecated but portable).
        #[allow(deprecated)]
        std::fs::soft_link(&target, &link_path)?;
    }

    eprintln!(
        "Linked {name}\n  {} → {}",
        link_path.display(),
        target.display()
    );
    Ok(())
}

/// Reject names that would collide with HF `owner/name` syntax or break
/// filesystem assumptions.
fn validate_name(name: &str) -> Result<(), Box<dyn std::error::Error>> {
    if name.is_empty() {
        return Err("name cannot be empty".into());
    }
    if name.contains('/') || name.contains(std::path::MAIN_SEPARATOR) {
        return Err(format!(
            "name `{name}` contains a path separator — use `--as` with a plain name"
        )
        .into());
    }
    if name.starts_with('.') {
        return Err(format!("name `{name}` cannot start with `.`").into());
    }
    Ok(())
}
