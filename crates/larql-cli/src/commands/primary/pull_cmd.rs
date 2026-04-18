//! `larql pull <model>` — download a vindex and cache it locally.
//!
//! Resolves `hf://owner/name`, `owner/name`, or an existing directory to a
//! local path. Files land in `~/.cache/huggingface/` (the hf-hub convention
//! already used by `larql_vindex::resolve_hf_vindex`).

use std::path::PathBuf;

use clap::Args;

#[derive(Args)]
pub struct PullArgs {
    /// `hf://owner/name[@rev]`, `owner/name`, or a local path.
    pub model: String,
}

pub fn run(args: PullArgs) -> Result<(), Box<dyn std::error::Error>> {
    let hf_path = normalise_hf_path(&args.model)?;
    eprintln!("Pulling {hf_path}...");
    let cached: PathBuf = larql_vindex::resolve_hf_vindex(&hf_path)?;
    eprintln!("Cached at: {}", cached.display());
    if let Ok(cfg) = larql_vindex::load_vindex_config(&cached) {
        eprintln!(
            "  {} layers, hidden_size={}, dtype={:?}",
            cfg.num_layers, cfg.hidden_size, cfg.dtype,
        );
    }
    Ok(())
}

fn normalise_hf_path(model: &str) -> Result<String, Box<dyn std::error::Error>> {
    if model.starts_with("hf://") {
        return Ok(model.to_string());
    }
    if model.contains('/') && !model.contains(std::path::MAIN_SEPARATOR) {
        return Ok(format!("hf://{model}"));
    }
    Err(format!(
        "pull expects `hf://owner/name` or `owner/name`, got: {model}"
    )
    .into())
}
