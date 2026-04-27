//! `larql show <model>` — print vindex metadata.
//!
//! Resolves the model the same way `run` does, then dumps `index.json` plus
//! file inventory (size per component) so you can see what's actually in
//! this vindex before you load it.

use clap::Args;

use crate::commands::primary::cache;

#[derive(Args)]
pub struct ShowArgs {
    /// Vindex directory, `hf://owner/name`, `owner/name`, or cache shorthand.
    pub model: String,
}

pub fn run(args: ShowArgs) -> Result<(), Box<dyn std::error::Error>> {
    let path = cache::resolve_model(&args.model)?;
    let cfg = larql_vindex::load_vindex_config(&path)?;

    println!("Model:      {}", args.model);
    println!("Path:       {}", path.display());
    println!("Layers:     {}", cfg.num_layers);
    println!("Hidden:     {}", cfg.hidden_size);
    println!("Dtype:      {:?}", cfg.dtype);
    println!("Quant:      {:?}", cfg.quant);

    println!("\nFiles:");
    let mut entries: Vec<_> = std::fs::read_dir(&path)?
        .filter_map(|e| e.ok())
        .filter(|e| e.metadata().map(|m| m.is_file()).unwrap_or(false))
        .collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let name = entry.file_name().to_string_lossy().to_string();
        let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
        println!("  {:<32} {:>12}", name, human_size(size));
    }
    Ok(())
}

fn human_size(bytes: u64) -> String {
    const K: u64 = 1024;
    const M: u64 = K * 1024;
    const G: u64 = M * 1024;
    if bytes >= G {
        format!("{:.2} GB", bytes as f64 / G as f64)
    } else if bytes >= M {
        format!("{:.1} MB", bytes as f64 / M as f64)
    } else if bytes >= K {
        format!("{:.1} KB", bytes as f64 / K as f64)
    } else {
        format!("{bytes} B")
    }
}
