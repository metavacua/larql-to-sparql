//! `larql list` — show cached vindexes.
//!
//! Walks both caches (HF hub + LARQL local registry) and lists every
//! cached vindex with its size, layer count, and hidden dim. See
//! [`crate::commands::primary::cache`] for the scan logic shared with
//! `run` / `show` / `rm` / `link`.

use clap::Args;

use crate::commands::primary::cache;

#[derive(Args)]
pub struct ListArgs {}

pub fn run(_args: ListArgs) -> Result<(), Box<dyn std::error::Error>> {
    let entries = cache::scan_cached_vindexes()?;

    if entries.is_empty() {
        println!(
            "No cached vindexes.\n\
             Try `larql pull hf://owner/name` (remote) or \
             `larql link <path>` (local)."
        );
        return Ok(());
    }

    println!(
        "{:<8}  {:<48}  {:>10}  {:>7}  {:>8}",
        "SOURCE", "MODEL", "SIZE (MB)", "LAYERS", "HIDDEN"
    );
    for entry in &entries {
        let (layers, hidden) = larql_vindex::load_vindex_config(&entry.snapshot)
            .map(|c| (c.num_layers, c.hidden_size))
            .unwrap_or((0, 0));
        println!(
            "{:<8}  {:<48}  {:>10.1}  {:>7}  {:>8}",
            entry.source.label(),
            entry.repo,
            entry.size_bytes as f64 / 1e6,
            layers,
            hidden,
        );
    }
    Ok(())
}
