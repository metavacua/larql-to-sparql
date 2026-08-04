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

    println!("Model:      {}", args.model);
    println!("Path:       {}", path.display());

    // Dispatch on the container's own discriminator, then let each generation
    // describe itself in its own vocabulary. Normalising VINDEX3 back into a
    // VINDEX2-shaped summary would drop exactly what a VINDEX3 user needs to
    // see — programme id, storage key, manifest validity.
    match larql_vindex::format::generation::detect_generation(&path)? {
        larql_vindex::format::generation::ContainerGeneration::V3 => show_v3(&path)?,
        larql_vindex::format::generation::ContainerGeneration::V2 => {
            let cfg = larql_vindex::load_vindex_config(&path)?;
            println!("Generation: VINDEX2");
            println!("Layers:     {}", cfg.num_layers);
            println!("Hidden:     {}", cfg.hidden_size);
            println!("Dtype:      {:?}", cfg.dtype);
            println!("Quant:      {:?}", cfg.quant);
        }
    }

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

/// Describe a VINDEX3 container in its own terms.
///
/// The per-layer lines are the ones with no VINDEX2 equivalent: which
/// programme interprets the bank, and which storage key it resolves to. Those
/// are what a binding failure is diagnosed from, so `show` is where they
/// belong.
fn show_v3(path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let container = larql_vindex::format::vindex3::Vindex3Container::open(path)?;
    let index = container.index();
    println!("Generation: VINDEX3 (index.json schema {})", index.version);
    println!("Layers:     {}", index.num_layers);
    println!("Hidden:     {}", index.hidden_size);
    println!("Family:     {}", index.family);
    println!("Profiles:   {}", index.profiles.join(", "));
    println!("Manifest:   {}", index.moe_manifest);

    println!("\nMoE layers:");
    for layer in &container.manifest().layers {
        let bank = &layer.routed_bank;
        let known = if bank.resolve_programme().is_some() {
            ""
        } else {
            "  [programme not implemented by this binary]"
        };
        println!(
            "  layer {:<3} {:<24} experts {:<4} storage {}{}",
            layer.layer, bank.programme, bank.experts, bank.storage, known
        );
    }

    let defects = container.verify();
    if defects.is_empty() {
        println!("\nStructure:  bindable (no defects)");
    } else {
        println!("\nStructure:  {} defect(s)", defects.len());
        for d in &defects {
            println!("  - {d}");
        }
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
