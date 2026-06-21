use std::path::PathBuf;

use clap::Args;
use larql_codebase::extract_codebase;
use larql_codebase::graph_to_weight_repr;
use larql_codebase::write_weight_repr_to_gguf;
use larql_codebase_core::basis::BitNetBasis;
use serde_json::json;

#[derive(Args)]
pub struct ExtractCodebaseArgs {
    /// Root directory of the codebase to extract.
    root: PathBuf,

    /// Output vindex directory (created if absent).
    #[arg(short, long, default_value = "codebase.vindex")]
    output: PathBuf,
}

pub fn run(args: ExtractCodebaseArgs) -> Result<(), Box<dyn std::error::Error>> {
    let root = args.root.canonicalize()?;
    eprintln!("Extracting codebase from: {}", root.display());

    let graph = extract_codebase(&root)?;
    eprintln!(
        "  {} nodes, {} edges extracted",
        graph.node_count(),
        graph.edge_count()
    );

    let repr = graph_to_weight_repr(&graph, &BitNetBasis);
    eprintln!("  {} weight tensors synthesised", repr.tensors.len());

    std::fs::create_dir_all(&args.output)?;

    // Write weights as a GGUF file inside the vindex directory.
    let gguf_path = args.output.join("weights.gguf");
    let model_name = root
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("larql-codebase");
    // Save arch before repr is consumed by the GGUF writer.
    let arch = repr.arch.clone();
    write_weight_repr_to_gguf(repr, model_name, &gguf_path)?;

    // Write a minimal vindex manifest so `larql show` can recognise the dir.
    let manifest = json!({
        "version": 1,
        "kind": "codebase",
        "source": root.to_string_lossy(),
        "weights": "weights.gguf",
        "arch": {
            "hidden_size": arch.hidden_size,
            "n_layers": arch.n_layers,
            "n_heads": arch.n_heads,
            "head_dim": arch.head_dim,
        }
    });
    std::fs::write(
        args.output.join("vindex.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;

    eprintln!("  vindex written to: {}", args.output.display());
    Ok(())
}
