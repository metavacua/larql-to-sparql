use std::path::PathBuf;

use clap::Args;
use larql_codebase::extract_codebase;
use larql_codebase::graph_to_weight_repr;
use larql_codebase_core::basis::BitNetBasis;
use larql_models::loading::gguf::{GgufTensor, GgufValue, GgufWriter};
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
    let mut writer = GgufWriter::new();
    writer.meta("general.architecture", GgufValue::String("bitnet".into()));
    writer.meta("general.name", GgufValue::String("larql-codebase".into()));
    writer.meta(
        "larql.hidden_size",
        GgufValue::U32(repr.arch.hidden_size as u32),
    );
    writer.meta("larql.n_layers", GgufValue::U32(repr.arch.n_layers as u32));
    writer.meta("larql.n_heads", GgufValue::U32(repr.arch.n_heads as u32));
    for t in &repr.tensors {
        writer.tensor(GgufTensor {
            name: t.name.clone(),
            dims: t.dims.clone(),
            ggml_type: t.ggml_type,
            data: t.data.clone(),
        });
    }
    writer.write_to_file(&gguf_path)?;

    // Write a minimal vindex manifest so `larql show` can recognise the dir.
    let manifest = json!({
        "version": 1,
        "kind": "codebase",
        "source": root.to_string_lossy(),
        "weights": "weights.gguf",
        "arch": {
            "hidden_size": repr.arch.hidden_size,
            "n_layers": repr.arch.n_layers,
            "n_heads": repr.arch.n_heads,
            "head_dim": repr.arch.head_dim,
        }
    });
    std::fs::write(
        args.output.join("vindex.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;

    eprintln!("  vindex written to: {}", args.output.display());
    Ok(())
}
