use std::path::{Path, PathBuf};

use clap::{Args, ValueEnum};
use larql_codebase::graph_to_weight_repr;
use larql_codebase::write_weight_repr_to_gguf;
use larql_codebase_core::basis::BitNetBasis;
use larql_core::io::load_graph;

#[derive(ValueEnum, Clone)]
pub enum ExportFormat {
    Gguf,
}

#[derive(Args)]
pub struct ExportArgs {
    /// Input: a .larql.json graph file.
    input: PathBuf,

    /// Output file path.
    #[arg(short, long)]
    output: PathBuf,

    /// Export format.
    #[arg(long, value_enum, default_value = "gguf")]
    format: ExportFormat,
}

pub fn export_to_gguf(input: &Path, output: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let graph = load_graph(input).map_err(|e| format!("load graph: {e}"))?;
    let repr = graph_to_weight_repr(&graph, &BitNetBasis);

    let model_name = output
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("larql-export");
    let n_tensors = repr.tensors.len();
    write_weight_repr_to_gguf(repr, model_name, output)?;
    eprintln!("Wrote {} tensors → {}", n_tensors, output.display());
    Ok(())
}

pub fn run(args: ExportArgs) -> Result<(), Box<dyn std::error::Error>> {
    match args.format {
        ExportFormat::Gguf => export_to_gguf(&args.input, &args.output),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_core::core::{edge::Edge, graph::Graph};
    use larql_core::io::save_graph;
    use tempfile::NamedTempFile;

    #[test]
    fn gguf_header_magic_present() {
        let mut g = Graph::new();
        for i in 0..20 {
            g.add_edge(Edge::new(
                format!("n{i}"),
                "calls",
                format!("n{}", (i + 1) % 20),
            ));
        }
        let graph_file = NamedTempFile::with_suffix(".larql.json").unwrap();
        save_graph(&g, graph_file.path()).unwrap();

        let out = NamedTempFile::with_suffix(".gguf").unwrap();
        export_to_gguf(graph_file.path(), out.path()).unwrap();

        let bytes = std::fs::read(out.path()).unwrap();
        // GGUF magic: "GGUF" in little-endian
        assert_eq!(&bytes[0..4], b"GGUF", "expected GGUF magic bytes");
    }
}
