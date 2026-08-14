//! `larql inspect-hf` — machine-readable architecture inventory of an HF
//! checkpoint directory.
//!
//! Thin wrapper over [`larql_models::inventory::build_inventory`]: reads
//! `config.json` and the safetensors shard headers (no tensor data), prints
//! the inventory as JSON. See the module docs there for what the report
//! carries and why unconsumed config keys are its central finding.

use std::path::PathBuf;

use clap::Args;

#[derive(Args)]
pub struct InspectHfArgs {
    /// HF checkpoint directory (config.json + *.safetensors).
    pub path: PathBuf,

    /// Write the inventory JSON here instead of stdout.
    #[arg(long)]
    pub output: Option<PathBuf>,

    /// Omit the per-tensor list (keep totals and per-prefix groups) for a
    /// compact report on large checkpoints.
    #[arg(long)]
    pub no_tensor_list: bool,
}

pub fn run(args: InspectHfArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut inventory = larql_models::inventory::build_inventory(&args.path)?;
    if args.no_tensor_list {
        inventory.tensors.tensors.clear();
    }
    let json = serde_json::to_string_pretty(&inventory)?;
    match &args.output {
        Some(path) => {
            std::fs::write(path, &json)?;
            eprintln!("inventory written to {}", path.display());
        }
        None => println!("{json}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn fixture_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::json!({
                "model_type": "muse_glimmer",
                "text_config": {
                    "model_type": "muse_glimmer_text",
                    "hidden_size": 64,
                    "num_hidden_layers": 2,
                    "intermediate_size": 256,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 2
                }
            })
            .to_string(),
        )
        .unwrap();
        let header = serde_json::json!({
            "w": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]}
        });
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let mut f = std::fs::File::create(dir.path().join("model.safetensors")).unwrap();
        f.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        f.write_all(&header_bytes).unwrap();
        dir
    }

    #[test]
    fn writes_inventory_to_output_file() {
        let dir = fixture_dir();
        let out = dir.path().join("inventory.json");
        run(InspectHfArgs {
            path: dir.path().to_path_buf(),
            output: Some(out.clone()),
            no_tensor_list: false,
        })
        .unwrap();
        let text = std::fs::read_to_string(&out).unwrap();
        let inv: larql_models::inventory::ArchitectureInventory =
            serde_json::from_str(&text).unwrap();
        assert_eq!(inv.identity.model_type, "muse_glimmer_text");
        assert_eq!(inv.tensors.total_tensors, 1);
        assert_eq!(inv.tensors.tensors.len(), 1);
    }

    #[test]
    fn no_tensor_list_keeps_totals_and_drops_the_list() {
        let dir = fixture_dir();
        let out = dir.path().join("inventory.json");
        run(InspectHfArgs {
            path: dir.path().to_path_buf(),
            output: Some(out.clone()),
            no_tensor_list: true,
        })
        .unwrap();
        let inv: larql_models::inventory::ArchitectureInventory =
            serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
        assert_eq!(inv.tensors.total_tensors, 1);
        assert!(inv.tensors.tensors.is_empty());
    }

    #[test]
    fn missing_directory_is_an_error() {
        let result = run(InspectHfArgs {
            path: PathBuf::from("/nonexistent/model/dir"),
            output: None,
            no_tensor_list: false,
        });
        assert!(result.is_err());
    }
}
