//! Convert an existing MoE vindex from BF16 monolithic blob (`experts_packed.bin`)
//! to per-layer Q4_K files (`layers/layer_{L:02}.weights`).
//!
//! Usage:
//!   cargo run --release --example convert_moe_to_per_layer -- <vindex_path>
//!
//! Reads `weight_manifest.json` for BF16 expert byte ranges, quantizes each
//! expert to Q4_K, writes the new binary format, then updates `index.json`
//! with `"ffn_layout": "per_layer"`.

use std::collections::HashMap;
use std::path::Path;

use larql_vindex::format::weights::write_layers::{
    quantize_moe_entries, write_layer_weights, LayerWeightFormat,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <vindex_path>", args[0]);
        std::process::exit(1);
    }
    let vindex_path = Path::new(&args[1]);

    // Load and parse index.json
    let index_path = vindex_path.join("index.json");
    let index_text = std::fs::read_to_string(&index_path)?;
    let mut config: serde_json::Value = serde_json::from_str(&index_text)?;

    let num_layers = config["num_layers"].as_u64().ok_or("missing num_layers")? as usize;
    let hidden = config["hidden_size"]
        .as_u64()
        .ok_or("missing hidden_size")? as usize;

    let moe_cfg = config["model_config"]["moe"]
        .as_object()
        .ok_or("not a MoE model (no model_config.moe)")?;
    let num_experts = moe_cfg["num_experts"]
        .as_u64()
        .ok_or("missing num_experts")? as usize;
    let moe_inter = moe_cfg["moe_intermediate_size"]
        .as_u64()
        .ok_or("missing moe_intermediate_size")? as usize;

    eprintln!(
        "Model: {num_layers} layers, hidden={hidden}, {num_experts} experts, inter={moe_inter}"
    );

    // Parse weight_manifest.json → BF16 byte ranges
    let manifest_text = std::fs::read_to_string(vindex_path.join("weight_manifest.json"))?;
    let manifest: Vec<serde_json::Value> = serde_json::from_str(&manifest_text)?;

    let mut bf16_ranges: HashMap<String, (String, usize, usize)> = HashMap::new();
    for entry in &manifest {
        if entry["kind"].as_str() != Some("packed_bf16") {
            continue;
        }
        let key = entry["key"].as_str().unwrap_or("").to_string();
        let file = entry["file"].as_str().unwrap_or("").to_string();
        let offset = entry["offset"].as_u64().unwrap_or(0) as usize;
        let length = entry["length"].as_u64().unwrap_or(0) as usize;
        bf16_ranges.insert(key, (file, offset, length));
    }

    if bf16_ranges.is_empty() {
        return Err("no packed_bf16 entries in weight_manifest.json — already converted?".into());
    }

    // Open source mmaps lazily
    let mut open_mmaps: HashMap<String, memmap2::Mmap> = HashMap::new();
    let get_bytes = |file: &str,
                     offset: usize,
                     length: usize,
                     mmaps: &mut HashMap<String, memmap2::Mmap>|
     -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        if !mmaps.contains_key(file) {
            let f = std::fs::File::open(vindex_path.join(file))?;
            mmaps.insert(file.to_string(), unsafe { memmap2::Mmap::map(&f)? });
        }
        Ok(mmaps[file][offset..offset + length].to_vec())
    };

    // Convert each layer
    let fmt = LayerWeightFormat::Q4_K;
    let t_start = std::time::Instant::now();
    for layer in 0..num_layers {
        let gu_key = format!("layers.{layer}.experts.gate_up_proj");
        let dn_key = format!("layers.{layer}.experts.down_proj");

        let (gu_file, gu_off, gu_len) = bf16_ranges
            .get(&gu_key)
            .ok_or_else(|| format!("missing {gu_key}"))?
            .clone();
        let (dn_file, dn_off, dn_len) = bf16_ranges
            .get(&dn_key)
            .ok_or_else(|| format!("missing {dn_key}"))?
            .clone();

        let gu_bytes = get_bytes(&gu_file, gu_off, gu_len, &mut open_mmaps)?;
        let dn_bytes = get_bytes(&dn_file, dn_off, dn_len, &mut open_mmaps)?;

        let entries =
            quantize_moe_entries(&gu_bytes, &dn_bytes, num_experts, moe_inter, hidden, fmt)?;
        write_layer_weights(vindex_path, layer, fmt, &entries, moe_inter, hidden)?;

        let elapsed = t_start.elapsed().as_secs_f64();
        let rate = (layer + 1) as f64 / elapsed;
        let eta = (num_layers - layer - 1) as f64 / rate;
        eprintln!(
            "  layer {:02}/{} ({:.1}s elapsed, ETA {:.0}s)",
            layer,
            num_layers - 1,
            elapsed,
            eta
        );
    }

    // Update index.json
    config["ffn_layout"] = serde_json::Value::String("per_layer".into());
    std::fs::write(&index_path, serde_json::to_string_pretty(&config)?)?;

    eprintln!(
        "\nDone in {:.1}s. layers/ ready. experts_packed.bin can be removed after validation.",
        t_start.elapsed().as_secs_f64()
    );
    Ok(())
}
