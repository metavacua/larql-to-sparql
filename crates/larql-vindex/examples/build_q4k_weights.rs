//! Build Q4_K attention weights + Q4_K/Q6_K FFN weights from vindex f32 data.
//!
//! Matches Ollama's quantization strategy:
//!   Attn Q/K/O: Q4_K
//!   Attn V:     Q6_K  (higher precision for value projection)
//!   FFN gate/up: Q4_K
//!   FFN down:    Q6_K  (higher precision for down projection)
//!
//! IMPORTANT: Uses larql_compute::cpu::ops::q4_common quantizers as single source of truth.
//! This ensures the format matches what the Metal/CPU kernels expect.
//!
//! Usage:
//!   cargo run --release -p larql-vindex --example build_q4k_weights -- <vindex_dir>

use std::io::Write;
use std::path::Path;
use std::time::Instant;

// Single source of truth for quantization — same functions used by compute kernels
use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args().nth(1)
        .unwrap_or_else(|| { eprintln!("Usage: build_q4k_weights <vindex_dir>"); std::process::exit(1); });
    let dir = Path::new(&dir);

    let manifest_path = dir.join("weight_manifest.json");
    if !manifest_path.exists() { return Err("weight_manifest.json not found".into()); }
    let manifest: Vec<serde_json::Value> = serde_json::from_str(
        &std::fs::read_to_string(&manifest_path)?
    )?;

    let t0 = Instant::now();
    println!("=== Building Q4_K/Q6_K weights (Ollama strategy) ===");
    println!("  Using larql_compute quantizers (single source of truth)\n");

    // Process attention weights: Q/K/O → Q4_K, V → Q6_K
    let attn_src = dir.join("attn_weights.bin");
    if attn_src.exists() {
        let file = std::fs::File::open(&attn_src)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let mut out = std::fs::File::create(dir.join("attn_weights_q4k.bin"))?;
        let mut q4k_manifest = Vec::new();
        let mut offset = 0usize;

        let entries: Vec<&serde_json::Value> = manifest.iter()
            .filter(|e| e.get("file").and_then(|f| f.as_str()) == Some("attn_weights.bin")
                && e.get("kind").and_then(|k| k.as_str()) == Some("tensor"))
            .collect();

        for entry in &entries {
            let key = entry["key"].as_str().unwrap_or("?");
            let file_offset = entry["offset"].as_u64().unwrap() as usize;
            let length = entry["length"].as_u64().unwrap() as usize;
            let shape = entry["shape"].as_array().unwrap();
            let rows = shape[0].as_u64().unwrap() as usize;
            let cols = shape[1].as_u64().unwrap() as usize;
            let num_floats = rows * cols;

            let f32_data = unsafe {
                let ptr = mmap[file_offset..file_offset + length].as_ptr() as *const f32;
                std::slice::from_raw_parts(ptr, num_floats)
            };

            // Pad to 256 for K-quant super-blocks
            let padded_len = (num_floats + 255) / 256 * 256;
            let padded = if padded_len != num_floats {
                let mut v = f32_data.to_vec();
                v.resize(padded_len, 0.0);
                v
            } else {
                f32_data.to_vec()
            };

            // V projection gets Q6_K (higher precision), others get Q4_K
            let is_v = key.contains("v_proj") || key.contains("attn_v");
            let (q_data, format) = if is_v {
                (quantize_q6_k(&padded), "Q6_K")
            } else {
                (quantize_q4_k(&padded), "Q4_K")
            };

            out.write_all(&q_data)?;
            q4k_manifest.push(serde_json::json!({
                "key": key, "shape": [rows, cols], "format": format,
                "offset": offset, "length": q_data.len(),
            }));
            offset += q_data.len();

            if offset < 100_000_000 {
                println!("  {key:45} [{rows},{cols}] → {format} {} bytes", q_data.len());
            }
        }

        std::fs::write(
            dir.join("attn_weights_q4k_manifest.json"),
            serde_json::to_string_pretty(&q4k_manifest)?,
        )?;
        println!("  Attention: {} entries, {} bytes total", q4k_manifest.len(), offset);
    } else {
        println!("  No attn_weights.bin found, skipping attention quantization");
    }

    // Process FFN interleaved: gate/up → Q4_K, down → Q6_K
    let interleaved_src = dir.join("interleaved.bin");
    if interleaved_src.exists() {
        let file = std::fs::File::open(&interleaved_src)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        let config_path = dir.join("index.json");
        let config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;
        let num_layers = config["num_layers"].as_u64().unwrap_or(0) as usize;
        let hidden = config["hidden_size"].as_u64().unwrap_or(0) as usize;
        let inter = config["intermediate_size"].as_u64().unwrap_or(config["num_features_per_layer"].as_u64().unwrap_or(0)) as usize;

        if num_layers > 0 && hidden > 0 && inter > 0 {
            let floats_per_matrix = inter * hidden;
            let bytes_per_matrix = floats_per_matrix * 4;
            let bytes_per_layer = bytes_per_matrix * 3; // gate, up, down

            let mut out = std::fs::File::create(dir.join("interleaved_q4k.bin"))?;
            let mut total_bytes = 0usize;

            for layer in 0..num_layers {
                let layer_offset = layer * bytes_per_layer;

                for (i, name) in ["gate", "up", "down"].iter().enumerate() {
                    let matrix_offset = layer_offset + i * bytes_per_matrix;
                    if matrix_offset + bytes_per_matrix > mmap.len() { break; }

                    let f32_data = unsafe {
                        let ptr = mmap[matrix_offset..matrix_offset + bytes_per_matrix].as_ptr() as *const f32;
                        std::slice::from_raw_parts(ptr, floats_per_matrix)
                    };

                    let padded_len = (floats_per_matrix + 255) / 256 * 256;
                    let padded = if padded_len != floats_per_matrix {
                        let mut v = f32_data.to_vec();
                        v.resize(padded_len, 0.0);
                        v
                    } else {
                        f32_data.to_vec()
                    };

                    // Down gets Q6_K, gate/up get Q4_K
                    let q_data = if *name == "down" {
                        quantize_q6_k(&padded)
                    } else {
                        quantize_q4_k(&padded)
                    };

                    out.write_all(&q_data)?;
                    total_bytes += q_data.len();
                }

                if layer < 3 || layer == num_layers - 1 {
                    println!("  Layer {layer}: gate(Q4_K) + up(Q4_K) + down(Q6_K)");
                } else if layer == 3 {
                    println!("  ...");
                }
            }
            println!("  FFN interleaved: {num_layers} layers, {total_bytes} bytes total");
        }
    } else {
        println!("  No interleaved.bin found, skipping FFN quantization");
    }

    println!("\nDone in {:.1}s", t0.elapsed().as_secs_f64());
    Ok(())
}
