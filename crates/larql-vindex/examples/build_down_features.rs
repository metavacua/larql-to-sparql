//! Build feature-major vectors from vindex weight files.
//! Transposes [hidden, intermediate] → [intermediate, hidden] per layer.
//!
//! Creates both:
//!   down_features.bin — from down_weights.bin
//!   up_features.bin   — from up_weights.bin (if present)
//!
//! Each feature's vector is contiguous for cache-friendly mmap access.
//!
//! Previously:
//!
//! Transposes [hidden, intermediate] → [intermediate, hidden] per layer,
//! so each feature's down vector is contiguous in memory. Stores as f32.
//!
//! Creates: down_features.bin (3.4GB for Gemma-3 4B)
//!          + updates weight_manifest.json with offsets
//!
//! Usage:
//!   cargo run --release -p larql-vindex --example build_down_features -- \
//!     /path/to/vindex/

use std::io::Write;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vindex_dir = std::env::args().nth(1)
        .ok_or("Usage: build_down_features <vindex_dir>")?;
    let dir = Path::new(&vindex_dir);

    let config_text = std::fs::read_to_string(dir.join("index.json"))?;
    let config: serde_json::Value = serde_json::from_str(&config_text)?;

    let num_layers = config["num_layers"].as_u64().unwrap() as usize;
    let hidden_size = config["hidden_size"].as_u64().unwrap() as usize;
    let intermediate_size = config["intermediate_size"].as_u64().unwrap() as usize;

    // Read dtype from config (down_weights may be f16 or f32)
    let manifest_text = std::fs::read_to_string(dir.join("weight_manifest.json"))?;
    let entries: Vec<serde_json::Value> = serde_json::from_str(&manifest_text)?;

    // Find down weight entries
    let down_entries: Vec<&serde_json::Value> = entries.iter()
        .filter(|e| {
            let key = e["key"].as_str().unwrap_or("");
            let file = e["file"].as_str().unwrap_or("");
            key.contains("down_proj") && file == "down_weights.bin"
        })
        .collect();

    if down_entries.is_empty() {
        return Err("No down_proj entries found in weight_manifest.json".into());
    }

    println!("=== Build Feature-Major Down Vectors ===\n");
    println!("Vindex: {}", dir.display());
    println!("Layers: {num_layers}, hidden: {hidden_size}, intermediate: {intermediate_size}");
    println!("Down entries: {}", down_entries.len());

    // Determine storage dtype from file size
    let down_path = dir.join("down_weights.bin");
    let down_file = std::fs::File::open(&down_path)?;
    let down_mmap = unsafe { memmap2::Mmap::map(&down_file)? };

    let expected_f16 = num_layers * hidden_size * intermediate_size * 2;
    let expected_f32 = num_layers * hidden_size * intermediate_size * 4;
    let is_f16 = down_mmap.len() == expected_f16;
    let is_f32 = down_mmap.len() == expected_f32;

    if !is_f16 && !is_f32 {
        println!("WARNING: down_weights.bin size {} doesn't match expected f16 ({}) or f32 ({})",
            down_mmap.len(), expected_f16, expected_f32);
        println!("  Falling back to per-entry size detection");
    }

    let dtype_str = if is_f16 { "f16" } else { "f32" };
    println!("Down weights dtype: {dtype_str}");
    println!("Down weights size: {:.1} MB\n", down_mmap.len() as f64 / 1e6);

    // Create feature-major output: [intermediate, hidden] per layer, all f32
    let out_path = dir.join("down_features.bin");
    let mut out_file = std::io::BufWriter::with_capacity(
        8 * 1024 * 1024, // 8MB buffer
        std::fs::File::create(&out_path)?
    );

    let t0 = Instant::now();
    let mut total_bytes: u64 = 0;

    for (layer_idx, entry) in down_entries.iter().enumerate() {
        let offset = entry["offset"].as_u64().unwrap() as usize;
        let length = entry["length"].as_u64().unwrap() as usize;
        let shape = entry["shape"].as_array().unwrap();
        let rows = shape[0].as_u64().unwrap() as usize; // hidden
        let cols = shape[1].as_u64().unwrap() as usize; // intermediate

        // Read the weight matrix (may be f16 or f32)
        let raw = &down_mmap[offset..offset + length];
        let floats: Vec<f32> = if is_f16 {
            larql_models::quant::half::decode_f16(raw)
        } else {
            unsafe {
                let ptr = raw.as_ptr() as *const f32;
                std::slice::from_raw_parts(ptr, rows * cols).to_vec()
            }
        };

        // Transpose: [hidden, intermediate] → [intermediate, hidden]
        // Input: row-major [rows=hidden, cols=intermediate]
        // Output: row-major [intermediate, hidden] — each row is one feature's down vector
        let mut transposed = vec![0.0f32; cols * rows];
        for r in 0..rows {
            for c in 0..cols {
                transposed[c * rows + r] = floats[r * cols + c];
            }
        }

        // Write as f32 bytes
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                transposed.as_ptr() as *const u8,
                transposed.len() * 4,
            )
        };
        out_file.write_all(bytes)?;
        total_bytes += bytes.len() as u64;

        if layer_idx % 10 == 0 || layer_idx == down_entries.len() - 1 {
            println!("  Layer {layer_idx}: [{rows}, {cols}] → [{cols}, {rows}], {:.1}MB",
                bytes.len() as f64 / 1e6);
        }
    }

    out_file.flush()?;

    let elapsed = t0.elapsed();
    println!("\nFeature-major file: {:.1} MB ({:.1}s)", total_bytes as f64 / 1e6, elapsed.as_secs_f64());
    println!("Layout: [intermediate={intermediate_size}, hidden={hidden_size}] per layer, f32");
    println!("Each feature's down vector: {hidden_size} contiguous f32 ({:.1}KB)",
        hidden_size as f64 * 4.0 / 1024.0);
    println!("\nFile: {}", out_path.display());
    println!("Done.");

    Ok(())
}
