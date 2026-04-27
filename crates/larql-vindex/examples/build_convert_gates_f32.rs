//! Convert gate_vectors.bin from f16 to f32 for zero-copy mmap access.
//!
//! Reads the existing f16 gate vectors, decodes to f32, writes a new file,
//! and updates index.json with the new dtype and byte offsets.
//!
//! Usage:
//!   cargo run --release -p larql-vindex --example convert_gates_f32 -- \
//!     /path/to/gemma3-4b-f16.vindex
//!
//! This modifies the vindex in-place (overwrites gate_vectors.bin and index.json).
//! Back up first if needed.

use std::io::Write;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vindex_dir = std::env::args().nth(1)
        .ok_or("Usage: convert_gates_f32 <vindex_dir>")?;
    let dir = Path::new(&vindex_dir);

    if !dir.join("index.json").exists() {
        return Err("Not a vindex directory (no index.json)".into());
    }

    // Read config
    let config_text = std::fs::read_to_string(dir.join("index.json"))?;
    let mut config: serde_json::Value = serde_json::from_str(&config_text)?;

    let dtype = config["dtype"].as_str().unwrap_or("f32");
    if dtype == "f32" {
        println!("Already f32. Nothing to do.");
        return Ok(());
    }
    if dtype != "f16" {
        return Err(format!("Unsupported dtype: {dtype}").into());
    }

    let num_layers = config["num_layers"].as_u64().unwrap() as usize;
    let hidden_size = config["hidden_size"].as_u64().unwrap() as usize;

    println!("=== Convert gate_vectors.bin: f16 → f32 ===\n");
    println!("Vindex: {}", dir.display());
    println!("Layers: {num_layers}, hidden: {hidden_size}");

    // Mmap the f16 file
    let gate_path = dir.join("gate_vectors.bin");
    let f16_file = std::fs::File::open(&gate_path)?;
    let f16_mmap = unsafe { memmap2::Mmap::map(&f16_file)? };
    let f16_size = f16_mmap.len();
    println!("F16 file: {:.1} MB", f16_size as f64 / 1e6);

    // Create output file
    let f32_path = dir.join("gate_vectors_f32.tmp");
    let mut f32_file = std::io::BufWriter::new(std::fs::File::create(&f32_path)?);

    let t0 = Instant::now();
    let mut new_offset: u64 = 0;

    let layers = config["layers"].as_array_mut()
        .ok_or("Missing layers array in index.json")?;

    for layer_info in layers.iter_mut() {
        let layer = layer_info["layer"].as_u64().unwrap() as usize;
        let num_features = layer_info["num_features"].as_u64().unwrap() as usize;
        let old_offset = layer_info["offset"].as_u64().unwrap() as usize;
        let old_length = layer_info["length"].as_u64().unwrap() as usize;

        if num_features == 0 {
            layer_info["offset"] = serde_json::json!(new_offset);
            layer_info["length"] = serde_json::json!(0);
            continue;
        }

        // Decode f16 → f32
        let f16_bytes = &f16_mmap[old_offset..old_offset + old_length];
        let f32_data = larql_models::quant::half::decode_f16(f16_bytes);

        // Write f32 bytes
        let f32_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                f32_data.as_ptr() as *const u8,
                f32_data.len() * 4,
            )
        };
        f32_file.write_all(f32_bytes)?;

        let new_length = f32_bytes.len() as u64;
        layer_info["offset"] = serde_json::json!(new_offset);
        layer_info["length"] = serde_json::json!(new_length);
        new_offset += new_length;

        if layer.is_multiple_of(10) || layer == num_layers - 1 {
            println!("  Layer {layer}/{num_layers}: {num_features} features, {:.1}MB",
                new_length as f64 / 1e6);
        }
    }

    f32_file.flush()?;
    drop(f32_file);
    drop(f16_mmap);
    drop(f16_file);

    // Atomic replace: rename tmp → gate_vectors.bin
    let backup_path = dir.join("gate_vectors_f16.bin.bak");
    std::fs::rename(&gate_path, &backup_path)?;
    std::fs::rename(&f32_path, &gate_path)?;

    let elapsed = t0.elapsed();
    let f32_size = new_offset;
    println!("\nF32 file: {:.1} MB ({:.1}s)", f32_size as f64 / 1e6, elapsed.as_secs_f64());

    // Update index.json
    config["dtype"] = serde_json::json!("f32");
    let new_config = serde_json::to_string_pretty(&config)?;
    std::fs::write(dir.join("index.json"), new_config)?;
    println!("Updated index.json: dtype=f32");

    println!("\nF16 backup: {}", backup_path.display());
    println!("Done. Gate vectors are now zero-copy mmap-able as f32.");

    Ok(())
}
