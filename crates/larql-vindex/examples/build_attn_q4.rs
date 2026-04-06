//! Convert attn_weights.bin (f32) → attn_weights_q4.bin (Q4_0).
//!
//! Uses weight_manifest.json for exact per-matrix sizes (handles GQA where K/V ≠ Q/O).
//! Layout per layer: [Q_q4 | K_q4 | V_q4 | O_q4] — QK norms are skipped (1D vectors).
//!
//! Usage:
//!   cargo run --release -p larql-vindex --example build_attn_q4 -- <vindex_dir>

use std::io::Write;
use std::path::Path;
use std::time::Instant;
use larql_compute::cpu::q4::quantize_q4_0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args().nth(1)
        .unwrap_or_else(|| { eprintln!("Usage: build_attn_q4 <vindex_dir>"); std::process::exit(1); });
    let dir = Path::new(&dir);

    let src = dir.join("attn_weights.bin");
    if !src.exists() {
        return Err("attn_weights.bin not found".into());
    }

    // Load manifest for exact shapes
    let manifest_path = dir.join("weight_manifest.json");
    if !manifest_path.exists() {
        return Err("weight_manifest.json not found — need exact matrix sizes for GQA".into());
    }
    let manifest: Vec<serde_json::Value> = serde_json::from_str(
        &std::fs::read_to_string(&manifest_path)?
    )?;

    let file = std::fs::File::open(&src)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };

    println!("=== Building attn_weights_q4.bin (GQA-aware) ===");
    println!("  Source: {} ({:.1} MB)", src.display(), mmap.len() as f64 / 1e6);

    let t0 = Instant::now();
    let out_path = dir.join("attn_weights_q4.bin");
    let mut out = std::fs::File::create(&out_path)?;
    let mut total_q4 = 0usize;
    let mut total_f32 = 0usize;

    // Process each entry from manifest that's in attn_weights.bin and is a tensor (not vector)
    let attn_entries: Vec<&serde_json::Value> = manifest.iter()
        .filter(|e| {
            e.get("file").and_then(|f| f.as_str()) == Some("attn_weights.bin")
                && e.get("kind").and_then(|k| k.as_str()) == Some("tensor")
        })
        .collect();

    println!("  Manifest: {} tensor entries in attn_weights.bin", attn_entries.len());

    for entry in &attn_entries {
        let key = entry["key"].as_str().unwrap_or("?");
        let offset = entry["offset"].as_u64().unwrap() as usize;
        let length = entry["length"].as_u64().unwrap() as usize;
        let shape = entry["shape"].as_array().unwrap();
        let rows = shape[0].as_u64().unwrap() as usize;
        let cols = shape[1].as_u64().unwrap() as usize;
        let num_floats = rows * cols;

        // Read f32 data from mmap
        let f32_data = unsafe {
            let ptr = mmap[offset..offset + length].as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, num_floats)
        };

        // Pad to multiple of 32
        let padded_len = (num_floats + 31) / 32 * 32;
        let data = if padded_len != num_floats {
            let mut v = f32_data.to_vec();
            v.resize(padded_len, 0.0);
            v
        } else {
            f32_data.to_vec()
        };

        let q4 = quantize_q4_0(&data);
        out.write_all(&q4)?;
        total_q4 += q4.len();
        total_f32 += num_floats * 4;

        // Print first few for verification
        if total_q4 < 100_000_000 {
            println!("    {} [{},{}] → {} bytes Q4", key, rows, cols, q4.len());
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let ratio = total_f32 as f64 / total_q4 as f64;
    println!("  Output: {} ({:.1} MB, {:.1}x compression)", out_path.display(), total_q4 as f64 / 1e6, ratio);
    println!("  Time: {:.1}s", elapsed);

    // Write a sidecar with per-matrix Q4 offsets for the inference pipeline
    let mut q4_offset = 0usize;
    let mut q4_manifest: Vec<serde_json::Value> = Vec::new();
    for entry in &attn_entries {
        let shape = entry["shape"].as_array().unwrap();
        let rows = shape[0].as_u64().unwrap() as usize;
        let cols = shape[1].as_u64().unwrap() as usize;
        let padded = ((rows * cols) + 31) / 32 * 32;
        let q4_bytes = padded / 32 * 18;
        q4_manifest.push(serde_json::json!({
            "key": entry["key"],
            "shape": [rows, cols],
            "q4_offset": q4_offset,
            "q4_length": q4_bytes,
        }));
        q4_offset += q4_bytes;
    }
    let manifest_out = dir.join("attn_weights_q4_manifest.json");
    std::fs::write(&manifest_out, serde_json::to_string_pretty(&q4_manifest)?)?;
    println!("  Manifest: {} ({} entries)", manifest_out.display(), q4_manifest.len());

    println!("=== Done ===");
    Ok(())
}
