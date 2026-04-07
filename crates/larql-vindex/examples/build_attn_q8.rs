//! Convert attn_weights.bin (f32) → attn_weights_q8.bin (Q8_0).
//!
//! Q8 for attention projections — higher precision than Q4 (matches llama.cpp strategy).
//! Uses weight_manifest.json for exact per-matrix sizes (handles GQA).
//!
//! Usage:
//!   cargo run --release -p larql-vindex --example build_attn_q8 -- <vindex_dir>

use std::io::Write;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args().nth(1)
        .unwrap_or_else(|| { eprintln!("Usage: build_attn_q8 <vindex_dir>"); std::process::exit(1); });
    let dir = Path::new(&dir);

    let src = dir.join("attn_weights.bin");
    if !src.exists() { return Err("attn_weights.bin not found".into()); }

    let manifest_path = dir.join("weight_manifest.json");
    if !manifest_path.exists() { return Err("weight_manifest.json not found".into()); }
    let manifest: Vec<serde_json::Value> = serde_json::from_str(
        &std::fs::read_to_string(&manifest_path)?
    )?;

    let file = std::fs::File::open(&src)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };

    println!("=== Building attn_weights_q8.bin ===");
    println!("  Source: {} ({:.1} MB)", src.display(), mmap.len() as f64 / 1e6);

    let t0 = Instant::now();
    let out_path = dir.join("attn_weights_q8.bin");
    let mut out = std::fs::File::create(&out_path)?;
    let mut total_q8 = 0usize;
    let mut total_f32 = 0usize;

    let attn_entries: Vec<&serde_json::Value> = manifest.iter()
        .filter(|e| {
            e.get("file").and_then(|f| f.as_str()) == Some("attn_weights.bin")
                && e.get("kind").and_then(|k| k.as_str()) == Some("tensor")
        })
        .collect();

    println!("  Manifest: {} tensor entries", attn_entries.len());

    // Q8_0 quantization: per block of 32 elements, find max abs, scale to [-127, 127]
    // Output: separate int8 values + separate f32 scales (our format, not GGML Q8_0)
    // Layout per matrix: [rows * cols int8 values] + [rows * cols / 32 f32 scales]
    let mut q8_manifest: Vec<serde_json::Value> = Vec::new();
    let mut q8_offset = 0usize;

    for entry in &attn_entries {
        let key = entry["key"].as_str().unwrap_or("?");
        let offset = entry["offset"].as_u64().unwrap() as usize;
        let length = entry["length"].as_u64().unwrap() as usize;
        let shape = entry["shape"].as_array().unwrap();
        let rows = shape[0].as_u64().unwrap() as usize;
        let cols = shape[1].as_u64().unwrap() as usize;
        let num_floats = rows * cols;
        let padded = (num_floats + 31) / 32 * 32;
        let n_blocks = padded / 32;

        let f32_data = unsafe {
            let ptr = mmap[offset..offset + length].as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, num_floats)
        };

        // Quantize to Q8: int8 values + f32 scales (separate arrays)
        let mut q8_vals = Vec::with_capacity(padded);
        let mut q8_scales = Vec::with_capacity(n_blocks);

        for b in 0..n_blocks {
            let start = b * 32;
            let _end = (start + 32).min(num_floats);
            let block: Vec<f32> = (start..start + 32).map(|i| {
                if i < num_floats { f32_data[i] } else { 0.0 }
            }).collect();

            let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = amax / 127.0;
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            q8_scales.push(scale);

            for j in 0..32 {
                let q = (block[j] * inv).round().clamp(-128.0, 127.0) as i8;
                q8_vals.push(q as u8);
            }
        }

        // Write: int8 values then f32 scales
        let vals_bytes = padded; // 1 byte per value
        let scales_bytes = n_blocks * 4; // f32 per block
        out.write_all(&q8_vals)?;
        let scale_bytes: Vec<u8> = q8_scales.iter().flat_map(|s| s.to_le_bytes()).collect();
        out.write_all(&scale_bytes)?;

        let entry_size = vals_bytes + scales_bytes;
        q8_manifest.push(serde_json::json!({
            "key": key,
            "shape": [rows, cols],
            "q8_offset": q8_offset,
            "q8_vals_len": vals_bytes,
            "q8_scales_len": scales_bytes,
            "q8_total_len": entry_size,
        }));

        total_q8 += entry_size;
        total_f32 += num_floats * 4;
        q8_offset += entry_size;

        if total_f32 < 400_000_000 {
            println!("    {} [{},{}] → {} bytes Q8 ({} vals + {} scales)",
                key, rows, cols, entry_size, vals_bytes, scales_bytes);
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let ratio = total_f32 as f64 / total_q8 as f64;
    println!("  Output: {} ({:.1} MB, {:.1}x compression)", out_path.display(), total_q8 as f64 / 1e6, ratio);
    println!("  Time: {:.1}s", elapsed);

    let manifest_out = dir.join("attn_weights_q8_manifest.json");
    std::fs::write(&manifest_out, serde_json::to_string_pretty(&q8_manifest)?)?;
    println!("  Manifest: {} ({} entries)", manifest_out.display(), q8_manifest.len());
    println!("=== Done ===");
    Ok(())
}
