//! Build Q4_K attention weights + Q4_K/Q6_K FFN weights from vindex f32 data.
//!
//! Matches Ollama's quantization strategy:
//!   Attn Q/K/O: Q4_K
//!   Attn V:     Q6_K  (higher precision for value projection)
//!   FFN gate/up: Q4_K
//!   FFN down:    Q6_K  (higher precision for down projection)
//!
//! Usage:
//!   cargo run --release -p larql-vindex --example build_q4k_weights -- <vindex_dir>

use std::io::Write;
use std::path::Path;
use std::time::Instant;

/// Q4_K super-block: 256 values → 148 bytes.
/// [0..1] f16 d, [2..3] f16 dmin, [4..15] 6-bit scales, [16..19] 4-bit mins, [20..147] 4-bit quants.
fn quantize_q4_k(data: &[f32]) -> Vec<u8> {
    assert!(data.len() % 256 == 0, "Q4_K requires multiple of 256 elements");
    let n_blocks = data.len() / 256;
    let mut out = Vec::with_capacity(n_blocks * 148);

    for sb in 0..n_blocks {
        let block = &data[sb * 256..(sb + 1) * 256];

        // Process 8 sub-blocks of 32 values each
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];
        let mut sub_maxes = [0.0f32; 8];
        let mut sub_mins = [0.0f32; 8];

        for j in 0..8 {
            let sub = &block[j * 32..(j + 1) * 32];
            let max_val = sub.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let min_val = sub.iter().copied().fold(f32::INFINITY, f32::min);
            sub_maxes[j] = max_val;
            sub_mins[j] = min_val;
        }

        // Compute super-block d and dmin
        let global_max = sub_maxes.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let global_min = sub_mins.iter().copied().fold(f32::INFINITY, f32::min).min(0.0);

        let d = if global_max > 0.0 { global_max / 15.0 / 63.0 } else { 0.0 };
        let dmin = if global_min < 0.0 { -global_min / 15.0 / 15.0 } else { 0.0 };

        // Compute per-sub-block scales and mins
        for j in 0..8 {
            if d > 0.0 {
                scales[j] = ((sub_maxes[j] / d / 15.0).round() as i32).clamp(0, 63) as u8;
            }
            if dmin > 0.0 {
                let sub = &block[j * 32..(j + 1) * 32];
                let sub_min = sub.iter().copied().fold(f32::INFINITY, f32::min).min(0.0);
                mins[j] = ((-sub_min / dmin / 15.0).round() as i32).clamp(0, 15) as u8;
            }
        }

        // Write header
        let d_f16 = larql_models::quant::half::f32_to_f16(d);
        let dmin_f16 = larql_models::quant::half::f32_to_f16(dmin);
        out.extend_from_slice(&d_f16.to_le_bytes());
        out.extend_from_slice(&dmin_f16.to_le_bytes());

        // Write 6-bit scales (12 bytes) — simplified: store lower 6 bits in first 8 bytes
        for j in 0..8 { out.push(scales[j] & 0x3F); }
        out.extend_from_slice(&[0u8; 4]); // padding for 12-byte scale field

        // Write 4-bit mins (4 bytes)
        for j in 0..4 {
            out.push(mins[j] | (mins[j + 4] << 4));
        }

        // Write 4-bit quantized values (128 bytes)
        for j in 0..8 {
            let sub = &block[j * 32..(j + 1) * 32];
            let sc = d * scales[j] as f32;
            let mn = dmin * mins[j] as f32;
            let inv_sc = if sc > 0.0 { 1.0 / sc } else { 0.0 };

            for i in 0..16 {
                let v0 = sub[i * 2];
                let v1 = sub[i * 2 + 1];
                let q0 = ((v0 + mn) * inv_sc).round().clamp(0.0, 15.0) as u8;
                let q1 = ((v1 + mn) * inv_sc).round().clamp(0.0, 15.0) as u8;
                out.push(q0 | (q1 << 4));
            }
        }
    }
    out
}

/// Q6_K super-block: 256 values → 210 bytes.
fn quantize_q6_k(data: &[f32]) -> Vec<u8> {
    assert!(data.len() % 256 == 0, "Q6_K requires multiple of 256 elements");
    let n_blocks = data.len() / 256;
    let mut out = Vec::with_capacity(n_blocks * 210);

    for sb in 0..n_blocks {
        let block = &data[sb * 256..(sb + 1) * 256];

        // Compute scales for 16 sub-blocks of 16 values
        let mut int_scales = [0i8; 16];
        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let d = amax / 32.0 / 127.0;
        let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };

        for j in 0..16 {
            let sub = &block[j * 16..(j + 1) * 16];
            let sub_max = sub.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let sc = if d > 0.0 { (sub_max / d / 32.0).round().clamp(-128.0, 127.0) } else { 0.0 };
            int_scales[j] = sc as i8;
        }

        // Lower 4 bits: 128 bytes
        let mut ql = vec![0u8; 128];
        // Upper 2 bits: 64 bytes
        let mut qh = vec![0u8; 64];

        for j in 0..16 {
            let sc = d * int_scales[j] as f32;
            let inv_sc = if sc.abs() > 0.0 { 1.0 / sc } else { 0.0 };
            for i in 0..16 {
                let idx = j * 16 + i;
                let q = (block[idx] * inv_sc).round().clamp(-32.0, 31.0) as i32 + 32;
                let lo4 = (q & 0x0F) as u8;
                let hi2 = ((q >> 4) & 0x03) as u8;
                // Pack lower 4 bits
                if idx % 2 == 0 {
                    ql[idx / 2] |= lo4;
                } else {
                    ql[idx / 2] |= lo4 << 4;
                }
                // Pack upper 2 bits
                let bit_offset = (idx % 4) * 2;
                qh[idx / 4] |= hi2 << bit_offset;
            }
        }

        out.extend_from_slice(&ql);
        out.extend_from_slice(&qh);
        for j in 0..16 { out.push(int_scales[j] as u8); }
        let d_f16 = larql_models::quant::half::f32_to_f16(d);
        out.extend_from_slice(&d_f16.to_le_bytes());
    }
    out
}

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

            // V projection gets Q6_K, others get Q4_K
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

        std::fs::write(dir.join("attn_weights_q4k_manifest.json"),
            serde_json::to_string_pretty(&q4k_manifest)?)?;
        println!("  Attention: {} entries, {:.1} MB", entries.len(), offset as f64 / 1e6);
    }

    // Process FFN interleaved: gate/up → Q4_K, down → Q6_K
    let interleaved_src = dir.join("interleaved.bin");
    if interleaved_src.exists() {
        let config: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("index.json"))?
        )?;
        let num_layers = config["num_layers"].as_u64().unwrap() as usize;
        let hidden = config["hidden_size"].as_u64().unwrap() as usize;
        let inter = config["intermediate_size"].as_u64().unwrap() as usize;

        let file = std::fs::File::open(&interleaved_src)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let mut out = std::fs::File::create(dir.join("interleaved_q4k.bin"))?;

        let matrix_floats = inter * hidden;
        let matrix_bytes = matrix_floats * 4;
        let layer_bytes = matrix_bytes * 3;
        let padded_floats = (matrix_floats + 255) / 256 * 256;
        let mut total = 0usize;

        for layer in 0..num_layers {
            let base = layer * layer_bytes;
            for (comp, format) in [(0, "Q4_K"), (1, "Q4_K"), (2, "Q6_K")] {
                let start = base + comp * matrix_bytes;
                let f32_data = unsafe {
                    let ptr = mmap[start..start + matrix_bytes].as_ptr() as *const f32;
                    std::slice::from_raw_parts(ptr, matrix_floats)
                };
                let mut padded = f32_data.to_vec();
                padded.resize(padded_floats, 0.0);

                let q_data = if format == "Q6_K" {
                    quantize_q6_k(&padded)
                } else {
                    quantize_q4_k(&padded)
                };
                out.write_all(&q_data)?;
                total += q_data.len();
            }
            if (layer + 1) % 10 == 0 { eprint!("\r  FFN layer {}/{}", layer + 1, num_layers); }
        }
        eprintln!();
        println!("  FFN interleaved: {num_layers} layers, {:.1} MB Q4_K/Q6_K", total as f64 / 1e6);
    }

    println!("  Time: {:.1}s", t0.elapsed().as_secs_f64());
    println!("=== Done ===");
    Ok(())
}
