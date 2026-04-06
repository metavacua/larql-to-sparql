//! Quantize interleaved.bin to Q4_0 format.
//!
//! Reads f32 interleaved [gate|up|down] per layer, quantizes each matrix
//! to Q4_0 (18 bytes per 32 elements), writes interleaved_q4.bin.
//!
//! Usage:
//!   cargo run --release -p larql-vindex --example build_q4_interleaved -- output/gemma3-4b-v2.vindex


use std::io::Write;
use std::path::Path;
use std::time::Instant;

use larql_models::quant::ggml::{quantize_q4_0, dequantize_q4_0};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args().nth(1).ok_or("Usage: build_q4_interleaved <vindex_dir>")?;
    let dir = Path::new(&dir);

    let config_text = std::fs::read_to_string(dir.join("index.json"))?;
    let config: serde_json::Value = serde_json::from_str(&config_text)?;
    let num_layers = config["num_layers"].as_u64().unwrap() as usize;
    let hidden_size = config["hidden_size"].as_u64().unwrap() as usize;
    let intermediate_size = config["intermediate_size"].as_u64().unwrap() as usize;

    let floats_per_matrix = intermediate_size * hidden_size;
    let bytes_per_matrix_f32 = floats_per_matrix * 4;
    let bytes_per_matrix_q4 = floats_per_matrix / 32 * 18; // Q4_0: 18 bytes per 32 elements
    let bytes_per_layer_f32 = bytes_per_matrix_f32 * 3;
    let bytes_per_layer_q4 = bytes_per_matrix_q4 * 3;

    println!("=== Build Q4_0 Interleaved Vindex ===\n");
    println!("Layers: {num_layers}, hidden: {hidden_size}, intermediate: {intermediate_size}");
    println!("Per matrix: {:.1}MB f32 → {:.1}MB Q4_0 ({:.1}x)",
        bytes_per_matrix_f32 as f64 / 1e6,
        bytes_per_matrix_q4 as f64 / 1e6,
        bytes_per_matrix_f32 as f64 / bytes_per_matrix_q4 as f64);
    println!("Per layer:  {:.1}MB → {:.1}MB",
        bytes_per_layer_f32 as f64 / 1e6,
        bytes_per_layer_q4 as f64 / 1e6);
    println!("Total:      {:.1}GB → {:.1}GB\n",
        (bytes_per_layer_f32 * num_layers) as f64 / 1e9,
        (bytes_per_layer_q4 * num_layers) as f64 / 1e9);

    // Open source file
    let src_path = dir.join("interleaved.bin");
    if !src_path.exists() {
        return Err("interleaved.bin not found. Run build_interleaved first.".into());
    }
    let src_file = std::fs::File::open(&src_path)?;
    let src_mmap = unsafe { memmap2::Mmap::map(&src_file)? };

    // Output
    let out_path = dir.join("interleaved_q4.bin");
    let mut out = std::io::BufWriter::with_capacity(
        16 * 1024 * 1024,
        std::fs::File::create(&out_path)?,
    );

    let t0 = Instant::now();
    let mut total_q4_bytes: u64 = 0;
    let mut max_error: f32 = 0.0;
    let mut total_rmse: f64 = 0.0;
    let mut total_elements: u64 = 0;

    for layer in 0..num_layers {
        let layer_offset = layer * bytes_per_layer_f32;

        for (comp, _comp_name) in [(0, "gate"), (1, "up"), (2, "down")] {
            let start = layer_offset + comp * bytes_per_matrix_f32;
            let end = start + bytes_per_matrix_f32;

            // Read f32 data
            let f32_data = unsafe {
                let ptr = src_mmap[start..end].as_ptr() as *const f32;
                std::slice::from_raw_parts(ptr, floats_per_matrix)
            };

            // Quantize
            let q4_bytes = quantize_q4_0(f32_data);

            // Measure error (on first + last layer)
            if layer == 0 || layer == num_layers - 1 {
                let reconstructed = dequantize_q4_0(&q4_bytes, floats_per_matrix).unwrap();
                let mut layer_rmse: f64 = 0.0;
                for i in 0..floats_per_matrix {
                    let err = (f32_data[i] - reconstructed[i]).abs();
                    if err > max_error { max_error = err; }
                    layer_rmse += (err as f64) * (err as f64);
                }
                layer_rmse = (layer_rmse / floats_per_matrix as f64).sqrt();
                total_rmse += layer_rmse;
                total_elements += 1;
            }

            out.write_all(&q4_bytes)?;
            total_q4_bytes += q4_bytes.len() as u64;
        }

        if layer % 10 == 0 || layer == num_layers - 1 {
            println!("  Layer {layer}: {:.1}MB Q4_0", bytes_per_layer_q4 as f64 / 1e6);
        }
    }

    out.flush()?;
    let elapsed = t0.elapsed();

    let avg_rmse = total_rmse / total_elements.max(1) as f64;
    println!("\nQ4_0 file: {:.1}MB ({:.1}s)",
        total_q4_bytes as f64 / 1e6, elapsed.as_secs_f64());
    println!("Compression: {:.1}x", (bytes_per_layer_f32 * num_layers) as f64 / total_q4_bytes as f64);
    println!("Max abs error: {max_error:.6}");
    println!("Avg RMSE per matrix: {avg_rmse:.6}");
    println!("File: {}", out_path.display());
    println!("Done.");

    Ok(())
}
