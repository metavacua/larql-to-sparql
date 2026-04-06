//! Convert up_weights.bin from f16 to f32 for zero-copy mmap access.
//!
//! Up weights are already [intermediate, hidden] per layer (feature-major).
//! No transpose needed — just f16 → f32 decode.
//!
//! Creates: up_features.bin
//!
//! Usage:
//!   cargo run --release -p larql-vindex --example build_up_features -- path/to/vindex/

use std::io::Write;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vindex_dir = std::env::args().nth(1)
        .ok_or("Usage: build_up_features <vindex_dir>")?;
    let dir = Path::new(&vindex_dir);

    let manifest_text = std::fs::read_to_string(dir.join("weight_manifest.json"))?;
    let entries: Vec<serde_json::Value> = serde_json::from_str(&manifest_text)?;

    let up_entries: Vec<&serde_json::Value> = entries.iter()
        .filter(|e| {
            let key = e["key"].as_str().unwrap_or("");
            let file = e["file"].as_str().unwrap_or("");
            key.contains("up_proj") && file == "up_weights.bin"
        })
        .collect();

    if up_entries.is_empty() {
        return Err("No up_proj entries found".into());
    }

    let up_path = dir.join("up_weights.bin");
    let up_file = std::fs::File::open(&up_path)?;
    let up_mmap = unsafe { memmap2::Mmap::map(&up_file)? };

    println!("=== Build f32 Up Features ===\n");
    println!("Up entries: {}, file: {:.1}MB", up_entries.len(), up_mmap.len() as f64 / 1e6);

    let out_path = dir.join("up_features.bin");
    let mut out_file = std::io::BufWriter::with_capacity(
        8 * 1024 * 1024,
        std::fs::File::create(&out_path)?,
    );

    let t0 = Instant::now();
    let mut total: u64 = 0;

    for (i, entry) in up_entries.iter().enumerate() {
        let offset = entry["offset"].as_u64().unwrap() as usize;
        let length = entry["length"].as_u64().unwrap() as usize;
        let shape = entry["shape"].as_array().unwrap();
        let rows = shape[0].as_u64().unwrap() as usize;
        let cols = shape[1].as_u64().unwrap() as usize;

        // Already [intermediate, hidden] — no transpose needed
        let raw = &up_mmap[offset..offset + length];
        let bpf = length / (rows * cols);
        let floats: Vec<f32> = if bpf == 2 {
            larql_models::quant::half::decode_f16(raw)
        } else {
            unsafe {
                let ptr = raw.as_ptr() as *const f32;
                std::slice::from_raw_parts(ptr, rows * cols).to_vec()
            }
        };

        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(floats.as_ptr() as *const u8, floats.len() * 4)
        };
        out_file.write_all(bytes)?;
        total += bytes.len() as u64;

        if i % 10 == 0 || i == up_entries.len() - 1 {
            println!("  Layer {i}: [{rows}, {cols}], {:.1}MB", bytes.len() as f64 / 1e6);
        }
    }

    out_file.flush()?;
    println!("\nf32 file: {:.1}MB ({:.1}s)", total as f64 / 1e6, t0.elapsed().as_secs_f64());
    println!("File: {}", out_path.display());
    Ok(())
}
