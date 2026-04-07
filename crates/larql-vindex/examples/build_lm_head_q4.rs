//! Convert lm_head.bin (f32) → lm_head_q4.bin (Q4_0).
//!
//! Usage:
//!   cargo run --release -p larql-vindex --example build_lm_head_q4 -- <vindex_dir>

use std::io::Write;
use std::path::Path;
use std::time::Instant;
use larql_compute::cpu::q4::quantize_q4_0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args().nth(1)
        .unwrap_or_else(|| { eprintln!("Usage: build_lm_head_q4 <vindex_dir>"); std::process::exit(1); });
    let dir = Path::new(&dir);

    let src = dir.join("lm_head.bin");
    if !src.exists() {
        return Err("lm_head.bin not found".into());
    }

    let file = std::fs::File::open(&src)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let num_floats = mmap.len() / 4;
    let f32_data = unsafe {
        std::slice::from_raw_parts(mmap.as_ptr() as *const f32, num_floats)
    };

    // Must be multiple of 32 for Q4 — pad if needed
    let padded_len = num_floats.div_ceil(32) * 32;
    let data = if padded_len != num_floats {
        let mut v = f32_data.to_vec();
        v.resize(padded_len, 0.0);
        v
    } else {
        f32_data.to_vec()
    };

    println!("=== Building lm_head_q4.bin ===");
    println!("  Source: {} ({:.1} MB, {} floats)", src.display(), mmap.len() as f64 / 1e6, num_floats);

    let t0 = Instant::now();
    let q4 = quantize_q4_0(&data);
    let elapsed = t0.elapsed().as_secs_f64();

    let out_path = dir.join("lm_head_q4.bin");
    let mut out = std::fs::File::create(&out_path)?;
    out.write_all(&q4)?;

    let ratio = mmap.len() as f64 / q4.len() as f64;
    println!("  Output: {} ({:.1} MB, {:.1}x compression)", out_path.display(), q4.len() as f64 / 1e6, ratio);
    println!("  Time: {:.2}s", elapsed);
    println!("=== Done ===");
    Ok(())
}
