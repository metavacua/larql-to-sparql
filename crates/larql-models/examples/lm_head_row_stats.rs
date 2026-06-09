//! Dump per-row L2 norm + max-abs of a Q4_K-quantised lm_head matrix.
//!
//! Diagnostic for the qwen35 forward "logit collapse" bug — if any rows in
//! lm_head have abnormally large magnitudes, the matvec output will favour
//! those rows' token IDs regardless of input, producing the consistent-garbage
//! output we see on Qwen 3.6 chat completions.
//!
//! Usage:
//!   cargo run --release --example lm_head_row_stats -- <vindex_dir>
//!
//! The vindex must have `lm_head_q4.bin` (Q4_K) and `weight_manifest.json`
//! describing it.

use std::fs;
use std::path::PathBuf;

use larql_models::quant::ggml::dequantize_q4_k;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: {} <vindex_dir>", args[0]);
        std::process::exit(2);
    }
    let dir = PathBuf::from(&args[1]);

    let manifest_path = dir.join("weight_manifest.json");
    let manifest_bytes =
        fs::read(&manifest_path).unwrap_or_else(|e| panic!("read manifest {manifest_path:?}: {e}"));
    let manifest: serde_json::Value =
        serde_json::from_slice(&manifest_bytes).unwrap_or_else(|e| panic!("parse manifest: {e}"));
    let weights = manifest
        .get("weights")
        .and_then(|w| w.as_array())
        .or_else(|| manifest.as_array())
        .expect("manifest weights array");

    let lm = weights
        .iter()
        .find(|w| w.get("key").and_then(|k| k.as_str()) == Some("lm_head.weight"))
        .expect("lm_head.weight entry not in manifest");
    let shape = lm
        .get("shape")
        .and_then(|s| s.as_array())
        .expect("lm_head shape");
    let rows = shape[0].as_u64().unwrap() as usize;
    let cols = shape[1].as_u64().unwrap() as usize;
    let file = lm.get("file").and_then(|f| f.as_str()).unwrap();
    let offset = lm.get("offset").and_then(|o| o.as_u64()).unwrap() as usize;
    let length = lm.get("length").and_then(|l| l.as_u64()).unwrap() as usize;

    println!("lm_head: rows={rows} cols={cols} file={file} offset={offset} length={length}");

    let bytes = fs::read(dir.join(file)).expect("read lm_head_q4.bin");
    let bytes = &bytes[offset..offset + length];
    assert_eq!(cols % 256, 0, "Q4_K requires cols%256==0, got {cols}");
    let row_bytes = (cols / 256) * 144;
    assert_eq!(bytes.len(), rows * row_bytes, "size mismatch");

    let mut norms: Vec<(usize, f32, f32)> = Vec::with_capacity(rows);
    let mut sum_l2 = 0.0_f64;
    for r in 0..rows {
        let row_slice = &bytes[r * row_bytes..(r + 1) * row_bytes];
        let f = dequantize_q4_k(row_slice, cols).expect("dequant row");
        let l2 = f.iter().map(|v| v * v).sum::<f32>().sqrt();
        let max_abs = f.iter().fold(0.0_f32, |a, &v| a.max(v.abs()));
        norms.push((r, l2, max_abs));
        sum_l2 += l2 as f64;
        if r % 10000 == 0 {
            eprintln!("  row {r}/{rows} l2={l2:.3} max_abs={max_abs:.3}");
        }
    }
    let mean_l2 = (sum_l2 / rows as f64) as f32;
    let max_entry = norms
        .iter()
        .copied()
        .fold((0usize, 0.0_f32, 0.0_f32), |acc, x| {
            if x.1 > acc.1 {
                x
            } else {
                acc
            }
        });
    let min_entry = norms
        .iter()
        .copied()
        .fold((0usize, f32::INFINITY, f32::INFINITY), |acc, x| {
            if x.1 < acc.1 {
                x
            } else {
                acc
            }
        });
    println!("---");
    println!("mean L2 = {mean_l2:.3}");
    println!(
        "max L2  = {:.3} at row {} (max_abs {:.3})",
        max_entry.1, max_entry.0, max_entry.2
    );
    println!("min L2  = {:.3} at row {}", min_entry.1, min_entry.0);

    // Top 20 outliers
    let mut sorted = norms.clone();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("--- top 20 rows by L2 ---");
    for (r, l2, max_abs) in &sorted[..20.min(sorted.len())] {
        println!("  row {r:>6}: l2={l2:.3} max_abs={max_abs:.3}");
    }

    // Specific check: rows 248040..248075 (around the special-token boundary)
    if rows >= 248075 {
        println!("--- special-token-band rows (248040..248075) ---");
        for r in 248040..248075usize {
            let (_, l2, m) = norms[r];
            println!("  row {r}: l2={l2:.3} max_abs={m:.3}");
        }
    }

    // Padded rows beyond the logical vocab 248044 — check for NaN/Inf
    let mut nan_rows = 0usize;
    let mut inf_rows = 0usize;
    let mut zero_rows = 0usize;
    for (r, l2, _) in &norms {
        if l2.is_nan() {
            nan_rows += 1;
            if nan_rows <= 5 {
                println!("  NaN row: {r}");
            }
        } else if l2.is_infinite() {
            inf_rows += 1;
            if inf_rows <= 5 {
                println!("  Inf row: {r}");
            }
        } else if *l2 == 0.0 {
            zero_rows += 1;
            if zero_rows <= 5 {
                println!("  zero row: {r}");
            }
        }
    }
    println!("NaN={nan_rows} Inf={inf_rows} zero={zero_rows}");
}
