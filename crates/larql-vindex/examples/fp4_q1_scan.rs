//! Experiment 26 / Q1 — Scan a LARQL vindex and measure the distribution of
//! per-sub-block max/min scale ratios. The DeepSeek-V4 FP4→FP8 lossless
//! dequant condition requires this ratio to stay below ~16 within each
//! FP8-sized block.
//!
//! The vindex stores per-feature vectors of length `hidden_size` (2560 on
//! Gemma 3 4B). DeepSeek's "FP8 block" is a 128×128 tile (16,384 elements)
//! which does not divide evenly into a 2560-wide feature vector, so we
//! report at two natural granularities:
//!
//! 1. **per-feature block**: one block = one whole feature vector
//!    (80 sub-blocks of 32 when hidden=2560). This is the natural unit of
//!    the per-feature vindex organisation and is the primary signal.
//! 2. **sub-feature tile**: one block = 16 sub-blocks = 512 elements,
//!    ⌊hidden/512⌋ tiles per feature (5 on Gemma 3 4B). Closer to the
//!    DeepSeek tile size; tighter bound, weaker signal.
//!
//! Scans `gate_vectors.bin`, `up_features.bin`, `down_features.bin`
//! directly via mmap, reinterprets bytes as f32 (dtype = "f32" per
//! `index.json`). No VectorIndex load is necessary.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release -p larql-vindex --example fp4_q1_scan -- \
//!   --vindex path/to/gemma3-4b-f16.vindex \
//!   --out    path/to/results.json
//! ```

use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

use memmap2::Mmap;
use rayon::prelude::*;
use serde_json::{json, Value};

const SUB_BLOCK_SIZE: usize = 32;
const DEFAULT_TILE_SUB_BLOCKS: usize = 16;
const COMPLIANCE_THRESHOLDS: &[f32] = &[2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0];
const TOP_K_OFFENDERS: usize = 32;

#[derive(Clone, Copy, PartialEq)]
enum Dtype {
    F32,
    F16,
    Bf16,
}

impl Dtype {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "f32" => Some(Dtype::F32),
            "f16" => Some(Dtype::F16),
            "bf16" => Some(Dtype::Bf16),
            _ => None,
        }
    }
    fn bytes_per_float(self) -> usize {
        match self {
            Dtype::F32 => 4,
            _ => 2,
        }
    }
}

/// `(projection_name, filename)` — scanner opportunistically skips missing files.
const PROJECTIONS: &[(&str, &str)] = &[
    ("gate", "gate_vectors.bin"),
    ("up", "up_features.bin"),
    ("down", "down_features.bin"),
];

#[derive(Debug, Clone, Default)]
struct Bucket {
    ratios: Vec<f32>,
    all_zero_blocks: u64,
    has_zero_blocks: u64,
}

impl Bucket {
    fn merge(&mut self, other: Bucket) {
        self.ratios.extend(other.ratios);
        self.all_zero_blocks += other.all_zero_blocks;
        self.has_zero_blocks += other.has_zero_blocks;
    }

    fn count(&self) -> usize {
        self.ratios.len() + self.all_zero_blocks as usize
    }

    fn summary(&self) -> Value {
        let mut sorted = self.ratios.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let percentile = |p: f64| -> f32 {
            if sorted.is_empty() {
                return f32::NAN;
            }
            let idx = (((sorted.len() - 1) as f64) * p).round() as usize;
            sorted[idx.min(sorted.len() - 1)]
        };
        let mean = if sorted.is_empty() {
            f32::NAN
        } else {
            sorted.iter().map(|&x| x as f64).sum::<f64>() as f32 / sorted.len() as f32
        };
        let total = self.count() as f64;
        let nonzero = sorted.len() as f64;
        let compliance: Value = COMPLIANCE_THRESHOLDS
            .iter()
            .map(|&t| {
                let under = sorted.iter().filter(|&&r| r < t).count() as f64;
                // Blocks with any all-zero: trivially lossless — count as compliant.
                let compliant_total = under + self.all_zero_blocks as f64;
                let frac = if total > 0.0 {
                    compliant_total / total
                } else {
                    0.0
                };
                json!({ "threshold": t, "compliant_fraction": frac })
            })
            .collect::<Vec<_>>()
            .into();
        json!({
            "total_blocks": total,
            "nonzero_ratio_blocks": nonzero,
            "all_zero_blocks": self.all_zero_blocks,
            "has_some_zero_blocks": self.has_zero_blocks,
            "mean": mean,
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "p999": percentile(0.999),
            "max": if sorted.is_empty() { f32::NAN } else { *sorted.last().unwrap() },
            "min": if sorted.is_empty() { f32::NAN } else { sorted[0] },
            "compliance": compliance,
        })
    }
}

#[derive(Debug, Clone, Default)]
struct Granularity {
    per_feature: Bucket,
    sub_feature_tile: Bucket,
}

/// Per-layer stats for one projection.
#[derive(Debug, Clone, Default)]
struct LayerStats {
    granularity: Granularity,
    /// Top offenders in this layer (per-feature granularity): (feat_idx, ratio).
    top_per_feature: Vec<(usize, f32)>,
    /// Top offenders in this layer (sub-feature tile granularity): (feat_idx, tile_idx, ratio).
    top_sub_feature: Vec<(usize, usize, f32)>,
}

/// Scan one feature vector (`hidden` f32s), record stats.
fn scan_feature_vector(
    vec: &[f32],
    feat_idx: usize,
    tile_sub_blocks: usize,
    gran: &mut Granularity,
    top_pf: &mut Vec<(usize, f32)>,
    top_sf: &mut Vec<(usize, usize, f32)>,
) {
    let hidden = vec.len();
    let sub_blocks = hidden / SUB_BLOCK_SIZE;
    if sub_blocks == 0 {
        return;
    }

    let mut scales = Vec::with_capacity(sub_blocks);
    for chunk in vec.chunks_exact(SUB_BLOCK_SIZE) {
        let s = chunk.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        scales.push(s);
    }

    // Per-feature block: one block covering all sub_blocks of this feature.
    record_block(&scales, &mut gran.per_feature, |r| {
        if let Some(r) = r {
            top_pf.push((feat_idx, r));
        }
    });

    // Sub-feature tiles: `tile_sub_blocks` contiguous sub-blocks each.
    for (tile_idx, tile_scales) in scales.chunks_exact(tile_sub_blocks).enumerate() {
        record_block(tile_scales, &mut gran.sub_feature_tile, |r| {
            if let Some(r) = r {
                top_sf.push((feat_idx, tile_idx, r));
            }
        });
    }
}

/// Compute the max/min(nonzero) ratio for one block of sub-block scales,
/// updating the bucket. `on_ratio` is called with Some(ratio) for non-zero
/// blocks and None for trivially-lossless all-zero blocks.
fn record_block(scales: &[f32], bucket: &mut Bucket, mut on_ratio: impl FnMut(Option<f32>)) {
    let mut mx = 0.0f32;
    let mut mn = f32::INFINITY;
    let mut any_zero = false;
    for &s in scales {
        if s > mx {
            mx = s;
        }
        if s > 0.0 && s < mn {
            mn = s;
        }
        if s == 0.0 {
            any_zero = true;
        }
    }
    if mx == 0.0 {
        bucket.all_zero_blocks += 1;
        on_ratio(None);
        return;
    }
    if any_zero {
        bucket.has_zero_blocks += 1;
    }
    let ratio = mx / mn;
    bucket.ratios.push(ratio);
    on_ratio(Some(ratio));
}

/// Keep only the top `k` largest values in a Vec, in descending order.
fn truncate_top<T: Clone>(v: &mut Vec<T>, k: usize, key: impl Fn(&T) -> f32) {
    v.sort_by(|a, b| {
        key(b)
            .partial_cmp(&key(a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    v.truncate(k);
}

fn log2_histogram(ratios: &[f32], max_bucket: usize) -> Vec<u64> {
    let mut buckets = vec![0u64; max_bucket + 1];
    for &r in ratios {
        if r <= 0.0 || !r.is_finite() {
            continue;
        }
        let b = r.log2().max(0.0) as usize;
        let idx = b.min(max_bucket);
        buckets[idx] += 1;
    }
    buckets
}

fn parse_args() -> (PathBuf, PathBuf, usize) {
    let args: Vec<String> = std::env::args().collect();
    let mut vindex = None;
    let mut out = None;
    let mut tile_sub_blocks = DEFAULT_TILE_SUB_BLOCKS;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--vindex" => {
                i += 1;
                vindex = Some(PathBuf::from(&args[i]));
            }
            "--out" => {
                i += 1;
                out = Some(PathBuf::from(&args[i]));
            }
            "--tile-sub-blocks" => {
                i += 1;
                tile_sub_blocks = args[i].parse().expect("integer");
            }
            _ => eprintln!("unknown arg: {}", args[i]),
        }
        i += 1;
    }
    let vindex = vindex.unwrap_or_else(|| {
        eprintln!("usage: fp4_q1_scan --vindex PATH --out PATH [--tile-sub-blocks N]");
        std::process::exit(1);
    });
    let out = out.unwrap_or_else(|| {
        eprintln!("usage: fp4_q1_scan --vindex PATH --out PATH [--tile-sub-blocks N]");
        std::process::exit(1);
    });
    (vindex, out, tile_sub_blocks)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (vindex_path, out_path, tile_sub_blocks) = parse_args();

    let index_json: Value =
        serde_json::from_str(&std::fs::read_to_string(vindex_path.join("index.json"))?)?;
    let num_layers = index_json["num_layers"].as_u64().ok_or("num_layers")? as usize;
    let hidden = index_json["hidden_size"].as_u64().ok_or("hidden_size")? as usize;
    let dtype_str = index_json["dtype"].as_str().unwrap_or("f32");
    let dtype =
        Dtype::from_str(dtype_str).ok_or_else(|| format!("unsupported dtype: {dtype_str}"))?;
    // Per-layer num_features (may vary — MoE / E2B-style layouts) and byte offsets.
    // The `layers` array in index.json is authoritative for gate_vectors.bin;
    // up_features.bin / down_features.bin use the same per-layer feature count.
    let layers_array = index_json["layers"]
        .as_array()
        .ok_or("index.json missing `layers` array")?;
    let layer_features: Vec<usize> = layers_array
        .iter()
        .map(|v| v["num_features"].as_u64().unwrap_or(0) as usize)
        .collect();
    let intermediate_max = layer_features.iter().copied().max().unwrap_or(0);
    let intermediate_total_floats: usize = layer_features.iter().sum::<usize>() * hidden;

    println!("== fp4_q1_scan ==");
    println!("  vindex       : {}", vindex_path.display());
    println!("  out          : {}", out_path.display());
    println!("  num_layers   : {num_layers}");
    println!("  hidden       : {hidden}");
    if layer_features.iter().all(|&n| n == intermediate_max) {
        println!("  intermediate : {intermediate_max} (uniform)");
    } else {
        let min = layer_features.iter().copied().min().unwrap_or(0);
        println!("  intermediate : {min}..{intermediate_max} (non-uniform)");
    }
    println!("  dtype        : {dtype_str}");
    println!("  sub_block    : {SUB_BLOCK_SIZE}");
    println!(
        "  tile (sub)   : {tile_sub_blocks} sub-blocks = {} elements",
        tile_sub_blocks * SUB_BLOCK_SIZE
    );
    println!();

    if !hidden.is_multiple_of(SUB_BLOCK_SIZE) {
        return Err(
            format!("hidden={hidden} is not divisible by sub-block {SUB_BLOCK_SIZE}").into(),
        );
    }

    // Results keyed: results[proj_idx][layer] = LayerStats. None if file missing.
    let mut proj_results: Vec<Option<Vec<LayerStats>>> = Vec::new();
    let mut scanned_projections: Vec<&str> = Vec::new();
    let bpf = dtype.bytes_per_float();
    let expected_total_bytes = intermediate_total_floats * bpf;

    // Pre-compute per-layer byte offsets and byte counts.
    let mut layer_byte_offsets: Vec<usize> = Vec::with_capacity(num_layers);
    let mut byte_cursor: usize = 0;
    for &nf in &layer_features {
        layer_byte_offsets.push(byte_cursor);
        byte_cursor += nf * hidden * bpf;
    }

    let t_total = Instant::now();
    for (proj_name, filename) in PROJECTIONS {
        let path = vindex_path.join(filename);
        if !path.exists() {
            println!("· skipping {proj_name} — {} not present", filename);
            proj_results.push(None);
            continue;
        }
        println!("→ scanning {proj_name} ({}, {dtype_str})", path.display());
        let file = File::open(&path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() != expected_total_bytes {
            return Err(format!(
                "{}: size {} != expected {}",
                filename,
                mmap.len(),
                expected_total_bytes
            )
            .into());
        }
        let bytes = &mmap[..];

        let t_proj = Instant::now();
        let layer_stats: Vec<LayerStats> = (0..num_layers)
            .into_par_iter()
            .map(|layer| {
                let nf = layer_features[layer];
                let layer_bytes_start = layer_byte_offsets[layer];
                let layer_bytes_len = nf * hidden * bpf;
                let layer_bytes = &bytes[layer_bytes_start..layer_bytes_start + layer_bytes_len];
                let floats: Vec<f32> = match dtype {
                    Dtype::F32 => {
                        // SAFETY: mmap'd region, f32 alignment matches u8 at read; no writes.
                        let view: &[f32] = unsafe {
                            std::slice::from_raw_parts(
                                layer_bytes.as_ptr() as *const f32,
                                nf * hidden,
                            )
                        };
                        view.to_vec()
                    }
                    Dtype::F16 => larql_models::quant::half::decode_f16(layer_bytes),
                    Dtype::Bf16 => larql_models::quant::half::decode_bf16(layer_bytes),
                };
                let mut stats = LayerStats::default();
                for feat in 0..nf {
                    let v = &floats[feat * hidden..(feat + 1) * hidden];
                    scan_feature_vector(
                        v,
                        feat,
                        tile_sub_blocks,
                        &mut stats.granularity,
                        &mut stats.top_per_feature,
                        &mut stats.top_sub_feature,
                    );
                    truncate_top(&mut stats.top_per_feature, TOP_K_OFFENDERS, |(_, r)| *r);
                    truncate_top(&mut stats.top_sub_feature, TOP_K_OFFENDERS, |(_, _, r)| *r);
                }
                stats
            })
            .collect();
        let elapsed = t_proj.elapsed();
        println!("  {proj_name} done in {:.1}s", elapsed.as_secs_f64());
        proj_results.push(Some(layer_stats));
        scanned_projections.push(proj_name);
    }
    println!(
        "all projections scanned in {:.1}s",
        t_total.elapsed().as_secs_f64()
    );

    // ── Aggregate ──────────────────────────────────────────────────────────
    let mut per_projection_agg: Vec<Granularity> = (0..PROJECTIONS.len())
        .map(|_| Granularity::default())
        .collect();
    let mut all_agg = Granularity::default();

    for (p, proj_layers) in proj_results.iter().enumerate() {
        let Some(proj_layers) = proj_layers else {
            continue;
        };
        for lstats in proj_layers {
            let mut copy = lstats.granularity.clone();
            per_projection_agg[p]
                .per_feature
                .merge(std::mem::take(&mut copy.per_feature));
            per_projection_agg[p]
                .sub_feature_tile
                .merge(std::mem::take(&mut copy.sub_feature_tile));
        }
    }

    for proj_gran in &per_projection_agg {
        all_agg
            .per_feature
            .ratios
            .extend(&proj_gran.per_feature.ratios);
        all_agg.per_feature.all_zero_blocks += proj_gran.per_feature.all_zero_blocks;
        all_agg.per_feature.has_zero_blocks += proj_gran.per_feature.has_zero_blocks;
        all_agg
            .sub_feature_tile
            .ratios
            .extend(&proj_gran.sub_feature_tile.ratios);
        all_agg.sub_feature_tile.all_zero_blocks += proj_gran.sub_feature_tile.all_zero_blocks;
        all_agg.sub_feature_tile.has_zero_blocks += proj_gran.sub_feature_tile.has_zero_blocks;
    }

    // Per-layer summary per projection.
    let mut per_layer_json: Vec<Value> = Vec::new();
    for (p, proj_layers) in proj_results.iter().enumerate() {
        let Some(proj_layers) = proj_layers else {
            continue;
        };
        let (proj_name, _) = PROJECTIONS[p];
        for (layer, lstats) in proj_layers.iter().enumerate() {
            per_layer_json.push(json!({
                "projection": proj_name,
                "layer": layer,
                "per_feature": lstats.granularity.per_feature.summary(),
                "sub_feature_tile": lstats.granularity.sub_feature_tile.summary(),
            }));
        }
    }

    // Worst offenders across the whole vindex (per granularity).
    let mut global_pf: Vec<(String, usize, usize, f32)> = Vec::new();
    let mut global_sf: Vec<(String, usize, usize, usize, f32)> = Vec::new();
    for (p, proj_layers) in proj_results.iter().enumerate() {
        let Some(proj_layers) = proj_layers else {
            continue;
        };
        let (proj_name, _) = PROJECTIONS[p];
        for (layer, lstats) in proj_layers.iter().enumerate() {
            for &(feat, r) in &lstats.top_per_feature {
                global_pf.push((proj_name.to_string(), layer, feat, r));
            }
            for &(feat, tile, r) in &lstats.top_sub_feature {
                global_sf.push((proj_name.to_string(), layer, feat, tile, r));
            }
        }
    }
    truncate_top(&mut global_pf, TOP_K_OFFENDERS, |(_, _, _, r)| *r);
    truncate_top(&mut global_sf, TOP_K_OFFENDERS, |(_, _, _, _, r)| *r);

    // ── Write JSON ─────────────────────────────────────────────────────────
    let histogram_pf = log2_histogram(&all_agg.per_feature.ratios, 24);
    let histogram_sf = log2_histogram(&all_agg.sub_feature_tile.ratios, 24);

    let projection_summary: Vec<Value> = per_projection_agg
        .iter()
        .enumerate()
        .filter(|(p, _)| proj_results[*p].is_some())
        .map(|(p, g)| {
            json!({
                "projection": PROJECTIONS[p].0,
                "per_feature": g.per_feature.summary(),
                "sub_feature_tile": g.sub_feature_tile.summary(),
            })
        })
        .collect();

    let report = json!({
        "experiment": "26_fp4_quantisation",
        "question":   "Q1",
        "config": {
            "vindex": vindex_path.display().to_string(),
            "num_layers": num_layers,
            "hidden": hidden,
            "layer_features": layer_features,
            "intermediate_max": intermediate_max,
            "dtype": dtype_str,
            "scanned_projections": scanned_projections,
            "sub_block_size": SUB_BLOCK_SIZE,
            "per_feature_sub_blocks": hidden / SUB_BLOCK_SIZE,
            "sub_feature_tile_sub_blocks": tile_sub_blocks,
            "sub_feature_tile_elements": tile_sub_blocks * SUB_BLOCK_SIZE,
            "compliance_thresholds": COMPLIANCE_THRESHOLDS,
        },
        "aggregate_all_projections": {
            "per_feature": all_agg.per_feature.summary(),
            "sub_feature_tile": all_agg.sub_feature_tile.summary(),
        },
        "per_projection": projection_summary,
        "per_layer_per_projection": per_layer_json,
        "log2_histogram_per_feature":      histogram_pf,
        "log2_histogram_sub_feature_tile": histogram_sf,
        "worst_offenders_per_feature": global_pf.iter().map(|(proj, layer, feat, r)| json!({
            "projection": proj, "layer": layer, "feature": feat, "ratio": r,
        })).collect::<Vec<_>>(),
        "worst_offenders_sub_feature_tile": global_sf.iter().map(|(proj, layer, feat, tile, r)| json!({
            "projection": proj, "layer": layer, "feature": feat, "tile": tile, "ratio": r,
        })).collect::<Vec<_>>(),
    });

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&out_path, serde_json::to_string_pretty(&report)?)?;
    println!();
    println!("→ wrote {}", out_path.display());

    // ── Short stdout summary ───────────────────────────────────────────────
    println!();
    println!("== aggregate (all projections) ==");
    let pf = &all_agg.per_feature;
    let sf = &all_agg.sub_feature_tile;
    let pf_sum = pf.summary();
    let sf_sum = sf.summary();
    println!(
        "per_feature      : total={:>10} p50={:.3} p95={:.3} p99={:.3} p99.9={:.3} max={:.3}",
        pf_sum["total_blocks"],
        pf_sum["p50"],
        pf_sum["p95"],
        pf_sum["p99"],
        pf_sum["p999"],
        pf_sum["max"]
    );
    println!(
        "sub_feature_tile : total={:>10} p50={:.3} p95={:.3} p99={:.3} p99.9={:.3} max={:.3}",
        sf_sum["total_blocks"],
        sf_sum["p50"],
        sf_sum["p95"],
        sf_sum["p99"],
        sf_sum["p999"],
        sf_sum["max"]
    );
    println!();
    println!("== compliance fraction at threshold ==");
    println!("threshold   per_feature   sub_feature_tile");
    let pf_comp = pf_sum["compliance"].as_array().unwrap();
    let sf_comp = sf_sum["compliance"].as_array().unwrap();
    for (a, b) in pf_comp.iter().zip(sf_comp.iter()) {
        let t = a["threshold"].as_f64().unwrap();
        let af = a["compliant_fraction"].as_f64().unwrap();
        let bf = b["compliant_fraction"].as_f64().unwrap();
        println!("  {:>6.1}       {:>6.4}         {:>6.4}", t, af, bf);
    }

    Ok(())
}

fn _assert_send_sync()
where
    LayerStats: Send + Sync,
{
}
