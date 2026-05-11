//! Sanity check: round-trip a few feature vectors through a converted
//! FP4 vindex and compare to the original. Independent verification that
//! fp4_convert didn't silently corrupt anything at the format or codec
//! level.
//!
//! Reports per-feature max, median, and RMS absolute error for a handful
//! of sample features across gate/up/down and across layers.
//!
//! Usage:
//! ```
//! cargo run --release -p larql-vindex --example fp4_verify -- \
//!   --src output/gemma3-4b-f16.vindex \
//!   --fp4 output/gemma3-4b-fp4.vindex
//! ```

use std::path::{Path, PathBuf};

use larql_models::quant::fp4_block::{
    decode_fp4_feature, decode_fp8_feature, fp4_feature_bytes, fp8_feature_bytes,
};
use larql_vindex::{Precision, VindexConfig};

fn parse_args() -> (PathBuf, PathBuf) {
    let args: Vec<String> = std::env::args().collect();
    let mut src = None;
    let mut fp4 = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--src" => {
                i += 1;
                src = Some(PathBuf::from(&args[i]));
            }
            "--fp4" => {
                i += 1;
                fp4 = Some(PathBuf::from(&args[i]));
            }
            _ => eprintln!("unknown arg: {}", args[i]),
        }
        i += 1;
    }
    (src.expect("--src"), fp4.expect("--fp4"))
}

fn load_source_feature(
    vindex_dir: &Path,
    proj_file: &str,
    dtype: &str,
    layer: usize,
    feat: usize,
    hidden: usize,
    per_layer_features: &[usize],
) -> Vec<f32> {
    let bpf = if dtype == "f32" { 4 } else { 2 };
    let mut cursor = 0usize;
    for (li, &n) in per_layer_features.iter().enumerate() {
        if li == layer {
            let feat_offset = cursor + feat * hidden * bpf;
            let feat_bytes = hidden * bpf;
            let bytes = &std::fs::read(vindex_dir.join(proj_file)).unwrap()
                [feat_offset..feat_offset + feat_bytes];
            return match dtype {
                "f32" => {
                    let v: &[f32] =
                        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, hidden) };
                    v.to_vec()
                }
                "f16" => larql_models::quant::half::decode_f16(bytes),
                "bf16" => larql_models::quant::half::decode_bf16(bytes),
                _ => panic!("unsupported source dtype {dtype}"),
            };
        }
        cursor += n * hidden * bpf;
    }
    panic!("layer {layer} out of range")
}

fn load_fp4_feature(
    vindex_dir: &Path,
    file: &str,
    precision: Precision,
    layer: usize,
    feat: usize,
    hidden: usize,
    per_layer_features: &[usize],
) -> Vec<f32> {
    let (per_feat, is_fp4) = match precision {
        Precision::Fp4 => (fp4_feature_bytes(hidden), true),
        Precision::Fp8 => (fp8_feature_bytes(hidden), false),
        _ => panic!("expected fp4 or fp8"),
    };
    let bytes = std::fs::read(vindex_dir.join(file)).unwrap();
    let mut cursor = 0usize;
    for (li, &n) in per_layer_features.iter().enumerate() {
        if li == layer {
            let start = cursor + feat * per_feat;
            let slice = &bytes[start..start + per_feat];
            let mut out = vec![0.0f32; hidden];
            if is_fp4 {
                decode_fp4_feature(slice, &mut out);
            } else {
                decode_fp8_feature(slice, &mut out);
            }
            return out;
        }
        cursor += n * per_feat;
    }
    panic!("layer {layer} out of range")
}

fn feature_errors(src: &[f32], decoded: &[f32]) -> (f32, f32, f32) {
    assert_eq!(src.len(), decoded.len());
    let mut max = 0.0f32;
    let mut sum = 0.0f32;
    let mut sum_sq = 0.0f32;
    for (&a, &b) in src.iter().zip(decoded.iter()) {
        let e = (a - b).abs();
        if e > max {
            max = e;
        }
        sum += e;
        sum_sq += e * e;
    }
    let n = src.len() as f32;
    (max, sum / n, (sum_sq / n).sqrt())
}

fn main() {
    let (src_dir, fp4_dir) = parse_args();

    let src_config: VindexConfig =
        serde_json::from_str(&std::fs::read_to_string(src_dir.join("index.json")).unwrap())
            .unwrap();
    let fp4_config: VindexConfig =
        serde_json::from_str(&std::fs::read_to_string(fp4_dir.join("index.json")).unwrap())
            .unwrap();
    let fp4_cfg = fp4_config.fp4.expect("no fp4 manifest in target");

    let hidden = src_config.hidden_size;
    let num_layers = src_config.num_layers;
    let per_layer_features: Vec<usize> = src_config.layers.iter().map(|l| l.num_features).collect();
    let src_dtype_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(src_dir.join("index.json")).unwrap())
            .unwrap();
    let src_dtype = src_dtype_json["dtype"]
        .as_str()
        .unwrap_or("f32")
        .to_string();

    println!("== fp4_verify ==");
    println!("  src    : {} ({src_dtype})", src_dir.display());
    println!("  fp4    : {}", fp4_dir.display());
    println!("  hidden : {hidden}");
    println!();

    let projections = [
        ("gate", "gate_vectors.bin", &fp4_cfg.projections.gate),
        ("up", "up_features.bin", &fp4_cfg.projections.up),
        ("down", "down_features.bin", &fp4_cfg.projections.down),
    ];

    // Sample a few (layer, feat) pairs across layers.
    let sample_layers = [
        0usize,
        num_layers / 4,
        num_layers / 2,
        3 * num_layers / 4,
        num_layers - 1,
    ];
    let sample_feats = [0usize, 1000, 5000, 9000];

    for (proj_name, src_file, proj) in projections.iter() {
        println!(
            "→ {proj_name} (source {src_file}, decoded {} ({:?}))",
            proj.file, proj.precision
        );

        let mut max_over_samples = 0.0f32;
        let mut sum_rms = 0.0f32;
        let mut count = 0;

        for &layer in &sample_layers {
            for &feat in &sample_feats {
                if feat >= per_layer_features[layer] {
                    continue;
                }
                let src = load_source_feature(
                    &src_dir,
                    src_file,
                    &src_dtype,
                    layer,
                    feat,
                    hidden,
                    &per_layer_features,
                );
                let dec = load_fp4_feature(
                    &fp4_dir,
                    &proj.file,
                    proj.precision,
                    layer,
                    feat,
                    hidden,
                    &per_layer_features,
                );
                let (max, mean, rms) = feature_errors(&src, &dec);
                let block_max = src.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                if max > max_over_samples {
                    max_over_samples = max;
                }
                sum_rms += rms;
                count += 1;
                println!(
                    "    L{layer:>2} f{feat:>5}: max_err={max:.4e} mean_err={mean:.4e} rms={rms:.4e}  block_max={block_max:.3}  max/block_max={:.2}%",
                    100.0 * max / block_max
                );
            }
        }
        println!(
            "  summary: max {:.4e}  mean rms {:.4e}  n={count}",
            max_over_samples,
            sum_rms / count as f32
        );
        println!();
    }
}
