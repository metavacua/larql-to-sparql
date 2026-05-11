//! Q1 compliance scan — measures the FP4/FP8 block-storage
//! distributional property on a vindex without quantising anything.
//!
//! Pure library: takes a vindex directory path + a `ScanConfig`,
//! returns a `VindexComplianceReport`. No I/O beyond mmap'ing the
//! projection files. No side effects.
//!
//! Consumers:
//! - `fp4_q1_scan` example binary (thin CLI wrapper).
//! - `quant::convert::vindex_to_fp4` (self-policing gate — projections
//!   targeted for FP4 that fall below the compliance floor get
//!   downgraded to the manifest's `fallback_precision`).
//!
//! Reports at two granularities:
//! - **per-feature block**: one feature vector = one block (natural
//!   unit of the per-feature vindex organisation).
//! - **sub-feature tile**: 16 sub-blocks per tile = 512 elements,
//!   multiple tiles per feature (closer to DeepSeek's 128×128).
//!
//! See `docs/specs/fp4-format-spec.md` §5 for the byte layout these
//! scales correspond to, and `~/chris-source/chris-experiments/routing/26_fp4_quantisation/SPEC.md`
//! for the theoretical framing.

use std::path::Path;

use memmap2::Mmap;
use rayon::prelude::*;
use serde_json::Value;

use crate::error::VindexError;
use crate::format::filenames::*;

/// Fixed block geometry for v1. `sub_block` matches MXFP4's 1×32.
pub const SUB_BLOCK_SIZE: usize = 32;

/// Sub-block count for the secondary "tile" granularity the scanner
/// reports (tile = `DEFAULT_TILE_SUB_BLOCKS * SUB_BLOCK_SIZE`
/// elements). `16 * 32 = 512`, matching the tile size pinned in
/// `fp4-format-spec.md` §4 as the chosen block granularity.
pub const DEFAULT_TILE_SUB_BLOCKS: usize = 16;

/// Canonical compliance thresholds Q1 reports always include.
/// Consumers can add custom thresholds; these are always measured.
pub const DEFAULT_COMPLIANCE_THRESHOLDS: &[f32] = &[2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0];

/// Default top-K offenders recorded per projection per granularity.
pub const DEFAULT_TOP_K_OFFENDERS: usize = 32;

/// Projections scanned. Missing files are skipped (not an error).
pub const PROJECTIONS: &[(&str, &str)] = &[
    ("gate", GATE_VECTORS_BIN),
    ("up", UP_FEATURES_BIN),
    ("down", DOWN_FEATURES_BIN),
];

/// Source dtype on disk. Q1 is always run on raw-float inputs; FP4
/// vindexes don't need a scan — they're the output of one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F16,
    Bf16,
}

impl Dtype {
    pub fn from_index_json(s: &str) -> Result<Self, String> {
        match s {
            "f32" => Ok(Dtype::F32),
            "f16" => Ok(Dtype::F16),
            "bf16" => Ok(Dtype::Bf16),
            _ => Err(format!("unsupported dtype for scan: {s}")),
        }
    }
    pub fn bytes_per_float(self) -> usize {
        match self {
            Dtype::F32 => 4,
            _ => 2,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Dtype::F32 => "f32",
            Dtype::F16 => "f16",
            Dtype::Bf16 => "bf16",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScanConfig {
    pub tile_sub_blocks: usize,
    pub compliance_thresholds: Vec<f32>,
    pub top_k_offenders: usize,
}

impl Default for ScanConfig {
    fn default() -> Self {
        Self {
            tile_sub_blocks: DEFAULT_TILE_SUB_BLOCKS,
            compliance_thresholds: DEFAULT_COMPLIANCE_THRESHOLDS.to_vec(),
            top_k_offenders: DEFAULT_TOP_K_OFFENDERS,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Bucket {
    pub ratios: Vec<f32>,
    pub all_zero_blocks: u64,
    pub has_zero_blocks: u64,
}

impl Bucket {
    pub fn count(&self) -> u64 {
        self.ratios.len() as u64 + self.all_zero_blocks
    }

    pub fn compliance_at(&self, threshold: f32) -> f64 {
        let total = self.count() as f64;
        if total == 0.0 {
            return 0.0;
        }
        let under = self.ratios.iter().filter(|&&r| r < threshold).count() as f64;
        (under + self.all_zero_blocks as f64) / total
    }

    fn percentile(sorted: &[f32], p: f64) -> f32 {
        if sorted.is_empty() {
            return f32::NAN;
        }
        let idx = (((sorted.len() - 1) as f64) * p).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    pub fn quantiles(&self) -> BucketQuantiles {
        let mut sorted = self.ratios.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        BucketQuantiles {
            total_blocks: self.count(),
            nonzero_ratio_blocks: sorted.len() as u64,
            all_zero_blocks: self.all_zero_blocks,
            has_some_zero_blocks: self.has_zero_blocks,
            mean: if sorted.is_empty() {
                f32::NAN
            } else {
                sorted.iter().map(|&x| x as f64).sum::<f64>() as f32 / sorted.len() as f32
            },
            p50: Self::percentile(&sorted, 0.50),
            p95: Self::percentile(&sorted, 0.95),
            p99: Self::percentile(&sorted, 0.99),
            p999: Self::percentile(&sorted, 0.999),
            min: sorted.first().copied().unwrap_or(f32::NAN),
            max: sorted.last().copied().unwrap_or(f32::NAN),
        }
    }

    fn merge_from(&mut self, other: &Bucket) {
        self.ratios.extend(&other.ratios);
        self.all_zero_blocks += other.all_zero_blocks;
        self.has_zero_blocks += other.has_zero_blocks;
    }
}

#[derive(Debug, Clone)]
pub struct BucketQuantiles {
    pub total_blocks: u64,
    pub nonzero_ratio_blocks: u64,
    pub all_zero_blocks: u64,
    pub has_some_zero_blocks: u64,
    pub mean: f32,
    pub p50: f32,
    pub p95: f32,
    pub p99: f32,
    pub p999: f32,
    pub min: f32,
    pub max: f32,
}

#[derive(Debug, Clone, Default)]
pub struct GranularityStats {
    pub per_feature: Bucket,
    pub sub_feature_tile: Bucket,
}

#[derive(Debug, Clone, Default)]
pub struct LayerStats {
    pub granularity: GranularityStats,
    pub top_per_feature: Vec<(usize, f32)>,
    pub top_sub_feature: Vec<(usize, usize, f32)>,
}

#[derive(Debug, Clone)]
pub struct ProjectionReport {
    pub name: String,
    pub layers: Vec<LayerStats>,
    pub aggregate: GranularityStats,
}

impl ProjectionReport {
    pub fn compliance_at(&self, threshold: f32) -> f64 {
        self.aggregate.per_feature.compliance_at(threshold)
    }
    pub fn sub_feature_compliance_at(&self, threshold: f32) -> f64 {
        self.aggregate.sub_feature_tile.compliance_at(threshold)
    }
}

/// (`threshold`, `compliant_fraction`) pair. Used in the sidecar JSON.
#[derive(Debug, Clone)]
pub struct ComplianceThreshold {
    pub threshold: f32,
    pub compliant_fraction: f64,
}

#[derive(Debug, Clone)]
pub struct VindexComplianceReport {
    pub config: ScanConfig,
    pub num_layers: usize,
    pub hidden: usize,
    pub layer_features: Vec<usize>,
    pub dtype: Dtype,
    pub projections: Vec<ProjectionReport>,
    pub aggregate: GranularityStats,
}

impl VindexComplianceReport {
    /// Find a projection report by name; None if this projection was
    /// skipped (file absent) during the scan.
    pub fn projection(&self, name: &str) -> Option<&ProjectionReport> {
        self.projections.iter().find(|p| p.name == name)
    }

    /// Per-projection compliance at the given ratio threshold.
    pub fn per_projection_compliance(&self, threshold: f32) -> Vec<(String, f64)> {
        self.projections
            .iter()
            .map(|p| (p.name.clone(), p.compliance_at(threshold)))
            .collect()
    }

    /// Canonical JSON dump — matches the shape the `fp4_q1_scan`
    /// example emits so sidecar consumers don't break across the
    /// example → library promotion.
    pub fn to_json(&self) -> Value {
        use serde_json::json;
        let thresholds = &self.config.compliance_thresholds;

        fn bucket_json(b: &Bucket, thresholds: &[f32]) -> Value {
            let q = b.quantiles();
            let compliance: Vec<Value> = thresholds
                .iter()
                .map(|&t| {
                    json!({
                        "threshold": t,
                        "compliant_fraction": b.compliance_at(t),
                    })
                })
                .collect();
            json!({
                "total_blocks": q.total_blocks as f64,
                "nonzero_ratio_blocks": q.nonzero_ratio_blocks as f64,
                "all_zero_blocks": q.all_zero_blocks,
                "has_some_zero_blocks": q.has_some_zero_blocks,
                "mean": q.mean,
                "p50": q.p50, "p95": q.p95, "p99": q.p99, "p999": q.p999,
                "min": q.min, "max": q.max,
                "compliance": compliance,
            })
        }

        let per_projection: Vec<Value> = self
            .projections
            .iter()
            .map(|p| {
                json!({
                    "projection": p.name,
                    "per_feature": bucket_json(&p.aggregate.per_feature, thresholds),
                    "sub_feature_tile": bucket_json(&p.aggregate.sub_feature_tile, thresholds),
                })
            })
            .collect();

        let mut per_layer_json: Vec<Value> = Vec::new();
        for p in &self.projections {
            for (layer, l) in p.layers.iter().enumerate() {
                per_layer_json.push(json!({
                    "projection": p.name,
                    "layer": layer,
                    "per_feature": bucket_json(&l.granularity.per_feature, thresholds),
                    "sub_feature_tile": bucket_json(&l.granularity.sub_feature_tile, thresholds),
                }));
            }
        }

        let mut pf: Vec<(String, usize, usize, f32)> = Vec::new();
        let mut sf: Vec<(String, usize, usize, usize, f32)> = Vec::new();
        for p in &self.projections {
            for (layer, l) in p.layers.iter().enumerate() {
                for &(feat, r) in &l.top_per_feature {
                    pf.push((p.name.clone(), layer, feat, r));
                }
                for &(feat, tile, r) in &l.top_sub_feature {
                    sf.push((p.name.clone(), layer, feat, tile, r));
                }
            }
        }
        pf.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
        pf.truncate(self.config.top_k_offenders);
        sf.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
        sf.truncate(self.config.top_k_offenders);

        json!({
            "config": {
                "num_layers": self.num_layers,
                "hidden": self.hidden,
                "layer_features": self.layer_features,
                "intermediate_max": self.layer_features.iter().copied().max().unwrap_or(0),
                "dtype": self.dtype.as_str(),
                "sub_block_size": SUB_BLOCK_SIZE,
                "per_feature_sub_blocks": self.hidden / SUB_BLOCK_SIZE,
                "sub_feature_tile_sub_blocks": self.config.tile_sub_blocks,
                "sub_feature_tile_elements": self.config.tile_sub_blocks * SUB_BLOCK_SIZE,
                "compliance_thresholds": thresholds,
            },
            "aggregate_all_projections": {
                "per_feature": bucket_json(&self.aggregate.per_feature, thresholds),
                "sub_feature_tile": bucket_json(&self.aggregate.sub_feature_tile, thresholds),
            },
            "per_projection": per_projection,
            "per_layer_per_projection": per_layer_json,
            "worst_offenders_per_feature": pf.iter().map(|(proj, layer, feat, r)| json!({
                "projection": proj, "layer": layer, "feature": feat, "ratio": r,
            })).collect::<Vec<_>>(),
            "worst_offenders_sub_feature_tile": sf.iter().map(|(proj, layer, feat, tile, r)| json!({
                "projection": proj, "layer": layer, "feature": feat, "tile": tile, "ratio": r,
            })).collect::<Vec<_>>(),
        })
    }
}

// ── Scan kernels ──────────────────────────────────────────────────────

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

fn scan_feature_vector(
    vec: &[f32],
    feat_idx: usize,
    tile_sub_blocks: usize,
    gran: &mut GranularityStats,
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
    record_block(&scales, &mut gran.per_feature, |r| {
        if let Some(r) = r {
            top_pf.push((feat_idx, r));
        }
    });
    for (tile_idx, tile_scales) in scales.chunks_exact(tile_sub_blocks).enumerate() {
        record_block(tile_scales, &mut gran.sub_feature_tile, |r| {
            if let Some(r) = r {
                top_sf.push((feat_idx, tile_idx, r));
            }
        });
    }
}

fn truncate_top<T: Clone>(v: &mut Vec<T>, k: usize, key: impl Fn(&T) -> f32) {
    v.sort_by(|a, b| {
        key(b)
            .partial_cmp(&key(a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    v.truncate(k);
}

// ── Public entry points ───────────────────────────────────────────────

pub fn scan_projection(
    path: &Path,
    name: &str,
    dtype: Dtype,
    layer_features: &[usize],
    hidden: usize,
    config: &ScanConfig,
) -> Result<ProjectionReport, VindexError> {
    if !hidden.is_multiple_of(SUB_BLOCK_SIZE) {
        return Err(VindexError::Parse(format!(
            "hidden {hidden} not divisible by sub-block size {SUB_BLOCK_SIZE}"
        )));
    }
    let bpf = dtype.bytes_per_float();
    let expected_bytes: usize = layer_features.iter().sum::<usize>() * hidden * bpf;

    let file = std::fs::File::open(path)
        .map_err(|e| VindexError::Parse(format!("open {}: {e}", path.display())))?;
    let mmap = unsafe { Mmap::map(&file).map_err(|e| VindexError::Parse(format!("mmap: {e}")))? };
    if mmap.len() != expected_bytes {
        return Err(VindexError::Parse(format!(
            "{}: size {} != expected {}",
            path.display(),
            mmap.len(),
            expected_bytes
        )));
    }
    let bytes = &mmap[..];

    let mut layer_byte_offsets = Vec::with_capacity(layer_features.len());
    let mut cursor = 0usize;
    for &nf in layer_features {
        layer_byte_offsets.push(cursor);
        cursor += nf * hidden * bpf;
    }

    let top_k = config.top_k_offenders;
    let tile_sub_blocks = config.tile_sub_blocks;

    let layer_stats: Vec<LayerStats> = (0..layer_features.len())
        .into_par_iter()
        .map(|layer| {
            let nf = layer_features[layer];
            let start = layer_byte_offsets[layer];
            let len = nf * hidden * bpf;
            let layer_bytes = &bytes[start..start + len];
            let floats: Vec<f32> = match dtype {
                Dtype::F32 => {
                    // SAFETY: mmap'd region, f32 alignment matches u8.
                    let view: &[f32] = unsafe {
                        std::slice::from_raw_parts(layer_bytes.as_ptr() as *const f32, nf * hidden)
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
                truncate_top(&mut stats.top_per_feature, top_k, |(_, r)| *r);
                truncate_top(&mut stats.top_sub_feature, top_k, |(_, _, r)| *r);
            }
            stats
        })
        .collect();

    let mut aggregate = GranularityStats::default();
    for l in &layer_stats {
        aggregate.per_feature.merge_from(&l.granularity.per_feature);
        aggregate
            .sub_feature_tile
            .merge_from(&l.granularity.sub_feature_tile);
    }

    Ok(ProjectionReport {
        name: name.to_string(),
        layers: layer_stats,
        aggregate,
    })
}

pub fn scan_vindex(
    vindex_dir: &Path,
    config: &ScanConfig,
) -> Result<VindexComplianceReport, VindexError> {
    let index_json: Value = serde_json::from_str(
        &std::fs::read_to_string(vindex_dir.join(INDEX_JSON))
            .map_err(|e| VindexError::Parse(format!("read index.json: {e}")))?,
    )
    .map_err(|e| VindexError::Parse(format!("parse index.json: {e}")))?;

    let num_layers = index_json["num_layers"]
        .as_u64()
        .ok_or_else(|| VindexError::Parse("index.json: missing num_layers".into()))?
        as usize;
    let hidden = index_json["hidden_size"]
        .as_u64()
        .ok_or_else(|| VindexError::Parse("index.json: missing hidden_size".into()))?
        as usize;
    let dtype_str = index_json["dtype"].as_str().unwrap_or("f32");
    let dtype = Dtype::from_index_json(dtype_str).map_err(VindexError::Parse)?;

    let layers_array = index_json["layers"]
        .as_array()
        .ok_or_else(|| VindexError::Parse("index.json: missing layers[]".into()))?;
    let layer_features: Vec<usize> = layers_array
        .iter()
        .map(|v| v["num_features"].as_u64().unwrap_or(0) as usize)
        .collect();

    let mut projections = Vec::new();
    for (name, filename) in PROJECTIONS {
        let path = vindex_dir.join(filename);
        if !path.exists() {
            continue;
        }
        projections.push(scan_projection(
            &path,
            name,
            dtype,
            &layer_features,
            hidden,
            config,
        )?);
    }

    let mut aggregate = GranularityStats::default();
    for p in &projections {
        aggregate.per_feature.merge_from(&p.aggregate.per_feature);
        aggregate
            .sub_feature_tile
            .merge_from(&p.aggregate.sub_feature_tile);
    }

    Ok(VindexComplianceReport {
        config: config.clone(),
        num_layers,
        hidden,
        layer_features,
        dtype,
        projections,
        aggregate,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_compliance_fraction() {
        let b = Bucket {
            ratios: vec![1.5, 2.0, 3.0, 18.0],
            all_zero_blocks: 1,
            ..Default::default()
        };
        // total = 5; under 16 = 3 non-zero + 1 all-zero = 4; 4/5 = 0.8.
        assert!((b.compliance_at(16.0) - 0.8).abs() < 1e-9);
        assert!((b.compliance_at(20.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn bucket_quantiles_empty_ok() {
        let b = Bucket::default();
        let q = b.quantiles();
        assert_eq!(q.total_blocks, 0);
        assert!(q.mean.is_nan());
    }

    #[test]
    fn config_defaults_pin_geometry() {
        let c = ScanConfig::default();
        assert_eq!(c.tile_sub_blocks, 16);
        assert_eq!(c.top_k_offenders, 32);
        assert_eq!(c.compliance_thresholds.len(), 8);
    }
}
