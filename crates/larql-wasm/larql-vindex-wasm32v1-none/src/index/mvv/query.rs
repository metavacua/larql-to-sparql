//! The MVV query kernel: a total (panic-free, UB-free) gate-KNN over slices.
//!
//! No `unsafe`, no `.unwrap()`, no recursion, no locks, no `ndarray`/BLAS, and
//! no `libm`/`FloatExt` — just bounded iteration and core f32 arithmetic. This
//! is the L0 floor of the dialect-stratified ladder (see the spec); higher
//! impls swap the matvec behind the same semantics + totality contract.
#[allow(unused_imports)]
use alloc::vec::Vec;

use crate::config::dtype::StorageDtype;
use crate::index::types::FeatureMeta;

use super::descriptor::MvvDescriptor;
use super::error::MvvError;

/// A minimal, portable, total vindex: a validated descriptor + the gate-vector
/// blob, plus optional per-layer/per-feature metadata.
pub struct MvvIndex {
    desc: MvvDescriptor,
    blob: Vec<u8>,
    /// `meta[layer][feature]` — empty outer vec means "no metadata loaded".
    meta: Vec<Vec<Option<FeatureMeta>>>,
}

impl MvvIndex {
    /// Build from an already-parsed descriptor + blob. Validates the blob
    /// against the descriptor's invariants (total: typed error on mismatch).
    pub fn new(desc: MvvDescriptor, blob: Vec<u8>) -> Result<Self, MvvError> {
        desc.validate(blob.len())?;
        Ok(Self {
            desc,
            blob,
            meta: Vec::new(),
        })
    }

    /// Build from raw `index.json` bytes + the gate blob. Total on all inputs.
    pub fn from_index_json(index_json: &[u8], blob: Vec<u8>) -> Result<Self, MvvError> {
        let desc = MvvDescriptor::from_index_json(index_json)?;
        Self::new(desc, blob)
    }

    /// Attach optional metadata (`meta[layer][feature]`).
    pub fn with_metadata(mut self, meta: Vec<Vec<Option<FeatureMeta>>>) -> Self {
        self.meta = meta;
        self
    }

    pub fn num_layers(&self) -> usize {
        self.desc.num_layers
    }
    pub fn hidden_size(&self) -> usize {
        self.desc.hidden_size
    }

    /// Feature count for `layer`. Total: 0 if out of range.
    pub fn num_features(&self, layer: usize) -> usize {
        self.desc.layers.get(layer).map_or(0, |l| l.num_features)
    }

    /// Metadata for one feature. Total: `None` if out of range or absent.
    pub fn feature_meta(&self, layer: usize, feature: usize) -> Option<&FeatureMeta> {
        self.meta
            .get(layer)
            .and_then(|fs| fs.get(feature))
            .and_then(|m| m.as_ref())
    }

    /// Top-K features at `layer` by |gate·residual|. Total on all inputs.
    pub fn gate_knn(
        &self,
        layer: usize,
        residual: &[f32],
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>, MvvError> {
        let n = self.num_features(layer);
        self.gate_knn_range(layer, residual, 0, n, top_k)
    }

    /// Top-K within a feature range `[feat_start, feat_end)` (MoE-expert scope).
    /// Total on all inputs.
    pub fn gate_knn_expert(
        &self,
        layer: usize,
        residual: &[f32],
        feat_start: usize,
        feat_end: usize,
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>, MvvError> {
        self.gate_knn_range(layer, residual, feat_start, feat_end, top_k)
    }

    fn gate_knn_range(
        &self,
        layer: usize,
        residual: &[f32],
        feat_start: usize,
        feat_end: usize,
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>, MvvError> {
        let hidden = self.desc.hidden_size;
        if residual.len() != hidden {
            return Err(MvvError::DimMismatch {
                expected: hidden,
                actual: residual.len(),
            });
        }
        let (floats, num_features) = self.layer_floats(layer)?;
        if feat_start > feat_end || feat_end > num_features {
            return Err(MvvError::FeatureRange {
                start: feat_start,
                end: feat_end,
                num_features,
            });
        }
        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(feat_end - feat_start);
        for f in feat_start..feat_end {
            // base = f * hidden; in-bounds because floats.len() == num_features *
            // hidden and f < num_features. `.get` keeps it total regardless.
            let base = f.saturating_mul(hidden);
            let end = base.saturating_add(hidden);
            let row = match floats.get(base..end) {
                Some(r) => r,
                None => {
                    return Err(MvvError::LayerLengthMismatch {
                        layer,
                        expected: num_features.saturating_mul(hidden),
                        declared: floats.len(),
                    })
                }
            };
            let score: f32 = row.iter().zip(residual.iter()).map(|(a, b)| a * b).sum();
            scored.push((f, score));
        }
        // Top-K by descending |score|, total order (NaN-safe via total_cmp).
        scored.sort_by(|a, b| fabs(b.1).total_cmp(&fabs(a.1)));
        scored.truncate(top_k);
        Ok(scored)
    }

    /// Decode a layer's gate matrix to f32 — checked, alignment-independent,
    /// no `unsafe`. Returns `(floats, num_features)`.
    fn layer_floats(&self, layer: usize) -> Result<(Vec<f32>, usize), MvvError> {
        let l = self
            .desc
            .layers
            .get(layer)
            .ok_or(MvvError::LayerOutOfRange {
                layer,
                num_layers: self.desc.num_layers,
            })?;
        let off = l.offset as usize;
        let len = l.length as usize;
        let end = off.checked_add(len).ok_or(MvvError::BlobLengthMismatch {
            expected: usize::MAX,
            actual: self.blob.len(),
        })?;
        let slice = self.blob.get(off..end).ok_or(MvvError::BlobLengthMismatch {
            expected: end,
            actual: self.blob.len(),
        })?;
        let floats = decode_checked(slice, self.desc.dtype, layer)?;
        let expected = l
            .num_features
            .checked_mul(self.desc.hidden_size)
            .ok_or(MvvError::LayerLengthMismatch {
                layer,
                expected: usize::MAX,
                declared: floats.len(),
            })?;
        if floats.len() != expected {
            return Err(MvvError::LayerLengthMismatch {
                layer,
                expected,
                declared: floats.len(),
            });
        }
        Ok((floats, l.num_features))
    }
}

/// Checked dtype decode: no `unsafe`, alignment-independent (`from_le_bytes`),
/// total — typed error on a misaligned byte length.
fn decode_checked(bytes: &[u8], dtype: StorageDtype, layer: usize) -> Result<Vec<f32>, MvvError> {
    match dtype {
        StorageDtype::F32 => {
            if bytes.len() % 4 != 0 {
                return Err(MvvError::Misaligned { layer });
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        StorageDtype::F16 => {
            if bytes.len() % 2 != 0 {
                return Err(MvvError::Misaligned { layer });
            }
            Ok(larql_models::quant::half::decode_f16(bytes))
        }
    }
}

/// |x| via sign-bit clear — pure core, no `libm`/`FloatExt`, total (NaN→NaN).
#[inline]
fn fabs(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0x7FFF_FFFF)
}
